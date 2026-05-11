"""kIK standalone reachability probe (9.4).

Drives ``RepositionIKTracker`` in free mode toward the verbatim targets
that the wrapper commanded during 9.3.4 verdict-A runs but the EE did
not reach. Isolates the kIK layer from the sampling-C3 wrapper, the
inner C3/C3+ solver, and the contact dynamics — the box is parked far
from the arm so manipuland coupling does not contribute.

Test targets (verbatim from 9.3.4 GS-tgt logs at step=800):
    W1: (-0.169, -0.043, 0.050)  — Path D α+C-fix overshoot case
    W2: (-0.065, -0.135, 0.050)  — Path A α+C-fix undershoot case

Per target: reset arm to INITIAL_ARM_Q, reset integrator, simulate 8 s
of control at dt_ctrl=0.01 s, log per-step diagnostics, report the
final EE position, settling time, ik failures, knot0-overshoot count
(IPOPT slow-solve proxy), and max torque.

Usage:
    python scripts/probe_9_4_kik_reachability.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow `from control.* import ...` and `from sim.* import ...` when this
# script is invoked directly (`python scripts/probe_9_4_*.py`). Python adds
# the script's own dir to sys.path, not the project root.
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pydrake.all as ad
import yaml

from control.sampling_c3.params import SamplingC3Params, RepositioningTrajectoryType
from control.sampling_c3.reposition_ik import RepositionIKTracker
from sim.env_builder import (
    EE_BODY_NAME,
    INITIAL_ARM_Q,
    build_environment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_task_cfg() -> dict:
    with open(PROJECT_ROOT / "config/tasks.yaml") as f:
        return yaml.safe_load(f)["tasks"]["pushing"]


def _resolve_scene_graph(diagram) -> ad.SceneGraph:
    sgs = [s for s in diagram.GetSystems() if isinstance(s, ad.SceneGraph)]
    assert len(sgs) == 1, f"expected 1 SceneGraph, found {len(sgs)}"
    return sgs[0]


def _set_state(plant, plant_ctx, obj_body, *, arm_q, obj_xyz):
    """Plant the arm at arm_q and park the box at obj_xyz (identity pose)."""
    n_q = plant.num_positions()
    q = np.zeros(n_q)
    q[:7] = arm_q
    s = obj_body.floating_positions_start()
    q[s + 0] = 1.0  # qw
    q[s + 4] = obj_xyz[0]
    q[s + 5] = obj_xyz[1]
    q[s + 6] = obj_xyz[2]
    plant.SetPositions(plant_ctx, q)
    plant.SetVelocities(plant_ctx, np.zeros(plant.num_velocities()))
    return q


def _run_target(label: str, p_target: np.ndarray, *, tracker, plant, ee_frame,
                world_frame, simulator, plant_ctx, obj_body, n_u: int,
                dt_ctrl: float = 0.01, sim_seconds: float = 8.0) -> dict:
    """Drive tracker toward p_target for sim_seconds, return summary dict."""
    print()
    print(f"=== {label}  p_target = ({p_target[0]:+.3f}, "
          f"{p_target[1]:+.3f}, {p_target[2]:+.3f}) ===")

    # Reset state: arm at INITIAL_ARM_Q, box parked far away.
    _set_state(plant, plant_ctx, obj_body,
               arm_q=INITIAL_ARM_Q, obj_xyz=(10.0, 10.0, 0.05))

    # Reset tracker integrator (sticky across runs in the wrapper; for a
    # clean per-target probe we explicitly zero it).
    tracker._integral = np.zeros_like(tracker._integral)

    # Reset simulator clock to t=0.
    sim_ctx = simulator.get_mutable_context()
    sim_ctx.SetTime(0.0)
    simulator.Initialize()

    # Trajectory log.
    n_q = plant.num_positions()
    ee_log: list[np.ndarray]   = []
    t_log: list[float]         = []
    dist_log: list[float]      = []
    u_norm_log: list[float]    = []
    infeas_count               = 0
    overshoot_count            = 0
    finished_first_time: float = float("nan")
    snapshot_at: list[tuple]   = []  # (t, ee, dist, |u|) at t=0..8s

    n_steps = int(round(sim_seconds / dt_ctrl))
    next_snap = 0  # next integer second to log

    for step in range(n_steps + 1):
        sim_time = step * dt_ctrl

        current_q = plant.GetPositions(plant_ctx).copy()
        current_v = plant.GetVelocities(plant_ctx).copy()

        u, diag = tracker.compute_torque(
            current_q=current_q,
            current_v=current_v,
            plant_ctx=plant_ctx,
            p_target=p_target,
            dt_ctrl=dt_ctrl,
        )

        # kIK tracker already includes grav-comp in u (see main.py:520-524).
        plant.get_actuation_input_port().FixValue(plant_ctx, u)

        ee_pos = plant.CalcPointsPositions(
            plant_ctx, ee_frame, np.zeros(3), world_frame
        ).flatten()

        d = float(np.linalg.norm(ee_pos - p_target))
        ee_log.append(ee_pos.copy())
        t_log.append(sim_time)
        dist_log.append(d)
        u_norm_log.append(float(np.linalg.norm(u)))

        if not bool(diag.get("knot0_feasible", True)):
            infeas_count += 1
        if float(diag.get("knot0_overshoot_ms", 0.0)) > 0.0:
            overshoot_count += 1
        if bool(diag.get("finished", False)) and np.isnan(finished_first_time):
            finished_first_time = sim_time

        if sim_time + 1e-9 >= next_snap and next_snap <= int(sim_seconds):
            snapshot_at.append((next_snap, ee_pos.copy(), d, float(np.linalg.norm(u))))
            print(f"  t={sim_time:4.2f}s  ee=({ee_pos[0]:+.3f}, {ee_pos[1]:+.3f}, "
                  f"{ee_pos[2]:+.3f})  dist_to_target={d:.4f}m  |u|={np.linalg.norm(u):.2f}Nm  "
                  f"feas={diag.get('knot0_feasible')}  finished={diag.get('finished')}")
            next_snap += 1

        # Advance the sim one control step.
        if step < n_steps:
            simulator.AdvanceTo(sim_time + dt_ctrl)

    # Settling time: first sample t* such that for all t ≥ t* within
    # [t*, t* + 0.1s] the EE moves < 1 mm between consecutive steps.
    settling_time = float("nan")
    window = max(1, int(round(0.10 / dt_ctrl)))  # 10 steps = 0.1 s
    for i in range(len(ee_log) - window):
        moves = [float(np.linalg.norm(ee_log[j + 1] - ee_log[j]))
                 for j in range(i, i + window)]
        if all(m < 1e-3 for m in moves):
            settling_time = t_log[i]
            break

    summary = dict(
        label                 = label,
        p_target              = p_target,
        ee_final              = ee_log[-1],
        dist_final            = dist_log[-1],
        settling_time         = settling_time,
        infeas_count          = infeas_count,
        overshoot_count       = overshoot_count,
        finished_first_time   = finished_first_time,
        max_u_norm            = max(u_norm_log),
        snapshot_at           = snapshot_at,
        n_steps               = n_steps + 1,
    )
    return summary


def main() -> int:
    task_cfg = _load_task_cfg()

    # Build the environment (box present, will be parked far away).
    diagram, plant, _panda, obj_model, _meshcat, _plant_ad, _ctx_ad = \
        build_environment(task_cfg)
    scene_graph = _resolve_scene_graph(diagram)

    obj_body = plant.GetBodyByName(task_cfg["link_name"], obj_model)
    ee_frame = plant.GetFrameByName(EE_BODY_NAME)
    world_frame = plant.world_frame()

    simulator = ad.Simulator(diagram)
    sim_ctx   = simulator.get_mutable_context()
    plant_ctx = plant.GetMyMutableContextFromRoot(sim_ctx)

    # Load the kIK YAML used by the verdict-A runs (so the IK params and
    # PD gains match exactly what step 9.3.4 used).
    params = SamplingC3Params.from_yaml(PROJECT_ROOT / "config/sampling_c3_kik.yaml")
    assert params.reposition_params.traj_type == RepositioningTrajectoryType.kIK, \
        "config/sampling_c3_kik.yaml must use traj_type: kIK"

    n_u = 7  # Franka arm DoFs

    tracker = RepositionIKTracker(
        plant=plant, ee_frame=ee_frame, obj_body=obj_body,
        n_arm_dofs=n_u,
        horizon=20,
        dt=0.05,
        repos_params=params.reposition_params,
        ik_params=params.repos_ik_params,
        diagram=diagram,
        scene_graph=scene_graph,
    )
    print(f"[probe-9.4] tracker constructed; repos_ik_params defaults in use")
    print(f"[probe-9.4] PD gains: Kp={params.reposition_params.Kp_q}  "
          f"Kd={params.reposition_params.Kd_q}  Ki={params.reposition_params.Ki_q}  "
          f"I_max={params.reposition_params.I_max}  "
          f"torque_limit={params.reposition_params.torque_limit}")

    targets = [
        ("W1_path_D_overshoot",  np.array([-0.169, -0.043, 0.050])),
        ("W2_path_A_undershoot", np.array([-0.065, -0.135, 0.050])),
    ]

    summaries = []
    for label, p_target in targets:
        summaries.append(_run_target(
            label, p_target,
            tracker=tracker, plant=plant, ee_frame=ee_frame,
            world_frame=world_frame, simulator=simulator,
            plant_ctx=plant_ctx, obj_body=obj_body, n_u=n_u,
        ))

    # Final report.
    print()
    print("=" * 78)
    print("BLOCK 4 REPORT")
    print("=" * 78)
    fmt = ("{label:>22} | dist_final={d:7.4f}m | settling_time={st:6.3f}s | "
           "infeas={inf:>4d} | knot0_overshoot={ov:>4d} | finished_at={ft} | "
           "max|u|={mu:.2f}Nm")
    for s in summaries:
        ft = f"{s['finished_first_time']:.3f}s" if not np.isnan(s['finished_first_time']) else "NEVER"
        print(fmt.format(
            label=s['label'],
            d=s['dist_final'],
            st=s['settling_time'] if not np.isnan(s['settling_time']) else float('nan'),
            inf=s['infeas_count'],
            ov=s['overshoot_count'],
            ft=ft,
            mu=s['max_u_norm'],
        ))
        ee = s['ee_final']
        print(f"{'':>24}ee_final=({ee[0]:+.4f}, {ee[1]:+.4f}, {ee[2]:+.4f})  "
              f"target=({s['p_target'][0]:+.4f}, {s['p_target'][1]:+.4f}, {s['p_target'][2]:+.4f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
