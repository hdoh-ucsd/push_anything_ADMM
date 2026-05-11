"""Integrator characterization probe (9.4.2).

9.4.1 found the kIK joint-PD integrator at step 700 of the 8-second 9.4
probe sat at magnitude 0.66–0.70 across joints, ~35% of I_max=2.0,
despite a sustained joint-1 q_err of 0.054 rad. Standard PI with no
anti-windup, fed sustained error, should accumulate to clamp. Code
inspection (control/sampling_c3/reposition_ik.py) confirms the
integrator update is `I += q_err · dt; clip(±I_max)` — no leak, no
back-calculation, no rate limit, no conditional update; only reset is
on target change > 1mm. Working hypothesis: H2 — slow update; the 8s
window in 9.4 was too short for the integrator to fully accumulate.

This probe drives the standalone kIK toward verdict-A target W1 for
30 simulated seconds (vs 8s in 9.4) and logs the integrator state,
q_err, tau_I, and tau_demand per joint per step. Snapshots at
t = 0, 1, 5, 10, 20, 30 reveal whether:

  - integrator reaches I_max (clamp)   → H2 confirmed (slow update)
  - integrator asymptotes sub-clamp    → equilibrium where q_err
                                          shrinks faster than the
                                          integrator grows; still H2
                                          but with closed-loop coupling
  - integrator decreases or oscillates → rules H2 out

Pure read-only instrumentation; no controller behavior changes.

Usage:
    python scripts/probe_9_4_2_integrator.py
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pydrake.all as ad
import yaml

from control.sampling_c3.params import SamplingC3Params, RepositioningTrajectoryType
from control.sampling_c3.reposition_ik import RepositionIKTracker
from sim.env_builder import EE_BODY_NAME, INITIAL_ARM_Q, build_environment


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_task_cfg() -> dict:
    with open(PROJECT_ROOT / "config/tasks.yaml") as f:
        return yaml.safe_load(f)["tasks"]["pushing"]


def _resolve_scene_graph(diagram) -> ad.SceneGraph:
    sgs = [s for s in diagram.GetSystems() if isinstance(s, ad.SceneGraph)]
    assert len(sgs) == 1
    return sgs[0]


def _set_state(plant, plant_ctx, obj_body, *, arm_q, obj_xyz):
    n_q = plant.num_positions()
    q = np.zeros(n_q)
    q[:7] = arm_q
    s = obj_body.floating_positions_start()
    q[s + 0] = 1.0
    q[s + 4] = obj_xyz[0]
    q[s + 5] = obj_xyz[1]
    q[s + 6] = obj_xyz[2]
    plant.SetPositions(plant_ctx, q)
    plant.SetVelocities(plant_ctx, np.zeros(plant.num_velocities()))
    return q


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sim-seconds", type=float, default=30.0,
                    help="Simulated duration of the constant-target run. "
                         "Default 30 (9.4.2 invocation). 9.4.3 uses 90 to "
                         "observe post-clamp behavior on joint 1.")
    args = ap.parse_args()
    sim_seconds = float(args.sim_seconds)

    task_cfg = _load_task_cfg()
    diagram, plant, _panda, obj_model, _meshcat, _plant_ad, _ctx_ad = \
        build_environment(task_cfg)
    scene_graph = _resolve_scene_graph(diagram)

    obj_body    = plant.GetBodyByName(task_cfg["link_name"], obj_model)
    ee_frame    = plant.GetFrameByName(EE_BODY_NAME)
    world_frame = plant.world_frame()

    simulator = ad.Simulator(diagram)
    sim_ctx   = simulator.get_mutable_context()
    plant_ctx = plant.GetMyMutableContextFromRoot(sim_ctx)

    params = SamplingC3Params.from_yaml(PROJECT_ROOT / "config/sampling_c3_kik.yaml")
    assert params.reposition_params.traj_type == RepositioningTrajectoryType.kIK

    n_j = 7
    tracker = RepositionIKTracker(
        plant=plant, ee_frame=ee_frame, obj_body=obj_body,
        n_arm_dofs=n_j, horizon=20, dt=0.05,
        repos_params=params.reposition_params,
        ik_params=params.repos_ik_params,
        diagram=diagram, scene_graph=scene_graph,
    )

    Kp = float(tracker.repos_params.Kp_q)
    Kd = float(tracker.repos_params.Kd_q)
    Ki = float(tracker.repos_params.Ki_q)
    I_max        = float(tracker.repos_params.I_max)
    torque_limit = float(tracker.repos_params.torque_limit)
    print(f"[probe-9.4.2] PD: Kp={Kp}  Kd={Kd}  Ki={Ki}  I_max={I_max}  "
          f"torque_limit={torque_limit}")

    # Verdict-A target W1 — same as 9.4 / 9.4.1.
    p_target = np.array([-0.169, -0.043, 0.050])

    _set_state(plant, plant_ctx, obj_body,
               arm_q=INITIAL_ARM_Q, obj_xyz=(10.0, 10.0, 0.05))
    tracker._integral = np.zeros_like(tracker._integral)
    sim_ctx.SetTime(0.0)
    simulator.Initialize()

    dt_ctrl = 0.01
    n_steps = int(round(sim_seconds / dt_ctrl))

    base_snaps = [0, 1, 5, 10, 20, 30]
    extra_snaps = [45, 60, 75, 90]
    snapshot_times = [t for t in base_snaps if t <= sim_seconds + 1e-9] + \
                     [t for t in extra_snaps if t <= sim_seconds + 1e-9]
    next_snap_idx = 0

    csv_name = f"probe_9_4_2_integrator_W1_{int(round(sim_seconds))}s.csv"
    csv_path = PROJECT_ROOT / "results" / csv_name
    csv_path.parent.mkdir(exist_ok=True)
    csv_fh = open(csv_path, "w", newline="")
    csv_w  = csv.writer(csv_fh)
    csv_w.writerow([
        "step", "time", "joint",
        "q_target", "q_now", "q_err", "v_now",
        "integral", "tau_P", "tau_I", "tau_D", "tau_grav",
        "tau_demand", "tau_clipped", "saturated",
    ])

    print()
    print(f"=== W1 30s integrator probe   p_target=({p_target[0]:+.3f}, "
          f"{p_target[1]:+.3f}, {p_target[2]:+.3f}) ===")

    # Snapshots of per-joint integrator + q_err + tau_I + tau_demand.
    snap_rows: list[tuple] = []

    for step in range(n_steps + 1):
        sim_time = step * dt_ctrl

        current_q = plant.GetPositions(plant_ctx).copy()
        current_v = plant.GetVelocities(plant_ctx).copy()

        q_arm_now_before = current_q[:n_j].copy()
        v_arm_now_before = current_v[:n_j].copy()

        u, _diag = tracker.compute_torque(
            current_q=current_q, current_v=current_v,
            plant_ctx=plant_ctx, p_target=p_target, dt_ctrl=dt_ctrl,
        )

        q_arm_target  = tracker.last_q_knots[:, 0].copy()
        integral_post = tracker._integral.copy()
        tau_grav_full = plant.CalcGravityGeneralizedForces(tracker._plant_ctx_ik)
        tau_grav      = tau_grav_full[:n_j].copy()

        q_err     = q_arm_target - q_arm_now_before
        tau_P     = Kp * q_err
        tau_I     = Ki * integral_post
        tau_D     = -Kd * v_arm_now_before
        tau_demand = tau_P + tau_I + tau_D + tau_grav
        tau_clipped = np.clip(tau_demand, -torque_limit, +torque_limit)

        assert np.allclose(tau_clipped, u, atol=1e-9), (
            f"breakdown mismatch step {step}: "
            f"max |delta| = {np.max(np.abs(tau_clipped - u)):.3e}"
        )

        saturated = np.abs(tau_demand) > torque_limit

        for j in range(n_j):
            csv_w.writerow([
                step, f"{sim_time:.4f}", j,
                f"{q_arm_target[j]:.6f}", f"{q_arm_now_before[j]:.6f}", f"{q_err[j]:.6f}",
                f"{v_arm_now_before[j]:.6f}",
                f"{integral_post[j]:.6f}",
                f"{tau_P[j]:.4f}", f"{tau_I[j]:.4f}", f"{tau_D[j]:.4f}", f"{tau_grav[j]:.4f}",
                f"{tau_demand[j]:.4f}", f"{tau_clipped[j]:.4f}",
                int(saturated[j]),
            ])

        plant.get_actuation_input_port().FixValue(plant_ctx, u)

        if (next_snap_idx < len(snapshot_times)
                and sim_time + 1e-9 >= snapshot_times[next_snap_idx]):
            ee_pos = plant.CalcPointsPositions(
                plant_ctx, ee_frame, np.zeros(3), world_frame).flatten()
            snap_rows.append((sim_time, q_err.copy(), integral_post.copy(),
                              tau_I.copy(), tau_demand.copy(),
                              saturated.copy(), ee_pos.copy()))
            next_snap_idx += 1

        if step < n_steps:
            simulator.AdvanceTo(sim_time + dt_ctrl)

    csv_fh.close()
    print(f"[probe-9.4.2] CSV written: {csv_path}")

    # Joint-1 focused report.
    print()
    print("=" * 78)
    print("BLOCK 3 REPORT — joint 1 (shoulder) trajectory")
    print("=" * 78)
    print(f"  {'t (s)':>5}  {'q_err (rad)':>12}  {'integral':>10}  "
          f"{'tau_I (Nm)':>11}  {'tau_demand (Nm)':>15}  {'sat?':>4}")
    for (t, qe, ig, ti, td, sat, ee) in snap_rows:
        print(f"  {t:>5.1f}  {qe[1]:>+12.5f}  {ig[1]:>+10.5f}  "
              f"{ti[1]:>+11.4f}  {td[1]:>+15.4f}  {int(sat[1]):>4d}")

    print()
    print("EE position + EE-to-target distance:")
    print(f"  {'t (s)':>5}  {'ee_x':>8}  {'ee_y':>8}  {'ee_z':>8}  "
          f"{'dist_to_target (m)':>20}")
    for (t, qe, ig, ti, td, sat, ee) in snap_rows:
        d = float(np.linalg.norm(ee - p_target))
        print(f"  {t:>5.1f}  {ee[0]:>+8.4f}  {ee[1]:>+8.4f}  {ee[2]:>+8.4f}  "
              f"{d:>20.4f}")

    print()
    print("All-joints integrator snapshot (signed):")
    print(f"  {'t (s)':>5}  " + "  ".join(f"j{j:1d}".rjust(8) for j in range(n_j)))
    for (t, qe, ig, ti, td, sat, ee) in snap_rows:
        print(f"  {t:>5.1f}  " + "  ".join(f"{ig[j]:+8.4f}" for j in range(n_j)))

    print()
    print("All-joints q_err snapshot:")
    print(f"  {'t (s)':>5}  " + "  ".join(f"j{j:1d}".rjust(9) for j in range(n_j)))
    for (t, qe, ig, ti, td, sat, ee) in snap_rows:
        print(f"  {t:>5.1f}  " + "  ".join(f"{qe[j]:+9.5f}" for j in range(n_j)))

    # Hypothesis classification helper.
    j1_traj = [(t, ig[1]) for (t, qe, ig, ti, td, sat, ee) in snap_rows]
    final_abs_I = abs(j1_traj[-1][1])
    print()
    print(f"Joint-1 integrator at t=30s:  {j1_traj[-1][1]:+.5f}  "
          f"(|I|={final_abs_I:.4f}, I_max={I_max})")
    if final_abs_I >= 0.99 * I_max:
        print("  → reached I_max clamp. H2 confirmed (slow update; just needs time).")
    elif final_abs_I > abs(j1_traj[2][1]):   # increased between t=1 and t=30
        print("  → still growing at t=30s; trajectory consistent with H2 but "
              "either clamps later or asymptotes sub-clamp.")
    else:
        print("  → did NOT grow past early-snapshot magnitude. H2 ruled out; "
              "investigate H1/H3/H4.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
