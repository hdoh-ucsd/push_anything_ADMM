"""kIK standalone reachability probe (9.4 / 9.4.5-A).

Drives ``RepositionIKTracker`` in free mode toward a list of test
targets. Isolates the kIK layer from the sampling-C3 wrapper, the
inner C3/C3+ solver, and the contact dynamics — the box is parked
far from the arm so manipuland coupling does not contribute.

Two scopes of tests are run in one pass:

  Tier 1 (horizontal only, varying distance — exposes Phase 2
  fixed-point as a function of distance):
    T1.1: home → ( +0.010, −0.001, 0.200)   1 cm,  direct-line
    T1.2: home → ( +0.050, −0.001, 0.200)   5 cm,  Phase 2 only
    T1.3: home → ( +0.100, −0.001, 0.200)  10 cm,  Phase 2 only

  Tier 2 (z-axis motion — phase isolation):
    T2.1: home → ( +0.000, −0.001, 0.050)   pure descent  (Phase 3)
    T2.2: home → ( +0.100, −0.001, 0.050)   traverse + descend
    T2.3: SKIPPED (lift + traverse) — needs a starting arm_q whose
          FK lands EE at z=0.05; no such pose is checked into the
          repo and adding one is out of scope for this read-only
          probe (per spec, Phase 1 coverage matters less because
          lift always runs first in any complex motion anyway).

  Tier 3 (verdict-A failing targets — sanity check 9.4 reproduces):
    T3.1: W2  ( −0.065, −0.135, 0.050)   Path A α+C-fix undershoot
    T3.2: W1  ( −0.169, −0.043, 0.050)   Path D α+C-fix overshoot

Per target: reset arm to its starting q, reset integrator, simulate
8 s of control at dt_ctrl=0.01 s, log per-step diagnostics, write a
trajectory CSV (results/probe_9_4_5_A_<label>.csv) and a torque-
breakdown CSV (results/probe_9_4_5_A_torque_<label>.csv). At the end,
print a summary table covering every target.

Usage:
    python scripts/probe_9_4_kik_reachability.py
"""
from __future__ import annotations

import csv
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
from control.sampling_c3.reposition import next_waypoint
from control.sampling_c3.reposition_ik import RepositionIKTracker
from sim.env_builder import (
    EE_BODY_NAME,
    INITIAL_ARM_Q,
    build_environment,
)


def _classify_phase(p_now:    np.ndarray,
                    p_target: np.ndarray,
                    *,
                    z_safe:   float,
                    straight_line_thresh: float,
                    finished_tol: float = 0.02,
                    z_eps:        float = 1e-4) -> str:
    """Mirror the rule-set used inside ``next_waypoint`` (single source of
    truth: ``control/sampling_c3/reposition.py: next_waypoint``). Returns
    one of {completed, direct-line, phase1-lift, phase2-traverse,
    phase3-descend}. ``completed`` overlays the next_waypoint phases when
    the EE is within the tracker's ``finished`` threshold (mirrors the
    2 cm ball-of-acceptance asserted at reposition_ik.py:1205).
    """
    direct = float(np.linalg.norm(p_target - p_now))
    if direct <= finished_tol:
        return "completed"
    if direct < straight_line_thresh:
        return "direct-line"
    xy_dist = float(np.linalg.norm(p_target[:2] - p_now[:2]))
    if xy_dist <= z_eps:
        return "phase3-descend"
    if p_now[2] < z_safe - z_eps:
        return "phase1-lift"
    return "phase2-traverse"


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
                breakdown_csv_path: Path, trajectory_csv_path: Path,
                start_arm_q: np.ndarray = INITIAL_ARM_Q,
                dt_ctrl: float = 0.01, sim_seconds: float = 8.0) -> dict:
    """Drive tracker toward p_target for sim_seconds, return summary dict.

    Writes two per-target CSVs:
      - ``trajectory_csv_path`` (one row per control step): time, EE
        position, EE-to-target distance, |u|, classified guide-path
        phase, and ``knot0 - ee_now`` offset (recomputed from the same
        ``next_waypoint`` rule the kIK uses for its warm-start build).
      - ``breakdown_csv_path`` (one row per (step, joint)): torque
        components recomputed from tracker read-only state — see the
        original 9.4.1 probe header for the reconstruction rationale.
    The reconstruction is sanity-checked against the tracker's returned
    ``u`` each step. No controller code is modified.
    """
    print()
    print(f"=== {label}  p_target = ({p_target[0]:+.3f}, "
          f"{p_target[1]:+.3f}, {p_target[2]:+.3f}) ===")

    # Reset state: arm at start_arm_q, box parked far away.
    _set_state(plant, plant_ctx, obj_body,
               arm_q=start_arm_q, obj_xyz=(10.0, 10.0, 0.05))

    # Reset tracker integrator (sticky across runs in the wrapper; for a
    # clean per-target probe we explicitly zero it).
    tracker._integral = np.zeros_like(tracker._integral)
    # Reset the tracker's "previous target" memo so the integrator-reset
    # branch inside compute_torque (reposition_ik.py:1093-1099) always
    # fires on the first step of every new target — otherwise the second
    # target onward inherits the prior target's memo and skips its own
    # reset, defeating the explicit zero above. Read-only field; setting
    # it to None does not change the controller's behaviour, only resets
    # the probe-side initial condition.
    tracker._prev_target_pos = None

    # Reset simulator clock to t=0.
    sim_ctx = simulator.get_mutable_context()
    sim_ctx.SetTime(0.0)
    simulator.Initialize()

    Kp = float(tracker.repos_params.Kp_q)
    Kd = float(tracker.repos_params.Kd_q)
    Ki = float(tracker.repos_params.Ki_q)
    torque_limit = float(tracker.repos_params.torque_limit)

    # Guide-path constants — same values that ``_build_guide_path`` reads
    # at every call (reposition_ik.py:889-892). Recomputing knot0 here
    # gives us the literal p_guide[:, 0] without changing controller code.
    z_safe = float(tracker.repos_params.pwl_waypoint_height)
    line_thresh = float(
        tracker.repos_params.use_straight_line_traj_under_piecewise_linear
    )
    ds = float(tracker.repos_params.speed) * float(tracker.dt)

    # Trajectory log.
    ee_log: list[np.ndarray]    = []
    t_log: list[float]          = []
    dist_log: list[float]       = []
    u_norm_log: list[float]     = []
    phase_log: list[str]        = []
    knot0_offset_log: list[np.ndarray] = []
    infeas_count                = 0
    overshoot_count             = 0
    finished_first_time: float  = float("nan")
    snapshot_at: list[tuple]    = []  # (t, ee, dist, |u|) at t=0..8s
    initial_ee:  np.ndarray     = np.zeros(3)  # filled on step 0 below

    # Per-joint accumulators across the run.
    n_j = n_u  # 7 Franka arm DoFs
    sat_count       = np.zeros(n_j, dtype=int)
    max_abs_demand  = np.zeros(n_j)
    max_abs_demand_at_step = np.full(n_j, -1, dtype=int)
    max_abs_demand_components = np.zeros((n_j, 4))  # P, I, D, grav at max-demand moment

    max_abs_P    = np.zeros(n_j)
    max_abs_I    = np.zeros(n_j)
    max_abs_D    = np.zeros(n_j)
    max_abs_grav = np.zeros(n_j)

    # Of saturated steps (per joint), how many had each component as the
    # dominant (largest |·|) contributor.
    dom_P    = np.zeros(n_j, dtype=int)
    dom_I    = np.zeros(n_j, dtype=int)
    dom_D    = np.zeros(n_j, dtype=int)
    dom_grav = np.zeros(n_j, dtype=int)

    n_steps = int(round(sim_seconds / dt_ctrl))
    next_snap = 0

    # CSV writer: one row per (step, joint).
    csv_fh = open(breakdown_csv_path, "w", newline="")
    csv_w  = csv.writer(csv_fh)
    csv_w.writerow([
        "step", "time", "joint",
        "q_target", "q_now", "q_err",
        "integral", "v_now",
        "tau_P", "tau_I", "tau_D", "tau_grav",
        "tau_demand_pre_clip", "tau_clipped", "saturated",
    ])

    # Trajectory CSV — one row per control step.
    traj_fh = open(trajectory_csv_path, "w", newline="")
    traj_w  = csv.writer(traj_fh)
    traj_w.writerow([
        "step", "time",
        "ee_x", "ee_y", "ee_z",
        "dist_to_target", "u_norm",
        "phase",
        "knot0_dx", "knot0_dy", "knot0_dz", "knot0_norm",
    ])

    for step in range(n_steps + 1):
        sim_time = step * dt_ctrl

        current_q = plant.GetPositions(plant_ctx).copy()
        current_v = plant.GetVelocities(plant_ctx).copy()

        q_arm_now_before = current_q[:n_j].copy()
        v_arm_now_before = current_v[:n_j].copy()

        u, diag = tracker.compute_torque(
            current_q=current_q,
            current_v=current_v,
            plant_ctx=plant_ctx,
            p_target=p_target,
            dt_ctrl=dt_ctrl,
        )

        # --- Torque-component breakdown (Option B: recompute from tracker
        #     read-only state) ---
        # last_q_knots[:, 0] is the IK-solved q_arm for knot 0 (the q_target
        # the PD law tracks). _integral is post-update, post-clip. The kIK's
        # _plant_ctx_ik is left with positions = q_full_target at exit (see
        # reposition_ik.py:1194), so CalcGravityGeneralizedForces on it
        # returns the exact tau_g_arm the kIK used.
        q_arm_target = tracker.last_q_knots[:, 0].copy()
        integral_post = tracker._integral.copy()
        tau_g_full = plant.CalcGravityGeneralizedForces(tracker._plant_ctx_ik)
        tau_grav   = tau_g_full[:n_j].copy()

        q_err     = q_arm_target - q_arm_now_before
        tau_P     = Kp * q_err
        tau_I     = Ki * integral_post
        tau_D     = -Kd * v_arm_now_before
        tau_demand = tau_P + tau_I + tau_D + tau_grav
        tau_clipped = np.clip(tau_demand, -torque_limit, +torque_limit)

        # Sanity: reconstruction must match the tracker's returned u.
        assert np.allclose(tau_clipped, u, atol=1e-9), (
            f"breakdown reconstruction mismatch at step {step}: "
            f"max |delta| = {np.max(np.abs(tau_clipped - u)):.3e}"
        )

        saturated = np.abs(tau_demand) > torque_limit
        sat_count += saturated.astype(int)

        # Per-joint accumulators.
        abs_demand = np.abs(tau_demand)
        for j in range(n_j):
            if abs_demand[j] > max_abs_demand[j]:
                max_abs_demand[j] = abs_demand[j]
                max_abs_demand_at_step[j] = step
                max_abs_demand_components[j, 0] = tau_P[j]
                max_abs_demand_components[j, 1] = tau_I[j]
                max_abs_demand_components[j, 2] = tau_D[j]
                max_abs_demand_components[j, 3] = tau_grav[j]

            if abs(tau_P[j])    > max_abs_P[j]:    max_abs_P[j]    = abs(tau_P[j])
            if abs(tau_I[j])    > max_abs_I[j]:    max_abs_I[j]    = abs(tau_I[j])
            if abs(tau_D[j])    > max_abs_D[j]:    max_abs_D[j]    = abs(tau_D[j])
            if abs(tau_grav[j]) > max_abs_grav[j]: max_abs_grav[j] = abs(tau_grav[j])

            if saturated[j]:
                comps = np.array([abs(tau_P[j]), abs(tau_I[j]),
                                  abs(tau_D[j]), abs(tau_grav[j])])
                k = int(np.argmax(comps))
                if   k == 0: dom_P[j]    += 1
                elif k == 1: dom_I[j]    += 1
                elif k == 2: dom_D[j]    += 1
                else:        dom_grav[j] += 1

            csv_w.writerow([
                step, f"{sim_time:.4f}", j,
                f"{q_arm_target[j]:.6f}", f"{q_arm_now_before[j]:.6f}", f"{q_err[j]:.6f}",
                f"{integral_post[j]:.6f}", f"{v_arm_now_before[j]:.6f}",
                f"{tau_P[j]:.4f}", f"{tau_I[j]:.4f}", f"{tau_D[j]:.4f}", f"{tau_grav[j]:.4f}",
                f"{tau_demand[j]:.4f}", f"{tau_clipped[j]:.4f}",
                int(saturated[j]),
            ])

        # --- Apply torque, advance sim, log ---
        plant.get_actuation_input_port().FixValue(plant_ctx, u)
        ee_pos = plant.CalcPointsPositions(
            plant_ctx, ee_frame, np.zeros(3), world_frame
        ).flatten()

        d = float(np.linalg.norm(ee_pos - p_target))
        ee_log.append(ee_pos.copy())
        t_log.append(sim_time)
        dist_log.append(d)
        u_norm_log.append(float(np.linalg.norm(u)))

        if step == 0:
            initial_ee = ee_pos.copy()

        # Reproduce the kIK's first guide-knot here (read-only). Same call
        # the tracker makes inside _build_guide_path with p_curr=ee_now;
        # equivalent to ``p_guide[:, 0]`` at this step.
        knot0 = next_waypoint(
            p_now=ee_pos,
            p_target=p_target,
            z_safe=z_safe,
            ds=ds,
            straight_line_thresh=line_thresh,
        )
        knot0_off = knot0 - ee_pos
        phase = _classify_phase(
            ee_pos, p_target,
            z_safe=z_safe,
            straight_line_thresh=line_thresh,
        )
        phase_log.append(phase)
        knot0_offset_log.append(knot0_off.copy())

        traj_w.writerow([
            step, f"{sim_time:.4f}",
            f"{ee_pos[0]:+.6f}", f"{ee_pos[1]:+.6f}", f"{ee_pos[2]:+.6f}",
            f"{d:.6f}", f"{np.linalg.norm(u):.4f}",
            phase,
            f"{knot0_off[0]:+.6f}", f"{knot0_off[1]:+.6f}", f"{knot0_off[2]:+.6f}",
            f"{float(np.linalg.norm(knot0_off)):.6f}",
        ])

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

        if step < n_steps:
            simulator.AdvanceTo(sim_time + dt_ctrl)

    csv_fh.close()
    traj_fh.close()

    # Settling = first step `i` such that for the next `window` steps the
    # EE moved <1mm consecutively. Returns the time of step `i` itself
    # (the first quiescent step), not the end of the window.
    settling_time = float("nan")
    settling_idx  = -1
    settling_ee:    np.ndarray | None = None
    settling_phase: str          = ""
    settling_knot0_offset:        np.ndarray | None = None
    window = max(1, int(round(0.10 / dt_ctrl)))
    for i in range(len(ee_log) - window):
        moves = [float(np.linalg.norm(ee_log[j + 1] - ee_log[j]))
                 for j in range(i, i + window)]
        if all(m < 1e-3 for m in moves):
            settling_time         = t_log[i]
            settling_idx          = i
            settling_ee           = ee_log[i].copy()
            settling_phase        = phase_log[i]
            settling_knot0_offset = knot0_offset_log[i].copy()
            break

    summary = dict(
        label                 = label,
        p_target              = p_target,
        initial_ee            = initial_ee,
        ee_final              = ee_log[-1],
        dist_final            = dist_log[-1],
        settling_time         = settling_time,
        settling_idx          = settling_idx,
        settling_ee           = settling_ee if settling_ee is not None else ee_log[-1],
        settling_phase        = settling_phase if settling_idx >= 0 else phase_log[-1],
        settling_knot0_offset = (settling_knot0_offset
                                 if settling_knot0_offset is not None
                                 else knot0_offset_log[-1]),
        infeas_count          = infeas_count,
        overshoot_count       = overshoot_count,
        finished_first_time   = finished_first_time,
        max_u_norm            = max(u_norm_log),
        snapshot_at           = snapshot_at,
        n_steps               = n_steps + 1,
        # Breakdown aggregates
        sat_count             = sat_count,
        max_abs_demand        = max_abs_demand,
        max_abs_demand_at_step = max_abs_demand_at_step,
        max_abs_demand_components = max_abs_demand_components,
        max_abs_P             = max_abs_P,
        max_abs_I             = max_abs_I,
        max_abs_D             = max_abs_D,
        max_abs_grav          = max_abs_grav,
        dom_P                 = dom_P,
        dom_I                 = dom_I,
        dom_D                 = dom_D,
        dom_grav              = dom_grav,
        torque_limit          = torque_limit,
        csv_path              = str(breakdown_csv_path),
        trajectory_csv_path   = str(trajectory_csv_path),
    )
    return summary


def _print_breakdown_summary(s: dict) -> None:
    """Block 4 report — torque-breakdown tables and saturation pattern."""
    print()
    print("-" * 78)
    print(f"BREAKDOWN: {s['label']}")
    print(f"  csv: {s['csv_path']}")
    print(f"  total steps: {s['n_steps']}   torque_limit: {s['torque_limit']:.1f} Nm")
    print()

    # Max-abs of each component, with WHICH joint and step.
    def _argmax_joint(arr): return int(np.argmax(arr))
    j_P    = _argmax_joint(s["max_abs_P"])
    j_I    = _argmax_joint(s["max_abs_I"])
    j_D    = _argmax_joint(s["max_abs_D"])
    j_g    = _argmax_joint(s["max_abs_grav"])

    print("Component maxima (across all steps & joints):")
    print(f"  tau_P    max={s['max_abs_P'][j_P]:7.2f} Nm  at joint {j_P}")
    print(f"  tau_I    max={s['max_abs_I'][j_I]:7.2f} Nm  at joint {j_I}")
    print(f"  tau_D    max={s['max_abs_D'][j_D]:7.2f} Nm  at joint {j_D}")
    print(f"  tau_grav max={s['max_abs_grav'][j_g]:7.2f} Nm  at joint {j_g}")
    print()

    print("Per-joint saturation pattern:")
    print(f"  {'j':>2}  {'sat_steps':>9}  {'max|demand|':>11}  "
          f"{'P':>7}  {'I':>7}  {'D':>7}  {'grav':>7}  {'dominant':>9}  "
          f"{'(P/I/D/g sat-frac)':>22}")
    for j in range(7):
        total_sat = s["sat_count"][j]
        comps = s["max_abs_demand_components"][j]
        # Components at the max-demand moment.
        c_P, c_I, c_D, c_g = comps
        dom_names = ["P", "I", "D", "grav"]
        dom_at_max = dom_names[int(np.argmax([abs(c_P), abs(c_I), abs(c_D), abs(c_g)]))]
        # Saturated-fraction breakdown.
        sums = total_sat if total_sat > 0 else 1
        fP, fI, fD, fg = (s["dom_P"][j] / sums, s["dom_I"][j] / sums,
                          s["dom_D"][j] / sums, s["dom_grav"][j] / sums)
        print(f"  {j:>2}  {total_sat:>9d}  {s['max_abs_demand'][j]:>11.2f}  "
              f"{c_P:>+7.2f}  {c_I:>+7.2f}  {c_D:>+7.2f}  {c_g:>+7.2f}  "
              f"{dom_at_max:>9}  ({fP:.2f}/{fI:.2f}/{fD:.2f}/{fg:.2f})")
    print()


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

    # 9.4.5-A target tiers (read top-of-file docstring for tier rationale).
    # Tuple: (label, p_target, start_arm_q). T2.3 deliberately skipped.
    targets = [
        # Tier 1 — horizontal only at z=0.20 (home pose plane)
        ("T1_1_xplus_1cm",  np.array([+0.010, -0.001, 0.200]), INITIAL_ARM_Q),
        ("T1_2_xplus_5cm",  np.array([+0.050, -0.001, 0.200]), INITIAL_ARM_Q),
        ("T1_3_xplus_10cm", np.array([+0.100, -0.001, 0.200]), INITIAL_ARM_Q),
        # Tier 2 — z-axis motion (T2.3 skipped: needs a sub-z_safe start
        # arm_q that is not checked into the repo)
        ("T2_1_descend",            np.array([+0.000, -0.001, 0.050]), INITIAL_ARM_Q),
        ("T2_2_traverse_descend",   np.array([+0.100, -0.001, 0.050]), INITIAL_ARM_Q),
        # Tier 3 — verdict-A failing targets, sanity-check 9.4 reproduces
        ("T3_1_W2_path_A_undershoot", np.array([-0.065, -0.135, 0.050]), INITIAL_ARM_Q),
        ("T3_2_W1_path_D_overshoot",  np.array([-0.169, -0.043, 0.050]), INITIAL_ARM_Q),
    ]
    print()
    print("[probe-9.4.5-A] T2.3 (lift-then-traverse) skipped — needs a "
          "starting arm_q with EE at z=0.05; no such pose checked into the "
          "repo. Per scope: Phase 1 coverage matters less because lift "
          "always runs first in any complex motion anyway.")

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    summaries = []
    for label, p_target, start_arm_q in targets:
        breakdown_csv = results_dir / f"probe_9_4_5_A_torque_{label}.csv"
        traj_csv      = results_dir / f"probe_9_4_5_A_{label}.csv"
        summaries.append(_run_target(
            label, p_target,
            tracker=tracker, plant=plant, ee_frame=ee_frame,
            world_frame=world_frame, simulator=simulator,
            plant_ctx=plant_ctx, obj_body=obj_body, n_u=n_u,
            breakdown_csv_path=breakdown_csv,
            trajectory_csv_path=traj_csv,
            start_arm_q=start_arm_q,
        ))

    # ------------------------------------------------------------------
    # Final report — original per-target detail block (kept for context).
    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("BLOCK 4 REPORT — per-target detail")
    print("=" * 78)
    fmt = ("{label:>30} | dist_final={d:7.4f}m | settling_time={st:6.3f}s | "
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
        ee0 = s['initial_ee']
        ee  = s['ee_final']
        print(f"{'':>32}ee_initial=({ee0[0]:+.4f}, {ee0[1]:+.4f}, {ee0[2]:+.4f})  "
              f"ee_final=({ee[0]:+.4f}, {ee[1]:+.4f}, {ee[2]:+.4f})  "
              f"target=({s['p_target'][0]:+.4f}, {s['p_target'][1]:+.4f}, {s['p_target'][2]:+.4f})")

    # ------------------------------------------------------------------
    # 9.4.5-A summary table — one row per target, scope-spec format.
    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("BLOCK 4 REPORT — 9.4.5-A summary table")
    print("=" * 78)
    print(f"{'Test':<32} {'Dist (m)':>10} {'Settled?':>9} {'EE→target (m)':>14} "
          f"{'Phase@settle':>16} {'knot0 - ee_now (mm)':>26}")
    print("-" * 110)
    for s in summaries:
        # Cartesian distance from this run's actual initial EE (not the
        # scope's stated home approximation) to the target.
        d_total = float(np.linalg.norm(s['p_target'] - s['initial_ee']))
        settled = "YES" if not np.isnan(s['settling_time']) else "NO"
        if not np.isnan(s['settling_time']):
            settled = f"YES@{s['settling_time']:5.2f}s"
        ko = s['settling_knot0_offset'] * 1e3  # mm
        ko_str = f"({ko[0]:+6.2f},{ko[1]:+6.2f},{ko[2]:+6.2f})"
        print(f"{s['label']:<32} {d_total:>10.4f} {settled:>9} "
              f"{s['dist_final']:>14.4f} {s['settling_phase']:>16} {ko_str:>26}")

    for s in summaries:
        _print_breakdown_summary(s)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
