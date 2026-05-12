"""9.4.5-A.1 — Hold-home-pose probe (Effect B isolation).

9.4.5-A baseline (commit 8827917) found that all 7 standalone kIK
tests converge to the same fixed point at EE ≈ (-0.016, -0.084, 0.025),
having fallen from home pose (z=0.2) within 14 control steps. The
report identified two coupled effects:

    Effect A: Hypothesis F per-stride truncation (already diagnosed)
    Effect B: home pose is not a static equilibrium under PD with
              torque-limited grav-comp

This probe isolates Effect B by driving the joint-PD law directly
with q_target = INITIAL_ARM_Q (the home pose), FIXED FOR THE ENTIRE
RUN. No kIK, no IK solves, no guide-path construction. If the arm
holds, Effect B is caused by the kIK's q_target sequence; if it
falls, Effect B is independent of the kIK and lives at the executor.

Implementation: Option A (recompute PD in probe). The kIK's PD law
is inline inside compute_torque (reposition_ik.py:1173-1202), not
exposed as a separate callable. Extracting it would require
modifying reposition_ik.py, which the 9.4.5-A.1 spec prohibits.

The PD law mirrors reposition_ik.py:1173-1202 verbatim:

    q_err = q_target - q_now
    integral += q_err * dt_ctrl
    integral = np.clip(integral, ±I_max)
    u_p = Kp * q_err
    u_i = Ki * integral
    u_d = -Kd * v_now
    tau_g = grav_comp(q_target)   # constant since q_target is fixed
    u = np.clip(tau_g + u_p + u_i + u_d, ±torque_limit)

Usage:
    python scripts/probe_9_4_5_A1_hold_home_pose.py
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pydrake.all as ad
import yaml

from control.sampling_c3.params import SamplingC3Params
from sim.env_builder import EE_BODY_NAME, INITIAL_ARM_Q, build_environment


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_task_cfg() -> dict:
    with open(PROJECT_ROOT / "config/tasks.yaml") as f:
        return yaml.safe_load(f)["tasks"]["pushing"]


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
    task_cfg = _load_task_cfg()
    diagram, plant, _panda, obj_model, _meshcat, _plant_ad, _ctx_ad = \
        build_environment(task_cfg)

    obj_body    = plant.GetBodyByName(task_cfg["link_name"], obj_model)
    ee_frame    = plant.GetFrameByName(EE_BODY_NAME)
    world_frame = plant.world_frame()

    simulator = ad.Simulator(diagram)
    sim_ctx   = simulator.get_mutable_context()
    plant_ctx = plant.GetMyMutableContextFromRoot(sim_ctx)

    # Auxiliary diagram context for grav-comp at q_target. Mirrors the
    # kIK's _plant_ctx_ik pattern (reposition_ik.py:1196-1199): mutate
    # an aux context to compute tau_g(q_target) without disturbing the
    # live sim state. Because q_target is fixed in this probe, we set
    # the aux context once and reuse the resulting tau_g.
    diag_ctx_aux  = diagram.CreateDefaultContext()
    plant_ctx_aux = plant.GetMyMutableContextFromRoot(diag_ctx_aux)

    # Load PD parameters from the kIK YAML so this probe tracks the
    # production gains. Same source-of-truth used by the 9.4.x probes.
    params = SamplingC3Params.from_yaml(PROJECT_ROOT / "config/sampling_c3_kik.yaml")
    rp = params.reposition_params
    Kp           = float(rp.Kp_q)
    Kd           = float(rp.Kd_q)
    Ki           = float(rp.Ki_q)
    I_max        = float(rp.I_max)
    torque_limit = float(rp.torque_limit)

    n_j = 7
    print(f"[probe-9.4.5-A.1] PD: Kp={Kp}  Kd={Kd}  Ki={Ki}  "
          f"I_max={I_max}  torque_limit={torque_limit}")
    print(f"[probe-9.4.5-A.1] q_target = INITIAL_ARM_Q (FIXED for the entire run)")

    # ------------------------------------------------------------------
    # Set initial state: arm at home, box parked far away.
    # ------------------------------------------------------------------
    q_target_arm  = np.asarray(INITIAL_ARM_Q, dtype=float).copy()
    q_full_init   = _set_state(
        plant, plant_ctx, obj_body, arm_q=q_target_arm, obj_xyz=(10.0, 10.0, 0.05)
    )
    # Also set aux context to q_target_full (with box parked) so its
    # CalcGravityGeneralizedForces returns the gravity load at q_target.
    plant.SetPositions(plant_ctx_aux, q_full_init)

    # tau_g at q_target — constant for the run (q_target is fixed).
    tau_g_full   = plant.CalcGravityGeneralizedForces(plant_ctx_aux)
    tau_g_target = tau_g_full[:n_j].copy()
    print(f"[probe-9.4.5-A.1] tau_g(q_target) per joint (Nm): " +
          " ".join(f"{v:+.2f}" for v in tau_g_target))

    sim_ctx.SetTime(0.0)
    simulator.Initialize()

    # ------------------------------------------------------------------
    # PD state and run loop.
    # ------------------------------------------------------------------
    integral = np.zeros(n_j)
    dt_ctrl = 0.01
    sim_seconds = 30.0
    n_steps = int(round(sim_seconds / dt_ctrl))

    snapshot_times = [0.0, 0.14, 1.0, 5.0, 10.0, 30.0]
    next_snap_idx  = 0
    snapshots: list[tuple] = []  # (t, ee, q_err, integral, tau_demand, tau_clipped, sat_mask)

    csv_path = PROJECT_ROOT / "results" / "probe_9_4_5_A1_hold_home_pose.csv"
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
    print("=== run ===")
    for step in range(n_steps + 1):
        sim_time = step * dt_ctrl

        current_q = plant.GetPositions(plant_ctx).copy()
        current_v = plant.GetVelocities(plant_ctx).copy()

        q_now = current_q[:n_j].copy()
        v_now = current_v[:n_j].copy()

        # PD law (mirrors reposition_ik.py:1173-1202).
        q_err = q_target_arm - q_now
        integral += q_err * dt_ctrl
        np.clip(integral, -I_max, +I_max, out=integral)

        u_p = Kp * q_err
        u_i = Ki * integral
        u_d = -Kd * v_now
        tau_demand  = tau_g_target + u_p + u_i + u_d
        tau_clipped = np.clip(tau_demand, -torque_limit, +torque_limit)
        saturated   = np.abs(tau_demand) > torque_limit

        # Apply to plant (the actuation port takes per-joint torques on
        # the 7 arm DoFs only — env_builder welds the box to a floating
        # base with no actuators, so the actuation vector size matches
        # n_arm_dofs = 7).
        plant.get_actuation_input_port().FixValue(plant_ctx, tau_clipped)

        ee_pos = plant.CalcPointsPositions(
            plant_ctx, ee_frame, np.zeros(3), world_frame
        ).flatten()

        # Snapshot.
        if (next_snap_idx < len(snapshot_times)
                and sim_time + 1e-9 >= snapshot_times[next_snap_idx]):
            snapshots.append((sim_time, ee_pos.copy(), q_err.copy(),
                              integral.copy(), tau_demand.copy(),
                              tau_clipped.copy(), saturated.copy()))
            next_snap_idx += 1

        for j in range(n_j):
            csv_w.writerow([
                step, f"{sim_time:.4f}", j,
                f"{q_target_arm[j]:.6f}", f"{q_now[j]:.6f}", f"{q_err[j]:.6f}",
                f"{v_now[j]:.6f}",
                f"{integral[j]:.6f}",
                f"{u_p[j]:.4f}", f"{u_i[j]:.4f}", f"{u_d[j]:.4f}", f"{tau_g_target[j]:.4f}",
                f"{tau_demand[j]:.4f}", f"{tau_clipped[j]:.4f}",
                int(saturated[j]),
            ])

        if step < n_steps:
            simulator.AdvanceTo(sim_time + dt_ctrl)

    csv_fh.close()
    print(f"[probe-9.4.5-A.1] CSV written: {csv_path}")

    # ------------------------------------------------------------------
    # EE trajectory table.
    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("EE trajectory under hold-home-pose")
    print("=" * 78)
    ee0 = snapshots[0][1]
    print(f"  {'t (s)':>5}  {'ee_x':>9}  {'ee_y':>9}  {'ee_z':>9}  "
          f"{'displacement from t=0 (mm)':>28}")
    for (t, ee, qe, ig, td, tc, sat) in snapshots:
        disp_mm = float(np.linalg.norm(ee - ee0)) * 1e3
        print(f"  {t:>5.2f}  {ee[0]:>+9.4f}  {ee[1]:>+9.4f}  {ee[2]:>+9.4f}  "
              f"{disp_mm:>28.2f}")

    # ------------------------------------------------------------------
    # Per-joint summary at t=30.
    # ------------------------------------------------------------------
    final = snapshots[-1]
    t_f, ee_f, qe_f, ig_f, td_f, tc_f, sat_f = final
    print()
    print(f"Per-joint summary at t={t_f:.1f} s:")
    print(f"  {'j':>2}  {'q_err (rad)':>12}  {'integral':>10}  "
          f"{'tau_grav':>9}  {'tau_demand':>11}  {'tau_clipped':>12}  {'sat?':>4}")
    for j in range(n_j):
        print(f"  {j:>2}  {qe_f[j]:>+12.5f}  {ig_f[j]:>+10.5f}  "
              f"{tau_g_target[j]:>+9.2f}  {td_f[j]:>+11.2f}  {tc_f[j]:>+12.2f}  "
              f"{int(sat_f[j]):>4d}")

    # Persistent saturation report — count joints saturated at t=30.
    n_sat_t30 = int(np.sum(sat_f))
    max_demand_t30 = float(np.max(np.abs(td_f)))
    print()
    print(f"At t={t_f:.1f}s: {n_sat_t30}/7 joints saturated; "
          f"max |tau_demand| = {max_demand_t30:.2f} Nm")

    # Headline classification helper.
    disp_t30_mm = float(np.linalg.norm(ee_f - ee0)) * 1e3
    print()
    if disp_t30_mm < 10.0:
        print(f"[CLASSIFY] EE displacement at t=30s = {disp_t30_mm:.2f} mm < 10 mm")
        print("           → arm HOLDS home pose. Effect B = caused by kIK q_target sequence.")
    elif disp_t30_mm > 100.0:
        print(f"[CLASSIFY] EE displacement at t=30s = {disp_t30_mm:.2f} mm > 100 mm")
        print("           → arm FALLS. Effect B = independent of kIK; lives at executor.")
    else:
        print(f"[CLASSIFY] EE displacement at t=30s = {disp_t30_mm:.2f} mm (10-100 mm)")
        print("           → PARTIAL hold. Executor can almost hold but not quite.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
