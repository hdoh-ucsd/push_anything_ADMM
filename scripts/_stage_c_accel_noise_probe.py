"""§7.34 STEP 0b — Feedforward-source NOISE probe (offline).

The reference OSC: yddot_cmd = yddot_des + Kp·error_y + Kd·error_ydot (PD +
feedforward).  The port: a_des = Kp_cart·p_err + Kd_cart·v_err (PD only).
The structural gap is the feedforward `yddot_des`.

In the port's EE-space planner the state is x = [box_q(7), p_ee(3),
box_v(6), v_ee(3)] — there is NO acceleration state slot, so the
candidate a_ff source is the FIRST FINITE DIFFERENCE of the velocity
slot across consecutive knots:

    a_ff[k] ≈ (x_seq[k+1][16:19] − x_seq[k][16:19]) / dt_planner

The planner is non-converged (25/25 every solve, §7.33 (4)).  A
non-converged solve might give a velocity slot that oscillates across
knots — the first difference would then amplify the noise to garbage
acceleration values.  This probe samples the candidate a_ff at a few
constructed states (separated and just-touching) and reports magnitude
+ smoothness so the build can pick a safe a_max clip.
"""
from __future__ import annotations

import os
import numpy as np
import yaml

# Enable always-on row (matching §7.32 setup).
os.environ["LCS_ALWAYS_ON_EE_BOX"]      = "1"
os.environ["LCS_NORMAL_PHI_CLAMP"]      = ""
os.environ["LCS_NORMAL_VELOCITY_LEVEL"] = "0"
os.environ["LCS_NORMAL_COMPLIANCE_K"]   = "0.0"
os.environ["LCS_EXPLICIT_BOX_GND"]      = "4"
os.environ["REF_RECONCILE_APPROACH"]    = "1"
if os.environ.get("LCS_NORMAL_PHI_CLAMP") == "":
    del os.environ["LCS_NORMAL_PHI_CLAMP"]

from sim.env_builder import build_environment
from control.lcs_formulator import LCSFormulator
from control.admm_solver  import C3Solver
from control.task_costs   import QuadraticManipulationCost


# ---------------------------------------------------------------------------
# 19-dim state poses at increasing EE-to-box separation.
def _x0(p_ee_x: float) -> np.ndarray:
    """Build a 19-dim EE-space state with box at origin and EE at (p_ee_x, 0, 0.05).
    box_q = [w=1, x=0, y=0, z=0, px=0, py=0, pz=0.05] — quaternion + position.
    All velocities zero."""
    return np.concatenate([
        np.array([1.0, 0.0, 0.0, 0.0]),  # quat
        np.array([0.0, 0.0, 0.05]),       # box pos
        np.array([p_ee_x, 0.0, 0.05]),    # EE pos
        np.zeros(6),                       # box v (omega + linear)
        np.zeros(3),                       # EE v
    ])


def probe(separation_mm: float, label: str):
    sep_m = separation_mm * 1e-3
    # Box face at +0.050; EE-sphere radius effectively zero in EE-space planner
    # (the LCS formulator's signed-distance already accounts for the pusher
    # radius via Drake). Place EE this far east of the box face.
    p_ee_x = 0.050 + sep_m
    x0 = _x0(p_ee_x)

    with open("config/tasks.yaml") as f:
        cfg = yaml.safe_load(f)
    task_cfg = cfg["tasks"]["pushing"]
    diagram, plant, panda, obj, _, plant_ad, ctx_ad = build_environment(
        task_cfg, time_step=0.001)
    plant_ctx = plant.GetMyContextFromRoot(diagram.CreateDefaultContext())
    plant.SetPositions(plant_ctx, obj,
                       np.concatenate([np.array([1., 0, 0, 0]), np.array([0, 0, 0.05])]))

    f_ = LCSFormulator(plant, mu=task_cfg["friction"], obj_body=plant.GetBodyByName("box_link", obj),
                       plant_ad=plant_ad, context_ad=ctx_ad, box_ground_drag=0.0)
    dt_planner = 0.05
    A, B_ctrl, D, d_const, E, F, H, c_lcs, *_ = f_.linearize_discrete_ee_space(
        plant_ctx, dt_planner, np.zeros(3))

    # Cost matrices (with reconcile flag → proxy off).
    qc = QuadraticManipulationCost(
        plant=plant, ee_frame_name="pusher",
        obj_body=plant.GetBodyByName("box_link", obj),
        cost_cfg=task_cfg["cost"], n_x=19, n_u=3, math_diag=False,
    )
    _r = qc.build_ee_space(
        target_xy=np.array([-0.30, 0.0]),
        plant_ctx=plant_ctx,
        current_q=plant.GetPositions(plant_ctx),
        target_yaw=0.0,
    )
    if len(_r) == 5:
        Q, R, QN, x_ref, u_ref = _r
    else:
        Q, R, QN, x_ref = _r
        u_ref = np.zeros(3)

    N = 20
    lambda_t_dim = (D.shape[1] // 6) * 4
    solver = C3Solver(n_x=19, n_u=3, rho=100.0, mode='c3plus', math_diag=False)
    solver.use_ee_space = True
    # Single full solve at admm_iter=25 (the canonical config).
    u_seq, x_seq, lam_n_first, *_extras = solver.solve(
        A, B_ctrl, D, d_const, E, F, H, c_lcs,
        x0, Q, R, QN, x_ref, u_ref,
        lambda_t_dim=lambda_t_dim, N=N, admm_iter=25, torque_limit=30.0,
    )

    # x_seq shape (N+1, 19).  Inspect the v_ee slot [16:19] across knots.
    v_traj = np.asarray([x_seq[k][16:19] for k in range(N + 1)])
    p_traj = np.asarray([x_seq[k][7:10]  for k in range(N + 1)])

    # Candidate a_ff at knot 1 = (v_traj[2] − v_traj[1]) / dt_planner.
    a_traj = (v_traj[1:] - v_traj[:-1]) / dt_planner  # shape (N, 3)
    a_candidate = a_traj[0]   # a_ff[k=1] — what the wrapper would feed

    # Summary stats across the whole horizon.
    a_max_abs = float(np.max(np.abs(a_traj)))
    a_l2 = np.linalg.norm(a_traj, axis=1)
    a_l2_max = float(a_l2.max())
    a_l2_med = float(np.median(a_l2))
    # Smoothness: variation in a_ff across consecutive knots.  If the
    # planner emits a smooth velocity profile this should be small.
    a_diff = a_traj[1:] - a_traj[:-1]
    a_diff_l2_max = float(np.linalg.norm(a_diff, axis=1).max())

    # Velocity bounds + position bounds for context.
    v_l2_max = float(np.linalg.norm(v_traj, axis=1).max())
    p_min_x = float(p_traj[:, 0].min())

    print(f"\n--- {label} (EE−face sep = {separation_mm:+.1f} mm) ---")
    print(f"  x0 EE = ({x0[7]:+.4f}, {x0[8]:+.4f}, {x0[9]:+.4f})")
    print(f"  knots N+1 = {N + 1}, dt_planner = {dt_planner:.3f}s")
    print(f"  v_l2_max across horizon   = {v_l2_max:+.4f} m/s")
    print(f"  p_traj[:, 0] min          = {p_min_x:+.4f} m  (negative = inside box)")
    print(f"  candidate a_ff[k=1] = ({a_candidate[0]:+.3f}, "
          f"{a_candidate[1]:+.3f}, {a_candidate[2]:+.3f}) m/s²")
    print(f"  |a|_max across horizon    = {a_max_abs:+.3f} m/s² (per-axis)")
    print(f"  |a|_2 max across horizon  = {a_l2_max:+.3f} m/s² (Euclidean)")
    print(f"  |a|_2 median (smoothness) = {a_l2_med:+.3f} m/s²")
    print(f"  |Δa|_2 max knot-to-knot   = {a_diff_l2_max:+.3f} m/s² "
          f"({'SMOOTH' if a_diff_l2_max < 30.0 else 'JITTERY'})")

    return dict(
        label=label,
        a_max_abs=a_max_abs,
        a_l2_max=a_l2_max,
        a_l2_med=a_l2_med,
        a_diff_l2_max=a_diff_l2_max,
        v_l2_max=v_l2_max,
        a_candidate=a_candidate.tolist(),
    )


def main():
    print("=" * 84)
    print("§7.34 STEP 0b — FEEDFORWARD-SOURCE NOISE PROBE")
    print("=" * 84)
    print()
    print("Source: a_ff = (x_seq[k+1][16:19] − x_seq[k][16:19]) / dt_planner")
    print("        (one finite difference of the v_ee state slot)")
    print()
    print("Pass criterion: |a|_2 max < 50 m/s² AND |Δa|_2 knot-to-knot < 30 m/s²")
    print("(loose, just rejects garbage; the build will clip at a_max anyway)")

    results = [
        probe(50.0,  "FAR    (50 mm separation, free-mode approach)"),
        probe(5.0,   "NEAR   ( 5 mm separation, c3-mode just before contact)"),
        probe(0.5,   "TOUCH  (0.5 mm separation, c3-mode at touching)"),
    ]

    print()
    print("=" * 84)
    print("VERDICT")
    print("=" * 84)
    pass_all = True
    a_max_observed = 0.0
    for r in results:
        ok = r["a_l2_max"] < 50.0 and r["a_diff_l2_max"] < 30.0
        pass_all = pass_all and ok
        a_max_observed = max(a_max_observed, r["a_l2_max"])
        print(f"  {r['label']:60s} : "
              f"|a|_2 max = {r['a_l2_max']:+.2f}  |Δa| = {r['a_diff_l2_max']:+.2f}  "
              f"→ {('PASS' if ok else 'GARBAGE')}")
    print()
    print(f"  Largest observed |a|_2 max = {a_max_observed:.2f} m/s²")
    a_clip_suggested = max(20.0, min(50.0, 1.5 * a_max_observed))
    print(f"  Suggested defensive a_max clip = {a_clip_suggested:.1f} m/s² "
          f"(1.5× largest observed, clamped to [20, 50])")
    print()
    print(f"  SOURCE NOISE: "
          f"{('PASS — proceed to STEP 1 with a_max clip' if pass_all else 'GARBAGE — STOP and reconsider')}")
    return 0 if pass_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
