"""Stage C contact-model falsification probe — does enabling LCS_EXPLICIT_BOX_GND
make the LCS match the plant?

Offline, on the captured seed0_full50.npz state. Tests BOTH axes:
  (a) VERTICAL: does the LCS-with-new-oracle predict ≈ 0 vertical motion
      (floor holds), matching Drake's 0?
  (b) HORIZONTAL: does the LCS-with-new-oracle predict horizontal box
      motion that Drake renders under push?

Uses Lemke (control.lcp_solver.solve_lcp) — NOT brute-force enumeration
(intractable at n_lambda ~ 78).
"""
from __future__ import annotations

import os
import sys
import numpy as np
import yaml

from pydrake.systems.analysis import Simulator

from sim.env_builder import build_environment, compute_prepositioned_arm_q
from control.lcp_solver import solve_lcp

DT_PLANNER = 0.05


def build_state(plant, panda_model, object_model, ee_frame, obj_body, task_cfg,
                x0):
    """Build a Drake context with state matching captured x0."""
    context = None  # placeholder — caller passes
    return None


def setup_at_captured_state(diagram, plant, panda_model, object_model,
                            ee_frame, obj_body, task_cfg, x0):
    """Set Drake state so:
       - box at captured x0[0:7]  (quat + position)
       - arm joints positioned so EE is at captured x0[7:10]
       - velocities from x0[10:19]
    Returns (context, plant_ctx).
    """
    context = diagram.CreateDefaultContext()
    plant_ctx = plant.GetMyContextFromRoot(context)

    # First place the box (compute_prepositioned_arm_q uses box position to
    # avoid arm-box collision in its IK seed).
    box_q = np.concatenate([x0[0:4], x0[4:7]])
    plant.SetPositions(plant_ctx, object_model, box_q)

    # Compute arm joints for EE at x0[7:10]; helper performs IK to position
    # the EE on the box side per task_cfg's task-direction, but accepts a
    # seed_arm_q. We use the home config as the seed.
    seed_arm_q = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.785])
    plant.SetPositions(plant_ctx, panda_model, seed_arm_q)

    # The helper does an IK that sometimes ignores the target; for our
    # purposes, we want EE exactly at x0[7:10]. Use Drake IK directly.
    from pydrake.multibody.inverse_kinematics import InverseKinematics
    ik = InverseKinematics(plant, plant_ctx, with_joint_limits=True)
    p_des = x0[7:10]
    p_tol = 1e-4
    ik.AddPositionConstraint(
        ee_frame, np.zeros(3), plant.world_frame(),
        p_des - p_tol, p_des + p_tol,
    )
    q0 = plant.GetPositions(plant_ctx)
    # use arm joints in q0 for the IK seed
    ik.get_mutable_prog().SetInitialGuess(ik.q(), q0)
    from pydrake.solvers import Solve
    result = Solve(ik.prog())
    if result.is_success():
        q_full = result.GetSolution(ik.q())
        plant.SetPositions(plant_ctx, q_full)
    else:
        print("[probe] IK to place EE at x0[7:10] failed; using seed config")

    plant.SetVelocities(plant_ctx, panda_model, np.zeros(7))
    box_v = np.concatenate([x0[10:13], x0[13:16]])
    plant.SetVelocities(plant_ctx, object_model, box_v)

    # Verify
    ee_pos_actual = plant.CalcPointsPositions(
        plant_ctx, ee_frame, np.zeros(3), plant.world_frame()).flatten()
    print(f"  EE pos achieved: {ee_pos_actual}    (target: {x0[7:10]})")
    print(f"  |Δ EE pos|: {np.linalg.norm(ee_pos_actual - x0[7:10])*1000:.4f} mm")
    return context, plant_ctx


def run_one_count(N_explicit: int):
    """Run the falsification at LCS_EXPLICIT_BOX_GND=N_explicit and return
    a dict of the read."""
    print(f"\n{'='*64}")
    print(f"LCS_EXPLICIT_BOX_GND = {N_explicit}")
    print(f"{'='*64}")

    os.environ["LCS_EXPLICIT_BOX_GND"] = str(N_explicit)
    # Force-reload the formulator module so the env var takes effect
    import importlib
    import control.lcs_formulator
    importlib.reload(control.lcs_formulator)
    from control.lcs_formulator import LCSFormulator

    with open("config/tasks.yaml") as f:
        cfg = yaml.safe_load(f)
    task_cfg = cfg["tasks"]["pushing"]

    diagram, plant, panda_model, object_model, _, plant_ad, ctx_ad = \
        build_environment(task_cfg, time_step=0.001)

    obj_body = plant.GetBodyByName("box_link", object_model)
    ee_frame = plant.GetFrameByName("pusher")

    d = np.load("stage_c/admm_dump/seed0_full50.npz", allow_pickle=True)
    x0 = d["x0"]

    context, plant_ctx = setup_at_captured_state(
        diagram, plant, panda_model, object_model, ee_frame, obj_body,
        task_cfg, x0)

    # Build LCS formulator and extract LCS at the captured state
    mu = task_cfg["friction"]
    formulator = LCSFormulator(plant, mu=mu, obj_body=obj_body,
                               plant_ad=plant_ad, context_ad=ctx_ad)
    print(f"  formulator.lcs_explicit_box_ground_contacts = {formulator.lcs_explicit_box_ground_contacts}")

    # Use EE-space variant to match captured n_x=19 layout
    # ([box_q (7), p_ee (3), box_v (6), v_ee (3)]) and n_u=3.
    u_lin = np.zeros(3)
    A, B_ctrl, D, d_const, E, F, H, c_lcs, J_n, J_t, phi_vec, mu_vec = \
        formulator.linearize_discrete_ee_space(plant_ctx, DT_PLANNER, u_lin)
    n_x = A.shape[0]
    n_lambda = D.shape[1]
    print(f"  Re-extracted LCS: n_x={n_x}, n_lambda={n_lambda}")
    print(f"  contacts found: {len(getattr(formulator, '_last_contact_info', []))}")
    if hasattr(formulator, '_last_contact_info'):
        for i, info in enumerate(formulator._last_contact_info or []):
            tag = info.get('tag', '?')
            dist = info.get('distance', None)
            print(f"    pair {i}: tag={tag}  dist={dist}")

    # Use the actual extracted x0 (matches the live machinery's slice)
    # — but for consistency with the original npz, use d["x0"]. Actually
    # the live machinery would have its own x0; the LCS dynamics matrices
    # depend on the state we linearized at. Use the same captured x0.
    x0_use = x0[:n_x] if n_x <= len(x0) else None
    if x0_use is None or n_x != len(x0):
        # If the new LCS has a different n_x (because EE space is different
        # or extra slots), we need a fresh x0 vector. The live planner's
        # x0 layout is the same in EE-space; the contact admission doesn't
        # change n_x. n_x should stay 19.
        print(f"  WARNING n_x={n_x} != captured n_x={len(x0)}")
        return None

    # Solve LCP for λ at u=0:  0 ≤ λ ⊥ (F·λ + E·x0 + H·0 + c) ≥ 0
    # M = F, q = E·x0 + c
    q_lcp = E @ x0_use + c_lcs
    print(f"\n  LCP M shape: F = {F.shape}")
    print(f"  LCP q stats : min={q_lcp.min():.4e} max={q_lcp.max():.4e}")
    try:
        lam_new, lcp_res = solve_lcp(F, q_lcp)
        print(f"  Lemke residual: {lcp_res:.3e}")
    except Exception as e:
        print(f"  Lemke FAILED: {e}")
        return {"count": N_explicit, "n_lambda": n_lambda,
                "feasible": False, "error": str(e)}

    # Verify feasibility
    w_check = F @ lam_new + q_lcp
    lam_pos = bool(np.all(lam_new >= -1e-6))
    w_pos = bool(np.all(w_check >= -1e-6))
    compl = float(np.max(np.abs(lam_new * w_check)))
    print(f"  Feasibility: λ≥0 {lam_pos}, w≥0 {w_pos}, max|λw|={compl:.4e}")

    print(f"\n  λ_n_max (max over normal slots): {float(lam_new.max()):.4f}")
    # Identify normal slots — usually slot 1 + every k after gamma block
    # but for the BOX-GND additions we don't know exact layout; just report.
    print(f"  λ nonzero indices: {np.where(lam_new > 1e-6)[0].tolist()}")

    # LCS-PREDICTED next state under new oracle
    # EE-space: n_u = 3 (EE force), not plant.num_actuators (=7 panda)
    x_next_LCS = A @ x0_use + B_ctrl @ np.zeros(B_ctrl.shape[1]) \
                 + D @ lam_new + d_const
    lcs_delta = x_next_LCS[4:7] - x0_use[4:7]
    lcs_v = x_next_LCS[13:16]

    print(f"\n  LCS-PREDICTED Δ box xyz: ({lcs_delta[0]*1000:+.3f}, "
          f"{lcs_delta[1]*1000:+.3f}, {lcs_delta[2]*1000:+.3f}) mm")
    print(f"  LCS-PREDICTED box vz   : {lcs_v[2]:+.5f} m/s")

    # Drake-rendered (we already computed this: 0mm) — re-confirm
    sim = Simulator(diagram, context)
    sim.Initialize()
    sim.AdvanceTo(DT_PLANNER)
    plant_ctx_after = plant.GetMyContextFromRoot(context)
    xyz_after = plant.EvalBodyPoseInWorld(plant_ctx_after, obj_body).translation()
    xyz_before = np.array(x0[4:7])
    drake_delta = xyz_after - xyz_before
    print(f"  Drake-RENDERED Δ box xyz: ({drake_delta[0]*1000:+.3f}, "
          f"{drake_delta[1]*1000:+.3f}, {drake_delta[2]*1000:+.3f}) mm")

    # Both-axes gate
    vertical_close = abs(lcs_delta[2] - drake_delta[2]) * 1000 < 1.0  # within 1 mm
    horiz_lcs = np.array([lcs_delta[0], lcs_delta[1]])
    horiz_drake = np.array([drake_delta[0], drake_delta[1]])
    horiz_close = np.linalg.norm(horiz_lcs - horiz_drake) * 1000 < 1.0
    print(f"\n  (a) VERTICAL  close? {vertical_close}  "
          f"(|LCS - Drake| in z = {abs(lcs_delta[2] - drake_delta[2])*1000:.3f} mm)")
    print(f"  (b) HORIZONTAL close? {horiz_close}  "
          f"(|LCS - Drake| in xy = {np.linalg.norm(horiz_lcs - horiz_drake)*1000:.3f} mm)")

    return {
        "count": N_explicit,
        "n_lambda": n_lambda,
        "feasible": True,
        "lcs_delta_mm": (lcs_delta * 1000).tolist(),
        "drake_delta_mm": (drake_delta * 1000).tolist(),
        "lcs_v_z": float(lcs_v[2]),
        "lcp_res": float(lcp_res),
        "compl": float(compl),
        "vertical_close": vertical_close,
        "horizontal_close": horiz_close,
    }


if __name__ == "__main__":
    print("STAGE C CONTACT-MODEL FALSIFICATION PROBE")
    print(f"Captured instance: stage_c/admm_dump/seed0_full50.npz")
    print(f"Planner Δt: {DT_PLANNER} s")

    results = []
    for N in [4, 12]:
        r = run_one_count(N)
        if r is not None:
            results.append(r)

    print(f"\n{'='*64}")
    print("SUMMARY")
    print(f"{'='*64}")
    for r in results:
        print(f"\nLCS_EXPLICIT_BOX_GND = {r['count']}  →  n_lambda = {r['n_lambda']}")
        if not r.get("feasible", False):
            print(f"  Lemke could not find feasible λ — error: {r.get('error')}")
            continue
        print(f"  LCS Δ box xyz  : ({r['lcs_delta_mm'][0]:+.3f}, "
              f"{r['lcs_delta_mm'][1]:+.3f}, {r['lcs_delta_mm'][2]:+.3f}) mm")
        print(f"  Drake Δ box xyz: ({r['drake_delta_mm'][0]:+.3f}, "
              f"{r['drake_delta_mm'][1]:+.3f}, {r['drake_delta_mm'][2]:+.3f}) mm")
        print(f"  vertical close: {r['vertical_close']}")
        print(f"  horizontal close: {r['horizontal_close']}")
