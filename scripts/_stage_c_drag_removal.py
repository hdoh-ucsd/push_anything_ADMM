"""Stage C drag-removal probe (offline, single instance).

Question: is `box_ground_drag` a redundant band-aid the count=4 fix
superseded? The leading hypothesis on the §7.11 horizontal gap: the
LCS A-matrix multiplier (1 - c·dt) = 0.5× halves box translational
velocity each planner step. With explicit floor contacts (count=4)
now balancing gravity and providing friction, the drag may be
double-counting ground interaction. The reference has NO drag term
(anitescu + explicit contacts).

Method
------
Construct the SAME contact state as `_stage_c_horizontal_consistency.py`
and reuse its IK helper. Build TWO temp formulators (drag=10.0 baseline
rerun + drag=0.0 test) under LCS_EXPLICIT_BOX_GND=4, extract LCS, solve
Lemke for the oracle, step the LCS once, and compare to a single Drake
forward step from the same state.

Reports (alongside the existing baseline=10 / new test=0 numbers):
  - new horizontal under-prediction factor (was 3.7×)
  - does vertical still hold with drag=0
  - which OUTCOME bucket the result lands in
    (WHOLE / HALF / REDHERRING / VERTICAL-BREAKS)

NO live-port change.  This script reads HEAD's default (10.0) and runs
both values offline; the live default is untouched.

Run: `python scripts/_stage_c_drag_removal.py` from repo root.
"""
from __future__ import annotations

import os
import numpy as np
import yaml

from pydrake.systems.analysis import Simulator
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.solvers import Solve

from sim.env_builder import build_environment
from control.lcp_solver import solve_lcp

DT_PLANNER = 0.05
BOX_HALF = 0.05
EE_RADIUS = 0.025

BOX_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
BOX_POS  = np.array([0.0, 0.0, 0.05])
EE_PEN_M = 0.0001
EE_POS   = np.array([BOX_HALF + EE_RADIUS - EE_PEN_M, 0.0, 0.05])
EE_VEL_X = -0.05  # westward push


def setup_contact_state(diagram, plant, panda_model, object_model, ee_frame):
    """Same construction as _stage_c_horizontal_consistency.setup_contact_state."""
    from pydrake.multibody.tree import JacobianWrtVariable

    context = diagram.CreateDefaultContext()
    plant_ctx = plant.GetMyContextFromRoot(context)

    box_q = np.concatenate([BOX_QUAT, BOX_POS])
    plant.SetPositions(plant_ctx, object_model, box_q)

    seed = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.785])
    plant.SetPositions(plant_ctx, panda_model, seed)

    ik = InverseKinematics(plant, plant_ctx, with_joint_limits=True)
    p_tol = 1e-4
    ik.AddPositionConstraint(
        ee_frame, np.zeros(3), plant.world_frame(),
        EE_POS - p_tol, EE_POS + p_tol,
    )
    q0 = plant.GetPositions(plant_ctx)
    ik.get_mutable_prog().SetInitialGuess(ik.q(), q0)
    result = Solve(ik.prog())
    if not result.is_success():
        print("[probe] IK FAILED")
        return None, None
    q_full = result.GetSolution(ik.q())
    plant.SetPositions(plant_ctx, q_full)

    plant.SetVelocities(plant_ctx, object_model, np.zeros(6))

    J_ee = plant.CalcJacobianTranslationalVelocity(
        plant_ctx, JacobianWrtVariable.kV, ee_frame, np.zeros(3),
        plant.world_frame(), plant.world_frame(),
    )
    panda_v_start = plant.GetJointByName("panda_joint1").velocity_start()
    n_arm = 7
    J_arm = J_ee[:, panda_v_start: panda_v_start + n_arm]
    q_arm_dot, *_ = np.linalg.lstsq(
        J_arm, np.array([EE_VEL_X, 0.0, 0.0]), rcond=None)
    plant.SetVelocities(plant_ctx, panda_model, q_arm_dot)

    ee_pos = plant.CalcPointsPositions(
        plant_ctx, ee_frame, np.zeros(3), plant.world_frame()).flatten()
    ee_v = plant.CalcJacobianTranslationalVelocity(
        plant_ctx, JacobianWrtVariable.kV, ee_frame, np.zeros(3),
        plant.world_frame(), plant.world_frame(),
    ) @ plant.GetVelocities(plant_ctx)
    print(f"  EE pos achieved: {ee_pos}  Δ {np.linalg.norm(ee_pos - EE_POS)*1000:.3f} mm")
    print(f"  EE vel achieved: {ee_v}")
    return context, plant_ctx


def run_lcs_test(label, box_ground_drag_value, plant, plant_ctx,
                 obj_body, ee_frame, plant_ad, ctx_ad, mu):
    """One LCS extraction + oracle + LCS step.  Returns the comparison
    quantities versus Drake (Drake step is done once by the caller).
    """
    from control.lcs_formulator import LCSFormulator
    formulator = LCSFormulator(plant, mu=mu, obj_body=obj_body,
                               plant_ad=plant_ad, context_ad=ctx_ad,
                               box_ground_drag=box_ground_drag_value)
    A, B_ctrl, D, d_const, E, F, H, c_lcs, J_n, J_t, phi_vec, mu_vec = \
        formulator.linearize_discrete_ee_space(plant_ctx, DT_PLANNER, np.zeros(3))
    n_x = A.shape[0]
    n_lambda = D.shape[1]
    print(f"\n[{label}] box_ground_drag = {box_ground_drag_value}")
    print(f"  Re-extracted LCS: n_x={n_x}, n_lambda={n_lambda}")
    contacts = getattr(formulator, '_last_contact_info', [])
    print(f"  contacts: {len(contacts)}    "
          f"tags: {[c.get('tag') for c in contacts]}")

    # x0 in EE-space layout (must match what live wrapper builds)
    box_q = np.concatenate([BOX_QUAT, BOX_POS])
    ee_pos_actual = plant.CalcPointsPositions(
        plant_ctx, ee_frame, np.zeros(3), plant.world_frame()).flatten()
    box_v_zero = np.zeros(6)
    ee_v_actual = np.array([EE_VEL_X, 0.0, 0.0])
    x0 = np.concatenate([box_q, ee_pos_actual, box_v_zero, ee_v_actual])
    assert x0.shape[0] == n_x

    # Lemke oracle
    q_lcp = E @ x0 + c_lcs
    lam, lcp_res = solve_lcp(F, q_lcp)
    w = F @ lam + q_lcp
    compl = float(np.max(np.abs(lam * w)))
    print(f"  Lemke residual: {lcp_res:.3e}    "
          f"max|λw|: {compl:.3e}    λ_max: {float(lam.max()):.4f}")
    print(f"  λ nonzero idx: {np.where(lam > 1e-6)[0].tolist()}")
    print(f"  λ nonzero val: {[f'{v:.4f}' for v in lam[lam > 1e-6]]}")

    # LCS step
    x_next = A @ x0 + B_ctrl @ np.zeros(3) + D @ lam + d_const
    lcs_dpos = x_next[4:7] - x0[4:7]
    lcs_dvel = x_next[13:16]  # box translational velocity (post-step)
    print(f"  LCS Δ box xyz : ({lcs_dpos[0]*1000:+.3f}, "
          f"{lcs_dpos[1]*1000:+.3f}, {lcs_dpos[2]*1000:+.3f}) mm")
    print(f"  LCS Δ box vel : ({lcs_dvel[0]:+.5f}, {lcs_dvel[1]:+.5f}, "
          f"{lcs_dvel[2]:+.5f}) m/s")
    return lcs_dpos, lcs_dvel, n_lambda, lcp_res


def main() -> int:
    print("=" * 72)
    print("STAGE C DRAG-REMOVAL PROBE — is box_ground_drag now redundant?")
    print("=" * 72)
    print(f"  state: box rest at (0,0,0.05), EE east face penetrating "
          f"{EE_PEN_M*1000:.2f}mm, EE vel ({EE_VEL_X},0,0), u=0")
    print(f"  Δt = {DT_PLANNER}s   LCS_EXPLICIT_BOX_GND = 4")
    print(f"  Test: box_ground_drag = 10.0 (baseline) vs 0.0 (drag-removed)")

    os.environ["LCS_EXPLICIT_BOX_GND"] = "4"
    # Force reload so the env knob takes effect on this run
    import importlib, control.lcs_formulator
    importlib.reload(control.lcs_formulator)

    with open("config/tasks.yaml") as f:
        cfg = yaml.safe_load(f)
    task_cfg = cfg["tasks"]["pushing"]
    mu = task_cfg["friction"]

    # Drake-side single step (the ground truth — runs ONCE; LCS tests
    # re-use the same constructed state from the offline plant)
    diagram, plant, panda_model, object_model, _, plant_ad, ctx_ad = \
        build_environment(task_cfg, time_step=0.001)
    obj_body = plant.GetBodyByName("box_link", object_model)
    ee_frame = plant.GetFrameByName("pusher")

    context, plant_ctx = setup_contact_state(
        diagram, plant, panda_model, object_model, ee_frame)
    if context is None:
        return 1

    # 1. BASELINE rerun (drag = 10.0)
    lcs_dpos_base, lcs_dvel_base, n_lambda_base, res_base = run_lcs_test(
        "BASELINE", 10.0, plant, plant_ctx, obj_body, ee_frame,
        plant_ad, ctx_ad, mu)

    # 2. DRAG-REMOVED (drag = 0.0)
    lcs_dpos_test, lcs_dvel_test, n_lambda_test, res_test = run_lcs_test(
        "DRAG-OFF", 0.0, plant, plant_ctx, obj_body, ee_frame,
        plant_ad, ctx_ad, mu)

    # 3. Drake forward step from the same state
    sim = Simulator(diagram, context)
    sim.Initialize()
    sim.AdvanceTo(DT_PLANNER)
    plant_ctx_after = plant.GetMyContextFromRoot(context)
    xyz_after = plant.EvalBodyPoseInWorld(plant_ctx_after, obj_body).translation()
    drake_dpos = xyz_after - BOX_POS
    drake_dvel = plant.EvalBodySpatialVelocityInWorld(
        plant_ctx_after, obj_body).translational()

    print()
    print("=" * 72)
    print("COMPARISON — both LCS results vs Drake")
    print("=" * 72)
    print(f"  Δ box xyz         BASELINE (drag=10): "
          f"({lcs_dpos_base[0]*1000:+.3f}, {lcs_dpos_base[1]*1000:+.3f}, "
          f"{lcs_dpos_base[2]*1000:+.3f}) mm")
    print(f"                    DRAG-OFF (drag=0):  "
          f"({lcs_dpos_test[0]*1000:+.3f}, {lcs_dpos_test[1]*1000:+.3f}, "
          f"{lcs_dpos_test[2]*1000:+.3f}) mm")
    print(f"                    DRAKE:              "
          f"({drake_dpos[0]*1000:+.3f}, {drake_dpos[1]*1000:+.3f}, "
          f"{drake_dpos[2]*1000:+.3f}) mm")

    # Push-axis (x) under-prediction factor: |Drake_dx / LCS_dx|
    def px_factor(lcs):
        if abs(lcs[0]) < 1e-9:
            return float("inf")
        return abs(drake_dpos[0] / lcs[0])
    factor_base = px_factor(lcs_dpos_base)
    factor_test = px_factor(lcs_dpos_test)
    print()
    print(f"  Push-axis (x) under-prediction factor (|Drake/LCS|):")
    print(f"    BASELINE (drag=10) : {factor_base:.2f}×  (§7.11 reported 3.7×)")
    print(f"    DRAG-OFF (drag=0)  : {factor_test:.2f}×")

    horiz_xy_gap_base = np.linalg.norm(
        lcs_dpos_base[:2] - drake_dpos[:2]) * 1000
    horiz_xy_gap_test = np.linalg.norm(
        lcs_dpos_test[:2] - drake_dpos[:2]) * 1000
    vert_gap_base = abs(lcs_dpos_base[2] - drake_dpos[2]) * 1000
    vert_gap_test = abs(lcs_dpos_test[2] - drake_dpos[2]) * 1000
    print()
    print(f"  Horizontal |LCS - Drake| xy:")
    print(f"    BASELINE : {horiz_xy_gap_base:.4f} mm   "
          f"(< 1.0 mm bar: {horiz_xy_gap_base < 1.0})")
    print(f"    DRAG-OFF : {horiz_xy_gap_test:.4f} mm   "
          f"(< 1.0 mm bar: {horiz_xy_gap_test < 1.0})")
    print(f"  Vertical   |LCS - Drake| z:")
    print(f"    BASELINE : {vert_gap_base:.4f} mm   "
          f"(< 1.0 mm bar: {vert_gap_base < 1.0})")
    print(f"    DRAG-OFF : {vert_gap_test:.4f} mm   "
          f"(< 1.0 mm bar: {vert_gap_test < 1.0})")

    print()
    print("=" * 72)
    print("STRUCTURAL READ — was box_ground_drag a redundant band-aid?")
    print("=" * 72)
    vert_holds_drag_off = vert_gap_test < 1.0
    horiz_close_drag_off = horiz_xy_gap_test < 1.0
    print(f"  Vertical with drag=0 (count=4 floor alone): "
          f"{'HOLDS' if vert_holds_drag_off else 'BREAKS'}")
    print(f"    → drag {'IS' if not vert_holds_drag_off else 'IS NOT'} "
          f"doing real support work")
    print(f"  Horizontal closes with drag=0: {horiz_close_drag_off}")

    # Three-outcome routing
    print()
    print("=" * 72)
    print("OUTCOME ROUTING (three-outcome pre-registration)")
    print("=" * 72)
    if not vert_holds_drag_off:
        outcome = "VERTICAL-BREAKS"
        print(f"  ► {outcome} — drag was doing real support work; count=4 "
              f"contacts do NOT fully replace it.")
        print(f"    The gap and support are entangled. Re-frame "
              f"(count=4 alone insufficient).")
        next_step = "re-frame: count=4 floor support is not equivalent to drag"
    elif factor_test < 1.5:
        outcome = "DRAG-IS-IT (OUTCOME-WHOLE)"
        print(f"  ► {outcome} — factor closes to ~1× (drag was dominant + "
              f"redundant; count=4 floor superseded it).")
        print(f"    Structural finding: drag was a band-aid the explicit "
              f"contacts SUPERSEDED.")
        print(f"    Move TOWARD the reference (which has NO drag term).")
        next_step = ("MODEL-FIXED-REAL (pending sanity at another contact "
                     "state); re-promote convergence; live verify count=4 + drag=0")
    elif factor_test < 2.5:
        outcome = "DRAG-IS-HALF (OUTCOME-HALF)"
        print(f"  ► {outcome} — factor closes to ~1.85× (drag was ~2×, "
              f"as predicted by 0.5× single-step arithmetic).")
        print(f"    SECOND contributor remains on the residual.")
        next_step = ("Δt sub-stepping test: re-extract at Δt=0.005 vs 0.05, "
                     "see whether sub-stepping closes the residual")
    else:
        outcome = "DRAG-NOT-IT (OUTCOME-REDHERRING)"
        print(f"  ► {outcome} — factor barely moves (0.5× multiplier was a "
              f"red herring; drag is NOT the operative term).")
        print(f"    The gap lives in integration (Δt) or friction "
              f"(Stewart-Trinkle vs compliant).")
        next_step = "Δt test next; friction audit held as deepest/last"

    print()
    print(f"  NEXT STEP (separate block): {next_step}")
    print()
    print(f"  CONVERGENCE: {'RE-PROMOTED' if outcome.startswith('DRAG-IS-IT') else 'HELD'}")
    print(f"  ANTI-STALE: {outcome}; do NOT mark MODEL-FIXED unless DRAG-IS-IT.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
