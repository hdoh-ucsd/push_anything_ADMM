"""Stage C Δt × drag 2×2 factorial probe (offline).

Question (§7.12 routed): is the 3.73× horizontal under-prediction the LCS's
discrete Euler step over Δt=0.05 missing contact-transition nonlinearities
that Drake's 1-ms substeps capture?

Why factorial: sub-stepping the LCS passes through v_box≠0 states where
`box_ground_drag` (which multiplies v_box) RE-ENTERS. A single Δt=0.005
test would conflate Δt with drag-on-a-moving-box. The 2×2 separates them
AND closes the §7.12 drag caveat ("does drag matter once moving?").

Cells:
  A  Δt = 0.05,  drag = 10.0   → KNOWN baseline -0.451 mm (single step)
  B  Δt = 0.05,  drag = 0.0    → KNOWN baseline -0.451 mm (single step)
  C  Δt = 0.005, drag = 10.0   → NEW: 10 sub-steps, RE-EXTRACTED each step
  D  Δt = 0.005, drag = 0.0    → NEW: 10 sub-steps, RE-EXTRACTED each step

Faithfulness: cells C/D RE-EXTRACT the LCS at each evolving sub-step state
(re-IK arm to LCS-predicted EE pos, set box pose/vel from LCS state, then
re-linearize). This is what a finer-Δt planner would actually do — re-
linearize as the state evolves. A FIXED-LCS variant is run alongside as a
reference (to make the re-extract-vs-fixed difference empirical, not
declared).

State layout (n_x=19, EE-space):
    [box_quat(4), box_pos(3), EE_pos(3), box_ω(3), box_v(3), EE_v(3)]

Common Drake reference step: a single 0.05-s Drake step from the SAME
initial state → ground-truth Δbox.
"""
from __future__ import annotations

import os
import importlib
import numpy as np
import yaml

from pydrake.systems.analysis import Simulator
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.multibody.tree import JacobianWrtVariable
from pydrake.solvers import Solve

from sim.env_builder import build_environment
from control.lcp_solver import solve_lcp

DT_BIG  = 0.05
DT_SMALL = 0.005
N_SUB = int(round(DT_BIG / DT_SMALL))  # 10

BOX_HALF  = 0.05
EE_RADIUS = 0.025
BOX_QUAT  = np.array([1.0, 0.0, 0.0, 0.0])
BOX_POS   = np.array([0.0, 0.0, 0.05])
EE_PEN_M  = 0.0001
EE_POS    = np.array([BOX_HALF + EE_RADIUS - EE_PEN_M, 0.0, 0.05])
EE_VEL_X  = -0.05


# --------------- contact-state setup (mirror of drag-removal probe) --------

def setup_contact_state(diagram, plant, panda_model, object_model, ee_frame):
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
    ik.get_mutable_prog().SetInitialGuess(ik.q(), plant.GetPositions(plant_ctx))
    result = Solve(ik.prog())
    if not result.is_success():
        print("[probe] IK FAILED")
        return None, None, None
    q_full = result.GetSolution(ik.q())
    plant.SetPositions(plant_ctx, q_full)
    q_arm_init = plant.GetPositions(plant_ctx, panda_model).copy()

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

    ee_pos_actual = plant.CalcPointsPositions(
        plant_ctx, ee_frame, np.zeros(3), plant.world_frame()).flatten()
    print(f"  setup: EE pos {ee_pos_actual} Δ {np.linalg.norm(ee_pos_actual - EE_POS)*1000:.3f} mm")
    return context, plant_ctx, q_arm_init


def write_state_to_plant(plant, plant_ctx, x, panda_model, object_model,
                         ee_frame, q_arm_warm):
    """Push EE-space state x (n_x=19) back into plant_ctx.
    Returns the new arm config (for warm-starting the next sub-step's IK)."""
    # 1. Box pose + velocity (direct write)
    plant.SetPositions(plant_ctx, object_model, x[0:7])
    plant.SetVelocities(plant_ctx, object_model, x[10:16])

    # 2. EE position via IK
    ee_pos_target = x[7:10]
    # Seed: previous arm config
    plant.SetPositions(plant_ctx, panda_model, q_arm_warm)
    ik = InverseKinematics(plant, plant_ctx, with_joint_limits=True)
    p_tol = 1e-5
    ik.AddPositionConstraint(
        ee_frame, np.zeros(3), plant.world_frame(),
        ee_pos_target - p_tol, ee_pos_target + p_tol,
    )
    ik.get_mutable_prog().SetInitialGuess(ik.q(), plant.GetPositions(plant_ctx))
    result = Solve(ik.prog())
    if not result.is_success():
        return None
    q_full = result.GetSolution(ik.q())
    plant.SetPositions(plant_ctx, q_full)
    q_arm_new = plant.GetPositions(plant_ctx, panda_model).copy()

    # 3. EE velocity via least-norm arm joint velocity
    J_ee = plant.CalcJacobianTranslationalVelocity(
        plant_ctx, JacobianWrtVariable.kV, ee_frame, np.zeros(3),
        plant.world_frame(), plant.world_frame(),
    )
    panda_v_start = plant.GetJointByName("panda_joint1").velocity_start()
    J_arm = J_ee[:, panda_v_start: panda_v_start + 7]
    q_arm_dot, *_ = np.linalg.lstsq(J_arm, x[16:19], rcond=None)
    plant.SetVelocities(plant_ctx, panda_model, q_arm_dot)
    return q_arm_new


def build_x0(plant, plant_ctx, ee_frame):
    ee_pos_actual = plant.CalcPointsPositions(
        plant_ctx, ee_frame, np.zeros(3), plant.world_frame()).flatten()
    return np.concatenate([
        BOX_QUAT, BOX_POS,            # 0..7
        ee_pos_actual,                # 7..10
        np.zeros(3),                  # box ω
        np.zeros(3),                  # box v
        np.array([EE_VEL_X, 0., 0.]), # EE v
    ])


def lcs_single_step(box_drag, plant, plant_ctx, ee_frame, obj_body,
                    plant_ad, ctx_ad, mu, dt):
    from control.lcs_formulator import LCSFormulator
    f = LCSFormulator(plant, mu=mu, obj_body=obj_body,
                      plant_ad=plant_ad, context_ad=ctx_ad,
                      box_ground_drag=box_drag)
    A, B_ctrl, D, d_const, E, F, H, c_lcs, *_ = \
        f.linearize_discrete_ee_space(plant_ctx, dt, np.zeros(3))
    x0 = build_x0(plant, plant_ctx, ee_frame)
    q_lcp = E @ x0 + c_lcs
    lam, res = solve_lcp(F, q_lcp)
    x_next = A @ x0 + B_ctrl @ np.zeros(3) + D @ lam + d_const
    return x_next - x0, lam, res, (A, B_ctrl, D, d_const, E, F, c_lcs)


def lcs_substep_refresh(box_drag, plant, plant_ctx, ee_frame, obj_body,
                        plant_ad, ctx_ad, mu, dt_sub, n_sub,
                        panda_model, object_model, q_arm_init):
    """Sub-step the LCS forward with FULL re-extraction at each step.
    Returns the cumulative box delta + per-step Δλ summary."""
    from control.lcs_formulator import LCSFormulator
    x_curr = build_x0(plant, plant_ctx, ee_frame)
    box_pos_0 = x_curr[4:7].copy()
    q_arm = q_arm_init.copy()
    lam_history = []
    fail_steps = 0
    for k in range(n_sub):
        f = LCSFormulator(plant, mu=mu, obj_body=obj_body,
                          plant_ad=plant_ad, context_ad=ctx_ad,
                          box_ground_drag=box_drag)
        A, B_ctrl, D, d_const, E, F, H, c_lcs, *_ = \
            f.linearize_discrete_ee_space(plant_ctx, dt_sub, np.zeros(3))
        q_lcp = E @ x_curr + c_lcs
        lam, res = solve_lcp(F, q_lcp)
        if res > 1e-3:
            fail_steps += 1
        lam_history.append((float(lam.max()), float(np.sum(lam > 1e-6))))
        x_next = A @ x_curr + B_ctrl @ np.zeros(3) + D @ lam + d_const
        if k < n_sub - 1:
            new_q_arm = write_state_to_plant(
                plant, plant_ctx, x_next, panda_model, object_model,
                ee_frame, q_arm)
            if new_q_arm is None:
                print(f"    [warn] sub-step {k+1}/{n_sub}: IK failed; "
                      f"falling back to fixed-LCS for the remainder")
                return _fallback_fixed(
                    A, B_ctrl, D, d_const, E, F, c_lcs,
                    x_curr, x_next, n_sub - k - 1, lam_history, box_pos_0)
            q_arm = new_q_arm
        x_curr = x_next
    return x_curr[4:7] - box_pos_0, lam_history, fail_steps


def _fallback_fixed(A, B_ctrl, D, d_const, E, F, c_lcs,
                    x_curr, x_next, remaining, lam_history, box_pos_0):
    x_curr = x_next
    for _ in range(remaining):
        q_lcp = E @ x_curr + c_lcs
        lam, _ = solve_lcp(F, q_lcp)
        lam_history.append((float(lam.max()), float(np.sum(lam > 1e-6))))
        x_curr = A @ x_curr + B_ctrl @ np.zeros(3) + D @ lam + d_const
    return x_curr[4:7] - box_pos_0, lam_history, -1


def lcs_substep_fixed(box_drag, plant, plant_ctx, ee_frame, obj_body,
                      plant_ad, ctx_ad, mu, dt_sub, n_sub):
    """Sub-step with FIXED A,B,D,d,E,F,c (no re-extraction).  For comparison."""
    from control.lcs_formulator import LCSFormulator
    f = LCSFormulator(plant, mu=mu, obj_body=obj_body,
                      plant_ad=plant_ad, context_ad=ctx_ad,
                      box_ground_drag=box_drag)
    A, B_ctrl, D, d_const, E, F, H, c_lcs, *_ = \
        f.linearize_discrete_ee_space(plant_ctx, dt_sub, np.zeros(3))
    x_curr = build_x0(plant, plant_ctx, ee_frame)
    box_pos_0 = x_curr[4:7].copy()
    for k in range(n_sub):
        q_lcp = E @ x_curr + c_lcs
        lam, _ = solve_lcp(F, q_lcp)
        x_curr = A @ x_curr + B_ctrl @ np.zeros(3) + D @ lam + d_const
    return x_curr[4:7] - box_pos_0


def main() -> int:
    print("=" * 72)
    print("STAGE C  Δt × drag  2×2 FACTORIAL  (offline, count=4)")
    print("=" * 72)
    print(f"  state: box rest (0,0,0.05), EE east face pen {EE_PEN_M*1000:.2f}mm,"
          f" EE vel ({EE_VEL_X},0,0), u=0")
    print(f"  Δt cells: {{{DT_BIG}, {DT_SMALL}}}    drag cells: {{10, 0}}")
    print(f"  Δt=0.005 cells: {N_SUB} sub-steps, FAITHFUL re-extraction each step")

    os.environ["LCS_EXPLICIT_BOX_GND"] = "4"
    import control.lcs_formulator
    importlib.reload(control.lcs_formulator)

    with open("config/tasks.yaml") as f:
        cfg = yaml.safe_load(f)
    task_cfg = cfg["tasks"]["pushing"]
    mu = task_cfg["friction"]

    # Drake ground truth (run ONCE; we rebuild the diagram per LCS test
    # because the diagram is mutated by the sub-step state-write helper).
    diagram_g, plant_g, panda_g, object_g, _, plant_ad_g, ctx_ad_g = \
        build_environment(task_cfg, time_step=0.001)
    obj_body_g = plant_g.GetBodyByName("box_link", object_g)
    ee_frame_g = plant_g.GetFrameByName("pusher")
    ctx_g, pctx_g, _ = setup_contact_state(
        diagram_g, plant_g, panda_g, object_g, ee_frame_g)
    sim = Simulator(diagram_g, ctx_g)
    sim.Initialize()
    sim.AdvanceTo(DT_BIG)
    drake_after = plant_g.GetMyContextFromRoot(ctx_g)
    drake_xyz = plant_g.EvalBodyPoseInWorld(drake_after, obj_body_g).translation()
    drake_delta = drake_xyz - BOX_POS
    print(f"\n  Drake ground truth (single 0.05-s step):")
    print(f"    Δ box xyz = ({drake_delta[0]*1000:+.3f}, "
          f"{drake_delta[1]*1000:+.3f}, {drake_delta[2]*1000:+.3f}) mm")

    cells = {}
    for label, drag_val in (("A_dt050_d10", 10.0), ("B_dt050_d00", 0.0)):
        diagram, plant, panda, object_, _, plant_ad, ctx_ad = \
            build_environment(task_cfg, time_step=0.001)
        obj_body = plant.GetBodyByName("box_link", object_)
        ee_frame = plant.GetFrameByName("pusher")
        ctx, pctx, _ = setup_contact_state(
            diagram, plant, panda, object_, ee_frame)
        delta, lam, res, _ = lcs_single_step(
            drag_val, plant, pctx, ee_frame, obj_body, plant_ad, ctx_ad,
            mu, DT_BIG)
        cells[label] = {"box_delta": delta[4:7].copy(),
                        "lambda_max": float(lam.max()),
                        "lcp_res": res, "kind": "single-step"}
        print(f"\n  CELL {label}  Δt={DT_BIG} drag={drag_val}  ({cells[label]['kind']})")
        print(f"    Δ box xyz = ({delta[4]*1000:+.3f}, "
              f"{delta[5]*1000:+.3f}, {delta[6]*1000:+.3f}) mm")

    for label, drag_val in (("C_dt005_d10", 10.0), ("D_dt005_d00", 0.0)):
        diagram, plant, panda, object_, _, plant_ad, ctx_ad = \
            build_environment(task_cfg, time_step=0.001)
        obj_body = plant.GetBodyByName("box_link", object_)
        ee_frame = plant.GetFrameByName("pusher")
        ctx, pctx, q_arm_init = setup_contact_state(
            diagram, plant, panda, object_, ee_frame)
        box_delta, lam_hist, fail_or_fallback = lcs_substep_refresh(
            drag_val, plant, pctx, ee_frame, obj_body, plant_ad, ctx_ad,
            mu, DT_SMALL, N_SUB, panda, object_, q_arm_init)
        cells[label] = {"box_delta": box_delta.copy(),
                        "lambda_first": lam_hist[0],
                        "lambda_last": lam_hist[-1],
                        "lcp_fail_steps": fail_or_fallback,
                        "kind": "10×sub-step RE-EXTRACTED"}
        print(f"\n  CELL {label}  Δt={DT_SMALL} drag={drag_val}  ({cells[label]['kind']})")
        print(f"    Δ box xyz (cumulative) = ({box_delta[0]*1000:+.3f}, "
              f"{box_delta[1]*1000:+.3f}, {box_delta[2]*1000:+.3f}) mm")
        print(f"    λ first sub-step: max={lam_hist[0][0]:.4f}, "
              f"nnz={int(lam_hist[0][1])}")
        print(f"    λ last  sub-step: max={lam_hist[-1][0]:.4f}, "
              f"nnz={int(lam_hist[-1][1])}")
        if fail_or_fallback == -1:
            print(f"    [FLAG] IK failed at some sub-step → fallback to FIXED-LCS")
        elif fail_or_fallback > 0:
            print(f"    [FLAG] {fail_or_fallback} sub-steps had Lemke residual > 1e-3")

    # Run the FIXED-LCS variant alongside as an empirical reference (so the
    # re-extract-vs-fixed difference is measured, not declared).
    fixed_results = {}
    for label, drag_val in (("Cf_fixed_d10", 10.0), ("Df_fixed_d00", 0.0)):
        diagram, plant, panda, object_, _, plant_ad, ctx_ad = \
            build_environment(task_cfg, time_step=0.001)
        obj_body = plant.GetBodyByName("box_link", object_)
        ee_frame = plant.GetFrameByName("pusher")
        ctx, pctx, _ = setup_contact_state(
            diagram, plant, panda, object_, ee_frame)
        bd = lcs_substep_fixed(
            drag_val, plant, pctx, ee_frame, obj_body, plant_ad, ctx_ad,
            mu, DT_SMALL, N_SUB)
        fixed_results[label] = bd
        print(f"\n  REF  {label}  Δt={DT_SMALL} drag={drag_val}  (10× FIXED-LCS)")
        print(f"    Δ box xyz (cumulative) = ({bd[0]*1000:+.3f}, "
              f"{bd[1]*1000:+.3f}, {bd[2]*1000:+.3f}) mm")

    # --------------- summary table -----------------------------------------
    print()
    print("=" * 72)
    print("SUMMARY — 4-cell factorial + 2 fixed-LCS reference cells")
    print("=" * 72)
    print(f"  Drake (ground truth)                : "
          f"({drake_delta[0]*1000:+.3f}, {drake_delta[1]*1000:+.3f}, "
          f"{drake_delta[2]*1000:+.3f}) mm")

    def fmt_cell(d):
        return f"({d[0]*1000:+.3f}, {d[1]*1000:+.3f}, {d[2]*1000:+.3f}) mm"

    def factor(d):
        return abs(drake_delta[0] / d[0]) if abs(d[0]) > 1e-9 else float("inf")

    rows = [
        ("A  Δt=0.05  drag=10   (single)        ", cells["A_dt050_d10"]["box_delta"]),
        ("B  Δt=0.05  drag= 0   (single)        ", cells["B_dt050_d00"]["box_delta"]),
        ("C  Δt=0.005 drag=10   (RE-EXTRACT)    ", cells["C_dt005_d10"]["box_delta"]),
        ("D  Δt=0.005 drag= 0   (RE-EXTRACT)    ", cells["D_dt005_d00"]["box_delta"]),
        ("Cf Δt=0.005 drag=10   (FIXED-LCS ref) ", fixed_results["Cf_fixed_d10"]),
        ("Df Δt=0.005 drag= 0   (FIXED-LCS ref) ", fixed_results["Df_fixed_d00"]),
    ]
    print()
    for name, d in rows:
        print(f"  {name}: {fmt_cell(d)}     factor (|Drake/LCS_x|): {factor(d):.2f}×")

    # ----- main effects + interaction -----
    print()
    print("=" * 72)
    print("MAIN EFFECTS + INTERACTION")
    print("=" * 72)

    # Δt main effect: avg(Δt=0.005) - avg(Δt=0.05) in box_delta_x
    avg_dt050 = 0.5 * (cells["A_dt050_d10"]["box_delta"][0]
                       + cells["B_dt050_d00"]["box_delta"][0])
    avg_dt005 = 0.5 * (cells["C_dt005_d10"]["box_delta"][0]
                       + cells["D_dt005_d00"]["box_delta"][0])
    dt_main = (avg_dt005 - avg_dt050) * 1000
    # Drag main: avg(drag=10) - avg(drag=0)
    avg_d10 = 0.5 * (cells["A_dt050_d10"]["box_delta"][0]
                     + cells["C_dt005_d10"]["box_delta"][0])
    avg_d00 = 0.5 * (cells["B_dt050_d00"]["box_delta"][0]
                     + cells["D_dt005_d00"]["box_delta"][0])
    drag_main = (avg_d10 - avg_d00) * 1000
    # Interaction: (C - D) - (A - B)
    interact = ((cells["C_dt005_d10"]["box_delta"][0]
                 - cells["D_dt005_d00"]["box_delta"][0])
                - (cells["A_dt050_d10"]["box_delta"][0]
                   - cells["B_dt050_d00"]["box_delta"][0])) * 1000

    print(f"  Δt main effect (avg Δt=0.005 - avg Δt=0.05) on box_x: "
          f"{dt_main:+.4f} mm")
    print(f"  drag main effect (avg drag=10 - avg drag=0) on box_x: "
          f"{drag_main:+.4f} mm")
    print(f"  Δt × drag interaction:                                  "
          f"{interact:+.4f} mm")

    print()
    print(f"  Re-extract vs Fixed-LCS difference (drag=10): "
          f"{(cells['C_dt005_d10']['box_delta'][0] - fixed_results['Cf_fixed_d10'][0])*1000:+.4f} mm")
    print(f"  Re-extract vs Fixed-LCS difference (drag=0):  "
          f"{(cells['D_dt005_d00']['box_delta'][0] - fixed_results['Df_fixed_d00'][0])*1000:+.4f} mm")

    # Vertical sanity
    print()
    print("VERTICAL SANITY (Δz, all cells should be near 0 with count=4)")
    for name, d in rows:
        print(f"  {name}: Δz = {d[2]*1000:+.4f} mm")

    # Three-outcome routing on Δt
    print()
    print("=" * 72)
    print("OUTCOME ROUTING (Δt main effect)")
    print("=" * 72)
    # Use the larger-magnitude of C/D as the "with sub-stepping" factor
    f_C = factor(cells["C_dt005_d10"]["box_delta"])
    f_D = factor(cells["D_dt005_d00"]["box_delta"])
    f_best_substep = min(f_C, f_D)
    print(f"  Best (smallest) under-prediction factor at Δt=0.005: "
          f"{f_best_substep:.2f}×  (was 3.73×)")
    if f_best_substep < 1.5:
        outcome = "Δt-IS-IT (WHOLE)"
    elif f_best_substep < 2.5:
        outcome = "Δt-IS-HALF"
    else:
        outcome = "Δt-NOT-IT (REDHERRING)"
    print(f"  ► OUTCOME: {outcome}")

    # Drag-caveat closure
    print()
    print(f"  Drag interaction at Δt=0.005 (C - D) on box_x: "
          f"{(cells['C_dt005_d10']['box_delta'][0] - cells['D_dt005_d00']['box_delta'][0])*1000:+.4f} mm")
    print(f"  Drag interaction at Δt=0.05  (A - B) on box_x: "
          f"{(cells['A_dt050_d10']['box_delta'][0] - cells['B_dt050_d00']['box_delta'][0])*1000:+.4f} mm")
    drag_at_substep = abs(cells['C_dt005_d10']['box_delta'][0]
                          - cells['D_dt005_d00']['box_delta'][0]) * 1000
    if drag_at_substep > 0.01:
        print(f"  ► DRAG-MATTERS-MOVING — drag-on/off cells diverge at Δt=0.005 "
              f"(v_box≠0): {drag_at_substep:.4f} mm")
    else:
        print(f"  ► drag remains INERT at v_box≠0 sub-stepping (cells identical "
              f"within {drag_at_substep:.4f} mm)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
