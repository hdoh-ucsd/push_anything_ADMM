"""Stage C Part A — force-level confirmation probe (offline).

Question (§7.16-aug routed): does the displacement-level rigid-vs-compliant
diagnosis hold AT THE FORCE LEVEL? §7.16 showed the LCS scales nearly
linearly with EE penetration while Drake's box motion stays stable —
crossing 1× at ~0.549 mm. This probe maps the FORCE PROFILES (not just
end-state displacements) at three depths (0.10 mm, 0.549 mm, 1.00 mm) —
under-spot, sweet-spot, over-spot — to confirm the rigid-impulsive
(LCS) vs soft-spread (Drake) signature.

Method (per depth):
  1. Set up the clean (box-pinned) state at this EE penetration.
  2. LCS: run 10 sub-steps at Δt=0.005, re-extract each step; dump per-
     sub-step λ_n for the EE↔box contact pair (Stewart-Trinkle
     impulse-like Lagrange multiplier; the §7.13 burst signal,
     read here AS a force/impulse profile).
  3. Drake: AdvanceTo(0.001 × k) for k = 1..50; at each tick read the
     pusher↔box ContactResults entry's normal force (the compliant
     point-contact normal force in N).
  4. Tabulate both profiles.

Pre-registered force-level signatures (from §7.16-aug (2)):
  LCS  expected: rigid-impulsive — sharp early λ_n spike scaling with
                 penetration depth, decaying as the box accelerates away.
  Drake expected: soft-spread — depth-stable peak force, distributed over
                  the contact stiffness time-constant, weaker depth-dep.

Three-outcome:
  CONFIRMED-COMPLIANCE — profiles match rigid-impulsive vs soft-spread
                         AND depth scaling is LCS-linear vs Drake-stable.
                         → Part B (anitescu scoping) opens.
  UNEXPECTED-MATCH     — profiles similar (LCS not impulsive OR Drake
                         not soft) despite §7.16's displacement
                         divergence → mechanism is something else; do
                         NOT proceed to anitescu.
  PARTIAL              — partial match; characterize residual before
                         anitescu.
"""
from __future__ import annotations

import os
import importlib
import numpy as np
import yaml

from pydrake.systems.analysis import Simulator
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.multibody.tree import JacobianWrtVariable, BodyIndex
from pydrake.solvers import Solve
from pydrake.geometry import Role
from pydrake.math import RotationMatrix

from sim.env_builder import build_environment
from control.lcp_solver import solve_lcp

# ---- Sweep cells: under-spot, sweet-spot, over-spot ----------------------
DEPTH_M = [0.0001, 0.000549, 0.001]   # 0.10 mm, 0.549 mm, 1.00 mm
DEPTH_TAG = ["UNDER (0.10mm)", "SWEET (0.549mm)", "OVER (1.00mm)"]

DT_BIG    = 0.05
DT_SUB    = 0.005
N_SUB     = int(round(DT_BIG / DT_SUB))    # 10
DT_DRAKE  = 0.001
N_DRAKE   = int(round(DT_BIG / DT_DRAKE))  # 50

BOX_HALF  = 0.05
EE_RADIUS = 0.025
BOX_QUAT  = np.array([1.0, 0.0, 0.0, 0.0])
BOX_POS   = np.array([0.0, 0.0, 0.05])
EE_VEL_X  = -0.05

BOX_POS_X_IDX  = 4
BOX_POS_Z_IDX  = 6

ARM_BOX_CLEARANCE_M = 0.005
ARM_LINKS_TO_CLEAR  = ["panda_link4", "panda_link5", "panda_link6", "panda_link7"]
POSTURE_NOMINAL = np.array([0.0, -0.4, 0.0, -1.8, 0.0, 1.4, 0.785])
POSTURE_WEIGHT = 5.0
IK_SEEDS = [
    np.array([0.0,  0.0,  0.0, -1.5,  0.0,  1.5,  0.785]),
    np.array([0.0, -0.3,  0.0, -1.7,  0.0,  1.4,  0.785]),
]


def _scene_graph_of(diagram):
    for sys in diagram.GetSystems():
        if 'SceneGraph' in type(sys).__name__:
            return sys
    raise RuntimeError("SceneGraph not found")


def _build_env(task_cfg):
    diagram, plant, panda, obj, meshcat, plant_ad, ctx_ad = \
        build_environment(task_cfg, time_step=0.001)
    sg = _scene_graph_of(diagram)
    return diagram, plant, sg, panda, obj, meshcat, plant_ad, ctx_ad


def _collect_geom_ids(plant, sg, model, body_names):
    q = sg.model_inspector()
    ids = []
    for bname in body_names:
        body = plant.GetBodyByName(bname, model)
        fid = plant.GetBodyFrameIdOrThrow(body.index())
        for gid in q.GetGeometries(fid, Role.kProximity):
            ids.append(gid)
    return ids


def _geom_ids_for_body(plant, sg, body):
    q = sg.model_inspector()
    fid = plant.GetBodyFrameIdOrThrow(body.index())
    return list(q.GetGeometries(fid, Role.kProximity))


def setup_state_at_depth(diagram, plant, sg, panda_model, object_model,
                         ee_frame, ee_pen_m):
    """Set up clean pusher-only state at this EE penetration."""
    world = plant.world_frame()
    ee_pos = np.array([BOX_HALF + EE_RADIUS - ee_pen_m, 0.0, 0.05])
    p_tol = 1e-5
    arm_geoms = _collect_geom_ids(plant, sg, panda_model, ARM_LINKS_TO_CLEAR)
    box_body = plant.GetBodyByName("box_link", object_model)
    box_geoms = _geom_ids_for_body(plant, sg, box_body)

    for seed_idx, seed in enumerate(IK_SEEDS):
        context = diagram.CreateDefaultContext()
        plant_ctx = plant.GetMyContextFromRoot(context)
        plant.SetPositions(plant_ctx, object_model,
                           np.concatenate([BOX_QUAT, BOX_POS]))
        plant.SetPositions(plant_ctx, panda_model, seed)
        ik = InverseKinematics(plant, plant_ctx, with_joint_limits=True)
        box_frame_pin = box_body.body_frame()
        ik.AddPositionConstraint(box_frame_pin, np.zeros(3), world,
                                 BOX_POS - 1e-5, BOX_POS + 1e-5)
        ik.AddOrientationConstraint(world, RotationMatrix(),
                                    box_frame_pin, RotationMatrix(), 0.001)
        ik.AddPositionConstraint(ee_frame, np.zeros(3), world,
                                 ee_pos - p_tol, ee_pos + p_tol)
        for a_gid in arm_geoms:
            for b_gid in box_geoms:
                ik.AddDistanceConstraint((a_gid, b_gid),
                                          distance_lower=ARM_BOX_CLEARANCE_M,
                                          distance_upper=10.0)
        q_dec = ik.q()
        j1 = plant.GetJointByName("panda_joint1").position_start()
        for k in range(7):
            ik.get_mutable_prog().AddQuadraticErrorCost(
                np.array([[POSTURE_WEIGHT]]),
                np.array([POSTURE_NOMINAL[k]]),
                np.array([q_dec[j1 + k]]))
        ik.get_mutable_prog().SetInitialGuess(ik.q(), plant.GetPositions(plant_ctx))
        res = Solve(ik.prog())
        if not res.is_success():
            continue
        plant.SetPositions(plant_ctx, res.GetSolution(ik.q()))
        q_arm = plant.GetPositions(plant_ctx, panda_model).copy()
        p_ee_actual = plant.CalcPointsPositions(
            plant_ctx, ee_frame, np.zeros(3), world).flatten()
        if np.linalg.norm(p_ee_actual - ee_pos) > 1e-3:
            continue
        plant.SetVelocities(plant_ctx, object_model, np.zeros(6))
        J = plant.CalcJacobianTranslationalVelocity(
            plant_ctx, JacobianWrtVariable.kV, ee_frame, np.zeros(3),
            world, world)
        s = plant.GetJointByName("panda_joint1").velocity_start()
        q_arm_dot, *_ = np.linalg.lstsq(J[:, s:s+7],
                                        np.array([EE_VEL_X, 0., 0.]),
                                        rcond=None)
        plant.SetVelocities(plant_ctx, panda_model, q_arm_dot)
        return context, plant_ctx, q_arm, ee_pos
    return None, None, None, None


def lcs_force_profile(plant, plant_ctx, ee_frame, obj_body, plant_ad, ctx_ad,
                      mu, panda_model, object_model, q_arm_init, ee_pos):
    """Run LCS sub-stepped Δt=0.005, drag=0, count=4. Dump per-sub-step
    λ_n for the EE↔box contact and the box state for each sub-step.

    Returns (lam_n_history (N_SUB,), dx (total)).
    """
    from control.lcs_formulator import LCSFormulator
    world = plant.world_frame()
    x_curr = np.concatenate([
        BOX_QUAT, BOX_POS, ee_pos, np.zeros(3), np.zeros(3),
        np.array([EE_VEL_X, 0., 0.])])
    box_x_0 = float(x_curr[BOX_POS_X_IDX])
    q_arm = q_arm_init.copy()
    lam_n_hist = np.full(N_SUB, np.nan)

    for k in range(N_SUB):
        f = LCSFormulator(plant, mu=mu, obj_body=obj_body,
                          plant_ad=plant_ad, context_ad=ctx_ad,
                          box_ground_drag=0.0)
        A, B_ctrl, D, d_const, E, F, H, c_lcs, *_ = \
            f.linearize_discrete_ee_space(plant_ctx, DT_SUB, np.zeros(3))
        # Identify EE-box λ_n index
        contacts = getattr(f, '_last_contact_info', [])
        ee_box_idx = None
        for ci, info in enumerate(contacts):
            if info.get('tag', '') == 'EE-BOX':
                ee_box_idx = ci
                break
        n_lam = D.shape[1]
        n_c = n_lam // 6  # γ, λ_n, 4·λ_t per contact
        lam_n_offset = n_c  # γ block first, then λ_n

        q_lcp = E @ x_curr + c_lcs
        lam, _ = solve_lcp(F, q_lcp)
        if ee_box_idx is not None:
            lam_n_hist[k] = float(lam[lam_n_offset + ee_box_idx])
        x_next = A @ x_curr + B_ctrl @ np.zeros(3) + D @ lam + d_const

        if k < N_SUB - 1:
            plant.SetPositions(plant_ctx, object_model, x_next[0:7])
            plant.SetVelocities(plant_ctx, object_model, x_next[10:16])
            plant.SetPositions(plant_ctx, panda_model, q_arm)
            ik = InverseKinematics(plant, plant_ctx, with_joint_limits=True)
            ik.AddPositionConstraint(ee_frame, np.zeros(3), world,
                                     x_next[7:10] - 1e-5, x_next[7:10] + 1e-5)
            ik.get_mutable_prog().SetInitialGuess(
                ik.q(), plant.GetPositions(plant_ctx))
            res = Solve(ik.prog())
            if not res.is_success():
                break
            plant.SetPositions(plant_ctx, res.GetSolution(ik.q()))
            q_arm = plant.GetPositions(plant_ctx, panda_model).copy()
            J = plant.CalcJacobianTranslationalVelocity(
                plant_ctx, JacobianWrtVariable.kV, ee_frame, np.zeros(3),
                world, world)
            s = plant.GetJointByName("panda_joint1").velocity_start()
            q_arm_dot, *_ = np.linalg.lstsq(J[:, s:s+7], x_next[16:19],
                                            rcond=None)
            plant.SetVelocities(plant_ctx, panda_model, q_arm_dot)
        x_curr = x_next
    dx = float(x_curr[BOX_POS_X_IDX]) - box_x_0
    return lam_n_hist, dx


def drake_force_profile(diagram, plant, ctx, obj_body, ee_body):
    """Run Drake forward 0.05 s in 1 ms ticks. At each tick read the
    pusher↔box pair's normal force projected on world +x.

    Returns (force_history (N_DRAKE+1,), dx_drake).
    """
    sim = Simulator(diagram, ctx)
    sim.Initialize()
    pctx0 = plant.GetMyContextFromRoot(ctx)
    box_x_0 = float(plant.EvalBodyPoseInWorld(pctx0, obj_body).translation()[0])

    F_pusher_n_x = np.zeros(N_DRAKE + 1)
    for k in range(N_DRAKE + 1):
        if k > 0:
            sim.AdvanceTo(k * DT_DRAKE)
        pctx_now = plant.GetMyContextFromRoot(ctx)
        cr = plant.get_contact_results_output_port().Eval(pctx_now)
        f_x = 0.0
        for i in range(cr.num_point_pair_contacts()):
            info = cr.point_pair_contact_info(i)
            bodyA = plant.get_body(info.bodyA_index())
            bodyB = plant.get_body(info.bodyB_index())
            if bodyB.index() == obj_body.index():
                other = bodyA
                F_box = info.contact_force()
            elif bodyA.index() == obj_body.index():
                other = bodyB
                F_box = -info.contact_force()
            else:
                continue
            if "pusher" not in other.name().lower():
                continue
            pp = info.point_pair()
            nhat = pp.nhat_BA_W
            f_norm_mag = float(F_box @ nhat)
            F_norm = f_norm_mag * nhat
            f_x += float(F_norm[0])  # normal-force x-component on box
        F_pusher_n_x[k] = f_x

    pctx_end = plant.GetMyContextFromRoot(ctx)
    box_x_end = float(plant.EvalBodyPoseInWorld(pctx_end, obj_body).translation()[0])
    return F_pusher_n_x, box_x_end - box_x_0


def main() -> int:
    print("=" * 80)
    print("STAGE C  FORCE-LEVEL PROBE — Part A (rigid-vs-compliant at the force level)")
    print("=" * 80)
    os.environ["LCS_EXPLICIT_BOX_GND"] = "4"
    import control.lcs_formulator
    importlib.reload(control.lcs_formulator)

    with open("config/tasks.yaml") as f:
        cfg = yaml.safe_load(f)
    task_cfg = cfg["tasks"]["pushing"]
    mu = task_cfg["friction"]
    print(f"  μ = {mu}, Δt_sub = {DT_SUB}s × {N_SUB} sub-steps, Drake 1ms ticks")
    print(f"  Depths: {[round(d*1000,3) for d in DEPTH_M]} mm  "
          f"({', '.join(DEPTH_TAG)})")
    print()

    results = {}
    for depth_idx, (pen_m, tag) in enumerate(zip(DEPTH_M, DEPTH_TAG)):
        pen_mm = pen_m * 1000
        print("─" * 80)
        print(f"DEPTH {tag}")
        print("─" * 80)

        # LCS λ_n history
        d, p, sg, panda, obj, _, p_ad, ctx_ad = _build_env(task_cfg)
        obj_body = p.GetBodyByName("box_link", obj)
        ee_frame = p.GetFrameByName("pusher")
        ctx, pctx, q_arm, ee_pos = setup_state_at_depth(
            d, p, sg, panda, obj, ee_frame, pen_m)
        if ctx is None:
            print(f"  re-pose FAIL at depth {pen_mm}mm — skip")
            continue
        lam_n_hist, dx_lcs = lcs_force_profile(
            p, pctx, ee_frame, obj_body, p_ad, ctx_ad, mu, panda, obj,
            q_arm, ee_pos)

        # Drake force history
        d2, p2, sg2, panda2, obj2, _, _, _ = _build_env(task_cfg)
        obj_body2 = p2.GetBodyByName("box_link", obj2)
        ee_frame2 = p2.GetFrameByName("pusher")
        ee_body2  = p2.GetBodyByName("pusher")
        ctx2, _, _, _ = setup_state_at_depth(
            d2, p2, sg2, panda2, obj2, ee_frame2, pen_m)
        F_drake, dx_drake = drake_force_profile(
            d2, p2, ctx2, obj_body2, ee_body2)

        results[pen_mm] = dict(
            lam_n_hist=lam_n_hist,
            F_drake=F_drake,
            dx_lcs=dx_lcs,
            dx_drake=dx_drake)

        # Print LCS λ_n profile
        print(f"  LCS λ_n history (per sub-step, EE↔box contact):")
        print(f"    sub-step: " + "  ".join(f"{i:>6d}" for i in range(N_SUB)))
        print(f"    λ_n     : " + "  ".join(
            f"{v:>6.3f}" if not np.isnan(v) else "   NaN" for v in lam_n_hist))
        peak_lam = np.nanmax(lam_n_hist) if not np.isnan(lam_n_hist).all() else float('nan')
        peak_lam_step = int(np.nanargmax(lam_n_hist)) if not np.isnan(lam_n_hist).all() else -1
        sum_lam = np.nansum(lam_n_hist)
        print(f"    peak λ_n = {peak_lam:.4f} at sub-step {peak_lam_step};  "
              f"Σλ_n = {sum_lam:.4f}")
        # First-vs-last ratio (rigid would be decaying spike)
        if not np.isnan(lam_n_hist[0]) and not np.isnan(lam_n_hist[-1]):
            ratio = lam_n_hist[0] / max(lam_n_hist[-1], 1e-6)
            print(f"    λ_n[0] / λ_n[-1] = {ratio:.2f}  "
                  f"(rigid-impulsive expects >> 1)")

        # Print Drake force profile (subsampled)
        peak_f_drake = np.max(np.abs(F_drake))
        peak_f_step = int(np.argmax(np.abs(F_drake)))
        avg_force = np.mean(np.abs(F_drake[1:]))  # exclude k=0 (sim not stepped)
        print(f"  Drake normal-force x-component on box (every 5 ms):")
        sub_idx = list(range(0, N_DRAKE + 1, 5))
        print(f"    t (ms)  : " + "  ".join(f"{i:>6d}" for i in sub_idx))
        print(f"    F_x (N) : " + "  ".join(
            f"{F_drake[i]:>+6.3f}" for i in sub_idx))
        print(f"    peak |F| = {peak_f_drake:.4f} N at tick {peak_f_step} "
              f"(t={peak_f_step*DT_DRAKE*1000:.1f}ms);  avg |F| = {avg_force:.4f} N")
        # First-vs-last ratio (compliant would be roughly stable)
        if abs(F_drake[1]) > 1e-6 and abs(F_drake[-1]) > 1e-6:
            ratio_drake = abs(F_drake[1]) / abs(F_drake[-1])
            print(f"    F[1ms] / F[50ms] = {ratio_drake:.2f}  "
                  f"(soft-spread expects ~1)")

        print(f"  Δbox_x  LCS = {dx_lcs*1000:+.4f}mm  Drake = {dx_drake*1000:+.4f}mm  "
              f"factor = {abs(dx_drake/dx_lcs) if abs(dx_lcs)>1e-9 else float('inf'):.3f}×")
        print()

    # ---- Cross-depth comparison ----
    print("=" * 80)
    print("CROSS-DEPTH COMPARISON")
    print("=" * 80)
    print()
    print(f"  Peak λ_n (LCS) vs depth — expected LINEAR scaling if rigid:")
    print(f"    {'depth (mm)':>11} {'peak λ_n':>12} {'Σλ_n':>10} {'λ_n[0]/λ_n[-1]':>16}")
    for pen_mm, r in results.items():
        peak = np.nanmax(r['lam_n_hist'])
        ssum = np.nansum(r['lam_n_hist'])
        ratio = (r['lam_n_hist'][0] /
                 max(r['lam_n_hist'][-1], 1e-6))
        print(f"    {pen_mm:>9.3f}   {peak:>12.4f} {ssum:>10.4f} {ratio:>15.2f}")

    print()
    print(f"  Peak Drake force vs depth — expected STABLE if compliant:")
    print(f"    {'depth (mm)':>11} {'peak |F| (N)':>15} {'avg |F| (N)':>14} {'F[1]/F[50]':>13}")
    for pen_mm, r in results.items():
        peak = float(np.max(np.abs(r['F_drake'])))
        avg  = float(np.mean(np.abs(r['F_drake'][1:])))
        if abs(r['F_drake'][1]) > 1e-6 and abs(r['F_drake'][-1]) > 1e-6:
            ratio = abs(r['F_drake'][1]) / abs(r['F_drake'][-1])
        else:
            ratio = float('nan')
        print(f"    {pen_mm:>9.3f}   {peak:>15.4f} {avg:>14.4f} {ratio:>13.2f}")

    # ---- ROUTE ----
    print()
    print("=" * 80)
    print("ROUTE (force-level confirmation)")
    print("=" * 80)
    pens = sorted(results.keys())
    if len(pens) < 2:
        print("  ► INSUFFICIENT DATA — fewer than 2 depths.")
        return 0

    peak_lcs = [np.nanmax(results[p]['lam_n_hist']) for p in pens]
    peak_drake = [float(np.max(np.abs(results[p]['F_drake']))) for p in pens]
    lcs_ratio_dyn = peak_lcs[-1] / max(peak_lcs[0], 1e-9)   # how much LCS λ_n grows
    drake_ratio_dyn = peak_drake[-1] / max(peak_drake[0], 1e-9)  # Drake force growth
    depth_ratio = pens[-1] / pens[0]   # how much depth grew (e.g., 1.0 / 0.1 = 10×)
    print(f"  Depth range: {pens[0]:.3f} → {pens[-1]:.3f} mm  ({depth_ratio:.1f}×)")
    print(f"  LCS peak λ_n: {peak_lcs[0]:.3f} → {peak_lcs[-1]:.3f}  "
          f"({lcs_ratio_dyn:.2f}× growth)")
    print(f"  Drake peak F: {peak_drake[0]:.3f} → {peak_drake[-1]:.3f}  "
          f"({drake_ratio_dyn:.2f}× growth)")
    print()
    print(f"  Rigid LCS signature: λ_n scales LINEARLY with depth → growth ratio ≈ depth ratio")
    print(f"    expected: LCS_growth ≈ {depth_ratio:.1f}×;  observed: {lcs_ratio_dyn:.2f}×")
    print(f"  Compliant Drake signature: peak force STABLE with depth → growth ratio ≈ 1")
    print(f"    expected: Drake_growth ≈ 1.0×;  observed: {drake_ratio_dyn:.2f}×")
    print()

    # Scoring: LCS should grow with depth (closer to depth_ratio than to 1)
    # Drake should NOT grow much with depth (closer to 1 than to depth_ratio)
    lcs_compliant_like = abs(lcs_ratio_dyn - 1.0) < abs(lcs_ratio_dyn - depth_ratio)
    drake_rigid_like = abs(drake_ratio_dyn - depth_ratio) < abs(drake_ratio_dyn - 1.0)
    lcs_rigid_like = not lcs_compliant_like
    drake_compliant_like = not drake_rigid_like

    if lcs_rigid_like and drake_compliant_like:
        route = "CONFIRMED-COMPLIANCE"
        msg = (
            f"  Force-level signature MATCHES the displacement-level diagnosis.\n"
            f"  LCS scales with depth (rigid); Drake is roughly depth-stable (compliant).\n"
            f"  The rigid-vs-compliant diagnosis is LOCKED at the FORCE level.\n"
            f"  NEXT (PART B in this same block): scope anitescu reformulation.")
    elif lcs_compliant_like and drake_compliant_like:
        route = "UNEXPECTED-MATCH"
        msg = (
            f"  Force-level signature does NOT match rigid-vs-compliant.\n"
            f"  Both LCS and Drake look depth-stable at the force level, yet §7.16\n"
            f"  showed displacement-level divergence. The mechanism is NOT purely\n"
            f"  normal compliance — re-examine before anitescu.\n"
            f"  ► PART B PAUSED.")
    elif lcs_rigid_like and drake_rigid_like:
        route = "UNEXPECTED-MATCH"
        msg = (
            f"  Force-level signature: BOTH LCS and Drake show rigid-impulsive\n"
            f"  scaling. The displacement-level divergence is NOT explained by\n"
            f"  rigid-vs-compliant; something else is in play.\n"
            f"  ► PART B PAUSED.")
    else:
        route = "PARTIAL"
        msg = (
            f"  Force-level signature partial match. One side aligns with rigid-vs-\n"
            f"  compliant, the other does not. Characterize the residual mechanism\n"
            f"  before scoping anitescu.")

    print(f"  ► ROUTE: {route}")
    print(msg)
    return 0 if route == "CONFIRMED-COMPLIANCE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
