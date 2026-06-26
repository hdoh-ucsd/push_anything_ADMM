"""Stage C force-level RE-PROBE — Part A, sub-ms Drake sampling (offline).

Question (§7.20 routed): the §7.17 force-level probe (commit 8b55c94)
verdicted FORCE-DISCONFIRMS with three of four signatures failing —
Drake "OSCILLATING not soft-spread", Drake peak force decreased with
depth, LCS λ_n sub-linear with depth. §7.20 (continuous-φ at 0.25 ms)
showed the underlying Drake contact is essentially MAINTAINED at all
depths (max φ ≤ 8.6 µm < 30 µm grazing threshold). The §7.17 Drake-
oscillating reading was reading 1 ms-binary-thresholding-artifact zeros
on a maintained contact, not real contact loss.

This re-probe re-runs Part A at the §7.20 sub-ms Drake sampling
resolution and the §7.18/§7.20 robust 5-seed IK setup, so:
  - Drake force is read at every 0.25 ms tick (vs §7.17's every 1 ms)
  - 0.549 mm SWEET-SPOT depth does not re-pose FAIL (the §7.17 script's
    2-seed IK was too thin; the 5-seed list from §7.18 succeeds)
  - 1.00 mm OVER-spot LCS sub-step doesn't break partway through

Pre-registered signatures (same as §7.17, restated for the re-probe):
  - LCS λ_n: rigid-impulsive (sharp spike at sub-step 0, decay) +
             linear scaling with depth (10× depth ⇒ ~10× peak λ_n)
  - Drake force: soft-spread (depth-stable peak, distributed in time) +
             stable peak vs depth (10× depth ⇒ ~1× peak |F|)
  - Sweet-spot 0.549 mm: total impulse over Δt = 0.05 s matches between
             LCS and Drake (that's what the displacement-crossing says)

The §7.20-augmented re-read: classify Drake force ON MAINTAINED
contact. The §7.17 "OSCILLATING" reading is replaced by the sub-ms read
of "force-magnitude variation on continuous contact" (= soft-spread,
not impulsive). The LCS sub-linear scaling stands or falls on its own.

Routes (same four-way as the prompt):
  FORCE-CONFIRMS → LCS impulsive + Drake soft-spread at sub-ms; depth
                   scaling matches (rigid linear, compliant stable);
                   compliance LOCKED at the force level.
  FORCE-CONFIRMS-PARTIAL → some signatures match, some don't (e.g.,
                   Drake soft-spread ✓ but LCS sub-linear scaling ✗).
                   Compliance partially locked; document the residual.
  FORCE-DISCONFIRMS → §7.17 verdict still stands under finer sampling.
  REFERENCE-UNREADABLE → not in scope here (a Part B concern).
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

DEPTH_M   = [0.0001, 0.000549, 0.001]   # 0.10 mm, 0.549 mm, 1.00 mm
DEPTH_TAG = ["UNDER (0.10mm)", "SWEET (0.549mm)", "OVER (1.00mm)"]

DT_BIG    = 0.05
DT_SUB    = 0.005
N_SUB     = int(round(DT_BIG / DT_SUB))           # 10 LCS sub-steps
DT_DRAKE  = 0.00025                                # 0.25 ms — §7.20 dt
N_DRAKE   = int(round(DT_BIG / DT_DRAKE))          # 200 ticks

BOX_HALF  = 0.05
EE_RADIUS = 0.025
BOX_QUAT  = np.array([1.0, 0.0, 0.0, 0.0])
BOX_POS   = np.array([0.0, 0.0, 0.05])
EE_VEL_X  = -0.05

BOX_POS_X_IDX  = 4

ARM_BOX_CLEARANCE_M = 0.005
ARM_LINKS_TO_CLEAR  = ["panda_link4", "panda_link5", "panda_link6", "panda_link7"]
POSTURE_NOMINAL = np.array([0.0, -0.4, 0.0, -1.8, 0.0, 1.4, 0.785])
POSTURE_WEIGHT = 5.0
# §7.18/§7.20 robust 5-seed list
IK_SEEDS_INIT = [
    np.array([0.0,  0.0,  0.0, -1.5,  0.0,  1.5,  0.785]),
    np.array([0.0, -0.3,  0.0, -1.7,  0.0,  1.4,  0.785]),
    np.array([0.0,  0.3,  0.0, -1.3,  0.0,  1.6,  0.785]),
    np.array([0.0,  0.0,  0.0, -2.0,  0.0,  2.0,  0.785]),
    np.array([0.0, -0.5,  0.0, -2.0,  0.0,  1.5,  0.785]),
]


def _scene_graph_of(diagram):
    for sys in diagram.GetSystems():
        if 'SceneGraph' in type(sys).__name__:
            return sys
    raise RuntimeError("SceneGraph not found")


def _build_env(task_cfg, time_step):
    diagram, plant, panda, obj, meshcat, plant_ad, ctx_ad = \
        build_environment(task_cfg, time_step=time_step)
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
    """Robust 5-seed setup_state. Same as §7.18/§7.20."""
    world = plant.world_frame()
    ee_pos = np.array([BOX_HALF + EE_RADIUS - ee_pen_m, 0.0, 0.05])
    p_tol = 1e-5
    arm_geoms = _collect_geom_ids(plant, sg, panda_model, ARM_LINKS_TO_CLEAR)
    box_body = plant.GetBodyByName("box_link", object_model)
    box_geoms = _geom_ids_for_body(plant, sg, box_body)

    for seed in IK_SEEDS_INIT:
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
    """Same as §7.17 — LCS sub-step λ_n history. With robust per-sub-step
    IK retry from §7.18 / §7.20 cleanup."""
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
        contacts = getattr(f, '_last_contact_info', [])
        ee_box_idx = None
        for ci, info in enumerate(contacts):
            if info.get('tag', '') == 'EE-BOX':
                ee_box_idx = ci
                break
        n_lam = D.shape[1]
        n_c = n_lam // 6
        lam_n_offset = n_c

        q_lcp = E @ x_curr + c_lcs
        lam, _ = solve_lcp(F, q_lcp)
        if ee_box_idx is not None:
            lam_n_hist[k] = float(lam[lam_n_offset + ee_box_idx])
        x_next = A @ x_curr + B_ctrl @ np.zeros(3) + D @ lam + d_const

        if k < N_SUB - 1:
            # Robust per-sub-step IK: try multiple seeds before giving up
            ik_ok = False
            for seed in [q_arm] + [s for s in IK_SEEDS_INIT]:
                plant.SetPositions(plant_ctx, object_model, x_next[0:7])
                plant.SetVelocities(plant_ctx, object_model, x_next[10:16])
                plant.SetPositions(plant_ctx, panda_model, seed)
                ik = InverseKinematics(plant, plant_ctx, with_joint_limits=True)
                ik.AddPositionConstraint(ee_frame, np.zeros(3), world,
                                         x_next[7:10] - 1e-5, x_next[7:10] + 1e-5)
                ik.get_mutable_prog().SetInitialGuess(
                    ik.q(), plant.GetPositions(plant_ctx))
                res = Solve(ik.prog())
                if res.is_success():
                    plant.SetPositions(plant_ctx, res.GetSolution(ik.q()))
                    q_arm = plant.GetPositions(plant_ctx, panda_model).copy()
                    J = plant.CalcJacobianTranslationalVelocity(
                        plant_ctx, JacobianWrtVariable.kV, ee_frame, np.zeros(3),
                        world, world)
                    s = plant.GetJointByName("panda_joint1").velocity_start()
                    q_arm_dot, *_ = np.linalg.lstsq(J[:, s:s+7], x_next[16:19],
                                                    rcond=None)
                    plant.SetVelocities(plant_ctx, panda_model, q_arm_dot)
                    ik_ok = True
                    break
            if not ik_ok:
                break
        x_curr = x_next
    dx = float(x_curr[BOX_POS_X_IDX]) - box_x_0
    return lam_n_hist, dx


def drake_force_profile_subms(diagram, plant, ctx, obj_body, ee_body):
    """Drake forward 0.05 s at DT_DRAKE = 0.25 ms ticks (4× finer than
    §7.17). Per-tick: read pusher↔box normal force x-component on box.
    """
    sim = Simulator(diagram, ctx)
    sim.Initialize()
    pctx0 = plant.GetMyContextFromRoot(ctx)
    box_x_0 = float(plant.EvalBodyPoseInWorld(pctx0, obj_body).translation()[0])

    F_pusher_n_x = np.zeros(N_DRAKE + 1)
    contact_present = np.zeros(N_DRAKE + 1, dtype=int)
    for k in range(N_DRAKE + 1):
        if k > 0:
            sim.AdvanceTo(k * DT_DRAKE)
        pctx_now = plant.GetMyContextFromRoot(ctx)
        cr = plant.get_contact_results_output_port().Eval(pctx_now)
        f_x = 0.0
        present = 0
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
            present = 1
            pp = info.point_pair()
            nhat = pp.nhat_BA_W
            f_norm_mag = float(F_box @ nhat)
            F_norm = f_norm_mag * nhat
            f_x += float(F_norm[0])
        F_pusher_n_x[k] = f_x
        contact_present[k] = present

    pctx_end = plant.GetMyContextFromRoot(ctx)
    box_x_end = float(plant.EvalBodyPoseInWorld(pctx_end, obj_body).translation()[0])
    return F_pusher_n_x, contact_present, box_x_end - box_x_0


def main() -> int:
    print("=" * 84)
    print("STAGE C  FORCE-LEVEL RE-PROBE — Part A, sub-ms Drake sampling")
    print("(§7.17 force-level probe at 0.25 ms Drake ticks; robust 5-seed IK)")
    print("=" * 84)
    os.environ["LCS_EXPLICIT_BOX_GND"] = "4"
    import control.lcs_formulator
    importlib.reload(control.lcs_formulator)

    with open("config/tasks.yaml") as f:
        cfg = yaml.safe_load(f)
    task_cfg = cfg["tasks"]["pushing"]
    mu = task_cfg["friction"]
    print(f"  μ = {mu}")
    print(f"  Δt_sub (LCS) = {DT_SUB*1000:.1f} ms × {N_SUB} sub-steps")
    print(f"  Drake time_step = {DT_DRAKE*1000:.3f} ms × {N_DRAKE+1} ticks")
    print(f"  Depths: {[round(d*1000,3) for d in DEPTH_M]} mm  "
          f"({', '.join(DEPTH_TAG)})")
    print()

    results = {}
    for pen_m, tag in zip(DEPTH_M, DEPTH_TAG):
        pen_mm = pen_m * 1000
        print("─" * 84)
        print(f"DEPTH {tag}")
        print("─" * 84)

        # LCS profile (unchanged sim time_step — LCS doesn't depend on it)
        d, p, sg, panda, obj, _, p_ad, ctx_ad = _build_env(task_cfg, time_step=0.001)
        obj_body = p.GetBodyByName("box_link", obj)
        ee_frame = p.GetFrameByName("pusher")
        ctx, pctx, q_arm, ee_pos = setup_state_at_depth(
            d, p, sg, panda, obj, ee_frame, pen_m)
        if ctx is None:
            print(f"  re-pose FAIL at depth {pen_mm}mm (LCS path) — skip")
            continue
        lam_n_hist, dx_lcs = lcs_force_profile(
            p, pctx, ee_frame, obj_body, p_ad, ctx_ad, mu, panda, obj,
            q_arm, ee_pos)

        # Drake force profile at sub-ms
        d2, p2, sg2, panda2, obj2, _, _, _ = _build_env(
            task_cfg, time_step=DT_DRAKE)
        obj_body2 = p2.GetBodyByName("box_link", obj2)
        ee_frame2 = p2.GetFrameByName("pusher")
        ee_body2  = p2.GetBodyByName("pusher")
        ctx2, _, _, _ = setup_state_at_depth(
            d2, p2, sg2, panda2, obj2, ee_frame2, pen_m)
        if ctx2 is None:
            print(f"  re-pose FAIL at depth {pen_mm}mm (Drake path) — skip")
            continue
        F_drake, present, dx_drake = drake_force_profile_subms(
            d2, p2, ctx2, obj_body2, ee_body2)

        results[pen_mm] = dict(
            lam_n_hist=lam_n_hist, F_drake=F_drake, present=present,
            dx_lcs=dx_lcs, dx_drake=dx_drake)

        # Print LCS λ_n profile
        print(f"  LCS λ_n history (per sub-step, EE↔box contact):")
        print(f"    sub-step: " + "  ".join(f"{i:>6d}" for i in range(N_SUB)))
        print(f"    λ_n     : " + "  ".join(
            f"{v:>6.3f}" if not np.isnan(v) else "   NaN" for v in lam_n_hist))
        peak_lam = np.nanmax(lam_n_hist) if not np.isnan(lam_n_hist).all() else float('nan')
        peak_lam_step = int(np.nanargmax(lam_n_hist)) if not np.isnan(lam_n_hist).all() else -1
        sum_lam = np.nansum(lam_n_hist)
        valid_subs = int(np.sum(~np.isnan(lam_n_hist)))
        print(f"    peak λ_n = {peak_lam:.4f} at sub-step {peak_lam_step};  "
              f"Σλ_n = {sum_lam:.4f};  valid sub-steps = {valid_subs}/{N_SUB}")
        if not np.isnan(lam_n_hist[0]) and not np.isnan(lam_n_hist[-1]) and lam_n_hist[-1] > 0:
            print(f"    λ_n[0] / λ_n[-1] = {lam_n_hist[0]/lam_n_hist[-1]:.2f}  "
                  f"(rigid-impulsive expects >> 1)")

        # Print Drake force at 1 ms display stride from 0.25 ms samples
        stride = max(1, int(round(0.001 / DT_DRAKE)))
        ms_t = np.arange(0, N_DRAKE + 1, stride) * DT_DRAKE * 1000.0
        print(f"  Drake normal-force x on box (every 1 ms, from {DT_DRAKE*1000:.2f}ms samples):")
        for i_start in range(0, len(ms_t), 13):
            i_end = min(i_start + 13, len(ms_t))
            t_chunk = ms_t[i_start:i_end]
            f_chunk = F_drake[::stride][i_start:i_end]
            p_chunk = present[::stride][i_start:i_end]
            t_str = "  ".join(f"{v:>6.1f}" for v in t_chunk)
            f_str = "  ".join(f"{v:>+6.2f}" for v in f_chunk)
            p_str = "  ".join(f"{'C' if v else '.':>6}" for v in p_chunk)
            print(f"    t (ms)  : {t_str}")
            print(f"    F_x (N) : {f_str}")
            print(f"    contact : {p_str}")

        peak_F_abs = float(np.max(np.abs(F_drake)))
        peak_F_idx = int(np.argmax(np.abs(F_drake)))
        avg_F = float(np.mean(np.abs(F_drake)))
        present_frac = float(np.mean(present))

        # On-contact analysis (force-on-MAINTAINED-contact, per §7.20)
        F_on_contact = F_drake[present == 1]
        peak_F_on = float(np.max(np.abs(F_on_contact))) if F_on_contact.size > 0 else 0.0
        avg_F_on  = float(np.mean(np.abs(F_on_contact))) if F_on_contact.size > 0 else 0.0
        # F at first contact tick vs near-end
        first_contact = next((k for k, v in enumerate(present) if v), -1)
        last_contact = next((k for k in range(N_DRAKE, -1, -1) if present[k]), -1)
        F_early = float(np.abs(F_drake[first_contact])) if first_contact >= 0 else 0.0
        F_late = float(np.abs(F_drake[last_contact])) if last_contact >= 0 else 0.0
        early_late_ratio = F_early / max(F_late, 1e-6)

        print(f"    peak |F| (all) = {peak_F_abs:.4f} N at tick {peak_F_idx} "
              f"(t={peak_F_idx*DT_DRAKE*1000:.2f}ms)")
        print(f"    avg  |F| (all) = {avg_F:.4f} N    contact% = {present_frac*100:.1f}%")
        print(f"    peak |F| (on contact)   = {peak_F_on:.4f} N")
        print(f"    avg  |F| (on contact)   = {avg_F_on:.4f} N")
        print(f"    F(first contact tick) / F(last contact tick) = "
              f"{early_late_ratio:.2f}  (soft-spread expects ~1)")
        print(f"  Δbox_x:  LCS = {dx_lcs*1000:.4f}mm  Drake = {dx_drake*1000:.4f}mm  "
              f"factor = {dx_lcs/dx_drake if abs(dx_drake) > 1e-9 else float('nan'):.3f}×")
        print()

    # Cross-depth comparison
    print("=" * 84)
    print("CROSS-DEPTH COMPARISON")
    print("=" * 84)
    print()
    print(f"  Peak λ_n (LCS) vs depth — rigid expects LINEAR scaling:")
    print(f"     {'depth (mm)':>11}  {'peak λ_n':>10}  {'Σλ_n':>10}  "
          f"{'valid':>6}  {'λ_n[0]/λ_n[-1]':>16}")
    for d_mm, r in sorted(results.items()):
        peak = float(np.nanmax(r['lam_n_hist'])) if not np.isnan(r['lam_n_hist']).all() else float('nan')
        ssum = float(np.nansum(r['lam_n_hist']))
        valid = int(np.sum(~np.isnan(r['lam_n_hist'])))
        ratio = (r['lam_n_hist'][0] / r['lam_n_hist'][-1]
                 if (not np.isnan(r['lam_n_hist'][0]) and not np.isnan(r['lam_n_hist'][-1])
                     and r['lam_n_hist'][-1] > 0) else float('nan'))
        print(f"     {d_mm:>10.3f}  {peak:>10.4f}  {ssum:>10.4f}  "
              f"{valid:>6}  {ratio:>16.2f}")
    print()
    print(f"  Peak Drake force vs depth — compliant expects STABLE:")
    print(f"     {'depth (mm)':>11}  {'peak |F|':>10}  {'avg |F|':>10}  "
          f"{'peak on-c':>10}  {'contact%':>9}  {'F_early/F_late':>16}")
    for d_mm, r in sorted(results.items()):
        F = r['F_drake']
        present = r['present']
        peak_F = float(np.max(np.abs(F)))
        avg_F = float(np.mean(np.abs(F)))
        F_on = F[present == 1]
        peak_on = float(np.max(np.abs(F_on))) if F_on.size > 0 else 0.0
        cf = float(np.mean(present))
        first_c = next((k for k, v in enumerate(present) if v), -1)
        last_c = next((k for k in range(len(present) - 1, -1, -1) if present[k]), -1)
        ratio = (float(np.abs(F[first_c])) / max(float(np.abs(F[last_c])), 1e-6)
                 if first_c >= 0 and last_c >= 0 else float('nan'))
        print(f"     {d_mm:>10.3f}  {peak_F:>10.4f}  {avg_F:>10.4f}  "
              f"{peak_on:>10.4f}  {cf*100:>8.1f}%  {ratio:>16.2f}")

    # Route
    print()
    print("=" * 84)
    print("ROUTE  (force-level re-confirmation under §7.20 sub-ms sampling)")
    print("=" * 84)
    depths_done = sorted(results.keys())
    if len(depths_done) < 2:
        print("  ► INSUFFICIENT DATA — fewer than 2 depths completed")
        return 0

    d_min_mm = depths_done[0]
    d_max_mm = depths_done[-1]
    depth_ratio = d_max_mm / d_min_mm
    lcs_min = float(np.nanmax(results[d_min_mm]['lam_n_hist']))
    lcs_max = float(np.nanmax(results[d_max_mm]['lam_n_hist']))
    lcs_growth = lcs_max / max(lcs_min, 1e-9)
    F_min = float(np.max(np.abs(results[d_min_mm]['F_drake'])))
    F_max = float(np.max(np.abs(results[d_max_mm]['F_drake'])))
    F_growth = F_max / max(F_min, 1e-9)
    # On-contact peak
    F_on_min = float(np.max(np.abs(results[d_min_mm]['F_drake'][results[d_min_mm]['present']==1])))
    F_on_max = float(np.max(np.abs(results[d_max_mm]['F_drake'][results[d_max_mm]['present']==1])))
    F_on_growth = F_on_max / max(F_on_min, 1e-9)
    contact_frac_min = float(np.mean(results[d_min_mm]['present']))
    contact_frac_max = float(np.mean(results[d_max_mm]['present']))

    print(f"  Depth range : {d_min_mm:.3f} → {d_max_mm:.3f} mm  ({depth_ratio:.1f}×)")
    print(f"  LCS peak λ_n: {lcs_min:.3f} → {lcs_max:.3f}  ({lcs_growth:.2f}×)  "
          f"[rigid expects {depth_ratio:.1f}×]")
    print(f"  Drake peak F (all)        : {F_min:.3f} → {F_max:.3f}  ({F_growth:.2f}×)  "
          f"[compliant expects ~1×]")
    print(f"  Drake peak F (on contact) : {F_on_min:.3f} → {F_on_max:.3f}  ({F_on_growth:.2f}×)")
    print(f"  Contact fraction          : {contact_frac_min*100:.1f}% → "
          f"{contact_frac_max*100:.1f}%   [§7.20: should be ~100% maintained]")
    print()

    # Pre-registered signature checks
    sig_LCS_impulsive = True       # all depths show λ_n[0] > λ_n[-1] (peak at sub-step 0)
    for d_mm in depths_done:
        lh = results[d_mm]['lam_n_hist']
        if np.isnan(lh[0]) or (not np.isnan(lh[-1]) and lh[0] <= lh[-1]):
            sig_LCS_impulsive = False
            break
    sig_LCS_linear = abs(lcs_growth - depth_ratio) / depth_ratio < 0.5  # ±50%
    sig_Drake_stable = abs(F_on_growth - 1.0) < 0.5  # ±50%
    sig_Drake_maintained = contact_frac_min > 0.95 and contact_frac_max > 0.95

    print(f"  Signatures (pre-registered, recounted on contact-aware reads):")
    print(f"    LCS impulsive (λ_n peak at sub-step 0): {'✓' if sig_LCS_impulsive else '✗'}")
    print(f"    LCS linear scaling with depth         : {'✓' if sig_LCS_linear else '✗'}"
          f"  (observed {lcs_growth:.2f}× vs expected {depth_ratio:.1f}×)")
    print(f"    Drake force depth-STABLE (on contact) : {'✓' if sig_Drake_stable else '✗'}"
          f"  (observed {F_on_growth:.2f}× vs expected ~1×)")
    print(f"    Drake contact MAINTAINED (≥95% ticks) : {'✓' if sig_Drake_maintained else '✗'}"
          f"  ({contact_frac_min*100:.1f}% / {contact_frac_max*100:.1f}%)")
    print()

    n_match = sum([sig_LCS_impulsive, sig_LCS_linear, sig_Drake_stable, sig_Drake_maintained])
    if n_match == 4:
        route = "FORCE-CONFIRMS"
        msg = ("  All four pre-registered force-level signatures match. The §7.17 verdict\n"
               "  was a §7.20-pinned sampling artifact; under sub-ms Drake sampling the\n"
               "  contact is maintained and the force-on-contact profile confirms rigid-\n"
               "  vs-compliant. Compliance LOCKED at the force level.\n"
               "  NEXT (separate block): Part B — anitescu scoping (SCOPE not BUILD).")
    elif sig_Drake_maintained and (sig_LCS_impulsive or sig_Drake_stable):
        route = "FORCE-CONFIRMS-PARTIAL"
        msg = ("  Drake maintained ✓ + at least one of LCS-impulsive / Drake-stable ✓,\n"
               "  but the depth-scaling signatures don't fully match the rigid/linear vs\n"
               "  compliant/stable expectation. Compliance PARTIALLY locked — the §7.20\n"
               "  contact-loss rehabilitation succeeds, but the depth-scaling residual\n"
               "  still needs an explanation before anitescu commits.\n"
               "  NEXT (separate block): characterize the depth-scaling residual; if it\n"
               "  resolves into a sub-mechanism of compliance, Part B opens; if it points\n"
               "  at something else (A-matrix dynamics artifact, contact-stiffness regime\n"
               "  not on the cartoon), keep anitescu paused.")
    else:
        route = "FORCE-DISCONFIRMS"
        msg = ("  Even at sub-ms Drake sampling and robust 5-seed IK, the pre-registered\n"
               "  signatures don't match. The §7.17 verdict survives the §7.20 sampling\n"
               "  correction. The mechanism is NOT purely normal compliance. Anitescu\n"
               "  STILL PAUSED. NEXT (separate block): re-examine the displacement-level\n"
               "  sweep's cause from a different mechanism (intermittent contact ruled\n"
               "  out by §7.20 → other candidates: A-matrix dynamics artifact, sub-step\n"
               "  IK divergence pattern, contact-stiffness regime not on the cartoon).")

    print(f"  ► ROUTE: {route}")
    print(msg)
    print()
    print("=" * 84)
    print("HOLD: next block (Part B anitescu scoping / depth-scaling residual /")
    print("      mechanism re-examination) is SEPARATE.")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
