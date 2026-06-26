"""Stage C LCS residual probe — Part A (clear 1mm partial confound under
§7.18 EXACT robust IK) + Part B (λ_t coupling check on the clean data).

Routed from §7.21 banking. At the force level we have 3-of-4 signatures
confirmed; the survivor is "LCS λ_n LINEAR depth-scaling" — peak λ_n
grows 2.65× over 10× depth (0.10mm → 1.00mm), not the 10× a pure-rigid
cartoon predicts. Two confounds:
  (a) the 1.00mm re-probe LCS run completed only 4/10 sub-steps (sub-
      step 4 IK fails); §7.18's exact robust-IK formulation was NOT
      matched in the re-probe.
  (b) the displacement-crossing depth may be dt-dependent (HOLD).

This probe addresses (a) and the residual mechanism, ONLY:

PART A — clear the 1mm partial confound. Replace the re-probe's fixed-
seed IK retry with §7.18 sweep-cleanup's EXACT warm-aware perturbation
recipe: [q_arm_warm, POSTURE_NOMINAL, q_arm_warm+rand1, +rand2, +rand3]
(rng seed=0, ±0.1 rad per joint). KEY READ: with full 10/10 sub-steps
at 1.00mm, does peak-λ_n depth-scaling stay ~2.65× or MOVE?

PART B — λ_t coupling check. Stewart-Trinkle couples normal and
tangential through the friction cone (Σλ_t ≤ μ·λ_n per contact).
HYPOTHESIS: at deeper penetration, more of the contact-impulse channel
into box-x velocity may route through the J_t^T·λ_t channel rather
than J_n^T·λ_n, so peak λ_n grows sub-linearly while total channel
contribution (D[v_box_x] @ λ) scales closer to linear. KEY READ at
sub-step 0 across depths: does the total D[v_box_x] @ λ scale linearly
while the normal-channel-only contribution scales sub-linearly?

Pre-registered routes (the next block scopes against these — this
probe does NOT execute the next block):
  RESOLVES-INTO-COMPLIANCE — Part A scaling stays sub-linear on clean
    data AND Part B λ_t accounts (total impulse channel ~linear, the
    normal-channel-only sub-linearity is the friction-cone coupling).
    HONESTY-FLAG: this does NOT mint a 4th "confirm" for Sig 2.
    Sig 2 as literally pre-registered ("LCS λ_n LINEAR normal scaling")
    STAYS DISCONFIRMED. The lock rests on the disconfirmation being
    UNDERSTOOD as a Stewart-Trinkle friction-cone refinement, not on a
    relabeling. The same anti-stale binding applied to "FORCE-CONFIRMS-
    PARTIAL" (§7.21 (3)) and "MOSTLY-CONTINUOUS-WITH-GRAZING" (§7.19
    (5)) applies here.
  IS-DYNAMICS — λ_t does NOT account → check A-matrix box-velocity
    propagation (the sub-linearity may live in the A·x channel rather
    than the D·λ channel — a dynamics-propagation effect, not a
    contact-model effect). This probe leaves the A-matrix check to the
    next block.
  PERSISTS-UNEXPLAINED — neither λ_t-coupling nor A-matrix accounts on
    clean data → mechanism reopens beyond pure normal compliance;
    anitescu stays paused.
  1mm-STILL-FAILS — Part A can't recover full 1mm even with §7.18 IK
    → sweep is BOUNDED to where IK completes; the 2.65× is undetermined
    at 1mm; characterization is bounded to ≤0.7mm.
"""
from __future__ import annotations

import os
import importlib
import numpy as np
import yaml

from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.multibody.tree import JacobianWrtVariable
from pydrake.solvers import Solve
from pydrake.geometry import Role
from pydrake.math import RotationMatrix

from sim.env_builder import build_environment
from control.lcp_solver import solve_lcp

DEPTH_M   = [0.0001, 0.000549, 0.001]
DEPTH_TAG = ["UNDER (0.10mm)", "SWEET (0.549mm)", "OVER (1.00mm)"]

DT_BIG    = 0.05
DT_SUB    = 0.005
N_SUB     = int(round(DT_BIG / DT_SUB))

BOX_HALF  = 0.05
EE_RADIUS = 0.025
BOX_QUAT  = np.array([1.0, 0.0, 0.0, 0.0])
BOX_POS   = np.array([0.0, 0.0, 0.05])
EE_VEL_X  = -0.05

BOX_POS_X_IDX = 4
BOX_V_X_IDX   = 13

ARM_BOX_CLEARANCE_M = 0.005
ARM_LINKS_TO_CLEAR  = ["panda_link4", "panda_link5", "panda_link6", "panda_link7"]
POSTURE_NOMINAL = np.array([0.0, -0.4, 0.0, -1.8, 0.0, 1.4, 0.785])
POSTURE_WEIGHT = 5.0
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
    """Same robust 5-seed re-pose used by §7.18/§7.20/§7.21."""
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


def _try_substep_ik_warm_aware(plant, plant_ctx, ee_frame, panda_model,
                                object_model, x_next, q_arm_warm, world):
    """§7.18 sweep-cleanup's EXACT recipe (warm + posture + 3 perturbations
    of warm, ±0.1 rad per joint, rng seed=0). This is the IK formulation
    the §7.21 re-probe DID NOT match.
    """
    seeds = [q_arm_warm.copy(), POSTURE_NOMINAL.copy()]
    rng = np.random.default_rng(seed=0)
    for _ in range(3):
        perturbed = q_arm_warm + rng.uniform(-0.1, 0.1, size=7)
        seeds.append(perturbed)

    for seed_idx, seed_q in enumerate(seeds):
        plant.SetPositions(plant_ctx, object_model, x_next[0:7])
        plant.SetVelocities(plant_ctx, object_model, x_next[10:16])
        plant.SetPositions(plant_ctx, panda_model, seed_q)
        ik = InverseKinematics(plant, plant_ctx, with_joint_limits=True)
        ik.AddPositionConstraint(ee_frame, np.zeros(3), world,
                                 x_next[7:10] - 1e-5, x_next[7:10] + 1e-5)
        ik.get_mutable_prog().SetInitialGuess(
            ik.q(), plant.GetPositions(plant_ctx))
        res = Solve(ik.prog())
        if res.is_success():
            plant.SetPositions(plant_ctx, res.GetSolution(ik.q()))
            q_arm_new = plant.GetPositions(plant_ctx, panda_model).copy()
            p_ee = plant.CalcPointsPositions(
                plant_ctx, ee_frame, np.zeros(3), world).flatten()
            if np.linalg.norm(p_ee - x_next[7:10]) < 1e-3:
                J = plant.CalcJacobianTranslationalVelocity(
                    plant_ctx, JacobianWrtVariable.kV, ee_frame, np.zeros(3),
                    world, world)
                s = plant.GetJointByName("panda_joint1").velocity_start()
                q_arm_dot, *_ = np.linalg.lstsq(J[:, s:s+7], x_next[16:19],
                                                rcond=None)
                plant.SetVelocities(plant_ctx, panda_model, q_arm_dot)
                return q_arm_new, seed_idx
    return None, -1


def lcs_residual_profile(plant, plant_ctx, ee_frame, obj_body, plant_ad,
                          ctx_ad, mu, panda_model, object_model,
                          q_arm_init, ee_pos):
    """Per-sub-step LCS pull at sub-step k:
       lam_n_hist[k]            : EE-BOX normal force
       lam_t_sum_hist[k]        : EE-BOX Σλ_t over the 4 polyhedral edges
       contrib_n_x_hist[k]      : D[box_v_x, n_c:2n_c] @ λ_n_vec   (box-v_x channel from normals)
       contrib_t_x_hist[k]      : D[box_v_x, 2n_c:6n_c] @ λ_t_vec  (box-v_x channel from tangents)
       contrib_tot_x_hist[k]    : D[box_v_x, :] @ λ                (total box-v_x channel from impulse)
    Uses §7.18 EXACT warm-aware IK retry.
    """
    from control.lcs_formulator import LCSFormulator
    world = plant.world_frame()
    x_curr = np.concatenate([
        BOX_QUAT, BOX_POS, ee_pos, np.zeros(3), np.zeros(3),
        np.array([EE_VEL_X, 0., 0.])])
    box_x_0 = float(x_curr[BOX_POS_X_IDX])
    q_arm = q_arm_init.copy()
    lam_n_hist        = np.full(N_SUB, np.nan)
    lam_t_sum_hist    = np.full(N_SUB, np.nan)
    contrib_n_x_hist  = np.full(N_SUB, np.nan)
    contrib_t_x_hist  = np.full(N_SUB, np.nan)
    contrib_tot_x_hist= np.full(N_SUB, np.nan)
    n_completed = 0
    fail_step = -1
    seeds_used = []

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
        SLN = n_c
        SLT = 2 * n_c

        q_lcp = E @ x_curr + c_lcs
        lam, _ = solve_lcp(F, q_lcp)
        if ee_box_idx is not None:
            lam_n_hist[k] = float(lam[SLN + ee_box_idx])
            lam_t_sum_hist[k] = float(np.sum(
                lam[SLT + 4 * ee_box_idx : SLT + 4 * (ee_box_idx + 1)]))
        # Channel decomposition into box_v_x (LCS state index 13)
        D_row = D[BOX_V_X_IDX, :]
        contrib_n_x_hist[k]   = float(D_row[SLN:SLT] @ lam[SLN:SLT])
        contrib_t_x_hist[k]   = float(D_row[SLT:]    @ lam[SLT:])
        contrib_tot_x_hist[k] = float(D_row @ lam)
        x_next = A @ x_curr + B_ctrl @ np.zeros(3) + D @ lam + d_const

        if k < N_SUB - 1:
            new_q, seed_idx = _try_substep_ik_warm_aware(
                plant, plant_ctx, ee_frame, panda_model, object_model,
                x_next, q_arm, world)
            seeds_used.append(seed_idx)
            if new_q is None:
                fail_step = k + 1
                break
            q_arm = new_q
        x_curr = x_next
        n_completed = k + 1

    status = "CLEAN" if n_completed == N_SUB else "PARTIAL"
    dx = float(x_curr[BOX_POS_X_IDX]) - box_x_0
    return dict(lam_n_hist=lam_n_hist, lam_t_sum_hist=lam_t_sum_hist,
                contrib_n_x_hist=contrib_n_x_hist,
                contrib_t_x_hist=contrib_t_x_hist,
                contrib_tot_x_hist=contrib_tot_x_hist,
                dx=dx, n_completed=n_completed, fail_step=fail_step,
                seeds_used=seeds_used, status=status, mu=mu)


def main() -> int:
    print("=" * 84)
    print("STAGE C  LCS RESIDUAL PROBE — Part A (1mm confound) + Part B (λ_t coupling)")
    print("(§7.18 EXACT warm-aware IK; LCS-only; sub-step 0 λ_t + box-v_x channel split)")
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
    print(f"  Depths: {[round(d*1000,3) for d in DEPTH_M]} mm  "
          f"({', '.join(DEPTH_TAG)})")
    print(f"  IK formulation: §7.18 warm-aware perturbation (warm + posture")
    print(f"                  + 3 random ±0.1 rad/joint perturbations of warm)")
    print()

    results = {}
    for pen_m, tag in zip(DEPTH_M, DEPTH_TAG):
        pen_mm = pen_m * 1000
        print("─" * 84)
        print(f"DEPTH {tag}")
        print("─" * 84)

        d, p, sg, panda, obj, _, p_ad, ctx_ad = _build_env(task_cfg, time_step=0.001)
        obj_body = p.GetBodyByName("box_link", obj)
        ee_frame = p.GetFrameByName("pusher")
        ctx, pctx, q_arm, ee_pos = setup_state_at_depth(
            d, p, sg, panda, obj, ee_frame, pen_m)
        if ctx is None:
            print(f"  re-pose FAIL at depth {pen_mm}mm — skip")
            continue
        res = lcs_residual_profile(
            p, pctx, ee_frame, obj_body, p_ad, ctx_ad, mu, panda, obj,
            q_arm, ee_pos)
        results[pen_mm] = res

        lh = res['lam_n_hist']
        lt = res['lam_t_sum_hist']
        cn = res['contrib_n_x_hist']
        ct = res['contrib_t_x_hist']
        cT = res['contrib_tot_x_hist']

        print(f"  status            : {res['status']}  ({res['n_completed']}/{N_SUB} sub-steps)")
        if res['fail_step'] >= 0:
            print(f"  fail at sub-step  : {res['fail_step']}  "
                  f"(IK couldn't recover via §7.18 5-seed retry)")
        print(f"  seeds used (-1=fail; 0=warm, 1=nominal, 2-4=perturbations):")
        print(f"    " + "  ".join(f"{s:>2d}" for s in res['seeds_used']))
        print()
        print(f"  per-sub-step λ + impulse-channel split into box_v_x:")
        print(f"    {'k':>3}  {'λ_n':>9}  {'Σλ_t':>9}  {'μ·λ_n':>9}  "
              f"{'D@λ_n→vx':>10}  {'D@λ_t→vx':>10}  {'D@λ→vx':>10}")
        for k in range(N_SUB):
            ln  = lh[k]; lts = lt[k]
            mln = mu * ln if not np.isnan(ln) else float('nan')
            cnk = cn[k]; ctk = ct[k]; cTk = cT[k]
            def fmt(v, w=9, dp=4):
                return f"{v:>{w}.{dp}f}" if not np.isnan(v) else (" " * (w-3) + "NaN")
            print(f"    {k:>3}  {fmt(ln)}  {fmt(lts)}  {fmt(mln)}  "
                  f"{fmt(cnk,10,5)}  {fmt(ctk,10,5)}  {fmt(cTk,10,5)}")
        peak_lam_n = float(np.nanmax(lh)) if not np.isnan(lh).all() else float('nan')
        sum_lam_n  = float(np.nansum(lh))
        sum_lam_t  = float(np.nansum(lt))
        print(f"  peak λ_n = {peak_lam_n:.4f}  ;  Σλ_n = {sum_lam_n:.4f}  ;  "
              f"Σ(Σλ_t) = {sum_lam_t:.4f}")
        print(f"  Δbox_x = {res['dx']*1000:.4f} mm")
        print()

    # ------- CROSS-DEPTH COMPARISON --------
    print("=" * 84)
    print("CROSS-DEPTH COMPARISON")
    print("=" * 84)
    depths_done = sorted(results.keys())
    if len(depths_done) < 2:
        print("  ► INSUFFICIENT DATA — fewer than 2 depths completed")
        return 0

    print()
    print(f"  Status summary:")
    print(f"    {'depth (mm)':>11}  {'status':>8}  {'completed':>9}  "
          f"{'fail@':>5}")
    for d_mm in depths_done:
        r = results[d_mm]
        fs = r['fail_step'] if r['fail_step'] >= 0 else -1
        print(f"    {d_mm:>10.3f}  {r['status']:>8}  {r['n_completed']:>9}  "
              f"{fs:>5}")
    print()

    # Part A core read: peak-λ_n scaling on CLEAN data
    clean_depths = [d for d in depths_done if results[d]['status'] == 'CLEAN']
    print(f"  CLEAN depths ({len(clean_depths)}/{len(depths_done)}): {clean_depths}")

    if len(clean_depths) >= 2:
        d_lo = clean_depths[0]; d_hi = clean_depths[-1]
        peak_n_lo = float(np.nanmax(results[d_lo]['lam_n_hist']))
        peak_n_hi = float(np.nanmax(results[d_hi]['lam_n_hist']))
        ratio_peak_n = peak_n_hi / max(peak_n_lo, 1e-9)
        depth_ratio = d_hi / d_lo
        print(f"  Peak λ_n scaling (CLEAN, {d_lo}mm → {d_hi}mm, {depth_ratio:.1f}× depth):")
        print(f"    peak λ_n  : {peak_n_lo:.4f} → {peak_n_hi:.4f}  "
              f"({ratio_peak_n:.2f}×)   [rigid expects {depth_ratio:.1f}×]")

    # Part A vs §7.21 re-probe baseline (which used fixed-seed IK)
    print()
    print(f"  Cross-reference vs §7.21 (force-level re-probe, fixed-seed IK):")
    print(f"    §7.21 peak λ_n: 0.10mm → 1.00mm = 3.470 → 9.200  (2.65× over 10× depth)")
    if 0.1 in [d for d in depths_done] and 1.0 in [d for d in depths_done]:
        p010 = float(np.nanmax(results[0.1]['lam_n_hist']))
        p100 = float(np.nanmax(results[1.0]['lam_n_hist']))
        ratio_010_100 = p100 / max(p010, 1e-9)
        print(f"    §7.22 peak λ_n: 0.10mm → 1.00mm = {p010:.3f} → {p100:.3f}  "
              f"({ratio_010_100:.2f}×)   [Part A comparison]")
        delta_pct = abs(ratio_010_100 - 2.65) / 2.65 * 100
        if results[1.0]['status'] == 'CLEAN':
            if delta_pct < 25:
                a_route = "STAYS"
                a_msg = (f"Part A: 1mm CLEAN 10/10, scaling STAYS ~2.65× "
                         f"({ratio_010_100:.2f}× vs 2.65×, Δ={delta_pct:.1f}%); "
                         "the residual is REAL on clean data — proceed to Part B.")
            else:
                a_route = "MOVES"
                a_msg = (f"Part A: 1mm CLEAN 10/10, but scaling MOVES "
                         f"({ratio_010_100:.2f}× vs 2.65×, Δ={delta_pct:.1f}%); "
                         "the 2.65× was partly partial-run artifact — "
                         "re-characterize the clean scaling.")
        else:
            a_route = "1mm-STILL-FAILS"
            a_msg = (f"Part A: 1mm did NOT recover full 10/10 sub-steps under "
                     f"§7.18 EXACT IK either (status={results[1.0]['status']}, "
                     f"completed {results[1.0]['n_completed']}/{N_SUB}). The 2.65× "
                     "is undetermined at 1mm; characterization is BOUNDED.")
    else:
        a_route = "INCOMPLETE-COVERAGE"
        a_msg = "Part A: 0.10mm or 1.00mm not in completed depths."

    # ------- PART B — λ_t-coupling read on sub-step 0, CLEAN data --------
    print()
    print("=" * 84)
    print("PART B — λ_t-coupling read at sub-step 0 (impulse channel into box_v_x)")
    print("=" * 84)
    print()
    print(f"  Sub-step 0 reads on the CLEAN depths:")
    print(f"    {'depth (mm)':>11}  {'λ_n[0]':>9}  {'Σλ_t[0]':>9}  "
          f"{'D@λ_n→vx':>10}  {'D@λ_t→vx':>10}  {'D@λ→vx':>10}  {'t/n':>6}")
    for d_mm in clean_depths:
        r = results[d_mm]
        ln0  = r['lam_n_hist'][0]; lt0 = r['lam_t_sum_hist'][0]
        cn0  = r['contrib_n_x_hist'][0]; ct0 = r['contrib_t_x_hist'][0]
        cT0  = r['contrib_tot_x_hist'][0]
        tn   = ct0 / cn0 if abs(cn0) > 1e-12 else float('nan')
        print(f"    {d_mm:>10.3f}  {ln0:>9.4f}  {lt0:>9.4f}  "
              f"{cn0:>10.5f}  {ct0:>10.5f}  {cT0:>10.5f}  {tn:>6.3f}")
    print()

    b_route = None
    b_msg   = ""
    if len(clean_depths) >= 2:
        d_lo = clean_depths[0]; d_hi = clean_depths[-1]
        depth_ratio = d_hi / d_lo
        r_lo = results[d_lo]; r_hi = results[d_hi]
        cn_lo = r_lo['contrib_n_x_hist'][0]
        ct_lo = r_lo['contrib_t_x_hist'][0]
        cT_lo = r_lo['contrib_tot_x_hist'][0]
        cn_hi = r_hi['contrib_n_x_hist'][0]
        ct_hi = r_hi['contrib_t_x_hist'][0]
        cT_hi = r_hi['contrib_tot_x_hist'][0]
        # Take absolute values because contrib magnitudes carry sign
        def g(a, b): return abs(b) / max(abs(a), 1e-12)
        g_n_only = g(cn_lo, cn_hi)
        g_t_only = g(ct_lo, ct_hi)
        g_total  = g(cT_lo, cT_hi)
        print(f"  Channel-scaling sub-step 0 ({d_lo}mm → {d_hi}mm, {depth_ratio:.1f}× depth):")
        print(f"    normal-channel into box_v_x   : "
              f"{abs(cn_lo):.5f} → {abs(cn_hi):.5f}  ({g_n_only:.2f}×)  "
              f"[rigid expects {depth_ratio:.1f}×]")
        print(f"    tangent-channel into box_v_x  : "
              f"{abs(ct_lo):.5f} → {abs(ct_hi):.5f}  ({g_t_only:.2f}×)")
        print(f"    TOTAL channel into box_v_x    : "
              f"{abs(cT_lo):.5f} → {abs(cT_hi):.5f}  ({g_total:.2f}×)  "
              f"[Part B: does this scale ~linear with depth?]")
        # ROUTE: λ_t-accounts iff total ~linear (within ±25% of depth-ratio)
        # AND normal-only is sub-linear (otherwise λ_t isn't the explanation).
        linear_total_dev = abs(g_total - depth_ratio) / depth_ratio
        normal_subscaling = g_n_only < 0.7 * depth_ratio
        if linear_total_dev < 0.25 and normal_subscaling:
            b_route = "λt-ACCOUNTS"
            b_msg = (f"Total channel scales ~linearly with depth ({g_total:.2f}× "
                     f"over {depth_ratio:.1f}× depth, Δ={linear_total_dev*100:.1f}%), "
                     f"while normal-only is sub-linear ({g_n_only:.2f}×) — "
                     "the sub-linearity routes through the friction-cone λ_t channel; "
                     "the rigid cartoon's 'linear normal scaling' refines to "
                     "'linear TOTAL impulse with a normal/tangential split.'")
        else:
            b_route = "λt-DOES-NOT-ACCOUNT"
            b_msg = (f"Total channel scales {g_total:.2f}× over {depth_ratio:.1f}× "
                     f"depth (Δ={linear_total_dev*100:.1f}% from linear) — λ_t-coupling "
                     "does NOT close the sub-linearity. The residual lives elsewhere "
                     "(A-matrix dynamics propagation is the next candidate).")
    else:
        b_route = "INSUFFICIENT-CLEAN-DATA"
        b_msg = "Need ≥2 CLEAN depths for the λ_t-coupling read."

    # ------- COMBINED ROUTE --------
    print()
    print("=" * 84)
    print("COMBINED ROUTE")
    print("=" * 84)
    print(f"  Part A : {a_route}")
    print(f"  Part B : {b_route}")
    print()
    print(f"  Part A reads: {a_msg}")
    print()
    print(f"  Part B reads: {b_msg}")
    print()

    if a_route == "STAYS" and b_route == "λt-ACCOUNTS":
        route = "RESOLVES-INTO-COMPLIANCE"
        verdict = (
            "  The sub-linear normal scaling is the Stewart-Trinkle friction-cone\n"
            "  coupling. Compliance is force-level-locked — but on the HONEST basis\n"
            "  that the literal Sig 2 ('LCS λ_n LINEAR normal scaling') STAYS\n"
            "  DISCONFIRMED. The lock rests on the disconfirmation being UNDERSTOOD\n"
            "  (a known physical mechanism consistent with compliance), NOT on minting\n"
            "  a relabeled 4th 'confirm.' Sig 2 was a CARTOON; the right refinement is\n"
            "  'linear TOTAL impulse with a normal/tangential split.' Recorded as a\n"
            "  refinement, NOT as a relabeled confirm.\n"
            "  NEXT (separate block): anitescu Part B RE-PROMOTES TO SCOPING (read the\n"
            "  reference's velocity-level convex compliance construction, scope the\n"
            "  lcs_formulator change as localized-vs-pervasive + flag-stageable,\n"
            "  describe offline validation = gap closes ACROSS depths).")
    elif a_route == "STAYS" and b_route == "λt-DOES-NOT-ACCOUNT":
        route = "IS-DYNAMICS"
        verdict = (
            "  Part A confirms the 2.65× residual on CLEAN data; Part B rules out λ_t-\n"
            "  coupling. The next candidate is A-matrix dynamics: the box-velocity\n"
            "  propagation after the sub-step-0 impulse is sub-linear in λ_n through\n"
            "  the A · x channel rather than the D · λ channel. NEXT (separate block):\n"
            "  measure A-matrix contribution to box_v_x at the linearization point.\n"
            "  Anitescu STAYS PAUSED.")
    elif a_route == "STAYS" and b_route in ("INSUFFICIENT-CLEAN-DATA",):
        route = "PARTIAL-PART-B-INCOMPLETE"
        verdict = "  Part A confirms 2.65× on clean data; Part B coverage insufficient."
    elif a_route == "MOVES":
        route = "PARTIAL-ARTIFACT"
        verdict = (
            "  Part A: the 2.65× was substantially a partial-run artifact; the clean\n"
            "  scaling is different. Re-characterize the clean scaling first; Part B's\n"
            "  λ_t check then runs against the new scaling number. Anitescu PAUSED.")
    elif a_route == "1mm-STILL-FAILS":
        route = "1mm-STILL-FAILS"
        verdict = (
            "  Even §7.18's EXACT robust IK can't recover full 10/10 sub-steps at\n"
            "  1.00mm. The sweep is BOUNDED to depths where IK completes. The 2.65×\n"
            "  is undetermined at 1mm from clean data; the scaling characterization is\n"
            "  bounded to the deepest CLEAN depth. Anitescu STAYS PAUSED.")
    else:
        route = "PERSISTS-UNEXPLAINED"
        verdict = ("  Neither λ_t-coupling nor a clear partial-artifact route fires;\n"
                   "  characterize further. Anitescu STAYS PAUSED.")

    print(f"  ► ROUTE: {route}")
    print(verdict)
    print()
    print("=" * 84)
    print("HOLD: next block (anitescu scoping / A-matrix check / re-characterize /")
    print("      bound-the-sweep) is SEPARATE.")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
