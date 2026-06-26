"""Stage C §7.16 sweep cleanup (offline).

Question (§7.17-aug routed): does the §7.16 monotonic 1× crossing
SURVIVE on clean full-trajectory runs, once partial-IK sub-step factors
are EXCLUDED (not annotated) from the crossing determination?

§7.17 surfaced the §7.16 confound: at EE_PEN_M = 1 mm the sub-step IK
fails at step 4, so §7.16's reported Δbox_x = -5.73 mm / factor 0.391×
came from a 3-4-substep PARTIAL trajectory, NOT the full 10-step
sub-stepping. The sweep printed `fail` counts but did NOT correlate
them with the reported factor; the partial factor entered the crossing
determination silently.

This cleanup probe:
  1. Re-runs the §7.16 sweep at the same depths
     [0.1, 0.2, 0.3, 0.5, 0.7, 1.0] mm.
  2. ROBUST SUB-STEP IK: per sub-step, try multiple seed warm-starts
     (current q_arm; prior sub-step q_arm; nominal posture) — accept
     the first that converges.
  3. ACCEPT-PARTIAL-AND-INVALIDATE: if all retries fail at some
     sub-step k < N_SUB-1, the trajectory is PARTIAL → its factor is
     INVALID and EXCLUDED from the crossing determination (NOT
     annotated as if clean).
  4. CORRELATE: per-depth row shows (factor, n_substeps_completed,
     fail_step, status: CLEAN / PARTIAL / FAILED). Only CLEAN rows
     enter route determination.
  5. Per-depth contact-pair gate still applies (§7.14 rule).

Pre-registered routes (§7.17-aug):
  CROSSING-SURVIVES — crossing holds on clean full runs
                      → next block: cleaned force-level re-characterization
                                    + mechanism-name determination (separate)
  CROSSING-WEAKENS  — crossing changes materially when partials excluded
                      → diagnosis reopens
  IK-STILL-FAILS-DEEP — deep depths can't complete even with robustness
                        → bounded sweep, crossing maybe undeterminable
  INTERMITTENT-DOMINATES — Drake bouncing dominant → re-examine test design
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

EE_PEN_VALUES_M = [0.0001, 0.0002, 0.0003, 0.0005, 0.0007, 0.001]
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
    world = plant.world_frame()
    ee_pos = np.array([BOX_HALF + EE_RADIUS - ee_pen_m, 0.0, 0.05])
    p_tol = 1e-5
    arm_geoms = _collect_geom_ids(plant, sg, panda_model, ARM_LINKS_TO_CLEAR)
    box_body = plant.GetBodyByName("box_link", object_model)
    box_geoms = _geom_ids_for_body(plant, sg, box_body)

    for seed_idx, seed in enumerate(IK_SEEDS_INIT):
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


def gate_check(diagram, plant, sg, ctx, obj_body, ee_body,
               threshold=0.010, gate_time=0.005):
    sim = Simulator(diagram, ctx)
    sim.Initialize()
    query_obj = sg.get_query_output_port().Eval(sg.GetMyContextFromRoot(ctx))
    inspector = query_obj.inspector()
    box_fid = plant.GetBodyFrameIdOrThrow(obj_body.index())
    box_gids = set(inspector.GetGeometries(box_fid, Role.kProximity))
    pusher_fid = plant.GetBodyFrameIdOrThrow(ee_body.index())
    pusher_gids = set(inspector.GetGeometries(pusher_fid, Role.kProximity))
    world_fid = plant.GetBodyFrameIdOrThrow(plant.world_body().index())
    world_gids = set(inspector.GetGeometries(world_fid, Role.kProximity))
    fid_to_body = {}
    for bi in range(plant.num_bodies()):
        body = plant.get_body(BodyIndex(bi))
        try:
            fid_to_body[plant.GetBodyFrameIdOrThrow(body.index())] = body
        except Exception:
            pass
    sd_pairs = query_obj.ComputeSignedDistancePairwiseClosestPoints(
        max_distance=threshold)
    n_pusher = n_floor = n_arm = 0
    for pp in sd_pairs:
        in_box = pp.id_A in box_gids or pp.id_B in box_gids
        if not in_box:
            continue
        other = pp.id_B if pp.id_A in box_gids else pp.id_A
        if other in pusher_gids:
            n_pusher += 1
        elif other in world_gids:
            n_floor += 1
        else:
            ob = fid_to_body.get(inspector.GetFrameId(other))
            if ob and "panda" in ob.name().lower():
                n_arm += 1
    sim.AdvanceTo(gate_time)
    pctx_now = plant.GetMyContextFromRoot(ctx)
    cr = plant.get_contact_results_output_port().Eval(pctx_now)
    n_dyn_arm = 0
    for i in range(cr.num_point_pair_contacts()):
        info = cr.point_pair_contact_info(i)
        a = plant.get_body(info.bodyA_index())
        b = plant.get_body(info.bodyB_index())
        if b.index() == obj_body.index():
            other = a
        elif a.index() == obj_body.index():
            other = b
        else:
            continue
        oname = other.name().lower()
        if "panda" in oname and "pusher" not in oname:
            n_dyn_arm += 1
    ok = (n_arm == 0) and (n_dyn_arm == 0)
    return ok, dict(static_arm=n_arm, dynamic_arm=n_dyn_arm,
                    static_pusher=n_pusher, static_floor=n_floor)


def _try_substep_ik(plant, plant_ctx, ee_frame, panda_model, object_model,
                     x_next, q_arm_warm, world):
    """Try multiple seed q_arm for sub-step IK. Returns new q_arm or None.

    Seeds tried, in order:
      1. q_arm_warm (the prior sub-step solution)
      2. POSTURE_NOMINAL
      3-5. Random perturbations of q_arm_warm (±0.1 rad each joint)
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
            # Verify EE constraint actually satisfied
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


def lcs_sub_stepped_robust(plant, plant_ctx, ee_frame, obj_body, plant_ad,
                            ctx_ad, mu, panda_model, object_model,
                            q_arm_init, ee_pos):
    """Robust sub-stepped LCS: retry sub-step IK with multiple seeds.

    Returns dict with: dx, dz, n_substeps_completed, fail_step (-1 if none),
                       seeds_used (per-substep), status ('CLEAN'/'PARTIAL').
    """
    from control.lcs_formulator import LCSFormulator
    world = plant.world_frame()
    x_curr = np.concatenate([
        BOX_QUAT, BOX_POS, ee_pos, np.zeros(3), np.zeros(3),
        np.array([EE_VEL_X, 0., 0.])])
    box_x_0 = float(x_curr[BOX_POS_X_IDX])
    box_z_0 = float(x_curr[BOX_POS_Z_IDX])
    q_arm = q_arm_init.copy()
    n_completed = 0
    fail_step = -1
    seeds_used = []

    for k in range(N_SUB):
        f = LCSFormulator(plant, mu=mu, obj_body=obj_body,
                          plant_ad=plant_ad, context_ad=ctx_ad,
                          box_ground_drag=0.0)
        A, B_ctrl, D, d_const, E, F, H, c_lcs, *_ = \
            f.linearize_discrete_ee_space(plant_ctx, DT_SUB, np.zeros(3))
        q_lcp = E @ x_curr + c_lcs
        lam, _ = solve_lcp(F, q_lcp)
        x_next = A @ x_curr + B_ctrl @ np.zeros(3) + D @ lam + d_const

        if k < N_SUB - 1:
            new_q, seed_idx = _try_substep_ik(
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
    dz = float(x_curr[BOX_POS_Z_IDX]) - box_z_0
    return dict(dx=dx, dz=dz, n_completed=n_completed, fail_step=fail_step,
                seeds_used=seeds_used, status=status)


def drake_reference(diagram, plant, ctx, obj_body):
    sim = Simulator(diagram, ctx)
    sim.Initialize()
    pctx0 = plant.GetMyContextFromRoot(ctx)
    p0 = plant.EvalBodyPoseInWorld(pctx0, obj_body).translation()
    sim.AdvanceTo(DT_BIG)
    pctx1 = plant.GetMyContextFromRoot(ctx)
    p1 = plant.EvalBodyPoseInWorld(pctx1, obj_body).translation()
    return float(p1[0] - p0[0]), float(p1[2] - p0[2])


def main() -> int:
    print("=" * 80)
    print("STAGE C  §7.16 SWEEP CLEANUP  (robust sub-step IK + partial-INVALIDATES)")
    print("=" * 80)
    os.environ["LCS_EXPLICIT_BOX_GND"] = "4"
    import control.lcs_formulator
    importlib.reload(control.lcs_formulator)

    with open("config/tasks.yaml") as f:
        cfg = yaml.safe_load(f)
    task_cfg = cfg["tasks"]["pushing"]
    mu = task_cfg["friction"]
    print(f"  μ = {mu}, Δt_sub = {DT_SUB}s × {N_SUB} = {DT_BIG}s, Drake 1ms substeps")
    print(f"  Sweep EE_PEN_M ∈ {[round(v*1000,2) for v in EE_PEN_VALUES_M]} mm")
    print(f"  Robust sub-step IK: 5 seeds per sub-step (warm/posture/perturbed)")
    print(f"  Discipline: PARTIAL trajectories EXCLUDED from crossing determination")
    print()

    rows = []
    for pen_m in EE_PEN_VALUES_M:
        pen_mm = pen_m * 1000
        print("─" * 80)
        print(f"DEPTH EE_PEN_M = {pen_mm:.2f} mm")
        print("─" * 80)

        # ---- LCS robust sub-stepped ----
        d, p, sg, panda, obj, _, p_ad, ctx_ad = _build_env(task_cfg)
        obj_body = p.GetBodyByName("box_link", obj)
        ee_frame = p.GetFrameByName("pusher")
        ee_body  = p.GetBodyByName("pusher")
        ctx, pctx, q_arm, ee_pos = setup_state_at_depth(
            d, p, sg, panda, obj, ee_frame, pen_m)
        if ctx is None:
            print(f"  re-pose FAIL — skip depth")
            rows.append(dict(pen_mm=pen_mm, status='REPOSE_FAIL'))
            continue

        # ---- Per-depth contact-pair gate ----
        d_g, p_g, sg_g, panda_g, obj_g, _, _, _ = _build_env(task_cfg)
        obj_body_g = p_g.GetBodyByName("box_link", obj_g)
        ee_frame_g = p_g.GetFrameByName("pusher")
        ee_body_g  = p_g.GetBodyByName("pusher")
        ctx_g, _, _, _ = setup_state_at_depth(
            d_g, p_g, sg_g, panda_g, obj_g, ee_frame_g, pen_m)
        gate_ok, gdiag = gate_check(d_g, p_g, sg_g, ctx_g,
                                     obj_body_g, ee_body_g)
        if not gate_ok:
            print(f"  GATE FAIL — arm↔box static={gdiag['static_arm']} "
                  f"dynamic={gdiag['dynamic_arm']}")
            rows.append(dict(pen_mm=pen_mm, status='GATE_FAIL', **gdiag))
            continue
        print(f"  GATE PASS (pusher={gdiag['static_pusher']}, "
              f"floor={gdiag['static_floor']}, arm static=0 dynamic=0)")

        # ---- Robust sub-stepped LCS ----
        lcs_res = lcs_sub_stepped_robust(
            p, pctx, ee_frame, obj_body, p_ad, ctx_ad, mu, panda, obj,
            q_arm, ee_pos)

        # ---- Drake reference ----
        d_d, p_d, sg_d, panda_d, obj_d, _, _, _ = _build_env(task_cfg)
        obj_body_d = p_d.GetBodyByName("box_link", obj_d)
        ee_frame_d = p_d.GetFrameByName("pusher")
        ctx_d, _, _, _ = setup_state_at_depth(
            d_d, p_d, sg_d, panda_d, obj_d, ee_frame_d, pen_m)
        dx_drk, dz_drk = drake_reference(d_d, p_d, ctx_d, obj_body_d)

        factor = (abs(dx_drk / lcs_res['dx'])
                  if abs(lcs_res['dx']) > 1e-9 else float('inf'))
        print(f"  LCS    Δbox_x  = {lcs_res['dx']*1000:+9.4f} mm  "
              f"(n_substeps_completed={lcs_res['n_completed']}/{N_SUB}, "
              f"fail_step={lcs_res['fail_step']}, status={lcs_res['status']})")
        print(f"  Drake  Δbox_x  = {dx_drk*1000:+9.4f} mm")
        print(f"  factor (computed) = {factor:.3f}×  — "
              f"{'CLEAN, enters crossing' if lcs_res['status']=='CLEAN' else 'PARTIAL, EXCLUDED from crossing'}")
        print(f"  seeds used per sub-step: {lcs_res['seeds_used']}")

        rows.append(dict(
            pen_mm=pen_mm,
            status=lcs_res['status'],
            dx_lcs=lcs_res['dx'], dx_drake=dx_drk,
            dz_lcs=lcs_res['dz'], dz_drake=dz_drk,
            n_completed=lcs_res['n_completed'],
            fail_step=lcs_res['fail_step'],
            factor=factor,
        ))
        print()

    # ---- Summary table ----
    print("=" * 80)
    print("SUMMARY  (per-depth: factor + n_substeps + fail_step + status)")
    print("=" * 80)
    print(f"  {'EE_PEN':>8} {'status':>10} {'n_sub':>7} {'fail_step':>10} "
          f"{'Drake Δx':>13} {'LCS Δx':>13} {'factor':>9}")
    for r in rows:
        if 'factor' in r:
            f_str = f"{r['factor']:>8.3f}×"
            print(f"  {r['pen_mm']:>6.2f}mm {r['status']:>10} "
                  f"{r['n_completed']:>7}/{N_SUB} {r['fail_step']:>10} "
                  f"{r['dx_drake']*1000:+11.4f}mm {r['dx_lcs']*1000:+11.4f}mm "
                  f"{f_str:>9}")
        else:
            print(f"  {r['pen_mm']:>6.2f}mm {r['status']:>10} -- (no factor) --")

    clean_rows = [r for r in rows if r.get('status') == 'CLEAN']
    partial_rows = [r for r in rows if r.get('status') == 'PARTIAL']
    gate_fail_rows = [r for r in rows if r.get('status') == 'GATE_FAIL']

    print()
    print(f"  CLEAN full-trajectory depths    : {len(clean_rows)}/{len(rows)}")
    print(f"  PARTIAL trajectory depths       : {len(partial_rows)} (EXCLUDED)")
    print(f"  GATE FAIL depths                : {len(gate_fail_rows)}")

    # ---- Crossing determination on CLEAN-only rows ----
    print()
    print("=" * 80)
    print("CROSSING DETERMINATION  (only CLEAN rows enter)")
    print("=" * 80)

    if len(clean_rows) < 2:
        print("  ► INSUFFICIENT CLEAN DATA — fewer than 2 clean full-trajectory depths.")
        print("  ► ROUTE: IK-STILL-FAILS-DEEP")
        print(f"    Max clean depth: "
              f"{max((r['pen_mm'] for r in clean_rows), default=0):.2f} mm")
        return 0

    clean_rows.sort(key=lambda r: r['pen_mm'])
    factors = [r['factor'] for r in clean_rows]
    pens = [r['pen_mm'] for r in clean_rows]
    f_max = max(factors)
    f_min = min(factors)
    spread = f_max - f_min

    crosses_1 = False
    cross_depth = None
    for i in range(len(factors) - 1):
        if (factors[i] - 1.0) * (factors[i + 1] - 1.0) < 0:
            crosses_1 = True
            f0, f1 = factors[i], factors[i + 1]
            d0, d1 = pens[i], pens[i + 1]
            cross_depth = d0 + (1.0 - f0) * (d1 - d0) / (f1 - f0)
            break
    diffs = [factors[i+1] - factors[i] for i in range(len(factors) - 1)]
    monotonic = all(d <= 1e-9 for d in diffs) or all(d >= -1e-9 for d in diffs)

    print(f"  CLEAN depths        : {pens}")
    print(f"  CLEAN factors       : {[f'{f:.3f}' for f in factors]}")
    print(f"  spread              : {spread:.3f}")
    print(f"  monotonic w/ depth? : {'YES' if monotonic else 'NO'}")
    print(f"  crosses 1×?         : {'YES' if crosses_1 else 'NO'}"
          + (f" at ~{cross_depth:.3f} mm" if crosses_1 else ""))

    # ---- §7.16 comparison ----
    print()
    print(f"  §7.16 ORIGINAL factors at these depths (for comparison):")
    sweep_orig = {0.1: 1.684, 0.2: 1.474, 0.3: 1.244, 0.5: 1.112,
                  0.7: 0.653, 1.0: 0.391}
    for r in rows:
        if 'factor' in r:
            orig = sweep_orig.get(round(r['pen_mm'], 2), float('nan'))
            print(f"    {r['pen_mm']:>5.2f}mm  CLEAN={r['factor']:.3f}×  "
                  f"§7.16-original={orig:.3f}×  "
                  f"diff={r['factor']-orig:+.3f}  status={r['status']}")

    # ---- Route ----
    print()
    print("=" * 80)
    print("ROUTE")
    print("=" * 80)
    deep_clean = [r for r in clean_rows if r['pen_mm'] >= 0.5]
    deep_partial = [r for r in partial_rows if r['pen_mm'] >= 0.5]
    over_predict_clean = [r for r in clean_rows if r['factor'] < 1.0]
    under_predict_clean = [r for r in clean_rows if r['factor'] > 1.0]

    if crosses_1 and monotonic and len(under_predict_clean) > 0 and len(over_predict_clean) > 0:
        route = "CROSSING-SURVIVES"
        msg = (
            f"  The monotonic 1× crossing SURVIVES on clean full-trajectory runs.\n"
            f"  Crossing at ~{cross_depth:.3f} mm.\n"
            f"  However: PARTIAL trajectories at {[r['pen_mm'] for r in partial_rows]} mm were\n"
            f"  EXCLUDED — the original §7.16 included them silently.\n"
            f"  NEXT (separate block): cleaned force-level re-characterization +\n"
            f"  mechanism-name determination. Do NOT auto-restore compliance — the\n"
            f"  intermittent-contact observation requires a separate probe.")
    elif len(under_predict_clean) > 0 and len(over_predict_clean) == 0:
        route = "CROSSING-WEAKENS"
        msg = (
            f"  ALL CLEAN depths under-predict (no over-prediction in clean data).\n"
            f"  The §7.16 over-prediction at deep depths was the partial-IK artifact.\n"
            f"  The monotonic crossing-of-1× does NOT survive cleanup.\n"
            f"  The diagnosis itself reopens. Compliance is no longer the leading\n"
            f"  hypothesis on this basis — the clean gap is a same-direction\n"
            f"  under-prediction across all measured depths.")
    elif len(under_predict_clean) == 0 and len(over_predict_clean) > 0:
        route = "CROSSING-WEAKENS"
        msg = (
            f"  ALL CLEAN depths over-predict. The crossing observed in §7.16 may\n"
            f"  reflect missing low-depth coverage rather than a real mechanism\n"
            f"  signature. Diagnosis reopens.")
    elif len(deep_clean) == 0 and len(deep_partial) > 0:
        route = "IK-STILL-FAILS-DEEP"
        msg = (
            f"  Even with robust IK, deep penetration depths {[r['pen_mm'] for r in deep_partial]}\n"
            f"  mm produced PARTIAL trajectories.\n"
            f"  The clean sweep is BOUNDED to shallow penetrations (max clean = \n"
            f"  {max((r['pen_mm'] for r in clean_rows), default=0):.2f} mm).\n"
            f"  The over-prediction depths are UNREACHABLE on clean trajectories —\n"
            f"  the §7.16 monotonic crossing is UNDETERMINABLE from clean data.")
    else:
        route = "UNDETERMINED"
        msg = (
            f"  Clean data does not produce a clean crossing or non-crossing\n"
            f"  determination. Re-examine.")

    print(f"  ► ROUTE: {route}")
    print(msg)
    print()
    print("=" * 80)
    print("HOLD: next block (mechanism-name determination / re-characterize / etc) SEPARATE.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
