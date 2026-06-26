"""Stage C penetration sweep (offline).

Question (§7.15-aug routed): is the ~1.70× sub-stepped residual at the
clean state NORMAL-CONTACT COMPLIANCE (rigid Stewart-Trinkle vs Drake's
compliant point-contact) or FRICTION (Stewart-Trinkle instantaneous λ_t
vs compliant tangential friction)?

The §7.15 (6) flagged inversion: factor 1.70× UNDER at EE_PEN_M=0.1 mm
crosses to 0.48× OVER at EE_PEN_M=1 mm. Crossing-of-1× across penetration
depth is in the NORMAL-direction state, NOT a tangent-force knob. This
sweep maps the gap factor vs depth to disambiguate.

Method:
  1. Vary EE_PEN_M across [0.1, 0.2, 0.3, 0.5, 0.7, 1.0] mm.
  2. AT EACH DEPTH: re-pose to a clean pusher-only state at that
     penetration via the box-pin IK from §7.15. Apply the §7.14 hard gate
     PER-DEPTH (deeper penetration changes arm pose; arm-link intrusion
     could creep back).
  3. AT GATED-CLEAN DEPTHS: LCS sub-stepped (Δt=0.005, 10 sub-steps,
     re-extracted each step, count=4) + Drake reference (1 ms substeps,
     0.05 s total) at the SAME state. Compute factor = |Drake / LCS|.
  4. Tabulate factor vs depth.

Pre-registered routes:
  NORMAL-COMPLIANCE — smooth crossing of 1× across depth → contact-model
                      reformulation conversation (compliance term, or
                      anitescu); anitescu re-promoted from parked.
  FRICTION          — flat ~1.7× across all gated depths, 1 mm inversion
                      is a gate-fail/numerical artifact → friction audit
                      on the clean residual.
  MIXED             — depth dependence but no smooth crossing → both
                      mechanisms contribute.
  GATE-FAILS-DEEP   — gate fails above some depth → bound the sweep, the
                      1 mm inversion may have been arm-contaminated.

This script does NOT execute the next step (characterize / friction
audit / anitescu scoping) — that is a SEPARATE block.
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

# ---- Sweep grid -----------------------------------------------------------
EE_PEN_VALUES_M = [0.0001, 0.0002, 0.0003, 0.0005, 0.0007, 0.001]   # 0.1..1.0 mm
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


def _collect_geom_ids(plant, scene_graph, model, body_names):
    q = scene_graph.model_inspector()
    ids = []
    for bname in body_names:
        body = plant.GetBodyByName(bname, model)
        fid = plant.GetBodyFrameIdOrThrow(body.index())
        for gid in q.GetGeometries(fid, Role.kProximity):
            ids.append(gid)
    return ids


def _geom_ids_for_body(plant, scene_graph, body):
    q = scene_graph.model_inspector()
    fid = plant.GetBodyFrameIdOrThrow(body.index())
    return list(q.GetGeometries(fid, Role.kProximity))


def setup_state_at_depth(diagram, plant, scene_graph, panda_model,
                         object_model, ee_frame, ee_pen_m, verbose=False):
    """Set up the clean pusher-only state at EE penetration ee_pen_m.

    Returns (context, plant_ctx, q_arm, ee_pos, seed_idx) on success or
    (None, None, None, None, None) on failure (no seed produces a
    physically-sane solution).
    """
    world = plant.world_frame()
    ee_pos = np.array([BOX_HALF + EE_RADIUS - ee_pen_m, 0.0, 0.05])
    p_tol = 1e-5

    arm_geoms = _collect_geom_ids(plant, scene_graph, panda_model,
                                   ARM_LINKS_TO_CLEAR)
    box_body = plant.GetBodyByName("box_link", object_model)
    box_geoms = _geom_ids_for_body(plant, scene_graph, box_body)

    for seed_idx, seed in enumerate(IK_SEEDS):
        context = diagram.CreateDefaultContext()
        plant_ctx = plant.GetMyContextFromRoot(context)
        plant.SetPositions(plant_ctx, object_model,
                           np.concatenate([BOX_QUAT, BOX_POS]))
        plant.SetPositions(plant_ctx, panda_model, seed)

        ik = InverseKinematics(plant, plant_ctx, with_joint_limits=True)

        # Pin box
        box_frame_pin = box_body.body_frame()
        ik.AddPositionConstraint(box_frame_pin, np.zeros(3), world,
                                 BOX_POS - 1e-5, BOX_POS + 1e-5)
        ik.AddOrientationConstraint(world, RotationMatrix(),
                                    box_frame_pin, RotationMatrix(), 0.001)

        # EE position constraint at this depth
        ik.AddPositionConstraint(ee_frame, np.zeros(3), world,
                                 ee_pos - p_tol, ee_pos + p_tol)

        # Per-pair arm-link↔box distance ≥ ARM_BOX_CLEARANCE_M
        for a_gid in arm_geoms:
            for b_gid in box_geoms:
                ik.AddDistanceConstraint((a_gid, b_gid),
                                          distance_lower=ARM_BOX_CLEARANCE_M,
                                          distance_upper=10.0)

        # Posture cost
        q_dec = ik.q()
        j1_idx = plant.GetJointByName("panda_joint1").position_start()
        for k in range(7):
            ik.get_mutable_prog().AddQuadraticErrorCost(
                np.array([[POSTURE_WEIGHT]]),
                np.array([POSTURE_NOMINAL[k]]),
                np.array([q_dec[j1_idx + k]]))

        ik.get_mutable_prog().SetInitialGuess(ik.q(), plant.GetPositions(plant_ctx))
        res = Solve(ik.prog())
        if not res.is_success():
            if verbose:
                print(f"      [seed {seed_idx}] IK FAIL")
            continue

        plant.SetPositions(plant_ctx, res.GetSolution(ik.q()))
        q_arm = plant.GetPositions(plant_ctx, panda_model).copy()
        p_ee_actual = plant.CalcPointsPositions(
            plant_ctx, ee_frame, np.zeros(3), world).flatten()
        ee_err_mm = np.linalg.norm(p_ee_actual - ee_pos) * 1000
        if ee_err_mm > 1.0:
            if verbose:
                print(f"      [seed {seed_idx}] IK reports success but EE_err = {ee_err_mm:.2f}mm > 1mm — REJECT")
            continue

        # Set EE velocity
        plant.SetVelocities(plant_ctx, object_model, np.zeros(6))
        J = plant.CalcJacobianTranslationalVelocity(
            plant_ctx, JacobianWrtVariable.kV, ee_frame, np.zeros(3),
            world, world)
        s = plant.GetJointByName("panda_joint1").velocity_start()
        q_arm_dot, *_ = np.linalg.lstsq(J[:, s:s+7],
                                        np.array([EE_VEL_X, 0., 0.]),
                                        rcond=None)
        plant.SetVelocities(plant_ctx, panda_model, q_arm_dot)
        return context, plant_ctx, q_arm, ee_pos, seed_idx

    return None, None, None, None, None


def gate_check(diagram, plant, scene_graph, ctx, obj_body, ee_body,
               threshold=0.010, gate_time=0.005):
    """The §7.14 per-state hard gate.

    Returns (ok, diag).
    """
    sim = Simulator(diagram, ctx)
    sim.Initialize()

    # ---- STATIC SignedDistance query at t=0 ----
    query_obj = scene_graph.get_query_output_port().Eval(
        scene_graph.GetMyContextFromRoot(ctx))
    inspector = query_obj.inspector()

    box_fid = plant.GetBodyFrameIdOrThrow(obj_body.index())
    box_gids_set = set(inspector.GetGeometries(box_fid, Role.kProximity))
    pusher_fid = plant.GetBodyFrameIdOrThrow(ee_body.index())
    pusher_gids_set = set(inspector.GetGeometries(pusher_fid, Role.kProximity))
    world_fid = plant.GetBodyFrameIdOrThrow(plant.world_body().index())
    world_gids_set = set(inspector.GetGeometries(world_fid, Role.kProximity))

    fid_to_body = {}
    for bi in range(plant.num_bodies()):
        body = plant.get_body(BodyIndex(bi))
        try:
            fid_to_body[plant.GetBodyFrameIdOrThrow(body.index())] = body
        except Exception:
            pass

    sd_pairs = query_obj.ComputeSignedDistancePairwiseClosestPoints(
        max_distance=threshold)
    n_pusher_box = n_floor_box = n_arm_box = n_other = 0
    d_pusher_box = d_floor_box = float('inf')
    min_arm_box = float('inf')
    arm_offender = None

    for pp in sd_pairs:
        in_box = pp.id_A in box_gids_set or pp.id_B in box_gids_set
        if not in_box:
            n_other += 1
            continue
        other = pp.id_B if pp.id_A in box_gids_set else pp.id_A
        if other in pusher_gids_set:
            n_pusher_box += 1
            d_pusher_box = min(d_pusher_box, pp.distance)
        elif other in world_gids_set:
            n_floor_box += 1
            d_floor_box = min(d_floor_box, pp.distance)
        else:
            other_body = fid_to_body.get(inspector.GetFrameId(other))
            oname = other_body.name().lower() if other_body else "?"
            if "panda" in oname:
                n_arm_box += 1
                if pp.distance < min_arm_box:
                    min_arm_box = pp.distance
                    arm_offender = (other_body.name(), pp.distance)
            else:
                n_other += 1

    # ---- DYNAMIC ContactResults at t=gate_time ----
    sim.AdvanceTo(gate_time)
    pctx_now = plant.GetMyContextFromRoot(ctx)
    cr = plant.get_contact_results_output_port().Eval(pctx_now)
    n_dyn_pusher = n_dyn_floor = n_dyn_arm = n_dyn_other = 0
    for i in range(cr.num_point_pair_contacts()):
        info = cr.point_pair_contact_info(i)
        bodyA = plant.get_body(info.bodyA_index())
        bodyB = plant.get_body(info.bodyB_index())
        if bodyB.index() == obj_body.index():
            other = bodyA
        elif bodyA.index() == obj_body.index():
            other = bodyB
        else:
            continue
        oname = other.name().lower()
        is_ground = (other.index() == plant.world_body().index()
                     or "world" in oname or "ground" in oname)
        is_pusher = "pusher" in oname
        is_arm = ("panda" in oname) and not is_pusher
        if is_pusher:
            n_dyn_pusher += 1
        elif is_arm:
            n_dyn_arm += 1
        elif is_ground:
            n_dyn_floor += 1
        else:
            n_dyn_other += 1

    ok = (n_arm_box == 0) and (n_dyn_arm == 0)
    return ok, dict(
        static=(n_pusher_box, n_floor_box, n_arm_box, n_other),
        dynamic=(n_dyn_pusher, n_dyn_floor, n_dyn_arm, n_dyn_other),
        d_pusher_box=d_pusher_box, d_floor_box=d_floor_box,
        min_arm_box=min_arm_box, arm_offender=arm_offender,
    )


def lcs_sub_stepped(plant, plant_ctx, ee_frame, obj_body, plant_ad, ctx_ad,
                     mu, panda_model, object_model, q_arm_init, ee_pos):
    """LCS Δt=0.005, 10 sub-steps, re-extract each step, drag=0 (Cell D
    setup from §7.15)."""
    from control.lcs_formulator import LCSFormulator
    world = plant.world_frame()

    x_curr = np.concatenate([
        BOX_QUAT, BOX_POS, ee_pos, np.zeros(3), np.zeros(3),
        np.array([EE_VEL_X, 0., 0.])])
    box_x_0 = float(x_curr[BOX_POS_X_IDX])
    box_z_0 = float(x_curr[BOX_POS_Z_IDX])
    q_arm = q_arm_init.copy()
    fail = 0

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
                fail += 1
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
    dz = float(x_curr[BOX_POS_Z_IDX]) - box_z_0
    return dx, dz, fail


def drake_reference(diagram, plant, ctx, obj_body):
    sim = Simulator(diagram, ctx)
    sim.Initialize()
    pctx0 = plant.GetMyContextFromRoot(ctx)
    p0 = plant.EvalBodyPoseInWorld(pctx0, obj_body).translation()
    sim.AdvanceTo(DT_BIG)
    pctx1 = plant.GetMyContextFromRoot(ctx)
    p1 = plant.EvalBodyPoseInWorld(pctx1, obj_body).translation()
    return float(p1[0] - p0[0]), float(p1[2] - p0[2])


# ---- main -----------------------------------------------------------------

def main() -> int:
    print("=" * 76)
    print("STAGE C  PENETRATION SWEEP  (offline, clean state, per-depth gate)")
    print("=" * 76)

    os.environ["LCS_EXPLICIT_BOX_GND"] = "4"
    import control.lcs_formulator
    importlib.reload(control.lcs_formulator)

    with open("config/tasks.yaml") as f:
        cfg = yaml.safe_load(f)
    task_cfg = cfg["tasks"]["pushing"]
    mu = task_cfg["friction"]

    print(f"  μ = {mu}, Δt_sub = {DT_SUB}s × {N_SUB} = {DT_BIG}s, Drake 1ms substeps")
    print(f"  Sweep EE_PEN_M ∈ {[round(v*1000,2) for v in EE_PEN_VALUES_M]} mm")
    print()

    rows = []
    for pen_m in EE_PEN_VALUES_M:
        pen_mm = pen_m * 1000
        print("─" * 76)
        print(f"DEPTH EE_PEN_M = {pen_mm:.2f} mm")
        print("─" * 76)

        # ---- Re-pose at this depth ----
        d, p, sg, panda, obj, _, p_ad, ctx_ad = _build_env(task_cfg)
        obj_body = p.GetBodyByName("box_link", obj)
        ee_frame = p.GetFrameByName("pusher")
        ee_body  = p.GetBodyByName("pusher")
        ctx, pctx, q_arm, ee_pos, seed = setup_state_at_depth(
            d, p, sg, panda, obj, ee_frame, pen_m, verbose=True)
        if ctx is None:
            print(f"  [depth {pen_mm:.2f}mm] RE-POSE FAILED on all seeds")
            rows.append(dict(pen_mm=pen_mm, status='REPOSE_FAIL'))
            continue
        print(f"  re-pose OK seed={seed}, q_arm={q_arm.round(3)}")

        # ---- Gate check ----
        d_g, p_g, sg_g, panda_g, obj_g, _, _, _ = _build_env(task_cfg)
        obj_body_g = p_g.GetBodyByName("box_link", obj_g)
        ee_frame_g = p_g.GetFrameByName("pusher")
        ee_body_g  = p_g.GetBodyByName("pusher")
        ctx_g, pctx_g, _, _, _ = setup_state_at_depth(
            d_g, p_g, sg_g, panda_g, obj_g, ee_frame_g, pen_m)
        ok, gdiag = gate_check(d_g, p_g, sg_g, ctx_g, obj_body_g, ee_body_g)
        s = gdiag['static']
        dyn = gdiag['dynamic']
        print(f"  GATE: STATIC pusher={s[0]} floor={s[1]} arm={s[2]} other={s[3]}    "
              f"DYNAMIC pusher={dyn[0]} floor={dyn[1]} arm={dyn[2]} other={dyn[3]}")
        print(f"        d_pusher_box={gdiag['d_pusher_box']*1000:+7.4f}mm  "
              f"d_floor_box={gdiag['d_floor_box']*1000:+7.4f}mm  "
              f"min_d_arm_box={gdiag['min_arm_box']*1000:+7.3f}mm")
        if not ok:
            print(f"  ► GATE FAIL — arm↔box pair found (static={s[2]}, dynamic={dyn[2]})")
            if gdiag['arm_offender']:
                bname, dist = gdiag['arm_offender']
                print(f"      offender: {bname} at {dist*1000:+.3f}mm")
            rows.append(dict(pen_mm=pen_mm, status='GATE_FAIL',
                             arm_box_count_static=s[2],
                             arm_box_count_dynamic=dyn[2]))
            continue
        print(f"  ► GATE PASS")

        # ---- LCS sub-stepped (re-extract Δt=0.005, drag=0) ----
        d_l, p_l, sg_l, panda_l, obj_l, _, p_ad_l, ctx_ad_l = _build_env(task_cfg)
        obj_body_l = p_l.GetBodyByName("box_link", obj_l)
        ee_frame_l = p_l.GetFrameByName("pusher")
        ctx_l, pctx_l, q_arm_l, ee_pos_l, _ = setup_state_at_depth(
            d_l, p_l, sg_l, panda_l, obj_l, ee_frame_l, pen_m)
        dx_lcs, dz_lcs, fail = lcs_sub_stepped(
            p_l, pctx_l, ee_frame_l, obj_body_l, p_ad_l, ctx_ad_l,
            mu, panda_l, obj_l, q_arm_l, ee_pos_l)

        # ---- Drake reference ----
        d_d, p_d, sg_d, panda_d, obj_d, _, _, _ = _build_env(task_cfg)
        obj_body_d = p_d.GetBodyByName("box_link", obj_d)
        ee_frame_d = p_d.GetFrameByName("pusher")
        ctx_d, _, _, _, _ = setup_state_at_depth(
            d_d, p_d, sg_d, panda_d, obj_d, ee_frame_d, pen_m)
        dx_drk, dz_drk = drake_reference(d_d, p_d, ctx_d, obj_body_d)

        # ---- Factor ----
        if abs(dx_lcs) < 1e-9:
            factor = float('inf')
        else:
            factor = abs(dx_drk / dx_lcs)
        direction = "UNDER" if factor > 1.0 else "OVER "
        print(f"  Drake Δbox_x = {dx_drk*1000:+.4f} mm  (Δz={dz_drk*1000:+.4f}mm)")
        print(f"  LCS sub-st   = {dx_lcs*1000:+.4f} mm  (Δz={dz_lcs*1000:+.4f}mm) "
              f"{'fail '+str(fail) if fail else ''}")
        print(f"  factor       = {factor:.3f}×  ({direction}-predicts Drake)")
        rows.append(dict(pen_mm=pen_mm, status='OK',
                         dx_drake=dx_drk, dx_lcs=dx_lcs,
                         dz_drake=dz_drk, dz_lcs=dz_lcs,
                         factor=factor, fail=fail))
        print()

    # ---- Final table + route ----
    print("=" * 76)
    print("SUMMARY  (sub-stepped Δt=0.005, drag=0, count=4)")
    print("=" * 76)
    print(f"  {'EE_PEN':>8} {'status':>12} {'Drake Δx':>12} {'LCS Δx':>12} "
          f"{'factor':>9} {'direction':>10}")
    for r in rows:
        if r['status'] == 'OK':
            d = "UNDER" if r['factor'] > 1.0 else "OVER"
            print(f"  {r['pen_mm']:>6.2f}mm {r['status']:>12} "
                  f"{r['dx_drake']*1000:+10.4f}mm {r['dx_lcs']*1000:+10.4f}mm "
                  f"{r['factor']:>8.3f}× {d:>10}")
        else:
            print(f"  {r['pen_mm']:>6.2f}mm {r['status']:>12} -- gate or repose fail --")

    ok_rows = [r for r in rows if r['status'] == 'OK']
    gate_fails = [r for r in rows if r['status'] == 'GATE_FAIL']
    print()
    print(f"  Gated-clean depths : {len(ok_rows)}/{len(rows)}")
    print(f"  Gate fails         : {len(gate_fails)}")
    if gate_fails:
        max_clean = max(
            (r['pen_mm'] for r in ok_rows), default=0.0)
        min_failed = min(r['pen_mm'] for r in gate_fails)
        print(f"  Max clean depth    : {max_clean:.2f} mm")
        print(f"  Min failed depth   : {min_failed:.2f} mm")

    # ---- Route the read ----
    print()
    print("=" * 76)
    print("ROUTE")
    print("=" * 76)
    if len(ok_rows) < 3:
        print("  ► INSUFFICIENT DATA — fewer than 3 gated-clean depths.")
        if gate_fails:
            print("    GATE-FAILS-DEEP: the sweep is bounded to shallow penetrations.")
            print("    Re-evaluate the 1mm sweet-spot claim — it may have been arm-contaminated.")
        return 0

    factors = [r['factor'] for r in ok_rows]
    pens = [r['pen_mm'] for r in ok_rows]
    f_max = max(factors)
    f_min = min(factors)
    spread = f_max - f_min

    # Crossing of 1: any pair where one is > 1 and next is < 1 (or vice versa)
    crosses_1 = False
    cross_depth = None
    for i in range(len(factors) - 1):
        if (factors[i] - 1.0) * (factors[i + 1] - 1.0) < 0:
            crosses_1 = True
            # Interpolate
            f0, f1 = factors[i], factors[i + 1]
            d0, d1 = pens[i], pens[i + 1]
            cross_depth = d0 + (1.0 - f0) * (d1 - d0) / (f1 - f0)
            break

    # Monotonicity check
    diffs = [factors[i+1] - factors[i] for i in range(len(factors) - 1)]
    monotonic = all(d <= 0 for d in diffs) or all(d >= 0 for d in diffs)

    print(f"  Factor spread     : {spread:.3f}  (max {f_max:.3f}×, min {f_min:.3f}×)")
    print(f"  Crosses 1×?       : {'YES' if crosses_1 else 'NO'}"
          + (f" at ~{cross_depth:.3f} mm" if crosses_1 else ""))
    print(f"  Monotonic w/ depth?: {'YES' if monotonic else 'NO'}")
    print()

    if crosses_1 and monotonic and spread > 0.5:
        route = "NORMAL-COMPLIANCE"
        msg = (
            f"  Factor crosses 1× SMOOTHLY at ~{cross_depth:.3f} mm penetration. The gap is\n"
            f"  in the NORMAL-direction state — consistent with rigid Stewart-Trinkle vs\n"
            f"  Drake's compliant point-contact normal force law. Friction is a sideshow.\n"
            f"  NEXT (separate block): characterize the normal compliance, scope anitescu\n"
            f"  re-promoted as the indicated direction (no longer parked-pending-FRICTION-\n"
            f"  BARELY). The sweet-spot depth (~{cross_depth:.3f} mm) is a quantitative handle.")
    elif spread < 0.3:
        route = "FRICTION"
        msg = (
            f"  Factor is FLAT across depth (spread {spread:.3f}). Normal compliance is\n"
            f"  ruled out — the residual is in the tangential direction. The §7.15-aug\n"
            f"  1mm inversion in the EARLIER probe was a gate-fail or numerical edge.\n"
            f"  NEXT (separate block): friction audit on the clean residual (§7.13 (9)\n"
            f"  three-outcome pre-registration). Anitescu STAYS PARKED.")
    elif crosses_1:
        route = "MIXED"
        msg = (
            f"  Factor varies with depth and crosses 1× but NOT smoothly/monotonically.\n"
            f"  Both normal compliance and another mechanism contribute.\n"
            f"  NEXT (separate block): characterize the depth dependence (normal compliance)\n"
            f"  first, then friction on the residual.")
    else:
        route = "GAP-PERSISTS-NO-CROSSING"
        msg = (
            f"  Factor varies with depth (spread {spread:.3f}) but does NOT cross 1× in\n"
            f"  this range. There is depth dependence (likely normal compliance) but the\n"
            f"  LCS doesn't quite reach Drake at any tested depth.\n"
            f"  NEXT (separate block): extend the sweep range OR characterize the depth\n"
            f"  dependence directly; friction audit later if a residual floor remains.")

    print(f"  ► ROUTE: {route}")
    print(msg)
    print()
    print("=" * 76)
    print("HOLD: next block (characterize / friction / anitescu) is SEPARATE.")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
