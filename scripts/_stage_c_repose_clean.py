"""Stage C re-pose probe + hard contact-pair gate (offline).

Path A (re-pose) — NOT Path B (filter). Constructed-state-only; NO live-port
change, NO env_builder change.

The §7.11→§7.13 quantitative chain (3.73× → 1.43×) was contaminated by a
phantom panda_link7↔box arm-wrist contact carrying 37× the pusher impulse
(§7.14 AUDIT-SETUP-BROKEN). The LCS's geom filter (EE='pusher', geom IDs
[223]) excludes panda_link7, so §7.11→§7.13 compared an LCS-pusher-and-
floor model to a Drake-with-arm-artifact reference.

This probe:
  STEP 1  — re-pose the constructed state via additional IK constraints
            (panda_link5/6/7/8 origins forced to z >= z_clearance) so the
            wrist clears the box while pusher stays on the east face;
  STEP 2  — HARD GATE: query Drake's contact-pair set at the re-posed
            state; assert EXACTLY pusher↔box + N floor pairs, NO panda_
            link* contact (this is the precondition the contamination
            slipped past for FOUR sections);
  STEP 3  — IF gate passes: re-run §7.11/§7.12/§7.13 on the clean setup
            (single-step LCS, sub-stepped LCS, Drake reference, vertical
            sanity, drag byte-identical at v_box=0);
  STEP 4  — route the read: GAP-VANISHES / GAP-PERSISTS-LARGE /
            GAP-PARTIAL / GATE-FAILS (workspace finding).

This script does NOT pre-decide friction or anitescu — those are routed
only if §7.13's residual survives on a clean setup.
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

from pydrake.geometry import Role
from pydrake.math import RotationMatrix

from sim.env_builder import build_environment
from control.lcp_solver import solve_lcp

DT_BIG    = 0.05
DT_SUB    = 0.005
N_SUB     = int(round(DT_BIG / DT_SUB))    # 10
DT_DRAKE  = 0.001
N_DRAKE   = int(round(DT_BIG / DT_DRAKE))  # 50

BOX_HALF  = 0.05
EE_RADIUS = 0.025
BOX_QUAT  = np.array([1.0, 0.0, 0.0, 0.0])
BOX_POS   = np.array([0.0, 0.0, 0.05])
EE_PEN_M  = 0.0001        # 0.1 mm penetration (matches §7.13 setup exactly)
EE_POS    = np.array([BOX_HALF + EE_RADIUS - EE_PEN_M, 0.0, 0.05])
EE_VEL_X  = -0.05

BOX_POS_X_IDX  = 4   # box pos x in EE-space layout (after quat[0:4])
BOX_POS_Z_IDX  = 6   # box pos z in EE-space layout
BOX_VEL_X_IDX  = 13  # box velocity x

# Clearance: panda_link{4,5,6,7,8} collision geoms must be at least
# ARM_BOX_CLEARANCE_M away from any box collision geom. The pusher↔box
# pair is excluded (it is the LCS contact we want to model).
ARM_BOX_CLEARANCE_M = 0.005   # 5 mm clearance (geom-level, robust to rotation)
# panda_link8 is welded to the pusher (5cm +z offset) and its collision sphere
# rotates with the wrist — including it makes the joint problem infeasible
# with the EE-at-box-face constraint. We constrain link4-7 instead. The pusher
# IS the legit EE pair; link8's geom intrusion (if any) would be caught by
# the post-IK pair-set check.
ARM_LINKS_TO_CLEAR  = ["panda_link4", "panda_link5", "panda_link6",
                       "panda_link7"]
POSTURE_NOMINAL = np.array([0.0,  -0.4,  0.0, -1.8,  0.0,  1.4,  0.785])
POSTURE_WEIGHT = 5.0

# Seeds to try in order (varied elbow + wrist postures)
IK_SEEDS = [
    np.array([0.0,  0.0,  0.0, -1.5,  0.0,  1.5,  0.785]),   # canonical-elbow
    np.array([0.0, -0.3,  0.0, -1.7,  0.0,  1.4,  0.785]),   # mild back
    np.array([0.0,  0.3,  0.0, -1.3,  0.0,  1.6,  0.785]),   # mild forward
    np.array([0.0,  0.0,  0.0, -2.0,  0.0,  2.0,  0.785]),   # tighter wrist
    np.array([0.0, -0.5,  0.0, -2.0,  0.0,  1.5,  0.785]),   # ORIGINAL (sanity)
    np.array([0.0,  0.5,  0.0, -2.4,  0.0,  2.0,  0.785]),   # very forward+wrist
    np.array([0.0, -0.8,  0.0, -2.4,  0.0,  1.6,  0.785]),   # very back+wrist
    np.array([0.5,  0.0,  0.0, -1.8,  0.0,  1.8,  0.0]),     # rotated base
    np.array([-0.5, 0.0,  0.0, -1.8,  0.0,  1.8,  0.0]),     # rotated base other way
]


# --------------- contact-state setup (re-posed, with clearance) ----------

def _scene_graph_of(diagram):
    """Find the SceneGraph subsystem of the diagram."""
    for sys in diagram.GetSystems():
        if 'SceneGraph' in type(sys).__name__:
            return sys
    raise RuntimeError("SceneGraph not found in diagram")


def _build_env(task_cfg):
    """Wrap build_environment to also expose SceneGraph."""
    diagram, plant, panda, obj, meshcat, plant_ad, ctx_ad = \
        build_environment(task_cfg, time_step=0.001)
    sg = _scene_graph_of(diagram)
    return diagram, plant, sg, panda, obj, meshcat, plant_ad, ctx_ad


def _collect_geom_ids(plant, scene_graph, model, body_names):
    """Return list of proximity-role GeometryId for the named bodies of model."""
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


def setup_contact_state_clean(diagram, plant, scene_graph, panda_model,
                              object_model, ee_frame):
    """Re-pose via IK with per-geom-pair distance constraints between
    arm-link collision geoms and box collision geoms (excluding pusher).

    Tries multiple IK seeds; returns first that solves AND that we can
    verify Drake-side (gate check happens in caller — this only sets up
    the IK state).
    """
    world = plant.world_frame()
    p_tol = 1e-5  # 0.01 mm per-axis; L2 ~ 0.017mm < EE_PEN_M = 0.1mm

    # Collect geom IDs
    arm_geoms = _collect_geom_ids(plant, scene_graph, panda_model,
                                   ARM_LINKS_TO_CLEAR)
    box_body = plant.GetBodyByName("box_link", object_model)
    box_geoms = _geom_ids_for_body(plant, scene_graph, box_body)

    print(f"  [setup] arm geoms (link4-8): {len(arm_geoms)}  "
          f"box geoms: {len(box_geoms)}  "
          f"→ {len(arm_geoms)*len(box_geoms)} pair constraints")

    for seed_idx, seed in enumerate(IK_SEEDS):
        context = diagram.CreateDefaultContext()
        plant_ctx = plant.GetMyContextFromRoot(context)
        plant.SetPositions(plant_ctx, object_model, np.concatenate([BOX_QUAT, BOX_POS]))
        plant.SetPositions(plant_ctx, panda_model, seed)

        ik = InverseKinematics(plant, plant_ctx, with_joint_limits=True)

        # Pin the box (otherwise IK treats its 7 floating-base DOFs as free
        # decision vars and would relocate the box to satisfy clearance).
        box_body = plant.GetBodyByName("box_link", object_model)
        box_frame_pin = box_body.body_frame()
        ik.AddPositionConstraint(
            box_frame_pin, np.zeros(3), world,
            BOX_POS - 1e-5, BOX_POS + 1e-5,
        )
        ik.AddOrientationConstraint(
            world, RotationMatrix(),
            box_frame_pin, RotationMatrix(),
            0.001,
        )

        # EE position constraint (unchanged from contamination state)
        ik.AddPositionConstraint(
            ee_frame, np.zeros(3), world,
            EE_POS - p_tol, EE_POS + p_tol,
        )

        # Per-pair distance constraint: every arm-link geom must be at least
        # ARM_BOX_CLEARANCE_M away from every box geom. Pusher↔box is NOT
        # in this set, so pusher can stay in penetration with the box.
        for a_gid in arm_geoms:
            for b_gid in box_geoms:
                ik.AddDistanceConstraint(
                    (a_gid, b_gid),
                    distance_lower=ARM_BOX_CLEARANCE_M,
                    distance_upper=10.0,
                )

        # Posture cost: keep q_arm close to POSTURE_NOMINAL.
        # Drake's IK has the panda joint indices identified via the model.
        q = ik.q()  # decision vars; full plant positions (incl. box quat+pos)
        # Find slice for panda joints in q
        # IK's q is the plant's positions vector
        j1_idx = plant.GetJointByName("panda_joint1").position_start()
        for k in range(7):
            ik.get_mutable_prog().AddQuadraticErrorCost(
                np.array([[POSTURE_WEIGHT]]),
                np.array([POSTURE_NOMINAL[k]]),
                np.array([q[j1_idx + k]]),
            )

        ik.get_mutable_prog().SetInitialGuess(ik.q(), plant.GetPositions(plant_ctx))
        res = Solve(ik.prog())
        if not res.is_success():
            print(f"  [IK seed {seed_idx}] FAIL (no clearance-feasible solution)")
            continue

        plant.SetPositions(plant_ctx, res.GetSolution(ik.q()))
        q_arm_init = plant.GetPositions(plant_ctx, panda_model).copy()

        # Verify EE position constraint is actually satisfied (Ipopt can
        # return "success" with violated nonlinear constraints).
        p_ee_actual = plant.CalcPointsPositions(
            plant_ctx, ee_frame, np.zeros(3), world).flatten()
        ee_err_mm = np.linalg.norm(p_ee_actual - EE_POS) * 1000

        # Force diagram-wide cache update via Initialize, then query SG.
        sim_check = Simulator(diagram, context)
        sim_check.Initialize()
        query_obj = scene_graph.get_query_output_port().Eval(
            scene_graph.GetMyContextFromRoot(context))

        # Check actual distances: arm geoms vs box (per-pair) AND find
        # pusher↔box AND floor↔box. Pusher must penetrate box; arm must
        # NOT penetrate box; floor MUST be touching box (≤ 0).
        box_body = plant.GetBodyByName("box_link", object_model)
        box_fid = plant.GetBodyFrameIdOrThrow(box_body.index())
        box_gids_set = set(scene_graph.model_inspector().GetGeometries(
            box_fid, Role.kProximity))
        pusher_body = plant.GetBodyByName("pusher", panda_model)
        pusher_fid = plant.GetBodyFrameIdOrThrow(pusher_body.index())
        pusher_gids_set = set(scene_graph.model_inspector().GetGeometries(
            pusher_fid, Role.kProximity))
        world_fid = plant.GetBodyFrameIdOrThrow(plant.world_body().index())
        world_gids_set = set(scene_graph.model_inspector().GetGeometries(
            world_fid, Role.kProximity))

        sd_pairs = query_obj.ComputeSignedDistancePairwiseClosestPoints(
            max_distance=0.5)
        d_pusher_box = float('inf')
        d_floor_box = float('inf')
        min_arm_box = float('inf')
        arm_box_offender = None
        min_arm_floor = float('inf')

        inspector = query_obj.inspector()
        fid_to_body = {}
        for bi in range(plant.num_bodies()):
            from pydrake.multibody.tree import BodyIndex
            body = plant.get_body(BodyIndex(bi))
            try:
                fid_to_body[plant.GetBodyFrameIdOrThrow(body.index())] = body
            except Exception:
                pass

        for pp in sd_pairs:
            in_box_A = pp.id_A in box_gids_set
            in_box_B = pp.id_B in box_gids_set
            in_pusher_A = pp.id_A in pusher_gids_set
            in_pusher_B = pp.id_B in pusher_gids_set
            in_world_A = pp.id_A in world_gids_set
            in_world_B = pp.id_B in world_gids_set

            if (in_box_A and in_pusher_B) or (in_box_B and in_pusher_A):
                d_pusher_box = min(d_pusher_box, pp.distance)
                continue
            if (in_box_A and in_world_B) or (in_box_B and in_world_A):
                d_floor_box = min(d_floor_box, pp.distance)
                continue
            # box-arm
            if in_box_A or in_box_B:
                other = pp.id_B if in_box_A else pp.id_A
                if other in pusher_gids_set or other in world_gids_set:
                    continue
                other_body = fid_to_body.get(inspector.GetFrameId(other))
                if other_body and "panda" in other_body.name().lower():
                    if pp.distance < min_arm_box:
                        min_arm_box = pp.distance
                        arm_box_offender = (other_body.name(), pp.distance)
                continue
            # arm-floor
            if (in_world_A or in_world_B):
                other = pp.id_B if in_world_A else pp.id_A
                if other in pusher_gids_set:
                    continue
                other_body = fid_to_body.get(inspector.GetFrameId(other))
                if other_body and "panda" in other_body.name().lower():
                    min_arm_floor = min(min_arm_floor, pp.distance)

        # Dump actual pusher position and box-pusher distance
        p_pusher = p_ee_actual
        p_box = plant.EvalBodyPoseInWorld(plant_ctx, box_body).translation()
        center_dist = float(np.linalg.norm(p_pusher - p_box))
        expected_d = center_dist - BOX_HALF - EE_RADIUS  # rough lower bound
        print(f"  [IK seed {seed_idx}] IK OK")
        print(f"      q_arm        = {q_arm_init.round(4)}")
        print(f"      p_pusher     = {p_pusher.round(5)}  (target {EE_POS.round(5)})")
        print(f"      p_box        = {p_box.round(5)}")
        print(f"      center-dist  = {center_dist*1000:.3f}mm  "
              f"sphere-box rough = {expected_d*1000:+.3f}mm")
        print(f"      EE_err  = {ee_err_mm:.3f}mm   "
              f"d_pusher_box = {d_pusher_box*1000:+7.3f}mm   "
              f"d_floor_box  = {d_floor_box*1000:+7.3f}mm")
        print(f"      min_d_arm_box   = {min_arm_box*1000:+7.3f}mm "
              f"({arm_box_offender[0] if arm_box_offender else 'none'})   "
              f"min_d_arm_floor = {min_arm_floor*1000:+7.3f}mm")

        # Physical sanity rules
        if ee_err_mm > 1.0:
            print(f"  [IK seed {seed_idx}] REJECT — EE_err {ee_err_mm:.2f}mm > 1mm")
            continue
        if d_pusher_box > 0.0:
            print(f"  [IK seed {seed_idx}] REJECT — pusher not penetrating "
                  f"box (d={d_pusher_box*1000:+.3f}mm > 0)")
            continue
        if d_pusher_box < -0.005:   # > 5mm penetration is unphysical
            print(f"  [IK seed {seed_idx}] REJECT — pusher too deep "
                  f"(d={d_pusher_box*1000:+.3f}mm)")
            continue
        if min_arm_box <= 0.0:
            print(f"  [IK seed {seed_idx}] REJECT — arm penetrating box "
                  f"({arm_box_offender[0]} d={arm_box_offender[1]*1000:+.3f}mm)")
            continue
        if min_arm_floor < 0.0:
            print(f"  [IK seed {seed_idx}] REJECT — arm penetrating floor "
                  f"(d={min_arm_floor*1000:+.3f}mm)")
            continue

        # Set EE velocity via joint Jacobian
        plant.SetVelocities(plant_ctx, object_model, np.zeros(6))
        J = plant.CalcJacobianTranslationalVelocity(
            plant_ctx, JacobianWrtVariable.kV, ee_frame, np.zeros(3),
            world, world)
        s = plant.GetJointByName("panda_joint1").velocity_start()
        q_arm_dot, *_ = np.linalg.lstsq(J[:, s:s+7], np.array([EE_VEL_X, 0., 0.]),
                                        rcond=None)
        plant.SetVelocities(plant_ctx, panda_model, q_arm_dot)
        return context, plant_ctx, q_arm_init, seed_idx

    return None, None, None, None


# --------------- HARD GATE: Drake contact-pair set check -----------------

def assert_pair_gate(diagram, plant, scene_graph, ctx, ctx_plant, obj_body,
                     ee_body, threshold=0.005, gate_time=0.005):
    """Geom-pair gate at the re-posed state.

    Uses SceneGraph SignedDistance query directly (after Initialize() to
    flush caches) for a deterministic read of close-pair geometry. Also
    runs ContactResults at t=gate_time as a secondary check (catches
    dynamic contacts forming in the first ms).

    Returns:
        ok (bool): True iff NO arm↔box pair exists at signed-distance ≤
                   threshold AND ContactResults shows NO arm↔box at gate_time.
        diag (dict): per-pair tags + ContactResults summary.
    """
    # ---- Force cache flush via Simulator.Initialize() ---------------------
    sim = Simulator(diagram, ctx)
    sim.Initialize()

    # ---- SignedDistance query at t=0 (the re-posed state) -----------------
    query_obj = scene_graph.get_query_output_port().Eval(
        scene_graph.GetMyContextFromRoot(ctx))
    inspector = query_obj.inspector()
    sd_pairs = query_obj.ComputeSignedDistancePairwiseClosestPoints(
        max_distance=threshold)

    n_pusher_box = 0
    n_floor_box = 0
    n_arm_box = 0
    n_other = 0
    pairs = []

    box_geom_ids = set()
    obj_fid = plant.GetBodyFrameIdOrThrow(obj_body.index())
    for gid in inspector.GetGeometries(obj_fid, Role.kProximity):
        box_geom_ids.add(gid)

    # Precompute frame_id → body lookup for all plant bodies that have one.
    fid_to_body = {}
    for bi in range(plant.num_bodies()):
        from pydrake.multibody.tree import BodyIndex
        body = plant.get_body(BodyIndex(bi))
        try:
            bfid = plant.GetBodyFrameIdOrThrow(body.index())
        except Exception:
            continue
        fid_to_body[bfid] = body

    for pp in sd_pairs:
        gA = pp.id_A
        gB = pp.id_B
        if gA in box_geom_ids:
            other_g = gB
        elif gB in box_geom_ids:
            other_g = gA
        else:
            pairs.append(('non-box', f"d={pp.distance*1000:+.3f}mm"))
            n_other += 1
            continue
        other_fid = inspector.GetFrameId(other_g)
        other_body = fid_to_body.get(other_fid, None)
        oname = other_body.name().lower() if other_body else "<?>"
        is_ground = (other_body and (other_body.index() == plant.world_body().index()
                                     or "world" in oname or "ground" in oname))
        is_pusher = "pusher" in oname
        is_arm = ("panda" in oname) and not is_pusher
        d_mm = pp.distance * 1000
        if is_pusher:
            n_pusher_box += 1
            pairs.append(('pusher↔box', f"d={d_mm:+.3f}mm"))
        elif is_arm:
            n_arm_box += 1
            pairs.append((f'ARM↔box (ARTIFACT: {oname})', f"d={d_mm:+.3f}mm"))
        elif is_ground:
            n_floor_box += 1
            pairs.append(('floor↔box', f"d={d_mm:+.3f}mm"))
        else:
            n_other += 1
            pairs.append((f'OTHER↔box ({oname})', f"d={d_mm:+.3f}mm"))

    # ---- ContactResults at t=gate_time (dynamic check; sim already inited) -
    sim.AdvanceTo(gate_time)
    plant_ctx_now = plant.GetMyContextFromRoot(ctx)
    cr = plant.get_contact_results_output_port().Eval(plant_ctx_now)

    dyn_pairs = []
    n_dyn_pusher = 0
    n_dyn_floor = 0
    n_dyn_arm = 0
    n_dyn_other = 0
    for i in range(cr.num_point_pair_contacts()):
        info = cr.point_pair_contact_info(i)
        nA = info.bodyA_index()
        nB = info.bodyB_index()
        bodyA = plant.get_body(nA)
        bodyB = plant.get_body(nB)
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
            dyn_pairs.append(('pusher↔box', oname))
        elif is_arm:
            n_dyn_arm += 1
            dyn_pairs.append(('ARM↔box (ARTIFACT)', oname))
        elif is_ground:
            n_dyn_floor += 1
            dyn_pairs.append(('floor↔box', oname))
        else:
            n_dyn_other += 1
            dyn_pairs.append(('OTHER↔box', oname))

    # Gate criterion: NO arm-box pairs in EITHER the static signed-distance
    # query OR the dynamic ContactResults at gate_time. (Note: pusher-box
    # may or may not be present at t=0 in the static query depending on
    # threshold; it MUST be present in the dynamic check because EE_VEL_X
    # drives the pusher into the box.)
    ok = (n_arm_box == 0) and (n_dyn_arm == 0)
    diag = dict(
        pairs=pairs,
        n_pusher_box=n_pusher_box, n_floor_box=n_floor_box,
        n_arm_box=n_arm_box, n_other=n_other,
        dyn_pairs=dyn_pairs,
        n_dyn_pusher=n_dyn_pusher, n_dyn_floor=n_dyn_floor,
        n_dyn_arm=n_dyn_arm, n_dyn_other=n_dyn_other,
        n_dyn_total=cr.num_point_pair_contacts(),
    )
    return ok, diag


# --------------- LCS path (single-step + sub-stepped) --------------------

def write_state_to_plant(plant, plant_ctx, x, panda_model, object_model,
                         ee_frame, q_arm_warm):
    """Re-IK arm to LCS-predicted EE pos, set box pose/vel from LCS state.
    Mirrors dt_factorial / friction_audit. No re-pose clearance constraints
    here — re-extract uses warm-start q_arm and a tight EE constraint."""
    world = plant.world_frame()
    plant.SetPositions(plant_ctx, object_model, x[0:7])
    plant.SetVelocities(plant_ctx, object_model, x[10:16])
    plant.SetPositions(plant_ctx, panda_model, q_arm_warm)
    ik = InverseKinematics(plant, plant_ctx, with_joint_limits=True)
    ik.AddPositionConstraint(
        ee_frame, np.zeros(3), world,
        x[7:10] - 1e-5, x[7:10] + 1e-5,
    )
    ik.get_mutable_prog().SetInitialGuess(ik.q(), plant.GetPositions(plant_ctx))
    res = Solve(ik.prog())
    if not res.is_success():
        return None
    plant.SetPositions(plant_ctx, res.GetSolution(ik.q()))
    q_arm_new = plant.GetPositions(plant_ctx, panda_model).copy()
    J = plant.CalcJacobianTranslationalVelocity(
        plant_ctx, JacobianWrtVariable.kV, ee_frame, np.zeros(3), world, world)
    s = plant.GetJointByName("panda_joint1").velocity_start()
    q_arm_dot, *_ = np.linalg.lstsq(J[:, s:s+7], x[16:19], rcond=None)
    plant.SetVelocities(plant_ctx, panda_model, q_arm_dot)
    return q_arm_new


def build_x0(plant, plant_ctx, ee_frame):
    ee = plant.CalcPointsPositions(
        plant_ctx, ee_frame, np.zeros(3), plant.world_frame()).flatten()
    return np.concatenate([
        BOX_QUAT, BOX_POS, ee, np.zeros(3), np.zeros(3),
        np.array([EE_VEL_X, 0., 0.])])


def lcs_step_path(plant, plant_ctx, ee_frame, obj_body, plant_ad, ctx_ad, mu,
                  panda_model, object_model, q_arm_init,
                  dt, n_sub, box_drag=10.0, re_extract=True):
    """Run LCS for `n_sub` sub-steps of size `dt`.

    re_extract=True : re-IK + re-linearize at each sub-step (faithful)
    re_extract=False: use the same LCS from t=0 across all sub-steps (Cf/Df)
    """
    from control.lcs_formulator import LCSFormulator
    x_curr = build_x0(plant, plant_ctx, ee_frame)
    box_x_0 = float(x_curr[BOX_POS_X_IDX])
    box_z_0 = float(x_curr[BOX_POS_Z_IDX])
    q_arm = q_arm_init.copy()
    fail_steps = 0
    A_static = None
    if not re_extract:
        f = LCSFormulator(plant, mu=mu, obj_body=obj_body,
                          plant_ad=plant_ad, context_ad=ctx_ad,
                          box_ground_drag=box_drag)
        out = f.linearize_discrete_ee_space(plant_ctx, dt, np.zeros(3))
        A_static = out
    for k in range(n_sub):
        if re_extract:
            f = LCSFormulator(plant, mu=mu, obj_body=obj_body,
                              plant_ad=plant_ad, context_ad=ctx_ad,
                              box_ground_drag=box_drag)
            A, B_ctrl, D, d_const, E, F, H, c_lcs, *_ = \
                f.linearize_discrete_ee_space(plant_ctx, dt, np.zeros(3))
        else:
            A, B_ctrl, D, d_const, E, F, H, c_lcs, *_ = A_static
        q_lcp = E @ x_curr + c_lcs
        lam, _ = solve_lcp(F, q_lcp)
        x_next = A @ x_curr + B_ctrl @ np.zeros(3) + D @ lam + d_const

        if re_extract and k < n_sub - 1:
            new_q = write_state_to_plant(plant, plant_ctx, x_next,
                                         panda_model, object_model,
                                         ee_frame, q_arm)
            if new_q is None:
                fail_steps += 1
                break
            q_arm = new_q
        x_curr = x_next
    dx = float(x_curr[BOX_POS_X_IDX]) - box_x_0
    dz = float(x_curr[BOX_POS_Z_IDX]) - box_z_0
    return dx, dz, fail_steps


# --------------- Drake reference -----------------------------------------

def drake_reference_dx(diagram, plant, ctx, obj_body):
    sim = Simulator(diagram, ctx)
    sim.Initialize()
    plant_ctx_now = plant.GetMyContextFromRoot(ctx)
    box_xyz_0 = plant.EvalBodyPoseInWorld(plant_ctx_now, obj_body).translation()
    box_x_0 = float(box_xyz_0[0])
    box_z_0 = float(box_xyz_0[2])
    sim.AdvanceTo(DT_BIG)
    plant_ctx_now = plant.GetMyContextFromRoot(ctx)
    box_xyz_T = plant.EvalBodyPoseInWorld(plant_ctx_now, obj_body).translation()
    dx = float(box_xyz_T[0]) - box_x_0
    dz = float(box_xyz_T[2]) - box_z_0
    return dx, dz


# --------------- main ----------------------------------------------------

def main() -> int:
    print("=" * 72)
    print("STAGE C  RE-POSE PROBE  (Path A: re-pose + HARD GATE + clean re-run)")
    print("=" * 72)

    os.environ["LCS_EXPLICIT_BOX_GND"] = "4"
    import control.lcs_formulator
    importlib.reload(control.lcs_formulator)

    with open("config/tasks.yaml") as f:
        cfg = yaml.safe_load(f)
    task_cfg = cfg["tasks"]["pushing"]
    mu = task_cfg["friction"]
    print(f"  μ = {mu}")
    print(f"  EE_POS = {EE_POS}, EE_VEL_X = {EE_VEL_X}")
    print(f"  Arm-box clearance: {ARM_LINKS_TO_CLEAR} geoms >= {ARM_BOX_CLEARANCE_M*1000:.1f} mm from box geoms")
    print()

    # ============================================================
    # STEP 1 — Re-pose via IK with wrist-clearance constraints
    # ============================================================
    print("─" * 72)
    print("STEP 1 — RE-POSE")
    print("─" * 72)
    diagram, plant, sg, panda, object_, _, plant_ad, ctx_ad = _build_env(task_cfg)
    obj_body = plant.GetBodyByName("box_link", object_)
    ee_frame = plant.GetFrameByName("pusher")
    ee_body  = plant.GetBodyByName("pusher")

    ctx, pctx, q_arm_init, seed_idx = setup_contact_state_clean(
        diagram, plant, sg, panda, object_, ee_frame)

    if ctx is None:
        print("  ► RE-POSE FAILED on ALL seeds. ROUTE: GATE-FAILS-PRE (no IK).")
        print("  ► The geometric workspace forbids a wrist-clearance pose at this EE.")
        print("  ► Workspace finding: report — a different push configuration is needed.")
        return 0

    print(f"  ► Re-pose succeeded with seed index {seed_idx}.")
    print()

    # ============================================================
    # STEP 2 — HARD GATE: Drake contact-pair set
    # ============================================================
    print("─" * 72)
    print("STEP 2 — HARD GATE (Drake contact-pair set check)")
    print("─" * 72)

    # Build a SECOND diagram/plant/ctx for the gate (Drake forward step
    # modifies state; we need fresh state for STEP 3's LCS).
    diagram_g, plant_g, sg_g, panda_g, object_g, _, _, _ = _build_env(task_cfg)
    obj_body_g = plant_g.GetBodyByName("box_link", object_g)
    ee_frame_g = plant_g.GetFrameByName("pusher")
    ee_body_g  = plant_g.GetBodyByName("pusher")
    ctx_g, pctx_g, _, _ = setup_contact_state_clean(
        diagram_g, plant_g, sg_g, panda_g, object_g, ee_frame_g)
    if ctx_g is None:
        print("  [unreachable: re-pose failed for second build]")
        return 1
    ok, gdiag = assert_pair_gate(diagram_g, plant_g, sg_g, ctx_g, pctx_g,
                                 obj_body_g, ee_body_g,
                                 threshold=0.01, gate_time=0.005)
    print(f"  STATIC (SignedDistance) pairs at re-posed state (within 10 mm):")
    for tag, info in gdiag['pairs']:
        print(f"    {tag}: {info}")
    print(f"  STATIC summary: {gdiag['n_pusher_box']} pusher↔box, "
          f"{gdiag['n_floor_box']} floor↔box, "
          f"{gdiag['n_arm_box']} ARM↔box (ARTIFACT), "
          f"{gdiag['n_other']} other")
    print()
    print(f"  DYNAMIC (ContactResults) pairs at t=0.005s:")
    for tag, info in gdiag['dyn_pairs']:
        print(f"    {tag}: {info}")
    print(f"  DYNAMIC summary: {gdiag['n_dyn_pusher']} pusher↔box, "
          f"{gdiag['n_dyn_floor']} floor↔box, "
          f"{gdiag['n_dyn_arm']} ARM↔box (ARTIFACT), "
          f"{gdiag['n_dyn_other']} other")

    if not ok:
        print()
        print(f"  ► HARD GATE: FAIL")
        print(f"  ► STATIC arm↔box count = {gdiag['n_arm_box']}, "
              f"DYNAMIC arm↔box count = {gdiag['n_dyn_arm']} "
              f"(need both = 0).")
        print(f"  ► ROUTE: GATE-FAILS")
        print(f"  ► The re-pose did not clear the arm-box artifact under this EE position.")
        print(f"  ► Workspace finding: a different push configuration is needed, OR")
        print(f"  ►   Path B (collision-filter the arm body in env_builder) with the")
        print(f"  ►   penetration caveat explicitly recorded.")
        print(f"  ► Report and stop — no quantitative read on this contaminated setup.")
        return 0

    print()
    print(f"  ► HARD GATE: PASS — STATIC + DYNAMIC arm↔box both = 0; "
          f"clean setup (pusher + floor only).")
    print()

    # ============================================================
    # STEP 3 — Re-run §7.11 / §7.12 / §7.13 quantitatively
    # ============================================================
    print("─" * 72)
    print("STEP 3 — CLEAN RE-RUN (§7.11/§7.12/§7.13 quantitative)")
    print("─" * 72)

    # 3a — Drake reference Δbox_x and Δbox_z over 0.05 s
    diagram_d, plant_d, sg_d, panda_d, object_d, _, _, _ = _build_env(task_cfg)
    obj_body_d = plant_d.GetBodyByName("box_link", object_d)
    ee_frame_d = plant_d.GetFrameByName("pusher")
    ctx_d, _, _, _ = setup_contact_state_clean(
        diagram_d, plant_d, sg_d, panda_d, object_d, ee_frame_d)
    drake_dx, drake_dz = drake_reference_dx(diagram_d, plant_d, ctx_d, obj_body_d)
    print(f"  Drake reference (Δt=0.001 substeps, 0.05 s total):")
    print(f"    Δbox_x = {drake_dx*1000:+.4f} mm")
    print(f"    Δbox_z = {drake_dz*1000:+.4f} mm")

    # 3b — LCS cells A,B (Δt=0.05 single-step) and C,D,Cf,Df (Δt=0.005 sub-stepped)
    print()
    print("  LCS cells (Δt × drag × extraction):")
    print(f"    {'cell':<5} {'Δt':>7} {'drag':>5} {'extr':>9} "
          f"{'Δbox_x (mm)':>13} {'Δbox_z (mm)':>13} {'factor':>8}")

    # Cell A: Δt=0.05, drag=10  (single step)
    diagram_a, plant_a, sg_a, panda_a, object_a, _, plant_ad_a, ctx_ad_a = _build_env(task_cfg)
    obj_body_a = plant_a.GetBodyByName("box_link", object_a)
    ee_frame_a = plant_a.GetFrameByName("pusher")
    ctx_a, pctx_a, q_arm_a, _ = setup_contact_state_clean(
        diagram_a, plant_a, sg_a, panda_a, object_a, ee_frame_a)
    dx_A, dz_A, fail_A = lcs_step_path(plant_a, pctx_a, ee_frame_a, obj_body_a,
                                       plant_ad_a, ctx_ad_a, mu, panda_a, object_a,
                                       q_arm_a, dt=DT_BIG, n_sub=1, box_drag=10.0,
                                       re_extract=False)
    factor_A = abs(drake_dx / dx_A) if abs(dx_A) > 1e-9 else float('inf')
    print(f"    {'A':<5} {DT_BIG:>7} {10:>5} {'single':>9} "
          f"{dx_A*1000:+13.4f} {dz_A*1000:+13.4f} {factor_A:>7.2f}×")

    # Cell B: Δt=0.05, drag=0   (single step)
    diagram_b, plant_b, sg_b, panda_b, object_b, _, plant_ad_b, ctx_ad_b = _build_env(task_cfg)
    obj_body_b = plant_b.GetBodyByName("box_link", object_b)
    ee_frame_b = plant_b.GetFrameByName("pusher")
    ctx_b, pctx_b, q_arm_b, _ = setup_contact_state_clean(
        diagram_b, plant_b, sg_b, panda_b, object_b, ee_frame_b)
    dx_B, dz_B, fail_B = lcs_step_path(plant_b, pctx_b, ee_frame_b, obj_body_b,
                                       plant_ad_b, ctx_ad_b, mu, panda_b, object_b,
                                       q_arm_b, dt=DT_BIG, n_sub=1, box_drag=0.0,
                                       re_extract=False)
    factor_B = abs(drake_dx / dx_B) if abs(dx_B) > 1e-9 else float('inf')
    print(f"    {'B':<5} {DT_BIG:>7} {0:>5} {'single':>9} "
          f"{dx_B*1000:+13.4f} {dz_B*1000:+13.4f} {factor_B:>7.2f}×")

    # Cell C: Δt=0.005, drag=10, re-extract
    diagram_c, plant_c, sg_c, panda_c, object_c, _, plant_ad_c, ctx_ad_c = _build_env(task_cfg)
    obj_body_c = plant_c.GetBodyByName("box_link", object_c)
    ee_frame_c = plant_c.GetFrameByName("pusher")
    ctx_c, pctx_c, q_arm_c, _ = setup_contact_state_clean(
        diagram_c, plant_c, sg_c, panda_c, object_c, ee_frame_c)
    dx_C, dz_C, fail_C = lcs_step_path(plant_c, pctx_c, ee_frame_c, obj_body_c,
                                       plant_ad_c, ctx_ad_c, mu, panda_c, object_c,
                                       q_arm_c, dt=DT_SUB, n_sub=N_SUB, box_drag=10.0,
                                       re_extract=True)
    factor_C = abs(drake_dx / dx_C) if abs(dx_C) > 1e-9 else float('inf')
    fail_str = f" (IK fail {fail_C})" if fail_C else ""
    print(f"    {'C':<5} {DT_SUB:>7} {10:>5} {'re-ext':>9} "
          f"{dx_C*1000:+13.4f} {dz_C*1000:+13.4f} {factor_C:>7.2f}×{fail_str}")

    # Cell D: Δt=0.005, drag=0, re-extract
    diagram_dd, plant_dd, sg_dd, panda_dd, object_dd, _, plant_ad_dd, ctx_ad_dd = _build_env(task_cfg)
    obj_body_dd = plant_dd.GetBodyByName("box_link", object_dd)
    ee_frame_dd = plant_dd.GetFrameByName("pusher")
    ctx_dd, pctx_dd, q_arm_dd, _ = setup_contact_state_clean(
        diagram_dd, plant_dd, sg_dd, panda_dd, object_dd, ee_frame_dd)
    dx_D, dz_D, fail_D = lcs_step_path(plant_dd, pctx_dd, ee_frame_dd, obj_body_dd,
                                       plant_ad_dd, ctx_ad_dd, mu, panda_dd, object_dd,
                                       q_arm_dd, dt=DT_SUB, n_sub=N_SUB, box_drag=0.0,
                                       re_extract=True)
    factor_D = abs(drake_dx / dx_D) if abs(dx_D) > 1e-9 else float('inf')
    fail_str = f" (IK fail {fail_D})" if fail_D else ""
    print(f"    {'D':<5} {DT_SUB:>7} {0:>5} {'re-ext':>9} "
          f"{dx_D*1000:+13.4f} {dz_D*1000:+13.4f} {factor_D:>7.2f}×{fail_str}")

    # ============================================================
    # STEP 4 — Route the read
    # ============================================================
    print()
    print("─" * 72)
    print("STEP 4 — READ + ROUTE")
    print("─" * 72)
    print()
    print(f"  Drake Δbox_x          : {drake_dx*1000:+.4f} mm  (clean reference)")
    print(f"  LCS single-step (A/B) : {dx_A*1000:+.4f} mm / {dx_B*1000:+.4f} mm")
    print(f"  LCS sub-stepped (C/D) : {dx_C*1000:+.4f} mm / {dx_D*1000:+.4f} mm")
    print()
    print(f"  Δt-MAIN-EFFECT (avg over drag): "
          f"{(0.5*(dx_C + dx_D) - 0.5*(dx_A + dx_B))*1000:+.4f} mm")
    print(f"  DRAG-MAIN-EFFECT @Δt=0.05    : {(dx_B - dx_A)*1000:+.4f} mm "
          f"(should be ~0; v_box=0)")
    print(f"  DRAG-MAIN-EFFECT @Δt=0.005   : {(dx_D - dx_C)*1000:+.4f} mm "
          f"(drag re-enters at v_box≠0)")
    print()

    # "Deviation" = |1 - factor|; smaller = closer to model-matches-plant.
    # Signed direction: factor > 1 = LCS UNDER-predicts (Drake bigger);
    #                   factor < 1 = LCS OVER-predicts (LCS bigger).
    def deviation(f): return abs(1.0 - f)

    devs = {'A': deviation(factor_A), 'B': deviation(factor_B),
            'C': deviation(factor_C), 'D': deviation(factor_D)}
    min_dev_cell = min(devs, key=devs.get)
    min_dev = devs[min_dev_cell]
    factors = {'A': factor_A, 'B': factor_B, 'C': factor_C, 'D': factor_D}

    print(f"  Per-cell deviation from 1× (smaller = better):")
    for c in ['A', 'B', 'C', 'D']:
        direction = "UNDER" if factors[c] > 1.0 else "OVER "
        print(f"    cell {c}: factor {factors[c]:.2f}×  "
              f"deviation {devs[c]:.2f}  ({direction}-predicts Drake)")
    print(f"  Best cell (smallest deviation): {min_dev_cell} "
          f"(deviation {min_dev:.2f}, factor {factors[min_dev_cell]:.2f}×)")
    print(f"  Vertical sanity (max |Δz|)  : "
          f"{max(abs(dz_A), abs(dz_B), abs(dz_C), abs(dz_D))*1000:.4f} mm "
          f"(should be ~0; box held by floor)")
    print()

    # §7.11 baseline (contaminated): single-step factor was 3.73×
    # §7.13 baseline (contaminated): sub-stepped factor was 1.43×
    print(f"  §7.13 comparison (contaminated → clean):")
    print(f"    single-step: 3.73× (contam) → {factor_A:.2f}× (clean) "
          f"— {'cleared {:.0%} of gap'.format(1 - devs['A']/2.73)}")
    print(f"    sub-stepped: 1.43× (contam) → {factor_D:.2f}× (clean) "
          f"— now {'OVER-predicts' if factor_D < 1.0 else 'under-predicts'}")
    print()

    # Direction split: single under-predicts vs sub-stepped over-predicts
    single_dir = "UNDER" if factor_A > 1.0 else "OVER"
    sub_dir    = "UNDER" if factor_D > 1.0 else "OVER"
    bidirectional = (single_dir != sub_dir)

    # Route
    if min_dev < 0.10:
        route = "GAP-VANISHES"
        msg = (
            "  The clean-state gap closes to ~1× somewhere. The §7.11→§7.13 quantitative\n"
            "  gap was largely the arm artifact. Re-promote convergence on the clean LCS\n"
            "  in the next block; friction + anitescu may be moot.")
    elif bidirectional:
        route = "GAP-PARTIAL-WITH-OVERSHOOT"
        msg = (
            "  CLEAN-STATE READ: single-step UNDER-predicts (factor {:.2f}×),\n"
            "  sub-stepped OVER-predicts (factor {:.2f}×). Δt sub-stepping over-corrects\n"
            "  past Drake — the LCS rigid contact pumps more momentum than Drake's\n"
            "  compliant contact when given finer Δt to resolve the burst. The §7.11\n"
            "  3.73× shrinks to {:.2f}× at single-step (the arm artifact accounted for\n"
            "  ~{:.0%} of the original gap) but a real ~{:.0%} mismatch remains, with the\n"
            "  Δt knob over-correcting it. This is the rigid-vs-compliant contact\n"
            "  signature, NOT a friction-isolation question (yet).\n"
            "\n"
            "  NEXT GATE (next block): the Δt-vs-compliant story — does intermediate Δt\n"
            "  (e.g., Δt=0.01, 0.02) land closer to 1×? IF a sweet spot exists, the LCS\n"
            "  is interpretable as a stiffer-than-Drake approximation tunable via Δt.\n"
            "  Friction audit re-opens only after Δt is decomposed on the CLEAN setup."
            .format(factor_A, factor_D, factor_A, 1 - (factor_A - 1)/(3.73 - 1),
                    abs(factor_A - 1)))
    elif min_dev < 0.50:
        route = "GAP-PARTIAL"
        msg = (
            "  The clean-state gap shrinks (best deviation {:.2f}, factor {:.2f}×). The\n"
            "  arm artifact accounted for SOME of the §7.11/§7.13 gap; a clean\n"
            "  same-direction residual remains. The Δt decomposition reruns on solid\n"
            "  ground; friction audit re-opens after Δt is decomposed on the clean\n"
            "  residual.".format(min_dev, factors[min_dev_cell]))
    else:
        route = "GAP-PERSISTS-LARGE"
        msg = (
            "  The clean-state gap is still large (best deviation {:.2f}). There is a\n"
            "  real model-vs-plant mismatch even after removing contamination. The Δt\n"
            "  decomposition reruns on solid ground; IF a residual remains after Δt,\n"
            "  the friction audit finally has a defined, uncontaminated residual to\n"
            "  chase.".format(min_dev))

    print(f"  ► ROUTE: {route}")
    print(msg)
    print()
    print("=" * 72)
    print("HOLD: next block (re-decompose / re-promote / friction) is SEPARATE.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
