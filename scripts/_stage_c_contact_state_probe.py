"""Stage C mechanism-name probe (offline).

Question (§7.18-aug routed): is the contact-axis residual COMPLIANCE
(continuous contact with stiffness mismatch) or INTERMITTENT BOUNCING
(disengage / re-engage dynamics)?

§7.16 displacement-level crossing at ~0.549 mm is REVALIDATED (§7.18).
§7.17 force-level showed Drake's force oscillating with zeros at
intermediate ticks — HINTING at intermittent contact, but a force VALUE
of zero is AMBIGUOUS (could be sampling noise on a continuous contact).
The CONTACT-STATE (geometrically engaged, yes/no) is the unambiguous
discriminator.

Method (per depth, on the §7.16 sweep grid):
  1. Set up the clean box-pinned state at this EE penetration.
  2. Per-depth §7.14 hard gate (pusher↔box + floor only, no arm-box).
  3. Run Drake AdvanceTo(0.001 * k) for k = 1..50; at each tick read
     Drake's ContactResults and CLASSIFY the pusher↔box pair:
       - PRESENT (the pair appears in ContactResults at this tick =
         geometrically engaged with measured penetration > 0) → 1
       - ABSENT (the pair does NOT appear in ContactResults = pusher
         and box are NOT penetrating) → 0
  4. From the per-tick binary trace, extract:
       - contact-engaged tick count (out of 50)
       - disengage events (1→0 transitions)
       - re-engage events (0→1 transitions)
       - longest contiguous engaged run

Read the PUSHER↔BOX PAIR SPECIFICALLY (not aggregated over all pairs;
a force VALUE of zero is ambiguous; PRESENCE in ContactResults is not).

Pre-registered routes (§7.18-aug):
  CONTINUOUS-COMPLIANT — pusher↔box engaged the full 50 ms at all depths
                         → compliance regains standing; re-do force-level
                         probe with this understanding.
  INTERMITTENT-DOMINATES — pusher↔box flickers (disengage/re-engage)
                            across most/all depths → dynamics are impulse-
                            and-recoil; rigid-vs-compliant is the wrong
                            frame; anitescu NOT indicated on this basis.
  DEPTH-DEPENDENT — continuity changes with depth (continuous at one end,
                    intermittent at the other) → the 0.549 mm crossing
                    may sit at a REGIME BOUNDARY.
  AMBIGUOUS — borderline cases; needs finer time resolution or a
              penetration-depth time series.
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

EE_PEN_VALUES_M = [0.0001, 0.0002, 0.0003, 0.0005, 0.0007, 0.001]
DT_BIG    = 0.05
DT_DRAKE  = 0.001
N_DRAKE   = int(round(DT_BIG / DT_DRAKE))   # 50 ticks

BOX_HALF  = 0.05
EE_RADIUS = 0.025
BOX_QUAT  = np.array([1.0, 0.0, 0.0, 0.0])
BOX_POS   = np.array([0.0, 0.0, 0.05])
EE_VEL_X  = -0.05

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
    """Set up clean box-pinned pusher-only state at this EE penetration."""
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


def gate_check(diagram, plant, sg, ctx, obj_body, ee_body,
               threshold=0.010, gate_time=0.005):
    """The §7.14 per-state hard gate. Run a separate Drake step on ITS OWN
    sim, so the main contact-state extraction starts from t=0 with a
    fresh sim."""
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
    return (n_arm == 0 and n_dyn_arm == 0), n_pusher, n_floor


def contact_state_trace(diagram, plant, ctx, obj_body, ee_body):
    """Run Drake AdvanceTo(0.001 * k) for k=1..50; at each tick, classify
    the pusher↔box pair as PRESENT in ContactResults (1) or ABSENT (0).

    Also captures, per tick:
      - normal force magnitude on box from pusher (for diagnostics)
      - signed distance pusher↔box (via SignedDistance query, the
        unambiguous geometric measure of contact).

    Returns dict with trace, contact_engaged, transitions, etc.
    """
    sim = Simulator(diagram, ctx)
    sim.Initialize()

    # Pre-extract pusher and box geom IDs from the inspector once
    sg = None
    for sys in diagram.GetSystems():
        if 'SceneGraph' in type(sys).__name__:
            sg = sys
            break
    query_obj = sg.get_query_output_port().Eval(sg.GetMyContextFromRoot(ctx))
    inspector = query_obj.inspector()
    box_fid = plant.GetBodyFrameIdOrThrow(obj_body.index())
    box_gids = set(inspector.GetGeometries(box_fid, Role.kProximity))
    pusher_fid = plant.GetBodyFrameIdOrThrow(ee_body.index())
    pusher_gids = set(inspector.GetGeometries(pusher_fid, Role.kProximity))

    trace = np.zeros(N_DRAKE + 1, dtype=int)      # binary contact present
    sd_trace = np.full(N_DRAKE + 1, np.nan)        # signed distance pusher↔box
    f_trace = np.zeros(N_DRAKE + 1)                # |F| pusher↔box

    for k in range(N_DRAKE + 1):
        if k > 0:
            sim.AdvanceTo(k * DT_DRAKE)
        pctx_now = plant.GetMyContextFromRoot(ctx)

        # ContactResults presence for pusher↔box pair
        cr = plant.get_contact_results_output_port().Eval(pctx_now)
        present = 0
        f_mag = 0.0
        for i in range(cr.num_point_pair_contacts()):
            info = cr.point_pair_contact_info(i)
            a = plant.get_body(info.bodyA_index())
            b = plant.get_body(info.bodyB_index())
            if (a.index() == obj_body.index() and b.index() == ee_body.index()) or \
               (b.index() == obj_body.index() and a.index() == ee_body.index()):
                present = 1
                F = info.contact_force()
                f_mag = float(np.linalg.norm(F))
                break
        trace[k] = present
        f_trace[k] = f_mag

        # SignedDistance for the pusher↔box pair (geometric, unambiguous)
        sg_ctx = sg.GetMyContextFromRoot(ctx)
        qo = sg.get_query_output_port().Eval(sg_ctx)
        for pp in qo.ComputeSignedDistancePairwiseClosestPoints(max_distance=0.05):
            in_box = pp.id_A in box_gids or pp.id_B in box_gids
            other = pp.id_B if pp.id_A in box_gids else pp.id_A
            if in_box and other in pusher_gids:
                sd_trace[k] = pp.distance
                break

    # Classify transitions (1->0 = disengage, 0->1 = re-engage)
    disengages = 0
    re_engages = 0
    for k in range(N_DRAKE):
        if trace[k] == 1 and trace[k+1] == 0:
            disengages += 1
        elif trace[k] == 0 and trace[k+1] == 1:
            re_engages += 1

    # Longest contiguous engaged run
    longest = 0
    cur = 0
    for v in trace:
        if v:
            cur += 1
            if cur > longest:
                longest = cur
        else:
            cur = 0

    engaged_fraction = float(np.mean(trace))

    return dict(
        trace=trace,
        sd_trace=sd_trace,
        f_trace=f_trace,
        engaged_ticks=int(np.sum(trace)),
        engaged_fraction=engaged_fraction,
        disengages=disengages,
        re_engages=re_engages,
        longest_engaged_run=longest,
    )


def main() -> int:
    print("=" * 84)
    print("STAGE C  CONTACT-STATE PROBE  (mechanism name: COMPLIANCE vs INTERMITTENT)")
    print("=" * 84)
    os.environ["LCS_EXPLICIT_BOX_GND"] = "4"
    import control.lcs_formulator
    importlib.reload(control.lcs_formulator)

    with open("config/tasks.yaml") as f:
        cfg = yaml.safe_load(f)
    task_cfg = cfg["tasks"]["pushing"]
    mu = task_cfg["friction"]
    print(f"  μ = {mu}, Drake 1ms ticks × {N_DRAKE+1} samples over {DT_BIG}s")
    print(f"  Sweep EE_PEN_M ∈ {[round(v*1000,2) for v in EE_PEN_VALUES_M]} mm")
    print(f"  Discriminator: ContactResults PRESENCE for pusher↔box pair "
          f"(NOT force value).")
    print()

    rows = []
    for pen_m in EE_PEN_VALUES_M:
        pen_mm = pen_m * 1000
        print("─" * 84)
        print(f"DEPTH EE_PEN_M = {pen_mm:.2f} mm")
        print("─" * 84)

        # Per-depth contact-pair gate
        d_g, p_g, sg_g, panda_g, obj_g, _, _, _ = _build_env(task_cfg)
        obj_body_g = p_g.GetBodyByName("box_link", obj_g)
        ee_frame_g = p_g.GetFrameByName("pusher")
        ee_body_g  = p_g.GetBodyByName("pusher")
        ctx_g, _, _, _ = setup_state_at_depth(
            d_g, p_g, sg_g, panda_g, obj_g, ee_frame_g, pen_m)
        if ctx_g is None:
            print(f"  re-pose FAIL — skip")
            rows.append(dict(pen_mm=pen_mm, status='REPOSE_FAIL'))
            continue
        gate_ok, n_pusher_init, n_floor_init = gate_check(
            d_g, p_g, sg_g, ctx_g, obj_body_g, ee_body_g)
        if not gate_ok:
            print(f"  GATE FAIL — arm↔box present at re-posed state")
            rows.append(dict(pen_mm=pen_mm, status='GATE_FAIL'))
            continue
        print(f"  GATE PASS (pusher={n_pusher_init}, floor={n_floor_init}, arm=0)")

        # Fresh build for the trace
        d, p, sg, panda, obj, _, _, _ = _build_env(task_cfg)
        obj_body = p.GetBodyByName("box_link", obj)
        ee_frame = p.GetFrameByName("pusher")
        ee_body  = p.GetBodyByName("pusher")
        ctx, _, _, _ = setup_state_at_depth(
            d, p, sg, panda, obj, ee_frame, pen_m)
        res = contact_state_trace(d, p, ctx, obj_body, ee_body)
        rows.append(dict(pen_mm=pen_mm, status='CLEAN', **res))

        # Print binary trace + summary
        binary_str = "".join(str(int(v)) for v in res['trace'])
        # Print SD trace at every 5th tick for compactness
        sd_str = " ".join(
            (f"{res['sd_trace'][i]*1000:+6.3f}" if not np.isnan(res['sd_trace'][i])
             else "  NaN ")
            for i in range(0, N_DRAKE + 1, 5))
        f_str = " ".join(f"{res['f_trace'][i]:>5.2f}"
                          for i in range(0, N_DRAKE + 1, 5))
        print(f"  pusher↔box PRESENCE per 1ms (51 ticks 0..50):")
        print(f"    [{binary_str}]")
        print(f"  pusher↔box signed-distance (mm) at every 5ms:")
        print(f"    {sd_str}")
        print(f"  pusher↔box |F| (N) at every 5ms:")
        print(f"    {f_str}")
        print(f"  engaged ticks: {res['engaged_ticks']}/{N_DRAKE+1}  "
              f"({res['engaged_fraction']*100:.1f}%)  "
              f"longest run: {res['longest_engaged_run']}")
        print(f"  disengages: {res['disengages']}  re-engages: {res['re_engages']}")
        print()

    # Summary table
    print("=" * 84)
    print("SUMMARY  (per-depth contact-state continuity)")
    print("=" * 84)
    print(f"  {'EE_PEN':>8} {'status':>10} {'engaged %':>11} "
          f"{'longest run':>13} {'disengage':>10} {'re-engage':>10}")
    for r in rows:
        if r['status'] == 'CLEAN':
            print(f"  {r['pen_mm']:>6.2f}mm {r['status']:>10} "
                  f"{r['engaged_fraction']*100:>10.1f}% "
                  f"{r['longest_engaged_run']:>13} "
                  f"{r['disengages']:>10} {r['re_engages']:>10}")
        else:
            print(f"  {r['pen_mm']:>6.2f}mm {r['status']:>10} -- (no trace) --")

    # Route
    print()
    print("=" * 84)
    print("ROUTE")
    print("=" * 84)
    clean_rows = [r for r in rows if r['status'] == 'CLEAN']
    if len(clean_rows) < 3:
        print("  ► INSUFFICIENT DATA")
        return 0

    engaged_fracs = [r['engaged_fraction'] for r in clean_rows]
    disengage_counts = [r['disengages'] for r in clean_rows]
    longest_runs = [r['longest_engaged_run'] for r in clean_rows]
    pens = [r['pen_mm'] for r in clean_rows]

    all_continuous = all(d == 0 for d in disengage_counts)
    all_intermittent = all(d >= 2 for d in disengage_counts)
    # Depth-dependent: continuity varies systematically with depth
    monotonic_disengage = all(disengage_counts[i] <= disengage_counts[i+1]
                              for i in range(len(disengage_counts) - 1)) or \
                          all(disengage_counts[i] >= disengage_counts[i+1]
                              for i in range(len(disengage_counts) - 1))
    spread_disengage = max(disengage_counts) - min(disengage_counts)

    print(f"  engaged fraction range: "
          f"{min(engaged_fracs)*100:.1f}% — {max(engaged_fracs)*100:.1f}%")
    print(f"  disengage count range : {min(disengage_counts)} — {max(disengage_counts)}")
    print(f"  longest run range     : {min(longest_runs)} — {max(longest_runs)}")
    print(f"  all continuous (zero disengages)? : {'YES' if all_continuous else 'NO'}")
    print(f"  all intermittent (≥2 disengages)? : {'YES' if all_intermittent else 'NO'}")
    print(f"  monotonic w/ depth?               : "
          f"{'YES' if monotonic_disengage else 'NO'}  (spread {spread_disengage})")
    print()

    if all_continuous:
        route = "CONTINUOUS-COMPLIANT"
        msg = (
            "  pusher↔box stays engaged the full 50 ms at every depth. The §7.17\n"
            "  force oscillation was sampling/noise on a continuous contact — force≈0\n"
            "  with contact present, NOT contact loss. Compliance REGAINS standing\n"
            "  as the mechanism name. NEXT (separate block): re-do the force-level\n"
            "  probe with this understanding (read force-with-contact-present). If\n"
            "  the force profile then confirms rigid-vs-compliant → anitescu RE-\n"
            "  PROMOTED, force-level-confirmed for real this time.")
    elif all_intermittent:
        route = "INTERMITTENT-DOMINATES"
        msg = (
            "  pusher↔box flickers (disengage/re-engage multiple times) at every\n"
            "  tested depth. The dynamics are IMPULSE-AND-RECOIL, NOT continuous\n"
            "  compliant push. 'Rigid-vs-compliant' is the WRONG (or incomplete)\n"
            "  frame; the fix question changes (it's about how the LCS handles a\n"
            "  bouncing contact, not normal-compliance reformulation). The 0.549 mm\n"
            "  crossing is then COINCIDENTAL impulse-matching, NOT a compliance\n"
            "  match. Anitescu (a compliance reformulation) is NOT the indicated\n"
            "  fix on this basis. NEXT (separate block): re-pose the fix question\n"
            "  in impulse-and-recoil terms.")
    elif monotonic_disengage and spread_disengage >= 2:
        route = "DEPTH-DEPENDENT"
        msg = (
            "  The contact-state continuity CHANGES systematically with penetration\n"
            "  depth. The 0.549 mm crossing may sit at a REGIME BOUNDARY between\n"
            "  continuous-compliant (one side) and impulse-recoil (the other).\n"
            "  'Compliance vs bouncing' is NOT a single answer across the sweep —\n"
            "  it's two regimes with a transition. NEXT (separate block):\n"
            "  characterize the transition; the fix question depends on WHICH\n"
            "  regime the LIVE push operates in.")
    else:
        route = "AMBIGUOUS / MIXED"
        msg = (
            "  The contact-state pattern is mixed — some depths continuous, some\n"
            "  intermittent, no monotonic depth dependence. May need finer time-\n"
            "  resolution OR a penetration-depth time series (continuous signed-\n"
            "  distance) rather than binary contact indicator. Re-examine.")

    print(f"  ► ROUTE: {route}")
    print(msg)
    print()
    print("=" * 84)
    print("HOLD: next block (mechanism-fix scoping / re-probe / etc) is SEPARATE.")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
