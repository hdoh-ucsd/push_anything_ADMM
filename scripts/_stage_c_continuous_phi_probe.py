"""Stage C finer-resolution probe — continuous-φ classification (offline).

Question (§7.19 routed): are the shallow-depth "disengages" in the §7.18-aug
contact-state probe REAL contact loss, or a 1 ms boolean thresholding
artifact on a continuous-but-near-zero signed-distance signal?

§7.19 found AMBIGUOUS-leans-DEPTH-DEPENDENT under a 1 ms binary indicator;
the signed-distance co-trace (5 ms stride) showed the "disengages" were
sub-30-micron positive separations (0.000-0.028 mm). The next gate:
  - DROP the binary indicator
  - Dump continuous φ at sub-ms resolution
  - Classify each tick by φ≤0 (in contact) vs φ>0 (separated)
  - For each φ>0 region, record max(φ) → is it sub-grazing or meaningful

Weighted on the 0.2-0.5 mm LIVE band (live runs penetrate ~0.2-0.5 mm,
sitting at the §7.16 0.549 mm crossing).

Pre-registered routes (§7.19-aug):
  GRAZING-IS-ARTIFACT — max(φ) < grazing threshold at all depths → the
                        binary intermittency was a sampling artifact;
                        mechanism collapses to CONTINUOUS-COMPLIANT.
  GRAZING-IS-REAL    — shallow depths have meaningful φ > grazing
                       threshold → genuine depth-dependent grazing on
                       continuous base.
  LIVE-BAND-DECIDES — operational cut: the 0.2-0.5 mm answer matters
                      regardless of the full sweep.
  STILL-AMBIGUOUS    — even continuous-φ is marginal (φ oscillates around
                       0 with no clean ≤0/>0 separation) → marginal
                       contact is itself a finding.

Method (per depth, on the §7.16 sweep grid):
  1. Build plant at time_step=0.00025 (4× finer than the original 1 ms).
  2. Set up clean box-pinned state via setup_state_at_depth (reused).
  3. §7.14 per-depth contact-pair gate (reused).
  4. Run Drake AdvanceTo(0.00025 * k) for k = 1..200; at each tick dump
     the pusher↔box signed-distance φ from the SignedDistance query.
  5. Classify each tick: φ ≤ 0 (in contact / penetrating) vs φ > 0
     (separated). For each φ>0 region, record max(φ), region duration.
  6. Compare against grazing threshold (0.030 mm = the §7.19 cut).
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
LIVE_BAND_M = [0.0002, 0.0003, 0.0005]   # 0.2, 0.3, 0.5 mm — live operational band

DT_BIG    = 0.05
DT_DRAKE  = 0.00025                      # 0.25 ms — 4× finer than original
N_DRAKE   = int(round(DT_BIG / DT_DRAKE)) # 200 ticks

GRAZING_THRESHOLD_MM = 0.030             # the §7.19 sub-grazing cut

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
        return context, plant_ctx
    return None, None


def gate_check(diagram, plant, sg, ctx, obj_body, ee_body,
               threshold=0.010, gate_time=0.005):
    """The §7.14 per-state hard gate. Drake step on its own sim — leaves
    the main φ-trace to start from t=0 on a fresh sim."""
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


def continuous_phi_trace(diagram, plant, ctx, obj_body, ee_body):
    """Drop the binary indicator. Dump CONTINUOUS signed-distance φ for
    the pusher↔box pair at every Drake tick (DT_DRAKE = 0.25 ms).

    Returns dict with phi_trace (mm), separation regions, max-φ-per-region.
    """
    sim = Simulator(diagram, ctx)
    sim.Initialize()

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

    phi_trace_mm = np.full(N_DRAKE + 1, np.nan)

    for k in range(N_DRAKE + 1):
        if k > 0:
            sim.AdvanceTo(k * DT_DRAKE)
        sg_ctx = sg.GetMyContextFromRoot(ctx)
        qo = sg.get_query_output_port().Eval(sg_ctx)
        # Larger max_distance to capture any positive separation
        for pp in qo.ComputeSignedDistancePairwiseClosestPoints(max_distance=0.05):
            in_box = pp.id_A in box_gids or pp.id_B in box_gids
            other = pp.id_B if pp.id_A in box_gids else pp.id_A
            if in_box and other in pusher_gids:
                phi_trace_mm[k] = pp.distance * 1000.0   # to mm
                break

    # Classify regions: φ ≤ 0 (in contact) vs φ > 0 (separated)
    sep_mask = phi_trace_mm > 0.0
    # Find contiguous φ>0 regions
    sep_regions = []  # list of (start_k, end_k, max_phi_mm, duration_ms)
    in_sep = False
    seg_start = -1
    seg_max = 0.0
    for k in range(N_DRAKE + 1):
        if sep_mask[k]:
            if not in_sep:
                in_sep = True
                seg_start = k
                seg_max = phi_trace_mm[k]
            else:
                if phi_trace_mm[k] > seg_max:
                    seg_max = phi_trace_mm[k]
        else:
            if in_sep:
                seg_end = k - 1
                duration_ms = (seg_end - seg_start + 1) * DT_DRAKE * 1000.0
                sep_regions.append((seg_start, seg_end, seg_max, duration_ms))
                in_sep = False
    if in_sep:
        seg_end = N_DRAKE
        duration_ms = (seg_end - seg_start + 1) * DT_DRAKE * 1000.0
        sep_regions.append((seg_start, seg_end, seg_max, duration_ms))

    sep_ticks = int(np.sum(sep_mask))
    contact_ticks = (N_DRAKE + 1) - sep_ticks
    max_phi_mm = float(np.nanmax(phi_trace_mm)) if phi_trace_mm.size else float('nan')
    max_phi_clipped = max(max_phi_mm, 0.0)   # only positive matters for "separation peak"

    return dict(
        phi_trace_mm=phi_trace_mm,
        sep_regions=sep_regions,
        n_sep_regions=len(sep_regions),
        sep_ticks=sep_ticks,
        contact_ticks=contact_ticks,
        contact_fraction=contact_ticks / (N_DRAKE + 1),
        max_phi_mm=max_phi_clipped,
    )


def main() -> int:
    print("=" * 84)
    print("STAGE C  FINER-RESOLUTION PROBE  (continuous-φ classification, sub-ms)")
    print("=" * 84)
    os.environ["LCS_EXPLICIT_BOX_GND"] = "4"
    import control.lcs_formulator
    importlib.reload(control.lcs_formulator)

    with open("config/tasks.yaml") as f:
        cfg = yaml.safe_load(f)
    task_cfg = cfg["tasks"]["pushing"]
    mu = task_cfg["friction"]
    print(f"  μ = {mu}, Drake time_step = {DT_DRAKE*1000:.2f} ms "
          f"× {N_DRAKE+1} samples over {DT_BIG*1000:.0f} ms")
    print(f"  Sweep EE_PEN_M ∈ {[round(v*1000,2) for v in EE_PEN_VALUES_M]} mm")
    print(f"  LIVE band (weighted) ∈ {[round(v*1000,2) for v in LIVE_BAND_M]} mm")
    print(f"  Discriminator: continuous signed-distance φ (mm), classified")
    print(f"                 φ ≤ 0 (in contact) vs φ > 0 (separated).")
    print(f"  Grazing cut    : {GRAZING_THRESHOLD_MM:.3f} mm "
          f"(sub-30-micron = grazing artifact; ≥ = real separation).")
    print()

    rows = []
    for pen_m in EE_PEN_VALUES_M:
        pen_mm = pen_m * 1000
        print("─" * 84)
        print(f"DEPTH EE_PEN_M = {pen_mm:.2f} mm "
              + ("[LIVE BAND]" if pen_m in LIVE_BAND_M else ""))
        print("─" * 84)

        # Per-depth contact-pair gate (separate env to keep the trace fresh)
        d_g, p_g, sg_g, panda_g, obj_g, _, _, _ = _build_env(
            task_cfg, time_step=DT_DRAKE)
        obj_body_g = p_g.GetBodyByName("box_link", obj_g)
        ee_frame_g = p_g.GetFrameByName("pusher")
        ee_body_g  = p_g.GetBodyByName("pusher")
        ctx_g, _ = setup_state_at_depth(
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

        # Fresh build for the continuous-φ trace
        d, p, sg, panda, obj, _, _, _ = _build_env(task_cfg, time_step=DT_DRAKE)
        obj_body = p.GetBodyByName("box_link", obj)
        ee_frame = p.GetFrameByName("pusher")
        ee_body  = p.GetBodyByName("pusher")
        ctx, _ = setup_state_at_depth(d, p, sg, panda, obj, ee_frame, pen_m)
        res = continuous_phi_trace(d, p, ctx, obj_body, ee_body)
        rows.append(dict(pen_mm=pen_mm, status='CLEAN', **res))

        # Compact print of φ at every 1 ms (every 4th tick) over 50 ms = 51 samples
        every_ms = max(1, int(round(0.001 / DT_DRAKE)))
        phi_sub = res['phi_trace_mm'][::every_ms]
        phi_str = " ".join(
            (f"{v:+6.3f}" if not np.isnan(v) else "  NaN ")
            for v in phi_sub)
        print(f"  pusher↔box φ (mm) at every 1 ms (sub-sampled from {DT_DRAKE*1000:.2f}ms):")
        # Wrap long line
        for i in range(0, len(phi_sub), 13):
            chunk = phi_sub[i:i+13]
            line = " ".join(
                (f"{v:+6.3f}" if not np.isnan(v) else "  NaN ")
                for v in chunk)
            print(f"    [{i:>3}..{i+len(chunk)-1:<3}] {line}")
        print(f"  contact ticks: {res['contact_ticks']}/{N_DRAKE+1}  "
              f"({res['contact_fraction']*100:.1f}%)")
        print(f"  separated regions: {res['n_sep_regions']}")
        for (s, e, mp, dur) in res['sep_regions']:
            cls = "GRAZING" if mp < GRAZING_THRESHOLD_MM else "MEANINGFUL"
            print(f"    region ticks [{s:>3}..{e:<3}]  max φ = {mp:+.4f} mm  "
                  f"dur = {dur:.2f} ms  [{cls}]")
        print(f"  max(φ) over full window: {res['max_phi_mm']:+.4f} mm "
              f"({'GRAZING' if res['max_phi_mm'] < GRAZING_THRESHOLD_MM else 'MEANINGFUL'})")
        print()

    # Summary table
    print("=" * 84)
    print("SUMMARY  (per-depth continuous-φ classification)")
    print("=" * 84)
    print(f"  {'EE_PEN':>8} {'status':>10} {'contact %':>11} "
          f"{'n_sep':>6} {'max φ (mm)':>11} {'verdict':>11}")
    for r in rows:
        if r['status'] == 'CLEAN':
            verdict = "GRAZING" if r['max_phi_mm'] < GRAZING_THRESHOLD_MM else "MEANINGFUL"
            print(f"  {r['pen_mm']:>6.2f}mm {r['status']:>10} "
                  f"{r['contact_fraction']*100:>10.1f}% "
                  f"{r['n_sep_regions']:>6} "
                  f"{r['max_phi_mm']:>+10.4f}  {verdict:>10}")
        else:
            print(f"  {r['pen_mm']:>6.2f}mm {r['status']:>10} -- (no trace) --")

    # LIVE-band summary
    live_rows = [r for r in rows if r['status'] == 'CLEAN' and (r['pen_mm']/1000.0) in LIVE_BAND_M]
    print()
    print("LIVE-band (0.2 / 0.3 / 0.5 mm) summary:")
    for r in live_rows:
        verdict = "MAINTAINED (φ≤0)" if r['max_phi_mm'] <= 0.0 else (
            "GRAZING (sub-cut)" if r['max_phi_mm'] < GRAZING_THRESHOLD_MM else
            "SEPARATING (meaningful)")
        print(f"  {r['pen_mm']:>6.2f}mm  contact% = {r['contact_fraction']*100:>5.1f}  "
              f"max φ = {r['max_phi_mm']:+.4f} mm  → {verdict}")

    # Route
    print()
    print("=" * 84)
    print("ROUTE")
    print("=" * 84)
    clean_rows = [r for r in rows if r['status'] == 'CLEAN']
    if len(clean_rows) < 3:
        print("  ► INSUFFICIENT DATA")
        return 0

    max_phi_all = [r['max_phi_mm'] for r in clean_rows]
    max_phi_live = [r['max_phi_mm'] for r in live_rows]

    all_below_grazing = all(p < GRAZING_THRESHOLD_MM for p in max_phi_all)
    any_above_grazing_shallow = any(
        r['max_phi_mm'] >= GRAZING_THRESHOLD_MM
        for r in clean_rows if r['pen_mm'] < 0.7)
    live_max = max(max_phi_live) if max_phi_live else 0.0
    live_all_maintained = all(p <= 0.0 for p in max_phi_live)
    live_all_grazing = all(p < GRAZING_THRESHOLD_MM for p in max_phi_live)

    print(f"  max(φ) range across full sweep: "
          f"{min(max_phi_all):+.4f} mm — {max(max_phi_all):+.4f} mm")
    print(f"  max(φ) range across LIVE band : "
          f"{min(max_phi_live):+.4f} mm — {max(max_phi_live):+.4f} mm")
    print(f"  grazing threshold              : {GRAZING_THRESHOLD_MM:.3f} mm")
    print(f"  all below grazing?             : "
          f"{'YES' if all_below_grazing else 'NO'}")
    print(f"  shallow has meaningful sep?    : "
          f"{'YES' if any_above_grazing_shallow else 'NO'}")
    print(f"  LIVE band all MAINTAINED (φ≤0)?: "
          f"{'YES' if live_all_maintained else 'NO'}")
    print(f"  LIVE band all sub-grazing?     : "
          f"{'YES' if live_all_grazing else 'NO'}")
    print()

    if all_below_grazing:
        route = "GRAZING-IS-ARTIFACT"
        msg = (
            "  All depths' max(φ) < grazing threshold. The §7.19 binary 'disengages'\n"
            "  are sub-30-micron POSITIVE φ excursions — at-the-threshold grazing,\n"
            "  NOT meaningful separations. The 1 ms boolean was thresholding a\n"
            "  continuous-but-near-zero φ at exactly its noisy value; under continuous-\n"
            "  φ classification, the contact is essentially MAINTAINED (φ ≤ 0 or only\n"
            "  trivially positive). Mechanism collapses to CONTINUOUS-COMPLIANT.\n"
            "  §7.17 force-disconfirms FULLY rehabilitates (|F|≈0 was force-on-\n"
            "  maintained-contact). NEXT (separate block): re-do force-level probe\n"
            "  reading force-on-continuous-contact; if it confirms rigid-vs-compliant,\n"
            "  anitescu Part B RE-PROMOTED, force-level-confirmed.")
    elif any_above_grazing_shallow:
        route = "GRAZING-IS-REAL"
        msg = (
            "  Shallow depths exhibit max(φ) ≥ grazing threshold — meaningful positive\n"
            "  separations, not sub-grazing. The contact genuinely separates and\n"
            "  re-closes at shallow penetrations. Genuine DEPTH-DEPENDENT grazing-on-\n"
            "  continuous-base. The fix question is how the LCS handles a contact that\n"
            "  grazes / separates near the §7.16 0.549 mm crossing (NOT a pure\n"
            "  compliance reformulation). NEXT (separate block): characterize the\n"
            "  live-band regime specifically.")
    elif live_all_maintained or live_all_grazing:
        route = "LIVE-BAND-DECIDES — MAINTAINED"
        msg = (
            "  The LIVE band (0.2–0.5 mm) is MAINTAINED (φ ≤ 0) or sub-grazing\n"
            "  regardless of full-sweep behaviour. The deep-vs-shallow distinction is\n"
            "  academic for the live system; treat the operational regime as\n"
            "  continuous-compliant. NEXT (separate block): re-do force-level probe\n"
            "  on the live band; if compliance confirms, anitescu Part B RE-PROMOTED.")
    else:
        route = "STILL-AMBIGUOUS"
        msg = (
            "  Even continuous-φ at sub-ms is ambiguous — φ genuinely oscillates with\n"
            "  no clean ≤0/>0 separation, OR the sim's internal contact resolution is\n"
            "  itself sub-tick. MARGINAL contact is itself a finding (a marginal\n"
            "  contact is a different mechanism than either compliance or bouncing).\n"
            "  NEXT (separate block): the reference's own contact-model behaviour at\n"
            "  this state, for comparison.")

    print(f"  ► ROUTE: {route}")
    print(msg)
    print()
    print("=" * 84)
    print("HOLD: next block (force-level re-probe / live-band characterization /")
    print("      reference-comparison) is SEPARATE.")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
