"""Stage C Candidate E validation — clamped-φ/dt saturating-stiffness on
EE-BOX-only, behind LCS_NORMAL_PHI_CLAMP env var. §7.27.

Reverse-engineered prediction from §7.25 linear fit
    box_v_x ≈ −0.84·(depth/dt) − 0.036
and §7.26 separability:
    With v_cap ≈ 0.034 m/s, depth=0.10mm passes unclamped → -0.0528;
    depth=0.549mm and 1.00mm clamp to (depth/dt)_eff = v_cap →
    box_v_x ≈ -0.0646 for both. All three in the §7.24 band.

This probe:
  • runs Drake on-the-fly at 5 depths {0.10, 0.25, 0.549, 0.75, 1.00}mm
    (anchors + held-out) with Drake dt=0.00025s, DT_SUB=5ms
  • runs the LCS once per (depth, v_cap) cell at v_cap ∈
    {0.025, 0.030, 0.034, 0.040, 0.050}
  • finds the best v_cap satisfying the §7.24 band at the 3 anchors
  • THEN checks held-out depths {0.25, 0.75}mm at that v_cap against Drake
    — the overfitting-shape check.

Pre-registered routes:
  E-PASSES                      — best v_cap satisfies band at all 3
                                  anchors AND held-out box_v_x agrees
                                  with Drake within DEEP_TOL
  E-PASSES-ANCHORS-FAILS-HELDOUT — band OK at anchors but held-out shape
                                  diverges from Drake → clamp is an
                                  interpolation, not the physics
  E-FAILS-ANCHORS               — no v_cap satisfies the band at all 3
                                  anchors → linear fit didn't hold under
                                  clamping (LCP active set shifted)
  E-BREAKS-VERTICAL / GATE-DIRTY — clamping perturbs BOX-VERT coupling
                                  / gate goes dirty / vertical regresses
                                  → diagnose before proceeding
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
from pydrake.systems.analysis import Simulator

from sim.env_builder import build_environment
from control.lcp_solver import solve_lcp

# §7.27: 3 anchor depths (used to fit v_cap) + 2 HELD-OUT depths (used
# only to check the clamp's transition SHAPE under partial clamping)
ANCHOR_DEPTHS_M = [0.0001, 0.000549, 0.001]
HELDOUT_DEPTHS_M = [0.00025, 0.00075]
ALL_DEPTHS_M = sorted(ANCHOR_DEPTHS_M + HELDOUT_DEPTHS_M)

# Reverse-engineered grid (centered on §7.25 fit's predicted ≈0.034)
V_CAP_GRID = [0.025, 0.030, 0.034, 0.040, 0.050]

DT_SUB = 0.005          # 5 ms LCS substep / Drake comparison horizon
DRAKE_DT = 0.00025      # Drake internal timestep — prompt-pinned
DRAKE_STEPS = int(round(DT_SUB / DRAKE_DT))  # 20

# §7.24 band (same as Candidate C validation)
SHALLOW_TOL = 0.05       # ±5% at 0.10 mm
DEEP_TOL    = 0.25       # ±25% at 0.549 and 1.00 mm
HELDOUT_TOL = 0.25       # ±25% at held-out depths vs Drake

BOX_HALF  = 0.05
EE_RADIUS = 0.025
BOX_QUAT  = np.array([1.0, 0.0, 0.0, 0.0])
BOX_POS   = np.array([0.0, 0.0, 0.05])
EE_VEL_X  = -0.05
BOX_V_X_IDX = 13         # LCS state index for box translational v_x

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
    """Re-pose the system so the EE is `ee_pen_m` inside the box surface,
    with the box pinned at BOX_POS and the EE moving at EE_VEL_X. The
    posture is constrained so no arm link is closer than ARM_BOX_CLEARANCE_M
    to the box (the §7.14 per-depth gate)."""
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
        return context, plant_ctx, ee_pos
    return None, None, None


def run_lcs_one_substep(plant, plant_ctx, obj_body, plant_ad, ctx_ad, mu, ee_pos,
                       initial_q, initial_v):
    """Reset state and run one LCS substep. Returns dict with box_v_x,
    lam_n on EE-BOX, gate counts."""
    from control.lcs_formulator import LCSFormulator
    plant.SetPositions(plant_ctx, initial_q)
    plant.SetVelocities(plant_ctx, initial_v)
    x_curr = np.concatenate([
        BOX_QUAT, BOX_POS, ee_pos, np.zeros(3), np.zeros(3),
        np.array([EE_VEL_X, 0., 0.])])
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
    q_lcp = E @ x_curr + c_lcs
    try:
        lam, lcp_info = solve_lcp(F, q_lcp)
        lcp_ok = lcp_info if isinstance(lcp_info, bool) else True
    except Exception:
        lam = np.zeros(n_lam)
        lcp_ok = False
    x_next = A @ x_curr + D @ lam + d_const
    box_v_x = float(x_next[BOX_V_X_IDX])
    lam_n_ee = (float(lam[SLN + ee_box_idx])
                if ee_box_idx is not None else float('nan'))
    n_pusher = sum(1 for c in contacts if c.get('tag', '') == 'EE-BOX')
    n_floor  = sum(1 for c in contacts
                   if c.get('tag', '') == 'BOX-GND'
                   or c.get('tag', '').startswith('BOX-VERT-'))
    n_arm    = sum(1 for c in contacts
                   if c.get('tag', '') not in ('EE-BOX', 'BOX-GND')
                   and not c.get('tag', '').startswith('BOX-VERT-'))
    gate_ok = (n_arm == 0 and n_pusher >= 1 and n_floor >= 1)
    return dict(box_v_x=box_v_x, lam_n_ee=lam_n_ee,
                n_pusher=n_pusher, n_floor=n_floor, n_arm=n_arm,
                gate_ok=gate_ok, lcp_ok=lcp_ok,
                phi_clamp=f._normal_phi_clamp_v_cap)


def run_drake_one_substep(diagram, plant, plant_ctx, context, obj_body,
                          initial_q, initial_v):
    """Reset state and run Drake forward DT_SUB. Returns drake box_v_x."""
    plant.SetPositions(plant_ctx, initial_q)
    plant.SetVelocities(plant_ctx, initial_v)
    sim = Simulator(diagram, context)
    sim.Initialize()
    sim.AdvanceTo(DT_SUB)
    plant_ctx_after = plant.GetMyContextFromRoot(context)
    box_v = plant.EvalBodySpatialVelocityInWorld(
        plant_ctx_after, obj_body).translational()
    return float(box_v[0])


def run_at_depth(task_cfg, pen_m, v_cap_grid):
    """For one depth: build env, IK-pose, compute Drake reference,
    run LCS at each v_cap (and at baseline v_cap=None for sanity).
    Returns dict {drake_box_v_x, baseline_box_v_x, results_by_v_cap}."""
    pen_mm = pen_m * 1000
    d, p, sg, panda, obj, _, p_ad, ctx_ad = _build_env(task_cfg,
                                                       time_step=DRAKE_DT)
    obj_body = p.GetBodyByName("box_link", obj)
    ee_frame = p.GetFrameByName("pusher")
    ctx, pctx, ee_pos = setup_state_at_depth(d, p, sg, panda, obj, ee_frame, pen_m)
    if ctx is None:
        return {"pen_mm": pen_mm, "failed": True}

    # Save initial state (LCS calls must reset to this since they share ctx)
    initial_q = p.GetPositions(pctx).copy()
    initial_v = p.GetVelocities(pctx).copy()

    # LCS baseline (no clamp) — sanity-check that flag default-OFF is
    # byte-identical to pre-§7.27 behaviour
    if "LCS_NORMAL_PHI_CLAMP" in os.environ:
        del os.environ["LCS_NORMAL_PHI_CLAMP"]
    os.environ["LCS_NORMAL_VELOCITY_LEVEL"] = "0"
    os.environ["LCS_NORMAL_COMPLIANCE_K"]   = "0.0"
    os.environ["LCS_EXPLICIT_BOX_GND"]      = "4"
    baseline = run_lcs_one_substep(p, pctx, obj_body, p_ad, ctx_ad,
                                    task_cfg["friction"], ee_pos,
                                    initial_q, initial_v)

    # LCS clamp sweep
    by_v_cap = {}
    for v_cap in v_cap_grid:
        os.environ["LCS_NORMAL_PHI_CLAMP"] = f"{v_cap:.6f}"
        r = run_lcs_one_substep(p, pctx, obj_body, p_ad, ctx_ad,
                                 task_cfg["friction"], ee_pos,
                                 initial_q, initial_v)
        by_v_cap[v_cap] = r
    if "LCS_NORMAL_PHI_CLAMP" in os.environ:
        del os.environ["LCS_NORMAL_PHI_CLAMP"]

    # Drake — done LAST since it mutates ctx
    drake_box_v_x = run_drake_one_substep(d, p, pctx, ctx, obj_body,
                                           initial_q, initial_v)

    return dict(pen_mm=pen_mm, failed=False, drake_box_v_x=drake_box_v_x,
                baseline=baseline, by_v_cap=by_v_cap)


def _in_band(lcs_v, drake_v, tol):
    target = abs(drake_v)
    return target * (1 - tol) <= abs(lcs_v) <= target * (1 + tol)


def main() -> int:
    print("=" * 84)
    print("STAGE C  CANDIDATE E VALIDATION — clamped-φ/dt saturating-stiffness")
    print("(EE-BOX only; v_cap-sweep + held-out shape check)")
    print("=" * 84)
    with open("config/tasks.yaml") as f:
        cfg = yaml.safe_load(f)
    task_cfg = cfg["tasks"]["pushing"]
    print(f"  μ = {task_cfg['friction']}  m_box = {task_cfg['mass']} kg")
    print(f"  Δt_sub = {DT_SUB*1000:.1f} ms   Drake dt = {DRAKE_DT*1e6:.1f} μs "
          f"({DRAKE_STEPS} steps)")
    print(f"  Anchor depths : {[f'{d*1000:.2f}mm' for d in ANCHOR_DEPTHS_M]}")
    print(f"  Held-out      : {[f'{d*1000:.2f}mm' for d in HELDOUT_DEPTHS_M]}")
    print(f"  v_cap grid    : {V_CAP_GRID}")
    print(f"  band — shallow ±{SHALLOW_TOL*100:.0f}%, deep ±{DEEP_TOL*100:.0f}%, "
          f"held-out ±{HELDOUT_TOL*100:.0f}%")
    print()

    # ---------------------------------------------------------------- per-depth
    per_depth = {}
    for pen_m in ALL_DEPTHS_M:
        pen_mm = pen_m * 1000
        is_anchor = pen_m in ANCHOR_DEPTHS_M
        tag = "ANCHOR" if is_anchor else "HELDOUT"
        print(f"[{tag}] depth = {pen_mm:.2f} mm")
        r = run_at_depth(task_cfg, pen_m, V_CAP_GRID)
        if r.get("failed"):
            print("  IK FAILED — could not pose the system at this depth")
            per_depth[pen_m] = None
            continue
        drake_v = r["drake_box_v_x"]
        bl = r["baseline"]
        gate_ok = bl["gate_ok"]
        gate_str = ("OK" if gate_ok else "FAIL") \
                   + f" (p={bl['n_pusher']},f={bl['n_floor']},a={bl['n_arm']})"
        print(f"  Drake box_v_x    = {drake_v:+.6f}")
        print(f"  baseline (rigid) = {bl['box_v_x']:+.6f}  λ_n={bl['lam_n_ee']:+.4f}  "
              f"gate={gate_str}")
        for v_cap in V_CAP_GRID:
            cell = r["by_v_cap"][v_cap]
            cstr = ("OK" if cell['gate_ok'] else "FAIL") \
                   + f" (p={cell['n_pusher']},f={cell['n_floor']},a={cell['n_arm']})"
            lcp_str = "OK" if cell['lcp_ok'] else "BAD"
            print(f"    v_cap={v_cap:.3f}  box_v_x = {cell['box_v_x']:+.6f}  "
                  f"λ_n={cell['lam_n_ee']:+.4f}  gate={cstr}  lcp={lcp_str}")
        per_depth[pen_m] = r
        print()

    # ---------------------------------------------------------------- band scan
    print("=" * 84)
    print("BAND CHECK — find best v_cap satisfying §7.24 band at all 3 anchors")
    print("=" * 84)
    anchor_drake = {d: per_depth[d]["drake_box_v_x"]
                    for d in ANCHOR_DEPTHS_M
                    if per_depth.get(d) is not None}

    if len(anchor_drake) < 3:
        print("  Cannot evaluate — one or more anchor depths failed setup")
        return 1

    # Per-v_cap pass/fail at each anchor
    band_table = {}
    for v_cap in V_CAP_GRID:
        cells = []
        passes = 0
        all_lcp_ok = True
        all_gate_ok = True
        for d in ANCHOR_DEPTHS_M:
            cell = per_depth[d]["by_v_cap"][v_cap]
            drake_v = per_depth[d]["drake_box_v_x"]
            tol = SHALLOW_TOL if d == 0.0001 else DEEP_TOL
            in_b = _in_band(cell["box_v_x"], drake_v, tol)
            cells.append((d, cell, drake_v, in_b))
            if in_b:
                passes += 1
            all_lcp_ok = all_lcp_ok and cell["lcp_ok"]
            all_gate_ok = all_gate_ok and cell["gate_ok"]
        band_table[v_cap] = dict(cells=cells, passes=passes,
                                  lcp_ok=all_lcp_ok, gate_ok=all_gate_ok)

    print()
    print(f"  {'v_cap':>6}  "
          f"{'0.10mm':>14}  {'in?':>3}  "
          f"{'0.549mm':>14}  {'in?':>3}  "
          f"{'1.00mm':>14}  {'in?':>3}  "
          f"{'gates':>5}  {'lcp':>4}")
    for v_cap in V_CAP_GRID:
        bt = band_table[v_cap]
        row = f"  {v_cap:>6.3f}  "
        for (d, cell, drake_v, in_b) in bt["cells"]:
            row += f"{cell['box_v_x']:>+14.6f}  {('Y' if in_b else 'N'):>3}  "
        row += f"{('OK' if bt['gate_ok'] else 'DIRTY'):>5}  "
        row += f"{('OK' if bt['lcp_ok'] else 'BAD'):>4}"
        print(row)
    print()
    print(f"  Drake (rigid Drake reference at anchors): "
          f"{anchor_drake[0.0001]:+.6f} / "
          f"{anchor_drake[0.000549]:+.6f} / "
          f"{anchor_drake[0.001]:+.6f}")
    print()

    # Pick the best v_cap — maximum anchors passing; if tie, the one whose
    # LCS box_v_x is closest (in L1 over anchors) to Drake
    best_v_cap = None
    best_passes = -1
    best_l1 = float('inf')
    for v_cap in V_CAP_GRID:
        bt = band_table[v_cap]
        if not bt["gate_ok"] or not bt["lcp_ok"]:
            continue
        l1 = sum(abs(cell["box_v_x"] - drake_v)
                 for (d, cell, drake_v, _) in bt["cells"])
        if bt["passes"] > best_passes or (bt["passes"] == best_passes and l1 < best_l1):
            best_v_cap = v_cap
            best_passes = bt["passes"]
            best_l1 = l1

    if best_v_cap is None:
        print("  NO v_cap is gate-clean AND LCP-feasible at all 3 anchors.")
        anchors_pass = False
    else:
        print(f"  Best v_cap = {best_v_cap:.3f}  (anchors passed = {best_passes}/3, "
              f"L1 = {best_l1:.5f})")
        anchors_pass = (best_passes == 3)

    print()

    # ---------------------------------------------------------------- held-out
    print("=" * 84)
    print(f"HELD-OUT SHAPE CHECK at depths {[f'{d*1000:.2f}mm' for d in HELDOUT_DEPTHS_M]}")
    print(f"(best v_cap = {best_v_cap}, ±{HELDOUT_TOL*100:.0f}% vs Drake)")
    print("=" * 84)
    heldout_pass = True
    heldout_diags = []
    if best_v_cap is None:
        print("  Skipped (no best v_cap).")
        heldout_pass = False
    else:
        for d in HELDOUT_DEPTHS_M:
            r = per_depth.get(d)
            if r is None:
                print(f"  {d*1000:.2f}mm: setup failed")
                heldout_pass = False
                continue
            cell = r["by_v_cap"][best_v_cap]
            drake_v = r["drake_box_v_x"]
            in_b = _in_band(cell["box_v_x"], drake_v, HELDOUT_TOL)
            heldout_pass = heldout_pass and in_b and cell["gate_ok"] and cell["lcp_ok"]
            heldout_diags.append((d, cell, drake_v, in_b))
            print(f"  {d*1000:.2f}mm  Drake={drake_v:+.6f}  "
                  f"LCS={cell['box_v_x']:+.6f}  "
                  f"diff={cell['box_v_x'] - drake_v:+.6f}  "
                  f"{('IN-BAND' if in_b else 'OUT')}  "
                  f"gate={'OK' if cell['gate_ok'] else 'DIRTY'}  "
                  f"lcp={'OK' if cell['lcp_ok'] else 'BAD'}")
    print()

    # ---------------------------------------------------------------- gate scan
    gate_clean_all = all(
        per_depth[d] is not None
        and per_depth[d]["baseline"]["gate_ok"]
        for d in ALL_DEPTHS_M
    )
    if best_v_cap is not None:
        clamp_gate_clean = all(
            per_depth[d]["by_v_cap"][best_v_cap]["gate_ok"]
            for d in ALL_DEPTHS_M
            if per_depth.get(d) is not None
        )
    else:
        clamp_gate_clean = True

    # ---------------------------------------------------------------- route
    print("=" * 84)
    print("ROUTE")
    print("=" * 84)
    if not gate_clean_all or not clamp_gate_clean:
        route = "E-BREAKS-VERTICAL / GATE-DIRTY"
        msg = ("  The per-depth §7.14 gate is dirty under the clamp at one or more\n"
               "  depths. Clamping the EE-BOX φ/dt perturbed the BOX-VERT coupling\n"
               "  enough to shift the contact set. Diagnose the coupling before\n"
               "  proceeding.")
    elif anchors_pass and heldout_pass:
        route = "E-PASSES"
        msg = (f"  best v_cap = {best_v_cap:.3f} m/s passes the §7.24 band at\n"
               "  all 3 anchors AND agrees with Drake within the band at both\n"
               "  held-out depths. The clamp recovered the saturation SHAPE, not\n"
               "  just the anchor magnitudes. The saturating-stiffness fix WORKS\n"
               "  as a one-parameter clamp.\n"
               "  NEXT (separate block): the LIVE flip — enable\n"
               f"  LCS_NORMAL_PHI_CLAMP={best_v_cap} in main.py / the full push sim\n"
               "  and answer the original no-push question.")
    elif anchors_pass and not heldout_pass:
        route = "E-PASSES-ANCHORS-FAILS-HELDOUT"
        msg = ("  best v_cap satisfies the band at the 3 anchors but DIVERGES from\n"
               "  Drake at one or both held-out depths. The one-param clamp matched\n"
               "  the anchor magnitudes but its transition SHAPE is wrong — the\n"
               "  clamp is an interpolation, not the physics.\n"
               "  NEXT (separate block): escalate to Candidate F (additive\n"
               "  saturating-penetration term: C-base + bounded onset+saturation\n"
               "  shape, fit to anchors + held-out).")
    elif not anchors_pass:
        route = "E-FAILS-ANCHORS"
        msg = ("  No v_cap in the grid passes the §7.24 band at all 3 anchors.\n"
               "  The §7.25 linear fit did NOT hold under clamping — the LCP active\n"
               "  set shifted at the deep geometry.\n"
               "  NEXT (separate block): characterize how clamping actually behaves\n"
               "  at depth, then escalate to Candidate F informed by the measurement.")
    else:
        route = "UNKNOWN"
        msg = "  Could not classify — inspect results table."

    print(f"  ► ROUTE: {route}")
    print(msg)
    print()
    print("=" * 84)
    print("HOLD: next block (live flip / Candidate F / characterize / diagnose")
    print("      coupling) is SEPARATE.")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
