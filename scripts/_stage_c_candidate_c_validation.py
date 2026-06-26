"""Stage C Candidate C validation — velocity-level normal formulation
(Anitescu-Potra, v_target=0) on EE-BOX-only, behind LCS_NORMAL_VELOCITY_LEVEL
env var. §7.25-routed.

Re-runs the §7.23 LCS-vs-Drake box-velocity comparison at depths
{0.10, 0.549, 1.00} mm with the velocity-level formulation enabled.
This is NOT a sweep — velocity-level has no tunable k (the §7.25 (5) (ii)
warning rules out β-scaling). Drake `dt = 0.25 ms` FIXED per §7.20.

Pre-registered routes:
  C-IN-BAND        — LCS box_v_x lands in the band at shallow AND deep
                     (the LCP off-diagonal coupling lifted it above the
                     naïve −0.036 m/s extrapolation)
  C-UNDER-PREDICTS — LCS box_v_x FLATTENS but UNDER-predicts (~−0.036),
                     fails both bands for being too small → Drake's
                     impulse needs a saturating-stiffness term ON TOP of
                     velocity-damping (a SEPARATE next block)
  C-DOESNT-FLATTEN — LCS box_v_x still scales with depth → dropping
                     phi/dt didn't kill depth-scaling, has a second
                     source (diagnose)
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

DT_SUB    = 0.005

# Pre-cached Drake-side box_v_x at t=5ms from §7.23 (Drake doesn't depend
# on the C-flag, so we don't re-run it).
DRAKE_BOX_V_X = {0.10: -0.053988, 0.549: -0.064133, 1.00: -0.064927}

SHALLOW_TOL = 0.05   # ±5% at 0.10 mm
DEEP_TOL    = 0.25   # ±25% at 0.549 and 1.00 mm

BOX_HALF  = 0.05
EE_RADIUS = 0.025
BOX_QUAT  = np.array([1.0, 0.0, 0.0, 0.0])
BOX_POS   = np.array([0.0, 0.0, 0.05])
EE_VEL_X  = -0.05
BOX_V_X_IDX = 13

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


def lcs_one_substep(plant, plant_ctx, obj_body, plant_ad, ctx_ad, mu, ee_pos):
    from control.lcs_formulator import LCSFormulator
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
    lam, lcp_info = solve_lcp(F, q_lcp)
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
    lcp_ok = lcp_info if isinstance(lcp_info, bool) else True
    return dict(box_v_x=box_v_x, lam_n_ee=lam_n_ee,
                n_pusher=n_pusher, n_floor=n_floor, n_arm=n_arm,
                gate_ok=gate_ok, lcp_ok=lcp_ok,
                vl_flag=f._normal_velocity_level)


def run_at_config(task_cfg, velocity_level, depths_m, depths_tag, results_out):
    os.environ["LCS_NORMAL_VELOCITY_LEVEL"] = "1" if velocity_level else "0"
    os.environ["LCS_NORMAL_COMPLIANCE_K"]   = "0.0"
    os.environ["LCS_EXPLICIT_BOX_GND"]      = "4"
    import control.lcs_formulator
    importlib.reload(control.lcs_formulator)
    importlib.reload(control.lcp_solver)
    tag_label = "C (velocity-level, drop φ/dt)" if velocity_level else "baseline (rigid)"
    print("=" * 84)
    print(f"  configuration: {tag_label}")
    print("=" * 84)
    for pen_m, tag in zip(depths_m, depths_tag):
        pen_mm = pen_m * 1000
        d, p, sg, panda, obj, _, p_ad, ctx_ad = _build_env(task_cfg,
                                                            time_step=0.001)
        obj_body = p.GetBodyByName("box_link", obj)
        ee_frame = p.GetFrameByName("pusher")
        ctx, pctx, q_arm, ee_pos = setup_state_at_depth(
            d, p, sg, panda, obj, ee_frame, pen_m)
        if ctx is None:
            print(f"  {tag}: re-pose FAIL")
            results_out[(velocity_level, pen_mm)] = None
            continue
        r = lcs_one_substep(p, pctx, obj_body, p_ad, ctx_ad,
                             task_cfg["friction"], ee_pos)
        results_out[(velocity_level, pen_mm)] = r
        gate_str = "OK" if r['gate_ok'] else "FAIL"
        lcp_str  = "OK" if r['lcp_ok'] else "BAD"
        print(f"  {tag}: box_v_x = {r['box_v_x']:+.6f}  λ_n = {r['lam_n_ee']:+.4f}"
              f"  gate(p={r['n_pusher']},f={r['n_floor']},a={r['n_arm']})={gate_str}"
              f"  lcp={lcp_str}  VL={r['vl_flag']}")


def main() -> int:
    print("=" * 84)
    print("STAGE C  CANDIDATE C VALIDATION — velocity-level normal on EE-BOX-only")
    print("(drop phi/dt for EE-BOX contact; v_target=0; no k-sweep — single config)")
    print("=" * 84)
    with open("config/tasks.yaml") as f:
        cfg = yaml.safe_load(f)
    task_cfg = cfg["tasks"]["pushing"]
    print(f"  μ = {task_cfg['friction']}  m_box = {task_cfg['mass']} kg")
    print(f"  Δt_sub = {DT_SUB*1000:.1f} ms")
    print(f"  Drake reference box_v_x: "
          f"{DRAKE_BOX_V_X[0.10]:+.6f} / {DRAKE_BOX_V_X[0.549]:+.6f} / "
          f"{DRAKE_BOX_V_X[1.00]:+.6f} at 0.10/0.549/1.00 mm")
    print()

    results = {}
    # Baseline (rigid) — sanity check that the flag default-OFF is byte-identical
    run_at_config(task_cfg, False, DEPTH_M, DEPTH_TAG, results)
    print()
    # Candidate C (velocity-level)
    run_at_config(task_cfg, True, DEPTH_M, DEPTH_TAG, results)
    print()

    # Band check
    print("=" * 84)
    print("BAND CHECK")
    print("=" * 84)
    print()
    print(f"  Shallow (0.10mm): ±{SHALLOW_TOL*100:.0f}% of {abs(DRAKE_BOX_V_X[0.10]):.6f} "
          f"→ |LCS| ∈ [{abs(DRAKE_BOX_V_X[0.10])*(1-SHALLOW_TOL):.6f}, "
          f"{abs(DRAKE_BOX_V_X[0.10])*(1+SHALLOW_TOL):.6f}]")
    print(f"  Deep    (≥0.549): ±{DEEP_TOL*100:.0f}% of {abs(DRAKE_BOX_V_X[1.00]):.6f} "
          f"→ |LCS| ∈ [{abs(DRAKE_BOX_V_X[1.00])*(1-DEEP_TOL):.6f}, "
          f"{abs(DRAKE_BOX_V_X[1.00])*(1+DEEP_TOL):.6f}]")
    print()
    print(f"  {'config':>14}  {'0.10mm':>11}  {'shal?':>5}  "
          f"{'0.549mm':>11}  {'deep?':>5}  "
          f"{'1.00mm':>11}  {'deep?':>5}  {'gate':>6}")
    in_shallow_C = False; in_549_C = False; in_1mm_C = False
    all_gates_C = True; all_lcp_C = True
    for vl_flag, label in [(False, "rigid (k=0)"), (True, "velocity-level")]:
        r010 = results.get((vl_flag, 0.10))
        r549 = results.get((vl_flag, 0.549))
        r1mm = results.get((vl_flag, 1.00))
        tgt_sh = abs(DRAKE_BOX_V_X[0.10])
        tgt_dp = abs(DRAKE_BOX_V_X[1.00])
        in_sh  = (r010 is not None and tgt_sh*(1-SHALLOW_TOL) <= abs(r010['box_v_x']) <= tgt_sh*(1+SHALLOW_TOL))
        in_549 = (r549 is not None and tgt_dp*(1-DEEP_TOL) <= abs(r549['box_v_x']) <= tgt_dp*(1+DEEP_TOL))
        in_1mm = (r1mm is not None and tgt_dp*(1-DEEP_TOL) <= abs(r1mm['box_v_x']) <= tgt_dp*(1+DEEP_TOL))
        gates_ok = all(r is not None and r['gate_ok']
                       for r in (r010, r549, r1mm))
        lcp_ok = all(r is not None and r['lcp_ok']
                     for r in (r010, r549, r1mm))
        if vl_flag:
            in_shallow_C = in_sh; in_549_C = in_549; in_1mm_C = in_1mm
            all_gates_C = gates_ok; all_lcp_C = lcp_ok
        print(f"  {label:>14}  "
              f"{r010['box_v_x']:>+11.6f}  {('Y' if in_sh else 'N'):>5}  "
              f"{r549['box_v_x']:>+11.6f}  {('Y' if in_549 else 'N'):>5}  "
              f"{r1mm['box_v_x']:>+11.6f}  {('Y' if in_1mm else 'N'):>5}  "
              f"{('OK' if gates_ok else 'FAIL'):>6}")
    print()

    # Route
    print("=" * 84)
    print("ROUTE")
    print("=" * 84)
    if not all_lcp_C:
        route = "C-LCP-INFEASIBLE"
        msg = ("  LCP solve failed at one or more depths under velocity-level —\n"
               "  dropping phi/dt may have made the LCP degenerate; diagnose.")
    elif not all_gates_C:
        route = "C-BREAKS-GATE"
        msg = ("  Per-depth §7.14 gate broke at one or more depths under VL —\n"
               "  unexpected coupling; diagnose.")
    elif in_shallow_C and in_549_C and in_1mm_C:
        route = "C-IN-BAND"
        msg = ("  Velocity-level lands in the band at ALL three depths. The LCP\n"
               "  off-diagonal coupling lifted box_v_x above the naïve −0.036\n"
               "  extrapolation. C works.\n"
               "  NEXT (separate block): the LIVE flip — enable\n"
               "  LCS_NORMAL_VELOCITY_LEVEL=1 in main.py / the full push sim and\n"
               "  answer the original no-push question.")
    else:
        # Determine sub-route: under-predicts (flattens too far) or doesn't flatten
        r549 = results[(True, 0.549)]; r1mm = results[(True, 1.00)]
        v549 = abs(r549['box_v_x']); v1mm = abs(r1mm['box_v_x'])
        # Does it flatten? Compare 549 vs 1mm
        if abs(v549 - v1mm) / max(v549, v1mm, 1e-9) < 0.25:
            # Flat — check magnitude
            tgt_dp = abs(DRAKE_BOX_V_X[1.00])
            if v1mm < tgt_dp * (1 - DEEP_TOL):
                route = "C-UNDER-PREDICTS"
                msg = ("  Velocity-level FLATTENS but UNDER-predicts (v_deep ≈\n"
                       f"  {v1mm:.4f} < Drake's {tgt_dp:.4f} band-lower-bound\n"
                       f"  {tgt_dp*(1-DEEP_TOL):.4f}). Drake's impulse needs a\n"
                       "  SATURATING-PENETRATION component (finite stiffness) ON\n"
                       "  TOP of velocity-damping. The fix is a finite-stiffness\n"
                       "  compliant contact — NEITHER constant-k soft-LCP NOR pure\n"
                       "  velocity-level.\n"
                       "  NEXT (separate block): scope the saturating-stiffness /\n"
                       "  compliant-contact build.")
            else:
                route = "C-FLAT-BUT-OUT"
                msg = ("  Velocity-level flattens but lands outside the band\n"
                       "  (likely over-predicts at deep). Diagnose.")
        else:
            route = "C-DOESNT-FLATTEN"
            msg = ("  Velocity-level still scales with depth (doesn't flatten).\n"
                   "  Dropping phi/dt didn't kill depth-scaling, so it has a\n"
                   "  SECOND source. Re-examine c_lcs decomposition / LCP coupling.")

    print(f"  ► ROUTE: {route}")
    print(msg)
    print()
    print("=" * 84)
    print("HOLD: next block (live flip / saturating-stiffness build / diagnose")
    print("      / cleaner removal) is SEPARATE.")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
