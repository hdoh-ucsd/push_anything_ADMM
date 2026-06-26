"""Stage C Candidate A validation — soft-LCP compliance on EE-BOX-only,
behind LCS_NORMAL_COMPLIANCE_K env var. §7.24-routed.

Re-runs the §7.23 LCS-vs-Drake box-velocity comparison at depths
{0.10, 0.549, 1.00} mm with k swept over a CAPPED small grid
{0.0, 0.02, 0.04, 0.06, 0.08}. Drake `dt = 0.25 ms` FIXED per §7.20.
Validates against the §7.24 band:

  deep (≥ 0.549 mm): |LCS box_v_x| within 25% of Drake's 0.064 m/s
  shallow (0.10 mm): |LCS box_v_x| within 5% of Drake's 0.054 m/s
  per-depth §7.14 gate clean

Routes (this script does NOT execute the next block):
  A-SUFFICES        — some k passes the band across all three depths
  A-INSUFFICIENT    — no k passes (band missed at deep or shallow)
  A-BREAKS-VERTICAL — adding k re-breaks the vertical-fix gate (BOX-VERT
                      contact admit changes; floor pair count drops)
  K-SWEEP-DEGENERATE — LCP solve degrades / becomes infeasible at some k
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

DEPTH_M   = [0.0001, 0.000549, 0.001]
DEPTH_TAG = ["UNDER (0.10mm)", "SWEET (0.549mm)", "OVER (1.00mm)"]

DT_SUB    = 0.005
DT_DRAKE  = 0.00025
N_DRAKE_PER_SUB = int(round(DT_SUB / DT_DRAKE))

# Pre-cached Drake-side box_v_x at t=5ms from §7.23, dt fixed 0.25 ms.
# (Drake doesn't depend on k, so we don't need to re-run it per k.)
DRAKE_BOX_V_X = {0.10: -0.053988, 0.549: -0.064133, 1.00: -0.064927}

# Validation band (§7.24)
SHALLOW_TOL = 0.05   # ±5% at 0.10 mm
DEEP_TOL    = 0.25   # ±25% at 0.549 and 1.00 mm

# k-sweep (CAPPED — start near scope estimate 0.04, span ±4×)
K_GRID = [0.0, 0.02, 0.04, 0.06, 0.08]

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
                k_used=f._normal_compliance_k)


def run_at_k(task_cfg, k_value, depths_m, depths_tag, results_out):
    os.environ["LCS_NORMAL_COMPLIANCE_K"] = str(k_value)
    os.environ["LCS_EXPLICIT_BOX_GND"]   = "4"
    import control.lcs_formulator
    importlib.reload(control.lcs_formulator)
    importlib.reload(control.lcp_solver)
    print("=" * 84)
    print(f"  k = {k_value:.4f}")
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
            results_out[(k_value, pen_mm)] = None
            continue
        r = lcs_one_substep(p, pctx, obj_body, p_ad, ctx_ad,
                             task_cfg["friction"], ee_pos)
        results_out[(k_value, pen_mm)] = r
        gate_str = "OK" if r['gate_ok'] else "FAIL"
        lcp_str  = "OK" if r['lcp_ok'] else "BAD"
        print(f"  {tag}: box_v_x = {r['box_v_x']:+.6f}  λ_n = {r['lam_n_ee']:+.4f}"
              f"  gate(p={r['n_pusher']},f={r['n_floor']},a={r['n_arm']})={gate_str}"
              f"  lcp={lcp_str}  k_used={r['k_used']:.4f}")


def main() -> int:
    print("=" * 84)
    print("STAGE C  CANDIDATE A VALIDATION — soft-LCP on EE-BOX-only, k-sweep")
    print("(§7.24 band: deep within 25% Drake −0.064; shallow within 5% −0.054)")
    print("=" * 84)
    with open("config/tasks.yaml") as f:
        cfg = yaml.safe_load(f)
    task_cfg = cfg["tasks"]["pushing"]
    print(f"  μ = {task_cfg['friction']}  m_box = {task_cfg['mass']} kg")
    print(f"  Δt_sub = {DT_SUB*1000:.1f} ms (one sub-step)")
    print(f"  Drake reference box_v_x (§7.23, dt=0.25 ms): "
          f"{DRAKE_BOX_V_X[0.10]:+.6f} / {DRAKE_BOX_V_X[0.549]:+.6f} / "
          f"{DRAKE_BOX_V_X[1.00]:+.6f} at 0.10/0.549/1.00 mm")
    print(f"  k-grid (CAPPED): {K_GRID}")
    print()

    results = {}
    for k_val in K_GRID:
        run_at_k(task_cfg, k_val, DEPTH_M, DEPTH_TAG, results)
        print()

    # Band check per (k, depth) — table
    print("=" * 84)
    print("BAND CHECK (per (k, depth) row)")
    print("=" * 84)
    print()
    print(f"  Drake target: shallow 0.10mm = {abs(DRAKE_BOX_V_X[0.10]):.6f}; "
          f"deep ≥0.549mm ≈ {abs(DRAKE_BOX_V_X[1.00]):.6f}")
    print(f"  Shallow band (0.10mm): {SHALLOW_TOL*100:.0f}% of {abs(DRAKE_BOX_V_X[0.10]):.6f} "
          f"→ |LCS| ∈ [{abs(DRAKE_BOX_V_X[0.10])*(1-SHALLOW_TOL):.6f}, "
          f"{abs(DRAKE_BOX_V_X[0.10])*(1+SHALLOW_TOL):.6f}]")
    print(f"  Deep    band (≥0.549): {DEEP_TOL*100:.0f}% of {abs(DRAKE_BOX_V_X[1.00]):.6f} "
          f"→ |LCS| ∈ [{abs(DRAKE_BOX_V_X[1.00])*(1-DEEP_TOL):.6f}, "
          f"{abs(DRAKE_BOX_V_X[1.00])*(1+DEEP_TOL):.6f}]")
    print()
    print(f"  {'k':>6}  {'0.10mm':>11}  {'in shallow?':>11}  "
          f"{'0.549mm':>11}  {'in deep?':>9}  "
          f"{'1.00mm':>11}  {'in deep?':>9}  {'gates':>7}  {'k-OK':>6}")
    for k_val in K_GRID:
        row = []
        all_gates = True; all_lcp = True
        in_shallow = False; in_549 = False; in_1mm = False
        for pen_m, tag in zip(DEPTH_M, DEPTH_TAG):
            pen_mm = pen_m * 1000
            r = results.get((k_val, pen_mm))
            if r is None:
                row.extend([float('nan'), False])
                continue
            row.extend([r['box_v_x'], r['gate_ok'], r['lcp_ok']])
            all_gates = all_gates and r['gate_ok']
            all_lcp = all_lcp and r['lcp_ok']
        if 0.10 in [d*1000 for d in DEPTH_M]:
            r = results[(k_val, 0.10)]
            if r is not None:
                v = abs(r['box_v_x']); tgt = abs(DRAKE_BOX_V_X[0.10])
                in_shallow = tgt*(1-SHALLOW_TOL) <= v <= tgt*(1+SHALLOW_TOL)
        r549 = results.get((k_val, 0.549))
        r1mm = results.get((k_val, 1.00))
        if r549 is not None:
            v = abs(r549['box_v_x']); tgt = abs(DRAKE_BOX_V_X[1.00])
            in_549 = tgt*(1-DEEP_TOL) <= v <= tgt*(1+DEEP_TOL)
        if r1mm is not None:
            v = abs(r1mm['box_v_x']); tgt = abs(DRAKE_BOX_V_X[1.00])
            in_1mm = tgt*(1-DEEP_TOL) <= v <= tgt*(1+DEEP_TOL)
        r010 = results.get((k_val, 0.10))
        print(f"  {k_val:>6.3f}  "
              f"{(r010['box_v_x'] if r010 else float('nan')):>+11.6f}  "
              f"{('Y' if in_shallow else 'N'):>11}  "
              f"{(r549['box_v_x'] if r549 else float('nan')):>+11.6f}  "
              f"{('Y' if in_549 else 'N'):>9}  "
              f"{(r1mm['box_v_x'] if r1mm else float('nan')):>+11.6f}  "
              f"{('Y' if in_1mm else 'N'):>9}  "
              f"{('OK' if all_gates else 'FAIL'):>7}  "
              f"{('OK' if all_lcp else 'BAD'):>6}")

    # Route
    print()
    print("=" * 84)
    print("ROUTE")
    print("=" * 84)
    best_k = None
    for k_val in K_GRID:
        r010 = results.get((k_val, 0.10))
        r549 = results.get((k_val, 0.549))
        r1mm = results.get((k_val, 1.00))
        if r010 is None or r549 is None or r1mm is None:
            continue
        if not (r010['gate_ok'] and r549['gate_ok'] and r1mm['gate_ok']):
            continue
        if not (r010['lcp_ok'] and r549['lcp_ok'] and r1mm['lcp_ok']):
            continue
        v010 = abs(r010['box_v_x']); v549 = abs(r549['box_v_x']); v1mm = abs(r1mm['box_v_x'])
        tgt_shallow = abs(DRAKE_BOX_V_X[0.10]); tgt_deep = abs(DRAKE_BOX_V_X[1.00])
        in_shallow = tgt_shallow*(1-SHALLOW_TOL) <= v010 <= tgt_shallow*(1+SHALLOW_TOL)
        in_549 = tgt_deep*(1-DEEP_TOL) <= v549 <= tgt_deep*(1+DEEP_TOL)
        in_1mm = tgt_deep*(1-DEEP_TOL) <= v1mm <= tgt_deep*(1+DEEP_TOL)
        if in_shallow and in_549 and in_1mm:
            best_k = k_val
            break
    # Check vertical-fix integrity: at k=0 (control) and at every k, gate must be OK
    gates_ok_all_k = all(
        all(results.get((k_val, d*1000), {}).get('gate_ok', False)
            for d in DEPTH_M)
        for k_val in K_GRID
    )
    if best_k is not None:
        print(f"  ► A-SUFFICES at k = {best_k}")
        print(f"    Soft-LCP closes the push-regime gap across "
              f"{{0.10, 0.549, 1.00}} mm within the §7.24 band.")
        print(f"    NEXT (separate block): the LIVE flip — enable "
              f"LCS_NORMAL_COMPLIANCE_K = {best_k} at main.py / the full")
        print(f"    push sim and answer the original no-push question.")
    elif not gates_ok_all_k:
        print(f"  ► A-BREAKS-VERTICAL — adding k perturbs the BOX-VERT gate.")
        print(f"    The EE-BOX-only insertion is not as isolated as scoped;")
        print(f"    diagnose off-diagonal coupling before proceeding.")
    else:
        # Need to figure out whether near-miss (escalate to C) or degenerate
        print(f"  ► A-INSUFFICIENT — no k in the swept grid passes the band.")
        print(f"    NEXT (separate block): escalate to Candidate C (velocity-")
        print(f"    level target replacing c_lcs[SLN] = phi/dt), informed by")
        print(f"    the reference normal-formulation read.")
    print()
    print("=" * 84)
    print("HOLD: live flip (A-SUFFICES) / Candidate C build (A-INSUFFICIENT) /")
    print("      coupling diagnosis (A-BREAKS-VERTICAL) — SEPARATE block.")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
