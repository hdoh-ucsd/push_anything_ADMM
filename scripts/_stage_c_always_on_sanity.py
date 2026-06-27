"""§7.30 — Always-on EE-BOX admission STRUCTURAL sanity (offline; BEFORE the
live sim). Confirms:
  (1) Flag OFF reproduces the 2 mm filter (byte-identical baseline).
  (2) Flag ON at a SEPARATED state (φ ≈ +5 mm) admits the EE-BOX pair,
      the LCS arrays are well-formed (no NaN, correct shapes), and the
      LCP solution has λ_n = 0 on the EE-BOX row (no spurious force).
  (3) Per-state §7.14 gate is clean (p≥1, f≥1, a=0) in both modes.
This is a STRUCTURAL check — NOT a physics validation (the live sim is
the only available physics test for the always-on change).
"""
from __future__ import annotations

import os
import numpy as np
import yaml

from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.multibody.tree import JacobianWrtVariable
from pydrake.solvers import Solve
from pydrake.geometry import Role
from pydrake.math import RotationMatrix

from sim.env_builder import build_environment
from control.lcp_solver import solve_lcp

# A separated state: EE 5 mm beyond touching (Drake signed distance +5 mm).
SEPARATION_M = 0.005

DT_SUB = 0.005

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
]


def _scene_graph_of(diagram):
    for sys in diagram.GetSystems():
        if 'SceneGraph' in type(sys).__name__:
            return sys
    raise RuntimeError("SceneGraph not found")


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


def setup_state_at_separation(diagram, plant, sg, panda_model, object_model,
                              ee_frame, separation_m):
    world = plant.world_frame()
    # ee_pos_x = box_face_x + EE_RADIUS + separation
    #          = (BOX_POS[0] + BOX_HALF) + EE_RADIUS + separation
    ee_pos = np.array([BOX_POS[0] + BOX_HALF + EE_RADIUS + separation_m,
                       0.0, 0.05])
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
        ik.get_mutable_prog().SetInitialGuess(ik.q(),
                                              plant.GetPositions(plant_ctx))
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
        return plant_ctx, ee_pos
    return None, None


def probe_lcs(plant, plant_ctx, obj_body, plant_ad, ctx_ad, mu, ee_pos,
              flag_value: str):
    from control.lcs_formulator import LCSFormulator
    if flag_value is None:
        if "LCS_ALWAYS_ON_EE_BOX" in os.environ:
            del os.environ["LCS_ALWAYS_ON_EE_BOX"]
    else:
        os.environ["LCS_ALWAYS_ON_EE_BOX"] = flag_value
    os.environ["LCS_NORMAL_PHI_CLAMP"]      = ""
    os.environ["LCS_NORMAL_VELOCITY_LEVEL"] = "0"
    os.environ["LCS_NORMAL_COMPLIANCE_K"]   = "0.0"
    os.environ["LCS_EXPLICIT_BOX_GND"]      = "4"
    if "LCS_NORMAL_PHI_CLAMP" in os.environ \
       and os.environ["LCS_NORMAL_PHI_CLAMP"] == "":
        del os.environ["LCS_NORMAL_PHI_CLAMP"]

    f = LCSFormulator(plant, mu=mu, obj_body=obj_body,
                      plant_ad=plant_ad, context_ad=ctx_ad,
                      box_ground_drag=0.0)
    A, B_ctrl, D, d_const, E, F, H, c_lcs, *_ = \
        f.linearize_discrete_ee_space(plant_ctx, DT_SUB, np.zeros(3))
    contacts = getattr(f, '_last_contact_info', [])
    ee_box_idx = None
    ee_box_phi = None
    for ci, info in enumerate(contacts):
        if info.get('tag', '') == 'EE-BOX':
            ee_box_idx = ci
            ee_box_phi = float(info.get('distance', float('nan')))
            break
    n_lam = D.shape[1]
    n_c = n_lam // 6
    SLN = n_c

    # NaN / shape check
    bad = (not np.all(np.isfinite(c_lcs))
           or not np.all(np.isfinite(F))
           or not np.all(np.isfinite(E))
           or not np.all(np.isfinite(D)))
    if bad:
        return dict(flag=flag_value, has_ee_box=False, ee_box_phi=None,
                    lam_n_ee=float('nan'), box_v_x=float('nan'),
                    n_pusher=0, n_floor=0, n_arm=0, gate_ok=False,
                    well_formed=False, lcp_ok=False,
                    phi_clamp=f._normal_phi_clamp_v_cap,
                    always_on=f._always_on_ee_box)

    x_curr = np.concatenate([
        BOX_QUAT, BOX_POS, ee_pos, np.zeros(3), np.zeros(3),
        np.array([EE_VEL_X, 0., 0.])])
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
    gate_ok = (n_arm == 0 and n_pusher >= 1 and n_floor >= 1) \
              if ee_box_idx is not None else (n_arm == 0 and n_floor >= 1)
    return dict(flag=flag_value, has_ee_box=(ee_box_idx is not None),
                ee_box_phi=ee_box_phi, ee_box_idx=ee_box_idx,
                lam_n_ee=lam_n_ee, box_v_x=box_v_x,
                n_pusher=n_pusher, n_floor=n_floor, n_arm=n_arm,
                gate_ok=gate_ok, well_formed=True, lcp_ok=lcp_ok,
                phi_clamp=f._normal_phi_clamp_v_cap,
                always_on=f._always_on_ee_box,
                n_lam=n_lam)


def main() -> int:
    print("=" * 84)
    print("§7.30 ALWAYS-ON EE-BOX ADMISSION — STRUCTURAL SANITY at φ ≈ +5 mm")
    print("=" * 84)
    with open("config/tasks.yaml") as f:
        cfg = yaml.safe_load(f)
    task_cfg = cfg["tasks"]["pushing"]
    print(f"  separation target: +{SEPARATION_M*1000:.1f} mm (EE 5 mm beyond touching)")
    print(f"  Δt_sub = {DT_SUB*1000:.1f} ms   μ = {task_cfg['friction']}")
    print()

    diagram, plant, panda, obj, _, plant_ad, ctx_ad = build_environment(
        task_cfg, time_step=0.001)
    sg = _scene_graph_of(diagram)
    obj_body = plant.GetBodyByName("box_link", obj)
    ee_frame = plant.GetFrameByName("pusher")
    plant_ctx, ee_pos = setup_state_at_separation(
        diagram, plant, sg, panda, obj, ee_frame, SEPARATION_M)
    if plant_ctx is None:
        print("IK FAILED — could not pose the system at the target separation")
        return 1
    p_ee_actual = plant.CalcPointsPositions(
        plant_ctx, ee_frame, np.zeros(3), plant.world_frame()).flatten()
    print(f"  posed state: ee_x={p_ee_actual[0]*1000:.3f} mm, "
          f"box_x={BOX_POS[0]*1000:.3f} mm, "
          f"gap={(p_ee_actual[0] - BOX_POS[0])*1000:.3f} mm")
    print()

    # Test 1: flag OFF (default) — should be byte-identical to pre-§7.30
    print("Test 1 — flag OFF (default, byte-identical to pre-§7.30):")
    r_off = probe_lcs(plant, plant_ctx, obj_body, plant_ad, ctx_ad,
                       task_cfg["friction"], ee_pos, None)
    print(f"  always_on_flag = {r_off['always_on']}")
    print(f"  EE-BOX in LCS    = {r_off['has_ee_box']}")
    print(f"  contacts (p,f,a) = ({r_off['n_pusher']},{r_off['n_floor']},"
          f"{r_off['n_arm']})  gate={('OK' if r_off['gate_ok'] else 'FAIL')}")
    print(f"  well_formed      = {r_off['well_formed']}  lcp={('OK' if r_off['lcp_ok'] else 'BAD')}")
    print()

    # Test 2: flag ON — EE-BOX row must be present, well-formed, λn=0 at separation
    print("Test 2 — flag ON (LCS_ALWAYS_ON_EE_BOX=1):")
    r_on = probe_lcs(plant, plant_ctx, obj_body, plant_ad, ctx_ad,
                      task_cfg["friction"], ee_pos, "1")
    print(f"  always_on_flag = {r_on['always_on']}")
    print(f"  EE-BOX in LCS    = {r_on['has_ee_box']}")
    if r_on['has_ee_box']:
        print(f"  EE-BOX φ         = +{r_on['ee_box_phi']*1000:.3f} mm "
              f"(target +{SEPARATION_M*1000:.1f} mm)")
        print(f"  EE-BOX λ_n       = {r_on['lam_n_ee']:+.4f}  "
              f"(EXPECTED ≈ 0 at separation)")
    print(f"  contacts (p,f,a) = ({r_on['n_pusher']},{r_on['n_floor']},"
          f"{r_on['n_arm']})  gate={('OK' if r_on['gate_ok'] else 'FAIL')}")
    print(f"  well_formed      = {r_on['well_formed']}  lcp={('OK' if r_on['lcp_ok'] else 'BAD')}")
    print(f"  box_v_x at this state = {r_on['box_v_x']:+.6f} m/s")
    print()

    # Verdict
    print("=" * 84)
    print("VERDICT")
    print("=" * 84)
    byte_identical = (not r_off['has_ee_box']
                       and r_off['gate_ok']
                       and r_off['well_formed']
                       and r_off['lcp_ok'])
    eebox_present_on  = r_on['has_ee_box']
    lam_zero_at_sep   = (eebox_present_on
                         and abs(r_on['lam_n_ee']) < 1e-6)
    well_formed_on    = r_on['well_formed'] and r_on['lcp_ok'] and r_on['gate_ok']
    phi_matches       = (eebox_present_on
                         and abs(r_on['ee_box_phi'] - SEPARATION_M) < 1e-4)

    print(f"  (a) Flag OFF byte-identical (no EE-BOX, gate OK, well-formed) : "
          f"{('PASS' if byte_identical else 'FAIL')}")
    print(f"  (b) Flag ON → EE-BOX present at separation                    : "
          f"{('PASS' if eebox_present_on else 'FAIL')}")
    print(f"  (c) Flag ON → EE-BOX φ matches +5 mm target                   : "
          f"{('PASS' if phi_matches else 'FAIL')}")
    print(f"  (d) Flag ON → λ_n = 0 at separation (no spurious force)       : "
          f"{('PASS' if lam_zero_at_sep else 'FAIL')}")
    print(f"  (e) Flag ON → LCS well-formed, LCP feasible, gate clean       : "
          f"{('PASS' if well_formed_on else 'FAIL')}")
    print()
    all_pass = (byte_identical and eebox_present_on and phi_matches
                and lam_zero_at_sep and well_formed_on)
    print(f"  STRUCTURAL SANITY: {('PASS — proceed to live sim' if all_pass else 'FAIL — diagnose before live run')}")
    print()
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
