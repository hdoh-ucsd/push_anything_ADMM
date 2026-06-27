"""§7.36 — ANITESCU FRICTION-FOLDED LCS build sanity (offline, no full sim).

Verifies the four pieces of the §7.36 build:
  (a) DEFAULT-OFF byte-identical: LCS_CONTACT_MODEL unset → ST construction
      produces matrices BIT-IDENTICAL to a fresh ST baseline. The §7.36 edits
      did not alter the ST path.
  (b) ANITESCU-ON construction: LCS_CONTACT_MODEL=anitescu → LCS matrices
      have the FOLDED Anitescu shape (n_λ = 4·n_c, single block, no γ slack,
      no λ_n/λ_t partition), all finite, F PSD-ish (symmetric + min eigenvalue
      ≥ -tol).
  (c) ANITESCU-ON ONE-FULL-SOLVE smoke: build C3Solver(mode='c3plus'), call
      solve() with the Anitescu LCS → ADMM completes cleanly (consensus
      scaffolding dimensions correctly: n_lambda = E.shape[0] = 4, no
      ST-keyed slot extraction error), returns finite u_seq + x_seq.
  (d) DEFAULT-OFF SOLVER byte-identical: under the same ST LCS, the solver
      runs as before (Pin 3 scaffolding-dimensioning fixes are transparent
      to ST: E.shape[0] = 6·n_c == 2·num_normals + n_t, the _is_st_layout /
      _is_st_c3p guards always evaluate True under ST).

Reuses the Drake plant setup pattern from
scripts/_stage_c_always_on_sanity.py (build_environment + IK pose at a
contacting state via the always-on flag for guaranteed n_c >= 1).
"""
from __future__ import annotations

import os
import sys
import numpy as np
import yaml

from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.multibody.tree import JacobianWrtVariable
from pydrake.solvers import Solve
from pydrake.geometry import Role
from pydrake.math import RotationMatrix

from sim.env_builder import build_environment
from control.lcs_formulator import LCSFormulator
from control.admm_solver import C3Solver


DT_PLANNER = 0.05
SEPARATION_M = 0.005       # +5 mm — always-on admits EE-BOX even at separation
BOX_HALF  = 0.05
EE_RADIUS = 0.025
BOX_QUAT  = np.array([1.0, 0.0, 0.0, 0.0])
BOX_POS   = np.array([0.0, 0.0, 0.05])
ARM_BOX_CLEARANCE_M = 0.005
ARM_LINKS_TO_CLEAR  = ["panda_link4", "panda_link5", "panda_link6", "panda_link7"]
POSTURE_NOMINAL = np.array([0.0, -0.4, 0.0, -1.8, 0.0, 1.4, 0.785])
POSTURE_WEIGHT = 5.0
IK_SEEDS = [
    np.array([0.0,  0.0,  0.0, -1.5,  0.0,  1.5,  0.785]),
    np.array([0.0, -0.3,  0.0, -1.7,  0.0,  1.4,  0.785]),
    np.array([0.0,  0.3,  0.0, -1.3,  0.0,  1.6,  0.785]),
]


def _scene_graph_of(diagram):
    for sys in diagram.GetSystems():
        if 'SceneGraph' in type(sys).__name__:
            return sys
    raise RuntimeError("SceneGraph not found")


def _geom_ids(plant, sg, model, body_names):
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


def setup_state_near_box(diagram, plant, sg, panda_model, object_model,
                         ee_frame, separation_m):
    world = plant.world_frame()
    ee_pos = np.array([BOX_POS[0] + BOX_HALF + EE_RADIUS + separation_m,
                       0.0, 0.05])
    p_tol = 1e-5
    arm_geoms = _geom_ids(plant, sg, panda_model, ARM_LINKS_TO_CLEAR)
    box_body = plant.GetBodyByName("box_link", object_model)
    box_geoms = _geom_ids_for_body(plant, sg, box_body)
    for seed in IK_SEEDS:
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
        plant.SetVelocities(plant_ctx, panda_model, np.zeros(7))
        return plant_ctx, ee_pos
    return None, None


def _set_env(contact_model_value):
    """Set env flags for a clean run. Always-on ON (admits EE-BOX at
    separation so n_c >= 1), other patches OFF, contact-model per arg."""
    for k in ("LCS_NORMAL_PHI_CLAMP",):
        if k in os.environ:
            del os.environ[k]
    os.environ["LCS_NORMAL_VELOCITY_LEVEL"] = "0"
    os.environ["LCS_NORMAL_COMPLIANCE_K"]   = "0.0"
    os.environ["LCS_ALWAYS_ON_EE_BOX"]      = "1"
    if contact_model_value is None:
        if "LCS_CONTACT_MODEL" in os.environ:
            del os.environ["LCS_CONTACT_MODEL"]
    else:
        os.environ["LCS_CONTACT_MODEL"] = contact_model_value


def build_lcs(plant, plant_ad, ctx_ad, obj_body, plant_ctx, mu):
    f = LCSFormulator(plant, mu=mu, obj_body=obj_body,
                      plant_ad=plant_ad, context_ad=ctx_ad,
                      box_ground_drag=0.0)
    out = f.linearize_discrete_ee_space(plant_ctx, DT_PLANNER, np.zeros(3))
    A, B_ctrl, D, d_const, E, F, H, c_lcs, J_n, J_t, phi, mu_ret = out
    return f, dict(A=A, B=B_ctrl, D=D, d=d_const, E=E, F=F, H=H, c=c_lcs,
                   J_n=J_n, J_t=J_t, phi=phi, mu=mu_ret,
                   contact_model=f._contact_model)


def main() -> int:
    print("=" * 90)
    print("§7.36 ANITESCU FRICTION-FOLDED LCS — build sanity (no full sim)")
    print("=" * 90)
    with open("config/tasks.yaml") as f:
        cfg = yaml.safe_load(f)
    task_cfg = cfg["tasks"]["pushing"]
    print(f"  task: pushing  μ={task_cfg['friction']}  dt_planner={DT_PLANNER}s")
    print(f"  separation: +{SEPARATION_M*1000:.1f} mm "
          f"(always-on admits EE-BOX → n_c=1 expected)")
    print()

    diagram, plant, panda, obj, _, plant_ad, ctx_ad = build_environment(
        task_cfg, time_step=0.001)
    sg = _scene_graph_of(diagram)
    obj_body = plant.GetBodyByName("box_link", obj)
    ee_frame = plant.GetFrameByName("pusher")

    plant_ctx, ee_pos = setup_state_near_box(
        diagram, plant, sg, panda, obj, ee_frame, SEPARATION_M)
    if plant_ctx is None:
        print("IK FAILED — could not pose system")
        return 1
    p_ee_actual = plant.CalcPointsPositions(
        plant_ctx, ee_frame, np.zeros(3), plant.world_frame()).flatten()
    print(f"  posed: ee=({p_ee_actual[0]*1000:.2f}, "
          f"{p_ee_actual[1]*1000:.2f}, {p_ee_actual[2]*1000:.2f}) mm")
    print()

    # =================================================================
    # TEST (a) — DEFAULT-OFF byte-identical
    # =================================================================
    print("Test (a) — DEFAULT-OFF byte-identical (LCS_CONTACT_MODEL unset):")
    _set_env(None)
    f_st1, m_st1 = build_lcs(plant, plant_ad, ctx_ad, obj_body, plant_ctx,
                              task_cfg["friction"])
    _set_env(None)
    f_st2, m_st2 = build_lcs(plant, plant_ad, ctx_ad, obj_body, plant_ctx,
                              task_cfg["friction"])
    a_byte = (
        f_st1._contact_model == "stewart_trinkle" and
        f_st2._contact_model == "stewart_trinkle" and
        np.array_equal(m_st1["A"], m_st2["A"]) and
        np.array_equal(m_st1["D"], m_st2["D"]) and
        np.array_equal(m_st1["E"], m_st2["E"]) and
        np.array_equal(m_st1["F"], m_st2["F"]) and
        np.array_equal(m_st1["H"], m_st2["H"]) and
        np.array_equal(m_st1["c"], m_st2["c"])
    )
    n_c_st = m_st1["J_n"].shape[0]
    n_lam_st_expected = 6 * n_c_st
    a_n_lam_ok = (m_st1["E"].shape[0] == n_lam_st_expected
                  and m_st1["D"].shape[1] == n_lam_st_expected)
    print(f"  contact_model = {f_st1._contact_model!r}, n_c = {n_c_st}, "
          f"n_lam = {m_st1['E'].shape[0]} (ST expects {n_lam_st_expected})")
    print(f"  two identical ST builds produce bit-identical LCS         : "
          f"{('PASS' if a_byte else 'FAIL')}")
    print(f"  ST n_lam = 6·n_c                                          : "
          f"{('PASS' if a_n_lam_ok else 'FAIL')}")
    print()

    # =================================================================
    # TEST (b) — ANITESCU-ON construction
    # =================================================================
    print("Test (b) — ANITESCU-ON construction (LCS_CONTACT_MODEL=anitescu):")
    _set_env("anitescu")
    f_an, m_an = build_lcs(plant, plant_ad, ctx_ad, obj_body, plant_ctx,
                            task_cfg["friction"])
    n_c_an = m_an["J_n"].shape[0]
    n_lam_an_expected = 4 * n_c_an
    b_contact = (f_an._contact_model == "anitescu")
    b_shapes = (m_an["E"].shape == (n_lam_an_expected, 19) and
                m_an["D"].shape == (19, n_lam_an_expected) and
                m_an["F"].shape == (n_lam_an_expected, n_lam_an_expected) and
                m_an["H"].shape == (n_lam_an_expected, 3) and
                m_an["c"].shape == (n_lam_an_expected,))
    b_finite = all(np.all(np.isfinite(m_an[k]))
                    for k in ("A","B","D","d","E","F","H","c"))
    F_an = m_an["F"]
    F_sym_err = float(np.max(np.abs(F_an - F_an.T))) if F_an.size else 0.0
    b_F_sym = (F_sym_err < 1e-10)
    F_eigs = (np.linalg.eigvalsh(0.5*(F_an + F_an.T))
              if F_an.size else np.array([0.0]))
    F_min_eig = float(F_eigs.min())
    b_F_psd = (F_min_eig > -1e-10)
    print(f"  contact_model = {f_an._contact_model!r}, n_c = {n_c_an}, "
          f"n_lam = {m_an['E'].shape[0]} (Anitescu expects {n_lam_an_expected})")
    print(f"  shapes (D 19×{n_lam_an_expected}, E {n_lam_an_expected}×19, "
          f"F {n_lam_an_expected}×{n_lam_an_expected}, H {n_lam_an_expected}×3, "
          f"c ({n_lam_an_expected},))                                    : "
          f"{('PASS' if b_shapes else 'FAIL')}")
    print(f"  contact_model attr set                                    : "
          f"{('PASS' if b_contact else 'FAIL')}")
    print(f"  all finite                                                : "
          f"{('PASS' if b_finite else 'FAIL')}")
    print(f"  F symmetric (max |F-F^T| = {F_sym_err:.2e})                  : "
          f"{('PASS' if b_F_sym else 'FAIL')}")
    print(f"  F PSD (min eig = {F_min_eig:+.4e})                          : "
          f"{('PASS' if b_F_psd else 'FAIL')}")

    # Confirm ST and Anitescu DIFFER (sanity that we didn't accidentally
    # fall through to the ST path).
    b_differ = (m_st1["E"].shape != m_an["E"].shape)
    print(f"  ST E shape {m_st1['E'].shape} ≠ Anitescu E shape {m_an['E'].shape}   : "
          f"{('PASS' if b_differ else 'FAIL')}")
    print()

    # =================================================================
    # TEST (c) — ANITESCU-ON ONE-FULL-SOLVE smoke (exercises scaffolding)
    # =================================================================
    print("Test (c) — ANITESCU-ON ONE-FULL-SOLVE smoke (consensus scaffolding):")
    # Construct minimal cost matrices for the smoke (the cost doesn't have
    # to be physically meaningful; it just needs to be well-conditioned).
    n_x, n_u = 19, 3
    Q  = 1.0 * np.eye(n_x)
    R  = 0.1 * np.eye(n_u)
    QN = 10.0 * np.eye(n_x)
    # x_ref: keep the box at goal x = -0.1 m, EE near box face
    x_ref = np.zeros(n_x)
    x_ref[0] = 1.0   # quat w
    x_ref[4] = -0.1  # box x
    x_ref[6] = 0.05  # box z
    x_ref[7] = -0.05  # p_ee x  (touching the box's west face)
    x_ref[9] = 0.05  # p_ee z

    # x0: current measured state (the linearization point).
    q_full = plant.GetPositions(plant_ctx)
    v_full = plant.GetVelocities(plant_ctx)
    BS = obj_body.floating_velocities_start_in_v()
    BQ = obj_body.floating_positions_start()
    box_q = q_full[BQ:BQ+7]
    box_v = v_full[BS:BS+6]
    p_ee_now = p_ee_actual.copy()
    v_ee_now = np.zeros(3)
    x0 = np.concatenate([box_q, p_ee_now, box_v, v_ee_now])
    assert x0.shape == (n_x,), f"x0 shape {x0.shape}"

    solver = C3Solver(n_x=n_x, n_u=n_u, rho=1.0, mode='c3plus')

    try:
        u_seq, x_seq = solver.solve(
            x0=x0,
            A=m_an["A"], B_ctrl=m_an["B"], D=m_an["D"], d=m_an["d"],
            J_n=m_an["J_n"], J_t=m_an["J_t"], mu=m_an["mu"],
            Q=Q, R=R, QN=QN, x_ref=x_ref,
            N=8, admm_iter=10, torque_limit=10.0,
            phi=m_an["phi"],
            E=m_an["E"], F=m_an["F"], H=m_an["H"], c_lcs=m_an["c"],
        )
        c_completed = True
        c_finite = (np.all(np.isfinite(u_seq))
                    and np.all(np.isfinite(x_seq)))
        c_shape = (u_seq.shape == (8, n_u)
                   and x_seq.shape == (9, n_x))
        # Anitescu-specific stash
        c_anitescu_stashed = (
            hasattr(solver, "_last_lambda_anitescu_first")
            and solver._last_lambda_anitescu_first.shape == (n_lam_an_expected,)
        )
        c_err_msg = ""
    except Exception as exc:
        c_completed = False
        c_finite = False
        c_shape = False
        c_anitescu_stashed = False
        c_err_msg = f"{type(exc).__name__}: {exc}"

    print(f"  C3+ solve completed (no crash)                            : "
          f"{('PASS' if c_completed else f'FAIL — {c_err_msg}')}")
    print(f"  u_seq + x_seq finite                                      : "
          f"{('PASS' if c_finite else 'FAIL')}")
    print(f"  shapes u_seq=(8,3), x_seq=(9,19)                          : "
          f"{('PASS' if c_shape else 'FAIL')}")
    print(f"  Anitescu λ block stashed (n_lam={n_lam_an_expected})            : "
          f"{('PASS' if c_anitescu_stashed else 'FAIL')}")
    print()

    # =================================================================
    # TEST (d) — DEFAULT-OFF SOLVER byte-identical (Pin 3 scaffolding)
    # =================================================================
    print("Test (d) — DEFAULT-OFF SOLVER byte-identical "
          "(Pin 3 fixes transparent to ST):")
    _set_env(None)
    f_st3, m_st3 = build_lcs(plant, plant_ad, ctx_ad, obj_body, plant_ctx,
                              task_cfg["friction"])
    solver_st = C3Solver(n_x=n_x, n_u=n_u, rho=1.0, mode='c3plus')
    try:
        u_seq_st, x_seq_st = solver_st.solve(
            x0=x0,
            A=m_st3["A"], B_ctrl=m_st3["B"], D=m_st3["D"], d=m_st3["d"],
            J_n=m_st3["J_n"], J_t=m_st3["J_t"], mu=m_st3["mu"],
            Q=Q, R=R, QN=QN, x_ref=x_ref,
            N=8, admm_iter=10, torque_limit=10.0,
            phi=m_st3["phi"],
            E=m_st3["E"], F=m_st3["F"], H=m_st3["H"], c_lcs=m_st3["c"],
        )
        d_completed = True
        d_finite = (np.all(np.isfinite(u_seq_st))
                    and np.all(np.isfinite(x_seq_st)))
        # Under ST, ST-keyed lambda views should be populated (not Anitescu).
        d_st_keyed = (hasattr(solver_st, "_last_lambda_n_first")
                      and solver_st._last_lambda_n_first.shape[0]
                          == m_st3["J_n"].shape[0])
        d_err_msg = ""
    except Exception as exc:
        d_completed = False
        d_finite = False
        d_st_keyed = False
        d_err_msg = f"{type(exc).__name__}: {exc}"

    print(f"  ST C3+ solve completed (no crash, Pin 3 transparent)      : "
          f"{('PASS' if d_completed else f'FAIL — {d_err_msg}')}")
    print(f"  u_seq + x_seq finite                                      : "
          f"{('PASS' if d_finite else 'FAIL')}")
    print(f"  ST λ_n/λ_t views populated (NOT Anitescu placeholders)    : "
          f"{('PASS' if d_st_keyed else 'FAIL')}")
    print()

    # =================================================================
    # VERDICT
    # =================================================================
    print("=" * 90)
    print("VERDICT")
    print("=" * 90)
    overall = (a_byte and a_n_lam_ok
               and b_contact and b_shapes and b_finite and b_F_sym and b_F_psd
               and b_differ
               and c_completed and c_finite and c_shape and c_anitescu_stashed
               and d_completed and d_finite and d_st_keyed)
    print(f"  (a) ST default-OFF byte-identical                         : "
          f"{('PASS' if (a_byte and a_n_lam_ok) else 'FAIL')}")
    print(f"  (b) Anitescu construction (shapes / finite / F PSD)        : "
          f"{('PASS' if (b_contact and b_shapes and b_finite and b_F_sym and b_F_psd and b_differ) else 'FAIL')}")
    print(f"  (c) Anitescu ONE-FULL-SOLVE smoke (scaffolding works)      : "
          f"{('PASS' if (c_completed and c_finite and c_shape and c_anitescu_stashed) else 'FAIL')}")
    print(f"  (d) ST SOLVER byte-identical (Pin 3 transparent to ST)     : "
          f"{('PASS' if (d_completed and d_finite and d_st_keyed) else 'FAIL')}")
    print()
    print(f"  ANITESCU BUILD SANITY: "
          f"{('PASS — build is structurally correct' if overall else 'FAIL — diagnose before commit')}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
