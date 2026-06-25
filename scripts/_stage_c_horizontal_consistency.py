"""Stage C horizontal model-plant consistency probe — does the LCS at
LCS_EXPLICIT_BOX_GND=4 predict the push Drake renders at an EE-in-contact
state?

Constructs a contact state offline:
  - Box at rest pose:  (0, 0, 0.05),  quat = identity
  - EE in slight penetration with the box east face (so the push direction
    is west, matching task-id 4)
  - EE moving westward at -0.05 m/s (an initial momentum injection — drives
    the contact and produces non-trivial push dynamics)
  - u = 0 (no commanded EE force; the only horizontal driver is the
    compliant / LCP contact under the EE momentum)

Tests BOTH axes:
  (a) HORIZONTAL: does the LCS-with-oracle predict box xy motion that
      Drake renders? (the push-axis analog of the §7.10 vertical
      confirmation)
  (b) VERTICAL still holds: the floor still keeps the box up under push
      conditions.
"""
from __future__ import annotations

import os
import numpy as np
import yaml

from pydrake.systems.analysis import Simulator
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.solvers import Solve

from sim.env_builder import build_environment
from control.lcp_solver import solve_lcp

DT_PLANNER = 0.05  # planner Δt the LCS A,B,D,d are discretized over
BOX_HALF = 0.05
EE_RADIUS = 0.025

# Contact state design:
#   Box at rest at origin, z = 0.05 (resting on table)
#   EE on east face (x = +BOX_HALF), penetrating by 0.1 mm
#   EE moving westward at 0.05 m/s
BOX_QUAT  = np.array([1.0, 0.0, 0.0, 0.0])
BOX_POS   = np.array([0.0, 0.0, 0.05])
EE_PEN_M  = 0.0001        # 0.1 mm penetration
EE_POS    = np.array([BOX_HALF + EE_RADIUS - EE_PEN_M, 0.0, 0.05])
EE_VEL_X  = -0.05         # m/s westward (push direction)


def setup_contact_state(diagram, plant, panda_model, object_model, ee_frame):
    """Set Drake state to the constructed contact configuration.
    Returns (context, plant_ctx)."""
    context = diagram.CreateDefaultContext()
    plant_ctx = plant.GetMyContextFromRoot(context)

    # 1. Place the box
    box_q = np.concatenate([BOX_QUAT, BOX_POS])
    plant.SetPositions(plant_ctx, object_model, box_q)

    # 2. IK to position EE at EE_POS
    ik = InverseKinematics(plant, plant_ctx, with_joint_limits=True)
    p_tol = 1e-4
    ik.AddPositionConstraint(
        ee_frame, np.zeros(3), plant.world_frame(),
        EE_POS - p_tol, EE_POS + p_tol,
    )
    q0 = plant.GetPositions(plant_ctx)
    seed = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.785])
    plant.SetPositions(plant_ctx, panda_model, seed)
    q0 = plant.GetPositions(plant_ctx)
    ik.get_mutable_prog().SetInitialGuess(ik.q(), q0)
    result = Solve(ik.prog())
    if not result.is_success():
        print("[probe] IK FAILED to put EE at contact pose")
        return None, None
    q_full = result.GetSolution(ik.q())
    plant.SetPositions(plant_ctx, q_full)

    # 3. Box velocity = 0
    plant.SetVelocities(plant_ctx, object_model, np.zeros(6))

    # 4. Arm joint velocities that give EE Cartesian velocity (-0.05, 0, 0)
    #    From J_ee_translational · q_dot = v_ee, solve least-norm.
    J_ee = plant.CalcJacobianTranslationalVelocity(
        plant_ctx,
        # Use the spatial-velocity Jacobian's translational part
        plant.GetFrameByName("world").body().model_instance() and
        # (the API needs JacobianWrtVariable arg)
        __import__("pydrake.multibody.tree", fromlist=["JacobianWrtVariable"]
                   ).JacobianWrtVariable.kV,
        ee_frame, np.zeros(3),
        plant.world_frame(), plant.world_frame(),
    )
    # J_ee is (3, n_v). The arm's velocity columns are the panda model
    # instance v indices.
    panda_v_start = plant.GetJointByName(
        "panda_joint1").velocity_start()  # 0
    n_arm = 7
    J_arm = J_ee[:, panda_v_start : panda_v_start + n_arm]
    v_ee_target = np.array([EE_VEL_X, 0.0, 0.0])
    # least-norm: q_dot = J^+ · v_target
    q_arm_dot, *_ = np.linalg.lstsq(J_arm, v_ee_target, rcond=None)
    plant.SetVelocities(plant_ctx, panda_model, q_arm_dot)

    # Verify
    ee_pos = plant.CalcPointsPositions(
        plant_ctx, ee_frame, np.zeros(3), plant.world_frame()).flatten()
    ee_v = plant.CalcJacobianTranslationalVelocity(
        plant_ctx,
        __import__("pydrake.multibody.tree", fromlist=["JacobianWrtVariable"]
                   ).JacobianWrtVariable.kV,
        ee_frame, np.zeros(3),
        plant.world_frame(), plant.world_frame(),
    ) @ plant.GetVelocities(plant_ctx)
    print(f"  EE pos achieved: {ee_pos} (target: {EE_POS})  Δ {np.linalg.norm(ee_pos - EE_POS)*1000:.3f} mm")
    print(f"  EE vel achieved: {ee_v} (target: {v_ee_target})")

    return context, plant_ctx


def run():
    print("=" * 72)
    print("STAGE C HORIZONTAL CONSISTENCY PROBE — contact state, count=4")
    print("=" * 72)
    print(f"  Constructed contact state:")
    print(f"    box position = {BOX_POS}    quat = {BOX_QUAT}")
    print(f"    EE position  = {EE_POS}  (east face, {EE_PEN_M*1000:.2f} mm penetration)")
    print(f"    EE velocity  = ({EE_VEL_X}, 0, 0) m/s  (westward push)")
    print(f"    u (EE force) = 0  (no commanded force; momentum drives contact)")

    os.environ["LCS_EXPLICIT_BOX_GND"] = "4"
    import importlib, control.lcs_formulator
    importlib.reload(control.lcs_formulator)
    from control.lcs_formulator import LCSFormulator

    with open("config/tasks.yaml") as f:
        cfg = yaml.safe_load(f)
    task_cfg = cfg["tasks"]["pushing"]

    diagram, plant, panda_model, object_model, _, plant_ad, ctx_ad = \
        build_environment(task_cfg, time_step=0.001)
    obj_body = plant.GetBodyByName("box_link", object_model)
    ee_frame = plant.GetFrameByName("pusher")

    context, plant_ctx = setup_contact_state(
        diagram, plant, panda_model, object_model, ee_frame)
    if context is None:
        return

    mu = task_cfg["friction"]
    formulator = LCSFormulator(plant, mu=mu, obj_body=obj_body,
                               plant_ad=plant_ad, context_ad=ctx_ad)
    u_lin = np.zeros(3)
    A, B_ctrl, D, d_const, E, F, H, c_lcs, J_n, J_t, phi_vec, mu_vec = \
        formulator.linearize_discrete_ee_space(plant_ctx, DT_PLANNER, u_lin)
    n_x = A.shape[0]
    n_lambda = D.shape[1]
    print(f"\n  Re-extracted LCS: n_x={n_x}, n_lambda={n_lambda}")
    contacts = getattr(formulator, '_last_contact_info', [])
    print(f"  contacts found: {len(contacts)}")
    for i, info in enumerate(contacts):
        print(f"    pair {i}: tag={info.get('tag')}  dist={info.get('distance')}")

    # Build x0 in the EE-space layout
    box_q = np.concatenate([BOX_QUAT, BOX_POS])
    ee_pos_actual = plant.CalcPointsPositions(
        plant_ctx, ee_frame, np.zeros(3), plant.world_frame()).flatten()
    box_v_zero = np.zeros(6)
    ee_v_actual = np.array([EE_VEL_X, 0.0, 0.0])
    x0 = np.concatenate([box_q, ee_pos_actual, box_v_zero, ee_v_actual])
    assert x0.shape[0] == n_x, f"x0 shape {x0.shape} != n_x {n_x}"

    # Solve LCP for λ at u=0:  0 ≤ λ ⊥ (F·λ + E·x0 + c) ≥ 0
    q_lcp = E @ x0 + c_lcs
    try:
        lam_new, lcp_res = solve_lcp(F, q_lcp)
        print(f"  Lemke residual: {lcp_res:.3e}")
    except Exception as e:
        print(f"  Lemke FAILED: {e}")
        return

    w_check = F @ lam_new + q_lcp
    lam_pos = bool(np.all(lam_new >= -1e-6))
    w_pos = bool(np.all(w_check >= -1e-6))
    compl = float(np.max(np.abs(lam_new * w_check)))
    print(f"  Feasibility: λ≥0 {lam_pos}, w≥0 {w_pos}, max|λw|={compl:.4e}")
    print(f"  λ_max = {float(lam_new.max()):.4f}")
    print(f"  λ nonzero indices: {np.where(lam_new > 1e-6)[0].tolist()}")
    print(f"  λ nonzero values : {lam_new[lam_new > 1e-6].tolist()}")

    # LCS-predicted next state
    x_next_LCS = A @ x0 + B_ctrl @ np.zeros(3) + D @ lam_new + d_const
    lcs_box_delta = x_next_LCS[4:7] - x0[4:7]
    lcs_box_vel = x_next_LCS[13:16]

    # Drake forward step
    sim = Simulator(diagram, context)
    sim.Initialize()
    sim.AdvanceTo(DT_PLANNER)
    plant_ctx_after = plant.GetMyContextFromRoot(context)
    xyz_after = plant.EvalBodyPoseInWorld(plant_ctx_after, obj_body).translation()
    drake_box_delta = xyz_after - BOX_POS
    drake_box_vel = plant.EvalBodySpatialVelocityInWorld(
        plant_ctx_after, obj_body).translational()

    print()
    print("=" * 72)
    print("RESULT — horizontal model-plant consistency at the contact state")
    print("=" * 72)
    print(f"  Δ box xyz       LCS:   ({lcs_box_delta[0]*1000:+.3f}, "
          f"{lcs_box_delta[1]*1000:+.3f}, {lcs_box_delta[2]*1000:+.3f}) mm")
    print(f"                 Drake:  ({drake_box_delta[0]*1000:+.3f}, "
          f"{drake_box_delta[1]*1000:+.3f}, {drake_box_delta[2]*1000:+.3f}) mm")
    print()
    print(f"  Δ box vel       LCS:   ({lcs_box_vel[0]:+.5f}, {lcs_box_vel[1]:+.5f}, "
          f"{lcs_box_vel[2]:+.5f}) m/s")
    print(f"                 Drake:  ({drake_box_vel[0]:+.5f}, {drake_box_vel[1]:+.5f}, "
          f"{drake_box_vel[2]:+.5f}) m/s")
    print()
    horiz_lcs = np.array([lcs_box_delta[0], lcs_box_delta[1]])
    horiz_drake = np.array([drake_box_delta[0], drake_box_delta[1]])
    horiz_close = np.linalg.norm(horiz_lcs - horiz_drake) * 1000 < 1.0
    vertical_close = abs(lcs_box_delta[2] - drake_box_delta[2]) * 1000 < 1.0
    print(f"  Horizontal |LCS - Drake| in xy : {np.linalg.norm(horiz_lcs - horiz_drake)*1000:.4f} mm")
    print(f"  Vertical   |LCS - Drake| in z  : {abs(lcs_box_delta[2] - drake_box_delta[2])*1000:.4f} mm")
    print(f"  HORIZONTAL CLOSE : {horiz_close}")
    print(f"  VERTICAL CLOSE   : {vertical_close}")

    print()
    print("=" * 72)
    print("ROUTE")
    print("=" * 72)
    if horiz_close and vertical_close:
        print("  → MODEL-FIXED-REAL — both axes consistent at the contact state.")
        print("    The contact-model fix (LCS_EXPLICIT_BOX_GND=4) is fully validated.")
        print("    Next: reference-settings convergence on the count=4 LCS becomes")
        print("    meaningful (re-promoted from §7.10).")
    elif vertical_close and not horiz_close:
        print("  → HORIZONTAL-GAP — vertical holds but horizontal does NOT match.")
        print("    Second contact-model gap on the push axis (EE-box dynamics,")
        print("    friction, or count=4 floor interacting with push).")
        print("    Investigate before re-promoting convergence.")
    elif not vertical_close and horiz_close:
        print("  → Unusual: horizontal close but vertical broke under push.")
        print("    Investigate the contact-model interaction.")
    else:
        print("  → STILL-BROKEN on both axes under push.")


if __name__ == "__main__":
    run()
