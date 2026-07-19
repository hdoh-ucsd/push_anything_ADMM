"""QP construction for the Operational Space Controller.

Decision variables (per OSC tick):
    v̇ ∈ ℝ^{n_v}    — generalized acceleration (arm + manipuland floating-body)
    τ ∈ ℝ^{n_u}    — joint torque (arm only)

Equality constraint (dynamics):
    M v̇ + C(q,v) v - τ_g(q) = B τ + J_full^T λ_planned

    λ_planned is the planner's desired feedforward contact force.
    Sign convention: J built from nhat_BA · (J_A − J_B), λ ≥ 0, and
    the term +J^T λ on the RHS represents the contact force pushing
    the box in the goal direction while reacting on the arm. The QP
    then solves for τ that produces EE motion consistent with that
    planned contact.

Inequality constraints:
    τ_min ≤ τ ≤ τ_max         per-joint URDF effort limits (87/87/87/87/12/12/12 Nm for Franka)

Costs:
    1. Task tracking — ‖J_v v̇ + J̇_v v − a_des‖²_W_track,
       a_des = Kp_cart · (p_des − p_now) + Kd_cart · (v_des − v_ee_now)
    2. Posture (nullspace via cost) — ‖v̇[:n_arm] − a_posture‖²_W_posture,
       a_posture = Kp_null · (q_nominal − q_arm) + Kd_null · (-v_arm)
    3. Torque regularization — w_torque · ‖τ‖²
    4. Acceleration regularization — w_acc · ‖v̇‖²

Notes:
    * v1 does NOT add contact forces as decision variables — the planner's
      λ is treated as a known external force on the RHS. Friction-cone
      constraints from dairlib's full formulation are skipped for v1
      because we don't decide the contact force; we only command τ.
    * Solver: Drake's OsqpSolver. Build → Solve is ~0.5–2 ms.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from pydrake.solvers import MathematicalProgram, Solve, OsqpSolver


@dataclass
class OscGains:
    """Numeric gains/weights consumed by `build_qp`."""
    # Cartesian PD (3D position tracking)
    Kp_cart:   np.ndarray   # (3,) N/m
    Kd_cart:   np.ndarray   # (3,) N·s/m

    # Posture (n_arm,) nullspace via cost
    Kp_null:   np.ndarray
    Kd_null:   np.ndarray

    # QP cost weights (scalars)
    W_track:   float
    W_posture: float
    W_torque:  float
    W_acc:     float

    # Force-tracking weight on ‖λ_ext − λ_des‖² (scalar). Only consulted
    # when `build_and_solve_qp` is called with use_force_tracking=True.
    # Mirrors the reference's `W_ee_lambda` (osc_params.W_ee_lambda).
    W_force:   float = 0.0

    # Joint-2 posture (dairlib `panda_joint2_target` analog). Reference:
    # franka_osc_controller.cc:159-165 (Kp=200, Kd=10, W=1, target=1.1 rad).
    # Cost: W_joint2·‖v̇[joint2_idx] − a_j2‖² where
    #   a_j2 = Kp_joint2·(target − q_arm[j2]) + Kd_joint2·(−v_arm[j2]).
    # Reproduce-dairlib Phase 1.
    Kp_joint2:         float = 0.0
    Kd_joint2:         float = 0.0
    W_joint2:          float = 0.0
    joint2_target_rad: float = 1.1
    joint2_idx:        int   = 1

    # 2.k — Rotation tracking (identity-quaternion hold). Reference:
    # osc_params.yaml EndEffectorRotW=diag(10), EndEffectorRotKp=diag(800),
    # EndEffectorRotKd=diag(40). Cost:
    #   W_rot·‖J_w·v̇ + J̇_w·v − a_rot‖², a_rot = Kp_rot·w_err + Kd_rot·(−w_now)
    # where w_err is the small-angle vector representation of the orientation
    # error between R_WE and R_WE_target. Skipped when W_rot == 0.
    Kp_rot:            np.ndarray = None
    Kd_rot:            np.ndarray = None
    W_rot:             float = 0.0

    # EE Cartesian accel command cap (reference SetCmdAccelerationBounds,
    # franka_osc_controller.cc:155-158). Reference osc_params.yaml has
    # `end_effector_acceleration: 10` applied as ±10 m/s² per axis. Any
    # (Kp·p_err + Kd·v_err) component beyond this bound is clamped before
    # the QP solves for v̇. Set 0.0 to disable (unbounded, port legacy).
    a_ee_cap:          float = 0.0


@dataclass
class OscLimits:
    """Inequality bounds for the QP."""
    tau_max:   np.ndarray   # (n_u,) per-joint effort limit (symmetric)


def build_and_solve_qp(
    *,
    M:              np.ndarray,      # (n_v, n_v)
    bias:           np.ndarray,      # (n_v,)  C v − τ_g  (sign-resolved)
    B:              np.ndarray,      # (n_v, n_u)
    n_arm:          int,
    J_v:            np.ndarray,      # (3, n_v) translational EE Jacobian
    Jdot_v_v:       np.ndarray,      # (3,) J̇_v v bias accel
    p_err:          np.ndarray,      # (3,)
    v_err:          np.ndarray,      # (3,)
    q_arm_err:      np.ndarray,      # (n_arm,)
    v_arm_err:      np.ndarray,      # (n_arm,)
    gains:          OscGains,
    limits:         OscLimits,
    F_ff_external:  np.ndarray,      # (n_v,) feedforward generalized force (+J^T λ_planned)
    solver:         Optional[OsqpSolver] = None,
    use_force_tracking: bool = False,
    lambda_des:     Optional[np.ndarray] = None,   # (3,) Cartesian force command for λ_ext tracking
    a_ff:           Optional[np.ndarray] = None,   # (3,) Cartesian feedforward acceleration (yddot_des analog)
    q_arm:          Optional[np.ndarray] = None,   # (n_arm,) arm joint positions (for joint-2 posture)
    v_arm:          Optional[np.ndarray] = None,   # (n_arm,) arm joint velocities
    J_w:            Optional[np.ndarray] = None,   # (3, n_v) EE angular Jacobian (world)
    Jdot_w_v:       Optional[np.ndarray] = None,   # (3,) angular bias accel
    w_err:          Optional[np.ndarray] = None,   # (3,) orientation error (small-angle vec)
    w_ee_now:       Optional[np.ndarray] = None,   # (3,) EE angular velocity in world
) -> Tuple[np.ndarray, np.ndarray, bool, str]:
    """Build and solve the OSC QP for one control tick.

    Returns
    -------
    u_opt       : (n_u,) commanded joint torques. Zero on failure.
    vdot_opt    : (n_v,) commanded generalized acceleration. Zero on failure.
    success     : True iff QP solved.
    result_str  : human-readable solution-result string.
    lam_ext_opt : (3,) solved external EE force. Zero on failure or when
                  use_force_tracking is False.
    """
    n_v = M.shape[0]
    n_u = B.shape[1]

    prog = MathematicalProgram()
    vdot = prog.NewContinuousVariables(n_v, "vdot")
    u    = prog.NewContinuousVariables(n_u, "u")

    if use_force_tracking:
        # λ_ext = external Cartesian force at the EE world point, tracked
        # as a *cost-only shadow variable* — mirrors the dairlib reference.
        # Reference dispatch (operational_space_control.cc:407) calls
        # `UpdateDynamics(x, active_contact_names, {})` — the empty third
        # arg means external forces (lambda_e_) are NEVER injected into
        # the dynamics constraint (inverse_dynamics_qp.cc:207-211 loop
        # skipped). Only the cost `W_ee_lambda · ‖λ_e − λ_des‖²` couples
        # λ_e to λ_des; the joint torques τ are computed as if no
        # external reaction existed. This prevents the phantom-reaction
        # slam when the planner routes a fictional u_sol force through
        # ExternalForceTrackingData (see 2026-07-19 arm-slam probe;
        # coupling λ_ext into dynamics made the arm accelerate opposite
        # to λ_des whenever real contact was absent).
        lam_ext = prog.NewContinuousVariables(3, "lambda_ext")
        # Dynamics constraint identical to the non-force-tracking path.
        A_eq = np.hstack([M, -B])                 # (n_v, n_v + n_u)
        b_eq = F_ff_external - bias               # (n_v,)
        prog.AddLinearEqualityConstraint(
            A_eq, b_eq, np.concatenate([vdot, u]))
    else:
        # Original path: λ_planned enters as a fixed RHS feedforward.
        lam_ext = None
        A_eq = np.hstack([M, -B])                 # (n_v, n_v + n_u)
        b_eq = F_ff_external - bias               # (n_v,)
        prog.AddLinearEqualityConstraint(A_eq, b_eq,
                                         np.concatenate([vdot, u]))

    # --- Box constraint: τ_min ≤ u ≤ τ_max ---
    prog.AddBoundingBoxConstraint(-limits.tau_max, limits.tau_max, u)

    # --- Cost 1: Cartesian tracking ‖J_v v̇ + J̇_v v − a_des‖² ---
    # Reference's `yddot_command = yddot_des + Kp · error_y + Kd · error_ydot`
    # (osc_tracking_data.cc:113-116). The port previously matched only the
    # PD half; `a_ff` adds the `yddot_des` feedforward term so the port's
    # a_des is PD + feedforward (faithful to the reference). When a_ff is
    # None the PD-only form is preserved byte-identically.
    if a_ff is None:
        a_des = gains.Kp_cart * p_err + gains.Kd_cart * v_err   # (3,)
    else:
        a_ff_v = np.asarray(a_ff, dtype=float).reshape(3)
        a_des = a_ff_v + gains.Kp_cart * p_err + gains.Kd_cart * v_err

    # Reference-parity accel cap (franka_osc_controller.cc:155-158,
    # SetCmdAccelerationBounds). Clamps each axis of a_des to
    # ±gains.a_ee_cap. Disabled when a_ee_cap == 0.0.
    if float(getattr(gains, "a_ee_cap", 0.0)) > 0.0:
        _cap = float(gains.a_ee_cap)
        a_des = np.clip(a_des, -_cap, +_cap)
    # residual_track = J_v @ vdot + Jdot_v_v - a_des
    # ‖residual‖² = vdot.T J_v.T J_v vdot + 2 vdot.T J_v.T (Jdot_v_v - a_des)
    #               + const
    Q_track = gains.W_track * (J_v.T @ J_v)
    b_track = gains.W_track * (J_v.T @ (Jdot_v_v - a_des))
    prog.AddQuadraticCost(2.0 * Q_track, 2.0 * b_track, vdot,
                          is_convex=True)

    # --- Cost 2: Posture (arm slice) ‖v̇_arm − a_posture‖² ---
    a_posture = gains.Kp_null * q_arm_err + gains.Kd_null * v_arm_err
    # Apply only on the first n_arm rows of vdot.
    I_arm = np.zeros((n_arm, n_v))
    I_arm[:n_arm, :n_arm] = np.eye(n_arm)
    Q_post = gains.W_posture * (I_arm.T @ I_arm)
    b_post = -gains.W_posture * (I_arm.T @ a_posture)
    prog.AddQuadraticCost(2.0 * Q_post, 2.0 * b_post, vdot,
                          is_convex=True)

    # --- Cost 3: Torque regularization w_torque · ‖u‖² ---
    if gains.W_torque > 0.0:
        prog.AddQuadraticCost(2.0 * gains.W_torque * np.eye(n_u),
                              np.zeros(n_u), u, is_convex=True)

    # --- Cost 4: Acceleration regularization w_acc · ‖v̇‖² ---
    if gains.W_acc > 0.0:
        prog.AddQuadraticCost(2.0 * gains.W_acc * np.eye(n_v),
                              np.zeros(n_v), vdot, is_convex=True)

    # --- Cost 5 (optional): Force tracking W_force·‖λ_ext − λ_des‖² ---
    # Expanded:  λ_ext^T (W_force·I) λ_ext − 2·W_force·λ_des^T·λ_ext + const.
    # Following the existing factor-of-2 convention
    # (`AddQuadraticCost(2·Q_user, 2·b_user, ...)` for cost
    # `v^T Q_user v + 2·b_user^T v`):
    #   Q_user = W_force·I_3,  b_user = −W_force·λ_des.
    if use_force_tracking and gains.W_force > 0.0:
        if lambda_des is None:
            lambda_des_v = np.zeros(3)
        else:
            lambda_des_v = np.asarray(lambda_des, dtype=float).reshape(3)
        Q_force = gains.W_force * np.eye(3)
        b_force = -gains.W_force * lambda_des_v
        prog.AddQuadraticCost(2.0 * Q_force, 2.0 * b_force, lam_ext,
                              is_convex=True)

    # --- Cost 6 (optional): Joint-2 posture ---
    # Mirrors dairlib's JointSpaceTrackingData("panda_joint2_target")
    # at franka_osc_controller.cc:159-165. Cost:
    #   W_joint2 · ‖v̇[joint2_idx] − a_j2‖²
    # with a_j2 = Kp_joint2·(target − q_arm[j2]) + Kd_joint2·(−v_arm[j2]).
    # Skipped when W_joint2 == 0 or q_arm/v_arm are None (byte-identical
    # to pre-Phase-1 behavior).
    if gains.W_joint2 > 0.0 and q_arm is not None and v_arm is not None:
        j2 = int(gains.joint2_idx)
        q_j2 = float(np.asarray(q_arm, dtype=float).reshape(-1)[j2])
        v_j2 = float(np.asarray(v_arm, dtype=float).reshape(-1)[j2])
        a_j2 = (gains.Kp_joint2 * (gains.joint2_target_rad - q_j2)
                + gains.Kd_joint2 * (-v_j2))
        e_j2 = np.zeros(n_v)
        e_j2[j2] = 1.0
        # ‖e_j2^T v̇ − a_j2‖² = v̇^T (e_j2 e_j2^T) v̇ − 2 a_j2 (e_j2^T v̇) + const
        Q_j2 = gains.W_joint2 * np.outer(e_j2, e_j2)
        b_j2 = -gains.W_joint2 * a_j2 * e_j2
        prog.AddQuadraticCost(2.0 * Q_j2, 2.0 * b_j2, vdot, is_convex=True)

    # --- Cost 7 (optional): Rotation tracking (identity-quaternion hold) ---
    # Reference osc_tracking_data.cc RotTaskSpaceTrackingData with the
    # yaml gains EndEffectorRotW=10·I, EndEffectorRotKp=800·I,
    # EndEffectorRotKd=40·I → compound rotational authority 8000/axis.
    # Cost: W_rot·‖J_w·v̇ + J̇_w·v − a_rot‖²
    # a_rot = Kp_rot·w_err + Kd_rot·(−w_ee_now)
    if (gains.W_rot > 0.0
            and J_w is not None and Jdot_w_v is not None
            and w_err is not None and w_ee_now is not None
            and gains.Kp_rot is not None and gains.Kd_rot is not None):
        _kp_rot = np.asarray(gains.Kp_rot, dtype=float).reshape(3)
        _kd_rot = np.asarray(gains.Kd_rot, dtype=float).reshape(3)
        a_rot = _kp_rot * np.asarray(w_err, dtype=float).reshape(3) \
                + _kd_rot * (-np.asarray(w_ee_now, dtype=float).reshape(3))
        Q_rot = gains.W_rot * (J_w.T @ J_w)
        b_rot = gains.W_rot * (J_w.T @ (np.asarray(Jdot_w_v).reshape(3) - a_rot))
        prog.AddQuadraticCost(2.0 * Q_rot, 2.0 * b_rot, vdot, is_convex=True)

    # --- Solve ---
    if solver is None:
        result = Solve(prog)
    else:
        result = solver.Solve(prog)

    success = result.is_success()
    result_str = str(result.get_solution_result())

    if success:
        u_opt    = result.GetSolution(u)
        vdot_opt = result.GetSolution(vdot)
        lam_ext_opt = (result.GetSolution(lam_ext)
                       if (use_force_tracking and lam_ext is not None)
                       else np.zeros(3))
    else:
        u_opt       = np.zeros(n_u)
        vdot_opt    = np.zeros(n_v)
        lam_ext_opt = np.zeros(3)

    # Stash on the result_str-adjacent return for callers that need it;
    # done as a fifth tuple element. Older callers can still unpack 4.
    return u_opt, vdot_opt, success, result_str, lam_ext_opt
