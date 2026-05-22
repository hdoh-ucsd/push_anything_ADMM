"""QP construction for the Operational Space Controller.

Decision variables (per OSC tick):
    v̇ ∈ ℝ^{n_v}    — generalized acceleration (arm + manipuland floating-body)
    τ ∈ ℝ^{n_u}    — joint torque (arm only)

Equality constraint (dynamics):
    M v̇ + C(q,v) v - τ_g(q) = B τ + J_full^T λ_planned

    λ_planned is the planner's desired feedforward contact force.
    Sign convention matches `ImpedanceController`: J built from
    nhat_BA · (J_A − J_B), λ ≥ 0, and the term +J^T λ on the RHS
    represents the contact force pushing the box in the goal direction
    while reacting on the arm. The QP then solves for τ that produces
    EE motion consistent with that planned contact.

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
        # λ_ext = external Cartesian force at the EE world point, treated
        # as a QP decision variable (mirrors dairlib reference
        # `inverse_dynamics_qp.cc:82` `lambda_e_`). Soft cost
        # `W_force·‖λ_ext − λ_des‖²` pulls it toward `lambda_des`; the QP
        # is forced to find a τ that produces that EE force.
        lam_ext = prog.NewContinuousVariables(3, "lambda_ext")
        # Dynamics:  M v̇ + bias = B u + F_ff_external + J_v^T λ_ext
        #         ⇔  M v̇ − B u − J_v^T λ_ext = F_ff_external − bias
        # J_v is the EE translational Jacobian (3, n_v) — its transpose
        # maps a 3-D world-frame EE force to a generalized force on the
        # arm DOFs (manipuland-DOF rows are zero since the EE point does
        # not move under box DOFs). This is the executor's commanded
        # contact force, separate from the LCS-reaction term in F_ff.
        A_eq = np.hstack([M, -B, -J_v.T])         # (n_v, n_v + n_u + 3)
        b_eq = F_ff_external - bias               # (n_v,)
        prog.AddLinearEqualityConstraint(
            A_eq, b_eq, np.concatenate([vdot, u, lam_ext]))
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
    a_des = gains.Kp_cart * p_err + gains.Kd_cart * v_err   # (3,)
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
