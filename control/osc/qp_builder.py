"""Builds the per-tick OSC QP.

Decision variables:
    v̇ ∈ ℝ^{n_v}   (generalized acceleration; arm + floating manipuland)
    τ ∈ ℝ^{n_a}   (arm joint torques, n_a = 7 for Franka)

Costs:
    W_task    · ‖J·v̇ + J̇·v − ẍ_cmd‖²
    W_posture · ‖(v̇ − v̇_posture_cmd)[arm]‖²
    w_τ_reg   · ‖τ‖²

Equality:
    M·v̇ + C·v − τ_g = B·τ
    (= dynamics, with τ_g the generalized gravity force —
     see dynamics_helpers.get_gravity for sign convention.)

Inequality:
    τ_min ≤ τ ≤ τ_max  (per-joint)

The task tracking is a soft cost (not a constraint), so the QP is
always feasible: if the task is unreachable in one tick, the QP
admits whatever the torque limits + dynamics allow, and the residual
shows up as nonzero task cost.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pydrake.all as ad
from pydrake.solvers import MathematicalProgram

from control.osc.dynamics_helpers import (
    get_actuation_matrix,
    get_bias_acceleration,
    get_coriolis_centrifugal,
    get_gravity,
    get_jacobian,
    get_mass_matrix,
)


def build_osc_qp(
    plant,
    ctx,
    ee_frame,
    *,
    x_des:           np.ndarray,
    xd_des:          np.ndarray,
    q_home_arm:      np.ndarray,
    q_now_arm:       np.ndarray,
    v_now_arm:       np.ndarray,
    Kp_task:         np.ndarray,
    Kd_task:         np.ndarray,
    Kp_posture:      np.ndarray,
    Kd_posture:      np.ndarray,
    W_task:          float,
    W_posture:       float,
    torque_limits:   np.ndarray,
    w_tau_reg:       float = 1e-4,
    n_arm:           int = 7,
) -> Tuple[MathematicalProgram, dict]:
    """Assemble the OSC QP. The plant ctx must be at (q_now, v_now) on
    entry; the dynamics terms (M, C·v, g) and kinematics terms (J, J̇·v)
    are evaluated at that state. Returns (prog, vars_dict) so the
    caller can solve and read solutions per call.

    Returns
    -------
    prog       : MathematicalProgram ready for OSC solver.
    vars_dict  : keys "qdd" (n_v decision variables), "tau" (n_a).
    """
    n_v = plant.num_velocities()
    n_a = int(n_arm)

    # ---- Plant queries -------------------------------------------------
    M    = get_mass_matrix(plant, ctx)                            # (n_v, n_v)
    Cv   = get_coriolis_centrifugal(plant, ctx)                   # (n_v,)
    tau_g = get_gravity(plant, ctx)                               # (n_v,)
    B    = get_actuation_matrix(plant)                            # (n_v, n_a)
    J    = get_jacobian(plant, ctx, ee_frame)                     # (3, n_v)
    Jdv  = get_bias_acceleration(plant, ctx, ee_frame)            # (3,)

    # ---- Current EE state for task feedback ----------------------------
    x_now = plant.CalcPointsPositions(
        ctx, ee_frame, np.zeros(3), plant.world_frame(),
    ).flatten()
    v_now_full = plant.GetVelocities(ctx)
    xd_now = J @ v_now_full

    # ---- Task and posture commanded accelerations ----------------------
    xdd_cmd = (
        Kp_task * (x_des - x_now)
        + Kd_task * (xd_des - xd_now)
    )                                                              # (3,)
    qdd_posture_cmd_arm = (
        Kp_posture * (q_home_arm - q_now_arm)
        + Kd_posture * (-v_now_arm)
    )                                                              # (n_a,)

    # ---- QP assembly ---------------------------------------------------
    prog = MathematicalProgram()
    qdd = prog.NewContinuousVariables(n_v, "qdd")
    tau = prog.NewContinuousVariables(n_a, "tau")

    # Use Add2NormSquaredCost (‖Ax − b‖²) for all three costs — this
    # cleanly handles the rank-deficient task Jacobian (J is 3×n_v;
    # JᵀJ is rank-3 PSD-but-not-PD, which Drake's strict convexity
    # check on AddQuadraticCost rejects with default settings).

    # Cost 1: task tracking. Residual r = J·qdd + Jdv − xdd_cmd.
    # Add2NormSquaredCost computes ‖A·x − b‖², so set
    # A = sqrt(W_task)·J,  b = sqrt(W_task)·(xdd_cmd − Jdv).
    sqrt_W_task = float(np.sqrt(W_task))
    prog.Add2NormSquaredCost(
        sqrt_W_task * J,
        sqrt_W_task * (xdd_cmd - Jdv),
        qdd,
    )

    # Cost 2: posture — arm DOFs only. ‖qdd_arm − qdd_posture_cmd‖²
    # weighted by W_posture.
    sqrt_W_posture = float(np.sqrt(W_posture))
    prog.Add2NormSquaredCost(
        sqrt_W_posture * np.eye(n_a),
        sqrt_W_posture * qdd_posture_cmd_arm,
        qdd[:n_a],
    )

    # Cost 3: torque regularization — tiny tie-breaker.
    sqrt_w_tau = float(np.sqrt(w_tau_reg))
    prog.Add2NormSquaredCost(
        sqrt_w_tau * np.eye(n_a),
        np.zeros(n_a),
        tau,
    )

    # Equality: M·qdd + C·v − τ_g − B·τ = 0
    # i.e. [M, −B] · [qdd; tau] = τ_g − C·v
    A_eq = np.hstack([M, -B])
    b_eq = tau_g - Cv
    prog.AddLinearEqualityConstraint(A_eq, b_eq, np.concatenate([qdd, tau]))

    # Inequality: τ bounded per joint.
    prog.AddBoundingBoxConstraint(-torque_limits, torque_limits, tau)

    return prog, {"qdd": qdd, "tau": tau, "M": M, "Cv": Cv, "tau_g": tau_g,
                  "B": B, "J": J, "Jdv": Jdv,
                  "x_now": x_now, "xd_now": xd_now, "xdd_cmd": xdd_cmd}
