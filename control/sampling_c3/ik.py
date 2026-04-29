"""
Damped pseudoinverse inverse kinematics — lifted from the existing
GlobalSamplingC3MPC wrapper (control/global_sampling_c3.py:240-280).

Same convergence/tolerance behaviour as the legacy implementation so
the IK-resolved q_seed values match the existing wrapper's output for
identical inputs.

`plant_ctx` is left at the last IK iterate's configuration on return —
caller must restore current_q/current_v if the context is shared with
downstream consumers.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pydrake.all as ad


def solve_ik_to_ee_pos(plant,
                       ee_frame,
                       p_target:    np.ndarray,
                       q_init:      np.ndarray,
                       plant_ctx,
                       n_arm_dofs:  int,
                       max_iter:    int   = 30,
                       tol:         float = 1e-3,
                       damping:     float = 0.05,
                       q_lo:        np.ndarray | None = None,
                       q_hi:        np.ndarray | None = None,
                       limit_margin: float = 0.005) -> Tuple[np.ndarray, float, int]:
    """Iterated DLS IK: find q s.t. FK_ee(q) ≈ p_target.

    Parameters
    ----------
    plant       : Drake MultibodyPlant (must be Finalized)
    ee_frame    : Drake Frame for the end-effector
    p_target    : (3,)  desired EE world-frame position
    q_init      : (n_q,) starting joint configuration
    plant_ctx   : Drake plant Context (will be modified in place)
    n_arm_dofs  : number of arm DOFs (first n_arm_dofs columns of J_ee)
    max_iter    : DLS iteration cap
    tol         : EE-position error norm at which to early-terminate (m)
    damping     : Levenberg-Marquardt damping parameter
    q_lo, q_hi  : (>= n_arm_dofs,) optional lower/upper joint limits. When
                  both are supplied the arm portion of `q` is clipped to
                  `[q_lo + limit_margin, q_hi - limit_margin]` after the
                  initial seed copy and after each DLS step. Pass `None`
                  (the default) to disable clipping — legacy callers see
                  identical behaviour.
    limit_margin : safety pad pulled in from each joint limit (rad). The
                  default of 5 mrad is small enough not to over-constrain
                  reachability but large enough that a downstream joint-
                  limit safety check at ±50 mrad does not flag a freshly
                  converged pose.

    Returns
    -------
    q        : (n_q,) converged (or best-effort) joint configuration
    err_norm : final EE-position error norm (m)
    iters    : number of iterations actually executed
    """
    world = plant.world_frame()
    q        = q_init.copy()
    err_norm = float("inf")
    iters    = 0

    use_limits = (q_lo is not None) and (q_hi is not None)
    if use_limits:
        q_lo_eff = q_lo[:n_arm_dofs] + limit_margin
        q_hi_eff = q_hi[:n_arm_dofs] - limit_margin
        np.clip(q[:n_arm_dofs], q_lo_eff, q_hi_eff, out=q[:n_arm_dofs])

    for i in range(max_iter):
        plant.SetPositions(plant_ctx, q)
        p_curr = plant.CalcPointsPositions(
            plant_ctx, ee_frame, np.zeros(3), world,
        ).flatten()
        err      = p_target - p_curr
        err_norm = float(np.linalg.norm(err))
        iters    = i + 1
        if err_norm < tol:
            break
        J_ee = plant.CalcJacobianTranslationalVelocity(
            plant_ctx, ad.JacobianWrtVariable.kV,
            ee_frame, np.zeros(3),
            world, world,
        )
        J_arm = J_ee[:, :n_arm_dofs]
        dq    = J_arm.T @ np.linalg.solve(
            J_arm @ J_arm.T + damping ** 2 * np.eye(3), err
        )
        q[:n_arm_dofs] += dq
        if use_limits:
            np.clip(q[:n_arm_dofs], q_lo_eff, q_hi_eff, out=q[:n_arm_dofs])
    return q, err_norm, iters


def ik_seed_one_step(plant,
                     ee_frame,
                     current_q:  np.ndarray,
                     target_3d:  np.ndarray,
                     plant_ctx,
                     n_arm_dofs: int,
                     damping:    float = 0.001) -> np.ndarray:
    """Single damped-pseudoinverse step from current_q toward target_3d.

    Cheaper than the iterated solver — useful as a warm-start before
    `solve_ik_to_ee_pos`. Does NOT modify plant_ctx.
    """
    world = plant.world_frame()
    ee_pos = plant.CalcPointsPositions(
        plant_ctx, ee_frame, np.zeros(3), world,
    ).flatten()
    J_ee = plant.CalcJacobianTranslationalVelocity(
        plant_ctx, ad.JacobianWrtVariable.kV,
        ee_frame, np.zeros(3),
        world, world,
    )
    J_arm  = J_ee[:, :n_arm_dofs]
    ee_err = target_3d - ee_pos
    JJT    = J_arm @ J_arm.T + damping * np.eye(3)
    dq     = J_arm.T @ np.linalg.solve(JJT, ee_err)

    q_k = current_q.copy()
    q_k[:n_arm_dofs] += dq
    return q_k
