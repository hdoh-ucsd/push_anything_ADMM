"""LCS forward simulator — Stage 2 of §9 Option B (reference cost-LCS ranking).

Provides a PGS-based LCP solver and a PD-with-feedforward LCS simulator,
mirroring the reference's `TrajectoryEvaluator::SimulatePDControlWithLCS`
(dairlib sampling_based_c3_controller.cc:571-590 for cost_type=5,
kSimImpedanceObjectCostOnly).

The reference builds a SEPARATE 5-pair cost-LCS (via
`resolve_contacts_to_for_cost=[0,2,3]` for push_t → 2 EE-T + 3 T-GND) and
forward-simulates the plan's u_seq (with PD tracking on the planner's x_seq)
on that cost-LCS. The simulated trajectory `XX_sim` is then scored by a pure
quadratic cost with robot pos/vel/torque entries of Q and R zeroed.

STAGE 2 SIMPLIFICATION: this port uses the SAME LCS for planning and cost
evaluation (single build, currently 1 EE-BOX + 3 T-GND from Option A). The
5-pair cost-LCS delta (top-2 EE-manipuland vs top-1) is left as a follow-up.
The core mechanism — forward simulate the plan, score object errors on the
simulated trajectory — is preserved.

LCP solve: projected Gauss-Seidel (PGS) with Tikhonov regularization (reg=1e-8
per reference `simulate_config.regularized=true, min_exp=-8`). Solves the
per-knot complementarity problem
    0 ≤ λ_k ⊥ (E · x_k + F · λ_k + H · u_k + c_lcs) ≥ 0
for the current knot's contact forces.
"""
from __future__ import annotations

import numpy as np


def solve_lcp_pgs(F: np.ndarray,
                  q: np.ndarray,
                  max_iter: int = 50,
                  tol: float = 1e-6,
                  reg: float = 1e-8) -> np.ndarray:
    """Projected Gauss-Seidel LCP solve.

    Find λ ≥ 0 s.t. F·λ + q ≥ 0 and λ_i · (F·λ + q)_i = 0.

    F : (n, n)   assumed PSD; reg > 0 makes it strictly PD (unique soln).
    q : (n,)
    max_iter, tol : convergence knobs.
    reg : Tikhonov regularizer (matches reference min_exp=-8).
    """
    n = int(q.shape[0])
    if n == 0:
        return np.zeros(0)
    lam = np.zeros(n)
    # Diag entries with regularization (avoid rebuilding F+regI).
    F_diag = np.diag(F).copy() + reg
    for _ in range(int(max_iter)):
        max_change = 0.0
        for i in range(n):
            f_ii = float(F_diag[i])
            if abs(f_ii) < 1e-14:
                continue
            # Residual excluding self: (F·λ + q)_i - F[i,i]·λ_i
            g_i = float(F[i, :] @ lam) - float(F[i, i]) * float(lam[i]) + float(q[i])
            new_lam_i = max(0.0, -g_i / f_ii)
            change = abs(new_lam_i - float(lam[i]))
            if change > max_change:
                max_change = change
            lam[i] = new_lam_i
        if max_change < tol:
            break
    return lam


def simulate_pd_control_with_lcs(
    x_plan: np.ndarray,          # (N+1, n_x) planner state trajectory
    u_plan: np.ndarray,          # (N,   n_u) planner control sequence
    A: np.ndarray, B: np.ndarray, D: np.ndarray, d: np.ndarray,
    E: np.ndarray, F: np.ndarray, H: np.ndarray, c_lcs: np.ndarray,
    Kp_ee: float, Kd_ee: float,
    x0_override: np.ndarray = None,
    lcp_max_iter: int = 50,
    lcp_tol: float = 1e-6,
    lcp_reg: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate PD-with-feedforward on an LCS. Mirrors reference
    TrajectoryEvaluator::SimulatePDControlWithLCS with LTI matrices.

    At each knot k:
        u_sim_k = u_plan_k + Kp · (p_ee_plan_k - p_ee_sim_k)
                           + Kd · (v_ee_plan_k - v_ee_sim_k)
        Solve LCP for λ_k given (x_sim_k, u_sim_k)
        x_sim_{k+1} = A · x_sim_k + B · u_sim_k + D · λ_k + d

    Port's EE-space state layout: x = [box_q (7), p_ee (3), box_v (6), v_ee (3)].
    PD applies on p_ee (7:10) and v_ee (16:19).

    Returns (XX_sim, UU_sim) with shapes (N+1, n_x) and (N, n_u).
    """
    N = int(u_plan.shape[0])
    n_x = int(A.shape[0])
    n_u = int(B.shape[1])
    XX = np.zeros((N + 1, n_x))
    UU = np.zeros((N, n_u))
    if x0_override is not None:
        XX[0] = np.asarray(x0_override, dtype=float).reshape(n_x)
    else:
        XX[0] = np.asarray(x_plan[0], dtype=float).reshape(n_x)
    for k in range(N):
        x_k = XX[k]
        p_ee_plan = x_plan[k, 7:10]
        v_ee_plan = x_plan[k, 16:19]
        p_ee_sim  = x_k[7:10]
        v_ee_sim  = x_k[16:19]
        u_pd  = Kp_ee * (p_ee_plan - p_ee_sim) + Kd_ee * (v_ee_plan - v_ee_sim)
        u_sim = np.asarray(u_plan[k], dtype=float).reshape(n_u) + u_pd
        UU[k] = u_sim
        # LCP solve for λ_k
        if F is not None and F.shape[0] > 0:
            q_lcp = E @ x_k + H @ u_sim + c_lcs
            lam_k = solve_lcp_pgs(F, q_lcp,
                                  max_iter=lcp_max_iter,
                                  tol=lcp_tol,
                                  reg=lcp_reg)
            XX[k + 1] = A @ x_k + B @ u_sim + D @ lam_k + d
        else:
            # No contacts → LCS reduces to affine dynamics.
            XX[k + 1] = A @ x_k + B @ u_sim + d
    return XX, UU
