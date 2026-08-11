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

LCP solve: Lemke pivoting via Drake's `MobyLcpSolver` — the reference's own
solver (`c3/core/lcs.cc LCS::Simulate` calls
`MobyLcpSolver::SolveLcpLemkeRegularized`). Solves the per-knot
complementarity problem
    0 ≤ λ_k ⊥ (E · x_k + F · λ_k + H · u_k + c_lcs) ≥ 0
for the current knot's contact forces.

The former projected-Gauss-Seidel path is retained behind `PORT_LCP_PGS=1`
as a falsification lever only. PGS requires F diagonally dominant or PSD;
the Anitescu friction F is neither, so PGS silently returned non-solutions
exactly on the high-λ knots that decide the contact (measured complementarity
residual max|λ·(Fλ+q)| up to 1.3e+04 where a solution gives ~1e-16). Those
bogus λ fed the divergence that made the cost-type-5 sample ranking score
numerical blow-up instead of task progress — see `DIAG_PD_ROLLOUT_SPECTRUM`.
"""
from __future__ import annotations

import os

import numpy as np

from pydrake.solvers import MathematicalProgram, MobyLcpSolver

# One solver instance; Drake's MobyLcpSolver is stateless across Solve calls.
_MOBY = MobyLcpSolver()


def solve_lcp_lemke(F: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, bool]:
    """Exact LCP solve by Lemke pivoting — mirrors the reference's
    `LCS::Simulate`.

    Find λ ≥ 0 s.t. F·λ + q ≥ 0 and λ_i · (F·λ + q)_i = 0.

    Unlike PGS this makes no PSD/diagonal-dominance assumption on F, so it is
    valid for the Anitescu friction blocks.

    Returns (λ, success). On failure λ is zeros and success is False — the
    caller is expected to react rather than score a non-solution.
    """
    n = int(q.shape[0])
    if n == 0:
        return np.zeros(0), True
    prog = MathematicalProgram()
    lam = prog.NewContinuousVariables(n, "lam")
    prog.AddLinearComplementarityConstraint(np.asarray(F, dtype=float),
                                            np.asarray(q, dtype=float), lam)
    res = _MOBY.Solve(prog)
    if not res.is_success():
        return np.zeros(n), False
    return np.asarray(res.GetSolution(lam), dtype=float), True


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


def zero_order_hold_trajectories(x_coarse: np.ndarray, u_coarse: np.ndarray,
                                 rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Mirror reference `TrajectoryEvaluator::ZeroOrderHoldTrajectories`.

    x_coarse (N+1, n_x), u_coarse (N, n_u) → (N·rate+1, n_x), (N·rate, n_u).
    The first N states are each repeated `rate` times and the terminal state is
    appended; controls are repeated `rate` times.
    """
    x_fine = np.concatenate([np.repeat(x_coarse[:-1], rate, axis=0),
                             x_coarse[-1][None, :]], axis=0)
    u_fine = np.repeat(u_coarse, rate, axis=0)
    return x_fine, u_fine


def downsample_trajectories(x_fine: np.ndarray, u_fine: np.ndarray,
                            rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Mirror reference `TrajectoryEvaluator::DownsampleTrajectories`.

    x_fine (Nf+1, n_x), u_fine (Nf, n_u) → (Nf/rate+1, n_x), (Nf/rate, n_u).
    Takes every `rate`-th entry, keeping the terminal state.
    """
    x_coarse = np.concatenate([x_fine[:-1][::rate], x_fine[-1][None, :]],
                              axis=0)
    return x_coarse, u_fine[::rate]


def simulate_pd_control_with_lcs(
    x_plan: np.ndarray,          # (N+1, n_x) planner state trajectory
    u_plan: np.ndarray,          # (N,   n_u) planner control sequence
    A: np.ndarray, B: np.ndarray, D: np.ndarray, d: np.ndarray,
    E: np.ndarray, F: np.ndarray, H: np.ndarray, c_lcs: np.ndarray,
    Kp_ee, Kd_ee,   # float or (3,) per-axis EE PD gains (elementwise ops)
    x0_override: np.ndarray = None,
    lcp_max_iter: int = 50,
    lcp_tol: float = 1e-6,
    lcp_reg: float = 1e-8,
    upsample_rate: int = 1,
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

    `upsample_rate` > 1 selects the reference's two-LCS (coarse plan / fine
    cost) path: the caller passes the FINE LCS (built at dt/rate, N·rate) and
    the coarse plan, which is zero-order-held onto the fine grid, simulated
    there, and downsampled back. This is not an optimization — the PD rollout
    is only inside its stability region at the fine dt. At the coarse planning
    dt, rho(A_cl) is 2.61 (dt=0.05) to 16.41 (dt=0.1) and the rollout diverges,
    making the ranking cost measure numerical blow-up instead of task progress.

    Returns (XX_sim, UU_sim) with shapes (N+1, n_x) and (N, n_u) at the COARSE
    discretization regardless of `upsample_rate`.
    """
    _rate = max(1, int(upsample_rate))
    if _rate > 1:
        x_plan, u_plan = zero_order_hold_trajectories(
            np.asarray(x_plan, dtype=float), np.asarray(u_plan, dtype=float),
            _rate)
    N = int(u_plan.shape[0])
    n_x = int(A.shape[0])
    n_u = int(B.shape[1])
    # DIAG_PD_ROLLOUT_SPECTRUM=1 — closed-loop stability of the contact-free
    # PD rollout. u_sim depends on x_sim as -(Kp·S_p + Kd·S_v)·x_sim, so
    # A_cl = A - B·(Kp·S_p + Kd·S_v). rho(A_cl) > 1 => the ranking cost is
    # scoring a divergent simulation, not the sample's task progress.
    _diag_sp = bool(os.environ.get("DIAG_PD_ROLLOUT_SPECTRUM", ""))
    if _diag_sp:
        _K = np.zeros((n_u, n_x))
        _K[:, 7:10] += Kp_ee * np.eye(3)[:n_u, :]
        _K[:, 16:19] += Kd_ee * np.eye(3)[:n_u, :]
        _rho = float(np.max(np.abs(np.linalg.eigvals(A - B @ _K))))
        # Reference seeds the rollout at x_plan[0] (traj_eval.cc: x[0] =
        # x_plan[0]) => knot-0 PD error is exactly zero. Port seeds at
        # x0_override. Any nonzero seed error is amplified rho^N.
        _x0 = (np.asarray(x0_override, dtype=float).reshape(n_x)
               if x0_override is not None else np.asarray(x_plan[0]))
        _e0 = _x0 - np.asarray(x_plan[0], dtype=float).reshape(n_x)
        print(f"[PD-SPECTRUM] rho(A_cl)={_rho:.4f} "
              f"rho(A)={float(np.max(np.abs(np.linalg.eigvals(A)))):.4f} "
              f"N={N} seed_err_ee_pos={np.linalg.norm(_e0[7:10]):.6f}m "
              f"seed_err_all={np.linalg.norm(_e0):.6f} "
              f"amp=rho^N={_rho ** N:.3e} "
              f"{'DIVERGENT' if _rho > 1.0 else 'stable'}", flush=True)
    # Reference solves each knot's LCP with Lemke pivoting (c3/core/lcs.cc
    # LCS::Simulate). PORT_LCP_PGS=1 restores the old projected-Gauss-Seidel
    # path as a falsification lever.
    _use_pgs = bool(os.environ.get("PORT_LCP_PGS", ""))
    _n_lcp_fail = 0
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
            if _use_pgs:
                lam_k = solve_lcp_pgs(F, q_lcp,
                                      max_iter=lcp_max_iter,
                                      tol=lcp_tol,
                                      reg=lcp_reg)
            else:
                lam_k, _ok = solve_lcp_lemke(F, q_lcp)
                if not _ok:
                    # Never score a non-solution: fall back to PGS and count
                    # it, so a silent bad λ can't masquerade as a valid knot.
                    lam_k = solve_lcp_pgs(F, q_lcp, max_iter=lcp_max_iter,
                                          tol=lcp_tol, reg=lcp_reg)
                    _n_lcp_fail += 1
            XX[k + 1] = A @ x_k + B @ u_sim + D @ lam_k + d
            if _diag_sp and k == 0:
                # Knot 0 has zero seed error and u_pd == 0, so ANY gap between
                # x[1] and x_plan[1] is pure planning-LCS vs cost-LCS model
                # mismatch. Decompose it by term to attribute the mismatch.
                _pl = slice(7, 10)
                print(f"[PD-K0] x1_sim_ee={XX[1][_pl]} "
                      f"x1_plan_ee={np.asarray(x_plan[1])[_pl]} "
                      f"err={np.linalg.norm(XX[1][_pl] - np.asarray(x_plan[1])[_pl]):.5f}m\n"
                      f"[PD-K0]   Ax={(A @ x_k)[_pl]} Bu={(B @ u_sim)[_pl]} "
                      f"Dlam={(D @ lam_k)[_pl]} d={d[_pl]}\n"
                      f"[PD-K0]   n_lam={lam_k.shape[0]} lam={np.array2string(lam_k, precision=3, max_line_width=200)}",
                      flush=True)
            if _diag_sp:
                _res = F @ lam_k + q_lcp
                print(f"[PD-KNOT] k={k} "
                      f"track_err_ee={np.linalg.norm(x_k[7:10] - p_ee_plan):.5f}m "
                      f"|u_pd|={np.linalg.norm(u_pd):.3f} "
                      f"|u_sim|={np.linalg.norm(u_sim):.3f} "
                      f"|lam|={np.linalg.norm(lam_k):.4f} "
                      f"lcp_compl={float(np.abs(lam_k * _res).max()):.3e} "
                      f"lcp_viol={float(np.minimum(_res, 0.0).min()):.3e}",
                      flush=True)
        else:
            # No contacts → LCS reduces to affine dynamics.
            XX[k + 1] = A @ x_k + B @ u_sim + d
    if _n_lcp_fail:
        print(f"[LCP-FAIL] Lemke failed on {_n_lcp_fail}/{N} knots; "
              f"fell back to PGS (those knots' λ are not LCP solutions)",
              flush=True)
    if _rate > 1:
        XX, UU = downsample_trajectories(XX, UU, _rate)
    return XX, UU
