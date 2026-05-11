"""
Linear Complementarity Problem solver — wraps Drake's UnrevisedLemkeSolver.

Used by the C3 path's δ-projection (Aydinoglu 2024 §V-B.3.b, Phase 2):
    δ_λ = argmin (ρ/2) ‖z − δ‖²   s.t.   δ ∈ H_k
where for the C3 LCP-projection method, the projection collapses (for fixed
δ_x, δ_u — they pass through) to a single LCP per timestep:

    q_lcp = E·δ_x + H·δ_u + c_lcs
    LCP:   0 ≤ λ ⊥ F·λ + q_lcp ≥ 0
    return δ_λ = solution.

Drake's UnrevisedLemkeSolver expects MathematicalProgram input; we build one
per call. For tight inner-loop usage this allocation is small (n_lambda is
typically 6·n_c ≈ 6 in single-contact scenarios), but we may revisit if
profiling shows it matters.

Notes
-----
- Lemke is well-tested but can fail on degenerate F. We add eps·I
  regularization (default 1e-8) to recover a feasible solution.
- For Anitescu LCS (F ⪰ 0) we could use the convex projection (Aydinoglu
  eq. 24) instead, but Phase 2 is built around Stewart-Trinkle where
  F is generally not PSD.
"""
import numpy as np
from pydrake.solvers import MathematicalProgram, UnrevisedLemkeSolver


_LEMKE = UnrevisedLemkeSolver()


def solve_lcp(F: np.ndarray, q: np.ndarray, eps_reg: float = 1e-8
              ) -> tuple[np.ndarray, float]:
    """
    Solve LCP(F, q): find λ ≥ 0 with F·λ + q ≥ 0 and λ^T(F·λ + q) = 0.

    Parameters
    ----------
    F       : (n, n)
    q       : (n,)
    eps_reg : Tikhonov regularisation added to F's diagonal before the
              solve. Lemke's pivot can fail on degenerate F; eps·I makes
              F a strictly P-matrix and guarantees a unique solution.

    Returns
    -------
    lam      : (n,)  solution vector (zeros if n == 0)
    residual : max(0, |λ^T (F λ + q)|)  — sanity check, ~0 on success
    """
    n = q.shape[0]
    if n == 0:
        return np.zeros(0), 0.0

    F_reg = F + eps_reg * np.eye(n)

    prog = MathematicalProgram()
    z    = prog.NewContinuousVariables(n, "z")
    prog.AddLinearComplementarityConstraint(F_reg, q, z)
    result = _LEMKE.Solve(prog)
    if not result.is_success():
        # Lemke failed — return zeros and let the caller decide whether
        # to escalate. Typical failures: ray termination, max-pivots
        # exceeded. The C3 path then falls back to whatever ω/δ state
        # already represented.
        return np.zeros(n), float("inf")

    lam = np.asarray(result.GetSolution(z), dtype=float)
    # Numerical floor — Lemke may produce tiny negative components.
    lam = np.maximum(lam, 0.0)
    w   = F_reg @ lam + q
    # Complementarity residual on the un-regularised (F, q): sanity check.
    residual = float(abs(lam @ (F @ lam + q)))
    return lam, residual
