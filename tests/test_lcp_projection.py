"""
Phase-2 acceptance tests for the LCP projection (Aydinoglu §V-B.3.b).

Verifies the LCP wrapper in control/lcp_solver.py against synthetic
(F, q) inputs covering interior, boundary, and mixed solutions.
"""
import os
import sys
import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

pytest.importorskip("pydrake", reason="Drake required for UnrevisedLemkeSolver")

from control.lcp_solver import solve_lcp


def _check_lcp_solution(F, q, lam, eps=1e-7):
    """Return (lam_ok, w_ok, comp_ok, w)."""
    w = F @ lam + q
    return (
        bool(np.all(lam >= -eps)),
        bool(np.all(w   >= -eps)),
        bool(abs(float(lam @ w)) < eps * max(1.0, np.linalg.norm(lam) * np.linalg.norm(w))),
        w,
    )


def test_lcp_zero_dimensional():
    lam, res = solve_lcp(np.zeros((0, 0)), np.zeros(0))
    assert lam.shape == (0,)
    assert res == 0.0


def test_lcp_interior_solution():
    """LCP(F, q) where the optimum has λ > 0, w = 0 — interior of the
    nonnegative orthant."""
    F = np.array([[2.0, 1.0], [1.0, 2.0]])
    q = np.array([-1.0, -1.0])
    lam, res = solve_lcp(F, q)
    print(f"lam={lam}, residual={res:.3e}")
    assert lam.shape == (2,)
    lam_ok, w_ok, comp_ok, w = _check_lcp_solution(F, q, lam)
    assert lam_ok and w_ok and comp_ok, f"lam={lam}, w={w}"
    # Known closed-form: λ = (1/3, 1/3)
    assert np.allclose(lam, [1/3, 1/3], atol=1e-6), f"got {lam}"


def test_lcp_boundary_solution_lambda_zero():
    """LCP(F, q) where q ≥ 0 — solution is λ = 0, w = q."""
    F = np.array([[1.0, 0.0], [0.0, 1.0]])
    q = np.array([1.0, 2.0])
    lam, res = solve_lcp(F, q)
    print(f"lam={lam}, residual={res:.3e}")
    assert np.allclose(lam, 0.0, atol=1e-9)


def test_lcp_mixed_solution():
    """LCP(F, q) with one λ active and one inactive."""
    F = np.array([[2.0, 0.0], [0.0, 1.0]])
    q = np.array([-2.0, 1.0])
    lam, res = solve_lcp(F, q)
    print(f"lam={lam}, residual={res:.3e}")
    lam_ok, w_ok, comp_ok, w = _check_lcp_solution(F, q, lam)
    assert lam_ok and w_ok and comp_ok, f"lam={lam}, w={w}"
    # Component 0: λ = 1, w = 0 (active)
    # Component 1: λ = 0, w = 1 (inactive)
    assert np.allclose(lam, [1.0, 0.0], atol=1e-6), f"got {lam}"


def test_lcp_random_psd_F():
    """Random LCPs with F = G^T G (PSD): should always have a solution."""
    rng = np.random.default_rng(0)
    for trial in range(8):
        n = rng.integers(2, 8)
        G = rng.normal(size=(n, n))
        F = G.T @ G + 1e-3 * np.eye(n)   # strictly PSD
        q = rng.normal(size=n)
        lam, res = solve_lcp(F, q)
        lam_ok, w_ok, comp_ok, w = _check_lcp_solution(F, q, lam, eps=1e-6)
        assert lam_ok and w_ok and comp_ok, (
            f"trial {trial} n={n}: λ={lam}, w={w}, residual={res:.3e}"
        )


def test_lcp_friction_cone_pyramid_block():
    """Synthetic Stewart-Trinkle-shaped LCP for one contact.
    Decision λ = [γ, λ_n, λ_t1, λ_t2, λ_t3, λ_t4]  (n_lambda = 6).

    Block structure (Aydinoglu eq. 9, with synthetic dynamics coupling):
      γ row:   F·λ + q ≥ 0  →  μ λ_n − Σ λ_ti ≥ 0   (no q term — γ has c=0)
      λ_n row: dynamics-coupled, q < 0 forces contact active
      λ_t row: depends on γ via E_t^T (forces λ_t = 0 when γ=0, else free)
    """
    mu = 0.4
    # Synthetic dynamics-coupling block (small dt·J_n·M^-1·J_c^T): identity-ish
    coupling = 0.05 * np.eye(5)   # for [λ_n; λ_t1..t4] vs same
    F = np.zeros((6, 6))
    # γ row: 0 + μ·λ_n − Σ λ_ti
    F[0, 1] = mu
    F[0, 2:6] = -1.0
    # λ_n row: 0 (γ no contribution) + coupling·[λ_n; λ_t]
    F[1, 1:6] = coupling[0, :]
    # λ_t rows: E_t^T γ + coupling
    for j in range(4):
        F[2 + j, 0]    = 1.0      # E_t^T column on γ
        F[2 + j, 1:6]  = coupling[1 + j, :]
    q = np.array([0.0, -0.5, 0.0, 0.0, 0.0, 0.0])
    lam, res = solve_lcp(F, q, eps_reg=1e-6)
    lam_ok, w_ok, comp_ok, w = _check_lcp_solution(F, q, lam, eps=1e-4)
    print(f"lam={lam}, w={w}, residual={res:.3e}")
    assert lam_ok and w_ok, f"infeasible: lam={lam}, w={w}"
    # λ_n should be positive (driven by q[λ_n] < 0)
    assert lam[1] > 0.0, f"expected λ_n > 0, got {lam[1]}"
    # Friction-cone constraint: μ λ_n ≥ Σ λ_ti
    assert mu * lam[1] >= float(lam[2:6].sum()) - 1e-6, (
        f"friction cone violated: μ λ_n={mu * lam[1]:.4f} < "
        f"Σ λ_t={float(lam[2:6].sum()):.4f}"
    )
