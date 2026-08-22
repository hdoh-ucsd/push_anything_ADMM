"""Equivalence pin for the C3+ projection case-histogram vectorization
(2026-08-21, GPU-ADMM plan Task 2).

The histogram in `_solve_c3plus`'s delta-update used to be a per-slot Python
scalar loop re-deriving cond1/cond2 one element at a time. It was replaced by
a vectorized np.where + np.bincount. It is NOT diagnostic-only: the resulting
`_last_proj_case_N` / `_last_proj_case_T` feed the [CONSENSUS-STEP] log line,
which prints on EVERY solve -- so the counts must be byte-identical, not
merely close.

These tests carry both implementations side by side and assert they agree on
randomized inputs, including the boundary cases (exact ties, zeros, negatives)
where the `>=` / `<` precedence between case 1 and case 2 decides the bucket.
"""
import numpy as np
import pytest


def _hist_scalar(lam_blk, eta_blk, sqrt_ratio, n_lambda, n_lo, n_hi,
                 acc_G, acc_N, acc_T):
    """The ORIGINAL implementation, verbatim in behavior."""
    for _j in range(n_lambda):
        _lo = float(lam_blk[_j])
        _eo = float(eta_blk[_j])
        _c1 = (_eo >= 0.0) and (_eo >= sqrt_ratio * _lo)
        _c2 = (_lo >= 0.0) and (_eo < sqrt_ratio * _lo)
        if _c1:
            _case_idx = 0
        elif _c2:
            _case_idx = 1
        else:
            _case_idx = 2
        if _j < n_lo:
            acc_G[_case_idx] += 1
        elif _j < n_hi:
            acc_N[_case_idx] += 1
        else:
            acc_T[_case_idx] += 1


def _hist_vector(lam_blk, eta_blk, sqrt_ratio, n_lambda, n_lo, n_hi,
                 acc_G, acc_N, acc_T):
    """The REPLACEMENT, verbatim from control/admm_solver.py."""
    _c1_v = (eta_blk >= 0.0) & (eta_blk >= sqrt_ratio * lam_blk)
    _c2_v = (lam_blk >= 0.0) & (eta_blk < sqrt_ratio * lam_blk)
    _case_v = np.where(_c1_v, 0, np.where(_c2_v, 1, 2))
    for _acc, _sl in ((acc_G, slice(0, n_lo)),
                      (acc_N, slice(n_lo, n_hi)),
                      (acc_T, slice(n_hi, None))):
        _bc = np.bincount(_case_v[_sl], minlength=3)
        _acc[0] += int(_bc[0])
        _acc[1] += int(_bc[1])
        _acc[2] += int(_bc[2])


def _run_both(lam, eta, ratio, n_lo, n_hi):
    n = lam.size
    a = ([0, 0, 0], [0, 0, 0], [0, 0, 0])
    b = ([0, 0, 0], [0, 0, 0], [0, 0, 0])
    _hist_scalar(lam, eta, ratio, n, n_lo, n_hi, *a)
    _hist_vector(lam, eta, ratio, n, n_lo, n_hi, *b)
    return a, b


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_random_agrees(seed):
    rng = np.random.default_rng(seed)
    num_normals = 4
    n_lambda = 6 * num_normals
    lam = rng.uniform(-5, 5, n_lambda)
    eta = rng.uniform(-5, 5, n_lambda)
    for ratio in (1.0, 0.5, 2.0, np.sqrt(1000.0 / 4.0)):
        a, b = _run_both(lam, eta, ratio, num_normals, 2 * num_normals)
        assert a == b, f"seed={seed} ratio={ratio}: {a} != {b}"


def test_boundary_values_agree():
    """Exact ties and zeros are where >= / < precedence decides the bucket."""
    num_normals = 2
    n_lambda = 6 * num_normals
    vals = np.array([0.0, 1.0, -1.0, 2.0, -2.0, 0.5,
                     0.0, 1.0, -1.0, 2.0, -2.0, 0.5])
    lam = np.concatenate([vals, vals])[:n_lambda]
    eta = np.concatenate([vals[::-1], vals])[:n_lambda]
    for ratio in (1.0, 0.5, 2.0):
        a, b = _run_both(lam, eta, ratio, num_normals, 2 * num_normals)
        assert a == b, f"ratio={ratio}: {a} != {b}"


def test_exact_tie_eta_equals_ratio_times_lam():
    """eta == ratio*lam exactly -> case 1 must win (>= beats <)."""
    num_normals = 1
    lam = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    eta = 1.5 * lam                       # exact tie at ratio = 1.5
    a, b = _run_both(lam, eta, 1.5, num_normals, 2 * num_normals)
    assert a == b
    totals = [a[0][i] + a[1][i] + a[2][i] for i in range(3)]
    assert totals[0] == lam.size, f"expected all case 1, got {totals}"


def test_accumulates_across_knots():
    """Accumulators are shared across knots; both must accumulate alike."""
    rng = np.random.default_rng(9)
    num_normals = 3
    n_lambda = 6 * num_normals
    acc_a = ([0, 0, 0], [0, 0, 0], [0, 0, 0])
    acc_b = ([0, 0, 0], [0, 0, 0], [0, 0, 0])
    for _ in range(10):                   # 10 knots
        lam = rng.uniform(-3, 3, n_lambda)
        eta = rng.uniform(-3, 3, n_lambda)
        _hist_scalar(lam, eta, 1.0, n_lambda, num_normals, 2 * num_normals, *acc_a)
        _hist_vector(lam, eta, 1.0, n_lambda, num_normals, 2 * num_normals, *acc_b)
    assert acc_a == acc_b
    assert sum(sum(x) for x in acc_a) == 10 * n_lambda


# ---------------------------------------------------------------------
# All-knots form: what control/admm_solver.py actually ships.
# Per-knot numpy measured SLOWER than the scalar loop it replaced (0.72x,
# numpy per-call overhead vs a 24-element loop); hoisting the histogram out
# of the knot loop and doing one pass over (N, n_lambda) is 5.87x.
# See scripts/gpu/hist_speedup.py.
# ---------------------------------------------------------------------
def _hist_all_knots(LAM, ETA, sqrt_ratio, n_lo, n_hi, acc_G, acc_N, acc_T):
    """Verbatim from control/admm_solver.py (post-2026-08-21)."""
    _c1_v = (ETA >= 0.0) & (ETA >= sqrt_ratio * LAM)
    _c2_v = (LAM >= 0.0) & (ETA < sqrt_ratio * LAM)
    _case_v = np.where(_c1_v, 0, np.where(_c2_v, 1, 2))
    for _acc, _sl in ((acc_G, slice(0, n_lo)),
                      (acc_N, slice(n_lo, n_hi)),
                      (acc_T, slice(n_hi, None))):
        _bc = np.bincount(_case_v[:, _sl].ravel(), minlength=3)
        _acc[0] += int(_bc[0])
        _acc[1] += int(_bc[1])
        _acc[2] += int(_bc[2])


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_all_knots_matches_scalar(seed):
    """The shipped all-knots form must equal the original per-knot scalar
    loop accumulated over the same knots."""
    rng = np.random.default_rng(seed)
    num_normals, n_knots = 4, 10
    n_lambda = 6 * num_normals
    LAM = rng.uniform(-5, 5, (n_knots, n_lambda))
    ETA = rng.uniform(-5, 5, (n_knots, n_lambda))
    for ratio in (1.0, 0.5, 2.0, float(np.sqrt(1000.0 / 4.0))):
        a = ([0, 0, 0], [0, 0, 0], [0, 0, 0])
        b = ([0, 0, 0], [0, 0, 0], [0, 0, 0])
        for k in range(n_knots):
            _hist_scalar(LAM[k], ETA[k], ratio, n_lambda,
                         num_normals, 2 * num_normals, *a)
        _hist_all_knots(LAM, ETA, ratio, num_normals, 2 * num_normals, *b)
        assert a == b, f"seed={seed} ratio={ratio}: {a} != {b}"
        assert sum(sum(x) for x in a) == n_knots * n_lambda


def test_all_knots_boundary_and_ties():
    """Zeros, negatives and exact ties, in the all-knots form."""
    num_normals, n_knots = 2, 4
    n_lambda = 6 * num_normals
    base = np.array([0.0, 1.0, -1.0, 2.0, -2.0, 0.5,
                     1.5, -0.5, 0.0, 3.0, -3.0, 1.0])
    LAM = np.stack([np.roll(base, k) for k in range(n_knots)])
    ETA = np.stack([np.roll(base[::-1], k) for k in range(n_knots)])
    for ratio in (1.0, 0.5, 1.5, 2.0):
        a = ([0, 0, 0], [0, 0, 0], [0, 0, 0])
        b = ([0, 0, 0], [0, 0, 0], [0, 0, 0])
        for k in range(n_knots):
            _hist_scalar(LAM[k], ETA[k], ratio, n_lambda,
                         num_normals, 2 * num_normals, *a)
        _hist_all_knots(LAM, ETA, ratio, num_normals, 2 * num_normals, *b)
        assert a == b, f"ratio={ratio}: {a} != {b}"
