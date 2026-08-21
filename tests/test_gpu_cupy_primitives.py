"""Phase 2F: CPU-vs-GPU parity for every batched CuPy primitive.

float64 throughout. The projection is compared with EXACT equality because
it selects existing values rather than computing new ones; the reductions
use a tight tolerance because float summation order legitimately differs
between a serial CPU norm and a parallel GPU one.

Inputs are both golden (from the Phase 2C fixtures) and randomized,
including the boundary cases where the projection's >= / < precedence
decides the branch.
"""
import glob
import os

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
pytestmark = pytest.mark.skipif(
    not cp.cuda.is_available(), reason="no CUDA device")

from control.admm_solver import project_C3Plus_eq12          # noqa: E402
from control.gpu.cupy_primitives import (                    # noqa: E402
    candidate_argmin, candidate_effort_cost, dual_update,
    project_C3Plus_batch, residuals, slot_view, to_device, to_host)

FIX = "audit_output/gpu_golden/independent_batch/fixtures.npz"
needs_fixtures = pytest.mark.skipif(
    not os.path.exists(FIX),
    reason="run scripts/gpu/make_golden_fixtures.py first")

WEIGHTS = [(1.0, 1.0), (1000.0, 4.0), (4.0, 90.0), (0.18, 1.0)]


# --------------------------------------------------------------- projection

@pytest.mark.parametrize("ul,ue", WEIGHTS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_projection_exact_vs_cpu(seed, ul, ue):
    rng = np.random.default_rng(seed)
    lam = rng.uniform(-5, 5, (6, 10, 20))       # B x N x n_lambda
    eta = rng.uniform(-5, 5, (6, 10, 20))
    g_l, g_e = project_C3Plus_batch(to_device(lam), to_device(eta), ul, ue)
    c_l, c_e = project_C3Plus_eq12(lam, eta, ul, ue)
    assert np.array_equal(to_host(g_l), c_l)
    assert np.array_equal(to_host(g_e), c_e)


def test_projection_boundary_grid_exact():
    """Zeros, exact ties and sign flips -- where >= / < decides the branch."""
    vals = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    lam = np.repeat(vals, vals.size).reshape(1, -1)
    eta = np.tile(vals, vals.size).reshape(1, -1)
    for ul, ue in WEIGHTS + [(2.25, 1.0)]:      # 2.25/1 -> ratio exactly 1.5
        g_l, g_e = project_C3Plus_batch(to_device(lam), to_device(eta),
                                        ul, ue)
        c_l, c_e = project_C3Plus_eq12(lam, eta, ul, ue)
        assert np.array_equal(to_host(g_l), c_l), (ul, ue)
        assert np.array_equal(to_host(g_e), c_e), (ul, ue)


@needs_fixtures
def test_projection_matches_golden_fixture():
    f = np.load(FIX)
    g_l, g_e = project_C3Plus_batch(to_device(f["proj_lam_in"]),
                                    to_device(f["proj_eta_in"]), 1.0, 1.0)
    assert np.array_equal(to_host(g_l), f["proj_lam_out"])
    assert np.array_equal(to_host(g_e), f["proj_eta_out"])


def test_projection_invariants():
    rng = np.random.default_rng(7)
    lam = to_device(rng.uniform(-9, 9, (4, 200)))
    eta = to_device(rng.uniform(-9, 9, (4, 200)))
    d_l, d_e = project_C3Plus_batch(lam, eta, 1.0, 1.0)
    assert bool((d_l * d_e == 0).all())          # complementarity
    assert bool((d_l >= 0).all()) and bool((d_e >= 0).all())


def test_projection_batch_independence():
    """Batching must not couple candidates."""
    rng = np.random.default_rng(3)
    lam = rng.uniform(-4, 4, (5, 20))
    eta = rng.uniform(-4, 4, (5, 20))
    full_l, full_e = project_C3Plus_batch(to_device(lam), to_device(eta),
                                          1000.0, 4.0)
    for b in range(5):
        one_l, one_e = project_C3Plus_batch(to_device(lam[b:b + 1]),
                                            to_device(eta[b:b + 1]),
                                            1000.0, 4.0)
        assert np.array_equal(to_host(full_l)[b], to_host(one_l)[0])
        assert np.array_equal(to_host(full_e)[b], to_host(one_e)[0])


# -------------------------------------------------------------- dual update

def test_dual_update_exact():
    rng = np.random.default_rng(11)
    om, z, dl = (rng.standard_normal((6, 639)) for _ in range(3))
    got = to_host(dual_update(to_device(om), to_device(z), to_device(dl)))
    assert np.allclose(got, om + (z - dl), rtol=0, atol=1e-15)


# ---------------------------------------------------------------- residuals

def test_slot_view_matches_cpu_concatenation():
    """The residual is defined over lambda/eta slots ONLY, in a specific
    per-knot order. A gather that permutes them would still 'look right'
    in a norm, so compare the gathered vector elementwise."""
    N, TOT, SL, SE, nl, n_x, n_u = 10, 62, 19, 42, 20, 19, 3
    total = N * TOT + n_x
    rng = np.random.default_rng(5)
    v = rng.standard_normal(total)
    cpu = np.concatenate([
        np.concatenate([v[i * TOT + SL: i * TOT + SL + nl],
                        v[i * TOT + SE: i * TOT + SE + nl]])
        for i in range(N)])
    gpu = to_host(slot_view(to_device(v), N, TOT, SL, SE, nl)).reshape(-1)
    assert np.array_equal(gpu, cpu)


def test_residuals_match_cpu():
    N, TOT, SL, SE, nl, n_x = 10, 62, 19, 42, 20, 19
    total = N * TOT + n_x
    rng = np.random.default_rng(9)
    z, dl, dp = (rng.standard_normal((6, total)) for _ in range(3))
    rho = 3.0
    pr, dr = residuals(to_device(z), to_device(dl), to_device(dp),
                       rho, N, TOT, SL, SE, nl)

    def cpu_vec(v):
        return np.concatenate([
            np.concatenate([v[i * TOT + SL: i * TOT + SL + nl],
                            v[i * TOT + SE: i * TOT + SE + nl]])
            for i in range(N)])
    for b in range(6):
        a, bb, cc = cpu_vec(z[b]), cpu_vec(dl[b]), cpu_vec(dp[b])
        assert np.isclose(to_host(pr)[b], np.linalg.norm(a - bb),
                          rtol=1e-13, atol=0)
        assert np.isclose(to_host(dr)[b], rho * np.linalg.norm(bb - cc),
                          rtol=1e-13, atol=0)


# ------------------------------------------------------- cost + selection

@needs_fixtures
def test_candidate_cost_and_argmin_match_golden():
    f = np.load(FIX)
    B = int(f["candidate_costs"].size)
    u = np.stack([f[f"u_seq_{k}"] for k in range(B)])
    costs = to_host(candidate_effort_cost(to_device(u)))
    assert np.allclose(costs, f["candidate_costs"], rtol=1e-13, atol=0)
    assert candidate_argmin(to_device(costs)) == int(f["best_candidate"])


def test_argmin_never_selects_non_finite():
    costs = to_device(np.array([np.nan, 5.0, np.inf, 2.0]))
    assert candidate_argmin(costs) == 3
