"""control.gpu.projection must agree EXACTLY with the numpy C3+ projection.

Both implementations select existing values via a where(), so this is exact
equality, not a tolerance comparison. Skips cleanly without CUDA.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="no CUDA device")

from control.gpu.legacy_torch.projection import project_C3Plus_batch  # noqa: E402


def _numpy_ref(lam, eta, u_lambda, u_eta):
    """Verbatim C3Solver._project_C3Plus (admm_solver.py:1035)."""
    sqrt_ratio = float(np.sqrt(u_lambda / u_eta))
    cond1 = (eta >= 0.0) & (eta >= sqrt_ratio * lam)
    cond2 = (lam >= 0.0) & (eta < sqrt_ratio * lam)
    return np.where(cond2, lam, 0.0), np.where(cond1, eta, 0.0)


WEIGHTS = [(1.0, 1.0), (1000.0, 4.0), (4.0, 90.0), (0.18, 1.0)]


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("ul,ue", WEIGHTS)
def test_matches_numpy_exactly(seed, ul, ue):
    rng = np.random.default_rng(seed)
    lam = rng.uniform(-5, 5, (6, 10, 20))      # (batch, knots, n_lambda)
    eta = rng.uniform(-5, 5, (6, 10, 20))
    d_l, d_e = project_C3Plus_batch(
        torch.as_tensor(lam, device="cuda"),
        torch.as_tensor(eta, device="cuda"), ul, ue)
    r_l, r_e = _numpy_ref(lam, eta, ul, ue)
    assert np.array_equal(d_l.cpu().numpy(), r_l)
    assert np.array_equal(d_e.cpu().numpy(), r_e)


def test_boundary_grid_exact():
    """Zeros, exact ties and sign flips across a dense grid."""
    vals = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    lam = np.repeat(vals, vals.size).reshape(1, -1)
    eta = np.tile(vals, vals.size).reshape(1, -1)
    for ul, ue in WEIGHTS + [(2.25, 1.0)]:      # 2.25/1 -> ratio exactly 1.5
        d_l, d_e = project_C3Plus_batch(
            torch.as_tensor(lam, device="cuda"),
            torch.as_tensor(eta, device="cuda"), ul, ue)
        r_l, r_e = _numpy_ref(lam, eta, ul, ue)
        assert np.array_equal(d_l.cpu().numpy(), r_l), (ul, ue)
        assert np.array_equal(d_e.cpu().numpy(), r_e), (ul, ue)


def test_output_invariants():
    """Post-projection: complementarity and non-negativity, as on CPU."""
    rng = np.random.default_rng(7)
    lam = torch.as_tensor(rng.uniform(-9, 9, (4, 200)), device="cuda")
    eta = torch.as_tensor(rng.uniform(-9, 9, (4, 200)), device="cuda")
    d_l, d_e = project_C3Plus_batch(lam, eta, 1.0, 1.0)
    assert torch.all(d_l * d_e == 0)
    assert torch.all(d_l >= 0) and torch.all(d_e >= 0)


def test_batch_independence():
    """Each batch element must be untouched by its neighbours."""
    rng = np.random.default_rng(3)
    lam = rng.uniform(-4, 4, (5, 20))
    eta = rng.uniform(-4, 4, (5, 20))
    full_l, full_e = project_C3Plus_batch(
        torch.as_tensor(lam, device="cuda"),
        torch.as_tensor(eta, device="cuda"), 1000.0, 4.0)
    for b in range(5):
        one_l, one_e = project_C3Plus_batch(
            torch.as_tensor(lam[b:b + 1], device="cuda"),
            torch.as_tensor(eta[b:b + 1], device="cuda"), 1000.0, 4.0)
        assert torch.equal(full_l[b], one_l[0])
        assert torch.equal(full_e[b], one_e[0])


def test_matches_solver_helper_on_corpus_weights():
    """Cross-check against the shipped module helper, not just a local copy."""
    try:
        from control.admm_solver import project_C3Plus_eq12
    except Exception:
        pytest.skip("control.admm_solver requires pydrake")
    rng = np.random.default_rng(11)
    lam = rng.uniform(-3, 3, (2, 10, 20))
    eta = rng.uniform(-3, 3, (2, 10, 20))
    for ul, ue in WEIGHTS:
        d_l, d_e = project_C3Plus_batch(
            torch.as_tensor(lam, device="cuda"),
            torch.as_tensor(eta, device="cuda"), ul, ue)
        r_l, r_e = project_C3Plus_eq12(lam, eta, ul, ue)
        assert np.array_equal(d_l.cpu().numpy(), r_l)
        assert np.array_equal(d_e.cpu().numpy(), r_e)
