import numpy as np
import pytest

# Inline the pure-numpy projection so the test runs without Drake
def project_lorentz(lam_n: float, lam_t: np.ndarray, mu: float):
    """Mirror of C3Solver._project_single_contact for isolated testing."""
    b_norm = float(np.linalg.norm(lam_t))
    if b_norm <= mu * lam_n + 1e-12:
        return float(lam_n), lam_t.copy()
    if mu * b_norm <= -lam_n + 1e-12:
        return 0.0, np.zeros_like(lam_t)
    s     = (lam_n + mu * b_norm) / (1.0 + mu * mu)
    t_new = (mu * s / b_norm) * lam_t
    return s, t_new


MU = 0.4

def test_inside_cone_unchanged():
    """Point already in the cone should be returned unchanged."""
    a, b = 1.0, np.array([0.1, 0.1])
    assert np.linalg.norm(b) < MU * a  # sanity: input is inside
    a_new, b_new = project_lorentz(a, b, MU)
    assert abs(a_new - a) < 1e-9
    assert np.allclose(b_new, b)


def test_on_cone_boundary_unchanged():
    """Point exactly on the cone surface should be unchanged."""
    a = 1.0
    b = np.array([MU * a, 0.0])
    a_new, b_new = project_lorentz(a, b, MU)
    assert abs(a_new - a) < 1e-9
    assert np.allclose(b_new, b, atol=1e-9)


def test_outside_cone_projects_to_surface():
    """Point outside cone must land on ||b_new|| = μ·a_new exactly."""
    a, b = 0.1, np.array([0.3, 0.0])
    assert np.linalg.norm(b) > MU * a  # sanity: input is outside
    a_new, b_new = project_lorentz(a, b, MU)
    b_norm = np.linalg.norm(b_new)
    expected = MU * a_new
    ratio = b_norm / expected if expected > 0 else float('inf')
    assert abs(ratio - 1.0) < 1e-6, (
        f"Projection not on cone surface. "
        f"||b_new||={b_norm:.6f}, μ·a_new={expected:.6f}, ratio={ratio:.4f}"
    )


def test_polar_cone_projects_to_origin():
    """Deep-polar point should project to the apex (0, 0)."""
    a, b = -1.0, np.array([0.1, 0.0])
    # Polar cone condition: μ·||b|| ≤ -a → 0.04 ≤ 1 ✓
    a_new, b_new = project_lorentz(a, b, MU)
    assert abs(a_new) < 1e-9
    assert np.linalg.norm(b_new) < 1e-9


def test_scan_ratios():
    """Scan a range of inputs outside the cone; all should produce ratio 1.0."""
    ratios = []
    rng = np.random.default_rng(0)
    for _ in range(50):
        a = rng.uniform(0.01, 1.0)
        b_dir = rng.normal(size=2)
        b_dir /= np.linalg.norm(b_dir)
        b_mag = rng.uniform(2 * MU * a, 10 * MU * a)  # well outside cone
        b = b_mag * b_dir
        a_new, b_new = project_lorentz(a, b, MU)
        b_norm = np.linalg.norm(b_new)
        if a_new > 1e-9:
            ratios.append(b_norm / (MU * a_new))
    ratios = np.array(ratios)
    assert abs(ratios.mean() - 1.0) < 1e-6, (
        f"Mean ratio = {ratios.mean():.4f}, std = {ratios.std():.4f}. "
        f"If ≈1.3, there is a (1+μ) vs (1+μ²) bug. "
        f"If ≈2.5, the μ·s scaling on b_new is missing."
    )


def test_four_component_tangent():
    """If the projection accepts 4-vectors, verify it still respects the cone.

    This test reveals whether project_lorentz handles polyhedral (4D) tangent
    vectors sensibly — the key diagnostic for the J_t dimensionality question.
    """
    a, b = 0.1, np.array([0.3, 0.0, 0.0, 0.0])
    try:
        a_new, b_new = project_lorentz(a, b, MU)
        b_norm = np.linalg.norm(b_new)
        assert b_norm <= MU * a_new + 1e-6, (
            f"4D input: ||b_new||={b_norm:.4f} > μ·a_new={MU*a_new:.4f}. "
            f"Lorentz projection is not correct for a polyhedral tangent basis."
        )
    except (ValueError, AssertionError, IndexError) as e:
        pytest.fail(f"project_lorentz raised {type(e).__name__} on 4D input: {e}")
