"""Unit tests for control.sampling_c3.sampling."""
import numpy as np
import pytest

from control.sampling_c3.params import SamplingParams, SamplingStrategy
from control.sampling_c3.sampling import (
    generate_samples,
    is_in_workspace,
)


def _params(**over) -> SamplingParams:
    s = SamplingParams()
    for k, v in over.items():
        setattr(s, k, v)
    return s


# ---------------------------------------------------------------------------
# Workspace filter
# ---------------------------------------------------------------------------

def test_is_in_workspace_inside():
    s = _params()
    assert is_in_workspace(np.array([0.0, -0.3, 0.05]), s) is True


def test_is_in_workspace_x_out_of_bounds():
    s = _params()
    assert is_in_workspace(np.array([1.0, -0.3, 0.05]), s) is False


def test_is_in_workspace_y_out_of_bounds():
    s = _params()
    assert is_in_workspace(np.array([0.0,  0.5, 0.05]), s) is False


def test_is_in_workspace_z_too_low():
    s = _params()
    assert is_in_workspace(np.array([0.0, -0.3, 0.00]), s) is False


def test_is_in_workspace_z_too_high():
    s = _params()
    assert is_in_workspace(np.array([0.0, -0.3, 1.00]), s) is False


def test_is_in_workspace_boundary_inclusive():
    s = _params()
    assert is_in_workspace(np.array([s.workspace_xy_min[0], s.workspace_xy_min[1], s.workspace_z_min]), s) is True
    assert is_in_workspace(np.array([s.workspace_xy_max[0], s.workspace_xy_max[1], s.workspace_z_max]), s) is True


def test_is_in_workspace_wrong_shape_raises():
    with pytest.raises(ValueError):
        is_in_workspace(np.array([0.0, 0.0]), _params())


# ---------------------------------------------------------------------------
# kRandomOnCircle
# ---------------------------------------------------------------------------

def test_random_on_circle_returns_requested_count():
    s = _params(filter_samples_for_safety=False)
    rng = np.random.default_rng(seed=0)
    out = generate_samples(SamplingStrategy.kRandomOnCircle, n_samples=4,
                           obj_xy=np.array([0.0, 0.0]), params=s, rng=rng)
    assert len(out) == 4


def test_random_on_circle_all_on_circle():
    """Every sample's xy distance from obj_xy must equal sampling_radius."""
    s = _params(filter_samples_for_safety=False, sampling_radius=0.18,
                sampling_height=0.05)
    rng = np.random.default_rng(seed=0)
    obj = np.array([0.10, 0.20])
    out = generate_samples(SamplingStrategy.kRandomOnCircle, n_samples=8,
                           obj_xy=obj, params=s, rng=rng)
    for p in out:
        d = float(np.linalg.norm(p[:2] - obj))
        assert d == pytest.approx(0.18, abs=1e-9)
        assert p[2] == pytest.approx(0.05, abs=1e-9)


def test_random_on_circle_proxy_seeded_first_when_g_hat_given():
    s = _params(filter_samples_for_safety=False, sampling_radius=0.18,
                sampling_height=0.05)
    rng = np.random.default_rng(seed=0)
    obj   = np.array([0.0, 0.0])
    g_hat = np.array([1.0, 0.0])   # push east
    out = generate_samples(SamplingStrategy.kRandomOnCircle, n_samples=4,
                           obj_xy=obj, params=s, rng=rng, g_hat=g_hat)
    # proxy = obj - r*g_hat = (-0.18, 0, 0.05)
    assert out[0][0] == pytest.approx(-0.18, abs=1e-9)
    assert out[0][1] == pytest.approx( 0.00, abs=1e-9)


def test_random_on_circle_deterministic_with_seeded_rng():
    s = _params(filter_samples_for_safety=False)
    obj = np.array([0.0, 0.0])
    out_a = generate_samples(SamplingStrategy.kRandomOnCircle, n_samples=4,
                             obj_xy=obj, params=s,
                             rng=np.random.default_rng(seed=42))
    out_b = generate_samples(SamplingStrategy.kRandomOnCircle, n_samples=4,
                             obj_xy=obj, params=s,
                             rng=np.random.default_rng(seed=42))
    for a, b in zip(out_a, out_b):
        np.testing.assert_array_equal(a, b)


def test_zero_samples_returns_empty():
    out = generate_samples(SamplingStrategy.kRandomOnCircle, n_samples=0,
                           obj_xy=np.array([0.0, 0.0]),
                           params=_params())
    assert out == []


# ---------------------------------------------------------------------------
# Workspace filter integration
# ---------------------------------------------------------------------------

def test_filter_drops_out_of_workspace_samples():
    """Setting an absurdly large radius must cause workspace filter to drop most samples."""
    s = _params(filter_samples_for_safety=True, sampling_radius=10.0,
                sampling_height=0.05)
    rng = np.random.default_rng(seed=0)
    obj = np.array([0.0, 0.0])
    out = generate_samples(SamplingStrategy.kRandomOnCircle, n_samples=8,
                           obj_xy=obj, params=s, rng=rng)
    # At radius 10 m, no samples should land inside the [-0.5,0.5]×[-0.7,0.0]
    # workspace
    assert len(out) == 0


def test_filter_off_keeps_all():
    s = _params(filter_samples_for_safety=False, sampling_radius=10.0)
    rng = np.random.default_rng(seed=0)
    out = generate_samples(SamplingStrategy.kRandomOnCircle, n_samples=8,
                           obj_xy=np.array([0.0, 0.0]),
                           params=s, rng=rng)
    assert len(out) == 8


# ---------------------------------------------------------------------------
# kRadiallySymmetric
# ---------------------------------------------------------------------------

def test_radially_symmetric_evenly_spaced():
    s = _params(filter_samples_for_safety=False, sampling_radius=0.18)
    out = generate_samples(SamplingStrategy.kRadiallySymmetric, n_samples=4,
                           obj_xy=np.array([0.0, 0.0]),
                           params=s, g_hat=np.array([1.0, 0.0]))
    assert len(out) == 4
    # Pairwise xy angles should be 90° apart
    angles = [float(np.arctan2(p[1], p[0])) for p in out]
    diffs  = sorted([(angles[i] - angles[0]) % (2 * np.pi) for i in range(4)])
    expected = [0.0, np.pi/2, np.pi, 3*np.pi/2]
    for d, e in zip(diffs, expected):
        assert d == pytest.approx(e, abs=1e-9)


# ---------------------------------------------------------------------------
# Unsupported strategies
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strat", [
    SamplingStrategy.kRandomOnSphere,
    SamplingStrategy.kRandomOnPerimeter,
    SamplingStrategy.kRandomOnShell,
    SamplingStrategy.kMeshNormal,
])
def test_unimplemented_strategy_raises(strat):
    with pytest.raises(NotImplementedError):
        generate_samples(strat, n_samples=4,
                         obj_xy=np.array([0.0, 0.0]),
                         params=_params())
