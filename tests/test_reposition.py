"""Unit tests for the pure-numpy trajectory generator in
control.sampling_c3.reposition.

The PD tracker is exercised end-to-end by the WEST/EAST validation runs;
unit-testing it would require firing up Drake which we avoid here.
"""
import numpy as np
import pytest

from control.sampling_c3.reposition import (
    is_at_target,
    next_waypoint,
)


# ---------------------------------------------------------------------------
# Phase 1: lift
# ---------------------------------------------------------------------------

def test_phase1_lift_partial_step():
    """When p_now is below z_safe, only z changes; xy stays put."""
    p_now    = np.array([0.10, -0.30, 0.05])
    p_target = np.array([0.20,  0.20, 0.05])
    out = next_waypoint(p_now, p_target, z_safe=0.20, ds=0.005)
    assert out[0] == pytest.approx(0.10, abs=1e-9)
    assert out[1] == pytest.approx(-0.30, abs=1e-9)
    assert out[2] == pytest.approx(0.055, abs=1e-9)


def test_phase1_lift_clamps_at_z_safe():
    """A large step must not overshoot z_safe."""
    p_now    = np.array([0.10, -0.30, 0.05])
    p_target = np.array([0.20,  0.20, 0.05])
    out = next_waypoint(p_now, p_target, z_safe=0.20, ds=10.0)
    assert out[2] == pytest.approx(0.20, abs=1e-9)
    # xy still untouched in phase 1
    assert out[0] == 0.10 and out[1] == -0.30


# ---------------------------------------------------------------------------
# Phase 2: traverse
# ---------------------------------------------------------------------------

def test_phase2_traverse_xy_only():
    """At z_safe, only xy moves; z holds at z_safe."""
    p_now    = np.array([0.00, -0.30, 0.20])
    p_target = np.array([0.10,  0.20, 0.05])
    out = next_waypoint(p_now, p_target, z_safe=0.20, ds=0.05)
    assert out[2] == pytest.approx(0.20, abs=1e-6)
    # Direction should be from (0,-0.30) to above (0.10, 0.20)
    direction = np.array([0.10, 0.50]) / np.linalg.norm([0.10, 0.50])
    expected_xy = p_now[:2] + 0.05 * direction
    assert out[0] == pytest.approx(expected_xy[0], abs=1e-9)
    assert out[1] == pytest.approx(expected_xy[1], abs=1e-9)


def test_phase2_traverse_clamps_at_above_target():
    """A step larger than the remaining xy distance must clamp at the
    above-target xy, not overshoot."""
    p_now    = np.array([0.00, 0.0, 0.20])
    p_target = np.array([0.05, 0.0, 0.05])
    out = next_waypoint(p_now, p_target, z_safe=0.20, ds=10.0)
    assert out[0] == pytest.approx(0.05, abs=1e-9)
    assert out[1] == pytest.approx(0.00, abs=1e-9)
    assert out[2] == pytest.approx(0.20, abs=1e-9)


# ---------------------------------------------------------------------------
# Phase 3: descend
# ---------------------------------------------------------------------------

def test_phase3_descend_partial_step():
    p_now    = np.array([0.10, 0.20, 0.20])
    p_target = np.array([0.10, 0.20, 0.05])
    out = next_waypoint(p_now, p_target, z_safe=0.20, ds=0.03)
    assert out[0] == pytest.approx(0.10, abs=1e-9)
    assert out[1] == pytest.approx(0.20, abs=1e-9)
    assert out[2] == pytest.approx(0.17, abs=1e-9)


def test_phase3_descend_clamps_at_target_z():
    p_now    = np.array([0.10, 0.20, 0.20])
    p_target = np.array([0.10, 0.20, 0.05])
    out = next_waypoint(p_now, p_target, z_safe=0.20, ds=10.0)
    assert out[2] == pytest.approx(0.05, abs=1e-9)


def test_already_at_target_returns_target():
    p_target = np.array([0.10, 0.20, 0.05])
    out = next_waypoint(p_target.copy(), p_target, z_safe=0.20, ds=0.01)
    np.testing.assert_array_equal(out, p_target)


# ---------------------------------------------------------------------------
# End-to-end PWL invariant: the path never enters a box bounding box
# ---------------------------------------------------------------------------

def test_full_path_clears_box_bounding_box():
    """Stepping the trajectory from start to end through a 0.10 m box at
    (0,0) must never put the EE inside the box's xy footprint at z below
    the box top. This is the key safety property the PWL design buys."""
    z_safe   = 0.20
    box_top  = 0.10
    p_target = np.array([0.20, 0.0, 0.05])
    p_now    = np.array([-0.20, 0.0, 0.05])

    waypoints = [p_now.copy()]
    for _ in range(2000):                 # plenty of iterations to converge
        wp = next_waypoint(waypoints[-1], p_target, z_safe=z_safe, ds=0.005)
        waypoints.append(wp)
        if np.linalg.norm(wp - p_target) < 1e-4:
            break

    box_half = 0.05    # half-edge of a 10 cm box at origin
    for wp in waypoints:
        in_xy_footprint = (abs(wp[0]) <= box_half) and (abs(wp[1]) <= box_half)
        if in_xy_footprint:
            assert wp[2] >= box_top - 1e-6, (
                f"PWL waypoint {wp} entered box footprint below box top "
                f"{box_top}; trajectory would clip the box.")


def test_path_terminates_at_target():
    """The PWL iteration must eventually reach the target (within tol)."""
    p_target = np.array([0.20, 0.20, 0.05])
    p_now    = np.array([-0.20, -0.20, 0.05])
    cur = p_now.copy()
    for _ in range(5000):
        cur = next_waypoint(cur, p_target, z_safe=0.20, ds=0.005)
        if np.linalg.norm(cur - p_target) < 1e-3:
            break
    assert is_at_target(cur, p_target, tol=1e-3)


# ---------------------------------------------------------------------------
# Straight-line shortcut
# ---------------------------------------------------------------------------

def test_short_distance_uses_direct_line_not_lift():
    """Below straight_line_thresh, must NOT detour through z_safe."""
    p_now    = np.array([0.00, 0.00, 0.05])
    p_target = np.array([0.005, 0.00, 0.05])   # 5 mm away
    out = next_waypoint(p_now, p_target, z_safe=0.20, ds=0.001,
                        straight_line_thresh=0.008)
    # z must NOT have lifted to z_safe; it should stay near 0.05
    assert out[2] == pytest.approx(0.05, abs=1e-9)
    # x advanced toward target
    assert 0.0 < out[0] < 0.005


def test_short_distance_clamps_at_target():
    p_now    = np.array([0.00, 0.00, 0.05])
    p_target = np.array([0.005, 0.00, 0.05])
    out = next_waypoint(p_now, p_target, z_safe=0.20, ds=10.0,
                        straight_line_thresh=0.008)
    np.testing.assert_array_almost_equal(out, p_target, decimal=9)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_invalid_shapes_raise():
    with pytest.raises(ValueError):
        next_waypoint(np.zeros(2), np.zeros(3), z_safe=0.2, ds=0.01)
    with pytest.raises(ValueError):
        next_waypoint(np.zeros(3), np.zeros(4), z_safe=0.2, ds=0.01)


def test_non_positive_ds_raises():
    with pytest.raises(ValueError):
        next_waypoint(np.zeros(3), np.array([0.1, 0.0, 0.05]),
                      z_safe=0.2, ds=0.0)


# ---------------------------------------------------------------------------
# is_at_target helper
# ---------------------------------------------------------------------------

def test_is_at_target_within_tol():
    p = np.array([0.10, 0.20, 0.05])
    assert is_at_target(p + 1e-4, p, tol=1e-3) is True
    assert is_at_target(p + 1e-2, p, tol=1e-3) is False
