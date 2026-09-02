import numpy as np
import pytest

from main import planar_tilt_angle


def test_planar_tilt_ignores_yaw():
    yaw = 2.4
    q = [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
    assert planar_tilt_angle(q) == pytest.approx(0.0)


def test_planar_tilt_measures_roll_and_normalizes_quaternion():
    roll = 1.2
    q = 3.0 * np.array(
        [np.cos(roll / 2.0), np.sin(roll / 2.0), 0.0, 0.0])
    assert planar_tilt_angle(q) == pytest.approx(roll)


def test_planar_tilt_detects_upside_down_and_invalid_state():
    assert planar_tilt_angle([0.0, 1.0, 0.0, 0.0]) == pytest.approx(np.pi)
    assert planar_tilt_angle([0.0, 0.0, 0.0, 0.0]) == float("inf")
