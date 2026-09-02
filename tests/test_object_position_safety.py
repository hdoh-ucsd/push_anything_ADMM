import numpy as np

from main import object_position_is_divergent


def test_continuous_goal_travel_is_not_numerical_blowup():
    initial = np.array([0.4, -0.3, 0.027])
    # Exact A-run state that the old Euclidean 0.5 m rule rejected.
    current = np.array([0.5108755024791172, 0.1891407473367681,
                        0.0234349988715954])
    assert np.linalg.norm(current - initial) > 0.5
    assert object_position_is_divergent(current, initial) is False


def test_large_planar_or_vertical_excursion_is_divergent():
    initial = np.array([0.4, -0.3, 0.027])
    assert object_position_is_divergent([1.41, -0.3, 0.027], initial) is True
    assert object_position_is_divergent([0.4, -0.3, -0.474], initial) is True


def test_limits_are_independent_not_a_euclidean_ball():
    initial = np.zeros(3)
    assert object_position_is_divergent([0.8, 0.5, 0.49], initial) is False
