"""Tests for the per-tick orientation lookahead (SLERP sub-goal).

Port of reference `GenerateLineTrajectoryWithLookahead`, orientation
branch (examples/sampling_c3/goal_generator.cc:408-437):

1. angle-axis of q_goal * q_now^-1, Eigen-canonical (angle in [0, pi];
   axis = vec/|vec| sign-flipped when w < 0);
2. 180-degree-singularity hysteresis: if axis.dot(last_axis) < 0 AND
   (pi - angle) < angle_hysteresis, take the complementary rotation
   (angle -> 2*pi - angle, axis -> -axis); store the post-flip axis;
3. clamp angle to lookahead_angle;
4. sub-goal = R(axis, angle_clamped) * q_now  (world-frame premultiply).

Reference parameter values (identical in all three demos' goal_params.yaml):
lookahead_angle: 2 rad, angle_hysteresis: 0.4 rad.
"""
import numpy as np
import pytest

from control.sampling_c3.goal_generator import (
    geodesic_angle,
    orientation_lookahead,
    quat_multiply,
)


def _axis_angle_quat(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    return np.concatenate([[np.cos(angle / 2.0)], np.sin(angle / 2.0) * axis])


# The failing-run pair (results/jack_rerun_180s.txt, seed 0): init pose is
# the BlueDown-adjacent rest, drawn goal is the AllDown tripod at
# 3.0154 rad reorientation demand.
Q_INIT_RUN = np.array([0.33469071, 0.02671006, -0.88767188, -0.31513067])
Q_GOAL_RUN = np.array([0.3498, -0.8853, 0.0704, 0.2983])


def test_within_lookahead_returns_goal_exactly():
    q_now = _axis_angle_quat([0.0, 0.0, 1.0], 0.3)
    q_goal = _axis_angle_quat([0.0, 0.0, 1.0], 1.5)   # 1.2 rad away < 2.0
    q_sub, _axis = orientation_lookahead(q_now, q_goal, np.zeros(3))
    assert geodesic_angle(q_sub, q_goal) < 1e-12
    # Hemisphere-canonical: the sub-goal sits in the current quat's
    # hemisphere (this is what the reference reconstruction produces).
    assert float(np.dot(q_sub, q_now)) >= 0.0


def test_hemisphere_canonicalization_flips_sign():
    q_now = _axis_angle_quat([0.0, 0.0, 1.0], 0.3)
    q_goal = -_axis_angle_quat([0.0, 0.0, 1.0], 1.5)  # negated representation
    q_sub, _axis = orientation_lookahead(q_now, q_goal, np.zeros(3))
    # Same rotation as the goal, but represented in q_now's hemisphere.
    assert geodesic_angle(q_sub, q_goal) < 1e-12
    assert float(np.dot(q_sub, q_now)) >= 0.0
    assert np.allclose(q_sub, -q_goal, atol=1e-12)


def test_identity_error_returns_current():
    q_now = _axis_angle_quat([1.0, 2.0, 3.0], 0.7)
    q_sub, _axis = orientation_lookahead(q_now, q_now, np.zeros(3))
    assert np.allclose(q_sub, q_now, atol=1e-12)


def test_clamp_beyond_lookahead_lands_on_geodesic():
    q_sub, _axis = orientation_lookahead(Q_INIT_RUN, Q_GOAL_RUN, np.zeros(3))
    demand = geodesic_angle(Q_GOAL_RUN, Q_INIT_RUN)
    assert demand == pytest.approx(3.0154, abs=1e-3)
    # Sub-goal is exactly lookahead_angle from the current orientation ...
    assert geodesic_angle(q_sub, Q_INIT_RUN) == pytest.approx(2.0, abs=1e-9)
    # ... and on the geodesic: remaining distance is demand - 2.0.
    assert geodesic_angle(q_sub, Q_GOAL_RUN) == pytest.approx(
        demand - 2.0, abs=1e-9)


def test_returned_axis_is_error_axis():
    q_sub, axis = orientation_lookahead(Q_INIT_RUN, Q_GOAL_RUN, np.zeros(3))
    # Reconstructing R(axis, 2.0) * q_now must reproduce the sub-goal.
    q_rel = _axis_angle_quat(axis, 2.0)
    assert np.allclose(q_sub, quat_multiply(q_rel, Q_INIT_RUN), atol=1e-9)


def test_first_tick_zero_last_axis_never_flips():
    # angle = 3.0 > pi - 0.4: inside the hysteresis band, but last_axis is
    # the zero vector (reference h:180 init) so dot == 0, not < 0.
    q_now = np.array([1.0, 0.0, 0.0, 0.0])
    q_goal = _axis_angle_quat([0.0, 0.0, 1.0], 3.0)
    q_sub, axis = orientation_lookahead(q_now, q_goal, np.zeros(3))
    assert np.allclose(axis, [0.0, 0.0, 1.0], atol=1e-12)
    assert np.allclose(q_sub, _axis_angle_quat([0.0, 0.0, 1.0], 2.0),
                       atol=1e-12)


def test_hysteresis_flip_near_pi():
    # Error is 3.0 rad about +z (pi - 3.0 = 0.14 < 0.4) and the previous
    # commitment was -z: the reference takes the complementary rotation
    # 2*pi - 3.0 about -z, clamped to 2.0 -> sub-goal rotates -z.
    q_now = np.array([1.0, 0.0, 0.0, 0.0])
    q_goal = _axis_angle_quat([0.0, 0.0, 1.0], 3.0)
    q_sub, axis = orientation_lookahead(
        q_now, q_goal, last_axis=np.array([0.0, 0.0, -1.0]))
    assert np.allclose(axis, [0.0, 0.0, -1.0], atol=1e-12)  # post-flip stored
    assert np.allclose(q_sub, _axis_angle_quat([0.0, 0.0, -1.0], 2.0),
                       atol=1e-12)


def test_no_flip_outside_hysteresis_band():
    # angle = 2.5: pi - 2.5 = 0.64 > 0.4, so even an opposing last axis
    # does not flip.
    q_now = np.array([1.0, 0.0, 0.0, 0.0])
    q_goal = _axis_angle_quat([0.0, 0.0, 1.0], 2.5)
    q_sub, axis = orientation_lookahead(
        q_now, q_goal, last_axis=np.array([0.0, 0.0, -1.0]))
    assert np.allclose(axis, [0.0, 0.0, 1.0], atol=1e-12)
    assert np.allclose(q_sub, _axis_angle_quat([0.0, 0.0, 1.0], 2.0),
                       atol=1e-12)


def test_consecutive_ticks_converge_to_goal():
    # Chasing the sub-goal tick after tick walks the full demand down:
    # simulate the object exactly reaching each sub-goal.
    q_now = Q_INIT_RUN.copy()
    last_axis = np.zeros(3)
    for _ in range(3):
        q_sub, last_axis = orientation_lookahead(q_now, Q_GOAL_RUN, last_axis)
        q_now = q_sub / np.linalg.norm(q_sub)
    # demand 3.0154: after one 2.0 step the rest (1.0154) is within the
    # lookahead, so tick 2 already returns the goal; tick 3 is stationary.
    assert geodesic_angle(q_now, Q_GOAL_RUN) < 1e-9
