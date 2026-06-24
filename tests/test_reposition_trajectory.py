"""Unit tests for control/sampling_c3/reposition_trajectory.py."""
from __future__ import annotations

import numpy as np
import pytest

from control.sampling_c3.reposition_trajectory import RepositionTrajectory


def test_three_leg_structure():
    """From (0,0,0.03) to (0.20,0,0.03) via z_safe=0.20 yields 4 knots."""
    traj = RepositionTrajectory(
        p_start=np.array([0.0, 0.0, 0.03]),
        p_target=np.array([0.20, 0.0, 0.03]),
        z_safe=0.20, speed=0.5, t_start=0.0,
    )
    assert traj.knot_positions.shape[0] == 3
    assert traj.knot_positions.shape[1] >= 4
    assert traj.knot_times.shape == (traj.knot_positions.shape[1],)
    assert np.allclose(traj.knot_positions[:, 0], [0.0, 0.0, 0.03])
    assert np.allclose(traj.knot_positions[:, -1], [0.20, 0.0, 0.03])
    assert np.isclose(traj.knot_positions[2, 1], 0.20)
    assert np.isclose(traj.knot_positions[2, -2], 0.20)


def test_eval_at_start_returns_start_point():
    traj = RepositionTrajectory(
        p_start=np.array([0.0, 0.0, 0.03]),
        p_target=np.array([0.20, 0.0, 0.03]),
        z_safe=0.20, speed=0.5, t_start=10.0,
    )
    p, v, done = traj.eval(10.0)
    assert np.allclose(p, [0.0, 0.0, 0.03])
    assert np.isclose(v[0], 0.0)
    assert np.isclose(v[1], 0.0)
    assert np.isclose(v[2], 0.5)
    assert done is False


def test_eval_past_end_returns_target_and_done():
    traj = RepositionTrajectory(
        p_start=np.array([0.0, 0.0, 0.03]),
        p_target=np.array([0.20, 0.0, 0.03]),
        z_safe=0.20, speed=0.5, t_start=0.0,
    )
    t_end = traj.knot_times[-1]
    p, v, done = traj.eval(t_end + 1.0)
    assert np.allclose(p, [0.20, 0.0, 0.03])
    assert np.allclose(v, [0.0, 0.0, 0.0])
    assert done is True


def test_eval_velocity_magnitude_equals_speed_on_each_leg():
    traj = RepositionTrajectory(
        p_start=np.array([0.0, 0.0, 0.03]),
        p_target=np.array([0.20, 0.0, 0.03]),
        z_safe=0.20, speed=0.5, t_start=0.0,
    )
    for i in range(len(traj.knot_times) - 1):
        t_mid = 0.5 * (traj.knot_times[i] + traj.knot_times[i + 1])
        p, v, done = traj.eval(t_mid)
        assert not done
        assert abs(np.linalg.norm(v) - 0.5) < 1e-9


def test_eval_position_continuous_at_knots():
    traj = RepositionTrajectory(
        p_start=np.array([0.0, 0.0, 0.03]),
        p_target=np.array([0.20, 0.0, 0.03]),
        z_safe=0.20, speed=0.5, t_start=0.0,
    )
    for kt in traj.knot_times[1:-1]:
        p_before, _, _ = traj.eval(kt - 1e-6)
        p_after, _, _  = traj.eval(kt + 1e-6)
        assert np.allclose(p_before, p_after, atol=1e-4)


def test_short_hop_under_threshold_uses_direct_line():
    traj = RepositionTrajectory(
        p_start=np.array([0.0, 0.0, 0.05]),
        p_target=np.array([0.003, 0.0, 0.05]),
        z_safe=0.20, speed=0.5, t_start=0.0,
        straight_line_thresh=0.008,
    )
    assert traj.knot_positions.shape[1] == 2
    assert np.allclose(traj.knot_positions[:, 0], [0.0, 0.0, 0.05])
    assert np.allclose(traj.knot_positions[:, -1], [0.003, 0.0, 0.05])


def test_finished_predicate_requires_both_time_and_ee_near_target():
    traj = RepositionTrajectory(
        p_start=np.array([0.0, 0.0, 0.03]),
        p_target=np.array([0.20, 0.0, 0.03]),
        z_safe=0.20, speed=0.5, t_start=0.0,
    )
    t_end = traj.knot_times[-1]
    assert not traj.is_finished(t_end + 1.0, np.array([0.0, 0.0, 0.03]),
                                tol=0.005)
    assert traj.is_finished(t_end + 1.0, np.array([0.20, 0.0, 0.03]),
                            tol=0.005)
    assert not traj.is_finished(t_end - 1.0, np.array([0.20, 0.0, 0.03]),
                                tol=0.005)


def test_kik_yaml_pwl_speed_matches_reference_push_t():
    """Stage A descent-leg alignment lock: kik.yaml's pwl_speed must equal
    the reference push_t value (0.18 m/s).

    Reference: dairlib examples/sampling_c3/push_t/parameters/reposition_params.yaml.
    Regression target: at speed=0.40 the PWL descent ran at vz≈0.44 m/s past
    phi=6mm and Drake compliant contact yawed the box (|qz| 8× baseline on
    seed 0). Aligning to the reference's per-leg-constant 0.18 m/s by
    construction is the fix.
    """
    from control.sampling_c3.params import SamplingC3Params
    params = SamplingC3Params.from_yaml("config/sampling_c3_kik.yaml")
    assert params.reposition_params.pwl_speed == pytest.approx(0.18), (
        f"pwl_speed diverged from reference push_t value 0.18: "
        f"got {params.reposition_params.pwl_speed}"
    )
    # Sanity: descent leg from z=0.15 (pwl_waypoint_height) to z=0.03 at
    # 0.18 m/s → 0.667 s = 67 ticks at dt=0.01 (vs the buggy 30 ticks @ 0.40).
    z_safe = float(params.reposition_params.pwl_waypoint_height)
    pwl_speed = float(params.reposition_params.pwl_speed)
    traj = RepositionTrajectory(
        p_start=np.array([0.0, 0.0, z_safe]),
        p_target=np.array([0.0, 0.0, 0.03]),
        z_safe=z_safe, speed=pwl_speed, t_start=0.0,
    )
    _, v_des, _ = traj.eval(0.05)  # interior of the descent leg
    assert np.linalg.norm(v_des) == pytest.approx(pwl_speed, rel=1e-6), (
        f"descent v_des magnitude diverged from configured pwl_speed: "
        f"|v_des|={np.linalg.norm(v_des)} pwl_speed={pwl_speed}"
    )
