"""NaN-velocity guard for degenerate RepositionTrajectory.

Bug (found via [QP-FAIL] diagnostic, p125/p126): when p_start ≈ p_target
(EE parked at/near the repos target), the straight-line and skip-lift
branches build a 2-knot trajectory whose single leg has zero length and
zero duration. `eval(t)` at `t <= t_start` — which is exactly what the
build tick evaluates (traj built with t_start=sim_t, then eval(sim_t)) —
divided (p1-p0)/seg_dt = 0/0 → NaN v_des → NaN v_err → OSC QP cost is
NaN → OSQP kIterationLimit → zero torque on every main tick. This was
the canonical stack's baseline 0.2-0.4% qp_failure source, amplified to
a 53-consecutive-tick freeze in the p125 stall.
"""
import numpy as np

from control.sampling_c3.reposition_trajectory import RepositionTrajectory


def _assert_all_finite(traj, t):
    p, v, done = traj.eval(t)
    assert np.all(np.isfinite(p)), f"p not finite at t={t}: {p}"
    assert np.all(np.isfinite(v)), f"v not finite at t={t}: {v}"
    return p, v, done


def test_degenerate_identical_start_target_eval_at_t_start():
    """p_start == p_target → zero-length single leg; eval at build time
    (t == t_start) must return finite (zero) velocity, not 0/0 NaN."""
    p = np.array([0.46, -0.056, 0.005])
    traj = RepositionTrajectory(
        p_start=p, p_target=p.copy(), z_safe=0.075, speed=0.18,
        t_start=21.0, straight_line_thresh=0.008, dt_plan=0.1)
    _, v, _ = _assert_all_finite(traj, 21.0)
    np.testing.assert_allclose(v, np.zeros(3))
    # Before start as well (same code path).
    _assert_all_finite(traj, 20.9)
    # And past end.
    _assert_all_finite(traj, 22.0)


def test_degenerate_near_identical_skip_lift_branch():
    """Sub-millimeter offset through the skip-lift branch (arm below
    z_safe, target below arm, xy close): duration ~5.6e-9 s — the guard
    must not blow v up to Δp/1e-12 either; |v| stays ≤ speed-scale."""
    p_start = np.array([0.46, -0.056, 0.0200000001])
    p_target = np.array([0.46, -0.056, 0.0199999999])
    traj = RepositionTrajectory(
        p_start=p_start, p_target=p_target, z_safe=0.075, speed=0.18,
        t_start=5.0, straight_line_thresh=0.008, dt_plan=0.1)
    _, v, _ = _assert_all_finite(traj, 5.0)
    assert float(np.linalg.norm(v)) <= 0.18 + 1e-6


def test_normal_trajectory_velocity_unchanged():
    """Non-degenerate PWL keeps its exact per-leg velocity at t_start."""
    traj = RepositionTrajectory(
        p_start=np.array([0.40, 0.00, 0.075]),
        p_target=np.array([0.50, 0.10, 0.005]),
        z_safe=0.075, speed=0.18, t_start=0.0,
        straight_line_thresh=0.008, dt_plan=0.1)
    p, v, done = traj.eval(0.0)
    assert not done
    # First leg is the traverse at z_safe; speed magnitude = 0.18.
    np.testing.assert_allclose(np.linalg.norm(v), 0.18, rtol=1e-9)
