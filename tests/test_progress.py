"""Unit tests for control.sampling_c3.progress."""
import pytest

from control.sampling_c3.params import ProgressMetric, ProgressParams
from control.sampling_c3.progress import ProgressTracker, StepMetrics


def _params(**over) -> ProgressParams:
    p = ProgressParams()
    for k, v in over.items():
        setattr(p, k, v)
    return p


def _step(c3=100.0, cfg=50.0, pos=0.30, rot=0.0):
    return StepMetrics(c3_cost=c3, config_cost=cfg, pos_error=pos, rot_error=rot)


# ---------------------------------------------------------------------------
# Empty / first-update behaviour
# ---------------------------------------------------------------------------

def test_no_updates_yet_returns_progressing():
    """Before any update, met_progress must return True (no timeout fires
    when there's no data)."""
    t = ProgressTracker(_params())
    assert t.met_progress(near_goal=False) is True


def test_single_update_keeps_progressing():
    t = ProgressTracker(_params())
    t.update(_step())
    assert t.met_progress(near_goal=False) is True


# ---------------------------------------------------------------------------
# kC3Cost variant
# ---------------------------------------------------------------------------

def test_kC3Cost_progress_when_decreasing():
    t = ProgressTracker(_params(track_c3_progress_via=ProgressMetric.kC3Cost,
                                num_control_loops_to_wait=5))
    for c in [100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0]:
        t.update(_step(c3=c))
    assert t.met_progress(near_goal=False) is True


def test_kC3Cost_timeout_after_no_improvement():
    t = ProgressTracker(_params(track_c3_progress_via=ProgressMetric.kC3Cost,
                                num_control_loops_to_wait=3))
    t.update(_step(c3=100.0))   # initial best
    # 4 consecutive non-improvements → counter = 4 ≥ wait=3 → timeout
    for _ in range(4):
        t.update(_step(c3=200.0))
    assert t.met_progress(near_goal=False) is False


def test_improvement_resets_the_counter():
    t = ProgressTracker(_params(track_c3_progress_via=ProgressMetric.kC3Cost,
                                num_control_loops_to_wait=3))
    t.update(_step(c3=100.0))
    for _ in range(5):
        t.update(_step(c3=200.0))
    assert t.met_progress(near_goal=False) is False
    # Now a new best — counter should reset
    t.update(_step(c3=80.0))
    assert t.met_progress(near_goal=False) is True


# ---------------------------------------------------------------------------
# kPosOrRotCost variant
# ---------------------------------------------------------------------------

def test_kPosOrRotCost_either_metric_keeps_progressing():
    t = ProgressTracker(_params(track_c3_progress_via=ProgressMetric.kPosOrRotCost,
                                num_control_loops_to_wait=3))
    t.update(_step(pos=0.30, rot=0.10))
    # pos stalls but rot keeps improving
    for r in [0.09, 0.08, 0.07, 0.06, 0.05]:
        t.update(_step(pos=0.30, rot=r))
    assert t.met_progress(near_goal=False) is True


def test_kPosOrRotCost_timeout_when_both_stagnate():
    t = ProgressTracker(_params(track_c3_progress_via=ProgressMetric.kPosOrRotCost,
                                num_control_loops_to_wait=3))
    t.update(_step(pos=0.30, rot=0.10))
    for _ in range(4):
        t.update(_step(pos=0.40, rot=0.20))   # both worse
    assert t.met_progress(near_goal=False) is False


# ---------------------------------------------------------------------------
# Position-variant (near_goal switches the wait length)
# ---------------------------------------------------------------------------

def test_near_goal_uses_position_wait_length():
    """When near_goal=True, num_control_loops_to_wait_position applies."""
    t = ProgressTracker(_params(
        track_c3_progress_via=ProgressMetric.kPosOrRotCost,
        num_control_loops_to_wait=10,
        num_control_loops_to_wait_position=2,
    ))
    t.update(_step(pos=0.10))
    # 3 stalls > position-wait=2 BUT < normal-wait=10
    for _ in range(3):
        t.update(_step(pos=0.20))
    assert t.met_progress(near_goal=True)  is False
    assert t.met_progress(near_goal=False) is True


# ---------------------------------------------------------------------------
# kConfigCostDrop variant (paper's strict mode)
# ---------------------------------------------------------------------------

def test_kConfigCostDrop_progress_if_drop_meets_threshold():
    t = ProgressTracker(_params(
        track_c3_progress_via=ProgressMetric.kConfigCostDrop,
        progress_enforced_over_n_loops=3,
        progress_enforced_cost_drop=10.0,
    ))
    # config cost trace 100 → 80 over 3 loops → drop=20 ≥ required=10
    for c in [100.0, 95.0, 88.0, 80.0]:
        t.update(_step(cfg=c))
    assert t.met_progress(near_goal=False) is True


def test_kConfigCostDrop_timeout_if_drop_below_threshold():
    t = ProgressTracker(_params(
        track_c3_progress_via=ProgressMetric.kConfigCostDrop,
        progress_enforced_over_n_loops=3,
        progress_enforced_cost_drop=10.0,
    ))
    # drop only 2 over the window
    for c in [100.0, 99.5, 99.0, 98.0]:
        t.update(_step(cfg=c))
    assert t.met_progress(near_goal=False) is False


def test_kConfigCostDrop_short_history_returns_progressing():
    """Insufficient history → benefit of the doubt → True."""
    t = ProgressTracker(_params(
        track_c3_progress_via=ProgressMetric.kConfigCostDrop,
        progress_enforced_over_n_loops=10,
        progress_enforced_cost_drop=10.0,
    ))
    t.update(_step(cfg=100.0))
    t.update(_step(cfg=99.0))
    assert t.met_progress(near_goal=False) is True


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

def test_reset_wipes_state():
    t = ProgressTracker(_params(num_control_loops_to_wait=2))
    t.update(_step(pos=0.10))
    for _ in range(5):
        t.update(_step(pos=0.20))
    assert t.met_progress(near_goal=False) is False
    t.reset()
    assert t.met_progress(near_goal=False) is True   # nothing to time out
    t.update(_step(pos=0.30))
    assert t.met_progress(near_goal=False) is True   # one update only


# ---------------------------------------------------------------------------
# Diagnostic API
# ---------------------------------------------------------------------------

def test_steps_since_improve_increments_correctly():
    t = ProgressTracker(_params(track_c3_progress_via=ProgressMetric.kC3Cost))
    t.update(_step(c3=100.0))
    assert t.steps_since_improve() == 0
    t.update(_step(c3=200.0))
    t.update(_step(c3=200.0))
    assert t.steps_since_improve() == 2
    t.update(_step(c3=50.0))
    assert t.steps_since_improve() == 0
