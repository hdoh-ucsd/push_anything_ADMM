"""Unit tests for control.sampling_c3.mode_switch."""
import pytest

from control.sampling_c3.params import ProgressParams
from control.sampling_c3.mode_switch import (
    SwitchReason,
    decide_mode,
    _hysteresis,
)


def _params(**over) -> ProgressParams:
    p = ProgressParams()
    for k, v in over.items():
        setattr(p, k, v)
    return p


# ---------------------------------------------------------------------------
# Hysteresis lookup
# ---------------------------------------------------------------------------

def test_hysteresis_absolute_uses_far_field_when_not_near_goal():
    p = _params(hyst_c3_to_repos=1234.0, hyst_c3_to_repos_position=9999.0)
    assert _hysteresis(p, "c3_to_repos", near_goal=False, ref_cost=0) == 1234.0


def test_hysteresis_absolute_uses_position_field_when_near_goal():
    p = _params(hyst_c3_to_repos=1234.0, hyst_c3_to_repos_position=9999.0)
    assert _hysteresis(p, "c3_to_repos", near_goal=True, ref_cost=0) == 9999.0


def test_hysteresis_relative_scales_with_ref_cost():
    p = _params(use_relative_hysteresis=True,
                hyst_repos_to_c3_frac=0.10,
                hyst_repos_to_c3_frac_position=0.50)
    assert _hysteresis(p, "repos_to_c3", near_goal=False, ref_cost=10000.0) == 1000.0
    assert _hysteresis(p, "repos_to_c3", near_goal=True,  ref_cost=10000.0) == 5000.0


def test_hysteresis_relative_with_negative_cost_uses_abs():
    """C3 costs can go negative (alignment-bonus minus baseline cost) —
    relative hysteresis should still produce a non-negative gap."""
    p = _params(use_relative_hysteresis=True, hyst_c3_to_repos_frac=0.10)
    assert _hysteresis(p, "c3_to_repos", near_goal=False, ref_cost=-50000.0) == 5000.0


def test_unknown_hysteresis_kind_raises():
    with pytest.raises(ValueError):
        _hysteresis(_params(), kind="garbage", near_goal=False, ref_cost=0)


# ---------------------------------------------------------------------------
# C3 → C3 (stay)
# ---------------------------------------------------------------------------

def test_stay_in_c3_when_no_other_sample_is_clearly_better():
    p = _params(hyst_c3_to_repos=1000.0)
    mode, reason = decide_mode(
        prev_mode="c3",
        c3_cost=10000.0,
        best_other_cost=9500.0,    # cheaper but not by enough (need < 9000)
        current_repos_cost=None,
        met_progress=True,
        near_goal=False,
        finished_repos=False,
        params=p,
    )
    assert mode   == "c3"
    assert reason == SwitchReason.kStayInC3


# ---------------------------------------------------------------------------
# C3 → free, kToReposCost
# ---------------------------------------------------------------------------

def test_c3_to_repos_when_cost_gap_exceeds_hysteresis():
    p = _params(hyst_c3_to_repos=1000.0)
    mode, reason = decide_mode(
        prev_mode="c3",
        c3_cost=10000.0,
        best_other_cost=8000.0,    # 2000 cheaper > 1000 gap
        current_repos_cost=None,
        met_progress=True,
        near_goal=False,
        finished_repos=False,
        params=p,
    )
    assert mode   == "free"
    assert reason == SwitchReason.kToReposCost


def test_position_hysteresis_used_near_goal():
    """When near_goal, the larger _position field applies — same gap that
    triggered a switch far from the goal must NOT trigger near it."""
    p = _params(hyst_c3_to_repos=1000.0, hyst_c3_to_repos_position=5000.0)
    args = dict(
        prev_mode="c3",
        c3_cost=10000.0,
        best_other_cost=8000.0,    # gap 2000
        current_repos_cost=None,
        met_progress=True,
        finished_repos=False,
        params=p,
    )
    mode_far,  _ = decide_mode(near_goal=False, **args)
    mode_near, _ = decide_mode(near_goal=True,  **args)
    assert mode_far  == "free"   # gap 2000 > hyst 1000
    assert mode_near == "c3"     # gap 2000 < hyst_position 5000


# ---------------------------------------------------------------------------
# C3 → free, kToReposUnproductive (timeout dominates cost)
# ---------------------------------------------------------------------------

def test_c3_to_repos_when_progress_times_out_even_if_costs_tie():
    p = _params(hyst_c3_to_repos=1000.0)
    mode, reason = decide_mode(
        prev_mode="c3",
        c3_cost=10000.0,
        best_other_cost=12000.0,   # WORSE — but progress has timed out
        current_repos_cost=None,
        met_progress=False,
        near_goal=False,
        finished_repos=False,
        params=p,
    )
    assert mode   == "free"
    assert reason == SwitchReason.kToReposUnproductive


def test_unproductive_takes_precedence_over_cost():
    """Both conditions met — kToReposUnproductive wins (it's checked first)."""
    p = _params(hyst_c3_to_repos=1000.0)
    mode, reason = decide_mode(
        prev_mode="c3",
        c3_cost=10000.0,
        best_other_cost=8000.0,
        current_repos_cost=None,
        met_progress=False,
        near_goal=False,
        finished_repos=False,
        params=p,
    )
    assert reason == SwitchReason.kToReposUnproductive


# ---------------------------------------------------------------------------
# Free → c3, kToC3ReachedReposTarget
# ---------------------------------------------------------------------------

def test_finished_repos_returns_to_c3_unconditionally():
    p = _params(hyst_repos_to_c3=99999.0)   # huge — wouldn't switch on cost alone
    mode, reason = decide_mode(
        prev_mode="free",
        c3_cost=20000.0,           # WORSE than current repos
        best_other_cost=20000.0,
        current_repos_cost=10000.0,
        met_progress=True,
        near_goal=False,
        finished_repos=True,       # forces switch
        params=p,
    )
    assert mode   == "c3"
    assert reason == SwitchReason.kToC3ReachedReposTarget


# ---------------------------------------------------------------------------
# Free → c3, kToC3Cost
# ---------------------------------------------------------------------------

def test_free_to_c3_when_c3_beats_repos_by_hysteresis():
    p = _params(hyst_repos_to_c3=1000.0)
    mode, reason = decide_mode(
        prev_mode="free",
        c3_cost=5000.0,
        best_other_cost=8000.0,    # repos cost 8000; c3+gap=6000 < 8000 → switch
        current_repos_cost=8000.0,
        met_progress=True,
        near_goal=False,
        finished_repos=False,
        params=p,
    )
    assert mode   == "c3"
    assert reason == SwitchReason.kToC3Cost


def test_free_to_c3_blocked_when_gap_below_hysteresis():
    p = _params(hyst_repos_to_c3=2000.0)
    mode, reason = decide_mode(
        prev_mode="free",
        c3_cost=5000.0,
        best_other_cost=6000.0,    # gap 1000 < hyst 2000
        current_repos_cost=6000.0,
        met_progress=True,
        near_goal=False,
        finished_repos=False,
        params=p,
    )
    assert mode != "c3"


# ---------------------------------------------------------------------------
# Free → free, kToBetterRepos
# ---------------------------------------------------------------------------

def test_repos_retarget_when_better_repos_appears():
    p = _params(hyst_repos_to_c3=1e9,         # never re-enter C3 on cost
                hyst_repos_to_repos=500.0)
    mode, reason = decide_mode(
        prev_mode="free",
        c3_cost=20000.0,
        best_other_cost=5000.0,    # better than current repos by 1000
        current_repos_cost=6000.0,
        met_progress=True,
        near_goal=False,
        finished_repos=False,
        params=p,
    )
    assert mode   == "free"
    assert reason == SwitchReason.kToBetterRepos


def test_repos_stay_when_no_better_target():
    p = _params(hyst_repos_to_repos=500.0)
    mode, reason = decide_mode(
        prev_mode="free",
        c3_cost=20000.0,
        best_other_cost=5800.0,    # only 200 better — below 500 hyst
        current_repos_cost=6000.0,
        met_progress=True,
        near_goal=False,
        finished_repos=False,
        params=p,
    )
    assert mode   == "free"
    assert reason == SwitchReason.kStayInRepos


def test_repos_stay_with_no_current_target():
    """When current_repos_cost is None (just entered free mode last loop)
    the kToBetterRepos check is skipped."""
    p = _params(hyst_repos_to_c3=1e9)
    mode, reason = decide_mode(
        prev_mode="free",
        c3_cost=20000.0,
        best_other_cost=5000.0,
        current_repos_cost=None,
        met_progress=True,
        near_goal=False,
        finished_repos=False,
        params=p,
    )
    assert mode   == "free"
    assert reason == SwitchReason.kStayInRepos


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_invalid_prev_mode_raises():
    with pytest.raises(ValueError):
        decide_mode(prev_mode="off",
                    c3_cost=0, best_other_cost=0,
                    current_repos_cost=None,
                    met_progress=True, near_goal=False,
                    finished_repos=False, params=_params())


# ---------------------------------------------------------------------------
# All six SwitchReason enum values are documented
# ---------------------------------------------------------------------------

def test_switch_reasons_have_distinct_int_values():
    vals = [r.value for r in SwitchReason]
    assert len(vals) == len(set(vals))
