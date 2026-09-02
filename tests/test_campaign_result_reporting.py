"""Regression coverage for continuous-goal campaign result reporting."""

from pathlib import Path

import pytest

from scripts import run_fig8_block_28_successes as campaign
from scripts import plot_fig8


def test_campaign_result_is_separate_from_final_pose_snapshot():
    """A reached campaign must not be inferred from the newly drawn goal."""
    source = (Path(__file__).parents[1] / "main.py").read_text()

    campaign = source.index("[CAMPAIGN-RESULT]")
    final_pose = source.index("[RESULT] method=", campaign)

    assert campaign < final_pose
    assert "_goal_gen.goals_reached >= _gg_n" in source
    assert "requested_goals={_gg_n}" in source
    assert source.count("_gg_n = int(os.environ.get(\"PORT_GOALGEN_N\"") == 1
    assert source.index("_gg_n = int(os.environ.get(\"PORT_GOALGEN_N\"") < campaign


def test_campaign_reporting_is_diagnostic_only():
    """The reporting block must not mutate controller or goal state."""
    source = (Path(__file__).parents[1] / "main.py").read_text()
    comment = source.index("Campaign outcome is distinct")
    start = source.index("if _gg_n > 0:", comment)
    end = source.index("print(f\"[RESULT] method=", start)
    block = source[start:end]

    assert "set_goal" not in block
    assert "reset_for_new_goal" not in block
    assert "force_regoal" not in block


def test_fig8_runner_uses_one_continuous_goal_sequence():
    source = (Path(__file__).parents[1] / "scripts" /
              "run_fig8_block_28_successes.py").read_text()

    assert '"PORT_GOALGEN_N": str(target)' in source
    assert '"PORT_GOAL_TIMEOUT_S": str(goal_timeout)' in source
    assert "args.max_time = args.target * args.goal_timeout" in source
    assert "next_seed" not in source
    assert "while successes" not in source
    assert "fig8_consecutive_28_c3plus" in source
    assert "A failed session is never replaced" in source
    assert 'default=1200.0' in source


def test_fig8_runner_does_not_abort_recoverable_tilt():
    """Canonical goals must run to their goal timeout, not a tilt heuristic.

    Recorded Fig. 8 runs include a successful 182.5-second tilt recovery, so
    the former 120-second reachability rule was not a valid safety boundary.
    The opt-in diagnostic remains available in ``main.py`` but must not be
    forced by the canonical campaign wrapper.
    """
    source = (Path(__file__).parents[1] / "scripts" /
              "run_fig8_block_28_successes.py").read_text()

    assert '"PORT_PLANAR_REACHABILITY_GUARD"' not in source
    assert '"PORT_PLANAR_TILT_LIMIT_RAD"' not in source
    assert '"PORT_PLANAR_TILT_HOLD_S"' not in source


def test_fig8_session_requires_all_goals_and_campaign_pass(tmp_path):
    log = tmp_path / "session.txt"
    log.write_text(
        "[GOAL-GEN] COMPLETE: 28 goals achieved (PORT_GOALGEN_N=28)\n"
        "[CAMPAIGN-RESULT] requested_goals=28 goals_reached=28 status=PASS\n"
    )
    assert campaign.classify(log, 28) == "success"

    log.write_text(
        "[GOAL-GEN] COMPLETE: 1 goals achieved (PORT_GOALGEN_N=1)\n"
        "[CAMPAIGN-RESULT] requested_goals=1 goals_reached=1 status=PASS\n"
    )
    assert campaign.classify(log, 28) == "failure"


@pytest.mark.parametrize(("marker", "category"), [
    ("[ABORT-UNREACHABLE] t=130.0s", "persistent_topple"),
    ("[WORKSPACE-VIOLATION] step=2\nTraceback (most recent call last)",
     "workspace_limit"),
    ("[ABORT] Object position blowup at t=4.0s", "numerical_divergence"),
    ("[WARN] NaN in state at t=4.0s", "nonfinite_state"),
    ("[RESULT] FAILURE: consecutive campaign wall timeout after 99.0s",
     "wall_clock_timeout"),
])
def test_fig8_failure_diagnosis_markers(tmp_path, marker, category):
    log = tmp_path / "session.txt"
    log.write_text(
        "[GOAL-GEN] goal #1 REACHED at t=2.0s -> new goal\n" + marker + "\n"
    )
    diagnosis = campaign.diagnose(log, 28)
    assert diagnosis["state"] == "failure"
    assert diagnosis["failure_category"] == category
    assert diagnosis["goals_reached"] == 1
    assert diagnosis["failed_goal_index"] == 2


def test_fig8_tight_tolerance_timeout_diagnosis(tmp_path):
    log = tmp_path / "session.txt"
    log.write_text(
        "[CAMPAIGN-RESULT] requested_goals=28 goals_reached=7 status=FAIL\n"
        "[RESULT] method=sampling-c3 translational_error=0.012m "
        "rotational_error=0.16rad tight_goal=FAIL(-) loose_goal=PASS\n"
    )
    diagnosis = campaign.diagnose(log, 28)
    assert diagnosis["failure_category"] == "tight_tolerance_timeout"
    assert diagnosis["goals_reached"] == 7
    assert diagnosis["failed_goal_index"] == 8


def test_fig8_plot_uses_consecutive_timestamp_differences(tmp_path, monkeypatch):
    task = plot_fig8.ORDER[0][1]
    log_dir = tmp_path / task
    log_dir.mkdir()
    lines = [f"[GOAL-GEN] goal #{i} REACHED at t={i * 2.5:.3f}s -> new goal"
             for i in range(1, 29)]
    lines.append("[GOAL-GEN] COMPLETE: 28 goals achieved (PORT_GOALGEN_N=28)")
    lines.append("[CAMPAIGN-RESULT] requested_goals=28 goals_reached=28 status=PASS")
    (log_dir / f"{task}_consecutive28.txt").write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(plot_fig8, "CAMPAIGN_DIR", str(tmp_path))

    data, records = plot_fig8.collect()

    assert data["Letter I"] == [2.5] * 28
    assert len(records) == 28
    assert records[-1]["goal_index"] == 28
    assert records[-1]["session_total_s"] == "70.000"


def test_failed_session_preserves_completed_goal_statistics(tmp_path, monkeypatch):
    task = plot_fig8.ORDER[0][1]
    log_dir = tmp_path / task
    log_dir.mkdir()
    (log_dir / f"{task}_consecutive28.txt").write_text(
        "[GOAL-GEN] goal #1 REACHED at t=3.000s -> new goal\n"
        "[GOAL-GEN] goal #2 REACHED at t=8.500s -> new goal\n"
        "[ABORT-UNREACHABLE] t=130.0s\n"
        "[CAMPAIGN-RESULT] requested_goals=28 goals_reached=2 status=FAIL\n"
    )
    monkeypatch.setattr(plot_fig8, "CAMPAIGN_DIR", str(tmp_path))

    data, records, sessions = plot_fig8.collect_sessions()

    assert data["Letter I"] == [3.0, 5.5]
    assert [row["session_outcome"] for row in records] == ["FAIL", "FAIL"]
    assert sessions[0]["outcome"] == "FAIL"
    assert sessions[0]["goals_reached"] == 2
    assert sessions[0]["failed_goal_index"] == 3
    assert sessions[0]["failure_category"] == "persistent_topple"


def test_per_goal_timeout_resets_only_after_goal_achievement():
    source = (Path(__file__).parents[1] / "main.py").read_text()
    timeout_check = source.index("[GOAL-TIMEOUT]")
    regoal = source.index("if _regoaled:", timeout_check)
    reset = source.index("_gg_goal_started_at = sim_time", regoal)
    assert timeout_check < regoal < reset
