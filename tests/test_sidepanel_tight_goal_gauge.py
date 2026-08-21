"""Regression coverage for Block-T sidepanel fields and timing."""

import pytest

from tools.visualizer.paint_log_sidepanel import (
    load_frame_times,
    parse_gauges,
    playback_rate,
    realtime_frame_indices,
)


def test_parse_block_t_tight_goal_fields(tmp_path):
    log = tmp_path / "block_t.log"
    log.write_text(
        "[STEP] step=17 mode=c3 t=1.200s "
        "ee=(+0.500,+0.100,+0.030) obj=(+0.480,+0.180,+0.020) "
        "goal_dist=0.015m final_goal_dist=0.0197m "
        "obj_yaw=-0.7012rad goal_yaw=-0.7379rad rot_err=0.0830rad "
        "switch=kStayInC3 lam_n=1.0 contact=Y f_cmd=(+1,+0,+0)\n"
    )

    gauge = parse_gauges(log)[17]

    assert gauge["goal_dist"] == "0.015"
    assert gauge["final_goal_dist"] == "0.0197"
    assert gauge["rot_err"] == "0.0830"
    assert gauge["obj_yaw"] == "-0.7012"
    assert gauge["goal_yaw"] == "-0.7379"


def test_historical_step_logs_remain_parseable(tmp_path):
    log = tmp_path / "old.log"
    log.write_text(
        "[STEP] step=3 mode=free t=0.150s "
        "ee=(+0.5,+0.1,+0.03) obj=(+0.4,+0.1,+0.02) "
        "goal_dist=0.120m switch=kStayInFree\n"
    )

    gauge = parse_gauges(log)[3]

    assert gauge["goal_dist"] == "0.120"
    assert "final_goal_dist" not in gauge
    assert "rot_err" not in gauge


def test_timeline_drives_playback_rate(tmp_path):
    (tmp_path / "mode_timeline.csv").write_text(
        "step,sim_t,mode,switch\n"
        "10,0.0000,free,kStayInFree\n"
        "12,0.1500,free,kStayInFree\n"
        "14,0.3000,c3,kToC3ReachedReposTarget\n"
    )
    times = load_frame_times(tmp_path)

    assert times == {10: 0.0, 12: 0.15, 14: 0.3}
    assert playback_rate([0.0, 0.15, 0.3], fps=10.0,
                         realtime=False) == pytest.approx(1.5)
    assert playback_rate([0.0, 0.15, 0.3], fps=27.0,
                         realtime=True) == 1.0


def test_realtime_cfr_schedule_uses_sim_timestamps():
    indices = realtime_frame_indices([0.0, 0.15, 0.30], fps=20.0)

    assert len(indices) == 9       # 0.45 s covered at 20 fps
    assert indices[:3] == [0, 0, 0]
    assert indices[3:6] == [1, 1, 1]
    assert indices[6:] == [2, 2, 2]
