"""Unit tests for scripts/_parse_stage_a_metrics.py.

Reads the purpose-built ``[STAGE-A-TRACE]`` lines + ``[ENTRY-GATE]`` +
``[STAGE-A-PWL] ... build`` lines emitted by the dispatcher.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Make scripts/ importable.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts._parse_stage_a_metrics import parse_run_log  # noqa: E402


def _trace(step: int, mode: str, phi: float, bx: float = 0.0, by: float = 0.0,
           lam: float = float("nan"), qy: float = 0.0, qz: float = 0.0,
           fin: int = 0, sim_t: float = None) -> str:
    if sim_t is None:
        sim_t = step * 0.01
    lam_str = "nan" if math.isnan(lam) else f"{lam:.4f}"
    phi_str = "nan" if math.isnan(phi) else f"{phi:.5f}"
    return (f"[STAGE-A-TRACE] step={step} sim_t={sim_t:.3f} mode={mode} "
            f"phi={phi_str} box_xy={bx:+.5f},{by:+.5f} "
            f"lam_n_ee_box={lam_str} qy={qy:+.5f} qz={qz:+.5f} "
            f"finished_repos={fin}")


def test_parser_extracts_admit_rate(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("\n".join([
        _trace(0, "free", phi=float("nan")),
        _trace(1, "c3", phi=0.0005, lam=3.5),
        _trace(2, "c3", phi=0.0008, lam=float("nan")),  # contact lost
    ]) + "\n")
    m = parse_run_log(log)
    assert m["c3_steps_total"] == 2
    assert m["c3_steps_with_admitted_ee_box"] == 1
    assert m["admit_rate"] == 0.5


def test_parser_windowed_landing_median_and_fraction(tmp_path):
    """Refinement 1: WINDOWED over the first c3-mode episode. phi is
    already surface-distance (no half-extent subtraction)."""
    # 5-step c3 episode with surface-distances: 0.0018, 0.0021, 0.0019, 0.0020, 0.0030
    # → median = 0.0020 m (≤ 2 mm),
    #   fraction-within-2 mm = 3/5 = 0.6 (0.0018, 0.0019, 0.0020).
    log_lines = [
        _trace(10, "free", phi=float("nan")),
        _trace(11, "c3", phi=0.0018, lam=1.0),
        _trace(12, "c3", phi=0.0021, lam=1.0),
        _trace(13, "c3", phi=0.0019, lam=1.0),
        _trace(14, "c3", phi=0.0020, lam=1.0),
        _trace(15, "c3", phi=0.0030, lam=1.0),
        _trace(16, "free", phi=float("nan")),
    ]
    log = tmp_path / "run.log"
    log.write_text("\n".join(log_lines) + "\n")
    m = parse_run_log(log, window_ticks_max=100, window_ticks_min=3)
    assert m["first_c3_entry_step"] == 11
    assert m["first_c3_episode_len_ticks"] == 5
    assert m["window_ticks_used"] == 5
    assert m["window_status"] == "OK"
    assert abs(m["landing_median_m"] - 0.0020) < 1e-9
    assert abs(m["landing_fraction_within_2mm"] - 0.6) < 1e-9


def test_parser_window_truncated_when_episode_short(tmp_path):
    """If first c3 episode shorter than WINDOW_MIN_TICKS, flag INSUFFICIENT_WINDOW."""
    log = tmp_path / "run.log"
    log.write_text("\n".join([
        _trace(10, "free", phi=float("nan")),
        _trace(11, "c3", phi=0.0018, lam=1.0),
        _trace(12, "c3", phi=0.0020, lam=1.0),
        _trace(13, "free", phi=float("nan")),
    ]) + "\n")
    m = parse_run_log(log, window_ticks_max=100, window_ticks_min=50)
    assert m["first_c3_episode_len_ticks"] == 2
    assert m["window_status"] == "INSUFFICIENT_WINDOW"


def test_parser_entry_gate_firing_rate(tmp_path):
    """Entry-gate candidates = free-mode ticks with finished_repos=1.
    Firings = [ENTRY-GATE] lines."""
    log = tmp_path / "run.log"
    log.write_text("\n".join([
        _trace(20, "free", phi=0.12, fin=1),
        "[ENTRY-GATE] step=20 ee_to_surf=120.0mm >= thr=60.0mm — block kToC3ReachedReposTarget",
        _trace(30, "free", phi=0.10, fin=1),
        _trace(40, "free", phi=0.11, fin=1),
        "[ENTRY-GATE] step=40 ee_to_surf=110.0mm >= thr=60.0mm — block kToC3ReachedReposTarget",
    ]) + "\n")
    m = parse_run_log(log)
    assert m["entry_gate_candidate_transitions"] == 3
    assert m["entry_gate_firings"] == 2
    assert abs(m["entry_gate_firing_rate"] - (2.0 / 3.0)) < 1e-9


def test_parser_rebuild_rate_hz(tmp_path):
    """Refinement 3: counts [STAGE-A-PWL] ... build lines / free seconds."""
    log = tmp_path / "run.log"
    log.write_text("\n".join([
        # 4 free ticks → 0.04 s of free mode.
        _trace(0, "free", phi=0.20),
        "[STAGE-A-PWL] step=0 sim_t=0.000 build p_start=(...)",
        _trace(1, "free", phi=0.18),
        _trace(2, "free", phi=0.10),
        "[STAGE-A-PWL] step=2 sim_t=0.020 build p_start=(...)",
        _trace(3, "free", phi=0.05),
        _trace(4, "c3", phi=0.001, lam=2.0),
    ]) + "\n")
    m = parse_run_log(log, dt_ctrl=0.01)
    assert m["pwl_rebuilds_total"] == 2
    assert m["free_mode_seconds_total"] == pytest.approx(0.04, abs=1e-9)
    assert m["rebuild_rate_hz"] == pytest.approx(50.0, abs=1e-6)
    assert m["rebuild_churn_flagged"] is True


def test_parser_zero_rebuilds_on_flag_off_baseline(tmp_path):
    """Baseline has no [STAGE-A-PWL] → rebuilds=0, churn flag False."""
    log = tmp_path / "run.log"
    log.write_text("\n".join([
        _trace(0, "free", phi=0.20),
        _trace(1, "free", phi=0.10),
        _trace(2, "c3", phi=0.001, lam=2.0),
    ]) + "\n")
    m = parse_run_log(log, dt_ctrl=0.01)
    assert m["pwl_rebuilds_total"] == 0
    assert m["rebuild_rate_hz"] == 0.0
    assert m["rebuild_churn_flagged"] is False


def test_parser_max_abs_qy_qz(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("\n".join([
        _trace(0, "free", phi=0.20, qy=0.01, qz=-0.02),
        _trace(1, "free", phi=0.10, qy=-0.05, qz=0.03),
        _trace(2, "c3",   phi=0.001, qy=0.02, qz=-0.04),
    ]) + "\n")
    m = parse_run_log(log)
    assert m["max_abs_qy"] == pytest.approx(0.05)
    assert m["max_abs_qz"] == pytest.approx(0.04)


def test_parser_goal_motion(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("\n".join([
        _trace(0, "free", phi=0.20, bx=0.0, by=0.0),
        _trace(1, "free", phi=0.10, bx=0.0, by=0.0),
        _trace(2, "c3",   phi=0.001, bx=0.03, by=0.04, lam=1.0),
    ]) + "\n")
    m = parse_run_log(log)
    # Distance from (0,0) to (0.03, 0.04) = 0.05.
    assert m["goal_motion_m"] == pytest.approx(0.05, abs=1e-9)


def test_parser_no_c3_entry(tmp_path):
    """If the run never enters c3, window_status = NO_C3_ENTRY and the
    windowed landing fields are None."""
    log = tmp_path / "run.log"
    log.write_text("\n".join([
        _trace(0, "free", phi=0.20),
        _trace(1, "free", phi=0.18),
    ]) + "\n")
    m = parse_run_log(log)
    assert m["first_c3_entry_step"] is None
    assert m["window_status"] == "NO_C3_ENTRY"
    assert m["landing_median_m"] is None
    assert m["landing_fraction_within_2mm"] is None
