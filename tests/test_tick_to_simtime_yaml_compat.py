"""Stage C tick→sim-t reconciliation — YAML back-compat shim unit tests.

The conversion renames tick-int fields to sim-time-float fields with
back-compat shims that auto-convert old YAML (int ticks) → new field
(float seconds) at load time, emitting a [YAML-COMPAT] log line.

These tests PIN that contract.
"""
from __future__ import annotations

import pytest
import yaml

from control.sampling_c3.params import SamplingC3Params


def _params_from_dict(d: dict) -> SamplingC3Params:
    """Helper: build a SamplingC3Params from a YAML-loaded dict."""
    return SamplingC3Params.from_dict(d)


def test_old_sample_buffer_lifetime_int_converts_to_seconds(capsys):
    """Pre-conversion YAML: `sample_buffer_lifetime: 30` (ticks)
    should auto-convert to `sample_buffer_lifetime_s = 0.30` (seconds)
    with a [YAML-COMPAT] log line."""
    d = {
        "sampling_params": {"sample_buffer_lifetime": 30},
        "reposition_params": {},
    }
    cfg = _params_from_dict(d)
    assert cfg.sampling_params.sample_buffer_lifetime_s == pytest.approx(0.30)
    captured = capsys.readouterr()
    assert "YAML-COMPAT" in captured.out


def test_new_sample_buffer_lifetime_s_loads_directly():
    """Post-conversion YAML: `sample_buffer_lifetime_s: 0.45` (seconds)
    loads directly without any shim."""
    d = {
        "sampling_params": {"sample_buffer_lifetime_s": 0.45},
        "reposition_params": {},
    }
    cfg = _params_from_dict(d)
    assert cfg.sampling_params.sample_buffer_lifetime_s == pytest.approx(0.45)


def test_old_contact_loss_threshold_default_converts(capsys):
    """`contact_loss_threshold_default: 5` (ticks) → `_default_s = 0.05`."""
    d = {
        "sampling_params": {},
        "reposition_params": {},
        "contact_loss_threshold_default": 5,
    }
    cfg = _params_from_dict(d)
    assert cfg.contact_loss_threshold_default_s == pytest.approx(0.05)
    captured = capsys.readouterr()
    assert "YAML-COMPAT" in captured.out


def test_old_num_control_loops_to_wait_converts(capsys):
    """`num_control_loops_to_wait: 60` (ticks) → `_to_wait_s = 0.60`."""
    d = {
        "sampling_params": {},
        "reposition_params": {},
        "progress_params": {"num_control_loops_to_wait": 60},
    }
    cfg = _params_from_dict(d)
    assert cfg.progress_params.num_control_loops_to_wait_s == pytest.approx(0.60)
    captured = capsys.readouterr()
    assert "YAML-COMPAT" in captured.out


def test_new_num_control_loops_to_wait_s_loads_directly():
    """`num_control_loops_to_wait_s: 0.75` loads directly."""
    d = {
        "sampling_params": {},
        "reposition_params": {},
        "progress_params": {"num_control_loops_to_wait_s": 0.75},
    }
    cfg = _params_from_dict(d)
    assert cfg.progress_params.num_control_loops_to_wait_s == pytest.approx(0.75)


def test_old_progress_enforced_over_n_loops_converts(capsys):
    """`progress_enforced_over_n_loops: 30` (ticks) → `_s = 0.30`."""
    d = {
        "sampling_params": {},
        "reposition_params": {},
        "progress_params": {"progress_enforced_over_n_loops": 30},
    }
    cfg = _params_from_dict(d)
    assert cfg.progress_params.progress_enforced_over_duration_s == pytest.approx(0.30)
    captured = capsys.readouterr()
    assert "YAML-COMPAT" in captured.out
