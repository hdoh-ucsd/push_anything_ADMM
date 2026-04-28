"""Unit tests for control.sampling_c3.params.

Pure-Python; no Drake required. Run via:

    python3 -m pytest tests/test_sampling_c3_params.py -v
"""
from pathlib import Path
import textwrap

import pytest
import yaml

from control.sampling_c3.params import (
    ProgressMetric,
    ProgressParams,
    RepositionParams,
    RepositioningTrajectoryType,
    SamplingC3Params,
    SamplingParams,
    SamplingStrategy,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YAML = REPO_ROOT / "config" / "sampling_c3_params.yaml"


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

def test_defaults_match_seed_values():
    """Constructed-with-defaults dataclass must match the seed values
    promised in the prompt."""
    p = ProgressParams()
    assert p.track_c3_progress_via                == ProgressMetric.kPosOrRotCost
    assert p.num_control_loops_to_wait            == 60
    assert p.num_control_loops_to_wait_position   == 30
    assert p.cost_switching_threshold_distance    == 0.05
    assert p.finished_reposition_cost             == 5000.0
    assert p.hyst_c3_to_repos                     == 1000.0
    assert p.hyst_c3_to_repos_position            == 5000.0
    assert p.hyst_repos_to_c3                     == 1000.0
    assert p.hyst_repos_to_c3_position            == 5000.0
    assert p.hyst_repos_to_repos                  == 500.0
    assert p.hyst_repos_to_repos_position         == 2500.0
    assert p.use_relative_hysteresis              is False
    # _frac_position (NOT _position_frac) — match upstream naming
    assert p.hyst_c3_to_repos_frac_position       == 0.10
    assert p.hyst_repos_to_repos_frac_position    == 0.05

    s = SamplingParams()
    assert s.sampling_strategy             == SamplingStrategy.kRandomOnCircle
    assert s.sampling_radius               == 0.18
    assert s.sampling_height               == 0.05
    assert s.N_sample_buffer               == 5
    assert s.pos_error_sample_retention    == 0.05
    assert s.ang_error_sample_retention    == 0.30
    assert s.workspace_xy_min              == [-0.5, -0.7]
    assert s.workspace_xy_max              == [ 0.5,  0.0]
    assert s.workspace_z_min               == 0.02
    assert s.workspace_z_max               == 0.30
    assert s.filter_samples_for_safety     is True

    r = RepositionParams()
    assert r.traj_type            == RepositioningTrajectoryType.kPiecewiseLinear
    assert r.speed                == 0.20
    assert r.pwl_waypoint_height  == 0.20
    assert r.Kp_q                 == 60.0
    assert r.Kd_q                 == 8.0
    assert r.Ki_q                 == 8.0
    assert r.I_max                == 0.5
    assert r.torque_limit         == 30.0


def test_top_level_defaults():
    p = SamplingC3Params()
    assert p.w_align              == 30000.0
    assert p.w_travel             == 200.0
    assert p.surrogate_admm_iters == 1
    assert isinstance(p.progress_params,   ProgressParams)
    assert isinstance(p.sampling_params,   SamplingParams)
    assert isinstance(p.reposition_params, RepositionParams)


# ---------------------------------------------------------------------------
# Enum coercion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("kPosOrRotCost",     ProgressMetric.kPosOrRotCost),
    ("kConfigCostDrop",   ProgressMetric.kConfigCostDrop),
    (0,                   ProgressMetric.kC3Cost),
    (3,                   ProgressMetric.kConfigCostDrop),
    ("3",                 ProgressMetric.kConfigCostDrop),
])
def test_progress_metric_coercion(raw, expected):
    p = ProgressParams.from_dict({"track_c3_progress_via": raw})
    assert p.track_c3_progress_via == expected


@pytest.mark.parametrize("raw, expected", [
    ("kRandomOnCircle",   SamplingStrategy.kRandomOnCircle),
    ("kFixed",            SamplingStrategy.kFixed),
    (1,                   SamplingStrategy.kRandomOnCircle),
])
def test_sampling_strategy_coercion(raw, expected):
    s = SamplingParams.from_dict({"sampling_strategy": raw})
    assert s.sampling_strategy == expected


@pytest.mark.parametrize("raw, expected", [
    ("kPiecewiseLinear",  RepositioningTrajectoryType.kPiecewiseLinear),
    ("kSpherical",        RepositioningTrajectoryType.kSpherical),
    (3,                   RepositioningTrajectoryType.kPiecewiseLinear),
])
def test_repos_traj_type_coercion(raw, expected):
    r = RepositionParams.from_dict({"traj_type": raw})
    assert r.traj_type == expected


def test_unknown_enum_value_raises():
    with pytest.raises(ValueError, match="Cannot coerce"):
        ProgressParams.from_dict({"track_c3_progress_via": "kBogus"})


# ---------------------------------------------------------------------------
# Default YAML loads & maps to expected values
# ---------------------------------------------------------------------------

def test_default_yaml_loads():
    """The shipped config/sampling_c3_params.yaml must load and reflect
    the on-disk values. Update this test when the YAML is intentionally
    re-tuned so re-tunings can't pass silently."""
    assert DEFAULT_YAML.exists(), f"missing {DEFAULT_YAML}"
    p = SamplingC3Params.from_yaml(DEFAULT_YAML)

    assert p.w_align == 30000.0
    assert p.w_travel == 200.0
    assert p.progress_params.track_c3_progress_via == ProgressMetric.kPosOrRotCost
    assert p.progress_params.hyst_c3_to_repos == 1000.0
    assert p.progress_params.cost_switching_threshold_distance == 0.05
    # Tuning post-WEST-v7: relative mode + finished_reposition_cost bump
    # to fix the cost-magnitude mismatch with the original seed values.
    assert p.progress_params.use_relative_hysteresis is True
    assert p.progress_params.finished_reposition_cost == 200000.0

    assert p.sampling_params.sampling_strategy == SamplingStrategy.kRandomOnCircle
    assert p.sampling_params.sampling_radius == 0.18
    assert p.sampling_params.workspace_xy_max == [0.5, 0.0]

    assert p.reposition_params.traj_type == RepositioningTrajectoryType.kPiecewiseLinear
    assert p.reposition_params.pwl_waypoint_height == 0.20
    assert p.reposition_params.Kp_q == 60.0


def test_unknown_yaml_field_warns_not_crashes(tmp_path, capsys):
    """An unknown YAML field should print a warning but not crash."""
    cfg = tmp_path / "test.yaml"
    cfg.write_text(textwrap.dedent("""
        progress_params:
          track_c3_progress_via: kPosOrRotCost
          some_field_that_does_not_exist: 42
        sampling_params:
          sampling_strategy: kRandomOnCircle
        reposition_params:
          traj_type: kPiecewiseLinear
    """))
    p = SamplingC3Params.from_yaml(cfg)
    assert p.progress_params.track_c3_progress_via == ProgressMetric.kPosOrRotCost
    out = capsys.readouterr().out
    assert "some_field_that_does_not_exist" in out
    assert "ignored" in out


def test_partial_yaml_uses_defaults_for_missing_fields(tmp_path):
    """A YAML with only some fields populated must inherit defaults."""
    cfg = tmp_path / "partial.yaml"
    cfg.write_text(textwrap.dedent("""
        progress_params:
          hyst_c3_to_repos: 9999.9
    """))
    p = SamplingC3Params.from_yaml(cfg)
    # overridden field
    assert p.progress_params.hyst_c3_to_repos == 9999.9
    # untouched field gets the default
    assert p.progress_params.hyst_repos_to_c3 == 1000.0
    # entire missing section gets defaults
    assert p.sampling_params.sampling_strategy == SamplingStrategy.kRandomOnCircle


def test_empty_yaml_uses_all_defaults(tmp_path):
    cfg = tmp_path / "empty.yaml"
    cfg.write_text("")
    p = SamplingC3Params.from_yaml(cfg)
    assert p == SamplingC3Params()


# ---------------------------------------------------------------------------
# Field-name regression guard (catches accidental name swaps)
# ---------------------------------------------------------------------------

def test_no_position_frac_typo():
    """Upstream uses _frac_position, NOT _position_frac. Regression guard
    against the seed-value-document drift flagged in Step 1."""
    p = ProgressParams()
    bad_names = [
        "hyst_c3_to_repos_position_frac",
        "hyst_repos_to_c3_position_frac",
        "hyst_repos_to_repos_position_frac",
    ]
    for n in bad_names:
        assert not hasattr(p, n), (
            f"ProgressParams has the wrong-order field {n!r}; "
            f"upstream uses _frac_position suffix.")
    good_names = [
        "hyst_c3_to_repos_frac_position",
        "hyst_repos_to_c3_frac_position",
        "hyst_repos_to_repos_frac_position",
    ]
    for n in good_names:
        assert hasattr(p, n), f"ProgressParams missing upstream field {n!r}"
