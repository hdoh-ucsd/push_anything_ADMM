"""
Parameter dataclasses + YAML loader for the sampling-C3 outer controller.

Mirrors the dairlib upstream parameter structs from
examples/sampling_c3/parameter_headers/{progress,sampling,reposition}_params.h
on branch hien/push_anything_with_nextgen_c3.

Field names are kept verbatim with upstream so configs are interpretable
side-by-side with dairlib YAMLs. A few upstream fields that are specific
to the C++ pipeline are intentionally omitted here:

  cost_type / cost_type_position
      Upstream selects between kSimLCS / kUseC3Plan / kSimImpedance / etc.
      to compute the C3 cost. This Python port always uses the same
      Σ (x_t-x_ref)^T Q (x_t-x_ref) + Σ u_t^T R u_t + terminal expression
      (see C3MPC + admm_solver). Single mode → no enum needed.

Two project-specific fields that have no upstream equivalent:

  w_align     project-specific alignment-bonus weight on sample cost
              (sample cost is reduced by w_align * max(0, n_hat · g_hat));
              empirically required to overcome friction-cone discretization
              bias on directional pushes (default 30000 — do not change
              without re-validating WEST task).
  w_travel    Cartesian travel penalty per metre (default 200).

Loading:

    params = SamplingC3Params.from_yaml("config/sampling_c3_params.yaml")
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import IntEnum
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Enums (numeric values match upstream c3_options.h ordering)
# ---------------------------------------------------------------------------

class ProgressMetric(IntEnum):
    """Match enum ProgressMetric in dairlib parameter_headers/progress_params.h."""
    kC3Cost          = 0
    kConfigCost      = 1
    kPosOrRotCost    = 2
    kConfigCostDrop  = 3


class SamplingStrategy(IntEnum):
    """Match enum SamplingStrategy in dairlib parameter_headers/sampling_params.h."""
    kRadiallySymmetric = 0
    kRandomOnCircle    = 1
    kRandomOnSphere    = 2
    kFixed             = 3
    kRandomOnPerimeter = 4
    kRandomOnShell     = 5
    kMeshNormal        = 6


class RepositioningTrajectoryType(IntEnum):
    """Match enum RepositioningTrajectoryType in dairlib reposition_params.h."""
    kSpline          = 0
    kSpherical       = 1
    kCircular        = 2
    kPiecewiseLinear = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_enum(enum_cls, raw):
    """YAML may give an int or a string like 'kRandomOnCircle' / 'kPosOrRotCost'."""
    if isinstance(raw, enum_cls):
        return raw
    if isinstance(raw, int):
        return enum_cls(raw)
    if isinstance(raw, str):
        if raw in enum_cls.__members__:
            return enum_cls[raw]
        try:
            return enum_cls(int(raw))
        except (ValueError, TypeError):
            pass
    raise ValueError(f"Cannot coerce {raw!r} to {enum_cls.__name__}")


def _filter_kwargs(cls, raw: dict) -> dict:
    """Drop unknown YAML keys instead of crashing — print a warning per key."""
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        for k in sorted(unknown):
            print(f"[sampling_c3.params] warning: unknown {cls.__name__} field {k!r} ignored")
    return {k: v for k, v in raw.items() if k in known}


# ---------------------------------------------------------------------------
# ProgressParams — matches SamplingC3ProgressParams in dairlib
# ---------------------------------------------------------------------------

@dataclass
class ProgressParams:
    # Which progress metric drives the timeout decision
    track_c3_progress_via:               ProgressMetric = ProgressMetric.kPosOrRotCost

    # Timeout (in control loops) for the timeout-based progress check
    num_control_loops_to_wait:           int   = 60
    num_control_loops_to_wait_position:  int   = 30

    # kConfigCostDrop variant: required object-config cost drop over N loops
    progress_enforced_cost_drop:         float = 0.0
    progress_enforced_over_n_loops:      int   = 30

    # Distance below which we use the _position hysteresis variant
    cost_switching_threshold_distance:   float = 0.05

    # Auto-end-reposition when total reposition cost falls below this
    finished_reposition_cost:            float = 5000.0

    # Absolute hysteresis (used when use_relative_hysteresis is False)
    hyst_c3_to_repos:                    float = 1000.0
    hyst_c3_to_repos_position:           float = 5000.0
    hyst_repos_to_c3:                    float = 1000.0
    hyst_repos_to_c3_position:           float = 5000.0
    hyst_repos_to_repos:                 float = 500.0
    hyst_repos_to_repos_position:        float = 2500.0

    # Relative hysteresis (used when use_relative_hysteresis is True)
    use_relative_hysteresis:             bool  = False
    # Note: upstream field names use _frac_position, NOT _position_frac. Keep verbatim.
    hyst_c3_to_repos_frac:               float = 0.05
    hyst_c3_to_repos_frac_position:      float = 0.10
    hyst_repos_to_c3_frac:               float = 0.05
    hyst_repos_to_c3_frac_position:      float = 0.10
    hyst_repos_to_repos_frac:            float = 0.02
    hyst_repos_to_repos_frac_position:   float = 0.05

    @classmethod
    def from_dict(cls, raw: dict) -> "ProgressParams":
        kw = _filter_kwargs(cls, raw)
        if "track_c3_progress_via" in kw:
            kw["track_c3_progress_via"] = _coerce_enum(
                ProgressMetric, kw["track_c3_progress_via"])
        return cls(**kw)


# ---------------------------------------------------------------------------
# SamplingParams — matches SamplingParams in dairlib (extended with workspace)
# ---------------------------------------------------------------------------

@dataclass
class SamplingParams:
    sampling_strategy:                   SamplingStrategy = SamplingStrategy.kRandomOnCircle

    # Total samples evaluated each control loop:
    #   1                                (current EE / "k=0")
    # + (1 if previous-repos target valid)
    # + num_additional_samples_c3        (during C3 mode)
    # + num_additional_samples_repos     (during repos mode)
    num_additional_samples_c3:           int   = 3
    num_additional_samples_repos:        int   = 1

    # Sample buffer
    consider_best_buffer_sample_when_leaving_c3: bool = True
    N_sample_buffer:                     int   = 5
    pos_error_sample_retention:          float = 0.05   # m
    ang_error_sample_retention:          float = 0.30   # rad

    # Geometry shared across multiple strategies
    sampling_radius:                     float = 0.18   # m, around object xy
    sampling_height:                     float = 0.05   # m, contact-plane EE z

    # Workspace bounds (kept here, not in a separate sampling_c3_options.yaml)
    workspace_xy_min:                    list  = field(default_factory=lambda: [-0.5, -0.7])
    workspace_xy_max:                    list  = field(default_factory=lambda: [ 0.5,  0.0])
    workspace_z_min:                     float = 0.02
    workspace_z_max:                     float = 0.30

    # Safety filter — drop samples that fail workspace or surface-clearance check
    filter_samples_for_safety:           bool  = True

    @classmethod
    def from_dict(cls, raw: dict) -> "SamplingParams":
        kw = _filter_kwargs(cls, raw)
        if "sampling_strategy" in kw:
            kw["sampling_strategy"] = _coerce_enum(
                SamplingStrategy, kw["sampling_strategy"])
        return cls(**kw)


# ---------------------------------------------------------------------------
# RepositionParams — matches SamplingC3RepositionParams in dairlib
# ---------------------------------------------------------------------------

@dataclass
class RepositionParams:
    traj_type:                                     RepositioningTrajectoryType = RepositioningTrajectoryType.kPiecewiseLinear
    speed:                                         float = 0.20  # m/s

    # Switching to straight-line under threshold (per-trajectory-type)
    use_straight_line_traj_under_spline:           float = 0.12
    use_straight_line_traj_within_angle:           float = 0.30
    use_straight_line_traj_under_piecewise_linear: float = 0.008

    # Spline-specific
    spline_width:                                  float = 0.17

    # Spherical-specific
    sphere_radius:                                 float = 0.12

    # Circular-specific
    circle_radius:                                 float = 0.20
    circle_height:                                 float = 0.00

    # Piecewise-linear-specific (the only type we currently implement)
    pwl_waypoint_height:                           float = 0.20  # safe-height m

    # Joint-PD control law for tracking the per-step waypoint
    Kp_q:                                          float = 60.0
    Kd_q:                                          float = 8.0
    Ki_q:                                          float = 8.0
    I_max:                                         float = 0.5
    torque_limit:                                  float = 30.0

    @classmethod
    def from_dict(cls, raw: dict) -> "RepositionParams":
        kw = _filter_kwargs(cls, raw)
        if "traj_type" in kw:
            kw["traj_type"] = _coerce_enum(
                RepositioningTrajectoryType, kw["traj_type"])
        return cls(**kw)


# ---------------------------------------------------------------------------
# Top-level wrapper
# ---------------------------------------------------------------------------

@dataclass
class SamplingC3Params:
    progress_params:    ProgressParams    = field(default_factory=ProgressParams)
    sampling_params:    SamplingParams    = field(default_factory=SamplingParams)
    reposition_params:  RepositionParams  = field(default_factory=RepositionParams)

    # Project-specific (no upstream equivalent)
    w_align:            float = 30_000.0
    w_travel:           float = 200.0

    # Inner-solver knobs
    surrogate_admm_iters: int = 1   # for the K-1 cheap sample evaluations

    @classmethod
    def from_dict(cls, raw: dict) -> "SamplingC3Params":
        return cls(
            progress_params   = ProgressParams.from_dict(raw.get("progress_params", {}) or {}),
            sampling_params   = SamplingParams.from_dict(raw.get("sampling_params", {}) or {}),
            reposition_params = RepositionParams.from_dict(raw.get("reposition_params", {}) or {}),
            w_align              = float(raw.get("w_align", 30_000.0)),
            w_travel             = float(raw.get("w_travel", 200.0)),
            surrogate_admm_iters = int(raw.get("surrogate_admm_iters", 1)),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SamplingC3Params":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)
