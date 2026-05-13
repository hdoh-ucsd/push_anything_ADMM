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
    """Match enum RepositioningTrajectoryType in dairlib reposition_params.h.

    kIK is a project-specific extension (no upstream equivalent) selecting
    the constrained-pydrake-IK planner in control/sampling_c3/reposition_ik.py
    instead of the per-step Cartesian PWL tracker.
    """
    kSpline          = 0
    kSpherical       = 1
    kCircular        = 2
    kPiecewiseLinear = 3
    kIK              = 4   # RepositionIKTracker — see reposition_ik.py


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
    sampling_radius:                     float = 0.13   # m, around object xy
    # 9.4.7 / F2: reduced from 0.18 (no documented rationale) to close
    # the 5mm geometric mismatch with Drake's 0.10m contact-extraction
    # threshold (lcs_formulator.py:181). Old value placed every strategy
    # sample at pusher-to-box surface clearance 0.105m — 5mm above the
    # threshold — so ee-box pairs never entered the project filter and
    # the LCS was empty at all commanded geometry (9.4.6 probe). New
    # value targets 0.055m clearance (inside the threshold) while
    # keeping 0.055m margin above the 0.075m hard collision floor
    # (box_half 0.05 + pusher_radius 0.025).
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

    # Joint-PD control law for tracking the per-step waypoint.
    # Defaults are calibrated to the operating regime measured in step 8;
    # see docs/reposition_ik.md §Refactor-protection notes for the receipts.
    # Kp_q = 60: at this gain ‖u‖_max ≈ √(5.14)·30 Nm — most joints already
    #   in the saturation regime. Doubling to 120 produces ‖u‖_max =
    #   √6·30 Nm exactly (6 of 7 joints clipping); proportional response is
    #   capped by torque_limit, not gain. Tracking does not improve.
    Kp_q:                                          float = 60.0
    # Kd_q = 8: damps absolute joint velocity. Note the D-term has no
    #   v_target component (u_d = -Kd_q·v_arm_now), so it damps motion
    #   toward the target as well as motion away from it. A future fix
    #   surface would compute v_target from consecutive IK knots and use
    #   u_d = -Kd_q·(v_arm_now - v_target); requires num_full_ik_knots ≥ 2.
    Kd_q:                                          float = 8.0
    # Ki_q = 8: integral gain. Combined with I_max below, max integral
    #   correction = Ki_q·I_max = 32 Nm (after 9.4.5-B Attempt 1; was
    #   16 Nm under step 8 Fix 6's I_max=2.0). Sized to the heaviest
    #   gravity-load mismatch the executor is asked to hold (home-pose
    #   joint 1 deficit ~33.65 Nm), not just the pushing-task equilibrium.
    Ki_q:                                          float = 8.0
    # I_max = 4.0 (raised from 2.0 in 9.4.5-B Attempt 1; previously raised
    #   from 0.5 in step 8 Fix 6).
    #
    #   Step 8 rationale (Fix 6, I_max 0.5 → 2.0): the integral converges
    #   to ~1.0 rad·s per joint at equilibrium under the pushing task,
    #   matching the measured 7.39 Nm gravity-load shift on q[1] (shoulder)
    #   between current_q and q_target to within 10%. With I_max = 0.5 the
    #   integral was clamped at 50% of its natural equilibrium, capping
    #   correction at 4 Nm — half of what the task requires.
    #
    #   9.4.5-B Attempt 1 rationale (I_max 2.0 → 4.0): the step 8 budget
    #   (Ki·I_max = 16 Nm) was sized to the pushing-task gravity load. The
    #   9.4.5-A.1 hold-home-pose probe (commit 1102939) measured a heavier
    #   home-hold load — 3 integrators clamped at I_max=2.0, joint 1
    #   q_err=-0.405 rad, EE displacement 197mm at t=30s. The step 8
    #   executor-tuning catalog (commit 22bfd4a) showed Ki·I_max needs to
    #   be sized to the heaviest load the executor is asked to hold, not
    #   the pushing-task equilibrium. Doubling I_max to 4.0 raises Ki·I_max
    #   from 16 Nm to 32 Nm, close to (but slightly under) the ~33.65 Nm
    #   deficit at joint 1's home-hold equilibrium. Validated against
    #   the 9.4.5-A.1 probe and the verdict-A regression (probe_5f_smoke
    #   paths A and D) — see commit message of this commit for results.
    I_max:                                         float = 4.0
    # torque_limit = 30 Nm: per-joint clip applied at reposition_ik.py:1190
    #   and reposition.py:230. The saturation signature ‖u‖_max ≈ √n·30 Nm
    #   in measured 7-joint torque norms indicates n joints simultaneously
    #   at this clip; raising Kp_q past the regime where this fires does
    #   not increase actual commanded torque.
    torque_limit:                                  float = 30.0

    @classmethod
    def from_dict(cls, raw: dict) -> "RepositionParams":
        kw = _filter_kwargs(cls, raw)
        if "traj_type" in kw:
            kw["traj_type"] = _coerce_enum(
                RepositioningTrajectoryType, kw["traj_type"])
        return cls(**kw)


# ---------------------------------------------------------------------------
# RepositionIKParams — project-specific (no upstream equivalent)
# ---------------------------------------------------------------------------

@dataclass
class RepositionIKParams:
    """Parameters for ``RepositionIKTracker`` (sibling planner — selected by
    ``RepositionParams.traj_type == kIK``).

    Orientation cone (orientation_cone_deg)
    --------------------------------------
    Defaults to 0.0 (disabled). With only the 3-DoF position constraint
    active, IK has 7 DoFs to fit a 3D target — leaving 4 DoFs of redundancy
    that the centering + smoothness costs use to keep warm-started solutions
    on a single IK branch (no q-jumps between adjacent knots). Enabling the
    cone consumes 1-2 redundant DoFs and increases solver-failure rate near
    workspace edges. Don't enable unless the downstream task actually
    requires a constrained EE orientation.

    Min-distance bounds (ik_min_distance_lower_bound + fk_min_distance)
    -------------------------------------------------------------------
    Two distinct knobs, intentionally split (5f V-7, 2026-05-08):

    * ``ik_min_distance_lower_bound`` (default 0.0): lower bound enforced
      INSIDE the per-knot IK solve via Drake's
      ``AddMinimumDistanceLowerBoundConstraint``. Default disables the
      constraint entirely. For typical pushing/manipulation tasks where
      the pusher must contact objects, keep this at 0.0 — every value
      > 0.0 causes the IK to reject any warm-start whose pusher is
      already at table contact, which is the common case at the start
      of a free-mode entry. Set positive only if the task does NOT
      require approach-to-contact.
    * ``fk_min_distance`` (default 0.0): min-distance threshold for the
      FK sweep on knots K..N-1. Default 0.0 disables FK-side clearance
      enforcement entirely; this matches the dairlib upstream precedent
      where reposition IK does not enforce per-knot collision avoidance
      and instead relies on the trajectory's geometric design (lift-
      traverse-descend with safe-height clearance) for safety. Set to a
      positive value only if your trajectory shape genuinely needs
      per-knot signed-distance verification — and budget for the ~19
      ``ComputeSignedDistancePairwiseClosestPoints`` calls per free-mode
      loop that the sweep performs (5f V-8 measurement: borderline
      overshoots of the 8 ms IK cap on a non-trivial fraction of loops).

    Old single ``min_distance_lower_bound`` field has been removed;
    YAMLs containing it raise a clear migration error in ``from_dict``.

    Knot horizon (num_full_ik_knots, "K")
    -------------------------------------
    K = 1 (default) means one full IK solve per control loop and N-1 knots
    filled by joint-space hold + FK signed-distance check. Diagnostic
    only — wrapper consumes only ``q_knots[:, 0]``. Raise K only after
    timing benchmarks show the per-knot budget is met.
    """
    # Constraints
    position_tolerance:                         float = 1e-3
    orientation_cone_deg:                       float = 0.0
    R_des_world_to_ee:                          list  = field(default_factory=list)  # 3x3 row-major; unused if cone == 0
    # Min-distance: split between IK-side and FK-sweep-side, both
    # default 0.0 (disabled) — matches dairlib upstream which relies
    # on lift-traverse-descend trajectory shape for safety. See class
    # docstring for why and when to opt in.
    ik_min_distance_lower_bound:                float = 0.0
    fk_min_distance:                            float = 0.0
    influence_distance_offset:                  float = 0.01

    # Costs
    joint_centering_weight:                     float = 1e-2
    joint_movement_weight:                      float = 1e-1
    q_nominal:                                  list  = field(default_factory=lambda: [
        0.0, -0.7853981633974483, 0.0, -2.356194490192345,
        0.0,  1.5707963267948966, 0.7853981633974483,
    ])  # Franka "ready" pose [0, -pi/4, 0, -3pi/4, 0, pi/2, pi/4]

    # Solver / timing
    per_knot_solve_timeout_s:                   float = 8e-3
    max_ipopt_iter:                             int   = 30  # structural cap — IPOPT max_iter; complements the wall-clock cap
    max_consecutive_failures_before_abort:      int   = 2   # only active when num_full_ik_knots >= 2
    num_full_ik_knots:                          int   = 1

    # IPOPT first-call cold-start can take ~15-25 ms (vs ~6 ms warm), so
    # the very first compute_torque() at t=0 would otherwise overshoot
    # the production wall-clock cap. RepositionIKTracker.__init__ runs a
    # one-shot warm-up Solve() at the end of construction (with a
    # trivially-feasible target = FK of the current arm pose) so the
    # in-loop solves all hit the warm path. Disable for tight test loops
    # where the cumulative warm-up cost across many tracker constructions
    # adds up.
    warm_up_on_construction:                    bool  = True

    # Infeasibility-poison interface to wrapper.py
    infeasibility_match_radius_m:               float = 0.01

    # Frames (informational; tracker resolves via the obj_body / ee_frame
    # objects passed at construction — these names are kept for parity with
    # upstream YAMLs and for potential debugging output)
    ee_frame_name:                              str = "pusher"
    object_body_name:                           str = ""

    @classmethod
    def from_dict(cls, raw: dict) -> "RepositionIKParams":
        # 5f V-7 migration: the old single field is split into two with
        # different defaults. Fail loudly so YAMLs that still set the
        # old name don't silently get the new (much looser) IK default.
        if "min_distance_lower_bound" in raw:
            raise ValueError(
                "RepositionIKParams.min_distance_lower_bound has been "
                "split into ik_min_distance_lower_bound (default 0.0, "
                "disables IK-side enforcement) and fk_min_distance "
                "(default 0.0, disables FK-sweep enforcement — matches "
                "dairlib upstream). Update your YAML to declare which "
                "one(s) you want. See the class docstring for the "
                "rationale."
            )
        kw = _filter_kwargs(cls, raw)
        return cls(**kw)


# ---------------------------------------------------------------------------
# Top-level wrapper
# ---------------------------------------------------------------------------

@dataclass
class SamplingC3Params:
    progress_params:    ProgressParams       = field(default_factory=ProgressParams)
    sampling_params:    SamplingParams       = field(default_factory=SamplingParams)
    reposition_params:  RepositionParams     = field(default_factory=RepositionParams)
    repos_ik_params:    RepositionIKParams   = field(default_factory=RepositionIKParams)

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
            repos_ik_params   = RepositionIKParams.from_dict(raw.get("repos_ik_params", {}) or {}),
            w_align              = float(raw.get("w_align", 30_000.0)),
            w_travel             = float(raw.get("w_travel", 200.0)),
            surrogate_admm_iters = int(raw.get("surrogate_admm_iters", 1)),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SamplingC3Params":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)
