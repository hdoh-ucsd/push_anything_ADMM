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
from typing import Any, Optional

import yaml

# NOTE: INITIAL_ARM_Q import removed — env_builder no longer exports it.
# The comment references at lines below are historical (documenting the
# rationale for q_nominal[1]=0.325 vs the old INITIAL_ARM_Q[1]=0.675).


# ---------------------------------------------------------------------------
# Enums (numeric values match upstream c3_options.h ordering)
# ---------------------------------------------------------------------------

class ProgressMetric(IntEnum):
    """Match enum ProgressMetric in dairlib parameter_headers/progress_params.h.

    kPosOnly is a project-specific extension (no upstream equivalent). Pushing
    tasks emit rot_error=0.0 constant; kPosOrRotCost's OR-aggregation can
    artificially extend met_progress=True via the rot branch (a latent bug
    unrelated to the current absolute-regression early-exit). kPosOnly
    reports the pos timer only.
    """
    kC3Cost          = 0
    kConfigCost      = 1
    kPosOrRotCost    = 2
    kConfigCostDrop  = 3
    kPosOnly         = 4


class SamplingStrategy(IntEnum):
    """Match enum SamplingStrategy in dairlib parameter_headers/sampling_params.h.

    kFaceNormal is the Push-Anything §IV-B1 paper-faithful sampler: sample a
    point on a stored box face, project it outward along the face's outward
    normal by `sampling_setback`, project to fixed world height. Specific to
    this project — upstream dairlib does not define this enum value.
    """
    kRadiallySymmetric = 0
    kRandomOnCircle    = 1
    kRandomOnSphere    = 2
    kFixed             = 3
    kRandomOnPerimeter = 4
    kRandomOnShell     = 5
    kMeshNormal        = 6
    kFaceNormal        = 7


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


def _resolve_legacy_int_to_seconds(raw: dict, old_key: str, new_key_prefix: str,
                                   default_ticks: int) -> float:
    """2026-06-25 reconciliation back-compat helper.

    If `<new_key_prefix>_s` is present, use it. Else if `<old_key>` is
    present (the OLD int-tick form), convert ticks × 0.01 and print a
    [YAML-COMPAT] log line. Else fall back to default_ticks × 0.01.
    """
    new_key = f"{new_key_prefix}_s"
    if new_key in raw:
        return float(raw[new_key])
    if old_key in raw:
        old = int(raw[old_key])
        new_val = float(old) * 0.01
        print(f"[YAML-COMPAT] {old_key}={old} (ticks @ 100 Hz) "
              f"→ {new_key}={new_val:.4f} s", flush=True)
        return new_val
    return float(default_ticks) * 0.01


def _filter_kwargs(cls, raw: dict) -> dict:
    """Drop unknown YAML keys instead of crashing — print a warning per key.

    Also coerce values for fields annotated as ``float`` or ``int`` from
    string form into the numeric type. PyYAML's safe_load parses YAML 1.1,
    which requires an explicit ``+``/``-`` exponent sign for scientific
    notation; ``1.0e9`` parses as the string ``"1.0e9"``, not a float.
    The cast here makes the loader resilient to that.
    """
    known = {f.name: f for f in fields(cls)}
    unknown = set(raw) - known.keys()
    if unknown:
        for k in sorted(unknown):
            print(f"[sampling_c3.params] warning: unknown {cls.__name__} field {k!r} ignored")
    out: dict = {}
    for k, v in raw.items():
        if k not in known:
            continue
        ann = known[k].type
        if isinstance(v, str) and ann in ("float", "int"):
            v = float(v) if ann == "float" else int(float(v))
        out[k] = v
    return out


# ---------------------------------------------------------------------------
# ProgressParams — matches SamplingC3ProgressParams in dairlib
# ---------------------------------------------------------------------------

@dataclass
class ProgressParams:
    # Which progress metric drives the timeout decision
    track_c3_progress_via:               ProgressMetric = ProgressMetric.kPosOrRotCost

    # Timeout (in seconds, sim-time) for the timeout-based progress check.
    # 2026-06-25 tick→sim-t reconciliation: source-of-truth now in seconds.
    # Defaults preserve the port's 100 Hz sim-time values (60/30 ticks ×
    # 10 ms = 600 ms / 300 ms). Reference (anything/c3plus_progress.yaml:40,41)
    # uses 5 ticks @ 1 kHz = 5 ms / 5 ms — a 120×/60× sim-time gap that is
    # an OPEN alignment question, not closed by this reconciliation.
    num_control_loops_to_wait_s:         float = 0.60
    num_control_loops_to_wait_position_s: float = 0.30

    # Absolute-regression early-exit threshold (metres). When the current
    # pos_error minus the best pos_error since reset() exceeds this, the
    # dispatcher forces met_progress=False at wrapper.py:979 — regardless
    # of which ProgressMetric variant is active or how many ticks the
    # no-improvement window allows. Catches runaway trajectories by their
    # actual signature (box moving AWAY from best) instead of by tick count.
    # Default 0.030m; CP1 of the combined-fix plan pins the YAML value
    # against working-seed wobble. Set <= 0 to disable.
    pos_regression_threshold:            float = 0.030

    # kConfigCostDrop variant: required object-config cost drop over a
    # sim-time window. 2026-06-25 reconciliation: source-of-truth in
    # seconds. Default preserves 100 Hz value (30 ticks × 10 ms = 300 ms).
    # Reference (anything/c3plus_progress.yaml:45) = 16 ticks @ 1 kHz = 16 ms
    # (19× sim-time gap — OPEN alignment question).
    progress_enforced_cost_drop:         float = 0.0
    progress_enforced_over_duration_s:   float = 0.30

    # Distance below which we use the _position hysteresis variant
    cost_switching_threshold_distance:   float = 0.05

    # Reference-aligned arrival penalty (dairlib
    # sampling_based_c3_controller.cc:604-608 and 769-776).  When the
    # IK tracker reports that the EE is within tolerance of the pursued
    # repos slot, this value is added to that slot's c_sample.  The
    # inflation is large enough (≫ typical c_sample magnitudes) that
    # `repos_target_cost > finished_reposition_cost` cleanly distinguishes
    # arrival from a stable raw-cost regime, which is what
    # decide_mode() uses to assign kToC3ReachedReposTarget vs kToC3Cost.
    finished_reposition_cost:            float = 1.0e9

    # Absolute hysteresis (used when use_relative_hysteresis is False)
    hyst_c3_to_repos:                    float = 1000.0
    hyst_c3_to_repos_position:           float = 5000.0
    hyst_repos_to_c3:                    float = 1000.0
    hyst_repos_to_c3_position:           float = 5000.0
    hyst_repos_to_repos:                 float = 500.0
    hyst_repos_to_repos_position:        float = 2500.0

    # Steps-since-improve watchdog (1d, 9.4.7 Option A re-test).
    # When > 0, the wrapper overrides mode_switch and forces "c3" with
    # SwitchReason.kForceC3Watchdog once steps_since_improve >= this
    # threshold. 0 disables (default). Set to 100 in
    # config/sampling_c3_{params,kik}.yaml for the F2-regime re-test.
    watchdog_steps_since_improve_threshold: int = 0

    # Relative hysteresis (used when use_relative_hysteresis is True)
    use_relative_hysteresis:             bool  = False
    # Note: upstream field names use _frac_position, NOT _position_frac. Keep verbatim.
    hyst_c3_to_repos_frac:               float = 0.05
    hyst_c3_to_repos_frac_position:      float = 0.10
    hyst_repos_to_c3_frac:               float = 0.9
    hyst_repos_to_c3_frac_position:      float = 0.9
    hyst_repos_to_repos_frac:            float = 0.02
    hyst_repos_to_repos_frac_position:   float = 0.05

    @classmethod
    def from_dict(cls, raw: dict) -> "ProgressParams":
        # 2026-06-25 reconciliation: back-compat shims for the three tick-int
        # fields that became sim-time floats. Old YAMLs auto-convert at load.
        for old_key, new_key in (
            ("num_control_loops_to_wait",          "num_control_loops_to_wait_s"),
            ("num_control_loops_to_wait_position", "num_control_loops_to_wait_position_s"),
            ("progress_enforced_over_n_loops",     "progress_enforced_over_duration_s"),
        ):
            if old_key in raw and new_key not in raw:
                old = int(raw[old_key])
                raw[new_key] = float(old) * 0.01
                print(f"[YAML-COMPAT] {old_key}={old} (ticks @ 100 Hz) "
                      f"→ {new_key}={raw[new_key]:.4f} s", flush=True)
                del raw[old_key]
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

    # Unsuccessful sample buffer (TODO #6 port — reference
    # generate_samples.cc:181-205 SampleAvoidsBadSpots + cc:2161-2205
    # AddToUnsuccessfulBuffer).  When a sample is judged unsuccessful
    # (reference: free→c3 transition; port also fires on
    # kToBetterRepos), its EE position is stored and future samples
    # within `unsuccessful_radius` are rejected.
    #
    # Defaults match reference `anything/parameters/sampling_params.yaml`
    # lines 32-36 (push_t doesn't override these).  Port initially used
    # 5-17× coarser values that made the buffer effectively broken —
    # positions 50 mm apart were treated as "same bad spot" and the
    # 100 mm pos-retention held entries across the whole tabletop.
    avoid_choosing_unsuccessful_samples:  bool  = True
    N_unsuccessful_sample_buffer:         int   = 10
    unsuccessful_radius:                  float = 0.010   # m — ref
    unsuccessful_pos_error_sample_retention:  float = 0.006   # m — ref
    unsuccessful_ang_error_sample_retention:  float = 0.05    # rad — ref

    # Geometry shared across multiple strategies
    sampling_radius:                     float = 0.13   # m, candidate-ring radius for cost eval (samples 1..n-1)
    repos_target_radius:                 float = 0.075  # m, IK proxy target — pusher just touching box
    # repos_target_radius derivation:
    #   box_half_extent (0.050) + pusher_radius (0.025) = 0.075 m
    # Target gap = 0 mm (just touching). IK tolerance is ±0.020 m, so the
    # actual EE landing falls in φ ∈ [-20, +20] mm. The formulator's
    # distance_threshold = 0.002 m admits the negative-φ landings; positive-φ
    # landings yield an empty LCS and the wrapper retries via
    # kToReposUnproductive. Self-correcting by design.
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
    # Reference sampling_params.yaml:50 `z_height` — the c3-mode EE z-freeze
    # plane AND the c3-entry altitude-ceiling base (cc:1290, cc:1759). The
    # reference keeps this SEPARATE from sampling_height (samples at 34mm
    # above ground, c3 tracks at 25mm, in reference-ground-relative terms).
    # None → fall back to sampling_height (legacy port behavior).
    z_height:                            Optional[float] = None

    # Face-normal projection (kFaceNormal strategy — Push-Anything §IV-B1):
    # samples are a point on a box face projected outward along the face's
    # outward normal by sampling_setback. Rejection step drops samples whose
    # post-projection xy is still within sample_reject_clearance of the box
    # surface.
    box_half_extent:                     float = 0.05   # m, half-extent of the cube
    sampling_setback:                    float = 0.030  # m, outward projection along face normal (pusher_radius 0.025 + 5 mm margin)
    sample_reject_clearance:             float = 0.005  # m, post-projection minimum gap to box surface
    # Per-tick collision clearance for the PURSUED repositioning target
    # (reference sample_projection_clearance,
    # push_t/parameters/sampling_params.yaml:30 = 0.02). Reference
    # cc:908-926 flags the retained target in_collision when its signed
    # distance to any object geometry is <= this; cc:931/cc:1205-1213 then
    # exclude/reject it so the tracker never presses the EE into a face
    # that rotated under the target. Distinct from generation-time
    # sample_reject_clearance above.
    sample_projection_clearance:         float = 0.02   # m — reference value

    # kRandomOnPerimeter sampling window (body-frame bounding box).
    # Reference push_t/parameters/sampling_params.yaml:39-40 grid_x/y_limits.
    # When None, sampling.py:709-716 falls back to shape-hardcoded defaults
    # ([-0.07, +0.13] × [-0.08, +0.08] for tshape) which do NOT match ref.
    # Setting explicit values via yaml override enables ref-conformant window.
    grid_x_limits:                       list | None = None
    grid_y_limits:                       list | None = None

    # Object-shape selector for the kFaceNormal sampler.
    #   "box"    → 4 cardinal ±x/±y body-frame face patches (unchanged, regression-safe)
    #   "tshape" → 8-patch T-shape face table (single-body-collapsed T; see
    #              sampling.py:_TSHAPE_FACE_TABLE for the geometry constants).
    # The reference's generic path is kMeshNormalMultiObject (mesh triangle-face
    # normals filtered by |n_z|² < 0.035) — a future-work generalization.
    # The port's hardcoded T table is faithful in OUTCOME for the T but not
    # to the reference's generic mesh sampler; that's a documented fidelity
    # boundary, not a shortcut. Adding the mesh sampler unlocks the shape zoo.
    object_shape:                        str   = "box"

    # Face-selection bias toward goal-aligned faces (Stage 2B Mode-B fix).
    # When > 0, each face's draw probability is weighted by
    #   w_i = 1 + face_bias_strength * max(0, -n_world_i . g_hat_xy)
    # so the face whose outward normal points opposite the goal direction
    # (i.e., contact on that face pushes the box toward the goal) is over-
    # represented. 0.0 -> uniform 1-of-4 (regression-safe identity).
    face_bias_strength:                  float = 0.0

    # T1c — kMeshNormal area-weighted face pick. Reference
    # generate_samples.cc:454 draws faces proportional to face area
    # (binary search into cumulative bins) and then uniform-within-face
    # (barycentric_bias=1). When True, replaces the uniform (or goal-align-
    # biased) face selection in _face_normal_projection with a categorical
    # distribution proportional to face area. Uniform-within-face is
    # unchanged (the port already samples uniformly on its rectangular
    # patches — matches barycentric_bias=1 semantics for its geometry).
    #
    # For box: all 4 faces have equal area, so area-weighted == uniform
    # (no-op).
    # For T: small side faces (half_len 0.02) get 3.85 % each, medium
    #        (half_len 0.03) get 5.77 %, large side + top faces (half_len
    #        0.08, area 0.0064 m²) get 15.4 % each — vs the pre-T1c uniform
    #        10 % per face across 10 faces.
    # Orthogonal to face_bias_strength (goal-align): when True, area-weighting
    # takes priority over goal-align; goal-align becomes a no-op.
    use_mesh_normal_area_weighting:      bool  = False

    # D.3 (2026-07-13) — yaw-torque alignment face bias. When > 0 AND
    # the task has a nonzero yaw_delta (target_yaw − current_yaw), each
    # face's draw probability is multiplied by
    #   w_i = 1 + yaw_face_bias_strength * max(0, sign(yaw_delta) · τ_z_i)
    # where τ_z_i = ry_i·nx_i − rx_i·ny_i is the per-unit-force z-torque
    # induced by a normal push at that face's world-frame centre. Faces
    # whose contact torque matches the goal-yaw direction are over-
    # represented; anti-aligned faces get no bonus. Multiplies onto the
    # existing area / goal-align weights (composes cleanly with either).
    # Default 0.0 → regression-safe identity. Set on the T-config to
    # break the seed-0 face-2/face-4 tie in favour of the goal-aligned
    # torque face (see 2026-07-13 D.2/D.3 writeup).
    # Box: box has 4 cardinal side faces with rx·ny − ry·nx = 0
    # identically (a single-face push cannot yaw a cube about its z-axis),
    # so the bias is a null identity for the box shape regardless of the
    # yaml value. Bit-identical for box.
    yaw_face_bias_strength:              float = 0.0

    # Workspace bounds (kept here, not in a separate sampling_c3_options.yaml)
    workspace_xy_min:                    list  = field(default_factory=lambda: [-0.5, -0.7])
    # F3 ship 2026-05-14: y_max raised from 0.0 to 0.13 to match sampling_radius.
    # See journal 2026-05-14 and the F3 commit message for the Phase 2 sweep receipts.
    workspace_xy_max:                    list  = field(default_factory=lambda: [ 0.5,  0.13])
    workspace_z_min:                     float = 0.02
    workspace_z_max:                     float = 0.30

    # Port of reference `robot_radius_limits` (dairlib
    # sampling_c3plus_options.yaml:32). Radius shell on EE sqrt(x²+y²).
    # Reference `SamplingC3Controller::CheckForWorkspaceLimitViolations`
    # (sampling_based_c3_controller.cc:1487-1493) DRAKE_DEMANDs
    # r_min² < x² + y² < r_max². Default here is effectively unbounded
    # (0..100 m) so legacy callers see no behaviour change unless the yaml
    # sets an explicit shell. push_t and jacktoy reference yaml both use
    # [0.25, 0.75]; anything c3plus uses [0.28, 0.75].
    robot_radius_limits:                 list  = field(
        default_factory=lambda: [0.0, 100.0])

    # Port of reference `CheckForWorkspaceLimitViolations` DRAKE_DEMAND
    # (sampling_based_c3_controller.cc:1476-1494). When True, the workspace
    # + radius check raises RuntimeError on first violation (matching
    # reference's abort-on-exit intent). When False, the port keeps the
    # existing soft-log `[WORKSPACE-VIOLATION]` behaviour (per port-todo #5
    # note: "Consider adding a hard-abort mode when running non-interactively
    # — a production run should fail-fast").
    strict_workspace:                    bool  = False

    # Safety filter — drop samples that fail workspace or surface-clearance check
    filter_samples_for_safety:           bool  = True

    # Number of control loops a random ring sample buffer persists before
    # DEAD FIELD (2026-07-28e): no longer read anywhere. The position cache
    # it governed was removed in the 2026-07-19 fresh-samples refactor
    # (reference GenerateSampleStates regenerates every tick, no caching);
    # the residual no-op machinery (_refresh_buffer_on_arrival,
    # _sample_buffer* fields) was deleted 2026-07-28e. Field + the
    # from_dict compat shim are kept only so old YAMLs still load.
    sample_buffer_lifetime_s:            float = 0.30

    # Retreat position when achieved_fixed_goal_ latches.
    # Reference sampling_based_c3_controller.cc:887-897: when the object
    # is at goal, all candidate_states are replaced with a fixed
    # "get out of the way" position `head(3) << 0.3, 0.4, 0.1`.
    # 2026-07-21: with Franka base moved to reference-conformant world
    # origin (was port-only y=-0.6) and T workspace widened to reference
    # (y∈-0.6..0.6, x∈0.15..0.75), the reference literal (0.3, 0.4, 0.1)
    # is now workspace-legal and adopted directly.
    retreat_ee_position:                 list  = field(
        default_factory=lambda: [0.3, 0.4, 0.1])

    @classmethod
    def from_dict(cls, raw: dict) -> "SamplingParams":
        # Back-compat shim: old YAML used `sample_buffer_lifetime` (int ticks).
        # Convert to `_s` (sim-time) so old YAMLs still load at 100 Hz semantics.
        if ("sample_buffer_lifetime" in raw
                and "sample_buffer_lifetime_s" not in raw):
            old = int(raw["sample_buffer_lifetime"])
            raw["sample_buffer_lifetime_s"] = float(old) * 0.01
            print(f"[YAML-COMPAT] sample_buffer_lifetime={old} (ticks @ 100 Hz) "
                  f"→ sample_buffer_lifetime_s={raw['sample_buffer_lifetime_s']:.4f} s",
                  flush=True)
            del raw["sample_buffer_lifetime"]
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
    pwl_waypoint_height:                           float = 0.15  # safe-height m (25 mm above 0.10 box top + 0.025 pusher; reclaims horizon budget — see kik.yaml)
    # Per-leg Cartesian speed for the gated RepositionTrajectory PWL path
    # (Stage A). Separate from `.speed` (which the legacy IK tracker
    # consumes as a planning-lookahead stride). Reference push_t value
    # is 0.18 m/s (examples/sampling_c3/push_t/parameters/reposition_params.yaml).
    pwl_speed:                                     float = 0.18  # m/s

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
# RepositionIKParams — DEAD CODE, kept only so any external caller loading
# an old yaml with `repos_ik_params:` doesn't crash. The kIK reposition
# tracker (control/sampling_c3/reposition_ik.py) was removed as part of
# the reference-only cleanup — the reference dairlib
# push_anything_dev@257e3ed only ships PiecewiseLinearTracker.
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
    q_nominal:                                  list  = field(
        default_factory=lambda: [+0.552150, +0.325037, +0.976275,
                                  -2.246164, -0.188979, +3.044706, +0.785000]
    )  # J2 reduced 0.35 rad from INITIAL_ARM_Q[1]=0.675 to 0.325 (≈19°).
       # Lowers IK's nominal shoulder posture so gravity load on J2 at q*
       # doesn't saturate the 30 N·m budget. Prior nominal (=INITIAL_ARM_Q)
       # produced q*[1]≈0.90 rad, where the gravity-comp term consumed ~28
       # of 30 N·m, leaving the PD only ~2 N·m of proportional headroom and
       # ~2.3° of unresolvable residual on J2 — see
       # results/tracker_bias_north.log. Other 6 joints unchanged.

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

    # Option A noise-floor reducer (default OFF — identity behavior):
    # on knot[0] IK failure, q_arm_sol == q_warm, so the default
    # p_des == FK(q_warm) ≈ ee_now commands the executor to stay put,
    # cascading a missed-motion tick into a 25-35cm trajectory bifurcation
    # (verified at step 103 of nondet_seed0_serial_16s_pair). Enabling this
    # flag substitutes p_des := self._last_good_p_des (cached previous
    # successful target) when knot[0] fails — keeps the executor on the
    # last-known reachable target rather than stalling. Does NOT confer
    # bit-determinism (Ipopt FP-noise is sub-ULP per call, not confined to
    # failures — see project_b3b_refuted_paired_bit_identical.md), but
    # removes the catastrophic 35cm failure-cascades, leaving only the
    # silent FP-drift floor. Use as a noise-floor reducer for ablation
    # studies / video runs.
    hold_last_good_p_des_on_failure:            bool  = False

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
    # Rotation-aware sample bonus (analog of w_align for pure-rotation tasks).
    # Rewards samples whose off-center contact produces a moment M_z = (r x n).z
    # that turns the box toward goal_yaw. Only active when both w_rot > 0 and
    # the task has w_yaw > 0 (i.e. cost has a rotation goal); inert otherwise.
    w_rot:              float = 0.0

    # Inner-solver knobs
    surrogate_admm_iters: int = 1   # for the K-1 cheap sample evaluations

    # ---- Parallel sample evaluation (port-todo #1) -----------------------
    # Port of reference `num_outer_threads`
    # (sampling_c3plus_options.yaml:6, sampling_based_c3_controller.cc:415-422,
    # :971 `#pragma omp parallel for num_threads(num_threads_to_use_)`).
    # When > 1, InnerSolver.evaluate_samples dispatches per-sample
    # C3 evaluations across a pool of thread-owned (plant_ctx, LCSFormulator,
    # C3Solver) kits via concurrent.futures.ThreadPoolExecutor. Default 1
    # preserves bit-identical serial behaviour. Reference push_t sets 4;
    # our default is 1 pending the pool warm-up cost measurement.
    num_threads_to_use: int = 1

    # ---- Explicit manipuland-ground contact synthesis (§9 Option A) ------
    # When > 0, LCSFormulator.extract_lcs_contacts appends N synthesized
    # manipuland-bottom-face ↔ ground contact rows (in addition to Drake's
    # EE-manipuland admits; the Drake-auto-admitted single BOX-GND pair is
    # DE-DUPLICATED to avoid double-counting). Matches reference push_t
    # resolve_contacts_to=[0,1,3] (3 T-ground pairs) — captures the T's
    # distributed bottom-face friction so the planner's LCS models real
    # torsional resistance to yaw. Default 0 preserves prior behavior.
    # Env var LCS_EXPLICIT_MANIPULAND_GND takes precedence (backward compat:
    # LCS_EXPLICIT_BOX_GND also honored).
    lcs_explicit_manipuland_ground_contacts: int = 0

    # ---- §9 Option B (Stage 2) — cost-LCS forward-sim ranking ------------
    # When use_cost_lcs_ranking=True, InnerSolver.evaluate_sample computes the
    # sample cost by (a) forward-simulating the planner's u_seq under
    # PD-with-feedforward on the LCS, and (b) scoring the SIMULATED
    # trajectory with an object-only quadratic cost. Mirrors reference
    # `TrajectoryEvaluator::SimulatePDControlWithLCS` + `CalcCost` with
    # cost_type=5 (kSimImpedanceObjectCostOnly). Disabled by default →
    # keeps Stage-1 behavior (planner's own x_seq + object-only Q).
    use_cost_lcs_ranking: bool = False
    # Reference push_t/parameters/sampling_c3plus_options.yaml:
    #   Kp_for_ee_pd_rollout: 100
    #   Kd_for_ee_pd_rollout: 0.5
    # Scalars broadcast to per-axis EE PD gains during cost simulation.
    Kp_for_ee_pd_rollout: float = 100.0
    Kd_for_ee_pd_rollout: float = 0.5
    # PGS LCP knobs (Tikhonov regularization matches reference
    # simulate_config.regularized=true, min_exp=-8).
    cost_lcs_pgs_max_iter: int = 50
    cost_lcs_pgs_tol: float = 1.0e-6
    cost_lcs_pgs_reg: float = 1.0e-8

    # ----- Planner workspace state constraints (reference cc:995-1025) ----
    # Reference sampling_c3_options.yaml:26-30 workspace_limits +
    # workspace_margins: every per-sample C3 object gets hard linear STATE
    # rows keeping EE position AND object position inside each half-space
    # (bounds widened by the margin). Port applies them as per-knot
    # BoundingBoxConstraints in _solve_c3plus via
    # C3Solver.state_position_bounds (main.py wiring; EE slots 0-2, object
    # position slots 7-9 of the EE-space state). None → no constraint
    # (legacy behavior for configs that do not opt in).
    planner_workspace_x: Optional[list] = None   # [lo, hi] m, world frame
    planner_workspace_y: Optional[list] = None
    planner_workspace_z: Optional[list] = None
    planner_workspace_margin: float = 0.02       # reference workspace_margins

    # ----- Executor -----
    # The wrapper always instantiates `OperationalSpaceController` (a
    # per-tick QP). The alternate closed-form impedance executor was
    # removed; see git history if a comparison/ablation is needed.
    osc_gains_yaml: str = "config/osc_franka.yaml"

    # ------------ Force-tracking executor knobs ---------------------------
    # When True, OSC promotes the planner's contact force to a QP decision
    # variable (λ_ext) softly tracked toward `lambda_des` instead of being
    # added as a fixed RHS feedforward. Mirrors dairlib reference's
    # ExternalForceTrackingData (franka_osc_controller.cc:168-188).
    use_force_tracking: bool = True
    # Soft cost weight on ‖λ_ext − λ_des‖². Reference's W_ee_lambda.
    # Comparable to W_track (100.0) so neither dominates.
    # Reference LambdaEndEffectorW = diag(1,1,1) (osc_params.yaml:74).
    # 2026-07-28 defaults flip: 100.0 → 1.0 (was the REFCONF_OSC_ALIGN
    # runtime override).
    W_force: float = 1.0
    # When the LCS has no admitted EE-BOX pair at knot 0, command this
    # magnitude of recoil force in the -g_hat direction so the executor
    # keeps pressing rather than letting the command collapse to zero.
    #
    # Reference c3-mode force target = u_sol (planner's 3D EE-space control),
    # QP-bounded by u_horizontal_limits=[-10,10] N and u_vertical_limits=
    # [-3,3] N (a HARD constraint rarely hit — average u_sol is smaller).
    # Port fabricates from -g_hat and caps here (R^7 mode can't use u_sol
    # directly). 5 N chosen empirically to match reference AVERAGE force
    # (not the [-10,10] hard cap — raising to 10 tumbled the box).
    nominal_push_force: float = 5.0
    # When λ_n is admitted but very small, floor the commanded magnitude
    # at this value so the executor doesn't fall below the friction
    # threshold during marginal contact predictions.
    min_push_force: float = 2.0

    # ------------ Stage A — Reposition PWL trajectory port ---------------
    # When True, the dispatcher bypasses the legacy per-tick setpoint
    # march + per-knot IK + joint-PD path (RepositionIKTracker /
    # PiecewiseLinearTracker) and instead builds a full N-knot Cartesian
    # PWL trajectory (RepositionTrajectory) at planner cadence, feeding
    # (p_des, v_des) to the OSC at each control tick. Default False →
    # legacy path. Read from env var REFCONF_REPOSITION_PWL=1 in main.py
    # at controller construction. See alignment plan §3 Stage A.
    # Reference uses derivative-carrying PWL trajectory (LcmTrajectoryReceiver
    # → FirstOrderHold PP) so OSC gets (p_des, v_des). Default ON.
    use_reposition_pwl_trajectory: bool = True

    # ------------ Contact-proximity entry-gate knobs ----------------------
    # When True, the kToC3ReachedReposTarget trigger requires both
    # IK-finished AND ee-to-box-center ≤ contact_entry_threshold. Without
    # this gate, ReachedReposTarget fires at the IK's 20mm tolerance to the
    # 30mm-setback target, leaving the EE ~35mm shy of contact -> LCS
    # admits no EE-BOX pair -> contact-loss exit after 5 steps. (12/13
    # canonical c3 sessions failed this way before the fix.)
    # Port-only gate — reference has no contact-proximity entry gate.
    # Reference dispatcher relies on cost hysteresis + progress tracker.
    use_contact_entry_gate: bool = False
    # Threshold on ‖ee_now − box_center‖ in meters. Default 0.090 m
    # (loosened from 0.080 after the both_fixes_20260521_193033 run
    # found that IK arrivals systematically land at 80-95 mm — a 0.080
    # gate blocked all 10 arrivals and zero c3 sessions started). With
    # force-tracking active the executor can bridge 5-10 mm of gap by
    # commanding sustained push, so admitting entries up to 90 mm is
    # safe and gives force-tracking a chance. The very-far entries
    # (>90 mm; 3 of 10 in that run) remain blocked. Drake's LCS admit
    # threshold sits at 50+25+2=77 mm — 90 mm leaves a 13 mm gap that
    # force-tracking must close.
    contact_entry_threshold: float = 0.090

    # Layer 2.6: surface-distance entry gate metric. The original
    # `contact_entry_threshold` measures ‖ee − box_center‖, which
    # penalises tangentially-offset (torque-producing) rotation samples
    # for being further from the CoM. The surface metric
    # ‖ee − box_center‖ − box_half_extent measures distance to the
    # nearest face, which is the geometrically relevant quantity for
    # contact admission. Default 0.060 m derived from:
    #   - Face-center sample at 30 mm setback → surface dist 30 mm
    #   - IK arrival tolerance up to 20 mm → worst-case surface 50 mm
    #   - Off-center sample (y_offset 30 mm) → surface dist ~35 mm,
    #     worst-case arrival 55 mm
    #   - 60 mm passes both with margin; blocks pathological >60 mm cases.
    # When `use_surface_entry_gate` is True (default), this metric
    # supersedes contact_entry_threshold. Goal-agnostic — applies to
    # translation tasks too; threshold chosen to preserve their
    # engagement behavior.
    # Port-only surface-distance entry gate — reference has no such gate.
    use_surface_entry_gate: bool = False
    contact_entry_surface_threshold: float = 0.060

    # Stage 2 L1: goal-aligned contact-normal requirement at admission.
    # Applies AFTER the distance check passes. With -nhat_xy·g_hat as the
    # alignment cosine (+1 = perfect goal-ward contact, 0 = perpendicular,
    # -1 = anti-goal): admit c3 only when align > entry_align_threshold.
    # 0.0 → identity / disabled (regression-safe default).
    # 0.7 → reuses GOAL_ALIGN_THRESHOLD convention (sampling.py:31, ~45°
    # cone), refuses wrong-face cardinal contact (alignment 0.04) AND
    # off-cardinal/edge contact (alignment ≤ 0.5) with one cosine check.
    entry_align_threshold: float = 0.0

    # Stage 2 L2: commit-trigger face gate. Evaluate the active
    # self._current_repos_target's face: n_face_out · g_hat must be
    # ≤ commit_face_gate_threshold for c3 to commit. Default +0.3
    # admits any face whose projection onto -g_hat is positive
    # (i.e. the EE is anywhere on the goal-anti hemisphere, not just
    # strictly on the goal-anti face). For push-west: +x (cos=-1) and
    # ±y (cos=0) admit; -x (cos=+1) refuses.
    #
    # Default +0.3 reverted from 0.0 after the Q7b OVERLAP read.
    # The 0.0 sweep fixed seed-2 (drift +0.250 -> -0.050, goal_dist 0.141)
    # but REGRESSED seed-0's clean win (drift +0.021 -> +0.198, goal_dist
    # 0.177 -> 0.234). The cheap read of seed-0's 0.0 [GATE-COMMIT-FACE]
    # refusals showed seed-0's productive entries at face_align = +0.025,
    # +0.175, +0.272, +0.273 — INTERLEAVED with seed-2's drift entries
    # at +0.190, +0.269. No threshold separates them (arithmetic-proven
    # OVERLAP): every threshold in (0, +0.3] either kills seed-0's win
    # or admits seed-2's drift. face_align is an INCOMPLETE predictor of
    # push direction; the same ~+0.2 entry geometry is productive for
    # seed-0 and drift-causing for seed-2. The residual is the EE
    # contact-point / off-equator offset — the SAME mechanism as
    # seed-4's face-correct (-0.74 to -0.99) drift entries.
    # Reverting to +0.3 keeps seed-0's validated clean win; seed-2 is
    # covered by the off-axis investigation alongside seed-4 (universal
    # residual). The -0.497 productive session survives at +0.3 margin
    # 0.797 (noregress pin).
    #
    # Two gate sites in wrapper.py:
    #   - pre-decide (line ~983): mutates finished_repos to suppress
    #     kToC3ReachedReposTarget.
    #   - post-decide (line ~1024, plan 2026-06-10): override mode
    #     -> "free", reason -> kStayInRepos when prev_mode=="free"
    #     and decide_mode returned "c3". Catches kToC3Cost and any
    #     other free->c3 transition the pre-decide site doesn't
    #     cover.
    #
    # Distinct from L1 (entry_align_threshold above): L1 keys on
    # formulator._last_contact_info which is empty 80 mm pre-contact;
    # L2 keys on _current_repos_target which is populated by
    # definition when finished_repos==True.
    # Port-only face-selection gate on c3 entry — reference doesn't gate by
    # face alignment; dispatcher lets the sample scorer sort candidates.
    use_commit_face_gate: bool = False
    commit_face_gate_threshold: float = 0.3

    # ------------ T1a — EE_z altitude mode-switch gate --------------------
    # Reference sampling_based_c3_controller.cc:1290-1293. Blocks c3 mode
    # entry from free while the pusher is above the sampling-height ceiling
    # (sampling_z + c3_min_clearance). Forces the pusher to descend before
    # c3 dispatches, so the first c3-mode tick starts near the contact plane
    # rather than dive-and-whack from 100+ mm above. Complementary to the
    # port's per-tick ADMIT-GUARD (LCS admission latch) and ALT-GATE (descent
    # permission) — those latch each tick; this is a one-shot mode-switch
    # gate applied AFTER decide_mode returns, covering BOTH
    # kToC3ReachedReposTarget (via finished_repos) and kToC3Cost (via cost
    # hysteresis). The port's pre-decide gates only cover the first path.
    #
    # Defaults match reference: ee_z_close=True, c3_min_clearance=0.01 m.
    ee_z_close: bool = True
    c3_min_clearance: float = 0.01

    # ------------ Contact-loss disengage thresholds -----------------------
    # The contact-loss gate exits c3 when `_no_ee_box_streak` consecutive
    # rich-mode steps had no admitted EE-BOX pair. Conditioned on whether
    # the approach-closing override (Lever 3) was firing on the previous
    # tick: when the override is actively trying to close a ~6 mm gap that
    # the OSC's tracking lag has not yet closed, give it more time before
    # bailing. When the override is NOT firing (no proximity reason to be
    # in c3), use the strict default. The `with_override` value is the
    # hard cap on grace — if the override fires for that many ticks
    # without LCS admitting a pair, the EE is structurally unable to
    # reach contact and we bail.
    # 2026-06-25 reconciliation: source-of-truth in seconds. Defaults
    # preserve 100 Hz values (5/12 ticks × 10 ms = 50/120 ms). NO reference
    # analog (reference has no contact-loss disengage counter). Port-only
    # candidate band-aid; alignment-status-OPEN.
    contact_loss_threshold_default_s: float = 0.05
    contact_loss_threshold_with_override_s: float = 0.12
    # LTD PHASE A traverse needs ~80-110 ticks at realized lateral rate
    # (~0.8 mm/tick observed) to cover the box_half + clearance ~ 75 mm
    # to W_side. With the `_with_override` value of 12 ticks, the gate
    # killed PHASE A 9× too early during LTD smoke tests. PHASE A holds
    # EE.z at z_safe (above box top) under active z-Kp tracking, so the
    # earlier objection to a longer timer ("EE has more time to fall
    # onto the top") does not apply in PHASE A specifically. The threshold
    # also acts as a stuck-watchdog: if PHASE A can't form contact by
    # this many ticks, the system gives up and the dispatcher routes to
    # free mode. 120 ≈ 1.5× the expected 80 ticks.
    contact_loss_threshold_phaseA_ltd_s: float = 1.20   # was 120 ticks @ 100 Hz
    # PHASE B (descend beside the face from z_safe down to face-centroid z)
    # needs ~215 ticks at the realized ~0.84 mm/tick rate to cover
    # ~150 mm of vertical travel. Per the PHASE-B lateral-clearance probe
    # (ee.x stays 4–16 mm east of the face plane through the entire
    # descent, monotonically drifting outward toward W_side), the
    # fall-onto-top objection that motivated the strict default does not
    # apply: EE is laterally outside the box footprint, so a free-mode
    # interlude would fall east of the box, not onto its top. Extend the
    # threshold with the same 1.5× watchdog margin as PHASE A.
    contact_loss_threshold_phaseB_ltd_s: float = 3.00   # was 300 ticks @ 100 Hz

    # ------------ PHASE C progress-gated exit (Layer 2.5/2.6) -------------
    # Once the EE is in PHASE C (pushing into the face), the contact-loss
    # tick-count gate is the wrong productivity metric: the EE may sit
    # one tick from LCS admission and need only a few more ticks of
    # convergence. The C gate keys on surf_dist progress instead.
    #   * phaseC_stall_threshold — consecutive C ticks without
    #     surf_dist improving by ≥ phaseC_progress_eps. Fires even
    #     when the absolute time budget is small.
    #   * phaseC_hard_cap — absolute max active C ticks. Bounds the
    #     worst case even when surf_dist creeps in but never closes.
    #     Also used as the contact-loss tick-count budget during C
    #     (the elif _approach_override_phase=='C_approach' branch in
    #     wrapper.py) so the existing tick-count gate doesn't
    #     pre-empt the progress gate.
    #   * phaseC_progress_eps — minimum surf_dist improvement (m) to
    #     count as progress. Default 0.0002 m = 0.2 mm ≈ 0.1 × LCS
    #     admission threshold (2 mm), so noise-level oscillation
    #     does not register as progress.
    phaseC_stall_threshold_s: float = 0.30   # was 30 ticks @ 100 Hz
    phaseC_hard_cap_s: float = 1.00          # was 100 ticks @ 100 Hz
    phaseC_progress_eps: float = 0.0002

    # ------------ Velocity feedforward to OSC (bounded re-enable) ---------
    # `v_ee_desired` was set to None at commit 02c48e9 (2026-05-20). Reason
    # from the code comment at wrapper.py around the executor call:
    #   "the IK knot spacing produces a much larger effective velocity
    #    than the task tracking can absorb without saturating every joint
    #    at URDF limits. Revisit once the OSC baseline (position-only
    #    tracking) is verified."
    # The closed-form decomp (audit_output/phaseC_gate_runs/seed4_diag.log,
    # parsed by /tmp/decomp_cmd_vs_realized.py) shows the PD law without
    # v_des settles at v_realized = -Kp·p_err/Kd = -0.1 m/s = -1 mm/tick
    # in PHASE B against a -10 mm/tick commanded delta — a 9.4% realized
    # ratio that EXACTLY matches the analytic prediction. Re-enabling v_des
    # closes this lag.
    #
    # BUT the original saturation concern is unobserved, not refuted: 0%
    # saturation today exists precisely BECAUSE v_des is off, which holds
    # the a_des target at Kp·p_err = 4 m/s². Re-enabling raises a_des
    # toward Kp·p_err + Kd·v_des ≈ 4 + 40·1 = 44 m/s² (full feedforward
    # against the 1 m/s commanded rate), well into the regime 02c48e9
    # feared. Implement as bounded feedforward — α·v_raw with α in (0,1]
    # — and sweep α with saturation as the load-bearing observable.
    #
    # Default off so the flag is opt-in; current behavior preserved bit-
    # identically when `use_velocity_feedforward=False`.
    use_velocity_feedforward: bool = False  # reference has ydot_des but at α=1 port saturates/spikes (per docstring). Kept off; would need bounded α.
    # Scale on v_des derived from successive p_ee_des. α=1.0 is full
    # feedforward (analytically eliminates the steady-state lag). Lower α
    # trades descent rate for actuator headroom.
    #
    # A/B sweep at α ∈ {0.25, 0.5, 1.0} (audit_output/phaseC_gate_runs/
    # vff_ab/, seed4 pushing-W, 4s, default-off baseline as control)
    # measured saturation 0.25%, 3.24%, 17.96% and final goal-ward box
    # Δx 1.1 mm, 16.6 mm, 34.4 mm. α=0.25 destabilizes approach without
    # producing push (0 PHASE-B/C ticks, transient contact bursts).
    # α=1.0 hits 18% saturation, materializing 02c48e9's safety concern
    # (joints sit at URDF cap on ~1/5 ticks). α=0.5 is the operating-
    # point pick: first in-regime SC3 push in the 5-50 mm bin (16.6 mm
    # goal-ward), bounded saturation (3.24%, p99 util 1.000 only at a
    # handful of ticks). Flagged PROVISIONAL — cross-track y is 18.3 mm
    # at α=0.5 (push direction ~48° off goal), so yaw direction-loss is
    # the next SC3 wall; revisit α after yaw is fixed.
    velocity_feedforward_alpha: float = 0.5
    # Per-axis clip on the raw |v_des| before α scaling. Prevents pathological
    # spikes when p_ee_des jumps across discontinuities (e.g. mode change,
    # phase change with target re-aim). 1.5 m/s leaves 50% headroom over
    # the LTD per-tick advance velocity (1.0 m/s = 10 mm / 0.01 s).
    velocity_feedforward_v_max: float = 1.5

    # ------------ T-architecture rate-split knobs (Stage 1 substrate) -----
    # Stage 1 introduces these as separate dials defaulting to dt_ctrl=0.01s.
    # Stage 2 will gate the CI-MPC re-solve on dt_mpc boundaries while the
    # OSC fires every dt_osc. With defaults equal to dt_ctrl, the system
    # behaves identically to pre-Stage-1 (tight coupling preserved).
    dt_osc: float = 0.01   # OSC tick period (sec); defaults to dt_ctrl
    dt_mpc: float = 0.01   # CI-MPC re-solve period (sec); defaults to dt_ctrl

    # ------------ Lift-Traverse-Descend (LTD) override geometry -----------
    # The contact-free override (wrapper.py face-picker block) used to aim
    # a direct line at the face centroid. From above-box starts that line
    # was 67° below horizontal → EE descended onto the box top before
    # reaching the side face. Stage-3 sweep with the directional picker
    # but legacy direct-line target: 30/30 EE-BOX events landed on TOP
    # face (nhat≈[0,0,+1]), 3.82 mm box motion across the one seed of 20
    # that completed.
    #
    # LTD routes the override's approach through a beside-box waypoint at
    # face mid-height, with a lift-above-box-top traverse phase if needed.
    # Three phases (stateless, decided per-tick from EE geometry):
    #   A: lift-and-traverse — aim above-and-beside box at face-x/y
    #   B: descend           — aim at W_side (beside box, face mid-height)
    #   C: approach          — aim at face centroid (z rigidly clamped)
    # Port-only lift-traverse-descend override for approach path. Reference
    # relies on the PWL reposition trajectory (which itself does lift/traverse/
    # descend via z_safe). Disabling to remove the redundant approach shaper.
    use_lift_traverse_descend_override: bool = False
    # PHASE B descent puts the sphere SURFACE at (clearance - PUSHER_RADIUS)
    # from the face plane. Floor is PUSHER_RADIUS + LCS_THRESHOLD + 5 mm
    # safety = 32 mm: smaller would admit contact mid-descent and re-
    # introduce the very bypass that motivated LTD. Asserted at every
    # override entry; never sweep below the floor.
    ltd_clearance: float = 0.050
    # PHASE A safe-traverse height above box top:
    #   z_safe = box.z + box_half + PUSHER_RADIUS + ltd_z_margin
    # Margin > LCS_THRESHOLD (2 mm) so accidental grazing doesn't admit.
    ltd_z_margin: float = 0.010
    # PHASE A → B transition: lateral distance to W_side below which the
    # override switches from lift-and-traverse to descend. Sized above
    # typical OSC steady-state xy error to prevent boundary ping-pong.
    ltd_xy_tol: float = 0.020
    # PHASE B → C transition: z above W_side at which the override
    # switches from descend to approach. Orthogonal to ltd_xy_tol so the
    # two boundaries cannot couple into a single oscillating state.
    ltd_z_band: float = 0.005

    @classmethod
    def from_dict(cls, raw: dict) -> "SamplingC3Params":
        return cls(
            progress_params   = ProgressParams.from_dict(raw.get("progress_params", {}) or {}),
            sampling_params   = SamplingParams.from_dict(raw.get("sampling_params", {}) or {}),
            reposition_params = RepositionParams.from_dict(raw.get("reposition_params", {}) or {}),
            repos_ik_params   = RepositionIKParams.from_dict(raw.get("repos_ik_params", {}) or {}),
            w_align              = float(raw.get("w_align", 30_000.0)),
            w_travel             = float(raw.get("w_travel", 200.0)),
            w_rot                = float(raw.get("w_rot", 0.0)),
            surrogate_admm_iters = int(raw.get("surrogate_admm_iters", 1)),
            num_threads_to_use   = int(raw.get(
                "num_threads_to_use",
                raw.get("num_outer_threads", 1))),  # ref name alias
            lcs_explicit_manipuland_ground_contacts = int(raw.get(
                "lcs_explicit_manipuland_ground_contacts", 0)),
            use_cost_lcs_ranking     = bool(raw.get("use_cost_lcs_ranking", False)),
            Kp_for_ee_pd_rollout     = float(raw.get("Kp_for_ee_pd_rollout", 100.0)),
            Kd_for_ee_pd_rollout     = float(raw.get("Kd_for_ee_pd_rollout", 0.5)),
            cost_lcs_pgs_max_iter    = int(raw.get("cost_lcs_pgs_max_iter", 50)),
            cost_lcs_pgs_tol         = float(raw.get("cost_lcs_pgs_tol", 1.0e-6)),
            cost_lcs_pgs_reg         = float(raw.get("cost_lcs_pgs_reg", 1.0e-8)),
            planner_workspace_x  = raw.get("planner_workspace_x", None),
            planner_workspace_y  = raw.get("planner_workspace_y", None),
            planner_workspace_z  = raw.get("planner_workspace_z", None),
            planner_workspace_margin = float(raw.get("planner_workspace_margin", 0.02)),
            osc_gains_yaml       = str(raw.get("osc_gains_yaml", "config/osc_franka.yaml")),
            use_force_tracking   = bool(raw.get("use_force_tracking", True)),
            W_force              = float(raw.get("W_force", 1.0)),
            nominal_push_force   = float(raw.get("nominal_push_force", 5.0)),
            min_push_force       = float(raw.get("min_push_force", 2.0)),
            use_reposition_pwl_trajectory = bool(raw.get(
                "use_reposition_pwl_trajectory", True)),
            use_contact_entry_gate    = bool(raw.get("use_contact_entry_gate", True)),
            contact_entry_threshold   = float(raw.get("contact_entry_threshold", 0.090)),
            use_surface_entry_gate         = bool(raw.get("use_surface_entry_gate", True)),
            contact_entry_surface_threshold = float(raw.get("contact_entry_surface_threshold", 0.060)),
            entry_align_threshold          = float(raw.get("entry_align_threshold", 0.0)),
            use_commit_face_gate           = bool(raw.get("use_commit_face_gate", True)),
            commit_face_gate_threshold     = float(raw.get("commit_face_gate_threshold", 0.3)),
            # T1a — EE_z altitude mode-switch gate (reference cc:1290-1293).
            ee_z_close                     = bool(raw.get("ee_z_close", True)),
            c3_min_clearance               = float(raw.get("c3_min_clearance", 0.01)),
            # 2026-06-25 reconciliation: tick-int → sim-time-float with
            # auto-conversion from old YAMLs. If the OLD int form is present
            # and the new _s float form is not, convert old × 0.01 (the
            # 100 Hz sim-time-equivalent) and print a [YAML-COMPAT] line.
            **{f"{new}_s": _resolve_legacy_int_to_seconds(raw, old, new, default_ticks)
               for old, new, default_ticks in (
                   ("contact_loss_threshold_default",       "contact_loss_threshold_default",       5),
                   ("contact_loss_threshold_with_override", "contact_loss_threshold_with_override", 12),
                   ("contact_loss_threshold_phaseA_ltd",    "contact_loss_threshold_phaseA_ltd",    120),
                   ("contact_loss_threshold_phaseB_ltd",    "contact_loss_threshold_phaseB_ltd",    300),
                   ("phaseC_stall_threshold",               "phaseC_stall_threshold",               30),
                   ("phaseC_hard_cap",                      "phaseC_hard_cap",                      100),
               )},
            phaseC_progress_eps    = float(raw.get("phaseC_progress_eps", 0.0002)),
            use_velocity_feedforward    = bool(raw.get("use_velocity_feedforward", False)),
            velocity_feedforward_alpha  = float(raw.get("velocity_feedforward_alpha", 0.5)),
            velocity_feedforward_v_max  = float(raw.get("velocity_feedforward_v_max", 1.5)),
            dt_osc = float(raw.get("dt_osc", 0.01)),
            dt_mpc = float(raw.get("dt_mpc", 0.01)),
            use_lift_traverse_descend_override = bool(raw.get("use_lift_traverse_descend_override", True)),
            ltd_clearance = float(raw.get("ltd_clearance", 0.050)),
            ltd_z_margin  = float(raw.get("ltd_z_margin",  0.010)),
            ltd_xy_tol    = float(raw.get("ltd_xy_tol",    0.020)),
            ltd_z_band    = float(raw.get("ltd_z_band",    0.005)),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SamplingC3Params":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)
