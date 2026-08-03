"""
Mode-switch decision — Venkatesh et al. RA-L 2025 §IV-D.

Inputs each control loop:

  prev_mode             "c3" or "free" (the mode active at the previous loop)
  c3_cost               c_sample at k=0 (current EE location), with the
                        alignment bonus already applied
  best_other_cost       lowest c_sample over all k != 0 candidates (the
                        most attractive repositioning target)
  current_repos_cost    c_sample of the sample we are currently navigating
                        toward in repos mode (None when prev_mode == "c3")
  met_progress          ProgressTracker.met_progress(near_goal) — True
                        when the configured progress metric has not timed out
  near_goal             box-to-goal Cartesian distance below the
                        cost_switching_threshold_distance threshold; selects
                        between the absolute-mode hysteresis variants
  finished_repos        True when reposition cost has fallen below
                        params.finished_reposition_cost (forces repos→c3)

Returns (mode, reason) where mode ∈ {"c3", "free"} and reason is the
SwitchReason enum identifying which branch of §IV-D was taken.

Pure-Python; no Drake dependency.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Optional

from control.sampling_c3.params import ProgressParams


class SwitchReason(IntEnum):
    """Why the controller chose the mode it did at this control loop.
    The string-form of each enum value is what gets logged in the [GS]
    diagnostic line and is what success-criteria checks grep for."""
    kStayInC3                = 0
    kStayInRepos             = 1
    kToReposCost             = 2   # c3 → free, lower-cost target found
    kToReposUnproductive     = 3   # c3 → free, progress-metric timed out
    kToC3Cost                = 4   # free → c3, current C3 plan beat repos
    kToC3ReachedReposTarget  = 5   # free → c3, reposition target reached
    kToBetterRepos           = 6   # free → free, switch repos target
    kForceC3Watchdog         = 7   # free → c3, steps_since_improve watchdog
                                   # (re-test of 1d under F2 regime, 9.4.7
                                   # Option A — see paper_alignment_plan
                                   # Item 2.1 post-F2 reframe)


class PursuedTargetSource(IntEnum):
    """Which sample the current reposition target came from (mirrors
    reference `enum PursuedTargetSource` at
    `dairlib_sampling_c3/systems/controllers/sampling_based_c3_controller.h:60`).
    Purely telemetry — no dispatcher branch consults this state.
    Derived from the port's existing string labels via
    `pursued_from_label(mode, best_src)` below."""
    kNoTarget   = 0   # in c3 mode (no reposition target)
    kPrevious   = 1   # reusing prior reposition target
    kNewSample  = 2   # freshly sampled
    kFromBuffer = 3   # reference cc:1192-1194 — pursued target came from the
                      # SAMPLE buffer (AugmentSamplesWithBuffer injection).
                      # STALE-COMMENT FIX 2026-08-03: the old note claimed the
                      # port "lacks the unsuccessful-sample buffer" — false
                      # since UnsuccessfulSampleBuffer landed (wired at
                      # sampling_based_c3_controller.py: instantiation,
                      # sample_avoids_bad_spots gate, prune+append on c3
                      # entry). Whether THIS enum value is emitted depends on
                      # the wrapper's label→source mapping below, not on that
                      # buffer.


def pursued_from_label(mode: str, best_src: str) -> "PursuedTargetSource":
    """Map the port's string label at each mode-switch tick to the reference
    PursuedTargetSource enum. In c3 mode, always kNoTarget. In free mode,
    'prev_repos' → kPrevious, 'strat_*' / other → kNewSample."""
    if mode == "c3":
        return PursuedTargetSource.kNoTarget
    if best_src == "prev_repos":
        return PursuedTargetSource.kPrevious
    return PursuedTargetSource.kNewSample


# ---------------------------------------------------------------------------
# Hysteresis lookup
# ---------------------------------------------------------------------------

def _hysteresis(params: ProgressParams,
                kind: str,             # "c3_to_repos" | "repos_to_c3" | "repos_to_repos"
                near_goal: bool,
                ref_cost: float) -> float:
    """Return the cost-gap threshold used by the configured hysteresis mode.

    Absolute mode: returns the hyst_* field (in cost units).
    Relative mode: returns hyst_*_frac (or _frac_position) × |ref_cost|.
    """
    # Reference (sampling_based_c3_controller.cc:1107-1114):
    #   if (!crossed_cost_switching_threshold_)         # far from goal
    #       hyst_c3_to_repos = hyst_c3_to_repos_position
    #   else                                            # near goal
    #       hyst_c3_to_repos = hyst_c3_to_repos
    # → _position variants apply FAR from goal (position-dominant regime);
    #   non-_position variants apply NEAR goal (pose-tracking regime).
    # Port previously inverted this (used _position when near_goal). Fixed.
    _use_position = not near_goal
    if params.use_relative_hysteresis:
        if kind == "c3_to_repos":
            frac = (params.hyst_c3_to_repos_frac_position if _use_position
                    else params.hyst_c3_to_repos_frac)
        elif kind == "repos_to_c3":
            frac = (params.hyst_repos_to_c3_frac_position if _use_position
                    else params.hyst_repos_to_c3_frac)
        elif kind == "repos_to_repos":
            frac = (params.hyst_repos_to_repos_frac_position if _use_position
                    else params.hyst_repos_to_repos_frac)
        else:
            raise ValueError(f"Unknown hysteresis kind: {kind}")
        return float(frac) * abs(float(ref_cost))

    # Absolute mode
    if kind == "c3_to_repos":
        return (params.hyst_c3_to_repos_position if _use_position
                else params.hyst_c3_to_repos)
    if kind == "repos_to_c3":
        return (params.hyst_repos_to_c3_position if _use_position
                else params.hyst_repos_to_c3)
    if kind == "repos_to_repos":
        return (params.hyst_repos_to_repos_position if _use_position
                else params.hyst_repos_to_repos)
    raise ValueError(f"Unknown hysteresis kind: {kind}")


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

def decide_mode(prev_mode:          str,
                c3_cost:            float,
                best_other_cost:    float,
                current_repos_cost: Optional[float],
                met_progress:       bool,
                near_goal:          bool,
                finished_repos:     bool,
                params:             ProgressParams,
                *,
                ee_z_gate_pass:     bool = True,
                ) -> tuple[str, SwitchReason]:
    """Return (next_mode, switch_reason). See module docstring for the
    contract on each input.

    T1a — ``ee_z_gate_pass`` (keyword-only, default True for backward
    compatibility) mirrors the reference's in-branch altitude condition at
    ``sampling_based_c3_controller.cc:1290-1293``. When False, both
    ``free → c3`` transitions (``kToC3ReachedReposTarget`` and ``kToC3Cost``)
    are suppressed inline — the code falls through to the repos re-target /
    stay-in-repos branches, matching the reference's implicit stay-in-repos
    behavior when the combined ``else if`` is false. This is the in-decision
    placement (not a post-decide override): decide_mode never returns c3
    when the gate is closed, so no mode-switch/hysteresis state ever latches
    from a c3 decision that would then be undone.
    """
    if prev_mode not in ("c3", "free"):
        raise ValueError(f"prev_mode must be 'c3' or 'free', got {prev_mode!r}")

    if prev_mode == "c3":
        # 1. Progress timeout — strongest reason to abandon C3
        if not met_progress:
            return "free", SwitchReason.kToReposUnproductive

        # 2. Cost-based switch — only if the gap exceeds hysteresis
        gap = _hysteresis(params, "c3_to_repos", near_goal, c3_cost)
        if best_other_cost + gap < c3_cost:
            return "free", SwitchReason.kToReposCost

        return "c3", SwitchReason.kStayInC3

    # prev_mode == "free"

    # Reference order (cc:1236-1243 then cc:1284-1293):
    #   1. Check repos-to-repos (would a new target be better?).
    #      If so, inflate best_other_cost IN-PLACE by the repos-to-repos
    #      hysteresis margin (reference cc:1241:
    #        best_other_cost += hyst_repos_to_repos_frac * repos_target_cost).
    #   2. Run c3-entry gate on the (possibly inflated) best_other_cost.
    #   3. If c3 gate fires → return c3.
    #   4. If c3 gate does NOT fire and the repos-to-repos condition held →
    #      return kToBetterRepos.
    #   5. Else → kStayInRepos.
    #
    # This ordering is critical: when finished_reposition_cost=1e9 inflates
    # c_samples[1] (the current repos target), a fresh strategy sample at
    # ~7000 becomes the new best_other.  Without inflation, c3 gate is
    # blocked (curr~5000 + 0.9*7000 = 11200 > 7000).  With the
    # repos-to-repos inflation applied first:
    #   best_other_inflated = 7000 + 0.3 * 1e9 ≈ 3e8
    #   c3 gate: 5000 + 0.9*3e8 = 2.7e8 < 3e8 → FIRES.
    # See diagnosis-v2.md §5 for the full derivation.

    _repos_to_repos_would_fire = False
    if current_repos_cost is not None:
        gap_repos = _hysteresis(params, "repos_to_repos",
                                near_goal, current_repos_cost)
        if best_other_cost + gap_repos < current_repos_cost:
            # Inflate best_other_cost in-place (mirrors reference cc:1241).
            best_other_cost += gap_repos
            _repos_to_repos_would_fire = True

    # c3-entry gate (reference cc:1284-1293) — runs AFTER potential inflation.
    if best_other_cost != float("inf") and ee_z_gate_pass:
        gap_back = _hysteresis(params, "repos_to_c3", near_goal,
                               best_other_cost)
        if c3_cost + gap_back < best_other_cost:
            _label = (SwitchReason.kToC3ReachedReposTarget
                      if finished_repos else SwitchReason.kToC3Cost)
            return "c3", _label

    # repos-to-repos: fire now if the condition held and c3 didn't win.
    if _repos_to_repos_would_fire:
        return "free", SwitchReason.kToBetterRepos

    return "free", SwitchReason.kStayInRepos
