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

    # Reference cc:1284-1303: c3 entry from free is COST-GATED, and the
    # label is chosen after the cost gate passes:
    #     if (best_other > curr + hyst_frac * best_other &&
    #         (ee_z < z_height + clearance || !ee_z_close)) {
    #         is_doing_c3_ = true;
    #         if (repos_target_cost > finished_reposition_cost) {
    #             reason = kToC3ReachedReposTarget;
    #         } else {
    #             reason = kToC3Cost;
    #         }
    #     }
    # Port previously had a SEPARATE direct-flag c3 entry (fired when
    # finished_repos flag set, bypassing cost gate). That fired c3 more
    # aggressively than reference. Now cost-gated with post-gate label
    # discrimination (finished_repos → kToC3ReachedReposTarget label).
    if best_other_cost != float("inf") and ee_z_gate_pass:
        gap_back = _hysteresis(params, "repos_to_c3", near_goal,
                               best_other_cost)
        if c3_cost + gap_back < best_other_cost:
            _label = (SwitchReason.kToC3ReachedReposTarget
                      if finished_repos else SwitchReason.kToC3Cost)
            return "c3", _label

    # 3. Re-target within repos mode (only if a current repos target exists)
    if current_repos_cost is not None:
        gap_repos = _hysteresis(params, "repos_to_repos",
                                near_goal, current_repos_cost)
        if best_other_cost + gap_repos < current_repos_cost:
            return "free", SwitchReason.kToBetterRepos

    return "free", SwitchReason.kStayInRepos
