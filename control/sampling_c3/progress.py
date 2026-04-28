"""
Progress tracker for the sampling-C3 mode-switch decision.

Implements the four ProgressMetric variants from
examples/sampling_c3/parameter_headers/progress_params.h:

  kC3Cost           tracks the running C3 trajectory cost; "progress" means
                    the C3 cost is still decreasing.
  kConfigCost       tracks the current object-config cost (~ box xy
                    distance² to goal); progress = decreased.
  kPosOrRotCost     tracks position OR rotation error to goal; progress
                    means either has improved within the window.
  kConfigCostDrop   stricter version of kConfigCost: requires the config
                    cost to drop by at least progress_enforced_cost_drop
                    within progress_enforced_over_n_loops.

For the project's push tasks there is no rotation goal, so kPosOrRotCost
collapses to "position improved within the window."

`met_progress(near_goal)` returns True when the configured metric is
still improving — meaning the no-progress timeout has NOT fired and we
should stay in C3 mode.
"""
from __future__ import annotations

from dataclasses import dataclass

from control.sampling_c3.params import ProgressMetric, ProgressParams


# ---------------------------------------------------------------------------
# Per-step metric snapshot
# ---------------------------------------------------------------------------

@dataclass
class StepMetrics:
    """Metrics evaluated at the current control loop.

    c3_cost     : c_sample[k=0] (post-alignment-bonus) — the running C3 cost
    config_cost : weighted squared distance of object to goal in q-space
                  (matches upstream kConfigCost — typically the box xy term)
    pos_error   : ||box_xy - goal_xy||₂ in metres
    rot_error   : geodesic angle to goal orientation in radians
                  (pass 0 for tasks without a rotational goal)
    """
    c3_cost:     float
    config_cost: float
    pos_error:   float
    rot_error:   float = 0.0


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class ProgressTracker:
    """Window-based progress tracking for one of the four ProgressMetric
    variants.

    Usage per control loop:

        tracker.update(StepMetrics(c3_cost=..., config_cost=...,
                                    pos_error=..., rot_error=...))
        progressing = tracker.met_progress(near_goal=...)

    `near_goal` should be (box_to_goal_dist < cost_switching_threshold_distance);
    when True, the position-variant timeout (num_control_loops_to_wait_position)
    is used instead of num_control_loops_to_wait.
    """

    # tolerance below which "no decrease" still counts as "decreased" — keeps
    # floating-point jitter from preventing met_progress from latching on
    _IMPROVEMENT_EPS = 1e-9

    def __init__(self, params: ProgressParams):
        self.params = params

        # rolling history (most recent at the end)
        self._c3_cost_history:     list[float] = []
        self._config_cost_history: list[float] = []
        self._pos_error_history:   list[float] = []
        self._rot_error_history:   list[float] = []

        # Best-ever values (used by metrics that track absolute improvement).
        # The "wait" counter counts consecutive steps where no new best is
        # achieved.
        self._best_c3_cost:           float = float("inf")
        self._best_config_cost:       float = float("inf")
        self._best_pos_error:         float = float("inf")
        self._best_rot_error:         float = float("inf")

        self._steps_since_c3_improve:     int = 0
        self._steps_since_config_improve: int = 0
        self._steps_since_pos_improve:    int = 0
        self._steps_since_rot_improve:    int = 0

        self._n_updates: int = 0

    # ------------------------------------------------------------------
    # Per-step update
    # ------------------------------------------------------------------

    def update(self, m: StepMetrics) -> None:
        self._n_updates += 1

        self._c3_cost_history.append(m.c3_cost)
        self._config_cost_history.append(m.config_cost)
        self._pos_error_history.append(m.pos_error)
        self._rot_error_history.append(m.rot_error)

        # Cap history length at the larger of the two relevant windows
        cap = max(self.params.num_control_loops_to_wait,
                  self.params.num_control_loops_to_wait_position,
                  self.params.progress_enforced_over_n_loops) + 1
        for hist in (self._c3_cost_history,
                     self._config_cost_history,
                     self._pos_error_history,
                     self._rot_error_history):
            if len(hist) > cap:
                del hist[: len(hist) - cap]

        # Update best-ever + steps-since-improve counters
        eps = self._IMPROVEMENT_EPS

        if m.c3_cost < self._best_c3_cost - eps:
            self._best_c3_cost = m.c3_cost
            self._steps_since_c3_improve = 0
        else:
            self._steps_since_c3_improve += 1

        if m.config_cost < self._best_config_cost - eps:
            self._best_config_cost = m.config_cost
            self._steps_since_config_improve = 0
        else:
            self._steps_since_config_improve += 1

        if m.pos_error < self._best_pos_error - eps:
            self._best_pos_error = m.pos_error
            self._steps_since_pos_improve = 0
        else:
            self._steps_since_pos_improve += 1

        if m.rot_error < self._best_rot_error - eps:
            self._best_rot_error = m.rot_error
            self._steps_since_rot_improve = 0
        else:
            self._steps_since_rot_improve += 1

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def met_progress(self, near_goal: bool) -> bool:
        """True ⇔ the configured metric has shown progress within the
        applicable window. False ⇔ no-progress timeout has fired."""
        if self._n_updates == 0:
            return True   # nothing to time-out yet

        wait = (self.params.num_control_loops_to_wait_position
                if near_goal else self.params.num_control_loops_to_wait)

        m = self.params.track_c3_progress_via

        if m == ProgressMetric.kC3Cost:
            return self._steps_since_c3_improve < wait

        if m == ProgressMetric.kConfigCost:
            return self._steps_since_config_improve < wait

        if m == ProgressMetric.kPosOrRotCost:
            # progress = either pos OR rot improved recently
            return (self._steps_since_pos_improve < wait
                    or self._steps_since_rot_improve < wait)

        if m == ProgressMetric.kConfigCostDrop:
            # over the last N loops, cost must have dropped by ≥ required
            n = self.params.progress_enforced_over_n_loops
            if len(self._config_cost_history) < n + 1:
                return True   # not enough history yet — give the benefit of the doubt
            window = self._config_cost_history[-(n + 1):]
            drop = window[0] - window[-1]
            return drop >= self.params.progress_enforced_cost_drop - self._IMPROVEMENT_EPS

        raise ValueError(f"Unknown ProgressMetric: {m}")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def steps_since_improve(self) -> int:
        """Steps since the configured metric last improved.

        For kPosOrRotCost returns min(pos, rot). For kConfigCostDrop returns
        the kConfigCost counter (the closest analogue).
        """
        m = self.params.track_c3_progress_via
        if m == ProgressMetric.kC3Cost:
            return self._steps_since_c3_improve
        if m == ProgressMetric.kConfigCost:
            return self._steps_since_config_improve
        if m == ProgressMetric.kPosOrRotCost:
            return min(self._steps_since_pos_improve,
                       self._steps_since_rot_improve)
        if m == ProgressMetric.kConfigCostDrop:
            return self._steps_since_config_improve
        return -1

    def reset(self) -> None:
        """Wipe history. Call when entering a fresh repos→c3 cycle so the
        timeout starts from scratch."""
        self._c3_cost_history.clear()
        self._config_cost_history.clear()
        self._pos_error_history.clear()
        self._rot_error_history.clear()
        self._best_c3_cost     = float("inf")
        self._best_config_cost = float("inf")
        self._best_pos_error   = float("inf")
        self._best_rot_error   = float("inf")
        self._steps_since_c3_improve     = 0
        self._steps_since_config_improve = 0
        self._steps_since_pos_improve    = 0
        self._steps_since_rot_improve    = 0
        self._n_updates                  = 0
