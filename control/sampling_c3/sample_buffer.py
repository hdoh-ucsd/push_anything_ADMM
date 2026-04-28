"""
Sample buffer — remembers good repositioning targets across control loops.

When the controller has been in C3 mode for many loops the recent C3 cost
trace alone may not contain the best repositioning candidate. The buffer
keeps a short history of high-quality samples (low c_sample) so that on
mode-switch we can warm-start with the historical best that is still
relevant under the current object pose.

Algorithm (per control loop):

  1. PRUNE: drop entries whose stored object pose has drifted from the
     current pose by more than (pos_error_sample_retention,
     ang_error_sample_retention).
  2. APPEND: add the current loop's best non-current sample.
  3. CAP at N_sample_buffer entries (FIFO).

`best_with_position()` returns the lowest-cost surviving entry, or None.

Pure-numpy; no Drake dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Single buffer entry
# ---------------------------------------------------------------------------

@dataclass
class BufferedSample:
    """One historical sample.

    position      : (3,) Cartesian EE target in world frame (m)
    cost          : float — the c_sample (post-alignment-bonus) value at insert
    obj_pos_xy    : (2,) box xy at insert time (used for pos-retention check)
    obj_quat      : (4,) box quaternion (qw, qx, qy, qz) at insert time;
                    pass None to disable the angular-retention check
    age_steps     : int — incremented each control loop (oldest dropped first
                    when capacity is exceeded)
    """
    position:      np.ndarray
    cost:          float
    obj_pos_xy:    np.ndarray
    obj_quat:      Optional[np.ndarray] = None
    age_steps:     int                  = 0


# ---------------------------------------------------------------------------
# Quaternion utilities
# ---------------------------------------------------------------------------

def _quat_geodesic_angle(q_a: np.ndarray, q_b: np.ndarray) -> float:
    """Smallest rotation angle between two unit quaternions (radians).

    Handles the q ≡ -q double-cover ambiguity by taking |dot|.
    """
    if q_a is None or q_b is None:
        return 0.0
    d = float(abs(np.dot(q_a, q_b)))
    d = min(1.0, max(-1.0, d))
    return 2.0 * float(np.arccos(d))


# ---------------------------------------------------------------------------
# Buffer
# ---------------------------------------------------------------------------

class SampleBuffer:
    """FIFO sample buffer with pose-based pruning."""

    def __init__(self,
                 capacity:           int   = 5,
                 pos_threshold:      float = 0.05,
                 ang_threshold:      float = 0.30):
        """
        Parameters
        ----------
        capacity        : max entries before FIFO eviction
        pos_threshold   : prune when |obj_pos_xy_now - sample.obj_pos_xy| > thresh (m)
        ang_threshold   : prune when geodesic_angle(q_now, sample.obj_quat) > thresh (rad)
                          (only applied when both quaternions are non-None)
        """
        self.capacity      = int(capacity)
        self.pos_threshold = float(pos_threshold)
        self.ang_threshold = float(ang_threshold)
        self._entries: list[BufferedSample] = []

    # -- core operations ----------------------------------------------------

    def prune(self,
              obj_pos_xy_now: np.ndarray,
              obj_quat_now:   Optional[np.ndarray] = None) -> int:
        """Drop entries whose stored pose has drifted past the retention
        thresholds. Returns the number of entries removed."""
        before = len(self._entries)
        kept: list[BufferedSample] = []
        for s in self._entries:
            d_pos = float(np.linalg.norm(obj_pos_xy_now - s.obj_pos_xy))
            if d_pos > self.pos_threshold:
                continue
            if obj_quat_now is not None and s.obj_quat is not None:
                d_ang = _quat_geodesic_angle(obj_quat_now, s.obj_quat)
                if d_ang > self.ang_threshold:
                    continue
            kept.append(s)
        self._entries = kept
        return before - len(self._entries)

    def append(self, sample: BufferedSample) -> None:
        """Append, then evict oldest while over capacity."""
        self._entries.append(sample)
        while len(self._entries) > self.capacity:
            self._entries.pop(0)

    def tick_age(self) -> None:
        """Increment age on every entry. Called once per control loop."""
        for s in self._entries:
            s.age_steps += 1

    def clear(self) -> None:
        self._entries.clear()

    # -- queries ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def best_with_position(self) -> Optional[BufferedSample]:
        """Lowest-cost surviving entry, or None if buffer is empty."""
        if not self._entries:
            return None
        return min(self._entries, key=lambda s: s.cost)

    def snapshot(self) -> list[BufferedSample]:
        """Defensive copy for inspection / diagnostics."""
        return list(self._entries)
