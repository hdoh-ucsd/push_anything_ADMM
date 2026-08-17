"""kRandom goal semantics for the jack — port of the reference
`SamplingC3GoalGenerator` random mode (examples/sampling_c3/
goal_generator.{h,cc}), single-object case.

Reference anatomy (all three demos ship goal_mode: 0 = kRandom):
- The INITIAL goal is the yaml's fixed_target_* pair (ctor cc:67-69) —
  identical to the port's existing fixed goal, so kRandom only changes
  what happens ON SUCCESS.
- Success gate (cc:135-154): xy position error < position_success_
  threshold (0.02; only_use_xy_position: true in the jacktoy yaml) AND
  geodesic angle error < orientation_success_threshold (0.1) ->
  OnGoalReached (cc:378-389) draws a new position + orientation.
- Position draw (cc:269-314): x ~ U over random_goal_x_limits
  [0.42, 0.5]; y ~ U over the object's sampling area — for a single
  object that is the whole random_goal_y_limits [0.02, 0.25]
  (goal_params.h:96-104). random_goal_radius_limits is parsed but never
  read by goal_generator.cc — dead config, deliberately not ported.
- Orientation draw (cc:319-344): uniform index over the 8 nominal
  tripod rests; if the index repeats the CURRENT goal's index, the yaw
  range shrinks to [pi/2, 3pi/2] and is applied on top of the PREVIOUS
  FULL GOAL QUAT (";= 90 deg away" rule, cc:330-336); otherwise
  yaw ~ U[0, 2pi) on the nominal. quat_final = R_z(yaw) * base
  (world-frame premultiply). orientation_index_ starts at -1 (h:182).
- The controller-side achieved_fixed_goal_ latch fires ONLY under
  kFixedGoal (systems/controllers/sampling_based_c3_controller.cc:
  914-916); under kRandom the controller never enters the
  achieved/force-free regime — main.py resets the port's latch on every
  re-goal.

Port deviation (deliberate): the reference seeds each draw from
std::random_device (goal_generator.cc:322 — non-reproducible); the port
draws from the run's --seed-derived numpy Generator so goal sequences
are reproducible per seed. Distribution semantics are identical.
"""
from __future__ import annotations

import numpy as np

# The 8 nominal tripod orientations, (w, x, y, z), copied verbatim from
# goal_generator.h:20-38 (Eigen::Quaterniond{w, x, y, z} literals).
KQUAT_ALL_UP = (
    0.88047623921714, 0.279848142333121, -0.36470519963100,
    -0.115916895959295)
KQUAT_RED_DOWN = (
    0.88047623921714, 0.279848142333121, 0.36470519963100,
    0.115916895959295)
KQUAT_BLUE_UP = (
    0.70455634261098, -0.060003000646865, 0.455768038939282,
    -0.5406250962371)
KQUAT_ALL_DOWN = (
    0.455768038939282, -0.54062509623716, 0.70455634261098,
    -0.0600030006468661)
KQUAT_GREEN_UP = (
    0.364705199631001, 0.115916895959295, 0.88047623921714,
    0.279848142333121)
KQUAT_BLUE_DOWN = (
    0.0600030006468662, 0.70455634261098, 0.5406250962371,
    0.45576803893928)
KQUAT_RED_UP = (
    -0.27984814233312, 0.88047623921714, -0.115916895959295,
    0.36470519963100)
KQUAT_GREEN_DOWN = (
    -0.82047323857028, 0.424708200277866, 0.17591989660616,
    0.33985114297998)

KNOMINAL_ORIENTATIONS_JACK = [
    np.array(KQUAT_ALL_UP),   np.array(KQUAT_RED_DOWN),
    np.array(KQUAT_BLUE_UP),  np.array(KQUAT_ALL_DOWN),
    np.array(KQUAT_GREEN_UP), np.array(KQUAT_BLUE_DOWN),
    np.array(KQUAT_RED_UP),   np.array(KQUAT_GREEN_DOWN),
]
KNOMINAL_NAMES_JACK = [
    "AllUp", "RedDown", "BlueUp", "AllDown",
    "GreenUp", "BlueDown", "RedUp", "GreenDown",
]


def quat_multiply(a, b) -> np.ndarray:
    """Hamilton product a*b, both (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def geodesic_angle(q_a, q_b) -> float:
    """Rotation angle of q_a * q_b^-1 in [0, pi].

    Matches Eigen 3.4 AngleAxisd(q): angle = 2*atan2(|vec|, |w|) —
    double-cover canonical, so q and -q are the same rotation.
    """
    q_a = np.asarray(q_a, dtype=float)
    q_b = np.asarray(q_b, dtype=float)
    q_a = q_a / np.linalg.norm(q_a)
    q_b = q_b / np.linalg.norm(q_b)
    q_b_inv = q_b * np.array([1.0, -1.0, -1.0, -1.0])
    d = quat_multiply(q_a, q_b_inv)
    return float(2.0 * np.arctan2(np.linalg.norm(d[1:]), abs(d[0])))


class JackRandomGoalGenerator:
    """Single-object kRandom re-goaler (reference jacktoy goal_params)."""

    def __init__(self, rng, initial_xy, initial_quat,
                 x_limits=(0.42, 0.5),          # random_goal_x_limits
                 y_limits=(0.02, 0.25),         # random_goal_y_limits
                 pos_success_threshold=0.02,    # position_success_threshold
                 ori_success_threshold=0.1):    # orientation_success_threshold
        self._rng = rng
        self.goal_xy = np.asarray(initial_xy, dtype=float).copy()
        q0 = np.asarray(initial_quat, dtype=float)
        self.goal_quat = q0 / np.linalg.norm(q0)
        self._x_limits = tuple(x_limits)
        self._y_limits = tuple(y_limits)
        self._pos_thr = float(pos_success_threshold)
        self._ori_thr = float(ori_success_threshold)
        self.orientation_index = -1     # reference h:182
        self.goals_reached = 0

    def check_and_regoal(self, obj_xy, obj_quat) -> bool:
        """Reference cc:135-154 gate + OnGoalReached. True if re-goaled."""
        pos_err = float(np.linalg.norm(
            np.asarray(obj_xy, dtype=float) - self.goal_xy))
        ang_err = geodesic_angle(self.goal_quat, obj_quat)
        if pos_err >= self._pos_thr or ang_err >= self._ori_thr:
            return False
        self.goals_reached += 1
        self._draw_position()
        self._draw_orientation()
        return True

    def force_regoal(self) -> None:
        """Diagnostic: draw a new goal regardless of the success gate.

        Used by main.py's DIAG_GOALGEN_FORCE_REGOAL_AT_STEP hook to
        exercise the live re-goal path (setters, ghost, latch reset)
        without waiting for a real goal achievement. Never called on the
        canonical path.
        """
        self.goals_reached += 1
        self._draw_position()
        self._draw_orientation()

    def _draw_position(self) -> None:
        # Single object: datum is always NaN at draw time -> plain uniform
        # box draw (cc:281-294 with <= 2 objects).
        self.goal_xy = np.array([
            self._rng.uniform(*self._x_limits),
            self._rng.uniform(*self._y_limits),
        ])

    def _draw_orientation(self) -> None:
        idx = int(self._rng.integers(0, len(KNOMINAL_ORIENTATIONS_JACK)))
        if idx == self.orientation_index:
            # Repeat draw: >= 90 deg of extra yaw on the PREVIOUS GOAL
            # QUAT (cc:330-336) so the new goal always needs real work.
            yaw_lo, yaw_hi = np.pi / 2, 3 * np.pi / 2
            base = self.goal_quat
        else:
            yaw_lo, yaw_hi = 0.0, 2.0 * np.pi
            base = KNOMINAL_ORIENTATIONS_JACK[idx]
        yaw = float(self._rng.uniform(yaw_lo, yaw_hi))
        qz = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
        q = quat_multiply(qz, base)
        self.goal_quat = q / np.linalg.norm(q)
        self.orientation_index = idx
