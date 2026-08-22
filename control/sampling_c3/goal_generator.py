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


def tripod_id(quat) -> tuple:
    """Resting-tripod identity of a jack orientation: the world-z signs of
    the three body capsule axes (columns of R), one of 2^3 = 8 patterns —
    exactly the 8 nominal rests. World-yaw invariant. At the rests each
    axis has |z-component| = 1/sqrt(3), so the signs are far from the
    flicker zone; mid-roll a component crosses zero and the id is
    transient — callers must apply persistence before acting on it.
    """
    q = np.asarray(quat, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    zrow = (2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y))
    return tuple(1 if v > 0.0 else -1 for v in zrow)


# id -> nominal-rest name, e.g. TRIPOD_NAMES[tripod_id(q)] == "AllDown".
TRIPOD_NAMES = {
    tripod_id(q): name
    for q, name in zip(KNOMINAL_ORIENTATIONS_JACK, KNOMINAL_NAMES_JACK)
}


def topple_roll_plan(obj_quat, obj_tripod, goal_tripod, half_len=0.0625,
                     prefer_direction=None):
    """Plan one goal-directed roll (topple driver + flip primitive).

    Rolling over the support edge formed by two ground tips toggles
    exactly the third capsule's tripod sign (3-cube adjacency). Returns
    (k, p_B, dir_W): capsule index, push point in BODY frame (the UP end
    cap centre of capsule k — world height CoM_z + h/sqrt(3) = 97 mm,
    above the 55.3 mm tip-before-slide critical height), and the world
    horizontal unit push direction (from above tip_k toward the midpoint
    of the other two support tips, i.e. over the toggling edge).

    Candidate selection among the differing signs: FIRST differing index
    by default; with `prefer_direction` (world xy(z ignored) vector, e.g.
    goal_xy - obj_xy) the candidate whose push direction best aligns with
    it — each roll translates the CoM along dir_W, so this walks the jack
    TOWARD the position goal while it flips.
    Returns None when the tripods already match. Never on the reference
    path.
    """
    obj_tripod = tuple(obj_tripod)
    goal_tripod = tuple(goal_tripod)
    diff = [i for i in range(3) if obj_tripod[i] != goal_tripod[i]]
    if not diff:
        return None
    q = np.asarray(obj_quat, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    e = np.eye(3)
    tips = [R @ (-obj_tripod[i] * half_len * e[i]) for i in range(3)]

    def _dir_for(k):
        i, j = [m for m in range(3) if m != k]
        d = 0.5 * (tips[i] + tips[j]) - tips[k]
        d[2] = 0.0
        return d / np.linalg.norm(d)

    if prefer_direction is not None and len(diff) > 1:
        p = np.array([float(prefer_direction[0]),
                      float(prefer_direction[1]), 0.0])
        n = np.linalg.norm(p)
        if n > 1e-9:
            p = p / n
            k = max(diff, key=lambda kk: float(np.dot(_dir_for(kk), p)))
        else:
            k = diff[0]
    else:
        k = diff[0]
    d = _dir_for(k)
    p_B = obj_tripod[k] * half_len * e[k]
    return k, p_B, d


def orientation_lookahead(q_now, q_goal, last_axis,
                          lookahead_angle=2.0,      # goal_params.yaml:18/20
                          angle_hysteresis=0.4):    # goal_params.yaml:24
    """Per-tick orientation sub-goal — reference
    `GenerateLineTrajectoryWithLookahead` orientation branch
    (goal_generator.cc:408-437).

    Angle-axis of q_goal * q_now^-1 in the Eigen canonical form
    (angle = 2*atan2(|vec|, |w|) in [0, pi]; axis sign flipped when
    w < 0), then the near-180-degree hysteresis (if the axis opposes
    `last_axis` and pi - angle < angle_hysteresis, take the complementary
    rotation 2*pi - angle about -axis so the committed turn direction is
    kept across the singularity), then angle clamped to lookahead_angle
    and applied to the CURRENT orientation (world-frame premultiply).

    Returns (q_subgoal, new_last_axis). The caller threads new_last_axis
    into the next tick (reference: mutable last_rotation_axis_, h:180,
    zero-initialized). When the demand is within lookahead_angle the
    sub-goal IS the final goal, represented in q_now's hemisphere.
    """
    q_now = np.asarray(q_now, dtype=float)
    q_now = q_now / np.linalg.norm(q_now)
    q_goal = np.asarray(q_goal, dtype=float)
    q_goal = q_goal / np.linalg.norm(q_goal)
    d = quat_multiply(q_goal, q_now * np.array([1.0, -1.0, -1.0, -1.0]))
    n = float(np.linalg.norm(d[1:]))
    if n < 1e-12:
        # Eigen AngleAxis(q) fallback for the identity rotation.
        angle, axis = 0.0, np.array([1.0, 0.0, 0.0])
    else:
        angle = float(2.0 * np.arctan2(n, abs(d[0])))
        axis = d[1:] / n if d[0] >= 0.0 else -d[1:] / n
    if (float(np.dot(axis, np.asarray(last_axis, dtype=float))) < 0.0
            and (np.pi - angle) < angle_hysteresis):
        angle = 2.0 * np.pi - angle
        axis = -axis
    new_last_axis = axis.copy()
    angle = min(angle, lookahead_angle)
    q_rel = np.concatenate([[np.cos(angle / 2.0)],
                            np.sin(angle / 2.0) * axis])
    q_sub = quat_multiply(q_rel, q_now)
    return q_sub / np.linalg.norm(q_sub), new_last_axis


class JackRandomGoalGenerator:
    """Single-object kRandom re-goaler (reference jacktoy goal_params)."""

    def __init__(self, rng, initial_xy, initial_quat,
                 x_limits=(0.42, 0.5),          # random_goal_x_limits
                 y_limits=(0.02, 0.25),         # random_goal_y_limits
                 pos_success_threshold=0.02,    # position_success_threshold
                 ori_success_threshold=0.1,     # orientation_success_threshold
                 nominal_orientations=None,     # None -> jack tripods; a
                                                # planar object passes
                                                # [identity] (its single
                                                # flat-resting nominal, cf.
                                                # reference
                                                # GetNominalOrientations)
                 nominal_names=None,
                 planar_yaw_step_max=None,       # optional reachable yaw step
                 success_mode="reference",      # "reference" | "flip"
                 flip_persistence=10,           # ticks of tripod match to latch
                 flip_event_persistence=3):     # ticks to log a tripod change
        self._rng = rng
        self.goal_xy = np.asarray(initial_xy, dtype=float).copy()
        q0 = np.asarray(initial_quat, dtype=float)
        self.goal_quat = q0 / np.linalg.norm(q0)
        self._x_limits = tuple(x_limits)
        self._y_limits = tuple(y_limits)
        self._pos_thr = float(pos_success_threshold)
        self._ori_thr = float(ori_success_threshold)
        self._nominals = (KNOMINAL_ORIENTATIONS_JACK
                          if nominal_orientations is None
                          else [np.asarray(q, dtype=float)
                                for q in nominal_orientations])
        self.nominal_names = (KNOMINAL_NAMES_JACK
                              if nominal_names is None else list(nominal_names))
        self._planar_yaw_step_max = (None if planar_yaw_step_max is None
                                     else float(planar_yaw_step_max))
        self.orientation_index = -1     # reference h:182
        self.goals_reached = 0
        # --- flip success mode (USER-DIRECTED DEVIATION 2026-08-17) ------
        # "flip": a goal is reached when the jack's resting tripod matches
        # the goal's tripod for flip_persistence consecutive checks —
        # position and yaw are IGNORED (replaces the reference gate).
        # Draws then avoid the current tripod so every goal demands a flip.
        if success_mode not in ("reference", "flip"):
            raise ValueError(f"unknown success_mode: {success_mode!r}")
        self.success_mode = success_mode
        self._flip_persist = int(flip_persistence)
        self._match_streak = 0
        # Tripod-change telemetry (active in BOTH modes): persisted current
        # tripod + candidate streak so one mid-roll flicker tick is not a
        # "flip". last_flip = (from_name, to_name) of the newest event.
        self._event_persist = int(flip_event_persistence)
        self._current_tripod = None
        self._cand_tripod = None
        self._cand_streak = 0
        self.flip_events = 0
        self.last_flip = None

    @property
    def current_tripod(self):
        """Persisted resting tripod (None until first settled observation)."""
        return self._current_tripod

    def _track_tripod(self, obj_tripod) -> None:
        if obj_tripod == self._current_tripod:
            self._cand_tripod = None
            self._cand_streak = 0
            return
        if obj_tripod == self._cand_tripod:
            self._cand_streak += 1
        else:
            self._cand_tripod = obj_tripod
            self._cand_streak = 1
        if self._cand_streak >= self._event_persist:
            if self._current_tripod is not None:
                self.flip_events += 1
                self.last_flip = (TRIPOD_NAMES[self._current_tripod],
                                  TRIPOD_NAMES[obj_tripod])
            self._current_tripod = obj_tripod
            self._cand_tripod = None
            self._cand_streak = 0

    def check_and_regoal(self, obj_xy, obj_quat) -> bool:
        """Success gate + OnGoalReached. True if re-goaled.

        reference mode: cc:135-154 (pos<thr AND geodesic<thr).
        flip mode: resting-tripod match held flip_persistence checks.
        """
        obj_tripod = tripod_id(obj_quat)
        self._track_tripod(obj_tripod)
        if self.success_mode == "flip":
            if obj_tripod == tripod_id(self.goal_quat):
                self._match_streak += 1
            else:
                self._match_streak = 0
            if self._match_streak < self._flip_persist:
                return False
            self._match_streak = 0
            self.goals_reached += 1
            self._draw_position()
            self._draw_orientation(avoid_tripod=obj_tripod)
            return True
        pos_err = float(np.linalg.norm(
            np.asarray(obj_xy, dtype=float) - self.goal_xy))
        ang_err = geodesic_angle(self.goal_quat, obj_quat)
        if pos_err >= self._pos_thr or ang_err >= self._ori_thr:
            return False
        self.goals_reached += 1
        self._draw_position()
        self._draw_orientation()
        return True

    def draw_initial_goal(self, avoid_tripod=None) -> None:
        """Draw goal #1 from the kRandom distribution (port option,
        user-directed 2026-08-16).

        The reference boots from the yaml's fixed_target (goal_generator.cc
        ctor :67-69) and only randomizes on success — but its steady-state
        task IS the random tripod x yaw chase, and the fixed boot goal is a
        one-time transient. This draws goal #1 from the same distribution
        the reference uses for every subsequent goal. Not counted as a
        reached goal. In flip mode pass avoid_tripod=tripod_id(init_quat)
        so goal #1 already demands a flip.
        """
        self._draw_position()
        self._draw_orientation(avoid_tripod=avoid_tripod)

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

    def _draw_orientation(self, avoid_tripod=None) -> None:
        # avoid_tripod (flip mode, jack-only): re-draw the nominal index
        # until its tripod differs from the given one, so the goal demands
        # a flip. Bounded like the reference's random_goal_gen_max_attempts.
        # Draws index over self._nominals (jack tripods by default; a
        # planar object has ONE nominal and never passes avoid_tripod).
        # A planar object has only one nominal orientation. Draw a bounded
        # signed yaw increment from the goal it just reached so every new
        # demand stays inside the controller's yaw-lookahead envelope. The
        # unconstrained reference rule can draw nearly pi radians; the first
        # 28-goal Block-T attempt then parked forever at a pi-radian error on
        # goal 2 despite having reached its position.
        if self._planar_yaw_step_max is not None and len(self._nominals) == 1:
            yaw_max = self._planar_yaw_step_max
            if yaw_max < np.pi / 2:
                raise ValueError("planar_yaw_step_max must be >= pi/2")
            yaw = float(self._rng.uniform(np.pi / 2, yaw_max))
            if int(self._rng.integers(0, 2)) == 0:
                yaw = -yaw
            qz = np.array([np.cos(yaw / 2.0), 0.0, 0.0,
                           np.sin(yaw / 2.0)])
            q = quat_multiply(qz, self.goal_quat)
            self.goal_quat = q / np.linalg.norm(q)
            self.orientation_index = 0
            return

        for _ in range(100):
            idx = int(self._rng.integers(0, len(self._nominals)))
            if (avoid_tripod is None
                    or tripod_id(self._nominals[idx])
                    != tuple(avoid_tripod)):
                break
        if idx == self.orientation_index:
            # Repeat draw: >= 90 deg of extra yaw on the PREVIOUS GOAL
            # QUAT (cc:330-336) so the new goal always needs real work.
            # A planar object has ONE nominal, so every draw after the
            # first takes this branch — reference semantics for the T.
            yaw_lo, yaw_hi = np.pi / 2, 3 * np.pi / 2
            base = self.goal_quat
        else:
            yaw_lo, yaw_hi = 0.0, 2.0 * np.pi
            base = self._nominals[idx]
        yaw = float(self._rng.uniform(yaw_lo, yaw_hi))
        qz = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
        q = quat_multiply(qz, base)
        self.goal_quat = q / np.linalg.norm(q)
        self.orientation_index = idx
