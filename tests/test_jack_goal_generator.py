"""kRandom goal semantics for the jack — port of the reference
SamplingC3GoalGenerator's random mode (goal_generator.{h,cc}).

Reference semantics (verified against the checkout 2026-08-16):
- 8 nominal tripod orientations `kNominalOrientationsJack` (h:20-38),
  order [AllUp, RedDown, BlueUp, AllDown, GreenUp, BlueDown, RedUp,
  GreenDown].
- Success gate (cc:135-154): xy position error < 0.02 (only_use_xy_
  position: true) AND geodesic angle error < 0.1 -> OnGoalReached draws a
  NEW goal (kRandom, cc:378-389). Continuous re-goaling.
- Position draw (cc:269-314, single object): x ~ U[0.42, 0.5],
  y ~ U[0.02, 0.25] (sampling_area_y_limits for one object = the whole
  random_goal_y_limits range, goal_params.h:96-104).
  random_goal_radius_limits is parsed but never used — dead config.
- Orientation draw (cc:319-344): index ~ U{0..7}; if the index REPEATS
  the current goal's index, yaw ~ U[pi/2, 3pi/2] applied on top of the
  PREVIOUS FULL GOAL QUAT (so the new goal always needs >= 90 deg of
  work); else yaw ~ U[0, 2pi) applied to the nominal.
  quat_final = yaw-about-world-z * base. orientation_index_ starts -1
  (h:182), so the first redraw never triggers the repeat rule.
- The controller-side achieved_fixed_goal_ latch fires ONLY under
  kFixedGoal (controller cc:914-916) — under kRandom it must stay off.
"""
import numpy as np
import pytest

from control.sampling_c3.goal_generator import (
    KNOMINAL_ORIENTATIONS_JACK,
    KNOMINAL_NAMES_JACK,
    JackRandomGoalGenerator,
    geodesic_angle,
    quat_multiply,
)

INITIAL_XY = np.array([0.45, 0.2])
KQUAT_ALL_UP = np.array(
    [0.88047623921714, 0.279848142333121, -0.36470519963100,
     -0.115916895959295])


def _mk(rng=None):
    return JackRandomGoalGenerator(
        rng=rng if rng is not None else np.random.default_rng(0),
        initial_xy=INITIAL_XY,
        initial_quat=KQUAT_ALL_UP,
    )


class _ScriptedRng:
    """Fake rng: scripted integer draws, recorded uniform calls."""

    def __init__(self, int_draws, uniform_returns):
        self.int_draws = list(int_draws)
        self.uniform_returns = list(uniform_returns)
        self.uniform_calls = []

    def integers(self, lo, hi):
        return self.int_draws.pop(0)

    def uniform(self, lo, hi):
        self.uniform_calls.append((lo, hi))
        return self.uniform_returns.pop(0)


# ---------------------------------------------------------------- literals
def test_nominal_orientations_count_names_unit_norm():
    assert len(KNOMINAL_ORIENTATIONS_JACK) == 8
    assert KNOMINAL_NAMES_JACK == [
        "AllUp", "RedDown", "BlueUp", "AllDown",
        "GreenUp", "BlueDown", "RedUp", "GreenDown"]
    for q in KNOMINAL_ORIENTATIONS_JACK:
        assert abs(np.linalg.norm(q) - 1.0) < 1e-9


def test_nominal_literals_match_reference_spot_checks():
    np.testing.assert_allclose(
        KNOMINAL_ORIENTATIONS_JACK[0], KQUAT_ALL_UP, atol=1e-14)
    np.testing.assert_allclose(
        KNOMINAL_ORIENTATIONS_JACK[3],
        [0.455768038939282, -0.54062509623716, 0.70455634261098,
         -0.0600030006468661], atol=1e-14)


def test_nominals_pairwise_distinct():
    for i in range(8):
        for j in range(i + 1, 8):
            assert geodesic_angle(
                KNOMINAL_ORIENTATIONS_JACK[i],
                KNOMINAL_ORIENTATIONS_JACK[j]) > 0.1


# ---------------------------------------------------------------- geodesic
def test_geodesic_double_cover_and_identity():
    q = np.array([0.5, 0.5, 0.5, 0.5])
    assert geodesic_angle(q, q) < 1e-12
    assert geodesic_angle(q, -q) < 1e-12   # same rotation, other cover
    # 90 deg about z vs identity
    qz90 = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    e = np.array([1.0, 0.0, 0.0, 0.0])
    assert abs(geodesic_angle(qz90, e) - np.pi / 2) < 1e-12


# ---------------------------------------------------------------- gate
def test_no_regoal_when_position_off():
    g = _mk()
    changed = g.check_and_regoal(INITIAL_XY + [0.05, 0.0], KQUAT_ALL_UP)
    assert changed is False
    assert g.goals_reached == 0
    np.testing.assert_allclose(g.goal_xy, INITIAL_XY)


def test_no_regoal_when_orientation_off():
    g = _mk()
    # 0.2 rad of world yaw on the goal quat -> angular error 0.2 > 0.1
    yaw = 0.2
    qz = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    q_off = quat_multiply(qz, KQUAT_ALL_UP)
    assert g.check_and_regoal(INITIAL_XY, q_off) is False


def test_regoal_on_success_draws_within_limits():
    g = _mk()
    changed = g.check_and_regoal(INITIAL_XY + [0.01, 0.0], KQUAT_ALL_UP)
    assert changed is True
    assert g.goals_reached == 1
    assert 0.42 <= g.goal_xy[0] <= 0.5
    assert 0.02 <= g.goal_xy[1] <= 0.25


def test_position_draws_stay_in_reference_box():
    g = _mk(np.random.default_rng(7))
    for _ in range(200):
        # Track the goal exactly so every call re-goals.
        assert g.check_and_regoal(g.goal_xy.copy(), g.goal_quat.copy())
        assert 0.42 <= g.goal_xy[0] <= 0.5, g.goal_xy
        assert 0.02 <= g.goal_xy[1] <= 0.25, g.goal_xy
        assert abs(np.linalg.norm(g.goal_quat) - 1.0) < 1e-9


# ---------------------------------------------------------- orientation draw
def test_fresh_index_uses_nominal_with_full_yaw_range():
    # index draw 2 (BlueUp) while orientation_index=-1; yaw scripted 0.7.
    # uniform calls: x, y (position), then yaw.
    rng = _ScriptedRng(int_draws=[2], uniform_returns=[0.45, 0.1, 0.7])
    g = _mk(rng)
    assert g.check_and_regoal(INITIAL_XY, KQUAT_ALL_UP)
    yaw_lo, yaw_hi = rng.uniform_calls[-1]
    assert (yaw_lo, yaw_hi) == (0.0, 2.0 * np.pi)
    qz = np.array([np.cos(0.35), 0.0, 0.0, np.sin(0.35)])
    expected = quat_multiply(qz, KNOMINAL_ORIENTATIONS_JACK[2])
    assert geodesic_angle(g.goal_quat, expected) < 1e-12
    assert g.orientation_index == 2


def test_repeat_index_forces_half_pi_yaw_on_previous_goal_quat():
    # First re-goal: index 2, yaw 0.7 (as above). Second re-goal: index 2
    # again -> yaw range [pi/2, 3pi/2] applied to the PREVIOUS GOAL QUAT
    # (reference cc:330-336), not the nominal.
    rng = _ScriptedRng(int_draws=[2, 2],
                       uniform_returns=[0.45, 0.1, 0.7,
                                        0.46, 0.11, 2.0])
    g = _mk(rng)
    assert g.check_and_regoal(INITIAL_XY, KQUAT_ALL_UP)
    prev_goal_quat = g.goal_quat.copy()
    assert g.check_and_regoal(g.goal_xy.copy(), g.goal_quat.copy())
    yaw_lo, yaw_hi = rng.uniform_calls[-1]
    assert abs(yaw_lo - np.pi / 2) < 1e-12
    assert abs(yaw_hi - 3 * np.pi / 2) < 1e-12
    qz = np.array([np.cos(1.0), 0.0, 0.0, np.sin(1.0)])
    expected = quat_multiply(qz, prev_goal_quat)
    assert geodesic_angle(g.goal_quat, expected) < 1e-12


# ---------------------------------------------------------------- diag
def test_force_regoal_draws_without_gate():
    # Diagnostic path (DIAG_GOALGEN_FORCE_REGOAL_AT_STEP): draw a new goal
    # regardless of the success gate, counting it as reached.
    g = _mk()
    g.force_regoal()
    assert g.goals_reached == 1
    assert 0.42 <= g.goal_xy[0] <= 0.5
    assert 0.02 <= g.goal_xy[1] <= 0.25
    assert g.orientation_index in range(8)


# ------------------------------------------------------- initial-goal draw
def test_draw_initial_goal_samples_krandom_distribution():
    # User directive 2026-08-16: the reference's steady-state task is the
    # random quaternion chase; goal #1 is drawn from the same distribution
    # (position box + tripod x yaw) at startup. Not counted as a success.
    g = _mk()
    g.draw_initial_goal()
    assert g.goals_reached == 0
    assert 0.42 <= g.goal_xy[0] <= 0.5
    assert 0.02 <= g.goal_xy[1] <= 0.25
    assert g.orientation_index in range(8)
    assert abs(np.linalg.norm(g.goal_quat) - 1.0) < 1e-9


def test_draw_initial_goal_is_seed_deterministic():
    g1 = _mk(np.random.default_rng(42))
    g2 = _mk(np.random.default_rng(42))
    g1.draw_initial_goal()
    g2.draw_initial_goal()
    np.testing.assert_allclose(g1.goal_xy, g2.goal_xy)
    np.testing.assert_allclose(g1.goal_quat, g2.goal_quat)
    assert g1.orientation_index == g2.orientation_index


# ---------------------------------------------------------------- config
def test_push_jack_task_config_enables_krandom():
    import yaml
    cfg = yaml.safe_load(open("config/tasks.yaml"))
    assert cfg["tasks"]["push_jack"].get("goal_mode") == "kRandom"
    # Parked to false with the 2026-08-19 canonical jack recipe (boot from
    # the fixed reference target; draw on success only). Stale `is True`
    # assertion corrected during the 2026-08-19 T+jack merge.
    assert cfg["tasks"]["push_jack"].get("krandom_draw_initial_goal") is False
