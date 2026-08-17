"""Tests for the flip success mode on the jack goal generator.

USER-DIRECTED DEVIATION (2026-08-17): under `goal_success_mode: flip` a
goal counts as reached when the jack's RESTING TRIPOD matches the goal's
tripod — position and yaw are ignored (replaces the reference gate
pos<0.02 AND rot<0.1). A tripod is identified by the world-z signs of the
three body capsule axes (2^3 = 8, one per nominal rest). The match must
persist `flip_persistence` consecutive checks so mid-roll sign flicker
cannot latch, and re-draws avoid the current tripod so every goal demands
a flip. The reference gate remains the code default.
"""
import numpy as np

from control.sampling_c3.goal_generator import (
    JackRandomGoalGenerator,
    KNOMINAL_ORIENTATIONS_JACK,
    KNOMINAL_NAMES_JACK,
    TRIPOD_NAMES,
    quat_multiply,
    tripod_id,
)


def _yawed(q, yaw):
    qz = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
    return quat_multiply(qz, np.asarray(q, dtype=float))


class _ScriptedRng:
    """integers()/uniform() from scripted queues (falls back to fixed)."""

    def __init__(self, integers=(), uniforms=()):
        self._ints = list(integers)
        self._unis = list(uniforms)

    def integers(self, lo, hi):
        return self._ints.pop(0) if self._ints else lo

    def uniform(self, lo, hi):
        return self._unis.pop(0) if self._unis else (lo + hi) / 2.0


def _gen(success_mode="flip", rng=None, goal_idx=3, persistence=3):
    # goal = nominal[goal_idx] (AllDown by default), arbitrary position.
    return JackRandomGoalGenerator(
        rng=rng or np.random.default_rng(0),
        initial_xy=(0.45, 0.10),
        initial_quat=KNOMINAL_ORIENTATIONS_JACK[goal_idx],
        success_mode=success_mode,
        flip_persistence=persistence,
    )


def test_tripod_ids_distinct_and_signed():
    ids = [tripod_id(q) for q in KNOMINAL_ORIENTATIONS_JACK]
    assert len(set(ids)) == 8
    for t in ids:
        assert len(t) == 3 and all(s in (-1, 1) for s in t)
    # Every id has a name, and names match the nominal list.
    for t, name in zip(ids, KNOMINAL_NAMES_JACK):
        assert TRIPOD_NAMES[t] == name


def test_tripod_id_yaw_invariant():
    for q in KNOMINAL_ORIENTATIONS_JACK:
        base = tripod_id(q)
        for yaw in (0.3, 1.7, -2.9):
            assert tripod_id(_yawed(q, yaw)) == base


def test_flip_mode_ignores_position_and_yaw():
    g = _gen(persistence=3)
    # Object ON the goal tripod (AllDown) but 40 cm away and 2.5 rad of
    # yaw off the goal quat: the reference gate would NEVER fire here.
    obj_q = _yawed(KNOMINAL_ORIENTATIONS_JACK[3], 2.5)
    far_xy = (0.05, 0.50)
    assert g.check_and_regoal(far_xy, obj_q) is False   # streak 1
    assert g.check_and_regoal(far_xy, obj_q) is False   # streak 2
    assert g.check_and_regoal(far_xy, obj_q) is True    # streak 3 -> latch
    assert g.goals_reached == 1


def test_flip_mode_flicker_does_not_latch():
    g = _gen(persistence=3)
    on = _yawed(KNOMINAL_ORIENTATIONS_JACK[3], 1.0)     # goal tripod
    off = KNOMINAL_ORIENTATIONS_JACK[5]                 # other tripod
    for _ in range(5):
        assert g.check_and_regoal((0.0, 0.0), on) is False
        assert g.check_and_regoal((0.0, 0.0), off) is False
    assert g.goals_reached == 0


def test_flip_mode_wrong_tripod_never_latches():
    g = _gen(persistence=2)
    obj_q = KNOMINAL_ORIENTATIONS_JACK[5]               # BlueDown != AllDown
    for _ in range(10):
        assert g.check_and_regoal(g.goal_xy, obj_q) is False
    assert g.goals_reached == 0


def test_reference_mode_unchanged_by_tripod_match():
    g = _gen(success_mode="reference", persistence=1)
    # Same tripod, big yaw error: reference gate must NOT fire.
    obj_q = _yawed(KNOMINAL_ORIENTATIONS_JACK[3], 2.5)
    for _ in range(5):
        assert g.check_and_regoal(g.goal_xy, obj_q) is False
    assert g.goals_reached == 0


def test_redraw_avoids_current_tripod():
    # Scripted rng first proposes index 3 (the tripod the jack now RESTS
    # on) twice; the draw must skip it and settle on index 5.
    rng = _ScriptedRng(integers=[3, 3, 5], uniforms=[0.45, 0.10, 1.0])
    g = _gen(rng=rng, persistence=1)
    obj_q = _yawed(KNOMINAL_ORIENTATIONS_JACK[3], 0.7)
    assert g.check_and_regoal((0.9, 0.9), obj_q) is True
    assert g.orientation_index == 5
    assert tripod_id(g.goal_quat) != tripod_id(obj_q)


def test_initial_draw_avoids_given_tripod():
    rng = _ScriptedRng(integers=[5, 5, 1], uniforms=[0.45, 0.10, 0.5])
    g = _gen(rng=rng, persistence=1)
    g.draw_initial_goal(
        avoid_tripod=tripod_id(KNOMINAL_ORIENTATIONS_JACK[5]))
    assert g.orientation_index == 1
    assert tripod_id(g.goal_quat) != tripod_id(KNOMINAL_ORIENTATIONS_JACK[5])


def test_flip_events_recorded_on_persisted_change():
    g = _gen(persistence=100)   # success gate effectively off
    a = KNOMINAL_ORIENTATIONS_JACK[5]                   # BlueDown
    b = _yawed(KNOMINAL_ORIENTATIONS_JACK[1], 0.4)      # RedDown
    for _ in range(5):
        g.check_and_regoal((0.0, 0.0), a)
    assert g.flip_events == 0
    # one flickering tick must not count as a flip ...
    g.check_and_regoal((0.0, 0.0), b)
    g.check_and_regoal((0.0, 0.0), a)
    for _ in range(4):
        g.check_and_regoal((0.0, 0.0), a)
    assert g.flip_events == 0
    # ... but a persisted change must.
    for _ in range(5):
        g.check_and_regoal((0.0, 0.0), b)
    assert g.flip_events == 1
    assert g.last_flip == (TRIPOD_NAMES[tripod_id(a)],
                           TRIPOD_NAMES[tripod_id(b)])


def test_success_streak_resets_after_regoal():
    rng = _ScriptedRng(integers=[5], uniforms=[0.45, 0.10, 1.0])
    g = _gen(rng=rng, persistence=2)
    obj_q = KNOMINAL_ORIENTATIONS_JACK[3]
    assert g.check_and_regoal((0.0, 0.0), obj_q) is False
    assert g.check_and_regoal((0.0, 0.0), obj_q) is True
    # New goal is BlueDown (idx 5); the same AllDown pose must now build a
    # fresh streak against it without instantly latching.
    assert g.check_and_regoal((0.0, 0.0), obj_q) is False
