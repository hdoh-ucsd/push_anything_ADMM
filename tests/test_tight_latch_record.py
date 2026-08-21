"""The sticky tight-achievement record vs the releasable dispatch pin.

Reference semantics (sampling_based_c3_controller.cc:887-897): once the
goal is achieved, the achievement is recorded forever for that goal. The
port's authorized achieved-goal-release deviation may clear the dispatch
pin (_achieved_fixed_goal) to re-engage and correct post-latch drift, but
it must not erase the record (_tight_ever_latched) — 2026-08-20 blockT
run3 latched joint-tight at t=122.4s, was released twice on a 0.4 mm
settle, and wrongly reported tight_goal=FAIL(-).

Pure-logic replica of the controller's latch/release block so the test
runs without pydrake.
"""


class _LatchSim:
    """Minimal replica of the latch + release + record state machine."""

    def __init__(self, release_loops):
        self._achieved_fixed_goal = False
        self._tight_ever_latched = False
        self._off_target_streak = 0
        self._release_loops = release_loops
        self._crossed = True          # pose regime: rot is checked

    def tick(self, dist, rot, pos_thr=0.02, rot_thr=0.10):
        # release block (controller cc-port :2168-2185)
        if self._achieved_fixed_goal and self._release_loops > 0:
            on_target = dist < pos_thr and (not self._crossed or rot < rot_thr)
            if on_target:
                self._off_target_streak = 0
            else:
                self._off_target_streak += 1
                if self._off_target_streak >= self._release_loops:
                    self._achieved_fixed_goal = False
                    self._off_target_streak = 0
        # latch block (:2185-2197)
        if not self._achieved_fixed_goal:
            on_target = dist < pos_thr and (not self._crossed or rot < rot_thr)
            if on_target:
                self._achieved_fixed_goal = True
                self._tight_ever_latched = True

    def reset_for_new_goal(self):
        self._achieved_fixed_goal = False
        self._off_target_streak = 0
        self._tight_ever_latched = False


def test_record_survives_release():
    """run3 anatomy: latch, drift out past release_loops, end just over."""
    s = _LatchSim(release_loops=20)
    s.tick(0.0197, 0.083)                 # t=122.4s: joint-tight -> latch
    assert s._achieved_fixed_goal and s._tight_ever_latched
    for _ in range(20):                    # 0.4mm settle: 20 off-target loops
        s.tick(0.0201, 0.077)
    assert not s._achieved_fixed_goal      # pin released (authorized)
    assert s._tight_ever_latched           # record survives -> PASS(latched)


def test_record_never_set_without_achievement():
    s = _LatchSim(release_loops=20)
    for _ in range(100):
        s.tick(0.0210, 0.05)               # never under 20 mm
    assert not s._tight_ever_latched
    assert not s._achieved_fixed_goal


def test_sticky_config_zero_is_reference_identical():
    s = _LatchSim(release_loops=0)
    s.tick(0.0197, 0.083)
    for _ in range(500):
        s.tick(0.0400, 0.30)               # gross drift: still no release
    assert s._achieved_fixed_goal and s._tight_ever_latched


def test_new_goal_clears_record():
    s = _LatchSim(release_loops=20)
    s.tick(0.0197, 0.083)
    assert s._tight_ever_latched
    s.reset_for_new_goal()
    assert not s._tight_ever_latched       # record is per-goal
    assert not s._achieved_fixed_goal


def test_relatch_after_release_keeps_record():
    """run3 latched twice; record must stay True throughout."""
    s = _LatchSim(release_loops=20)
    s.tick(0.0197, 0.083)                  # latch 1
    for _ in range(20):
        s.tick(0.0201, 0.077)              # release 1
    s.tick(0.0198, 0.074)                  # latch 2
    assert s._achieved_fixed_goal and s._tight_ever_latched
    for _ in range(20):
        s.tick(0.0200, 0.074)              # release 2
    assert not s._achieved_fixed_goal
    assert s._tight_ever_latched
