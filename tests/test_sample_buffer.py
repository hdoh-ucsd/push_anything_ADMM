"""Unit tests for control.sampling_c3.sample_buffer."""
import numpy as np
import pytest

from control.sampling_c3.sample_buffer import (
    BufferedSample,
    SampleBuffer,
    _quat_geodesic_angle,
)


def _entry(pos=(0.1, 0.0, 0.05), cost=100.0, obj_xy=(0.0, 0.0), quat=None):
    return BufferedSample(
        position   = np.array(pos, dtype=float),
        cost       = float(cost),
        obj_pos_xy = np.array(obj_xy, dtype=float),
        obj_quat   = None if quat is None else np.array(quat, dtype=float),
    )


# ---------------------------------------------------------------------------
# Quaternion geodesic angle
# ---------------------------------------------------------------------------

def test_quat_angle_identity_is_zero():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    assert _quat_geodesic_angle(q, q) == pytest.approx(0.0)


def test_quat_angle_handles_double_cover():
    """q ≡ -q for SO(3); angle must be 0 between them, not 2π."""
    q  = np.array([1.0, 0.0, 0.0, 0.0])
    qn = -q
    assert _quat_geodesic_angle(q, qn) == pytest.approx(0.0)


def test_quat_angle_90deg_about_z():
    half = np.sqrt(0.5)
    q1 = np.array([1.0, 0.0, 0.0, 0.0])
    q2 = np.array([half, 0.0, 0.0, half])  # 90° rotation about z
    assert _quat_geodesic_angle(q1, q2) == pytest.approx(np.pi / 2.0, abs=1e-6)


def test_quat_angle_with_none_returns_zero():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    assert _quat_geodesic_angle(q, None) == 0.0
    assert _quat_geodesic_angle(None, q) == 0.0


# ---------------------------------------------------------------------------
# Append + capacity
# ---------------------------------------------------------------------------

def test_empty_buffer_has_no_best():
    b = SampleBuffer()
    assert len(b) == 0
    assert b.best_with_position() is None


def test_append_increases_length():
    b = SampleBuffer(capacity=5)
    b.append(_entry(cost=100.0))
    b.append(_entry(cost=50.0))
    assert len(b) == 2


def test_capacity_evicts_oldest_first():
    b = SampleBuffer(capacity=3)
    for c in [10.0, 20.0, 30.0, 40.0, 50.0]:   # insert 5 into capacity-3
        b.append(_entry(cost=c))
    snap = b.snapshot()
    assert len(snap) == 3
    # Oldest two (10, 20) should have been evicted
    assert [s.cost for s in snap] == [30.0, 40.0, 50.0]


def test_best_returns_lowest_cost():
    b = SampleBuffer(capacity=5)
    for c in [40.0, 10.0, 70.0, 20.0]:
        b.append(_entry(cost=c))
    best = b.best_with_position()
    assert best is not None
    assert best.cost == 10.0


def test_clear_empties_buffer():
    b = SampleBuffer()
    b.append(_entry())
    b.append(_entry())
    b.clear()
    assert len(b) == 0


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

def test_prune_no_op_when_object_static():
    b = SampleBuffer(pos_threshold=0.05)
    b.append(_entry(obj_xy=(0.10, 0.20)))
    b.append(_entry(obj_xy=(0.10, 0.20)))
    removed = b.prune(np.array([0.10, 0.20]))
    assert removed == 0
    assert len(b) == 2


def test_prune_drops_stale_pos_entries():
    b = SampleBuffer(pos_threshold=0.05)
    # one entry at (0,0), one at (0.20, 0)
    b.append(_entry(obj_xy=(0.0, 0.0)))
    b.append(_entry(obj_xy=(0.20, 0.0)))
    # prune relative to (0.0, 0.0): the second is 0.20 m away → drop
    removed = b.prune(np.array([0.0, 0.0]))
    assert removed == 1
    assert len(b) == 1
    assert b.snapshot()[0].obj_pos_xy[0] == 0.0


def test_prune_threshold_inclusive_boundary():
    b = SampleBuffer(pos_threshold=0.05)
    # entry at (0.05, 0) is exactly at the threshold → keep
    b.append(_entry(obj_xy=(0.05, 0.0)))
    removed = b.prune(np.array([0.0, 0.0]))
    assert removed == 0
    assert len(b) == 1


def test_prune_drops_stale_ang_entries():
    b = SampleBuffer(pos_threshold=10.0, ang_threshold=0.30)
    # both at same xy; one upright, one rotated 90°
    half = np.sqrt(0.5)
    b.append(_entry(obj_xy=(0, 0), quat=(1.0, 0.0, 0.0, 0.0)))
    b.append(_entry(obj_xy=(0, 0), quat=(half, 0.0, 0.0, half)))
    removed = b.prune(np.array([0.0, 0.0]),
                      obj_quat_now=np.array([1.0, 0.0, 0.0, 0.0]))
    assert removed == 1
    surviving = b.snapshot()[0]
    assert surviving.obj_quat[0] == 1.0


def test_prune_skips_ang_check_if_either_quat_missing():
    b = SampleBuffer(pos_threshold=10.0, ang_threshold=0.01)
    half = np.sqrt(0.5)
    # entry has quat, but caller passes None for obj_quat_now → no ang check
    b.append(_entry(obj_xy=(0, 0), quat=(half, 0.0, 0.0, half)))
    removed = b.prune(np.array([0.0, 0.0]), obj_quat_now=None)
    assert removed == 0
    assert len(b) == 1


# ---------------------------------------------------------------------------
# Aging
# ---------------------------------------------------------------------------

def test_tick_age_increments_all():
    b = SampleBuffer()
    b.append(_entry())
    b.append(_entry())
    b.tick_age()
    b.tick_age()
    for s in b.snapshot():
        assert s.age_steps == 2


# ---------------------------------------------------------------------------
# Iteration
# ---------------------------------------------------------------------------

def test_iter_yields_all_entries_in_order():
    b = SampleBuffer()
    for c in [5.0, 6.0, 7.0]:
        b.append(_entry(cost=c))
    costs = [s.cost for s in b]
    assert costs == [5.0, 6.0, 7.0]
