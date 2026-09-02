"""_update_buffer must not re-append the augmented buffer entry.

Reference call order (push_anything_dev@257e3ed,
systems/controllers/sampling_based_c3_controller.cc):
  cc:1094  MaintainSampleBuffers(x_lcs_curr);      // prune + append FIRST
  cc:1097  AugmentSamplesWithBuffer(c3_objects);   // THEN inject buffer best

so the buffer-append loop never sees the augmented stale entry, and combined
with the exit-side removal (cc:1196-1198) a cached promise is citable at most
once. The port maintains AFTER augmenting, so without an explicit skip the
stale SampleResult re-enters the buffer with its original cost every c3 tick
— the entry self-replicates and the exit-side removal is defeated (p135:
all 121 c3 exits cited the SAME step-2 promise c=1454.81 for 180 s).
"""
import numpy as np

from control.sampling_c3.inner_solve import SampleResult
from control.sampling_c3.params import SamplingC3Params
from control.sampling_c3.sample_buffer import (
    BufferedSample,
    SampleBuffer,
    UnsuccessfulSampleBuffer,
)
from control.sampling_c3.sampling_based_c3_controller import SamplingC3Controller


def _result(pos, cost, feasible=True):
    p = np.asarray(pos, dtype=float)
    return SampleResult(
        sample_pos      = p,
        is_current_ee   = False,
        q_seed          = np.zeros(7),
        ee_pos_resolved = p.copy(),
        ik_err          = 0.0,
        ik_iters        = 1,
        feasible        = feasible,
        c_C3_raw        = float(cost),
        align_score     = 0.0,
        align_bonus     = 0.0,
        travel_dist     = 0.0,
        travel_penalty  = 0.0,
        rot_score       = 0.0,
        rot_bonus       = 0.0,
        c_sample        = float(cost),
        u_seq           = None,
        x_seq           = None,
    )


def _controller(prev_mode="c3"):
    """Bare controller with just the state _update_buffer touches."""
    c = SamplingC3Controller.__new__(SamplingC3Controller)
    c.params = SamplingC3Params()
    c.buffer = SampleBuffer(capacity=10,
                            pos_threshold=10.0, ang_threshold=10.0)
    c.unsuccessful_buffer = UnsuccessfulSampleBuffer()
    c._prev_mode = prev_mode
    c.log_diag = False
    c._step = 100
    return c


_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
_OBJ_XY = np.array([0.5, 0.0])


def test_buffer_labeled_entry_is_not_reappended():
    c = _controller(prev_mode="c3")
    stale = _result((0.470, 0.040, 0.034), 1454.8)

    # Candidate list as it stands AFTER AugmentSamplesWithBuffer on a c3
    # tick: [current, strat_0, <stale buffer re-injection>].
    results = [_result((0.5, -0.1, 0.034), 7000.0),
               _result((0.56, 0.04, 0.034), 7400.0),
               stale]
    labels = ["current", "strat_0", "buffer"]

    c._update_buffer(results, _OBJ_XY, _QUAT, labels=labels)

    stored = [s.result for s in c.buffer]
    assert all(r is not stale for r in stored), (
        "augmented buffer entry was re-appended — stale promise "
        "self-replicates (reference maintains BEFORE augmenting, cc:1094/1097)")
    # The genuinely fresh non-current sample IS appended.
    assert len(c.buffer) == 1
    assert c.buffer.best_with_position().result is results[1]


def test_fresh_samples_still_appended_without_labels():
    """Back-compat: callers that pass no labels keep the old behavior for
    fresh strategy samples (skip current, append the rest)."""
    c = _controller(prev_mode="c3")
    results = [_result((0.5, -0.1, 0.034), 7000.0),
               _result((0.56, 0.04, 0.034), 7400.0)]
    c._update_buffer(results, _OBJ_XY, _QUAT)
    assert len(c.buffer) == 1
    assert c.buffer.best_with_position().result is results[1]


def test_remove_near_purges_all_cached_failed_contact_neighbors():
    b = SampleBuffer(capacity=10, pos_threshold=10.0, ang_threshold=10.0)
    for x in (0.645, 0.651, 0.720):
        b.append(BufferedSample(
            position=np.array([x, -0.185, 0.04]), cost=x,
            obj_pos_xy=_OBJ_XY.copy(), obj_quat=_QUAT.copy()))
    removed = b.remove_near(np.array([0.645, -0.189, 0.04]), 0.05)
    assert removed == 2
    assert len(b) == 1
    assert b.best_with_position().position[0] == 0.720


def test_body_relative_bad_spot_follows_object_translation_and_yaw():
    b = UnsuccessfulSampleBuffer(unsuccessful_radius=0.02)
    b.append(BufferedSample(
        position=np.array([0.60, 0.00, 0.04]), cost=1.0,
        obj_pos_xy=np.array([0.50, 0.00]), obj_quat=_QUAT.copy(),
        position_body_xy=np.array([0.10, 0.00])))

    # Object moved and rotated +90 degrees: local +x is now world +y.
    q_yaw_90 = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    obj_now = np.array([0.70, -0.20])
    assert not b.sample_avoids_bad_spots(
        np.array([0.70, -0.10, 0.04]), obj_now, q_yaw_90)
    assert b.sample_avoids_bad_spots(
        np.array([0.60, 0.00, 0.04]), obj_now, q_yaw_90)

    # Pose retention must not erase a body-relative memory as it moves.
    assert b.prune(obj_now, q_yaw_90) == 0
    assert len(b) == 1


def test_world_relative_reposition_stall_is_pose_pruned():
    """Planner/arm stalls must expire as the object moves away."""
    b = UnsuccessfulSampleBuffer(
        unsuccessful_pos_retention=0.05,
        unsuccessful_ang_retention=0.50,
    )
    b.append(BufferedSample(
        position=np.array([0.60, 0.00, 0.04]), cost=1.0,
        obj_pos_xy=np.array([0.50, 0.00]), obj_quat=_QUAT.copy(),
        position_body_xy=None))
    assert b.prune(np.array([0.56, 0.00]), _QUAT) == 1
    assert len(b) == 0
