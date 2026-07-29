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
