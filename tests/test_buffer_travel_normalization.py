"""Buffer travel-cost normalization — reference cc:2064-2072 / cc:2113-2118.

The reference stores buffer costs TRAVEL-FREE at append:
  cc:2064-2072  sample_costs_buffer_[i] = all_sample_costs_[i] - travel_cost
and re-adds CURRENT-EE travel at augmentation:
  cc:2113-2118  lowest_buffer_cost += travel_cost_per_meter * ||buf - ee_now||

The port previously stored c_sample verbatim (append-time travel baked in)
and replayed it verbatim at augmentation — stale travel, flagged in the p132
anatomy ("t=0 cost replayed verbatim incl. stale travel"). Inert at the
current configs (w_travel = 0 everywhere) but wrong for any travel-weighted
config. Distance convention: the port's own 3D ||sample - ee|| (matching
inner_solve travel_dist, so subtract/re-add are self-consistent); the
reference uses xy head(2) — a pre-existing metric divergence out of scope
here.
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


def _result(pos, c_sample, travel_penalty=0.0, feasible=True):
    p = np.asarray(pos, dtype=float)
    return SampleResult(
        sample_pos      = p,
        is_current_ee   = False,
        q_seed          = np.zeros(7),
        ee_pos_resolved = p.copy(),
        ik_err          = 0.0,
        ik_iters        = 1,
        feasible        = feasible,
        c_C3_raw        = float(c_sample),
        align_score     = 0.0,
        align_bonus     = 0.0,
        travel_dist     = 0.0,
        travel_penalty  = float(travel_penalty),
        rot_score       = 0.0,
        rot_bonus       = 0.0,
        c_sample        = float(c_sample),
        u_seq           = None,
        x_seq           = None,
    )


def _controller(prev_mode="c3", w_travel=200.0):
    c = SamplingC3Controller.__new__(SamplingC3Controller)
    c.params = SamplingC3Params()
    c.params.w_travel = float(w_travel)
    c.buffer = SampleBuffer(capacity=10,
                            pos_threshold=10.0, ang_threshold=10.0)
    c.unsuccessful_buffer = UnsuccessfulSampleBuffer()
    c._prev_mode = prev_mode
    c.log_diag = False
    c._step = 100
    return c


_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
_OBJ_XY = np.array([0.5, 0.0])


def test_append_stores_travel_free_cost():
    """cc:2064-2072: append-time travel is stripped before storage."""
    c = _controller()
    r = _result((0.6, 0.1, 0.034), c_sample=1000.0, travel_penalty=300.0)
    c._update_buffer([_result((0.5, 0.0, 0.034), 5000.0), r],
                     _OBJ_XY, _QUAT, labels=["current", "strat_0"])
    assert len(c.buffer) == 1
    assert c.buffer.best_with_position().cost == 700.0   # 1000 - 300


def test_augment_cost_readds_current_travel():
    """cc:2113-2118: augmentation prices the stored travel-free cost plus
    CURRENT-EE travel."""
    c = _controller(w_travel=200.0)
    entry = BufferedSample(
        position   = np.array([0.6, 0.0, 0.034]),
        cost       = 700.0,                       # travel-free
        obj_pos_xy = _OBJ_XY.copy(),
    )
    ee_now = np.array([0.5, 0.0, 0.034])          # 0.10 m away
    priced = c._buffer_cost_with_current_travel(entry, ee_now)
    assert priced == 700.0 + 200.0 * 0.10


def test_augment_cost_zero_travel_weight_is_identity():
    c = _controller(w_travel=0.0)
    entry = BufferedSample(
        position   = np.array([0.9, 0.9, 0.034]),
        cost       = 700.0,
        obj_pos_xy = _OBJ_XY.copy(),
    )
    priced = c._buffer_cost_with_current_travel(
        entry, np.array([0.0, 0.0, 0.0]))
    assert priced == 700.0
