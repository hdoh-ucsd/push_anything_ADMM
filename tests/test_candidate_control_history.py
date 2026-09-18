"""Seed-42 candidate independence, without an external/deleted solve corpus.

This dispatch harness maps each sample ID to a fixed synthetic planar-object
C3+ problem, and runs the real C3Solver, ADMM, projection, and ranking cost.
The state is [object_x, object_y, object_yaw, vx, vy, yaw_rate]. Different
candidate actuation matrices represent different hypothetical approaches.
Rank flags (current-EE/full-iters) intentionally do not change these fixed
problems: B must remain the same numerical candidate when tested alone.
This is a solver/batch-history regression, not a simulator geometry fixture.
"""
import copy
import queue
import threading
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pydrake")

from control.admm_solver import C3Solver
from control.sampling_c3.inner_solve import InnerSolver, traj_cost


def _planar_problems():
    rng = np.random.default_rng(42)
    dt, mass, nx, nu, nl, horizon = 0.05, 0.2, 6, 3, 6, 4
    A = np.eye(nx)
    A[:3, 3:] = dt * np.eye(3)
    D = np.zeros((nx, nl))
    D[3, 1] = dt / mass
    E = np.zeros((nl, nx))
    E[1, 3] = 1.0
    # One Stewart-Trinkle normal with four tangent slots. This is the same
    # synthetic contact construction as test_c3plus_vs_c3_smoke, in SE(2).
    F = np.zeros((nl, nl))
    F[0, 1], F[0, 2:], F[1, 1], F[2:, 0] = 0.4, -1.0, dt / mass, 1.0
    Q = np.diag([100.0, 100.0, 30.0, 1.0, 1.0, 1.0])
    common = dict(
        x0=np.zeros(nx), A=A, D=D, d=np.zeros(nx), E=E, F=F,
        c_lcs=np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0]),
        J_n=np.array([[1.0, 0.0, 0.0]]), J_t=np.zeros((4, 3)), mu=0.4,
        Q=Q, R=np.eye(nu), QN=10 * Q,
        x_ref=np.array([0.3, 0.1, 0.08, 0.0, 0.0, 0.0]),
        N=horizon, admm_iter=3, torque_limit=30.0, phi=np.array([0.005]),
    )
    problems = []
    for _ in range(3):
        B = np.zeros((nx, nu))
        B[3:, :] = dt / mass * (np.eye(3) + rng.normal(0, 0.1, (3, 3)))
        H = np.zeros((nl, nu))
        H[1, :] = B[3, :]
        problems.append(dict(common, B_ctrl=B, H=H))
    return problems, rng.normal(0, 0.3, (horizon, nu))


class _NumericalCandidates(InnerSolver):
    """Exercise the production dispatch with fixed numerical LCS inputs."""

    def __init__(self, problems, entry_history, threads=1):
        self.problems = problems
        self.solver = C3Solver(6, 3, mode="c3plus", penalize_input_change=True)
        self.solver._u_prev_solve = entry_history
        self._num_threads_to_use = threads
        self._diagram = object()  # contexts are unused by this numeric fixture
        self.plant = SimpleNamespace(SetPositions=lambda *_: None,
                                     SetVelocities=lambda *_: None)
        self._worker_kits = []
        self._worker_queue = None
        self._worker_lock = threading.Lock()
        self.seen_histories = []
        self.fail_candidate = None

    def _lazy_init_worker_kits(self, n_workers):
        # Keep workers alive across dispatches, as the production pool does.
        if not self._worker_kits:
            for _ in range(n_workers):
                clone = copy.copy(self)
                clone.solver = C3Solver(6, 3, mode="c3plus",
                                        penalize_input_change=True)
                # Deliberately stale worker history must never enter a solve.
                clone.solver._u_prev_solve = np.full((4, 3), 19.0)
                clone.seen_histories = []
                self._worker_kits.append((clone, object()))
        self._worker_queue = queue.Queue()
        for kit in self._worker_kits:
            self._worker_queue.put(kit)

    def evaluate_sample(self, sample_pos, **_kwargs):
        candidate = int(sample_pos[0])
        ref = self.solver._u_prev_solve
        self.seen_histories.append(None if ref is None else ref.copy())
        if candidate == self.fail_candidate:
            if ref is not None:
                ref[:] += 1000.0  # exercise copy isolation as well as finally
            raise RuntimeError("synthetic candidate failure")
        p = self.problems[candidate]
        u_seq, x_seq = self.solver.solve(**p)
        cost = traj_cost(x_seq, u_seq, p["Q"], p["R"], p["QN"], p["x_ref"])
        return SimpleNamespace(c_C3_raw=cost, c_sample=cost, u_seq=u_seq,
                               x_seq=x_seq, candidate=candidate, J_n=None)


def _evaluate(inner, order, *, threaded=False):
    results = inner.evaluate_samples(
        samples=[np.array([k, 0.0, 0.0]) for k in order],
        current_q=np.zeros(6), current_v=np.zeros(3), plant_ctx=None,
        target_xy=np.array([0.3, 0.1]), ee_pos_now=np.zeros(3),
        g_hat_3d=np.array([1.0, 0.0, 0.0]), use_threading=threaded,
    )
    return {r.candidate: r for r in results}


def _assert_same_candidate(a, b):
    # Tight across identical numerical problems, independent of the looser
    # physical solver stopping tolerance: no objective or ordering differs.
    np.testing.assert_allclose(a.c_C3_raw, b.c_C3_raw, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(a.u_seq[0], b.u_seq[0], rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(a.x_seq[-1, :3], b.x_seq[-1, :3],
                               rtol=1e-11, atol=1e-11)


@pytest.fixture(autouse=True)
def _default_candidate_mode(monkeypatch):
    monkeypatch.delenv("PORT_CANDIDATE_WARMSTART", raising=False)
    monkeypatch.delenv("PORT_CANDIDATE_ORDER", raising=False)
    monkeypatch.delenv("DIAG_SAMP_LCS_DUMP", raising=False)
    monkeypatch.delenv("DIAG_COST_BD_AT_TICK", raising=False)


def test_candidate_b_is_invariant_after_a_after_c_and_alone():
    problems, entry = _planar_problems()
    inner = _NumericalCandidates(problems, entry)
    b_after_a = _evaluate(inner, [0, 1, 2])[1]
    b_after_c = _evaluate(inner, [2, 1, 0])[1]
    b_alone = _evaluate(inner, [1])[1]
    _assert_same_candidate(b_after_a, b_after_c)
    _assert_same_candidate(b_after_a, b_alone)
    assert inner.solver._u_prev_solve is entry
    for seen in inner.seen_histories:
        np.testing.assert_array_equal(seen, entry)


def test_diagnostic_permutation_preserves_all_candidates(monkeypatch):
    problems, entry = _planar_problems()
    inner = _NumericalCandidates(problems, entry)
    expected = _evaluate(inner, [0, 1, 2])
    monkeypatch.setenv("PORT_CANDIDATE_ORDER", "reversed")
    got = _evaluate(inner, [0, 1, 2])
    for k in expected:
        _assert_same_candidate(expected[k], got[k])


def test_legacy_replay_demonstrates_fixture_catches_the_original_coupling(monkeypatch):
    monkeypatch.setenv("PORT_CANDIDATE_WARMSTART", "legacy_ordered")
    problems, entry = _planar_problems()
    b_after_a = _evaluate(_NumericalCandidates(problems, entry.copy()), [0, 1, 2])[1]
    b_after_c = _evaluate(_NumericalCandidates(problems, entry.copy()), [2, 1, 0])[1]
    assert abs(b_after_a.c_C3_raw - b_after_c.c_C3_raw) > 0.1
    assert np.linalg.norm(b_after_a.u_seq[0] - b_after_c.u_seq[0]) > 0.1
    assert np.linalg.norm(b_after_a.x_seq[-1, :3] - b_after_c.x_seq[-1, :3]) > 1e-3


def test_input_change_penalty_remains_and_committed_solve_advances_history():
    problems, entry = _planar_problems()
    inner = _NumericalCandidates(problems, entry)
    before = _evaluate(inner, [1])[1]
    without_history = _evaluate(_NumericalCandidates(problems, None), [1])[1]
    assert np.linalg.norm(before.u_seq[0] - without_history.u_seq[0]) > 1e-3
    # This direct solve stands for the existing base-MPC execution solve.
    committed_u, _ = inner.solver.solve(**problems[0])
    np.testing.assert_array_equal(inner.solver._u_prev_solve, committed_u)
    after = _evaluate(inner, [0, 1, 2])[1]
    standalone = _evaluate(_NumericalCandidates(problems, committed_u.copy()), [1])[1]
    _assert_same_candidate(after, standalone)
    assert np.linalg.norm(after.u_seq[0] - before.u_seq[0]) > 1e-3
    np.testing.assert_array_equal(inner.solver._u_prev_solve, committed_u)


@pytest.mark.parametrize("threads", [1, 2])
def test_parallel_workers_use_common_parent_history_and_survive_reuse(threads):
    problems, entry = _planar_problems()
    inner = _NumericalCandidates(problems, entry, threads=threads)
    expected = _evaluate(_NumericalCandidates(problems, entry.copy()), [1])[1]
    _assert_same_candidate(_evaluate(inner, [0, 1, 2], threaded=True)[1], expected)
    _assert_same_candidate(_evaluate(inner, [2, 1, 0], threaded=True)[1], expected)
    assert inner.solver._u_prev_solve is entry
    for clone, _ in inner._worker_kits:
        for seen in clone.seen_histories:
            np.testing.assert_array_equal(seen, entry)

    committed_u, _ = inner.solver.solve(**problems[2])
    expected_next = _evaluate(_NumericalCandidates(problems, committed_u.copy()), [1])[1]
    _assert_same_candidate(_evaluate(inner, [0, 1, 2], threaded=True)[1], expected_next)
    np.testing.assert_array_equal(inner.solver._u_prev_solve, committed_u)


@pytest.mark.parametrize("threaded", [False, True])
def test_candidate_exception_restores_executed_history_and_does_not_alias(threaded):
    problems, entry = _planar_problems()
    inner = _NumericalCandidates(problems, entry, threads=1)
    saved = entry.copy()
    inner.fail_candidate = 1
    with pytest.raises(RuntimeError, match="synthetic candidate failure"):
        _evaluate(inner, [0, 1, 2], threaded=threaded)
    assert inner.solver._u_prev_solve is entry
    np.testing.assert_array_equal(entry, saved)
    for clone, _ in inner._worker_kits:
        np.testing.assert_array_equal(clone.solver._u_prev_solve,
                                      np.full((4, 3), 19.0))


def test_no_prior_committed_control_stays_none_after_hypothetical_batch():
    problems, _ = _planar_problems()
    inner = _NumericalCandidates(problems, None)
    combined = _evaluate(inner, [0, 1, 2])[1]
    alone = _evaluate(inner, [1])[1]
    _assert_same_candidate(combined, alone)
    assert inner.solver._u_prev_solve is None
    assert all(seen is None for seen in inner.seen_histories)


def test_reference_reset_compatibility_keeps_committed_history(monkeypatch):
    monkeypatch.setenv("PORT_CANDIDATE_WARMSTART", "reference_reset")
    problems, entry = _planar_problems()
    inner = _NumericalCandidates(problems, entry)
    combined = _evaluate(inner, [0, 1, 2])[1]
    alone = _evaluate(inner, [1])[1]
    _assert_same_candidate(combined, alone)
    assert inner.solver._u_prev_solve is entry
    assert all(seen is None for seen in inner.seen_histories)
