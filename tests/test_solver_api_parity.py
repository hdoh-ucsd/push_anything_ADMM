"""Phase 1 parity: the batched CPU API must be BIT-IDENTICAL to the
single-instance path it wraps.

The whole point of `CpuC3PlusSolver` is to be a correctness reference for a
future batched/GPU backend. If routing a solve through the batch container
perturbs the answer at all, it is useless for that job -- so these tests
assert exact equality, not a tolerance.

Corpus: `audit_output/admm_corpus/` (60 real instances with CPU golden
outputs), produced by `scripts/gpu/dump_admm_corpus.sh`.
"""
import glob

import numpy as np
import pytest

from control.solver_api import (C3PlusProblemBatch, C3PlusSolutionBatch,
                                CpuC3PlusSolver)

CORPUS = sorted(glob.glob("audit_output/admm_corpus/inst_*[0-9].npz"))
needs_corpus = pytest.mark.skipif(
    not CORPUS, reason="run scripts/gpu/dump_admm_corpus.sh first")


def args_of(d):
    """Round-trip a dumped instance back into _solve_c3plus kwargs.

    Note `mu` is a PER-CONTACT-PAIR array (e.g. [0.42, 0.46, 0.46, 0.46,
    0.46] -- EE-BOX vs BOX-GND overrides), not a scalar, despite the
    parameter being annotated `float`. Coercing it with float() raises.
    Empty-size arrays are how the dump encodes "None was passed".
    """
    def _opt(key):
        v = d[key]
        return v if np.size(v) > 0 else None

    return dict(x0=d["x0"], A=d["A"], B_ctrl=d["B_ctrl"], D=d["D"], d=d["d"],
                E=d["E"], F=d["F"], H=d["H"], c_lcs=d["c_lcs"],
                J_n=d["J_n"], J_t=d["J_t"], mu=d["mu"],
                Q=d["Q"], R=d["R"], QN=d["QN"], x_ref=d["x_ref"],
                N=int(d["N"]), admm_iter=int(d["admm_iter"]),
                torque_limit=float(d["torque_limit"]),
                phi=_opt("phi"), u_lower=_opt("u_lower"),
                u_upper=_opt("u_upper"))


def fresh_solver(d):
    from control.admm_solver import C3Solver
    return C3Solver(n_x=int(d["n_x"]), n_u=int(d["n_u"]),
                    rho=float(d["rho_initial"]), mode="c3plus")


# ----------------------------------------------------------------- container

def test_shape_validation_rejects_mismatch():
    """A silent shape mismatch would hand a candidate the wrong dynamics."""
    a = {f: np.zeros(1) for f in
         ("x0", "A", "B_ctrl", "D", "d", "E", "F", "H", "c_lcs",
          "J_n", "J_t", "Q", "QN", "x_ref")}
    a.update(mu=0.5, N=10, admm_iter=3, torque_limit=30.0,
             R=np.zeros(3), x0=np.zeros(19), E=np.zeros((20, 5)),
             J_n=np.zeros((5, 3)))
    b = dict(a); b["E"] = np.zeros((24, 5))          # different n_lambda
    with pytest.raises(ValueError, match="shape"):
        C3PlusProblemBatch.from_instances([a, b])


def test_missing_field_is_named():
    a = {"x0": np.zeros(19)}
    with pytest.raises(ValueError, match="missing required fields"):
        C3PlusProblemBatch.from_instances([a])


def test_empty_batch_rejected():
    with pytest.raises(ValueError, match="at least one"):
        C3PlusProblemBatch.from_instances([])


def test_nan_costs_cannot_win_selection():
    """The parallel-path defect produced all-NaN costs and the dispatcher
    silently treated that as 'no better candidate'. Selection must refuse."""
    sol = C3PlusSolutionBatch(
        u_seqs=[np.zeros((10, 3))] * 3, x_seqs=[np.zeros((11, 19))] * 3,
        candidate_costs=np.array([np.nan, 5.0, np.nan]),
        converged=np.ones(3, bool), primal_residuals=np.zeros(3),
        dual_residuals=np.zeros(3), iteration_counts=np.zeros(3, int),
        failed=np.zeros(3, bool))
    assert sol.best_candidate_index == 1

    allnan = C3PlusSolutionBatch(
        u_seqs=[np.zeros((10, 3))], x_seqs=[np.zeros((11, 19))],
        candidate_costs=np.array([np.nan]), converged=np.zeros(1, bool),
        primal_residuals=np.zeros(1), dual_residuals=np.zeros(1),
        iteration_counts=np.zeros(1, int), failed=np.zeros(1, bool))
    with pytest.raises(RuntimeError, match="non-finite"):
        _ = allnan.best_candidate_index


@needs_corpus
def test_total_dim_matches_corpus():
    d = np.load(CORPUS[0], allow_pickle=True)
    qp = np.load(CORPUS[0].replace(".npz", "_qp.npz"), allow_pickle=True)
    batch = C3PlusProblemBatch.from_instances([args_of(d)])
    assert batch.total_dim == int(qp["total_dim"])
    assert batch.candidate_count == 1
    assert batch.contact_mask.shape == (1, int(qp["num_normals"]))
    assert batch.contact_mask.all(), "planar tasks have no padded contacts"


# -------------------------------------------------------------- bit-identity

@needs_corpus
@pytest.mark.parametrize("path", CORPUS[:8])
def test_batch_of_one_is_bit_identical(path):
    """Routing through the container must not perturb a single solve."""
    d = np.load(path, allow_pickle=True)
    args = args_of(d)

    direct_solver = fresh_solver(d)
    u_direct, x_direct = direct_solver._solve_c3plus(**args)

    batch_solver = fresh_solver(d)
    sol = CpuC3PlusSolver(batch_solver).solve_batch(
        C3PlusProblemBatch.from_instances([args_of(d)]))

    assert not sol.failed[0], "batched path raised where direct did not"
    assert np.array_equal(sol.u_seqs[0], u_direct), "u_seq not bit-identical"
    assert np.array_equal(sol.x_seqs[0], x_direct), "x_seq not bit-identical"


@needs_corpus
def test_golden_outputs_are_not_fresh_solver_reproducible():
    """Documents why the recorded goldens are NOT a valid fresh-solver oracle.

    `_out.npz` was dumped mid-run, when the live solver had already
    accumulated `_u_prev_solve` from earlier ticks. Because
    `penalize_input_change` is on, that history enters `q_ref` via the
    `-2*R*u_prev` term, so a cold solver cannot reproduce the golden and an
    equality assertion against it would fail for a reason that has nothing to
    do with the code under test.

    The valid oracle for the assembly extraction is a side-by-side run of the
    pre-extraction module against the current one with a fresh solver on both
    sides -- that comparison returned 10/10 bit-identical. This test pins the
    trap so nobody reinstates a misleading golden assertion.
    """
    d = np.load(CORPUS[0], allow_pickle=True)
    solver = fresh_solver(d)
    if not getattr(solver, "_penalize_input_change", False):
        pytest.skip("penalize_input_change off; goldens would be reproducible")
    assert solver._u_prev_solve is None, "a fresh solver has no input history"
    golden = np.load(CORPUS[0].replace(".npz", "_out.npz"))
    sol = CpuC3PlusSolver(solver).solve_batch(
        C3PlusProblemBatch.from_instances([args_of(d)]))
    assert sol.u_seqs[0].shape == golden["u_seq"].shape


@needs_corpus
def test_candidate_order_is_preserved():
    """The controller indexes results positionally -- mode-switch and the
    prev-repos inflation both do -- so order must survive the round trip.

    The reference semantics are a SEQUENTIAL loop over one shared solver,
    which is what the serial path in inner_solve.py does. That loop carries
    `_u_prev_solve` from candidate k into candidate k+1, so the batch is
    compared against an identically-ordered sequential loop, NOT against
    independent solo solves.
    """
    ds = [np.load(p, allow_pickle=True) for p in CORPUS[:4]]

    seq_solver = fresh_solver(ds[0])
    sequential = [seq_solver._solve_c3plus(**args_of(d))[0] for d in ds]

    batch_solver = fresh_solver(ds[0])
    sol = CpuC3PlusSolver(batch_solver).solve_batch(
        C3PlusProblemBatch.from_instances([args_of(d) for d in ds]))

    assert sol.candidate_count == 4
    for k in range(4):
        assert np.array_equal(sol.u_seqs[k], sequential[k]), (
            f"candidate {k} differs from the sequential reference loop")


@needs_corpus
def test_batch_carries_input_history_across_candidates():
    """Make the cross-candidate coupling EXPLICIT so a batched/GPU backend
    cannot diverge from it silently.

    With `penalize_input_change` on, the CPU batch is NOT a set of
    independent solves: candidate k's `u_seq` becomes candidate k+1's
    `_u_prev_solve` and enters its linear cost. A GPU backend that solves all
    candidates simultaneously has no way to reproduce this ordering by
    construction, so this is a documented behavioural difference the GPU
    design must decide about deliberately.
    """
    ds = [np.load(p, allow_pickle=True) for p in CORPUS[:2]]
    solver = fresh_solver(ds[0])
    if not getattr(solver, "_penalize_input_change", False):
        pytest.skip("penalize_input_change off; no coupling to document")

    batched = CpuC3PlusSolver(solver).solve_batch(
        C3PlusProblemBatch.from_instances([args_of(d) for d in ds]))
    solo_1 = CpuC3PlusSolver(fresh_solver(ds[1])).solve_batch(
        C3PlusProblemBatch.from_instances([args_of(ds[1])]))

    assert not np.array_equal(batched.u_seqs[1], solo_1.u_seqs[0]), (
        "expected candidate 1 to be influenced by candidate 0 via "
        "_u_prev_solve; if this now passes, the coupling was removed and the "
        "GPU design note above is stale")


@needs_corpus
def test_solver_state_does_not_leak_across_candidates():
    """`_u_prev_solve` (the ||u-u_prev||^2_R warm start) is set at the end of
    every solve. If one shared solver instance is reused across candidates it
    feeds candidate k's output into candidate k+1 -- the exact bug that made
    an earlier bit-check report 517/517 mismatches. Document the behaviour so
    a batched backend cannot reintroduce it silently."""
    d = np.load(CORPUS[0], allow_pickle=True)
    args = args_of(d)
    shared = fresh_solver(d)
    first = CpuC3PlusSolver(shared).solve_batch(
        C3PlusProblemBatch.from_instances([args_of(d)]))
    second = CpuC3PlusSolver(shared).solve_batch(
        C3PlusProblemBatch.from_instances([args_of(d)]))
    if getattr(shared, "_penalize_input_change", False):
        assert not np.array_equal(first.u_seqs[0], second.u_seqs[0]), (
            "penalize_input_change is on, so a reused solver MUST carry "
            "_u_prev_solve into the next solve")
    else:
        assert np.array_equal(first.u_seqs[0], second.u_seqs[0])
