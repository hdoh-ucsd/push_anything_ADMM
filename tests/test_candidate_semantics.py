"""Phase 2A/2B: candidate warm-start semantics, and the order-invariance
property that licenses batched or concurrent candidate evaluation.

The distinction is not cosmetic. Under LEGACY_ORDERED the candidate loop
reuses one solver whose `_u_prev_solve` is rewritten after every solve, so
candidate k warm-starts k+1 and the loop is order-dependent -- a parallel
batch cannot reproduce it. INDEPENDENT_BATCH and REFERENCE_RESET both give
every candidate the same initialization, so solve order cannot matter.

These tests operate at the solver level on real corpus instances, which is
the only place the property can be checked exactly (a sim log diff cannot:
two runs of identical code already diverge because wall clock feeds the
x-pred clamp).
"""
import glob
import itertools

import numpy as np
import pytest

from control.solver_api import (CandidateSemantics, C3PlusProblemBatch,
                                CpuC3PlusSolver)

CORPUS = sorted(glob.glob("audit_output/admm_corpus/inst_*[0-9].npz"))
needs_corpus = pytest.mark.skipif(
    not CORPUS, reason="run scripts/gpu/dump_admm_corpus.sh first")


def args_of(d):
    def _opt(k):
        v = d[k]
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


def solve_candidates(instances, semantics, order, u_prev_entry=None):
    """Solve `instances` in `order` under `semantics`, returning results
    keyed by ORIGINAL candidate index. Mirrors the serial loop in
    inner_solve.evaluate_samples."""
    sem = CandidateSemantics.coerce(semantics)
    solver = fresh_solver(instances[0]["_meta"])
    solver._u_prev_solve = u_prev_entry
    out = {}
    for k in order:
        if sem is CandidateSemantics.INDEPENDENT_BATCH:
            solver._u_prev_solve = u_prev_entry
        elif sem is CandidateSemantics.REFERENCE_RESET:
            solver._u_prev_solve = None
        # LEGACY_ORDERED: leave whatever the previous candidate wrote.
        out[k] = solver._solve_c3plus(**instances[k]["args"])[0]
    return out


def load_candidates(n=5):
    """A deterministic stand-in candidate set: n distinct real instances."""
    cands = []
    for p in CORPUS[:n]:
        d = np.load(p, allow_pickle=True)
        cands.append({"args": args_of(d), "_meta": d})
    return cands


# ------------------------------------------------------------------ 2A: API

def test_semantics_names_and_aliases():
    CS = CandidateSemantics
    assert CS.coerce("legacy_ordered") is CS.LEGACY_ORDERED
    assert CS.coerce("ordered") is CS.LEGACY_ORDERED
    assert CS.coerce("reference_reset") is CS.REFERENCE_RESET
    assert CS.coerce("reset") is CS.REFERENCE_RESET
    assert CS.coerce("independent_batch") is CS.INDEPENDENT_BATCH
    assert CS.coerce("independent") is CS.INDEPENDENT_BATCH
    assert CS.coerce(CS.INDEPENDENT_BATCH) is CS.INDEPENDENT_BATCH
    with pytest.raises(ValueError, match="unknown candidate semantics"):
        CS.coerce("nope")


def test_only_legacy_is_order_dependent():
    CS = CandidateSemantics
    assert not CS.LEGACY_ORDERED.is_order_invariant
    assert CS.REFERENCE_RESET.is_order_invariant
    assert CS.INDEPENDENT_BATCH.is_order_invariant


@needs_corpus
def test_reference_reset_equals_zero_u_prev():
    """REFERENCE_RESET must reproduce the C++ semantics, where a fresh C3 has
    u_sol_ = zeros. The port encodes 'no history' as None; the two must be
    arithmetically identical, since the term is -2*R@0 = 0."""
    d = np.load(CORPUS[0], allow_pickle=True)
    a = args_of(d)
    s1 = fresh_solver(d)
    s1._u_prev_solve = None
    u_none = s1._solve_c3plus(**args_of(d))[0]
    s2 = fresh_solver(d)
    s2._u_prev_solve = np.zeros((int(d["N"]), int(d["n_u"])))
    u_zero = s2._solve_c3plus(**args_of(d))[0]
    assert np.array_equal(u_none, u_zero), (
        "None and zeros must be identical or REFERENCE_RESET does not "
        "reproduce the reference")


# ------------------------------------------------- 2B: order invariance

PERMUTATIONS = [
    [0, 1, 2, 3, 4],
    [4, 3, 2, 1, 0],
    [2, 0, 4, 1, 3],       # fixed "random" A
    [1, 4, 0, 3, 2],       # fixed "random" B
]


@needs_corpus
@pytest.mark.parametrize("semantics",
                         ["independent_batch", "reference_reset"])
def test_order_invariance(semantics):
    """Each candidate's own result must not depend on when it was solved."""
    cands = load_candidates(5)
    if len(cands) < 5:
        pytest.skip("need 5 corpus instances")
    u_prev_entry = np.full((int(cands[0]["_meta"]["N"]),
                            int(cands[0]["_meta"]["n_u"])), 0.3)

    base = solve_candidates(cands, semantics, PERMUTATIONS[0], u_prev_entry)
    for perm in PERMUTATIONS[1:]:
        got = solve_candidates(cands, semantics, perm, u_prev_entry)
        for k in range(5):
            assert np.array_equal(base[k], got[k]), (
                f"{semantics}: candidate {k} changed under permutation "
                f"{perm} -- candidates are NOT independent, so a batched "
                f"backend would not be safe. Look for shared mutable state.")


@needs_corpus
def test_legacy_ordered_IS_order_dependent():
    """The converse, pinned deliberately: LEGACY_ORDERED must change under
    permutation. If this ever passes silently, the coupling was removed and
    the semantics documentation is stale."""
    cands = load_candidates(5)
    if len(cands) < 5:
        pytest.skip("need 5 corpus instances")
    base = solve_candidates(cands, "legacy_ordered", PERMUTATIONS[0])
    rev = solve_candidates(cands, "legacy_ordered", PERMUTATIONS[1])
    differs = any(not np.array_equal(base[k], rev[k]) for k in range(5))
    assert differs, (
        "LEGACY_ORDERED became order-invariant; if the _u_prev_solve chain "
        "was intentionally removed, update CandidateSemantics docs")


@needs_corpus
def test_independent_batch_selection_is_order_invariant():
    """The decision that actually reaches the robot -- which candidate wins
    -- must be permutation-stable under INDEPENDENT_BATCH."""
    cands = load_candidates(5)
    if len(cands) < 5:
        pytest.skip("need 5 corpus instances")
    u_prev_entry = np.zeros((int(cands[0]["_meta"]["N"]),
                             int(cands[0]["_meta"]["n_u"])))

    picks = []
    for perm in PERMUTATIONS:
        res = solve_candidates(cands, "independent_batch", perm, u_prev_entry)
        # deterministic surrogate objective: control effort over the horizon
        costs = np.array([float(np.sum(res[k] ** 2)) for k in range(5)])
        picks.append(int(np.argmin(costs)))
    assert len(set(picks)) == 1, (
        f"selected candidate varied with solve order: {picks}")


@needs_corpus
def test_independent_batch_ignores_entry_history_differences_per_candidate():
    """Every candidate must see the SAME u_prev -- not its own history."""
    cands = load_candidates(3)
    if len(cands) < 3:
        pytest.skip("need 3 corpus instances")
    entry = np.full((int(cands[0]["_meta"]["N"]),
                     int(cands[0]["_meta"]["n_u"])), 0.25)
    res = solve_candidates(cands, "independent_batch", [0, 1, 2], entry)

    # Solving candidate 2 alone with the same entry u_prev must match the
    # batched result exactly -- proof that candidates 0 and 1 contributed
    # nothing to it.
    solo = fresh_solver(cands[2]["_meta"])
    solo._u_prev_solve = entry
    u_solo = solo._solve_c3plus(**cands[2]["args"])[0]
    assert np.array_equal(res[2], u_solo)
