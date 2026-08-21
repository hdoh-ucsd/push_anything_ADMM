"""Backend-neutral batched C3+ solver API (GPU-acceleration plan, Phase 1).

Push Anything evaluates several candidate end-effector placements per MPC
tick and solves an independent C3+ problem for each. Today that is a Python
loop in `control/sampling_c3/inner_solve.py`. This module gives that loop a
*named, backend-neutral interface* so a batched backend can be substituted
without touching the controller:

    batch  = C3PlusProblemBatch.from_instances([...])
    sol    = CpuC3PlusSolver(solver).solve_batch(batch)
    k_star = sol.best_candidate_index

Design constraints this file honours:

* **No PyDrake objects cross this boundary.** Everything is plain NumPy, so
  the same container can be uploaded to a device without translation.
* **Fixed shapes.** Every candidate in a batch shares
  ``(N, n_x, n_u, n_lambda)``. Measured on the box task, contact counts do
  NOT vary: 246/246 candidates over 41 ticks had ``n_c = 5``. A
  ``contact_mask`` is carried anyway so multi-contact and SE(3) work later
  can pad to ``C_max`` without changing this interface.
* **The CPU backend is the reference.** It loops internally and calls the
  existing, validated `C3Solver._solve_c3plus` unchanged, so adopting this
  API changes no numerics.

Deliberately NOT here: cost/ranking logic (that lives in the controller and
mixes in alignment/travel terms), and any solver parameter. This container
transports problems, it does not decide anything.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

# The LCS + cost arrays one C3+ instance needs. These names match the
# keyword arguments of C3Solver._solve_c3plus exactly, so an instance dict
# can be splatted into it without a translation layer.
_REQUIRED = (
    "x0", "A", "B_ctrl", "D", "d", "E", "F", "H", "c_lcs",
    "J_n", "J_t", "mu", "Q", "R", "QN", "x_ref", "N", "admm_iter",
    "torque_limit",
)
_OPTIONAL = ("phi", "u_lower", "u_upper", "ee_velocity_bounds",
             "ee_vel_state_indices")


@dataclass(frozen=True)
class C3PlusQPData:
    """The assembled numeric QP for ONE C3+ instance.

    This is the output of `C3Solver._assemble_c3plus_qp`, extracted so the
    CPU path and any future batched/GPU backend consume one assembly rather
    than maintaining two that can drift.

    `box_blocks` is an ORDERED tuple of `(idx, lo, hi)` triples, one per
    original `AddBoundingBoxConstraint` call, deliberately NOT merged into a
    single selector. Drake/OSQP row ordering follows the order constraints
    are added, and merging them would change the row layout and therefore
    the arithmetic -- which would break the byte-identity this refactor is
    required to preserve.

    `P` is the un-augmented cost Hessian; `P_sym` is
    `0.5 (P + rho*aug + transpose) + 1e-8 I` as the solver builds it. Both
    are carried because the ADMM loop's rho ramp needs each.
    """

    P: np.ndarray
    P_sym: np.ndarray
    q_ref: np.ndarray
    C_eq: np.ndarray
    b_eq: np.ndarray
    box_blocks: tuple
    n_x: int
    n_u: int
    n_lambda: int
    N: int
    TOT: int
    total_dim: int
    SX: int
    SL: int
    SU: int
    SE: int


@dataclass(frozen=True)
class C3PlusProblemBatch:
    """B independent C3+ problems that share shapes.

    `instances` holds one kwargs dict per candidate, in candidate order.
    Candidate order is load-bearing: the controller indexes results
    positionally (mode-switch and the prev-repos inflation both do), so a
    backend MUST return results in the order it received them.
    """

    instances: Sequence[dict]
    contact_mask: np.ndarray          # (B, C) bool; all-True on planar tasks
    n_x: int
    n_u: int
    n_lambda: int
    horizon: int
    warm_start: Optional[dict] = None
    labels: Sequence[str] = field(default_factory=tuple)

    @property
    def candidate_count(self) -> int:
        return len(self.instances)

    @property
    def total_dim(self) -> int:
        """Length of one candidate's stacked decision vector z."""
        tot = self.n_x + 2 * self.n_lambda + self.n_u
        return self.horizon * tot + self.n_x

    @classmethod
    def from_instances(cls, instances: Sequence[dict],
                       labels: Sequence[str] = (),
                       warm_start: Optional[dict] = None
                       ) -> "C3PlusProblemBatch":
        """Validate shape uniformity and build the batch.

        Raises rather than padding: a silent shape mismatch would give the
        wrong candidate the wrong dynamics, which is exactly the class of
        bug that made the parallel path return NaN costs for a whole run.
        """
        if not instances:
            raise ValueError("C3PlusProblemBatch needs at least one instance")
        for k, inst in enumerate(instances):
            missing = [f for f in _REQUIRED if f not in inst]
            if missing:
                raise ValueError(
                    f"candidate {k} is missing required fields: {missing}")

        ref = instances[0]
        n_x = int(np.asarray(ref["x0"]).shape[0])
        n_u = int(np.asarray(ref["R"]).shape[0])
        n_lambda = int(np.asarray(ref["E"]).shape[0])
        horizon = int(ref["N"])

        for k, inst in enumerate(instances[1:], start=1):
            got = (int(np.asarray(inst["x0"]).shape[0]),
                   int(np.asarray(inst["R"]).shape[0]),
                   int(np.asarray(inst["E"]).shape[0]),
                   int(inst["N"]))
            if got != (n_x, n_u, n_lambda, horizon):
                raise ValueError(
                    f"candidate {k} shape {got} != candidate 0 "
                    f"{(n_x, n_u, n_lambda, horizon)}; group candidates by "
                    f"shape before batching, or pad to C_max")

        n_c = int(np.asarray(ref["J_n"]).shape[0])
        mask = np.ones((len(instances), n_c), dtype=bool)
        return cls(instances=list(instances), contact_mask=mask, n_x=n_x,
                   n_u=n_u, n_lambda=n_lambda, horizon=horizon,
                   warm_start=warm_start, labels=tuple(labels))


@dataclass
class C3PlusSolutionBatch:
    """Per-candidate results, in the order the batch supplied them."""

    u_seqs: list                       # B x (N, n_u)
    x_seqs: list                       # B x (N+1, n_x)
    candidate_costs: np.ndarray        # (B,)
    converged: np.ndarray              # (B,) bool
    primal_residuals: np.ndarray       # (B,)
    dual_residuals: np.ndarray         # (B,)
    iteration_counts: np.ndarray       # (B,) int
    failed: np.ndarray                 # (B,) bool -- solver raised

    @property
    def candidate_count(self) -> int:
        return len(self.u_seqs)

    @property
    def best_candidate_index(self) -> int:
        """argmin over finite costs. NaN costs cannot win -- that failure
        mode silently disabled repositioning for an entire run once."""
        costs = np.where(np.isfinite(self.candidate_costs),
                         self.candidate_costs, np.inf)
        if not np.any(np.isfinite(costs)):
            raise RuntimeError("every candidate cost is non-finite; "
                               "no candidate can be selected")
        return int(np.argmin(costs))

    @property
    def best_first_control(self) -> np.ndarray:
        return np.asarray(self.u_seqs[self.best_candidate_index])[0]

    def selected_trajectory(self) -> np.ndarray:
        return np.asarray(self.x_seqs[self.best_candidate_index])


class CpuC3PlusSolver:
    """Reference backend. Loops over candidates and calls the existing,
    validated `C3Solver._solve_c3plus` with unchanged arguments.

    This exists to give the GPU backend something to be correct *against*,
    not to be fast. It deliberately adds no numerics of its own.
    """

    name = "cpu"

    def __init__(self, solver: Any):
        self._solver = solver

    def solve_batch(self, batch: C3PlusProblemBatch) -> C3PlusSolutionBatch:
        B = batch.candidate_count
        u_seqs: list = [None] * B
        x_seqs: list = [None] * B
        costs = np.full(B, np.nan)
        conv = np.zeros(B, dtype=bool)
        pres = np.full(B, np.nan)
        dres = np.full(B, np.nan)
        iters = np.zeros(B, dtype=int)
        failed = np.zeros(B, dtype=bool)

        for k, inst in enumerate(batch.instances):
            try:
                u_seq, x_seq = self._solver._solve_c3plus(**inst)
            except Exception:
                # Mirror the controller's tolerance for a single bad
                # candidate, but record it: a swallowed failure that looks
                # like a valid NaN cost is the defect this field exists to
                # make visible.
                failed[k] = True
                continue
            u_seqs[k] = u_seq
            x_seqs[k] = x_seq
            conv[k] = bool(getattr(self._solver, "_last_converged", False))
            pres[k] = float(getattr(self._solver, "_last_pr_final", np.nan))
            dres[k] = float(getattr(self._solver, "_last_dr_final", np.nan))
            iters[k] = int(getattr(self._solver, "_last_iters_used", 0))

        return C3PlusSolutionBatch(
            u_seqs=u_seqs, x_seqs=x_seqs, candidate_costs=costs,
            converged=conv, primal_residuals=pres, dual_residuals=dres,
            iteration_counts=iters, failed=failed)
