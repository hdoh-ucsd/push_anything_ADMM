# Candidate warm-start semantics

How the input-change warm start (`u_prev`) reaches each candidate decides
whether the candidate loop is order-dependent, and therefore whether a
batched or GPU backend can reproduce it at all.

API: `control/solver_api.py :: CandidateSemantics`
Runtime gate: `PORT_CANDIDATE_WARMSTART` (default `legacy_ordered`, inert).

---

## 1. C++ reference behaviour

Verified by reading the reference sources directly, not inferred.

`reference_repos/dairlib_sampling_c3/systems/controllers/sampling_based_c3_controller.cc`

```
 989   std::vector<std::shared_ptr<C3>> c3_objects(num_total_samples, nullptr);
 998   #pragma omp parallel for num_threads(num_threads_to_use_)
 999   for (int i = 0; i < num_total_samples; i++) {
1007     std::shared_ptr<C3> test_c3_object;      <- FRESH, declared inside
1084     test_c3_object->Solve(test_state);
1099     c3_objects.at(i) = test_c3_object;
```

`reference_repos/c3/core/c3.cc`

```
  91-97   u_sol_ initialised to VectorXd::Zero(n_u_) per knot
 269      void C3::Solve(const VectorXd& x0) {
 340-346    if (options_.penalize_input_change)
              input_costs_[i]->UpdateCoefficients(2*R, -2*R*u_sol_->at(i));
 488        u_sol_->at(i) updated after the solve
```

Line 340 runs **once at the top of each `Solve`**. Combined with a fresh C3
per candidate per tick (`u_sol_` = zeros), the linear term is always
`-2·R·0 = 0`.

> **`penalize_input_change` is effectively a no-op for sampled candidates in
> the reference.** There is no candidate-to-candidate chaining, and none
> across ticks either.

One exception: `cc:2187` re-pushes the retained `c3_buffer_plan_` as an extra
candidate, so the buffer candidate *does* carry state.

## 2. Legacy port behaviour

`_u_prev_solve = u_seq.copy()` at the end of every solve
(`admm_solver.py:2748`), read by the next solve as
`q_ref[u] += -2·R·u_prev` (`:1122`), gated on `penalize_input_change` —
which is ON for both canonical tasks (off only for legacy `push_t`).

The serial candidate loop reuses **one** solver, so candidate *k*
warm-starts *k+1*. This is **order-dependent** and **not** reference
behaviour. Calling it "reference ordered" is wrong.

The parallel path differs again: worker clones persist across ticks with
their own `_u_prev_solve`, and `ThreadPoolExecutor` decides which candidate
lands on which worker — so `threads>1` is *nondeterministically* coupled.

## 3. The three semantics

| | `u_prev` each candidate sees | Order-invariant | Reference |
|---|---|---|---|
| `legacy_ordered` | previous candidate's solution | **No** | No |
| `reference_reset` | zeros (no history) | Yes | **Yes** |
| `independent_batch` | the tick-entry `u_prev`, broadcast | Yes | No |

`reference_reset` uses `_u_prev_solve = None`, which is bit-identical to
zeros because the guard skips a term equal to `-2·R·0`.

## 4. Measured characterization

Box 60 s gate / T 180 s canonical, seed 0. **All six runs reached
tight PASS(final).**

| | box ms/step | T ms/step | argmin agree vs legacy (box / T) |
|---|---|---|---|
| `legacy_ordered` | 488.0 | 235.6 | — |
| `independent_batch` | 474.2 | 240.1 | 67.8% / 94.2% |
| `reference_reset` | 485.5 | **416.3** | 70.5% / 94.4% |

- Sequential chaining provides **no** measured performance benefit
  (independent is 2.8% faster on box, 1.9% slower on T).
- Selection changes materially under independent batching — ~32% of box
  ticks, 5.8% of T ticks — but task success does not.
- Divergence **grows with chain depth** (box, by the candidate legacy
  picked): 24.5% (k=0) → 47.4 → 57.1 → 75.0 → **86.7%** (k=4).
  k=0 24.5% vs k>0 55.8%. So the coupling's harm worsens exactly where
  batching becomes worthwhile.

## 5. The T latency-tail finding

`reference_reset` is **77% slower on T** while doing the same solve count.
Root-caused with `DIAG_OSQP_ITERS`: it is genuinely more solver work, and the
signature is a **tail**, not a mean shift.

| | ordered | reset |
|---|---|---|
| in-loop OSQP mean / **max** | 100.0 / **100** | 101.4 / **700** |
| final-QP mean / **max** | 226.9 / **300** | 264.1 / **1200** |

Means barely move; maxima blow up 7–12×. An average would have hidden this
entirely — hence the Phase 2I rule that every timing report carries
mean/median/p90/p95/p99/max.

Box shows no such penalty (485.5 vs 488.0): the effect is **task-dependent**.

## 6. GPU batch semantics

**`independent_batch`.** The useful warm start is *temporal* (previous MPC
tick → this tick), not *sequential* (candidate k → k+1). Independent batch
keeps the former and drops the latter, so a GPU backend broadcasts one
tick-entry `u_prev` to every candidate lane.

Requirements, enforced by `tests/test_candidate_semantics.py`:

1. capture `u_prev` exactly once at tick entry;
2. broadcast the same value to every candidate;
3. no candidate may mutate another's initialization;
4. candidates may be solved in any order;
5. results are identical under permutation.

Not chosen: `legacy_ordered` cannot be expressed in a batch and degrades
with candidate count. `reference_reset` is expressible and is the conformant
option, but carries T's latency tail.

## 7. CPU/GPU comparison policy

Keep these separate and never mix them in one comparison:

| Purpose | Semantics | `check_termination` |
|---|---|---|
| Reference conformance | `reference_reset` | 100 |
| Fast development | explicitly selected | 25 if explicitly enabled |
| GPU research | `independent_batch` | matched to the CPU baseline |

A GPU `independent_batch` / `check=25` result compared against a CPU
`reference_reset` / `check=100` result is **not** a GPU speedup. For any
performance claim, match: semantics, `check_termination`, candidate count,
horizon, config, seed, and stopping rule.
