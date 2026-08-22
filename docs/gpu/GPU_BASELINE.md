# GPU baseline — frozen

**Conclusion, stated up front:**

> The current planar Push Anything workload is **too small and too
> CPU-efficient for primitive-level GPU acceleration.** GPU development is
> paused, not abandoned. This document freezes the measurements so the
> question can be re-answered cheaply after the algorithm changes.

Frozen at commit `070c83c` (Phase 2) plus the hygiene work recorded below.
Everything here is measured on this machine; nothing is estimated except
where explicitly labelled *extrapolated*.

---

## 1. Environment

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti, 16 GB |
| Compute capability | **sm_120** (Blackwell) |
| Driver | 580.88 |
| CUDA runtime | 12.8 (via framework wheels) |
| CUDA toolkit / `nvcc` | **absent** — driver runtime only |
| CuPy | **14.2.0** — verified on sm_120; cuBLAS + cuSOLVER reachable |
| PyTorch | 2.11.0+cu128 — *legacy only*, see §8 |
| JAX | absent |
| pybind11 | absent |
| Python / NumPy / SciPy | 3.10.20 / 2.2.6 / 1.15.3 |
| CPU | Ryzen 7 7800X3D, 8c / 16t |

**fp64 caveat.** GeForce Blackwell runs float64 at 1/64 the fp32 rate
(~0.7 TFLOPS), roughly at parity with this CPU's theoretical peak. There is
no raw-throughput headroom; any GPU win must come from parallelism and
overhead elimination, not FLOPs.

## 2. Candidate semantics

Full treatment in `docs/gpu/candidate_semantics.md`. Summary:

| | `u_prev` per candidate | Order-invariant | Reference |
|---|---|---|---|
| `legacy_ordered` (default) | previous candidate's solution | No | No |
| `reference_reset` | zeros | Yes | **Yes** |
| `independent_batch` | tick-entry `u_prev`, broadcast | Yes | No |

The GPU target is **`independent_batch`**. The port's default
`legacy_ordered` is a port-specific divergence, not reference behaviour.

## 3. Problem dimensions (measured from the corpus)

| symbol | value |
|---|---|
| B — candidates per tick | **6** (5 sampled + current) |
| N — horizon | 10 |
| C — contacts | **5**, uniform across 246/246 candidates |
| n_x / n_u / n_lambda | 19 / 3 / 20 |
| QP dimension n | **639** |
| QP constraint rows m | 439 |
| `P_sym` nnz / density | 659 / **0.161 %** |
| `C_eq` nnz / density | 9,729 / 3.72 % |
| reduced KKT density | 8.98 % |
| ADMM iterations | 3 (fixed) |
| batch memory, fp64, B=6 | ~65 MB (0.4 % of VRAM) |

## 4. Where runtime actually goes (Task 4)

Wall-clock section profile, box task, seed 0, `PORT_SECTION_TIMING=1`
`PORT_SECTION_DISTRIBUTIONS=1`. Sections cover **96 %** of the measured
472 ms/tick, so this is the whole budget rather than a sample.
`avg_per_step_ms` in these runs: 479.2.

**Per call, milliseconds:**

| section | calls | mean | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| `admm.osqp_solve` | 3894 | 12.21 | 12.38 | — | 21.01 | 24.00 | **32.95** |
| `admm.final_qp` | 1298 | 18.76 | 19.03 | — | 24.96 | 28.76 | **31.88** |
| `admm.qp_build` | 5192 | **2.66** | **0.65** | 8.39 | 9.11 | 10.93 | **46.93** |
| `lcs.extract_dynamics` | 2596 | 1.29 | 1.23 | — | 2.87 | 4.41 | 6.94 |
| `admm.z_update` (projections) | 3894 | 0.13 | 0.12 | — | 0.19 | 0.28 | 0.59 |

Aggregated per `_solve_c3plus` (6.5 calls per control tick):

| component | ms/solve | % of solve |
|---|---|---|
| in-loop OSQP (3 solves) | 39.16 | 56.0 % |
| final QP (1 solve) | 19.50 | 27.9 % |
| QP assembly + update | 8.19 | 11.7 % |
| Drake LCS linearization | 2.71 | 3.9 % |
| projections | 0.41 | **0.6 %** |

Candidate generation, geometry, candidate scoring, simulation and logging
together fall inside the unaccounted ~4 %.

**Headline: ~8,000 OSQP inner iterations per control tick at 47.5 µs each —
81 % of wall time.** The port is an iteration-count machine.

**Two things only the distribution shows:**

1. `admm.qp_build` is **bimodal** — mean 2.66 but p50 0.65. It is called 4×
   per solve: one full assembly plus three `UpdateCoefficients` pushes. That
   *is* the setup-vs-solve split, and the mean is meaningless for it.
2. Tails are 2–3× the median (`osqp_solve` p50 12.38 → max 32.95). Any
   real-time claim must quote p95/p99, not a mean.

## 5. OSQP / global solve characterization (Task 5)

| question | answer |
|---|---|
| QP dimension | n = 639, m = 439 |
| sparsity | Hessian **0.161 %**, KKT 8.98 % |
| solves per candidate | 4 (3 ADMM iterations + 1 final QP) |
| solves per tick | ~26 (6.5 candidates × 4) |
| setup vs solve | assembly ≈ 0.65–2.66 ms/call vs solve 12.2 ms/call → solve dominates ~5–20× |
| factorization reuse | Drake/OSQP refactors when ρ changes; ρ changes **every** ADMM iteration |
| matrix reuse across candidates | **none** — every candidate is linearized at a different EE pose (distinct `J_n` hashes) |
| matrix reuse across ticks | none — a fresh `MathematicalProgram` per solve |
| what differs across candidates | everything numeric (A, B, D, d, E, F, H, c); only *structure* is shared |
| only q/bounds change? | **no** — `rho_scale=3` adds `2·Δρ·g` to the P diagonal each iteration, so the matrix changes (diagonally) too |
| iteration distribution | in-loop mean 275 (max 500 at `check=100`); final QP mean 409 |
| warm-start effect | OSQP `warm_starting=1` within a solve; the *candidate* warm start is the `u_prev` term, characterized separately |
| `check_termination` effect | QPs converge at ~68 iterations but pay ~247 at the reference value 100. Authorized A/B measured **2.03×** end-to-end on box, 1.21× on T, both tight PASS(final). Not landed — off-reference |
| worst-case tail | `osqp_solve` max 32.95 ms vs p50 12.38 ms |

**Future strategies, assessed — none implemented:**

| | strategy | assessment |
|---|---|---|
| A | batched direct solve | Measured. Loses: the bare GPU kernel-launch floor at these dimensions is **74.8 µs**, already 1.6× Drake's *entire* 47.5 µs iteration. |
| B | matrix-free PCG | Not viable now: `cond(K) ≈ 1.2e7` demands strong preconditioning, and the per-iteration work is already below a kernel launch. |
| C | GPU-native QP | OSQP 1.x has a `cuda` algebra backend, but `algebras_available()` returns `['builtin']`, the wheel ships no `ext_cuda`, and `cuosqp` has no installable distribution. Drake also vendors its own OSQP. Even if reachable, it solves one problem at a time and so sits at or above the launch floor. |
| D | **retain CPU OSQP** | **Current position.** It exploits sparsity the GPU cannot, and is 11× fewer flops per matvec than the dense equivalent. |
| E | hybrid CPU/GPU | Not justified while the GPU-suitable fraction is 0.6 %. |

## 6. GPU primitives and crossover

Chain: projection + dual update + residuals — the genuinely parallel part
(candidate × knot × contact). fp64, 200 reps.

| B | CPU p50 (ms) | GPU e2e p50 | GPU kernel p50 | speedup |
|---|---|---|---|---|
| 1 | 0.019 | 0.914 | 0.410 | 0.02× |
| 4 | 0.025 | 0.777 | 0.338 | 0.03× |
| 8 | 0.034 | 0.799 | 0.341 | 0.04× |
| 16 | 0.052 | 0.809 | 0.337 | 0.06× |
| 32 | 0.084 | 0.849 | 0.338 | 0.10× |
| 64 | 0.153 | 0.910 | 0.333 | 0.14× |

**The GPU never wins over the swept range — not even kernel-only.**

GPU kernel time is **flat** (0.337 ms mean over B=4..64, spread 0.0075 ms) —
purely launch-bound. CPU scales linearly at **2.12 µs/candidate + 16.9 µs**.

*Extrapolated* crossover: **B ≈ 151** kernel-only (data already resident),
**B ≈ 389** end-to-end. The port generates **B = 6**, ~25× below even the
kernel-only figure.

And the axis these primitives accelerate is **0.6 %** of solve time, so an
infinite speedup there is ~1.006× end-to-end. Both facts point the same way.

*Artifact note:* the B=1 GPU `max` of 265 ms is CUDA context initialization
on the first rep. Medians are robust; that mean and max are not.

## 7. Numerical parity

| primitive | agreement |
|---|---|
| complementarity projection | **exact** (selects existing values) |
| dual update | 1e-15 |
| residuals | 1e-13 (float summation order differs) |
| candidate cost / argmin | 1e-13 / exact |

Order-invariance under `independent_batch` and `reference_reset` is
**bit-identical** across 4 permutations, including the selected candidate.

## 8. Framework position

**CuPy is the GPU direction.** PyTorch remains installed solely because the
quarantined feasibility benchmarks in `control/gpu/legacy_torch/` and
`scripts/gpu/*.py` need it. Audited 2026-08-21:

| classification | what |
|---|---|
| active required | **none** |
| research legacy | `control/gpu/legacy_torch/`, 7 `scripts/gpu` benchmarks, 2 test files |
| unrelated ML dependency | **none** — no ML component in this repo uses torch |
| removable | yes, once the feasibility numbers are considered closed |

The active solver path imports **neither** framework — pinned by
`tests/test_gpu_optional_backend.py`.

## 9. Reproduction commands

```bash
# corpus (prerequisite for most of the below)
scripts/gpu/dump_admm_corpus.sh audit_output/admm_corpus

# CPU golden fixtures for independent_batch
PYTHONPATH=. python3 scripts/gpu/make_golden_fixtures.py

# candidate-count crossover (§6)
PYTHONPATH=. python3 scripts/gpu/primitive_sweep.py \
    --candidate-counts 1 4 8 16 32 64

# runtime breakdown with tails (§4)
PORT_SECTION_TIMING=1 PORT_SECTION_DISTRIBUTIONS=1 \
    scripts/gpu/run_gate.sh 25 /tmp/dist.log

# OSQP iteration distributions (§5)
DIAG_OSQP_ITERS=1 scripts/gpu/run_gate.sh 20 /tmp/iters.log

# candidate semantics A/B/C
scripts/gpu/warmstart_sweep.sh box 60 /tmp
PYTHONPATH=. python3 scripts/gpu/warmstart_compare.py \
    A=/tmp/ws_box_ordered.log B=/tmp/ws_box_independent.log \
    C=/tmp/ws_box_reset.log

# before/after Path B (§10)
PYTHONPATH=. python3 scripts/gpu/perf_snapshot.py --label before_pathb
PYTHONPATH=. python3 scripts/gpu/perf_snapshot.py --label after_pathb
PYTHONPATH=. python3 scripts/gpu/perf_snapshot.py --compare before_pathb after_pathb

# parity + hygiene tests (no GPU needed for the last one)
python3 -m pytest tests/test_gpu_cupy_primitives.py \
    tests/test_candidate_semantics.py tests/test_gpu_optional_backend.py -q
```

## 10. Re-profiling after Path B (Task 6)

`scripts/gpu/perf_snapshot.py` writes one JSON per run capturing **shape and
latency together** — a latency change is uninterpretable without knowing
whether the problem changed underneath it. Recorded: B, n_x, n_u, n_lambda,
QP dimension, nonzeros/density, C3+ iterations, solves/tick,
`avg_per_step_ms`, and mean/p50/p90/p95/p99/max per section, plus commit,
semantics, `check_termination`, seed and environment.

Baseline captured: `audit_output/perf_snapshots/baseline_pre_pathb.json`.

`--compare` prints shape and latency side by side and ends with the warning
that matters:

> the GPU crossover estimate is only valid for the shape it was measured at.

**Do not assume B ≈ 389 survives Path B.**

## 11. Conditions for resuming GPU development (Task 7)

Resume when a **measured** snapshot shows one of these — each is a threshold
derived from a measurement in this document, not a guess:

1. **Candidate count** rises toward the measured crossover. Current B = 6;
   kernel-only crossover ≈ 151. A material step (B ≥ ~32) justifies re-running
   `primitive_sweep.py`, not yet a port.
2. **QP dimension** grows materially. At n = 639 a dense GPU solve loses to
   sparse CPU; the measured (n, density) map showed the sign flipping by
   n ≈ 1278 at our density, and immediately at 5 % density.
3. **`n_lambda` / contact count** grows substantially — it drives both the QP
   dimension and the projection workload, the two axes that move the
   crossover at once.
4. **Hessian density** rises above ~5 %. The `P_sym` 0.161 % density is the
   single strongest reason the CPU wins; a coupled cost changes that.
5. **SE(3)** raises n_x, n_lambda and contact count together — the most
   likely trigger, and the reason this baseline exists.
6. **Path B adds meaningful per-candidate computation** that is itself
   parallel over candidates.
7. **Profiling shows a GPU-suitable global solve dominating**, i.e. the
   dominant cost stops being launch-bound small sparse solves.
8. **The GPU-suitable fraction exceeds ~10 %** of runtime. It is 0.6 % today;
   below ~10 % Amdahl caps any achievable win at ~1.1×.

Conversely, do **not** resume merely because new hardware arrives: the
binding constraints here are problem *size* and *sparsity*, plus fp64 rate on
consumer silicon — none of which a faster GeForce changes.

## 12. Related commits and documents

- `070c83c` — Phase 2: candidate semantics API, CuPy primitives, crossover
- `8f338da` — T warm-start sweep; reference `penalize_input_change` is a no-op
- `f9d461b` — reference has no cross-candidate chaining (C++ verified)
- `5a171a0` — `check_termination` A/B, 2.03× canonical
- `952d288` — CPU assembly extraction, 10/10 bit-identical
- `docs/gpu/candidate_semantics.md`, `docs/gpu/phase2_gpu_prototype.md`
- `docs/superpowers/plans/gpu-admm-baseline.txt` — raw measurement log
