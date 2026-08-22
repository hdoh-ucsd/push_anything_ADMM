# Phase 2 — GPU primitive prototype

Status: **GPU PRIMITIVES NOT READY** (see the gate at the end).

Everything here is measured on this machine at the commit recorded in
`audit_output/gpu_golden/independent_batch/metadata.json`.

---

## Environment (Phase 2D)

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti, **sm_120** (Blackwell), 16 GB |
| Driver | 580.88 |
| CUDA toolkit / `nvcc` | **absent** — driver runtime only |
| CuPy | **14.2.0**, working on sm_120; cuBLAS + cuSOLVER reachable |
| PyTorch | 2.11.0+cu128, working (installed during the earlier feasibility arc) |
| JAX | absent — not installed, not needed |
| pybind11 | absent |
| Python / NumPy / SciPy | 3.10.20 / 2.2.6 / 1.15.3 |
| CPU | Ryzen 7 7800X3D, 8c/16t |

**Framework choice: CuPy.** Verified on this GPU before committing to it
(`cp.cuda.Device(0).compute_capability == '120'`, batched Cholesky OK). It is
the smallest dependency change for a NumPy-shaped codebase and leaves the
`RawKernel` door open.

Note the package now carries torch-based modules
(`control/gpu/batched_qp.py`, `control/gpu/projection.py`) from the earlier
feasibility arc that concluded a full GPU solver loses at this problem size.
They are research artifacts, unused by Phase 2, and should be retired or
ported once the GPU direction is settled — two CUDA frameworks in one process
is not a state to keep.

**fp64 caveat.** GeForce Blackwell runs float64 at 1/64 the fp32 rate
(~0.7 TFLOPS), roughly at parity with this CPU. There is no raw-throughput
headroom here; any GPU win must come from parallelism and overhead
elimination.

## Batch representation (Phase 2E)

The CPU/GPU boundary is the extracted assembly,
`C3Solver._assemble_c3plus_qp` → `C3PlusQPData`. **No PyDrake object crosses
it.**

```
PyDrake / CPU  --(linearize)-->  C3PlusQPData (plain NumPy)  --(upload)-->  GPU
```

Measured dimensions (box task):

| symbol | value | note |
|---|---|---|
| B (candidates) | **6** | 5 sampled + current; 40 of 41 ticks had exactly 6 |
| N (horizon) | 10 | |
| C (contacts) | **5** | uniform: 246/246 candidates over 41 ticks |
| n_x / n_u / n_lambda | 19 / 3 / 20 | |
| TOT / total_dim | 62 / **639** | `N·TOT + n_x` |
| constraint rows m | 439 | 409 equality + 30 box |
| batch memory (fp64) | ~65 MB at B=6 | 0.4% of VRAM |

**Contact counts do not vary**, so fixed-shape batching needs no padding
today. `contact_mask` is carried in `C3PlusProblemBatch` anyway and is
all-True; multi-contact and SE(3) work can pad to `C_max` without changing
the interface.

Layout: structure-of-arrays. The projection axis is
`(B, N, n_lambda)` contiguous in the trailing dimension, so the whole
candidate × knot × slot space is one elementwise kernel.

## Primitives (Phase 2F)

Checked which ones actually exist in the C3+ path rather than assuming:

| primitive | exists? | notes |
|---|---|---|
| complementarity projection | **yes** | Bui eq.(12) — the only projection |
| dual update | **yes** | `omega += z - delta` |
| residual calculation | **yes** | over λ/η slots only, not all of `z` |
| candidate cost / argmin | partly | the controller's ranking cost adds alignment/travel terms computed *outside* the solver |
| friction / SOC projection | **no** | Lorentz code belongs to the falsified `mode='c3'` lineage |
| Drake 4D tangent-basis projection | **no** | not in this path |

So two of the brief's listed primitives have no C3+ counterpart to port.

Implemented in `control/gpu/cupy_primitives.py`; parity in
`tests/test_gpu_cupy_primitives.py` (21 tests):

- projection — **exact equality** (it selects existing values), across 4
  G-weight pairs, 3 seeds, a dense boundary grid including exact ties, and
  the golden fixture
- dual update — 1e-15
- residuals — 1e-13 (float summation order legitimately differs)
- cost / argmin — matches golden; argmin never selects a non-finite cost

## Candidate-count crossover (Phase 2G) and tails (Phase 2I)

Chain measured: projection + dual update + residuals, fp64, 200 reps.

| B | CPU p50 (ms) | GPU e2e p50 | GPU kernel p50 | speedup |
|---|---|---|---|---|
| 1 | 0.019 | 0.914 | 0.410 | 0.02× |
| 4 | 0.025 | 0.777 | 0.338 | 0.03× |
| 8 | 0.034 | 0.799 | 0.341 | 0.04× |
| 16 | 0.052 | 0.809 | 0.337 | 0.06× |
| 32 | 0.084 | 0.849 | 0.338 | 0.10× |
| 64 | 0.153 | 0.910 | 0.333 | 0.14× |

**The GPU never wins over the swept range — not even kernel-only.**

The shape of the result is the useful part: GPU kernel time is **flat**
(0.337 ms mean over B=4..64, spread 0.0075 ms) — purely launch-bound — while
CPU scales linearly at **2.12 µs per candidate + 16.9 µs fixed**.
Extrapolating:

- kernel-only crossover (data already resident): **B ≈ 151**
- end-to-end crossover (including transfer): **B ≈ 389**

The port generates **B = 6**, about **25× below** even the kernel-only
crossover.

Tail distributions are in the sweep output. One artifact to note honestly:
the B=1 GPU `max` of 265 ms is CUDA context initialization on the first rep;
medians are robust to it, means and maxima at B=1 are not.

And the decisive context: **projections are 0.6% of solve time**
(`admm.z_update`, 0.41 ms of a 70.0 ms solve). Even an infinite speedup on
this chain yields ~1.006× end-to-end.

## Do not port OSQP blindly (Phase 2H)

Already characterized:

- 81% of wall time is OSQP **inner iterations** (~8,023 per control tick at
  47.5 µs each)
- QP is n=639, m=439; Hessian **0.16% dense**, reduced KKT 8.98% dense
- matrices differ per candidate (distinct `J_n` hashes) — no shared
  factorization across candidates
- `rho` changes the matrix every ADMM iteration, but **diagonally only**
- a bare GPU kernel-launch floor at these dimensions is 74.8 µs — already
  1.6× Drake's *entire* 47.5 µs iteration

So the global solve is the bottleneck, and it is the part a GPU is *least*
suited to at this size.

## Conformance policy (Phase 2J)

See `docs/gpu/candidate_semantics.md` §7. Never compare a GPU
`independent_batch`/`check=25` result against a CPU
`reference_reset`/`check=100` result and call the difference a GPU speedup.

## Phase 2K gate

| requirement | status |
|---|---|
| `independent_batch` CPU mode is explicit | ✅ `CandidateSemantics` |
| `independent_batch` is candidate-order invariant | ✅ 4 permutations, exact |
| CPU `independent_batch` golden fixtures exist | ✅ 17.1 KB + metadata |
| GPU environment documented | ✅ above |
| batched GPU data representation exists | ✅ `C3PlusQPData` / `C3PlusProblemBatch` |
| projection primitives match CPU | ✅ exact |
| residual primitives match CPU | ✅ 1e-13 |
| candidate cost matches CPU | ✅ 1e-13 |
| candidate argmin matches CPU | ✅ |
| float64 parity characterized | ✅ |
| candidate-count crossover measured | ✅ — **and it is negative** |

Every box ticks except the one that decides the question: the measured
crossover says the GPU primitives do not pay off at B=6, and would not at
B=64 either.

> **GPU PRIMITIVES NOT READY**

Not "not working" — they are correct and parity-clean. Not *worth shipping*:
the crossover is ~25× above the port's candidate count, and the axis they
accelerate is 0.6% of runtime.

Revisit when the problem grows: SE(3) (next quarter) raises n_x, n_lambda and
contact count together, which moves both the per-candidate work and the
crossover in the favourable direction.
