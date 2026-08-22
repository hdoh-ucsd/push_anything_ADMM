"""QUARANTINED: PyTorch-based GPU research code from the feasibility arc.

**Nothing in the active solver path imports anything here, and nothing
should.** These modules exist only so the published feasibility measurements
stay reproducible.

Why they are quarantined rather than deleted
--------------------------------------------
They produced the numbers behind the decision to *stop* the GPU port:

  * `batched_qp.py` -- batched dense OSQP iteration with the explicit-inverse
    GEMV (amendment A1). Used by `scripts/gpu/hotloop_benchmark.py`,
    `gpu_osqp_ceiling.py`, `iteration_count_probe.py`, `crossover_sweep.py`,
    which measured the hot loop at 0.10x, the 74.8 us kernel-launch floor,
    and the (n, density) crossover map.
  * `projection.py` -- torch transcription of the Bui eq.(12) projection.
    Superseded by `control/gpu/cupy_primitives.py`.

Deleting them would make those findings unreproducible, which is a real cost
for a negative result that future work will want to re-check.

Framework direction
-------------------
**CuPy is the current GPU direction** (`control/gpu/cupy_primitives.py`).
Two CUDA frameworks in one process is not a state to keep: each carries its
own context and VRAM reservation. Torch remains installed only because these
legacy benchmarks need it.

Classification of torch usage in this repository, audited 2026-08-21:

    active required          none
    research legacy          everything here + scripts/gpu/*.py benchmarks
    unrelated ML dependency  none -- no ML component in this repo uses torch
    removable                yes, once the legacy benchmarks are retired

If the GPU direction is settled and the feasibility numbers are considered
closed, this subpackage, the torch benchmark scripts, and the `torch`
dependency itself can all be dropped in one change.
"""
