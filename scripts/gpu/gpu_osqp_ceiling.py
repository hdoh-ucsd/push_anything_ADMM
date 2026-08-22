"""What is the per-iteration FLOOR for any GPU OSQP at our problem size?

A GPU OSQP (cuOSQP / OSQP 1.x cuda algebra) solves ONE problem at a time. Our
batched solver amortises kernel launches across 6 samples, so it is strictly
the GPU-favourable case -- if a single-problem GPU iteration cannot beat
Drake's CPU iteration, no GPU OSQP can help this port.

Measured reference points:
  Drake CPU OSQP, in-run   : 47.5 us per iteration  (section timer / iters)

This measures:
  1. our batched GPU iteration at batch=6 and batch=1
  2. the bare kernel-launch floor: the minimum GPU op sequence an OSQP
     iteration needs (2 matvecs + clamp + 2 axpy) at n=639, batch=1
"""
import time

import numpy as np
import torch

from control.gpu.legacy_torch.batched_qp import BatchedBoxQP

N, M = 639, 439
REPS = 300
DRAKE_US_PER_ITER = 47.5


def make(batch):
    rng = np.random.default_rng(0)
    Mm = rng.standard_normal((batch, N, N))
    P = Mm @ Mm.transpose(0, 2, 1) + N * np.eye(N)
    A = rng.standard_normal((batch, M, N))
    lo = np.full((batch, M), -1.0)
    hi = np.full((batch, M), 1.0)
    q = rng.standard_normal((batch, N))
    return BatchedBoxQP(P, A, lo, hi, n_eq_rows=210), q


def timed_iters(batch, iters=REPS):
    qp, q = make(batch)
    qp.solve(q, max_iter=5, eps=0.0, check_every=10 ** 9)   # warm up
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    qp.solve(q, max_iter=iters, eps=0.0, check_every=10 ** 9)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return dt * 1e6 / batch          # us per SAMPLE-iteration


def launch_floor():
    """Bare minimum GPU work an OSQP iteration needs, batch=1."""
    kw = dict(dtype=torch.float64, device="cuda")
    K = torch.randn(1, N, N, **kw)
    A = torch.randn(1, M, N, **kw)
    x = torch.randn(1, N, **kw)
    z = torch.randn(1, M, **kw)
    lo, hi = -torch.ones(1, M, **kw), torch.ones(1, M, **kw)

    def step():
        r = torch.bmm(K, x.unsqueeze(-1)).squeeze(-1)
        t = torch.bmm(A, r.unsqueeze(-1)).squeeze(-1)
        zz = torch.clamp(t + z, lo, hi)
        return zz + z

    step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPS):
        step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / REPS * 1e6


def main():
    print(f"n={N}, m={M}, fp64, RTX 5070 Ti\n")
    print(f"  {'configuration':38s} {'us / sample-iteration':>22s}")
    print(f"  {'Drake CPU OSQP (measured in-run)':38s} {DRAKE_US_PER_ITER:>22.1f}")
    for b in (6, 1):
        us = timed_iters(b)
        tag = f"our batched GPU, batch={b}"
        print(f"  {tag:38s} {us:>22.1f}   "
              f"({DRAKE_US_PER_ITER / us:.2f}x vs Drake)")
    fl = launch_floor()
    print(f"  {'bare GPU launch floor, batch=1':38s} {fl:>22.1f}   "
          f"({DRAKE_US_PER_ITER / fl:.2f}x vs Drake)")
    print("\n  The launch floor is the best ANY single-problem GPU OSQP can do\n"
          "  per iteration at this size -- it is just the kernel dispatches,\n"
          "  with no solver logic, no preconditioner and no convergence check.")


if __name__ == "__main__":
    main()
