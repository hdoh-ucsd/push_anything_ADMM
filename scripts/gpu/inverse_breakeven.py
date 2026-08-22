"""Break-even analysis for the precomputed-inverse design.

loop_microbench.py showed the GPU loses with cholesky_solve (0.59x) and wins
with a precomputed inverse + GEMV (4.41x). The inverse costs MORE to build
than a Cholesky factor, and it is rebuilt once per C3+ ADMM iteration (the
rho_scale ramp changes K's diagonal every iteration -- correction C2).

So the design only pays if the inner OSQP loop runs enough iterations to
amortize the extra setup. This computes that break-even count.
"""
import time

import numpy as np
import torch

N, BATCH, REPS = 719, 6, 30
SAVED_PER_ITER_MS = 1.2651 - 0.2508     # from loop_microbench.py


def bench(fn, reps=REPS):
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


def main():
    rng = np.random.default_rng(0)
    Mm = rng.standard_normal((BATCH, N, N))
    K = torch.as_tensor(Mm @ Mm.transpose(0, 2, 1) + N * np.eye(N),
                        dtype=torch.float64, device="cuda")

    chol = bench(lambda: torch.linalg.cholesky(K))
    inv = bench(lambda: torch.linalg.inv(K))
    chinv = bench(lambda: torch.cholesky_inverse(torch.linalg.cholesky(K)))

    print(f"batch={BATCH} n={N} fp64, GPU setup cost per C3+ iteration")
    print(f"  batched cholesky        : {chol:7.3f} ms  (baseline, needed anyway)")
    print(f"  batched linalg.inv      : {inv:7.3f} ms")
    print(f"  cholesky + chol_inverse : {chinv:7.3f} ms")
    print(f"\n  saving per inner iter   : {SAVED_PER_ITER_MS:.4f} ms"
          f"  (CPU chol-loop vs GPU inv-loop)")
    print()
    for setup, name in ((chinv, "chol_inverse"), (inv, "linalg.inv")):
        extra = max(setup - chol, 0.0)
        print(f"  {name:12s}: extra setup {extra:6.2f} ms "
              f"-> break-even at {extra / SAVED_PER_ITER_MS:6.1f} inner iters")

    # Conditioning sanity: how much accuracy does the explicit inverse cost?
    rhs = torch.randn(BATCH, N, dtype=torch.float64, device="cuda")
    L = torch.linalg.cholesky(K)
    a = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
    b = torch.bmm(torch.cholesky_inverse(L), rhs.unsqueeze(-1)).squeeze(-1)
    print(f"\n  cond(K) ~ {torch.linalg.cond(K).max().item():.3e}")
    print(f"  chol_inverse vs cholesky_solve rel err: "
          f"{((a - b).norm() / a.norm()).item():.3e}")


if __name__ == "__main__":
    main()
