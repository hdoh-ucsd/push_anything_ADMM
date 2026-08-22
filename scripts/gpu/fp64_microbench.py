"""Decide whether fp64 batched dense linear algebra on THIS GPU beats this
CPU at OUR problem size. Run before writing any solver code.

Task 1 of docs/superpowers/plans/2026-08-20-gpu-admm.md. The gate:
  chol_solve speedup >= 3x  -> proceed as planned
  chol_solve speedup 1x-3x  -> proceed, mixed precision (Task 13) required
  chol_solve speedup <  1x  -> STOP and report; the thread lever is the
                               correct response, not a GPU port.

`chol_solve` is the number that decides because the inner QP does ONE
factorization plus ~100-2000 triangular solves per C3+ iteration.
"""
import time

import numpy as np
import torch

N_TOT, BATCH, REPS = 719, 6, 50   # total_dim ~719, 5-6 samples per tick


def _bench(fn, reps=REPS):
    fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3   # ms


def main():
    assert torch.cuda.is_available(), "CUDA not visible"
    print(f"torch {torch.__version__}  |  {torch.cuda.get_device_name(0)}")
    print(f"cuda {torch.version.cuda}  |  threads {torch.get_num_threads()}")

    M = np.random.default_rng(0).standard_normal((BATCH, N_TOT, N_TOT))
    K_np = M @ M.transpose(0, 2, 1) + N_TOT * np.eye(N_TOT)
    b_np = np.random.default_rng(1).standard_normal((BATCH, N_TOT, 1))

    K_g = torch.as_tensor(K_np, dtype=torch.float64, device="cuda")
    b_g = torch.as_tensor(b_np, dtype=torch.float64, device="cuda")
    K_c = torch.as_tensor(K_np, dtype=torch.float64)
    b_c = torch.as_tensor(b_np, dtype=torch.float64)

    gpu_chol = _bench(lambda: torch.linalg.cholesky(K_g))
    cpu_chol = _bench(lambda: torch.linalg.cholesky(K_c))
    L_g, L_c = torch.linalg.cholesky(K_g), torch.linalg.cholesky(K_c)
    gpu_slv = _bench(lambda: torch.cholesky_solve(b_g, L_g))
    cpu_slv = _bench(lambda: torch.cholesky_solve(b_c, L_c))

    print(f"\nbatch={BATCH} n={N_TOT} fp64")
    print(f"  cholesky  : GPU {gpu_chol:8.3f} ms | CPU {cpu_chol:8.3f} ms"
          f" | speedup {cpu_chol / gpu_chol:6.2f}x")
    print(f"  chol_solve: GPU {gpu_slv:8.3f} ms | CPU {cpu_slv:8.3f} ms"
          f" | speedup {cpu_slv / gpu_slv:6.2f}x   <-- THE GATE")

    K32, b32 = K_g.float(), b_g.float()
    f32 = _bench(lambda: torch.cholesky_solve(b32, torch.linalg.cholesky(K32)))
    print(f"  fp32 factor+solve: GPU {f32:8.3f} ms"
          f" (fp64 equivalent {gpu_chol + gpu_slv:8.3f} ms,"
          f" ratio {(gpu_chol + gpu_slv) / f32:5.2f}x)")

    ratio = cpu_slv / gpu_slv
    verdict = ("PROCEED as planned" if ratio >= 3.0
               else "PROCEED, Task 13 mixed precision REQUIRED" if ratio >= 1.0
               else "STOP - GPU slower; use the thread lever instead")
    print(f"\nGATE VERDICT (chol_solve {ratio:.2f}x): {verdict}")


if __name__ == "__main__":
    main()
