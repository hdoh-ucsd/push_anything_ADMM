"""Does Drake's OSQP solve release the GIL?

The parallel sample-eval path is a ThreadPoolExecutor, and Python threads only
parallelize work that releases the GIL. After the correctness fix, threads=4
gave ~3% wall improvement with user~=real -- the signature of GIL-bound
execution. This measures it directly.

Run with BLAS pinned to one thread, otherwise the numpy control is invalid
(multi-threaded BLAS already saturates cores in the "serial" leg):

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      PYTHONPATH=. python3 scripts/gpu/gil_probe.py

Controls:
  sleep       -- unambiguously releases the GIL; must show ~Nx
  blas 1thr   -- releases the GIL inside BLAS; should show ~Nx
  drake solve -- the thing we actually care about
"""
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pydrake.all as ad

NT = 4


def bench(fn, n):
    fn(0)
    t0 = time.perf_counter()
    for i in range(n):
        fn(i)
    serial = time.perf_counter() - t0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=NT) as pool:
        list(pool.map(fn, range(n)))
    return serial, time.perf_counter() - t0


def main():
    sa, ta = bench(lambda i: time.sleep(0.02), 16)
    print(f"control sleep      : serial {sa:.3f}s threaded {ta:.3f}s "
          f"-> {sa / ta:5.2f}x")

    A = np.random.default_rng(0).standard_normal((700, 700))
    sb, tb = bench(lambda i: float((A @ A).sum()), 16)
    print(f"control blas       : serial {sb:.3f}s threaded {tb:.3f}s "
          f"-> {sb / tb:5.2f}x")

    N, K = 400, 32
    progs = []
    for k in range(K):
        rng = np.random.default_rng(k)
        M = rng.standard_normal((N, N))
        pr = ad.MathematicalProgram()
        z = pr.NewContinuousVariables(N, "z")
        pr.AddQuadraticCost(M @ M.T + N * np.eye(N),
                            rng.standard_normal(N), z, is_convex=True)
        pr.AddBoundingBoxConstraint(-np.ones(N), np.ones(N), z)
        progs.append(pr)
    solvers = [ad.OsqpSolver() for _ in range(NT)]
    sc, tc = bench(lambda i: solvers[i % NT].Solve(progs[i % K]).is_success(), K)
    print(f"drake OsqpSolver   : serial {sc:.3f}s threaded {tc:.3f}s "
          f"-> {sc / tc:5.2f}x")


if __name__ == "__main__":
    main()
