"""Where is the CPU/GPU crossover for an ADMM inner iteration?

Our QP is (small, sparse): n=639, Hessian 0.16% dense. GPUs win on (large,
dense). This maps that plane so "when would a GPU be right?" has a number
instead of an opinion.

For each (n, density) cell:
  CPU leg -- build the QP in Drake, solve with the port's exact OSQP options,
             divide wall time by the iteration count OSQP reports
             => microseconds per ADMM iteration
  GPU leg -- same matrices through control/gpu/batched_qp.py at batch=6, with
             the convergence check disabled so it is pure per-iteration cost
             => microseconds per SAMPLE-iteration

Both legs report one ADMM iteration for one sample, the unit the port's time
budget is denominated in (47.5 us on CPU at our real size).
"""
import time

import numpy as np
import pydrake.all as ad
import torch

from control.gpu.batched_qp import BatchedBoxQP

BATCH = 6
GPU_ITERS = 120
CPU_REPS = 3
SIZES = [639, 1278, 2556]
DENSITIES = [0.0016, 0.05, 0.5, 1.0]
REAL = (639, 0.0016)


def make_qp(n, density, seed=0):
    """P at the requested density; A = banded equality rows + box selectors."""
    rng = np.random.default_rng(seed)
    if density >= 0.999:
        M = rng.standard_normal((n, n))
        P = M @ M.T + n * np.eye(n)
    else:
        nnz = max(int(density * n * n), n)
        P = np.diag(rng.uniform(1.0, 4.0, n))
        extra = (nnz - n) // 2
        if extra > 0:
            i = rng.integers(0, n, extra)
            j = rng.integers(0, n, extra)
            v = rng.standard_normal(extra) * 0.05
            P[i, j] += v
            P[j, i] += v
        P += n * np.eye(n)                   # keep it SPD
    m_eq = max(n // 3, 4)
    m_box = max(n // 7, 4)
    A_eq = np.zeros((m_eq, n))
    for r in range(m_eq):                    # banded, like dynamics rows
        c = (r * 3) % max(n - 5, 1)
        A_eq[r, c:c + 5] = rng.standard_normal(min(5, n - c))
    idx = rng.choice(n, m_box, replace=False)
    A = np.vstack([A_eq, np.eye(n)[idx]])
    b_eq = rng.standard_normal(m_eq) * 0.01
    lo = np.concatenate([b_eq, -np.ones(m_box)])
    hi = np.concatenate([b_eq, np.ones(m_box)])
    return P, rng.standard_normal(n) * 0.1, A, lo, hi, m_eq


def _opts():
    so = ad.SolverOptions()
    sid = ad.OsqpSolver().solver_id()
    for k, v in (("polishing", 0), ("warm_starting", 1), ("scaling", 1),
                 ("adaptive_rho", 1), ("check_termination", 100),
                 ("max_iter", 2000)):
        so.SetOption(sid, k, v)
    for k, v in (("rho", 0.1), ("sigma", 1e-5), ("alpha", 1.6),
                 ("eps_abs", 1e-5), ("eps_rel", 1e-5)):
        so.SetOption(sid, k, v)
    return so


def cpu_leg(P, q, A, lo, hi, m_eq):
    so, solver = _opts(), ad.OsqpSolver()
    box_cols = np.argmax(A[m_eq:], axis=1)
    best, iters = float("inf"), 0
    for _ in range(CPU_REPS):
        prog = ad.MathematicalProgram()
        z = prog.NewContinuousVariables(len(q), "z")
        prog.AddLinearEqualityConstraint(A[:m_eq], lo[:m_eq], z)
        prog.AddBoundingBoxConstraint(lo[m_eq:], hi[m_eq:], z[box_cols])
        prog.AddQuadraticCost(P, q, z, is_convex=True)
        t0 = time.perf_counter()
        res = solver.Solve(prog, None, so)
        dt = time.perf_counter() - t0
        it = max(int(res.get_solver_details().iter), 1)
        best = min(best, dt / it)
        iters = it
    return best * 1e6, iters


def gpu_leg(P, q, A, lo, hi, m_eq):
    rep = lambda a: np.repeat(a[None], BATCH, 0)      # noqa: E731
    qp = BatchedBoxQP(rep(P), rep(A), rep(lo), rep(hi), n_eq_rows=m_eq)
    qb = rep(q)
    qp.solve(qb, max_iter=5, eps=0.0, check_every=10 ** 9)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    qp.solve(qb, max_iter=GPU_ITERS, eps=0.0, check_every=10 ** 9)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / GPU_ITERS * 1e6 / BATCH


def main():
    print(f"us per ADMM iteration per sample; GPU batch={BATCH}, fp64")
    print("ratio > 1.00x means the GPU iteration is CHEAPER than Drake's\n")
    print("  " + " " * 6 + "".join(f"{d * 100:>9.2f}%" for d in DENSITIES))
    print("  " + "-" * (6 + 10 * len(DENSITIES)))
    results = {}
    for n in SIZES:
        row = []
        for d in DENSITIES:
            P, q, A, lo, hi, m_eq = make_qp(n, d)
            c, it = cpu_leg(P, q, A, lo, hi, m_eq)
            g = gpu_leg(P, q, A, lo, hi, m_eq)
            results[(n, d)] = (c, g, it)
            row.append(c / g)
        print(f"  n={n:<4d}" + "".join(f"{r:>9.2f}x" for r in row))

    print(f"\n  {'n':>6s} {'density':>9s} {'CPU us':>9s} {'GPU us':>9s} "
          f"{'ratio':>8s} {'cpu iters':>10s}")
    for (n, d), (c, g, it) in sorted(results.items()):
        star = "  <-- our problem" if (n, round(d, 4)) == REAL else ""
        print(f"  {n:>6d} {d * 100:>8.2f}% {c:>9.2f} {g:>9.2f} "
              f"{c / g:>7.2f}x {it:>10d}{star}")


if __name__ == "__main__":
    main()
