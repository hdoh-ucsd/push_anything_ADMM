"""Isolated timing of the projection case-histogram, scalar vs vectorized.

Wall-clock of a full sim CANNOT measure this: two runs of identical code
diverge (filtered_solve_time -> x-pred clamp), so trajectories differ and the
work differs. This times the histogram itself at realistic shapes.
"""
import time

import numpy as np

N_KNOTS, NUM_NORMALS = 10, 4
N_LAMBDA = 6 * NUM_NORMALS
ADMM_ITERS, SAMPLES = 3, 6
REPS = 300


def scalar(lam, eta, ratio, n_lo, n_hi, G, Nn, T):
    for j in range(lam.size):
        lo, eo = float(lam[j]), float(eta[j])
        c1 = (eo >= 0.0) and (eo >= ratio * lo)
        c2 = (lo >= 0.0) and (eo < ratio * lo)
        idx = 0 if c1 else (1 if c2 else 2)
        (G if j < n_lo else (Nn if j < n_hi else T))[idx] += 1


def vector(lam, eta, ratio, n_lo, n_hi, G, Nn, T):
    c1 = (eta >= 0.0) & (eta >= ratio * lam)
    c2 = (lam >= 0.0) & (eta < ratio * lam)
    case = np.where(c1, 0, np.where(c2, 1, 2))
    for acc, sl in ((G, slice(0, n_lo)), (Nn, slice(n_lo, n_hi)),
                    (T, slice(n_hi, None))):
        bc = np.bincount(case[sl], minlength=3)
        acc[0] += int(bc[0]); acc[1] += int(bc[1]); acc[2] += int(bc[2])


def bench(fn):
    rng = np.random.default_rng(0)
    blocks = [(rng.uniform(-5, 5, N_LAMBDA), rng.uniform(-5, 5, N_LAMBDA))
              for _ in range(N_KNOTS)]
    t0 = time.perf_counter()
    for _ in range(REPS):
        G, Nn, T = [0, 0, 0], [0, 0, 0], [0, 0, 0]
        for lam, eta in blocks:
            fn(lam, eta, 1.0, NUM_NORMALS, 2 * NUM_NORMALS, G, Nn, T)
    return (time.perf_counter() - t0) / REPS * 1e6      # us per ADMM iteration


def main():
    s, v = bench(scalar), bench(vector)
    print(f"shapes: N={N_KNOTS} n_lambda={N_LAMBDA} "
          f"(one ADMM iteration = {N_KNOTS} knot-blocks)")
    print(f"  scalar    : {s:9.1f} us / ADMM iteration")
    print(f"  vectorized: {v:9.1f} us / ADMM iteration")
    print(f"  speedup   : {s / v:9.2f}x   saving {s - v:7.1f} us/iter")
    per_tick = (s - v) * ADMM_ITERS * SAMPLES
    print(f"\n  per control tick ({ADMM_ITERS} iters x {SAMPLES} samples): "
          f"{per_tick:.0f} us saved")
    print(f"  at 13.3 Hz over a 180 s run: "
          f"{per_tick * 13.3 * 180 / 1e6:.2f} s saved")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------
# Variant C: one pass over ALL knots at once (amortizes numpy call overhead)
# ---------------------------------------------------------------------
def bench_allknots():
    rng = np.random.default_rng(0)
    LAM = rng.uniform(-5, 5, (N_KNOTS, N_LAMBDA))
    ETA = rng.uniform(-5, 5, (N_KNOTS, N_LAMBDA))
    t0 = time.perf_counter()
    for _ in range(REPS):
        c1 = (ETA >= 0.0) & (ETA >= 1.0 * LAM)
        c2 = (LAM >= 0.0) & (ETA < 1.0 * LAM)
        case = np.where(c1, 0, np.where(c2, 1, 2))
        G = np.bincount(case[:, :NUM_NORMALS].ravel(), minlength=3)
        Nn = np.bincount(case[:, NUM_NORMALS:2 * NUM_NORMALS].ravel(), minlength=3)
        T = np.bincount(case[:, 2 * NUM_NORMALS:].ravel(), minlength=3)
    return (time.perf_counter() - t0) / REPS * 1e6


if __name__ != "__main__":
    pass
