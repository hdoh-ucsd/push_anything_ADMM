"""Phase 2G/2I: candidate-count sweep for the batched GPU primitives.

Measures the projection + dual-update + residual chain -- the part of the
C3+ ADMM iteration that is genuinely parallel over
candidate x knot x contact -- across candidate counts, and reports the
CROSSOVER where GPU beats CPU.

Phase 2I rule: means are not enough. Every timing reports
mean / median / p90 / p95 / p99 / max, because the REFERENCE_RESET
experiment showed a change that barely moved a mean while blowing the
maximum up 7-12x.

Timing method:
  * GPU kernel time via CUDA events (device-side, excludes host overhead)
  * end-to-end via perf_counter around H2D + kernels + D2H, which is what a
    real caller pays
  * transfer measured separately so the two can be attributed
"""
import argparse
import time

import numpy as np

import cupy as cp

from control.gpu.cupy_primitives import (dual_update, project_C3Plus_batch,
                                         residuals)

N, TOT, SL, SE, N_LAMBDA, N_X = 10, 62, 19, 42, 20, 19
TOTAL = N * TOT + N_X
RHO = 3.0


def pct(a):
    a = np.asarray(a, dtype=float)
    return dict(mean=a.mean(), median=np.median(a), p90=np.percentile(a, 90),
                p95=np.percentile(a, 95), p99=np.percentile(a, 99),
                max=a.max())


def fmt(tag, d):
    return (f"    {tag:22s} mean {d['mean']:8.3f}  med {d['median']:8.3f}  "
            f"p90 {d['p90']:8.3f}  p95 {d['p95']:8.3f}  "
            f"p99 {d['p99']:8.3f}  max {d['max']:8.3f}")


def cpu_chain(lam, eta, om, z, dl, dp):
    """The same arithmetic on the host, as the CPU actually does it."""
    sqrt_ratio = 1.0
    c1 = (eta >= 0.0) & (eta >= sqrt_ratio * lam)
    c2 = (lam >= 0.0) & (eta < sqrt_ratio * lam)
    d_l = np.where(c2, lam, 0.0)
    d_e = np.where(c1, eta, 0.0)
    om2 = om + (z - dl)

    def vec(v):
        k = v[..., :N * TOT].reshape(v.shape[0], N, TOT)
        return np.concatenate([k[..., SL:SL + N_LAMBDA],
                               k[..., SE:SE + N_LAMBDA]],
                              axis=-1).reshape(v.shape[0], -1)
    pr = np.linalg.norm(vec(z) - vec(dl), axis=-1)
    dr = RHO * np.linalg.norm(vec(dl) - vec(dp), axis=-1)
    return d_l, d_e, om2, pr, dr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-counts", type=int, nargs="+",
                    default=[1, 4, 8, 16, 32, 64])
    ap.add_argument("--reps", type=int, default=200)
    args = ap.parse_args()

    print(f"batched primitive chain: projection + dual update + residuals")
    print(f"shapes per candidate: N={N} n_lambda={N_LAMBDA} total_dim={TOTAL}"
          f"   fp64   reps={args.reps}")
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}\n")

    print(f"{'B':>4s} {'CPU ms (p50)':>13s} {'GPU e2e (p50)':>14s} "
          f"{'GPU kernel(p50)':>16s} {'H2D+D2H':>9s} {'speedup':>8s}")
    print("-" * 72)
    rows = []
    for B in args.candidate_counts:
        rng = np.random.default_rng(0)
        lam = rng.uniform(-5, 5, (B, N, N_LAMBDA))
        eta = rng.uniform(-5, 5, (B, N, N_LAMBDA))
        om, z, dl, dp = (rng.standard_normal((B, TOTAL)) for _ in range(4))

        # ---- CPU ----
        cpu = []
        cpu_chain(lam, eta, om, z, dl, dp)
        for _ in range(args.reps):
            t0 = time.perf_counter()
            cpu_chain(lam, eta, om, z, dl, dp)
            cpu.append((time.perf_counter() - t0) * 1e3)

        # ---- GPU end-to-end (transfer included -- what a caller pays) ----
        e2e, kern, xfer = [], [], []
        ev0, ev1 = cp.cuda.Event(), cp.cuda.Event()
        for _ in range(args.reps):
            t0 = time.perf_counter()
            g_lam = cp.asarray(lam); g_eta = cp.asarray(eta)
            g_om = cp.asarray(om); g_z = cp.asarray(z)
            g_dl = cp.asarray(dl); g_dp = cp.asarray(dp)
            cp.cuda.runtime.deviceSynchronize()
            t_xfer = time.perf_counter()

            ev0.record()
            d_l, d_e = project_C3Plus_batch(g_lam, g_eta, 1.0, 1.0)
            om2 = dual_update(g_om, g_z, g_dl)
            pr, dr = residuals(g_z, g_dl, g_dp, RHO, N, TOT, SL, SE, N_LAMBDA)
            ev1.record()
            ev1.synchronize()
            k_ms = cp.cuda.get_elapsed_time(ev0, ev1)

            out = (cp.asnumpy(d_l), cp.asnumpy(d_e), cp.asnumpy(om2),
                   cp.asnumpy(pr), cp.asnumpy(dr))
            cp.cuda.runtime.deviceSynchronize()
            t1 = time.perf_counter()

            e2e.append((t1 - t0) * 1e3)
            kern.append(k_ms)
            xfer.append((t_xfer - t0) * 1e3)

        c, g, k, x = pct(cpu), pct(e2e), pct(kern), pct(xfer)
        rows.append((B, c, g, k, x))
        print(f"{B:>4d} {c['median']:>13.3f} {g['median']:>14.3f} "
              f"{k['median']:>16.4f} {x['median']:>9.3f} "
              f"{c['median']/g['median']:>7.2f}x")

    print("\nfull distributions (ms):")
    for B, c, g, k, x in rows:
        print(f"  B={B}")
        print(fmt("CPU total", c))
        print(fmt("GPU end-to-end", g))
        print(fmt("GPU kernel only", k))
        print(fmt("H2D transfer", x))

    print("\ncrossover:")
    win = [B for B, c, g, _, _ in rows if c["median"] > g["median"]]
    if win:
        print(f"  GPU end-to-end wins from B = {min(win)}")
    else:
        print("  GPU end-to-end NEVER wins over the swept range")
    kwin = [B for B, c, _, k, _ in rows if c["median"] > k["median"]]
    print(f"  GPU kernel-only would win from B = "
          f"{min(kwin) if kwin else 'never'}"
          f"   (kernel-only ignores transfer -- only reachable if the data "
          f"already lives on-device)")


if __name__ == "__main__":
    main()
