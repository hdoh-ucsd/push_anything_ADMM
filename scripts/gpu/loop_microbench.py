"""Faithful benchmark of ONE full OSQP inner iteration, batched, GPU vs CPU.

fp64_microbench.py measured torch.cholesky_solve in isolation and the GPU
lost (0.58x). That is one op; the real inner loop is

    rhs = sigma*x - q + At @ (rho*z - y)      bmm      GPU-friendly
    xt  = solve(K, rhs)                        tri-solve  GPU-HOSTILE (sequential)
    zt  = A @ xt                               bmm      GPU-friendly
    x,z,y updates                              elementwise  GPU-friendly

so the composition matters. This benchmark runs the true iteration body and
also tests a variant that replaces the triangular solve with a batched GEMV
against a PRECOMPUTED INVERSE -- same FLOPs, but parallel instead of
sequential, and legitimate here because K is refactored only once per C3+
iteration while the RHS changes every inner iteration.

Reported: ms per iteration, batched over samples, fp64.
"""
import time

import numpy as np
import torch

N, M_EQ, M_BOX, BATCH = 719, 210, 90, 6     # ~box/T canonical shapes
ITERS = 200


def _bench(fn, reps=ITERS, cuda=False):
    fn()
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    if cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


def build(device):
    kw = dict(dtype=torch.float64, device=device)
    rng = np.random.default_rng(0)
    Mm = rng.standard_normal((BATCH, N, N))
    P = Mm @ Mm.transpose(0, 2, 1) + N * np.eye(N)
    A = rng.standard_normal((BATCH, M_EQ + M_BOX, N))
    d = {}
    d["P"] = torch.as_tensor(P, **kw)
    d["A"] = torch.as_tensor(A, **kw)
    d["At"] = d["A"].transpose(1, 2).contiguous()
    m = M_EQ + M_BOX
    rho = torch.full((m,), 0.1, **kw)
    rho[:M_EQ] = 100.0
    d["rho"], d["rho_inv"] = rho, 1.0 / rho
    d["lo"] = torch.full((BATCH, m), -1.0, **kw)
    d["hi"] = torch.full((BATCH, m), 1.0, **kw)
    K = d["P"] + 1e-5 * torch.eye(N, **kw) + d["At"] @ (rho[:, None] * d["A"])
    d["L"] = torch.linalg.cholesky(K)
    d["Kinv"] = torch.linalg.inv(K)
    d["q"] = torch.zeros(BATCH, N, **kw)
    d["x"] = torch.zeros(BATCH, N, **kw)
    d["z"] = torch.zeros(BATCH, m, **kw)
    d["y"] = torch.zeros(BATCH, m, **kw)
    return d


def make_iter(d, mode):
    alpha, sigma = 1.6, 1e-5

    def step():
        x, z, y = d["x"], d["z"], d["y"]
        rhs = sigma * x - d["q"] + torch.bmm(
            d["At"], (d["rho"] * z - y).unsqueeze(-1)).squeeze(-1)
        if mode == "chol":
            xt = torch.cholesky_solve(rhs.unsqueeze(-1), d["L"]).squeeze(-1)
        else:                                   # precomputed inverse -> GEMV
            xt = torch.bmm(d["Kinv"], rhs.unsqueeze(-1)).squeeze(-1)
        zt = torch.bmm(d["A"], xt.unsqueeze(-1)).squeeze(-1)
        xn = alpha * xt + (1.0 - alpha) * x
        zn = torch.clamp(alpha * zt + (1.0 - alpha) * z + d["rho_inv"] * y,
                         d["lo"], d["hi"])
        yn = y + d["rho"] * (alpha * zt + (1.0 - alpha) * z - zn)
        d["x"], d["z"], d["y"] = xn, zn, yn
    return step


def main():
    assert torch.cuda.is_available()
    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}"
          f" | cpu threads {torch.get_num_threads()}")
    print(f"batch={BATCH} n={N} m={M_EQ + M_BOX} fp64, {ITERS} iters\n")

    rows = []
    for mode in ("chol", "inv"):
        g = _bench(make_iter(build("cuda"), mode), cuda=True)
        c = _bench(make_iter(build("cpu"), mode), cuda=False)
        rows.append((mode, g, c, c / g))
        print(f"  full iteration [{mode:4s}]: GPU {g:7.4f} ms | "
              f"CPU {c:7.4f} ms | speedup {c / g:6.2f}x")

    # Accuracy check: inverse-GEMV must agree with the triangular solve.
    d = build("cuda")
    rhs = torch.randn(BATCH, N, dtype=torch.float64, device="cuda")
    a = torch.cholesky_solve(rhs.unsqueeze(-1), d["L"]).squeeze(-1)
    b = torch.bmm(d["Kinv"], rhs.unsqueeze(-1)).squeeze(-1)
    rel = ((a - b).norm() / a.norm()).item()
    print(f"\n  inverse-vs-cholesky relative error: {rel:.3e}")

    best = max(rows, key=lambda r: r[3])
    print(f"\nBEST: {best[0]} at {best[3]:.2f}x "
          f"({'GPU wins' if best[3] > 1.0 else 'CPU wins'})")


if __name__ == "__main__":
    main()
