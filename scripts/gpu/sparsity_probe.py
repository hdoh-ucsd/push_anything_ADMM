"""How sparse is the C3+ QP? This decides whether a DENSE batched GPU solve
can ever beat Drake's SPARSE LDL' on the CPU.

The GPU path solves dense n x n systems. Drake's OSQP factors the KKT
sparsely. If the KKT is only a few percent dense, the CPU is doing far fewer
flops per iteration than the GPU is, and the GPU's raw throughput advantage
(already only ~1/64 rate for fp64 on GeForce Blackwell) has to overcome that
gap before it wins anything.
"""
import glob

import numpy as np

from scripts.gpu.hotloop_benchmark import load


def main():
    f = sorted(glob.glob("audit_output/admm_corpus/inst_*_qp.npz"))[0]
    d = load(f)
    P, C = d["P"], d["C_eq"]
    n = d["total_dim"]
    nb = d["idx"].size
    sel = np.zeros((nb, n))
    sel[np.arange(nb), d["idx"]] = 1.0
    A = np.vstack([C, sel])
    rho = np.full(A.shape[0], 0.1)
    rho[:C.shape[0]] = 100.0
    K = P + 1e-5 * np.eye(n) + A.T @ (rho[:, None] * A)

    for name, M in (("P_sym", P), ("A (C_eq+box)", A), ("reduced KKT", K)):
        nz = np.count_nonzero(np.abs(M) > 1e-14)
        print(f"  {name:14s} shape={str(M.shape):12s} nnz={nz:8d} "
              f"density={100 * nz / M.size:6.2f}%")

    dense_flops = n ** 3 / 3
    nzK = np.count_nonzero(np.abs(K) > 1e-14)
    print(f"\n  dense Cholesky   ~ n^3/3 = {dense_flops / 1e6:8.1f} MFLOP")
    print(f"  KKT density {100 * nzK / K.size:.2f}%")
    print(f"  dense per-iteration matvec ~ 2*n^2 = {2 * n * n / 1e6:.2f} MFLOP;"
          f" sparse ~ 2*nnz = {2 * nzK / 1e6:.2f} MFLOP"
          f"  ({K.size / max(nzK, 1):.1f}x fewer)")


if __name__ == "__main__":
    main()
