"""Emit a coarse occupancy grid of the REAL P_sym / KKT sparsity patterns,
for honest use in the findings report (no invented patterns).
"""
import glob
import json

import numpy as np

from scripts.gpu.hotloop_benchmark import load

G = 48          # grid cells per side


def occupancy(M, g=G):
    n = M.shape[0]
    nz = (np.abs(M) > 1e-14)
    edges = np.linspace(0, n, g + 1).astype(int)
    out = []
    for i in range(g):
        row = []
        for j in range(g):
            blk = nz[edges[i]:edges[i + 1], edges[j]:edges[j + 1]]
            row.append(int(blk.any()))
        out.append(row)
    return out


def main():
    d = load(sorted(glob.glob("audit_output/admm_corpus/inst_*_qp.npz"))[0])
    P, C = d["P"], d["C_eq"]
    n = d["total_dim"]
    nb = d["idx"].size
    sel = np.zeros((nb, n))
    sel[np.arange(nb), d["idx"]] = 1.0
    A = np.vstack([C, sel])
    rho = np.full(A.shape[0], 0.1)
    rho[:C.shape[0]] = 100.0
    K = P + 1e-5 * np.eye(n) + A.T @ (rho[:, None] * A)
    out = {
        "n": n,
        "grid": G,
        "P": occupancy(P),
        "K": occupancy(K),
        "P_density": round(100 * np.count_nonzero(np.abs(P) > 1e-14) / P.size, 3),
        "K_density": round(100 * np.count_nonzero(np.abs(K) > 1e-14) / K.size, 3),
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
