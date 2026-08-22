"""Why is the GPU hot loop slower despite winning per-iteration?

loop_microbench.py measured 4.41x PER ITERATION. hotloop_benchmark.py then
measured 0.10x for the whole tick. The only way both are true is if the GPU
solver needs far MORE iterations. This counts them on real corpus QPs.

Drake's OsqpSolver runs with scaling=1 (Ruiz equilibration) and
adaptive_rho=1. control/gpu/legacy_torch/batched_qp.py implements the plain OSQP
iteration with neither. The corpus instances have cond(K) ~ 1.2e7, and
preconditioning is exactly what makes first-order methods tolerable on
ill-conditioned problems.
"""
import glob

import numpy as np
import pydrake.all as ad
import torch

from control.gpu.legacy_torch.batched_qp import BatchedBoxQP
from scripts.gpu.hotloop_benchmark import load


def main():
    qps = sorted(glob.glob("audit_output/admm_corpus/inst_*_qp.npz"))[:6]
    insts = [load(f) for f in qps]

    print("CPU: Drake OSQP iterations to converge (eps=1e-5)")
    solver = ad.OsqpSolver()
    so = ad.SolverOptions()
    sid = ad.OsqpSolver().solver_id()
    for k, v in (("polishing", 0), ("warm_starting", 1), ("scaling", 1),
                 ("adaptive_rho", 1), ("check_termination", 100),
                 ("max_iter", 2000)):
        so.SetOption(sid, k, v)
    for k, v in (("rho", 0.1), ("sigma", 1e-5), ("alpha", 1.6),
                 ("eps_abs", 1e-5), ("eps_rel", 1e-5)):
        so.SetOption(sid, k, v)
    cpu_iters = []
    for d in insts:
        prog = ad.MathematicalProgram()
        z = prog.NewContinuousVariables(d["total_dim"], "z")
        prog.AddLinearEqualityConstraint(d["C_eq"], d["b_eq"], z)
        prog.AddBoundingBoxConstraint(d["lo"], d["hi"], z[d["idx"]])
        prog.AddQuadraticCost(d["P"], d["q"], z, is_convex=True)
        res = solver.Solve(prog, None, so)
        it = res.get_solver_details().iter
        cpu_iters.append(it)
        print(f"    n={d['total_dim']}  iters={it:5d}  "
              f"status={res.get_solution_result()}")

    print("\nGPU: control/gpu/legacy_torch/batched_qp.py iterations (same eps)")
    P = np.stack([d["P"] for d in insts])
    q = np.stack([d["q"] for d in insts])
    n_box = insts[0]["idx"].size
    total = insts[0]["total_dim"]
    sel = np.zeros((n_box, total))
    sel[np.arange(n_box), insts[0]["idx"]] = 1.0
    A = np.stack([np.vstack([d["C_eq"], sel]) for d in insts])
    lo = np.stack([np.concatenate([d["b_eq"], d["lo"]]) for d in insts])
    hi = np.stack([np.concatenate([d["b_eq"], d["hi"]]) for d in insts])
    qp = BatchedBoxQP(P, A, lo, hi, n_eq_rows=insts[0]["C_eq"].shape[0])
    x, z, y = qp.solve(q, max_iter=20000, eps=1e-5)
    print(f"    batch of {len(insts)}: iters={qp.last_iters} "
          f"(cap 20000)")

    Ax = torch.bmm(qp.A, x.unsqueeze(-1)).squeeze(-1)
    r_p = (Ax - z).abs().amax().item()
    qt = torch.as_tensor(q, dtype=torch.float64, device=qp.dev)
    r_d = (torch.bmm(qp.P, x.unsqueeze(-1)).squeeze(-1) + qt
           + torch.bmm(qp.At, y.unsqueeze(-1)).squeeze(-1)).abs().amax().item()
    print(f"    final residuals: primal {r_p:.3e}  dual {r_d:.3e}")
    print(f"\n  CPU median iters {int(np.median(cpu_iters))} vs GPU "
          f"{qp.last_iters} -> {qp.last_iters / max(np.median(cpu_iters),1):.0f}x "
          f"more iterations")
    print("  Missing from the GPU solver: Ruiz equilibration (scaling=1) and")
    print("  adaptive rho -- both ON in Drake. cond(K) ~ 1.2e7.")


if __name__ == "__main__":
    main()
