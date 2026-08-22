"""Measured head-to-head on REAL corpus QPs: one control tick's worth of
C3+ ADMM hot loop, CPU (Drake OSQP, sequential) vs GPU (batched).

A control tick evaluates ~6 samples, each running admm_iter=3 outer C3+
iterations, each of which builds/updates a QP and solves it. That inner
machinery is what the GPU path replaces; the Drake LCS linearization around
it stays on the CPU either way.

CPU leg reproduces what _solve_c3plus does per iteration: push (P_sym, q)
into the cost binding via UpdateCoefficients, then Solve. GPU leg runs the
batched solver over all 6 samples at once, refactoring K^-1 once per outer
iteration to mirror the rho_scale ramp.

Both legs solve the SAME matrices, taken from the corpus.
"""
import glob
import time

import numpy as np
import pydrake.all as ad
import torch

from control.gpu.legacy_torch.batched_qp import BatchedBoxQP

SAMPLES_PER_TICK = 6      # num_additional_samples_c3=5 + current
ADMM_ITERS = 3            # surrogate_admm_iters / base_mpc.admm_iter
REPS = 5


def load(qf):
    d = np.load(qf, allow_pickle=True)
    total_dim, TOT = int(d["total_dim"]), int(d["TOT"])
    SX, SU, n_lambda = int(d["SX"]), int(d["SU"]), int(d["n_lambda"])
    n_u = int(d["u_lo"].size)
    N = (total_dim - (TOT - 2 * n_lambda - n_u)) // TOT
    idx, lo, hi = [], [], []
    for i in range(N):
        for j in range(n_u):
            idx.append(i * TOT + SU + j)
            lo.append(float(d["u_lo"][j])); hi.append(float(d["u_hi"][j]))
    for row in d["spb"]:
        s, l, h = int(row[0]), float(row[1]), float(row[2])
        for i in range(N):
            idx.append(i * TOT + SX + s); lo.append(l); hi.append(h)
        idx.append(N * TOT + s); lo.append(l); hi.append(h)
    if d["ee_vel_bounds"].size == 2:
        l, h = float(d["ee_vel_bounds"][0]), float(d["ee_vel_bounds"][1])
        for s in d["ee_vel_idx"].tolist():
            for i in range(N):
                idx.append(i * TOT + SX + int(s)); lo.append(l); hi.append(h)
            idx.append(N * TOT + int(s)); lo.append(l); hi.append(h)
    return dict(P=d["P_sym"], q=d["q_ref"], C_eq=d["C_eq"], b_eq=d["b_eq"],
                idx=np.asarray(idx), lo=np.asarray(lo), hi=np.asarray(hi),
                total_dim=total_dim, g=d["g_diag"], use_g=bool(d["use_g"]))


def cpu_tick(insts):
    """Drake OSQP, sequential over samples, admm_iter outer iterations."""
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
    t0 = time.perf_counter()
    for d in insts:
        prog = ad.MathematicalProgram()
        z = prog.NewContinuousVariables(d["total_dim"], "z")
        prog.AddLinearEqualityConstraint(d["C_eq"], d["b_eq"], z)
        prog.AddBoundingBoxConstraint(d["lo"], d["hi"], z[d["idx"]])
        cost = prog.AddQuadraticCost(d["P"], np.zeros(d["total_dim"]), z,
                                     is_convex=True)
        P = d["P"].copy()
        rho = 1.0
        for _ in range(ADMM_ITERS):
            cost.evaluator().UpdateCoefficients(P, d["q"], 0.0, True)
            solver.Solve(prog, None, so)
            drho = rho * 2.0                      # rho_scale = 3 ramp
            rho *= 3.0
            aug = 2.0 * drho * (d["g"] if d["use_g"]
                                else np.ones(d["total_dim"]))
            np.fill_diagonal(P, P.diagonal() + aug)
    return time.perf_counter() - t0


def gpu_tick(insts):
    """Batched over samples, all iterates on-device."""
    P = np.stack([d["P"] for d in insts])
    q = np.stack([d["q"] for d in insts])
    n_box = insts[0]["idx"].size
    total = insts[0]["total_dim"]
    sel = np.zeros((n_box, total))
    sel[np.arange(n_box), insts[0]["idx"]] = 1.0
    A = np.stack([np.vstack([d["C_eq"], sel]) for d in insts])
    lo = np.stack([np.concatenate([d["b_eq"], d["lo"]]) for d in insts])
    hi = np.stack([np.concatenate([d["b_eq"], d["hi"]]) for d in insts])
    g = torch.as_tensor(np.stack([d["g"] for d in insts]),
                        dtype=torch.float64, device="cuda")
    use_g = insts[0]["use_g"]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    qp = BatchedBoxQP(P, A, lo, hi, n_eq_rows=insts[0]["C_eq"].shape[0])
    x = z = y = None
    rho = 1.0
    for _ in range(ADMM_ITERS):
        x, z, y = qp.solve(q, max_iter=2000, eps=1e-5, x0=x, z0=z, y0=y)
        drho = rho * 2.0
        rho *= 3.0
        qp.refactor_diag(2.0 * drho * g if use_g
                         else drho * torch.ones_like(g))
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main():
    qps = sorted(glob.glob("audit_output/admm_corpus/inst_*_qp.npz"))
    assert len(qps) >= SAMPLES_PER_TICK, "run dump_admm_corpus.sh first"
    insts = [load(f) for f in qps[:SAMPLES_PER_TICK]]
    print(f"one control tick: {SAMPLES_PER_TICK} samples x {ADMM_ITERS} "
          f"C3+ iters, n={insts[0]['total_dim']}, "
          f"m={insts[0]['C_eq'].shape[0] + insts[0]['idx'].size}")
    cpu_tick(insts); gpu_tick(insts)                     # warm up
    c = min(cpu_tick(insts) for _ in range(REPS)) * 1e3
    g = min(gpu_tick(insts) for _ in range(REPS)) * 1e3
    print(f"  CPU (Drake OSQP, sequential) : {c:8.1f} ms / tick")
    print(f"  GPU (batched, on-device)     : {g:8.1f} ms / tick")
    print(f"  speedup on the replaced part : {c / g:8.2f}x")
    print(f"\n  measured tick budget = 472.0 ms/step (threads=1, 60 s gate)")
    saved = c - g
    print(f"  saving {saved:.1f} ms/tick -> projected "
          f"{472.0 - saved:.1f} ms/step = {472.0 / max(472.0 - saved, 1e-9):.2f}x end-to-end")


if __name__ == "__main__":
    main()
