"""How much work does check_termination=100 waste?

The profile shows the port's runtime is essentially the COUNT of OSQP inner
iterations (final-QP time/in-loop time = 1.494 vs iteration ratio 1.488).

Drake runs OSQP with check_termination=100, so convergence is only tested
every 100 iterations and reported counts are quantized to 100/200/300/...
A solve that truly converges at iteration 105 still pays 200.

This re-solves the REAL corpus QPs at several check_termination values and
reports the true convergence point vs what the current setting pays. It
changes nothing in the port -- it only measures the headroom, so the
reference-conformance question can be decided on evidence.
"""
import glob

import numpy as np
import pydrake.all as ad

from scripts.gpu.hotloop_benchmark import load

CHECKS = [100, 25, 1]


def build(d):
    prog = ad.MathematicalProgram()
    z = prog.NewContinuousVariables(d["total_dim"], "z")
    prog.AddLinearEqualityConstraint(d["C_eq"], d["b_eq"], z)
    prog.AddBoundingBoxConstraint(d["lo"], d["hi"], z[d["idx"]])
    prog.AddQuadraticCost(d["P"], d["q"], z, is_convex=True)
    return prog, z


def opts(check):
    so = ad.SolverOptions()
    sid = ad.OsqpSolver().solver_id()
    for k, v in (("polishing", 0), ("polish_refine_iter", 1),
                 ("warm_starting", 1), ("scaled_termination", 1),
                 ("scaling", 1), ("adaptive_rho", 1),
                 ("adaptive_rho_interval", 0), ("adaptive_rho_tolerance", 5),
                 ("check_termination", check), ("max_iter", 2000)):
        so.SetOption(sid, k, v)
    for k, v in (("adaptive_rho_fraction", 0.4), ("rho", 0.1), ("sigma", 1e-5),
                 ("alpha", 1.6), ("delta", 1e-6),
                 ("eps_abs", 1e-5), ("eps_rel", 1e-5)):
        so.SetOption(sid, k, v)
    return so


def main():
    files = sorted(glob.glob("audit_output/admm_corpus/inst_*_qp.npz"))
    insts = [load(f) for f in files]
    solver = ad.OsqpSolver()
    print(f"{len(insts)} real corpus QPs, n={insts[0]['total_dim']}, "
          f"eps=1e-5\n")
    ref_sol = {}
    print(f"{'check_termination':>18s} {'mean iters':>11s} {'max':>6s} "
          f"{'ms/solve':>9s} {'vs check=1':>11s}")
    base = None
    for chk in CHECKS:
        so = opts(chk)
        iters, sols = [], []
        import time
        t0 = time.perf_counter()
        for d in insts:
            prog, z = build(d)
            res = solver.Solve(prog, None, so)
            iters.append(int(res.get_solver_details().iter))
            sols.append(res.GetSolution(z))
        ms = (time.perf_counter() - t0) / len(insts) * 1e3
        if chk == 1:
            base = float(np.mean(iters))
        ref_sol[chk] = sols
        print(f"{chk:>18d} {np.mean(iters):>11.1f} {max(iters):>6d} "
              f"{ms:>9.2f} "
              f"{('%.2fx' % (np.mean(iters) / base)) if base else '':>11s}")

    # Do the looser check settings change the ANSWER?
    print("\n  solution agreement vs check_termination=1 (max abs diff):")
    for chk in CHECKS:
        if chk == 1:
            continue
        d = max(np.max(np.abs(a - b))
                for a, b in zip(ref_sol[chk], ref_sol[1]))
        print(f"    check={chk:<4d} -> {d:.3e}")


if __name__ == "__main__":
    main()
