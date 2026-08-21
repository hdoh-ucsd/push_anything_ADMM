"""Roll the measured section timings up into a per-tick time budget.

Inputs are the wall-clock section totals from a PORT_SECTION_TIMING=1 run
(25 s box gate, seed 0, HEAD) plus the OSQP iteration counts from
DIAG_OSQP_ITERS=1. Nothing here is estimated except where marked.
"""

# ms per _solve_c3plus, from the Section Timer Report (2169 solves / 25 s)
PER_SOLVE = {
    "inloop_osqp": 39.16,   # 3 calls x 13.05 ms  (275 OSQP iters each)
    "final_qp":    19.50,   # 1 call             (409 OSQP iters)
    "qp_build":     8.19,   # 4 calls x 2.05 ms  (P/q assembly + push)
    "lcs":          2.71,   # extract_dynamics + geometry_query + jacobians
    "projection":   0.41,   # 3 calls x 0.137 ms (Bui eq.12 delta-update)
}
SOLVES_PER_TICK = 6.5       # 2169 solves / 332 ticks
TICK_MS = 472.0             # measured avg_per_step_ms, threads=1 60 s gate
ITERS_INLOOP, ITERS_FINAL = 275.0, 409.3
CONVERGE_AT, PAID_AT = 68.5, 246.7   # check_termination_probe.py


def main():
    tot = sum(PER_SOLVE.values())
    tick = tot * SOLVES_PER_TICK
    print(f"per _solve_c3plus : {tot:5.1f} ms")
    print(f"per control tick  : {tick:5.0f} ms "
          f"({100 * tick / TICK_MS:.0f}% of the measured {TICK_MS:.0f} ms)\n")
    print(f"  {'section':14s} {'ms/solve':>9s} {'% solve':>8s} {'% tick':>8s}")
    for k, v in sorted(PER_SOLVE.items(), key=lambda x: -x[1]):
        print(f"  {k:14s} {v:9.2f} {100 * v / tot:7.1f}% "
              f"{100 * v * SOLVES_PER_TICK / TICK_MS:7.1f}%")

    iters = 3 * ITERS_INLOOP + ITERS_FINAL
    osqp_ms = PER_SOLVE["inloop_osqp"] + PER_SOLVE["final_qp"]
    print(f"\n  OSQP iterations per solve : {iters:.0f}")
    print(f"  OSQP iterations per tick  : {iters * SOLVES_PER_TICK:.0f}")
    print(f"  cost per OSQP iteration   : {osqp_ms / iters * 1000:.1f} us")
    print(f"  share of tick that is OSQP iterations: "
          f"{100 * osqp_ms * SOLVES_PER_TICK / TICK_MS:.0f}%")

    print(f"\n  converges at {CONVERGE_AT:.1f} iters (check_termination=1)")
    print(f"  actually pays {PAID_AT:.1f} iters (check_termination=100)")
    print(f"  => {100 * (1 - CONVERGE_AT / PAID_AT):.0f}% of every OSQP "
          f"iteration runs PAST convergence")


if __name__ == "__main__":
    main()
