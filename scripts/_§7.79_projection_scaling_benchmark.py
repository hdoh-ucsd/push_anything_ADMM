"""§7.79 — Parametrized projection benchmark over contact count (n_λ).

Extends §7.78 in two ways:

  1. Multi-contact — accepts multiple LCS dumps (one per contact count) and
     runs the same four-method timing on each. Reports a scaling table showing
     the MIQP/eq-12 gap widen with n_λ.

  2. Feasibility-verified instances — replaces §7.78's synthetic q_lcp
     (which failed 17/30 instances at n_λ=6 due to LCP infeasibility) with:
       (a) real LCS from a dumped solve (F is authentic, not random);
       (b) q_lcp chosen so the LCP has a verified Lemke solution;
       (c) z_λ = Lemke's δ_λ + noise — near-feasible target, mirroring the
           ADMM's z-around-δ pattern (real ADMM behavior, not adversarial).

Methods timed (unchanged from §7.78):
  1. C3-MIQP (Aydinoglu Big-M)          — Drake MixedIntegerBranchAndBound + OSQP
  2. C3+ eq-12 (Bui componentwise)      — port's C3Solver._project_componentwise
  3. LCP-Lemke (Aydinoglu §V-B.3.b)     — port's control.lcp_solver.solve_lcp
  4. Lorentz-cone (Aydinoglu closed)    — port's C3Solver._lorentz_project

MIQP runaway prevention: `max_explored_nodes` capped (env `MAX_MIQP_NODES`,
default 100_000). Instances that hit the cap are counted separately.

Bui's headline: gap widens from ~3 orders (1 contact, n_λ=6) toward ~4-5
orders (3+ contacts, n_λ ≥ 18). This script prepares the tooling to
measure that curve; the actual multi-contact sweep is a later 'go'.

Usage:
    # Self-test: single dump
    python scripts/_§7.79_projection_scaling_benchmark.py \\
        --dump stage_c/admm_dump/seed0_full50.npz

    # Full sweep: directory of dumps (one per n_λ)
    python scripts/_§7.79_projection_scaling_benchmark.py \\
        --dumps §7.79_instances

Env overrides:
    BENCH_N_INSTANCES  (default 30)   — instances per dump
    BENCH_N_REPS       (default 100)  — reps per instance per method
    BENCH_SEED         (default 0)    — RNG seed
    MAX_MIQP_NODES     (default 100000) — B&B node cap per MIQP solve
    BENCH_OUT_CSV      (default off)  — path to write per-dump CSV
    Q_NOISE_SCALE      (default 0.1)  — noise scale for q_lcp perturbation
    Z_NOISE_SCALE      (default 2.0)  — noise scale for z_λ around Lemke soln
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control.admm_solver import C3Solver         # noqa: E402
from control.lcp_solver   import solve_lcp       # noqa: E402

from pydrake.solvers import (                    # noqa: E402
    MathematicalProgram,
    MixedIntegerBranchAndBound,
    OsqpSolver,
)


N_INSTANCES     = int(os.environ.get("BENCH_N_INSTANCES", "30"))
N_REPS          = int(os.environ.get("BENCH_N_REPS",       "100"))
RNG_SEED        = int(os.environ.get("BENCH_SEED",         "0"))
MAX_MIQP_NODES  = int(os.environ.get("MAX_MIQP_NODES",     "100000"))
Q_NOISE_SCALE   = float(os.environ.get("Q_NOISE_SCALE",    "0.1"))
Z_NOISE_SCALE   = float(os.environ.get("Z_NOISE_SCALE",    "2.0"))

BIG_M = 100.0


# ---------------------------------------------------------------------------
# Method 1: MIQP with node cap
# ---------------------------------------------------------------------------
def project_miqp(z_lam: np.ndarray,
                 F: np.ndarray,
                 q_lcp: np.ndarray,
                 big_m: float = BIG_M,
                 max_nodes: int = MAX_MIQP_NODES
                 ) -> tuple[np.ndarray, bool]:
    """
    Return (δ, hit_cap). hit_cap=True if the B&B stopped early on the node cap.

    Objective  min ||δ - z||²  (unweighted; matches §7.78 baseline).
    Constraints (per row i):
        δ[i] ≥ 0
        (F·δ + q)[i] ≥ 0
        δ[i] ≤ M · b[i]
        (F·δ + q)[i] ≤ M · (1 - b[i])
        b[i] ∈ {0, 1}
    """
    n = z_lam.shape[0]
    prog  = MathematicalProgram()
    delta = prog.NewContinuousVariables(n, "delta")
    b     = prog.NewBinaryVariables(n, "b")

    Q_obj = 2.0 * np.eye(n)
    prog.AddQuadraticCost(Q_obj, -2.0 * z_lam, delta)

    for i in range(n):
        prog.AddLinearConstraint(delta[i] >= 0.0)
        row = F[i, :]
        prog.AddLinearConstraint(row @ delta + q_lcp[i] >= 0.0)

    for i in range(n):
        prog.AddLinearConstraint(delta[i] <= big_m * b[i])
        row = F[i, :]
        prog.AddLinearConstraint(row @ delta + q_lcp[i] <= big_m * (1.0 - b[i]))

    options = MixedIntegerBranchAndBound.Options()
    options.max_explored_nodes = int(max_nodes)
    bnb = MixedIntegerBranchAndBound(prog, OsqpSolver.id(), options)
    _ = bnb.Solve()

    d = np.asarray(bnb.GetSolution(delta), dtype=float)
    # Drake doesn't expose an explicit "hit cap" flag; approximate via
    # GetOptimalCost NaN check (no integral solution found).
    try:
        _ = float(bnb.GetOptimalCost())
        hit_cap = False
    except Exception:
        hit_cap = True
    return d, hit_cap


# ---------------------------------------------------------------------------
# Method 2: componentwise eq-12
# ---------------------------------------------------------------------------
def project_componentwise(z_lam: np.ndarray,
                          eta_seed: np.ndarray,
                          u_lam: float = 1.0,
                          u_eta: float = 1.0) -> np.ndarray:
    delta_lam, _ = C3Solver._project_componentwise(z_lam, eta_seed, u_lam, u_eta)
    return delta_lam


# ---------------------------------------------------------------------------
# Method 3: LCP / Lemke
# ---------------------------------------------------------------------------
def project_lcp_lemke(z_lam: np.ndarray,
                      F: np.ndarray,
                      q_lcp: np.ndarray) -> np.ndarray:
    delta_lam, _ = solve_lcp(F, q_lcp)
    return delta_lam


# ---------------------------------------------------------------------------
# Method 4: Lorentz-cone (per-contact friction cone)
# ---------------------------------------------------------------------------
def project_lorentz(z_lam: np.ndarray,
                    num_normals: int,
                    mu: float) -> np.ndarray:
    """
    Layout expected: [γ*num_normals, λ_n*num_normals, λ_t*(4*num_normals)].
    Drops the γ block (first num_normals slots) and passes the rest.
    """
    lam_no_gamma = z_lam[num_normals:]
    return C3Solver._lorentz_project(lam_no_gamma, num_normals, mu)


# ---------------------------------------------------------------------------
# Feasibility-verified instance generator
# ---------------------------------------------------------------------------
def generate_legacy_synthetic_instances(F: np.ndarray,
                                        c_lcs: np.ndarray,
                                        n_instances: int,
                                        rng: np.random.Generator
                                        ) -> tuple[list, int]:
    """
    §7.78's original synthetic recipe — kept for the self-test that must
    reproduce §7.78's exact 1543× / 3.19 orders headline.

    z_λ ~ N(0, 5); q_lcp = c_lcs + N(0, 0.5); η_seed ~ N(0, 5).

    No feasibility pre-check → matches §7.78's raw distribution (17/30
    infeasible warm-up failures included).
    """
    n_lambda = F.shape[0]
    instances = []
    for _ in range(n_instances):
        z_lam = rng.normal(0.0, 5.0, size=n_lambda)
        q     = c_lcs + rng.normal(0.0, 0.5, size=n_lambda)
        eta_s = rng.normal(0.0, 5.0, size=n_lambda)
        instances.append((z_lam, q, eta_s))
    return instances, 0


def generate_feasible_instances(F: np.ndarray,
                                c_lcs: np.ndarray,
                                n_instances: int,
                                rng: np.random.Generator,
                                q_noise: float = Q_NOISE_SCALE,
                                z_noise: float = Z_NOISE_SCALE,
                                max_resample: int = 20
                                ) -> tuple[list, int]:
    """
    Return (instances, n_resamples_total). Each instance = (z_λ, q_lcp, η_seed).

    Instance recipe (paper-grade, replaces §7.78's synthetic q_lcp):
      1. Sample q_lcp = c_lcs + N(0, q_noise · ‖c_lcs‖).
      2. Verify LCP(F, q_lcp) has a solution via Lemke. If not, resample
         (up to `max_resample` tries) — cleans out the infeasible q_lcp that
         choked §7.78's MIQP.
      3. Let λ★ = Lemke's δ_λ (the verified LCP solution).
      4. Set z_λ = λ★ + N(0, z_noise). This mirrors what the port's ADMM
         actually feeds the projection: z = x + ω where x is the QP primal
         and ω is the ADMM dual — clustered around a nearby LCP-feasible
         point, not adversarial noise.
      5. Sample η_seed ~ N(0, z_noise) for eq-12's second argument.

    Returns (instances, total_resamples) — instances always has n_instances
    entries; total_resamples measures how many draws were rejected.
    """
    n_lambda = F.shape[0]
    q_scale  = float(max(1e-3, np.linalg.norm(c_lcs))) * q_noise
    instances = []
    total_resamples = 0
    while len(instances) < n_instances:
        q = c_lcs + rng.normal(0.0, q_scale, size=n_lambda)

        # Lemke feasibility pre-check
        lam_star, lcp_res = solve_lcp(F, q)
        feasible = (lcp_res < 1e-4
                    and float(np.min(lam_star)) >= -1e-6
                    and float(np.min(F @ lam_star + q)) >= -1e-4)
        if not feasible:
            total_resamples += 1
            if total_resamples > max_resample * n_instances:
                # Give up and use whatever we have; the noise scale may be
                # too aggressive for this LCS.
                break
            continue

        z_lam = lam_star + rng.normal(0.0, z_noise, size=n_lambda)
        eta_s = rng.normal(0.0, z_noise, size=n_lambda)
        instances.append((z_lam, q, eta_s))

    return instances, total_resamples


# ---------------------------------------------------------------------------
# Timing driver — parametrized over n_λ (inferred from F.shape)
# ---------------------------------------------------------------------------
def time_method(name: str, fn, instances, n_reps: int) -> tuple[list, int, int]:
    """Return (times_seconds, n_ok, n_cap_hits)."""
    all_times: list[float] = []
    n_cap = 0
    for inst in instances:
        try:
            out = fn(inst)
            if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], bool):
                if out[1]:
                    n_cap += 1
        except Exception as e:
            print(f"    {name}: warm-up FAILED: {type(e).__name__}: {e}")
            continue

        for _ in range(n_reps):
            t0 = time.perf_counter()
            try:
                out = fn(inst)
                if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], bool):
                    if out[1]:
                        n_cap += 1
            except Exception:
                continue
            all_times.append(time.perf_counter() - t0)
    return all_times, len(all_times), n_cap


def summarize(times_s: list[float]) -> dict:
    if not times_s:
        return {"count": 0, "median_us": float("nan"), "p95_us": float("nan"),
                "min_us": float("nan"), "max_us": float("nan")}
    arr = np.asarray(times_s) * 1e6
    return {
        "count":     len(times_s),
        "median_us": float(np.median(arr)),
        "p95_us":    float(np.percentile(arr, 95)),
        "min_us":    float(arr.min()),
        "max_us":    float(arr.max()),
    }


# ---------------------------------------------------------------------------
# One-dump run
# ---------------------------------------------------------------------------
def benchmark_dump(dump_path: Path,
                   n_instances: int,
                   n_reps: int,
                   rng: np.random.Generator,
                   max_nodes: int,
                   instance_mode: str = "feasible") -> dict:
    """Run the 4-method benchmark on one dump. Return a result dict.

    instance_mode:
      "feasible" (default) — paper-grade recipe: verified LCP-feasible q_lcp;
                             z_λ = Lemke solution + noise.
      "synthetic"          — §7.78's original recipe: random N(0,·) for
                             (z_λ, q_lcp, η_seed). Used for the self-test
                             that must reproduce §7.78's 1543× headline.
    """
    raw = np.load(dump_path, allow_pickle=True)
    F      = raw["F"]
    c_lcs  = raw["c_lcs"]
    mu     = float(raw["mu"])
    num_normals = int(raw["J_n"].shape[0])
    n_lambda    = int(F.shape[0])

    print(f"  Dump:           {dump_path}")
    print(f"  n_lambda:       {n_lambda}   ({num_normals} contact(s), layout γ+λ_n+4λ_t per contact)")
    print(f"  mu:             {mu}")
    print(f"  ‖c_lcs‖:        {float(np.linalg.norm(c_lcs)):.4f}")
    print(f"  Instance mode:  {instance_mode}")
    print(f"  Instances:      {n_instances}")

    if instance_mode == "feasible":
        instances, n_resamples = generate_feasible_instances(
            F, c_lcs, n_instances, rng
        )
        print(f"  Feasible draws: {len(instances)}/{n_instances}  "
              f"(rejected {n_resamples} infeasible before acceptance)")
    elif instance_mode == "synthetic":
        instances, n_resamples = generate_legacy_synthetic_instances(
            F, c_lcs, n_instances, rng
        )
        print(f"  Synthetic draws: {len(instances)}/{n_instances}  "
              f"(no feasibility pre-check — §7.78 legacy recipe)")
    else:
        raise ValueError(f"Unknown instance_mode: {instance_mode}")
    if len(instances) < n_instances:
        print(f"  WARNING: only {len(instances)} instances generated; "
              "consider tightening Q_NOISE_SCALE.")

    if not instances:
        return {"dump": str(dump_path), "n_lambda": n_lambda,
                "num_contacts": num_normals,
                "miqp": summarize([]), "eq12": summarize([]),
                "lcp": summarize([]), "lor": summarize([]),
                "n_resamples": n_resamples, "n_cap_hits": 0}

    print(f"  Running MIQP (max_nodes={max_nodes}) ...", flush=True)
    miqp_times, miqp_n, miqp_cap = time_method(
        "MIQP",
        lambda inst: project_miqp(inst[0], F, inst[1], BIG_M, max_nodes),
        instances, n_reps,
    )
    print(f"  Running eq-12 ...", flush=True)
    eq12_times, _, _ = time_method(
        "eq-12",
        lambda inst: project_componentwise(inst[0], inst[2]),
        instances, n_reps,
    )
    print(f"  Running LCP-Lemke ...", flush=True)
    lcp_times, _, _ = time_method(
        "LCP-Lemke",
        lambda inst: project_lcp_lemke(inst[0], F, inst[1]),
        instances, n_reps,
    )
    print(f"  Running Lorentz-cone ...", flush=True)
    lor_times, _, _ = time_method(
        "Lorentz",
        lambda inst: project_lorentz(inst[0], num_normals, mu),
        instances, n_reps,
    )

    return {
        "dump": str(dump_path),
        "n_lambda":       n_lambda,
        "num_contacts":   num_normals,
        "n_resamples":    n_resamples,
        "n_cap_hits":     miqp_cap,
        "miqp":           summarize(miqp_times),
        "eq12":           summarize(eq12_times),
        "lcp":            summarize(lcp_times),
        "lor":            summarize(lor_times),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_result_block(res: dict) -> None:
    n_lam = res["n_lambda"]
    print()
    print("=" * 82)
    print(f"RESULTS — n_λ = {n_lam}  ({res['num_contacts']} contact(s))")
    print("=" * 82)
    header = (f"  {'method':<32s} {'median':>10s} {'p95':>10s} "
              f"{'min':>10s} {'max':>10s} {'n':>8s}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, key in [("1. C3-MIQP (Aydinoglu Big-M)",   "miqp"),
                       ("2. C3+ eq-12 (Bui componentwise)", "eq12"),
                       ("3. LCP-Lemke (Aydinoglu §V-B.3)", "lcp"),
                       ("4. Lorentz-cone (Aydinoglu)",      "lor")]:
        s = res[key]
        if s["count"] == 0:
            print(f"  {label:<32s} {'---':>10s} {'---':>10s} "
                  f"{'---':>10s} {'---':>10s} {0:>8d}")
        else:
            print(f"  {label:<32s} "
                  f"{s['median_us']:>10.3f} {s['p95_us']:>10.3f} "
                  f"{s['min_us']:>10.3f} {s['max_us']:>10.3f} "
                  f"{s['count']:>8d}")
    if res["miqp"]["count"] > 0 and res["eq12"]["count"] > 0:
        ratio = res["miqp"]["median_us"] / res["eq12"]["median_us"]
        oom   = float(np.log10(ratio))
        print(f"\n  MIQP / eq-12 ratio (median): {ratio:>10.1f}×   "
              f"({oom:+.2f} orders of magnitude)")
    if res["n_cap_hits"] > 0:
        print(f"  MIQP node cap hits: {res['n_cap_hits']}  (of {res['miqp']['count']} timed reps)")


def print_scaling_summary(results: list[dict]) -> None:
    if len(results) < 2:
        return
    print()
    print("=" * 82)
    print("SCALING SUMMARY — MIQP-vs-eq12 gap widening with n_λ")
    print("=" * 82)
    header = f"  {'n_λ':>4s} {'contacts':>9s} {'MIQP (µs)':>14s} {'eq-12 (µs)':>14s} {'ratio':>12s} {'orders':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for res in results:
        if res["miqp"]["count"] > 0 and res["eq12"]["count"] > 0:
            miqp = res["miqp"]["median_us"]
            eq12 = res["eq12"]["median_us"]
            ratio = miqp / eq12
            oom = float(np.log10(ratio))
            print(f"  {res['n_lambda']:>4d} {res['num_contacts']:>9d} "
                  f"{miqp:>14.3f} {eq12:>14.3f} {ratio:>12.1f} {oom:>+8.2f}")
        else:
            print(f"  {res['n_lambda']:>4d} {res['num_contacts']:>9d}  "
                  f"(insufficient data)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dump", type=str, default=None,
                   help="Single .npz dump to benchmark (self-test path).")
    p.add_argument("--dumps", type=str, default=None,
                   help="Directory of .npz dumps (multi-contact sweep).")
    p.add_argument("--out-csv", type=str, default="",
                   help="Optional CSV output for scaling summary.")
    p.add_argument("--instance-mode", choices=["feasible", "synthetic"],
                   default="feasible",
                   help="feasible (default) = Lemke-verified LCP-feasible "
                        "q_lcp + z_λ around Lemke solution (paper-grade). "
                        "synthetic = §7.78 legacy recipe (raw N(0,·)); use "
                        "for the reproduction self-test.")
    args = p.parse_args()

    if not args.dump and not args.dumps:
        p.error("provide --dump PATH or --dumps DIR")

    dump_paths: list[Path] = []
    if args.dump:
        dump_paths.append(Path(args.dump))
    if args.dumps:
        d = Path(args.dumps)
        dump_paths.extend(sorted(d.glob("*.npz")))

    print("=" * 82)
    print("§7.79 projection scaling benchmark — parametrized over n_λ")
    print("=" * 82)
    print(f"Dumps to benchmark: {len(dump_paths)}")
    for p_ in dump_paths:
        print(f"  {p_}")
    print(f"Instances / dump:   {N_INSTANCES}")
    print(f"Reps / instance:    {N_REPS}")
    print(f"Seed:               {RNG_SEED}")
    print(f"MIP backend:        Drake MixedIntegerBranchAndBound + OsqpSolver")
    print(f"MIQP node cap:      {MAX_MIQP_NODES}")
    print(f"Q noise scale:      {Q_NOISE_SCALE}  (× ‖c_lcs‖)")
    print(f"Z noise scale:      {Z_NOISE_SCALE}  (around Lemke solution)")

    rng = np.random.default_rng(RNG_SEED)
    results: list[dict] = []

    def _flush_csv(path: str, rs: list[dict]) -> None:
        rs_sorted = sorted(rs, key=lambda r: r["n_lambda"])
        with open(path, "w") as f:
            f.write("n_lambda,num_contacts,method,median_us,p95_us,min_us,max_us,count\n")
            for res in rs_sorted:
                for label, key in [("miqp", "miqp"), ("eq12", "eq12"),
                                   ("lcp_lemke", "lcp"), ("lorentz", "lor")]:
                    s = res[key]
                    f.write(f"{res['n_lambda']},{res['num_contacts']},{label},"
                            f"{s['median_us']:.6f},{s['p95_us']:.6f},"
                            f"{s['min_us']:.6f},{s['max_us']:.6f},{s['count']}\n")

    for dp in dump_paths:
        if not dp.exists():
            print(f"\nWARNING: {dp} not found — skipping.")
            continue
        print()
        print("-" * 82)
        print(f"BENCHMARKING: {dp}")
        print("-" * 82)
        res = benchmark_dump(dp, N_INSTANCES, N_REPS, rng, MAX_MIQP_NODES,
                             instance_mode=args.instance_mode)
        print_result_block(res)
        results.append(res)
        if args.out_csv:
            _flush_csv(args.out_csv, results)
            print(f"[CHECKPOINT] CSV updated -> {args.out_csv} "
                  f"({len(results)} n_lambda point(s))", flush=True)

    if len(results) > 1:
        results.sort(key=lambda r: r["n_lambda"])
        print_scaling_summary(results)

    if args.out_csv and results:
        _flush_csv(args.out_csv, results)
        print(f"\nCSV written to {args.out_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
