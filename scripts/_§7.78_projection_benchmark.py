"""§7.78 — Standalone projection benchmark (Bui 2026 Table III reproduction).

Times four projection methods PER SUB-PROBLEM on identical instances built
from the captured live ADMM dump (`stage_c/admm_dump/seed0_full50.npz`):

  1. C3-MIQP (Aydinoglu 2024)         — Big-M mixed-integer QP for LCP projection.
                                        Reference C3's projection sub-step (per timestep,
                                        per ADMM iter). Solved via Drake's
                                        MixedIntegerBranchAndBound + OsqpSolver.
  2. C3+ componentwise eq-12 (Bui)    — O(n_λ) scalar case-test, no solver call.
                                        Port's default C3+ projection path
                                        (admm_solver.py:_project_componentwise).
  3. LCP / Lemke                       — Aydinoglu §V-B.3.b retrofit into C3+
                                        (port's `--c3plus-projection lcp` variant).
                                        Drake UnrevisedLemkeSolver wrapper.
  4. Lorentz-cone                      — Per-contact friction-cone projection
                                        (Aydinoglu 2024, closed-form via Moreau
                                        decomposition). NOT the same set as LCP —
                                        included as a reference for closed-form
                                        speed floor. Port's C3 mode uses this.

Sub-problem: min ||δ_λ - z_λ||²  s.t. LCP(δ_λ; F, q_lcp)   (methods 1, 3)
             or projection onto per-scalar-pair complementarity (method 2)
             or projection onto friction cone (method 4).

Instances are built from the dump's F (n_λ=6, one EE-BOX contact,
Stewart-Trinkle stack: γ + λ_n + 4 λ_t) and randomly synthesized
(z_λ, q_lcp) pairs with realistic magnitudes.

The Bui Table III claim: MIQP is ~4-5 orders slower than eq-12 (per projection).
Report medians + p95 + count. Print the ratio.

Usage:
    python scripts/_§7.78_projection_benchmark.py

Env overrides:
    BENCH_N_INSTANCES  (default 30)   — how many distinct (z_λ, q_lcp) pairs
    BENCH_N_REPS       (default 100)  — repetitions per instance per method
    BENCH_SEED         (default 0)    — RNG seed
    BENCH_OUT_CSV      (default off)  — path to write per-method timings CSV
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control.admm_solver import C3Solver         # noqa: E402  (port projections)
from control.lcp_solver   import solve_lcp       # noqa: E402  (port LCP wrapper)

from pydrake.solvers import (                    # noqa: E402
    MathematicalProgram,
    MixedIntegerBranchAndBound,
    OsqpSolver,
)


DUMP_PATH = Path("stage_c/admm_dump/seed0_full50.npz")

N_INSTANCES = int(os.environ.get("BENCH_N_INSTANCES", "30"))
N_REPS      = int(os.environ.get("BENCH_N_REPS",       "100"))
RNG_SEED    = int(os.environ.get("BENCH_SEED",         "0"))
OUT_CSV     = os.environ.get("BENCH_OUT_CSV", "")

# Big-M constant for the MIQP LCP encoding. Larger = weaker relaxation,
# smaller = potentially infeasible if the true λ_n exceeds M. From §7.72
# empirical, λ_n_max ≈ 7.5, so M=100 is a safe upper bound.
BIG_M = 100.0


# ---------------------------------------------------------------------------
# Method 1: MIQP LCP projection via Drake B&B
# ---------------------------------------------------------------------------
def project_miqp(z_lam: np.ndarray,
                 F: np.ndarray,
                 q_lcp: np.ndarray,
                 big_m: float = BIG_M) -> np.ndarray:
    """
    Solve:  min  (δ - z)ᵀ (δ - z)   (unweighted; matches Aydinoglu 2024 eq (11)
                                     at G_λ = I)
            s.t. δ ≥ 0
                 w = F·δ + q ≥ 0
                 δᵢ ≤ M · bᵢ
                 wᵢ ≤ M · (1 - bᵢ)
                 bᵢ ∈ {0, 1}
    """
    n = z_lam.shape[0]
    prog  = MathematicalProgram()
    delta = prog.NewContinuousVariables(n, "delta")
    b     = prog.NewBinaryVariables(n, "b")

    # Objective: min ||δ - z||²  (Drake wants the Hessian; write as
    # 0.5 δᵀ (2I) δ - 2 z·δ  = ||δ-z||² - ||z||² up to a constant)
    Q_obj = 2.0 * np.eye(n)
    prog.AddQuadraticCost(Q_obj, -2.0 * z_lam, delta)

    # δ ≥ 0 and w = F·δ + q ≥ 0
    for i in range(n):
        prog.AddLinearConstraint(delta[i] >= 0.0)
        row = F[i, :]
        prog.AddLinearConstraint(row @ delta + q_lcp[i] >= 0.0)

    # Big-M complementarity
    for i in range(n):
        prog.AddLinearConstraint(delta[i] <= big_m * b[i])
        row = F[i, :]
        prog.AddLinearConstraint(row @ delta + q_lcp[i] <= big_m * (1.0 - b[i]))

    bnb = MixedIntegerBranchAndBound(prog, OsqpSolver.id())
    _ = bnb.Solve()
    return np.asarray(bnb.GetSolution(delta), dtype=float)


# ---------------------------------------------------------------------------
# Method 2: C3+ componentwise eq-12 (Bui)
# ---------------------------------------------------------------------------
def project_componentwise(z_lam: np.ndarray,
                          eta_seed: np.ndarray,
                          u_lam: float = 1.0,
                          u_eta: float = 1.0) -> np.ndarray:
    """
    C3+ eq-12 (Bui 2026). Operates on (λ°, η°) pairs elementwise —
    no F, no q. Runs the port's exact implementation.

    Note: eq-12 projects on the (λ, η) complementarity set. Since
    the C3+ ADMM's z-vector includes both λ and η blocks, we pass
    both. The eta_seed here corresponds to what the ADMM would have
    passed as z_η + ω_η at this iter.
    """
    delta_lam, _ = C3Solver._project_componentwise(z_lam, eta_seed, u_lam, u_eta)
    return delta_lam


# ---------------------------------------------------------------------------
# Method 3: LCP / Lemke (port's --c3plus-projection lcp variant)
# ---------------------------------------------------------------------------
def project_lcp_lemke(z_lam: np.ndarray,
                      F: np.ndarray,
                      q_lcp: np.ndarray) -> np.ndarray:
    """
    Port's LCP-projection retrofit (Aydinoglu §V-B.3.b): the δ_λ is the
    LCP solution given (F, q_lcp = E·δ_x + H·δ_u + c). Independent of
    z_lam — this is the same "projection" the port uses in c3plus lcp
    mode inside `admm_solver._solve_c3plus`.

    Returns just δ_λ; the port derives δ_η = max(F·δ_λ + q, 0) from it.
    """
    delta_lam, _ = solve_lcp(F, q_lcp)
    return delta_lam


# ---------------------------------------------------------------------------
# Method 4: Lorentz-cone friction projection (port C3 mode)
# ---------------------------------------------------------------------------
def project_lorentz(z_lam: np.ndarray,
                    num_normals: int,
                    mu: float) -> np.ndarray:
    """
    Per-contact friction cone projection. Layout expected:
        [λ_n_0, ..., λ_n_{K-1}, λ_t_0(4), ..., λ_t_{K-1}(4)]
    where K = num_normals. NOTE: the dump's n_lambda=6 layout is
    (γ, λ_n, λ_t×4) — different from the Lorentz call's assumed layout.
    We drop γ and pass the remaining 5 as (λ_n=1, λ_t=4).
    """
    lam_no_gamma = z_lam[1:]  # drop γ slot (0)
    return C3Solver._lorentz_project(lam_no_gamma, num_normals, mu)


# ---------------------------------------------------------------------------
# Instance generator
# ---------------------------------------------------------------------------
def generate_instances(F: np.ndarray,
                       c_lcs: np.ndarray,
                       n_instances: int,
                       rng: np.random.Generator) -> list[tuple[np.ndarray,
                                                               np.ndarray,
                                                               np.ndarray]]:
    """
    Generate n_instances of (z_λ, q_lcp, eta_seed) tuples.

    z_λ magnitude: from §7.72 log, λ_n_max ≈ 5-8, so sample z_λ ~ N(0, 5).
    q_lcp: sample as c_lcs + perturbation ~ N(0, 0.5) — the ADMM's q_lcp
           inside the projection is E·δ_x + H·δ_u + c, which fluctuates
           around c_lcs with amplitude related to the state error.
    eta_seed: for eq-12, η° = z_η + ω_η at the current ADMM iter. Sample
              magnitude-matched to z_λ.
    """
    n_lambda = F.shape[0]
    instances = []
    for _ in range(n_instances):
        z_lam = rng.normal(0.0, 5.0, size=n_lambda)
        q     = c_lcs + rng.normal(0.0, 0.5, size=n_lambda)
        eta_s = rng.normal(0.0, 5.0, size=n_lambda)
        instances.append((z_lam, q, eta_s))
    return instances


# ---------------------------------------------------------------------------
# Timing driver
# ---------------------------------------------------------------------------
def time_method(name: str,
                fn,
                instances: list,
                n_reps: int) -> tuple[list[float], int]:
    """
    Time `fn(instance)` for each instance × n_reps. Return (all_times_seconds,
    n_reps_effective).

    First rep on each instance is discarded as warm-up (JIT / cache).
    Timing uses time.perf_counter().
    """
    all_times: list[float] = []
    for inst in instances:
        # Warm-up rep, discarded
        try:
            _ = fn(inst)
        except Exception as e:
            print(f"    {name}: warm-up FAILED on instance: {type(e).__name__}: {e}")
            continue

        for _ in range(n_reps):
            t0 = time.perf_counter()
            try:
                _ = fn(inst)
            except Exception:
                # Solver may fail on some instances (LCP infeasible etc);
                # skip the timing for this rep.
                continue
            all_times.append(time.perf_counter() - t0)
    return all_times, len(all_times)


def summarize(times_s: list[float]) -> dict:
    if not times_s:
        return {"count": 0, "median_us": float("nan"), "p95_us": float("nan"),
                "min_us": float("nan"), "max_us": float("nan")}
    arr = np.asarray(times_s) * 1e6  # to microseconds
    return {
        "count":      len(times_s),
        "median_us":  float(np.median(arr)),
        "p95_us":     float(np.percentile(arr, 95)),
        "min_us":     float(arr.min()),
        "max_us":     float(arr.max()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    if not DUMP_PATH.exists():
        print(f"ERROR: dump not found at {DUMP_PATH}")
        return 2

    raw = np.load(DUMP_PATH, allow_pickle=True)
    F      = raw["F"]
    c_lcs  = raw["c_lcs"]
    mu     = float(raw["mu"])
    num_normals = int(raw["J_n"].shape[0])
    n_lambda    = F.shape[0]

    print("=" * 78)
    print("§7.78 projection benchmark — Bui Table III reproduction")
    print("=" * 78)
    print(f"Dump:          {DUMP_PATH}")
    print(f"n_lambda:      {n_lambda}   (layout: γ×{num_normals} + λ_n×{num_normals} + λ_t×{4*num_normals})")
    print(f"μ (friction):  {mu}")
    print(f"F PSD?         {'yes' if np.min(np.linalg.eigvalsh(0.5*(F+F.T))) >= -1e-9 else 'no (copositive-plus)'}")
    print(f"Instances:     {N_INSTANCES}")
    print(f"Reps/instance: {N_REPS}")
    print(f"Seed:          {RNG_SEED}")
    print(f"MIP backend:   Drake MixedIntegerBranchAndBound + OsqpSolver")
    print()

    rng = np.random.default_rng(RNG_SEED)
    instances = generate_instances(F, c_lcs, N_INSTANCES, rng)

    print("Timing methods (warm-up rep discarded, all timings in µs)...")
    print()

    # Method 1: MIQP
    print("  running MIQP ...", flush=True)
    miqp_times, miqp_n = time_method(
        "MIQP",
        lambda inst: project_miqp(inst[0], F, inst[1]),
        instances, N_REPS,
    )
    miqp = summarize(miqp_times)

    # Method 2: componentwise eq-12
    print("  running eq-12 componentwise ...", flush=True)
    eq12_times, eq12_n = time_method(
        "eq-12",
        lambda inst: project_componentwise(inst[0], inst[2]),
        instances, N_REPS,
    )
    eq12 = summarize(eq12_times)

    # Method 3: LCP-Lemke
    print("  running LCP-Lemke ...", flush=True)
    lcp_times, lcp_n = time_method(
        "LCP-Lemke",
        lambda inst: project_lcp_lemke(inst[0], F, inst[1]),
        instances, N_REPS,
    )
    lcp = summarize(lcp_times)

    # Method 4: Lorentz-cone
    print("  running Lorentz-cone ...", flush=True)
    lor_times, lor_n = time_method(
        "Lorentz",
        lambda inst: project_lorentz(inst[0], num_normals, mu),
        instances, N_REPS,
    )
    lor = summarize(lor_times)

    # Table
    print()
    print("=" * 78)
    print("PER-PROJECTION TIMING (µs)")
    print("=" * 78)
    header = f"  {'method':<30s} {'median':>10s} {'p95':>10s} {'min':>10s} {'max':>10s} {'n':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, s in [("1. C3-MIQP (Aydinoglu Big-M)", miqp),
                     ("2. C3+ eq-12 (Bui componentwise)", eq12),
                     ("3. LCP-Lemke (Aydinoglu §V-B.3.b)", lcp),
                     ("4. Lorentz-cone (Aydinoglu closed)", lor)]:
        if s["count"] == 0:
            print(f"  {label:<30s} {'---':>10s} {'---':>10s} {'---':>10s} {'---':>10s} {0:>8d}")
        else:
            print(f"  {label:<30s} "
                  f"{s['median_us']:>10.3f} {s['p95_us']:>10.3f} "
                  f"{s['min_us']:>10.3f} {s['max_us']:>10.3f} "
                  f"{s['count']:>8d}")

    # Headline ratio
    print()
    print("=" * 78)
    print("BUI TABLE III HEADLINE — MIQP vs eq-12 speed ratio")
    print("=" * 78)
    if miqp["count"] > 0 and eq12["count"] > 0:
        ratio_median = miqp["median_us"] / eq12["median_us"]
        ratio_p95    = miqp["p95_us"]    / eq12["p95_us"]
        oom_median   = np.log10(ratio_median)
        print(f"  median: MIQP / eq-12 = {ratio_median:.1f}×  ({oom_median:.2f} orders of magnitude)")
        print(f"  p95:    MIQP / eq-12 = {ratio_p95:.1f}×")
    else:
        print("  Cannot compute — one or both methods had zero successful runs.")

    # Optional CSV
    if OUT_CSV:
        with open(OUT_CSV, "w") as f:
            f.write("method,median_us,p95_us,min_us,max_us,count\n")
            for label, s in [("miqp", miqp), ("eq12", eq12),
                             ("lcp_lemke", lcp), ("lorentz", lor)]:
                f.write(f"{label},{s['median_us']:.6f},{s['p95_us']:.6f},"
                        f"{s['min_us']:.6f},{s['max_us']:.6f},{s['count']}\n")
        print(f"\nCSV written to {OUT_CSV}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
