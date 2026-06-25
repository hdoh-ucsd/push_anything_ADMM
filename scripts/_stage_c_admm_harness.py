"""Stage C disambiguation probe — offline ADMM harness.

Reads stage_c/admm_dump/seed0_full50.npz (the captured c3-mode tick 50
ADMM instance from the live pipeline) and runs three checks to pin the
non-convergence to one of the deck's three layers:

  (i)  ITER × ρ SWEEP — replay control.admm_solver.C3Solver._solve_c3plus
       on the dumped instance for max_iter ∈ {25, 100, 500, 1000} ×
       {ρ-as-shipped=100, ρ-fixed-at-various-values}. Reads per-iter
       primal/dual residual trajectories from primal_hist/dual_hist
       (already populated by the solver) and reports the SHAPE: does any
       cell drive pr/dr toward tol=1e-3, or do all plateau above?
       Routes layers 1 (impl) vs 2 (tuning) vs 3 (modeling).

  (ii) DIRECT FIXED-POINT EXISTENCE — encode the captured instance as a
       Linear Complementarity Problem (LCP) in the contact force λ and
       call a reference LCP solver independent of the ADMM. Drake's
       MobyLcpSolver is the available reference. If the LCP has a
       solution, layer 3 is ruled out (the modeling admits a feasible
       point; the ADMM just isn't finding it). If no LCP solution
       exists, layer 3 is the gate (no solver can find what doesn't
       exist).

  (iii) E-MATRIX STRUCTURE — inspect E for zero rows and report which
        slot positions they correspond to (γ slack vs λ_n vs λ_t).
        Verifies the deck's "tangent rows zeroed" claim against the
        captured live instance.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Add repo root so we can import control.admm_solver.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control.admm_solver import C3Solver  # noqa: E402


DUMP_PATH = Path("stage_c/admm_dump/seed0_full50.npz")
TOL = 1e-3

# Brute-force LCP oracle from (ii) — the ground-truth λ for the captured
# knot-0 LCP at u=0. Recorded here so the isolation cells can be diffed
# against a single canonical reference.
ORACLE_LAMBDA = np.array([0.146119, 0.583936, 0.0, 0.116787, 0.0, 0.116787])
ORACLE_LAMBDA_N = 0.583936  # the physical normal-force value


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_dump(path: Path) -> dict:
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


# ---------------------------------------------------------------------------
# (iii) E-matrix structure
# ---------------------------------------------------------------------------
def inspect_E(dump: dict) -> dict:
    """Report E zero rows + the slot they correspond to.

    n_lambda layout per the C3+ Stewart-Trinkle stack
    (admm_solver.py:852 `n_lambda = 2 * num_normals + n_t`):
        slots [0 : num_normals)          -> γ (friction-cone slack)
        slots [num_normals : 2*num_normals) -> λ_n (normal force)
        slots [2*num_normals : n_lambda) -> λ_t (tangent forces)
    """
    E = dump["E"]
    num_normals = int(dump["J_n"].shape[0])
    n_t = int(dump["J_t"].shape[0])
    n_lambda = E.shape[0]
    assert n_lambda == 2 * num_normals + n_t, (
        f"layout assumption violated: n_lambda={n_lambda}, "
        f"expected {2*num_normals + n_t}")

    slot_label = []
    for k in range(num_normals):
        slot_label.append(f"gamma[{k}]")
    for k in range(num_normals):
        slot_label.append(f"lambda_n[{k}]")
    for k in range(n_t):
        slot_label.append(f"lambda_t[{k}]")

    zero_rows = []
    for r in range(n_lambda):
        nz = np.count_nonzero(np.abs(E[r]) > 1e-12)
        if nz == 0:
            zero_rows.append(r)

    return dict(
        E_shape=E.shape,
        num_normals=num_normals,
        n_t=n_t,
        n_lambda=n_lambda,
        slot_labels=slot_label,
        zero_rows=zero_rows,
        zero_row_labels=[slot_label[r] for r in zero_rows],
        deck_claim_tangent_rows_zeroed=any(
            "lambda_t" in slot_label[r] for r in zero_rows),
        actual_zero_row_kind=(
            "gamma_slack_only" if all("gamma" in slot_label[r] for r in zero_rows)
            else "tangent_rows" if all("lambda_t" in slot_label[r] for r in zero_rows)
            else "mixed"
        ),
    )


# ---------------------------------------------------------------------------
# (i) ITER × ρ SWEEP via replay of C3Solver._solve_c3plus
# ---------------------------------------------------------------------------
def replay(dump: dict, *, max_iter: int, rho_override: float,
           projection: str = "componentwise") -> dict:
    """Replay the captured instance with given max_iter + rho override + projection.

    `projection` ∈ {"componentwise", "lcp", "lorentz_on_lambda"}.
    - "componentwise": the C3+ baseline (Bui eq 12).
    - "lcp": the LCP-per-knot via Lemke (the in-tree alternative path).
    - "lorentz_on_lambda": monkey-patch the componentwise call site to use
      the C3 Lorentz projection on the λ slot and pass-through on η. This
      is the literal "swap to Aydinoglu Lorentz" for the user's primary
      experiment (the C3-not-plus formulation has no η, so we approximate
      by leaving η at its pre-projection value within the C3+ z-vector).

    Returns dict with terminal state AND first-knot λ (for oracle diff)."""
    n_x = int(dump["n_x"])
    n_u = int(dump["n_u"])
    c3p_mode = "componentwise" if projection in ("componentwise", "lorentz_on_lambda") else projection
    solver = C3Solver(n_x=n_x, n_u=n_u, rho=rho_override, mode="c3plus",
                       c3plus_projection=c3p_mode)

    # Monkey-patch the componentwise projection to do Lorentz-on-λ
    # if requested. The C3 Lorentz projection operates on a flat
    # [λ_n_0..λ_n_{K-1}, λ_t_0..λ_t_{K-1}] layout per knot, with
    # num_normals normals and 4·num_normals tangents (Stewart-Trinkle).
    # The C3+ z-vector's λ slot at a knot is shape (n_lambda,) =
    # (2·num_normals + n_t,) — interpret slots [0:num_normals) as γ
    # slack (NOT in the C3 LCP), slots [num_normals:2·num_normals) as
    # λ_n, slots [2·num_normals:n_lambda) as λ_t. Map to the Lorentz
    # function's expected layout: pass [λ_n, λ_t]; leave γ unmodified
    # (γ is a Stewart-Trinkle slack, not in the Aydinoglu C3 formulation,
    # so identity-projection on γ is the natural pass-through).
    if projection == "lorentz_on_lambda":
        num_normals = int(dump["J_n"].shape[0])
        n_t = int(dump["J_t"].shape[0])
        mu_val = float(dump["mu"])

        from control.admm_solver import C3Solver as _C3S

        def _lorentz_swap(lam, eta, u_lambda=1.0, u_eta=1.0):
            # lam is the (2*num_normals + n_t,) C3+ slot.
            # Extract λ_n and λ_t, run C3 Lorentz, write back, pass-through γ.
            lam = np.asarray(lam, dtype=float)
            eta = np.asarray(eta, dtype=float)
            d_lam = lam.copy()
            d_eta = eta.copy()  # pass-through (C3 has no η)
            # Build C3 layout: [λ_n_0..λ_n_{K-1}, λ_t_0..λ_t_{K-1}]
            lam_for_lorentz = np.empty(num_normals + n_t)
            lam_for_lorentz[:num_normals] = lam[num_normals : 2*num_normals]   # λ_n
            lam_for_lorentz[num_normals:] = lam[2*num_normals : 2*num_normals + n_t]  # λ_t
            projected = _C3S._lorentz_project(lam_for_lorentz, num_normals, mu_val)
            d_lam[num_normals : 2*num_normals] = projected[:num_normals]
            d_lam[2*num_normals : 2*num_normals + n_t] = projected[num_normals:]
            # γ slot (d_lam[0:num_normals]) and η pass through unchanged.
            return d_lam, d_eta

        # Patch the instance method via bound function.
        solver._project_componentwise = lambda lam, eta, u_lambda, u_eta: \
            _lorentz_swap(lam, eta, u_lambda, u_eta)

    phi_arg = dump["phi"] if dump["phi"].size > 0 else None
    ul = dump["u_lower"]; ul = ul if ul.size > 0 else None
    uu = dump["u_upper"]; uu = uu if uu.size > 0 else None

    t0 = time.perf_counter()
    out = solver._solve_c3plus(
        x0=dump["x0"], A=dump["A"], B_ctrl=dump["B_ctrl"], D=dump["D"],
        d=dump["d"], E=dump["E"], F=dump["F"], H=dump["H"], c_lcs=dump["c_lcs"],
        J_n=dump["J_n"], J_t=dump["J_t"], mu=float(dump["mu"]),
        Q=dump["Q"], R=dump["R"], QN=dump["QN"], x_ref=dump["x_ref"],
        N=int(dump["N"]),
        admm_iter=max_iter,
        torque_limit=float(dump["torque_limit"]),
        phi=phi_arg, u_lower=ul, u_upper=uu,
    )
    elapsed = time.perf_counter() - t0

    pr_final = float(getattr(solver, "_last_pr_final", float("nan")))
    dr_final = float(getattr(solver, "_last_dr_final", float("nan")))
    iters    = int(getattr(solver, "_last_iters_used", 0))
    conv     = bool(getattr(solver, "_last_converged", False))

    # Pull first-knot λ for oracle diff. _solve_c3plus stores
    # `_last_lambda_n_first` (just λ_n vector) AND `_last_lambda_n_first_delta`
    # (delta view), `_last_lambda_n_first_zsol` (z view). Pull the full λ
    # block (γ, λ_n, λ_t) from z_sol via delta if available — but the
    # solver only exposes λ_n. For the diff against the oracle we use
    # the λ_n component which is the physically meaningful comparator.
    lam_n_first = getattr(solver, "_last_lambda_n_first_delta", None)
    if lam_n_first is None:
        lam_n_first = getattr(solver, "_last_lambda_n_first", None)
    lam_n_first_val = (
        float(lam_n_first[0]) if (lam_n_first is not None
                                   and hasattr(lam_n_first, "__len__")
                                   and len(lam_n_first) > 0)
        else float("nan")
    )

    return dict(
        projection=projection,
        max_iter=max_iter,
        rho_initial=rho_override,
        pr_final=pr_final,
        dr_final=dr_final,
        iters_used=iters,
        converged=conv,
        wall_s=elapsed,
        lam_n_first=lam_n_first_val,
        oracle_lam_n=ORACLE_LAMBDA_N,
        lam_n_diff=abs(lam_n_first_val - ORACLE_LAMBDA_N),
    )


def sweep(dump: dict) -> list:
    rho_grid = [100.0, 10.0, 1.0, 0.1]
    iter_grid = [25, 100, 500, 1000]
    cells = []
    for rho in rho_grid:
        for it in iter_grid:
            cell = replay(dump, max_iter=it, rho_override=rho)
            print(f"  rho={rho:>6.1f}  max_iter={it:>5}  "
                  f"iters_used={cell['iters_used']:>5}/{it}  "
                  f"pr={cell['pr_final']:.4e}  dr={cell['dr_final']:.4e}  "
                  f"conv={cell['converged']}  wall={cell['wall_s']:.2f}s")
            cells.append(cell)
    return cells


# ---------------------------------------------------------------------------
# Projection-swap isolation (the primary cut)
# ---------------------------------------------------------------------------
def projection_swap_isolation(dump: dict, max_iter: int = 1000,
                              rho: float = 100.0) -> list:
    """Three-cell projection comparison on the captured instance.

    Cell A — Bui componentwise (the live C3+ baseline, already known to
             oscillate per the disambiguation iter×ρ sweep).
    Cell B — LCP-per-knot via the in-tree Lemke solver (the strongest
             "correct projection" comparator in the codebase — uses a
             reference LCP solver inside the ADMM δ-step).
    Cell C — Aydinoglu Lorentz on the λ slot (the user's literal swap to
             the C3-not-plus per-contact friction-cone projection,
             approximated within the C3+ z-vector formulation by leaving
             γ and η pass-through; γ is a Stewart-Trinkle slack absent
             from the C3 formulation, so identity-projection on it is
             the closest available Aydinoglu-style operator).

    All three cells share: same captured instance, same rho, same
    max_iter, same OSQP block construction, same ρ-adaptation rule.
    The ONLY difference between A and B is the δ-projection function.
    """
    cells = []
    for proj in ["componentwise", "lcp", "lorentz_on_lambda"]:
        cell = replay(dump, max_iter=max_iter, rho_override=rho,
                      projection=proj)
        cells.append(cell)
        oracle_match = abs(cell["lam_n_first"] - ORACLE_LAMBDA_N) < 1e-3
        print(f"  projection={proj:>20}  iters={cell['iters_used']:>5}/{max_iter}  "
              f"pr={cell['pr_final']:.4e}  dr={cell['dr_final']:.4e}  "
              f"conv={cell['converged']}  "
              f"λ_n_first={cell['lam_n_first']:+.4f}  "
              f"oracle_diff={cell['lam_n_diff']:.4e}  "
              f"oracle_match={oracle_match}  wall={cell['wall_s']:.2f}s")
    return cells


# ---------------------------------------------------------------------------
# (ii) DIRECT FIXED-POINT EXISTENCE — LCP encoding + reference solver
# ---------------------------------------------------------------------------
def build_lcp(dump: dict):
    """Build the per-knot LCP at knot 0 — the smallest test of LCS
    complementarity feasibility for the captured instance.

    The per-knot LCS at knot 0 is:
        x_1   = A x_0 + B u_0 + D λ_0 + d
        0 ≤ λ_0 ⊥ (E x_0 + F λ_0 + H u_0 + c) ≥ 0
        u_0 free (unbounded; the dynamics fix x_0; the LCP is in λ_0
                  conditional on a chosen u_0).

    Decoupling u_0: in the live solve u_0 is a decision variable, but to
    test the LCP existence we treat u_0 = 0 (the most conservative test:
    if a feasible λ exists at u=0 and the captured x_0, the constraint
    set is non-empty; with the live u_0 chosen by the QP, feasibility
    can only widen). This is a sufficient test, not necessary — if
    the u=0 LCP is infeasible, try the captured u_0 from the live
    solve trajectory if available.

    Returns (M_lcp, q_lcp) where the LCP is:
        find λ s.t.  0 ≤ λ ⊥ (M_lcp λ + q_lcp) ≥ 0
    """
    E, F, H, c_lcs = dump["E"], dump["F"], dump["H"], dump["c_lcs"]
    x0 = dump["x0"]
    u0_test = np.zeros(int(dump["n_u"]))   # conservative test
    M_lcp = F
    q_lcp = E @ x0 + H @ u0_test + c_lcs
    return M_lcp, q_lcp


def solve_lcp_reference(M_lcp: np.ndarray, q_lcp: np.ndarray) -> dict:
    """Brute-force LCP solver — basis enumeration.

    For a small LCP `0 ≤ λ ⊥ (Mλ+q) ≥ 0` with dim n, enumerate the 2^n
    complementary bases. For each pattern z ∈ {0,1}^n:
        z_i = 0 means λ_i = 0, w_i free (i in W-basis)
        z_i = 1 means w_i = 0, λ_i free (i in Z-basis)
    With λ_i = 0 on a subset, solve the remaining linear system for the
    free λ_j's: M_{ZZ} λ_Z = -q_Z. Then check λ_Z ≥ 0 and w_W ≥ 0.
    Returns the first feasible basis found.

    For n=6 this is 64 LU solves — trivial. Robust against Drake API
    drift (modern MobyLcpSolver via MathematicalProgram returns
    overflow garbage on the simple test; not reliable here).
    """
    n = M_lcp.shape[0]
    assert q_lcp.shape == (n,)

    best = None
    feasible_bases = []
    for mask in range(1 << n):
        z_in = [(mask >> i) & 1 == 1 for i in range(n)]  # z_in[i]=True -> λ_i free
        z_idx = [i for i, b in enumerate(z_in) if b]
        w_idx = [i for i, b in enumerate(z_in) if not b]
        lam = np.zeros(n)

        if len(z_idx) > 0:
            M_zz = M_lcp[np.ix_(z_idx, z_idx)]
            q_z  = q_lcp[z_idx]
            try:
                lam_z = np.linalg.solve(M_zz, -q_z)
            except np.linalg.LinAlgError:
                continue
            lam[z_idx] = lam_z

        w = M_lcp @ lam + q_lcp
        # Feasibility: λ_z ≥ 0 (for active Z-basis), w_w ≥ 0 (for active W-basis),
        # AND complementarity by construction: λ_W = 0 and w_Z = 0 hold up to FP.
        lam_z_nonneg = bool(all(lam[i] >= -1e-9 for i in z_idx)) if z_idx else True
        w_w_nonneg   = bool(all(w[i]   >= -1e-9 for i in w_idx)) if w_idx else True
        # Also check the "forced zeros" honor the LCP equality
        compl_resid = max(
            (abs(w[i]) for i in z_idx),  # w should be ~0 on Z-basis
            default=0.0,
        )
        # (λ on W-basis is zero by construction, not measured)

        if lam_z_nonneg and w_w_nonneg and compl_resid < 1e-6:
            feasible_bases.append(dict(
                mask=mask,
                z_idx=z_idx,
                w_idx=w_idx,
                lam=lam.copy(),
                w=w.copy(),
                lam_nonneg=lam_z_nonneg,
                w_nonneg=w_w_nonneg,
                compl_resid=compl_resid,
            ))
            if best is None:
                best = feasible_bases[-1]

    if best is None:
        return dict(
            ok=True, method="brute_force_basis_enumeration_64",
            feasible=False,
            n_bases_searched=1 << n,
            n_feasible=0,
            error="no feasible complementary basis exists",
        )

    return dict(
        ok=True, method="brute_force_basis_enumeration_64",
        feasible=True,
        n_bases_searched=1 << n,
        n_feasible=len(feasible_bases),
        lam=best["lam"].tolist(),
        w=best["w"].tolist(),
        z_basis=best["z_idx"],
        w_basis=best["w_idx"],
        compl_resid_max=float(best["compl_resid"]),
        lam_nonneg=True,
        w_nonneg=True,
        nonneg_violation=0.0,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"[HARNESS] loading {DUMP_PATH}")
    dump = load_dump(DUMP_PATH)
    print(f"[HARNESS]   x0  ={dump['x0']}")
    print(f"[HARNESS]   phi ={dump['phi']}")
    print(f"[HARNESS]   admm_iter (captured) = {int(dump['admm_iter'])}")
    print(f"[HARNESS]   rho_initial (captured) = {float(dump['rho_initial'])}")

    print()
    print("=" * 70)
    print("(iii) E-MATRIX STRUCTURE")
    print("=" * 70)
    e_info = inspect_E(dump)
    print(f"  E shape           : {e_info['E_shape']}")
    print(f"  num_normals       : {e_info['num_normals']}")
    print(f"  n_t (tangents)    : {e_info['n_t']}")
    print(f"  n_lambda          : {e_info['n_lambda']}")
    print(f"  zero rows         : {e_info['zero_rows']}")
    print(f"  zero row labels   : {e_info['zero_row_labels']}")
    print(f"  deck claim tangent: {e_info['deck_claim_tangent_rows_zeroed']}")
    print(f"  actual kind       : {e_info['actual_zero_row_kind']}")

    print()
    print("=" * 70)
    print("(i) ITER × ρ SWEEP")
    print("=" * 70)
    cells = sweep(dump)
    any_converge = any(c["converged"] for c in cells)
    print(f"\n  any cell converged? {any_converge}")
    if any_converge:
        conv_cells = [c for c in cells if c["converged"]]
        print(f"  converging cells: {len(conv_cells)} / {len(cells)}")
        for c in conv_cells:
            print(f"    rho={c['rho_initial']:>6.1f} max_iter={c['max_iter']:>5} "
                  f"iters={c['iters_used']} pr={c['pr_final']:.2e} dr={c['dr_final']:.2e}")
    else:
        # Look at residual floor across cells
        pr_mins = sorted(c["pr_final"] for c in cells)
        dr_mins = sorted(c["dr_final"] for c in cells)
        print(f"  pr_final min/max across all cells: {pr_mins[0]:.4e} / {pr_mins[-1]:.4e}")
        print(f"  dr_final min/max across all cells: {dr_mins[0]:.4e} / {dr_mins[-1]:.4e}")

    print()
    print("=" * 70)
    print("(iv) PROJECTION-SWAP ISOLATION — 3 cells, rho=100, max_iter=1000")
    print("     (the primary cut: holds everything fixed except the δ-projection)")
    print("=" * 70)
    iso_cells = projection_swap_isolation(dump, max_iter=1000, rho=100.0)

    print()
    print("=" * 70)
    print("(ii) DIRECT FIXED-POINT EXISTENCE — LCP at knot 0 (u_0 = 0)")
    print("=" * 70)
    M_lcp, q_lcp = build_lcp(dump)
    print(f"  M_lcp (= F)  shape={M_lcp.shape}")
    print(f"  q_lcp        = {q_lcp}")
    lcp_result = solve_lcp_reference(M_lcp, q_lcp)
    if not lcp_result["ok"]:
        print(f"  ERROR: {lcp_result['error']}")
    else:
        print(f"  method               : {lcp_result['method']}")
        print(f"  lam                  : {np.round(lcp_result['lam'], 6).tolist()}")
        print(f"  w  (=Mλ+q)           : {np.round(lcp_result['w'],   6).tolist()}")
        print(f"  λ ≥ 0                : {lcp_result['lam_nonneg']}")
        print(f"  w ≥ 0                : {lcp_result['w_nonneg']}")
        print(f"  max|λ_i · w_i|       : {lcp_result['compl_resid_max']:.4e}")
        print(f"  min nonneg violation : {lcp_result['nonneg_violation']:.4e}")
        print(f"  FEASIBLE LCP SOLUTION: {lcp_result['feasible']}")

    print()
    print("=" * 70)
    print("LAYER DISAMBIGUATION + COMPONENT ISOLATION")
    print("=" * 70)
    # Match cells by projection name for the routing logic.
    by_proj = {c["projection"]: c for c in iso_cells}
    cw = by_proj.get("componentwise", {})
    lcp_cell = by_proj.get("lcp", {})
    lor_cell = by_proj.get("lorentz_on_lambda", {})

    def _oracle_match(c):
        """Primary signal: does the solver find the oracle λ_n? The formal
        pr/dr-vs-tol convergence is a SECONDARY signal — it can lag the
        actual solution discovery (e.g. an ADMM that has found the
        complementarity-feasible λ but whose dual variable ω is still
        adjusting will show pr/dr above tol even though λ is correct)."""
        return c.get("lam_n_diff", 1.0) < 1e-3

    cw_match  = _oracle_match(cw)
    lcp_match = _oracle_match(lcp_cell)
    lor_match = _oracle_match(lor_cell)

    print(f"  Cell A (componentwise — current C3+ baseline) converges-to-oracle: {cw_match}")
    print(f"  Cell B (LCP-per-knot Lemke               ) converges-to-oracle: {lcp_match}")
    print(f"  Cell C (Lorentz-on-λ Aydinoglu-style     ) converges-to-oracle: {lor_match}")

    print()
    if not cw_match and (lcp_match or lor_match):
        if lcp_match and lor_match:
            print("  -> (1a) PROJECTION ISOLATED: Cell A fails; both alt-projections")
            print("     (LCP-per-knot AND Lorentz-on-λ) succeed. The defect is in")
            print("     Bui componentwise eq (12) on the Stewart-Trinkle slot layout.")
            print("     C3-vs-C3+ comparative: C3 Lorentz projection converges where")
            print("     C3+ componentwise does not — INVERTS the deck's '4-5x faster'")
            print("     framing on this instance (correctness > speed).")
        elif lcp_match and not lor_match:
            print("  -> PROJECTION-CLASS ISOLATED: Cell A and Cell C both fail;")
            print("     only the in-tree Lemke LCP-per-knot path (Cell B) converges.")
            print("     The defect is BROADER than just the eq (12) formula — both")
            print("     the live componentwise and the Lorentz-on-λ approximation")
            print("     fail, suggesting the C3+ z-vector formulation (with η slack)")
            print("     interacts poorly with both componentwise and per-contact")
            print("     projections; only direct LCP solve inside the δ-step works.")
        elif lor_match and not lcp_match:
            print("  -> Unusual route: Lorentz-on-λ converges, LCP-per-knot does not.")
            print("     Inspect LCP-per-knot path implementation.")
    elif cw_match:
        print("  -> Unexpected: Cell A converged to the oracle at max_iter=1000.")
        print("     The iter×ρ sweep above showed 0/16 convergence at max_iter≤1000.")
        print("     Re-check the dump or the live-vs-replay parity.")
    else:
        # All three fail → not a projection problem
        print("  -> (1b)/(1c) PROJECTION RULED OUT: all three projections (componentwise,")
        print("     LCP-per-knot, Lorentz-on-λ) fail to converge to the oracle.")
        print("     The defect is NOT the projection function; the OSQP block")
        print("     construction (1b) or ρ-adaptation pathology (1c) is the cause.")
        print("     Next: instrument per-iteration internals (λ, δ, ω, η, OSQP outputs,")
        print("     live ρ) and diff against the oracle to localize 1b vs 1c.")

    # Echo the original layer-2/3 disambiguation for reference.
    print()
    print("  (original disambiguation context)")
    if any_converge:
        print("  iter×ρ sweep — some cell converged → layer 2 not the sole gate.")
    elif lcp_result.get("ok") and lcp_result.get("feasible"):
        print("  iter×ρ sweep 0/16 converged + LCP at u=0 feasible (oracle exists).")
        print("  -> modeling REFUTED; tuning REFUTED; ADMM iteration scheme is the gate.")
    else:
        print("  iter×ρ sweep 0/16 converged + LCP at u=0 infeasible — modeling-leaning.")


if __name__ == "__main__":
    main()
