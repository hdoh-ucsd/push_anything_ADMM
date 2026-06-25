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
def replay(dump: dict, *, max_iter: int, rho_override: float) -> dict:
    """Replay the captured instance with given max_iter + rho override.

    Returns dict with the per-iteration primal/dual histories AND the
    terminal state."""
    n_x = int(dump["n_x"])
    n_u = int(dump["n_u"])
    solver = C3Solver(n_x=n_x, n_u=n_u, rho=rho_override, mode="c3plus")

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
    return dict(
        max_iter=max_iter,
        rho_initial=rho_override,
        pr_final=pr_final,
        dr_final=dr_final,
        iters_used=iters,
        converged=conv,
        wall_s=elapsed,
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
    print("LAYER DISAMBIGUATION")
    print("=" * 70)
    if any_converge:
        print("  LAYER 1-2 (FIXABLE): some (rho, max_iter) cell converged.")
        print("  -> tune ρ + raise max_iter in the live solver; REOPEN ALIGNMENT.")
    elif lcp_result.get("ok") and lcp_result.get("feasible"):
        print("  AMBIGUOUS / leans LAYER 2: LCP has a feasible solution at u=0,")
        print("  but the ADMM cannot reach it within the swept budgets.")
        print("  -> the modeling admits a fixed point; ADMM tuning / projection")
        print("     details are blocking — investigate further before declaring research.")
    else:
        print("  LAYER 3 (MODELING) leans CONFIRMED: no swept ADMM cell converges")
        print("  AND the direct LCP solver could not find a feasible point at u=0.")
        print("  -> the §0 #2 research target is the gate; spin out the research")
        print("     workstream.")


if __name__ == "__main__":
    main()
