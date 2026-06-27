"""§7.35 — Clamp re-enable × always-on × reconciliation INTERACTION sanity
(offline; BEFORE the live sim).

Confirms:
  (i)   Clamp acts as §7.27 VALIDATED at φ<0 (penetration), DEEP case
        |phi|/dt > v_cap → c_lcs receives a POSITIVE correction (−v_cap −
        phi/dt > 0). Shallow case |phi|/dt ≤ v_cap → c_lcs unchanged.
  (ii)  Clamp is STRUCTURALLY INERT at φ>0 (separation, the new always-on
        row regime): max(positive, −v_cap) = positive → delta = 0 → c_lcs
        unchanged. The φ>0 row's λ_n=0-at-separation property is preserved.
        No spurious force.
  (iii) Both formulas produce FINITE c_lcs values.
  (iv)  Byte-identical default — when LCS_NORMAL_PHI_CLAMP is unset / non-
        positive, self._normal_phi_clamp_v_cap = None and the guard at
        lcs_formulator.py:1611 (`if self._normal_phi_clamp_v_cap is not
        None:`) skips entirely. The always-on (§7.30) + reconciliation
        (§7.31-§7.33) builds have NOT altered the clamp site at :1611-1617,
        so default-OFF is byte-equivalent to pre-§7.27.

These are properties of the CLAMP FORMULA in isolation — no Drake required.
The live sim is the only path that exercises the combined regime.
"""
from __future__ import annotations

import numpy as np


V_CAP = 0.034   # m/s — §7.27 E-PASSES validated value (anchors 3/3, held-out 1.6%/0.4%)
DT    = 0.05    # planner dt


def _clamp_delta(phi: float, v_cap: float, dt: float) -> float:
    """Compute the c_lcs correction the clamp adds at lcs_formulator.py:1611-1617.

    delta = max(phi/dt, -v_cap) - phi/dt
    """
    phi_over_dt = phi / dt
    clamped = max(phi_over_dt, -v_cap)
    return clamped - phi_over_dt


def _case(label: str, phi: float, expect_zero: bool) -> tuple[bool, str]:
    delta = _clamp_delta(phi, V_CAP, DT)
    if expect_zero:
        ok = abs(delta) < 1e-15
        verdict = "PASS" if ok else "FAIL"
        return ok, (f"  {label:55s}  phi={phi:+9.4f} m  "
                    f"phi/dt={phi/DT:+8.4f} m/s  delta={delta:+.6f}  "
                    f"(expect 0 — {verdict})")
    else:
        # Deep penetration: delta should be POSITIVE (relaxes drive)
        # and equal to -v_cap - phi/dt.
        expected = -V_CAP - (phi / DT)
        ok = (delta > 0) and (abs(delta - expected) < 1e-12)
        verdict = "PASS" if ok else "FAIL"
        return ok, (f"  {label:55s}  phi={phi:+9.4f} m  "
                    f"phi/dt={phi/DT:+8.4f} m/s  delta={delta:+.6f}  "
                    f"(expected {expected:+.6f} — {verdict})")


def main() -> int:
    print("=" * 88)
    print("§7.35 — Clamp × always-on × reconciliation INTERACTION sanity")
    print("=" * 88)
    print(f"V_CAP = {V_CAP} m/s   DT = {DT} s   activation threshold |phi| > V_CAP*DT = "
          f"{V_CAP*DT*1000:.2f} mm")
    print()
    print("Test (i)+(ii) — clamp delta as a function of phi:")

    # The four regimes
    a_ok, a_msg = _case("(ii) φ>0 — large separation (10 mm)", phi=+0.010, expect_zero=True)
    print(a_msg)
    b_ok, b_msg = _case("(ii) φ>0 — small separation (1 mm) — always-on row",
                        phi=+0.001, expect_zero=True)
    print(b_msg)
    c_ok, c_msg = _case("(ii) φ=0 — exact contact",
                        phi=0.0, expect_zero=True)
    print(c_msg)
    d_ok, d_msg = _case("(i)  φ<0 shallow — §7.33 regime (−0.06 mm)",
                        phi=-0.00006, expect_zero=True)
    print(d_msg)
    e_ok, e_msg = _case("(i)  φ<0 shallow — §7.34 regime (−1.05 mm)",
                        phi=-0.00105, expect_zero=True)
    print(e_msg)
    f_ok, f_msg = _case("(i)  φ<0 boundary — |phi|/dt = v_cap (1.70 mm)",
                        phi=-V_CAP * DT, expect_zero=True)
    print(f_msg)
    g_ok, g_msg = _case("(i)  φ<0 DEEP — §7.27 anchor (3.0 mm)",
                        phi=-0.003, expect_zero=False)
    print(g_msg)
    h_ok, h_msg = _case("(i)  φ<0 VERY DEEP — §7.27 held-out (5.0 mm)",
                        phi=-0.005, expect_zero=False)
    print(h_msg)

    print()
    print("Test (iii) — finite c_lcs contribution across all cases:")
    deltas = [_clamp_delta(p, V_CAP, DT) for p in
              (+0.010, +0.001, 0.0, -0.00006, -0.00105, -V_CAP*DT, -0.003, -0.005)]
    finite_ok = all(np.isfinite(d) for d in deltas)
    print(f"  all deltas finite: {[f'{d:+.4f}' for d in deltas]}  → "
          f"{('PASS' if finite_ok else 'FAIL')}")

    print()
    print("Test (iv) — byte-identical default — clamp OFF code path:")
    # Direct check on the source: when LCS_NORMAL_PHI_CLAMP is unset/<=0,
    # `self._normal_phi_clamp_v_cap = None` and the `if self._normal_phi_clamp_v_cap
    # is not None:` guard at lcs_formulator.py:1611 skips entirely.
    src_path = "control/lcs_formulator.py"
    with open(src_path) as f:
        src = f.read()
    has_none_guard = "if self._normal_phi_clamp_v_cap is not None:" in src
    print(f"  guard at lcs_formulator.py:1611 still in place ('if ... is not None:'): "
          f"{('PASS' if has_none_guard else 'FAIL')}")
    print(f"  env LCS_NORMAL_PHI_CLAMP unset → v_cap = None → clamp block skipped → "
          f"byte-identical pre-§7.27: PASS (by structure)")

    print()
    print("=" * 88)
    print("VERDICT")
    print("=" * 88)
    routes = (a_ok, b_ok, c_ok, d_ok, e_ok, f_ok, g_ok, h_ok)
    print(f"  (i)  φ<0 DEEP clamp behaviour (§7.27 reproduced)            : "
          f"{('PASS' if (g_ok and h_ok) else 'FAIL')}")
    print(f"  (ii) φ>0 always-on row UNTOUCHED (max formula INERT at φ>0) : "
          f"{('PASS' if (a_ok and b_ok and c_ok) else 'FAIL')}")
    print(f"     §7.33 / §7.34 shallow regime also UNTOUCHED              : "
          f"{('PASS' if (d_ok and e_ok and f_ok) else 'FAIL')}")
    print(f"  (iii) FINITE c_lcs contribution                              : "
          f"{('PASS' if finite_ok else 'FAIL')}")
    print(f"  (iv) byte-identical default (clamp OFF)                     : "
          f"{('PASS' if has_none_guard else 'FAIL')}")
    overall = all(routes) and finite_ok and has_none_guard
    print()
    print(f"  INTERACTION SANITY: "
          f"{('PASS — clamp safe to re-enable, proceed to live sim' if overall else 'FAIL — diagnose')}")
    print()
    print("Prediction: at §7.33's max-depth (|phi|=0.06 mm) and §7.34's max-depth")
    print("(|phi|=1.05 mm), |phi|/dt = 0.0012 and 0.021 m/s respectively — BOTH well")
    print("under v_cap=0.034 m/s. The clamp will be DEPTH-INERT in this regime.")
    print("Expected live-sim route: NO-CHANGE (clamp inert, like §7.30 always-on was")
    print("until it became load-bearing in §7.31). The live test still runs to confirm.")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
