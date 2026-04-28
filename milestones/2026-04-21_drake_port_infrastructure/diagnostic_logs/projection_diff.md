# Projection Diff: Drake vs Sandbox

## Verdict: IDENTICAL — no algorithmic difference

Both implementations use the same 3-case projection onto
K = {λ_n ≥ 0, ‖λ_t‖₂ ≤ μ·λ_n} with identical formulas.

---

## Drake: control/admm_solver.py — C3Solver._project_single_contact

```python
@staticmethod
def _project_single_contact(lam_n: float,
                            lam_t: np.ndarray,
                            mu: float) -> tuple[float, np.ndarray]:
    b_norm = float(np.linalg.norm(lam_t))

    if b_norm <= mu * lam_n + 1e-12:           # Case 1: inside cone
        return float(lam_n), lam_t.copy()

    if mu * b_norm <= -lam_n + 1e-12:           # Case 2: polar cone -> apex
        return 0.0, np.zeros_like(lam_t)

    # Case 3: project onto cone surface
    s     = (lam_n + mu * b_norm) / (1.0 + mu * mu)
    t_new = (mu * s / b_norm) * lam_t

    t_norm = float(np.linalg.norm(t_new))
    if t_norm > mu * s + 1e-8:
        raise AssertionError(...)
    return s, t_new
```

## Sandbox: planar_sandbox/c3_solver.py — _project_single_contact

```python
def _project_single_contact(
    lam_n: float, lam_t: np.ndarray, mu: float
) -> tuple[float, np.ndarray]:
    b_norm = float(np.linalg.norm(lam_t))

    if b_norm <= mu * lam_n + 1e-12:           # Case 1: inside
        return float(lam_n), lam_t.copy()

    if mu * b_norm <= -lam_n + 1e-12:          # Case 2: polar -> apex
        return 0.0, np.zeros_like(lam_t)

    # Case 3: surface
    s     = (lam_n + mu * b_norm) / (1.0 + mu * mu)
    t_new = (mu * s / b_norm) * lam_t

    assert float(np.linalg.norm(t_new)) <= mu * s + 1e-6, ...
    return s, t_new
```

---

## Case-by-case comparison

| Case | Condition | Drake | Sandbox | Match? |
|------|-----------|-------|---------|--------|
| 1 — inside | `‖λ_t‖ ≤ μ·λ_n + 1e-12` | return unchanged | return unchanged | ✓ |
| 2 — polar  | `μ·‖λ_t‖ ≤ −λ_n + 1e-12` | return (0, 0) | return (0, 0) | ✓ |
| 3 — surface | else | `s=(a+μb)/(1+μ²)`, `t*=μs(b/‖b‖)` | same formula | ✓ |

Both use `(1 + μ²)` denominator (not `(1 + μ)²`). Formulas are identical.

## Minor differences (non-algorithmic)
| Item | Drake | Sandbox |
|------|-------|---------|
| Tolerance in Case 1/2 | `1e-12` | `1e-12` |
| Surface sanity check | `raise AssertionError` | `assert` statement |
| Surface check tolerance | `1e-8` | `1e-6` |

None of these affect the projection result.

## Conclusion
The projection port is correct. The Branch B failure is NOT caused by a
wrong projection formula. Root cause is in the approach/proxy layer:
the arm freezes at perp=0.078m before making contact, so the ADMM
(and projection) never even runs.
