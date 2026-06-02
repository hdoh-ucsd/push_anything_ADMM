"""LCP-residual fork: distinguish 'planner-wants-harder' (benign-ish) from
'projection-failing / fictional-lambda re-emerging' (serious).

Re-parses lcp_residual_probe/seed{2,4}_lcp_residual.log produced AFTER the
admm_solver.py instrumentation commit (d316462). The [C3+] line now ends
with `lcp_res_max=X.XXe-Y` — the per-step LCP residual stashed at
admm_solver.py:1056.

Pre-registered fork:
  * residual ~1e-7 at violation ticks  → BENIGN-ish (projection working,
    planner genuinely wants >5N under stuck contact).
  * residual >> 1e-7 at violation ticks → SERIOUS (LCP solution itself
    drifting; fictional-lambda basin re-emerging in this regime).
"""

from __future__ import annotations
import re
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, "/root/push_anything_ADMM/scripts")
from probe_lambda_regression import (
    parse_log as _parse_old, contact_windows, RE_GATE, LAM_VIOL,
)

REPO = Path("/root/push_anything_ADMM")
PROBE = REPO / "lcp_residual_probe"

# Updated regex includes lcp_res_max=X.XXe-Y
RE_C3_LCP = re.compile(
    r"^\[C3\+\]\s+step=(\d+)\s+\|u\[0\]\|=([0-9.eE+-]+)N\s+"
    r"λ_n_max=([0-9.eE+-]+)\s+η_n_max=([0-9.eE+-]+)\s+"
    r"primal=([0-9.eE+-]+)\s+iters=\d+/\d+\s+"
    r"lcp_res_max=([0-9.eE+-]+)"
)
RE_ADMM = re.compile(
    r"^\[ADMM-C3\+\]\s+primal:\s*([0-9.eE+-]+)->([0-9.eE+-]+)\s+"
    r"dual:\s*([0-9.eE+-]+)->([0-9.eE+-]+)"
)


from dataclasses import dataclass, field

@dataclass
class Inner:
    c3_step: int
    lam_n_max: float
    primal_final: float
    dual_final: float
    lcp_res_max: float

@dataclass
class Tick:
    gate_step: int
    A_is_ee: int
    box_xy: Tuple[float, float]
    inner: List[Inner] = field(default_factory=list)

    @property
    def lam_n_max(self) -> float:
        return max((s.lam_n_max for s in self.inner), default=0.0)

    @property
    def violation(self) -> bool:
        return self.lam_n_max >= LAM_VIOL

    @property
    def worst(self) -> Inner | None:
        if not self.inner:
            return None
        return max(self.inner, key=lambda s: s.lam_n_max)


def parse(path: Path) -> List[Tick]:
    ticks: List[Tick] = []
    pending_admm: Tuple[float, float] | None = None
    pending_inner: List[Inner] = []
    with path.open() as f:
        for line in f:
            m = RE_ADMM.match(line)
            if m:
                pending_admm = (float(m.group(2)), float(m.group(4)))
                continue
            m = RE_C3_LCP.match(line)
            if m:
                pf = pending_admm[0] if pending_admm else float('nan')
                df = pending_admm[1] if pending_admm else float('nan')
                pending_inner.append(Inner(
                    c3_step=int(m.group(1)),
                    lam_n_max=float(m.group(3)),
                    primal_final=pf,
                    dual_final=df,
                    lcp_res_max=float(m.group(6)),
                ))
                pending_admm = None
                continue
            m = RE_GATE.match(line)
            if m:
                box_parts = [float(x) for x in m.group(3).split(",")]
                ticks.append(Tick(
                    gate_step=int(m.group(1)),
                    A_is_ee=int(m.group(2)),
                    box_xy=(box_parts[0], box_parts[1]),
                    inner=pending_inner,
                ))
                pending_inner = []
    return ticks


def summarize(label: str, path: Path):
    if not path.exists():
        print(f"  [{label}] missing: {path}")
        return None
    ticks = parse(path)
    n_total = len(ticks)
    n_contact = sum(1 for t in ticks if t.A_is_ee == 1)
    viol = [t for t in ticks if t.violation]
    n_v = len(viol)
    runs = contact_windows(ticks)

    # all inner solves with their residuals — flat
    all_inner = [s for t in ticks for s in t.inner]
    n_inner = len(all_inner)
    n_inner_viol = sum(1 for s in all_inner if s.lam_n_max >= LAM_VIOL)

    def quantiles(xs):
        if not xs:
            return None
        xs = sorted(xs)
        n = len(xs)
        def q(p):
            i = max(0, min(n - 1, int(round(p * (n - 1)))))
            return xs[i]
        return {
            "min": xs[0], "p25": q(0.25), "med": q(0.5),
            "p75": q(0.75), "p90": q(0.9), "p99": q(0.99), "max": xs[-1],
        }

    # Residual distribution split by violation status (inner-solve level)
    res_viol = [s.lcp_res_max for s in all_inner if s.lam_n_max >= LAM_VIOL]
    res_bounded_contact = [
        s.lcp_res_max for t in ticks if t.A_is_ee == 1 and not t.violation
        for s in t.inner if s.lam_n_max < LAM_VIOL
    ]
    res_no_contact = [
        s.lcp_res_max for t in ticks if t.A_is_ee == 0
        for s in t.inner
    ]
    qv = quantiles(res_viol)
    qb = quantiles(res_bounded_contact)
    qn = quantiles(res_no_contact)

    print(f"=== {label} :: {path.name} ===")
    print(f"  ticks={n_total}  contact_ticks={n_contact}  viol_ticks={n_v}")
    print(f"  inner-solves total={n_inner}  inner-solves with λ_n>=5: {n_inner_viol}")
    print(f"  contact_runs={len(runs)}  max_run_len={max((e-s+1 for s,e in runs), default=0)}")
    print()
    print(f"  LCP residual distribution (lcp_res_max):")
    def fmt(d, name):
        if d is None:
            print(f"    {name:24s}  (no samples)")
            return
        print(f"    {name:24s}  min={d['min']:.2e}  p25={d['p25']:.2e}  med={d['med']:.2e}  "
              f"p75={d['p75']:.2e}  p90={d['p90']:.2e}  p99={d['p99']:.2e}  max={d['max']:.2e}")
    fmt(qn, "no-contact inner")
    fmt(qb, "contact-bounded inner")
    fmt(qv, "violation inner (λ_n≥5)")
    print()

    # Diagnostic: at each violation, what was the residual?
    if viol:
        print(f"  first 6 violation ticks (worst inner solve per tick):")
        for i, t in enumerate(viol[:6]):
            w = t.worst
            print(f"    viol[{i}] gate_step={t.gate_step} lam_n_max={w.lam_n_max:.3f}  "
                  f"lcp_res_max={w.lcp_res_max:.2e}  primal_admm={w.primal_final:.2e}  "
                  f"dual_admm={w.dual_final:.2e}  box_xy=({t.box_xy[0]:+.4f},{t.box_xy[1]:+.4f})")
    print()
    return {
        "n_contact": n_contact, "n_v": n_v,
        "res_viol": res_viol, "res_bounded_contact": res_bounded_contact,
        "res_no_contact": res_no_contact,
    }


def main():
    # Reference: baseline no-contact ticks always near machine precision
    print(f"Reference: lcp_res_max threshold for 'machine precision' ≈ 1e-7\n"
          f"Pre-registered fork:\n"
          f"  res_viol ~1e-7 → BENIGN-ish (planner genuinely wants >5N)\n"
          f"  res_viol >>1e-7 → SERIOUS (projection failing in this regime)\n")
    paths = {
        "seed2 (Mode B wrong-face route)": PROBE / "seed2_lcp_residual.log",
        "seed4 (Mode A overshoot route)":  PROBE / "seed4_lcp_residual.log",
    }
    results = {}
    for label, p in paths.items():
        results[label] = summarize(label, p)

    # Direct comparison across seeds (etiology test)
    print()
    print("=== Cross-seed comparison (etiology test) ===")
    for label, r in results.items():
        if r is None:
            continue
        rv = r["res_viol"]
        rb = r["res_bounded_contact"]
        if rv and rb:
            med_v = sorted(rv)[len(rv) // 2]
            med_b = sorted(rb)[len(rb) // 2]
            ratio = med_v / med_b if med_b else float('inf')
            print(f"  {label}")
            print(f"    median residual: violation={med_v:.2e}  bounded-contact={med_b:.2e}  ratio={ratio:.2f}×")
        elif not rv:
            print(f"  {label}: no violations in this re-run (regression non-deterministic?)")


if __name__ == "__main__":
    main()
