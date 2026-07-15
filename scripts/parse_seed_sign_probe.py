#!/usr/bin/env python3
"""Parse seed-sign probe logs and emit the per-seed table + IC-vs-outcome
cross-tab. See scripts/run_seed_sign_probe.sh for the probe spec.

Per-run extraction:
  initial_EE_y      : ee_p.y at first [GATE-CONTACT] step (step=1)
  first_sample_y    : strat_0 sample.y at first [GS-table] dump (step=0,
                      via "k=N (strat_0   ) pos=(x,y,z)") — the
                      seed-RNG-influenced IC PROXY (literal initial-EE-y
                      is deterministic across seeds because --seed only
                      seeds SamplingC3MPC._rng, not env init)
  contact_ticks     : count of [GATE-CONTACT] lines with A_is_ee=1
                      (Drake-realized EE-box contact, not LCS admit)
  EE_y_avg_contact  : mean ee_p.y over contact ticks (or initial if none)
  delta_box_y       : box_p.y at last [GATE-CONTACT] minus initial box_p.y
                      (initial box_p.y is 0.0 by tasks.yaml init_xyz)
  magnitude         : abs(delta_box_y) in mm

Verdict logic (printed at the end):
  Cross-tab rows = sign(first_sample_y); cols = sign(delta_box_y).
  If column collapses to one sign across all IC-sign rows: always-southy
  (verdict ii). If correlates: verdict (i). Else: verdict (iii).
"""
from __future__ import annotations
import argparse
import re
import sys
import pathlib
from dataclasses import dataclass
from collections import Counter

GATE_RE = re.compile(
    r"\[GATE-CONTACT\] step=(\d+) .*?"
    r"A_is_ee=(\d+) .*?"
    r"box_p=\(([+\-\d.]+),([+\-\d.]+),([+\-\d.]+)\) "
    r"ee_p=\(([+\-\d.]+),([+\-\d.]+),([+\-\d.]+)\)"
)
GS_TABLE_HEADER_RE = re.compile(r"\[GS-table\] step=(\d+)")
GS_TABLE_ROW_RE = re.compile(
    r"k=(\d+) \((\S+?)\s*\) pos=\(([+\-\d.]+),([+\-\d.]+),([+\-\d.]+)\)"
)
RESULT_RE = re.compile(
    r"\[RESULT\].*?final_obj_xy=\(([+\-\d.]+), ([+\-\d.]+)\)"
)
SAMPLING_SEED_RE = re.compile(r"\[OVERRIDE\] seed=(\d+)")


def sgn(x: float, eps: float = 1e-6) -> str:
    if abs(x) < eps:
        return "0"
    return "+" if x > 0 else "-"


@dataclass
class RunMetrics:
    path: str
    seed: int
    arm: str
    rc_ok: bool
    initial_ee_y: float
    initial_ee_y_sign: str
    first_strat0_y: float | None
    first_strat0_y_sign: str
    n_contact_ticks: int
    ee_y_avg_contact: float
    delta_box_y: float
    delta_box_y_mag_mm: float
    delta_box_y_sign: str
    final_obj_xy: tuple[float, float] | None


def parse_log(path: pathlib.Path) -> RunMetrics:
    seed = None
    arm = None
    # Filename: seed{N}_{arm}.log
    m = re.match(r"seed(\d+)_(off|alpha05)\.log", path.name)
    if m:
        seed = int(m.group(1))
        arm  = m.group(2)
    text = path.read_text(errors="replace")

    # First [GATE-CONTACT] row → initial EE-y (proxy for IC, deterministic
    # across seeds — see module docstring).
    first_gate = None
    last_gate  = None
    contact_ee_ys = []
    n_contact = 0
    initial_box_y = None
    for line in text.splitlines():
        gm = GATE_RE.search(line)
        if not gm:
            continue
        step = int(gm.group(1))
        a_is_ee = int(gm.group(2))
        box_y   = float(gm.group(4))
        ee_y    = float(gm.group(7))
        if first_gate is None:
            first_gate = (step, ee_y, box_y)
            initial_box_y = box_y
        last_gate = (step, ee_y, box_y)
        if a_is_ee == 1:
            n_contact += 1
            contact_ee_ys.append(ee_y)

    # First [GS-table] dump (step=0) → strat_0 y-coordinate (seed-perturbed IC).
    first_strat0_y = None
    in_table = False
    for line in text.splitlines():
        hm = GS_TABLE_HEADER_RE.search(line)
        if hm:
            if first_strat0_y is not None:
                break  # already grabbed the first table; stop
            in_table = True
            continue
        if in_table:
            rm = GS_TABLE_ROW_RE.search(line)
            if rm:
                label = rm.group(2)
                if label == "strat_0":
                    first_strat0_y = float(rm.group(4))
            elif line.strip() == "" or line.startswith("["):
                in_table = False  # left the table block

    rm = RESULT_RE.search(text)
    final_obj = (float(rm.group(1)), float(rm.group(2))) if rm else None

    if first_gate is None:
        # Run crashed before first step. Mark as missing.
        return RunMetrics(
            path=str(path), seed=seed if seed is not None else -1, arm=arm or "?",
            rc_ok=False,
            initial_ee_y=float("nan"), initial_ee_y_sign="?",
            first_strat0_y=None, first_strat0_y_sign="?",
            n_contact_ticks=0, ee_y_avg_contact=float("nan"),
            delta_box_y=float("nan"), delta_box_y_mag_mm=float("nan"),
            delta_box_y_sign="?", final_obj_xy=final_obj,
        )

    initial_ee_y = first_gate[1]
    ee_y_avg = (sum(contact_ee_ys) / len(contact_ee_ys)
                if contact_ee_ys else initial_ee_y)
    # Use final box-y from [RESULT] if available (more authoritative than
    # last [GATE-CONTACT], which is one tick earlier).
    if final_obj is not None:
        final_box_y = final_obj[1]
    else:
        final_box_y = last_gate[2]
    delta_box_y = final_box_y - (initial_box_y or 0.0)

    return RunMetrics(
        path=str(path), seed=seed if seed is not None else -1, arm=arm or "?",
        rc_ok=True,
        initial_ee_y=initial_ee_y,
        initial_ee_y_sign=sgn(initial_ee_y),
        first_strat0_y=first_strat0_y,
        first_strat0_y_sign=(sgn(first_strat0_y) if first_strat0_y is not None
                             else "?"),
        n_contact_ticks=n_contact,
        ee_y_avg_contact=ee_y_avg,
        delta_box_y=delta_box_y,
        delta_box_y_mag_mm=abs(delta_box_y) * 1000.0,
        delta_box_y_sign=sgn(delta_box_y),
        final_obj_xy=final_obj,
    )


def emit_table(metrics: list[RunMetrics]) -> None:
    print()
    print("=== PER-RUN TABLE ===")
    hdr = (f"{'seed':>4s} {'arm':>7s}  "
           f"{'sgn(EE0y)':>9s}  {'EE0_y':>10s}  "
           f"{'sgn(s0y)':>8s}  {'strat0_y':>10s}  "
           f"{'sgn(EE.y_c)':>11s}  {'sgn(dBy)':>8s}  "
           f"{'|dBy|mm':>8s}  {'n_cont':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for m in sorted(metrics, key=lambda r: (r.arm, r.seed)):
        ee0 = f"{m.initial_ee_y:+.5f}" if m.initial_ee_y == m.initial_ee_y else "NaN"
        s0  = (f"{m.first_strat0_y:+.5f}" if m.first_strat0_y is not None
               else "  -    ")
        eyc = sgn(m.ee_y_avg_contact) if m.n_contact_ticks > 0 else "?"
        print(f"{m.seed:4d} {m.arm:>7s}  "
              f"{m.initial_ee_y_sign:>9s}  {ee0:>10s}  "
              f"{m.first_strat0_y_sign:>8s}  {s0:>10s}  "
              f"{eyc:>11s}  {m.delta_box_y_sign:>8s}  "
              f"{m.delta_box_y_mag_mm:8.3f}  {m.n_contact_ticks:6d}")


def emit_crosstab(metrics: list[RunMetrics], ic_field: str, label: str) -> None:
    """Cross-tab IC sign (rows) vs delta_box_y sign (cols)."""
    print()
    print(f"=== CROSS-TAB: {label} (rows) × sgn(delta_box_y) (cols) ===")
    rows = {}
    for m in metrics:
        if not m.rc_ok:
            continue
        ic = getattr(m, ic_field)
        out = m.delta_box_y_sign
        rows.setdefault((m.arm, ic), Counter())[out] += 1
    arms = sorted({a for a, _ in rows})
    for arm in arms:
        print(f"  [{arm}]")
        print(f"    {'IC':>4s}  {'+':>3s}  {'-':>3s}  {'0':>3s}  total")
        for ic in ['+', '-', '0', '?']:
            row = rows.get((arm, ic))
            if row is None:
                continue
            tot = sum(row.values())
            print(f"    {ic:>4s}  {row.get('+',0):3d}  {row.get('-',0):3d}  "
                  f"{row.get('0',0):3d}  {tot:3d}")


def emit_verdict(metrics: list[RunMetrics]) -> None:
    print()
    print("=== VERDICT (per arm, on the SEED-VARYING proxy IC = sgn(strat0_y)) ===")
    for arm in sorted({m.arm for m in metrics if m.rc_ok}):
        outs   = [m.delta_box_y_sign       for m in metrics if m.rc_ok and m.arm == arm]
        ics    = [m.first_strat0_y_sign    for m in metrics if m.rc_ok and m.arm == arm]
        n      = len(outs)
        ne_outs = [o for o in outs if o != '0']
        n_nonzero = len(ne_outs)
        if n_nonzero == 0:
            print(f"  [{arm}]: all delta_box_y ~ 0 — no measurable slide; verdict UNDETERMINED")
            continue
        pos = sum(1 for o in ne_outs if o == '+')
        neg = sum(1 for o in ne_outs if o == '-')
        # Skip degenerate seeds (IC '?' or '0')
        paired = [(ic, o) for ic, o in zip(ics, outs)
                  if ic in ('+', '-') and o in ('+', '-')]
        if not paired:
            print(f"  [{arm}]: no informative pairs; pos={pos} neg={neg} of {n}")
            continue
        # Correlation count: IC sign == out sign
        same = sum(1 for ic, o in paired if ic == o)
        diff = sum(1 for ic, o in paired if ic != o)
        n_pair = len(paired)
        # Verdict heuristic:
        #   - if one out-sign dominates (≥0.9 of nonzero): (ii) always-south/north
        #   - else if same/diff strongly correlates (≥0.8 either way): (i) IC-driven
        #   - else: (iii) random / non-convergence symptom
        dom = max(pos, neg) / n_nonzero
        corr = max(same, diff) / n_pair
        if dom >= 0.9:
            dominant = '+' if pos > neg else '-'
            verd = f"(ii) ALWAYS-{dominant} ({pos}+/{neg}- of {n_nonzero}) — hidden asymmetry channel"
        elif corr >= 0.8:
            sense = "correlated" if same > diff else "anti-correlated"
            verd = (f"(i) IC-DRIVEN {sense} (same={same}, diff={diff} of {n_pair}) — "
                    "make planner robust to IC sign")
        else:
            verd = (f"(iii) UNCORRELATED (same={same}, diff={diff} of {n_pair}; "
                    f"pos={pos} neg={neg} of {n_nonzero}) — "
                    "solver-nondeterminism / non-convergence symptom; "
                    "connect to ADMM 25/25 wall")
        print(f"  [{arm}]: {verd}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="probe output directory")
    ap.add_argument("--min-contact", type=int, default=0,
                    help="Restrict cross-tab + verdict to runs with "
                         "n_contact_ticks >= N (signal-bearing seeds only). "
                         "Default 0 = no filter (original behavior).")
    args = ap.parse_args()
    d = pathlib.Path(args.dir)
    logs = sorted(d.glob("seed*_*.log"))
    if not logs:
        print(f"No seed*_*.log files in {d}", file=sys.stderr)
        sys.exit(2)
    metrics = [parse_log(p) for p in logs]
    emit_table(metrics)

    if args.min_contact > 0:
        thr = args.min_contact
        cleared = [m for m in metrics if m.rc_ok and m.n_contact_ticks >= thr]
        print()
        print(f"=== CONTACT-FILTER: n_contact_ticks >= {thr} ===")
        per_arm = Counter(m.arm for m in cleared)
        per_arm_total = Counter(m.arm for m in metrics if m.rc_ok)
        for arm in sorted(per_arm_total):
            print(f"  [{arm}] cleared {per_arm.get(arm,0):2d} / {per_arm_total[arm]:2d} seeds")
        if not cleared:
            print(f"  NO SIGNAL — zero seeds cleared n_contact >= {thr} across all arms.")
            print(f"  Verdict UNANSWERABLE from this sweep; sign-question requires")
            print(f"  reliable contact formation first.")
            return
        per_arm_cleared_seeds = {}
        for m in cleared:
            per_arm_cleared_seeds.setdefault(m.arm, []).append(m.seed)
        for arm in sorted(per_arm_cleared_seeds):
            print(f"  [{arm}] cleared seeds: {sorted(per_arm_cleared_seeds[arm])}")
        metrics_for_tab = cleared
        suffix = f" [FILTERED n_cont >= {thr}]"
    else:
        metrics_for_tab = metrics
        suffix = ""

    emit_crosstab(metrics_for_tab, "initial_ee_y_sign",
                  "sgn(initial_EE_y)  [literal IC — expected constant across seeds]"
                  + suffix)
    emit_crosstab(metrics_for_tab, "first_strat0_y_sign",
                  "sgn(first_strat0_y)  [seed-varying IC PROXY — load-bearing]"
                  + suffix)
    emit_verdict(metrics_for_tab)


if __name__ == "__main__":
    main()
