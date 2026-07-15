#!/usr/bin/env python3
"""Aggregate parse_beta_contact_events.py output across a paired seed sweep.

Reads results/beta_seed_sweep_<TS>/stepc_seed{N}_{beta_on|beta_off}.log,
runs parse_beta_contact_events.py on each, and prints a side-by-side
table: per seed the verdict bin + best-event direction-fix ratio + best
recoil fraction, with β on vs off in adjacent rows.

Usage:
  python scripts/aggregate_beta_sweep.py results/beta_seed_sweep_<TS>/
"""
from __future__ import annotations
import re
import sys
import subprocess
from pathlib import Path


VERDICT_RE = re.compile(r"^VERDICT:\s*(PASS|PARTIAL|FAIL)\b")
PEAK_RE    = re.compile(r"peak_disp_along_goal = ([+\-\d.]+) mm")
FINAL_RE   = re.compile(r"final_disp_along_goal .* = ([+\-\d.]+) mm")
RECOIL_RE  = re.compile(r"recoil_frac .* = ([+\-\d.nan]+)")
FCMD_RE    = re.compile(r"direction-fix ratio  F_goal/\|F\| \(f_cmd\)  = ([+\-\d.nan]+)")
FDRAKE_RE  = re.compile(r"direction-fix ratio  F_goal/\|F\| \(F_drake mean\) = "
                        r"([+\-\d.nan]+)  \(peak ([+\-\d.nan]+)\)")
NEVENTS_RE = re.compile(r"n_contact_events=(\d+)")


def analyze(log_path: Path) -> dict:
    """Run parser, scrape best per-event numbers + verdict."""
    out = subprocess.run(
        ["python", "scripts/parse_beta_contact_events.py", str(log_path)],
        capture_output=True, text=True,
    )
    text = out.stdout

    n_events = 0
    if (m := NEVENTS_RE.search(text)):
        n_events = int(m.group(1))

    verdict = "FAIL"
    if (m := VERDICT_RE.search(text, re.MULTILINE)):
        verdict = m.group(1)

    # Track best (largest |peak| and |final|) across events.
    peaks   = [float(x) for x in PEAK_RE.findall(text)]
    finals  = [float(x) for x in FINAL_RE.findall(text)]
    recoils = [float(x) for x in RECOIL_RE.findall(text)
               if x not in ("nan", "+nan", "-nan")]
    fdrakes = [float(g[0]) for g in FDRAKE_RE.findall(text)
               if g[0] not in ("nan", "+nan", "-nan")]

    return dict(
        log         = log_path.name,
        n_events    = n_events,
        verdict     = verdict,
        best_peak_mm  = max((abs(x) for x in peaks),  default=float("nan")),
        best_final_mm = max((abs(x) for x in finals), default=float("nan")),
        worst_recoil  = max(recoils, default=float("nan")),
        best_fdrake_dir = max(fdrakes, default=float("nan")),
    )


def main():
    if len(sys.argv) < 2:
        print("usage: aggregate_beta_sweep.py <sweep-dir>", file=sys.stderr)
        sys.exit(2)
    root = Path(sys.argv[1])
    logs = sorted(root.glob("stepc_seed*_beta_*.log"))
    if not logs:
        print(f"no logs in {root}", file=sys.stderr)
        sys.exit(1)
    rows = [analyze(p) for p in logs]

    print(f"=== {root} — paired β sweep ===")
    print()
    print(f"{'log':40s}  {'n_ev':>4s}  {'verdict':8s}  "
          f"{'peak_mm':>8s}  {'final_mm':>8s}  {'recoil':>8s}  "
          f"{'dirfix':>8s}")
    print("-" * 96)
    for r in rows:
        def _f(x, p=3):
            return f"{x:>8.3f}" if x == x else f"{'nan':>8s}"
        print(f"{r['log']:40s}  {r['n_events']:>4d}  {r['verdict']:8s}  "
              f"{_f(r['best_peak_mm'])}  {_f(r['best_final_mm'])}  "
              f"{_f(r['worst_recoil'])}  {_f(r['best_fdrake_dir'])}")

    # Summary: contact-rate, PASS-rate per arm
    on  = [r for r in rows if "beta_on"  in r["log"]]
    off = [r for r in rows if "beta_off" in r["log"]]
    def _summary(arm, label):
        if not arm:
            return f"  {label}: no runs"
        n_contact = sum(1 for r in arm if r["n_events"] > 0)
        n_pass    = sum(1 for r in arm if r["verdict"] == "PASS")
        n_partial = sum(1 for r in arm if r["verdict"] == "PARTIAL")
        return (f"  {label}: seeds={len(arm)}  contact-formed={n_contact}/{len(arm)}  "
                f"PASS={n_pass}  PARTIAL={n_partial}  FAIL={len(arm)-n_pass-n_partial}")
    print()
    print(_summary(on,  "β on "))
    print(_summary(off, "β off"))


if __name__ == "__main__":
    main()
