#!/usr/bin/env python3
"""Aggregate parse_beta_contact_events.py output across a set of seeded
runs. Reads pre-saved [STEP]+[GATE-CONTACT] logs from
``audit_output/seed_sweep/<config>_s<n>.txt`` (or any explicit list of
paths) and emits a single table.

Output per (config, seed):
  bin verdict, n_events, contact_rate (events / c3-steps), best
  peak/final displacement, mean direction-fix (Drake F_on_box), mean
  recoil fraction.
"""
from __future__ import annotations
import sys
import importlib.util
import pathlib

_HERE = pathlib.Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "_pbce", _HERE / "parse_beta_contact_events.py")
pbce = importlib.util.module_from_spec(_SPEC)
sys.modules["_pbce"] = pbce       # dataclasses lookup needs us in sys.modules
_SPEC.loader.exec_module(pbce)


def summarize(path: str) -> dict:
    rows, goal_xy = pbce.parse_log(path)
    events = pbce.group_contact_events(rows)
    metrics = [pbce.event_metrics(e, goal_xy) for e in events]
    n_step = len(rows)
    n_c3 = sum(1 for r in rows if r.mode == "c3")
    n_contact = sum(1 for r in rows if r.mode == "c3" and r.contact == "Y")
    has_contact = any(m["peak_disp_mm"] != 0 or m["n_steps"] > 0 for m in metrics)
    best_peak = max((m["peak_disp_mm"] for m in metrics), default=0.0)
    best_final = max((m["final_disp_mm"] for m in metrics), default=0.0)
    # Means weighted by event step count.
    total = sum(m["n_steps"] for m in metrics) if metrics else 1
    def _wmean(field):
        if not metrics: return float("nan")
        s = 0.0
        ws = 0.0
        for m in metrics:
            v = m[field]
            w = m["n_steps"]
            if v == v:  # not NaN
                s += v * w
                ws += w
        return s / ws if ws else float("nan")
    mean_fcmd_ratio = _wmean("fcmd_ratio_mean")
    mean_fbox_ratio = _wmean("fbox_ratio_mean")
    mean_recoil    = _wmean("recoil_frac")
    mean_fbox_mag  = _wmean("fbox_mag_mean")
    verdict = pbce.score(metrics)
    return dict(
        path           = path,
        n_step         = n_step,
        n_c3           = n_c3,
        n_contact_step = n_contact,
        n_events       = len(events),
        best_peak_mm   = best_peak,
        best_final_mm  = best_final,
        mean_fcmd_ratio = mean_fcmd_ratio,
        mean_fbox_ratio = mean_fbox_ratio,
        mean_recoil    = mean_recoil,
        mean_fbox_mag  = mean_fbox_mag,
        verdict        = verdict,
    )


def main():
    if len(sys.argv) < 2:
        print("usage: summarize_seed_sweep.py <log1> [<log2> ...]",
              file=sys.stderr)
        sys.exit(2)
    print(f"{'path':45s}  {'#step':>5s}  {'#c3':>4s}  {'#cont':>5s}  "
          f"{'#evt':>4s}  {'peak_mm':>8s}  {'final_mm':>8s}  "
          f"{'fcmd':>6s}  {'fbox':>6s}  {'recoil':>7s}  {'|F|':>6s}  verdict")
    for p in sys.argv[1:]:
        try:
            s = summarize(p)
        except FileNotFoundError:
            print(f"{p:45s}  MISSING")
            continue
        print(f"{p:45s}  {s['n_step']:5d}  {s['n_c3']:4d}  "
              f"{s['n_contact_step']:5d}  {s['n_events']:4d}  "
              f"{s['best_peak_mm']:+8.3f}  {s['best_final_mm']:+8.3f}  "
              f"{s['mean_fcmd_ratio']:+6.3f}  {s['mean_fbox_ratio']:+6.3f}  "
              f"{s['mean_recoil']:+7.3f}  {s['mean_fbox_mag']:6.3f}  "
              f"{s['verdict']}")


if __name__ == "__main__":
    main()
