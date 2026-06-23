"""Parse a Stage A run.log and emit metrics for the bar evaluation.

Parser input is the purpose-built ``[STAGE-A-TRACE]`` per-tick line in
control/sampling_c3/sampling_based_c3_controller.py (added in the
2026-06-23 pre-flight emit check), plus the existing ``[ENTRY-GATE]``
log (already emitted by the dispatcher) and the flag-ON-only
``[STAGE-A-PWL] step=N sim_t=T build ...`` lines (added by Task 4).

Usage:
    python3 scripts/_parse_stage_a_metrics.py path/to/run.log [--out metrics.json]

Metrics produced (see docs/superpowers/plans/2026-06-23-stage-a-...):

    Windowed landing (Refinement 1):
      first_c3_entry_step, first_c3_episode_len_ticks, window_ticks_used,
      window_status, landing_series_m, landing_median_m,
      landing_fraction_within_2mm

    Admit:
      c3_steps_total, c3_steps_with_admitted_ee_box, admit_rate

    Entry-gate:
      entry_gate_candidate_transitions, entry_gate_firings,
      entry_gate_firing_rate

    Orientation:
      max_abs_qy, max_abs_qz

    Goal motion (informational):
      goal_motion_m

    Rebuild-churn (Refinement 3, Stage A only — zero on baseline):
      pwl_rebuilds_total, free_mode_seconds_total, rebuild_rate_hz,
      rebuild_churn_flagged
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Optional


# Single per-tick trace line carrying everything we need.
_RE_TRACE = re.compile(
    r"^\[STAGE-A-TRACE\] step=(?P<step>\d+) "
    r"sim_t=(?P<sim_t>[+\-\d.eE]+) "
    r"mode=(?P<mode>c3|free) "
    r"phi=(?P<phi>[+\-\d.eE]+|nan) "
    r"box_xy=(?P<bx>[+\-\d.eE]+),(?P<by>[+\-\d.eE]+) "
    r"lam_n_ee_box=(?P<lam>[+\-\d.eE]+|nan) "
    r"qy=(?P<qy>[+\-\d.eE]+) "
    r"qz=(?P<qz>[+\-\d.eE]+) "
    r"finished_repos=(?P<fin>[01])"
)
_RE_ENTRY_GATE = re.compile(r"^\[ENTRY-GATE\] step=\d+")
_RE_PWL_BUILD = re.compile(r"^\[STAGE-A-PWL\] step=\d+ sim_t=[+\-\d.eE]+ build")


LANDING_THRESHOLD_M = 0.002       # 2 mm (cond 1)
WINDOW_TICKS_MAX_DEFAULT = 100    # 1.0 s at dt_ctrl=0.01
WINDOW_TICKS_MIN_DEFAULT = 50     # 0.5 s
REBUILD_RATE_HZ_BAR = 1.0         # Stage A churn flag


def _to_float(s: str) -> float:
    s = s.strip().lower()
    if s in ("nan", "+nan", "-nan"):
        return float("nan")
    return float(s)


def parse_run_log(path: Path,
                  window_ticks_max: int = WINDOW_TICKS_MAX_DEFAULT,
                  window_ticks_min: int = WINDOW_TICKS_MIN_DEFAULT,
                  dt_ctrl: float = 0.01) -> dict:
    """Parse a Stage A run.log into a metrics dict."""
    path = Path(path)
    lines = path.read_text().splitlines()

    # Per-step records keyed by integer step.
    step_mode: dict[int, str] = {}
    step_phi: dict[int, float] = {}
    step_lam_n: dict[int, float] = {}
    step_finished_repos: dict[int, bool] = {}
    box_xy_first: Optional[tuple[float, float]] = None
    box_xy_last: Optional[tuple[float, float]] = None
    max_abs_qy = 0.0
    max_abs_qz = 0.0
    entry_gate_firings = 0
    pwl_rebuilds_total = 0

    for ln in lines:
        m = _RE_TRACE.match(ln)
        if m:
            s = int(m["step"])
            step_mode[s] = m["mode"]
            step_phi[s] = _to_float(m["phi"])
            step_lam_n[s] = _to_float(m["lam"])
            step_finished_repos[s] = (m["fin"] == "1")
            bx = float(m["bx"]); by = float(m["by"])
            if box_xy_first is None:
                box_xy_first = (bx, by)
            box_xy_last = (bx, by)
            qy = abs(float(m["qy"])); qz = abs(float(m["qz"]))
            if qy > max_abs_qy: max_abs_qy = qy
            if qz > max_abs_qz: max_abs_qz = qz
            continue
        if _RE_ENTRY_GATE.match(ln):
            entry_gate_firings += 1
            continue
        if _RE_PWL_BUILD.match(ln):
            pwl_rebuilds_total += 1
            continue

    # --- First c3 entry + first-c3-episode length ---
    sorted_steps = sorted(step_mode.keys())
    first_c3_entry: Optional[int] = None
    prev = None
    for s in sorted_steps:
        if prev == "free" and step_mode[s] == "c3":
            first_c3_entry = s
            break
        prev = step_mode[s]

    first_c3_episode_len = 0
    if first_c3_entry is not None:
        idx = sorted_steps.index(first_c3_entry)
        while (idx < len(sorted_steps)
               and step_mode[sorted_steps[idx]] == "c3"):
            first_c3_episode_len += 1
            idx += 1

    window_ticks_used = min(window_ticks_max, first_c3_episode_len)
    if first_c3_entry is None:
        window_status = "NO_C3_ENTRY"
    elif window_ticks_used < window_ticks_min:
        window_status = "INSUFFICIENT_WINDOW"
    else:
        window_status = "OK"

    # --- Windowed landing series (phi is already surface-distance) ---
    landing_series_m: list[float] = []
    if first_c3_entry is not None and window_ticks_used > 0:
        idx = sorted_steps.index(first_c3_entry)
        window_steps = sorted_steps[idx: idx + window_ticks_used]
        for s in window_steps:
            phi = step_phi.get(s, float("nan"))
            if not math.isnan(phi):
                landing_series_m.append(float(phi))

    if landing_series_m:
        landing_median_m = float(statistics.median(landing_series_m))
        landing_fraction_within_2mm = float(
            sum(1 for d in landing_series_m
                if d <= LANDING_THRESHOLD_M) / len(landing_series_m)
        )
    else:
        landing_median_m = None
        landing_fraction_within_2mm = None

    # --- Admit rate (across entire 12 s run) ---
    c3_steps = [s for s, m in step_mode.items() if m == "c3"]
    c3_total = len(c3_steps)
    c3_admitted = sum(
        1 for s in c3_steps
        if not math.isnan(step_lam_n.get(s, float("nan")))
        and step_lam_n[s] > 0.0
    )
    admit_rate = (c3_admitted / c3_total) if c3_total > 0 else 0.0

    # --- Entry-gate candidate transitions + firing rate ---
    # Candidate: free-mode tick with finished_repos=1.
    cand = sum(
        1 for s in sorted_steps
        if step_mode[s] == "free" and step_finished_repos.get(s, False)
    )
    gate_rate = (entry_gate_firings / cand) if cand > 0 else 0.0

    # --- Goal motion (informational; A→E cumulative) ---
    if box_xy_first is not None and box_xy_last is not None:
        dx = box_xy_last[0] - box_xy_first[0]
        dy = box_xy_last[1] - box_xy_first[1]
        goal_motion_m = float((dx * dx + dy * dy) ** 0.5)
    else:
        goal_motion_m = None

    # --- Rebuild-churn (Refinement 3, Stage A only) ---
    free_ticks_total = sum(1 for s, m in step_mode.items() if m == "free")
    free_mode_seconds_total = float(free_ticks_total) * float(dt_ctrl)
    if free_mode_seconds_total > 0.0:
        rebuild_rate_hz = float(pwl_rebuilds_total) / free_mode_seconds_total
    else:
        rebuild_rate_hz = 0.0
    rebuild_churn_flagged = bool(rebuild_rate_hz >= REBUILD_RATE_HZ_BAR)

    return dict(
        first_c3_entry_step=first_c3_entry,
        first_c3_episode_len_ticks=first_c3_episode_len,
        window_ticks_used=window_ticks_used,
        window_status=window_status,
        landing_series_m=landing_series_m,
        landing_median_m=landing_median_m,
        landing_fraction_within_2mm=landing_fraction_within_2mm,
        c3_steps_total=c3_total,
        c3_steps_with_admitted_ee_box=c3_admitted,
        admit_rate=admit_rate,
        entry_gate_candidate_transitions=cand,
        entry_gate_firings=entry_gate_firings,
        entry_gate_firing_rate=gate_rate,
        max_abs_qy=max_abs_qy,
        max_abs_qz=max_abs_qz,
        goal_motion_m=goal_motion_m,
        pwl_rebuilds_total=pwl_rebuilds_total,
        free_mode_seconds_total=free_mode_seconds_total,
        rebuild_rate_hz=rebuild_rate_hz,
        rebuild_churn_flagged=rebuild_churn_flagged,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--window-ticks-max", type=int,
                    default=WINDOW_TICKS_MAX_DEFAULT)
    ap.add_argument("--window-ticks-min", type=int,
                    default=WINDOW_TICKS_MIN_DEFAULT)
    ap.add_argument("--dt-ctrl", type=float, default=0.01)
    args = ap.parse_args()
    metrics = parse_run_log(
        args.log,
        window_ticks_max=args.window_ticks_max,
        window_ticks_min=args.window_ticks_min,
        dt_ctrl=args.dt_ctrl,
    )
    out = args.out or args.log.with_suffix(".metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(f"[parse_stage_a_metrics] wrote {out}")
    summary = dict(metrics)
    if isinstance(summary.get("landing_series_m"), list):
        summary["landing_series_m"] = (
            f"<{len(summary['landing_series_m'])} samples>"
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
