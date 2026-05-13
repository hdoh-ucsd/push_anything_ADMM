"""F3 Phase 2 — per-log metric extractor.

For each input log, computes:
  M1 — strat_0 generation rate: fraction of [GS-table] blocks containing
       at least one row whose label starts with 'strat_'.
  M2 — strat_0 win-rate when generated: among blocks with strat_*, fraction
       where the winning row (← WIN) has label starting 'strat_'.
  M3 — final obj_xy, distance-to-goal, direction summary, switches, watchdog.

Usage:
    python scripts/probe_f3_p2_metrics.py results/f3_p2_*_stdout.log
"""
from __future__ import annotations

import re
import sys
import math
from pathlib import Path


_re_table  = re.compile(r"^\[GS-table\]\s+step=(?P<step>\d+)\s*$")
_re_row    = re.compile(
    r"^\s*k=(?P<k>\d+)\s+\((?P<label>[^)]+?)\s*\)\s+"
    r"pos=\([^)]*\)\s+"
    r"c_C3=\s*(?P<c_C3>-?[\d.]+)\s+"
    r"align=(?P<align_score>[\d.]+)\(bonus=\s*(?P<align_bonus>-?[\d.]+)\)\s+"
    r"travel=(?P<travel>[\d.]+)m\(pen=\s*(?P<travel_pen>-?[\d.]+)\)\s+"
    r"c_sample=\s*(?P<c_sample>-?[\d.]+)\s+"
    r"feas=(?P<feas>[YN])\s+"
    r"ik_err=(?P<ik_err>[\d.]+)m"
    r"(?P<win>\s+←\s+WIN)?",
)
_re_result   = re.compile(
    r"\[RESULT\]\s+method=\S+\s+final_obj_xy=\((?P<x>-?[\d.]+),\s*(?P<y>-?[\d.]+)\)\s+"
    r"goal_dist=(?P<d>[\d.]+)m\s+success=(?P<succ>YES|NO)"
)
_re_perf     = re.compile(
    r"\[GS-perf\]\s+avg_per_step_ms=(?P<ms>[\d.]+)\s+full_solves=(?P<full>\d+)\s+"
    r"cheap_solves=(?P<cheap>\d+)\s+switches=(?P<sw>\d+)"
)
_re_watchdog = re.compile(r"^\[GS-watchdog-summary\].*")
_re_override = re.compile(r"^\[OVERRIDE\]\s+workspace_xy_max\[1\]=(?P<v>-?[\d.]+)")
_re_goal     = re.compile(r"goal_xy:\s*\[\s*(?P<gx>-?[\d.]+),\s*(?P<gy>-?[\d.]+)\s*\]")


def parse_blocks(lines):
    """Yield (rows) lists per [GS-table] block."""
    cur = None
    for ln in lines:
        if _re_table.match(ln):
            if cur is not None:
                yield cur
            cur = []
            continue
        if cur is not None:
            m = _re_row.match(ln)
            if m:
                cur.append({
                    "label":  m.group("label").strip(),
                    "winner": bool(m.group("win")),
                })
            elif ln.startswith("[") or ln.startswith("  t="):
                if cur:
                    yield cur
                cur = None
    if cur:
        yield cur


def direction(x, y, gx, gy):
    """Categorize obj displacement vs goal."""
    # Obj starts at (0, 0). Goal at (gx, gy). Displacement = (x, y).
    # Project onto goal direction; if positive and small fraction → toward.
    gn = math.hypot(gx, gy)
    if gn < 1e-6:
        return "no-goal"
    along = (x * gx + y * gy) / gn
    perp  = (x * gy - y * gx) / gn  # signed lateral
    if along > 0.005:
        # Moved toward
        if abs(perp) < abs(along) * 0.5:
            return f"toward+goal({along:+.3f}m)"
        return f"diagonal+goal({along:+.3f},{perp:+.3f})"
    if abs(x) < 0.01 and abs(y) < 0.01:
        return "no-motion"
    # Did not move toward goal at all
    if abs(x) > abs(y):
        d = "+x" if x > 0 else "-x"
    else:
        d = "+y" if y > 0 else "-y"
    return f"wrong/{d}({x:+.3f},{y:+.3f})"


def analyze(path: Path):
    text = path.read_text(errors="replace")
    lines = text.splitlines()

    blocks = list(parse_blocks(lines))
    n_blocks = len(blocks)
    n_with_strat = 0
    n_strat_wins = 0
    for rows in blocks:
        has_strat = any(r["label"].startswith("strat_") for r in rows)
        if has_strat:
            n_with_strat += 1
            for r in rows:
                if r["winner"] and r["label"].startswith("strat_"):
                    n_strat_wins += 1
                    break

    m1 = (n_with_strat / n_blocks) if n_blocks else 0.0
    m2 = (n_strat_wins / n_with_strat) if n_with_strat else float("nan")

    # Final pose / switches
    final_x, final_y, final_d, succ = None, None, None, None
    sw = full = cheap = ms = None
    watchdog_line = None
    override_v = None
    goal = None
    for ln in lines:
        m = _re_result.search(ln)
        if m:
            final_x, final_y, final_d = float(m["x"]), float(m["y"]), float(m["d"])
            succ = m["succ"]
        m = _re_perf.search(ln)
        if m:
            ms   = float(m["ms"])
            full = int(m["full"])
            cheap = int(m["cheap"])
            sw    = int(m["sw"])
        if _re_watchdog.match(ln):
            watchdog_line = ln.strip()
        m = _re_override.match(ln)
        if m:
            override_v = float(m["v"])
        m = _re_goal.search(ln)
        if m and goal is None:
            goal = (float(m["gx"]), float(m["gy"]))

    if goal is None:
        # Pushing default
        goal = (0.30, 0.0)

    direction_str = direction(final_x or 0.0, final_y or 0.0, *goal)

    return {
        "name":       path.stem,
        "blocks":     n_blocks,
        "with_strat": n_with_strat,
        "strat_wins": n_strat_wins,
        "m1":         m1,
        "m2":         m2,
        "final_xy":   (final_x, final_y),
        "goal_dist":  final_d,
        "success":    succ,
        "direction":  direction_str,
        "switches":   sw,
        "full":       full,
        "watchdog":   watchdog_line,
        "override":   override_v,
        "goal":       goal,
    }


def fmt(r):
    fx, fy = r["final_xy"]
    fxs = f"{fx:+.3f}" if fx is not None else "?"
    fys = f"{fy:+.3f}" if fy is not None else "?"
    ds  = f"{r['goal_dist']:.3f}" if r['goal_dist'] is not None else "?"
    m1s = f"{r['m1']*100:5.1f}% ({r['with_strat']}/{r['blocks']})"
    if r['with_strat']:
        m2s = f"{r['m2']*100:5.1f}% ({r['strat_wins']}/{r['with_strat']})"
    else:
        m2s = "n/a (0 gen)"
    sws = str(r['switches']) if r['switches'] is not None else "?"
    return (f"{r['name']:50s} | M1={m1s:18s} | M2={m2s:18s} | "
            f"obj=({fxs},{fys}) d={ds} sw={sws} dir={r['direction']}")


def main():
    paths = [Path(p) for p in sys.argv[1:]]
    results = [analyze(p) for p in paths]
    for r in results:
        print(fmt(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
