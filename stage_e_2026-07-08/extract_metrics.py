#!/usr/bin/env python3
"""Stage E metric extractor.

Parses a run log and emits a JSON metrics record:
  goal_dist_final, goal_motion (0.30 - goal_dist_final)
  |qy|_max, |qz|_max (motion guards)
  n_c3_ticks, n_ee_box_admits (lam_n_ee_box > 0 while mode=c3)
  ee_box_admit_pct (over c3 ticks)
  free_mode_motion (box motion during free-mode ticks)
  c3_mode_motion   (box motion during c3-mode ticks)
  n_free_ticks, n_c3_ticks
  wall_seconds
"""
import json
import math
import re
import sys
from pathlib import Path


TRACE_RE = re.compile(
    r"\[STAGE-A-TRACE\] step=(\d+) sim_t=([\d.]+) mode=(\w+) phi=([-\d.]+) "
    r"box_xy=([+\-\d.]+),([+\-\d.]+) lam_n_ee_box=([-\d.enaN]+) "
    r"qy=([+\-\d.]+) qz=([+\-\d.]+) finished_repos=(\d)"
)
STEP_EE_RE = re.compile(r"\[STEP\] .* ee=\(([+\-\d.]+),([+\-\d.]+),([+\-\d.]+)\)")
STEP_GOAL_RE = re.compile(r"goal_dist=([\d.]+)m")
STEP_MODE_RE = re.compile(r"\[STEP\] step=\d+ mode=(\w+)")
ELAPSED_RE = re.compile(r"Elapsed \(wall clock\) time.*?:\s+([\d:.]+)")


def parse_hms(s):
    parts = s.split(":")
    if len(parts) == 3:
        h, m, sec = parts
        return int(h) * 3600 + int(m) * 60 + float(sec)
    if len(parts) == 2:
        m, sec = parts
        return int(m) * 60 + float(sec)
    return float(parts[0])


def extract(log_path: Path, time_path: Path | None = None) -> dict:
    qy_max = 0.0
    qz_max = 0.0
    n_free = 0
    n_c3 = 0
    n_ee_box_admit = 0
    prev_box_xy = None
    free_motion = 0.0
    c3_motion = 0.0
    lam_n_first = None
    lam_n_last = None
    ee_z_max = 0.0
    for line in log_path.read_text().splitlines():
        m_ee = STEP_EE_RE.search(line)
        if m_ee:
            ee_z_max = max(ee_z_max, float(m_ee.group(3)))
        m = TRACE_RE.search(line)
        if not m:
            continue
        mode = m.group(3)
        bx = float(m.group(5))
        by = float(m.group(6))
        lam_str = m.group(7)
        try:
            lam = float(lam_str)
        except ValueError:
            lam = float("nan")
        qy = float(m.group(8))
        qz = float(m.group(9))
        qy_max = max(qy_max, abs(qy))
        qz_max = max(qz_max, abs(qz))
        if mode == "free":
            n_free += 1
        elif mode == "c3":
            n_c3 += 1
            if not math.isnan(lam) and lam > 1e-6:
                n_ee_box_admit += 1
                if lam_n_first is None:
                    lam_n_first = lam
                lam_n_last = lam
        if prev_box_xy is not None:
            dx = bx - prev_box_xy[0]
            dy = by - prev_box_xy[1]
            step_motion = math.sqrt(dx * dx + dy * dy)
            if mode == "free":
                free_motion += step_motion
            elif mode == "c3":
                c3_motion += step_motion
        prev_box_xy = (bx, by)

    # final goal_dist from last [STEP] line
    goal_dist_final = None
    for line in reversed(log_path.read_text().splitlines()):
        m = STEP_GOAL_RE.search(line)
        if m:
            goal_dist_final = float(m.group(1))
            break

    goal_motion = None
    if goal_dist_final is not None:
        goal_motion = 0.300 - goal_dist_final  # initial distance is 0.30 m

    wall = None
    if time_path is not None and time_path.exists():
        for line in time_path.read_text().splitlines():
            m = ELAPSED_RE.search(line)
            if m:
                wall = parse_hms(m.group(1))
                break

    ee_box_admit_pct = None
    if n_c3 > 0:
        ee_box_admit_pct = 100.0 * n_ee_box_admit / n_c3

    # Bar per seed (Stage E cumulative): |qy|<0.10 AND |qz|<0.10 AND
    # EE-BOX >= 60% AND goal_motion >= 20 mm
    pass_qy = qy_max < 0.10
    pass_qz = qz_max < 0.10
    pass_admit = (ee_box_admit_pct is not None and ee_box_admit_pct >= 60.0)
    pass_motion = (goal_motion is not None and goal_motion >= 0.020)
    passed = pass_qy and pass_qz and pass_admit and pass_motion

    return {
        "log": str(log_path),
        "goal_dist_final_m": goal_dist_final,
        "goal_motion_m": goal_motion,
        "qy_max": qy_max,
        "qz_max": qz_max,
        "n_free_ticks": n_free,
        "n_c3_ticks": n_c3,
        "n_ee_box_admit_ticks": n_ee_box_admit,
        "ee_box_admit_pct_of_c3": ee_box_admit_pct,
        "free_mode_box_motion_m": free_motion,
        "c3_mode_box_motion_m": c3_motion,
        "lam_n_first_admit": lam_n_first,
        "lam_n_last_admit": lam_n_last,
        "ee_z_max_m": ee_z_max,
        "wall_seconds": wall,
        "pass_qy_lt_0.10": pass_qy,
        "pass_qz_lt_0.10": pass_qz,
        "pass_admit_ge_60pct": pass_admit,
        "pass_motion_ge_20mm": pass_motion,
        "PASS": passed,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: extract_metrics.py LOG [TIME_LOG]", file=sys.stderr)
        sys.exit(1)
    log = Path(sys.argv[1])
    tlog = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    print(json.dumps(extract(log, tlog), indent=2))
