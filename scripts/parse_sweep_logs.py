#!/usr/bin/env python3
"""Parse contact-free sweep logs → per-run CSV with T1-T5 thresholds.

Reads results/contact_free_sweep/logs/sweep_d{DD}cm_s{S}.log
Writes results/contact_free_sweep/summary.csv

Tag formats it expects (already emitted by main.py / SamplingC3MPC):
  [RESULT] method=... final_obj_xy=(x,y) goal_dist=d success=YES|NO
  [GS-tgt] step=N ee=(x,y,z) ...
  [EErel] along_push=...m ... obj=[x y] ...
  [GS] step=N mode=free|c3 ...
  '  t=Ts | ee=(x,y,z) | obj=(x,y,z) | |u|=U Nm | goal_dist=d m'
"""
from __future__ import annotations
import csv
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOG_DIR = REPO / "results" / "contact_free_sweep" / "logs"
OUT_CSV = REPO / "results" / "contact_free_sweep" / "summary.csv"

# Regexes -------------------------------------------------------------
RE_RESULT = re.compile(
    r"^\[RESULT\] method=(?P<method>\S+)\s+final_obj_xy=\("
    r"(?P<fx>-?\d+\.\d+),\s*(?P<fy>-?\d+\.\d+)\)\s+"
    r"goal_dist=(?P<gd>\d+\.\d+)m\s+success=(?P<succ>YES|NO)"
)
RE_GS_TGT = re.compile(
    r"^\[GS-tgt\]\s+step=(?P<step>\d+)\s+ee=\("
    r"(?P<ex>[-+]?\d+\.\d+),(?P<ey>[-+]?\d+\.\d+),(?P<ez>[-+]?\d+\.\d+)\)"
)
# [EErel] obj=[x y]  — the array is whitespace-separated, optional sign
RE_EEREL_OBJ = re.compile(
    r"^\[EErel\].*obj=\[\s*(?P<ox>-?\d+\.?\d*(?:[eE][-+]?\d+)?)"
    r"\s+(?P<oy>-?\d+\.?\d*(?:[eE][-+]?\d+)?)\s*\]"
)
RE_GS_MODE = re.compile(r"^\[GS\]\s+step=\d+\s+mode=(?P<mode>\w+)")
RE_T_STEP = re.compile(
    r"^\s+t=(?P<t>\d+\.\d+)s\s+\|\s+ee=\("
    r"(?P<ex>[-+]?\d+\.\d+),\s*(?P<ey>[-+]?\d+\.\d+),\s*(?P<ez>[-+]?\d+\.\d+)\)"
    r"\s+\|\s+obj=\(([-+]?\d+\.\d+),\s*([-+]?\d+\.\d+),\s*([-+]?\d+\.\d+)\)"
    r"\s+\|\s+\|u\|=(?P<u>\d+\.\d+)\s+Nm"
)
RE_OVERRIDE_GOAL = re.compile(
    r"^\[OVERRIDE\]\s+goal_xy=\[(?P<gx>[-+]?\d+\.?\d*),\s*(?P<gy>[-+]?\d+\.?\d*)\]"
)
RE_FNAME = re.compile(r"sweep_d(?P<dd>\d{2})cm_s(?P<s>\d)\.log$")


def parse_log(path: Path) -> dict:
    m = RE_FNAME.search(path.name)
    if not m:
        return {}
    distance_cm = int(m.group("dd"))
    seed = int(m.group("s"))

    goal_xy = None
    final_obj = None
    final_dist = None
    success = None

    ee_z_series: list[float] = []
    obj_xy_series: list[tuple[float, float]] = []
    u_peak = 0.0
    n_c3 = 0
    n_steps_gs = 0
    final_ee_z = None

    if not path.exists():
        return {
            "distance_cm": distance_cm, "seed": seed,
            "status": "missing_log",
        }

    text = path.read_text(errors="replace").splitlines()
    for line in text:
        if line.startswith("[OVERRIDE]"):
            mm = RE_OVERRIDE_GOAL.match(line)
            if mm:
                goal_xy = (float(mm.group("gx")), float(mm.group("gy")))
            continue
        if line.startswith("[GS-tgt]"):
            mm = RE_GS_TGT.match(line)
            if mm:
                ee_z_series.append(float(mm.group("ez")))
                final_ee_z = float(mm.group("ez"))
            continue
        if line.startswith("[EErel]"):
            mm = RE_EEREL_OBJ.match(line)
            if mm:
                obj_xy_series.append((float(mm.group("ox")), float(mm.group("oy"))))
            continue
        if line.startswith("[GS]"):
            mm = RE_GS_MODE.match(line)
            if mm:
                n_steps_gs += 1
                if mm.group("mode") == "c3":
                    n_c3 += 1
            continue
        if line.startswith("  t="):
            mm = RE_T_STEP.match(line)
            if mm:
                u = float(mm.group("u"))
                if u > u_peak:
                    u_peak = u
            continue
        if line.startswith("[RESULT]"):
            mm = RE_RESULT.match(line)
            if mm:
                final_obj = (float(mm.group("fx")), float(mm.group("fy")))
                final_dist = float(mm.group("gd"))
                success = mm.group("succ") == "YES"
            continue

    # Status
    if final_obj is None:
        # No RESULT line; check for timeout / partial
        if any("Simulation complete" in l for l in text):
            status = "partial_no_result"
        else:
            status = "timeout_or_crash"
    else:
        status = "ok"

    # Init box xy (first sample) — use first [EErel] obj or assume (0,0)
    if obj_xy_series:
        init_obj = obj_xy_series[0]
    else:
        init_obj = (0.0, 0.0)

    # Goal: from override line, else fall back to derived distance.
    if goal_xy is None:
        goal_xy = (init_obj[0] + distance_cm / 100.0, init_obj[1])

    gx, gy = goal_xy
    g_vec_x = gx - init_obj[0]
    g_vec_y = gy - init_obj[1]
    g_mag = (g_vec_x**2 + g_vec_y**2) ** 0.5
    if g_mag < 1e-9:
        ghat = (1.0, 0.0)
    else:
        ghat = (g_vec_x / g_mag, g_vec_y / g_mag)
    # Perp (90 deg CCW)
    ph = (-ghat[1], ghat[0])

    # Total motion / direction projections
    if final_obj is None and obj_xy_series:
        final_obj = obj_xy_series[-1]

    if final_obj is None:
        final_obj = init_obj

    dx = final_obj[0] - init_obj[0]
    dy = final_obj[1] - init_obj[1]
    total_motion_mm = ((dx**2 + dy**2) ** 0.5) * 1000.0
    motion_dir_mm = (dx * ghat[0] + dy * ghat[1]) * 1000.0
    motion_perp_mm = abs(dx * ph[0] + dy * ph[1]) * 1000.0

    # final_dist fallback
    if final_dist is None:
        final_dist = ((final_obj[0] - gx) ** 2 + (final_obj[1] - gy) ** 2) ** 0.5

    # EE z froze: min(ee_z) < 0.03 AND ee_z stayed < 0.05 for >= 200 consec steps
    ee_z_froze = False
    if ee_z_series:
        min_z = min(ee_z_series)
        # longest consecutive run below 0.05
        longest = 0
        cur = 0
        for z in ee_z_series:
            if z < 0.05:
                cur += 1
                if cur > longest:
                    longest = cur
            else:
                cur = 0
        ee_z_froze = (min_z < 0.03) and (longest >= 200)
    else:
        min_z = None

    if final_ee_z is None and ee_z_series:
        final_ee_z = ee_z_series[-1]

    # Thresholds T1-T5
    goal_distance_m = distance_cm / 100.0
    err = final_dist if final_dist is not None else float("inf")
    motion_dir_m = motion_dir_mm / 1000.0
    motion_mag_m = total_motion_mm / 1000.0
    # cos(angle to goal direction) of net motion
    if motion_mag_m > 1e-9:
        cos_to_goal = (dx * ghat[0] + dy * ghat[1]) / motion_mag_m
    else:
        cos_to_goal = 0.0

    T1 = int(err <= 0.005)                          # within 5 mm
    T2 = int(err <= 0.10 * goal_distance_m)         # within 10% of goal dist
    T3 = int(motion_dir_m >= 0.80 * goal_distance_m)
    T4 = int(motion_dir_m >= 0.50 * goal_distance_m)
    # T5: moved > 5 mm AND net motion within 30 deg of goal direction
    T5 = int((motion_mag_m > 0.005) and (cos_to_goal >= 0.866))  # cos(30°)

    return {
        "distance_cm": distance_cm,
        "seed": seed,
        "status": status,
        "final_box_err_m": round(final_dist, 4) if final_dist is not None else None,
        "total_motion_mm": round(total_motion_mm, 2),
        "motion_in_dir_mm": round(motion_dir_mm, 2),
        "motion_perp_mm": round(motion_perp_mm, 2),
        "ee_z_froze": int(ee_z_froze),
        "final_ee_z": round(final_ee_z, 4) if final_ee_z is not None else None,
        "min_ee_z": round(min_z, 4) if min_z is not None else None,
        "peak_tau_nm": round(u_peak, 2),
        "n_c3_entries": n_c3,
        "n_steps_gs": n_steps_gs,
        "T1": T1, "T2": T2, "T3": T3, "T4": T4, "T5": T5,
    }


def main():
    rows = []
    for d_cm in (5, 10, 15, 20, 30):
        for s in (0, 1, 2):
            name = f"sweep_d{d_cm:02d}cm_s{s}.log"
            path = LOG_DIR / name
            if not path.exists():
                rows.append({
                    "distance_cm": d_cm, "seed": s,
                    "status": "missing_log",
                })
                continue
            rows.append(parse_log(path))

    if not rows:
        print("[parse] no logs", file=sys.stderr)
        return 1

    keys = [
        "distance_cm", "seed", "status",
        "final_box_err_m", "total_motion_mm",
        "motion_in_dir_mm", "motion_perp_mm",
        "ee_z_froze", "final_ee_z", "min_ee_z",
        "peak_tau_nm", "n_c3_entries", "n_steps_gs",
        "T1", "T2", "T3", "T4", "T5",
    ]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

    print(f"[parse] wrote {OUT_CSV.relative_to(REPO)}  ({len(rows)} rows)")

    # Echo a quick table to stdout
    print()
    print(f"{'d_cm':>4} {'s':>1} {'status':<22} {'err_m':>7} "
          f"{'motion_mm':>9} {'dir_mm':>7} {'perp_mm':>7} "
          f"{'froze':>5} {'fin_ez':>6} "
          f"{'T1':>2} {'T2':>2} {'T3':>2} {'T4':>2} {'T5':>2}")
    for r in rows:
        print(
            f"{r.get('distance_cm','?'):>4} {r.get('seed','?'):>1} "
            f"{r.get('status','?'):<22} "
            f"{(r.get('final_box_err_m') if r.get('final_box_err_m') is not None else '-'):>7} "
            f"{r.get('total_motion_mm','-'):>9} "
            f"{r.get('motion_in_dir_mm','-'):>7} "
            f"{r.get('motion_perp_mm','-'):>7} "
            f"{r.get('ee_z_froze','-'):>5} "
            f"{(r.get('final_ee_z') if r.get('final_ee_z') is not None else '-'):>6} "
            f"{r.get('T1','-'):>2} {r.get('T2','-'):>2} "
            f"{r.get('T3','-'):>2} {r.get('T4','-'):>2} {r.get('T5','-'):>2}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
