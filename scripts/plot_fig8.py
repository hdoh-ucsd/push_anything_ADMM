#!/usr/bin/env python3
"""Fig 8-style time-to-goal figure for all recorded successful runs.

Every completed log in a task's configured Fig. 8 result directory whose
filename starts with the task name is considered. A run is recorded only when
it contains an
``ACHIEVED-FIXED-GOAL`` latch; unsuccessful runs are neither assigned a timeout
value nor drawn as censored observations.  The y-axis expands to the measured
success times, including successes later than 180 s.
"""
import csv
import os
import re
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "results", "fig8_objects")
BLOCK_OUT_DIR = os.path.join(REPO, "results", "fig8_blocks")

ORDER = [
    ("Letter I", "I_shape_texture"), ("Letter C", "C_shape_texture"),
    ("Letter R", "R_shape_texture"), ("Letter A", "A_shape_video"),
    ("Letter Y", "Y_shape_video"), ("Letter G", "G_shape_video"),
    ("Letter B", "B_shape_video"), ("Letter 3", "3_shape_video"),
    ("Letter H", "H_shape_texture"), ("Letter E", "E_shape_video"),
    ("Expo Box", "expo_box"), ("Lotion", "lotion"),
    ("Wood Block", "wood_block"), ("Tape", "tape"),
    ("Eraser", "eraser"), ("Milk Bottle", "milk"),
    ("Clamp", "clamp"), ("Chicken Broth", "chicken_broth"),
    ("Egg Carton", "egg_carton"), ("Book", "book"),
    ("Baby Toy", "baby_toy"), ("Gallon Milk", "gallon_milk"),
    ("Xbox", "xbox"), ("Push T", "push_t"),
]
TASK_LOG_DIR = {"push_t": BLOCK_OUT_DIR}
STEP_DT = 0.075
SUCCESS_CSV = os.path.join(REPO, "FIG8_SUCCESS_RUNS.csv")

# Neutral marks + one semantic accent (trial dots), text in ink tones.
INK = "#1f2430"
MUTED = "#6b7280"
GRID = "#e5e7eb"
BOX = "#9aa2af"
ORANGE = "#e8710a"


def collect():
    data = OrderedDict((disp, []) for disp, _ in ORDER)
    records = []
    seen_logs = set()

    # The manifest is tracked while bulky raw logs are not. Preserve its
    # validated successes when regenerating in a checkout where an older log
    # has been archived or removed, then append any newly completed logs.
    if os.path.exists(SUCCESS_CSV):
        with open(SUCCESS_CSV, newline="") as f:
            for row in csv.DictReader(f):
                disp = row.get("object", "")
                log = row.get("log", "")
                if disp not in data or not log:
                    continue
                try:
                    time_s = float(row["time_to_goal_s"])
                except (KeyError, TypeError, ValueError):
                    continue
                data[disp].append(time_s)
                records.append(row)
                seen_logs.add(log)

    for disp, task in ORDER:
        log_dir = TASK_LOG_DIR.get(task, OUT_DIR)
        for fn in sorted(os.listdir(log_dir)):
            if not (fn.startswith(f"{task}_") and fn.endswith(".log")):
                continue
            txt = open(os.path.join(log_dir, fn), errors="replace").read()
            if "[RESULT]" not in txt:
                continue          # incomplete/crashed run: not a trial
            m = re.search(r"ACHIEVED-FIXED-GOAL\] step=(\d+)", txt)
            if m:
                rel_log = os.path.relpath(os.path.join(log_dir, fn), REPO)
                if rel_log in seen_logs:
                    continue
                step = int(m.group(1))
                time_s = step * STEP_DT
                data[disp].append(time_s)
                meta = re.search(r"\[RUN-META\]\s+git=(\S+)\s+seed=(\S+)", txt)
                records.append({
                    "object": disp,
                    "task": task,
                    "log": rel_log,
                    "commit": meta.group(1) if meta else "",
                    "seed": meta.group(2) if meta else "",
                    "first_goal_step": step,
                    "time_to_goal_s": f"{time_s:.3f}",
                })
                seen_logs.add(rel_log)
    return data, records


def write_success_records(records):
    fields = ["object", "task", "log", "commit", "seed",
              "first_goal_step", "time_to_goal_s"]
    with open(SUCCESS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def main():
    data, records = collect()
    write_success_records(records)
    n_obj = len(data)
    fig, ax = plt.subplots(figsize=(12.5, 5.2), dpi=200)

    rng = np.random.default_rng(0)
    for i, (disp, times) in enumerate(data.items(), start=1):
        if times:
            bp = ax.boxplot(
                [times], positions=[i], widths=0.55, patch_artist=True,
                orientation="vertical", showfliers=False, zorder=2,
                boxprops=dict(facecolor="none", edgecolor=BOX, lw=1.2),
                whiskerprops=dict(color=BOX, lw=1.2),
                capprops=dict(color=BOX, lw=1.2),
                medianprops=dict(color=INK, lw=1.6),
            )
            jitter = rng.uniform(-0.10, 0.10, size=len(times))
            ax.scatter(np.full(len(times), i) + jitter, times,
                       s=34, color=ORANGE, zorder=3, edgecolor="white",
                       linewidth=0.7)

    ax.set_xlim(0.3, n_obj + 0.7)
    max_time = max((t for times in data.values() for t in times), default=1.0)
    ax.set_ylim(0, max(30.0, max_time * 1.12))
    ax.set_xticks(range(1, n_obj + 1))
    ax.set_xticklabels(list(data.keys()), rotation=45, ha="right",
                       fontsize=8.5, color=INK)
    ax.set_ylabel("Time-to-goal (s)", fontsize=10, color=INK)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)

    n_trials = len(records)
    ax.set_title(
        "Time-to-goal by object — recorded successful runs only  "
        f"(n={n_trials}; each dot is a fixed-goal success)",
        fontsize=9.5, color=INK, loc="left", pad=10)
    fig.text(0.005, 0.005,
             "Time-to-goal = first achieved-fixed-goal latch (geodesic). "
             "Unsuccessful and incomplete runs are omitted, not censored at 180 s; "
             "see fig8_success_runs.csv for run provenance.",
             fontsize=7, color=MUTED)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = os.path.join(OUT_DIR, "fig8_time_to_goal.png")
    fig.savefig(out, facecolor="white")
    print(f"wrote {out}")
    print(f"wrote {SUCCESS_CSV} ({len(records)} successful runs)")


if __name__ == "__main__":
    main()
