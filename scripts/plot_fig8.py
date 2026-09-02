#!/usr/bin/env python3
"""Fig. 8 per-goal timing for uninterrupted 28-goal C3+ sessions.

Each plotted point is one goal-to-goal segment from a single continuous session.
Failed or incomplete sessions are reported but never success-filtered or combined.
"""
import csv
import os
import re
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scripts.run_fig8_block_28_successes import diagnose
except ModuleNotFoundError:  # Direct execution: python scripts/plot_fig8.py
    from run_fig8_block_28_successes import diagnose

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "results", "fig8_consecutive_28_c3plus")
CAMPAIGN_DIR = os.path.join(REPO, "results", "fig8_consecutive_28_c3plus")

ORDER = [
    ("Letter I", "I_shape_texture"),
    ("Letter C", "C_shape_texture"),
    ("Letter R", "R_shape_texture"),
    ("Letter A", "A_shape_video"),
    ("Letter Y", "Y_shape_video"),
    ("Letter G", "G_shape_video"),
    ("Letter B", "B_shape_video"),
    ("Letter 3", "3_shape_video"),
    ("Letter H", "H_shape_texture"),
    ("Letter E", "E_shape_video"),
    ("Letter S", "S_shape"),
    ("Expo Box", "expo_box"),
    ("Lotion", "lotion_block"),
    ("Wood Block", "wood_block"),
    ("Tape", "tape"),
    ("Eraser", "eraser"),
    ("Milk Bottle", "milk"),
    ("Clamp", "clamp_block"),
    ("Chicken Broth", "chicken_broth"),
    ("Egg Carton", "egg_carton"),
    ("Book", "book_block"),
    ("Baby Toy", "baby_toy_block"),
    ("Gallon Milk", "gallon_milk"),
    ("Xbox", "xbox"),
    ("Push T", "push_t"),
]
TARGET = 28
SUCCESS_CSV = os.path.join(REPO, "FIG8_CONSECUTIVE_28_GOALS.csv")
SESSION_CSV = os.path.join(REPO, "FIG8_CONSECUTIVE_28_SESSIONS.csv")
GOAL_RE = re.compile(r"\[GOAL-GEN\] goal #(\d+) REACHED at t=([\d.]+)s")

# Neutral marks + one semantic accent (trial dots), text in ink tones.
INK = "#1f2430"
MUTED = "#6b7280"
GRID = "#e5e7eb"
BOX = "#9aa2af"
ORANGE = "#e8710a"
RED = "#c2413b"


def collect_sessions():
    data = OrderedDict((disp, []) for disp, _ in ORDER)
    records = []
    sessions = []

    for disp, task in ORDER:
        log_dir = os.path.join(CAMPAIGN_DIR, task)
        if not os.path.isdir(log_dir):
            continue
        for fn in sorted(os.listdir(log_dir)):
            if fn != f"{task}_consecutive{TARGET}.txt":
                continue
            path = os.path.join(log_dir, fn)
            txt = open(path, errors="replace").read()
            diagnosis = diagnose(Path(path), TARGET)
            if diagnosis["state"] not in ("success", "failure"):
                continue
            reached = [(int(n), float(t)) for n, t in GOAL_RE.findall(txt)]
            if [n for n, _ in reached] != list(range(1, len(reached) + 1)):
                continue
            rel_log = os.path.relpath(path, REPO)
            meta = re.search(r"\[RUN-META\]\s+git=(\S+)\s+seed=(\S+)", txt)
            outcome = "PASS" if diagnosis["state"] == "success" else "FAIL"
            sessions.append({
                "object": disp, "task": task, "log": rel_log,
                "commit": meta.group(1) if meta else "",
                "seed": meta.group(2) if meta else "",
                "outcome": outcome, "goals_reached": diagnosis["goals_reached"],
                "requested_goals": TARGET,
                "failed_goal_index": diagnosis["failed_goal_index"],
                "failure_category": diagnosis["failure_category"],
                "translational_error_m": diagnosis["translational_error_m"],
                "rotational_error_rad": diagnosis["rotational_error_rad"],
                "loose_goal": diagnosis["loose_goal"],
            })
            previous = 0.0
            total = reached[-1][1] if reached else 0.0
            for goal_index, timestamp in reached:
                segment = timestamp - previous
                previous = timestamp
                data[disp].append(segment)
                records.append({
                    "object": disp, "task": task, "log": rel_log,
                    "commit": meta.group(1) if meta else "",
                    "seed": meta.group(2) if meta else "",
                    "goal_index": goal_index,
                    "goal_timestamp_s": f"{timestamp:.3f}",
                    "segment_time_to_goal_s": f"{segment:.3f}",
                    "session_total_s": f"{total:.3f}",
                    "session_outcome": outcome,
                    "failure_category": diagnosis["failure_category"] or "",
                })
    return data, records, sessions


def collect():
    data, records, _ = collect_sessions()
    return data, records


def write_success_records(records):
    fields = ["object", "task", "log", "commit", "seed", "goal_index",
              "goal_timestamp_s", "segment_time_to_goal_s", "session_total_s",
              "session_outcome", "failure_category"]
    with open(SUCCESS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def write_session_records(sessions):
    fields = ["object", "task", "log", "commit", "seed", "outcome",
              "goals_reached", "requested_goals", "failed_goal_index",
              "failure_category", "translational_error_m",
              "rotational_error_rad", "loose_goal"]
    with open(SESSION_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sessions)


def main():
    data, records, sessions = collect_sessions()
    if not sessions:
        raise RuntimeError(
            "no terminal consecutive sessions found; preserving existing figures"
        )
    write_success_records(records)
    write_session_records(sessions)
    os.makedirs(OUT_DIR, exist_ok=True)
    n_obj = len(data)
    fig, ax = plt.subplots(figsize=(12.5, 5.2), dpi=200)

    rng = np.random.default_rng(0)
    session_by_object = {row["object"]: row for row in sessions}
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
            color = ORANGE if session_by_object[disp]["outcome"] == "PASS" else RED
            ax.scatter(np.full(len(times), i) + jitter, times,
                       s=34, color=color, zorder=3, edgecolor="white",
                       linewidth=0.7)

    ax.set_xlim(0.3, n_obj + 0.7)
    max_time = max((t for times in data.values() for t in times), default=1.0)
    ax.set_ylim(0, max(30.0, max_time * 1.12))
    ax.set_xticks(range(1, n_obj + 1))
    labels = [f"{name}\n{session_by_object[name]['outcome']}"
              if name in session_by_object else name for name in data]
    ax.set_xticklabels(labels, rotation=45, ha="right",
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
        "Figure 8 — 28 consecutive SE(2) goals per uninterrupted object session  "
        f"(n={n_trials} goal transitions)",
        fontsize=9.5, color=INK, loc="left", pad=10)
    fig.text(0.005, 0.005,
             "Time-to-goal is measured between consecutive tight-goal timestamps; "
             "the first interval begins at t=0. Failed sessions are not replaced or pooled. "
             "Orange=28/28 PASS; red=partial-goal FAIL. See goal and session CSVs.",
             fontsize=7, color=MUTED)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = os.path.join(OUT_DIR, "fig8_consecutive_time_to_goal.png")
    fig.savefig(out, facecolor="white")
    print(f"wrote {out}")
    print(f"wrote {SUCCESS_CSV} ({len(records)} consecutive goal transitions)")
    print(f"wrote {SESSION_CSV} ({len(sessions)} terminal sessions)")


if __name__ == "__main__":
    main()
