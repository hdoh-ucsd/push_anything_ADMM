#!/usr/bin/env python3
"""Fig 8-style time-to-goal figure across the anything object set.

Grammar (per the campaign spec): per-object boxplots showing median and
IQR, orange dots for individual trial data points, y-axis truncated at
the 180 s protocol cap, and a shaded region above the truncation marking
trials that never reached the goal (right-censored, not measured values).
With n=1 trials per object the boxes degenerate onto the single dot;
the script draws real boxes automatically once more trials accumulate
in results/fig8_objects/<task>_seed*.log.
"""
import csv
import os
import re
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "results", "fig8_objects")

ORDER = [
    ("I", "I_shape_texture"), ("C", "C_shape_texture"),
    ("R", "R_shape_texture"), ("A", "A_shape_video"),
    ("Y", "Y_shape_video"), ("G", "G_shape_video"),
    ("B", "B_shape_video"), ("3", "3_shape_video"),
    ("H", "H_shape_texture"), ("E", "E_shape_video"),
    ("Expo Box", "expo_box"), ("Lotion", "lotion"),
    ("Wood Block", "wood_block"), ("Tape", "tape"),
    ("Eraser", "eraser"), ("Milk Bottle", "milk"),
    ("Clamp", "clamp"), ("Chicken Broth", "chicken_broth"),
    ("Egg Carton", "egg_carton"), ("Book", "book"),
    ("Baby Toy", "baby_toy"), ("Gallon Milk", "gallon_milk"),
    ("Xbox", "xbox"),
]
T_CAP = 180.0

# Neutral marks + one semantic accent (trial dots), text in ink tones.
INK = "#1f2430"
MUTED = "#6b7280"
GRID = "#e5e7eb"
BOX = "#9aa2af"
ORANGE = "#e8710a"
BAND = "#eceef1"


def collect():
    data = OrderedDict()
    for disp, task in ORDER:
        times, censored = [], 0
        for fn in sorted(os.listdir(OUT_DIR)):
            if not re.match(rf"{re.escape(task)}_seed\d+\.log$", fn):
                continue
            txt = open(os.path.join(OUT_DIR, fn), errors="replace").read()
            if "[RESULT]" not in txt:
                continue          # incomplete/crashed run: not a trial
            m = re.search(r"ACHIEVED-FIXED-GOAL\] step=(\d+)", txt)
            if m:
                times.append(int(m.group(1)) * 0.075)
            else:
                censored += 1
        data[disp] = (times, censored)
    return data


def main():
    data = collect()
    n_obj = len(data)
    fig, ax = plt.subplots(figsize=(12.5, 5.2), dpi=200)

    # Truncation band: trials with no goal reach within the cap live here.
    band_lo, band_hi = T_CAP, T_CAP * 1.22
    ax.axhspan(band_lo, band_hi, color=BAND, zorder=0)
    ax.axhline(T_CAP, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)

    rng = np.random.default_rng(0)
    for i, (disp, (times, censored)) in enumerate(data.items(), start=1):
        if times:
            bp = ax.boxplot(
                [times], positions=[i], widths=0.55, patch_artist=True,
                showfliers=False, zorder=2,
                boxprops=dict(facecolor="none", edgecolor=BOX, lw=1.2),
                whiskerprops=dict(color=BOX, lw=1.2),
                capprops=dict(color=BOX, lw=1.2),
                medianprops=dict(color=INK, lw=1.6),
            )
            jitter = rng.uniform(-0.10, 0.10, size=len(times))
            ax.scatter(np.full(len(times), i) + jitter, times,
                       s=34, color=ORANGE, zorder=3, edgecolor="white",
                       linewidth=0.7)
        if censored:
            # Right-censored trials: presence markers in the band's lower
            # third (the label owns the top edge).
            ys = np.linspace(band_lo + 6, band_lo + (band_hi - band_lo) * 0.45,
                             censored + 1)[:-1] if censored > 1 else \
                np.array([band_lo + (band_hi - band_lo) * 0.30])
            ax.scatter(np.full(censored, i), ys, s=30, facecolor="none",
                       edgecolor=ORANGE, linewidth=1.3, zorder=3)

    ax.set_xlim(0.3, n_obj + 0.7)
    ax.set_ylim(0, band_hi)
    ax.set_xticks(range(1, n_obj + 1))
    ax.set_xticklabels(list(data.keys()), rotation=45, ha="right",
                       fontsize=8.5, color=INK)
    ax.set_yticks([0, 30, 60, 90, 120, 150, 180])
    ax.set_ylabel("Time-to-goal (s)", fontsize=10, color=INK)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)

    ax.text(0.99, (band_hi - 3) / band_hi,
            "no goal reach within 180 s (censored)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color=MUTED, style="italic")

    n_trials = sum(len(t) + c for t, c in data.values())
    ax.set_title(
        "Time-to-goal by object — single-object pushing, port replication  "
        f"(n={n_trials} trials, 1 per object; filled dot = goal reached, "
        "open dot = censored at 180 s)",
        fontsize=9.5, color=INK, loc="left", pad=10)
    fig.text(0.005, 0.005,
             "Protocol: canonical fixed goal (Δ=0.187 m, Δyaw=−0.74 rad) for "
             "all objects; seed 0; time-to-goal = first achieved-goal latch "
             "(geodesic). Paper protocol uses randomized goals.",
             fontsize=7, color=MUTED)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = os.path.join(OUT_DIR, "fig8_time_to_goal.png")
    fig.savefig(out, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
