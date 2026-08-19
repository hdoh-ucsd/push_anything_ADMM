#!/usr/bin/env python3
"""Render the reference-vs-port goal-error comparison as a PNG.

Two stacked panels (position error, rotation error — different scales,
never dual-axis), one line per system. Palette = validated categorical
slots 1-2 (light mode); tight-goal thresholds as muted dashed guides.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#1a1a19"
INK_2 = "#5f5e56"
GRID = "#e8e7e0"
C_REF = "#2a78d6"   # categorical slot 1 (blue)  — reference
C_PORT = "#eb6834"  # categorical slot 2 (orange) — port


def _thin(traj, max_pts=2000):
    if len(traj) <= max_pts:
        return traj
    step = len(traj) / max_pts
    return [traj[int(i * step)] for i in range(max_pts)] + [traj[-1]]


def render(ref, port, out_path):
    ref = _thin(ref)
    port = _thin(port)
    fig, (ax_p, ax_r) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, facecolor=SURFACE)

    panels = [
        (ax_p, 1, "Position error (m)", 0.02, "tight 0.02 m"),
        (ax_r, 2, "Rotation error (rad)", 0.10, "tight 0.1 rad"),
    ]
    for ax, idx, ylabel, thr, thr_label in panels:
        ax.set_facecolor(SURFACE)
        for series, color, label in (
                (ref, C_REF, "reference"), (port, C_PORT, "port")):
            ax.plot([s[0] for s in series], [s[idx] for s in series],
                    color=color, linewidth=2, label=label,
                    solid_capstyle="round")
            ax.annotate(
                f"{label} {series[-1][idx]:.3f}",
                xy=(series[-1][0], series[-1][idx]),
                xytext=(4, 0), textcoords="offset points",
                color=INK, fontsize=9, va="center")
        ax.axhline(thr, color=INK_2, linewidth=1, linestyle=(0, (4, 4)))
        ax.annotate(thr_label, xy=(0, thr), xytext=(4, 4),
                    textcoords="offset points", color=INK_2, fontsize=8)
        ax.set_ylabel(ylabel, color=INK)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(INK_2)
        ax.tick_params(colors=INK_2, labelcolor=INK)
        ax.set_ylim(bottom=0)
        ax.margins(x=0.08)

    ax_p.legend(loc="lower left", frameon=False, labelcolor=INK)
    ax_r.set_xlabel("time (s)  —  reference time is loop-index mapped "
                    "(realtime sim, quasi-constant loop rate)", color=INK_2)
    ax_p.set_title("push_t: goal error, reference vs port (same fixed goal)",
                   color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
