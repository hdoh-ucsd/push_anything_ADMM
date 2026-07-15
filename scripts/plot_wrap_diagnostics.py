"""Visualize the wrong-direction failure mode of the wrapper-on push run.

Parses results/video_wrap_s0.log into a per-step DataFrame and renders
a 2x2 diagnostic figure to results/wrap_diagnostics.png.

Panels:
  1. Top-down XY trajectory (box, ee, proxy, goal) with time-color
  2. Position drift over time (box.x, box.y, ee.z)
  3. Wrapper mode dispatch + sample-winner over time
  4. Contact-normal alignment with goal direction
"""
import re
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


LOG = Path("results/video_wrap_s0.log")
OUT = Path("results/wrap_diagnostics.png")

DT = 0.01  # outer-loop control period (probe_hold_pose: dt_ctrl=0.01)

# Pre-compiled regexes
_EE  = re.compile(
    r"\[EE-COST\] obj=\(([+-][0-9.]+),([+-][0-9.]+)\) "
    r"ghat=\(([+-][0-9.]+),([+-][0-9.]+)\) "
    r"ee=\(([+-][0-9.]+),([+-][0-9.]+),([+-][0-9.]+)\) "
    r"ee_to_box=([0-9.]+)mm "
    r"stage=([0-9]) "
    r"proxy=\(([+-][0-9.]+),([+-][0-9.]+),([+-][0-9.]+)\)"
)
_CT  = re.compile(
    r"\[CONTACTS\] step=\d+ n_c=([0-9]+).*?"
    r"sd=([+-][0-9.]+)mm nhat=\(([+-][0-9.]+),([+-][0-9.]+),([+-][0-9.]+)\)"
)
_GS  = re.compile(
    r"\[GS\] step=([0-9]+) mode=(\w+) switch=(\w+) "
    r"best_k=([0-9]+) best_src=(\w+)"
)


def parse_log(path: Path) -> dict:
    ee, ct, gs = [], [], []
    with path.open() as f:
        for line in f:
            if "[EE-COST]" in line:
                m = _EE.search(line)
                if m:
                    g = m.groups()
                    ee.append((float(g[0]), float(g[1]),     # box_x, box_y
                               float(g[2]), float(g[3]),     # ghat_x, ghat_y
                               float(g[4]), float(g[5]),     # ee_x, ee_y
                               float(g[6]),                  # ee_z
                               int(g[8]),                    # stage
                               float(g[9]), float(g[10]),    # proxy_x, proxy_y
                               float(g[11])))                # proxy_z
            elif "[CONTACTS]" in line:
                m = _CT.search(line)
                if m:
                    g = m.groups()
                    ct.append((int(g[0]), float(g[1]),       # n_c, sd_mm
                               float(g[2]), float(g[3]), float(g[4])))
            elif line.startswith("[GS] step="):
                m = _GS.search(line)
                if m:
                    g = m.groups()
                    gs.append((int(g[0]), g[1], g[2], int(g[3]), g[4]))

    print(f"  ee rows : {len(ee)}")
    print(f"  ct rows : {len(ct)}")
    print(f"  gs rows : {len(gs)}")
    n = min(len(ee), len(ct), len(gs))
    ee_a = np.array(ee[:n])
    ct_a = np.array(ct[:n])
    gs_step = np.array([row[0] for row in gs[:n]], dtype=int)
    gs_mode = np.array([row[1] for row in gs[:n]])
    gs_switch = np.array([row[2] for row in gs[:n]])
    gs_src  = np.array([row[4] for row in gs[:n]])

    d = {
        "t":       np.arange(n) * DT,
        "box_x":   ee_a[:, 0],  "box_y":  ee_a[:, 1],
        "g_hat_x": ee_a[:, 2],  "g_hat_y": ee_a[:, 3],
        "ee_x":    ee_a[:, 4],  "ee_y":   ee_a[:, 5], "ee_z": ee_a[:, 6],
        "stage":   ee_a[:, 7].astype(int),
        "proxy_x": ee_a[:, 8],  "proxy_y": ee_a[:, 9], "proxy_z": ee_a[:, 10],
        "n_c":     ct_a[:, 0].astype(int),
        "sd_mm":   ct_a[:, 1],
        "nhat_x":  ct_a[:, 2],  "nhat_y": ct_a[:, 3], "nhat_z": ct_a[:, 4],
        "gs_step": gs_step,
        "mode":    gs_mode,
        "switch":  gs_switch,
        "best_src": gs_src,
    }
    d["align"] = d["nhat_x"] * d["g_hat_x"] + d["nhat_y"] * d["g_hat_y"]
    return d


def add_colored_line(ax, x, y, t, cmap="viridis", lw=2.0, label=None,
                     ls="-"):
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    norm = plt.Normalize(t.min(), t.max())
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=lw,
                        linestyle=ls)
    lc.set_array(t[:-1])
    ax.add_collection(lc)
    if label:
        ax.plot([], [], color=plt.get_cmap(cmap)(0.5), lw=lw, ls=ls,
                label=label)
    return lc


def plot_panel1_xy(ax, d):
    add_colored_line(ax, d["box_x"], d["box_y"], d["t"],
                     cmap="viridis", lw=2.5, label="box")
    add_colored_line(ax, d["ee_x"], d["ee_y"], d["t"],
                     cmap="plasma", lw=1.2, ls="--", label="EE")
    ax.scatter(d["proxy_x"], d["proxy_y"], c=d["t"], cmap="cividis",
               s=4, alpha=0.35, label="proxy")
    ax.plot(0.30, 0.0, marker="*", color="green", markersize=18,
            label="goal", zorder=5)
    ax.plot(0.0, 0.0, marker="s", color="black", markersize=8,
            label="box start", zorder=5)
    ax.plot(-0.076, 0.0, marker="o", color="dimgray", markersize=8,
            label="EE start", zorder=5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Top-down XY trajectory (color = time)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)


def plot_panel2_drift(ax, d):
    ax.plot(d["t"], d["box_y"] * 1000, label="box.y", color="tab:red", lw=2)
    ax.plot(d["t"], d["box_x"] * 1000, label="box.x", color="tab:blue", lw=2)
    ax.plot(d["t"], d["ee_z"] * 1000, label="ee.z", color="tab:green",
            lw=1.2, alpha=0.7)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("position (mm)")
    ax.set_title("Position drift over time")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)


def plot_panel3_mode(ax, d):
    free_mask = (d["mode"] == "free").astype(int)
    ax.fill_between(d["t"], 0, free_mask, where=free_mask.astype(bool),
                    color="tab:gray", alpha=0.4, label="mode=free",
                    step="post")

    src_palette = {
        "current":    "tab:green",
        "prev_repos": "tab:orange",
        "strat_0":    "tab:purple",
        "strat_1":    "tab:brown",
        "strat_2":    "tab:pink",
    }
    for src in np.unique(d["best_src"]):
        mask = d["best_src"] == src
        col = src_palette.get(str(src), "black")
        ax.scatter(d["t"][mask], np.full(mask.sum(), 1.05),
                   s=4, c=col, label=f"best_src={src}")

    ax.set_ylim(-0.05, 1.20)
    ax.set_yticks([0.5, 1.05])
    ax.set_yticklabels(["mode", "best_src"])
    ax.set_xlabel("time (s)")
    ax.set_title("Wrapper dispatch over time")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3, axis="x")


def plot_panel4_align(ax, d):
    ax.plot(d["t"], d["align"], color="tab:blue", lw=1.2)
    ax.axhline(+0.5, color="green", ls="--", lw=0.8,
               label="productive threshold (+0.5)")
    ax.axhline(0.0, color="gray", ls=":", lw=0.8, label="orthogonal")
    ax.axhline(-0.5, color="red", ls="--", lw=0.8,
               label="anti-aligned threshold (-0.5)")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("nhat_xy · g_hat")
    mean_align = d["align"].mean()
    min_align = d["align"].min()
    ax.set_title(f"Contact-normal alignment "
                 f"(mean={mean_align:+.3f}, min={min_align:+.3f})")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)


def main():
    print(f"Parsing {LOG} ...")
    d = parse_log(LOG)
    n = len(d["t"])
    print(f"\nDataset: {n} rows × {len(d)} cols")
    print(f"columns: {sorted(d.keys())}")
    print(f"\nhead (first 3):")
    for i in range(3):
        print(f"  t={d['t'][i]:.2f}  box=({d['box_x'][i]:+.4f},{d['box_y'][i]:+.4f})"
              f"  ee=({d['ee_x'][i]:+.4f},{d['ee_y'][i]:+.4f},{d['ee_z'][i]:+.4f})"
              f"  ghat=({d['g_hat_x'][i]:+.4f},{d['g_hat_y'][i]:+.4f})"
              f"  sd={d['sd_mm'][i]:+.1f}mm  align={d['align'][i]:+.4f}"
              f"  mode={d['mode'][i]}  best_src={d['best_src'][i]}")
    print(f"\ntail (last 3):")
    for i in range(n-3, n):
        print(f"  t={d['t'][i]:.2f}  box=({d['box_x'][i]:+.4f},{d['box_y'][i]:+.4f})"
              f"  ee=({d['ee_x'][i]:+.4f},{d['ee_y'][i]:+.4f},{d['ee_z'][i]:+.4f})"
              f"  ghat=({d['g_hat_x'][i]:+.4f},{d['g_hat_y'][i]:+.4f})"
              f"  sd={d['sd_mm'][i]:+.1f}mm  align={d['align'][i]:+.4f}"
              f"  mode={d['mode'][i]}  best_src={d['best_src'][i]}")

    nan_keys = [k for k, v in d.items()
                if isinstance(v, np.ndarray) and v.dtype.kind == "f"
                and np.isnan(v).any()]
    if nan_keys:
        print(f"\n[WARN] NaN in: {nan_keys}")
    else:
        print("\nNo NaN in any numeric column.")
    print(f"\nmode value counts:     {dict(Counter(d['mode']))}")
    print(f"best_src value counts: {dict(Counter(d['best_src']))}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    plot_panel1_xy(axes[0, 0], d)
    plot_panel2_drift(axes[0, 1], d)
    plot_panel3_mode(axes[1, 0], d)
    plot_panel4_align(axes[1, 1], d)
    fig.suptitle("Wrong-direction failure mechanism — wrapper-on, seed 0",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
