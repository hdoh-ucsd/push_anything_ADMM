#!/usr/bin/env python3
"""Render static frames from a JSONL telemetry file at key moments.

Produces PNGs that can be visually compared to Drake's verdict.mp4 frames.
"""
import argparse
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrow

ATTR_COLORS = {
    "planned_productive":   "#4caf50",
    "planned_unproductive": "#ff9800",
    "accidental_contact":   "#f44336",
    "no_contact_free":      "#888888",
}


def load_jsonl(path):
    with open(path) as f:
        lines = f.readlines()
    meta = json.loads(lines[0]).get("_meta", {})
    records = [json.loads(l) for l in lines[1:]]
    return meta, records


def render_frame(meta, records, idx, out_path, title=None, show_target_chase=True):
    r = records[idx]
    goal = meta.get("goal", [-0.3, 0.0])
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#1a1a1a")
    ax.set_facecolor("#232323")

    all_x = [rr["ee_pos"][0] for rr in records if rr.get("ee_pos")] + \
            [rr["obj_pos"][0] for rr in records if rr.get("obj_pos")] + [goal[0]]
    all_y = [rr["ee_pos"][1] for rr in records if rr.get("ee_pos")] + \
            [rr["obj_pos"][1] for rr in records if rr.get("obj_pos")] + [goal[1]]
    if show_target_chase:
        all_x += [rr["p_repos"][0] for rr in records if rr.get("p_repos")]
        all_y += [rr["p_repos"][1] for rr in records if rr.get("p_repos")]
    minx, maxx = min(all_x) - 0.08, max(all_x) + 0.08
    miny, maxy = min(all_y) - 0.08, max(all_y) + 0.08
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", color="#aaa")
    ax.set_ylabel("y (m)", color="#aaa")
    ax.tick_params(colors="#aaa")

    ax.plot(goal[0], goal[1], marker="*", color="#ffd700", markersize=20, zorder=5)
    ax.annotate("GOAL", xy=goal, xytext=(goal[0], goal[1] - 0.025),
                color="#ffd700", ha="center", fontsize=9)

    box_trail_x = [records[i]["obj_pos"][0] for i in range(idx + 1) if records[i].get("obj_pos")]
    box_trail_y = [records[i]["obj_pos"][1] for i in range(idx + 1) if records[i].get("obj_pos")]
    if box_trail_x:
        ax.plot(box_trail_x, box_trail_y, color="white", alpha=0.4, linewidth=1)

    for i in range(1, idx + 1):
        a, b = records[i - 1], records[i]
        if not a.get("ee_pos") or not b.get("ee_pos"):
            continue
        ax.plot(
            [a["ee_pos"][0], b["ee_pos"][0]],
            [a["ee_pos"][1], b["ee_pos"][1]],
            color=ATTR_COLORS.get(b.get("attribution"), "#777"),
            linewidth=1.5, alpha=0.75,
        )

    if r.get("obj_pos"):
        ox, oy = r["obj_pos"]
        rect = Rectangle((ox - 0.05, oy - 0.05), 0.10, 0.10,
                         facecolor="#bbb", alpha=0.3, edgecolor="white", linewidth=1.5)
        ax.add_patch(rect)

    if r.get("ee_pos"):
        ex, ey = r["ee_pos"][0], r["ee_pos"][1]
        ee_color = ATTR_COLORS.get(r.get("attribution"), "#777")
        ee_circle = Circle((ex, ey), 0.012, facecolor=ee_color, edgecolor="white", linewidth=1, zorder=4)
        ax.add_patch(ee_circle)
        ax.plot(ex, ey, marker="o", color="white", markersize=2, zorder=5)

    if r.get("contact_active") and r.get("contact_normal") and r.get("ee_pos"):
        nx, ny = r["contact_normal"][0], r["contact_normal"][1]
        ex, ey = r["ee_pos"][0], r["ee_pos"][1]
        arrow = FancyArrow(ex, ey, -nx * 0.04, -ny * 0.04,
                           color="#ffeb3b", width=0.001, head_width=0.006,
                           length_includes_head=True)
        ax.add_patch(arrow)

    if r.get("g_hat") and r.get("obj_pos"):
        ox, oy = r["obj_pos"][0], r["obj_pos"][1]
        gx, gy = r["g_hat"]
        ax.annotate("", xytext=(ox, oy), xy=(ox + gx * 0.05, oy + gy * 0.05),
                    arrowprops=dict(arrowstyle="->", color="#ffd700", alpha=0.6, lw=1.5))

    # Target-chase overlay: draws the EE's current reposition target, the
    # EE→target gap, and flags re-target events. Guarded so logs without
    # p_repos still render normally.
    ee_to_target_str = "—"
    if show_target_chase and r.get("p_repos") and r.get("ee_pos"):
        tx, ty = r["p_repos"][0], r["p_repos"][1]
        ex, ey = r["ee_pos"][0], r["ee_pos"][1]
        gap = r.get("ee_to_target")
        # Gap line: color/width scaled by gap magnitude (larger gap = redder,
        # thicker). Threshold 0.025 m ≈ IK "finished" tol; 0.10 m ≈ dig's mean.
        if gap is not None:
            if gap < 0.025:
                gap_color, gap_lw = "#4caf50", 1.0   # green: at-target
            elif gap < 0.080:
                gap_color, gap_lw = "#ffeb3b", 1.5   # yellow: closing
            else:
                gap_color, gap_lw = "#ff5252", 2.2   # red: still chasing
            ee_to_target_str = f"{gap*1000:.0f}mm"
        else:
            gap_color, gap_lw = "#aaaaaa", 1.0
        ax.plot([ex, tx], [ey, ty], color=gap_color, alpha=0.85,
                linewidth=gap_lw, linestyle="--", zorder=3)
        # Target marker: open diamond. Highlight with red ring on re-target.
        retargeted = bool(r.get("target_changed"))
        marker_edge = "#ff1744" if retargeted else "#00e5ff"
        marker_lw = 2.6 if retargeted else 1.6
        ax.plot(tx, ty, marker="D", markerfacecolor="none",
                markeredgecolor=marker_edge, markeredgewidth=marker_lw,
                markersize=11, zorder=6)
        if retargeted:
            ax.annotate("RE-TARGET", xy=(tx, ty),
                        xytext=(tx + 0.012, ty + 0.012),
                        color="#ff1744", fontsize=8, fontweight="bold")
        else:
            label = r.get("target_label")
            if label:
                ax.annotate(label, xy=(tx, ty),
                            xytext=(tx + 0.010, ty - 0.018),
                            color="#00e5ff", fontsize=7, alpha=0.8)

    mode = r.get("mode") or "?"
    attr = r.get("attribution") or "?"
    lam_n = r.get("lambda_n_max")
    lam_n_str = f"{lam_n:.2f}" if lam_n is not None else "—"

    title_text = title or f"t={r['sim_t']:.2f}s  step={r['step']}"
    info = f"mode={mode}  attr={attr}  λ_n={lam_n_str} N  ee→tgt={ee_to_target_str}"

    ax.set_title(f"{title_text}\n{info}", color="white", fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100, facecolor="#1a1a1a")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Render static frames from a JSONL telemetry file.")
    parser.add_argument("jsonl", type=Path, help="Path to JSONL telemetry file.")
    parser.add_argument("out_dir", type=Path, nargs="?", default=None,
                        help="Output directory for PNG frames.")
    parser.add_argument("--show-target-chase", dest="show_target_chase",
                        action="store_true", default=True,
                        help="Overlay current target + EE→target gap (default on).")
    parser.add_argument("--no-target-chase", dest="show_target_chase",
                        action="store_false",
                        help="Disable target-chase overlay.")
    args = parser.parse_args()

    jsonl = args.jsonl
    out_dir = args.out_dir if args.out_dir is not None else jsonl.parent / "frames"
    out_dir.mkdir(exist_ok=True, parents=True)

    meta, records = load_jsonl(jsonl)
    print(f"Loaded {len(records)} records from {jsonl}")

    n = len(records)

    rich_entry_idx = None
    rich_exit_idx = None
    for i, r in enumerate(records):
        if r.get("mode") == "c3" and rich_entry_idx is None:
            rich_entry_idx = i
        if r.get("mode") == "c3":
            rich_exit_idx = i

    target_indices = set()
    target_indices.add(0)
    target_indices.add(n - 1)
    if rich_entry_idx is not None:
        target_indices.add(max(0, rich_entry_idx - 1))
        target_indices.add(rich_entry_idx)
        target_indices.add(min(n - 1, rich_entry_idx + 5))
    if rich_exit_idx is not None and rich_exit_idx != rich_entry_idx:
        target_indices.add(rich_exit_idx)
        target_indices.add(min(n - 1, rich_exit_idx + 5))
    for s in range(0, n, 50):
        target_indices.add(s)

    sorted_indices = sorted(target_indices)
    for i, idx in enumerate(sorted_indices):
        r = records[idx]
        label = None
        if idx == rich_entry_idx:
            label = f"RICH-MODE ENTRY  t={r['sim_t']:.2f}s"
        elif idx == rich_exit_idx:
            label = f"RICH-MODE EXIT  t={r['sim_t']:.2f}s"
        elif idx == 0:
            label = f"START  t={r['sim_t']:.2f}s"
        elif idx == n - 1:
            label = f"END  t={r['sim_t']:.2f}s"

        out_path = out_dir / f"frame_{i:02d}_step{r['step']:04d}_t{r['sim_t']:.2f}s.png"
        render_frame(meta, records, idx, out_path, title=label,
                     show_target_chase=args.show_target_chase)
        print(f"  Rendered: {out_path}  ({label or 'sample'})")

    print(f"\nDone. {len(sorted_indices)} frames written to {out_dir}/")


if __name__ == "__main__":
    main()
