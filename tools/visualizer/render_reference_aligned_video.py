#!/usr/bin/env python3
"""§7.73b — render the reference-aligned seed-0 push as two mp4 views.

Parses the §7.72/73 seed-0 log for per-tick pose (box_p, box_q, ee_p) from
[GATE-CONTACT] lines plus mode + f_cmd from [STEP] lines and Drake normal
from [DRAKE-CONTACT] lines. Renders:

  1. Top-down: goal ★, box (rotated rectangle by yaw), EE ○, force arrow
     during contact. Matches the deck's usual top-down convention.
  2. Perspective (3D): shows the ~90° forward pitch — the top-down view
     would flatten this.

Outputs mp4 via ffmpeg to the Windows sink (/d/projects/ERL/push_anything_ADMM/).

No Drake / no sim / no substrate exposure. Log parse only.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrow, RegularPolygon
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


BOX_HALF = 0.05
PUSHER_R = 0.025
GOAL     = np.array([-0.3, 0.0])


# ---------- log parsing ----------

RE_GATE = re.compile(
    r"^\[GATE-CONTACT\] step=(\d+).*"
    r"box_q=\(([^)]+)\) "
    r"box_p=\(([^)]+)\) "
    r"ee_p=\(([^)]+)\)"
)
RE_STEP = re.compile(
    r"^\[STEP\] step=(\d+) mode=(\w+).*"
)
RE_STEP_FCMD = re.compile(r"f_cmd=\(([^)]+)\)")
RE_DRAKE = re.compile(
    r"^\[DRAKE-CONTACT\] step=(\d+).*ee_box_normal=([\d.]+)"
)


def _parse_triple(s: str) -> np.ndarray:
    return np.array([float(x) for x in s.split(",")], dtype=float)


def parse_log(path: Path) -> Dict[int, dict]:
    """Return dict step → {box_p, box_q, ee_p, mode, drake_normal, f_cmd}."""
    frames: Dict[int, dict] = {}
    with open(path) as f:
        for line in f:
            if line.startswith("[GATE-CONTACT]"):
                m = RE_GATE.match(line)
                if not m:
                    continue
                step = int(m.group(1))
                box_q = _parse_triple(m.group(2))  # (w, x, y, z)
                box_p = _parse_triple(m.group(3))
                ee_p  = _parse_triple(m.group(4))
                frames.setdefault(step, {})
                frames[step].update(box_q=box_q, box_p=box_p, ee_p=ee_p)
            elif line.startswith("[STEP]"):
                m = RE_STEP.match(line)
                if not m:
                    continue
                step = int(m.group(1))
                frames.setdefault(step, {})
                frames[step]["mode"] = m.group(2)
                mf = RE_STEP_FCMD.search(line)
                if mf:
                    frames[step]["f_cmd"] = _parse_triple(mf.group(1))
            elif line.startswith("[DRAKE-CONTACT]"):
                m = RE_DRAKE.match(line)
                if not m:
                    continue
                step = int(m.group(1))
                frames.setdefault(step, {})
                frames[step]["drake_normal"] = float(m.group(2))
    return frames


# ---------- geometry helpers ----------

def quat_to_yaw(q: np.ndarray) -> float:
    """Extract yaw angle (rad) from quaternion (w, x, y, z). For visualising a
    top-down rectangle. Uses ZYX Euler."""
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """3×3 rotation from quaternion (w, x, y, z)."""
    w, x, y, z = q
    n = w*w + x*x + y*y + z*z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s*(y*y + z*z),     s*(x*y - z*w),     s*(x*z + y*w)],
        [    s*(x*y + z*w), 1 - s*(x*x + z*z),     s*(y*z - x*w)],
        [    s*(x*z - y*w),     s*(y*z + x*w), 1 - s*(x*x + y*y)],
    ])


def box_corners_3d(box_p: np.ndarray, box_q: np.ndarray) -> np.ndarray:
    """8 corners of the box in world frame (3, 8)."""
    R = quat_to_rot(box_q)
    signs = np.array([
        [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
        [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
    ], dtype=float) * BOX_HALF
    return (R @ signs.T + box_p.reshape(3, 1))


BOX_FACES = [
    (0, 1, 2, 3),  # bottom (-z)
    (4, 5, 6, 7),  # top (+z)
    (0, 1, 5, 4),  # -y
    (2, 3, 7, 6),  # +y
    (1, 2, 6, 5),  # +x
    (0, 3, 7, 4),  # -x
]


# ---------- top-down renderer ----------

def render_topdown(frames: Dict[int, dict], out_dir: Path,
                   stride: int = 3) -> int:
    """Render top-down PNGs (goal ★, box □, EE ○, force arrow). Returns count."""
    step_keys = sorted(k for k, v in frames.items()
                       if "box_p" in v and "ee_p" in v)
    all_bx = [frames[k]["box_p"][0] for k in step_keys]
    all_by = [frames[k]["box_p"][1] for k in step_keys]
    all_ex = [frames[k]["ee_p"][0] for k in step_keys]
    all_ey = [frames[k]["ee_p"][1] for k in step_keys]
    xmin = min(min(all_bx), min(all_ex), GOAL[0]) - 0.10
    xmax = max(max(all_bx), max(all_ex), GOAL[0]) + 0.10
    ymin = min(min(all_by), min(all_ey), GOAL[1]) - 0.10
    ymax = max(max(all_by), max(all_ey), GOAL[1]) + 0.10

    # Force axes to look 4:3, centered on plot span
    span_x = xmax - xmin
    span_y = ymax - ymin
    span = max(span_x, span_y * 4/3)
    cx = 0.5 * (xmin + xmax); cy = 0.5 * (ymin + ymax)
    xmin, xmax = cx - span/2, cx + span/2
    ymin, ymax = cy - span * 3/8, cy + span * 3/8

    n = 0
    for i, step in enumerate(step_keys):
        if i % stride != 0:
            continue
        rec = frames[step]
        box_p, box_q = rec["box_p"], rec["box_q"]
        ee_p = rec["ee_p"]
        mode = rec.get("mode", "?")
        drake = rec.get("drake_normal", 0.0)
        f_cmd = rec.get("f_cmd", None)
        yaw = quat_to_yaw(box_q)
        sim_t = step * 0.01

        fig, ax = plt.subplots(figsize=(9.6, 5.4), facecolor="#1a1a1a", dpi=120)
        ax.set_facecolor("#232323")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")

        # Box trail
        trail_x = [frames[k]["box_p"][0] for k in step_keys if k <= step]
        trail_y = [frames[k]["box_p"][1] for k in step_keys if k <= step]
        if len(trail_x) > 1:
            ax.plot(trail_x, trail_y, color="#e8b57e", alpha=0.55,
                    linewidth=1.5, zorder=2)

        # Goal ★
        ax.plot(GOAL[0], GOAL[1], marker="*", color="#ffd700",
                markersize=26, markeredgecolor="#7c5a00",
                markeredgewidth=1, zorder=6)
        ax.text(GOAL[0], GOAL[1] - 0.035, "GOAL", color="#ffd700",
                ha="center", va="top", fontsize=10)

        # Box □ (top-down projection: rotated square)
        box_color = "#ff9f43" if drake > 0.1 else "#f5c56b"
        box_edge  = "#ffffff" if drake > 0.1 else "#c48737"
        box_side = 2 * BOX_HALF
        # rectangle bottom-left in box body frame:
        rect = Rectangle(
            (box_p[0] - BOX_HALF, box_p[1] - BOX_HALF),
            box_side, box_side,
            angle=np.degrees(yaw), rotation_point='center',
            facecolor=box_color, edgecolor=box_edge, linewidth=2.0,
            alpha=0.85, zorder=4)
        ax.add_patch(rect)

        # Face-normal indicator (short arrow from box center to +x body face)
        # Shows box orientation reliably even when yaw looks small.
        R2d = np.array([[np.cos(yaw), -np.sin(yaw)],
                        [np.sin(yaw),  np.cos(yaw)]])
        nvec = R2d @ np.array([BOX_HALF * 1.4, 0.0])
        ax.annotate("", xy=(box_p[0] + nvec[0], box_p[1] + nvec[1]),
                    xytext=(box_p[0], box_p[1]),
                    arrowprops=dict(arrowstyle="-|>", color="#ffffff",
                                    lw=1.2, alpha=0.6), zorder=5)

        # EE ○
        ee_color = "#4dc4ff" if mode == "c3" else "#77bdff"
        ax.add_patch(Circle((ee_p[0], ee_p[1]), PUSHER_R,
                            facecolor=ee_color, edgecolor="#eef8ff",
                            linewidth=1.6, alpha=0.9, zorder=4))

        # f_cmd arrow when in c3 and force ≥ ~2N
        if mode == "c3" and f_cmd is not None:
            fmag = float(np.hypot(f_cmd[0], f_cmd[1]))
            if fmag > 2.0:
                # Arrow scaled: 50N → 6cm arrow
                s = 0.06 / 50.0
                ax.annotate(
                    "",
                    xy=(ee_p[0] + f_cmd[0]*s, ee_p[1] + f_cmd[1]*s),
                    xytext=(ee_p[0], ee_p[1]),
                    arrowprops=dict(arrowstyle="-|>", color="#ff6b6b",
                                    lw=2.4, alpha=0.9), zorder=7)

        # Goal-distance readout + status
        goal_dist = float(np.hypot(box_p[0] - GOAL[0], box_p[1] - GOAL[1]))
        drake_label = f"Drake N: {drake:5.1f} N" if drake > 0.1 else "Drake N:  ---"
        cmode = "C3 push" if mode == "c3" else "reposition"
        ax.text(0.02, 0.97,
                f"t = {sim_t:5.2f} s   |   goal_dist = {goal_dist*1000:5.1f} mm\n"
                f"mode = {cmode:10s}   {drake_label}",
                transform=ax.transAxes, color="#eef", fontsize=11,
                va="top", ha="left", family="monospace",
                bbox=dict(facecolor="#111", edgecolor="#666",
                          boxstyle="round,pad=0.35", alpha=0.85))

        ax.set_xlabel("x (m)  — west ← 0 → east", color="#bbb")
        ax.set_ylabel("y (m)  — south ← 0 → north", color="#bbb")
        ax.tick_params(colors="#bbb")
        for sp in ax.spines.values():
            sp.set_color("#666")
        ax.set_title(
            "§7.73  reference-aligned push  —  seed 0  —  top-down "
            "(box □, EE ○, goal ★, f_cmd →)",
            color="#f0f0f0", fontsize=12, pad=8)
        ax.text(0.98, 0.03,
                "source: full_reference_aligned_772/seed0.log\n"
                "config: B1-A + §7.70 gains + §7.63 channel + bound=50",
                transform=ax.transAxes, color="#8a8", fontsize=8,
                va="bottom", ha="right", family="monospace", alpha=0.85)
        fig.tight_layout()
        fig.savefig(out_dir / f"td_{n:04d}.png",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        n += 1
    return n


# ---------- perspective (3D) renderer ----------

def render_perspective(frames: Dict[int, dict], out_dir: Path,
                       stride: int = 3) -> int:
    """Render 3D PNGs — shows the ~90° forward pitch that top-down flattens."""
    step_keys = sorted(k for k, v in frames.items()
                       if "box_p" in v and "ee_p" in v)
    all_bx = [frames[k]["box_p"][0] for k in step_keys]
    all_by = [frames[k]["box_p"][1] for k in step_keys]
    xmin = min(min(all_bx), GOAL[0]) - 0.10
    xmax = max(max(all_bx), GOAL[0]) + 0.10
    ymin = min(min(all_by), GOAL[1]) - 0.15
    ymax = max(max(all_by), GOAL[1]) + 0.10

    n = 0
    for i, step in enumerate(step_keys):
        if i % stride != 0:
            continue
        rec = frames[step]
        box_p, box_q = rec["box_p"], rec["box_q"]
        ee_p = rec["ee_p"]
        mode = rec.get("mode", "?")
        drake = rec.get("drake_normal", 0.0)
        f_cmd = rec.get("f_cmd", None)
        sim_t = step * 0.01

        fig = plt.figure(figsize=(9.6, 5.4), facecolor="#1a1a1a", dpi=120)
        ax = fig.add_subplot(111, projection="3d",
                             computed_zorder=False)
        ax.set_facecolor("#232323")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(0.0, 0.30)
        ax.view_init(elev=22, azim=-145)

        # Ground plane (semi-transparent)
        gx = np.array([[xmin, xmax], [xmin, xmax]])
        gy = np.array([[ymin, ymin], [ymax, ymax]])
        gz = np.zeros_like(gx)
        ax.plot_surface(gx, gy, gz, color="#2a2a2a", alpha=0.4,
                        edgecolor="#4a4a4a", linewidth=0.4, zorder=0)

        # Goal star (top-down projection onto z=0)
        ax.scatter([GOAL[0]], [GOAL[1]], [0.001], marker="*",
                   color="#ffd700", s=280, zorder=3,
                   edgecolor="#7c5a00", linewidth=0.8)

        # Box faces from quaternion (this is where the pitch shows)
        corners = box_corners_3d(box_p, box_q)  # (3, 8)
        polys = []
        for face in BOX_FACES:
            polys.append([corners[:, idx] for idx in face])
        box_color = "#ff9f43" if drake > 0.1 else "#f5c56b"
        box_edge  = "#ffffff" if drake > 0.1 else "#c48737"
        pc = Poly3DCollection(polys, facecolor=box_color, edgecolor=box_edge,
                              linewidths=1.0, alpha=0.9)
        pc.set_zorder(2)
        ax.add_collection3d(pc)

        # EE sphere (approx as a small marker + wireframe sphere)
        u_sph, v_sph = np.mgrid[0:2*np.pi:14j, 0:np.pi:7j]
        sx = ee_p[0] + PUSHER_R * np.cos(u_sph) * np.sin(v_sph)
        sy = ee_p[1] + PUSHER_R * np.sin(u_sph) * np.sin(v_sph)
        sz = ee_p[2] + PUSHER_R * np.cos(v_sph)
        ee_color = "#4dc4ff" if mode == "c3" else "#77bdff"
        ax.plot_surface(sx, sy, sz, color=ee_color, alpha=0.65,
                        linewidth=0.4, edgecolor="#eef8ff", zorder=4)

        # Force arrow (3D)
        if mode == "c3" and f_cmd is not None:
            fmag = float(np.linalg.norm(f_cmd))
            if fmag > 2.0:
                s = 0.06 / 50.0
                ax.quiver(ee_p[0], ee_p[1], ee_p[2],
                          f_cmd[0]*s, f_cmd[1]*s, f_cmd[2]*s,
                          color="#ff6b6b", linewidth=2.0,
                          arrow_length_ratio=0.28, zorder=5)

        goal_dist = float(np.hypot(box_p[0] - GOAL[0], box_p[1] - GOAL[1]))
        drake_label = f"Drake N: {drake:5.1f} N" if drake > 0.1 else "Drake N:  ---"
        cmode = "C3 push" if mode == "c3" else "reposition"
        ax.text2D(0.02, 0.97,
                  f"t = {sim_t:5.2f} s   |   goal_dist = {goal_dist*1000:5.1f} mm\n"
                  f"mode = {cmode:10s}   {drake_label}",
                  transform=ax.transAxes, color="#eef", fontsize=11,
                  va="top", ha="left", family="monospace",
                  bbox=dict(facecolor="#111", edgecolor="#666",
                            boxstyle="round,pad=0.35", alpha=0.85))

        ax.set_xlabel("x (m)", color="#bbb")
        ax.set_ylabel("y (m)", color="#bbb")
        ax.set_zlabel("z (m)", color="#bbb")
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_tick_params(colors="#bbb")
        ax.set_title(
            "§7.73  reference-aligned push  —  seed 0  —  perspective "
            "(shows the ~90° forward roll)",
            color="#f0f0f0", fontsize=12, pad=6)
        ax.text2D(0.98, 0.03,
                  "source: full_reference_aligned_772/seed0.log\n"
                  "config: B1-A + §7.70 gains + §7.63 channel + bound=50",
                  transform=ax.transAxes, color="#8a8", fontsize=8,
                  va="bottom", ha="right", family="monospace",
                  alpha=0.85)

        fig.tight_layout()
        fig.savefig(out_dir / f"pv_{n:04d}.png",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        n += 1
    return n


# ---------- ffmpeg ----------

def ffmpeg_encode(pattern: str, out_path: Path, fps: int = 30) -> None:
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-framerate", str(fps),
        "-i", pattern,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "20", "-preset", "medium",
        "-movflags", "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path,
                    help="Path to the seed 0 reference-aligned log")
    ap.add_argument("--sink", type=Path,
                    default=Path("/d/projects/ERL/push_anything_ADMM"),
                    help="Output mp4 directory (default: Windows-side sink)")
    ap.add_argument("--stride", type=int, default=3,
                    help="Sim-tick stride (3 → ~200 frames of 6-s run → "
                         "30-fps mp4 plays close to real-time)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--basename", type=str,
                    default="seed0_reference_aligned")
    args = ap.parse_args()

    frames = parse_log(args.log)
    n_pose = sum(1 for v in frames.values() if "box_p" in v)
    print(f"[parse] {len(frames)} step keys, {n_pose} with pose",
          flush=True)
    if n_pose < 100:
        print("ERROR: insufficient pose data", file=sys.stderr)
        sys.exit(1)

    args.sink.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_root:
        tmp = Path(tmp_root)
        td_dir = tmp / "td"
        pv_dir = tmp / "pv"
        td_dir.mkdir()
        pv_dir.mkdir()

        print("[render] top-down frames ...", flush=True)
        n_td = render_topdown(frames, td_dir, stride=args.stride)
        print(f"[render] wrote {n_td} top-down PNGs", flush=True)

        print("[render] perspective frames ...", flush=True)
        n_pv = render_perspective(frames, pv_dir, stride=args.stride)
        print(f"[render] wrote {n_pv} perspective PNGs", flush=True)

        td_mp4 = args.sink / f"{args.basename}_topdown.mp4"
        pv_mp4 = args.sink / f"{args.basename}_perspective.mp4"
        print(f"[encode] {td_mp4}", flush=True)
        ffmpeg_encode(str(td_dir / "td_%04d.png"), td_mp4, fps=args.fps)
        print(f"[encode] {pv_mp4}", flush=True)
        ffmpeg_encode(str(pv_dir / "pv_%04d.png"), pv_mp4, fps=args.fps)

        for p in (td_mp4, pv_mp4):
            sz = p.stat().st_size
            print(f"[done] {p}  ({sz/1024:.0f} KB)", flush=True)


if __name__ == "__main__":
    main()
