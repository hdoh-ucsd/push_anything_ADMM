#!/usr/bin/env python3
"""Render one presentation-ready video per goal in a consecutive session."""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DT = 0.075
BOOT_RE = re.compile(
    r"goal #1 .*?xy=\(([+-]?[\d.]+),([+-]?[\d.]+)\) "
    r"quat=\[([+-]?[\d.]+) [+-]?[\d.]+ [+-]?[\d.]+ ([+-]?[\d.]+)\]")
REACHED_RE = re.compile(
    r"goal #(\d+) REACHED at t=([\d.]+)s -> new goal "
    r"xy=\(([+-]?[\d.]+),([+-]?[\d.]+)\).*?"
    r"quat=\[([+-]?[\d.]+) [+-]?[\d.]+ [+-]?[\d.]+ ([+-]?[\d.]+)\]")


def yaw(w: float, z: float) -> float:
    return math.atan2(2.0 * w * z, 1.0 - 2.0 * z * z)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--target-frames", type=int, default=120)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    text = args.log.read_text(errors="replace")
    boot = BOOT_RE.search(text)
    reached = list(REACHED_RE.finditer(text))
    if boot is None or [int(m.group(1)) for m in reached] != list(range(1, 29)):
        raise RuntimeError("log is not a complete sequential 28-goal session")

    goals = [(float(boot.group(1)), float(boot.group(2)),
              yaw(float(boot.group(3)), float(boot.group(4))))]
    # The goal drawn after reaching N is the target for segment N+1.
    goals.extend((float(m.group(3)), float(m.group(4)),
                  yaw(float(m.group(5)), float(m.group(6))))
                 for m in reached[:-1])
    ends = [float(m.group(2)) for m in reached]
    starts = [0.0, *ends[:-1]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    env = os.environ.copy()
    env.setdefault("PORT_CAM_EYE", "0.85,-0.38,0.38")
    env.setdefault("PORT_CAM_TARGET", "0.50,0.05,0.035")

    for number, (start, end, goal) in enumerate(zip(starts, ends, goals), 1):
        output = args.output_dir / f"goal_{number:02d}.mp4"
        start_step = max(0, round(start / DT))
        end_step = round(end / DT)
        stride = max(1, math.ceil((end_step - start_step + 1) / args.target_frames))
        rows.append({
            "goal": number, "start_s": f"{start:.3f}", "end_s": f"{end:.3f}",
            "time_to_goal_s": f"{end-start:.3f}", "target_x": f"{goal[0]:.4f}",
            "target_y": f"{goal[1]:.4f}", "target_yaw_rad": f"{goal[2]:.6f}",
            "video": output.name,
        })
        if output.exists() and not args.force:
            print(f"keeping {output}")
            continue
        with tempfile.TemporaryDirectory(prefix=f"goal-{number:02d}-", dir="/tmp") as tmp:
            frames = Path(tmp)
            subprocess.run([
                sys.executable, "tools/visualizer/render_log_drake_scene.py",
                str(args.log), "--task", args.task, "--out-dir", str(frames),
                "--min-step", str(start_step), "--max-step", str(end_step),
                "--stride", str(stride), "--goal-xy", f"{goal[0]},{goal[1]}",
                "--goal-yaw", str(goal[2]),
            ], cwd=ROOT, env=env, check=True)
            subprocess.run([
                sys.executable, "tools/visualizer/paint_log_sidepanel.py",
                "--frames-dir", str(frames), "--log-path", str(args.log),
                "--output", str(output), "--fps", str(args.fps),
            ], cwd=ROOT, check=True)
        print(f"rendered {output} ({end-start:.3f}s segment)")

    with (args.output_dir / "goal_video_manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
