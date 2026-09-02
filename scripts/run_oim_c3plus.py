#!/usr/bin/env python3
"""Run the existing C3+ consensus solver in a wrapped OIM Drake scene."""

from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control.oim_c3plus_architecture import build_oim_sampling_c3plus_architecture
from sim.oim_xarm6_tabletop import (
    SCENE_NAMES,
    oim_goal_reached,
    register_oim_pose_for_dairlab,
)


SCENE_STEP_BUDGETS = {
    "open_table": 8_000,
    "single_obstacle": 10_000,
    "shelf_gap": 12_000,
    "ycb_clutter": 16_000,
    "icra_sign": 12_000,
}


def wrapped_angle(value: float) -> float:
    return float(np.arctan2(np.sin(value), np.cos(value)))


def render_rollout(states, output: Path, dt: float,
                   compute_times=None, scene="open_table") -> None:
    """Render the OIM rollout with a dedicated controller side panel."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.patches import Circle, Polygon

    fig = plt.figure(figsize=(12.8, 7.2), dpi=120)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.75, 1.0), wspace=0.12)
    ax = fig.add_subplot(grid[0, 0])
    panel = fig.add_subplot(grid[0, 1])
    writer = FFMpegWriter(fps=max(5, min(30, round(1.0 / dt))), bitrate=1800)
    object_path = np.array([s.object_pose[:2] for s in states])
    tip_path = np.array([s.tip_position[:2] for s in states])
    goal = np.asarray(states[0].goal_pose, dtype=float)
    position_errors = np.array([
        np.linalg.norm(np.asarray(s.object_pose, dtype=float)[:2] - goal[:2])
        for s in states
    ])
    orientation_errors = np.array([
        abs(wrapped_angle(float(s.object_pose[2]) - goal[2])) for s in states
    ])
    compute_times = ([] if compute_times is None else list(compute_times))
    # Exact top-view union outline of tee.xml's two collision boxes:
    # crossbar size=(.0445,.0099) at y=.0099 and stem
    # size=(.0099,.0397) at y=-.0397 (MuJoCo sizes are half-extents).
    t_footprint = np.array([
        [-0.0445, 0.0000], [-0.0445, 0.0198],
        [ 0.0445, 0.0198], [ 0.0445, 0.0000],
        [ 0.0099, 0.0000], [ 0.0099, -0.0794],
        [-0.0099, -0.0794], [-0.0099, 0.0000],
    ])

    def transformed_t(pose):
        x, y, yaw = pose
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)],
                             [np.sin(yaw), np.cos(yaw)]])
        return t_footprint @ rotation.T + np.array([x, y])

    with writer.saving(fig, str(output), dpi=120):
        for index, state in enumerate(states):
            ax.clear()
            panel.clear()
            ax.set_xlim(0.0, 0.8)
            ax.set_ylim(-0.55, 0.55)
            ax.set_aspect("equal")
            ax.set_facecolor("#e8e2d3")
            ax.set_title(f"OIM {scene.replace('_', ' ').title()} | Sampling-C3+")
            ax.set_xlabel("world x [m]")
            ax.set_ylabel("world y [m]")
            ax.grid(alpha=0.18)
            ax.plot(object_path[:index + 1, 0], object_path[:index + 1, 1],
                    color="#1677ff", lw=2, label="object path")
            ax.plot(tip_path[:index + 1, 0], tip_path[:index + 1, 1],
                    color="#7b2cbf", lw=1.5, label="stick-tip path")
            ax.add_patch(Circle(goal[:2], 0.05, fill=False, ls="--", lw=2,
                                color="#16a34a", label="goal tolerance"))
            ax.add_patch(Polygon(transformed_t(goal), fill=False, ls="--",
                                 lw=2, edgecolor="#16a34a"))
            ax.add_patch(Polygon(transformed_t(state.object_pose),
                                 color="#60a5fa", alpha=0.9))
            ax.add_patch(Circle(state.tip_position[:2], 0.009,
                                color="#7b2cbf"))
            ax.legend(loc="lower left")

            pos_error = position_errors[index]
            theta_error = orientation_errors[index]
            successful = pos_error <= 0.05 and theta_error <= 0.1
            panel.set_facecolor("#111827")
            panel.set_xlim(0.0, 1.0)
            panel.set_ylim(0.0, 1.0)
            panel.set_xticks([])
            panel.set_yticks([])
            for spine in panel.spines.values():
                spine.set_visible(False)
            panel.text(0.07, 0.94, "RUN TELEMETRY", color="white",
                       fontsize=17, weight="bold", va="top")
            panel.text(0.07, 0.885,
                       f"step {index:04d}   t = {index * dt:6.2f} s",
                       color="#cbd5e1", fontsize=11, family="monospace")
            status_color = "#22c55e" if successful else "#f59e0b"
            panel.text(0.07, 0.82,
                       "GOAL REACHED" if successful else "RUNNING",
                       color=status_color, fontsize=14, weight="bold")
            panel.text(0.07, 0.745,
                       f"position error     {pos_error:7.3f} m\n"
                       f"position tolerance {0.05:7.3f} m\n\n"
                       f"orientation error  {theta_error:7.3f} rad\n"
                       f"orientation tol.    {0.10:7.3f} rad",
                       color="white", fontsize=11, family="monospace",
                       va="top", linespacing=1.35)
            panel.text(0.07, 0.49,
                       f"object  x {state.object_pose[0]:+.3f}\n"
                       f"        y {state.object_pose[1]:+.3f}\n"
                       f"      yaw {state.object_pose[2]:+.3f}\n\n"
                       f"goal    x {goal[0]:+.3f}\n"
                       f"        y {goal[1]:+.3f}\n"
                       f"      yaw {goal[2]:+.3f}",
                       color="#cbd5e1", fontsize=10.5, family="monospace",
                       va="top", linespacing=1.25)
            if compute_times and index > 0:
                rate = 1.0 / max(np.mean(compute_times[:index]), 1e-12)
                panel.text(0.07, 0.245, f"mean planning rate  {rate:5.2f} Hz",
                           color="#93c5fd", fontsize=10.5, family="monospace")

            inset = panel.inset_axes([0.09, 0.055, 0.84, 0.14])
            elapsed = np.arange(index + 1) * dt
            inset.plot(elapsed, position_errors[:index + 1],
                       color="#60a5fa", lw=1.8, label="position")
            inset.plot(elapsed, orientation_errors[:index + 1],
                       color="#f59e0b", lw=1.8, label="orientation")
            inset.axhline(0.05, color="#60a5fa", ls="--", lw=0.8, alpha=0.7)
            inset.axhline(0.10, color="#f59e0b", ls="--", lw=0.8, alpha=0.7)
            inset.set_xlim(0, max(dt, (len(states) - 1) * dt))
            inset.set_ylim(0, max(0.2, float(max(position_errors.max(),
                                                 orientation_errors.max())) * 1.05))
            inset.tick_params(labelsize=7, colors="#cbd5e1")
            inset.set_facecolor("#1f2937")
            for spine in inset.spines.values():
                spine.set_color("#475569")
            inset.legend(loc="upper right", fontsize=7, framealpha=0.4)
            writer.grab_frame()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=SCENE_NAMES, default="open_table")
    parser.add_argument(
        "--steps", type=int,
        help=("maximum control steps; omitted uses the scenario budget "
              "(open_table=8000, single_obstacle=10000, shelf_gap=12000, "
              "ycb_clutter=16000, icra_sign=12000); 0 runs until success "
              "or runtime failure"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start", help="OIM start-pose key; unset draws by seed")
    parser.add_argument("--goal", help="OIM goal-pose key; unset draws by seed")
    args = parser.parse_args()
    if args.steps is not None and args.steps < 0:
        parser.error("--steps must be nonnegative (0 means unlimited)")
    if args.steps is None:
        args.steps = SCENE_STEP_BUDGETS[args.scene]

    output_dir = args.output_dir
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    poses_path = (Path(__file__).resolve().parents[1] / "external" /
                  "Object-Informed-Manipulation-MJX" / "examples" /
                  "poses" / f"{args.scene}.yaml")
    start_index = goal_index = None
    raw_start_pose = raw_goal_pose = None
    start_pose = goal_pose = None
    if poses_path.is_file():
        pose_data = yaml.safe_load(poses_path.read_text())
        rng = np.random.default_rng(args.seed)
        start_keys = tuple(str(k) for k in pose_data["starts"])
        goal_keys = tuple(str(k) for k in pose_data["goals"])
        start_index = (str(args.start) if args.start is not None else
                       start_keys[int(rng.integers(len(start_keys)))])
        goal_index = (str(args.goal) if args.goal is not None else
                      goal_keys[int(rng.integers(len(goal_keys)))])
        if start_index not in pose_data["starts"]:
            parser.error(f"unknown start {start_index!r}; choices: {start_keys}")
        if goal_index not in pose_data["goals"]:
            parser.error(f"unknown goal {goal_index!r}; choices: {goal_keys}")
        raw_start_pose = np.asarray(
            pose_data["starts"][start_index], dtype=float
        )
        raw_goal_pose = np.asarray(pose_data["goals"][goal_index], dtype=float)
        start_pose = register_oim_pose_for_dairlab(raw_start_pose)
        goal_pose = register_oim_pose_for_dairlab(raw_goal_pose)
    architecture = build_oim_sampling_c3plus_architecture(
        args.scene, start_pose=start_pose, goal_pose=goal_pose
    )
    compute_times = []
    states = [architecture.adapter.observe()]
    records = []
    solver_trace = io.StringIO()
    first_success = None
    failure = None
    step_numbers = (itertools.count(1) if args.steps == 0
                    else range(1, args.steps + 1))
    for step in step_numbers:
        start = time.perf_counter()
        try:
            with contextlib.redirect_stdout(solver_trace):
                states.append(architecture.step())
        except RuntimeError as exc:
            compute_times.append(time.perf_counter() - start)
            failure = str(exc)
            solver_trace.write(f"[RUN-ABORT] step={step} {failure}\n")
            break
        compute_times.append(time.perf_counter() - start)
        state = states[-1]
        pos_error = float(np.linalg.norm(state.object_pose[:2] - state.goal_pose[:2]))
        theta_error = abs(wrapped_angle(state.object_pose[2] - state.goal_pose[2]))
        records.append({
            "step": step,
            "sim_time": step * architecture.config.planning_dt,
            "compute_time": compute_times[-1],
            "object_pose": state.object_pose.tolist(),
            "goal_pose": state.goal_pose.tolist(),
            "arm_positions": state.arm_positions.tolist(),
            "arm_velocities": state.arm_velocities.tolist(),
            "tip_position": state.tip_position.tolist(),
            "ee_force": architecture.controller.last_ee_force_command.tolist(),
            "joint_velocity_command": architecture.controller._last_u.tolist(),
            "position_error": pos_error,
            "orientation_error": theta_error,
            "qp_failures": architecture.solver.qp_failures,
        })
        if first_success is None and oim_goal_reached(pos_error, theta_error):
            first_success = step
            break

    pos_errors = np.array([
        np.linalg.norm(s.object_pose[:2] - s.goal_pose[:2]) for s in states
    ])
    theta_errors = np.array([
        abs(wrapped_angle(s.object_pose[2] - s.goal_pose[2])) for s in states
    ])
    result = {
        "scene": args.scene,
        "seed": args.seed,
        "start_index": start_index,
        "goal_index": goal_index,
        "pose_frame": "dairlab_registered_x_plus_0.10_y_times_0.50",
        "raw_start_pose": (None if raw_start_pose is None
                           else raw_start_pose.tolist()),
        "raw_goal_pose": (None if raw_goal_pose is None
                          else raw_goal_pose.tolist()),
        "start_pose": None if start_pose is None else start_pose.tolist(),
        "goal_pose": None if goal_pose is None else goal_pose.tolist(),
        "success": first_success is not None,
        "eps_d": float(np.mean(pos_errors)),
        "theta": float(np.mean(theta_errors)),
        "steps": (first_success if first_success is not None
                  else len(states) - 1),
        "steps_run": len(states) - 1,
        "f_hz": float(1.0 / np.mean(compute_times)),
        "T_s": float((len(states) - 1) * architecture.config.planning_dt),
        "final_position_error": float(pos_errors[-1]),
        "final_orientation_error": float(theta_errors[-1]),
        "aborted": failure is not None,
        "failure": failure,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if output_dir is not None:
        (output_dir / "summary.json").write_text(rendered + "\n")
        (output_dir / "steps.jsonl").write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
        )
        (output_dir / "solver.log").write_text(solver_trace.getvalue())
        if args.video:
            subprocess.run([
                sys.executable,
                str(Path(__file__).with_name("render_oim_xarm_rollout.py")),
                str(output_dir),
                "--output", str(output_dir / "rollout.mp4"),
            ], check=True)
    print(rendered)


if __name__ == "__main__":
    main()
