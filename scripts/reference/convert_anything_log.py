#!/usr/bin/env python3
"""Convert native DAIRLab ``anything`` [TRAJ] telemetry for video tools."""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np


TRAJ = re.compile(
    r"^\[TRAJ\] t=([\d.eE+-]+) wall=([\d.eE+-]+) obj_pos=\s*"
    r"([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+) obj_quat=\s*"
    r"([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)"
)
ARM_Q = re.compile(r"^\[ARM-Q\] t=([\d.eE+-]+) q=\s*(.*)$")
PARKED_EE = np.array([0.30, -0.45, 0.18])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--goal-xy", required=True, nargs=2, type=float)
    parser.add_argument("--goal-quat", required=True, nargs=4, type=float)
    parser.add_argument("--object", default="I_shape_texture")
    parser.add_argument("--arm-log", type=Path)
    parser.add_argument("--scene")
    parser.add_argument("--scene-sdf", type=Path)
    args = parser.parse_args()

    records = []
    succeeded = False
    success_record_count = None
    with args.source.open(errors="replace") as stream:
        for line in stream:
            match = TRAJ.match(line)
            if match:
                values = [float(value) for value in match.groups()]
                records.append((values[0], np.array(values[2:5]),
                                np.array(values[5:9])))
            if ("All objects on target" in line or
                    "All objects at fixed goals" in line):
                succeeded = True
                if success_record_count is None:
                    success_record_count = len(records)
    if not records:
        raise SystemExit("no [TRAJ] records found")
    # Native harnesses continue running after the fixed goal is reached.
    # A result video should end at that first outcome, not append a long
    # post-success hold which makes time/step telemetry misleading.
    if success_record_count:
        records = records[:success_record_count]

    arm_records = []
    if args.arm_log:
        with args.arm_log.open(errors="replace") as stream:
            for line in stream:
                match = ARM_Q.match(line)
                if match:
                    q = np.fromstring(match.group(2), sep=" ")
                    if q.size == 7:
                        arm_records.append((float(match.group(1)), q))

    goal = np.asarray(args.goal_xy)
    goal_quat = np.asarray(args.goal_quat, dtype=float)
    goal_quat /= max(float(np.linalg.norm(goal_quat)), 1e-12)

    def planar_yaw(quat):
        w, x, y, z = quat
        return math.atan2(2.0 * (w * z + x * y),
                          1.0 - 2.0 * (y * y + z * z))

    goal_yaw = planar_yaw(goal_quat)
    lines = [
        "[RUN-META] git=DAIRLab-257e3ede "
        f"task={args.object} flags=[solver=c3plus native_cpp=1 "
        f"success={int(succeeded)}]",
        f"[ENV]  Goal coords: [{goal[0]}, {goal[1]}]",
        "[GOAL-QUAT] goal_quat=[{}]".format(
            " ".join(str(value) for value in args.goal_quat)),
        "[TASK] Native DAIRLab C++ C3+ via multiyaml import adapter",
        "[TASK] ARM NOT RECORDED by compact reference telemetry; shown parked",
    ]
    if args.scene:
        lines.append(f"[SCENARIO] OIM {args.scene} | T object | native C3+")
    if arm_records:
        lines[-1] = "[TASK] Native Franka joint telemetry replay enabled"
    if args.scene and args.scene_sdf:
        lines.append(
            f"[OIM-SCENE] name={args.scene} sdf={args.scene_sdf.resolve()}")
    arm_i = 0
    for step, (sim_t, pos, quat) in enumerate(records, 1):
        quat = quat / max(float(np.linalg.norm(quat)), 1e-12)
        distance = float(np.linalg.norm(pos[:2] - goal))
        theta_error = abs(math.atan2(
            math.sin(planar_yaw(quat) - goal_yaw),
            math.cos(planar_yaw(quat) - goal_yaw),
        ))
        lines.append(
            f"[STEP] step={step} mode=c3 t={sim_t:.3f}s "
            f"ee=({PARKED_EE[0]:+.4f},{PARKED_EE[1]:+.4f},{PARKED_EE[2]:+.4f}) "
            f"obj=({pos[0]:+.5f},{pos[1]:+.5f},{pos[2]:+.5f}) "
            f"goal_dist={distance:.5f}m switch=kStayInC3 "
            f"rot_err={theta_error:.5f}rad")
        lines.append(
            f"[GATE-CONTACT] step={step} F_W=(+0,+0,+0) F_on_box=(+0,+0,+0) "
            f"n_face_out=(+0,+0,+1) A_is_ee=0 "
            f"box_q=({quat[0]:+.7f},{quat[1]:+.7f},{quat[2]:+.7f},{quat[3]:+.7f}) "
            f"box_p=({pos[0]:+.7f},{pos[1]:+.7f},{pos[2]:+.7f}) "
            f"ee_p=({PARKED_EE[0]:+.7f},{PARKED_EE[1]:+.7f},{PARKED_EE[2]:+.7f})")
        if arm_records:
            while (arm_i + 1 < len(arm_records) and
                   abs(arm_records[arm_i + 1][0] - sim_t) <=
                   abs(arm_records[arm_i][0] - sim_t)):
                arm_i += 1
            q_text = ",".join(f"{v:+.8f}" for v in arm_records[arm_i][1])
            lines.append(f"[ARM-Q] step={step} q=({q_text})")
    if succeeded:
        sim_t, pos, quat = records[-1]
        quat = quat / max(float(np.linalg.norm(quat)), 1e-12)
        distance = float(np.linalg.norm(pos[:2] - goal))
        theta_error = abs(math.atan2(
            math.sin(planar_yaw(quat) - goal_yaw),
            math.cos(planar_yaw(quat) - goal_yaw),
        ))
        lines.append(
            "[GOAL-GEN] goal #1 REACHED (native fixed-goal marker) "
            f"at t={sim_t:.1f}s step={len(records)}")
        lines.append(
            f"[RESULT] SUCCESS scenario={args.scene or 'unknown'} "
            f"pos_err={distance:.5f}m theta_err={theta_error:.5f}rad "
            f"execution_time={sim_t:.1f}s steps_to_goal={len(records)}")
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}: {len(records)} frames success={succeeded}")


if __name__ == "__main__":
    main()
