"""
Forward-kinematics checker for candidate Franka arm poses.

Checks the spherical pusher body (rigidly welded to panda_link8 at +5 cm),
which is the dedicated contact end-effector for all three tasks.

Usage (from project root):
    python scripts/check_pose.py

For each candidate pose prints:
  - pusher centre position in world frame
  - pusher_z  (target: > 0.025 m to clear table; ideal ≈ 0.05 m for box mid-height)
  - pusher–box XY distance  (target: < 0.35 m, arm extended toward box)
  - panda_link8 position for reference

Pass criteria (all must hold):
  ✓  pusher_z  ≥ 0.025 m   (puck above table surface — sphere radius = 0.025)
  ✓  pusher_z  ≤ 0.10 m    (not too high for box contact — box top at z=0.10)
  ✓  pusher–box XY dist < 0.35 m
"""
import sys
import numpy as np

sys.path.insert(0, ".")   # run from project root

import yaml
import pydrake.all as ad
from sim.env_builder import build_environment, EE_BODY_NAME


def load_task_cfg(task_name: str = "pushing") -> dict:
    with open("config/tasks.yaml") as f:
        return yaml.safe_load(f)["tasks"][task_name]


def check_poses():
    task_cfg = load_task_cfg("pushing")
    diagram, plant, _, _, _ = build_environment(task_cfg)
    ctx       = diagram.CreateDefaultContext()
    plant_ctx = plant.GetMyMutableContextFromRoot(ctx)

    n_q = plant.num_positions()

    # Box floating-base initial positions (qw,qx,qy,qz,x,y,z)
    box_q = np.array([1.0, 0.0, 0.0, 0.0,
                      task_cfg["init_xyz"][0],
                      task_cfg["init_xyz"][1],
                      task_cfg["init_xyz"][2]])

    pusher_body = plant.GetBodyByName(EE_BODY_NAME)
    link8_body  = plant.GetBodyByName("panda_link8")

    candidates = {
        "current (confirmed working)": [0.0,  0.4,  0.0, -2.8,  0.0,  3.2,  0.785],
        "slightly_higher_1":           [0.0,  0.3,  0.0, -2.8,  0.0,  3.2,  0.785],
        "slightly_higher_2":           [0.0,  0.2,  0.0, -2.8,  0.0,  3.2,  0.785],
        "hand_down_1":                 [0.0,  0.4,  0.0, -2.4,  0.0,  2.8,  0.785],
        "hand_down_2":                 [0.0,  0.5,  0.0, -2.2,  0.0,  2.4,  0.785],
        "hand_forward_1":              [0.0,  0.4,  0.0, -2.6,  0.0,  2.6,  0.0  ],
        "hand_forward_2":              [0.0,  0.5,  0.0, -2.4,  0.0,  2.2,  0.0  ],
        "hand_forward_3":              [0.0,  0.3,  0.0, -2.6,  0.0,  3.0,  0.0  ],
    }

    box_xy    = np.array([task_cfg["init_xyz"][0], task_cfg["init_xyz"][1]])
    box_top_z = task_cfg["init_xyz"][2] * 2.0    # box CoM at z=0.05 → top at z=0.10
    pusher_r  = 0.025                             # sphere radius

    print(f"Box at init_xyz={task_cfg['init_xyz']}  box_top_z={box_top_z:.3f}")
    print(f"Pusher radius: {pusher_r} m  "
          f"(pusher_z must be ≥{pusher_r} above table, ideally ≈0.05)")
    print(f"Robot base at Y=-0.6 (X=0, Z=0)")
    print("=" * 80)

    for name, arm_q in candidates.items():
        finger_q = np.array([0.04, 0.04])
        q_full   = np.concatenate([arm_q, finger_q, box_q])
        if len(q_full) != n_q:
            # If Panda arm has no hand/finger DOFs (panda_arm.urdf only has 7)
            q_full = np.concatenate([arm_q, box_q])
        if len(q_full) != n_q:
            print(f"[SKIP] {name}: q_full len {len(q_full)} != n_q {n_q}")
            continue

        plant.SetPositions(plant_ctx, q_full)

        pusher_pos = plant.EvalBodyPoseInWorld(plant_ctx, pusher_body).translation()
        link8_pos  = plant.EvalBodyPoseInWorld(plant_ctx, link8_body).translation()

        pusher_z         = float(pusher_pos[2])
        pusher_box_dist  = float(np.linalg.norm(pusher_pos[:2] - box_xy))

        z_ok   = pusher_r <= pusher_z <= box_top_z
        d_ok   = pusher_box_dist < 0.35
        above  = pusher_z >= pusher_r   # puck doesn't hit table

        verdict = "✓ CANDIDATE" if (z_ok and d_ok) else "✗"
        print(f"{verdict}  [{name}]")
        print(f"  arm_q   = {arm_q}")
        print(f"  pusher  xyz = {pusher_pos.round(4)}")
        print(f"  link8   xyz = {link8_pos.round(4)}")
        print(f"  pusher_z    = {pusher_z:.4f}  "
              f"(want {pusher_r:.3f}–{box_top_z:.3f}) {'✓' if z_ok else '✗'}")
        print(f"  pusher–box  = {pusher_box_dist:.3f} m  (want <0.35) {'✓' if d_ok else '✗'}")
        if pusher_z < pusher_r:
            print(f"  *** COLLISION WITH TABLE: pusher_z={pusher_z:.4f} < r={pusher_r} ***")
        print()


if __name__ == "__main__":
    check_poses()
