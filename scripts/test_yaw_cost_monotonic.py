"""Unit test: verify yaw-error cost decreases monotonically as box yaw -> goal_yaw.

Builds a QuadraticManipulationCost with target_yaw=+45deg, w_yaw=10, w_obj_xy=0,
then evaluates the quadratic cost  (x - x_ref)^T Q (x - x_ref)  at three states
where only the box quaternion differs:
  (a) yaw=0        (far from goal)   -> highest yaw cost
  (b) yaw=+22.5deg (halfway)         -> medium
  (c) yaw=+45deg   (at goal)         -> lowest (~0)

The cost is exactly  w_yaw * sin^2((psi - alpha)/2)  for upright yaw-only states,
so we expect 10 * sin^2(pi/8), 10 * sin^2(pi/16), 0 = ~1.464, ~0.381, 0.

GATE: assert c0 > c1 > c2.
"""
import sys
import numpy as np

sys.path.insert(0, ".")

from main import load_task, EE_BODY_NAME
from sim.env_builder import build_environment
from control.task_costs import QuadraticManipulationCost


def quat_yaw(psi: float) -> np.ndarray:
    """Yaw-only rotation about world z: returns [qw, qx, qy, qz]."""
    return np.array([np.cos(psi / 2), 0.0, 0.0, np.sin(psi / 2)])


def main():
    # Build a Drake plant via the cube_turning task config (just for plant + obj_body).
    task_cfg = load_task("cube_turning")
    diagram, plant, panda_model, object_model, meshcat, plant_ad, context_ad = \
        build_environment(task_cfg)

    obj_body = plant.GetBodyByName(task_cfg["link_name"])
    n_q = plant.num_positions()
    n_v = plant.num_velocities()
    n_u = plant.num_actuators()
    n_x = n_q + n_v
    print(f"[INFO] n_q={n_q} n_v={n_v} n_u={n_u} n_x={n_x}")

    cost_cfg = dict(task_cfg["cost"])
    print(f"[INFO] cost_cfg w_yaw={cost_cfg['w_yaw']} w_obj_xy={cost_cfg['w_obj_xy']}")

    cost = QuadraticManipulationCost(
        plant=plant, ee_frame_name=EE_BODY_NAME, obj_body=obj_body,
        cost_cfg=cost_cfg, n_x=n_x, n_u=n_u,
    )

    ps = cost._obj_ps
    target_yaw = float(task_cfg["goal_yaw"])      # pi/4
    target_xy  = np.array(task_cfg["goal_xy"], dtype=float)  # (0,0)
    print(f"[INFO] target_yaw={target_yaw:.4f} rad ({np.degrees(target_yaw):.1f} deg)")

    Q, R, QN, x_ref = cost.build(target_xy, target_yaw=target_yaw)
    print(f"[INFO] x_ref quaternion slots: "
          f"qw={x_ref[ps+0]:+.4f} qx={x_ref[ps+1]:+.4f} "
          f"qy={x_ref[ps+2]:+.4f} qz={x_ref[ps+3]:+.4f}")
    print(f"[INFO] Q quat-block 4x4:")
    print(Q[ps:ps+4, ps:ps+4])

    def make_state(psi: float) -> np.ndarray:
        x = np.zeros(n_x)
        q_box = quat_yaw(psi)
        x[ps + 0] = q_box[0]
        x[ps + 1] = q_box[1]
        x[ps + 2] = q_box[2]
        x[ps + 3] = q_box[3]
        x[cost._obj_x_idx] = target_xy[0]
        x[cost._obj_y_idx] = target_xy[1]
        x[cost._obj_z_idx] = cost.z_ref
        return x

    yaws = [0.0, np.pi / 8, np.pi / 4]
    labels = ["yaw=0.00 (far)", "yaw=pi/8 (halfway)", "yaw=pi/4 (at goal)"]
    costs = []
    for psi, lbl in zip(yaws, labels):
        x = make_state(psi)
        dx = x - x_ref
        c = float(dx @ Q @ dx)
        predicted = float(cost.w_yaw) * np.sin((psi - target_yaw) / 2) ** 2
        costs.append(c)
        print(f"[COST] {lbl:25s}: cost={c:.6f}  predicted={predicted:.6f}")

    c0, c1, c2 = costs
    print()
    print(f"yaw=0   cost: {c0:.6f}")
    print(f"yaw=22  cost: {c1:.6f}")
    print(f"yaw=45  cost: {c2:.6f}")

    assert c0 > c1 > c2, (
        f"yaw cost does not decrease monotonically — INVALID. "
        f"Got {c0:.6f}, {c1:.6f}, {c2:.6f}"
    )
    print()
    print("PASS: yaw cost decreases monotonically toward goal_yaw")


if __name__ == "__main__":
    main()
