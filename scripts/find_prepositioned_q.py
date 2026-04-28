"""Find INITIAL_ARM_Q such that the pusher touches the box's west face."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from pydrake.all import InverseKinematics, Solve
from sim.env_builder import build_environment, EE_BODY_NAME

import yaml
with open(Path(__file__).resolve().parent.parent / "config" / "tasks.yaml") as f:
    task_cfg = yaml.safe_load(f)["tasks"]["pushing"]

diagram, plant, _, _, _ = build_environment(task_cfg)
ctx = diagram.CreateDefaultContext()
plant_ctx = plant.GetMyMutableContextFromRoot(ctx)

# Target: pusher center at (-0.085, 0, 0.05) — 1cm west of box's west face for clearance
# (box west face at x=-0.05, pusher radius 0.025 → touching at x=-0.075, +1cm = -0.085)
target_xyz = np.array([-0.085, 0.0, 0.05])   # 1cm west of box west face for clearance
pusher_body = plant.GetBodyByName(EE_BODY_NAME)

# Try several seed poses; pick the first that converges within 3mm
SEEDS = [
    [0.0,  0.4, 0.0, -2.8, 0.0, 3.2, 0.785],   # current default
    [0.0,  0.2, 0.0, -2.5, 0.0, 2.8, 0.785],
    [0.0,  0.0, 0.0, -2.2, 0.0, 2.2, 0.785],
    [0.3, -0.2, 0.0, -2.4, 0.0, 2.2, 0.785],
    [-0.3, 0.2, 0.0, -2.6, 0.0, 2.8, 0.785],
]

n_q = plant.num_positions()
best_q = None
best_err = float("inf")

for i, seed_arm in enumerate(SEEDS):
    ik = InverseKinematics(plant, plant_ctx)
    ik.AddPositionConstraint(
        frameA=plant.world_frame(),
        p_AQ_lower=target_xyz - 0.003,
        p_AQ_upper=target_xyz + 0.003,
        frameB=pusher_body.body_frame(),
        p_BQ=np.zeros(3),
    )
    q_seed = np.zeros(n_q)
    q_seed[:7] = seed_arm
    q_seed[-7:] = [1, 0, 0, 0, 0, 0, 0.05]   # box quaternion + position
    ik.get_mutable_prog().SetInitialGuess(ik.q(), q_seed)

    res = Solve(ik.prog())
    if res.is_success():
        q_sol = res.GetSolution(ik.q())
        plant.SetPositions(plant_ctx, q_sol)
        pos = plant.EvalBodyPoseInWorld(plant_ctx, pusher_body).translation()
        err = float(np.linalg.norm(pos - target_xyz))
        print(f"[seed {i}] IK success. arm_q={list(q_sol[:7])}")
        print(f"          puck pos: {pos}  err={err*1000:.2f}mm")
        if err < best_err:
            best_err = err
            best_q = list(q_sol[:7])
            if err < 0.002:
                break
    else:
        print(f"[seed {i}] IK failed")

if best_q is None:
    print("ALL IK ATTEMPTS FAILED. Cannot proceed.")
    sys.exit(1)

print()
print("=" * 60)
print(f"Best error: {best_err*1000:.2f}mm")
print("PREPOSITIONED_ARM_Q = [")
for v in best_q:
    print(f"    {v:+.6f},")
print("]")
print("=" * 60)
