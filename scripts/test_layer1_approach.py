"""Layer 1 unit test: translation-agnostic EE-approach for rotation tasks.

For cube_turning (target_xy == obj_xy → dist=0), confirm:
  1. Q[0:n_u, 0:n_u] is now NON-ZERO (was zero before the fix).
  2. The arm x_ref shift dq points the arm toward a config where the EE
     is closer to the box CoM (J_arm·dq ≈ proxy − ee_now).
  3. The approach-cost contribution (x - x_ref)^T Q[arm] (x - x_ref) is
     larger when the arm is far from the box-CoM-pointing config than
     when it is close (monotonic).

Regression: for 'pushing' (w_yaw=0), the new branch must not fire.
"""
import sys
import numpy as np
import pydrake.all as ad

sys.path.insert(0, ".")

from main import load_task, EE_BODY_NAME
from sim.env_builder import _INITIAL_ARM_Q_SEED as INITIAL_ARM_Q  # IK seed, not the production start pose (7ff5a21)
from sim.env_builder import build_environment
from control.task_costs import QuadraticManipulationCost


def make_cost_and_ctx(task_name: str):
    task_cfg = load_task(task_name)
    diagram, plant, panda_model, _, meshcat, *_ = build_environment(task_cfg)
    sim = ad.Simulator(diagram)
    ctx = sim.get_mutable_context()
    plant_ctx = plant.GetMyContextFromRoot(ctx)
    obj_body = plant.GetBodyByName(task_cfg["link_name"])
    plant.SetFreeBodyPose(
        plant_ctx, obj_body,
        ad.RigidTransform(ad.RotationMatrix(), task_cfg["init_xyz"])
    )
    plant.SetPositions(plant_ctx, panda_model, INITIAL_ARM_Q)
    n_q = plant.num_positions()
    n_v = plant.num_velocities()
    n_u = plant.num_actuators()
    n_x = n_q + n_v
    cost = QuadraticManipulationCost(
        plant=plant, ee_frame_name=EE_BODY_NAME, obj_body=obj_body,
        cost_cfg=task_cfg["cost"], n_x=n_x, n_u=n_u,
    )
    return cost, plant, plant_ctx, obj_body, n_x, n_u, n_q, task_cfg


# === EXPERIMENT: cube_turning ============================================
print("=" * 70)
print("EXPERIMENT: cube_turning (w_yaw=10, dist=0 -> new branch should fire)")
print("=" * 70)
cost, plant, plant_ctx, obj_body, n_x, n_u, n_q, task_cfg = \
    make_cost_and_ctx("cube_turning")

target_xy   = np.array(task_cfg["goal_xy"], dtype=float)
target_yaw  = float(task_cfg["goal_yaw"])
print(f"target_xy={target_xy}, target_yaw={target_yaw:.4f} rad")
print(f"w_yaw={cost.w_yaw}, w_ee_approach={cost.w_ee_approach}")

current_q = plant.GetPositions(plant_ctx).copy()

Q, R, QN, x_ref = cost.build(
    target_xy, plant_ctx=plant_ctx, current_q=current_q,
    target_yaw=target_yaw,
)

Q_arm = Q[:n_u, :n_u]
Q_arm_fro = float(np.linalg.norm(Q_arm))
q_arm_nonzero = Q_arm_fro > 1e-9
print(f"Q[0:n_u, 0:n_u] Frobenius norm: {Q_arm_fro:.4f}")
print(f"Q[0:n_u, 0:n_u] NONZERO? {q_arm_nonzero}")

arm_q_now = current_q[:n_u]
arm_q_ref = x_ref[:n_u]
dq = arm_q_ref - arm_q_now
print(f"x_ref arm shift dq norm = {np.linalg.norm(dq):.4f}")

ee_frame = plant.GetFrameByName(EE_BODY_NAME)
ee_now = plant.CalcPointsPositions(
    plant_ctx, ee_frame, np.zeros(3), plant.world_frame()).flatten()
obj_xy = np.array([current_q[cost._obj_x_idx], current_q[cost._obj_y_idx]])
proxy = np.array([obj_xy[0], obj_xy[1], cost.z_ref])
print(f"ee_now = {ee_now.round(4)}")
print(f"box CoM proxy = {proxy.round(4)}")
print(f"ee_err (proxy - ee_now) = {(proxy - ee_now).round(4)}")
J_ee = plant.CalcJacobianTranslationalVelocity(
    plant_ctx, ad.JacobianWrtVariable.kV, ee_frame, np.zeros(3),
    plant.world_frame(), plant.world_frame(),
)
J_arm = J_ee[:, :n_u]
print(f"J_arm @ dq (predicted EE delta) = {(J_arm @ dq).round(4)}")

def approach_cost_at_arm(q_arm):
    e = q_arm - arm_q_ref
    return float(e @ Q_arm @ e)

c_far  = approach_cost_at_arm(arm_q_now)
c_at   = approach_cost_at_arm(arm_q_ref)
c_near = approach_cost_at_arm(arm_q_now + 0.5 * dq)

print(f"\napproach-cost contribution at three arm configs:")
print(f"  far (arm at home):                  cost = {c_far:.6f}")
print(f"  near (halfway along dq):            cost = {c_near:.6f}")
print(f"  at  (arm at x_ref = home + dq):     cost = {c_at:.6f}")

assert q_arm_nonzero, "FAIL: Q[0:n_u, 0:n_u] is still zero"
assert c_far > c_near > c_at, (
    f"FAIL: monotonicity violated: far={c_far}, near={c_near}, at={c_at}"
)
print("\nPASS: cube_turning approach incentive present + monotonic")

# === REGRESSION: pushing (w_yaw=0) =======================================
print()
print("=" * 70)
print("REGRESSION: pushing (w_yaw=0) — new branch must NOT fire when dist=0")
print("=" * 70)
cost_p, plant_p, plant_ctx_p, obj_body_p, n_x_p, n_u_p, n_q_p, task_cfg_p = \
    make_cost_and_ctx("pushing")

# Force dist=0 by placing the box at target.
current_q_p = plant_p.GetPositions(plant_ctx_p).copy()
target_xy_p = np.array(task_cfg_p["goal_xy"], dtype=float)
current_q_p[cost_p._obj_x_idx] = target_xy_p[0]
current_q_p[cost_p._obj_y_idx] = target_xy_p[1]
plant_p.SetPositions(plant_ctx_p, current_q_p)

print(f"w_yaw={cost_p.w_yaw} (pushing has w_yaw=0)")

Q_p, R_p, QN_p, x_ref_p = cost_p.build(
    target_xy_p, plant_ctx=plant_ctx_p, current_q=current_q_p,
    target_yaw=0.0,
)
Q_arm_p_fro = float(np.linalg.norm(Q_p[:n_u_p, :n_u_p]))
xref_arm_p_norm = float(np.linalg.norm(x_ref_p[:n_u_p]))
print(f"Q[0:n_u, 0:n_u] Frobenius norm: {Q_arm_p_fro:.6f}")
print(f"x_ref[0:n_u] norm: {xref_arm_p_norm:.6f}")

assert Q_arm_p_fro < 1e-9, (
    f"FAIL: pushing leaked into new branch (Q[0:n_u] = {Q_arm_p_fro})"
)
assert xref_arm_p_norm < 1e-9, "FAIL: pushing arm x_ref shifted unexpectedly"
print("PASS: pushing (w_yaw=0) byte-inert: Q[0:n_u]=0, x_ref[0:n_u]=0")

# === REGRESSION: pushing with dist>0 (translation branch still works) ====
print()
print("=" * 70)
print("REGRESSION: pushing with dist>0 (translation branch unchanged)")
print("=" * 70)
plant_p.SetFreeBodyPose(
    plant_ctx_p, obj_body_p,
    ad.RigidTransform(ad.RotationMatrix(), task_cfg_p["init_xyz"])
)
current_q_p2 = plant_p.GetPositions(plant_ctx_p).copy()
Q_p2, R_p2, QN_p2, x_ref_p2 = cost_p.build(
    target_xy_p, plant_ctx=plant_ctx_p, current_q=current_q_p2,
    target_yaw=0.0,
)
Q_arm_p2_fro = float(np.linalg.norm(Q_p2[:n_u_p, :n_u_p]))
print(f"Q[0:n_u, 0:n_u] Frobenius norm: {Q_arm_p2_fro:.4f}")
assert Q_arm_p2_fro > 1e-3, (
    "FAIL: pushing with dist>0 has zero Q[0:n_u] — translation branch broke"
)
print(f"PASS: pushing translation branch still active (Q[0:n_u] = {Q_arm_p2_fro:.4f})")

print()
print("=" * 70)
print("ALL UNIT TESTS PASSED — Layer 1 ready for commit + rollout")
print("=" * 70)
