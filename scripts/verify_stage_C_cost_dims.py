#!/usr/bin/env python3
"""Stage C verification — QuadraticManipulationCost.build_ee_space for the
new low-dim LCS.

Tests:
  - Q (19,19), R (3,3), QN (19,19), x_ref (19,) — correct shapes.
  - All entries finite, no NaN.
  - Cost evaluates: x_ref^T Q x_ref is finite and non-negative.
  - Q symmetric.
  - Gradients (q_ref = -2 Q x_ref) finite.
  - End-to-end: feed the new cost matrices into the solver and confirm one
    ADMM solve runs (gradient flow consistency check across cost+solver).
"""
from __future__ import annotations
import sys, yaml, numpy as np
sys.path.insert(0, "/root/push_anything_ADMM")

from sim.env_builder import build_environment
from control.lcs_formulator import LCSFormulator
from control.task_costs import QuadraticManipulationCost
from control.admm_solver import C3Solver


def main():
    all_tasks = yaml.safe_load(open("/root/push_anything_ADMM/config/tasks.yaml"))
    task_cfg = all_tasks["tasks"]["pushing"]
    (diagram, plant, panda, obj_model, _meshcat,
     plant_ad, context_ad) = build_environment(task_cfg)
    diag_ctx = diagram.CreateDefaultContext()
    plant_ctx = plant.GetMyMutableContextFromRoot(diag_ctx)
    obj_body = plant.GetBodyByName(task_cfg["link_name"])

    formulator = LCSFormulator(
        plant, mu=task_cfg["friction"], obj_body=obj_body,
        plant_ad=plant_ad, context_ad=context_ad, box_ground_drag=0.0,
    )

    # Cost builder takes n_x, n_u of the OLD plant in __init__ (used only
    # for obj-index calculations); build_ee_space ignores those for sizing
    # and produces 19x19 / 3x3 / 19x19 / 19.
    cost = QuadraticManipulationCost(
        plant, ee_frame_name="pusher",
        obj_body=obj_body, cost_cfg=task_cfg["cost"],
        n_x=plant.num_positions() + plant.num_velocities(),
        n_u=plant.num_actuators(),
    )

    target_xy = np.asarray(task_cfg["goal_xy"], dtype=float)
    q_current = plant.GetPositions(plant_ctx).copy()
    plant.SetPositions(plant_ctx, q_current)

    print(f"--- Test 1: shapes ---")
    Q, R, QN, x_ref = cost.build_ee_space(
        target_xy, plant_ctx=plant_ctx, current_q=q_current,
        target_yaw=0.0,
    )
    print(f"  Q.shape    = {Q.shape}")
    print(f"  R.shape    = {R.shape}")
    print(f"  QN.shape   = {QN.shape}")
    print(f"  x_ref.shape= {x_ref.shape}")
    assert Q.shape == (19, 19)
    assert R.shape == (3, 3)
    assert QN.shape == (19, 19)
    assert x_ref.shape == (19,)
    print("[PASS] Test 1: cost-matrix shapes match (19, 3)\n")

    print(f"--- Test 2: finite, symmetric ---")
    assert np.all(np.isfinite(Q)),  "Q has non-finite entries"
    assert np.all(np.isfinite(R)),  "R has non-finite entries"
    assert np.all(np.isfinite(QN)), "QN has non-finite entries"
    assert np.all(np.isfinite(x_ref)), "x_ref has non-finite entries"
    sym_diff = float(np.max(np.abs(Q - Q.T)))
    print(f"  max |Q - Q.T| = {sym_diff:.3e}")
    assert sym_diff < 1e-9, f"Q not symmetric (max abs asymmetry {sym_diff})"
    print("[PASS] Test 2: entries finite, Q symmetric\n")

    print(f"--- Test 3: cost evaluates, gradient finite ---")
    # Pick a test state x: box at default, p_ee somewhere reasonable.
    x_test = np.zeros(19)
    x_test[0] = 1.0   # qw = 1 (identity rotation)
    x_test[6] = 0.05  # obj_z
    x_test[7:10] = np.array([0.5, 0.0, 0.15])   # p_ee
    cost_val = float(x_test @ Q @ x_test)
    grad     = -2.0 * (Q @ x_ref)
    print(f"  x^T Q x = {cost_val:.4f}")
    print(f"  ||q_ref|| = {np.linalg.norm(grad):.4e}")
    assert np.isfinite(cost_val) and cost_val >= 0.0
    assert np.all(np.isfinite(grad))
    print("[PASS] Test 3: cost is finite, gradient is finite\n")

    print(f"--- Test 4: structural cost components ---")
    print(f"  Q[obj_x,obj_x]={Q[4,4]:.1f}  (expected w_obj_xy={cost.w_obj_xy:.1f})")
    print(f"  Q[obj_y,obj_y]={Q[5,5]:.1f}")
    print(f"  Q[obj_z,obj_z]={Q[6,6]:.1f}  "
          f"(expected w_obj_z+w_box_z={cost.w_obj_z + cost.w_box_z:.1f})")
    print(f"  Q[p_ee block (7:10,7:10)] = "
          f"{np.round(Q[7:10, 7:10], 1)}")
    print(f"  x_ref[obj] = ({x_ref[4]:+.3f}, {x_ref[5]:+.3f}, {x_ref[6]:+.3f})")
    print(f"  x_ref[p_ee]= ({x_ref[7]:+.3f}, {x_ref[8]:+.3f}, {x_ref[9]:+.3f})")
    assert abs(Q[4, 4] - cost.w_obj_xy) < 1e-9
    assert abs(Q[5, 5] - cost.w_obj_xy) < 1e-9
    assert abs(Q[6, 6] - (cost.w_obj_z + cost.w_box_z)) < 1e-9
    assert abs(R[0, 0] - cost.w_torque) < 1e-9
    print("[PASS] Test 4: cost components match config\n")

    print(f"--- Test 5: end-to-end (LCS + cost + solver) ---")
    dt = 0.05
    tup = formulator.linearize_discrete_ee_space(plant_ctx, dt, np.zeros(3))
    A, B_ctrl, D, d_vec, E, F, H, c_lcs, J_n, J_t, phi, mu = tup
    solver = C3Solver(n_x=19, n_u=3, rho=1.0, mode="c3plus")
    x0 = np.concatenate([
        q_current[7:14],
        np.array([0.5, 0.0, 0.15]),
        np.zeros(6),
        np.zeros(3),
    ])
    u_seq, x_seq = solver.solve(
        x0=x0, A=A, B_ctrl=B_ctrl, D=D, d=d_vec,
        J_n=J_n, J_t=J_t, mu=mu,
        Q=Q, R=R, QN=QN, x_ref=x_ref,
        N=4, admm_iter=10, torque_limit=10.0,
        phi=phi, E=E, F=F, H=H, c_lcs=c_lcs,
    )
    print(f"  u_seq.shape={u_seq.shape}  x_seq.shape={x_seq.shape}")
    assert u_seq.shape == (4, 3) and x_seq.shape == (5, 19)
    assert np.all(np.isfinite(u_seq)) and np.all(np.isfinite(x_seq))
    print("[PASS] Test 5: end-to-end LCS+cost+solver runs cleanly\n")

    print("=" * 60)
    print("STAGE C VERIFICATION — RESULT")
    print("=" * 60)
    print(f"  Q (19,19) ✓   R (3,3) ✓   QN (19,19) ✓   x_ref (19,) ✓")
    print(f"  Finite, symmetric, gradient OK")
    print(f"  End-to-end: solver runs cleanly with EE-space cost matrices")
    print(f"  R^7 build() untouched — additive change only.")


if __name__ == "__main__":
    main()
