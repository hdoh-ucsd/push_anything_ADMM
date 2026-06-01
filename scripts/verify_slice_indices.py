#!/usr/bin/env python3
"""Slice-index confirmation BEFORE the wrapper edit.

The wrapper edit will read p_ee_des = x_seq[1][7:10] and v_ee_des = x_seq[1][16:19]
under use_ee_space=True. Those slices are assumptions about the new 19-dim
state layout. A silent off-by-block (e.g. reading box-pos instead of EE-pos)
runs cleanly, returns finite numbers, and silently confounds the rewrite
comparison.

This script CONFIRMS the layout by:
  - Building the controller with use_ee_space=True.
  - Computing one tick to get last_x_seq.
  - Recording the GROUND TRUTH at the linearization point: the CURRENT
    EE position (from FK on q_arm) and CURRENT EE velocity (from J·v).
  - x_seq[0] MUST equal the constructed x0 — and x0 was built as
    [box_q, p_ee_now, box_v, v_ee_now]. So x_seq[0][7:10] == p_ee_now
    and x_seq[0][16:19] == v_ee_now is a bit-equal check, not just
    "values look like positions."

Pass criteria:
  - max |x_seq[0][7:10]  - p_ee_now|  < 1e-12  (bit-equal at linearization pt)
  - max |x_seq[0][16:19] - v_ee_now|  < 1e-12
  - max |x_seq[0][0:7]   - box_q_now| < 1e-12
  - max |x_seq[0][10:16] - box_v_now| < 1e-12
"""
from __future__ import annotations
import sys, yaml, numpy as np
sys.path.insert(0, "/root/push_anything_ADMM")
import pydrake.all as ad

from sim.env_builder import build_environment
from control.lcs_formulator import LCSFormulator
from control.task_costs import QuadraticManipulationCost
from control.admm_solver import C3Solver
from control.ci_mpc_c3plus import C3PlusMPC


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
        plant_ad=plant_ad, context_ad=context_ad,
    )
    solver = C3Solver(n_x=19, n_u=3, rho=100.0, mode="c3plus")
    quad_cost = QuadraticManipulationCost(
        plant, "pusher", obj_body, task_cfg["cost"],
        n_x=plant.num_positions() + plant.num_velocities(),
        n_u=plant.num_actuators(),
    )
    mpc = C3PlusMPC(
        formulator=formulator, solver=solver, quadratic_cost=quad_cost,
        horizon=8, dt=0.05, torque_limit=20.0, admm_iter=10,
        use_ee_space=True,
    )

    target_xy = np.asarray(task_cfg["goal_xy"], dtype=float)
    current_q = plant.GetPositions(plant_ctx).copy()
    current_v = plant.GetVelocities(plant_ctx).copy()

    # Ground truth at linearization point.
    plant.SetPositions(plant_ctx, current_q)
    plant.SetVelocities(plant_ctx, current_v)
    BOX_Q_START = obj_body.floating_positions_start()
    BOX_V_START = obj_body.floating_velocities_start_in_v()
    box_q_now = current_q[BOX_Q_START : BOX_Q_START + 7].copy()
    box_v_now = current_v[BOX_V_START : BOX_V_START + 6].copy()
    ee_body = plant.GetBodyByName('pusher')
    p_ee_now = plant.CalcPointsPositions(
        plant_ctx, ee_body.body_frame(), np.zeros((3, 1)),
        plant.world_frame(),
    ).flatten().copy()
    J_ee_full = plant.CalcJacobianTranslationalVelocity(
        plant_ctx, ad.JacobianWrtVariable.kV,
        ee_body.body_frame(), np.zeros(3),
        plant.world_frame(), plant.world_frame(),
    )
    v_ee_now = (J_ee_full @ current_v).copy()

    print("=== Ground truth at linearization point ===")
    print(f"  box_q_now (7) = {np.round(box_q_now, 5)}")
    print(f"  p_ee_now  (3) = {np.round(p_ee_now, 5)}")
    print(f"  box_v_now (6) = {np.round(box_v_now, 5)}")
    print(f"  v_ee_now  (3) = {np.round(v_ee_now, 5)}")

    _ = mpc.compute_control(current_q, current_v, plant_ctx, target_xy)

    xs = mpc.last_x_seq  # (N+1, 19)
    print(f"\n=== last_x_seq shape: {xs.shape} ===")
    print(f"  x_seq[0] (first knot — should bit-equal x0 = linearization-point state):")
    print(f"    [0:7]   box_q = {np.round(xs[0, 0:7], 5)}")
    print(f"    [7:10]  p_ee  = {np.round(xs[0, 7:10], 5)}")
    print(f"    [10:16] box_v = {np.round(xs[0, 10:16], 5)}")
    print(f"    [16:19] v_ee  = {np.round(xs[0, 16:19], 5)}")

    print(f"\n=== Bit-equality check x_seq[0] vs linearization point ===")
    d_box_q = float(np.max(np.abs(xs[0, 0:7]   - box_q_now)))
    d_p_ee  = float(np.max(np.abs(xs[0, 7:10]  - p_ee_now)))
    d_box_v = float(np.max(np.abs(xs[0, 10:16] - box_v_now)))
    d_v_ee  = float(np.max(np.abs(xs[0, 16:19] - v_ee_now)))
    print(f"  max |x_seq[0][0:7]   - box_q_now| = {d_box_q:.3e}")
    print(f"  max |x_seq[0][7:10]  - p_ee_now|  = {d_p_ee:.3e}  ← p_ee slice")
    print(f"  max |x_seq[0][10:16] - box_v_now| = {d_box_v:.3e}")
    print(f"  max |x_seq[0][16:19] - v_ee_now|  = {d_v_ee:.3e}  ← v_ee slice")

    TOL = 1e-12
    ok_box_q = d_box_q < TOL
    ok_p_ee  = d_p_ee  < TOL
    ok_box_v = d_box_v < TOL
    ok_v_ee  = d_v_ee  < TOL

    print(f"\n=== SLICE-INDEX VERDICT ===")
    print(f"  x_seq[0][0:7]  ≡ box_q       : {'PASS' if ok_box_q else 'FAIL'}")
    print(f"  x_seq[0][7:10] ≡ p_ee        : {'PASS' if ok_p_ee  else 'FAIL'}")
    print(f"  x_seq[0][10:16]≡ box_v       : {'PASS' if ok_box_v else 'FAIL'}")
    print(f"  x_seq[0][16:19]≡ v_ee        : {'PASS' if ok_v_ee  else 'FAIL'}")
    assert ok_box_q and ok_p_ee and ok_box_v and ok_v_ee, (
        "Slice indices do NOT match the documented layout — wrapper edit "
        "would silently feed the executor the wrong setpoint. Abort."
    )
    print(f"\n  CONFIRMED: wrapper can safely use x_seq[k][7:10] for p_ee_des")
    print(f"             and       x_seq[k][16:19] for v_ee_des  under use_ee_space.")


if __name__ == "__main__":
    main()
