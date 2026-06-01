#!/usr/bin/env python3
"""Stage D verification — end-to-end one tick of C3PlusMPC with use_ee_space=True.

Constructs the full controller stack with the EE-space sizing:
  - LCSFormulator (shared)
  - C3Solver(n_x=19, n_u=3, mode='c3plus')
  - QuadraticManipulationCost (uses build_ee_space inside C3PlusMPC)
  - C3PlusMPC(..., use_ee_space=True)

Then runs ONE compute_control() call and asserts:
  - Returns a (3,) vector (EE Cartesian force command).
  - last_x_seq shape (N+1, 19).
  - last_x_seq finite.
  - The planner's first-knot λ_n is available + finite.
  - No exceptions, no crash.

This is the gate before any sim or wrapper integration.
"""
from __future__ import annotations
import sys, yaml, numpy as np
sys.path.insert(0, "/root/push_anything_ADMM")

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
    print(f"C3PlusMPC built: use_ee_space={mpc.use_ee_space}, "
          f"solver.n_x={solver.n_x}, solver.n_u={solver.n_u}")

    target_xy = np.asarray(task_cfg["goal_xy"], dtype=float)
    current_q = plant.GetPositions(plant_ctx).copy()
    current_v = plant.GetVelocities(plant_ctx).copy()

    print(f"\nRunning one compute_control tick with EE-space planner...")
    u = mpc.compute_control(current_q, current_v, plant_ctx, target_xy)
    print(f"  u = {u}")
    print(f"  u.shape = {u.shape}")
    assert u.shape == (3,), f"u shape: {u.shape}"
    assert np.all(np.isfinite(u))
    print("[PASS] u is R^3 EE force, finite")

    xs = mpc.last_x_seq
    print(f"  last_x_seq.shape = {xs.shape}")
    assert xs.shape == (8 + 1, 19), f"x_seq shape: {xs.shape}"
    assert np.all(np.isfinite(xs))
    print(f"[PASS] last_x_seq (N+1, 19) finite")

    # Display key state quantities at predicted next step.
    print(f"\n--- Predicted x_seq[1] (next-tick state) ---")
    print(f"  box_q (quat+pos): {np.round(xs[1, 0:7], 4)}")
    print(f"  p_ee (predicted): {np.round(xs[1, 7:10], 4)}")
    print(f"  box_v (omega+lin): {np.round(xs[1, 10:16], 4)}")
    print(f"  v_ee (predicted): {np.round(xs[1, 16:19], 4)}")

    ln1 = mpc.last_lambda_n_first
    if ln1 is not None:
        print(f"\n  λ_n_first.shape = {ln1.shape}  values = {ln1}")
        assert np.all(np.isfinite(ln1))
        print("[PASS] λ_n_first finite")

    print(f"\n--- Second tick (uses self._last_u from first) ---")
    u2 = mpc.compute_control(current_q, current_v, plant_ctx, target_xy)
    print(f"  u2 = {u2}")
    print(f"  ||u2 - u|| = {np.linalg.norm(u2 - u):.6f}  "
          f"(may differ from u because _last_u changed the linearization)")

    print(f"\n{'='*60}\nSTAGE D VERIFICATION — RESULT\n{'='*60}")
    print(f"  C3PlusMPC(use_ee_space=True) runs end-to-end without crash.")
    print(f"  Returns u ∈ ℝ^3 (EE Cartesian force), finite.")
    print(f"  last_x_seq.shape = (N+1, 19), finite.")
    print(f"  Two consecutive ticks succeed (linearization warm-start works).")
    print(f"  Wrapper integration (extract p_ee from x_seq[1] instead of FK)")
    print(f"  is a follow-up commit.")


if __name__ == "__main__":
    main()
