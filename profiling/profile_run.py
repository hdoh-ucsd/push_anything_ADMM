"""
C3 Pipeline Profiler.

Runs cProfile + SectionTimer together for N control steps and emits:
  - profiling/results/<task>_cprofile.prof  (loadable by snakeviz / pstats)
  - profiling/results/<task>_cprofile.txt   (human-readable sorted summary)
  - profiling/results/<task>_sections.txt   (SectionTimer per-section report)

Usage
-----
    python profiling/profile_run.py [task] [n_steps]

    task    : pushing (default) | hard_pushing | shepherding
    n_steps : number of full control-loop iterations to profile (default 5)

Reading results
---------------
    snakeviz profiling/results/pushing_cprofile.prof   # interactive flame graph
    python -m pstats profiling/results/pushing_cprofile.prof

Section names (SectionTimer)
----------------------------
    lcs.extract_dynamics    CalcMassMatrixViaInverseDynamics + bias
    lcs.geometry_query      ComputeSignedDistancePairwiseClosestPoints
    lcs.calc_jacobians      CalcJacobianTranslationalVelocity (per contact)
    admm.qp_build           MathematicalProgram setup + cost update
    admm.osqp_solve         OSQP Solve() call
    admm.z_update           delta/omega update + Lorentz projection
"""

import cProfile
import pstats
import io
import os
import sys
import time
import yaml
import numpy as np
import pydrake.all as ad

# Make sure project root is on the path when running from profiling/
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import profiling.section_timer as ST
ST.ENABLED = True   # enable fine-grained section timing

from sim.env_builder import build_environment, INITIAL_ARM_Q, EE_BODY_NAME
from control.lcs_formulator import LCSFormulator
from control.admm_solver import C3Solver
from control.task_costs import QuadraticManipulationCost
from control.ci_mpc_c3 import C3MPC


# ---------------------------------------------------------------------------
# Build environment (identical to main.py)
# ---------------------------------------------------------------------------

def _setup(task_name: str):
    with open(os.path.join(_ROOT, "config", "tasks.yaml")) as f:
        cfg = yaml.safe_load(f)
    task_cfg = cfg["tasks"][task_name]

    diagram, plant, panda_model, _, meshcat = build_environment(task_cfg)

    simulator = ad.Simulator(diagram)
    context   = simulator.get_mutable_context()
    plant_ctx = plant.GetMyContextFromRoot(context)

    link_name = task_cfg["link_name"]
    obj_body  = plant.GetBodyByName(link_name)
    pos_start = obj_body.floating_positions_start()
    obj_x_idx = pos_start + 4
    obj_y_idx = pos_start + 5

    plant.SetPositions(plant_ctx, panda_model, INITIAL_ARM_Q)
    plant.SetFreeBodyPose(
        plant_ctx, obj_body,
        ad.RigidTransform(ad.RotationMatrix(), task_cfg["init_xyz"])
    )

    n_q = plant.num_positions()
    n_v = plant.num_velocities()
    n_u = plant.num_actuators()
    n_x = n_q + n_v

    formulator = LCSFormulator(plant, mu=task_cfg["friction"], obj_body=obj_body)
    solver     = C3Solver(n_x=n_x, n_u=n_u, rho=100.0)
    quad_cost  = QuadraticManipulationCost(
        plant, EE_BODY_NAME, obj_body, task_cfg["cost"], n_x, n_u
    )
    mpc = C3MPC(
        formulator=formulator,
        solver=solver,
        quadratic_cost=quad_cost,
        horizon=20,
        dt=0.05,
        torque_limit=30.0,
        admm_iter=80,
    )

    target_xy = np.array(task_cfg["goal_xy"], dtype=float)

    return (simulator, plant, plant_ctx, panda_model, mpc,
            target_xy, n_u, obj_x_idx, obj_y_idx)


# ---------------------------------------------------------------------------
# Profiled control loop (N steps only)
# ---------------------------------------------------------------------------

def _run_steps(simulator, plant, plant_ctx, panda_model, mpc,
               target_xy, n_u, obj_x_idx, obj_y_idx, n_steps: int):
    sim_time = 0.0
    dt_ctrl  = 0.01

    for step in range(n_steps):
        current_q = plant.GetPositions(plant_ctx)
        current_v = plant.GetVelocities(plant_ctx)
        tau_g     = plant.CalcGravityGeneralizedForces(plant_ctx)

        with ST.timed("compute_control [total]"):
            u_opt = mpc.compute_control(current_q, current_v, plant_ctx, target_xy)

        total_torque = tau_g[:n_u] + u_opt
        plant.get_actuation_input_port().FixValue(plant_ctx, total_torque)

        sim_time += dt_ctrl
        simulator.AdvanceTo(sim_time)

        obj_x = plant.GetPositions(plant_ctx)[obj_x_idx]
        obj_y = plant.GetPositions(plant_ctx)[obj_y_idx]
        dist  = np.linalg.norm(np.array([obj_x, obj_y]) - target_xy)
        print(f"  step {step+1:>2}/{n_steps} | "
              f"obj=({obj_x:.3f}, {obj_y:.3f}) | "
              f"goal_dist={dist:.3f} m")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    task_name = sys.argv[1] if len(sys.argv) > 1 else "pushing"
    n_steps   = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    results_dir = os.path.join(_ROOT, "profiling", "results")
    os.makedirs(results_dir, exist_ok=True)
    prof_path = os.path.join(results_dir, f"{task_name}_cprofile.prof")
    txt_path  = os.path.join(results_dir, f"{task_name}_cprofile.txt")
    sec_path  = os.path.join(results_dir, f"{task_name}_sections.txt")

    print(f"[Profiler] Task: {task_name}  |  Steps: {n_steps}")
    print("[Profiler] Building environment ...")
    setup_args = _setup(task_name)

    print(f"[Profiler] Running {n_steps} step(s) under cProfile ...")
    ST.reset()
    profiler = cProfile.Profile()

    t_wall_start = time.perf_counter()
    profiler.enable()
    _run_steps(*setup_args, n_steps=n_steps)
    profiler.disable()
    t_wall_total = time.perf_counter() - t_wall_start

    print(f"\n[Profiler] Wall time: {t_wall_total:.2f}s  "
          f"({t_wall_total/n_steps:.2f}s/step)")

    # ---- cProfile: save .prof ----
    profiler.dump_stats(prof_path)
    print(f"[Profiler] Saved: {prof_path}")
    print(f"           View:  snakeviz {prof_path}")

    # ---- cProfile: save text summary (top 40 by cumulative time) ----
    buf = io.StringIO()
    ps  = pstats.Stats(profiler, stream=buf)
    ps.sort_stats("cumulative")
    ps.print_stats(40)
    txt_report = buf.getvalue()

    with open(txt_path, "w") as f:
        f.write(f"Task: {task_name}  |  Steps: {n_steps}  |  "
                f"Wall: {t_wall_total:.2f}s\n\n")
        f.write(txt_report)
    print(f"[Profiler] Saved: {txt_path}")

    # Print top-15 to console
    buf2 = io.StringIO()
    pstats.Stats(profiler, stream=buf2).sort_stats("cumulative").print_stats(15)
    print("\n--- cProfile top-15 (cumulative) ---")
    for line in buf2.getvalue().splitlines()[4:]:
        print(line)

    # ---- SectionTimer report ----
    sec_text = ST.report()
    with open(sec_path, "w") as f:
        f.write(f"Task: {task_name}  |  Steps: {n_steps}  |  "
                f"Wall: {t_wall_total:.2f}s\n")
        f.write(sec_text)
    print(f"[Profiler] Saved: {sec_path}")


if __name__ == "__main__":
    main()
