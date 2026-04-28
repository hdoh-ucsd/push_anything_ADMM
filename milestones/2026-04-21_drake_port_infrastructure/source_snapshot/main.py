"""
C3 Contact-Implicit MPC — main entry point.

Usage
-----
    python main.py [task] [--save-video [OUTPUT.mp4]]

Task options
------------
    pushing       (default) — light box (0.2 kg, mu=0.4)
    hard_pushing            — heavy box (1.5 kg, mu=0.8)
    shepherding             — rolling ball (0.15 kg, mu=0.2)

Flags
-----
    --save-video            Save top-down MP4 to results/<task>.mp4
    --save-video OUT.mp4    Save to a specific path

Visualisation: Meshcat at http://127.0.0.1:7000

MPC parameters (hardcoded):
    horizon    = 8     steps  (8 × 0.03 s = 0.24 s lookahead)
    admm_iter  = 10    ADMM iterations per control step
    dt         = 0.03  s  planning timestep
    dt_ctrl    = 0.01  s  real sim control rate
    torque_lim = 30    Nm
    rho        = 1.0   ADMM penalty
"""
import argparse
import sys
from pathlib import Path
import yaml
import numpy as np
import pydrake.all as ad

from sim.env_builder import build_environment, INITIAL_ARM_Q, PREPOSITIONED_ARM_Q, EE_BODY_NAME
from sim.video_recorder import ExperimentRecorder
from control.lcs_formulator import LCSFormulator
from control.admm_solver import C3Solver
from control.task_costs import QuadraticManipulationCost
from control.ci_mpc_c3 import C3MPC


# ---------------------------------------------------------------------------
# Output tee — writes all print() calls to both terminal and results/output.txt
# ---------------------------------------------------------------------------

class _Tee:
    """Mirrors every write to a list of file-like objects."""
    def __init__(self, *files):
        self._files = files

    def write(self, data: str) -> None:
        for f in self._files:
            f.write(data)

    def flush(self) -> None:
        for f in self._files:
            f.flush()

    # Make this object behave as a proper stream (needed by some Drake output)
    def fileno(self):
        return self._files[0].fileno()


# ---------------------------------------------------------------------------
# Meshcat visualisation helpers
# ---------------------------------------------------------------------------

def _setup_meshcat_markers(meshcat, target_xy: np.ndarray, task_cfg: dict) -> None:
    """
    Add persistent Meshcat markers:
      /goal_marker  — semi-transparent ghost of the object at the goal position
    """
    init_z = task_cfg["init_xyz"][2]
    if task_cfg["object_type"] == "box":
        sx, sy, sz = task_cfg["size"]
        shape = ad.Box(sx, sy, sz)
    else:
        shape = ad.Sphere(task_cfg["radius"])

    meshcat.SetObject("/goal_marker", shape, ad.Rgba(0.1, 0.9, 0.1, 0.35))
    meshcat.SetTransform(
        "/goal_marker",
        ad.RigidTransform(ad.RotationMatrix(), [target_xy[0], target_xy[1], init_z]),
    )


def _update_predicted_trajectory(
    meshcat,
    x_seq: np.ndarray,        # (N+1, n_x)
    obj_x_idx: int,
    obj_y_idx: int,
    obj_z_idx: int,
) -> None:
    """
    Draw the MPC-predicted object trajectory each control step:
      /predicted_obj/line  — orange line connecting all N+1 predicted positions
      /predicted_obj/tip   — bright sphere at the terminal (N-th) predicted position
    """
    N = len(x_seq) - 1

    # Build 3×(N+1) vertex array
    pts = np.array(
        [[s[obj_x_idx], s[obj_y_idx], s[obj_z_idx]] for s in x_seq]
    ).T  # (3, N+1)

    meshcat.SetLine(
        "/predicted_obj/line", pts,
        line_width=4.0,
        rgba=ad.Rgba(1.0, 0.55, 0.0, 0.85),
    )

    # Terminal point: larger, brighter sphere
    meshcat.SetObject(
        "/predicted_obj/tip",
        ad.Sphere(0.022),
        ad.Rgba(1.0, 0.85, 0.0, 0.9),
    )
    meshcat.SetTransform(
        "/predicted_obj/tip",
        ad.RigidTransform(ad.RotationMatrix(), pts[:, -1].tolist()),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_task(task_name: str) -> dict:
    with open("config/tasks.yaml") as f:
        cfg = yaml.safe_load(f)
    tasks = cfg.get("tasks", {})
    if task_name not in tasks:
        raise ValueError(
            f"Unknown task '{task_name}'. Valid options: {list(tasks.keys())}"
        )
    return tasks[task_name]


def _obj_size_from_cfg(task_cfg: dict) -> float:
    if task_cfg["object_type"] == "sphere":
        return float(task_cfg["radius"]) * 2.0
    return float(task_cfg["size"][0])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="C3 Contact-Implicit MPC",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "task", nargs="?", default="pushing",
        choices=["pushing", "hard_pushing", "shepherding"],
        help="Task to run (default: pushing)",
    )
    parser.add_argument(
        "--save-video",
        metavar="OUTPUT.mp4",
        nargs="?",
        const="",
        default=None,
        help="Save top-down MP4.  Omit path to use results/<task>.mp4",
    )
    parser.add_argument(
        "--reset-every",
        metavar="N",
        type=int,
        default=None,
        help="Reset arm + box to initial pose every N control steps "
             "(useful to test first-contact behaviour repeatedly)",
    )
    parser.add_argument("--prepositioned", action="store_true",
                        help="Start arm in contact with box (diagnostic only)")
    args = parser.parse_args()

    task_name   = args.task
    video_path  = args.save_video
    reset_every = args.reset_every
    init_q = PREPOSITIONED_ARM_Q if args.prepositioned else INITIAL_ARM_Q
    if video_path is not None and video_path == "":
        video_path = f"results/{task_name}.mp4"

    Path("results").mkdir(exist_ok=True)
    from datetime import datetime
    _ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_path = f"results/{task_name}_{_ts}.txt"
    _log      = open(_log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, _log)
    print(f"[C3] Log: {_log_path}")

    print(f"[C3] Task: {task_name}")
    if video_path:
        print(f"[C3] Video output: {video_path}")

    task_cfg = load_task(task_name)

    # ---- Structured log header -------------------------------------------
    _cost = task_cfg.get("cost", {})
    print(f"[ENV]  Mass: {task_cfg.get('mass', '?')} kg   "
          f"Friction mu: {task_cfg.get('friction', '?')}")
    print(f"[MPC]  Horizon: 20   dt: 0.05 s")
    print(f"[MPC]  ADMM max iters: 3   rho_init: 100.0")
    print(f"[MPC]  Force limit: 30.0 Nm")
    print(f"[COST] w_obj_xy:      {_cost.get('w_obj_xy', '?')}")
    print(f"[COST] w_obj_z:       {_cost.get('w_obj_z', '?')}")
    print(f"[COST] w_box_z:       {_cost.get('w_box_z', '?')}")
    print(f"[COST] w_box_rp:      {_cost.get('w_box_rp', '?')}")
    print(f"[COST] w_terminal:    {_cost.get('w_terminal', '?')}  (QN = w_terminal * Q)")
    print(f"[COST] w_ee_approach: {_cost.get('w_ee_approach', '?')}")
    print(f"[COST] w_torque:      {_cost.get('w_torque', '?')}")
    print(f"[ENV]  INITIAL_ARM_Q: {'PREPOSITIONED' if args.prepositioned else 'DEFAULT'}")

    # ------------------------------------------------------------------
    # Build Drake environment
    # ------------------------------------------------------------------
    print("[C3] Building Drake environment ...")
    diagram, plant, panda_model, _, meshcat = build_environment(task_cfg)

    simulator = ad.Simulator(diagram)
    context   = simulator.get_mutable_context()
    plant_ctx = plant.GetMyContextFromRoot(context)

    # ------------------------------------------------------------------
    # Locate object body & position indices in q
    # Drake floating-body layout: [qw, qx, qy, qz, x, y, z]
    # ------------------------------------------------------------------
    link_name = task_cfg["link_name"]
    obj_body  = plant.GetBodyByName(link_name)
    pos_start = obj_body.floating_positions_start()
    obj_x_idx = pos_start + 4
    obj_y_idx = pos_start + 5
    obj_z_idx = pos_start + 6

    # ------------------------------------------------------------------
    # Set initial state
    # ------------------------------------------------------------------
    plant.SetPositions(plant_ctx, panda_model, init_q)
    plant.SetFreeBodyPose(
        plant_ctx, obj_body,
        ad.RigidTransform(ad.RotationMatrix(), task_cfg["init_xyz"])
    )

    # ------------------------------------------------------------------
    # System dimensions
    # ------------------------------------------------------------------
    n_q = plant.num_positions()
    n_v = plant.num_velocities()
    n_u = plant.num_actuators()
    n_x = n_q + n_v
    print(f"[C3] DOFs: n_q={n_q}, n_v={n_v}, n_u={n_u}, n_x={n_x}")
    print(f"[C3] '{link_name}': q[{obj_x_idx}]=x, q[{obj_y_idx}]=y")

    # ------------------------------------------------------------------
    # Controller pipeline
    # ------------------------------------------------------------------
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
        admm_iter=3,
    )

    target_xy   = np.array(task_cfg["goal_xy"], dtype=float)
    ee_frame    = plant.GetFrameByName(EE_BODY_NAME)
    world_frame = plant.world_frame()

    print(f"[C3] Goal: {target_xy}  |  Meshcat: {meshcat.web_url()}")

    # Goal ghost + trajectory markers (set up once before sim starts)
    _setup_meshcat_markers(meshcat, target_xy, task_cfg)

    print("[C3] Running simulation ...")

    # ------------------------------------------------------------------
    # Optional video recorder
    # ------------------------------------------------------------------
    recorder: ExperimentRecorder | None = None
    if video_path:
        recorder = ExperimentRecorder(
            output_path=video_path,
            fps=30,
            task_name=task_name,
            goal_xy=target_xy.tolist(),
            obj_shape=task_cfg["object_type"],
            obj_size=_obj_size_from_cfg(task_cfg),
        )

    # ------------------------------------------------------------------
    # Joint limit constants for arm safety check
    # ------------------------------------------------------------------
    _Q_LO = np.array([-2.897, -1.763, -2.897, -3.072, -2.897, -0.0175, -2.897])
    _Q_HI = np.array([ 2.897,  1.763,  2.897, -0.0698, 2.897,  3.752,   2.897])

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------
    sim_time      = 0.0
    dt_ctrl       = 0.01
    max_time      = 8.0
    step          = 0
    _record_every = max(1, round(1.0 / (20.0 * dt_ctrl)))

    if reset_every is not None:
        print(f"[C3] Reset mode: resetting arm + box every {reset_every} steps "
              f"({reset_every * dt_ctrl:.2f} s)")

    while sim_time < max_time:
        # ---- periodic state reset ----------------------------------------
        if reset_every is not None and step > 0 and step % reset_every == 0:
            print(f"[C3] Reset at step {step} (t={sim_time:.2f}s)")
            plant.SetPositions(plant_ctx, panda_model, init_q)
            plant.SetVelocities(plant_ctx, np.zeros(n_v))
            plant.SetFreeBodyPose(
                plant_ctx, obj_body,
                ad.RigidTransform(ad.RotationMatrix(), task_cfg["init_xyz"])
            )

        current_q = plant.GetPositions(plant_ctx)
        current_v = plant.GetVelocities(plant_ctx)

        # ---- NaN / joint-limit safety check ------------------------------
        if not (np.all(np.isfinite(current_q)) and np.all(np.isfinite(current_v))):
            print(f"[WARN] NaN in state at t={sim_time:.3f}s — stopping.")
            break
        arm_q = current_q[:n_u]
        if np.any(arm_q < _Q_LO - 0.05) or np.any(arm_q > _Q_HI + 0.05):
            violating = np.where(
                (arm_q < _Q_LO - 0.05) | (arm_q > _Q_HI + 0.05)
            )[0]
            print(f"[WARN] Joint limit violated at t={sim_time:.3f}s  "
                  f"joints={violating.tolist()}  q={arm_q.round(3)}")

        tau_g = plant.CalcGravityGeneralizedForces(plant_ctx)
        u_opt = mpc.compute_control(current_q, current_v, plant_ctx, target_xy)

        # Update predicted-trajectory markers in Meshcat
        if mpc.last_x_seq is not None:
            _update_predicted_trajectory(
                meshcat, mpc.last_x_seq, obj_x_idx, obj_y_idx, obj_z_idx
            )

        total_torque = tau_g[:n_u] + u_opt
        plant.get_actuation_input_port().FixValue(plant_ctx, total_torque)

        ee_pos = plant.CalcPointsPositions(
            plant_ctx, ee_frame, np.zeros(3), world_frame
        ).flatten()

        if step % 50 == 0:
            obj_x = current_q[obj_x_idx]
            obj_y = current_q[obj_y_idx]
            obj_z = current_q[obj_z_idx]
            dist  = np.linalg.norm(np.array([obj_x, obj_y]) - target_xy)
            print(
                f"  t={sim_time:.2f}s | "
                f"ee=({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}) | "
                f"obj=({obj_x:.3f}, {obj_y:.3f}, {obj_z:.3f}) | "
                f"|u|={np.linalg.norm(u_opt):.2f} Nm | "
                f"goal_dist={dist:.3f} m"
            )

        if recorder is not None and step % _record_every == 0:
            obj_xy = np.array([current_q[obj_x_idx], current_q[obj_y_idx]])
            recorder.record(sim_time, ee_pos[:2], obj_xy)

        sim_time += dt_ctrl
        step     += 1
        simulator.AdvanceTo(sim_time)

    print("[C3] Simulation complete.")

    if recorder is not None:
        recorder.save()


if __name__ == "__main__":
    main()
