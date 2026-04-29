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
    --save-video            Auto-name MP4 as results/<stem>.mp4
    --save-video OUT.mp4    Save to a specific path
    --video-path            Auto-name HTML as results/<stem>.html
    --video-path PATH.html  Save Meshcat replay HTML to a specific path
    --no-record             Disable both MP4 and HTML recording
    --name BASENAME         Shared <stem> for all outputs (txt + mp4 + html)

Default behavior records both MP4 and Meshcat HTML, sharing the same
stem as the _Tee log (results/<stem>.txt). The stem is BASENAME when
--name is given, else <task>_<timestamp>.

Visualisation: Meshcat at http://127.0.0.1:7000

MPC parameters:
    horizon    = 20    steps  (20 × 0.05 s = 1.0 s lookahead)
    admm_iter  = 3     ADMM iterations per control step (override with --admm-iter)
    dt         = 0.05  s  planning timestep
    dt_ctrl    = 0.01  s  real sim control rate
    torque_lim = 30    Nm
    rho        = 100   ADMM penalty (initial; adaptive every 10 iters)
"""
import argparse
import sys
from pathlib import Path
import yaml
import numpy as np
import pydrake.all as ad

from sim.env_builder import (
    build_environment,
    INITIAL_ARM_Q,
    EE_BODY_NAME,
    compute_prepositioned_arm_q,
)
from sim.video_recorder import ExperimentRecorder
from control.lcs_formulator import LCSFormulator
from control.admm_solver import C3Solver
from control.task_costs import QuadraticManipulationCost
from control.ci_mpc_c3 import C3MPC
from control.sampling_c3 import SamplingC3MPC, SamplingC3Params


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
        const="AUTO",
        default="AUTO",
        help="Save mp4 of sim. With no arg or absent, auto-names "
             "results/<task>_<timestamp>.mp4. Use --no-record to disable.",
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
    parser.add_argument("--task-id", type=int, choices=[1, 2, 3, 4], default=None,
                        help="Directional task ID from config/directional_tasks.json "
                             "(1=north, 2=east, 3=south, 4=west). Overrides tasks.yaml goal.")
    parser.add_argument("--video-path", type=str, nargs="?",
                        const="AUTO", default="AUTO", metavar="PATH.html",
                        help="Save Meshcat HTML replay. With no arg or absent, "
                             "auto-names results/<task>_<timestamp>.html. "
                             "Use --no-record to disable.")
    parser.add_argument("--no-record", action="store_true",
                        help="Disable both video and html recording. "
                             "Speeds up smoke tests.")
    parser.add_argument("--max-time", type=float, default=None,
                        help="Override simulation duration in seconds (default: 8.0).")
    parser.add_argument("--math-diag", action="store_true",
                        help="Print math-level solver diagnostics ([MATH.*] tags). "
                             "Zero overhead when off.")
    parser.add_argument("--admm-iter", type=int, default=3, metavar="N",
                        help="ADMM iterations per control step (default 3). "
                             "Higher values let the dual variable accumulate "
                             "before the QP-vs-cone projection re-fixes-point, "
                             "improving friction-cone feasibility. The README "
                             "notes adaptive-ρ fires every 10 iters, so values "
                             "≥ 10 also enable rho adaptation. Diagnostic use; "
                             "increases per-step solve time roughly linearly.")
    parser.add_argument("--cost-bias", action="store_true",
                        help="Enable C3 face-transition cost bias (lift/approach/push "
                             "state machine for sequential contact on correct cube face).")
    parser.add_argument("--name", type=str, default=None, metavar="BASENAME",
                        help="Shared basename (no extension) for all run outputs in "
                             "results/: <BASENAME>.txt, <BASENAME>.mp4, <BASENAME>.html. "
                             "When omitted, falls back to <task>_<timestamp>. "
                             "Explicit --save-video PATH / --video-path PATH still "
                             "override their respective files.")
    parser.add_argument("--sampling-c3", type=str, nargs="?",
                        const="config/sampling_c3_params.yaml", default=None,
                        metavar="PATH.yaml",
                        help="Enable Venkatesh-2025 sampling-C3 outer controller.\n"
                             "Optional PATH = YAML config "
                             "(default: config/sampling_c3_params.yaml).\n"
                             "Cannot be combined with --cost-bias.")
    args = parser.parse_args()

    if args.sampling_c3 is not None and args.cost_bias:
        parser.error("--sampling-c3 and --cost-bias are mutually exclusive. "
                     "Use one or the other.")

    task_name   = args.task
    reset_every = args.reset_every
    # init_q is computed below, after plant_ctx is staged with the object pose.
    # When --prepositioned, it comes from a push-direction-aware IK cascade so
    # the cost differential alone selects c3 mode at step 1.

    Path("results").mkdir(exist_ok=True)
    from datetime import datetime
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Shared basename: --name BASENAME wins; otherwise <task>_<timestamp>.
    if args.name is not None:
        stem = args.name
    else:
        stem = f"{task_name}_{run_stamp}"

    # Resolve recording paths.  --no-record wins; otherwise AUTO sentinels
    # produce shared-stem filenames, "" maps to legacy default-named files,
    # and any other string is taken as an explicit path.
    if args.no_record:
        video_path = None
        html_path  = None
    else:
        if args.save_video == "AUTO":
            video_path = f"results/{stem}.mp4"
        elif args.save_video == "":
            video_path = f"results/{task_name}.mp4"
        else:
            video_path = args.save_video

        if args.video_path == "AUTO":
            html_path = f"results/{stem}.html"
        elif args.video_path == "":
            html_path = f"results/{task_name}.html"
        else:
            html_path = args.video_path

    _log_path = f"results/{stem}.txt"
    _log      = open(_log_path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, _log)
    print(f"[C3] Log: {_log_path}")

    print(f"[C3] Task: {task_name}")
    if video_path:
        print(f"[C3] Video output: {video_path}")
    if html_path:
        print(f"[C3] HTML replay output: {html_path}")

    task_cfg = load_task(task_name)

    # Directional task override
    if args.task_id is not None:
        import json
        dir_path = Path(__file__).resolve().parent / "config" / "directional_tasks.json"
        with open(dir_path) as f:
            dir_cfg = json.load(f)
        task_entry = dir_cfg["tasks"][str(args.task_id)]
        task_cfg["goal_xy"] = task_entry["goal"]
        print(f"[ENV]  Directional task: id={args.task_id} name={task_entry['name']}")
        print(f"[ENV]  Goal coords: {task_cfg['goal_xy']}")
        print(f"[ENV]  Description: {task_entry['description']}")
    else:
        print(f"[ENV]  Goal coords: {task_cfg.get('goal_xy', 'default')}")

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
    # Stage object pose first so the prepositioned IK can see the object,
    # then resolve init_q (IK-derived if --prepositioned, default otherwise),
    # then set arm positions.
    # ------------------------------------------------------------------
    plant.SetFreeBodyPose(
        plant_ctx, obj_body,
        ad.RigidTransform(ad.RotationMatrix(), task_cfg["init_xyz"])
    )
    if args.prepositioned:
        init_q = compute_prepositioned_arm_q(
            plant, plant_ctx, panda_model,
            ee_frame=plant.GetFrameByName(EE_BODY_NAME),
            obj_body=obj_body,
            task_cfg=task_cfg,
        )
    else:
        init_q = INITIAL_ARM_Q
    plant.SetPositions(plant_ctx, panda_model, init_q)

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
    solver     = C3Solver(n_x=n_x, n_u=n_u, rho=100.0,
                          math_diag=args.math_diag)
    quad_cost  = QuadraticManipulationCost(
        plant, EE_BODY_NAME, obj_body, task_cfg["cost"], n_x, n_u,
        math_diag=args.math_diag,
        cost_bias=args.cost_bias,
    )
    mpc = C3MPC(
        formulator=formulator,
        solver=solver,
        quadratic_cost=quad_cost,
        horizon=20,
        dt=0.05,
        torque_limit=30.0,
        admm_iter=args.admm_iter,
        math_diag=args.math_diag,
    )

    target_xy   = np.array(task_cfg["goal_xy"], dtype=float)
    ee_frame    = plant.GetFrameByName(EE_BODY_NAME)
    world_frame = plant.world_frame()

    # ------------------------------------------------------------------
    # Optional sampling-C3 outer controller (Venkatesh 2025 §IV-D port)
    # ------------------------------------------------------------------
    if args.sampling_c3 is not None:
        _yaml_path = args.sampling_c3
        sc3_params = SamplingC3Params.from_yaml(_yaml_path)
        # With the new IK-based --prepositioned pose, k=0 captures the
        # ~30k alignment bonus (vs zero contact for every k>=1 at sampling
        # radius 0.18m), so decide_mode picks "c3" via kToC3Cost on step 1
        # under its own cost differential. Forcing the initial mode would
        # mask whether the pose actually does what we want.
        mpc = SamplingC3MPC(
            base_mpc=mpc,
            plant=plant,
            ee_frame=ee_frame,
            obj_body=obj_body,
            params=sc3_params,
            log_diag=True,
            start_in_c3_mode=False,
        )
        print(f"[GS] SamplingC3MPC enabled (config: {_yaml_path})")
        print(f"[GS]   strategy={sc3_params.sampling_params.sampling_strategy.name} "
              f"num_add_c3={sc3_params.sampling_params.num_additional_samples_c3} "
              f"num_add_repos={sc3_params.sampling_params.num_additional_samples_repos}")
        print(f"[GS]   w_align={sc3_params.w_align}  w_travel={sc3_params.w_travel}")
        print(f"[GS]   reposition: traj_type={sc3_params.reposition_params.traj_type.name} "
              f"z_safe={sc3_params.reposition_params.pwl_waypoint_height}m "
              f"speed={sc3_params.reposition_params.speed}m/s")

    print(f"[C3] Goal: {target_xy}  |  Meshcat: {meshcat.web_url()}")

    # Goal ghost + trajectory markers (set up once before sim starts)
    _setup_meshcat_markers(meshcat, target_xy, task_cfg)

    print("[C3] Running simulation ...")

    if html_path:
        meshcat.StartRecording()

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
    max_time      = args.max_time if args.max_time is not None else 8.0
    step          = 0
    if args.max_time is not None:
        print(f"[ENV]  Sim duration overridden: max_time={max_time}s")
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

        # SamplingC3MPC returns self-contained torque in free mode — its
        # tracker already includes gravity compensation. Adding tau_g again
        # here would double-compensate.
        if isinstance(mpc, SamplingC3MPC) and mpc.last_mode == "free":
            plant.get_actuation_input_port().FixValue(plant_ctx, u_opt)
        else:
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
    if isinstance(mpc, SamplingC3MPC):
        mpc.print_perf_summary()

    # ------------------------------------------------------------------
    # Final result summary
    # ------------------------------------------------------------------
    final_q      = plant.GetPositions(plant_ctx)
    final_obj_xy = np.array([final_q[obj_x_idx], final_q[obj_y_idx]])
    final_dist   = float(np.linalg.norm(final_obj_xy - target_xy))
    if args.sampling_c3 is not None:
        _method = "sampling-c3"
    else:
        _method = "baseline-C3"
    print(f"[RESULT] method={_method}  "
          f"final_obj_xy=({final_obj_xy[0]:.4f}, {final_obj_xy[1]:.4f})  "
          f"goal_dist={final_dist:.4f}m  "
          f"success={'YES' if final_dist < 0.05 else 'NO'}")

    if html_path:
        meshcat.StopRecording()
        meshcat.PublishRecording()
        html = meshcat.StaticHtml()
        html_dir = Path(html_path).parent
        html_dir.mkdir(parents=True, exist_ok=True)
        with open(html_path, "w") as f:
            f.write(html)
        print(f"[VIDEO] Saved replay to {html_path}")

    if recorder is not None:
        recorder.save()


if __name__ == "__main__":
    main()
