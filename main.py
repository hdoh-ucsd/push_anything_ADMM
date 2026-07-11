"""
C3 Contact-Implicit MPC — main entry point.

Usage
-----
    python main.py [task] [--drake-frames-dir DIR] [--video-path PATH.html]

Task options
------------
    pushing       (default) — light box (0.2 kg, mu=0.4)
    hard_pushing            — heavy box (1.5 kg, mu=0.8)
    shepherding             — rolling ball (0.15 kg, mu=0.2)

Flags
-----
    --drake-frames-dir DIR  OOM-safe Drake VTK PNG-per-tick capture.
                            Encode to MP4 offline via ffmpeg (or use
                            tools/visualizer/paint_mode_text.py for
                            the annotated-render pipeline).
    --video-path            Auto-name HTML as results/<stem>.html
    --video-path PATH.html  Save Meshcat replay HTML to a specific path
    --no-record             Disable HTML recording
    --name BASENAME         Shared <stem> for all outputs (txt + html)

Default behavior records the Meshcat HTML replay sharing the same
stem as the _Tee log (results/<stem>.txt). The stem is BASENAME when
--name is given, else <task>_<timestamp>. MP4 capture is opt-in via
--drake-frames-dir.

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
import os
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
from control.lcs_formulator import LCSFormulator
from control.admm_solver import C3Solver
from control.task_costs import QuadraticManipulationCost
from control.ci_mpc_c3 import C3MPC
from control.ci_mpc_c3plus import C3PlusMPC
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
        meshcat.SetObject("/goal_marker", shape, ad.Rgba(0.1, 0.9, 0.1, 0.35))
        meshcat.SetTransform(
            "/goal_marker",
            ad.RigidTransform(ad.RotationMatrix(), [target_xy[0], target_xy[1], init_z]),
        )
    elif task_cfg["object_type"] == "tshape":
        # Two-box ghost, yawed to the goal orientation so operator sees the
        # target POSE (not just position). Matches env_builder's Drake-VTK ghost.
        _goal_yaw = float(task_cfg.get("goal_yaw", 0.0))
        _R_goal = ad.RotationMatrix.MakeZRotation(_goal_yaw)
        _T_goal = ad.RigidTransform(_R_goal, [target_xy[0], target_xy[1], init_z])
        for _local_x, _local_yaw, _tag in (
            (+0.05, 0.0,    "/goal_marker/vbar"),
            (-0.05, 1.5708, "/goal_marker/hbar"),
        ):
            _R_local = ad.RotationMatrix.MakeZRotation(_local_yaw)
            _T_local = ad.RigidTransform(_R_local, [_local_x, 0.0, 0.0])
            meshcat.SetObject(_tag, ad.Box(0.16, 0.04, 0.04),
                              ad.Rgba(0.1, 0.9, 0.1, 0.35))
            meshcat.SetTransform(_tag, _T_goal.multiply(_T_local))
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
    if task_cfg["object_type"] == "tshape":
        # Rough T size: max linear extent (crossbar tip to stem back) = 0.20 m.
        # Used only for meshcat camera framing / visual-only helpers, not
        # dynamics — an approximation is fine.
        return 0.20
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
        choices=["pushing", "hard_pushing", "shepherding", "cube_turning", "push_t"],
        help="Task to run (default: pushing)",
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
                        help="Disable HTML recording. Speeds up smoke tests.")
    parser.add_argument("--drake-frames-dir", type=str, default=None,
                        metavar="DIR",
                        help="OOM-safe Drake VTK frame capture: write one PNG per "
                             "control step to DIR (real 3D Drake scene render). "
                             "Encode to MP4 after the sim with ffmpeg.")
    parser.add_argument("--drake-frames-stride", type=int, default=3,
                        metavar="N",
                        help="Capture one Drake VTK frame every N control steps "
                             "(default 3 -> 100Hz/3 ~= 33 fps).")
    parser.add_argument("--max-time", type=float, default=None,
                        help="Override simulation duration in seconds (default: 8.0).")
    parser.add_argument("--early-exit-goal-d", type=float, default=None,
                        metavar="METRES",
                        help="Goal-reached threshold: mark reach when "
                             "goal_dist <= this value (typical 0.05 m). "
                             "Combined with --goal-settle-time, the sim continues "
                             "past the reach for the settle window before breaking. "
                             "When omitted, the loop runs to --max-time.")
    parser.add_argument("--goal-settle-time", type=float, default=1.0,
                        metavar="SEC",
                        help="After the goal is reached, continue the sim for SEC "
                             "seconds so the video shows the arrival before ending. "
                             "Default 1.0 s.  Requires --early-exit-goal-d.")
    parser.add_argument("--force-save-video", action="store_true",
                        help="Encode the Drake-frames mp4 even if the goal was NOT "
                             "reached. Default: encode only on goal-reach.")
    parser.add_argument("--video-out", type=str, default=None, metavar="PATH.mp4",
                        help="Output path for the encoded Drake-frames mp4 "
                             "(runs paint_mode_text.py post-sim). Requires "
                             "--drake-frames-dir.")
    parser.add_argument("--sampling-height", type=float, default=None, metavar="Z",
                        help="Override task_cfg['sampling_height'] in-memory. "
                             "Must be >= sphere radius 0.025 m.")
    parser.add_argument("--extra-log-path", type=str, default=None, metavar="PATH",
                        help="Additionally copy the run log to PATH at end "
                             "(e.g., a Windows-side results sink).")
    parser.add_argument("--pitch-probe", action="store_true",
                        help="Emit per-tick [PITCH-PROBE] rows: EE_z, box pitch "
                             "angle, contact z on box face. For §7.74 diagnosis.")
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
                             "results/: <BASENAME>.txt and <BASENAME>.html. "
                             "When omitted, falls back to <task>_<timestamp>. "
                             "Explicit --video-path PATH still overrides its file.")
    parser.add_argument("--sampling-c3", type=str, nargs="?",
                        const="config/sampling_c3_params.yaml", default=None,
                        metavar="PATH.yaml",
                        help="Enable Venkatesh-2025 sampling-C3 outer controller.\n"
                             "Optional PATH = YAML config "
                             "(default: config/sampling_c3_params.yaml).\n"
                             "Cannot be combined with --cost-bias.")
    parser.add_argument("--solver", choices=["c3", "c3plus"], default="c3",
                        help="Inner ADMM solver. c3=Aydinoglu 2024 (default; "
                             "DEPRECATED — gated diagnostic, shares the LCS "
                             "with c3plus and carries the current contact "
                             "model, but the Lorentz-cone projection is no "
                             "longer the recommended path; prefer c3plus). "
                             "c3plus=Bui 2026 with slack variable η = E x + "
                             "F λ + H u + c (eq. 5c) and Bui eq (12) "
                             "componentwise δ-projection. v1 implements "
                             "normal-direction complementarity only — "
                             "friction LCS is a TODO.")
    parser.add_argument("--c3plus-projection", choices=["componentwise", "lcp"],
                        default="componentwise",
                        help="C3+ δ-step projection variant. "
                             "'componentwise' = Bui 2026 eq (12) per-pair test "
                             "(paper's no-feasibility-guarantee class; FAST). "
                             "'lcp' = Aydinoglu §V-B.3.b LCP-projection retrofit "
                             "applied to the C3+ η-slack structure "
                             "(feasibility-guaranteed; SLOWER per iter). "
                             "Only consulted when --solver c3plus.")
    parser.add_argument("--ee-space", action="store_true",
                        help="Use the paper-aligned EE-space LCS planner "
                             "(Push-Anything §IV-A): x ∈ ℝ^19, u ∈ ℝ^3 EE force. "
                             "Solver dims auto-set to n_x=19, n_u=3. Downstream OSC "
                             "still handles the J^-T mapping to joint torques. "
                             "Replaces the R^7-joint-torque LCS path. Force-limit "
                             "(--torque-limit semantics) is now Newtons.")
    parser.add_argument("--workspace-y-max", type=float, default=None,
                        metavar="YMAX",
                        help="F3 sweep override: override sampling_params."
                             "workspace_xy_max[1] in-memory after yaml load. "
                             "Valid range [-1.0, +1.0]. Only takes effect with "
                             "--sampling-c3.")
    parser.add_argument("--goal-xy", type=str, default=None, metavar="X,Y",
                        help="Contact-free sweep override: override task_cfg "
                             "['goal_xy'] in-memory after load_task(). Format "
                             "'X,Y' in meters (comma-separated). Overrides any "
                             "value from tasks.yaml or --task-id.")
    parser.add_argument("--seed", type=int, default=None, metavar="INT",
                        help="Contact-free sweep: seed the SamplingC3MPC rng "
                             "for deterministic sampling-circle angle draws. "
                             "Only takes effect with --sampling-c3.")
    args = parser.parse_args()

    if args.workspace_y_max is not None and not (-1.0 <= args.workspace_y_max <= 1.0):
        parser.error(f"--workspace-y-max {args.workspace_y_max} out of "
                     f"sane band [-1.0, +1.0].")

    if args.sampling_c3 is not None and args.cost_bias:
        parser.error("--sampling-c3 and --cost-bias are mutually exclusive. "
                     "Use one or the other.")

    # §7.42 — PUSHA_REF_OSC_ALIGN: bundle four matched-reference defaults
    # behind one default-OFF env flag. When set, activates (via setdefault, so
    # individual env vars still override for surgical tests):
    #   * PUSHA_FORCE_ROUTING=u_sol         — route planner u_sol → OSC λ_des
    #     (sampling_based_c3_controller.py:562)
    #   * PUSHA_STAGE5_U_HORIZONTAL=10      — per-axis Fx,Fy bound (admm_solver.py:1054,
    #     ci_mpc_c3plus.py:308; matches anything/sampling_c3plus_options.yaml:34)
    #   * PUSHA_STAGE5_U_VERTICAL=3         — per-axis Fz bound (matches yaml:35)
    #   * PUSHA_STAGE5_R_VECTOR=0.1,0.1,10  — planner R cost = w_R_position·diag(r_vector_position)
    #     = 10·diag(0.01,0.01,1)  (task_costs.py:796, matches yaml:120,125)
    # W_force=1.0 is read at sampling_based_c3_controller.py:258 (construction
    # time) when this flag is set, matching osc_params.yaml LambdaEndEffectorW.
    # Default-OFF preserves all prior behaviour byte-identically.  ORTHOGONAL
    # to LCS_CONTACT_MODEL / LCS_SAMPLE_AWARE_EE / LCS_NORMAL_* / REF_RECONCILE_*.
    if os.environ.get("PUSHA_REF_OSC_ALIGN", "0") == "1":
        os.environ.setdefault("PUSHA_FORCE_ROUTING",       "u_sol")
        os.environ.setdefault("PUSHA_STAGE5_U_HORIZONTAL", "10")
        os.environ.setdefault("PUSHA_STAGE5_U_VERTICAL",   "3")
        os.environ.setdefault("PUSHA_STAGE5_R_VECTOR",     "0.1,0.1,10")
        print("[§7.42] PUSHA_REF_OSC_ALIGN=1 active — "
              "FORCE_ROUTING=u_sol, u_bounds=±10/±3, R=diag(0.1,0.1,10), "
              "W_force→1.0 (matched to dairlib anything/sampling_c3plus_options.yaml + "
              "shared_parameters/osc_params.yaml)",
              flush=True)

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
    # and any other string is taken as an explicit path.  MP4 capture is
    # opt-in via --drake-frames-dir; this block governs Meshcat HTML only.
    if args.no_record:
        html_path  = None
    else:
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

    if args.goal_xy is not None:
        try:
            _gx, _gy = [float(s) for s in args.goal_xy.split(",")]
        except ValueError:
            parser.error(f"--goal-xy {args.goal_xy!r} must be 'X,Y' in meters.")
        _was_goal = task_cfg.get("goal_xy")
        task_cfg["goal_xy"] = [_gx, _gy]
        print(f"[OVERRIDE] goal_xy=[{_gx}, {_gy}] (was {_was_goal})")

    if args.sampling_height is not None:
        if args.sampling_height < 0.025:
            parser.error(f"--sampling-height {args.sampling_height} < "
                         f"sphere radius 0.025 m — pusher would clip the table.")
        _was_sh = task_cfg.get("sampling_height")
        task_cfg["sampling_height"] = float(args.sampling_height)
        print(f"[OVERRIDE] sampling_height={args.sampling_height:.4f} "
              f"(was {_was_sh})")

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
    _add_cam = args.drake_frames_dir is not None
    diagram, plant, panda_model, _, meshcat, plant_ad, context_ad = \
        build_environment(task_cfg, add_camera=_add_cam)

    drake_cam = None
    drake_frames_dir = None
    mode_timeline_fp = None
    if _add_cam:
        drake_frames_dir = Path(args.drake_frames_dir)
        drake_frames_dir.mkdir(parents=True, exist_ok=True)
        drake_cam = diagram.GetSubsystemByName("drake_render_camera")
        # mode_timeline.csv lives next to the frames so the post-sim
        # paint_mode_text.py step can frame-sync without re-parsing run.log.
        mode_timeline_fp = open(
            drake_frames_dir / "mode_timeline.csv", "w", buffering=1)
        mode_timeline_fp.write("step,sim_t,mode,switch\n")
        print(f"[C3] Drake VTK frames -> {drake_frames_dir}  "
              f"(stride={args.drake_frames_stride})")

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
    # §9 Option A: pre-read the sampling-c3 YAML to feed manipuland-ground
    # contact count + object_shape into the LCSFormulator. Falls back to
    # defaults when --sampling-c3 is not provided (env var
    # LCS_EXPLICIT_MANIPULAND_GND overrides regardless).
    _pre_manipuland_gnd = 0
    # d.1 — reference-conforming EE-manipuland admission for the planner LCS.
    # Pre-read from yaml so LCSFormulator gets it at construction.
    _pre_ref_pair_admission_planner = False
    if args.sampling_c3 is not None:
        try:
            import yaml as _yaml_pre
            with open(args.sampling_c3) as _f_pre:
                _raw_pre = _yaml_pre.safe_load(_f_pre) or {}
            _pre_manipuland_gnd = int(_raw_pre.get(
                "lcs_explicit_manipuland_ground_contacts", 0))
            _pre_ref_pair_admission_planner = bool(_raw_pre.get(
                "use_reference_pair_admission_planner_lcs", False))
        except Exception:
            _pre_manipuland_gnd = 0
    formulator = LCSFormulator(
        plant, mu=task_cfg["friction"], obj_body=obj_body,
        plant_ad=plant_ad, context_ad=context_ad,
        lcs_explicit_manipuland_ground_contacts=_pre_manipuland_gnd,
        object_shape=str(task_cfg.get("object_type", "box")),
        ref_pair_admission_planner_lcs=_pre_ref_pair_admission_planner,
    )

    # EE-space planner: solver/cost get the low-dim sizing (n_x=19, n_u=3).
    # R^7 path remains the default unless --ee-space is passed.
    # Supported on BOTH solvers — c3plus (componentwise projection) and c3
    # (LCP projection, Aydinoglu §V-B.3.b feasibility-guaranteed) — so the
    # projection variant can be flipped as a clean CLI test holding all
    # other dimensions (state, input, cost) constant.
    if args.ee_space:
        _solver_n_x, _solver_n_u = 19, 3
        _torque_limit = 30.0    # Newtons under EE-space (EE-force cap)
    else:
        _solver_n_x, _solver_n_u = n_x, n_u
        _torque_limit = 30.0    # Nm under R^7 (joint-torque cap)
    solver     = C3Solver(n_x=_solver_n_x, n_u=_solver_n_u, rho=100.0,
                          math_diag=args.math_diag,
                          mode=args.solver,
                          c3plus_projection=args.c3plus_projection)
    _proj_label = (args.c3plus_projection
                   if args.solver == "c3plus" else "n/a (mode=c3)")
    print(f"[C3] Solver mode: {args.solver}  "
          f"(planner: {'EE-space (R^3 force)' if args.ee_space else 'R^7 joint torque'}, "
          f"c3+ projection: {_proj_label})")
    # Geometry hints for PUSHA_BOX_DPUSH_FIX (default-OFF, box-shape only).
    _obj_shape = str(task_cfg.get("object_type", ""))
    if _obj_shape == "box":
        _obj_half_extent = float(task_cfg["size"][0]) * 0.5
    else:
        _obj_half_extent = None
    from sim.env_builder import PUSHER_RADIUS as _EFFECTIVE_PUSHER_RADIUS
    quad_cost  = QuadraticManipulationCost(
        plant, EE_BODY_NAME, obj_body, task_cfg["cost"], n_x, n_u,
        math_diag=args.math_diag,
        cost_bias=args.cost_bias,
        object_shape=_obj_shape,
        object_half_extent=_obj_half_extent,
        pusher_radius=_EFFECTIVE_PUSHER_RADIUS,
    )
    _MPCClass = C3PlusMPC if args.solver == "c3plus" else C3MPC
    _c3plus_N = int(os.environ.get("PUSHA_C3PLUS_N", "20"))
    if _c3plus_N != 20:
        print(f"[HORIZON-PROBE] PUSHA_C3PLUS_N={_c3plus_N} → "
              f"c3plus horizon override 20 → {_c3plus_N} "
              f"(lookahead {_c3plus_N * 0.05:.2f}s at dt=0.05)", flush=True)
    _mpc_kwargs = dict(
        formulator=formulator,
        solver=solver,
        quadratic_cost=quad_cost,
        horizon=_c3plus_N,
        dt=0.05,
        torque_limit=_torque_limit,
        admm_iter=args.admm_iter,
        math_diag=args.math_diag,
    )
    if args.ee_space:
        _mpc_kwargs["use_ee_space"] = True
    mpc = _MPCClass(**_mpc_kwargs)

    target_xy   = np.array(task_cfg["goal_xy"], dtype=float)
    target_yaw  = float(task_cfg.get("goal_yaw", 0.0))   # radians; 0 for legacy tasks
    ee_frame    = plant.GetFrameByName(EE_BODY_NAME)
    world_frame = plant.world_frame()

    # ------------------------------------------------------------------
    # Optional sampling-C3 outer controller (Venkatesh 2025 §IV-D port)
    # ------------------------------------------------------------------
    if args.sampling_c3 is not None:
        _yaml_path = args.sampling_c3
        sc3_params = SamplingC3Params.from_yaml(_yaml_path)
        if args.workspace_y_max is not None:
            _was = sc3_params.sampling_params.workspace_xy_max[1]
            sc3_params.sampling_params.workspace_xy_max[1] = args.workspace_y_max
            print(f"[OVERRIDE] workspace_xy_max[1]={args.workspace_y_max} (was {_was})")
        # D6: per-task sampling_height override. pushing/hard_pushing set 0.03
        # (sub-CoM, restoring tip moment); cube_turning/shepherding read the
        # sampler default. tasks.yaml is the source of truth; absent → no override.
        _task_sample_h = task_cfg.get("sampling_height")
        if _task_sample_h is not None:
            _was_sh = sc3_params.sampling_params.sampling_height
            sc3_params.sampling_params.sampling_height = float(_task_sample_h)
            print(f"[OVERRIDE] sampling_height={float(_task_sample_h):.3f} "
                  f"(was {_was_sh:.3f}, per-task '{task_name}')")
        # With the new IK-based --prepositioned pose, k=0 captures the
        # ~30k alignment bonus (vs zero contact for every k>=1 at sampling
        # radius 0.18m), so decide_mode picks "c3" via kToC3Cost on step 1
        # under its own cost differential. Forcing the initial mode would
        # mask whether the pose actually does what we want.
        # Stage A — Reposition PWL trajectory port (alignment plan §3).
        # Env var PUSHA_REPOSITION_PWL=1 flips the dispatcher to the new
        # full-N-knot Cartesian PWL trajectory path (fed to existing OSC
        # position-tracking). Default OFF → legacy per-tick march path.
        _env_pwl = os.environ.get("PUSHA_REPOSITION_PWL", None)
        if _env_pwl == "1":
            sc3_params.use_reposition_pwl_trajectory = True
            print("[STAGE-A-PWL] PUSHA_REPOSITION_PWL=1 → using "
                  "RepositionTrajectory + OSC position-tracking "
                  "(Stage A path).", flush=True)
        elif _env_pwl == "0":
            sc3_params.use_reposition_pwl_trajectory = False
            print("[STAGE-A-PWL] PUSHA_REPOSITION_PWL=0 → legacy "
                  "per-tick reposition path", flush=True)
        _rng = np.random.default_rng(args.seed) if args.seed is not None else None
        if args.seed is not None:
            print(f"[OVERRIDE] seed={args.seed} (rng=np.random.default_rng)")
        # Stage C cadence discriminator — PUSHA_CONTROL_HZ env gate.
        # Default 100 → dt_ctrl=0.01 (baseline); set to 1000 → dt_ctrl=0.001
        # to measure whether the ADMM converges when the state moves ~10×
        # less per control tick (reference's actual cadence). Override
        # sc3_params.dt_osc/dt_mpc BEFORE wrapper construction so the
        # wrapper's _dt_osc/_dt_mpc fields pick up the new rate. The PWL
        # reposition is sim-t-parameterized (eval(sim_t)) → naturally lands
        # the EE the same way at 1 kHz; only the control/ADMM cadence
        # changes (Stage A reposition isolation guard preserved).
        import os as _os_cad
        _ctrl_hz = float(_os_cad.environ.get("PUSHA_CONTROL_HZ", "100"))
        _dt_ctrl_pass = 1.0 / _ctrl_hz
        if abs(_ctrl_hz - 100.0) > 1e-9:
            sc3_params.dt_osc = _dt_ctrl_pass
            sc3_params.dt_mpc = _dt_ctrl_pass
            print(f"[ENV]  PUSHA_CONTROL_HZ={_ctrl_hz} → "
                  f"dt_ctrl={_dt_ctrl_pass:.4f}s, "
                  f"sc3_params.dt_osc={sc3_params.dt_osc:.4f}s, "
                  f"sc3_params.dt_mpc={sc3_params.dt_mpc:.4f}s")
        mpc = SamplingC3MPC(
            base_mpc=mpc,
            plant=plant,
            ee_frame=ee_frame,
            obj_body=obj_body,
            params=sc3_params,
            dt_ctrl=_dt_ctrl_pass,
            log_diag=True,
            start_in_c3_mode=False,
            rng=_rng,
            diagram=diagram,   # required if reposition_params.traj_type==kIK
        )
        print(f"[GS] SamplingC3MPC enabled (config: {_yaml_path})")
        print(f"[GS]   strategy={sc3_params.sampling_params.sampling_strategy.name} "
              f"num_add_c3={sc3_params.sampling_params.num_additional_samples_c3} "
              f"num_add_repos={sc3_params.sampling_params.num_additional_samples_repos}")
        print(f"[GS]   w_align={sc3_params.w_align}  w_travel={sc3_params.w_travel}")
        print(f"[GS]   reposition: traj_type={sc3_params.reposition_params.traj_type.name} "
              f"z_safe={sc3_params.reposition_params.pwl_waypoint_height}m "
              f"speed={sc3_params.reposition_params.speed}m/s")
        if getattr(sc3_params, "use_cost_lcs_ranking", False):
            print(f"[GS]   cost-LCS ranking: use_cost_lcs_ranking=True "
                  f"n_ee_top_k=2 force_top_k_ee_box=True "
                  f"Kp_ee_pd={sc3_params.Kp_for_ee_pd_rollout} "
                  f"Kd_ee_pd={sc3_params.Kd_for_ee_pd_rollout} "
                  f"per_sample_context=True "
                  f"(reference push_t "
                  f"resolve_contacts_to_for_cost=[0,2,3] → "
                  f"2 EE-T + 3 T-GND cost-LCS, "
                  f"reference UpdateContext per sample)")

    print(f"[C3] Goal: {target_xy}  |  Meshcat: {meshcat.web_url()}")

    # Goal ghost + trajectory markers (set up once before sim starts)
    _setup_meshcat_markers(meshcat, target_xy, task_cfg)

    print("[C3] Running simulation ...")

    if html_path:
        meshcat.StartRecording()

    # ------------------------------------------------------------------
    # Joint limit constants for arm safety check
    # ------------------------------------------------------------------
    _Q_LO = np.array([-2.897, -1.763, -2.897, -3.072, -2.897, -0.0175, -2.897])
    _Q_HI = np.array([ 2.897,  1.763,  2.897, -0.0698, 2.897,  3.752,   2.897])

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------
    sim_time      = 0.0
    # Stage C cadence discriminator — read PUSHA_CONTROL_HZ AGAIN here for
    # the simulator loop period. The same gate was applied above (before
    # the SamplingC3MPC construction) to set sc3_params.dt_osc/dt_mpc.
    import os as _os_cad2
    dt_ctrl       = 1.0 / float(_os_cad2.environ.get("PUSHA_CONTROL_HZ", "100"))
    max_time      = args.max_time if args.max_time is not None else 8.0
    step          = 0
    # §7.74 goal-reach-then-settle: on first tick with goal_dist <=
    # --early-exit-goal-d, record the reach step and continue for
    # --goal-settle-time seconds so the video captures the arrival.
    _goal_reach_step = None
    _goal_reach_t = None
    _settle_until_t = None
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

        # Negated: Drake returns generalized gravity force (the force gravity
        # exerts), we want compensation torque. See scripts/test_gravity_sign.py.
        tau_g = -plant.CalcGravityGeneralizedForces(plant_ctx)
        # === Stage 2: lookahead sub-goal (env-gated PUSHA_LOOKAHEAD_STEP) ===
        # Reference: dairlib_sampling_c3 goal_generator.cc:401-407 +
        # anything/parameters/goal_params.yaml:19 (lookahead_step_size: 0.15).
        # Plans toward a REACHABLE 15cm sub-goal each tick rather than the
        # distant final goal. When PUSHA_LOOKAHEAD_STEP is unset / 0, this is
        # a no-op (passes target_xy unchanged — preserves current behavior).
        _lookahead = float(os.environ.get("PUSHA_LOOKAHEAD_STEP", "0") or "0")
        if _lookahead > 0.0:
            _obj_xy_now = np.array([current_q[obj_x_idx], current_q[obj_y_idx]])
            _delta_vec  = target_xy - _obj_xy_now
            _dist       = float(np.linalg.norm(_delta_vec))
            if _dist > 1e-9:
                _step   = min(_lookahead, _dist)
                _effective_target_xy = _obj_xy_now + (_delta_vec / _dist) * _step
            else:
                _effective_target_xy = target_xy
            if step % 50 == 0:
                print(f"[LOOKAHEAD] step={step} obj=({_obj_xy_now[0]:+.3f},{_obj_xy_now[1]:+.3f}) "
                      f"sub_goal=({_effective_target_xy[0]:+.3f},{_effective_target_xy[1]:+.3f}) "
                      f"step_size={_step:.3f}m dist_remaining={_dist:.3f}m", flush=True)
        else:
            _effective_target_xy = target_xy
        u_opt = mpc.compute_control(current_q, current_v, plant_ctx,
                                    _effective_target_xy, target_yaw=target_yaw)
        # === end Stage 2 ===

        # Update predicted-trajectory markers in Meshcat
        if mpc.last_x_seq is not None:
            _update_predicted_trajectory(
                meshcat, mpc.last_x_seq, obj_x_idx, obj_y_idx, obj_z_idx
            )

        # Singular gravity-comp ownership: the main loop always applies
        # tau_g[:n_u] + u_opt. Executors/trackers strip their internal
        # gravity-comp (OSC: bias = Cv only; PWL: u_raw = u_pd only).
        # Replaces the stale free-vs-c3 conditional, which assumed only
        # the FREE-mode tracker was self-contained — but the OSC executor
        # was also self-contained in c3 mode, and the old else-branch
        # double-counted gravity (see audit_output/wire_probe/
        # CLIMB_SOURCE_REPORT.md).
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

        # Drake VTK frame capture: one PNG per stride ticks (streamed to disk
        # so RAM stays flat — OOM-safe path for 16s runs).  mode_timeline.csv
        # gets one row per captured frame for paint_mode_text.py to consume.
        if drake_cam is not None and step % args.drake_frames_stride == 0:
            _cam_ctx = drake_cam.GetMyContextFromRoot(context)
            _img = drake_cam.color_image_output_port().Eval(_cam_ctx)
            _arr = np.asarray(_img.data, dtype=np.uint8)  # (H, W, 4) RGBA
            try:
                from PIL import Image as _PILImage
                _PILImage.fromarray(_arr, mode="RGBA").save(
                    drake_frames_dir / f"frame_{step:06d}.png", optimize=False)
            except ImportError:
                import imageio.v2 as _imageio
                _imageio.imwrite(
                    drake_frames_dir / f"frame_{step:06d}.png", _arr)
            _mode = getattr(mpc, "last_mode", "n/a")
            _switch = getattr(mpc, "last_switch_reason", None)
            _switch_name = _switch.name if _switch is not None else ""
            mode_timeline_fp.write(
                f"{step},{sim_time:.4f},{_mode},{_switch_name}\n")

        sim_time += dt_ctrl
        step     += 1
        simulator.AdvanceTo(sim_time)

        # Sink 3 vs Sink 4 diagnostic: Drake-realized contact force on the
        # EE-box pair. Filters out box-ground using LCS formulator's geom IDs.
        try:
            _cr = plant.get_contact_results_output_port().Eval(plant_ctx)
            _eebox_fmag = 0.0
            _gate_F_W = None
            _gate_n_BA = None
            _gate_ia_ee = None
            _pp_contact_pt = None
            _n_pairs = _cr.num_point_pair_contacts()
            for _i in range(_n_pairs):
                _info = _cr.point_pair_contact_info(_i)
                _pp = _info.point_pair()
                _ia, _ib = _pp.id_A, _pp.id_B
                _is_ee_box = (
                    (_ia in formulator._ee_geom_ids and _ib in formulator._manipuland_geom_ids)
                    or (_ib in formulator._ee_geom_ids and _ia in formulator._manipuland_geom_ids)
                )
                if _is_ee_box:
                    _Fvec = _info.contact_force()
                    _eebox_fmag = float(np.linalg.norm(_Fvec))
                    # GATE-only: capture force vector + contact normal +
                    # which side is the EE so we can sign-correct n_BA.
                    _gate_F_W   = np.asarray(_Fvec, dtype=float).reshape(3)
                    _gate_n_BA  = np.asarray(_pp.nhat_BA_W, dtype=float).reshape(3)
                    _gate_ia_ee = (_ia in formulator._ee_geom_ids)
                    # §7.74: contact point in world for pitch-probe.
                    try:
                        _pp_contact_pt = np.asarray(
                            _info.contact_point(), dtype=float).reshape(3)
                    except Exception:
                        _pp_contact_pt = None
                    break
            print(f"[DRAKE-CONTACT] step={step} n_pairs={_n_pairs} "
                  f"ee_box_normal={_eebox_fmag:.3f}", flush=True)
            # [GATE-CONTACT] one line per step. Always emitted (zero vec when
            # no EE-box contact). Box quat lives at pos_start+[0..3].
            if _gate_F_W is None:
                _gate_F_W  = np.zeros(3)
                _gate_n_BA = np.zeros(3)
                _gate_ia_ee = False
            _box_q = current_q[pos_start : pos_start + 4]
            # Drake convention (PointPairContactInfo):
            #   contact_force() = force ON body B at the contact point
            #   nhat_BA_W       = unit normal in world frame, B → A
            # If A_is_ee: A = EE, B = box → F_W is already force on box,
            #             and nhat_BA_W points box→EE = OUT of box surface.
            # If !A_is_ee: A = box, B = EE → F_W is force on EE
            #              (flip for box), and nhat_BA_W points EE→box =
            #              INTO box surface (flip for "out of box").
            _F_on_box   = _gate_F_W  if _gate_ia_ee else -_gate_F_W
            _n_face_out = _gate_n_BA if _gate_ia_ee else -_gate_n_BA
            print(
                f"[GATE-CONTACT] step={step} "
                f"F_W=({_gate_F_W[0]:+.4f},{_gate_F_W[1]:+.4f},{_gate_F_W[2]:+.4f}) "
                f"F_on_box=({_F_on_box[0]:+.4f},{_F_on_box[1]:+.4f},{_F_on_box[2]:+.4f}) "
                f"n_face_out=({_n_face_out[0]:+.4f},{_n_face_out[1]:+.4f},{_n_face_out[2]:+.4f}) "
                f"A_is_ee={int(_gate_ia_ee)} "
                f"box_q=({_box_q[0]:+.5f},{_box_q[1]:+.5f},{_box_q[2]:+.5f},{_box_q[3]:+.5f}) "
                f"box_p=({current_q[obj_x_idx]:+.5f},{current_q[obj_y_idx]:+.5f},{current_q[obj_z_idx]:+.5f}) "
                f"ee_p=({ee_pos[0]:+.5f},{ee_pos[1]:+.5f},{ee_pos[2]:+.5f})",
                flush=True,
            )

            # §7.74 [PITCH-PROBE]: EE_z, box CoM z, contact z on box face
            # (world frame), tip angle from vertical.  Tip = angle between
            # box body +Z and world +Z, computed as acos(R(q)[2,2]) which is
            # gimbal-lock-free at 90 deg (the observed §7.73 outcome).
            if args.pitch_probe:
                _qw = float(_box_q[0]); _qx = float(_box_q[1])
                _qy = float(_box_q[2]); _qz = float(_box_q[3])
                _r22 = 1.0 - 2.0 * (_qx*_qx + _qy*_qy)
                _r22 = max(-1.0, min(1.0, _r22))
                _tip_deg = float(np.degrees(np.arccos(_r22)))
                _ee_z_now = float(ee_pos[2])
                _box_z_com = float(current_q[obj_z_idx])
                if _pp_contact_pt is not None:
                    _cz = float(_pp_contact_pt[2])
                    _cz_rel = _cz - _box_z_com
                    _has_contact = 1
                    _cz_str = f"{_cz:+.5f}"
                    _cz_rel_str = f"{_cz_rel:+.5f}"
                else:
                    _has_contact = 0
                    _cz_str = "nan"
                    _cz_rel_str = "nan"
                print(f"[PITCH-PROBE] step={step} t={sim_time:.3f}s "
                      f"ee_z={_ee_z_now:+.5f} box_z_com={_box_z_com:+.5f} "
                      f"contact_z={_cz_str} contact_z_rel_com={_cz_rel_str} "
                      f"has_contact={_has_contact} tip_deg={_tip_deg:.2f}",
                      flush=True)
        except Exception as _e:
            print(f"[DRAKE-CONTACT] step={step} ERROR={type(_e).__name__}: {_e}", flush=True)

        # Goal-reach + settle (opt-in via --early-exit-goal-d).  Mark the
        # first tick that hits the threshold, then continue for
        # --goal-settle-time seconds so the video shows the arrival before
        # breaking.  Pre-reach behavior unchanged.
        if args.early_exit_goal_d is not None:
            _ex_obj_xy = np.array([current_q[obj_x_idx], current_q[obj_y_idx]])
            _ex_gd = float(np.linalg.norm(_ex_obj_xy - target_xy))
            if _goal_reach_step is None and _ex_gd <= args.early_exit_goal_d:
                _goal_reach_step = step
                _goal_reach_t = sim_time
                _settle_until_t = sim_time + float(args.goal_settle_time)
                print(f"[GOAL-REACH] step={step} t={sim_time:.3f}s "
                      f"goal_d={_ex_gd:.4f}m <= threshold={args.early_exit_goal_d:.4f}m "
                      f"-> settling +{args.goal_settle_time:.2f}s then breaking")
            if _goal_reach_step is not None and sim_time >= _settle_until_t:
                print(f"[GOAL-SETTLE-DONE] step={step} t={sim_time:.3f}s "
                      f"(reached step={_goal_reach_step} at t={_goal_reach_t:.3f}s) "
                      f"-> breaking sim loop")
                break

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

    if mode_timeline_fp is not None:
        mode_timeline_fp.close()
        print(f"[VIDEO] mode_timeline -> {drake_frames_dir / 'mode_timeline.csv'}")

    # §7.74 auto-encode: run paint_mode_text.py on the captured Drake frames
    # when either the goal was reached OR --force-save-video was set.
    # Non-success runs stay silent unless --force-save-video, to keep sweep
    # output clean.
    _goal_reached = _goal_reach_step is not None
    if drake_frames_dir is not None and (args.force_save_video or _goal_reached):
        _video_out = args.video_out or f"results/{stem}.mp4"
        _video_out_path = Path(_video_out)
        _video_out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[VIDEO] encoding {drake_frames_dir} -> {_video_out_path}  "
              f"(goal_reached={_goal_reached}, "
              f"force_save={args.force_save_video})")
        import subprocess as _sp
        _paint = (Path(__file__).resolve().parent
                  / "tools" / "visualizer" / "paint_mode_text.py")
        _rc = _sp.run([
            sys.executable, str(_paint),
            "--frames-dir", str(drake_frames_dir),
            "--output", str(_video_out_path),
            "--fps", "30",
        ]).returncode
        if _rc == 0:
            print(f"[VIDEO] mp4 saved -> {_video_out_path}")
        else:
            print(f"[VIDEO] paint_mode_text.py failed rc={_rc}")

    # §7.74 side-load the log to a Windows sink (or any secondary path).
    if args.extra_log_path is not None:
        import shutil as _sh
        _dst = Path(args.extra_log_path)
        _dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            _sh.copy2(_log_path, _dst)
            print(f"[LOG-COPY] {_log_path} -> {_dst}")
        except Exception as _e:
            print(f"[LOG-COPY] failed: {type(_e).__name__}: {_e}")


if __name__ == "__main__":
    main()
