"""
C3 Contact-Implicit MPC — main entry point.

Usage
-----
    python main.py [pushing|push_t] [--task-id N] [--max-time SEC]
                   [--sampling-c3 PATH.yaml] [--name BASENAME] [--seed INT]

Canonical launches (flag-minimal since the 2026-08-05 CLI prune):
    python main.py push_t  --max-time 600 --sampling-c3 config/sampling_c3_kik_t.yaml --name X
    python main.py pushing --task-id 4 --max-time 180 --sampling-c3 config/sampling_c3_kik.yaml --name X

Outputs: results/<stem>.txt run log (stem = --name or <task>_<timestamp>).
Replay video (reference-shaped, post-hoc from the log — the reference's
process_lcm_logs.py paradigm): scripts/make_run_video.sh <stem>.
Live view during the run: Meshcat at http://127.0.0.1:7000

MPC parameters (reference-conformant defaults, dairlib push_anything_dev@257e3ed):
    horizon    = 7     steps (c3plus; c3 uses 5)   sampling_c3plus_options.yaml:20
    admm_iter  = 3     ADMM iters per solve (override with --admm-iter)
    dt         = 0.075 s planning timestep (c3plus; c3 uses 0.1)
                       sampling_c3plus_options.yaml:62 (planning_dt_position/pose)
    dt_ctrl    = 0.075 s planner cadence (=dt) → 13.3 Hz outer loop
    dt_osc     = 0.001 s OSC cadence            → 1000 Hz inner loop
                       matches osc_params.yaml:2 controller_frequency
    torque_lim = URDF per-joint effort limits (87/87/87/87/12/12/12 Nm)
    rho        = 3     ADMM penalty init (sampling_c3plus_options.yaml rho_init)
"""
import argparse
import os
import stat
import sys
from pathlib import Path
import yaml
import numpy as np
import pydrake.all as ad

from sim.env_builder import (
    build_environment,
    _INITIAL_ARM_Q_SEED,
    EE_BODY_NAME,
    JACK_GHOST_CAPS,
    compute_safe_init_arm_q,
    init_rotation,
)
from control.lcs_formulator import LCSFormulator
from control.admm_solver import C3Solver
from control.task_costs import QuadraticManipulationCost
from control.ci_mpc_c3 import C3MPC
from control.ci_mpc_c3plus import C3PlusMPC
from control.sampling_c3 import SamplingC3Controller, SamplingC3Params
from control.sampling_c3.goal_generator import (
    JackRandomGoalGenerator,
    KNOMINAL_NAMES_JACK,
    TRIPOD_NAMES,
    geodesic_angle,
    orientation_lookahead,
    topple_roll_plan,
    tripod_id,
)


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

def _update_jack_goal_marker(meshcat, target_xy, goal_quat, z) -> None:
    """Move the jack's three-capsule ghost to a (possibly new) goal pose.

    Split out of _setup_meshcat_markers so kRandom re-goaling can relocate
    the ghost without re-creating the meshcat objects.
    """
    _T_goal = ad.RigidTransform(
        ad.RotationMatrix(ad.Quaternion(
            goal_quat[0], goal_quat[1], goal_quat[2], goal_quat[3])),
        [target_xy[0], target_xy[1], z])
    for _rpy, _rgba, _tag in JACK_GHOST_CAPS:
        meshcat.SetTransform(
            f"/goal_marker/{_tag}", _T_goal.multiply(ad.RigidTransform(
                ad.RotationMatrix(ad.RollPitchYaw(*_rpy)), [0.0, 0.0, 0.0])))


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
    elif task_cfg["object_type"] == "jack":
        # Three-capsule ghost at the goal POSE (full quaternion, not a yaw).
        _gq = np.asarray(task_cfg.get("goal_quat", [1.0, 0.0, 0.0, 0.0]), float)
        _gq = _gq / float(np.linalg.norm(_gq))
        for _rpy, _rgba, _tag in JACK_GHOST_CAPS:
            # Pastel per-capsule tints (shared JACK_GHOST_CAPS table): colour
            # is the only signal for WHICH capsule goes where under the goal
            # quaternion; whitened + translucent so the ghost never reads as
            # the saturated real jack (2026-08-16 green-blob legibility fix).
            meshcat.SetObject(f"/goal_marker/{_tag}", ad.Capsule(0.025, 0.125),
                              ad.Rgba(*_rgba[:3], 0.35))
        _update_jack_goal_marker(meshcat, target_xy, _gq, init_z)
    elif task_cfg["object_type"] == "hshape":
        # Three-box ghost matching _hshape_sdf's decomposition.
        _goal_yaw = float(task_cfg.get("goal_yaw", 0.0))
        _T_goal = ad.RigidTransform(ad.RotationMatrix.MakeZRotation(_goal_yaw),
                                    [target_xy[0], target_xy[1], init_z])
        for _lx, _sz, _tag in (
            (-0.044, (0.024, 0.128, 0.032), "/goal_marker/lbar"),
            (+0.044, (0.024, 0.128, 0.032), "/goal_marker/rbar"),
            ( 0.000, (0.064, 0.024, 0.032), "/goal_marker/cbar"),
        ):
            meshcat.SetObject(_tag, ad.Box(*_sz), ad.Rgba(0.1, 0.9, 0.1, 0.35))
            meshcat.SetTransform(_tag, _T_goal.multiply(
                ad.RigidTransform(ad.RotationMatrix(), [_lx, 0.0, 0.0])))
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


def build_planner_workspace_bounds(sc3_params) -> list:
    """Planner workspace state rows (reference cc:995-1025) for the EE-space
    LCS: [(state_idx, lo, hi), ...] bounding the EE position AND object
    position slots, widened by the margin. Returns [] when the config does
    not opt in (planner_workspace_x is None)."""
    if sc3_params.planner_workspace_x is None:
        return []
    _pw_m = float(sc3_params.planner_workspace_margin)
    _pw = []
    # EE-space state is x = [box_q(quat 0-3 + pos 4-6), p_ee(7-9), box_v,
    # v_ee] (lcs_formulator.linearize_discrete_ee_space) — box FIRST, unlike
    # the reference's EE-first layout, so the reference slot numbers (EE 0-2,
    # obj 7-9) do not transfer. p142 regression: copying them bounded the
    # box quaternion and made every QP primal-infeasible.
    for _slot0 in (7, 4):   # EE pos, object pos
        for _axis, _lims in enumerate((sc3_params.planner_workspace_x,
                                       sc3_params.planner_workspace_y,
                                       sc3_params.planner_workspace_z)):
            if _lims is None:
                continue
            _pw.append((_slot0 + _axis,
                        float(_lims[0]) - _pw_m,
                        float(_lims[1]) + _pw_m))
    return _pw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # PORT_SECTION_TIMING=1: enable the existing profiling/section_timer
    # accumulators (admm.qp_build / admm.osqp_solve / admm.z_update /
    # admm.final_qp / lcs.*) on a normal run and print the report at exit.
    # Default-off; zero per-call overhead when disabled.
    if os.environ.get("PORT_SECTION_TIMING", "0") == "1":
        import atexit

        import profiling.section_timer as _ST
        _ST.ENABLED = True
        # PORT_SECTION_DISTRIBUTIONS=1 additionally retains every per-call
        # sample so the exit report can show p50/p90/p95/p99/max. Totals
        # alone hide tails, and tails are what matter for real-time MPC.
        _ST.DISTRIBUTIONS = (
            os.environ.get("PORT_SECTION_DISTRIBUTIONS", "0") == "1")
        atexit.register(lambda: print(_ST.report(top_n=30), flush=True))
        if _ST.DISTRIBUTIONS:
            atexit.register(
                lambda: print(_ST.distribution_report(top_n=30), flush=True))

    parser = argparse.ArgumentParser(
        description="C3 Contact-Implicit MPC",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # CLI prune 2026-08-05: removed the stale/off-reference flags —
    # --reset-every, --drake-frames-dir/-stride, --force-save-video,
    # --video-out, --hud-video (in-sim video pipeline superseded by
    # scripts/make_run_video.sh), --early-exit-goal-d/--goal-settle-time/
    # --early-exit-orient-err (superseded by the in-controller ACHIEVED
    # latch), --sampling-height/--workspace-y-max/--goal-xy (off-reference
    # override levers), --pitch-probe (pre-DIAG_* diagnostic),
    # --extra-log-path (superseded by scripts/sync_results_to_d.sh),
    # --ee-space (no-op; the ee_space ATTRIBUTE is still derived from
    # --r7 below).
    # Task choices come from config/tasks.yaml so imported anything objects
    # (Fig 8 campaign 2026-08-15) are runnable without touching argparse.
    # NOTE: that means ALL tasks.yaml keys are selectable — including the
    # stale hard_pushing / shepherding / cube_turning entries. The comment
    # here used to claim they "remain excluded via the legacy allowlist
    # union"; no such union exists. The hardcoded list below is only the
    # fallback for when tasks.yaml cannot be read.
    try:
        import yaml as _yaml_choices
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "config", "tasks.yaml")) as _f:
            _task_choices = sorted(_yaml_choices.safe_load(_f)["tasks"].keys())
    except Exception:
        _task_choices = ["pushing", "push_t", "push_h", "push_jack"]
    parser.add_argument(
        "task", nargs="?", default="pushing",
        choices=_task_choices,
        help="Task to run (default: pushing)",
    )
    parser.add_argument("--task-id", type=int, choices=[1, 2, 3, 4], default=None,
                        help="Directional task ID from config/directional_tasks.json "
                             "(1=north, 2=east, 3=south, 4=west). Overrides tasks.yaml goal.")
    # (--video-path / --no-record removed 2026-08-05 with the StaticHtml
    #  retirement — replay video = scripts/make_run_video.sh over the log.)
    parser.add_argument("--max-time", type=float, default=None,
                        help="Override simulation duration in seconds "
                             "(default: task max_time, otherwise 8.0).")
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
                             "(default: config/sampling_c3_params.yaml).")
    # 2026-08-05 CLI prune: default flipped c3 → c3plus. The c3 default
    # contradicted its own DEPRECATED help text; reference push_t AND
    # anything both run C3+ (projection_type 'C3+'). The c3 choice stays
    # as the Lorentz-projection falsification lever.
    parser.add_argument("--solver", choices=["c3", "c3plus"], default="c3plus",
                        help="Inner ADMM solver. c3plus=Bui 2026 (DEFAULT; "
                             "reference projection for both tasks). "
                             "c3=Aydinoglu 2024 Lorentz-cone projection — "
                             "DEPRECATED falsification lever. "
                             "c3plus uses slack variable η = E x + "
                             "F λ + H u + c (eq. 5c) and Bui eq (12) "
                             "componentwise δ-projection. v1 implements "
                             "normal-direction complementarity only — "
                             "friction LCS is a TODO.")
    # NOTE (2026-07-22): --c3plus-projection removed. The port previously
    # exposed a `lcp` variant (Aydinoglu §V-B.3.b LCP retrofit on the
    # C3+ η-slack structure) alongside the paper's `componentwise`
    # (Bui 2026 eq 12) projection. The LCP variant was port-added
    # (reference push_t uses projection_type: 'C3+' == componentwise
    # exclusively) and empirically convergence-limited at reference's
    # admm_iter=3 (p28 dashboard capture: gap_lam median 12.8, tight_goal
    # FAIL). Componentwise is the reference C3+ projection and delivers
    # tight_goal PASS at admm_iter=3. The LCP path is gone; only the
    # componentwise projection remains inside _solve_c3plus.
    # 2026-07-28 divergence removal: the EE-space LCS planner is the
    # DEFAULT for ALL tasks. The reference plans every sampling-c3 task
    # with the point-EE simple model (franka_sampling_c3_controller.cc:
    # 143-146 DRAKE_DEMANDs no orientation because the LCS plant IS the
    # simple EE model); the R^7 full-plant planner was a port-era
    # divergence (p106 arc) with no reference analog. --r7 opts back into
    # the legacy R^7 joint-torque planner for falsification runs only
    # (the --ee-space no-op flag was pruned 2026-08-05; the args.ee_space
    # ATTRIBUTE below is still the load-bearing planner selector).
    parser.add_argument("--r7", action="store_true",
                        help="LEGACY: use the port-only R^7 joint-torque "
                             "full-plant LCS planner instead of the "
                             "reference EE-space planner. Falsification "
                             "runs only — no reference analog.")
    parser.add_argument("--seed", type=int, default=None, metavar="INT",
                        help="Contact-free sweep: seed the SamplingC3Controller rng "
                             "for deterministic sampling-circle angle draws. "
                             "Only takes effect with --sampling-c3.")
    args = parser.parse_args()

    # 2026-07-28 divergence removal: EE-space is the default planner;
    # --r7 is the explicit legacy opt-out (see the --ee-space help text).
    args.ee_space = not args.r7

    # (Reference-alignment env bundle removed; all downstream reference
    # settings are now unconditional defaults.)

    task_name   = args.task
    # init_q is computed below via compute_safe_init_arm_q — IK-solved so
    # EE starts OPPOSITE goal direction at safe altitude above object top.
    # Prior INITIAL_ARM_Q placed EE directly over box CoM, causing PWL
    # descent to pass through box top (impact tumble).

    Path("results").mkdir(exist_ok=True)
    from datetime import datetime
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Shared basename: --name BASENAME wins; otherwise <task>_<timestamp>.
    if args.name is not None:
        stem = args.name
    else:
        stem = f"{task_name}_{run_stamp}"

    # 2026-07-22: [RUN-META] once per run so the dashboard's RUN region
    # has a durable file-log source for git/seed/flags (no live-process
    # reads inside paint_log_dashboard.py).
    try:
        import subprocess as _sp_meta
        _git_head = _sp_meta.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=_sp_meta.DEVNULL).decode().strip()
    except Exception:
        _git_head = "unknown"
    _seed_str = str(args.seed) if args.seed is not None else "unseeded"
    _flags = (f"solver={args.solver} ee_space={args.ee_space} "
              f"admm_iter={args.admm_iter} "
              f"max_time={args.max_time}")
    print(f"[RUN-META] git={_git_head} seed={_seed_str} task={task_name} "
          f"stem={stem} flags=[{_flags}]", flush=True)
    # 2026-07-27: full launch provenance. The short flags list above hides
    # both the argv tail (--sampling-c3 YAML path!) and the env-gated
    # feature set — reconstructing them from banners cost real time twice:
    # the YAML mismatch (p106 recipe hunt) and the missed
    # REFCONF_USE_G_MATRIX=1, which silently ran p106-p111 on the G-off
    # rho=100 ADMM and contaminated the series comparison. Dump both,
    # verbatim, once per run.
    _env_gates = {k: v for k, v in sorted(os.environ.items())
                  if k.startswith(("REFCONF_", "PORT_", "DIAG_"))}
    print(f"[RUN-META-ARGV] {' '.join(sys.argv)}", flush=True)
    print(f"[RUN-META-ENV] "
          f"{' '.join(f'{k}={v}' for k, v in _env_gates.items()) or '(none)'}",
          flush=True)

    # (Meshcat StaticHtml "replay" emission retired 2026-08-05: it had NO
    # reference analog — the reference records nothing live and renders
    # video post-hoc from logs (process_lcm_logs.py) — and it never worked:
    # every html ever written was a static scene snapshot (zero animation
    # tracks, byte-identical sizes). The port's reference-shaped replay
    # path is scripts/make_run_video.sh over the run log.)

    _log_path = f"results/{stem}.txt"
    # Guard against the shell ALSO redirecting stdout to this same path
    # (`main.py --name X > results/X.txt`). That is the natural thing to type,
    # because --name X is what makes us write results/X.txt -- but it gives two
    # independent file handles at independent offsets writing one file, which
    # shreds the log: lines spliced mid-token, whole lines lost, invalid UTF-8
    # from half-written glyphs, trailing NUL blocks. It cost a day of analysis
    # on 2026-08-09 (counts undercounted, a banner "missing" that was really
    # overwritten). If stdout already points at this file, skip the tee so
    # there is exactly one writer.
    _tee_stdout = True
    try:
        _st_out = os.fstat(sys.__stdout__.fileno())
        if stat.S_ISREG(_st_out.st_mode):
            _st_log = os.stat(_log_path) if os.path.exists(_log_path) else None
            if (_st_log is not None
                    and (_st_out.st_dev, _st_out.st_ino)
                    == (_st_log.st_dev, _st_log.st_ino)):
                _tee_stdout = False
    except (OSError, ValueError, AttributeError):
        pass    # non-file stdout (tty/pipe) — tee normally

    if _tee_stdout:
        _log = open(_log_path, "w", buffering=1)
        sys.stdout = _Tee(sys.__stdout__, _log)
    else:
        print(f"[C3] NOTE: stdout is already redirected to {_log_path}; "
              f"skipping the internal tee to avoid double-writing it "
              f"(drop the shell redirect — --name already writes this file).",
              flush=True)
    print(f"[C3] Log: {_log_path}")

    print(f"[C3] Task: {task_name}")

    task_cfg = load_task(task_name)

    # ------------------------------------------------------------------
    # kRandom goal semantics (reference goal_params.yaml goal_mode: 0).
    # Constructed HERE — before build_environment — so an initial-goal
    # draw (krandom_draw_initial_goal, user directive 2026-08-16: the
    # random-quaternion chase IS the jacktoy task) mutates task_cfg and
    # every downstream consumer (Drake-scene ghost, meshcat markers,
    # cost, dispatcher, log header) sees the drawn goal with no extra
    # wiring. Reference boots from the fixed target and randomizes only
    # on success (ctor cc:67-69; verified against the captured
    # REF_jacktoy run) — the draw samples the reference's steady-state
    # goal distribution from t=0 instead of its one-time boot transient.
    # ------------------------------------------------------------------
    # PORT_GOAL_MODE: env override to run a task under kRandom re-goaling
    # (the reference's consecutive-goals protocol, goal_params goal_mode: 0)
    # without touching its canonical yaml. Planar tasks get their boot
    # quat synthesized from goal_yaw (flat rest: yaw quat ≡ full goal).
    _env_goal_mode = os.environ.get("PORT_GOAL_MODE")
    if _env_goal_mode:
        task_cfg["goal_mode"] = _env_goal_mode
    # The generator works in quaternions internally. Planar tasks do NOT
    # get task_cfg["goal_quat"] set — they stay on the target_yaw path so
    # the reference yaw lookahead (goal_params lookahead_angle 2 +
    # angle_hysteresis 0.4; cc:427 min(angle, lookahead)) chunks large
    # re-goal demands into sub-goals. Installing a goal quat bypasses that
    # machinery (main.py GOAL-QUAT note) — the first 5-goal attempt stalled
    # 390 s on a 2.66 rad static demand exactly this way.
    _gg_boot_quat = task_cfg.get("goal_quat")
    if _gg_boot_quat is None and "goal_yaw" in task_cfg:
        _hy = 0.5 * float(task_cfg["goal_yaw"])
        _gg_boot_quat = [float(np.cos(_hy)), 0.0, 0.0, float(np.sin(_hy))]
    _goal_gen = None
    _initial_goal_drawn = False
    if (str(task_cfg.get("goal_mode", "")) == "kRandom"
            and _gg_boot_quat is not None):
        # Planar (tshape) objects have ONE flat-resting nominal orientation
        # (reference GetNominalOrientations) — pass [identity] so every
        # re-draw after the first applies >=90 deg of yaw to the previous
        # goal quat (cc:330-336). The jack keeps its tripod nominals.
        _gg_kwargs = {}
        if task_cfg.get("object_type") == "tshape":
            _gg_kwargs["nominal_orientations"] = [
                np.array([1.0, 0.0, 0.0, 0.0])]
            _gg_kwargs["nominal_names"] = ["planar"]
            _gg_kwargs["planar_yaw_step_max"] = 2.0
        if task_cfg.get("random_goal_x_limits") is not None:
            _gg_kwargs["x_limits"] = tuple(
                float(v) for v in task_cfg["random_goal_x_limits"])
        if task_cfg.get("random_goal_y_limits") is not None:
            _gg_kwargs["y_limits"] = tuple(
                float(v) for v in task_cfg["random_goal_y_limits"])
        # goal_success_mode "flip" (USER-DIRECTED DEVIATION 2026-08-17,
        # jack-only): success = the jack's resting tripod matches the
        # goal's tripod (position and yaw ignored; replaces the reference
        # pos<0.02 AND rot<0.1 gate). Absent/"reference" keeps the
        # reference gate.
        _goal_success_mode = str(
            task_cfg.get("goal_success_mode", "reference"))
        _goal_gen = JackRandomGoalGenerator(
            rng=np.random.default_rng(
                None if args.seed is None else [args.seed, 0x60A1]),
            initial_xy=np.asarray(task_cfg["goal_xy"], dtype=float),
            initial_quat=np.asarray(_gg_boot_quat, dtype=float),
            success_mode=_goal_success_mode,
            **_gg_kwargs,
        )
        _gg_planar = (task_cfg.get("object_type") == "tshape")
        _draw_initial_goal = (
            bool(task_cfg.get("krandom_draw_initial_goal", False))
            or os.environ.get("PORT_GOAL_DRAW_INITIAL", "0") == "1"
        )
        if _draw_initial_goal:
            _avoid = (tripod_id(np.asarray(task_cfg.get(
                          "init_quat", [1.0, 0.0, 0.0, 0.0]), dtype=float))
                      if _goal_success_mode == "flip" else None)
            _goal_gen.draw_initial_goal(avoid_tripod=_avoid)
            task_cfg["goal_xy"] = [float(_goal_gen.goal_xy[0]),
                                   float(_goal_gen.goal_xy[1])]
            task_cfg["goal_quat"] = [float(v) for v in _goal_gen.goal_quat]
            if _gg_planar:
                # Planar tasks consume goal_yaw (target_yaw path — keeps
                # the reference yaw-lookahead machinery active). Mirror
                # the re-goal planar conversion for the drawn boot goal;
                # without this the drawn quat would be ignored and the
                # boot yaw would silently stay at the task literal.
                _q0 = _goal_gen.goal_quat
                task_cfg["goal_yaw"] = float(np.arctan2(
                    2.0 * (_q0[0] * _q0[3] + _q0[1] * _q0[2]),
                    1.0 - 2.0 * (_q0[2] ** 2 + _q0[3] ** 2)))
            _initial_goal_drawn = True
            _gq0 = _goal_gen.goal_quat
            print(f"[GOAL-GEN] initial goal DRAWN "
                  f"(krandom_draw_initial_goal, seed={args.seed}): "
                  f"xy=({task_cfg['goal_xy'][0]:+.3f},"
                  f"{task_cfg['goal_xy'][1]:+.3f}) "
                  f"tripod={KNOMINAL_NAMES_JACK[_goal_gen.orientation_index]} "
                  f"quat=[{_gq0[0]:+.4f} {_gq0[1]:+.4f} "
                  f"{_gq0[2]:+.4f} {_gq0[3]:+.4f}]")

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
    # Read PORT_RHO once for both the log header and the C3Solver ctor.
    _rho_init = float(os.environ.get("PORT_RHO", "100.0"))
    print(f"[ENV]  Mass: {task_cfg.get('mass', '?')} kg   "
          f"Friction mu: {task_cfg.get('friction', '?')}")
    print(f"[MPC]  ADMM max iters: {args.admm_iter}   rho_init: {_rho_init}")
    print(f"[MPC]  Force limit: 30.0 Nm")
    print(f"[COST] w_obj_xy:      {_cost.get('w_obj_xy', '?')}")
    print(f"[COST] w_obj_z:       {_cost.get('w_obj_z', '?')}")
    print(f"[COST] w_box_z:       {_cost.get('w_box_z', '?')}")
    print(f"[COST] w_box_rp:      {_cost.get('w_box_rp', '?')}")
    print(f"[COST] w_terminal:    {_cost.get('w_terminal', '?')}  (QN = w_terminal * Q)")
    print(f"[COST] w_ee_approach: {_cost.get('w_ee_approach', '?')}")
    print(f"[COST] w_torque:      {_cost.get('w_torque', '?')}")
    if task_cfg.get("q_init_franka") is not None:
        print(f"[ENV]  init arm q: reference q_init_franka (tasks.yaml)")
    else:
        print(f"[ENV]  init arm q: SAFE-OFFSET (IK-derived, opposite goal direction)")

    # ------------------------------------------------------------------
    # Build Drake environment
    # ------------------------------------------------------------------
    print("[C3] Building Drake environment ...")
    # In-sim frame capture pruned 2026-08-05 (scripts/make_run_video.sh is
    # the post-hoc render path); build without the render camera.
    (diagram, plant, panda_model, _, meshcat, plant_ad, context_ad,
     drake_video_writer) = build_environment(task_cfg, add_camera=False)

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
    # Stage object pose first so compute_safe_init_arm_q's IK can see it,
    # then resolve init_q (IK-derived safe-offset pose), then set arm.
    # ------------------------------------------------------------------
    # Initial object orientation. Flat-resting tasks (box, T, H) spawn upright
    # and omit `init_quat`. The jack has no flat face -- it must be spawned
    # already balanced on a tripod, so tasks.yaml gives the full quaternion
    # (reference jacktoy/parameters/sim_params.yaml q_init_object[0:4]).
    plant.SetFreeBodyPose(
        plant_ctx, obj_body,
        ad.RigidTransform(init_rotation(task_cfg), task_cfg["init_xyz"])
    )
    init_q = compute_safe_init_arm_q(
        plant, plant_ctx, panda_model,
        ee_frame=plant.GetFrameByName(EE_BODY_NAME),
        obj_body=obj_body,
        task_cfg=task_cfg,
    )
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
    # Reference-conformant LCS: Anitescu contact model + always-on
    # EE-manipuland pair + shape-appropriate manipuland-ground witnesses
    # (3 for tshape, 4 for box). All configured by LCSFormulator defaults.
    _obj_shape_for_defaults = str(task_cfg.get("object_type", "box"))
    # Reference sampling_c3plus_options.yaml:44 mu_per_pair_type[1]:
    # the EE-manipuland contact uses the Drake-combined friction
    # `2·μ_ee·μ_manip / (μ_ee + μ_manip)`, not the raw manipuland
    # material μ. Port previously passed `task_cfg["friction"]` (raw
    # manipuland μ) which under-estimates the friction cone at the
    # primary push contact — for push_t (μ_manip=0.3, μ_ee=1.0) the
    # combined is 0.4615 vs the passed 0.3 (35% under-estimation).
    # For box (μ_manip=1.0, μ_ee=1.0) combined equals 1.0 → byte-identical.
    _mu_manip  = float(task_cfg["friction"])
    _mu_pusher = float(task_cfg.get("pusher_friction", 1.0))
    _mu_lcs    = 2.0 * _mu_manip * _mu_pusher / (_mu_manip + _mu_pusher)
    _mu_per_pair = task_cfg.get("mu_per_pair_type", None)
    if _mu_per_pair is not None:
        print(f"[TASK] mu_per_pair_type override active: {_mu_per_pair} "
              f"(scalar fallback mu={_mu_lcs:.4f})", flush=True)
    formulator = LCSFormulator(
        plant, mu=_mu_lcs, obj_body=obj_body,
        plant_ad=plant_ad, context_ad=context_ad,
        object_shape=_obj_shape_for_defaults,
        mu_per_pair_type=_mu_per_pair,
        # Reference plans with a SEPARATE, heavier object model than it
        # simulates (jack_control.sdf 0.99 kg vs jack.sdf 0.156 kg;
        # H_shape_texture_controller.sdf 1.0 kg vs 0.05 kg). push_t is the
        # only task where the two coincide. See _controller_inertia_scope.
        controller_object_mass=task_cfg.get("controller_mass", None),
        # 2026-08-15 leg 2: reference plans with the controller SDF's
        # DECLARED inertia tensor (not a mass-scaled sim tensor). Emitted
        # into tasks.yaml by scripts/emit_controller_inertia.py from each
        # object's *_controller.sdf <inertial> block.
        controller_inertia=task_cfg.get("controller_inertia", None),
        # Fallback for object_sdf tasks WITHOUT a per-task witness key:
        # port-computed mesh-footprint table (see _tshape_vertex_set_body_frame).
        tshape_mesh_witnesses=bool(task_cfg.get("object_sdf", None)),
        # Per-task witness triangle from the object's reference
        # *_controller.sdf spheres (imported anything objects + push_t).
        mesh_ground_witnesses_body=task_cfg.get(
            "ground_witness_points_body", None),
    )

    # EE-space planner: solver/cost get the low-dim sizing (n_x=19, n_u=3).
    # DEFAULT since the 2026-07-28 divergence removal (reference plans all
    # tasks with the point-EE simple model); --r7 opts out to the legacy
    # port-only full-plant planner.
    # Supported on BOTH solvers — c3plus (componentwise projection) and c3
    # (LCP projection, Aydinoglu §V-B.3.b feasibility-guaranteed) — so the
    # projection variant can be flipped as a clean CLI test holding all
    # other dimensions (state, input, cost) constant.
    if args.ee_space:
        _solver_n_x, _solver_n_u = 19, 3
        # 2026-07-22 qvector migration attempts (p44-p46) all failed:
        # coordinated 4-tuple (m_ee=0.057, u=50, qvec=True, w_Q=50) at
        # admm_iter=25 caused dual residual to explode 1000×/solve.
        # Root cause: port uses scalar ρ ADMM augmentation; reference
        # uses per-slot G matrix. Without matching the G structure, the
        # amplified qvector cost cannot be balanced. Reverted to p38.
        _torque_limit = 30.0    # Newtons under EE-space (EE-force cap)
    else:
        _solver_n_x, _solver_n_u = n_x, n_u
        _torque_limit = 30.0    # Nm under R^7 (joint-torque cap)
    # Reference rho_scale=3 (adaptive per iter, applied in _solve_c3plus).
    # Initial rho matches port's tuned C3+ constant; adaptive schedule
    # multiplies over the 3 admm iterations.
    # penalize_input_change: reference push_t/sampling_c3plus_options.yaml
    # sets `false`; reference anything/*.yaml sets `true`. Task-gated.
    _penalize_input_change = (task_name != "push_t")
    # _rho_init is defined earlier (log-header block); PORT_RHO env override
    # for the item-#7 ρ-sweep investigation. See
    # docs/superpowers/investigations/2026-07-23-item7-deep-investigation.md
    # (arc 1). Default 100.0 preserves the port's tuned regime.
    solver     = C3Solver(n_x=_solver_n_x, n_u=_solver_n_u, rho=_rho_init,
                          math_diag=args.math_diag,
                          mode=args.solver,
                          penalize_input_change=_penalize_input_change)
    print(f"[C3] Solver mode: {args.solver}  "
          f"(planner: {'EE-space (R^3 force)' if args.ee_space else 'R^7 joint torque'}, "
          f"c3+ projection: {'C3+ (Bui 2026 eq 12 case-analysis)' if args.solver == 'c3plus' else 'n/a (mode=c3)'})")
    # Geometry hints for PORT_BOX_DPUSH_FIX (default-OFF, box-shape only).
    _obj_shape = str(task_cfg.get("object_type", ""))
    if _obj_shape == "box":
        _obj_half_extent = float(task_cfg["size"][0]) * 0.5
    else:
        _obj_half_extent = None
    from sim.env_builder import PUSHER_RADIUS as _EFFECTIVE_PUSHER_RADIUS
    quad_cost  = QuadraticManipulationCost(
        plant, EE_BODY_NAME, obj_body, task_cfg["cost"], n_x, n_u,
        math_diag=args.math_diag,
        object_shape=_obj_shape,
        object_half_extent=_obj_half_extent,
        pusher_radius=_EFFECTIVE_PUSHER_RADIUS,
    )
    _MPCClass = C3PlusMPC if args.solver == "c3plus" else C3MPC
    # Reference-conformant horizon+dt per solver:
    #   c3plus: anything/parameters/sampling_c3plus_options.yaml:20,62
    #           N=7, planning_dt_position=planning_dt_pose=0.075
    #   c3    : anything/parameters/sampling_c3_options.yaml (N=5, dt=0.1)
    if args.solver == "c3plus":
        # 2026-07-18 iter4: push_t OVERRIDES anything's defaults —
        # push_t/parameters/sampling_c3plus_options.yaml:20,53-54:
        #   N: 5 (anything: 7)
        #   planning_dt_position: 0.1 (anything: 0.075)
        #   planning_dt_pose: 0.05
        # Task-conditional so box path (uses anything defaults) stays put.
        if task_name == "push_t" or task_cfg.get("object_sdf"):
            # 2026-08-11 L2 (anything-N1 lineage): multiyaml_rewrite.py
            # PLANNING_HORIZON_CONFIGS {1: 10} + uniform planning_dt 0.075
            # (anything/sampling_c3plus_options.yaml post-rewrite). The old
            # 5 / 0.1 / 0.05 came from the bit-rotted push_t demo yaml —
            # see docs/anything-n1-config-delta-audit.md (L2).
            # Fig 8 campaign: every imported anything object (object_sdf
            # set) is anything-N1 lineage and takes this branch.
            _c3plus_N  = 10
            _c3plus_dt = 0.075
            _c3plus_dt_pose = 0.075
        elif task_name == "push_h":
            # The H is a LETTER-family object (reference anything/ loads the
            # *_shape_* meshes; letter_settings.yaml lists H_shape_texture).
            # anything/parameters/sampling_c3plus_options.yaml:20,62-63:
            #   N: 7, planning_dt_position: 0.075, planning_dt_pose: 0.075
            # -- NOT push_t's N=5 / 0.1 / 0.05.
            _c3plus_N  = 7
            _c3plus_dt = 0.075
            _c3plus_dt_pose = 0.075
        elif task_name == "push_jack":
            # jacktoy/parameters/sampling_c3plus_options.yaml:16,61-62:
            #   N: 5, planning_dt_position: 0.1, planning_dt_pose: 0.05
            # -- same cadence as push_t, not the letter family's 7/0.075.
            _c3plus_N  = 5
            _c3plus_dt = 0.1
            _c3plus_dt_pose = 0.05
        else:
            # 2026-08-11 box-lineage: multiyaml PLANNING_HORIZON_CONFIGS
            # {1: 10} + uniform 0.075 (the old 7 / 0.075 / 0.05 mixed the
            # shipped 4-object N with push_t's pose dt).
            _c3plus_N  = 10
            _c3plus_dt = 0.075
            _c3plus_dt_pose = 0.075
    else:
        _c3plus_N  = 5
        _c3plus_dt = 0.1
        _c3plus_dt_pose = 0.1  # C3 baseline path — no regime swap
    print(f"[MPC]  Horizon: {_c3plus_N}   dt: {_c3plus_dt} s   "
          f"dt_pose: {_c3plus_dt_pose} s")
    # The base MPC and optional Sampling-C3 wrapper share one planner cadence.
    # Define it unconditionally so plain `main.py push_t` runs are valid too.
    _dt_ctrl_pass = _c3plus_dt if args.solver == "c3plus" else 0.1
    _mpc_kwargs = dict(
        formulator=formulator,
        solver=solver,
        quadratic_cost=quad_cost,
        horizon=_c3plus_N,
        dt=_c3plus_dt,
        torque_limit=_torque_limit,
        admm_iter=args.admm_iter,
        math_diag=args.math_diag,
    )
    if args.solver == "c3plus":
        _mpc_kwargs["dt_pose"] = _c3plus_dt_pose
    if args.ee_space:
        _mpc_kwargs["use_ee_space"] = True
        # Reference ee_velocity_limits state constraint (added in commit
        # b877785) is available via ee_velocity_bounds kwarg. Disabled by
        # default 2026-07-18: analysis of results/push_t_evel_20260718_155846
        # showed the cap exposes the port's LCS-vs-Drake phantom-contact
        # divergence — arm settles hovering above T (Drake F_W=0) because
        # LCS predicts contact and the velocity cap prevents the QP from
        # commanding the arm downward. Re-enable by adding kwarg per-task.
    mpc = _MPCClass(**_mpc_kwargs)

    target_xy   = np.array(task_cfg["goal_xy"], dtype=float)
    target_yaw  = float(task_cfg.get("goal_yaw", 0.0))   # radians; 0 for legacy tasks
    # ------------------------------------------------------------------
    # Optional full goal quaternion. Flat-resting tasks (box, T, H) reach
    # every attainable orientation by yaw alone, so they specify goal_yaw and
    # goal_quat is None. The jack rests on a tripod of tip spheres and
    # reorients by ROLLING onto a different tripod -- its goal tilts out of
    # the plane and is not any yaw, so tasks.yaml gives the quaternion
    # (reference jacktoy/parameters/goal_params.yaml fixed_target_orientation).
    # ------------------------------------------------------------------
    _pending_goal_quat = None
    target_quat = task_cfg.get("goal_quat", None)
    if target_quat is not None:
        target_quat = np.asarray(target_quat, dtype=float)
        target_quat = target_quat / float(np.linalg.norm(target_quat))
        quad_cost.set_goal_quat(target_quat)
        _pending_goal_quat = target_quat
        # Quaternion goals get the per-tick SLERP lookahead in the sim loop
        # (orientation_lookahead, reference goal_generator.cc:408-437), so
        # demands beyond lookahead_angle run against a moving sub-goal.
        # The startup demand print below is informational.
        _q0 = np.asarray(task_cfg.get("init_quat", [1.0, 0.0, 0.0, 0.0]), float)
        _q0 = _q0 / float(np.linalg.norm(_q0))
        _R0 = ad.RotationMatrix(ad.Quaternion(_q0[0], _q0[1], _q0[2], _q0[3])).matrix()
        _Rg = ad.RotationMatrix(ad.Quaternion(
            target_quat[0], target_quat[1], target_quat[2], target_quat[3])).matrix()
        _demand = float(np.arccos(np.clip(
            (np.trace(_R0.T @ _Rg) - 1.0) * 0.5, -1.0, 1.0)))
        print(f"[GOAL-QUAT] goal_quat=[{target_quat[0]:+.4f} {target_quat[1]:+.4f} "
              f"{target_quat[2]:+.4f} {target_quat[3]:+.4f}]  "
              f"reorientation demand = {_demand:.4f} rad ({np.degrees(_demand):.1f} deg)")
        if _demand > 2.0:
            print(f"[GOAL-QUAT] demand {_demand:.3f} rad > 2.0 rad "
                  f"lookahead_angle — per-tick SLERP sub-goal active "
                  f"(reference goal_generator.cc:408-437)")
    # _goal_gen was constructed just after load_task (initial-draw path
    # mutates task_cfg before build_environment). Banner here, where the
    # normalized target_quat exists.
    if _goal_gen is not None:
        _gg_q_str = ("target_yaw path (planar; reference yaw lookahead active)"
                     if target_quat is None else
                     f"quat=[{target_quat[0]:+.4f} {target_quat[1]:+.4f} "
                     f"{target_quat[2]:+.4f} {target_quat[3]:+.4f}]")
        print(f"[GOAL-GEN] kRandom re-goaling ACTIVE (reference goal_mode 0): "
              f"goal #1 {'DRAWN' if _initial_goal_drawn else 'fixed (reference boot value)'} "
              f"xy=({target_xy[0]:+.3f},{target_xy[1]:+.3f}) {_gg_q_str}")
        if _goal_gen.success_mode == "flip":
            print(f"[GOAL-GEN] success mode = FLIP (user-directed deviation "
                  f"2026-08-17): goal reached when resting tripod matches "
                  f"goal tripod {TRIPOD_NAMES[tripod_id(target_quat)]} "
                  f"(pos/yaw ignored; reference gate pos<0.02 AND rot<0.1 "
                  f"replaced)")
            if (bool(task_cfg.get("flip_primitive", False))
                    and os.environ.get("PORT_FLIP_PRIMITIVE", "1") != "0"):
                mpc._flip_primitive_enabled = True
                print(f"[FLIP-PRIM] ENABLED (task flip_primitive): the "
                      f"controller stages the EE at the toggling capsule's "
                      f"up-tip (97mm > 55.3mm tip-before-slide) and pushes "
                      f"through it — controller-induced flips, no external "
                      f"forces", flush=True)
    # Diagnostic (default-inert): force one re-goal at planner step N to
    # exercise the live re-goal path without a real goal achievement.
    _goal_gen_force_step = int(
        os.environ.get("DIAG_GOALGEN_FORCE_REGOAL_AT_STEP", "-1") or -1)
    ee_frame    = plant.GetFrameByName(EE_BODY_NAME)
    world_frame = plant.world_frame()

    # ------------------------------------------------------------------
    # Optional sampling-C3 outer controller (Venkatesh 2025 §IV-D port)
    # ------------------------------------------------------------------
    if args.sampling_c3 is not None:
        _yaml_path = args.sampling_c3
        sc3_params = SamplingC3Params.from_yaml(_yaml_path)
        # object_shape is a property of the TASK, not of the controller config,
        # but the sampler reads it from the sampling-c3 yaml while the LCS and
        # the cost read task_cfg["object_type"]. Two sources of truth for the
        # same fact: a mismatch silently hands the sampler the WRONG face table
        # (e.g. running push_h against sampling_c3_kik_t.yaml would project H
        # samples off the T's outline) with no error anywhere. tasks.yaml wins,
        # same precedent as the sampling_height override below.
        # lcs_explicit_manipuland_ground_contacts is parsed into
        # SamplingC3Params, but the LCSFormulator is constructed EARLIER (before
        # the sampling yaml is read) and hardcodes `3 if tshape else 4`. The
        # yaml key was therefore INERT -- unnoticed because push_t's configured
        # 3 coincides with the hardcoded value, so the T looked correct while
        # any other value was silently discarded. Push the configured count
        # through here.
        # Guarded on > 0: the dataclass default is 0, and blindly assigning that
        # would disable ground contacts entirely for configs omitting the key.
        _n_gnd = int(getattr(sc3_params,
                             "lcs_explicit_manipuland_ground_contacts", 0) or 0)
        if (_n_gnd > 0
                and _n_gnd != formulator.lcs_explicit_manipuland_ground_contacts):
            print(f"[OVERRIDE] lcs_explicit_manipuland_ground_contacts={_n_gnd} "
                  f"(was {formulator.lcs_explicit_manipuland_ground_contacts}, "
                  f"from {_yaml_path})")
            formulator.lcs_explicit_manipuland_ground_contacts = _n_gnd

        _task_shape = str(task_cfg.get("object_type", "") or "")
        if _task_shape:
            _was_shape = str(sc3_params.sampling_params.object_shape)
            if _was_shape != _task_shape:
                sc3_params.sampling_params.object_shape = _task_shape
                print(f"[OVERRIDE] object_shape={_task_shape} "
                      f"(was '{_was_shape}', per-task '{task_name}')")
        # D6: per-task sampling_height override. pushing/hard_pushing set 0.03
        # (sub-CoM, restoring tip moment); cube_turning/shepherding read the
        # sampler default. tasks.yaml is the source of truth; absent → no override.
        _task_sample_h = task_cfg.get("sampling_height")
        if _task_sample_h is not None:
            _was_sh = sc3_params.sampling_params.sampling_height
            sc3_params.sampling_params.sampling_height = float(_task_sample_h)
            print(f"[OVERRIDE] sampling_height={float(_task_sample_h):.3f} "
                  f"(was {_was_sh:.3f}, per-task '{task_name}')")
        # Per-task grid_x/y_limits override. The perimeter-sampler
        # draw grid is a PER-DEMO reference literal (push_t [-0.12,0.08] x
        # [-0.08,0.08]; anything [-0.11,0.11]^2) and each object belongs to
        # lineage is push_t for the reference Block-T and anything for the
        # imported objects. Absent -> yaml value.
        for _gk in ("grid_x_limits", "grid_y_limits"):
            _task_g = task_cfg.get(_gk)
            if _task_g is not None:
                _was_g = getattr(sc3_params.sampling_params, _gk, None)
                setattr(sc3_params.sampling_params, _gk,
                        [float(_task_g[0]), float(_task_g[1])])
                print(f"[OVERRIDE] {_gk}={_task_g} "
                      f"(was {_was_g}, per-task '{task_name}')")
        # 2026-08-17: per-task sampling strategy + kMeshNormal literals.
        # The strategy is a per-demo reference literal: push_t runs
        # kRandomOnPerimeter (kik_t default), the ANYTHING lineage runs
        # kMeshNormal(MultiObject) — anything sampling_params.yaml:14
        # sampling_strategy: 7, N=1 ≡ port kMeshNormal.
        _task_strat = task_cfg.get("sampling_strategy")
        if _task_strat is not None:
            from control.sampling_c3.params import SamplingStrategy as _SS
            _was_s = sc3_params.sampling_params.sampling_strategy
            sc3_params.sampling_params.sampling_strategy = _SS[_task_strat]
            print(f"[OVERRIDE] sampling_strategy={_task_strat} "
                  f"(was {_was_s.name}, per-task '{task_name}')")
        for _sk, _cast in (("sample_projection_clearance", float),
                           ("buffer_distance", float),
                           ("max_attempts", int),
                           ("barycentric_bias", float)):
            _task_v = task_cfg.get(_sk)
            if _task_v is not None:
                _was_v = getattr(sc3_params.sampling_params, _sk, None)
                setattr(sc3_params.sampling_params, _sk, _cast(_task_v))
                print(f"[OVERRIDE] {_sk}={_task_v} "
                      f"(was {_was_v}, per-task '{task_name}')")
        # kMeshNormal face preprocessing — AFTER the overrides so the
        # reference z-filter uses the final buffer/clearance literals.
        _mesh_faces_for_task = None
        from control.sampling_c3.params import SamplingStrategy as _SSm
        if (sc3_params.sampling_params.sampling_strategy
                == _SSm.kMeshNormal):
            from control.sampling_c3.sampling import load_mesh_faces
            from pathlib import Path as _PathM
            _obj_p = (_PathM(task_cfg["object_sdf"]).parent
                      / f"{task_cfg['link_name']}.obj")
            _spm = sc3_params.sampling_params
            _mesh_faces_for_task = load_mesh_faces(
                str(_obj_p), _spm.buffer_distance,
                _spm.sample_projection_clearance)
            print(f"[MESH-NORMAL] {_obj_p.name}: "
                  f"{len(_mesh_faces_for_task['areas'])} side-wall faces, "
                  f"total_area={_mesh_faces_for_task['total_area']:.4f} m^2")
        # 2026-07-19: per-task pwl_waypoint_height override. Object top varies
        # by task (T top 0.04, box top 0.10), so the PWL traverse height must
        # be per-object to keep sphere-bottom above object top. tasks.yaml
        # holds the geometric truth; sampling-c3 yaml value is inert when
        # the per-task override is present.
        _task_wp_h = task_cfg.get("pwl_waypoint_height")
        if _task_wp_h is not None:
            _was_wh = sc3_params.reposition_params.pwl_waypoint_height
            sc3_params.reposition_params.pwl_waypoint_height = float(_task_wp_h)
            print(f"[OVERRIDE] pwl_waypoint_height={float(_task_wp_h):.3f} "
                  f"(was {_was_wh:.3f}, per-task '{task_name}')")
        # With safe-offset init (IK-derived), EE starts opposite goal at
        # safe altitude above object top — free-mode PWL descent has
        # clear space, arm doesn't clip through object on the way down.
        # Reposition PWL trajectory is the reference path (LcmTrajectoryReceiver
        # + FirstOrderHold PP → OSC position tracking with velocity feedforward).
        # Default True in SamplingC3Params.
        _rng = np.random.default_rng(args.seed) if args.seed is not None else None
        if args.seed is not None:
            print(f"[OVERRIDE] seed={args.seed} (rng=np.random.default_rng)")
        # Planner cadence: 1/planning_dt Hz (matches reference — dairlib's C3
        # planner is LCM-driven at 1 kHz but effectively bounded by ADMM solve
        # time; here we tick the port planner once per planning_dt of sim time
        # so each planner tick corresponds to one LCS horizon step forward).
        # OSC runs at 1 kHz between planner ticks (compute_control_osc_only).
        # W_force reference value (LambdaEndEffectorW = I_3, scalar 1.0).
        sc3_params.W_force = 1.0
        # Planner workspace state constraints (reference cc:995-1025): hard
        # per-knot bounds on the EE position and object position slots of the
        # EE-space state, widened by the margin — applied to every solve on
        # this solver instance (full + surrogate), exactly like the
        # reference's per-sample C3 objects. EE-space layout only; the --r7
        # falsification planner has a different state layout.
        if args.ee_space:
            _pw = build_planner_workspace_bounds(sc3_params)
            if _pw:
                solver.state_position_bounds = _pw
                print(f"[OVERRIDE] planner workspace state bounds ON "
                      f"({len(_pw)} rows, "
                      f"margin={sc3_params.planner_workspace_margin}m, "
                      f"ref cc:995-1025)")
        mpc = SamplingC3Controller(
            base_mpc=mpc,
            plant=plant,
            ee_frame=ee_frame,
            obj_body=obj_body,
            params=sc3_params,
            dt_ctrl=_dt_ctrl_pass,
            log_diag=True,
            # User directive 2026-07-17: revert the start_in_c3_mode=True
            # bootstrap (commit 3bf3452) — test whether dispatcher enters c3
            # naturally now that today's yaml/script fixes (progress-drop,
            # cost-switching threshold, u-limits, task-sampling-height) are
            # in place. Reverts to reference `sampling_based_c3_controller.h`
            # NON-bootstrap behavior for the port (reference starts c3 but
            # port default was False since inception; this line restores that).
            start_in_c3_mode=False,
            rng=_rng,
            diagram=diagram,
            # Fig 8 sampler fix (2026-08-15): imported mesh
            # tasks use the reference geometry-generic perimeter sampler
            # (interior draw + signed-distance projection off the real
            # collision geometry). Analytic tasks retain their shape tables.
            # Key this from the selected strategy instead of object_sdf:
            # canonical push_t now loads the reference SDF but remains an
            # analytic kRandomOnPerimeter task with no OBJ dependency.
            use_geometry_perimeter_sampling=(_mesh_faces_for_task is not None),
            # kMeshNormal (anything lineage): preprocess the object's
            # full-resolution OBJ into the reference face set. Path
            # convention mirrors the reference's
            # "urdf/<base_name>/<base_name>.obj"
            # (sampling_based_c3_controller.cc:435-438).
            mesh_faces=_mesh_faces_for_task,
        )
        # SE(3) tasks: hand the controller the full goal orientation so its
        # rotation error is the geodesic angle rather than a yaw difference.
        if _pending_goal_quat is not None:
            mpc.set_goal_quat(_pending_goal_quat)
        # Flip primitive: re-apply on the WRAPPER — `mpc` was rebound above,
        # so the earlier set on the base controller object is inert.
        # PORT_FLIP_PRIMITIVE=0 disables it for emergent-toppling
        # experiments (paper-era MPC alone, no scripted pushes).
        if (_goal_gen is not None and _goal_gen.success_mode == "flip"
                and bool(task_cfg.get("flip_primitive", False))
                and os.environ.get("PORT_FLIP_PRIMITIVE", "1") != "0"):
            mpc._flip_primitive_enabled = True
        print(f"[GS] SamplingC3Controller enabled (config: {_yaml_path})")
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

    # ------------------------------------------------------------------
    # Joint limit constants for arm safety check
    # ------------------------------------------------------------------
    _Q_LO = np.array([-2.897, -1.763, -2.897, -3.072, -2.897, -0.0175, -2.897])
    _Q_HI = np.array([ 2.897,  1.763,  2.897, -0.0698, 2.897,  3.752,   2.897])

    # ------------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------------
    sim_time      = 0.0
    # Planner cadence set from _dt_ctrl_pass above so a single source-of-truth
    # (_c3plus_dt) drives both the LCS discretization step and the outer loop.
    # For c3plus: 0.075 s → planner ticks ~13.3 Hz (matches reference
    # sampling_c3plus_options.yaml planning_dt_position/pose = 0.075).
    dt_ctrl       = float(_dt_ctrl_pass)
    # Yaw sub-goal clip hysteresis state (reference goal_params.yaml:24
    # angle_hysteresis: 0.4). Once we've entered the clip regime, require
    # |Δyaw| to fall below (lookahead_angle - hysteresis) before un-clipping.
    # Prevents sub-goal orientation flip near the 180° error singularity.
    _yaw_clip_active = False
    # Quaternion-goal SLERP lookahead state (reference goal_generator.h:180
    # last_rotation_axis_, zero-initialized) + clamp-transition log latch.
    _last_rot_axis = np.zeros(3)
    _lookahead_was_clamped = False
    # Tripod-change ([FLIP]) log high-water mark.
    _flip_events_seen = 0
    # ------------------------------------------------------------------
    # DIAG_JACK_TOPPLE_DRIVER=<n> (diagnostic, default OFF): induce up to
    # n goal-directed tripod flips by applying an external horizontal
    # force at the UP end of the toggling capsule (97 mm — above the
    # 55.3 mm tip-before-slide critical height, so ~1 N tips instead of
    # sliding; see jack-topple-mechanics analysis 2026-08-17). Exercises
    # the flip-goal loop the CONTROLLER cannot yet drive: settle -> push
    # over the toggling support edge -> [FLIP] -> goal reached -> redraw.
    # Never active on the canonical path.
    # ------------------------------------------------------------------
    _topple_n = int(os.environ.get("DIAG_JACK_TOPPLE_DRIVER", "0") or 0)
    _topple = {"phase": "idle", "cooldown_until": 0.0, "push_until": 0.0,
               "fmag": 1.5, "tripod_at_push": None}
    _topple_body = None
    if _topple_n > 0 and _goal_gen is not None:
        _topple_body = plant.GetBodyByName(task_cfg["link_name"])
        print(f"[TOPPLE-DRIVER] ACTIVE (diagnostic): target flips="
              f"{_topple_n} f0=1.5N push_point=up-tip (97mm)", flush=True)

    def _topple_set_force(fvec_W, p_B):
        f = ad.ExternallyAppliedSpatialForce()
        f.body_index = _topple_body.index()
        f.p_BoBq_B = np.asarray(p_B, dtype=float)
        f.F_Bq_W = ad.SpatialForce(np.zeros(3), np.asarray(fvec_W, float))
        plant.get_applied_spatial_force_input_port().FixValue(plant_ctx, [f])
    # 1 kHz OSC decoupling — mirror dairlib's LcmDrivenLoop where the OSC
    # subscribes to the last-published planner trajectory and ticks at
    # osc_params.yaml:2 `controller_frequency: 1000`. Every outer iteration
    # runs planner + OSC once, then the OSC-only inner loop advances the sim
    # in 1 ms sub-steps using the cached planner output.
    _DT_OSC         = 0.001                              # 1 kHz OSC
    _N_OSC_PER_OUTER = int(round(dt_ctrl / _DT_OSC))     # 75 for c3plus
    max_time      = (args.max_time if args.max_time is not None
                     else float(task_cfg.get("max_time", 8.0)))
    step          = 0
    if args.max_time is not None:
        print(f"[ENV]  Sim duration overridden: max_time={max_time}s")
    elif "max_time" in task_cfg:
        print(f"[ENV]  Task sim duration: max_time={max_time}s")

    # PORT_EXIT_ON_TIGHT=1 (2026-08-20 Fig-8 block-T campaign): end the run
    # after a one-second simulated-time settling hold following the first
    # joint-tight achievement (_tight_ever_latched, the sticky record of
    # 16f03ac), instead of running to max_time. Measurement-harness gate for
    # time-to-goal campaigns — default OFF, byte-identical unset.
    _exit_on_tight = os.environ.get("PORT_EXIT_ON_TIGHT", "0") == "1"
    _tight_hold_s = 1.0
    _tight_exit_at = None

    while True:
        if _exit_on_tight and getattr(mpc, "_tight_ever_latched", False):
            if _tight_exit_at is None:
                _latch_t = getattr(mpc, "_tight_first_latch_sim_t", None)
                if _latch_t is None:
                    _latch_t = sim_time
                _tight_exit_at = float(_latch_t) + _tight_hold_s
                print(f"[TIGHT-GOAL-HOLD] step={step} t={sim_time:.3f}s — "
                      f"holding simulation through t={_tight_exit_at:.3f}s "
                      f"(1.000s after first latch)", flush=True)
            if sim_time + 1e-12 >= _tight_exit_at:
                print(f"[EXIT-ON-TIGHT] step={step} t={sim_time:.3f}s — "
                      f"1.000s post-goal hold complete, ending run "
                      f"(PORT_EXIT_ON_TIGHT=1)", flush=True)
                break
        # A tight latch observed at the configured time limit is allowed to
        # finish its settling hold. Without an active hold, max_time retains
        # its existing meaning.
        if sim_time >= max_time and _tight_exit_at is None:
            break
        current_q = plant.GetPositions(plant_ctx)
        current_v = plant.GetVelocities(plant_ctx)

        # ---- NaN / joint-limit safety check ------------------------------
        if not (np.all(np.isfinite(current_q)) and np.all(np.isfinite(current_v))):
            print(f"[WARN] NaN in state at t={sim_time:.3f}s — stopping.")
            break
        # ---- Object-position blowup abort (Bug 4 safety net) --------------
        # 2026-07-22: p41 diverged silently over 145s because obj_z hit
        # -107km without triggering NaN. Any LCS-parameter experiment can
        # produce sim divergence; without this abort the run wastes 10+ min
        # of compute and produces misleading logs. Threshold TIGHTENED
        # 1.0 → 0.5m after p44 attempts got WSL-killed at drift ~0.9m before
        # Bug 4 could fire (Drake contact solver bogs down as T sinks,
        # allowing WSL OOM/CPU kill before the sim reaches the abort).
        # Table-top push should never displace object > 0.5m; anything past
        # that is divergent. TEMPORARY: remove once qvector migration is
        # implemented properly (per user directive 2026-07-22).
        _obj_now = np.array([current_q[obj_x_idx],
                             current_q[obj_y_idx],
                             current_q[obj_z_idx]])
        _obj_init = np.asarray(task_cfg["init_xyz"], dtype=float)
        _drift = float(np.linalg.norm(_obj_now - _obj_init))
        if _drift > 0.5:
            raise RuntimeError(
                f"[ABORT] Object position blowup at t={sim_time:.3f}s: "
                f"obj={_obj_now.tolist()} vs init={_obj_init.tolist()} "
                f"drift={_drift:.3f}m > 1.0m — sim numerically diverged. "
                "See main.py Bug 4 safety net."
            )
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
        # Lookahead sub-goal (reference anything/goal_params.yaml:19
        # lookahead_step_size: 0.15). Plans toward a REACHABLE 15 cm
        # sub-goal each tick rather than the distant final goal.
        _lookahead = 0.15
        _obj_xy_now = np.array([current_q[obj_x_idx], current_q[obj_y_idx]])
        # kRandom re-goaling (reference goal_generator.cc:135-154 success
        # gate + :378-389 OnGoalReached). On success: new goal to the cost
        # + dispatcher; ghost moves; the achieved latch is reset because the
        # reference latch is kFixedGoal-only (controller cc:914-916).
        # target_yaw stays stale — inert for quaternion tasks (goal_quat
        # takes precedence in the cost and the dispatcher rot error).
        if _goal_gen is not None:
            _obj_quat_now = np.array(
                [current_q[pos_start + _i] for _i in range(4)])
            _regoaled = _goal_gen.check_and_regoal(_obj_xy_now, _obj_quat_now)
            if _goal_gen.flip_events > _flip_events_seen:
                _flip_events_seen = _goal_gen.flip_events
                print(f"[FLIP] #{_goal_gen.flip_events} at t={sim_time:.3f}s "
                      f"{_goal_gen.last_flip[0]} -> {_goal_gen.last_flip[1]}",
                      flush=True)
            if not _regoaled and step == _goal_gen_force_step:
                _goal_gen.force_regoal()
                _regoaled = True
                print(f"[GOAL-GEN] DIAG forced re-goal at step={step}",
                      flush=True)
            if _regoaled:
                target_xy = _goal_gen.goal_xy.copy()
                _gg_new_quat = _goal_gen.goal_quat.copy()
                if _gg_planar:
                    # Planar task: stay on the target_yaw path — the
                    # per-tick yaw lookahead below (reference
                    # goal_params lookahead_angle 2 / angle_hysteresis
                    # 0.4, cc:427 min(angle, lookahead)) then chunks the
                    # new demand into <=2 rad sub-goals. Installing the
                    # quat would bypass it (static-goal deviation — the
                    # first 5-goal attempt stalled 390 s on 2.66 rad).
                    target_yaw = float(np.arctan2(
                        2.0 * (_gg_new_quat[0] * _gg_new_quat[3]),
                        1.0 - 2.0 * (_gg_new_quat[3] ** 2)))
                else:
                    target_quat = _gg_new_quat
                    quad_cost.set_goal_quat(target_quat)
                    if hasattr(mpc, "set_goal_quat"):
                        mpc.set_goal_quat(target_quat)
                # Reference goal-change reset (cc:827-845): position regime,
                # achieved latch, progress, and BOTH sample buffers.
                if hasattr(mpc, "reset_for_new_goal"):
                    mpc.reset_for_new_goal()
                elif hasattr(mpc, "_achieved_fixed_goal"):
                    mpc._achieved_fixed_goal = False
                    mpc._off_target_streak = 0
                _update_jack_goal_marker(meshcat, target_xy, _gg_new_quat,
                                         task_cfg["init_xyz"][2])
                _gg_demand = geodesic_angle(_gg_new_quat, _obj_quat_now)
                print(f"[GOAL-GEN] goal #{_goal_gen.goals_reached} REACHED "
                      f"at t={sim_time:.3f}s -> new goal "
                      f"xy=({target_xy[0]:+.3f},{target_xy[1]:+.3f}) "
                      f"tripod={_goal_gen.nominal_names[_goal_gen.orientation_index]} "
                      f"quat=[{_gg_new_quat[0]:+.4f} {_gg_new_quat[1]:+.4f} "
                      f"{_gg_new_quat[2]:+.4f} {_gg_new_quat[3]:+.4f}] "
                      f"demand={_gg_demand:.3f} rad"
                      + ("  [> 2.0 rad lookahead_angle -- SLERP sub-goal "
                         "will clamp]"
                         if (_gg_demand > 2.0 and not _gg_planar) else ""),
                      flush=True)
                # PORT_GOALGEN_N: stop the run once N goals have been
                # achieved (achievements counted by the generator, boot
                # goal included). Diagnostic-class env gate for the
                # consecutive-goals protocol; unset = run to max-time.
                _gg_n = int(os.environ.get("PORT_GOALGEN_N", "0") or 0)
                if _gg_n > 0 and _goal_gen.goals_reached >= _gg_n:
                    print(f"[GOAL-GEN] COMPLETE: {_goal_gen.goals_reached} "
                          f"goals achieved (PORT_GOALGEN_N={_gg_n}) at "
                          f"t={sim_time:.3f}s — ending run", flush=True)
                    break
            # DIAG topple driver (see init above): settle -> push the
            # toggling capsule's up end over the support edge -> release
            # on raw tripod change -> cooldown; escalate force on a
            # failed window; inert once the target flip count is reached.
            if _topple_body is not None and _goal_gen.flip_events < _topple_n:
                _raw_trip = tripod_id(_obj_quat_now)
                _obj_w = float(np.linalg.norm(current_v[n_u:n_u + 3]))
                _obj_vl = float(np.linalg.norm(current_v[n_u + 3:n_u + 6]))
                if _topple["phase"] == "push":
                    _flipped = _raw_trip != _topple["tripod_at_push"]
                    if _flipped or sim_time >= _topple["push_until"]:
                        _topple_set_force(np.zeros(3), np.zeros(3))
                        _topple["phase"] = "idle"
                        _topple["cooldown_until"] = sim_time + 1.5
                        _topple["fmag"] = (1.5 if _flipped else
                                           min(_topple["fmag"] * 1.5, 8.0))
                        print(f"[TOPPLE-DRIVER] release t={sim_time:.3f}s "
                              f"{'FLIPPED' if _flipped else 'no flip'} "
                              f"next_f={_topple['fmag']:.2f}N", flush=True)
                elif (sim_time >= _topple["cooldown_until"]
                      and _obj_w < 0.5 and _obj_vl < 0.05
                      and _goal_gen.current_tripod == _raw_trip):
                    _plan = topple_roll_plan(
                        _obj_quat_now, _raw_trip,
                        tripod_id(_goal_gen.goal_quat))
                    if _plan is not None:
                        _tk, _tp_B, _td_W = _plan
                        _topple_set_force(_topple["fmag"] * _td_W, _tp_B)
                        _topple["phase"] = "push"
                        _topple["push_until"] = sim_time + 0.8
                        _topple["tripod_at_push"] = _raw_trip
                        print(f"[TOPPLE-DRIVER] push t={sim_time:.3f}s "
                              f"capsule={_tk} f={_topple['fmag']:.2f}N "
                              f"dir=({_td_W[0]:+.2f},{_td_W[1]:+.2f})",
                              flush=True)
            elif _topple_body is not None and _topple["phase"] == "push":
                _topple_set_force(np.zeros(3), np.zeros(3))
                _topple["phase"] = "done"
                print(f"[TOPPLE-DRIVER] target reached; driver inert",
                      flush=True)
        _delta_vec  = target_xy - _obj_xy_now
        _dist       = float(np.linalg.norm(_delta_vec))
        if _dist > 1e-9:
            _step   = min(_lookahead, _dist)
            _effective_target_xy = _obj_xy_now + (_delta_vec / _dist) * _step
        else:
            _effective_target_xy = target_xy
        # Yaw sub-goal — reference anything/goal_params.yaml:20 and
        # push_t/goal_params.yaml:18 `lookahead_angle: 2 rad`. Clip the
        # planner's yaw target to at most `lookahead_angle` from current
        # object yaw so a distant orientation goal doesn't force the
        # planner to reason over a large rotation in one solve. Yaw
        # extraction from box quaternion (qw, qx, qy, qz) at pos_start.
        _lookahead_angle = 2.0  # rad, reference goal_params.yaml:18/20
        _yaw_hysteresis  = 0.4  # rad, reference goal_params.yaml:24
        _qw = float(current_q[pos_start + 0])
        _qx = float(current_q[pos_start + 1])
        _qy = float(current_q[pos_start + 2])
        _qz = float(current_q[pos_start + 3])
        _yaw_now = float(np.arctan2(
            2.0 * (_qw * _qz + _qx * _qy),
            1.0 - 2.0 * (_qy * _qy + _qz * _qz),
        ))
        _dyaw = float(np.arctan2(np.sin(target_yaw - _yaw_now),
                                 np.cos(target_yaw - _yaw_now)))
        # Hysteresis: enter clip at |Δyaw| > lookahead_angle; only exit clip
        # when |Δyaw| falls below (lookahead_angle - hysteresis).
        if _yaw_clip_active:
            if abs(_dyaw) < (_lookahead_angle - _yaw_hysteresis):
                _yaw_clip_active = False
        else:
            if abs(_dyaw) > _lookahead_angle:
                _yaw_clip_active = True
        if _yaw_clip_active:
            _effective_target_yaw = float(_yaw_now
                                          + np.sign(_dyaw) * _lookahead_angle)
        else:
            _effective_target_yaw = float(target_yaw)
        # Quaternion-goal SLERP lookahead (reference goal_generator.cc:
        # 408-437): the COST chases a sub-goal at most lookahead_angle
        # (2 rad) along the geodesic from the CURRENT orientation,
        # recomputed every tick, with the near-180° axis hysteresis.
        # Progress metrics, the achievement latch and RESULT stay on the
        # final goal (reference current_*_error_ vs x_lcs_final_des,
        # controller cc:780-805; only x_desired uses the lookahead target,
        # cc:1009). mpc.set_goal_quat therefore keeps the FINAL quat.
        if target_quat is not None:
            _q_obj_now = np.array([_qw, _qx, _qy, _qz])
            _sub_quat, _last_rot_axis = orientation_lookahead(
                _q_obj_now, target_quat, _last_rot_axis)
            quad_cost.set_goal_quat(_sub_quat)
            _full_err = geodesic_angle(target_quat, _q_obj_now)
            _clamped = _full_err > 2.0
            _clamp_transition = _clamped != _lookahead_was_clamped
            _lookahead_was_clamped = _clamped
            if _clamp_transition or (_clamped and step % 100 == 0):
                print(f"[LOOKAHEAD] step={step} full_err={_full_err:.4f}rad "
                      f"clamped={'Y' if _clamped else 'N'} "
                      f"sub_err={geodesic_angle(_sub_quat, _q_obj_now):.4f}rad "
                      f"axis=({_last_rot_axis[0]:+.3f},"
                      f"{_last_rot_axis[1]:+.3f},"
                      f"{_last_rot_axis[2]:+.3f})", flush=True)
        _control_kwargs = {"target_yaw": _effective_target_yaw}
        # The Sampling-C3 wrapper needs the unclipped terminal target for
        # goal-transition logging; the base MPC API does not accept it.
        if args.sampling_c3 is not None:
            _control_kwargs["final_target_xy"] = target_xy
        u_opt = mpc.compute_control(current_q, current_v, plant_ctx,
                                    _effective_target_xy, **_control_kwargs)
        # === end Stage 2 ===

        # Update predicted-trajectory markers in Meshcat
        if mpc.last_x_seq is not None:
            _update_predicted_trajectory(
                meshcat, mpc.last_x_seq, obj_x_idx, obj_y_idx, obj_z_idx
            )

        # Tune-3: back to universal-add. OSC bias=Cv only, so gravity comp
        # comes from main.py's tau_g addition. Matches dd2294d proven
        # closure.
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

        # 1 kHz OSC inner loop. Sub-step 0 uses the u_opt just computed by
        # the full planner+OSC path (still applied via FixValue above); each
        # subsequent sub-step re-reads the plant state, calls
        # mpc.compute_control_osc_only using the cached planner output, and
        # re-applies torque. This mirrors dairlib's decoupled OSC ticking at
        # osc_params.yaml `controller_frequency: 1000`.
        # [F1K] 1 kHz contact-force aggregation. The physics and OSC run at
        # 1 kHz but [GATE-CONTACT] samples force once per planner tick
        # (13 Hz) — a 75x decimation that hides the stick-slip transients
        # which are the only thing that moves the T (delivered mean 2.5 N <
        # stiction 4.53 N; motion happens in unseen spikes). Aggregate the
        # EE-object |F| across every sub-step and report min/mean/max plus
        # the fraction of sub-steps above the stiction threshold.
        _f1k_vals = []
        for _osc_i in range(_N_OSC_PER_OUTER):
            sim_time += _DT_OSC
            simulator.AdvanceTo(sim_time)
            try:
                _cr1k = plant.get_contact_results_output_port().Eval(plant_ctx)
                _fm = 0.0
                for _ci in range(_cr1k.num_point_pair_contacts()):
                    _in1k = _cr1k.point_pair_contact_info(_ci)
                    _pa, _pb = _in1k.point_pair().id_A, _in1k.point_pair().id_B
                    if ((_pa in formulator._ee_geom_ids
                         and _pb in formulator._manipuland_geom_ids)
                            or (_pb in formulator._ee_geom_ids
                                and _pa in formulator._manipuland_geom_ids)):
                        _fm = float(np.linalg.norm(_in1k.contact_force()))
                        break
                _f1k_vals.append(_fm)
            except Exception:
                pass
            if _osc_i == _N_OSC_PER_OUTER - 1:
                break
            _cur_q = plant.GetPositions(plant_ctx)
            _cur_v = plant.GetVelocities(plant_ctx)
            _tau_g_i = -plant.CalcGravityGeneralizedForces(plant_ctx)
            _u_osc = mpc.compute_control_osc_only(
                _cur_q, _cur_v, plant_ctx, sim_time)
            _total_i = _tau_g_i[:n_u] + _u_osc
            plant.get_actuation_input_port().FixValue(plant_ctx, _total_i)
        if _f1k_vals:
            _fa = np.asarray(_f1k_vals)
            _n_contact_1k = int((_fa > 1e-6).sum())
            if _n_contact_1k > 0:
                _stick_thr = 4.53   # mu_comb(0.3,1.0)*m*g, report-only
                print(f"[F1K] step={step} sub={len(_fa)} "
                      f"contact_sub={_n_contact_1k} "
                      f"F(min/mean/max)=({_fa.min():.2f}/"
                      f"{_fa[_fa>1e-6].mean():.2f}/{_fa.max():.2f})N "
                      f"above_stiction={int((_fa > _stick_thr).sum())}",
                      flush=True)
        step     += 1

        # Sink 3 vs Sink 4 diagnostic: Drake-realized contact force on the
        # EE-box pair. Filters out box-ground using LCS formulator's geom IDs.
        try:
            _cr = plant.get_contact_results_output_port().Eval(plant_ctx)
            _eebox_fmag = 0.0
            _gate_F_W = None
            _gate_n_BA = None
            _gate_ia_ee = None
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

            # [OSC-DEBUG] face-normal / tangential decomposition — the
            # port's analog of the reference's osc_debug tracking channels.
            # Quantifies FACE-SKATING: how much of the delivered force and
            # of the EE's actual motion is press (along −n_face_out, into
            # the face) vs slide (tangential). Emitted only when a real
            # EE-object contact exists this tick.
            if _gate_F_W is not None and _n_face_out is not None:
                _n_hat = np.asarray(_n_face_out, dtype=float).reshape(3)
                _n_nrm = float(np.linalg.norm(_n_hat))
                if _n_nrm > 1e-9:
                    _n_hat = _n_hat / _n_nrm
                    # Force ON THE BOX: press = component along −n (into
                    # the face); tangential = the rest (friction drag).
                    _Fb = np.asarray(_F_on_box, dtype=float).reshape(3)
                    _F_press = float(-np.dot(_Fb, _n_hat))
                    _F_tan   = float(np.linalg.norm(
                        _Fb + _F_press * _n_hat))
                    # EE motion this tick: press-directed vs tangential.
                    _ee_prev = getattr(mpc, "_oscdbg_prev_ee", None)
                    _d_ee = (np.asarray(ee_pos) - _ee_prev
                             if _ee_prev is not None else np.zeros(3))
                    _d_press = float(-np.dot(_d_ee, _n_hat))
                    _d_tan   = float(np.linalg.norm(
                        _d_ee + _d_press * _n_hat))
                    print(f"[OSC-DEBUG] step={step} "
                          f"F_press={_F_press:+.3f}N F_tan={_F_tan:.3f}N "
                          f"dEE_press={_d_press*1000:+.3f}mm "
                          f"dEE_tan={_d_tan*1000:.3f}mm "
                          f"n=({_n_hat[0]:+.3f},{_n_hat[1]:+.3f},"
                          f"{_n_hat[2]:+.3f})", flush=True)
            mpc._oscdbg_prev_ee = np.asarray(ee_pos, dtype=float).copy()

        except Exception as _e:
            print(f"[DRAKE-CONTACT] step={step} ERROR={type(_e).__name__}: {_e}", flush=True)

    print("[C3] Simulation complete.")
    if isinstance(mpc, SamplingC3Controller):
        mpc.print_perf_summary()

    # ------------------------------------------------------------------
    # Final result summary
    # ------------------------------------------------------------------
    final_q      = plant.GetPositions(plant_ctx)
    final_obj_xy = np.array([final_q[obj_x_idx], final_q[obj_y_idx]])
    # Reference-conformant 3D goal distance:
    #   sampling_based_c3_controller.cc:768-770 uses .norm() on a 3-element
    #   position segment (obj_x, obj_y, obj_z), summed across objects.
    # Prior port dropped Z. For a box on the ground the Z contribution is
    # small (obj_z stays near box_init_z=0.05), but including it matches
    # reference's position_error metric exactly and catches lift-off cases.
    _init_xyz = task_cfg.get("init_xyz", [0.0, 0.0, 0.05])
    target_xyz = np.array(
        [target_xy[0], target_xy[1], float(_init_xyz[2])], dtype=float)
    final_obj_xyz = np.array(
        [final_q[obj_x_idx], final_q[obj_y_idx], final_q[obj_z_idx]],
        dtype=float)
    final_dist   = float(np.linalg.norm(final_obj_xyz - target_xyz))
    # Geodesic orientation error vs goal: acos((tr(R_goal^T R_final)-1)/2).
    # Reference success gate for push_t is position<0.02m AND orient<0.1 rad.
    _final_quat = final_q[pos_start:pos_start + 4]  # [qw, qx, qy, qz]
    # Drake plant state drifts slightly off unit norm; ad.Quaternion(w,x,y,z)
    # rejects non-normalized input, so renormalize here.
    _qn = float(np.linalg.norm(_final_quat))
    _final_quat = _final_quat / _qn if _qn > 0 else np.array([1.0, 0.0, 0.0, 0.0])
    _R_final = ad.RotationMatrix(ad.Quaternion(
        _final_quat[0], _final_quat[1], _final_quat[2], _final_quat[3]
    )).matrix()
    if target_quat is not None:
        _R_goal_mat = ad.RotationMatrix(ad.Quaternion(
            target_quat[0], target_quat[1],
            target_quat[2], target_quat[3])).matrix()
    else:
        _R_goal_mat = ad.RotationMatrix.MakeZRotation(target_yaw).matrix()
    _tr = float(np.trace(_R_goal_mat.T @ _R_final))
    orient_err = float(np.arccos(np.clip((_tr - 1.0) * 0.5, -1.0, 1.0)))
    if args.sampling_c3 is not None:
        _method = "sampling-c3"
    else:
        _method = "baseline-C3"
    _tight_final = (final_dist < 0.02 and orient_err < 0.1)
    _loose       = (final_dist < 0.05 and orient_err < 0.4)
    # Reference sampling_c3/goal_generator.cc:151 sets
    # `reached_goal_[i] = true` when both position_success_threshold and
    # orientation_success_threshold are simultaneously satisfied — a
    # LATCHING achievement, not re-evaluated on subsequent drift. The
    # controller's `_achieved_fixed_goal` flag mirrors this semantics
    # (sampling_based_c3_controller.py:1539-1543 uses the same 0.02/0.10
    # thresholds). Once the latch fires, both criteria WERE inside their
    # thresholds simultaneously at that instant — the reference considers
    # this "reached." Port previously required both criteria at
    # end-of-sim, which failed by physics settling drift (2 mm typical)
    # after the retreat fix (commit 17efb4e) parks the arm.
    # _tight_ever_latched is the sticky achievement RECORD (reference
    # cc:887-897): set the instant both criteria hold, never cleared by the
    # authorized achieved-goal-release deviation (which clears only the
    # dispatch pin _achieved_fixed_goal to re-engage on post-latch drift).
    # Fall back to the pin for controllers predating the record.
    _tight_latched = bool(getattr(mpc, "_tight_ever_latched", False)
                          or getattr(mpc, "_achieved_fixed_goal", False))
    _tight = _tight_final or _tight_latched
    _tight_reason = ("final" if _tight_final
                     else ("latched" if _tight_latched else "-"))
    if _goal_gen is not None:
        # kRandom headline: the reference task is continuous re-goaling, so
        # goals_reached is the success count; RESULT below is measured
        # against the LAST active goal only.
        if _goal_gen.success_mode == "flip":
            print(f"[GOAL-GEN] flips={_goal_gen.flip_events} "
                  f"(flip success mode: goals_reached counts goal-tripod "
                  f"matches; flips counts ALL persisted tripod changes)")
        print(f"[GOAL-GEN] goals_reached={_goal_gen.goals_reached} "
              f"(kRandom re-goaling; RESULT metrics are vs the final goal)")
    print(f"[RESULT] method={_method}  "
          f"final_obj_xy=({final_obj_xy[0]:.4f}, {final_obj_xy[1]:.4f})  "
          f"translational_error={final_dist:.4f}m  "
          f"rotational_error={orient_err:.4f}rad  "
          f"success={'YES' if final_dist < 0.05 else 'NO'}  "
          f"tight_goal={'PASS' if _tight else 'FAIL'}({_tight_reason})  "
          f"loose_goal={'PASS' if _loose else 'FAIL'}")

    # 2026-07-22: reference-style pydrake.visualization.VideoWriter finalize.
    # Mirrors process_lcm_logs.py:501-509 make_video pattern — Drake wrote
    # frames into memory during sim via ConnectRgbdSensor; Save() flushes
    # them to disk as MP4. Filename comes from env var (default set in
    # env_builder.py), letting run scripts point at their `_ref.mp4` output.
    if drake_video_writer is not None:
        try:
            drake_video_writer.Save()
            print(f"[VIDEO] pydrake.visualization.VideoWriter → "
                  f"{drake_video_writer._filename} saved  "
                  f"(ref process_lcm_logs.py:501-509)", flush=True)
        except Exception as _vw_err:
            print(f"[VIDEO] VideoWriter.Save() failed: "
                  f"{type(_vw_err).__name__}: {_vw_err}", flush=True)

    # (In-sim frame encode + extra-log copy pruned 2026-08-05:
    #  scripts/make_run_video.sh renders post-hoc from the run log;
    #  scripts/sync_results_to_d.sh mirrors results/ to the D sink.)


if __name__ == "__main__":
    main()
