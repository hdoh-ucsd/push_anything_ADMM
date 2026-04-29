#!/usr/bin/env python3
"""
Standalone pretests for the --prepositioned diagnostic chain.

Each pretest isolates one suspect from the diagnosis sequence and runs
in seconds (no sim loop, no Meshcat, no video). Output is written as
JSON to results/pretest_<N>.json so runs can be diffed after config
changes.

The script is intentionally standalone:
  - Does NOT import main.py or anything that runs a Drake simulation.
  - Imports only build_environment, LCSFormulator, the IK solver, and
    sampling-C3 params/utility modules.
  - Disables Meshcat by patching ad.StartMeshcat to a no-op before
    build_environment runs.

Usage
-----

    python profiling/pretest_prepositioned.py --test 1
    python profiling/pretest_prepositioned.py --test 2 [--task pushing]
    python profiling/pretest_prepositioned.py --test 1 --task-id 0   # east
    python profiling/pretest_prepositioned.py --test 1 --all-tasks   # 1, 2, 3, 4

Each test produces results/pretest_<N>.json (or
results/pretest_<N>_<task>_<dir>.json when iterating).

Tests
-----
  1 : FK of INITIAL_ARM_Q. Where does the EE actually start?
      Compares against env_builder.py's docstring claim and against
      the robot base position to determine if the EE is "in front of"
      the robot or "beside" it.

  2 : Reachability sweep. From INITIAL_ARM_Q, can IK reach common
      reposition targets (proxy points for each push direction, and
      generic safe-height waypoints)? Reports per-target ik_err,
      iters, joint-limit margin, and pass/fail.

  3 : (TODO) Arm Jacobian conditioning at the prepositioned pose.
  4 : (TODO) Single-step Cartesian-force realisation via inverse dynamics.
  5 : (TODO) One C3MPC.compute_control() call, dump all internals.
  6 : (TODO) Apply u_seq[0] for one Drake step, compare LCS prediction.
  7 : PD tracker convergence to a FIXED target. Closed-loop Drake sim
      driven only by PiecewiseLinearTracker (no outer controller, no
      sample resampling). Tests whether the tracker can drive the EE
      toward a stationary goal at all.
  8 : Sample variance per loop. Calls generate_samples 100 times from
      the same state and reports xy variance + angular spread on the
      0.18 m circle. Quantifies how much the random target jitters.

The remaining TODOs are stubbed so the CLI is stable; they print a clear
"not implemented yet" and exit 0.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pydrake.all as ad
import yaml

# ---------------------------------------------------------------------------
# Path setup so we can import sim/, control/ from the repo root
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Suppress Meshcat startup before any module touches it. build_environment
# calls ad.StartMeshcat() unconditionally; for a headless pretest we want
# that to be a no-op.
class _NullMeshcat:
    """Stub returned by the patched StartMeshcat. Drake's MeshcatVisualizer
    builder is happy enough with this since we never inspect/connect it."""
    def __getattr__(self, name):
        # Any attribute access returns a no-op callable
        return lambda *a, **kw: None

_orig_start_meshcat = ad.StartMeshcat
def _stub_start_meshcat(*args, **kwargs):
    return _NullMeshcat()
ad.StartMeshcat = _stub_start_meshcat

# Likewise, MeshcatVisualizer.AddToBuilder needs to accept _NullMeshcat
# without complaint. Patch its `AddToBuilder` to a no-op when the third
# arg is our stub.
_orig_add_to_builder = ad.MeshcatVisualizer.AddToBuilder
def _patched_add_to_builder(builder, scene_graph, meshcat, *args, **kwargs):
    if isinstance(meshcat, _NullMeshcat):
        return None
    return _orig_add_to_builder(builder, scene_graph, meshcat, *args, **kwargs)
ad.MeshcatVisualizer.AddToBuilder = staticmethod(_patched_add_to_builder)

# Now safe to import repo modules
from sim.env_builder import (                                          # noqa: E402
    build_environment,
    INITIAL_ARM_Q,
    EE_BODY_NAME,
    PUSHER_RADIUS,
    ROBOT_BASE_XYZ,
    compute_prepositioned_arm_q,
)
from control.sampling_c3.ik import solve_ik_to_ee_pos                  # noqa: E402


# ---------------------------------------------------------------------------
# Common setup helpers
# ---------------------------------------------------------------------------

def load_task_cfg(task_name: str) -> dict:
    """Load one task's config block from config/tasks.yaml."""
    cfg_path = REPO_ROOT / "config" / "tasks.yaml"
    with open(cfg_path) as f:
        all_cfg = yaml.safe_load(f)
    if task_name not in all_cfg["tasks"]:
        raise ValueError(f"task {task_name!r} not in tasks.yaml; "
                         f"available: {list(all_cfg['tasks'])}")
    return all_cfg["tasks"][task_name]


def build_for_pretest(task_cfg: dict, override_goal_xy: list[float] | None = None):
    """Run build_environment + assemble plant context + obj body handle.

    Returns
    -------
    plant       : MultibodyPlant (Finalized)
    plant_ctx   : Drake Context for the plant inside the diagram
    diagram_ctx : Drake diagram context (kept alive so plant_ctx stays valid)
    panda_model : ModelInstanceIndex
    obj_body    : Drake Body for the manipulated object
    ee_frame    : Drake Frame for the welded pusher
    """
    if override_goal_xy is not None:
        task_cfg = {**task_cfg, "goal_xy": list(override_goal_xy)}

    diagram, plant, panda_model, object_model, _meshcat = build_environment(task_cfg)

    diagram_ctx = diagram.CreateDefaultContext()
    plant_ctx   = plant.GetMyMutableContextFromRoot(diagram_ctx)

    obj_body = plant.GetBodyByName(task_cfg["link_name"], object_model)
    ee_frame = plant.GetFrameByName(EE_BODY_NAME)

    # Stage the object pose so any IK that probes it sees it at the task init
    plant.SetFreeBodyPose(
        plant_ctx, obj_body,
        ad.RigidTransform(ad.RotationMatrix(), task_cfg["init_xyz"]),
    )

    return plant, plant_ctx, diagram_ctx, panda_model, obj_body, ee_frame


def fk_ee(plant, plant_ctx, ee_frame, q_full: np.ndarray) -> np.ndarray:
    """Forward kinematics: pusher centre position in world frame."""
    plant.SetPositions(plant_ctx, q_full)
    return plant.CalcPointsPositions(
        plant_ctx, ee_frame, np.zeros(3), plant.world_frame()
    ).flatten()


def joint_limits_arm(plant, n_arm: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (q_lo, q_hi) sliced to the arm's first n_arm DOFs."""
    return (
        plant.GetPositionLowerLimits()[:n_arm].copy(),
        plant.GetPositionUpperLimits()[:n_arm].copy(),
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy types into JSON-serialisable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def write_result(test_id: int, result: dict, suffix: str = "") -> Path:
    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    name = f"pretest_{test_id}{('_' + suffix) if suffix else ''}.json"
    path = out_dir / name
    with open(path, "w") as f:
        json.dump(to_jsonable(result), f, indent=2)
    return path


# ===========================================================================
# Pretest 1 — FK of INITIAL_ARM_Q
# ===========================================================================

def pretest_1(task_cfg: dict) -> dict:
    """Where does INITIAL_ARM_Q put the EE?

    Reads INITIAL_ARM_Q, builds the plant, runs FK on the welded pusher,
    and computes interpretable diagnostics:
      - EE position in world frame
      - displacement from robot base
      - displacement from object init position
      - "is the EE in front of the robot" check (signed dot with +y, since
        the Panda is welded with its base at y=-0.6 and faces +y)
      - per-joint values vs joint-limit boundaries (margin to limit)
    """
    plant, plant_ctx, _ctx, panda_model, _obj, ee_frame = build_for_pretest(task_cfg)

    n_arm = plant.num_actuators()
    q_lo_arm, q_hi_arm = joint_limits_arm(plant, n_arm)

    plant.SetPositions(plant_ctx, panda_model, INITIAL_ARM_Q)
    p_ee = plant.CalcPointsPositions(
        plant_ctx, ee_frame, np.zeros(3), plant.world_frame()
    ).flatten()

    base = np.asarray(ROBOT_BASE_XYZ, dtype=float)
    obj_init = np.asarray(task_cfg["init_xyz"], dtype=float)

    # Per-joint margin to nearest limit (rad). Negative = over the limit.
    per_joint = []
    for i, q_i in enumerate(INITIAL_ARM_Q):
        margin_lo = float(q_i - q_lo_arm[i])
        margin_hi = float(q_hi_arm[i] - q_i)
        per_joint.append({
            "joint": i,
            "q":               float(q_i),
            "lo":              float(q_lo_arm[i]),
            "hi":              float(q_hi_arm[i]),
            "margin_to_lo":    margin_lo,
            "margin_to_hi":    margin_hi,
            "in_limits":       (margin_lo > 0) and (margin_hi > 0),
        })

    # Forward-direction check: Panda is welded with link0 at base. The
    # default Panda URDF orientation places its working volume in +y of
    # link0. With base at (0, -0.6, 0), "forward" relative to the robot
    # is +y in world frame. So if EE.y > base.y, the EE is in front of
    # the robot; if EE.y ≈ base.y, the EE is sideways (collapsed onto
    # the base column).
    ee_minus_base = p_ee - base
    forward_signed = float(ee_minus_base[1])     # >0 = forward, <0 = behind
    sideways = float(np.linalg.norm(ee_minus_base[[0, 2]]))   # x,z displacement

    # Distance to object — the planner needs the EE to traverse this
    ee_to_obj = float(np.linalg.norm(p_ee - obj_init))
    ee_to_obj_xy = float(np.linalg.norm(p_ee[:2] - obj_init[:2]))

    return {
        "test": "pretest_1_fk_initial_arm_q",
        "task": task_cfg.get("link_name", "?"),
        "INITIAL_ARM_Q": INITIAL_ARM_Q.tolist(),
        "ee_position_world":  p_ee.tolist(),
        "robot_base_world":   base.tolist(),
        "object_init_world":  obj_init.tolist(),
        "ee_minus_base":      ee_minus_base.tolist(),
        "forward_signed_y":   forward_signed,
        "sideways_xz_norm":   sideways,
        "ee_to_obj":          ee_to_obj,
        "ee_to_obj_xy":       ee_to_obj_xy,
        "verdict": (
            "EE is forward of base" if forward_signed > 0.05
            else "EE is approximately at base column (sideways)" if abs(forward_signed) < 0.05
            else "EE is BEHIND base (worse than expected)"
        ),
        "per_joint": per_joint,
        "all_joints_in_limits": all(j["in_limits"] for j in per_joint),
        "expected_per_env_builder_comment": [0.347, -0.600, 0.097],
        "ee_matches_comment_within_5mm": bool(
            np.linalg.norm(p_ee - np.array([0.347, -0.600, 0.097])) < 0.005
        ),
    }


# ===========================================================================
# Pretest 2 — Reachability sweep
# ===========================================================================

def pretest_2(task_cfg: dict) -> dict:
    """From INITIAL_ARM_Q, can IK reach a battery of relevant targets?

    Targets:
      - Sampling-radius circle proxy points for all 4 push directions
        (east/north/west/south), at the standard z=0.05 (sampling_height).
      - Lifted "safe-height" waypoints at z=0.20 above the same xy
        points (matches PiecewiseLinearTracker's pwl_waypoint_height).
      - The contact-face point itself (1 mm gap on the push axis) for
        the default east push.

    For each target the test reports:
      ik_err           : final EE position error (m)
      iters            : DLS iterations used
      converged        : ik_err < 5 mm
      q_arm            : converged arm config
      joint_violations : list of joints outside limits ± 0.005 rad margin
      reachable        : converged AND joints within limits
    """
    plant, plant_ctx, _ctx, panda_model, obj_body, ee_frame = build_for_pretest(task_cfg)

    n_arm = plant.num_actuators()
    q_lo_arm, q_hi_arm = joint_limits_arm(plant, n_arm)

    obj_xy = np.array(task_cfg["init_xyz"][:2])
    z_circle = 0.05    # sampling_params.sampling_height default
    z_lift   = 0.20    # reposition_params.pwl_waypoint_height default
    r_circle = 0.18    # sampling_params.sampling_radius default

    # Object half-extent on each axis (so the contact-face target works
    # for any direction, not just east; box is axis-aligned at init).
    obj_type = task_cfg["object_type"]
    if obj_type == "box":
        sx, sy, _ = task_cfg["size"]
        half_x, half_y = sx / 2.0, sy / 2.0
    else:
        r = float(task_cfg["radius"])
        half_x = half_y = r

    targets: list[dict] = []

    # 4 directional proxy points on the sampling circle
    for name, g_hat in [("east",  [+1, 0]),
                        ("north", [0, +1]),
                        ("west",  [-1, 0]),
                        ("south", [0, -1])]:
        gx, gy = g_hat
        # Proxy: behind the box on the push axis, at z=sampling_height
        targets.append({
            "name":   f"proxy_{name}",
            "p":      [obj_xy[0] - r_circle * gx,
                       obj_xy[1] - r_circle * gy,
                       z_circle],
        })
        # Lifted waypoint above the proxy
        targets.append({
            "name":   f"safe_lift_{name}",
            "p":      [obj_xy[0] - r_circle * gx,
                       obj_xy[1] - r_circle * gy,
                       z_lift],
        })

    # Contact-face target for east push (1mm gap)
    half_along_g_east = abs(+1.0) * half_x + abs(0.0) * half_y
    contact_offset = half_along_g_east + PUSHER_RADIUS + 0.001
    targets.append({
        "name": "contact_face_east",
        "p":    [obj_xy[0] - contact_offset, obj_xy[1], z_circle],
    })

    # Sanity target: directly above the box at safe-height
    targets.append({
        "name": "above_box",
        "p":    [obj_xy[0], obj_xy[1], z_lift],
    })

    # Sanity target: directly above the robot base
    targets.append({
        "name": "above_base",
        "p":    [ROBOT_BASE_XYZ[0], ROBOT_BASE_XYZ[1] + 0.30, z_lift],
    })

    n_q = plant.num_positions()

    results = []
    for t in targets:
        p_target = np.asarray(t["p"], dtype=float)

        # Stage the plant: arm at INITIAL_ARM_Q, full q vector.
        plant.SetPositions(plant_ctx, panda_model, INITIAL_ARM_Q)
        q_full = plant.GetPositions(plant_ctx).copy()

        q_out, ik_err, iters = solve_ik_to_ee_pos(
            plant, ee_frame,
            p_target = p_target,
            q_init   = q_full,
            plant_ctx= plant_ctx,
            n_arm_dofs= n_arm,
            max_iter = 80,
            tol      = 1e-3,
            damping  = 0.05,
            q_lo     = q_lo_arm,
            q_hi     = q_hi_arm,
        )
        q_arm_out = q_out[:n_arm]

        # FK at the converged config to get actual EE position
        plant.SetPositions(plant_ctx, q_out)
        p_actual = plant.CalcPointsPositions(
            plant_ctx, ee_frame, np.zeros(3), plant.world_frame()
        ).flatten()

        joint_violations = []
        for i in range(n_arm):
            if q_arm_out[i] < q_lo_arm[i] - 0.005:
                joint_violations.append(
                    {"joint": i, "q": float(q_arm_out[i]),
                     "lo": float(q_lo_arm[i]), "amount_below": float(q_lo_arm[i] - q_arm_out[i])})
            elif q_arm_out[i] > q_hi_arm[i] + 0.005:
                joint_violations.append(
                    {"joint": i, "q": float(q_arm_out[i]),
                     "hi": float(q_hi_arm[i]), "amount_above": float(q_arm_out[i] - q_hi_arm[i])})

        converged = bool(ik_err < 5e-3)
        results.append({
            "name":             t["name"],
            "p_target":         p_target.tolist(),
            "p_actual":         p_actual.tolist(),
            "ik_err_m":         float(ik_err),
            "iters":            int(iters),
            "converged":        converged,
            "q_arm":            q_arm_out.tolist(),
            "joint_violations": joint_violations,
            "reachable":        converged and len(joint_violations) == 0,
        })

    n_reachable = sum(1 for r in results if r["reachable"])
    return {
        "test":               "pretest_2_reachability_sweep",
        "task":               task_cfg.get("link_name", "?"),
        "INITIAL_ARM_Q_seed": INITIAL_ARM_Q.tolist(),
        "n_targets":          len(results),
        "n_reachable":        n_reachable,
        "n_unreachable":      len(results) - n_reachable,
        "targets":            results,
    }


# ===========================================================================
# TODOs (3-6) — stubbed for stable CLI
# ===========================================================================

def pretest_3(task_cfg: dict) -> dict:
    return {"test": "pretest_3_jacobian_conditioning", "status": "not_implemented_yet"}

def pretest_4(task_cfg: dict) -> dict:
    return {"test": "pretest_4_force_realisation", "status": "not_implemented_yet"}

def pretest_5(task_cfg: dict) -> dict:
    return {"test": "pretest_5_one_compute_control", "status": "not_implemented_yet"}

def pretest_6(task_cfg: dict) -> dict:
    return {"test": "pretest_6_lcs_drake_step", "status": "not_implemented_yet"}


# ===========================================================================
# Pretest 7 — PD tracker convergence to a FIXED target
# ===========================================================================

def pretest_7(task_cfg: dict, target: list[float] | None = None,
              n_steps: int = 200, dt_ctrl: float = 0.01) -> dict:
    """Drive the arm with PiecewiseLinearTracker only (no outer loop).

    Setup:
      - Arm at INITIAL_ARM_Q.
      - PiecewiseLinearTracker built from RepositionParams() defaults.
      - Target FIXED for the entire run (no resampling). Default
        target is the east proxy point (-0.18, 0, 0.05); override
        with --target X Y Z.
      - Drake simulator advances dt_ctrl each step (default 10 ms,
        matching main.py's control loop period).
      - Object held at task_cfg init pose; if it moves we'll see it.

    Records per-step:
      t, q_arm, ee_xyz, q_err_norm, integral_norm, |τ|.

    Reports:
      - converged_within_5mm: did the EE reach |target| within 5 mm?
      - n_steps_to_converge: when (if ever) it first dropped below 5 mm.
      - final_ee_pos, final_ee_err.
      - net_ee_displacement: how far the EE actually moved (any direction).
      - drift_perpendicular_to_target: was the motion in the right direction?
        Computed as the projection of net displacement onto target_dir vs
        its perpendicular complement.
      - τ statistics: max, mean across the run.

    Reads
    -----
      If converged: tracker works. Suspect 3 is purely about the outer
      loop's resampling rate (test 8 then quantifies that).
      If not converged but EE moved monotonically toward target: tracker
      is just slow; turn up Kp/Ki or run longer.
      If EE drifted away from target: tracker has its own bug
      (gravity-comp, sign convention, IK-vs-PD mismatch).
    """
    # Local import keeps test 1/2 free of the reposition.py dep
    from control.sampling_c3.reposition import PiecewiseLinearTracker     # noqa: WPS433
    from control.sampling_c3.params import RepositionParams                # noqa: WPS433

    # --- Build env, including the simulator/diagram needed for AdvanceTo --
    diagram, plant, panda_model, object_model, _meshcat = build_environment(task_cfg)
    simulator = ad.Simulator(diagram)
    diagram_ctx = simulator.get_mutable_context()
    plant_ctx   = plant.GetMyMutableContextFromRoot(diagram_ctx)

    obj_body = plant.GetBodyByName(task_cfg["link_name"], object_model)
    ee_frame = plant.GetFrameByName(EE_BODY_NAME)

    # Stage initial state
    plant.SetPositions(plant_ctx, panda_model, INITIAL_ARM_Q)
    plant.SetFreeBodyPose(
        plant_ctx, obj_body,
        ad.RigidTransform(ad.RotationMatrix(), task_cfg["init_xyz"]),
    )
    # Zero velocities everywhere
    plant.SetVelocities(plant_ctx, np.zeros(plant.num_velocities()))

    # Wire the actuation port: the tracker's torque is applied through
    # plant.get_actuation_input_port(panda_model). We call FixValue
    # each control step with the new torque before AdvanceTo.
    n_arm = plant.num_actuators()
    actuation_port = plant.get_actuation_input_port(panda_model)
    # Initialise to zero so the first AdvanceTo doesn't fail on a
    # disconnected port (Drake aborts if any input is unfixed).
    actuation_port.FixValue(plant_ctx, np.zeros(n_arm))

    # --- Tracker --------------------------------------------------------
    repo_params = RepositionParams()       # defaults from params.py
    tracker = PiecewiseLinearTracker(
        plant       = plant,
        ee_frame    = ee_frame,
        n_arm_dofs  = n_arm,
        params      = repo_params,
    )

    # Default target: east proxy point (matches pretest 2's first entry)
    if target is None:
        obj_xy = np.array(task_cfg["init_xyz"][:2])
        p_target = np.array([obj_xy[0] - 0.18, obj_xy[1], 0.05])
    else:
        p_target = np.asarray(target, dtype=float)

    # --- Initial FK reading --------------------------------------------
    p_ee_init = plant.CalcPointsPositions(
        plant_ctx, ee_frame, np.zeros(3), plant.world_frame()
    ).flatten()

    # --- Closed-loop run ------------------------------------------------
    history: list[dict] = []
    n_steps_to_converge = None
    sim_time = 0.0

    for step in range(n_steps):
        # Read current state
        q_full_now = plant.GetPositions(plant_ctx).copy()
        v_full_now = plant.GetVelocities(plant_ctx).copy()

        # Compute torque toward p_target (held constant)
        u, diag = tracker.compute_torque(
            current_q = q_full_now,
            current_v = v_full_now,
            plant_ctx = plant_ctx,
            p_target  = p_target,
            dt_ctrl   = dt_ctrl,
        )

        # Apply torque and advance
        actuation_port.FixValue(plant_ctx, u)
        sim_time += dt_ctrl
        simulator.AdvanceTo(sim_time)

        # Record
        q_full_after = plant.GetPositions(plant_ctx)
        plant.SetPositions(plant_ctx, q_full_after)   # explicit, no-op effectively
        ee_now = plant.CalcPointsPositions(
            plant_ctx, ee_frame, np.zeros(3), plant.world_frame()
        ).flatten()
        ee_err = float(np.linalg.norm(ee_now - p_target))

        history.append({
            "step":            step,
            "t":               sim_time,
            "ee_xyz":          ee_now.tolist(),
            "ee_err":          ee_err,
            "q_arm":           q_full_after[:n_arm].tolist(),
            "qerr_norm":       float(diag["qerr_norm"]),
            "integral_norm":   float(np.linalg.norm(tracker._integral)),
            "u_norm":          float(diag["uclip_norm"]),
            "ik_err":          float(diag["ik_err"]),
        })

        if n_steps_to_converge is None and ee_err < 5e-3:
            n_steps_to_converge = step

    # --- Aggregate diagnostics -----------------------------------------
    final = history[-1]
    final_ee = np.array(final["ee_xyz"])
    net_disp = final_ee - p_ee_init
    target_dir = p_target - p_ee_init
    target_dist0 = float(np.linalg.norm(target_dir))
    if target_dist0 > 1e-6:
        target_unit = target_dir / target_dist0
        proj_along = float(np.dot(net_disp, target_unit))
        proj_perp  = float(np.linalg.norm(net_disp - proj_along * target_unit))
    else:
        proj_along = 0.0
        proj_perp  = float(np.linalg.norm(net_disp))

    u_norms = [h["u_norm"] for h in history]
    ee_errs = [h["ee_err"] for h in history]

    # Sample the trajectory at 5 evenly-spaced points for the JSON
    # (full history is 200 entries — too verbose to dump in full but we
    # keep it under a key in case a follow-up script wants it).
    sample_indices = np.linspace(0, n_steps - 1, 5).astype(int).tolist()
    sampled = [history[i] for i in sample_indices]

    return {
        "test":                 "pretest_7_pd_tracker_fixed_target",
        "task":                 task_cfg.get("link_name", "?"),
        "p_target":             p_target.tolist(),
        "p_ee_init":            p_ee_init.tolist(),
        "p_ee_final":           final_ee.tolist(),
        "n_steps":              n_steps,
        "dt_ctrl":              dt_ctrl,
        "total_sim_time":       sim_time,
        "initial_ee_err":       float(np.linalg.norm(p_ee_init - p_target)),
        "final_ee_err":         float(final["ee_err"]),
        "min_ee_err":           float(min(ee_errs)),
        "min_ee_err_at_step":   int(np.argmin(ee_errs)),
        "converged_within_5mm": n_steps_to_converge is not None,
        "n_steps_to_converge":  n_steps_to_converge,
        "net_ee_displacement":  float(np.linalg.norm(net_disp)),
        "displacement_along_target":      proj_along,
        "displacement_perpendicular":     proj_perp,
        "fraction_progress_along_target": (proj_along / target_dist0) if target_dist0 > 1e-6 else 0.0,
        "u_norm_max":           float(max(u_norms)),
        "u_norm_mean":          float(np.mean(u_norms)),
        "trajectory_samples":   sampled,
        "full_history":         history,
    }


# ===========================================================================
# Pretest 8 — Sample variance per outer-loop call
# ===========================================================================

def pretest_8(task_cfg: dict, n_calls: int = 100, seed: int | None = None) -> dict:
    """How much does generate_samples jitter target positions per loop?

    Calls generate_samples(strategy=kRandomOnCircle, ...) n_calls times
    from the same state, with the same g_hat. Records:
      - All sampled (x, y, z) positions for k=0 (the proxy) and any
        additional strategy samples.
      - For the proxy (which is deterministic when g_hat is fixed):
        std should be ≈ 0.
      - For the additional random samples on the circle: mean, std,
        full xy range, angular distribution.

    Reads
    -----
      If proxy std ≈ 0 and additional-sample std is large (~5+ cm):
      this is the structural problem. Every loop the outer controller
      hands a different "best other" target to the tracker, the
      tracker resets its integrator, and progress can't accumulate.
      If both stds are ≈ 0: sampling is deterministic for some reason
      (RNG seeded to a constant, num_additional == 0, etc.) — the
      problem is elsewhere.
    """
    from control.sampling_c3.params import SamplingParams, SamplingStrategy
    from control.sampling_c3.sampling import generate_samples

    sp = SamplingParams()    # defaults; matches what main.py uses unless
                             # the YAML override changes them.

    # Pick the same g_hat the wrapper would for this task
    obj_xy = np.array(task_cfg["init_xyz"][:2], dtype=float)
    goal_xy = np.array(task_cfg["goal_xy"], dtype=float)
    g_vec = goal_xy - obj_xy
    g_norm = float(np.linalg.norm(g_vec))
    g_hat = g_vec / g_norm if g_norm > 1e-9 else np.array([1.0, 0.0])

    # Match wrapper.py: prev_mode='free' uses num_additional_samples_repos
    # (default 1). Test in repos mode since that's where the problem lives.
    n_strategy = sp.num_additional_samples_repos

    rng = np.random.default_rng(seed)

    sample_records: list[dict] = []
    for call_idx in range(n_calls):
        positions = generate_samples(
            strategy  = sp.sampling_strategy,
            n_samples = n_strategy,
            obj_xy    = obj_xy,
            params    = sp,
            rng       = rng,
            g_hat     = g_hat,
        )
        sample_records.append({
            "call":       call_idx,
            "n_returned": len(positions),
            "positions":  [p.tolist() for p in positions],
        })

    # Aggregate per-slot stats. Slot 0 is the proxy (deterministic);
    # slot 1+ are the random circle samples.
    all_positions = np.array([
        rec["positions"] for rec in sample_records
        if len(rec["positions"]) >= 1
    ])    # shape (n_calls, n_strategy, 3)

    per_slot = []
    for slot in range(all_positions.shape[1] if all_positions.ndim == 3 else 0):
        col = all_positions[:, slot, :]            # (n_calls, 3)
        xy_mean = col[:, :2].mean(axis=0)
        xy_std  = col[:, :2].std(axis=0)
        # Angle of each sample relative to obj_xy
        angles = np.arctan2(col[:, 1] - obj_xy[1], col[:, 0] - obj_xy[0])
        angles_deg = np.degrees(angles)

        per_slot.append({
            "slot":               slot,
            "label":              "proxy" if slot == 0 else f"random_{slot - 1}",
            "n_samples":          len(col),
            "xy_mean":            xy_mean.tolist(),
            "xy_std":             xy_std.tolist(),
            "xy_std_norm":        float(np.linalg.norm(xy_std)),
            "x_range":            [float(col[:, 0].min()), float(col[:, 0].max())],
            "y_range":            [float(col[:, 1].min()), float(col[:, 1].max())],
            "z":                  float(col[0, 2]),    # constant by design
            "angle_deg_mean":     float(angles_deg.mean()),
            "angle_deg_std":      float(angles_deg.std()),
            "angle_deg_range":    [float(angles_deg.min()), float(angles_deg.max())],
        })

    return {
        "test":            "pretest_8_sample_variance_per_loop",
        "task":            task_cfg.get("link_name", "?"),
        "n_calls":         n_calls,
        "n_strategy":      n_strategy,
        "obj_xy":          obj_xy.tolist(),
        "goal_xy":         goal_xy.tolist(),
        "g_hat":           g_hat.tolist(),
        "sampling_strategy": sp.sampling_strategy.name,
        "sampling_radius": sp.sampling_radius,
        "sampling_height": sp.sampling_height,
        "rng_seed":        seed,
        "per_slot_stats":  per_slot,
    }


# ===========================================================================
# CLI
# ===========================================================================

PRETESTS = {1: pretest_1, 2: pretest_2,
            3: pretest_3, 4: pretest_4, 5: pretest_5, 6: pretest_6,
            7: pretest_7, 8: pretest_8}


def main():
    parser = argparse.ArgumentParser(
        description="Standalone pretests for --prepositioned diagnosis. "
                    "JSON output written to results/pretest_<N>[_suffix].json.")
    parser.add_argument("--test", type=int, required=True, choices=PRETESTS.keys(),
                        help="Which pretest to run (1-8).")
    parser.add_argument("--task", type=str, default="pushing",
                        help="Task name from config/tasks.yaml (default: pushing).")
    parser.add_argument("--quiet", action="store_true",
                        help="Don't echo the JSON to stdout, only write the file.")

    # Test 7 specifics
    parser.add_argument("--target", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="(test 7) Override default tracker target. "
                             "Default: east proxy point at obj_xy − [0.18, 0, 0] + [0,0,0.05].")
    parser.add_argument("--n-steps", type=int, default=200,
                        help="(test 7) Control steps to run (default 200 = 2 s at dt_ctrl=0.01).")
    parser.add_argument("--dt-ctrl", type=float, default=0.01,
                        help="(test 7) Control loop period in seconds (default 0.01 = 100 Hz).")

    # Test 8 specifics
    parser.add_argument("--n-calls", type=int, default=100,
                        help="(test 8) Number of generate_samples invocations (default 100).")
    parser.add_argument("--seed", type=int, default=None,
                        help="(test 8) RNG seed for reproducibility (default: nondeterministic).")

    args = parser.parse_args()

    task_cfg = load_task_cfg(args.task)
    fn = PRETESTS[args.test]
    if args.test == 7:
        result = fn(task_cfg,
                    target  = args.target,
                    n_steps = args.n_steps,
                    dt_ctrl = args.dt_ctrl)
    elif args.test == 8:
        result = fn(task_cfg,
                    n_calls = args.n_calls,
                    seed    = args.seed)
    else:
        result = fn(task_cfg)
    out_path = write_result(args.test, result, suffix=args.task)
    print(f"[pretest_{args.test}] wrote {out_path}")
    if not args.quiet:
        print(json.dumps(to_jsonable(result), indent=2))


if __name__ == "__main__":
    main()
