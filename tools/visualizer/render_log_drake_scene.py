"""§7.73d — render Drake-shaded VTK PNGs from the honest seed-0 log.

Take-1's format used main.py's Drake render camera + paint_mode_text.py's
CONTACT-FREE / CONTACT-RICH mode banner. To reproduce that format without
re-running the sim, this script parses the log's [GATE-CONTACT] pose (box_p,
box_q, ee_p) and [STEP] mode, then:

  1. Builds the same Drake scene_graph env_builder produces (Franka arm +
     red box manipuland + green goal ghost + blue pusher sphere).
  2. Per stride tick: SetFreeBodyPose(box) + DLS-IK to place EE at log's ee_p
     + query the drake_render_camera output port + save frame_NNNNNN.png.
  3. Writes mode_timeline.csv (step, sim_t, mode, switch) parsed from [STEP].

paint_mode_text.py then paints the mode banner and encodes the MP4.

Usage
-----
    PORT_CAMERA_PERSPECTIVE=1 python3 tools/visualizer/render_log_drake_scene.py \\
        full_reference_aligned_772/seed0.log \\
        --out-dir /tmp/seed0_ref_drake_frames \\
        --stride 3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import yaml
import pydrake.all as ad
from PIL import Image

# Force perspective camera pose BEFORE env_builder imports pydrake render
# helpers.  env_builder reads this env var when add_camera=True.
os.environ.setdefault("PORT_CAMERA_PERSPECTIVE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sim.env_builder import (  # noqa: E402
    build_environment, compute_safe_init_arm_q,
    EE_BODY_NAME, _INITIAL_ARM_Q_SEED as INITIAL_ARM_Q,
)
from control.sampling_c3.ik import solve_ik_to_ee_pos  # noqa: E402

# Vertical-tool rotation target for the per-frame IK — identical to the
# sim's _R_vert (sim/env_builder.py) that the OSC holds at Kp_rot=800.
_R_VERT = ad.RotationMatrix()


RE_GATE = re.compile(
    r"^\[GATE-CONTACT\] step=(\d+).*?"
    r"box_q=\(([^)]+)\)\s+"
    r"box_p=\(([^)]+)\)\s+"
    r"ee_p=\(([^)]+)\)"
)
RE_STEP_MODE = re.compile(
    r"^\[STEP\] step=(\d+) mode=(\w+) t=([\d.]+)s.*?switch=(\w+)"
)


def _parse_triple(s: str) -> np.ndarray:
    return np.array([float(x) for x in s.split(",")], dtype=float)


def parse_log(path: Path):
    """Return dict step → {box_p, box_q, ee_p, mode, switch, sim_t}."""
    frames = {}
    # errors="replace" to match the other log readers here and in
    # paint_log_sidepanel.py: the 1 kHz OSC sub-loop can interleave a write
    # mid-way through a multi-byte glyph (the logs carry φ/λ/ρ), leaving a
    # stray continuation byte that would otherwise abort the whole render.
    with open(path, errors="replace") as f:
        for line in f:
            if line.startswith("[GATE-CONTACT]"):
                m = RE_GATE.match(line)
                if not m:
                    continue
                step = int(m.group(1))
                frames.setdefault(step, {}).update(
                    box_q=_parse_triple(m.group(2)),
                    box_p=_parse_triple(m.group(3)),
                    ee_p =_parse_triple(m.group(4)),
                )
            elif line.startswith("[STEP]"):
                m = RE_STEP_MODE.match(line)
                if not m:
                    continue
                step = int(m.group(1))
                frames.setdefault(step, {}).update(
                    mode=m.group(2),
                    switch=m.group(4),
                    # sim_t from the log's own t= field — the prior
                    # step*0.01 assumed a 10 ms planner tick and was wrong
                    # for every other cadence (jack/push_t tick at 0.1 s).
                    sim_t=float(m.group(3)),
                )
    return frames


def parse_log_goal(log_path: Path):
    """Extract the run's ACTUAL goal from its own `[ENV]  Goal coords:` line.

    2026-08-05 fix (reference-shaped): the reference's video pipeline takes
    goal poses from the LOG (process_lcm_logs.py make_video: goals_by_obj
    parsed from LCM messages), never from config files. The prior port
    behavior — unconditionally overriding with directional_tasks.json at
    the --task-id DEFAULT (4 = box-west) — painted a fabricated goal on
    every render whose run didn't use that task-id (p152: goal drawn at
    (0.30, 0) while the run pushed toward (0.482, 0.187)).
    """
    pat = re.compile(r"\[ENV\]\s+Goal coords:\s*\[([-\d.eE+]+),\s*([-\d.eE+]+)\]")
    # SE(3) tasks (jack) also log the ACTIVE goal quaternion; with
    # krandom_draw_initial_goal the drawn quat differs from tasks.yaml's
    # boot value, so the ghost must take it from the log too.
    pat_q = re.compile(
        r"\[GOAL-QUAT\] goal_quat=\[([-+\d.eE]+)\s+([-+\d.eE]+)\s+"
        r"([-+\d.eE]+)\s+([-+\d.eE]+)\]")
    goal_xy, goal_quat = None, None
    with open(log_path, errors="replace") as f:
        for _ in range(500):
            line = f.readline()
            if not line:
                break
            m = pat.search(line)
            if m and goal_xy is None:
                goal_xy = [float(m.group(1)), float(m.group(2))]
            m = pat_q.search(line)
            if m and goal_quat is None:
                goal_quat = [float(m.group(i)) for i in (1, 2, 3, 4)]
            if goal_xy is not None and goal_quat is not None:
                break
    return goal_xy, goal_quat


RE_REGOAL = re.compile(
    r"\[GOAL-GEN\] goal #\d+ REACHED at t=([\d.]+)s -> new goal "
    r"xy=\(([-+\d.]+),([-+\d.]+)\) tripod=\w+ "
    r"quat=\[([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)\]")


def parse_log_regoals(log_path: Path):
    """kRandom re-goal events: [{t, xy, quat}] in log order.

    Each one starts a new goal SEGMENT — the goal ghost is anchored world
    geometry, so the render env must be rebuilt with the new goal pose.
    """
    out = []
    with open(log_path, errors="replace") as f:
        for line in f:
            m = RE_REGOAL.search(line)
            if m:
                out.append({
                    "t": float(m.group(1)),
                    "xy": [float(m.group(2)), float(m.group(3))],
                    "quat": [float(m.group(i)) for i in range(4, 8)],
                })
    return out


def load_task_cfg(task_name: str, task_id, log_goal_pair):
    """main.py's load_task; goal priority: explicit --task-id > log > yaml."""
    log_goal, log_goal_quat = log_goal_pair
    root = Path(__file__).resolve().parents[2]
    with open(root / "config" / "tasks.yaml") as f:
        all_tasks = yaml.safe_load(f)["tasks"]
    task_cfg = dict(all_tasks[task_name])
    if log_goal_quat is not None:
        task_cfg["goal_quat"] = log_goal_quat
        print(f"[render-log-drake] goal quat from log [GOAL-QUAT] line: "
              f"{log_goal_quat}")
    if task_id is not None:
        with open(root / "config" / "directional_tasks.json") as f:
            dir_cfg = json.load(f)
        task_cfg["goal_xy"] = dir_cfg["tasks"][str(task_id)]["goal"]
        print(f"[render-log-drake] goal from --task-id {task_id}: "
              f"{task_cfg['goal_xy']}")
    elif log_goal is not None:
        task_cfg["goal_xy"] = log_goal
        print(f"[render-log-drake] goal from log [ENV] line: {log_goal}")
    else:
        print(f"[render-log-drake] goal from tasks.yaml: "
              f"{task_cfg.get('goal_xy')}")
    return task_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path,
                    help="Path to seed-0 reference-aligned log")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/tmp/seed0_ref_drake_frames"),
                    help="Where to write frame_NNNNNN.png + mode_timeline.csv")
    ap.add_argument("--stride", type=int, default=3,
                    help="Sim-tick stride (3 → ~200 frames of a 6s run → "
                         "30fps mp4 plays close to real-time)")
    ap.add_argument("--task", default="pushing")
    ap.add_argument("--task-id", type=int, default=None,
                    help="EXPLICIT directional-goal override only; when "
                         "omitted the goal comes from the log's [ENV] line "
                         "(per-run ground truth, reference-shaped).")
    ap.add_argument("--max-step", type=int, default=None,
                    help="Optional cap on the last rendered step")
    ap.add_argument("--min-step", type=int, default=None,
                    help="Optional floor on the first rendered step "
                         "(with --max-step: render a window, e.g. a "
                         "highlight clip of one episode)")
    ap.add_argument("--goal-xy", default=None, metavar="X,Y",
                    help="Override the goal position the ghost is drawn at "
                         "(kRandom multi-goal logs: render each goal segment "
                         "with its own ghost via --min/--max-step windows)")
    ap.add_argument("--goal-yaw", type=float, default=None,
                    help="Override the goal yaw for the ghost (radians)")
    ap.add_argument("--interp", type=int, default=1,
                    help="Sub-frames per log step (poses SLERP/lerp "
                         "interpolated toward the next step). Frame index "
                         "= step*interp + j; pass the SAME --interp to "
                         "paint_log_sidepanel.py so it maps frames back "
                         "to log steps.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[render-log-drake] parsing {args.log}", flush=True)
    frames = parse_log(args.log)
    valid = sorted(k for k, v in frames.items()
                   if "box_p" in v and "ee_p" in v and "mode" in v)
    if args.max_step is not None:
        valid = [k for k in valid if k <= args.max_step]
    if args.min_step is not None:
        valid = [k for k in valid if k >= args.min_step]
    print(f"[render-log-drake] {len(valid)} valid steps in log "
          f"(first={valid[0]}, last={valid[-1]})", flush=True)

    print("[render-log-drake] loading task config", flush=True)
    task_cfg = load_task_cfg(args.task, args.task_id, parse_log_goal(args.log))
    # Per-segment goal override (kRandom multi-goal renders): the ghost is
    # anchored geometry, so a multi-goal log is rendered segment-by-segment,
    # each with its goal injected here.
    if args.goal_xy is not None:
        task_cfg["goal_xy"] = [float(v) for v in args.goal_xy.split(",")]
        print(f"[render-log-drake] goal override: xy={task_cfg['goal_xy']}")
    if args.goal_yaw is not None:
        task_cfg["goal_yaw"] = float(args.goal_yaw)
        task_cfg.pop("goal_quat", None)
        print(f"[render-log-drake] goal override: yaw={args.goal_yaw}")

    # --- goal segments: the ghost is ANCHORED world geometry, so every
    # kRandom re-goal needs an env rebuild with that segment's goal pose.
    # Single-goal logs collapse to one segment (previous behavior).
    seg_cfgs = [dict(task_cfg)]
    seg_starts = [valid[0]]
    for rg in parse_log_regoals(args.log):
        b = next((k for k in valid
                  if frames[k].get("sim_t", -1.0) >= rg["t"]), None)
        if b is None:
            continue
        cfg = dict(task_cfg)
        cfg["goal_xy"] = rg["xy"]
        cfg["goal_quat"] = rg["quat"]
        seg_cfgs.append(cfg)
        seg_starts.append(b)
    if len(seg_starts) > 1:
        print(f"[render-log-drake] {len(seg_starts)} goal segments "
              f"starting at steps {seg_starts}", flush=True)

    def _build_scene(cfg):
        print("[render-log-drake] building Drake env (add_camera=True, "
              f"goal_xy={cfg.get('goal_xy')})", flush=True)
        diagram, plant, panda_model, _obj, _mc, _pad, _cad, _vw = \
            build_environment(cfg, add_camera=True)
        simulator = ad.Simulator(diagram)
        context = simulator.get_mutable_context()
        plant_ctx = plant.GetMyContextFromRoot(context)
        drake_cam = diagram.GetSubsystemByName("drake_render_camera")
        obj_body = plant.GetBodyByName(cfg["link_name"])
        ee_frame = plant.GetFrameByName(EE_BODY_NAME)
        n_arm = plant.num_actuators()
        q_lo = plant.GetPositionLowerLimits()[:n_arm]
        q_hi = plant.GetPositionUpperLimits()[:n_arm]
        # Seed the arm with the standard INITIAL_ARM_Q so the first IK
        # pass starts from a legal home pose.
        plant.SetPositions(plant_ctx, panda_model, INITIAL_ARM_Q)
        return (simulator, context, plant, panda_model, plant_ctx,
                drake_cam, obj_body, ee_frame, n_arm, q_lo, q_hi)

    # Timeline CSV — one row per rendered frame; paint_mode_text.py looks up
    # frame_NNNNNN.png's step in this file to pick the banner label.
    timeline_path = args.out_dir / "mode_timeline.csv"
    tl_fp = open(timeline_path, "w", buffering=1)
    tl_fp.write("step,sim_t,mode,switch\n")

    def _slerp(q0, q1, a):
        q0 = q0 / np.linalg.norm(q0)
        q1 = q1 / np.linalg.norm(q1)
        if float(np.dot(q0, q1)) < 0.0:
            q1 = -q1
        d = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
        if d > 0.9995:
            q = (1.0 - a) * q0 + a * q1
            return q / np.linalg.norm(q)
        th = np.arccos(d)
        return (np.sin((1.0 - a) * th) * q0 + np.sin(a * th) * q1) \
            / np.sin(th)

    prev_arm_q = INITIAL_ARM_Q.copy()
    kept = 0
    ik_failures = 0
    seg_idx = -1
    K = max(1, args.interp)
    rendered = [k for k in valid if k % args.stride == 0]
    for r_i, step in enumerate(rendered):
        _entered_new_seg = False
        while (seg_idx + 1 < len(seg_starts)
               and step >= seg_starts[seg_idx + 1]):
            seg_idx += 1
            _entered_new_seg = True
        if _entered_new_seg:
            (_sim, context, plant, panda_model, plant_ctx, drake_cam,
             obj_body, ee_frame, n_arm_dofs, q_lo_arm, q_hi_arm) = \
                _build_scene(seg_cfgs[seg_idx])
            prev_arm_q = INITIAL_ARM_Q.copy()
        rec = frames[step]
        nxt = frames[rendered[r_i + 1]] if r_i + 1 < len(rendered) else None

        # --interp K emits K sub-frames per log step, poses interpolated
        # toward the next step (SLERP box quat, lerp positions) so a
        # 0.2-0.4 s tripod roll spans enough frames to read as a roll
        # instead of a 2-frame snap. Frame index = step*K + j; painters
        # recover the log step as index // K (pass the same --interp).
        for j in range(K if nxt is not None else 1):
            a = j / float(K)
            if a == 0.0:
                bq, bp, eep = rec["box_q"], rec["box_p"], rec["ee_p"]
            else:
                bq = _slerp(rec["box_q"], nxt["box_q"], a)
                bp = (1.0 - a) * rec["box_p"] + a * nxt["box_p"]
                eep = (1.0 - a) * rec["ee_p"] + a * nxt["ee_p"]
            _n = float(np.linalg.norm(bq))
            if _n < 1e-9:
                bq_n = np.array([1.0, 0.0, 0.0, 0.0])
            else:
                bq_n = bq / _n
            X_WO = ad.RigidTransform(
                ad.Quaternion(float(bq_n[0]), float(bq_n[1]),
                              float(bq_n[2]), float(bq_n[3])),
                np.asarray(bp, dtype=float).tolist(),
            )
            plant.SetFreeBodyPose(plant_ctx, obj_body, X_WO)

            # DLS IK so the pusher sphere sits at the (interpolated) ee
            # position. Warm-start from the previous arm q. 6-DOF with the
            # tool held vertical (identity R — sim/env_builder.py _R_vert,
            # the OSC's own Kp_rot=800 hold): position-only IK leaves a
            # 4-dim null space that drifts across joint branches on long
            # continuous renders (full-600s render: the wrist flipped
            # during the t=440-457s flip burst and stayed wrong for the
            # remaining 140 s). The small home-pull on the seed anchors the
            # remaining 1-DOF elbow redundancy without visible snapping.
            seed_q = 0.98 * prev_arm_q + 0.02 * INITIAL_ARM_Q
            plant.SetPositions(plant_ctx, panda_model, seed_q)
            q_full = plant.GetPositions(plant_ctx).copy()
            q_sol, err, it = solve_ik_to_ee_pos(
                plant, ee_frame, eep, q_full, plant_ctx,
                n_arm_dofs=n_arm_dofs, max_iter=60, damping=0.05,
                q_lo=q_lo_arm, q_hi=q_hi_arm, R_target=_R_VERT,
            )
            if err > 5e-3:
                # Retry once from the standard home pose.
                plant.SetPositions(plant_ctx, panda_model, INITIAL_ARM_Q)
                q_full = plant.GetPositions(plant_ctx).copy()
                q_sol, err, it = solve_ik_to_ee_pos(
                    plant, ee_frame, eep, q_full, plant_ctx,
                    n_arm_dofs=n_arm_dofs, max_iter=120, damping=0.02,
                    q_lo=q_lo_arm, q_hi=q_hi_arm, R_target=_R_VERT,
                )
                if err > 5e-3:
                    ik_failures += 1
            prev_arm_q = q_sol[:n_arm_dofs].copy()

            # Re-set the box pose after IK (solve_ik_to_ee_pos may have
            # written back a slightly perturbed context).
            plant.SetFreeBodyPose(plant_ctx, obj_body, X_WO)

            cam_ctx = drake_cam.GetMyContextFromRoot(context)
            img = drake_cam.color_image_output_port().Eval(cam_ctx)
            arr = np.asarray(img.data, dtype=np.uint8)  # (H, W, 4) RGBA
            fidx = step * K + j
            out_png = args.out_dir / f"frame_{fidx:06d}.png"
            Image.fromarray(arr, mode="RGBA").save(out_png, optimize=False)

            _t = rec["sim_t"] + (0.1 * a if nxt is not None else 0.0)
            tl_fp.write(f"{fidx},{_t:.4f},{rec['mode']},{rec['switch']}\n")

            kept += 1
            if kept % 50 == 0:
                print(f"[render-log-drake] step={step:04d} "
                      f"ik_err={err*1000:.2f}mm frames={kept} "
                      f"ik_fail_far={ik_failures}", flush=True)

    tl_fp.close()
    print(f"[render-log-drake] wrote {kept} frames + {timeline_path}",
          flush=True)
    print(f"[render-log-drake] IK failures (err>5mm): {ik_failures}/{kept}",
          flush=True)


if __name__ == "__main__":
    main()
