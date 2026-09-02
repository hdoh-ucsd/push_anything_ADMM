#!/usr/bin/env python3
"""Render an OIM rollout with its real xArm, object, and scene geometry."""

from __future__ import annotations

import argparse
import bisect
from collections import Counter
import csv
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.oim_xarm6_tabletop import (
    build_oim_tabletop_scene, set_oim_free_body_pose,
    set_oim_start_configuration,
)


def font(size: int):
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    return ImageFont.truetype(path, size)


def set_record(scene, context, record):
    plant_context = scene.plant.GetMyMutableContextFromRoot(context)
    q = scene.plant.GetPositions(plant_context).copy()
    arm = np.asarray(record["arm_positions"], dtype=float)
    if arm.shape != (6,):
        raise ValueError("arm_positions must contain six native xArm joints")
    for i, value in enumerate(arm, 1):
        q[scene.arm_positions[f"xarm6_joint{i}"]] = value
    scene.plant.SetPositions(plant_context, q)

    pose = np.asarray(record["object_pose"], dtype=float)
    body = scene.plant.GetBodyByName("block")
    if body.is_floating_base_body():
        set_oim_free_body_pose(scene, context, "block", pose)
    else:
        q = scene.plant.GetPositions(plant_context).copy()
        for name, value in (("T_x", pose[0]), ("T_y", pose[1]),
                            ("T_z", pose[2])):
            q[scene.plant.GetJointByName(name).position_start()] = value
        scene.plant.SetPositions(plant_context, q)

    goal = np.asarray(record["goal_pose"], dtype=float)
    goal_body = scene.plant.GetBodyByName("goal")
    if goal_body.is_floating_base_body():
        set_oim_free_body_pose(scene, context, "goal", goal)


def load_control_log(path):
    joint_names = [f"xarm6_joint{i}" for i in range(1, 7)]
    rows = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            rows.append({
                "time": float(row["time_s"]),
                "qdot": [float(row[f"qdot_command_{name}"])
                         for name in joint_names],
                "tau": [float(row[f"tau_command_{name}"])
                        for name in joint_names],
            })
    if not rows:
        raise ValueError(f"control log has no data rows: {path}")
    return rows


def load_planner_phases(path):
    pattern = re.compile(
        r"full_sampling_c3plus_measurement_refresh=PASS phase=(\S+).*"
        r"latest_utime=(\d+)"
    )
    phases = []
    for line in path.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            phases.append((int(match.group(2)) / 1e6, match.group(1)))
    return phases


def load_planner_summary(path):
    admission_pattern = re.compile(
        r"full_sampling_c3plus_initial_candidate_admission=(\S+).*"
        r"dynamic_candidates=(\d+).*lateral_safe_candidates=(\d+).*"
        r"acquisition_feasible_candidates=(\d+).*"
        r"admitted_candidates=(\d+)"
    )
    outcome_pattern = re.compile(
        r"full_sampling_c3plus_acceptance_gate=(\S+).*reason=(\S+?)"
        r"(?:\s.*?updates=(\d+) budget=(\d+))?$"
    )
    terminal_pattern = re.compile(
        r"full_sampling_c3plus_open_table_terminal=(\S+)"
        r".*?translation_error_m=([\d.eE+-]+)"
        r".*?orientation_error_rad=([\d.eE+-]+)"
    )
    latest_time = None
    latest_time_pattern = re.compile(r"latest_utime=(\d+)")
    summary = {}
    for line in path.read_text(errors="replace").splitlines():
        latest_time_match = latest_time_pattern.search(line)
        if latest_time_match:
            latest_time = int(latest_time_match.group(1)) / 1e6
        admission = admission_pattern.search(line)
        if admission:
            summary["admission"] = {
                "status": admission.group(1),
                "dynamic": int(admission.group(2)),
                "lateral": int(admission.group(3)),
                "acquisition": int(admission.group(4)),
                "admitted": int(admission.group(5)),
            }
        terminal = terminal_pattern.search(line)
        if terminal:
            summary["terminal"] = {
                "success": terminal.group(1) == "PASS",
                "translation_error_m": float(terminal.group(2)),
                "orientation_error_rad": float(terminal.group(3)),
            }
        outcome = outcome_pattern.search(line)
        if outcome:
            summary["outcome"] = {
                "status": outcome.group(1),
                "reason": outcome.group(2),
                "updates": (int(outcome.group(3))
                            if outcome.group(3) is not None else None),
                "budget": (int(outcome.group(4))
                           if outcome.group(4) is not None else None),
                "sim_time": latest_time,
            }
    return summary


def load_planner_updates(path):
    """Returns time-aligned full-execution updates and the requested budget."""
    latest_time_pattern = re.compile(r"latest_utime=(\d+)")
    update_pattern = re.compile(r"(?:^| )updates=(\d+)(?: |$)")
    budget_pattern = re.compile(r"(?:^| )budget=(\d+)(?: |$)")
    latest_time = None
    budget = None
    observations = []
    for line in path.read_text(errors="replace").splitlines():
        latest_time_match = latest_time_pattern.search(line)
        if latest_time_match:
            latest_time = int(latest_time_match.group(1)) / 1e6
        budget_match = budget_pattern.search(line)
        if budget_match:
            budget = int(budget_match.group(1))
        update_match = update_pattern.search(line)
        if update_match and latest_time is not None:
            observations.append((latest_time, int(update_match.group(1))))

    # Multiple receipts may share one state timestamp. Keep the greatest
    # monotonic update count at that timestamp so a diagnostic sub-counter can
    # never make the displayed full-execution counter move backward.
    updates_by_time = {}
    running_max = 0
    for sim_time, update in sorted(observations):
        running_max = max(running_max, update)
        updates_by_time[sim_time] = running_max
    return sorted(updates_by_time.items()), budget


def load_contact_intervals(path):
    begin_pattern = re.compile(
        r"xarm_t_physical_contact_begin=PASS episode=(\d+) "
        r"sim_time_s=([\d.eE+-]+) force_N=([\d.eE+-]+)"
    )
    end_pattern = re.compile(
        r"xarm_t_physical_contact_end episode=(\d+) "
        r"sim_time_s=([\d.eE+-]+)"
    )
    contacts = {}
    for line in path.read_text(errors="replace").splitlines():
        begin = begin_pattern.search(line)
        if begin:
            episode = int(begin.group(1))
            contacts[episode] = {
                "episode": episode,
                "begin": float(begin.group(2)),
                "end": float("inf"),
                "force": float(begin.group(3)),
            }
        end = end_pattern.search(line)
        if end and int(end.group(1)) in contacts:
            contacts[int(end.group(1))]["end"] = float(end.group(2))
    return [contacts[key] for key in sorted(contacts)]


def load_contact_samples(path):
    if not path.is_file():
        return []
    samples = [json.loads(line) for line in path.read_text().splitlines()
               if line.strip()]
    samples.sort(key=lambda item: float(item["sim_time"]))
    return samples


def load_contact_transactions(path):
    number = r"([\d.eE+-]+)"
    pattern = re.compile(
        r"full_sampling_c3plus_contact_transaction=ACTIVE "
        r"latest_utime=(\d+) phase=(\S+) sample=(\S+) "
        r"boundary_W_x=" + number + r" boundary_W_y=" + number +
        r" normal_W_x=" + number + r" normal_W_y=" + number +
        r" predicted_prefix_owner=(\d+) geometric_contact=(\d+) "
        r"dwell_owner=(\d+) dwell_steps=(\d+) signed_gap_m=" + number +
        r" tip_hold_error_m=" + number
    )
    transactions = []
    for line in path.read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            transactions.append({
                "sim_time": int(match.group(1)) / 1e6,
                "phase": match.group(2),
                "sample": match.group(3),
                "boundary_W": [float(match.group(4)), float(match.group(5))],
                "normal_W": [float(match.group(6)), float(match.group(7))],
                "predicted_prefix_owner": bool(int(match.group(8))),
                "geometric_contact": bool(int(match.group(9))),
                "dwell_owner": bool(int(match.group(10))),
                "dwell_steps": int(match.group(11)),
                "signed_gap_m": float(match.group(12)),
                "tip_hold_error_m": float(match.group(13)),
            })
    transactions.sort(key=lambda item: item["sim_time"])
    return transactions


def infer_transaction_match_window(transactions, contact_period):
    unique_times = sorted(set(item["sim_time"] for item in transactions))
    deltas = np.diff(unique_times)
    deltas = deltas[deltas > 0.0]
    if deltas.size == 0:
        return 0.0
    return 0.5 * float(np.median(deltas)) + float(contact_period)


def join_contact_transactions(contact_samples, transactions, pusher_radius,
                              activation_tolerance, contact_period):
    if not transactions:
        for contact in contact_samples:
            contact["_join"] = {
                "accepted_contact": False,
                "dwell_owned_contact": False,
                "reason": "no_controller_transaction",
            }
        return
    transaction_times = [item["sim_time"] for item in transactions]
    match_window = infer_transaction_match_window(
        transactions, contact_period)
    face_limit = 2.0 * pusher_radius + activation_tolerance
    for contact in contact_samples:
        contact_time = float(contact["sim_time"])
        index = bisect.bisect_left(transaction_times, contact_time)
        candidates = [i for i in (index - 1, index)
                      if 0 <= i < len(transactions)]
        if not candidates:
            contact["_join"] = {
                "accepted_contact": False,
                "dwell_owned_contact": False,
                "reason": "no_controller_transaction",
            }
            continue
        index = min(candidates,
                    key=lambda i: abs(transaction_times[i] - contact_time))
        transaction = transactions[index]
        time_error = abs(transaction["sim_time"] - contact_time)
        if time_error > match_window:
            contact["_join"] = {
                "accepted_contact": False,
                "dwell_owned_contact": False,
                "reason": "transaction_time_mismatch",
                "time_error_s": time_error,
            }
            continue
        point = np.asarray(contact["point_W"], dtype=float)[:2]
        boundary = np.asarray(transaction["boundary_W"], dtype=float)
        normal = np.asarray(transaction["normal_W"], dtype=float)
        face_point_error = float(np.linalg.norm(point - boundary))
        face_identity = face_point_error <= face_limit
        force = contact.get("force_on_object_W")
        normal_force = None
        normal_polarity = False
        if force is not None:
            normal_force = float(-normal.dot(np.asarray(force)[:2]))
            normal_polarity = normal_force > 0.0
        tip_side = contact.get("classification") == "tip_side_candidate"
        accepted = (tip_side and face_identity and normal_polarity and
                    transaction["predicted_prefix_owner"] and
                    transaction["geometric_contact"])
        checks = (
            (tip_side, "not_tip_side"),
            (face_identity, "selected_face_mismatch"),
            (force is not None, "missing_force_vector"),
            (normal_polarity, "wrong_force_polarity"),
            (transaction["predicted_prefix_owner"], "no_prefix_owner"),
            (transaction["geometric_contact"], "no_geometric_owner"),
        )
        reason = next((reason for passed, reason in checks if not passed),
                      "accepted")
        contact["_join"] = {
            "accepted_contact": accepted,
            "dwell_owned_contact": accepted and transaction["dwell_owner"],
            "reason": reason,
            "time_error_s": time_error,
            "face_point_error_m": face_point_error,
            "face_limit_m": face_limit,
            "normal_force_on_object_N": normal_force,
            "transaction": transaction,
        }


def contact_join_summary(contact_samples):
    reasons = Counter(contact.get("_join", {}).get("reason", "not_joined")
                      for contact in contact_samples)
    controller_dwell_claims = sum(
        bool(contact.get("_join", {}).get("transaction", {}).get(
            "dwell_owner")) for contact in contact_samples)
    dwell_owned = sum(
        bool(contact.get("_join", {}).get("dwell_owned_contact"))
        for contact in contact_samples)
    return {
        "raw_contact_observations": len(contact_samples),
        "accepted_contact_observations": sum(
            bool(contact.get("_join", {}).get("accepted_contact"))
            for contact in contact_samples),
        "controller_dwell_claim_observations": controller_dwell_claims,
        "dwell_owned_contact_observations": dwell_owned,
        "rejected_controller_dwell_claim_observations":
            controller_dwell_claims - dwell_owned,
        "reasons": dict(sorted(reasons.items())),
    }


def select_records(records, stride, contact_samples, contact_detail):
    if stride <= 0:
        raise ValueError("stride must be positive")
    indices = set(range(0, len(records), stride))
    indices.add(len(records) - 1)
    if contact_detail and contact_samples:
        record_times = [float(record["sim_time"]) for record in records]
        for contact in contact_samples:
            contact_time = float(contact["sim_time"])
            index = bisect.bisect_left(record_times, contact_time)
            if index == len(records):
                index -= 1
            elif (index > 0 and
                  contact_time - record_times[index - 1] <=
                  record_times[index] - contact_time):
                index -= 1
            indices.update(i for i in (index - 1, index, index + 1)
                           if 0 <= i < len(records))
    return [records[index] for index in sorted(indices)]


def clip_records_at_or_after(records, stop_time):
    """Retains the first telemetry sample at/after an external terminal time."""
    if stop_time is None:
        return records
    record_times = [float(record["sim_time"]) for record in records]
    stop_index = bisect.bisect_left(record_times, float(stop_time))
    stop_index = min(stop_index, len(records) - 1)
    return records[:stop_index + 1]


def add_log_context(records, control_rows, phases, contacts, contact_samples,
                    planner_summary, contact_match_window,
                    planner_updates=None, planner_budget=None):
    control_times = [row["time"] for row in control_rows]
    phase_times = [item[0] for item in phases]
    planner_updates = planner_updates or []
    update_times = [item[0] for item in planner_updates]
    contact_times = [float(item["sim_time"]) for item in contact_samples]
    for record in records:
        sim_time = float(record["sim_time"])
        control_index = bisect.bisect_left(control_times, sim_time)
        if control_index == len(control_rows):
            control_index -= 1
        elif (control_index > 0 and
              sim_time - control_times[control_index - 1] <=
              control_times[control_index] - sim_time):
            control_index -= 1
        record["_control"] = control_rows[control_index]
        phase_index = bisect.bisect_right(phase_times, sim_time) - 1
        record["_planner_phase"] = (
            phases[phase_index][1] if phase_index >= 0 else "initialization"
        )
        record["_planner_summary"] = planner_summary
        update_index = bisect.bisect_right(update_times, sim_time) - 1
        if update_index < 0:
            record["_controller_update"] = 0
        elif update_index + 1 < len(planner_updates):
            # Long dwell and acquisition stretches emit no updates= receipts,
            # and the counter genuinely advances about one update per
            # execution period through them, so interpolate between the
            # bracketing observations instead of holding a stale count for
            # tens of seconds.
            t0, u0 = planner_updates[update_index]
            t1, u1 = planner_updates[update_index + 1]
            alpha = 0.0 if t1 <= t0 else (sim_time - t0) / (t1 - t0)
            record["_controller_update"] = int(round(
                u0 + min(max(alpha, 0.0), 1.0) * (u1 - u0)))
        else:
            record["_controller_update"] = planner_updates[update_index][1]
        record["_controller_budget"] = planner_budget
        record["_contact"] = None
        if contact_samples:
            contact_index = bisect.bisect_left(contact_times, sim_time)
            candidates = [index for index in
                          (contact_index - 1, contact_index)
                          if 0 <= index < len(contact_samples)]
            if candidates:
                contact_index = min(
                    candidates,
                    key=lambda index: abs(contact_times[index] - sim_time),
                )
                if abs(contact_times[contact_index] - sim_time) <= \
                        contact_match_window:
                    record["_contact"] = contact_samples[contact_index]
        else:
            record["_contact"] = next(
                (contact for contact in contacts
                 if contact["begin"] <= sim_time <= contact["end"]),
                None,
            )


CAMERA_EYE_W = np.array([1.05, -0.72, 0.68])
CAMERA_TARGET_W = np.array([0.40, 0.0, 0.04])
CAMERA_FOV_Y = np.deg2rad(50.0)


def project_world_point(point_W, width, height):
    forward = CAMERA_TARGET_W - CAMERA_EYE_W
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation_WC = np.column_stack((right, down, forward))
    point_C = rotation_WC.T @ (np.asarray(point_W) - CAMERA_EYE_W)
    if point_C[2] <= 0.0:
        return None
    focal = 0.5 * height / np.tan(0.5 * CAMERA_FOV_Y)
    pixel = (width / 2.0 + focal * point_C[0] / point_C[2],
             height / 2.0 + focal * point_C[1] / point_C[2])
    if not (-20 <= pixel[0] < width + 20 and
            -20 <= pixel[1] < height + 20):
        return None
    return tuple(round(value) for value in pixel)


def build_trial_record(run_dir, planner_summary):
    """One trial row for the ledger, from a run's parsed planner receipts."""
    terminal = planner_summary.get("terminal")
    outcome = planner_summary.get("outcome")
    if terminal is None or outcome is None:
        raise SystemExit(
            "run has no terminal/acceptance receipts; cannot build a trial")
    success = bool(terminal["success"])
    budget = outcome.get("budget")
    updates = outcome.get("updates")
    return {
        "run": Path(run_dir).name,
        "success": success,
        "pos_err": float(terminal["translation_error_m"]),
        "theta_err": float(terminal["orientation_error_rad"]),
        "execution_time": float(outcome.get("sim_time") or 0.0),
        # steps-to-goal is censored at the budget for unsuccessful trials.
        "steps_to_goal": int(updates if success and updates is not None
                             else (budget or updates or 0)),
    }


def update_trial_ledger(ledger_path, trial):
    ledger_path = Path(ledger_path)
    trials = (json.loads(ledger_path.read_text())
              if ledger_path.exists() else [])
    trials = [t for t in trials if t.get("run") != trial["run"]]
    trials.append(trial)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(trials, indent=2, sort_keys=True))
    return trials


def aggregate_trial_stats(trials):
    """Aggregate ledger trials into the requested result schema."""
    def moments(values):
        if not values:
            return {"mean": None, "std": None}
        data = np.asarray(values, dtype=float)
        return {"mean": float(data.mean()), "std": float(data.std())}

    successes = [t for t in trials if t["success"]]
    pos_err = moments([t["pos_err"] for t in trials])
    pos_err_s = moments([t["pos_err"] for t in successes])
    theta_err = moments([t["theta_err"] for t in trials])
    theta_err_s = moments([t["theta_err"] for t in successes])
    exec_time = moments([t["execution_time"] for t in trials])
    steps_goal = moments([t["steps_to_goal"] for t in trials])
    steps_goal_s = moments([t["steps_to_goal"] for t in successes])
    result = {
        "n_trials": len(trials),
        "success_rate": len(successes) / len(trials),
        "pos_err_mean": pos_err["mean"],
        "pos_err_std": pos_err["std"],
        "pos_err_mean_success": pos_err_s["mean"],
        "pos_err_std_success": pos_err_s["std"],
        "theta_err_mean": theta_err["mean"],
        "theta_err_std": theta_err["std"],
        "theta_err_mean_success": theta_err_s["mean"],
        "theta_err_std_success": theta_err_s["std"],
        "mean_execution_time": exec_time["mean"],
        "std_execution_time": exec_time["std"],
        "mean_steps_to_goal": steps_goal["mean"],
        "std_steps_to_goal": steps_goal["std"],
        "mean_steps_to_goal_success": steps_goal_s["mean"],
    }
    return result


def add_trial_stats_overlay(view, stats):
    """Draw the aggregate result dict in the camera pane's lower-left."""
    def fmt(value):
        if value is None:
            return "n/a"
        if isinstance(value, int):
            return str(value)
        return f"{value:.4f}"

    keys = list(stats.keys())
    columns = (keys[:8], keys[8:])
    overlay = Image.new("RGBA", view.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    header_font = font(15)
    row_font = font(13)
    row_height = 17
    box_top = view.height - (24 + row_height * 8 + 14)
    draw.rectangle((6, box_top, 6 + 640, view.height - 6),
                   fill=(7, 16, 24, 215), outline=(51, 65, 85, 255))
    draw.text((16, box_top + 6), "result", font=header_font,
              fill=(229, 244, 255, 255))
    for column_index, column in enumerate(columns):
        x = 16 + column_index * 320
        for row_index, key in enumerate(column):
            y = box_top + 26 + row_index * row_height
            draw.text((x, y), f"{key}:", font=row_font,
                      fill=(147, 197, 253, 255))
            draw.text((x + 218, y), fmt(stats[key]), font=row_font,
                      fill=(219, 234, 254, 255))
    return Image.alpha_composite(view.convert("RGBA"), overlay).convert("RGB")


def add_contact_overlay(view, record):
    overlay = Image.new("RGBA", view.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    start = record.get("capsule_start_W")
    end = record.get("capsule_end_W")
    tip = record.get("tip_position")
    if start is not None and end is not None:
        p0 = project_world_point(start, view.width, view.height)
        p1 = project_world_point(end, view.width, view.height)
        if p0 is not None and p1 is not None:
            draw.line((p0, p1), fill=(167, 139, 250, 150), width=9)
    if tip is not None:
        p_tip = project_world_point(tip, view.width, view.height)
        if p_tip is not None:
            draw.ellipse((p_tip[0] - 6, p_tip[1] - 6,
                          p_tip[0] + 6, p_tip[1] + 6),
                         outline=(250, 204, 21, 255), width=3)
    contact = record.get("_contact")
    if contact is not None and "point_W" in contact:
        p_contact = project_world_point(
            contact["point_W"], view.width, view.height)
        if p_contact is not None:
            contact_class = contact.get("classification", "raw_contact")
            join = contact.get("_join", {})
            accepted = join.get("accepted_contact", False)
            explicit_rejection = join.get("reason") not in (
                None, "no_controller_transaction", "transaction_time_mismatch")
            color = ((34, 197, 94, 255) if accepted else
                     (244, 63, 94, 255) if explicit_rejection else
                     (250, 204, 21, 255)
                     if contact_class == "tip_side_candidate" else
                     (244, 63, 94, 255))
            draw.ellipse((p_contact[0] - 9, p_contact[1] - 9,
                          p_contact[0] + 9, p_contact[1] + 9),
                         outline=color, width=4)
            draw.line((p_contact[0] - 12, p_contact[1],
                       p_contact[0] + 12, p_contact[1]), fill=color, width=2)
            draw.line((p_contact[0], p_contact[1] - 12,
                       p_contact[0], p_contact[1] + 12), fill=color, width=2)
    return Image.alpha_composite(view.convert("RGBA"), overlay).convert("RGB")


def panel(record, summary, height):
    width = 560
    image = Image.new("RGB", (width, height), "#071018")
    draw = ImageDraw.Draw(image)
    title = font(23)
    text = font(16)
    small = font(14)
    x = 24
    draw.text((x, 24), "OIM xARM6 TELEMETRY", font=title, fill="#e5f4ff")
    draw.text((x, 62), f"scene: {summary['scene']}", font=text, fill="#93c5fd")
    draw.text((x, 88), "robot: xArm6 (6 actuated joints)", font=text,
              fill="#93c5fd")
    draw.line((x, 122, width - x, 122), fill="#334155", width=2)
    pos = float(record["position_error"])
    theta = float(record["orientation_error"])
    success = pos <= 0.05 and theta <= 0.10
    planner_summary = record.get("_planner_summary", {})
    planner_outcome = planner_summary.get("outcome")
    planner_failed = (
        planner_outcome is not None and
        planner_outcome["status"] == "FAIL" and
        planner_outcome.get("sim_time") is not None and
        float(record["sim_time"]) >= planner_outcome["sim_time"]
    )
    status = ("GOAL REACHED" if success else
              "TERMINAL GATE FAILED" if planner_failed else "RUNNING")
    draw.text((x, 145), status, font=title,
              fill=("#22c55e" if success else
                    "#f87171" if planner_failed else "#f59e0b"))
    controller_update = int(record.get("_controller_update", 0))
    controller_budget = record.get("_controller_budget")
    update_text = (f"{controller_update:6d} / {controller_budget}"
                   if controller_budget is not None
                   else f"{controller_update:6d}")
    fields = [
        f"telemetry sample  {record['step']:6d}",
        f"controller update {update_text}",
        f"simulation time   {record['sim_time']:6.2f} s",
        "",
        f"position error    {pos:7.4f} m",
        "position limit     0.0500 m",
        f"orientation error {theta:7.4f} rad",
        "orientation limit  0.1000 rad",
        "",
        "object [x y yaw]",
        "  " + " ".join(f"{v:+.3f}" for v in record["object_pose"]),
        "goal [x y yaw]",
        "  " + " ".join(f"{v:+.3f}" for v in record["goal_pose"]),
        "",
        "xArm joints [rad]",
    ]
    y = 185
    for line in fields:
        draw.text((x, y), line, font=text, fill="#dbeafe")
        y += 20
    for first in (0, 3):
        values = record["arm_positions"][first:first + 3]
        draw.text((x, y), "  " + " ".join(f"{v:+.3f}" for v in values),
                  font=small, fill="#a7f3d0")
        y += 19
    control = record.get("_control")
    if control is not None:
        phase = record.get("_planner_phase", "unknown")
        contact = record.get("_contact")
        if contact is not None:
            phase = contact.get("_join", {}).get("transaction", {}).get(
                "phase", phase)
        contact_text = "none"
        contact_color = "#94a3b8"
        if contact is not None:
            force = contact.get("force_N", contact.get("force", 0.0))
            contact_class = contact.get("classification", "legacy_latched")
            contact_text = (f"ep {contact['episode']} {force:.2f} N "
                            f"{contact_class}")
            join = contact.get("_join", {})
            accepted = join.get("accepted_contact", False)
            explicit_rejection = join.get("reason") not in (
                None, "no_controller_transaction", "transaction_time_mismatch")
            contact_color = ("#4ade80" if accepted else
                             "#fb7185" if explicit_rejection else
                             "#facc15" if
                             contact_class == "tip_side_candidate" else
                             "#fb7185")
        draw.text((x, y), f"planner phase: {phase[:34]}", font=small,
                  fill="#fcd34d")
        y += 21
        admission = planner_summary.get("admission")
        if admission is not None:
            draw.text(
                (x, y),
                ("admission dyn/lat/ik/pass: "
                 f"{admission['dynamic']}/{admission['lateral']}/"
                 f"{admission['acquisition']}/{admission['admitted']}"),
                font=small, fill="#fca5a5")
            y += 21
        outcome = planner_summary.get("outcome")
        if outcome is not None:
            display_reason = outcome["reason"].replace(
                "initial_candidate_admission_failed",
                "initial_admission_failed")
            draw.text((x, y),
                      f"planner result: {outcome['status']} "
                      f"{display_reason[:28]}",
                      font=small,
                      fill=("#86efac" if outcome["status"] == "PASS"
                            else "#fca5a5"))
            y += 21
        draw.text((x, y), f"raw contact: {contact_text[:43]}", font=small,
                  fill=contact_color)
        y += 21
        if contact is not None and "relative_height_m" in contact:
            draw.text(
                (x, y),
                ("  height/tip distance: "
                 f"{contact['relative_height_m']:+.4f}/"
                 f"{contact['point_to_tip_m']:.4f} m"),
                font=small, fill=contact_color)
            y += 21
        if contact is not None and "_join" in contact:
            join = contact["_join"]
            transaction = join.get("transaction", {})
            join_text = ("accepted" if join["accepted_contact"] else
                         join["reason"])
            controller_dwell = bool(transaction.get("dwell_owner", False))
            draw.text(
                (x, y),
                f"  face join: {join_text[:31]}",
                font=small, fill=contact_color)
            y += 21
            draw.text(
                (x, y),
                (f"  sample={transaction.get('sample', 'none')[:23]} "
                 f"ctrl/join dwell={int(controller_dwell)}/"
                 f"{int(join['dwell_owned_contact'])}"),
                font=small, fill=contact_color)
            y += 21
        draw.text((x, y), "qdot cmd " +
                  " ".join(f"{v:+.2f}" for v in control["qdot"]),
                  font=small, fill="#c4b5fd")
        y += 21
        draw.text((x, y), "tau cmd  " +
                  " ".join(f"{v:+.1f}" for v in control["tau"]),
                  font=small, fill="#67e8f9")
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--control-log", type=Path)
    parser.add_argument("--planner-log", type=Path)
    parser.add_argument("--sim-log", type=Path)
    parser.add_argument("--contact-join-report", type=Path)
    parser.add_argument(
        "--stop-at-planner-outcome", action="store_true",
        help=("end at the first telemetry sample at or after the "
              "timestamped planner acceptance outcome"),
    )
    parser.add_argument(
        "--no-contact-detail", dest="contact_detail", action="store_false",
        help="do not retain extra state frames around high-rate contact samples",
    )
    parser.add_argument(
        "--trial-stats-ledger", type=Path,
        help=("append this run as one trial to the JSON ledger and render "
              "the aggregate result statistics in the camera pane"),
    )
    parser.set_defaults(contact_detail=True)
    args = parser.parse_args()
    summary = json.loads((args.run_dir / "summary.json").read_text())
    records = [json.loads(line) for line in
               (args.run_dir / "steps.jsonl").read_text().splitlines()]
    if not records:
        raise SystemExit("rollout contains no recorded steps")
    if "arm_positions" not in records[0]:
        raise SystemExit(
            "rollout predates native xArm telemetry; rerun the experiment"
        )
    result_dir = args.run_dir.parent
    control_path = args.control_log or result_dir / "xarm_control.csv"
    planner_path = args.planner_log or result_dir / "planner.log"
    sim_path = args.sim_log or result_dir / "sim.log"
    planner_summary = load_planner_summary(planner_path)
    if args.stop_at_planner_outcome:
        records = clip_records_at_or_after(
            records, planner_summary.get("outcome", {}).get("sim_time"))
    contact_samples = load_contact_samples(args.run_dir / "contacts.jsonl")
    selected = select_records(
        records, args.stride, contact_samples, args.contact_detail)
    transactions = load_contact_transactions(planner_path)
    join_contact_transactions(
        contact_samples, transactions,
        float(summary.get("pusher_radius", 0.00555)),
        float(summary.get("contact_activation_tolerance", 0.003)),
        float(summary.get("contact_period", 0.0)),
    )
    join_summary = contact_join_summary(contact_samples)
    if args.contact_join_report is not None:
        if args.contact_join_report.exists():
            raise SystemExit(
                "refusing to overwrite existing contact-join report: "
                f"{args.contact_join_report}")
        args.contact_join_report.parent.mkdir(parents=True, exist_ok=True)
        args.contact_join_report.write_text(
            json.dumps(join_summary, indent=2, sort_keys=True) + "\n")
    planner_updates, planner_budget = load_planner_updates(planner_path)
    add_log_context(
        selected,
        load_control_log(control_path),
        load_planner_phases(planner_path),
        load_contact_intervals(sim_path),
        contact_samples,
        planner_summary,
        0.5 * float(summary.get("record_period", 0.1)) +
        float(summary.get("contact_period", 0.0)),
        planner_updates,
        planner_budget,
    )

    scene = build_oim_tabletop_scene(
        summary["scene"], sampling_c3plus_object=summary["scene"] == "open_table",
        add_camera=True,
    )
    context = scene.diagram.CreateDefaultContext()
    set_oim_start_configuration(scene, context)
    camera_context = scene.render_camera.GetMyContextFromRoot(context)
    output = args.output or args.run_dir / "xarm_rollout.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing video: {output}")

    trial_stats = None
    if args.trial_stats_ledger is not None:
        trial = build_trial_record(args.run_dir.parent, planner_summary)
        trials = update_trial_ledger(args.trial_stats_ledger, trial)
        trial_stats = aggregate_trial_stats(trials)
        stats_path = args.run_dir.parent / "trial_stats.json"
        stats_path.write_text(json.dumps(trial_stats, indent=2))
        print("trial stats: " + json.dumps(trial_stats, sort_keys=True))

    with tempfile.TemporaryDirectory(prefix="oim_xarm_frames_") as tmp:
        tmp = Path(tmp)
        for index, record in enumerate(selected):
            set_record(scene, context, record)
            scene.diagram.ForcedPublish(context)
            rgb = scene.render_camera.color_image_output_port().Eval(
                camera_context).data[:, :, :3]
            view = Image.fromarray(rgb).convert("RGB")
            view = add_contact_overlay(view, record)
            if trial_stats is not None:
                view = add_trial_stats_overlay(view, trial_stats)
            side = panel(record, summary, view.height)
            frame = Image.new("RGB", (view.width + side.width, view.height))
            frame.paste(view, (0, 0))
            frame.paste(side, (view.width, 0))
            frame.save(tmp / f"frame_{index:06d}.png")
        subprocess.run([
            "ffmpeg", "-loglevel", "error", "-n", "-framerate", str(args.fps),
            "-i", str(tmp / "frame_%06d.png"), "-c:v", "libx264", "-pix_fmt",
            "yuv420p", str(output),
        ], check=True)
    print(f"wrote {output}: {len(selected)} frames, real OIM xArm geometry")
    print("contact join: " + json.dumps(join_summary, sort_keys=True))


if __name__ == "__main__":
    main()
