#!/usr/bin/env python3
"""Read-only forensic extraction for final-clean obstacle-study failures.

The script never launches a process from the robotics stack and never edits an
existing campaign artifact.  It reads the immutable manifest, receipts, logs,
and forensic CSV/JSONL streams and writes derived tables beneath
``forensic_audit``.
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "results/c3plus_final_clean_obstacle_ranking"
OUT = CAMPAIGN / "forensic_audit"
INVENTORY = OUT / "nonadmitted_failure_inventory.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", errors="replace") as stream:
        return list(csv.DictReader(stream))


def iter_csv(path: Path) -> Iterable[dict[str, str]]:
    with path.open(newline="", errors="replace") as stream:
        yield from csv.DictReader(stream)


def number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def finite(value: Any) -> bool:
    return math.isfinite(number(value))


def quantile(values: list[float], q: float) -> float:
    values = sorted(x for x in values if math.isfinite(x))
    if not values:
        return math.nan
    p = (len(values) - 1) * q
    lo, hi = math.floor(p), math.ceil(p)
    return values[lo] if lo == hi else values[lo] * (hi - p) + values[hi] * (p - lo)


def iso_elapsed(start: str, end: str) -> float:
    try:
        return (datetime.strptime(end, "%Y-%m-%dT%H:%M:%S%z") -
                datetime.strptime(start, "%Y-%m-%dT%H:%M:%S%z")).total_seconds()
    except (TypeError, ValueError):
        return math.nan


def json_numbers_are_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(json_numbers_are_finite(v) for v in value.values())
    if isinstance(value, list):
        return all(json_numbers_are_finite(v) for v in value)
    return True


def last_complete_cycles(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    required = ("time", "obj_x", "obj_y", "obj_z", "obj_yaw", "ee_x", "ee_y", "ee_z")
    for row in iter_csv(path):
        if all(finite(row.get(key)) for key in required):
            rows.append(row)
    return rows


def derivative(last: dict[str, str], previous: dict[str, str], prefix: str) -> tuple[float, ...]:
    dt = number(last["time"]) - number(previous["time"])
    keys = ("x", "y", "z") if prefix in {"obj", "ee"} else ()
    if not keys or dt <= 0:
        return (math.nan, math.nan, math.nan)
    return tuple((number(last[f"{prefix}_{key}"]) - number(previous[f"{prefix}_{key}"])) / dt
                 for key in keys)


def last_selected_candidate(path: Path) -> tuple[dict[str, str] | None, int, int]:
    last: dict[str, str] | None = None
    nonfinite_obstacle = 0
    nonfinite_rank = 0
    for row in iter_csv(path):
        if row.get("selected") != "1" or not finite(row.get("time")):
            continue
        if not finite(row.get("J_obstacle_applied_live")):
            nonfinite_obstacle += 1
        if not finite(row.get("J_rank_code_total")):
            nonfinite_rank += 1
        if last is None or number(row["time"]) >= number(last["time"]):
            last = row
    return last, nonfinite_obstacle, nonfinite_rank


def last_planner_event(text: str) -> str:
    ignored = ("warning]", "Format:", "Error: /usr/lib", "terminate called", "what():")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    meaningful = [line for line in lines if not any(token in line for token in ignored)]
    return meaningful[-1] if meaningful else "UNAVAILABLE"


def workspace_guard(text: str) -> tuple[str, str]:
    if ".cc:4352" in text:
        return "LOWER_CARTESIAN_BOUND", "systems/controllers/sampling_based_c3_controller.cc:4352"
    if ".cc:4355" in text:
        return "UPPER_CARTESIAN_BOUND", "systems/controllers/sampling_based_c3_controller.cc:4355"
    if ".cc:4360" in text:
        return "MINIMUM_PLANAR_RADIUS", "systems/controllers/sampling_based_c3_controller.cc:4360"
    if ".cc:4363" in text:
        return "MAXIMUM_PLANAR_RADIUS", "systems/controllers/sampling_based_c3_controller.cc:4363"
    return "UNKNOWN_WORKSPACE_GUARD", "systems/controllers/sampling_based_c3_controller.cc:4348-4366"


def boundary_metrics(row: dict[str, str]) -> tuple[str, float, float]:
    x, y, z = (number(row[f"ee_{axis}"]) for axis in "xyz")
    radius = math.hypot(x, y)
    margins = {
        "x_lower": x - 0.15, "x_upper": 0.75 - x,
        "y_lower": y + 0.60, "y_upper": 0.60 - y,
        "z_lower": z + 0.024, "z_upper": 0.30 - z,
        "radius_lower": radius - 0.25, "radius_upper": 0.70 - radius,
    }
    closest = min(margins, key=margins.get)
    return closest, margins[closest], radius


def candidate_in_workspace(row: dict[str, str] | None) -> str:
    if row is None:
        return "UNKNOWN"
    x, y, z = (number(row.get(f"candidate_ee_{axis}")) for axis in "xyz")
    if not all(math.isfinite(v) for v in (x, y, z)):
        return "UNKNOWN"
    radius = math.hypot(x, y)
    return str(0.15 < x < 0.75 and -0.60 < y < 0.60 and -0.024 < z < 0.30 and
               0.25 < radius < 0.70).upper()


def load_status_events() -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in (CAMPAIGN / "state/status_events.jsonl").read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[str(event.get("run_id", ""))].append(event)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    attempts = read_csv(CAMPAIGN / "metrics/attempt_inventory.csv")
    terminal = [row for row in attempts
                if row["terminal_for_manifest_entry"] == "1" and
                row["scientifically_admitted"] == "0"]
    if len(terminal) != 75:
        raise RuntimeError(f"expected 75 non-admitted terminal rows, found {len(terminal)}")
    status_events = load_status_events()
    inventory: list[dict[str, Any]] = []

    for entry in terminal:
        attempt = ROOT / entry["attempt_path"]
        receipt = json.loads((attempt / "run_receipt.json").read_text())
        cleanup = json.loads((attempt / "owned_cleanup.json").read_text())
        planner_path = attempt / "live/planner.log"
        planner = planner_path.read_text(errors="replace")
        cycles = last_complete_cycles(attempt / "forensics/forensics_cycle.csv")
        last = cycles[-1]
        previous = cycles[-2] if len(cycles) > 1 else cycles[-1]
        obj_v = derivative(last, previous, "obj")
        ee_v = derivative(last, previous, "ee")
        dt = number(last["time"]) - number(previous["time"])
        yaw_rate = ((number(last["obj_yaw"]) - number(previous["obj_yaw"])) / dt
                    if dt > 0 else math.nan)
        selected, bad_obs, bad_rank = last_selected_candidate(
            attempt / "forensics/forensics_candidates.csv")
        clearances = [number(row["physical_object_obstacle_clearance"]) for row in cycles
                      if finite(row.get("physical_object_obstacle_clearance"))]
        recent = cycles[-min(20, len(cycles)):]
        contact_duty = statistics.mean(number(row["physical_contact"]) for row in recent)
        state_rows = 0
        state_all_finite = True
        with (attempt / "live/state_trace.jsonl").open(errors="replace") as stream:
            for line in stream:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                state_rows += 1
                state_all_finite = state_all_finite and json_numbers_are_finite(obj)
        step_tail: list[dict[str, Any]] = []
        with (attempt / "live/steps_raw.jsonl").open(errors="replace") as stream:
            for line in stream:
                try:
                    step = json.loads(line)
                except json.JSONDecodeError:
                    continue
                step_tail.append(step)
                if len(step_tail) > 100:
                    step_tail.pop(0)
        robot_q = [abs(number(x)) for step in step_tail for x in step.get("robot_q", [])]
        robot_v = [abs(number(x)) for step in step_tail for x in step.get("robot_v", [])]
        robot_u = [abs(number(x)) for step in step_tail for x in step.get("robot_u", [])]
        q_step_deltas = []
        for a, b in zip(step_tail, step_tail[1:]):
            qa, qb = a.get("robot_q", []), b.get("robot_q", [])
            if len(qa) == len(qb):
                q_step_deltas.extend(abs(number(y) - number(x)) for x, y in zip(qa, qb))
        quaternion_norm_errors = []
        for step in step_tail:
            for obj in step.get("objects", {}).values():
                if isinstance(obj, list) and len(obj) >= 4:
                    quaternion_norm_errors.append(abs(math.sqrt(sum(number(x) ** 2 for x in obj[:4])) - 1.0))
        lambda_recent = [number(row.get("planner_pusher_object_lambda_norm")) for row in recent
                         if finite(row.get("planner_pusher_object_lambda_norm"))]
        guard_type, guard_location = workspace_guard(planner)
        closest_boundary, boundary_margin, radius = boundary_metrics(last)
        helper_codes = [event.get("helper_returncode") for event in status_events[entry["run_id"]]
                        if "helper_returncode" in event]
        planner_outcome = (
            "UNCAUGHT_DRAKE_ASSERTION; raw child return code not persisted"
            if entry["terminal_reason_audited"] in {"PLANNER_ABORT_ALLFINITE_QV", "WORKSPACE_ABORT"}
            else "OWNED_SUPERVISOR_TIMEOUT; raw child return code not persisted"
        )
        exact_guard = (
            "Drake MultibodyPlant::SetPositionsAndVelocities AllFinite(q_v), "
            "called by common/update_context.cc:29"
            if entry["terminal_reason_audited"] == "PLANNER_ABORT_ALLFINITE_QV"
            else guard_location if entry["terminal_reason_audited"] == "WORKSPACE_ABORT"
            else "scripts/run_c3plus_geometry_rotation_cost_replication.py:346"
        )
        inventory.append({
            "manifest_id": entry["run_id"], "config": entry["config_id"],
            "scene": entry["scene"], "start": entry["start_id"],
            "rotation": entry["rotation_label"], "seed": entry["seed"],
            "failure_reason": entry["terminal_reason_audited"],
            "failure_timestamp_proxy_planner_log_mtime": datetime.fromtimestamp(
                planner_path.stat().st_mtime).astimezone().isoformat(),
            "simulation_time_last_valid_s": number(last["time"]),
            "wall_time_start_to_cleanup_s": iso_elapsed(receipt.get("started_at", ""), cleanup.get("completed_at", "")),
            "position_error_m": number(last["position_error"]),
            "orientation_error_rad": number(last["yaw_error"]),
            "object_x": number(last["obj_x"]), "object_y": number(last["obj_y"]),
            "object_z": number(last["obj_z"]), "object_yaw": number(last["obj_yaw"]),
            "object_vx_derived": obj_v[0], "object_vy_derived": obj_v[1],
            "object_vz_derived": obj_v[2], "object_yaw_rate_derived": yaw_rate,
            "ee_x": number(last["ee_x"]), "ee_y": number(last["ee_y"]), "ee_z": number(last["ee_z"]),
            "ee_vx_derived": ee_v[0], "ee_vy_derived": ee_v[1], "ee_vz_derived": ee_v[2],
            "selected_candidate": last.get("selected_candidate_id", ""),
            "selected_contact_sector": last.get("selected_contact_sector", ""),
            "mode": last.get("mode_name", ""), "mode_age_cycles": last.get("mode_age_cycles", ""),
            "last_selected_J_obstacle": number(selected.get("J_obstacle_applied_live")) if selected else math.nan,
            "last_selected_J_rank": number(selected.get("J_rank_code_total")) if selected else math.nan,
            "last_selected_rank_reconstruction_valid": selected.get("reconstruction_valid", "") if selected else "",
            "last_selected_min_predicted_clearance_m": number(selected.get("min_pred_obj_obs_clearance")) if selected else math.nan,
            "minimum_physical_clearance_before_failure_m": min(clearances) if clearances else math.nan,
            "previous_logged_controller_cycles": len(cycles),
            "last_valid_planner_event": last_planner_event(planner),
            "last_valid_qp_status": "NOT_EXPLICITLY_LOGGED; last candidate reconstruction=" +
                                    (selected.get("reconstruction_valid", "UNKNOWN") if selected else "UNKNOWN"),
            "recent_20_cycle_contact_duty": contact_duty,
            "transaction_count": sum(1 for _ in iter_csv(attempt / "forensics/forensics_transactions.csv")),
            "planner_process_outcome": planner_outcome,
            "supervisor_helper_returncode": helper_codes[-1] if helper_codes else "NOT_RECORDED",
            "exact_source_guard": exact_guard,
            "external_state_trace_rows": state_rows,
            "external_state_trace_all_finite": state_all_finite,
            "last_100_steps_max_abs_robot_q": max(robot_q) if robot_q else math.nan,
            "last_100_steps_max_abs_robot_v": max(robot_v) if robot_v else math.nan,
            "last_100_steps_max_abs_robot_u": max(robot_u) if robot_u else math.nan,
            "last_100_steps_max_abs_delta_robot_q": max(q_step_deltas) if q_step_deltas else math.nan,
            "last_100_steps_max_object_quaternion_norm_error": (
                max(quaternion_norm_errors) if quaternion_norm_errors else math.nan),
            "recent_20_cycles_max_planner_lambda_norm": max(lambda_recent) if lambda_recent else math.nan,
            "planner_unsuccessful_buffer_overflow_warnings": planner.count("Unsuccessful sample buffer overflow"),
            "planner_no_progress_repositions": planner.count("Repositioning after not making progress in C3"),
            "planner_cost_repositions": planner.count("Repositioning because found good sample"),
            "selected_rows_nonfinite_obstacle_count": bad_obs,
            "selected_rows_nonfinite_rank_count": bad_rank,
            "workspace_guard_type": guard_type if entry["terminal_reason_audited"] == "WORKSPACE_ABORT" else "",
            "closest_workspace_boundary_last_valid": closest_boundary if entry["terminal_reason_audited"] == "WORKSPACE_ABORT" else "",
            "workspace_margin_last_valid_m": boundary_margin if entry["terminal_reason_audited"] == "WORKSPACE_ABORT" else math.nan,
            "ee_planar_radius_last_valid_m": radius,
            "selected_candidate_inside_declared_workspace": candidate_in_workspace(selected),
            "receipt_path": str((attempt / "run_receipt.json").relative_to(ROOT)),
            "trace_path": str((attempt / "live/state_trace.jsonl").relative_to(ROOT)),
            "attempt_path": str(attempt.relative_to(ROOT)),
        })

    write_csv(INVENTORY, inventory)

    manifest = read_csv(CAMPAIGN / "manifest/final_126_run_manifest.csv")
    terminal_by_run = {row["run_id"]: row for row in attempts if row["terminal_for_manifest_entry"] == "1"}
    status: dict[str, str] = {}
    for row in manifest:
        attempt = terminal_by_run[row["run_id"]]
        status[row["run_id"]] = ("ADMITTED" if attempt["scientifically_admitted"] == "1"
                                 else attempt["terminal_reason_audited"])

    rates: list[dict[str, Any]] = []
    for config in ("OPEN-A1", "B1", "B2", "B3"):
        rows = [row for row in manifest if row["config_id"] == config]
        counts = Counter(status[row["run_id"]] for row in rows)
        rates.append({
            "config": config, "planned": len(rows), "admitted": counts["ADMITTED"],
            "allfinite_aborts": counts["PLANNER_ABORT_ALLFINITE_QV"],
            "allfinite_fraction": counts["PLANNER_ABORT_ALLFINITE_QV"] / len(rows),
            "workspace_aborts": counts["WORKSPACE_ABORT"],
            "workspace_fraction": counts["WORKSPACE_ABORT"] / len(rows),
            "supervisor_timeouts": counts["SUPERVISOR_WALL_TIMEOUT"],
            "nonadmitted": len(rows) - counts["ADMITTED"],
            "admission_fraction": counts["ADMITTED"] / len(rows),
        })
    write_csv(OUT / "failure_rate_by_config.csv", rates)

    breakdown: list[dict[str, Any]] = []
    for dimension in ("config_id", "scene", "start_id", "rotation_label", "seed"):
        for value in sorted({row[dimension] for row in manifest}):
            rows = [row for row in manifest if row[dimension] == value]
            counts = Counter(status[row["run_id"]] for row in rows)
            breakdown.append({"dimension": dimension, "value": value, "planned": len(rows),
                              "admitted": counts["ADMITTED"],
                              "allfinite": counts["PLANNER_ABORT_ALLFINITE_QV"],
                              "workspace": counts["WORKSPACE_ABORT"],
                              "timeout": counts["SUPERVISOR_WALL_TIMEOUT"]})
    write_csv(OUT / "failure_breakdown.csv", breakdown)

    matched: dict[tuple[str, str, str, str], dict[str, str]] = defaultdict(dict)
    run_ids: dict[tuple[str, str, str, str], dict[str, str]] = defaultdict(dict)
    for row in manifest:
        if row["config_id"] not in {"B1", "B2", "B3"}:
            continue
        key = (row["scene"], row["start_id"], row["rotation_label"], row["seed"])
        matched[key][row["config_id"]] = status[row["run_id"]]
        run_ids[key][row["config_id"]] = row["run_id"]
    matched_rows: list[dict[str, Any]] = []
    patterns = Counter()
    for key in sorted(matched):
        values = matched[key]
        pattern = (values["B1"], values["B2"], values["B3"])
        patterns[pattern] += 1
        matched_rows.append({"scene": key[0], "start": key[1], "rotation": key[2], "seed": key[3],
                             "B1": values["B1"], "B2": values["B2"], "B3": values["B3"],
                             "B1_run_id": run_ids[key]["B1"], "B2_run_id": run_ids[key]["B2"],
                             "B3_run_id": run_ids[key]["B3"]})
    write_csv(OUT / "matched_task_failure_classifications.csv", matched_rows)
    write_csv(OUT / "matched_failure_pattern_counts.csv",
              [{"B1": key[0], "B2": key[1], "B3": key[2], "count": count}
               for key, count in patterns.most_common()])

    write_csv(OUT / "workspace_abort_details.csv",
              [row for row in inventory if row["failure_reason"] == "WORKSPACE_ABORT"])
    write_csv(OUT / "allfinite_last_valid_event.csv",
              [row for row in inventory if row["failure_reason"] == "PLANNER_ABORT_ALLFINITE_QV"])

    timing_summary: list[dict[str, Any]] = []
    for reason in sorted({row["failure_reason"] for row in inventory}):
        for config in ("ALL", "OPEN-A1", "B1", "B2", "B3"):
            rows = [row for row in inventory if row["failure_reason"] == reason and
                    (config == "ALL" or row["config"] == config)]
            if not rows:
                continue
            values = [number(row["simulation_time_last_valid_s"]) for row in rows]
            timing_summary.append({"failure_reason": reason, "config": config, "N": len(values),
                                   "min_sim_time_s": min(values), "q25_sim_time_s": quantile(values, .25),
                                   "median_sim_time_s": statistics.median(values),
                                   "q75_sim_time_s": quantile(values, .75), "max_sim_time_s": max(values),
                                   "immediate_le_10s": sum(v <= 10 for v in values),
                                   "early_le_60s": sum(v <= 60 for v in values),
                                   "late_gt_300s": sum(v > 300 for v in values)})
    write_csv(OUT / "failure_timing_summary.csv", timing_summary)

    state_summary: list[dict[str, Any]] = []
    summary_fields = ["position_error_m", "orientation_error_rad", "object_vx_derived",
                      "object_vy_derived", "object_yaw_rate_derived", "ee_vx_derived",
                      "ee_vy_derived", "ee_vz_derived", "minimum_physical_clearance_before_failure_m",
                      "recent_20_cycle_contact_duty", "transaction_count"]
    for reason in sorted({row["failure_reason"] for row in inventory}):
        rows = [row for row in inventory if row["failure_reason"] == reason]
        record: dict[str, Any] = {"failure_reason": reason, "N": len(rows)}
        for field in summary_fields:
            values = [number(row[field]) for row in rows if finite(row[field])]
            record[field + "_median"] = statistics.median(values) if values else math.nan
            record[field + "_q25"] = quantile(values, .25)
            record[field + "_q75"] = quantile(values, .75)
        state_summary.append(record)
    write_csv(OUT / "state_at_failure_summary.csv", state_summary)

    missing: list[dict[str, Any]] = []
    for row in matched_rows:
        b1, b2 = row["B1"], row["B2"]
        if b1 == "ADMITTED" and b2 == "ADMITTED":
            action = "PAIR_COMPLETE"
        elif b1 == "PLANNER_ABORT_ALLFINITE_QV" or b2 == "PLANNER_ABORT_ALLFINITE_QV":
            action = "RERUN_ALLFINITE_MEMBER_AFTER_APPROVED_FIX"
        elif b1 == "WORKSPACE_ABORT" or b2 == "WORKSPACE_ABORT":
            action = "RECLASSIFY_WORKSPACE_AS_ALGORITHM_SAFETY_FAILURE_NO_RERUN"
        elif b1 == "SUPERVISOR_WALL_TIMEOUT" or b2 == "SUPERVISOR_WALL_TIMEOUT":
            action = "AUDIT_AS_CAP_TIMEOUT_NO_RERUN_IF_TIMING_RECONSTRUCTED"
        else:
            action = "REVIEW"
        missing.append({**row, "both_admitted": b1 == "ADMITTED" and b2 == "ADMITTED",
                        "recommended_action": action})
    write_csv(OUT / "b1_b2_pair_recovery.csv", missing)

    # Streaming B1 exponential-overflow audit across every recorded candidate
    # knot in the 16 B1 AllFinite-abort runs (~1.3 GB); no rows are retained.
    min_d = math.inf
    finite_knots = 0
    nonfinite_d = 0
    b1_runs = [row for row in terminal if row["config_id"] == "B1" and
               row["terminal_reason_audited"] == "PLANNER_ABORT_ALLFINITE_QV"]
    for entry in b1_runs:
        path = ROOT / entry["attempt_path"] / "forensics/forensics_candidate_knots.csv"
        for row in iter_csv(path):
            d = number(row.get("object_obstacle_signed_clearance"))
            if math.isfinite(d):
                finite_knots += 1
                min_d = min(min_d, d)
            else:
                nonfinite_d += 1
    weight = 764.3857383844456
    sigma = 0.015
    max_argument = -min_d / sigma
    overflow_distance = -sigma * math.log(float.fromhex("0x1.fffffffffffffp1023") / weight)
    overflow = {
        "B1_allfinite_runs": len(b1_runs), "candidate_knots_scanned": finite_knots,
        "nonfinite_distance_rows": nonfinite_d, "minimum_signed_distance_m": min_d,
        "maximum_exp_argument": max_argument,
        "maximum_single_knot_weighted_cost": weight * math.exp(max_argument),
        "double_overflow_requires_distance_below_m": overflow_distance,
        "overflow_mechanism_excluded": min_d > overflow_distance,
    }
    (OUT / "b1_exponential_overflow_audit.json").write_text(json.dumps(overflow, indent=2) + "\n")

    allfinite = [row for row in inventory if row["failure_reason"] == "PLANNER_ABORT_ALLFINITE_QV"]
    workspace = [row for row in inventory if row["failure_reason"] == "WORKSPACE_ABORT"]
    recovery_allfinite_members = sum(
        (row["B1"] == "PLANNER_ABORT_ALLFINITE_QV") +
        (row["B2"] == "PLANNER_ABORT_ALLFINITE_QV") for row in matched_rows)
    summary = {
        "terminal_nonadmitted": len(inventory),
        "failure_counts": Counter(row["failure_reason"] for row in inventory),
        "allfinite_external_traces_all_finite": sum(bool(row["external_state_trace_all_finite"])
                                                     for row in allfinite),
        "allfinite_runs": len(allfinite),
        "allfinite_runs_with_any_nonfinite_selected_obstacle": sum(
            int(row["selected_rows_nonfinite_obstacle_count"]) > 0 for row in allfinite),
        "allfinite_runs_with_any_nonfinite_selected_rank": sum(
            int(row["selected_rows_nonfinite_rank_count"]) > 0 for row in allfinite),
        "workspace_guard_counts": Counter(row["workspace_guard_type"] for row in workspace),
        "workspace_mode_counts": Counter(row["mode"] for row in workspace),
        "workspace_selected_candidate_outside_count": sum(
            row["selected_candidate_inside_declared_workspace"] == "FALSE" for row in workspace),
        "B1_B2_jointly_admitted_pairs": Counter(
            row["scene"] for row in matched_rows if row["B1"] == row["B2"] == "ADMITTED"),
        "B1_B2_missing_pairs": Counter(
            row["scene"] for row in matched_rows if not (row["B1"] == row["B2"] == "ADMITTED")),
        "B1_B2_allfinite_members_requiring_targeted_rerun": recovery_allfinite_members,
        "overflow": overflow,
    }
    (OUT / "forensic_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    try:
        import matplotlib.pyplot as plt
        groups = [("AllFinite", [number(r["simulation_time_last_valid_s"]) for r in allfinite]),
                  ("Workspace", [number(r["simulation_time_last_valid_s"]) for r in workspace])]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist([g[1] for g in groups], bins=range(0, 421, 20), label=[g[0] for g in groups],
                alpha=.72, stacked=False)
        ax.set(xlabel="last valid simulation time (s)", ylabel="terminal runs",
               title="Non-admitted failure-time distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / "failure_time_distribution.png", dpi=160)
        plt.close(fig)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
