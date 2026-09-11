#!/usr/bin/env python3
"""Build a read-only inventory of historical TR07/TR08 obstacle evidence.

The input trees are never modified.  This script only materializes a new audit
artifact under results/c3plus_historical_obstacle_reliability_audit/.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/c3plus_historical_obstacle_reliability_audit"
TASK = ROOT / "results/c3plus_task_cost_ratio"
OBS = ROOT / "results/c3plus_obstacle_cost_tuning_v2"
CAPACITY = ROOT / "results/c3plus_concurrency_capacity"

FIELDS = [
    "source_study", "attempt_id", "task_base", "alpha_p", "alpha_theta",
    "obstacle_config", "obstacle_family", "obstacle_range_m",
    "obstacle_weight", "obstacle_distance_geometry", "soft_rank_cost_active",
    "inner_soft_obstacle_mode", "lcs_contact_active", "scene", "start",
    "goal", "rotation_deg", "seed", "attempt_number", "success_observed",
    "retained_success", "T_goal_s", "best_position_error_m",
    "final_position_error_m", "best_orientation_error_rad",
    "final_orientation_error_rad", "timing_valid", "concurrency",
    "runtime_wall_s", "scientific_status", "terminal_classification",
    "failure_class", "F1", "F2",
    "F3", "CHOMP_normalized", "minimum_clearance_m", "transactions",
    "median_transaction_displacement_m", "physical_contact_fraction",
    "contactless_c3_fraction", "median_J_obstacle_over_J_task",
    "controller_source_sha256", "controller_binary_sha256", "receipt_path",
    "trace_path", "notes",
]


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def scene_identity(scene: str) -> tuple[str, str, str]:
    if scene == "box_clutter":
        return "box_real_start", "box_real_goal", "+90"
    return "s01", "g02", "0"


def flags(failure: str) -> tuple[str, str, str]:
    return (
        "1" if failure.startswith("F1") else "0",
        "1" if failure.startswith("F2") else "0",
        "1" if failure.startswith("F3") else "0",
    )


def trace_summary(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    first_success = None
    best_p = float("inf")
    best_a = float("inf")
    final = None
    with path.open() as stream:
        for line in stream:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            p = float(item.get("pos_err", float("inf")))
            a = float(item.get("ang_err", float("inf")))
            best_p = min(best_p, p)
            best_a = min(best_a, a)
            if first_success is None and p <= 0.05 and a <= 0.1:
                first_success = item.get("t")
            final = item
    if final is None:
        return {}
    return {
        "success_observed": "1" if first_success is not None else "0",
        "T_goal_s": "" if first_success is None else str(first_success),
        "best_position_error_m": str(best_p),
        "final_position_error_m": str(final.get("pos_err", "")),
        "best_orientation_error_rad": str(best_a),
        "final_orientation_error_rad": str(final.get("ang_err", "")),
    }


def main() -> int:
    inventory: list[dict[str, str]] = []

    # Historical task-ratio runs.  The source fallback is configured in YAML,
    # but lcs_contact suppresses it; candidate logs record live obstacle cost 0.
    task_metrics = rows(TASK / "metrics/per_run_results.csv")
    for metric in task_metrics:
        tr = metric["condition_id"]
        if tr not in {"TR07", "TR08"}:
            continue
        scene = metric["scenario"]
        attempt = TASK / "runs" / tr / scene / "attempts/clean_attempt_01"
        receipt = read_json(attempt / "run_receipt.json")
        complete = read_json(attempt / "RUN_COMPLETE.json")
        start, goal, rotation = scene_identity(scene)
        failure = metric["failure_class"] or (
            "SUCCESS" if metric["success"] == "1" else metric["runtime_status"])
        f1, f2, f3 = flags(failure)
        inventory.append({
            "source_study": "c3plus_task_cost_ratio",
            "attempt_id": f"{tr}/BASELINE_SUPPRESSED/{scene}/clean_attempt_01",
            "task_base": tr, "alpha_p": metric["alpha_p"],
            "alpha_theta": metric["alpha_theta"],
            "obstacle_config": "HISTORICAL_BASELINE_SUPPRESSED",
            "obstacle_family": "center_exponential_fallback",
            "obstacle_range_m": "0.04", "obstacle_weight": "5000",
            "obstacle_distance_geometry": "object_center_to_disc_clearance",
            "soft_rank_cost_active": "0", "inner_soft_obstacle_mode": "none",
            "lcs_contact_active": "1", "scene": scene, "start": start,
            "goal": goal, "rotation_deg": rotation, "seed": metric["seed"],
            "attempt_number": "clean_attempt_01",
            "success_observed": metric["success"], "retained_success": "",
            "T_goal_s": metric["T_goal"],
            "best_position_error_m": metric["best_position_error"],
            "final_position_error_m": metric["final_position_error"],
            "best_orientation_error_rad": metric["best_orientation_error"],
            "final_orientation_error_rad": metric["final_orientation_error"],
            "timing_valid": metric["timing_valid"],
            "concurrency": metric["concurrency_level"],
            "runtime_wall_s": str(complete.get("wall_time_s", "")),
            "scientific_status": "ADMITTED",
            "terminal_classification": metric["runtime_status"],
            "failure_class": failure,
            "F1": f1, "F2": f2, "F3": f3,
            "CHOMP_normalized": metric["CHOMP_normalized"],
            "minimum_clearance_m": metric["min_physical_clearance"],
            "transactions": "", "median_transaction_displacement_m": "",
            "physical_contact_fraction": metric["physical_contact_fraction"],
            "contactless_c3_fraction": metric["contactless_c3_fraction"],
            "median_J_obstacle_over_J_task": "0",
            "controller_source_sha256": receipt.get("controller_source_sha256", ""),
            "controller_binary_sha256": receipt.get("controller_binary_sha256", ""),
            "receipt_path": rel(attempt / "run_receipt.json"),
            "trace_path": rel(attempt / "live/state_trace.jsonl"),
            "notes": "Historical center/disc fallback was suppressed by lcs_contact; live candidate obstacle cost is zero.",
        })

    # One explicitly recorded, contaminated TR07 diagnostic attempt.
    inventory.append({
        "source_study": "c3plus_task_cost_ratio/reproducibility",
        "attempt_id": "TR07/diagnostic/perimeter_candidate_identity/attempt_04",
        "task_base": "TR07", "alpha_p": "2", "alpha_theta": "0.5",
        "obstacle_config": "NOT_APPLICABLE", "obstacle_family": "",
        "scene": "open_task", "start": "s01", "goal": "g02",
        "rotation_deg": "0", "seed": "", "attempt_number": "4",
        "scientific_status": "CONTAMINATED",
        "terminal_classification": "CONTAMINATED",
        "failure_class": "EXCLUDED_CONTROL_RATE_CONTAMINATION",
        "notes": "Ten-second diagnostic; unrelated two-lane live campaign was active; no scientific admission.",
    })

    obstacle_metrics = rows(OBS / "metrics/per_run_results.csv")
    metric_by_attempt = {
        str((ROOT / item["admitted_attempt"]).resolve()): item
        for item in obstacle_metrics
        if item["obstacle_config_id"] in {"OC01", "OC04", "OC08"}
    }
    for attempt in sorted((OBS / "runs").glob("TR*/**/OC*/**/attempts/attempt_*")):
        if not attempt.is_dir() or not any(
                f"/{oc}/" in str(attempt) for oc in ("OC01", "OC04", "OC08")):
            continue
        receipt_path = attempt / "run_receipt.json"
        if not receipt_path.is_file():
            continue
        receipt = read_json(receipt_path)
        manifest = receipt.get("manifest_row", {})
        tr = manifest.get("task_cost_id", "")
        oc = manifest.get("obstacle_config_id", "")
        scene = manifest.get("scenario", "")
        complete_path = attempt / "RUN_COMPLETE.json"
        complete = read_json(complete_path) if complete_path.is_file() else {}
        admitted_path = attempt.parents[1] / "ADMITTED_RUN.json"
        admitted_attempt = ""
        if admitted_path.is_file():
            admitted_attempt = read_json(admitted_path).get("attempt", "")
        marker_selects = admitted_attempt.endswith(attempt.name)

        key = (tr, oc, scene, attempt.name)
        if key == ("TR07", "OC08", "box_clutter", "attempt_01"):
            status = "ORPHANED"
            note = ("Recorder stopped at the 600-s wall cap (last sim time 367.578 s); "
                    "unowned controller/simulator/OSC continued to about 2544 s wall and "
                    "forensic sim time 1576.559 s. Post-cap data excluded.")
        elif key == ("TR08", "OC04", "ycb_clutter", "attempt_01"):
            status = "WORKSPACE_ABORT"
            note = "Launcher completion missing; stale same-port OSC/recorder survived into attempt_02."
        elif key == ("TR08", "OC04", "ycb_clutter", "attempt_02"):
            status = "CONTAMINATED"
            note = "ADMITTED_RUN marker is overridden: attempt_01 OSC/recorder remained alive on port 24132 at prelaunch."
        elif key == ("TR07", "OC04", "open_task", "attempt_01"):
            status = "NEVER_RUN"
            note = "Receipt exists, but there are no live/forensic data and no completion marker."
        elif complete_path.is_file() and marker_selects:
            status = "ADMITTED"
            note = "Clean serial timing/logger gates passed."
        elif complete_path.is_file():
            status = "RUNTIME_FAILURE"
            note = "Completion exists but this attempt was not selected by the admission marker."
        else:
            status = "RUNTIME_FAILURE"
            note = "No completion or admission evidence."

        metric = metric_by_attempt.get(str(attempt.resolve()), {})
        trace_path = attempt / "live/state_trace.jsonl"
        observed = trace_summary(trace_path)
        failure = metric.get("failure_class", "")
        if status == "ORPHANED":
            failure = "TIMEOUT_ORPHAN"
            terminal = "ORPHANED"
        elif status == "WORKSPACE_ABORT":
            failure = "WORKSPACE_ABORT"
            terminal = "WORKSPACE_ABORT"
        elif status == "CONTAMINATED":
            failure = "CONTAMINATED"
            terminal = "CONTAMINATED"
        elif status == "NEVER_RUN":
            failure = "NEVER_RUN"
            terminal = "NEVER_RUN"
        else:
            terminal = metric.get("runtime_status", "") or (
                "ADMITTED" if status == "ADMITTED" else "RUNTIME_FAILURE")
        f1, f2, f3 = flags(failure)
        start, goal, rotation = scene_identity(scene)
        env = receipt.get("environment", {})
        family = manifest.get("family", "")
        inventory.append({
            "source_study": "c3plus_obstacle_cost_tuning_v2",
            "attempt_id": f"{tr}/{oc}/{scene}/{attempt.name}",
            "task_base": tr, "alpha_p": str(manifest.get("alpha_p", "")),
            "alpha_theta": str(manifest.get("alpha_theta", "")),
            "obstacle_config": oc, "obstacle_family": family,
            "obstacle_range_m": manifest.get("range_parameter_m", ""),
            "obstacle_weight": manifest.get("raw_weight", ""),
            "obstacle_distance_geometry": "orientation_aware_footprint_min_sdf",
            "soft_rank_cost_active": "0" if scene in {"open_task", "open_task_negative_control"} else "1",
            "inner_soft_obstacle_mode": "none", "lcs_contact_active": "1",
            "scene": scene, "start": start, "goal": goal,
            "rotation_deg": rotation, "seed": str(manifest.get("seed", "")),
            "attempt_number": attempt.name,
            "success_observed": metric.get("success", observed.get("success_observed", "")),
            "retained_success": metric.get("retained_success", ""),
            "T_goal_s": metric.get("T_goal", observed.get("T_goal_s", "")),
            "best_position_error_m": metric.get("best_position_error", observed.get("best_position_error_m", "")),
            "final_position_error_m": metric.get("final_position_error", observed.get("final_position_error_m", "")),
            "best_orientation_error_rad": metric.get("best_yaw_error", observed.get("best_orientation_error_rad", "")),
            "final_orientation_error_rad": metric.get("final_yaw_error", observed.get("final_orientation_error_rad", "")),
            "timing_valid": str(complete.get("timing", {}).get("timing_valid", "")),
            "concurrency": str(complete.get("timing", {}).get("concurrency_level", "")),
            "runtime_wall_s": "2544" if status == "ORPHANED" else str(complete.get("wall_time_s", "")),
            "scientific_status": status, "terminal_classification": terminal,
            "failure_class": failure,
            "F1": f1, "F2": f2, "F3": f3,
            "CHOMP_normalized": metric.get("CHOMP_normalized", ""),
            "minimum_clearance_m": metric.get("minimum_physical_clearance", ""),
            "transactions": metric.get("transactions", ""),
            "median_transaction_displacement_m": metric.get("median_transaction_displacement", ""),
            "physical_contact_fraction": metric.get("physical_contact_fraction", ""),
            "contactless_c3_fraction": metric.get("contactless_c3_fraction", ""),
            "median_J_obstacle_over_J_task": metric.get("median_Jobs_Jtask", ""),
            "controller_source_sha256": receipt.get("controller_source_sha256", ""),
            "controller_binary_sha256": receipt.get("controller_binary_sha256", ""),
            "receipt_path": rel(receipt_path),
            "trace_path": rel(trace_path) if trace_path.is_file() else "",
            "notes": note,
        })

    # The capacity study later reused TR07+OC08 as a timing workload.  These
    # attempts belong in a complete execution inventory, but their deliberately
    # short caps and changing affinity/thread/concurrency treatments make them
    # contaminated for an obstacle-performance comparison.
    for receipt_path in sorted(CAPACITY.glob("**/run_receipt.json")):
        receipt = read_json(receipt_path)
        fixed = receipt.get("fixed_cost", {})
        if (fixed.get("task_cost_id") not in {"TR07", "TR08"} or
                fixed.get("obstacle_config_id") not in {"OC01", "OC04", "OC08"}):
            continue
        run = receipt_path.parent
        timing_path = run / "timing.json"
        timing = read_json(timing_path) if timing_path.is_file() else {}
        trace_path = run / "live/state_trace.jsonl"
        observed = trace_summary(trace_path)
        tx_path = run / "forensics/forensics_transactions.csv"
        tx = rows(tx_path) if tx_path.is_file() else []
        displacements = sorted(float(x["object_displacement"]) for x in tx)
        median_displacement = ""
        if displacements:
            n = len(displacements)
            median_displacement = str(
                displacements[n // 2] if n % 2 else
                (displacements[n // 2 - 1] + displacements[n // 2]) / 2)
        scene = receipt.get("scene", "")
        start, goal, rotation = scene_identity(scene)
        inventory.append({
            "source_study": "c3plus_concurrency_capacity",
            "attempt_id": rel(run),
            "task_base": fixed.get("task_cost_id", ""),
            "alpha_p": str(fixed.get("alpha_p", "")),
            "alpha_theta": str(fixed.get("alpha_theta", "")),
            "obstacle_config": fixed.get("obstacle_config_id", ""),
            "obstacle_family": fixed.get("family", ""),
            "obstacle_range_m": str(fixed.get("epsilon_m", fixed.get("sigma_m", ""))),
            "obstacle_weight": str(fixed.get("weight", "")),
            "obstacle_distance_geometry": "orientation_aware_footprint_min_sdf",
            "soft_rank_cost_active": "1" if scene != "open_task" else "0",
            "inner_soft_obstacle_mode": "none", "lcs_contact_active": "1",
            "scene": scene, "start": start, "goal": goal,
            "rotation_deg": rotation, "seed": str(receipt.get("seed", "")),
            "attempt_number": receipt.get("run_id", run.name),
            "success_observed": observed.get("success_observed", ""),
            "retained_success": "", "T_goal_s": observed.get("T_goal_s", ""),
            "best_position_error_m": observed.get("best_position_error_m", ""),
            "final_position_error_m": observed.get("final_position_error_m", ""),
            "best_orientation_error_rad": observed.get("best_orientation_error_rad", ""),
            "final_orientation_error_rad": observed.get("final_orientation_error_rad", ""),
            "timing_valid": str(timing.get("timing_valid", "")),
            "concurrency": str(receipt.get("concurrency_level", "")),
            "runtime_wall_s": str(timing.get("wall_time_s", "")),
            "scientific_status": "CONTAMINATED",
            "terminal_classification": "CONTAMINATED",
            "failure_class": "CAPACITY_PROBE_NOT_OBSTACLE_SCIENCE",
            "F1": "", "F2": "", "F3": "", "CHOMP_normalized": "",
            "minimum_clearance_m": "", "transactions": str(len(tx)),
            "median_transaction_displacement_m": median_displacement,
            "physical_contact_fraction": "", "contactless_c3_fraction": "",
            "median_J_obstacle_over_J_task": "",
            "controller_source_sha256": receipt.get("controller_source_sha256", ""),
            "controller_binary_sha256": receipt.get("controller_binary_sha256", ""),
            "receipt_path": rel(receipt_path),
            "trace_path": rel(trace_path) if trace_path.is_file() else "",
            "notes": ("Timing/capacity workload only: short cap and deliberately varied "
                      "CPU affinity, worker caps, or concurrency; excluded from obstacle efficacy."),
        })

    OUT.mkdir(parents=True, exist_ok=True)
    inventory_path = OUT / "historical_run_inventory.csv"
    with inventory_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for item in inventory:
            writer.writerow({field: item.get(field, "") for field in FIELDS})

    hashes = {}
    for path in [
        TASK / "metrics/per_run_results.csv",
        TASK / "metrics/admission_audit.csv",
        TASK / "reproducibility/attempt_inventory.csv",
        OBS / "metrics/per_run_results.csv",
        OBS / "selection/stage1_experiment_manifest.csv",
        OBS / "forensics/historical_failure_clearance.csv",
        ROOT / "results/c3plus_concurrency_capacity/report/C3PLUS_CONCURRENCY_CAPACITY_REPORT_ATTEMPT3.md",
        ROOT / "results/c3plus_inner_outer_cost_boundary_audit/source_trace/obstacle_trace.md",
        ROOT / "results/c3plus_inner_outer_cost_boundary_audit/report/C3PLUS_INNER_OUTER_COST_BOUNDARY_AUDIT.md",
    ]:
        hashes[rel(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    admitted_obstacle = sum(
        x.get("scientific_status") == "ADMITTED"
        and x.get("obstacle_config") in {"OC01", "OC04", "OC08"}
        and x.get("scene") != "open_task"
        for x in inventory)
    admitted_controls = sum(
        x.get("scientific_status") == "ADMITTED"
        and x.get("obstacle_config") in {"OC01", "OC04", "OC08"}
        and x.get("scene") == "open_task"
        for x in inventory)
    provenance = {
        "mode": "existing-results-only; no live simulation",
        "inventory_rows": len(inventory),
        "admitted_historical_task_rows": sum(
            x.get("source_study") == "c3plus_task_cost_ratio"
            and x.get("scientific_status") == "ADMITTED" for x in inventory),
        "admitted_target_obstacle_bearing_rows": admitted_obstacle,
        "admitted_open_negative_controls": admitted_controls,
        "input_sha256": hashes,
    }
    (OUT / "audit_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps(provenance, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
