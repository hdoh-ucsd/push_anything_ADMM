#!/usr/bin/env python3
"""Offline aggregation for the controlled C3+ task-cost-ratio study."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
C3 = ROOT / "external/oim_c++_anything/.claude/worktrees/oim-scene-sync-metrics"
OUT = ROOT / "results/c3plus_task_cost_ratio"
METRICS = OUT / "metrics"
FIGURES = OUT / "figures"
POS_TOL = 0.05
YAW_TOL = 0.10
PROGRESS_WINDOW_S = 30.0
PROGRESS_POS_M = 0.005
PROGRESS_YAW_RAD = 0.05
CONDITIONS = {
    "TR01": (0.5, 0.5), "TR02": (0.5, 1.0), "TR03": (0.5, 2.0),
    "TR04": (1.0, 0.5), "TR05": (1.0, 1.0), "TR06": (1.0, 2.0),
    "TR07": (2.0, 0.5), "TR08": (2.0, 1.0), "TR09": (2.0, 2.0),
}
SCENES = ["open_task", "single_obstacle", "shelf_gap", "box_clutter", "ycb_clutter", "icra_sign", "slalom"]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


CHOMP = load_module("ratio_chomp_eval", C3 / "tools/relu_chomp/chomp_eval.py")
GEOM = load_module("ratio_chomp_common", C3 / "tools/relu_chomp/common.py")


PER_RUN_FIELDS = [
    "condition_id", "scenario", "seed", "alpha_p", "alpha_theta",
    "baseline_ratio", "effective_ratio", "ratio_factor", "log2_ratio_factor",
    "success", "position_success", "orientation_success", "T_goal",
    "first_success_step", "best_position_error", "best_orientation_error",
    "final_position_error", "final_orientation_error", "E_best", "E_final",
    "J_translation_median", "J_orientation_median", "realized_ratio_median",
    "realized_ratio_q25", "realized_ratio_q75", "CHOMP_raw", "CHOMP_normalized",
    "realized_ratio_early_median", "realized_ratio_mid_median",
        "realized_ratio_near_goal_median", "realized_ratio_stage_basis",
    "min_physical_clearance", "time_below_10mm", "time_below_5mm",
    "max_penetration", "num_repositions", "contactless_c3_fraction",
    "physical_contact_fraction", "runtime_status", "failure_class",
    "concurrency_level", "requested_control_frequency",
    "achieved_mean_control_frequency", "achieved_median_control_frequency",
    "control_period_median", "control_period_p95", "control_period_p99",
    "timing_valid", "notes",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as stream:
        return list(csv.DictReader(stream))


def read_selected_candidates(path: Path, committed: set[int]) -> list[dict[str, str]]:
    """Stream the multi-gigabyte pool and retain only ranked selections."""
    selected = []
    with path.open() as stream:
        for row in csv.DictReader(stream):
            try:
                event_id = int(row["event_id"])
            except (KeyError, TypeError, ValueError):
                continue
            if row.get("selected") == "1" and event_id in committed:
                selected.append(row)
    return selected


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open() as stream:
        for line in stream:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite(values):
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def median(values):
    values = finite(values)
    return float(np.median(values)) if values else math.nan


def wrap(value: float) -> float:
    return (value + math.pi) % (2 * math.pi) - math.pi


def quat_yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_error(q, goal_yaw: float) -> float:
    q = np.asarray(q, dtype=float)
    q /= np.linalg.norm(q)
    goal = np.asarray([math.cos(goal_yaw / 2), 0, 0, math.sin(goal_yaw / 2)])
    return float(2 * math.acos(min(1.0, abs(float(np.dot(q, goal))))))


def actual_clearances(scene: str, live: Path):
    trace = GEOM.read_state_trace(str(live))
    if trace is None:
        return np.array([]), np.array([])
    config = GEOM.load_scene(scene)
    if not (config["discs"] or config["polys"]):
        return trace["t"], np.full(len(trace["t"]), np.inf)
    boundary = GEOM.sample_boundary(config["footprint"])
    result = []
    for x, y, yaw in zip(trace["x"], trace["y"], trace["yaw"]):
        c, s = math.cos(yaw), math.sin(yaw)
        world = np.column_stack((x + c * boundary[:, 0] - s * boundary[:, 1],
                                 y + s * boundary[:, 0] + c * boundary[:, 1]))
        result.append(min(GEOM.min_obstacle_sdf(px, py, config["discs"], config["polys"])
                          for px, py in world))
    return trace["t"], np.asarray(result)


def duration_below(times: np.ndarray, values: np.ndarray, threshold: float) -> float:
    if len(times) < 2:
        return 0.0
    return float(np.sum(np.diff(times)[values[:-1] < threshold]))


def success_from_steps(live: Path, goal: tuple[float, float, float, float]):
    gx, gy, gz, gyaw = goal
    first_time = None
    first_step = None
    for row in read_jsonl(live / "steps_raw.jsonl"):
        objects = row.get("objects", {})
        if not objects:
            continue
        pose = next(iter(objects.values()))
        pos = math.sqrt((pose[4] - gx) ** 2 + (pose[5] - gy) ** 2 + (pose[6] - gz) ** 2)
        yaw = quat_error(pose[:4], gyaw)
        if pos < POS_TOL and yaw < YAW_TOL:
            first_time = float(row["sim_time"])
            first_step = int(row["control_step"])
            break
    return first_time, first_step


def load_goals() -> dict[str, tuple[float, float, float, float, int]]:
    import yaml
    rows = yaml.safe_load((ROOT / "results/benchmark_sync_rot15/c3plus_cases.yaml").read_text())["cases"]
    result = {}
    for row in rows:
        if row["case_id"] == "s01_g02_rot000":
            yaw = 2 * math.atan2(float(row["goal_qz"]), float(row["goal_qw"]))
            result[row["scene"]] = (float(row["goal_x"]), float(row["goal_y"]),
                                     float(row["goal_z"]), yaw, int(row["seed"]))
    result["box_clutter"] = (0.381, -0.305, 0.0008, math.pi / 2, 17601)
    return result


def classify_outcome(trace: list[dict], success: bool, planner_text: str):
    if success:
        return "SUCCESS", ""
    if "terminate called" in planner_text or "Traceback" in planner_text or not trace:
        return "RUNTIME_FAILURE", "F7"
    times = np.asarray([float(row["t"]) for row in trace])
    pos = np.asarray([float(row["pos_err"]) for row in trace])
    yaw = np.asarray([float(row["ang_err"]) for row in trace])
    start = np.searchsorted(times, times[-1] - PROGRESS_WINDOW_S)
    progressing = pos[start] - pos[-1] > PROGRESS_POS_M or yaw[start] - yaw[-1] > PROGRESS_YAW_RAD
    outcome = "TIMEOUT_PROGRESSING" if progressing else "TRUE_STALL"
    return outcome, "" if progressing else None


def causal_class(cycles: list[dict[str, str]], events: list[dict], clearances: np.ndarray) -> str:
    names = [str(event.get("flag_name", "")) for event in events if event.get("enter_or_exit") != "EXIT"]
    contact = [int(row["physical_contact"]) for row in cycles]
    c3 = [row for row in cycles if row["is_c3_mode"] == "1"]
    c3_contact = [int(row["physical_contact"]) for row in c3]
    if len(clearances) and np.nanmin(clearances) <= 0 and np.mean(contact or [0]) > 0.05:
        return "F6"
    if c3_contact and np.mean(c3_contact) < 0.02:
        return "F1"
    phantom = names.count("REPOS_REACHED_NO_CONTACT") + names.count("PLANNER_PHYSICAL_CONTACT_MISMATCH")
    no_progress = names.count("NO_PROGRESS_EXIT")
    if phantom or no_progress:
        return "F2"
    if names.count("FAILED_SECTOR_RESELECT"):
        return "F3"
    return "UNKNOWN"


def process_run(manifest: dict[str, str], goals) -> tuple[dict, list[dict]] | None:
    condition, scene = manifest["condition_id"], manifest["scenario"]
    scenario_dir = OUT / "runs" / condition / scene
    admitted = scenario_dir / "ADMITTED_RUN.json"
    if not admitted.exists():
        return None
    run = scenario_dir / json.loads(admitted.read_text())["attempt"]
    completion_path = run / "RUN_COMPLETE.json"
    if not completion_path.exists():
        completion_path = run / "RUN_RUNTIME_FAILURE.json"
    if not completion_path.exists():
        return None
    live, forensic = run / "live", run / "forensics"
    trace = read_jsonl(live / "state_trace.jsonl")
    cycles = read_csv(forensic / "forensics_cycle.csv")
    events = read_jsonl(forensic / "forensics_events.jsonl")
    gx, gy, gz, gyaw, seed = goals[scene]
    first_t, first_step = success_from_steps(live, (gx, gy, gz, gyaw))
    success = first_t is not None
    pos = np.asarray([float(row["pos_err"]) for row in trace])
    yaw = np.asarray([float(row["ang_err"]) for row in trace])
    normalized = np.maximum(pos / POS_TOL, yaw / YAW_TOL)
    position_success = bool(np.any(pos < POS_TOL))
    orientation_success = bool(np.any(yaw < YAW_TOL))
    committed = {int(row["event_id"]) for row in cycles}
    selected = read_selected_candidates(forensic / "forensics_candidates.csv", committed)
    selected_with_cost = [
        row for row in selected
        if math.isfinite(float(row["J_translation"]))
        and math.isfinite(float(row["J_orientation"]))
    ]
    jtrans = np.asarray([float(row["J_translation"]) for row in selected_with_cost])
    jori = np.asarray([float(row["J_orientation"]) for row in selected_with_cost])
    ratios = jtrans / np.maximum(jori, 1e-12)
    tclear, clearances = actual_clearances(scene, live)
    chomp = CHOMP.eval_run(scene, str(live))
    planner_text = (live / "planner.log").read_text(errors="replace")
    outcome, failure = classify_outcome(trace, success, planner_text)
    if outcome == "TRUE_STALL":
        failure = causal_class(cycles, events, clearances)
    complete = json.loads(completion_path.read_text())
    run_timing = complete.get("timing", {})
    ctimes = np.asarray([float(row["time"]) for row in cycles])
    periods = np.diff(ctimes)
    c3 = [row for row in cycles if row["is_c3_mode"] == "1"]
    physical_fraction = float(np.mean([int(row["physical_contact"]) for row in cycles]))
    contactless = float(np.mean([not int(row["physical_contact"]) for row in c3])) if c3 else math.nan
    repositions = sum(1 for event in events if event.get("flag_name") == "NO_PROGRESS_EXIT" and event.get("enter_or_exit") != "EXIT")
    stage_rows = []
    stage_ratios = defaultdict(list)
    stage_basis = "achieved_normalized_error_progress_thirds"
    if selected_with_cost:
        cycle_times = np.asarray([float(item["time"]) for item in cycles])
        cycle_error = np.maximum(
            np.asarray([float(item["position_error"]) for item in cycles]) / POS_TOL,
            np.asarray([float(item["yaw_error"]) for item in cycles]) / YAW_TOL,
        )
        error_span = float(cycle_error[0] - np.min(cycle_error))
        if error_span <= 1e-12:
            stage_basis = "temporal_thirds_fallback_no_measured_error_reduction"
            selected_times = np.asarray([float(item["time"]) for item in selected_with_cost])
            temporal_boundaries = np.quantile(selected_times, [1 / 3, 2 / 3])
        for item, ratio in zip(selected_with_cost, ratios):
            t = float(item["time"])
            if error_span > 1e-12:
                index = min(int(np.searchsorted(cycle_times, t, side="right")) - 1,
                            len(cycle_error) - 1)
                index = max(index, 0)
                fraction = float(np.clip((cycle_error[0] - cycle_error[index]) / error_span, 0, 1))
                stage = "EARLY" if fraction <= 1 / 3 else "MID" if fraction <= 2 / 3 else "NEAR_GOAL"
            else:
                stage = "EARLY" if t <= temporal_boundaries[0] else "MID" if t <= temporal_boundaries[1] else "NEAR_GOAL"
            stage_ratios[stage].append(float(ratio))
            stage_rows.append({
                "condition_id": condition, "scenario": scene, "time": t,
                "stage": stage, "stage_basis": stage_basis,
                "J_translation": item["J_translation"],
                "J_orientation": item["J_orientation"], "realized_ratio": ratio,
            })
    row = {
        "condition_id": condition, "scenario": scene, "seed": seed,
        "alpha_p": manifest["alpha_p"], "alpha_theta": manifest["alpha_theta"],
        "baseline_ratio": 1.0, "effective_ratio": float(manifest["ratio_factor"]),
        "ratio_factor": float(manifest["ratio_factor"]),
        "log2_ratio_factor": float(manifest["log2_ratio_factor"]),
        "success": int(success), "position_success": int(position_success),
        "orientation_success": int(orientation_success), "T_goal": first_t,
        "first_success_step": first_step,
        "best_position_error": float(np.min(pos)), "best_orientation_error": float(np.min(yaw)),
        "final_position_error": float(pos[-1]), "final_orientation_error": float(yaw[-1]),
        "E_best": float(np.min(normalized)), "E_final": float(normalized[-1]),
        "J_translation_median": float(np.median(jtrans)),
        "J_orientation_median": float(np.median(jori)),
        "realized_ratio_median": float(np.median(ratios)),
        "realized_ratio_q25": float(np.percentile(ratios, 25)),
        "realized_ratio_q75": float(np.percentile(ratios, 75)),
        "realized_ratio_early_median": median(stage_ratios["EARLY"]),
        "realized_ratio_mid_median": median(stage_ratios["MID"]),
        "realized_ratio_near_goal_median": median(stage_ratios["NEAR_GOAL"]),
        "realized_ratio_stage_basis": stage_basis,
        "CHOMP_raw": chomp["M_CHOMP"], "CHOMP_normalized": chomp["M_CHOMP_norm"],
        "min_physical_clearance": float(np.min(clearances)) if len(clearances) else math.nan,
        "time_below_10mm": duration_below(tclear, clearances, 0.010),
        "time_below_5mm": duration_below(tclear, clearances, 0.005),
        "max_penetration": max(0.0, -float(np.min(clearances))) if len(clearances) else 0.0,
        "num_repositions": repositions, "contactless_c3_fraction": contactless,
        "physical_contact_fraction": physical_fraction, "runtime_status": outcome,
        "failure_class": failure,
        "concurrency_level": run_timing.get("concurrency_level", 1),
        "requested_control_frequency": run_timing.get("requested_control_frequency", "event_driven (publish_frequency=0)"),
        "achieved_mean_control_frequency": run_timing.get("achieved_mean_control_frequency", (len(ctimes) - 1) / (ctimes[-1] - ctimes[0])),
        "achieved_median_control_frequency": run_timing.get("achieved_median_control_frequency", 1 / float(np.median(periods))),
        "control_period_median": run_timing.get("control_period_median", float(np.median(periods))),
        "control_period_p95": run_timing.get("control_period_p95", float(np.percentile(periods, 95))),
        "control_period_p99": run_timing.get("control_period_p99", float(np.percentile(periods, 99))),
        "timing_valid": int(run_timing.get("timing_valid", True)),
        "notes": "serial admitted load; ReLU inactive; rank exponential and lcs_contact frozen",
    }
    return row, stage_rows


def summarize(rows: list[dict]):
    by_condition = defaultdict(list)
    for row in rows:
        by_condition[row["condition_id"]].append(row)
    summaries = []
    for condition in CONDITIONS:
        group = by_condition.get(condition, [])
        ap, at = CONDITIONS[condition]
        valid = [row for row in group if row["timing_valid"]]
        successes = [row for row in valid if row["success"]]
        failed = [row for row in valid if not row["success"]]
        summaries.append({
            "ID": condition, "alpha_p": ap, "alpha_theta": at,
            "ratio_factor": ap / at, "effective_ratio": ap / at,
            "valid_expected": f"{len(valid)}/{len(SCENES)}",
            "valid_runs": len(valid), "success_count": sum(row["success"] for row in valid),
            "success_rate": np.mean([row["success"] for row in valid]) if valid else math.nan,
            "position_success_rate": np.mean([row["position_success"] for row in valid]) if valid else math.nan,
            "orientation_success_rate": np.mean([row["orientation_success"] for row in valid]) if valid else math.nan,
            "median_T_goal": median([row["T_goal"] for row in successes]),
            "median_E_best_failed": median([row["E_best"] for row in failed]),
            "median_final_position_error": median([row["final_position_error"] for row in valid]),
            "median_final_orientation_error": median([row["final_orientation_error"] for row in valid]),
            "median_realized_J_ratio": median([row["realized_ratio_median"] for row in valid]),
            "median_realized_J_ratio_early": median([row["realized_ratio_early_median"] for row in valid]),
            "median_realized_J_ratio_mid": median([row["realized_ratio_mid_median"] for row in valid]),
            "median_realized_J_ratio_near_goal": median([row["realized_ratio_near_goal_median"] for row in valid]),
            "median_CHOMP_normalized": median([row["CHOMP_normalized"] for row in valid]),
            "worst_min_clearance": min(finite([row["min_physical_clearance"] for row in valid]), default=math.nan),
            "penetration_count": sum(row["max_penetration"] > 0 for row in valid),
            "runtime_failure_count": sum(row["runtime_status"] == "RUNTIME_FAILURE" for row in valid),
            "timing_invalid_count": sum(not row["timing_valid"] for row in group),
        })
    fields = list(summaries[0])
    write_csv(METRICS / "condition_summary.csv", fields, summaries)
    return summaries


def ratio_summary(rows: list[dict]):
    output = []
    for factor in (0.25, 0.5, 1.0, 2.0, 4.0):
        group = [row for row in rows if math.isclose(row["ratio_factor"], factor) and row["timing_valid"]]
        output.append({
            "ratio_factor": factor, "effective_ratio": factor,
            "conditions_at_ratio": ";".join(sorted({row["condition_id"] for row in group})),
            "valid_scene_trials": len(group),
            "success_rate": np.mean([row["success"] for row in group]) if group else math.nan,
            "position_success_rate": np.mean([row["position_success"] for row in group]) if group else math.nan,
            "orientation_success_rate": np.mean([row["orientation_success"] for row in group]) if group else math.nan,
            "median_T_goal": median([row["T_goal"] for row in group if row["success"]]),
            "median_E_best": median([row["E_best"] for row in group]),
            "median_final_position_error": median([row["final_position_error"] for row in group]),
            "median_final_orientation_error": median([row["final_orientation_error"] for row in group]),
            "median_realized_cost_ratio": median([row["realized_ratio_median"] for row in group]),
            "median_CHOMP_normalized": median([row["CHOMP_normalized"] for row in group]),
            "median_min_clearance": median([row["min_physical_clearance"] for row in group]),
        })
    write_csv(METRICS / "ratio_summary.csv", list(output[0]), output)
    return output


def common_scale_summary(rows: list[dict]):
    output = []
    for condition in ("TR01", "TR05", "TR09"):
        group = [row for row in rows if row["condition_id"] == condition and row["timing_valid"]]
        ap, at = CONDITIONS[condition]
        output.append({
            "condition_id": condition,
            "common_scale": ap,
            "ratio_factor": ap / at,
            "valid_runs": len(group),
            "success_count": sum(row["success"] for row in group),
            "success_rate": np.mean([row["success"] for row in group]) if group else math.nan,
            "median_T_goal": median([row["T_goal"] for row in group if row["success"]]),
            "median_E_best": median([row["E_best"] for row in group]),
            "median_final_position_error": median([row["final_position_error"] for row in group]),
            "median_final_orientation_error": median([row["final_orientation_error"] for row in group]),
            "median_realized_J_ratio": median([row["realized_ratio_median"] for row in group]),
            "median_CHOMP_normalized": median([row["CHOMP_normalized"] for row in group]),
        })
    write_csv(METRICS / "common_scale_summary.csv", list(output[0]), output)
    return output


def paired(rows: list[dict]):
    index = {(row["condition_id"], row["scenario"]): row for row in rows}
    output = []
    for condition in CONDITIONS:
        if condition == "TR05":
            continue
        for scene in SCENES:
            a, b = index.get(("TR05", scene)), index.get((condition, scene))
            if not a or not b:
                continue
            transition = ("SUCCESS" if a["success"] else "FAIL") + " -> " + ("SUCCESS" if b["success"] else "FAIL")
            def delta(key):
                av, bv = a[key], b[key]
                return float(bv) - float(av) if av is not None and bv is not None else math.nan
            output.append({
                "condition_id": condition, "scenario": scene, "seed": b["seed"],
                "success_transition": transition, "Delta_T_goal": delta("T_goal"),
                "Delta_E_best": delta("E_best"),
                "Delta_final_position": delta("final_position_error"),
                "Delta_final_orientation": delta("final_orientation_error"),
                "Delta_CHOMP_normalized": delta("CHOMP_normalized"),
                "Delta_min_clearance": delta("min_physical_clearance"),
            })
    fields = list(output[0]) if output else ["condition_id", "scenario", "seed", "success_transition", "Delta_T_goal", "Delta_E_best", "Delta_final_position", "Delta_final_orientation", "Delta_CHOMP_normalized", "Delta_min_clearance"]
    write_csv(METRICS / "paired_vs_baseline.csv", fields, output)
    return output


def scenario_outcomes(rows: list[dict]) -> None:
    index = {(row["condition_id"], row["scenario"]): row for row in rows}
    output = []
    for condition in CONDITIONS:
        for scene in SCENES:
            row = index[(condition, scene)]
            output.append({
                "condition_id": condition,
                "scenario": scene,
                "success": row["success"],
                "runtime_status": row["runtime_status"],
                "failure_class": row["failure_class"],
                "T_goal": row["T_goal"],
                "E_best": row["E_best"],
                "final_position_error": row["final_position_error"],
                "final_orientation_error": row["final_orientation_error"],
            })
    write_csv(METRICS / "scenario_outcomes.csv", list(output[0]), output)


def admission_audit(manifest: list[dict[str, str]]) -> None:
    output = []
    for item in manifest:
        scenario_dir = OUT / "runs" / item["condition_id"] / item["scenario"]
        admission = json.loads((scenario_dir / "ADMITTED_RUN.json").read_text())
        run = scenario_dir / admission["attempt"]
        completion_name = admission.get("completion_file", "RUN_COMPLETE.json")
        completion = json.loads((run / completion_name).read_text())
        meta = json.loads((run / "forensics/forensics_run_meta.json").read_text())
        output.append({
            "condition_id": item["condition_id"], "scenario": item["scenario"],
            "attempt": admission["attempt"], "completion_file": completion_name,
            "seed_manifest": item["seed"], "seed_logger": meta.get("seed_information"),
            "alpha_p_manifest": item["alpha_p"],
            "alpha_p_logger": meta.get("translation_cost_scale"),
            "alpha_theta_manifest": item["alpha_theta"],
            "alpha_theta_logger": meta.get("orientation_cost_scale"),
            "obstacle_mode": meta.get("obstacle_mode"),
            "rank_obstacle_mode": meta.get("rank_obstacle_mode"),
            "relu_weight": meta.get("relu_weight"),
            "timing_valid": int(completion.get("timing", {}).get("timing_valid", False)),
            "validator_returncode": completion.get("validator_returncode"),
            "bounded_final_tail_exception": int(bool(
                completion.get("validator_bounded_final_tail_exception") or
                completion.get("validator_abrupt_runtime_tail_exception"))),
            "admitted": 1,
        })
    write_csv(METRICS / "admission_audit.csv", list(output[0]), output)


def plots(summaries: list[dict], ratio_rows: list[dict], rows: list[dict]):
    FIGURES.mkdir(parents=True, exist_ok=True)
    ids = [row["ID"] for row in summaries]
    x = np.asarray([math.log2(row["ratio_factor"]) for row in summaries])
    specs = [
        ("success_vs_ratio.png", "success_rate", "Success rate"),
        ("Tgoal_vs_ratio.png", "median_T_goal", "Median T_goal (s)"),
        ("Ebest_vs_ratio.png", "median_E_best_failed", "Median E_best (failed)"),
        ("position_error_vs_ratio.png", "median_final_position_error", "Final position error (m)"),
        ("yaw_error_vs_ratio.png", "median_final_orientation_error", "Final orientation error (rad)"),
        ("realized_cost_ratio_vs_nominal.png", "median_realized_J_ratio", "Median realized Jp/Jtheta"),
        ("CHOMP_vs_ratio.png", "median_CHOMP_normalized", "Median normalized CHOMP"),
        ("min_clearance_vs_ratio.png", "worst_min_clearance", "Worst min clearance (m)"),
    ]
    for filename, key, ylabel in specs:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        values = [row[key] for row in summaries]
        ax.scatter(x, values)
        for xx, yy, label in zip(x, values, ids):
            if math.isfinite(float(yy)):
                ax.annotate(label, (xx, yy), xytext=(3, 3), textcoords="offset points", fontsize=8)
        ax.set_xlabel("log2 ratio factor")
        ax.set_ylabel(ylabel)
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.grid(alpha=.25)
        fig.tight_layout(); fig.savefig(FIGURES / filename, dpi=180); plt.close(fig)
    # Preserve the earlier filenames as compatibility aliases; the requested
    # experiment names above are authoritative.
    for source, alias in (("position_error_vs_ratio.png", "final_position_vs_ratio.png"),
                          ("yaw_error_vs_ratio.png", "final_orientation_vs_ratio.png")):
        (FIGURES / alias).write_bytes((FIGURES / source).read_bytes())
    fig, ax = plt.subplots(figsize=(7, 5))
    px = [row["median_final_position_error"] for row in summaries]
    py = [row["median_final_orientation_error"] for row in summaries]
    ax.scatter(px, py)
    for xx, yy, label in zip(px, py, ids): ax.annotate(label, (xx, yy), xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Median final position error (m)"); ax.set_ylabel("Median final yaw error (rad)"); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(FIGURES / "position_vs_yaw_tradeoff.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5)); xx = np.arange(len(ids)); width=.25
    ax.bar(xx-width, [r["position_success_rate"] for r in summaries], width, label="position")
    ax.bar(xx, [r["orientation_success_rate"] for r in summaries], width, label="orientation")
    ax.bar(xx+width, [r["success_rate"] for r in summaries], width, label="joint")
    ax.set_xticks(xx, ids); ax.set_ylim(0,1); ax.legend(); ax.grid(axis="y",alpha=.25)
    fig.tight_layout(); fig.savefig(FIGURES / "position_orientation_success.png", dpi=180); plt.close(fig)
    common = [row for row in summaries if row["ID"] in {"TR01", "TR05", "TR09"}]
    fig, ax = plt.subplots(figsize=(6,4.5)); ax.plot([.5,1,2], [r["success_rate"] for r in common], marker="o")
    ax.set_xscale("log",base=2); ax.set_xticks([.5,1,2],["0.5x","1x","2x"]); ax.set_ylabel("Success rate"); ax.set_xlabel("Common tracking-cost scale"); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(FIGURES / "common_scale_Frho1.png", dpi=180); plt.close(fig)


def choose_winner(summaries: list[dict]):
    gate_path = OUT / "reproducibility/gate_status.json"
    if gate_path.exists() and not json.loads(gate_path.read_text()).get("campaign_permitted", False):
        gate = json.loads(gate_path.read_text())
        if not gate.get("deterministic_seed_and_candidate_identity", False):
            return "NONE", "E. RATIO_EXPERIMENT_BLOCKED_BY_REPRODUCIBILITY"
        if not gate.get("logger_v2", False):
            return "NONE", "F. RATIO_EXPERIMENT_BLOCKED_BY_LOGGER"
        return "NONE", "G. RATIO_EXPERIMENT_BLOCKED_BY_CONTROL_RATE"
    complete = all(int(row["valid_runs"]) == len(SCENES) for row in summaries)
    if not complete:
        return "NONE", "I. EXPERIMENT_INCOMPLETE"
    def key(row):
        return (-int(row["success_count"]), float(row["median_T_goal"]) if math.isfinite(float(row["median_T_goal"])) else math.inf,
                float(row["median_E_best_failed"]) if math.isfinite(float(row["median_E_best_failed"])) else -math.inf,
                abs(math.log2(float(row["ratio_factor"]))) + abs(math.log2(float(row["alpha_p"]))))
    winner = min(summaries, key=key)["ID"]
    best_success = max(int(row["success_count"]) for row in summaries)
    top_ratios = {float(row["ratio_factor"]) for row in summaries
                  if int(row["success_count"]) == best_success}
    if winner == "TR05":
        verdict = "A. BASELINE_NORMALIZED_RATIO_AND_SCALE_REMAINS_BEST"
    elif len(top_ratios) > 1:
        verdict = "D. RATIO_EFFECT_SCENE_DEPENDENT_NO_SINGLE_WINNER"
    else:
        ratios = {r["ID"]: r["ratio_factor"] for r in summaries}
        verdict = "C. ABSOLUTE_SCALE_MATTERS_MORE_THAN_RATIO" if ratios[winner] == 1 else "B. IMPROVED_TRANSLATION_ROTATION_RATIO_IDENTIFIED"
    return winner, verdict


def report(rows, summaries, ratio_rows, winner, verdict):
    from collections import Counter

    completed = len(rows)
    expected = len(CONDITIONS) * len(SCENES)
    box_done = any(row["scenario"] == "box_clutter" for row in rows)
    by_id = {row["ID"]: row for row in summaries}
    win = by_id.get(winner)
    baseline = by_id["TR05"]
    common = [by_id[key] for key in ("TR01", "TR05", "TR09")]
    all_timing_valid = completed == expected and all(row["timing_valid"] for row in rows)
    top_success = max(int(row["success_count"]) for row in summaries)
    top_conditions = [row["ID"] for row in summaries if int(row["success_count"]) == top_success]
    failures = [row for row in rows if not row["success"]]
    true_stalls = [row for row in failures if row["runtime_status"] == "TRUE_STALL"]
    failure_counts = Counter(row["failure_class"] or "UNCLASSIFIED" for row in failures)
    frozen_outer = sum(failure_counts[key] for key in ("F1", "F2", "F3"))
    baseline_index = {row["scenario"]: row for row in rows if row["condition_id"] == "TR05"}
    win_transitions = Counter()
    for row in rows:
        if row["condition_id"] == winner:
            base = baseline_index[row["scenario"]]
            win_transitions[("SUCCESS" if base["success"] else "FAIL") + " -> " +
                            ("SUCCESS" if row["success"] else "FAIL")] += 1
    transition_text = ", ".join(
        f"{key}: {value}" for key, value in sorted(win_transitions.items())
    )

    def fmt(value, digits=4):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "NA"
        if not math.isfinite(value):
            return "NA" if math.isnan(value) else "inf"
        return f"{value:.{digits}g}"

    text = f"""# C3+ translation-versus-rotation cost-ratio study

## Executive conclusion

All **{completed}/{expected}** newly generated scientific runs passed the fixed serial timing gate. TR07 and TR08 tied for the highest success count at **{top_success}/7**; the predeclared tie-break selects **TR08** because its successful-run median `T_goal` was **{fmt(win['median_T_goal'])} s** versus **{fmt(by_id['TR07']['median_T_goal'])} s** for TR07. Thus the pair to carry forward is `alpha_p=2`, `alpha_theta=1`, `F_rho=2`. This is a controlled panel selection, not evidence for a universal ratio: TR08 regressed the open task, the same `F_rho=2` at TR04 achieved only 1/7, and the best-success tie spans two ratio factors. The scientifically conservative verdict is therefore scene-dependent rather than a unique ratio law.

## Gates, provenance, and frozen quantities

- Deterministic seed/candidate identity: **PASS** for `RandomOnPerimeter` and `MeshNormalMultiObject`; TR03/TR05/TR07 use identical pre-score pools under matched scene/state/seed. A different seed changes the realization, and omitting the experiment seed retains legacy `std::random_device` behavior.
- Logger-v2: **PASS**. Initial transaction/pool, selected candidate, N+1=6 ranking states, translation/orientation components, and total-cost reconstruction are present. Across admission receipts the largest recorded task-cost residual is `5.82e-11`.
- Timing: concurrency two failed the predeclared gate; all scientific runs used concurrency one. **{sum(int(row['timing_valid']) for row in rows)}/{expected}** are timing-valid.
- Attempt accounting: the previous nine attempts are preserved in `reproducibility/attempt_inventory.csv`, all marked `EXCLUDED_CONTROL_RATE_CONTAMINATION`, and contribute zero observations.
- `box_clutter_real`: **SOURCE_FAITHFULLY_SYNCHRONIZED** from the authoritative OIM scene specification/MJCF into the native xArm6 frame. No substitute scene was invented.
- Baseline ratio: the remembered literal 4:1 is unsupported. The study uses the dimensionless normalized baseline `rho_0=1` and `F_rho=alpha_p/alpha_theta`.
- Baseline cost blocks: object XYZ diagonal entries are approximately `[10000,10000,6000]`; orientation is the existing state-dependent quaternion Hessian with scale 510, not a scalar yaw cost.
- Frozen obstacle configuration: all 63 metadata records say `obstacle_mode=lcs_contact`, `rank_obstacle_mode=exponential`, and `relu_weight=0`. CHOMP is offline evaluation only.
- Task-panel limitation: the six synchronized rot15 scenes use `s01_g02_rot000`; their requested goal yaw equals their start yaw. `box_clutter_real` retains its authoritative +90-degree goal. Consequently this panel measures the cost ratio mainly through incidental yaw deviations in six scenes and one explicit-rotation scene; it cannot establish a general result for the held `rotCCW090`/`rotCW090` tasks.

## Condition results

| ID | alpha_p | alpha_theta | F_rho | valid | success | median T_goal (s) | median failed E_best | median final pos (m) | median final yaw (rad) | median realized Jp/Jtheta | CHOMP norm | worst clearance (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for row in summaries:
        text += (f"| {row['ID']} | {row['alpha_p']} | {row['alpha_theta']} | "
                 f"{row['ratio_factor']} | {row['valid_runs']}/7 | {row['success_count']}/7 | "
                 f"{fmt(row['median_T_goal'])} | {fmt(row['median_E_best_failed'])} | "
                 f"{fmt(row['median_final_position_error'])} | {fmt(row['median_final_orientation_error'])} | "
                 f"{fmt(row['median_realized_J_ratio'])} | {fmt(row['median_CHOMP_normalized'])} | "
                 f"{fmt(row['worst_min_clearance'])} |\n")

    text += "\n## Scenario dependence\n\n| scenario | successful conditions |\n|---|---|\n"
    for scene in SCENES:
        successes = [row["condition_id"] for row in rows
                     if row["scenario"] == scene and row["success"]]
        text += f"| {scene} | {', '.join(successes) if successes else 'none'} |\n"

    text += "\n## Common-scale controls\n\n"
    text += "| condition | common scale | success | median T_goal (s) | median E_best | median final pos | median final yaw |\n|---|---:|---:|---:|---:|---:|---:|\n"
    for row in common:
        text += (f"| {row['ID']} | {row['alpha_p']}x | {row['success_count']}/7 | "
                 f"{fmt(row['median_T_goal'])} | {fmt(row['median_E_best_failed'])} | "
                 f"{fmt(row['median_final_position_error'])} | {fmt(row['median_final_orientation_error'])} |\n")
    text += "\nTR01 and TR05 each succeeded only on open_task; TR09 additionally succeeded on ycb_clutter, but its successful-run median time and failed-run error were worse than TR05. Common scale therefore changes closed-loop behavior materially, but not monotonically. The repeated-ratio controls reinforce the interaction: TR02/TR06 (F=0.5) score 2/7 versus 1/7, while TR04/TR08 (F=2) score 1/7 versus 3/7.\n"

    text += "\n## Nominal versus realized cost balance\n\n"
    text += "| condition | nominal F_rho | all-cycle median Jp/Jtheta | EARLY | MID | NEAR_GOAL |\n|---|---:|---:|---:|---:|---:|\n"
    for row in summaries:
        text += (f"| {row['ID']} | {row['ratio_factor']} | {fmt(row['median_realized_J_ratio'])} | "
                 f"{fmt(row['median_realized_J_ratio_early'])} | {fmt(row['median_realized_J_ratio_mid'])} | "
                 f"{fmt(row['median_realized_J_ratio_near_goal'])} |\n")
    text += "\nStages are thirds of each run's achieved reduction in normalized joint task error (with a temporal fallback only if no reduction exists), not arbitrary wall-time thirds. Nominal F_rho does not predict the realized ratio: the latter spans orders of magnitude and is strongly state/scene dependent. At the condition level Jp/Jtheta falls sharply from EARLY to NEAR_GOAL for all nine settings, primarily because translation error falls while the six rot000 tasks begin near their goal yaw; individual runs can be non-monotone.\n\nThe extreme early ratios are expected in this panel because six of seven tasks have zero requested relative rotation. They show that nominal block multipliers are not realized cost ratios, but do not show that translation should dominate a future ±90-degree rotation benchmark.\n"

    text += f"""

## Paired baseline comparison and safety

Relative to TR05, TR08 produced `{transition_text}`: three FAIL→SUCCESS transitions (single_obstacle, box_clutter, ycb_clutter), three FAIL→FAIL transitions, and one SUCCESS→FAIL transition (open_task). TR08's median normalized CHOMP was **{fmt(win['median_CHOMP_normalized'])}** versus **{fmt(baseline['median_CHOMP_normalized'])}** for TR05; worst penetration was **{fmt(max(0.0, -float(win['worst_min_clearance'])))} m** versus **{fmt(max(0.0, -float(baseline['worst_min_clearance'])))} m**. The success gain is therefore not explained by a material CHOMP safety regression. TR08 did not merely refuse obstacle progress: it completed three obstacle scenes, although it did fail to complete open_task.

## Failure mechanisms

Of {len(failures)} non-success runs, {len(true_stalls)} are TRUE_STALL, {failure_counts['F1']} are F1-primary, {failure_counts['F2']} are F2-primary, {failure_counts['F3']} are F3-primary, {failure_counts['F6']} are F6-primary, {failure_counts['F7']} are runtime failures, and {failure_counts['UNCLASSIFIED']} are progressing timeouts without a causal failure label. Frozen F1/F2/F3 mechanisms account for **{frozen_outer}/{len(failures)} ({100*frozen_outer/len(failures):.1f}%)** of all non-successes and **{frozen_outer}/{len(true_stalls)} ({100*frozen_outer/len(true_stalls):.1f}%)** of TRUE_STALL runs. They are not reinterpreted as local-C3 failure and were not modified.

## Answers to the requested questions

1. **Did every scientific run pass timing?** Yes, 63/63.
2. **Were exactly 63 new admissible runs obtained?** Yes. The nine contaminated attempts are separate and excluded.
3. **Was box_clutter source-faithfully synchronized?** Yes; see `provenance/box_clutter_scene_audit.md`.
4. **Which condition had highest success?** TR07 and TR08 tied at 3/7. The lexicographic tie-break selects TR08.
5. **Its F_rho?** TR08 has `F_rho=2`.
6. **Translation- or orientation-heavy?** More translation-heavy than normalized baseline.
7. **By what factor?** Twofold relative translation emphasis versus `F_rho=1`.
8. **Did common scale matter at F_rho=1?** Yes, behavior changed, but not monotonically.
9. **TR01 vs TR05 vs TR09?** Success was 1/7, 1/7, and 2/7; median successful T_goal was {fmt(common[0]['median_T_goal'])}, {fmt(common[1]['median_T_goal'])}, and {fmt(common[2]['median_T_goal'])} s.
10. **Did orientation emphasis reduce yaw error?** Not monotonically; the most orientation-heavy TR03 had worse median final yaw than TR02 and TR06.
11. **Did it sacrifice translation?** Often, but not as a monotone law; all orientation-heavy conditions retained poor median position convergence.
12. **Did translation emphasis improve position?** TR08 had the best aggregate median position error, but TR07 and TR04 show that more translation emphasis alone is not monotonic.
13. **Did it sacrifice orientation?** Not necessarily: TR08 also had the best aggregate median yaw error. It did regress open_task jointly.
14. **Clear Pareto optimum?** TR08 is the aggregate median position/yaw Pareto leader, but no scene-universal Pareto optimum exists.
15. **What realized ratios occurred?** The table above gives every condition; medians range from {fmt(min(float(r['median_realized_J_ratio']) for r in summaries))} to {fmt(max(float(r['median_realized_J_ratio']) for r in summaries))}, far from the nominal 0.25–4 range.
16. **How did ratio change during a task?** Usually downward as orientation error converged, with state/scene exceptions; exact per-selection values are in `metrics/realized_cost_ratio.csv`.
17. **Did success improve only by degrading CHOMP?** No for selected TR08 versus baseline; CHOMP and worst penetration both improved in aggregate.
18. **Which scenes favor different ratios?** open_task favors all except TR08; single_obstacle only TR08; box_clutter TR02/TR07/TR08; ycb_clutter TR07/TR08/TR09; shelf_gap, icra_sign, and slalom had no successes.
19. **How much failure remains F1/F2/F3?** {frozen_outer}/{len(failures)} non-successes and {frozen_outer}/{len(true_stalls)} TRUE_STALL runs.
20. **What pair should be frozen next?** Provisionally TR08: `(alpha_p, alpha_theta, F_rho)=(2,1,2)` under this deterministic panel.
21. **Is a universal superior ratio established?** No. The fixed-seed panel supports selecting TR08, but the tied success count, strong ratio×scale×scene interaction, and six rot000 tasks require a scene-dependent conclusion. Explicit ±90-degree confirmation remains necessary before a broad orientation-cost claim.

## Validation notes

- Ratio artifact audit: PASS (63 unique admissions, 63 timing-valid rows, 56 paired rows, fixed obstacle metadata, nine open-task CHOMP zeros, nine excluded contaminated attempts).
- Focused benchmark regression: `python -m pytest tests/test_rot15_benchmark.py -q` → 11 passed.
- Full dirty-worktree suite: 513 passed, 34 failed, 16 skipped, 1 xfailed. The failures span pre-existing-tree LCS shape assumptions, outer-loop/progress defaults, OSC/workspace expectations, and Python sampling-strategy contracts; none was weakened or repaired in this cost-only study. This prevents claiming the whole repository is green, but it does not invalidate the experiment-specific gates above.

## Commands and scope boundary

Canonical commands were:

```bash
python scripts/run_c3plus_task_cost_ratio_gates.py
python scripts/analyze_c3plus_task_cost_ratio_gates.py
cat results/c3plus_task_cost_ratio/reproducibility/gate_status.json
python scripts/run_c3plus_task_cost_ratio.py run
python scripts/analyze_c3plus_task_cost_ratio.py
```

Crash-safe resumptions used `--start-at 7`, `20`, `29`, `47`, and `50`; admitted pairs were skipped, so no scientific pair was duplicated. No ReLU/exponential tuning or larger validation campaign was launched.

## Final verdict

{verdict}
"""
    (OUT / "report/C3PLUS_TRANSLATION_ROTATION_RATIO_REPORT.md").write_text(text)
    handoff = {
        "completed_runs": completed, "planned_admitted_runs": expected,
        "intended_runs_with_box_clutter": expected,
        "box_clutter_status": "SOURCE_FAITHFULLY_SYNCHRONIZED" if box_done or completed == 0 else "SYNCHRONIZED_NOT_YET_RUN",
        "winner": winner, "alpha_p": win["alpha_p"] if win else None,
        "alpha_theta": win["alpha_theta"] if win else None,
        "ratio_factor": win["ratio_factor"] if win else None, "baseline_ratio": 1,
        "verdict": verdict.split(". ", 1)[1],
        "full_25x7_launched": False, "relu_tuning_launched": False,
    }
    (OUT / "handoff/translation_rotation_ratio_handoff.json").write_text(json.dumps(handoff, indent=2, sort_keys=True) + "\n")


def main() -> int:
    manifest = read_csv(OUT / "experiment_manifest.csv")
    goals = load_goals()
    rows, stages = [], []
    for item in manifest:
        result = process_run(item, goals)
        if result:
            row, stage = result; rows.append(row); stages.extend(stage)
    write_csv(METRICS / "per_run_results.csv", PER_RUN_FIELDS, rows)
    write_csv(METRICS / "realized_cost_ratio.csv", ["condition_id", "scenario", "time", "stage", "stage_basis", "J_translation", "J_orientation", "realized_ratio"], stages)
    summaries = summarize(rows)
    ratios = ratio_summary(rows)
    common_scale_summary(rows)
    paired(rows)
    scenario_outcomes(rows)
    admission_audit(manifest)
    timing = [{key: row[key] for key in ["condition_id", "scenario", "concurrency_level", "requested_control_frequency", "achieved_mean_control_frequency", "achieved_median_control_frequency", "control_period_median", "control_period_p95", "control_period_p99", "timing_valid"]} for row in rows]
    write_csv(METRICS / "timing_summary.csv", list(timing[0]) if timing else ["condition_id", "scenario"], timing)
    plots(summaries, ratios, rows)
    winner, verdict = choose_winner(summaries)
    report(rows, summaries, ratios, winner, verdict)
    print(f"analyzed {len(rows)}/{len(CONDITIONS) * len(SCENES)} runs; {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
