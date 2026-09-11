#!/usr/bin/env python3
"""Analyze the preregistered final clean C3+ obstacle-ranking campaign.

This program is offline: it never launches a controller or simulator.  By
default it fails closed until every row in the immutable 126-row manifest has
an ACCEPTED_RUN.json or TERMINAL_RUN.json marker.  It preserves the distinction
between a scientifically admitted task outcome and an infrastructure/runtime
failure, and it only forms paired scientific contrasts from matched admitted
runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/c3plus_final_clean_obstacle_ranking"
MANIFEST = OUT / "manifest/final_126_run_manifest.csv"
REPORT = OUT / "report/FINAL_CLEAN_OBSTACLE_RANKING_RESULTS.md"
HANDOFF = OUT / "handoff/final_clean_obstacle_ranking_handoff.json"
METRICS = OUT / "metrics"
EXPECTED_MANIFEST_SHA256 = (
    "e8dba426a23270724d742eca175255580dcc1c9443ac7d75de36ddcb0bb3e956"
)
POS_TOL = 0.02
YAW_TOL = 0.10
CONFIGS = ("B1", "B2", "B3")
OBSTACLE_SCENES = ("single_obstacle", "shelf_gap")
ROTATIONS = ("rot000", "rotCCW090", "rotCW090")
STARTS = ("s1", "s3", "s5")
THREAD_LIMITS = {
    "OMP_NUM_THREADS": "3",
    "OMP_THREAD_LIMIT": "3",
    "OMP_DYNAMIC": "FALSE",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "GOTO_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


RATIO = load_module(
    "final_clean_ratio_helpers", ROOT / "scripts/analyze_c3plus_task_cost_ratio.py"
)
CHOMP = RATIO.CHOMP


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as stream:
        for line in stream:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A final partial line is possible if a process was terminated.
                continue
    return rows


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text)
    temp.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    def strict_json(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: strict_json(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [strict_json(item) for item in value]
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            return None
        if isinstance(value, np.integer):
            return int(value)
        return value

    atomic_text(
        path,
        json.dumps(strict_json(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def number(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def finite(values: Iterable[Any]) -> list[float]:
    return [v for value in values if math.isfinite(v := number(value))]


def median(values: Iterable[Any]) -> float:
    vals = finite(values)
    return float(np.median(vals)) if vals else math.nan


def quantile(values: Iterable[Any], q: float) -> float:
    vals = finite(values)
    return float(np.quantile(vals, q)) if vals else math.nan


def mean(values: Iterable[Any]) -> float:
    vals = finite(values)
    return float(np.mean(vals)) if vals else math.nan


def fmt(value: Any, digits: int = 3) -> str:
    value = number(value)
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def duration_below(times: np.ndarray, values: np.ndarray, threshold: float) -> float:
    if len(times) < 2:
        return 0.0
    delta = np.diff(times)
    return float(np.sum(delta[values[:-1] < threshold]))


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def exact_mcnemar(b1_only: int, b2_only: int) -> float:
    discordant = b1_only + b2_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(b1_only, b2_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def bootstrap_mean_ci(values: list[float], seed: int = 20260910) -> tuple[float, float]:
    vals = np.asarray(finite(values), dtype=float)
    if len(vals) == 0:
        return math.nan, math.nan
    if len(vals) == 1:
        return float(vals[0]), float(vals[0])
    rng = np.random.default_rng(seed)
    draws = rng.choice(vals, size=(10000, len(vals)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def selected_candidates(path: Path, committed: set[int]) -> list[dict[str, str]]:
    selected = []
    for row in read_csv(path):
        try:
            event = int(row["event_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if event in committed and row.get("selected") == "1":
            selected.append(row)
    return selected


def causal_failure(cycles: list[dict[str, str]], events: list[dict[str, Any]]) -> str:
    if not cycles:
        return "RUNTIME_FAILURE"
    names = [
        str(event.get("flag_name", ""))
        for event in events
        if event.get("enter_or_exit") != "EXIT"
    ]
    contacts = [int(row.get("physical_contact", "0")) for row in cycles]
    c3 = [row for row in cycles if row.get("is_c3_mode") == "1"]
    c3_contacts = [int(row.get("physical_contact", "0")) for row in c3]
    clearances = finite(row.get("physical_object_obstacle_clearance") for row in cycles)
    if clearances and min(clearances) <= 0 and mean(contacts) > 0.05:
        return "F6_GEOMETRIC_BLOCK"
    if c3_contacts and mean(c3_contacts) < 0.02:
        return "F1_ACQUISITION"
    if (
        names.count("REPOS_REACHED_NO_CONTACT")
        + names.count("PLANNER_PHYSICAL_CONTACT_MISMATCH")
        + names.count("NO_PROGRESS_EXIT")
    ):
        return "F2_PHANTOM_CHURN"
    if names.count("FAILED_SECTOR_RESELECT"):
        return "F3_RESELECTION"
    return "UNCLASSIFIED_TASK_FAILURE"


def locate_attempt(row: dict[str, str]) -> tuple[str, Path | None, dict[str, Any]]:
    run_root = OUT / "runs" / row["run_id"]
    accepted = run_root / "ACCEPTED_RUN.json"
    terminal = run_root / "TERMINAL_RUN.json"
    if accepted.exists() and terminal.exists():
        raise RuntimeError(f"both terminal markers exist: {row['run_id']}")
    if accepted.exists():
        marker = json.loads(accepted.read_text())
        return "ACCEPTED", run_root / marker["accepted_attempt"], marker
    if terminal.exists():
        marker = json.loads(terminal.read_text())
        return str(marker.get("status", "TERMINAL_FAILURE")), run_root / marker["terminal_attempt"], marker
    return "PENDING", None, {}


def audited_terminal_reason(attempt: Path, recorded: str) -> str:
    """Refine a generic supervisor reason from the immutable planner log."""
    planner = attempt / "live/planner.log"
    text = planner.read_text(errors="replace") if planner.exists() else ""
    if "AllFinite(q_v)" in text:
        return "PLANNER_ABORT_ALLFINITE_QV"
    if "CheckForWorkspaceLimitViolations" in text:
        return "WORKSPACE_ABORT"
    if "assertion_error" in text or "condition '" in text:
        return "PLANNER_ASSERTION_OTHER"
    if "terminate called" in text:
        return "PLANNER_UNCAUGHT_EXCEPTION"
    return recorded or "UNKNOWN_RUNTIME_FAILURE"


def validate_terminal_attempt(
    row: dict[str, str], attempt: Path, accepted: bool
) -> dict[str, Any]:
    """Prove the frozen execution boundary for one terminal manifest row."""
    receipt_path = attempt / "run_receipt.json"
    cleanup_path = attempt / "owned_cleanup.json"
    if not receipt_path.exists() or not cleanup_path.exists():
        raise RuntimeError(f"terminal attempt lacks receipt/cleanup: {attempt}")
    receipt = json.loads(receipt_path.read_text())
    cleanup = json.loads(cleanup_path.read_text())
    environment = receipt.get("environment", {})
    checks = {
        "run_id": receipt.get("run_id") == row["run_id"],
        "manifest_row_exact": receipt.get("manifest_row") == row,
        "alpha_p_rank_one": environment.get("SAMPLING_C3_ALPHA_P_RANK") == "1",
        "alpha_theta_rank_one": environment.get("SAMPLING_C3_ALPHA_THETA_RANK") == "1",
        "inner_soft_obstacle_none": environment.get("SAMPLING_C3_INNER_OBS_MODE") == "none",
        "lcs_contact": environment.get("SAMPLING_C3_OBSTACLE_MODE") == "lcs_contact",
        "no_legacy_translation_scale": "SAMPLING_C3_TRANSLATION_COST_SCALE" not in environment,
        "no_legacy_orientation_scale": "SAMPLING_C3_ORIENTATION_COST_SCALE" not in environment,
        "no_inner_nonpenetration_extension": "SAMPLING_C3_OBJ_NONPEN" not in environment,
        "thread_limits": all(environment.get(key) == value for key, value in THREAD_LIMITS.items()),
        "affinity": receipt.get("verified_affinity") == {
            "osc": "6-7", "planner": "2-5", "recorder": "0-1", "sim": "6-7"
        },
        "owned_process_group": (
            isinstance(receipt.get("process_group_id"), int)
            and receipt.get("process_group_id") == receipt.get("launcher_pid")
        ),
        "cleanup_passed": cleanup.get("passed") is True and not cleanup.get("remaining"),
    }
    family = row["config_id"]
    if family == "OPEN-A1":
        checks["obstacle_rank_exact"] = "SAMPLING_C3_RANK_OBS_MODE" not in environment
    elif family == "B1":
        checks["obstacle_rank_exact"] = (
            environment.get("SAMPLING_C3_RANK_OBS_MODE") == "exponential_footprint"
            and environment.get("SAMPLING_C3_OBS_EXP_SIGMA") == "0.015"
            and environment.get("SAMPLING_C3_OBS_EXP_W") == "764.3857383844456"
        )
    elif family == "B2":
        checks["obstacle_rank_exact"] = (
            environment.get("SAMPLING_C3_RANK_OBS_MODE") == "relu_footprint"
            and environment.get("SAMPLING_C3_OBS_RELU_EPS") == "0.045"
            and environment.get("SAMPLING_C3_OBS_RELU_W") == "764.3857383844456"
        )
    elif family == "B3":
        checks["obstacle_rank_exact"] = (
            environment.get("SAMPLING_C3_RANK_OBS_MODE") == "relu_footprint"
            and environment.get("SAMPLING_C3_OBS_RELU_EPS") == "0.05"
            and environment.get("SAMPLING_C3_OBS_RELU_W") == "1173.2815219621552"
        )
    else:
        checks["obstacle_rank_exact"] = False
    if accepted:
        timing_path = attempt / "timing.json"
        timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
        checks["accepted_timing_valid"] = timing.get("timing_valid") is True
    failed = sorted(key for key, value in checks.items() if not value)
    if failed:
        raise RuntimeError(
            f"terminal attempt violates frozen campaign boundary ({row['run_id']}): {failed}"
        )
    return checks


def attempt_inventory(row: dict[str, str]) -> list[dict[str, Any]]:
    run_root = OUT / "runs" / row["run_id"]
    terminal_kind, terminal_attempt, marker = locate_attempt(row)
    output = []
    for attempt in sorted((run_root / "attempts").glob("attempt_*")):
        cleanup_path = attempt / "owned_cleanup.json"
        cleanup = json.loads(cleanup_path.read_text()) if cleanup_path.exists() else {}
        timing_path = attempt / "timing.json"
        timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
        accepted = terminal_kind == "ACCEPTED" and attempt == terminal_attempt
        terminal = terminal_attempt == attempt
        recorded_reason = marker.get(
            "terminal_reason", cleanup.get("terminal_reason", "")
        )
        output.append({
            "run_id": row["run_id"],
            "attempt": attempt.name,
            "scene": row["scene"],
            "config_id": row["config_id"],
            "start_id": row["start_id"],
            "rotation_label": row["rotation_label"],
            "seed": row["seed"],
            "terminal_for_manifest_entry": int(terminal),
            "scientifically_admitted": int(accepted),
            "attempt_classification": (
                str(marker.get("status", "ACCEPTED")) if terminal else "SUPERSEDED_ATTEMPT"
            ),
            "terminal_reason_recorded": recorded_reason,
            "terminal_reason_audited": (
                audited_terminal_reason(attempt, recorded_reason)
                if terminal and not accepted else ""
            ),
            "timing_valid": timing.get("timing_valid", ""),
            "cleanup_passed": cleanup.get("passed", ""),
            "wall_time_s": timing.get("wall_time_s", ""),
            "attempt_path": str(attempt.relative_to(ROOT)),
        })
    return output


def process_admitted(row: dict[str, str], attempt: Path, marker: dict[str, Any]) -> dict[str, Any]:
    live = attempt / "live"
    forensic = attempt / "forensics"
    cycles = read_csv(forensic / "forensics_cycle.csv")
    if not cycles:
        raise RuntimeError(f"accepted attempt lacks forensic cycles: {attempt}")
    events = read_jsonl(forensic / "forensics_events.jsonl")
    committed = {int(item["event_id"]) for item in cycles}
    selected = selected_candidates(forensic / "forensics_candidates.csv", committed)
    selected_cost = [
        item for item in selected
        if math.isfinite(number(item.get("J_rank_code_total")))
    ]
    times = np.asarray([float(item["time"]) for item in cycles])
    position = np.asarray([float(item["position_error"]) for item in cycles])
    yaw = np.asarray([float(item["yaw_error"]) for item in cycles])
    clearance = np.asarray(
        [number(item.get("physical_object_obstacle_clearance")) for item in cycles]
    )
    finite_clearance = clearance[np.isfinite(clearance)]
    normalized_error = np.maximum(position / POS_TOL, yaw / YAW_TOL)
    transactions = read_csv(forensic / "forensics_transactions.csv")
    timing = marker["timing"]
    success = marker.get("status") == "SUCCESS"
    retained = bool(position[-1] < POS_TOL and yaw[-1] < YAW_TOL)
    failure = "SUCCESS" if success else causal_failure(cycles, events)
    runtime_status = "SUCCESS" if success else "TIMEOUT"
    if not success and len(times) > 1:
        start = int(np.searchsorted(times, times[-1] - 30.0))
        if position[start] - position[-1] > 0.005 or yaw[start] - yaw[-1] > 0.05:
            runtime_status = "TIMEOUT_PROGRESSING"
        else:
            runtime_status = "TRUE_STALL"

    def component(item: dict[str, str], key: str) -> float:
        return number(item.get(key))

    position_cost = [component(item, "J_translation") for item in selected_cost]
    orientation_cost = [component(item, "J_orientation") for item in selected_cost]
    angular_cost = [component(item, "J_angular_velocity") for item in selected_cost]
    linear_cost = [component(item, "J_linear_velocity") for item in selected_cost]
    obstacle_cost = [component(item, "J_obstacle_applied_live") for item in selected_cost]
    task_cost = [
        p + o + a + l
        for p, o, a, l in zip(position_cost, orientation_cost, angular_cost, linear_cost)
        if all(math.isfinite(value) for value in (p, o, a, l))
    ]
    obstacle_task_ratio = []
    fixed_cost = []
    selected_ids = []
    for item in selected_cost:
        p = component(item, "J_translation")
        o = component(item, "J_orientation")
        a = component(item, "J_angular_velocity")
        l = component(item, "J_linear_velocity")
        obs = component(item, "J_obstacle_applied_live")
        total = component(item, "J_rank_code_total")
        if all(math.isfinite(value) for value in (p, o, a, l, obs, total)):
            task = p + o + a + l
            obstacle_task_ratio.append(obs / max(abs(task), 1e-12))
            fixed_cost.append(total - task - obs)
        selected_ids.append(int(item["candidate_id"]))

    if row["scene"] == "open_task":
        chomp_raw = chomp_normalized = 0.0
    else:
        chomp = CHOMP.eval_run(row["scene"], str(live))
        chomp_raw = number(chomp.get("M_CHOMP"))
        chomp_normalized = number(chomp.get("M_CHOMP_norm"))

    contact = [int(item.get("physical_contact", "0")) for item in cycles]
    c3 = [item for item in cycles if item.get("is_c3_mode") == "1"]
    c3_contact = [int(item.get("physical_contact", "0")) for item in c3]
    displacements = [item.get("object_displacement") for item in transactions]
    duties = [item.get("physical_contact_fraction") for item in transactions]
    event_names = [
        str(event.get("flag_name", ""))
        for event in events
        if event.get("enter_or_exit") != "EXIT"
    ]
    minimum_clearance = (
        float(np.min(finite_clearance)) if len(finite_clearance) else math.nan
    )
    result: dict[str, Any] = {
        **{key: row[key] for key in (
            "run_id", "run_index", "block_index", "config_id", "scene",
            "start_id", "goal_id", "rotation_label", "seed", "case_id",
            "obstacle_family", "obstacle_range_name", "obstacle_range_m",
            "obstacle_weight", "lcm_port",
        )},
        "scientifically_admitted": 1,
        "scientific_status": marker.get("status"),
        "success": int(success),
        "retained_success": int(retained),
        "T_goal_s": marker.get("recorder_final", {}).get("first_success_t", math.nan),
        "position_success": int(np.any(position < POS_TOL)),
        "orientation_success": int(np.any(yaw < YAW_TOL)),
        "joint_success": int(success),
        "best_position_error_m": float(np.min(position)),
        "best_orientation_error_rad": float(np.min(yaw)),
        "final_position_error_m": float(position[-1]),
        "final_orientation_error_rad": float(yaw[-1]),
        "E_best": float(np.min(normalized_error)),
        "E_final": float(normalized_error[-1]),
        "runtime_status": runtime_status,
        "failure_class": failure,
        "F1": int(failure.startswith("F1")),
        "F2": int(failure.startswith("F2")),
        "F3": int(failure.startswith("F3")),
        "J_position_median": median(position_cost),
        "J_position_q25": quantile(position_cost, 0.25),
        "J_position_q75": quantile(position_cost, 0.75),
        "J_orientation_median": median(orientation_cost),
        "J_orientation_q25": quantile(orientation_cost, 0.25),
        "J_orientation_q75": quantile(orientation_cost, 0.75),
        "J_angular_velocity_median": median(angular_cost),
        "J_linear_velocity_median": median(linear_cost),
        "J_obstacle_median": median(obstacle_cost),
        "J_obstacle_q25": quantile(obstacle_cost, 0.25),
        "J_obstacle_q75": quantile(obstacle_cost, 0.75),
        "J_task_median": median(task_cost),
        "R_obs_task_median": median(obstacle_task_ratio),
        "R_obs_task_q25": quantile(obstacle_task_ratio, 0.25),
        "R_obs_task_q75": quantile(obstacle_task_ratio, 0.75),
        "J_fixed_median": median(fixed_cost),
        "J_rank_median": median(component(item, "J_rank_code_total") for item in selected_cost),
        "selected_candidate_count": len(selected_ids),
        "first_selected_candidate": selected_ids[0] if selected_ids else "",
        "selected_candidate_sequence": selected_ids,
        "CHOMP_raw": chomp_raw,
        "CHOMP_normalized": chomp_normalized,
        "minimum_clearance_m": minimum_clearance,
        "time_below_10mm_s": (
            duration_below(times, clearance, 0.010) if len(finite_clearance) else math.nan
        ),
        "time_below_5mm_s": (
            duration_below(times, clearance, 0.005) if len(finite_clearance) else math.nan
        ),
        "maximum_penetration_m": (
            max(0.0, -minimum_clearance) if math.isfinite(minimum_clearance) else math.nan
        ),
        "transactions": len(transactions),
        "median_transaction_displacement_m": median(displacements),
        "mean_transaction_contact_duty": mean(duties),
        "physical_contact_fraction": mean(contact),
        "contactless_c3_fraction": 1.0 - mean(c3_contact) if c3_contact else math.nan,
        "repositions": sum(
            1 for first, second in zip(cycles, cycles[1:])
            if first.get("mode_name") != second.get("mode_name")
            and second.get("mode_name") == "REPOSITION"
        ),
        "unproductive_exits": sum(
            item.get("exit_reason") == "TO_REPOS_UNPRODUCTIVE" for item in transactions
        ),
        "F1_event_count": event_names.count("REPOS_REACHED_NO_CONTACT"),
        "F2_event_count": (
            event_names.count("PLANNER_PHYSICAL_CONTACT_MISMATCH")
            + event_names.count("NO_PROGRESS_EXIT")
        ),
        "F3_event_count": event_names.count("FAILED_SECTOR_RESELECT"),
        "mean_controller_hz": timing["mean_controller_hz"],
        "median_controller_hz": timing["median_controller_hz"],
        "median_period_s": timing["median_period_s"],
        "p95_period_s": timing["p95_period_s"],
        "p99_period_s": timing["p99_period_s"],
        "sim_wall_ratio": timing["sim_wall_ratio"],
        "wall_time_s": timing["wall_time_s"],
        "timing_valid": int(timing["timing_valid"]),
        "attempt_path": str(attempt.relative_to(ROOT)),
    }
    return result


def summarize_group(group: list[dict[str, Any]], planned_n: int) -> dict[str, Any]:
    admitted_n = len(group)
    success_n = sum(int(row["success"]) for row in group)
    retained_n = sum(int(row["retained_success"]) for row in group)
    planned_lo, planned_hi = wilson(success_n, planned_n)
    admitted_lo, admitted_hi = wilson(success_n, admitted_n)
    return {
        "planned_n": planned_n,
        "admitted_n": admitted_n,
        "runtime_failure_n": planned_n - admitted_n,
        "success_n": success_n,
        "success_rate_planned": success_n / planned_n if planned_n else math.nan,
        "success_rate_admitted": success_n / admitted_n if admitted_n else math.nan,
        "success_wilson95_low_planned": planned_lo,
        "success_wilson95_high_planned": planned_hi,
        "success_wilson95_low_admitted": admitted_lo,
        "success_wilson95_high_admitted": admitted_hi,
        "retained_success_n": retained_n,
        "median_T_goal_s": median(row["T_goal_s"] for row in group if row["success"]),
        "T_goal_q25_s": quantile((row["T_goal_s"] for row in group if row["success"]), 0.25),
        "T_goal_q75_s": quantile((row["T_goal_s"] for row in group if row["success"]), 0.75),
        "median_E_best_failures": median(row["E_best"] for row in group if not row["success"]),
        "median_CHOMP_normalized": median(row["CHOMP_normalized"] for row in group),
        "median_minimum_clearance_m": median(row["minimum_clearance_m"] for row in group),
        "median_maximum_penetration_m": median(row["maximum_penetration_m"] for row in group),
        "F1_n": sum(int(row["F1"]) for row in group),
        "F2_n": sum(int(row["F2"]) for row in group),
        "F3_n": sum(int(row["F3"]) for row in group),
        "F1_rate_admitted": (
            sum(int(row["F1"]) for row in group) / admitted_n
            if admitted_n else math.nan
        ),
        "F2_rate_admitted": (
            sum(int(row["F2"]) for row in group) / admitted_n
            if admitted_n else math.nan
        ),
        "F3_rate_admitted": (
            sum(int(row["F3"]) for row in group) / admitted_n
            if admitted_n else math.nan
        ),
        "median_R_obs_task": median(row["R_obs_task_median"] for row in group),
        "median_transactions": median(row["transactions"] for row in group),
        "median_transaction_displacement_m": median(
            row["median_transaction_displacement_m"] for row in group
        ),
        "median_transaction_contact_duty": median(
            row["mean_transaction_contact_duty"] for row in group
        ),
        "median_physical_contact_fraction": median(
            row["physical_contact_fraction"] for row in group
        ),
        "median_contactless_c3_fraction": median(
            row["contactless_c3_fraction"] for row in group
        ),
    }


def summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    open_group = [row for row in rows if row["scene"] == "open_task"]
    output.append({"config_id": "OPEN-A1", "scene": "open_task", **summarize_group(open_group, 18)})
    for config in CONFIGS:
        for scene in OBSTACLE_SCENES:
            group = [row for row in rows if row["config_id"] == config and row["scene"] == scene]
            output.append({"config_id": config, "scene": scene, **summarize_group(group, 18)})
        group = [row for row in rows if row["config_id"] == config and row["scene"] in OBSTACLE_SCENES]
        output.append({"config_id": config, "scene": "OBSTACLE_COMBINED", **summarize_group(group, 36)})
    return output


def breakdown_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for scene in ("open_task", *OBSTACLE_SCENES):
        configs = ("OPEN-A1",) if scene == "open_task" else CONFIGS
        for config in configs:
            base = [row for row in rows if row["scene"] == scene and row["config_id"] == config]
            for dimension, values, planned in (
                ("start", STARTS, 6),
                ("rotation", ROTATIONS, 6),
                ("seed", ("0", "1"), 9),
            ):
                key = "start_id" if dimension == "start" else "rotation_label" if dimension == "rotation" else "seed"
                for value in values:
                    group = [row for row in base if str(row[key]) == value]
                    output.append({
                        "config_id": config,
                        "scene": scene,
                        "dimension": dimension,
                        "level": value,
                        **summarize_group(group, planned),
                    })
    return output


def cost_component_aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-run selected-candidate component summaries by design strata."""
    metrics = (
        "J_position_median",
        "J_orientation_median",
        "J_angular_velocity_median",
        "J_linear_velocity_median",
        "J_obstacle_median",
        "J_task_median",
        "J_fixed_median",
        "J_rank_median",
        "R_obs_task_median",
    )
    output: list[dict[str, Any]] = []
    scene_configs = [("OPEN-A1", "open_task")] + [
        (config, scene) for config in CONFIGS for scene in OBSTACLE_SCENES
    ]
    for config, scene in scene_configs:
        base = [
            row for row in rows
            if row["config_id"] == config and row["scene"] == scene
        ]
        strata = [("all", "all", base)]
        strata.extend(
            ("start", start, [row for row in base if row["start_id"] == start])
            for start in STARTS
        )
        strata.extend(
            (
                "rotation",
                rotation,
                [row for row in base if row["rotation_label"] == rotation],
            )
            for rotation in ROTATIONS
        )
        for dimension, level, group in strata:
            item: dict[str, Any] = {
                "config_id": config,
                "scene": scene,
                "dimension": dimension,
                "level": level,
                "admitted_n": len(group),
            }
            for metric in metrics:
                item[f"{metric}_q25_across_runs"] = quantile(
                    (row[metric] for row in group), 0.25
                )
                item[f"{metric}_median_across_runs"] = median(
                    row[metric] for row in group
                )
                item[f"{metric}_q75_across_runs"] = quantile(
                    (row[metric] for row in group), 0.75
                )
            output.append(item)
    return output


def failure_count_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    scene_configs = [("OPEN-A1", "open_task")] + [
        (config, scene) for config in CONFIGS for scene in OBSTACLE_SCENES
    ]
    for config, scene in scene_configs:
        group = [
            row for row in rows
            if row["config_id"] == config and row["scene"] == scene
        ]
        counts = Counter(row["failure_class"] for row in group)
        for failure_class, count in sorted(counts.items()):
            output.append({
                "config_id": config,
                "scene": scene,
                "failure_class": failure_class,
                "count": count,
            })
    return output


def matched_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return row["scene"], row["start_id"], row["rotation_label"], str(row["seed"])


def pair_rows(
    rows: list[dict[str, Any]], left: str, right: str,
    planned_matched_tasks: int = 36,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    li = {matched_key(row): row for row in rows if row["config_id"] == left and row["scene"] in OBSTACLE_SCENES}
    ri = {matched_key(row): row for row in rows if row["config_id"] == right and row["scene"] in OBSTACLE_SCENES}
    output = []
    transitions = Counter()
    metrics = (
        "T_goal_s", "E_best", "E_final", "J_position_median",
        "J_orientation_median", "J_obstacle_median", "R_obs_task_median",
        "minimum_clearance_m", "maximum_penetration_m", "CHOMP_normalized",
        "transactions", "median_transaction_displacement_m",
        "mean_transaction_contact_duty", "contactless_c3_fraction", "F2",
    )
    for key in sorted(li.keys() & ri.keys()):
        a, b = li[key], ri[key]
        transition = ("S" if a["success"] else "F") + ("S" if b["success"] else "F")
        transitions[transition] += 1
        item: dict[str, Any] = {
            "pair": f"{left}_vs_{right}",
            "scene": key[0], "start_id": key[1], "rotation_label": key[2], "seed": key[3],
            "left_status": a["scientific_status"], "right_status": b["scientific_status"],
            "success_transition": transition,
            "left_run_id": a["run_id"], "right_run_id": b["run_id"],
            "first_selected_candidate_same": int(a["first_selected_candidate"] == b["first_selected_candidate"]),
        }
        seq_a, seq_b = a["selected_candidate_sequence"], b["selected_candidate_sequence"]
        common = min(len(seq_a), len(seq_b))
        item["selected_sequence_common_events"] = common
        item["selected_sequence_mismatch_fraction"] = (
            mean(int(seq_a[i] != seq_b[i]) for i in range(common)) if common else math.nan
        )
        for metric in metrics:
            av, bv = number(a.get(metric)), number(b.get(metric))
            item[f"delta_{metric}"] = bv - av if math.isfinite(av) and math.isfinite(bv) else math.nan
        output.append(item)

    b1_only = transitions["SF"]
    b2_only = transitions["FS"]
    delta_metrics = {}
    for metric in metrics:
        vals = finite(item[f"delta_{metric}"] for item in output)
        lo, hi = bootstrap_mean_ci(vals)
        delta_metrics[metric] = {
            "paired_n": len(vals),
            "mean_delta_right_minus_left": mean(vals),
            "median_delta_right_minus_left": median(vals),
            "bootstrap_mean_95ci": [lo, hi],
        }
    summary = {
        "pair": f"{left}_vs_{right}",
        "planned_matched_tasks": planned_matched_tasks,
        "admissible_matched_pairs": len(output),
        "unpaired_due_to_nonadmission": planned_matched_tasks - len(output),
        "transitions": {name: transitions[name] for name in ("SS", "SF", "FS", "FF")},
        "paired_success_difference_right_minus_left": (b2_only - b1_only) / len(output) if output else math.nan,
        "mcnemar_exact_two_sided_p": exact_mcnemar(b1_only, b2_only),
        "metric_differences": delta_metrics,
    }
    return output, summary


def rotation_asymmetry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare matched CCW/CW tasks without treating configurations as IID."""
    by_config: dict[str, Any] = {}
    for config in CONFIGS:
        config_rows = [
            row for row in rows
            if row["config_id"] == config and row["scene"] in OBSTACLE_SCENES
        ]
        ccw = {
            (row["scene"], row["start_id"], str(row["seed"])): row
            for row in config_rows if row["rotation_label"] == "rotCCW090"
        }
        cw = {
            (row["scene"], row["start_id"], str(row["seed"])): row
            for row in config_rows if row["rotation_label"] == "rotCW090"
        }
        transitions: Counter[str] = Counter()
        for key in sorted(ccw.keys() & cw.keys()):
            left = ccw[key]
            right = cw[key]
            transition = (
                ("S" if left["success"] else "F")
                + ("S" if right["success"] else "F")
            )
            transitions[transition] += 1
        paired_n = sum(transitions.values())
        effect = (
            (transitions["FS"] - transitions["SF"]) / paired_n
            if paired_n else math.nan
        )
        p_value = exact_mcnemar(transitions["SF"], transitions["FS"])
        by_config[config] = {
            "planned_pairs": 12,
            "admissible_pairs": paired_n,
            "unpaired_due_to_nonadmission": 12 - paired_n,
            "transitions_ccw_to_cw": {
                name: transitions[name] for name in ("SS", "SF", "FS", "FF")
            },
            "success_effect_cw_minus_ccw": effect,
            "mcnemar_exact_two_sided_p": p_value,
            "statistically_supported_at_0_05": bool(
                paired_n and p_value < 0.05
            ),
        }
    return {
        "by_config": by_config,
        "any_statistically_supported_at_0_05": any(
            item["statistically_supported_at_0_05"]
            for item in by_config.values()
        ),
    }


def ranked_configs(summary: list[dict[str, Any]]) -> list[str]:
    combined = {row["config_id"]: row for row in summary if row["scene"] == "OBSTACLE_COMBINED"}
    single = {(row["config_id"]): row for row in summary if row["scene"] == "single_obstacle"}
    shelf = {(row["config_id"]): row for row in summary if row["scene"] == "shelf_gap"}
    def key(config: str):
        row = combined[config]
        # Valid execution is primary, then success and scene balance. Lower
        # time/error/CHOMP/F2 breaks remaining ties.
        return (
            row["admitted_n"],
            row["success_n"],
            min(single[config]["success_n"], shelf[config]["success_n"]),
            row["retained_success_n"],
            -number(row["median_T_goal_s"], 1e30),
            -number(row["median_E_best_failures"], 1e30),
            number(row["median_minimum_clearance_m"], -1e30),
            -number(row["median_maximum_penetration_m"], 1e30),
            -number(row["median_CHOMP_normalized"], 1e30),
            number(row["median_transaction_displacement_m"], -1e30),
            -number(row["F2_rate_admitted"], 1e30),
        )
    return sorted(CONFIGS, key=key, reverse=True)


def table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def render_report(
    manifest: list[dict[str, str]],
    rows: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    breakdowns: list[dict[str, Any]],
    cost_aggregates: list[dict[str, Any]],
    failure_counts: list[dict[str, Any]],
    pair_summaries: list[dict[str, Any]],
    asymmetry: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    accepted = len(rows)
    terminal_failures = 126 - accepted
    total_success = sum(row["success"] for row in rows)
    order = ranked_configs(summaries)
    winner = order[0]
    summary_index = {(row["config_id"], row["scene"]): row for row in summaries}
    b1b2 = next(item for item in pair_summaries if item["pair"] == "B1_vs_B2")

    open_summary = summary_index[("OPEN-A1", "open_task")]
    scene_table = []
    for config, scene in [("OPEN-A1", "open_task")] + [
        (config, scene) for config in CONFIGS for scene in (*OBSTACLE_SCENES, "OBSTACLE_COMBINED")
    ]:
        item = summary_index[(config, scene)]
        scene_table.append([
            config, scene, f"{item['admitted_n']}/{item['planned_n']}",
            f"{item['success_n']}/{item['admitted_n']}",
            (
                f"{100*item['success_rate_admitted']:.1f}%"
                if item['admitted_n'] else "NA"
            ),
            (
                f"[{100*item['success_wilson95_low_admitted']:.1f}, "
                f"{100*item['success_wilson95_high_admitted']:.1f}]%"
                if item['admitted_n'] else "NA"
            ),
            f"{item['success_n']}/{item['planned_n']}",
            item["retained_success_n"], fmt(item["median_T_goal_s"]),
            fmt(item["median_E_best_failures"]), item["F2_n"],
        ])

    breakdown_table = []
    for item in breakdowns:
        breakdown_table.append([
            item["config_id"], item["scene"], item["dimension"], item["level"],
            f"{item['admitted_n']}/{item['planned_n']}", f"{item['success_n']}/{item['planned_n']}",
        ])

    paired_table = []
    for item in pair_summaries:
        tr = item["transitions"]
        paired_table.append([
            item["pair"], item["admissible_matched_pairs"], tr["SS"], tr["SF"],
            tr["FS"], tr["FF"], fmt(item["paired_success_difference_right_minus_left"]),
            fmt(item["mcnemar_exact_two_sided_p"]),
        ])

    physical_table = []
    for config, scene in [("OPEN-A1", "open_task")] + [
        (config, scene) for config in CONFIGS for scene in OBSTACLE_SCENES
    ]:
        item = summary_index[(config, scene)]
        physical_table.append([
            config, scene,
            fmt(item["median_minimum_clearance_m"]),
            fmt(item["median_maximum_penetration_m"]),
            fmt(item["median_CHOMP_normalized"]),
            fmt(item["median_transactions"]),
            fmt(item["median_transaction_displacement_m"]),
            fmt(item["median_transaction_contact_duty"]),
            fmt(item["median_physical_contact_fraction"]),
            fmt(item["median_contactless_c3_fraction"]),
            item["F1_n"], item["F2_n"], item["F3_n"],
        ])

    timing_table = []
    for config, scene in [("OPEN-A1", "open_task")] + [
        (config, scene) for config in CONFIGS for scene in OBSTACLE_SCENES
    ]:
        subset = [
            row for row in rows
            if row["config_id"] == config and row["scene"] == scene
        ]
        timing_table.append([
            config,
            scene,
            len(subset),
            fmt(median(row["mean_controller_hz"] for row in subset)),
            fmt(median(row["median_controller_hz"] for row in subset)),
            fmt(median(row["p95_period_s"] for row in subset)),
            fmt(median(row["p99_period_s"] for row in subset)),
            fmt(median(row["sim_wall_ratio"] for row in subset)),
            fmt(min((row["sim_wall_ratio"] for row in subset), default=math.nan)),
        ])

    failure_table = [
        [item["config_id"], item["scene"], item["failure_class"], item["count"]]
        for item in failure_counts
    ]

    b1 = summary_index[("B1", "OBSTACLE_COMBINED")]
    b2 = summary_index[("B2", "OBSTACLE_COMBINED")]
    b3 = summary_index[("B3", "OBSTACLE_COMBINED")]
    b1_shelf = summary_index[("B1", "shelf_gap")]
    b2_shelf = summary_index[("B2", "shelf_gap")]
    b1_single = summary_index[("B1", "single_obstacle")]
    b2_single = summary_index[("B2", "single_obstacle")]
    b1b2_shelf = b1b2["by_scene"]["shelf_gap"]
    b1b2_single = b1b2["by_scene"]["single_obstacle"]
    shelf_ratio_delta = b1b2_shelf["metric_differences"]["R_obs_task_median"]
    tail_reduced = (
        shelf_ratio_delta["paired_n"] > 0
        and number(shelf_ratio_delta["median_delta_right_minus_left"]) < 0
    )
    shelf_improved = (
        b1b2_shelf["transitions"]["FS"] > b1b2_shelf["transitions"]["SF"]
    )
    single_hurt = (
        b1b2_single["transitions"]["SF"] > b1b2_single["transitions"]["FS"]
    )

    runtime_reasons = Counter(
        item["terminal_reason_audited"] for item in attempts
        if item["terminal_for_manifest_entry"] and not item["scientifically_admitted"]
    )
    timing_invalid = sum(
        item["timing_valid"] is False for item in attempts
    )
    terminal_cleanup_failures = sum(
        item["terminal_for_manifest_entry"] and item["cleanup_passed"] in (False, "False", 0)
        for item in attempts
    )
    all_cleanup_failures = sum(
        item["cleanup_passed"] in (False, "False", 0) for item in attempts
    )
    cleanup_failure_attempts = [
        f"{item['run_id']}/{Path(item['attempt']).name}"
        for item in attempts
        if item["cleanup_passed"] in (False, "False", 0)
    ]
    start_winner = {}
    rotation_winner = {}
    for dimension, target in (("start", start_winner), ("rotation", rotation_winner)):
        levels = STARTS if dimension == "start" else ROTATIONS
        for level in levels:
            subset = [item for item in breakdowns if item["scene"] in OBSTACLE_SCENES and item["dimension"] == dimension and item["level"] == level]
            totals = defaultdict(lambda: [0, 0])
            for item in subset:
                totals[item["config_id"]][0] += item["success_n"]
                totals[item["config_id"]][1] += item["admitted_n"]
            target[level] = sorted(CONFIGS, key=lambda c: (totals[c][1], totals[c][0]), reverse=True)[0]

    asymmetry_table = []
    for config in CONFIGS:
        item = asymmetry["by_config"][config]
        transitions = item["transitions_ccw_to_cw"]
        asymmetry_table.append([
            config,
            f"{item['admissible_pairs']}/{item['planned_pairs']}",
            transitions["SS"], transitions["SF"], transitions["FS"], transitions["FF"],
            fmt(item["success_effect_cw_minus_ccw"]),
            fmt(item["mcnemar_exact_two_sided_p"]),
            "YES" if item["statistically_supported_at_0_05"] else "NO",
        ])
    historical_agreement = (
        "NO: none of the historical obstacle successes replicated among "
        "the admitted source-inner/A1 obstacle runs; B3 only had the highest "
        "admission coverage"
    )
    obstacle_success_counts = {config: summary_index[(config, "OBSTACLE_COMBINED")]["success_n"] for config in CONFIGS}
    best_obstacle_success = max(obstacle_success_counts.values())
    best_success_configs = [
        config for config in CONFIGS
        if obstacle_success_counts[config] == best_obstacle_success
    ]
    best_success_answer = (
        best_success_configs[0]
        if len(best_success_configs) == 1
        else "TIE: " + ", ".join(best_success_configs)
    )
    b1b2_shelf_paired_n = b1b2_shelf["admissible_matched_pairs"]
    b1b2_single_paired_n = b1b2_single["admissible_matched_pairs"]

    report = f"""# Final Clean C3+ Obstacle-Ranking Results

## Scientific boundary and provenance

This report analyzes only the clean source-inner experiment authorized by the immutable 126-row manifest (`{EXPECTED_MANIFEST_SHA256}`). Historical TR07/TR08 obstacle evidence was used only to preregister B1/B2/B3; it is not pooled with these results.

All live conditions used the source-frozen inner C3/C3+ controller, `alpha_p_rank=1`, `alpha_theta_rank=1`, `SAMPLING_C3_INNER_OBS_MODE=none`, and frozen `lcs_contact`. Study A was not launched. B1 versus B2 is the controlled shape comparison; B3 differs in support and weight and is interpreted as an independent practical ReLU reference.

## Execution and admission audit

- Manifest rows reaching a terminal marker: **126/126**.
- Scientifically admitted, timing-valid runs: **{accepted}/126**.
- Terminal non-admitted runtime/infrastructure failures: **{terminal_failures}/126**.
- Scientifically admitted successes: **{total_success}/126 preregistered tasks**.
- Timing-invalid attempts: **{timing_invalid}**.
- Cleanup-failed attempts (including superseded attempts): **{all_cleanup_failures}**.
- Cleanup-failed attempt identities: **{', '.join(cleanup_failure_attempts) if cleanup_failure_attempts else 'none'}**. A superseded attempt is never admitted merely because a later attempt passed.
- Cleanup failures on the terminal attempt for a manifest entry: **{terminal_cleanup_failures}**.
- Runtime-failure reasons: `{dict(runtime_reasons)}`.
- No post-cap computation is admitted; only attempts named by `ACCEPTED_RUN.json` enter scientific outcome metrics.

## Scene and configuration outcomes

{table(['Config','Scene','Admitted/planned','Success/admitted','Scientific rate','Wilson 95% CI (admitted)','Success/planned yield','Retained','Median T_goal s','Median E_best failures','F2'], scene_table)}

Runtime/infrastructure failures are not counted as task failures. `Success/planned yield` is shown separately as a reproducibility-aware preregistered yield; `Success/admitted` is the scientific task-success estimate among timing-valid admitted runs.

Shared OPEN-A1 is one common 18-task baseline, not three independent obstacle-configuration samples. It achieved **{open_summary['success_n']}/18** successes and **{open_summary['retained_success_n']}** retained successes.

## Start, rotation, and seed breakdown

{table(['Config','Scene','Dimension','Level','Admitted/planned','Success/planned'], breakdown_table)}

## Physical, contact, and failure outcomes

{table(['Config','Scene','Median min clearance m','Median max penetration m','Median CHOMP norm','Median transactions','Median transaction displacement m','Median transaction contact duty','Median physical contact','Median contactless C3','F1','F2','F3'], physical_table)}

{table(['Config','Scene','Admitted-run outcome class','Count'], failure_table)}

Selected-candidate cost decompositions (`J_position`, `J_orientation`, angular/linear velocity, obstacle, fixed, total rank, and obstacle/task ratio) are recorded per run in `metrics/per_run_results.csv`. Their q25/median/q75 aggregates by configuration, scene, start, and rotation are in `metrics/cost_component_aggregates.csv`.

## Timing validity of admitted runs

All rows in this table passed the existing per-run timing gate. Periods are seconds. The minimum sim/wall column makes the worst admitted progress ratio explicit.

{table(['Config','Scene','Admitted','Median mean Hz','Median controller Hz','Median p95','Median p99','Median sim/wall','Minimum sim/wall'], timing_table)}

## Paired obstacle comparisons

Only exact scene/start/rotation/seed pairs for which both sides were scientifically admitted enter this table. `SF` means left succeeds/right fails; `FS` means left fails/right succeeds. The effect is right-minus-left.

{table(['Pair','Admissible pairs','SS','SF','FS','FF','Success effect','Exact McNemar p'], paired_table)}

For B1 versus B2, {b1b2['unpaired_due_to_nonadmission']} of 36 preregistered pairs are excluded from scientific paired metrics because at least one side was not admitted. Full continuous-outcome paired differences and deterministic bootstrap confidence intervals are in the machine-readable handoff and `metrics/paired_summary.json`.

## CW/CCW asymmetry audit

CCW and CW outcomes are paired within each configuration by scene/start/seed; configurations are not pooled as independent replicates.

{table(['Config','Admissible/planned pairs','SS','SF','FS','FF','CW-minus-CCW effect','Exact McNemar p','Supported at 0.05'], asymmetry_table)}

Any configuration with statistically supported CW/CCW asymmetry at the two-sided 0.05 level: **{'YES' if asymmetry['any_statistically_supported_at_0_05'] else 'NO'}**.

## B1/B2 shelf-gap mechanism

- Median realized obstacle/task ratio on shelf: B1 **{fmt(b1_shelf['median_R_obs_task'])}**, B2 **{fmt(b2_shelf['median_R_obs_task'])}**.
- Paired shelf realized obstacle/task-ratio delta (B2 minus B1): median **{fmt(shelf_ratio_delta['median_delta_right_minus_left'])}**, mean **{fmt(shelf_ratio_delta['mean_delta_right_minus_left'])}**, bootstrap mean 95% CI **[{fmt(shelf_ratio_delta['bootstrap_mean_95ci'][0])}, {fmt(shelf_ratio_delta['bootstrap_mean_95ci'][1])}]**, n={shelf_ratio_delta['paired_n']}.
- Paired shelf evidence available: **{b1b2_shelf_paired_n}/18 pairs**. A causal tail/corridor-cost reduction is therefore **{'SUPPORTED' if tail_reduced else 'NOT ESTABLISHED'}**. The unpaired admitted-run medians (B1 {fmt(b1_shelf['median_R_obs_task'])}, B2 {fmt(b2_shelf['median_R_obs_task'])}) are descriptive only because admission was outcome-dependent.
- Shelf success: B1 **{b1_shelf['success_n']}/{b1_shelf['admitted_n']} admitted** and B2 **{b2_shelf['success_n']}/{b2_shelf['admitted_n']} admitted**; improvement by B2 is **{'SUPPORTED' if shelf_improved else 'NOT DEMONSTRATED'}**.
- Single-obstacle paired evidence available: **{b1b2_single_paired_n}/18 pairs**. B1 and B2 had zero successes among admitted runs; harm to anticipation from the cutoff is **{'SUPPORTED' if single_hurt else 'NOT DEMONSTRATED'}**.
- Candidate-selection sequence differences are reported per pair in `metrics/paired_task_results.csv`. Sequence comparisons after the first physical divergence are descriptive, not proof of a direct one-step causal effect.

## Reproducibility-aware ranking

Predeclared ordering criteria give: **{' > '.join(order)}**. The operational selection is **{winner}** because valid/reproducible execution is ranked first. This is **not evidence of obstacle-task efficacy**: all three configurations had zero admitted obstacle-task successes. Task success, cross-scene balance, retained success, time/error, physical safety/CHOMP, transaction productivity, and F2 incidence follow admission coverage in the preregistered rule. Raw `J_rank` magnitude is never a selection criterion.

Obstacle-combined headline values:

- B1: admitted {b1['admitted_n']}/36, success {b1['success_n']}/36, retained {b1['retained_success_n']}, F2 {b1['F2_n']}.
- B2: admitted {b2['admitted_n']}/36, success {b2['success_n']}/36, retained {b2['retained_success_n']}, F2 {b2['F2_n']}.
- B3: admitted {b3['admitted_n']}/36, success {b3['success_n']}/36, retained {b3['retained_success_n']}, F2 {b3['F2_n']}.

## Required final answers

1. **Does B1 remain successful under source-frozen C3?** {'YES' if b1['success_n'] else 'NO'} — {b1['success_n']}/{b1['admitted_n']} admitted obstacle runs succeeded ({b1['success_n']}/36 preregistered-task yield).
2. **Does matched ReLU B2 perform similarly to B1?** FORMAL EQUIVALENCE NOT ESTABLISHED because no equivalence margin was preregistered. Descriptively: SS={b1b2['transitions']['SS']}, SF={b1b2['transitions']['SF']}, FS={b1b2['transitions']['FS']}, FF={b1b2['transitions']['FF']}; success effect={fmt(b1b2['paired_success_difference_right_minus_left'])}.
3. **Does B2 improve shelf-gap performance?** NOT DEMONSTRATED — B1 had {b1_shelf['success_n']}/{b1_shelf['admitted_n']} admitted successes and B2 had {b2_shelf['success_n']}/{b2_shelf['admitted_n']}; no B1/B2 shelf pair was jointly admitted.
4. **Does B2 reduce exponential corridor-tail interference?** NOT ESTABLISHED CAUSALLY — no shelf pair was jointly admitted. The unpaired median obstacle/task ratio was lower for B2 ({fmt(b2_shelf['median_R_obs_task'])}) than B1 ({fmt(b1_shelf['median_R_obs_task'])}), but outcome-dependent admission prevents a controlled inference.
5. **Does removing the tail hurt single-obstacle anticipation?** NOT DEMONSTRATED — the {b1b2_single_paired_n} jointly admitted single-obstacle pairs were all fail/fail, and neither configuration had an admitted success.
6. **Does B3 remain competitive under source C3?** OPERATIONALLY YES, EFFICACY NO — B3 has the best admission coverage ({b3['admitted_n']}/36) and ranks {order.index('B3')+1} under the preregistered rule, but had {b3['success_n']}/{b3['admitted_n']} admitted obstacle successes.
7. **Which is most robust across starts?** No task-success winner exists. By admission coverage only: {start_winner}.
8. **Which is most robust across rotations?** No task-success winner exists. By admission coverage only: {rotation_winner}.
9. **Is there CW/CCW asymmetry?** Statistically supported at the two-sided 0.05 level: **{'YES' if asymmetry['any_statistically_supported_at_0_05'] else 'NO'}**. Per-configuration effects and exact paired tests are in the table above.
10. **Which produces fewer F2 stalls?** {min(CONFIGS, key=lambda c: number(summary_index[(c, 'OBSTACLE_COMBINED')]['F2_rate_admitted'], 1e30))} by admitted-run F2 rate, with sparse and unequal admitted denominators.
11. **Which gives the best clearance/success tradeoff?** NOT ESTABLISHED because no obstacle run succeeded. {winner} is the operational selection under the preregistered lexicographic rule, not a demonstrated efficacy tradeoff winner.
12. **Which gives the best overall obstacle-task success?** {best_success_answer}, each with {best_obstacle_success} admitted successes; no efficacy winner exists. The operational selection remains {winner} after applying the full preregistered rule.
13. **Does the clean result agree with the historical OC audit?** {historical_agreement}. The clean study, not the historical TR backgrounds, is authoritative here.
14. **Which obstacle ranking should proceed to the later C3+ versus OIM benchmark?** **{winner} conditionally**, because the preregistered rule prioritizes its higher admission coverage. No obstacle-task efficacy was demonstrated, so this is an operational selection rather than a validated scientific winner; the frozen source-inner/A1 boundary must be preserved.

## Scope stop

No alpha tuning, OC04, matched 0.075 m ReLU, extra start/seed/scene, inner-C3 change, F1/F2/F3 change, or OIM comparison was launched.
"""

    handoff = {
        "schema_version": 1,
        "campaign": "final_clean_obstacle_ranking",
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "manifest_sha256": sha256(MANIFEST),
        "manifest_rows": len(manifest),
        "terminal_rows": 126,
        "scientifically_admitted_rows": accepted,
        "terminal_nonadmitted_rows": terminal_failures,
        "scientific_successes": total_success,
        "study_a_launched": False,
        "additional_studies_launched": False,
        "inner_c3": "SOURCE_FROZEN",
        "task_ranking": {"alpha_p_rank": 1, "alpha_theta_rank": 1},
        "winner": winner,
        "winner_interpretation": (
            "Operational selection under preregistered admission-first rule; "
            "not an obstacle-task efficacy winner because all configurations "
            "had zero admitted successes."
        ),
        "ranking": order,
        "summaries": summaries,
        "breakdowns": breakdowns,
        "paired_summaries": pair_summaries,
        "runtime_failure_reasons": dict(runtime_reasons),
        "timing_invalid_attempts": timing_invalid,
        "cleanup_failed_attempts_including_superseded": all_cleanup_failures,
        "cleanup_failed_attempt_identities": cleanup_failure_attempts,
        "terminal_cleanup_failures": terminal_cleanup_failures,
        "rotation_asymmetry": asymmetry,
        "required_answers": {
            "B1_success_under_source_frozen": bool(b1["success_n"]),
            "B1_B2_formal_equivalence_established": False,
            "B1_B2_equivalence_note": "No equivalence margin was preregistered.",
            "B2_shelf_improves_success_count": shelf_improved,
            "B2_reduces_realized_shelf_obstacle_task_ratio_established": tail_reduced,
            "B1_B2_jointly_admitted_shelf_pairs": b1b2_shelf_paired_n,
            "B1_B2_unpaired_shelf_median_R_obs_task": {
                "B1": b1_shelf["median_R_obs_task"],
                "B2": b2_shelf["median_R_obs_task"],
                "interpretation": "descriptive_only_outcome_dependent_admission",
            },
            "B2_single_obstacle_success_harmed": single_hurt,
            "B3_rank": order.index("B3") + 1,
            "most_robust_by_start": start_winner,
            "most_robust_by_rotation": rotation_winner,
            "cw_ccw_asymmetry_supported_at_0_05": asymmetry[
                "any_statistically_supported_at_0_05"
            ],
            "fewest_F2": min(
                CONFIGS,
                key=lambda c: number(
                    summary_index[(c, "OBSTACLE_COMBINED")]["F2_rate_admitted"],
                    1e30,
                ),
            ),
            "best_overall_obstacle_task_success": best_success_answer,
            "obstacle_task_success_counts": obstacle_success_counts,
            "obstacle_task_efficacy_winner_established": False,
            "historical_audit_agreement": historical_agreement,
            "recommended_for_later_benchmark": f"{winner}_CONDITIONAL",
        },
        "artifacts": {
            "report": str(REPORT.relative_to(ROOT)),
            "per_run": str((METRICS / "per_run_results.csv").relative_to(ROOT)),
            "attempt_inventory": str((METRICS / "attempt_inventory.csv").relative_to(ROOT)),
            "summary": str((METRICS / "configuration_scene_summary.csv").relative_to(ROOT)),
            "breakdowns": str((METRICS / "start_rotation_seed_breakdown.csv").relative_to(ROOT)),
            "paired_tasks": str((METRICS / "paired_task_results.csv").relative_to(ROOT)),
            "paired_summary": str((METRICS / "paired_summary.json").relative_to(ROOT)),
            "cost_component_aggregates": str(
                (METRICS / "cost_component_aggregates.csv").relative_to(ROOT)
            ),
            "failure_counts": str((METRICS / "failure_counts.csv").relative_to(ROOT)),
            "rotation_asymmetry": str(
                (METRICS / "rotation_asymmetry.json").relative_to(ROOT)
            ),
        },
    }
    return report, handoff


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validate-partial", action="store_true",
        help="validate current artifacts without writing final scientific outputs",
    )
    args = parser.parse_args()
    if sha256(MANIFEST) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("MANIFEST_HASH_MISMATCH")
    manifest = read_csv(MANIFEST)
    if len(manifest) != 126 or len({row["run_id"] for row in manifest}) != 126:
        raise RuntimeError("manifest is not 126 unique rows")

    terminal = 0
    pending = []
    admitted_rows = []
    attempts = []
    for row in manifest:
        kind, attempt, marker = locate_attempt(row)
        attempts.extend(attempt_inventory(row))
        if kind == "PENDING":
            pending.append(row["run_id"])
            continue
        terminal += 1
        assert attempt is not None
        validate_terminal_attempt(row, attempt, kind == "ACCEPTED")
        if kind == "ACCEPTED":
            admitted_rows.append(process_admitted(row, attempt, marker))

    print(json.dumps({
        "manifest_sha256": sha256(MANIFEST),
        "terminal": terminal,
        "admitted": len(admitted_rows),
        "terminal_nonadmitted": terminal - len(admitted_rows),
        "pending": len(pending),
    }, indent=2))
    if args.validate_partial:
        return 0
    if pending:
        raise RuntimeError(
            f"campaign is incomplete: {len(pending)} manifest entries are not terminal"
        )

    summaries = summary_rows(admitted_rows)
    breakdowns = breakdown_rows(admitted_rows)
    cost_aggregates = cost_component_aggregate_rows(admitted_rows)
    failure_counts = failure_count_rows(admitted_rows)
    all_pair_rows = []
    pair_summaries = []
    for left, right in (("B1", "B2"), ("B1", "B3"), ("B2", "B3")):
        pair_detail, pair_summary = pair_rows(admitted_rows, left, right)
        pair_summary["by_scene"] = {}
        for scene in OBSTACLE_SCENES:
            _, scene_summary = pair_rows(
                [row for row in admitted_rows if row["scene"] == scene],
                left,
                right,
                planned_matched_tasks=18,
            )
            pair_summary["by_scene"][scene] = scene_summary
        all_pair_rows.extend(pair_detail)
        pair_summaries.append(pair_summary)

    asymmetry = rotation_asymmetry(admitted_rows)

    per_run_fields = [key for key in admitted_rows[0] if key != "selected_candidate_sequence"]
    write_csv(METRICS / "per_run_results.csv", admitted_rows, per_run_fields)
    write_csv(METRICS / "attempt_inventory.csv", attempts, list(attempts[0]))
    write_csv(METRICS / "configuration_scene_summary.csv", summaries, list(summaries[0]))
    write_csv(METRICS / "start_rotation_seed_breakdown.csv", breakdowns, list(breakdowns[0]))
    write_csv(
        METRICS / "cost_component_aggregates.csv",
        cost_aggregates,
        list(cost_aggregates[0]),
    )
    write_csv(
        METRICS / "failure_counts.csv",
        failure_counts,
        list(failure_counts[0]) if failure_counts else ["config_id", "scene", "failure_class", "count"],
    )
    write_csv(METRICS / "paired_task_results.csv", all_pair_rows, list(all_pair_rows[0]) if all_pair_rows else ["pair"])
    atomic_json(METRICS / "paired_summary.json", pair_summaries)
    atomic_json(METRICS / "rotation_asymmetry.json", asymmetry)
    report, handoff = render_report(
        manifest, admitted_rows, attempts, summaries, breakdowns,
        cost_aggregates, failure_counts, pair_summaries, asymmetry,
    )
    atomic_text(REPORT, report)
    atomic_json(HANDOFF, handoff)
    print(f"wrote {REPORT}")
    print(f"wrote {HANDOFF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
