#!/usr/bin/env python3
"""Fail-closed serial supervisor for the final clean 126-run obstacle study.

Without ``--execute`` this performs preflight only.  Live execution also
requires a manifest-checksummed ``USER_APPROVAL.json`` created after explicit
user approval.  The imported serial supervisor owns and cleans one process
group per run; this wrapper freezes the source inner/task cost and changes only
the outer obstacle-ranking family.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import statistics
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/c3plus_final_clean_obstacle_ranking"
MANIFEST = OUT / "manifest/final_126_run_manifest.csv"
CONFIG = OUT / "config/final_clean_obstacle_study.json"
APPROVAL = OUT / "USER_APPROVAL.json"
PROVENANCE = OUT / "provenance/preparation_receipt.json"
EXECUTION_PROVENANCE = OUT / "provenance/execution_infrastructure_patch.json"
OPEN_PROOF = OUT / "validation/open_task_equivalence.json"
BASE_PATH = ROOT / "scripts/run_c3plus_geometry_rotation_cost_replication.py"

spec = importlib.util.spec_from_file_location("c3plus_serial_supervisor", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import serial supervisor: {BASE_PATH}")
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)
supervisor.OUT = OUT
supervisor.MANIFEST = MANIFEST
supervisor.CONFIG = CONFIG
supervisor.APPROVAL = APPROVAL
supervisor.OWNED_RUNNER = Path(__file__).resolve()


def read_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 126 or len({row["run_id"] for row in rows}) != 126:
        raise RuntimeError("final clean obstacle manifest must have 126 unique runs")
    return rows


def run_environment(row: dict[str, str], attempt: Path) -> dict[str, str]:
    # No inherited experiment switch is allowed to leak into this campaign.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("SAMPLING_C3_")}
    env_file = supervisor.WT / row["env_file"] if row["env_file"] else None
    env = supervisor.source_environment(env, env_file)
    unexpected = sorted(key for key in env if key.startswith("SAMPLING_C3_")
                        and key != "SAMPLING_C3_OBS_BOXES")
    if unexpected:
        raise RuntimeError(f"uncontrolled SAMPLING_C3 variables from scene env: {unexpected}")

    tmp = attempt / "tmp"
    forensics = attempt / "forensics"
    tmp.mkdir()
    forensics.mkdir()
    env.update(supervisor.THREAD_LIMITS)
    env.update({
        "TMPDIR": str(tmp),
        "SAMPLING_C3_EXPERIMENT_SEED": row["seed"],
        "SAMPLING_C3_ALPHA_P_RANK": "1",
        "SAMPLING_C3_ALPHA_THETA_RANK": "1",
        "SAMPLING_C3_INNER_OBS_MODE": "none",
        "SAMPLING_C3_FORENSICS_LOG_DIR": str(forensics),
        "SAMPLING_C3_FORENSICS_TEXT_EVENTS": "0",
        "SAMPLING_C3_FORENSICS_GIT_COMMIT": supervisor.subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=supervisor.WT,
            text=True).strip(),
        "SAMPLING_C3_FORENSICS_TASK_ID": row["run_id"],
        "SAMPLING_C3_FORENSICS_ROBOT_MODEL": "xarm6",
        "SAMPLING_C3_OBSTACLE_MODE": "lcs_contact",
        "SAMPLING_C3_STOP_RECORDER_ON_PLANNER_EXIT": "1",
    })
    family = row["obstacle_family"]
    if family == "none":
        if row["scene"] != "open_task":
            raise RuntimeError("no-obstacle ranking is allowed only for OPEN-A1")
    elif family == "exponential_footprint":
        env.update({
            "SAMPLING_C3_RANK_OBS_MODE": "exponential_footprint",
            "SAMPLING_C3_OBS_EXP_SIGMA": row["obstacle_range_m"],
            "SAMPLING_C3_OBS_EXP_W": row["obstacle_weight"],
        })
    elif family == "relu_footprint":
        env.update({
            "SAMPLING_C3_RANK_OBS_MODE": "relu_footprint",
            "SAMPLING_C3_OBS_RELU_EPS": row["obstacle_range_m"],
            "SAMPLING_C3_OBS_RELU_W": row["obstacle_weight"],
        })
    else:
        raise RuntimeError(f"unknown obstacle family: {family}")

    forbidden = {
        "SAMPLING_C3_TRANSLATION_COST_SCALE",
        "SAMPLING_C3_ORIENTATION_COST_SCALE",
        "SAMPLING_C3_OBJ_NONPEN",
    }
    leaked = sorted(forbidden & env.keys())
    if leaked:
        raise RuntimeError(f"legacy/inner experimental variables leaked: {leaked}")
    return env


def check_shared_open_result(block_index: str,
                             rows: list[dict[str, str]]) -> None:
    open_rows = [row for row in rows
                 if row["block_index"] == block_index
                 and row["config_id"] == "OPEN-A1"]
    if len(open_rows) != 1:
        raise RuntimeError(f"block {block_index} does not contain one OPEN-A1")
    row = open_rows[0]
    attempt = supervisor.accepted_attempt(row)
    if attempt is None:
        return
    output = OUT / "validation/live_open_zero" / f"block_{int(block_index):02d}.json"
    if output.exists():
        if not json.loads(output.read_text()).get("passed"):
            raise RuntimeError(f"prior OPEN-A1 validation failed: {output}")
        return
    receipt = json.loads((attempt / "run_receipt.json").read_text())
    environment = receipt["environment"]
    zero = True
    candidate_rows = []
    with (attempt / "forensics/forensics_candidates.csv").open(newline="") as stream:
        candidate_rows = list(csv.DictReader(stream))
    incomplete = []
    candidates = 0
    for index, candidate in enumerate(candidate_rows):
        value = candidate.get("J_obstacle_applied_live")
        selected = candidate.get("selected")
        if value in (None, "") or selected not in {"0", "1"}:
            incomplete.append(index)
            continue
        candidates += 1
        if abs(float(value)) > 1e-12:
                zero = False
                break
    # SIGTERM after recorder success can interrupt the final CSV write.  A
    # single structurally incomplete last record is not scientific data and is
    # ignored; any interior or multiple partial records fail closed.
    trailing_partial_only = not incomplete or incomplete == [len(candidate_rows) - 1]
    checks = {
        "all_recorded_obstacle_costs_zero": zero,
        "candidate_rows_checked_positive": candidates > 0,
        "no_interior_or_multiple_partial_candidate_rows": trailing_partial_only,
        "alpha_p_rank_one": environment.get("SAMPLING_C3_ALPHA_P_RANK") == "1",
        "alpha_theta_rank_one": environment.get("SAMPLING_C3_ALPHA_THETA_RANK") == "1",
        "inner_soft_obstacle_none": environment.get("SAMPLING_C3_INNER_OBS_MODE") == "none",
        "no_experimental_obstacle_rank_mode": "SAMPLING_C3_RANK_OBS_MODE" not in environment,
    }
    payload = {"run_id": row["run_id"], "attempt": str(attempt),
               "candidate_rows_total": len(candidate_rows),
               "candidate_rows_complete": candidates,
               "trailing_partial_rows_ignored": len(incomplete) if trailing_partial_only else 0,
               "checks": checks, "passed": all(checks.values())}
    supervisor.atomic_json(output, payload)
    if not payload["passed"]:
        supervisor.atomic_json(
            OUT / "state/PAUSED_OPEN_TASK_EQUIVALENCE_FAILURE.json", payload)
        raise RuntimeError("OPEN_TASK_EQUIVALENCE_FAILURE")


def accepted_attempts(rows: list[dict[str, str]]) -> list[Path]:
    return [attempt for row in rows
            if (attempt := supervisor.accepted_attempt(row)) is not None]


def terminal_failure_marker(row: dict[str, str]) -> Path:
    return OUT / "runs" / row["run_id"] / "TERMINAL_RUN.json"


def finalize_runtime_failure(row: dict[str, str]) -> None:
    run_root = OUT / "runs" / row["run_id"]
    attempts = sorted((run_root / "attempts").glob("attempt_*"))
    if not attempts:
        raise RuntimeError(f"runtime failure has no attempt directory: {row['run_id']}")
    attempt = attempts[-1]
    cleanup_path = attempt / "owned_cleanup.json"
    cleanup = json.loads(cleanup_path.read_text()) if cleanup_path.exists() else {}
    payload = {
        "status": "RUNTIME_FAILURE",
        "scientifically_admitted": False,
        "terminal_attempt": str(attempt.relative_to(run_root)),
        "terminal_reason": cleanup.get("terminal_reason", "UNKNOWN"),
        "cleanup_passed": cleanup.get("passed", False),
        "finalized_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if not payload["cleanup_passed"]:
        raise RuntimeError(f"cannot finalize runtime failure without cleanup: {row['run_id']}")
    supervisor.atomic_json(terminal_failure_marker(row), payload)
    supervisor.append_event(
        row["run_id"], "FINALIZED_RUNTIME_FAILURE",
        terminal_attempt=payload["terminal_attempt"],
        terminal_reason=payload["terminal_reason"],
    )


def write_eta_checkpoint(rows: list[dict[str, str]]) -> None:
    attempts = accepted_attempts(rows)
    count = len(attempts)
    if count == 0 or count % 18:
        return
    output = OUT / "state" / f"eta_after_{count:03d}_accepted.json"
    if output.exists():
        return
    runtimes = []
    for attempt in attempts:
        timing = json.loads((attempt / "timing.json").read_text())
        runtimes.append(float(timing["wall_time_s"]))
    mean_s = statistics.mean(runtimes)
    runs_per_hour = 3600.0 / mean_s
    payload = {
        "accepted_runs": count,
        "remaining_runs": 126 - count,
        "mean_runtime_s": mean_s,
        "median_runtime_s": statistics.median(runtimes),
        "observed_runs_per_hour": runs_per_hour,
        "estimated_remaining_hours": (126 - count) / runs_per_hour,
        "parameters_changed": False,
        "informational_only": True,
    }
    supervisor.atomic_json(output, payload)


def preflight() -> int:
    rows = read_manifest()
    config = json.loads(CONFIG.read_text())
    provenance = json.loads(PROVENANCE.read_text())
    execution_provenance = (
        json.loads(EXECUTION_PROVENANCE.read_text())
        if EXECUTION_PROVENANCE.exists() else {}
    )
    proof = json.loads(OPEN_PROOF.read_text())
    errors: list[str] = []

    counts = {
        "shared_open": sum(row["config_id"] == "OPEN-A1" for row in rows),
        **{
            f"{config_id}_{scene}": sum(
                row["config_id"] == config_id and row["scene"] == scene
                for row in rows)
            for config_id in ("B1", "B2", "B3")
            for scene in ("single_obstacle", "shelf_gap")
        },
    }
    if counts["shared_open"] != 18 or any(
            value != 18 for key, value in counts.items() if key != "shared_open"):
        errors.append(f"run-count matrix differs: {counts}")
    if config["unique_live_runs"] != 126 or config["max_concurrent_live_simulations"] != 1:
        errors.append("config does not freeze 126 unique serial runs")
    if {row["config_id"] for row in rows} != {"OPEN-A1", "B1", "B2", "B3"}:
        errors.append("manifest condition set differs")
    if {row["start_id"] for row in rows} != {"s1", "s3", "s5"}:
        errors.append("start set differs from s1/s3/s5")
    if {row["rotation_label"] for row in rows} != {
            "rot000", "rotCCW090", "rotCW090"}:
        errors.append("rotation set differs")
    if {row["seed"] for row in rows} != {"0", "1"}:
        errors.append("seed set differs from 0/1")
    if len({row["lcm_port"] for row in rows}) != 126:
        errors.append("LCM ports are not unique")
    if any(row["alpha_p_rank"] != "1.0" or row["alpha_theta_rank"] != "1.0"
           for row in rows):
        errors.append("A1 ranking is not frozen")
    if any(row["inner_controller"] != "SOURCE_FROZEN" for row in rows):
        errors.append("inner controller is not source-frozen")
    if any(row["source_obstacle_id"] in {"OC04", "MATCHED_OC04_RELU"}
           for row in rows):
        errors.append("predeclared dropped branch appears in manifest")
    for block in {row["block_index"] for row in rows}:
        block_rows = [row for row in rows if row["block_index"] == block]
        if len(block_rows) != 7 or sum(row["config_id"] == "OPEN-A1"
                                      for row in block_rows) != 1:
            errors.append(f"block {block} is not one open plus six obstacle runs")
            break
        for scene in ("single_obstacle", "shelf_gap"):
            if {row["config_id"] for row in block_rows if row["scene"] == scene} != {
                    "B1", "B2", "B3"}:
                errors.append(f"block {block}/{scene} lacks a matched B triplet")
                break
    if proof.get("status") != "PASS" or not proof.get("selected_candidate_identical"):
        errors.append("offline shared-open equivalence proof is not PASS")
    checksums = {
        "controller_source_sha256": supervisor.WT /
            "systems/controllers/sampling_based_c3_controller.cc",
        "ranking_helper_sha256": supervisor.WT /
            "systems/controllers/ranking_task_cost.h",
        "controller_binary_sha256": supervisor.BIN / "franka_sampling_c3_controller",
        "sim_binary_sha256": supervisor.BIN / "franka_sim",
        "osc_binary_sha256": supervisor.BIN / "franka_osc_controller",
        "manifest_sha256": MANIFEST,
        "config_sha256": CONFIG,
        "open_equivalence_sha256": OPEN_PROOF,
        "serial_supervisor_sha256": BASE_PATH,
        "campaign_runner_sha256": Path(__file__).resolve(),
    }
    if execution_provenance and execution_provenance.get(
            "manifest_sha256") != supervisor.sha256(MANIFEST):
        errors.append("execution infrastructure receipt targets another manifest")
    for key, path in checksums.items():
        expected = execution_provenance.get(key, provenance.get(key))
        if not path.is_file() or expected != supervisor.sha256(path):
            errors.append(f"{key} differs from preparation receipt")
    if any(not (supervisor.WT / "examples/sampling_c3" / row["demo_name"]).is_dir()
           for row in rows):
        errors.append("a materialized synchronized demo is missing")
    started = bool(list((OUT / "runs").glob("**/attempt_*"))) if (OUT / "runs").exists() else False

    payload: dict[str, Any] = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "rows": len(rows),
        "run_counts": counts,
        "manifest_sha256": supervisor.sha256(MANIFEST),
        "open_equivalence": proof.get("status"),
        "approval_present": APPROVAL.exists(),
        "scientific_runs_started": started,
        "machine_gate": supervisor.machine_gate(),
    }
    supervisor.atomic_json(OUT / "provenance/latest_preflight.json", payload)
    print(json.dumps(payload, sort_keys=True))
    return 0 if not errors else 1


def execute() -> int:
    if not APPROVAL.exists():
        raise RuntimeError(f"explicit approval receipt absent: {APPROVAL}")
    approval = json.loads(APPROVAL.read_text())
    if approval.get("approved") is not True or \
            approval.get("manifest_sha256") != supervisor.sha256(MANIFEST):
        raise RuntimeError("approval does not authorize this exact manifest")
    if preflight() != 0:
        raise RuntimeError("preflight failed")
    rows = read_manifest()
    queue = deque(row for row in rows
                  if supervisor.accepted_attempt(row) is None
                  and not terminal_failure_marker(row).exists())
    consecutive_invalid = 0
    while queue:
        row = queue.popleft()
        outcome = supervisor.launch_one(row)
        if outcome in {"SUCCESS", "TIMEOUT", "ALREADY_ACCEPTED"}:
            consecutive_invalid = 0
            check_shared_open_result(row["block_index"], rows)
            write_eta_checkpoint(rows)
        elif outcome == "TIMING_INVALID":
            consecutive_invalid += 1
            queue.append(row)
            if consecutive_invalid >= 2:
                raise RuntimeError("PAUSED_AFTER_REPEATED_TIMING_FAILURES")
        elif outcome == "RUNTIME_FAILURE":
            consecutive_invalid = 0
            finalize_runtime_failure(row)
        else:
            raise RuntimeError(f"campaign paused after {outcome}")
    return 0


supervisor.read_manifest = read_manifest
supervisor.run_environment = run_environment
supervisor.check_open_negative_control = check_shared_open_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--owned-run", type=Path)
    parser.add_argument("--attempt", type=Path)
    args = parser.parse_args()
    if args.owned_run:
        if args.attempt is None:
            parser.error("--owned-run requires --attempt")
        return supervisor.owned_run(args.owned_run, args.attempt.resolve())
    if args.execute:
        return execute()
    return preflight()


if __name__ == "__main__":
    raise SystemExit(main())
