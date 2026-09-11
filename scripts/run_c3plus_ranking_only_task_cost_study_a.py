#!/usr/bin/env python3
"""Fail-closed serial supervisor for ranking-only Study A.

Without ``--execute`` this performs preflight only. Live execution additionally
requires a manifest-specific ``USER_APPROVAL.json``. Process ownership,
affinity, timing validation, and exact cleanup reuse the validated serial
supervisor; this module changes only the manifest and run environment.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/c3plus_ranking_only_task_cost_study_a"
MANIFEST = OUT / "manifest/study_a_manifest.csv"
CONFIG = OUT / "config/study_a_config.json"
APPROVAL = OUT / "USER_APPROVAL.json"
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
    if len(rows) != 27 or len({row["run_id"] for row in rows}) != 27:
        raise RuntimeError("Study A manifest must contain 27 unique run IDs")
    return rows


def run_environment(row: dict[str, str], attempt: Path) -> dict[str, str]:
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("SAMPLING_C3_OBS_RELU") or \
           key.startswith("SAMPLING_C3_OBS_EXP") or key in {
               "SAMPLING_C3_INNER_OBS_MODE", "SAMPLING_C3_OBJ_NONPEN",
               "SAMPLING_C3_RANK_OBS_MODE", "SAMPLING_C3_EXPERIMENT_SEED",
               "SAMPLING_C3_TRANSLATION_COST_SCALE",
               "SAMPLING_C3_ORIENTATION_COST_SCALE",
               "SAMPLING_C3_ALPHA_P_RANK", "SAMPLING_C3_ALPHA_THETA_RANK",
               "SAMPLING_C3_FORENSICS_LOG_DIR",
           }:
            env.pop(key)
    env_file = supervisor.WT / row["env_file"] if row["env_file"] else None
    env = supervisor.source_environment(env, env_file)
    # Fail closed if a scene environment tries to reactivate an inner soft
    # obstacle potential or the old shared-Q alpha plumbing.
    forbidden = {
        "SAMPLING_C3_TRANSLATION_COST_SCALE",
        "SAMPLING_C3_ORIENTATION_COST_SCALE",
    }
    leaked = sorted(key for key in forbidden if key in env and env[key] != "1")
    if leaked:
        raise RuntimeError(f"legacy shared-Q alpha leaked from scene env: {leaked}")
    tmp = attempt / "tmp"
    forensics = attempt / "forensics"
    tmp.mkdir()
    forensics.mkdir()
    env.update(supervisor.THREAD_LIMITS)
    env.update({
        "TMPDIR": str(tmp),
        "SAMPLING_C3_EXPERIMENT_SEED": row["seed"],
        "SAMPLING_C3_ALPHA_P_RANK": row["alpha_p_rank"],
        "SAMPLING_C3_ALPHA_THETA_RANK": row["alpha_theta_rank"],
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
    env.pop("SAMPLING_C3_TRANSLATION_COST_SCALE", None)
    env.pop("SAMPLING_C3_ORIENTATION_COST_SCALE", None)
    return env


def first_event_knots(attempt: Path) -> list[dict[str, str]]:
    path = attempt / "forensics/forensics_candidate_knots.csv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        first = next(reader)
        event_id = first["event_id"]
        rows = [first]
        for row in reader:
            if row["event_id"] != event_id:
                break
            rows.append(row)
    return rows


def close(a: str, b: str, tolerance: float = 1e-10) -> bool:
    try:
        return math.isclose(float(a), float(b), abs_tol=tolerance, rel_tol=tolerance)
    except ValueError:
        return a == b


def check_frozen_first_event(block_index: str,
                             rows: list[dict[str, str]]) -> None:
    block = [row for row in rows if row["block_index"] == block_index]
    attempts = {row["condition_id"]: supervisor.accepted_attempt(row)
                for row in block}
    if len(attempts) != 3 or any(value is None for value in attempts.values()):
        return
    output = OUT / "frozen_inner_checks" / f"block_{int(block_index):02d}.json"
    if output.exists():
        if not json.loads(output.read_text()).get("passed"):
            raise RuntimeError(f"prior frozen-inner check failed: {output}")
        return
    candidate_rows: dict[str, list[dict[str, str]]] = {}
    knot_rows: dict[str, list[dict[str, str]]] = {}
    for condition, attempt in attempts.items():
        assert attempt is not None
        _, candidate_rows[condition] = supervisor.first_event(attempt)
        knot_rows[condition] = first_event_knots(attempt)
    baseline_candidates = candidate_rows["A1"]
    baseline_knots = knot_rows["A1"]
    candidate_identity_fields = (
        "candidate_id", "candidate_source", "candidate_object_frame_x",
        "candidate_object_frame_y", "face_id", "face_bin_id",
        "contact_sector_id", "J_translation", "J_orientation",
        "J_angular_velocity", "J_linear_velocity",
    )
    knot_identity_fields = tuple(
        field for field in baseline_knots[0]
        if field not in {"time", "event_id", "selected"}
    )
    checks: dict[str, bool] = {}
    for condition in ("A2", "A3"):
        current_candidates = candidate_rows[condition]
        current_knots = knot_rows[condition]
        checks[f"{condition}_candidate_count"] = \
            len(current_candidates) == len(baseline_candidates)
        checks[f"{condition}_knot_count"] = len(current_knots) == len(baseline_knots)
        checks[f"{condition}_candidate_pool_and_source_components"] = (
            len(current_candidates) == len(baseline_candidates) and all(
                all(close(left[field], right[field])
                    for field in candidate_identity_fields)
                for left, right in zip(baseline_candidates, current_candidates)
            )
        )
        checks[f"{condition}_forward_rollout"] = (
            len(current_knots) == len(baseline_knots) and all(
                all(close(left[field], right[field])
                    for field in knot_identity_fields)
                for left, right in zip(baseline_knots, current_knots)
            )
        )
    payload = {
        "block_index": int(block_index),
        "conditions": sorted(attempts),
        "checks": checks,
        "passed": all(checks.values()),
        "scope": "first control event before cost-driven closed-loop divergence",
    }
    supervisor.atomic_json(output, payload)
    if not payload["passed"]:
        supervisor.atomic_json(
            OUT / "state/PAUSED_INNER_C3_NOT_FROZEN.json", payload)
        raise RuntimeError("INNER_C3_NOT_FROZEN")


def preflight() -> int:
    rows = read_manifest()
    config = json.loads(CONFIG.read_text())
    receipt = json.loads((OUT / "provenance/preparation_receipt.json").read_text())
    errors: list[str] = []
    if config["scientific_runs"] != 27:
        errors.append("config run count is not 27")
    if {row["condition_id"] for row in rows} != {"A1", "A2", "A3"}:
        errors.append("condition set differs from A1/A2/A3")
    if {row["seed"] for row in rows} != {"0"}:
        errors.append("seed set differs from {0}")
    if {row["start_id"] for row in rows} != {"s1", "s3", "s5"}:
        errors.append("start set differs from s1/s3/s5")
    if {row["rotation_label"] for row in rows} != {
            "rot000", "rotCCW090", "rotCW090"}:
        errors.append("rotation set differs from 0/+90/-90")
    if any(row["scene"] != "open_task" for row in rows):
        errors.append("non-open task present")
    if any(row["alpha_p_rank"] != "1.0" for row in rows):
        errors.append("alpha_p_rank is not frozen at 1")
    expected_theta = {"A1": "1.0", "A2": "0.5", "A3": "0.25"}
    if any(row["alpha_theta_rank"] != expected_theta[row["condition_id"]]
           for row in rows):
        errors.append("alpha_theta_rank mapping differs")
    if any(row["inner_soft_obstacle_mode"] != "none" for row in rows):
        errors.append("inner soft obstacle mode is not none")
    if receipt["manifest_sha256"] != supervisor.sha256(MANIFEST):
        errors.append("manifest checksum differs from preparation receipt")
    checksum_paths = {
        "controller_source_sha256": supervisor.WT /
            "systems/controllers/sampling_based_c3_controller.cc",
        "ranking_helper_sha256": supervisor.WT /
            "systems/controllers/ranking_task_cost.h",
        "controller_binary_sha256": supervisor.BIN /
            "franka_sampling_c3_controller",
        "config_sha256": CONFIG,
        "serial_supervisor_sha256": BASE_PATH,
    }
    for field, path in checksum_paths.items():
        if not path.is_file() or receipt[field] != supervisor.sha256(path):
            errors.append(f"{field} differs from preparation receipt")
    if any(not (supervisor.WT / "examples/sampling_c3" /
                row["demo_name"]).is_dir() for row in rows):
        errors.append("materialized demo missing")
    if not (supervisor.BIN / "franka_sampling_c3_controller").is_file():
        errors.append("controller binary missing")
    payload: dict[str, Any] = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "rows": len(rows),
        "manifest_sha256": supervisor.sha256(MANIFEST),
        "approval_present": APPROVAL.exists(),
        "scientific_runs_started": False,
        "machine_gate": supervisor.machine_gate(),
    }
    supervisor.atomic_json(OUT / "provenance/latest_preflight.json", payload)
    print(json.dumps(payload, sort_keys=True))
    return 0 if not errors else 1


supervisor.read_manifest = read_manifest
supervisor.run_environment = run_environment
supervisor.check_open_negative_control = check_frozen_first_event


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
        return supervisor.execute()
    return preflight()


if __name__ == "__main__":
    raise SystemExit(main())
