#!/usr/bin/env python3
"""Prepare, but never launch, the final serial C3+ cost-replication study.

The generated manifest is immutable experiment input.  Re-running this script
accepts an existing artifact only when its bytes are identical.  Native C3+
demo directories are materialized additively from the audited rot15 adapters;
an existing directory is validated and never overwritten.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.rot15.benchmark import validate_artifacts
from benchmarks.rot15.launcher import materialize_c3plus_case


WT = ROOT / "external/oim_c++_anything/.claude/worktrees/oim-scene-sync-metrics"
OUT = ROOT / "results/c3plus_geometry_rotation_cost_replication"
SYNC = ROOT / "results/benchmark_sync_rot15"
TUNING = ROOT / "results/c3plus_obstacle_cost_tuning_v2"
SCENES = ("open_task", "single_obstacle", "shelf_gap")
SEEDS = (0, 1)
CAP_S = 600
PORT_BASE = 27000

COSTS: tuple[dict[str, Any], ...] = (
    {"config_id": "C1", "source_id": "OC01", "task_base": "TR08", "alpha_p": 2.0,
     "alpha_theta": 1.0, "F_rho": 2.0, "family": "exponential_footprint",
     "range_name": "sigma", "range_m": 0.015, "weight": 764.3857383844456},
    {"config_id": "C2", "source_id": "MATCHED_OC01", "task_base": "TR08", "alpha_p": 2.0,
     "alpha_theta": 1.0, "F_rho": 2.0, "family": "relu_footprint",
     "range_name": "epsilon", "range_m": 0.045, "weight": 764.3857383844456},
    {"config_id": "C3", "source_id": "OC04", "task_base": "TR07", "alpha_p": 2.0,
     "alpha_theta": 0.5, "F_rho": 4.0, "family": "exponential_footprint",
     "range_name": "sigma", "range_m": 0.025, "weight": 778.8735787922371},
    {"config_id": "C4", "source_id": "MATCHED_OC04", "task_base": "TR07", "alpha_p": 2.0,
     "alpha_theta": 0.5, "F_rho": 4.0, "family": "relu_footprint",
     "range_name": "epsilon", "range_m": 0.075, "weight": 778.8735787922371},
    {"config_id": "C5", "source_id": "OC08", "task_base": "TR07", "alpha_p": 2.0,
     "alpha_theta": 0.5, "F_rho": 4.0, "family": "relu_footprint",
     "range_name": "epsilon", "range_m": 0.050, "weight": 1173.2815219621552},
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_or_identical(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f"refusing to overwrite non-identical artifact: {path}")
        return
    path.write_text(text)


def csv_text(rows: list[dict[str, Any]]) -> str:
    from io import StringIO
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def load_cases() -> list[dict[str, Any]]:
    return [
        dict(row) for row in yaml.safe_load((SYNC / "c3plus_cases.yaml").read_text())["cases"]
        if row["scene"] in SCENES
    ]


def scalar_values(path: Path, key: str) -> list[float]:
    match = re.search(rf"(?m)^{re.escape(key)}:\s*\[([^\]]+)\]", path.read_text())
    if not match:
        raise AssertionError(f"{path}: missing {key}")
    return [float(item) for item in match.group(1).split(",")]


def validate_materialized(case: Mapping[str, Any]) -> None:
    demo = WT / "examples/sampling_c3" / str(case["generated_demo"])
    params = demo / "parameters"
    if not params.is_dir():
        raise AssertionError(f"missing materialized demo: {demo}")
    actual_start = scalar_values(params / "sim_params.yaml", "q_init_object")
    expected_start = [float(case[f"start_q{k}"]) for k in "wxyz"] + [
        float(case[f"start_{axis}"]) for axis in "xyz"
    ]
    actual_goal = scalar_values(params / "goal_params.yaml", "fixed_target_position")
    expected_goal = [float(case[f"goal_{axis}"]) for axis in "xyz"]
    actual_quat = scalar_values(params / "goal_params.yaml", "fixed_target_orientation")
    expected_quat = [float(case[f"goal_q{k}"]) for k in "wxyz"]
    for label, actual, expected in (
        ("start", actual_start, expected_start), ("goal", actual_goal, expected_goal),
        ("goal quaternion", actual_quat, expected_quat),
    ):
        if len(actual) != len(expected) or max(abs(a - b) for a, b in zip(actual, expected)) > 1e-12:
            raise AssertionError(f"{demo}: {label} differs from audited adapter")


def ensure_demos(cases: list[dict[str, Any]]) -> tuple[int, int]:
    made = existing = 0
    for case in cases:
        destination = WT / "examples/sampling_c3" / str(case["generated_demo"])
        if destination.exists():
            existing += 1
        else:
            materialize_c3plus_case(case, WT)
            made += 1
        validate_materialized(case)
    return made, existing


def tuning_rows() -> list[dict[str, str]]:
    path = TUNING / "metrics/per_run_results.csv"
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def pairing_evidence(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for oc in ("OC01", "OC04", "OC08"):
        for task in ("TR07", "TR08"):
            selected = [r for r in rows if r["obstacle_config_id"] == oc and r["task_cost_id"] == task]
            valid = [r for r in selected if r["timing_valid"] in {"1", "True", "true"}]
            successes = [r for r in valid if r["success"] in {"1", "True", "true"}]
            retained = [r for r in valid if r["retained_success"] in {"1", "True", "true"}]
            times = [float(r["T_goal"]) for r in successes if r["T_goal"] and r["T_goal"].lower() != "nan"]
            result.append({
                "obstacle_config_id": oc, "task_base": task, "rows": len(selected),
                "timing_valid_rows": len(valid), "successes": len(successes),
                "retained_successes": len(retained),
                "successful_T_goal_s": ";".join(f"{value:.6g}" for value in times),
                "scenes": ";".join(r["scenario"] for r in valid),
                "failure_classes": ";".join(r["failure_class"] for r in valid),
            })
    return result


def manifest(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {(str(row["scene"]), str(row["case_id"])): row for row in cases}
    rows: list[dict[str, Any]] = []
    run_index = 0
    block_index = 0
    rotations = ("rot000", "rotCCW090", "rotCW090")
    rotation_delta = {"rot000": 0.0, "rotCCW090": math.pi / 2.0, "rotCW090": -math.pi / 2.0}
    for scene in SCENES:
        for start in range(1, 6):
            for rotation in rotations:
                case_id = f"s{start:02d}_g02_{rotation}"
                case = index[(scene, case_id)]
                for seed in SEEDS:
                    block_index += 1
                    rotation_offset = (block_index - 1) % len(COSTS)
                    ordered = COSTS[rotation_offset:] + COSTS[:rotation_offset]
                    for cost_order, cost in enumerate(ordered, start=1):
                        run_index += 1
                        config_id = str(cost["config_id"])
                        rows.append({
                            "run_index": run_index,
                            "block_index": block_index,
                            "cost_order_in_block": cost_order,
                            "run_id": f"GRC3P_{run_index:03d}_{scene}_{case_id}_seed{seed}_{config_id}",
                            "config_id": config_id,
                            "source_obstacle_id": cost["source_id"],
                            "task_base": cost["task_base"],
                            "alpha_p": cost["alpha_p"],
                            "alpha_theta": cost["alpha_theta"],
                            "F_rho": cost["F_rho"],
                            "obstacle_family": cost["family"],
                            "obstacle_range_name": cost["range_name"],
                            "obstacle_range_m": cost["range_m"],
                            "obstacle_weight": cost["weight"],
                            "scene": scene,
                            "case_id": case_id,
                            "start_id": f"s{start}",
                            "goal_id": "g2",
                            "rotation_label": rotation,
                            "delta_yaw_rad": rotation_delta[rotation],
                            "seed": seed,
                            "demo_name": case["generated_demo"],
                            "goal_x": case["goal_x"], "goal_y": case["goal_y"],
                            "goal_z": case["goal_z"],
                            "goal_yaw_rad": 2.0 * math.atan2(float(case["goal_qz"]), float(case["goal_qw"])),
                            "env_file": case.get("env_file", ""),
                            "lcm_port": PORT_BASE + run_index,
                            "cap_wall_s": CAP_S,
                            "planned_status": "PENDING",
                        })
    if len(rows) != 450:
        raise AssertionError(f"expected 450 manifest rows, got {len(rows)}")
    return rows


def main() -> int:
    validate_artifacts(SYNC)
    cases = load_cases()
    if len(cases) != 45:
        raise AssertionError(f"expected 45 geometry cases, got {len(cases)}")
    made, existing = ensure_demos(cases)
    evidence = pairing_evidence(tuning_rows())
    rows = manifest(cases)

    config_payload = {
        "campaign": "final_xarm6_c3plus_geometry_rotation_cost_replication",
        "max_concurrent_live_simulations": 1,
        "capacity_verdict": "MAX_SAFE_LANES_1",
        "costs": COSTS,
        "seeds": SEEDS,
        "cap_wall_s": CAP_S,
        "affinity": {"system": "0-1,8-15", "planner": "2-5", "osc": "6-7", "sim": "6-7", "recorder": "0-1"},
        "thread_limits": {"OMP_NUM_THREADS": 3, "OMP_THREAD_LIMIT": 3, "OMP_DYNAMIC": "FALSE",
                          "OPENBLAS_NUM_THREADS": 1, "MKL_NUM_THREADS": 1, "NUMEXPR_NUM_THREADS": 1,
                          "BLIS_NUM_THREADS": 1, "GOTO_NUM_THREADS": 1, "VECLIB_MAXIMUM_THREADS": 1},
        "solver_num_threads_yaml_unchanged": 5,
    }
    write_new_or_identical(OUT / "configs/cost_configurations.json", json.dumps(config_payload, indent=2, sort_keys=True) + "\n")
    write_new_or_identical(OUT / "audit/task_cost_pairing_evidence.csv", csv_text(evidence))
    write_new_or_identical(OUT / "manifest/experiment_manifest.csv", csv_text(rows))
    initial = [{"run_id": row["run_id"], "status": "PENDING", "accepted_attempt": ""} for row in rows]
    write_new_or_identical(OUT / "state/initial_manifest_status.csv", csv_text(initial))
    provenance = {
        "prepared_at": subprocess.check_output(["date", "--iso-8601=seconds"], text=True).strip(),
        "outer_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "c3plus_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=WT, text=True).strip(),
        "c3plus_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=WT, text=True).strip()),
        "controller_source_sha256": sha256(WT / "systems/controllers/sampling_based_c3_controller.cc"),
        "controller_binary_sha256": sha256(WT / "bazel-bin/examples/sampling_c3/franka_sampling_c3_controller"),
        "sim_binary_sha256": sha256(WT / "bazel-bin/examples/sampling_c3/franka_sim"),
        "osc_binary_sha256": sha256(WT / "bazel-bin/examples/sampling_c3/franka_osc_controller"),
        "canonical_cases_sha256": sha256(SYNC / "canonical_cases.csv"),
        "adapter_sha256": sha256(SYNC / "c3plus_cases.yaml"),
        "pairing_evidence_sha256": sha256(TUNING / "metrics/per_run_results.csv"),
        "manifest_sha256": sha256(OUT / "manifest/experiment_manifest.csv"),
        "materialized_demos_created": made,
        "materialized_demos_preexisting_and_validated": existing,
        "scientific_runs_launched": 0,
        "approval_required": True,
    }
    write_new_or_identical(OUT / "provenance/preparation_receipt.json", json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    write_new_or_identical(OUT / "APPROVAL_REQUIRED", "No scientific run may start until explicit user approval is recorded.\n")
    print(json.dumps({"prepared_runs": len(rows), "demos_created": made, "demos_validated_existing": existing,
                      "scientific_runs_launched": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
