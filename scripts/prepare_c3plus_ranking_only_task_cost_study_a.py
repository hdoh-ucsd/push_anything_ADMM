#!/usr/bin/env python3
"""Prepare the 27-run ranking-only Study A manifest without launching it."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
WT = ROOT / "external/oim_c++_anything/.claude/worktrees/oim-scene-sync-metrics"
SYNC = ROOT / "results/benchmark_sync_rot15"
OUT = ROOT / "results/c3plus_ranking_only_task_cost_study_a"
SCENE = "open_task"
STARTS = (1, 3, 5)
ROTATIONS = ("rot000", "rotCCW090", "rotCW090")
CONDITIONS: tuple[dict[str, Any], ...] = (
    {"condition_id": "A1", "alpha_p_rank": 1.0,
     "alpha_theta_rank": 1.0, "F": 1.0},
    {"condition_id": "A2", "alpha_p_rank": 1.0,
     "alpha_theta_rank": 0.5, "F": 2.0},
    {"condition_id": "A3", "alpha_p_rank": 1.0,
     "alpha_theta_rank": 0.25, "F": 4.0},
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
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def load_cases() -> dict[str, dict[str, Any]]:
    payload = yaml.safe_load((SYNC / "c3plus_cases.yaml").read_text())
    cases = {
        str(row["case_id"]): dict(row)
        for row in payload["cases"]
        if row["scene"] == SCENE
    }
    requested = {
        f"s{start:02d}_g02_{rotation}"
        for start in STARTS for rotation in ROTATIONS
    }
    if not requested <= cases.keys():
        raise RuntimeError(f"missing synchronized cases: {sorted(requested - cases.keys())}")
    return {case_id: cases[case_id] for case_id in sorted(requested)}


def manifest(cases: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_index = 0
    block_index = 0
    delta = {
        "rot000": 0.0,
        "rotCCW090": math.pi / 2.0,
        "rotCW090": -math.pi / 2.0,
    }
    for start in STARTS:
        for rotation in ROTATIONS:
            block_index += 1
            case_id = f"s{start:02d}_g02_{rotation}"
            case = cases[case_id]
            offset = (block_index - 1) % len(CONDITIONS)
            order = CONDITIONS[offset:] + CONDITIONS[:offset]
            for condition_order, condition in enumerate(order, start=1):
                run_index += 1
                rows.append({
                    "run_index": run_index,
                    "block_index": block_index,
                    "condition_order_in_block": condition_order,
                    "run_id": f"ROA_{run_index:03d}_{case_id}_{condition['condition_id']}",
                    **condition,
                    "scene": SCENE,
                    "start_id": f"s{start}",
                    "goal_id": "g2",
                    "rotation_label": rotation,
                    "delta_yaw_rad": delta[rotation],
                    "seed": 0,
                    "case_id": case_id,
                    "demo_name": case["generated_demo"],
                    "goal_x": case["goal_x"],
                    "goal_y": case["goal_y"],
                    "goal_z": case["goal_z"],
                    "goal_yaw_rad": 2.0 * math.atan2(
                        float(case["goal_qz"]), float(case["goal_qw"])),
                    "env_file": case.get("env_file", ""),
                    "lcm_port": 28000 + run_index,
                    "cap_wall_s": 600,
                    "max_live_concurrency": 1,
                    "inner_soft_obstacle_mode": "none",
                    "ranking_obstacle_mode": "none_open_task",
                    "status": "PENDING",
                })
    if len(rows) != 27:
        raise AssertionError(f"expected 27 rows, got {len(rows)}")
    return rows


def main() -> int:
    cases = load_cases()
    for case in cases.values():
        demo = WT / "examples/sampling_c3" / str(case["generated_demo"])
        if not demo.is_dir():
            raise FileNotFoundError(f"synchronized demo is not materialized: {demo}")
    rows = manifest(cases)
    config = {
        "study": "A_outer_ranking_translation_orientation",
        "scientific_runs": 27,
        "scene": SCENE,
        "starts": ["s1", "s3", "s5"],
        "goal": "g2",
        "rotations": ["0", "+90", "-90"],
        "seed": 0,
        "conditions": CONDITIONS,
        "inner_c3": {
            "boundary": "source_faithful_frozen",
            "pose_object_position_diag": [10000, 10000, 6000],
            "position_object_position_diag": [12500, 12500, 12500],
            "quaternion_hessian_scale": 510,
            "R_diag": [0.01, 0.01, 0.01],
            "N": 5,
            "gamma": 1,
            "dt_position": 0.1,
            "dt_pose": 0.05,
            "cost_switch_m": 0.50,
            "admm_iterations": 3,
            "rho_scale": 3,
            "soft_obstacle_mode": "none",
        },
        "affinity": {
            "system": "0-1,8-15", "planner": "2-5", "osc": "6-7",
            "sim": "6-7", "recorder": "0-1",
        },
        "thread_limits": {
            "OMP_NUM_THREADS": 3, "OMP_THREAD_LIMIT": 3,
            "OMP_DYNAMIC": "FALSE", "OPENBLAS_NUM_THREADS": 1,
            "MKL_NUM_THREADS": 1, "NUMEXPR_NUM_THREADS": 1,
            "BLIS_NUM_THREADS": 1, "GOTO_NUM_THREADS": 1,
            "VECLIB_MAXIMUM_THREADS": 1,
        },
    }
    write_new_or_identical(
        OUT / "config/study_a_config.json",
        json.dumps(config, indent=2, sort_keys=True) + "\n",
    )
    write_new_or_identical(
        OUT / "manifest/study_a_manifest.csv", csv_text(rows))
    write_new_or_identical(
        OUT / "state/initial_status.csv",
        csv_text([{"run_id": row["run_id"], "status": "PENDING"}
                  for row in rows]),
    )
    receipt = {
        "outer_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "controller_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=WT, text=True).strip(),
        "controller_source_sha256": sha256(
            WT / "systems/controllers/sampling_based_c3_controller.cc"),
        "ranking_helper_sha256": sha256(
            WT / "systems/controllers/ranking_task_cost.h"),
        "controller_binary_sha256": sha256(
            WT / "bazel-bin/examples/sampling_c3/franka_sampling_c3_controller"),
        "config_sha256": sha256(OUT / "config/study_a_config.json"),
        "controller_worktree_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=WT, text=True).strip()),
        "serial_supervisor_sha256": sha256(
            ROOT / "scripts/run_c3plus_geometry_rotation_cost_replication.py"),
        "manifest_sha256": sha256(OUT / "manifest/study_a_manifest.csv"),
        "scientific_runs_launched": 0,
        "approval_required": True,
    }
    write_new_or_identical(
        OUT / "provenance/preparation_receipt.json",
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    )
    write_new_or_identical(
        OUT / "APPROVAL_REQUIRED",
        "Study A may not start until explicit user approval is recorded.\n",
    )
    print(json.dumps({"prepared_runs": 27, "scientific_runs_launched": 0},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
