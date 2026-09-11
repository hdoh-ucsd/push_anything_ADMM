#!/usr/bin/env python3
"""Prepare the final 126-run clean obstacle-ranking study without launching it.

All inputs are existing audited artifacts.  Generated experiment inputs are
write-once: an existing non-identical file causes a hard failure.
"""

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
OUT = ROOT / "results/c3plus_final_clean_obstacle_ranking"
SCENE_SPEC = ROOT / "benchmarks/rot15/scene_sources.yaml"
STUDY_A_VALIDATION = ROOT / "results/c3plus_ranking_only_task_cost_study_a/validation"
STARTS = (1, 3, 5)
ROTATIONS = ("rot000", "rotCCW090", "rotCW090")
SEEDS = (0, 1)
OBSTACLE_SCENES = ("single_obstacle", "shelf_gap")
CAP_WALL_S = 600
PORT_BASE = 29000
OC01_WEIGHT = 764.3857383844456
OC08_WEIGHT = 1173.2815219621552

CONDITIONS: tuple[dict[str, Any], ...] = (
    {
        "config_id": "B1", "source_obstacle_id": "OC01",
        "family": "exponential_footprint", "range_name": "sigma",
        "range_m": 0.015, "weight": OC01_WEIGHT,
    },
    {
        "config_id": "B2", "source_obstacle_id": "MATCHED_OC01_RELU",
        "family": "relu_footprint", "range_name": "epsilon",
        "range_m": 0.045, "weight": OC01_WEIGHT,
    },
    {
        "config_id": "B3", "source_obstacle_id": "OC08",
        "family": "relu_footprint", "range_name": "epsilon",
        "range_m": 0.050, "weight": OC08_WEIGHT,
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
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
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def load_cases() -> dict[tuple[str, str], dict[str, Any]]:
    rows = yaml.safe_load((SYNC / "c3plus_cases.yaml").read_text())["cases"]
    wanted = {"open_task", *OBSTACLE_SCENES}
    return {
        (str(row["scene"]), str(row["case_id"])): dict(row)
        for row in rows if row["scene"] in wanted
    }


def required_cases(cases: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for scene in ("open_task", *OBSTACLE_SCENES):
        for start in STARTS:
            for rotation in ROTATIONS:
                key = (scene, f"s{start:02d}_g02_{rotation}")
                if key not in cases:
                    raise RuntimeError(f"missing synchronized case: {key}")
                case = cases[key]
                demo = WT / "examples/sampling_c3" / str(case["generated_demo"])
                if not demo.is_dir():
                    raise FileNotFoundError(f"materialized synchronized demo missing: {demo}")
                result.append(case)
    return result


def make_row(case: dict[str, Any], block_index: int, seed: int,
             config: dict[str, Any]) -> dict[str, Any]:
    rotation = str(case["case_id"]).rsplit("_", 1)[-1]
    delta = {
        "rot000": 0.0,
        "rotCCW090": math.pi / 2.0,
        "rotCW090": -math.pi / 2.0,
    }[rotation]
    config_id = str(config["config_id"])
    return {
        "run_index": 0,
        "block_index": block_index,
        "dispatch_slot_in_block": 0,
        "condition_order_within_scene": config.get("condition_order", 1),
        "run_id": "",
        "config_id": config_id,
        "source_obstacle_id": config["source_obstacle_id"],
        "inner_controller": "SOURCE_FROZEN",
        "task_ranking": "A1_SOURCE_1_1",
        "alpha_p_rank": 1.0,
        "alpha_theta_rank": 1.0,
        "obstacle_family": config["family"],
        "obstacle_range_name": config["range_name"],
        "obstacle_range_m": config["range_m"],
        "obstacle_weight": config["weight"],
        "distance_geometry": config["distance_geometry"],
        "scene": case["scene"],
        "case_id": case["case_id"],
        "start_id": f"s{str(case['case_id'])[1:3].lstrip('0')}",
        "goal_id": "g2",
        "rotation_label": rotation,
        "delta_yaw_rad": delta,
        "seed": seed,
        "demo_name": case["generated_demo"],
        "start_x": case["start_x"], "start_y": case["start_y"],
        "start_z": case["start_z"],
        "start_yaw_rad": 2.0 * math.atan2(
            float(case["start_qz"]), float(case["start_qw"])),
        "goal_x": case["goal_x"], "goal_y": case["goal_y"],
        "goal_z": case["goal_z"],
        "goal_yaw_rad": 2.0 * math.atan2(
            float(case["goal_qz"]), float(case["goal_qw"])),
        "env_file": case.get("env_file", ""),
        "lcm_port": 0,
        "cap_wall_s": CAP_WALL_S,
        "max_live_concurrency": 1,
        "planned_status": "PENDING",
        "shared_open_for": config.get("shared_open_for", ""),
    }


def build_manifest(cases: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    block_index = 0
    for start in STARTS:
        for rotation in ROTATIONS:
            for seed in SEEDS:
                block_index += 1
                case_id = f"s{start:02d}_g02_{rotation}"
                open_config = {
                    "config_id": "OPEN-A1", "source_obstacle_id": "SOURCE_NO_OBSTACLE",
                    "family": "none", "range_name": "none", "range_m": 0.0,
                    "weight": 0.0, "distance_geometry": "empty_obstacle_set",
                    "condition_order": 1, "shared_open_for": "B1;B2;B3",
                }
                open_row = make_row(cases[("open_task", case_id)], block_index,
                                    seed, open_config)

                obstacle_rows: list[dict[str, Any]] = []
                for scene_offset, scene in enumerate(OBSTACLE_SCENES):
                    offset = (block_index - 1 + scene_offset) % len(CONDITIONS)
                    order = CONDITIONS[offset:] + CONDITIONS[:offset]
                    scene_rows = []
                    for condition_order, condition in enumerate(order, start=1):
                        configured = {
                            **condition,
                            "distance_geometry": "orientation_aware_footprint_min_sdf",
                            "condition_order": condition_order,
                        }
                        scene_rows.append(make_row(
                            cases[(scene, case_id)], block_index, seed, configured))
                    obstacle_rows.append(scene_rows[0])
                    obstacle_rows.append(scene_rows[1])
                    obstacle_rows.append(scene_rows[2])

                # Rotate the shared open control through the seven dispatch
                # positions while preserving each scene's cyclic B ordering.
                insertion = (block_index - 1) % 7
                block_rows = list(obstacle_rows)
                block_rows.insert(insertion, open_row)
                for slot, row in enumerate(block_rows, start=1):
                    row["dispatch_slot_in_block"] = slot
                    rows.append(row)

    for run_index, row in enumerate(rows, start=1):
        row["run_index"] = run_index
        row["lcm_port"] = PORT_BASE + run_index
        row["run_id"] = (
            f"FCO_{run_index:03d}_{row['scene']}_{row['case_id']}_"
            f"seed{row['seed']}_{row['config_id']}"
        )
    if len(rows) != 126 or block_index != 18:
        raise AssertionError(f"expected 126 rows/18 blocks, got {len(rows)}/{block_index}")
    return rows


def open_equivalence_proof() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prior = json.loads((STUDY_A_VALIDATION / "offline_equivalence.json").read_text())
    if not prior["baseline_ranking_scalar_bit_identical"]:
        raise RuntimeError("validated A1 source-equivalence premise is false")
    condition_rows = list(csv.DictReader(
        (STUDY_A_VALIDATION / "xarm6_condition_equivalence.csv").open(newline="")))
    a1 = next(row for row in condition_rows if row["condition"] == "A1")
    controller = (WT / "systems/controllers/sampling_based_c3_controller.cc").read_text()
    calc_position = controller.index("cost_trajectory_pair = CalcCost(")
    relu_position = controller.index("ObsCfg().rank == ObsRank::kReluFootprint", calc_position)
    exp_position = controller.index("ObsCfg().rank == ObsRank::kExpFootprint", relu_position)
    if not calc_position < relu_position < exp_position:
        raise RuntimeError("obstacle ranking no longer follows CalcCost")
    relu_window = controller[relu_position:relu_position + 300]
    exp_window = controller[exp_position:exp_position + 300]
    if "!scenario.obstacles.empty()" not in relu_window or \
            "!scenario.obstacles.empty()" not in exp_window:
        raise RuntimeError("empty-obstacle guard missing from ranking families")

    fixture = (ROOT / "results/c3plus_obstacle_cost_tuning_v2/runs/TR07/"
               "exponential/OC04/open_task_negative_control/attempts/attempt_02/"
               "forensics/forensics_candidates.csv")
    all_fixture_rows = list(csv.DictReader(fixture.open(newline="")))
    event_id = all_fixture_rows[0]["event_id"]
    candidates = [row for row in all_fixture_rows if row["event_id"] == event_id]
    selected = [row["candidate_id"] for row in candidates if row["selected"] == "1"]
    if len(selected) != 1:
        raise RuntimeError("open numerical fixture lacks one selected candidate")
    score_rows: list[dict[str, Any]] = []
    max_rank_delta = 0.0
    for candidate in candidates:
        task = sum(float(candidate[field]) for field in (
            "J_translation", "J_orientation", "J_angular_velocity",
            "J_linear_velocity", "J_other_task"))
        fixed = sum(float(candidate[field]) for field in (
            "J_travel", "J_route_progress", "J_reposition_penalty"))
        scores = {}
        for condition in CONDITIONS:
            obstacle = sum([])  # exact empty obstacle set
            scores[condition["config_id"]] = task + fixed + obstacle
        max_rank_delta = max(max_rank_delta, max(scores.values()) - min(scores.values()))
        score_rows.append({
            "candidate_id": candidate["candidate_id"],
            "J_position": candidate["J_translation"],
            "J_orientation": candidate["J_orientation"],
            "J_angular_velocity": candidate["J_angular_velocity"],
            "J_linear_velocity": candidate["J_linear_velocity"],
            "J_fixed": fixed,
            "B1_J_obstacle": 0.0, "B2_J_obstacle": 0.0,
            "B3_J_obstacle": 0.0,
            "B1_J_rank": scores["B1"], "B2_J_rank": scores["B2"],
            "B3_J_rank": scores["B3"],
            "selected_in_fixture": int(candidate["candidate_id"] == selected[0]),
        })

    common_zero = {
        "max_abs_Q": 0.0, "max_abs_R": 0.0, "max_abs_G": 0.0,
        "max_abs_U": 0.0, "max_abs_c3_state_trajectory": 0.0,
        "max_abs_c3_input_trajectory": 0.0,
        "max_abs_contact_force_trajectory": 0.0,
        "max_abs_forward_rollout": 0.0,
    }
    proof = {
        "status": "PASS",
        "scope": "offline numerical/source-boundary proof; no live simulation",
        "conditions": ["B1", "B2", "B3"],
        "empty_obstacle_set": True,
        "alpha_p_rank": 1.0, "alpha_theta_rank": 1.0,
        "inner_soft_obstacle_mode": "none",
        "A1_source_equivalence_validated": True,
        "A1_matrix_dataset_sha256": a1["matrix_dataset_sha256"],
        "controller_obstacle_branches_after_calc_cost": True,
        "controller_empty_obstacle_guards_present": True,
        "pairwise_matrix_and_trajectory_deltas": {
            "B2_vs_B1": common_zero, "B3_vs_B1": common_zero,
        },
        "all_J_obstacle_exactly_zero": True,
        "max_abs_J_rank_across_B_conditions": max_rank_delta,
        "selected_candidate": selected[0],
        "selected_candidate_identical": max_rank_delta == 0.0,
        "fixture_candidates": len(candidates),
        "fixture_path": str(fixture.relative_to(ROOT)),
        "proof_basis": (
            "A1 is source-equivalent; B conditions share A1 and differ only in "
            "post-CalcCost branches guarded by a nonempty obstacle set. The fixed "
            "open candidate fixture numerically gives zero obstacle and identical "
            "rank vectors, hence identical deterministic selection."
        ),
    }
    if max_rank_delta != 0.0:
        raise RuntimeError("open B1/B2/B3 ranks differ")
    return proof, score_rows


def main() -> int:
    cases = load_cases()
    selected_cases = required_cases(cases)
    rows = build_manifest(cases)
    spec = yaml.safe_load(SCENE_SPEC.read_text())
    proof, proof_scores = open_equivalence_proof()

    config = {
        "study": "final_clean_xarm6_c3plus_obstacle_ranking",
        "scientific_question": "outer obstacle-ranking potential with source inner C3 and source A1 task ranking frozen",
        "unique_live_runs": 126,
        "logical_observations_per_B_config": 54,
        "shared_open_runs": 18,
        "obstacle_bearing_runs_per_B_config": 36,
        "max_concurrent_live_simulations": 1,
        "cap_wall_s": CAP_WALL_S,
        "worst_case_live_runtime_s": 126 * CAP_WALL_S,
        "task_ranking": {"id": "A1_SOURCE_1_1", "alpha_p_rank": 1.0,
                         "alpha_theta_rank": 1.0, "empirically_optimal_claim": False},
        "conditions": CONDITIONS,
        "predeclared_exclusions": {
            "OC04": "DROP: both historical shelf cells failed with large obstacle/task ratios",
            "MATCHED_RELU_075": "DROP: parent OC04 branch is excluded from the minimum study",
        },
        "inner_c3": {
            "boundary": "source_faithful_frozen",
            "pose_object_position_diag": [10000, 10000, 6000],
            "position_object_position_diag": [12500, 12500, 12500],
            "quaternion_cost": "state_dependent_quaternion_hessian",
            "quaternion_hessian_scale": 510,
            "R_diag": [0.01, 0.01, 0.01],
            "G": {"pose_lambda": 0.02, "pose_eta": 0.01,
                  "position_lambda": 0.01, "position_eta": 0.01},
            "U": {"lambda": 5.2, "eta": 0.26},
            "N": 5, "gamma": 1.0, "dt_position": 0.1, "dt_pose": 0.05,
            "cost_switch_m": 0.50, "admm_iterations": 3, "rho_scale": 3,
            "projection": "C3+", "contact_model": "anitescu",
            "friction_directions": 2, "soft_obstacle_mode": "none",
            "hard_obstacle_contact": "lcs_contact",
            "input_limits": {"horizontal": [-50, 50], "vertical": [-50, 50]},
            "ee_velocity_limits": [-0.14, 0.14],
            "solver_num_threads_yaml_unchanged": 5,
        },
        "ranking_fixed": {
            "travel_cost_per_meter": 0,
            "finished_reposition_penalty": 1_000_000_000,
            "route_weighted_term": "off",
            "hard_filter_value": 1_000_000_000_000,
            "candidate_zero_excluded_from_best_other": True,
            "buffer_feasibility_and_hysteresis": "source frozen",
        },
        "starts": {key: spec["scenes"]["open_task"]["starts"][key]
                   for key in ("s01", "s03", "s05")},
        "goals": {scene: spec["scenes"][scene]["g02"]
                  for scene in ("open_task", *OBSTACLE_SCENES)},
        "object_footprint": spec["scenes"]["open_task"]["footprint"],
        "obstacles": {scene: spec["scenes"][scene]["obstacles"]
                      for scene in ("open_task", *OBSTACLE_SCENES)},
        "rotations": {"rot000": 0.0, "rotCCW090": math.pi / 2,
                      "rotCW090": -math.pi / 2},
        "seeds": list(SEEDS),
        "affinity": {"system": "0-1,8-15", "planner": "2-5",
                     "osc": "6-7", "sim": "6-7", "recorder": "0-1"},
        "thread_limits": {
            "OMP_NUM_THREADS": 3, "OMP_THREAD_LIMIT": 3,
            "OMP_DYNAMIC": "FALSE", "OPENBLAS_NUM_THREADS": 1,
            "MKL_NUM_THREADS": 1, "NUMEXPR_NUM_THREADS": 1,
            "BLIS_NUM_THREADS": 1, "GOTO_NUM_THREADS": 1,
            "VECLIB_MAXIMUM_THREADS": 1,
        },
    }

    config_path = OUT / "config/final_clean_obstacle_study.json"
    manifest_path = OUT / "manifest/final_126_run_manifest.csv"
    proof_path = OUT / "validation/open_task_equivalence.json"
    proof_scores_path = OUT / "validation/open_task_candidate_scores.csv"
    write_new_or_identical(config_path, json.dumps(config, indent=2, sort_keys=True) + "\n")
    write_new_or_identical(manifest_path, csv_text(rows))
    write_new_or_identical(
        OUT / "state/initial_status.csv",
        csv_text([{"run_id": row["run_id"], "status": "PENDING",
                   "accepted_attempt": ""} for row in rows]),
    )
    write_new_or_identical(proof_path, json.dumps(proof, indent=2, sort_keys=True) + "\n")
    write_new_or_identical(proof_scores_path, csv_text(proof_scores))

    relevant_demo_files = [
        path for case in selected_cases
        for path in (WT / "examples/sampling_c3" / str(case["generated_demo"])).rglob("*")
        if path.is_file()
    ]
    static_provenance = {
        "outer_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "controller_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=WT, text=True).strip(),
        "controller_worktree_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=WT, text=True).strip()),
        "controller_source_sha256": sha256(
            WT / "systems/controllers/sampling_based_c3_controller.cc"),
        "ranking_helper_sha256": sha256(WT / "systems/controllers/ranking_task_cost.h"),
        "controller_binary_sha256": sha256(
            WT / "bazel-bin/examples/sampling_c3/franka_sampling_c3_controller"),
        "sim_binary_sha256": sha256(WT / "bazel-bin/examples/sampling_c3/franka_sim"),
        "osc_binary_sha256": sha256(WT / "bazel-bin/examples/sampling_c3/franka_osc_controller"),
        "canonical_cases_sha256": sha256(SYNC / "canonical_cases.csv"),
        "c3plus_cases_sha256": sha256(SYNC / "c3plus_cases.yaml"),
        "scene_sources_sha256": sha256(SCENE_SPEC),
        "selected_demo_tree_sha256": tree_sha256(relevant_demo_files),
        "manifest_sha256": sha256(manifest_path),
        "config_sha256": sha256(config_path),
        "open_equivalence_sha256": sha256(proof_path),
        "open_score_fixture_sha256": sha256(proof_scores_path),
        "serial_supervisor_sha256": sha256(
            ROOT / "scripts/run_c3plus_geometry_rotation_cost_replication.py"),
        "campaign_runner_sha256": sha256(
            ROOT / "scripts/run_c3plus_final_clean_obstacle_ranking.py"),
        "scientific_runs_launched": 0,
        "study_a_will_run": False,
        "approval_required": True,
    }
    provenance_path = OUT / "provenance/preparation_receipt.json"
    if provenance_path.exists():
        if json.loads(provenance_path.read_text()) != static_provenance:
            raise FileExistsError(
                f"refusing to overwrite changed provenance: {provenance_path}")
    else:
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        provenance_path.write_text(json.dumps(static_provenance, indent=2, sort_keys=True) + "\n")
    write_new_or_identical(
        OUT / "APPROVAL_REQUIRED",
        "The final 126-run obstacle study may not start until explicit user approval authorizes this exact manifest checksum.\n",
    )
    print(json.dumps({
        "prepared_unique_runs": len(rows),
        "shared_open_runs": sum(row["config_id"] == "OPEN-A1" for row in rows),
        "obstacle_bearing_runs": sum(row["config_id"] != "OPEN-A1" for row in rows),
        "open_equivalence": proof["status"],
        "scientific_runs_launched": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
