from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/c3plus_final_clean_obstacle_ranking"
MANIFEST = OUT / "manifest/final_126_run_manifest.csv"
CONFIG = OUT / "config/final_clean_obstacle_study.json"
PROOF = OUT / "validation/open_task_equivalence.json"
RUNNER = ROOT / "scripts/run_c3plus_final_clean_obstacle_ranking.py"


def manifest_rows() -> list[dict[str, str]]:
    with MANIFEST.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_manifest_is_the_exact_shared_open_126_run_design() -> None:
    rows = manifest_rows()
    assert len(rows) == 126
    assert len({row["run_id"] for row in rows}) == 126
    assert len({row["lcm_port"] for row in rows}) == 126
    assert {row["start_id"] for row in rows} == {"s1", "s3", "s5"}
    assert {row["goal_id"] for row in rows} == {"g2"}
    assert {row["rotation_label"] for row in rows} == {
        "rot000", "rotCCW090", "rotCW090"
    }
    assert {row["seed"] for row in rows} == {"0", "1"}
    assert all(row["alpha_p_rank"] == "1.0" for row in rows)
    assert all(row["alpha_theta_rank"] == "1.0" for row in rows)
    assert all(row["inner_controller"] == "SOURCE_FROZEN" for row in rows)
    assert all(row["max_live_concurrency"] == "1" for row in rows)
    assert sum(row["config_id"] == "OPEN-A1" for row in rows) == 18
    for config in ("B1", "B2", "B3"):
        for scene in ("single_obstacle", "shelf_gap"):
            assert sum(row["config_id"] == config and row["scene"] == scene
                       for row in rows) == 18
    assert not any(row["source_obstacle_id"] in {
        "OC04", "MATCHED_OC04_RELU"} for row in rows)


def test_every_task_block_has_one_open_and_two_matched_B_triplets() -> None:
    rows = manifest_rows()
    blocks = {row["block_index"] for row in rows}
    assert len(blocks) == 18
    for block in blocks:
        selected = [row for row in rows if row["block_index"] == block]
        assert len(selected) == 7
        assert sum(row["config_id"] == "OPEN-A1" for row in selected) == 1
        for scene in ("single_obstacle", "shelf_gap"):
            assert {row["config_id"] for row in selected
                    if row["scene"] == scene} == {"B1", "B2", "B3"}


def test_exact_costs_and_open_equivalence_are_frozen() -> None:
    config = json.loads(CONFIG.read_text())
    conditions = {row["config_id"]: row for row in config["conditions"]}
    assert conditions["B1"] == {
        "config_id": "B1", "source_obstacle_id": "OC01",
        "family": "exponential_footprint", "range_name": "sigma",
        "range_m": 0.015, "weight": 764.3857383844456,
    }
    assert conditions["B2"] == {
        "config_id": "B2", "source_obstacle_id": "MATCHED_OC01_RELU",
        "family": "relu_footprint", "range_name": "epsilon",
        "range_m": 0.045, "weight": 764.3857383844456,
    }
    assert conditions["B3"] == {
        "config_id": "B3", "source_obstacle_id": "OC08",
        "family": "relu_footprint", "range_name": "epsilon",
        "range_m": 0.05, "weight": 1173.2815219621552,
    }
    proof = json.loads(PROOF.read_text())
    assert proof["status"] == "PASS"
    assert proof["all_J_obstacle_exactly_zero"]
    assert proof["max_abs_J_rank_across_B_conditions"] == 0.0
    assert proof["selected_candidate_identical"]
    for comparison in proof["pairwise_matrix_and_trajectory_deltas"].values():
        assert set(comparison.values()) == {0.0}


def test_runner_is_fail_closed_and_uses_rank_only_A1() -> None:
    source = RUNNER.read_text()
    assert 'SAMPLING_C3_ALPHA_P_RANK": "1"' in source
    assert 'SAMPLING_C3_ALPHA_THETA_RANK": "1"' in source
    assert 'SAMPLING_C3_INNER_OBS_MODE": "none"' in source
    assert "SAMPLING_C3_TRANSLATION_COST_SCALE" in source
    assert "SAMPLING_C3_ORIENTATION_COST_SCALE" in source
    assert "if not APPROVAL.exists()" in source
    assert "approval.get(\"manifest_sha256\")" in source
    assert "MAX_CONCURRENT_LIVE_SIMULATIONS = 1" in (
        ROOT / "scripts/run_c3plus_geometry_rotation_cost_replication.py").read_text()
