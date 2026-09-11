from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WT = ROOT / "external/oim_c++_anything/.claude/worktrees/oim-scene-sync-metrics"
CONTROLLER = WT / "systems/controllers/sampling_based_c3_controller.cc"
MANIFEST = ROOT / "results/c3plus_ranking_only_task_cost_study_a/manifest/study_a_manifest.csv"


def function_body(source: str, signature: str, next_signature: str) -> str:
    begin = source.index(signature)
    end = source.index(next_signature, begin + len(signature))
    return source[begin:end]


def test_ranking_only_parameters_are_downstream_of_calc_cost() -> None:
    source = CONTROLLER.read_text()
    update = function_body(
        source,
        "void SamplingC3Controller::UpdateCostMatrices(",
        "std::vector<SortedPair<GeometryId>>",
    )
    assert "alpha_p_rank_" not in update
    assert "alpha_theta_rank_" not in update

    calc_call = source.index("cost_trajectory_pair = CalcCost(")
    rank_adjustment = source.index("ApplyRankingTaskScales(")
    sample_total = source.index("all_sample_costs_[i] =", rank_adjustment)
    assert calc_call < rank_adjustment < sample_total


def test_study_a_manifest_is_exact_balanced_27_run_design() -> None:
    with MANIFEST.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 27
    assert {row["condition_id"] for row in rows} == {"A1", "A2", "A3"}
    assert {row["start_id"] for row in rows} == {"s1", "s3", "s5"}
    assert {row["goal_id"] for row in rows} == {"g2"}
    assert {row["rotation_label"] for row in rows} == {
        "rot000", "rotCCW090", "rotCW090"
    }
    assert {row["seed"] for row in rows} == {"0"}
    assert {row["scene"] for row in rows} == {"open_task"}
    assert all(row["alpha_p_rank"] == "1.0" for row in rows)
    theta = {"A1": "1.0", "A2": "0.5", "A3": "0.25"}
    assert all(row["alpha_theta_rank"] == theta[row["condition_id"]]
               for row in rows)
    assert all(row["inner_soft_obstacle_mode"] == "none" for row in rows)
    blocks = {row["block_index"] for row in rows}
    assert len(blocks) == 9
    assert all(sum(row["block_index"] == block for row in rows) == 3
               for block in blocks)
