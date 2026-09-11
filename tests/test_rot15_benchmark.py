import csv
import math

import pytest

from benchmarks.rot15 import benchmark as rot15
from benchmarks.rot15 import launcher


def test_rot15_matrix_is_exactly_90_relative_yaw_cases():
    rows = rot15.build_cases()
    assert len(rows) == 90
    assert {row["scene"] for row in rows} == set(rot15.SCENES)
    assert len({(row["scene"], row["case_id"]) for row in rows}) == 90
    for scene in rot15.SCENES:
        scene_rows = [row for row in rows if row["scene"] == scene]
        assert len(scene_rows) == 15
        assert {row["start_id"] for row in scene_rows} == set(rot15.START_IDS)
        for start_id in rot15.START_IDS:
            triplet = [row for row in scene_rows if row["start_id"] == start_id]
            assert {row["rotation_label"] for row in triplet} == set(rot15.ROTATIONS)
            assert {row["goal_position_id"] for row in triplet} == {"g02"}
            assert len({(row["goal_x"], row["goal_y"], row["goal_z"]) for row in triplet}) == 1
            assert len({row["seed"] for row in triplet}) == 1


@pytest.mark.parametrize(
    ("label", "expected"),
    (("rot000", 0.0), ("rotCCW090", math.pi / 2), ("rotCW090", -math.pi / 2)),
)
def test_yaw_is_relative_and_direction_is_unambiguous(label, expected):
    rows = rot15.build_cases()
    for row in rows:
        if row["rotation_label"] != label:
            continue
        measured = rot15.angle_difference(row["goal_yaw_rad"], row["start_yaw_rad"])
        assert measured == pytest.approx(expected, abs=1e-12)
    # Explicitly guard the 40-degree example against absolute-yaw regression.
    start = math.radians(40.0)
    assert math.degrees(rot15.wrap_to_pi(start + expected)) == pytest.approx(40.0 + math.degrees(expected))


def test_quaternion_order_round_trips_for_both_adapter_conventions():
    for row in rot15.build_cases():
        yaw = row["goal_yaw_rad"]
        assert rot15.angle_difference(rot15.yaw_from_wxyz(rot15.yaw_to_wxyz(yaw)), yaw) == pytest.approx(0.0, abs=1e-12)
        assert rot15.angle_difference(rot15.yaw_from_xyzw(rot15.yaw_to_xyzw(yaw)), yaw) == pytest.approx(0.0, abs=1e-12)


def test_geometry_preflight_has_no_penetrating_goal():
    spec = rot15.load_spec()
    preflight = rot15.geometry_preflight(rot15.build_cases(spec), spec)
    assert len(preflight) == 90
    assert all(row["raw_min_signed_distance"] >= 0.0 for row in preflight)
    icra = [row for row in preflight if row["scene"] == "icra_sign"]
    assert len(icra) == 15
    assert {row["footprint_type"] for row in icra} == {"c_glyph_union_3_boxes_controller_boundary"}


def test_cross_method_pose_and_relative_yaw_parity():
    spec = rot15.load_spec()
    canonical = rot15.build_cases(spec)
    c3plus, oim = rot15.build_adapters(canonical, spec)
    parity = rot15.cross_method_parity(canonical, c3plus, oim)
    assert len(parity) == 90
    assert {row["parity_status"] for row in parity} == {"PASS"}
    numeric = (
        "start_xyz_max_abs_diff", "start_yaw_abs_diff",
        "goal_xyz_max_abs_diff", "goal_yaw_abs_diff",
        "c3plus_relative_yaw_error", "oim_relative_yaw_error",
        "oim_xyzw_oracle_yaw_error",
    )
    assert all(float(row[field]) <= rot15.TOL for row in parity for field in numeric)
    assert all(row["obstacle_geometry_match"] for row in parity)
    assert all(row["manipulated_object_match"] for row in parity)
    assert all(row["case_identity_match"] for row in parity)


def test_cross_method_non_numeric_mismatch_fails_parity():
    spec = rot15.load_spec()
    canonical = rot15.build_cases(spec)
    c3plus, oim = rot15.build_adapters(canonical, spec)
    oim[0]["footprint_type"] = "wrong-footprint"
    with pytest.raises(AssertionError, match="cross-method parity failed"):
        rot15.cross_method_parity(canonical, c3plus, oim)


def test_generated_artifacts_validate_without_overwrite(tmp_path):
    result = rot15.generate(tmp_path)
    assert result["cases"] == 90
    assert rot15.validate_artifacts(tmp_path) == {
        "cases": 90, "geometry": "PASS", "parity": "PASS"
    }
    # Identical regeneration is allowed and does not mutate evidence.
    before = (tmp_path / "canonical_cases.csv").read_bytes()
    rot15.generate(tmp_path)
    assert (tmp_path / "canonical_cases.csv").read_bytes() == before
    with (tmp_path / "scene_case_summary.csv").open() as stream:
        summary = list(csv.DictReader(stream))
    assert [int(row["case_count"]) for row in summary] == [15] * 6


def test_generation_refuses_to_overwrite_changed_artifact(tmp_path):
    rot15.generate(tmp_path)
    path = tmp_path / "canonical_cases.csv"
    path.write_text("scientific evidence must not be overwritten\n")
    with pytest.raises(FileExistsError):
        rot15.generate(tmp_path)


def test_full_campaign_gate_precedes_c3_materialization(tmp_path):
    output = tmp_path / "artifacts"
    c3_root = tmp_path / "c3"
    rot15.generate(output)
    with pytest.raises(RuntimeError, match="full campaign safety gate"):
        launcher.main([
            "--method", "c3plus", "--all", "--materialize-c3plus",
            "--output", str(output), "--c3-root", str(c3_root),
        ])
    assert not c3_root.exists()
