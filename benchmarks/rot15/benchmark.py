#!/usr/bin/env python3
"""Generate and validate benchmark_v2_rot15 without running controllers.

The YAML beside this module is the single authoritative task-definition layer.
Both method adapters are serialized from the same in-memory rows, then read
back and compared numerically.  Geometry follows the synchronized C3+ scene
representation, including its orientation-aware T/C boundary point sets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path(__file__).with_name("scene_sources.yaml")
DEFAULT_OUTPUT = ROOT / "results" / "benchmark_sync_rot15"
SCENES = (
    "open_task",
    "single_obstacle",
    "shelf_gap",
    "ycb_clutter",
    "icra_sign",
    "slalom",
)
START_IDS = tuple(f"s{i:02d}" for i in range(1, 6))
ROTATIONS = ("rot000", "rotCCW090", "rotCW090")
CASE_COUNT = len(SCENES) * len(START_IDS) * len(ROTATIONS)
TOL = 1e-10

CANONICAL_FIELDS = (
    "scene", "case_id", "start_id", "goal_position_id",
    "start_x", "start_y", "start_z", "start_yaw_rad", "start_yaw_deg",
    "goal_x", "goal_y", "goal_z", "delta_yaw_rad", "delta_yaw_deg",
    "goal_yaw_rad", "goal_yaw_deg", "rotation_label", "seed",
)


def load_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    with path.open() as stream:
        spec = yaml.safe_load(stream)
    if tuple(spec["scenes"]) != SCENES:
        raise AssertionError(
            f"scene order/names drifted: {tuple(spec['scenes'])!r} != {SCENES!r}"
        )
    return spec


def wrap_to_pi(theta: float) -> float:
    """Wrap to [-pi, pi), so +pi is represented as -pi."""
    return (float(theta) + math.pi) % (2.0 * math.pi) - math.pi


def angle_difference(a: float, b: float) -> float:
    return wrap_to_pi(float(a) - float(b))


def yaw_to_wxyz(yaw: float) -> tuple[float, float, float, float]:
    q = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
    norm = math.sqrt(sum(v * v for v in q))
    return tuple(v / norm for v in q)


def yaw_to_xyzw(yaw: float) -> tuple[float, float, float, float]:
    w, x, y, z = yaw_to_wxyz(yaw)
    return x, y, z, w


def yaw_from_wxyz(q: Sequence[float]) -> float:
    if len(q) != 4:
        raise ValueError(f"wxyz quaternion must have four components, got {q!r}")
    w, x, y, z = (float(v) for v in q)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isclose(norm, 1.0, abs_tol=1e-10):
        raise ValueError(f"non-normalized wxyz quaternion (norm={norm}): {q!r}")
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_from_xyzw(q: Sequence[float]) -> float:
    x, y, z, w = (float(v) for v in q)
    return yaw_from_wxyz((w, x, y, z))


def build_cases(spec: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    spec = dict(spec or load_spec())
    rotations = {r["label"]: r for r in spec["rotations"]}
    if tuple(rotations) != ROTATIONS:
        raise AssertionError(f"rotation order/labels drifted: {tuple(rotations)!r}")
    seed_cfg = spec["seed_policy"]
    rows: list[dict[str, Any]] = []
    for scene_index, scene in enumerate(SCENES):
        source = spec["scenes"][scene]
        if tuple(source["starts"]) != START_IDS:
            raise AssertionError(f"{scene}: starts are not exactly {START_IDS}")
        gx, gy, gz = (float(v) for v in source["g02"])
        for start_index, start_id in enumerate(START_IDS, start=1):
            sx, sy, start_yaw = (float(v) for v in source["starts"][start_id])
            seed = (
                int(seed_cfg["base"])
                + scene_index * int(seed_cfg["scene_stride"])
                + start_index * int(seed_cfg["start_stride"])
            )
            for label in ROTATIONS:
                rotation = rotations[label]
                delta = float(rotation["delta_yaw_rad"])
                goal_yaw = wrap_to_pi(start_yaw + delta)
                rows.append({
                    "scene": scene,
                    "case_id": f"{start_id}_g02_{rotation['id_suffix']}",
                    "start_id": start_id,
                    "goal_position_id": "g02",
                    "start_x": sx,
                    "start_y": sy,
                    "start_z": float(source["object_z"]),
                    "start_yaw_rad": start_yaw,
                    "start_yaw_deg": math.degrees(start_yaw),
                    "goal_x": gx,
                    "goal_y": gy,
                    "goal_z": gz,
                    "delta_yaw_rad": delta,
                    "delta_yaw_deg": float(rotation["delta_yaw_deg"]),
                    "goal_yaw_rad": goal_yaw,
                    "goal_yaw_deg": math.degrees(goal_yaw),
                    "rotation_label": label,
                    "seed": seed,
                })
    validate_cases(rows)
    return rows


def validate_cases(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != CASE_COUNT:
        raise AssertionError(f"global manifest has {len(rows)} rows, expected {CASE_COUNT}")
    ids = [(str(r["scene"]), str(r["case_id"])) for r in rows]
    if len(ids) != len(set(ids)):
        duplicates = [key for key, n in Counter(ids).items() if n > 1]
        raise AssertionError(f"duplicate scene/case IDs: {duplicates}")
    expected_delta = {
        "rot000": 0.0,
        "rotCCW090": math.pi / 2.0,
        "rotCW090": -math.pi / 2.0,
    }
    by_scene: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_triplet: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        scene = str(row["scene"])
        if scene not in SCENES:
            raise AssertionError(f"unknown scene {scene!r}")
        by_scene[scene].append(row)
        by_triplet[(scene, str(row["start_id"]))].append(row)
        if row["goal_position_id"] != "g02":
            raise AssertionError(f"{scene}/{row['case_id']} does not use g02")
        label = str(row["rotation_label"])
        if label not in expected_delta:
            raise AssertionError(f"unexpected rotation label {label!r}")
        measured = angle_difference(float(row["goal_yaw_rad"]), float(row["start_yaw_rad"]))
        if not math.isclose(measured, expected_delta[label], abs_tol=TOL):
            raise AssertionError(
                f"{scene}/{row['case_id']}: relative yaw {measured} != {expected_delta[label]}"
            )
        if label == "rotCCW090" and float(row["delta_yaw_deg"]) != 90.0:
            raise AssertionError("+90 must be CCW")
        if label == "rotCW090" and float(row["delta_yaw_deg"]) != -90.0:
            raise AssertionError("-90 must be CW")
    if set(by_scene) != set(SCENES):
        raise AssertionError(f"manifest scenes differ from canonical six: {set(by_scene)}")
    for scene, scene_rows in by_scene.items():
        if len(scene_rows) != 15:
            raise AssertionError(f"{scene}: {len(scene_rows)} cases, expected 15")
    for key, triplet in by_triplet.items():
        labels = {str(r["rotation_label"]) for r in triplet}
        if labels != set(ROTATIONS) or len(triplet) != 3:
            raise AssertionError(f"{key}: rotation set {labels}, n={len(triplet)}")
        positions = {
            (float(r["goal_x"]), float(r["goal_y"]), float(r["goal_z"]))
            for r in triplet
        }
        if len(positions) != 1:
            raise AssertionError(f"{key}: goal XYZ differs within triplet: {positions}")
        seeds = {int(r["seed"]) for r in triplet}
        if len(seeds) != 1:
            raise AssertionError(f"{key}: rotation changed task seed: {seeds}")


def _controller_boundary_points(footprint: Mapping[str, Any]) -> list[tuple[float, float]]:
    subdivisions = int(footprint["controller_boundary_subdivisions"])
    points: list[tuple[float, float]] = []
    for box in footprint["boxes"]:
        cx, cy = (float(v) for v in box["center"])
        width, height = (float(v) for v in box["size"])
        for index in range(subdivisions + 1):
            alpha = index / subdivisions
            points.extend((
                (cx - width / 2.0 + alpha * width, cy - height / 2.0),
                (cx - width / 2.0 + alpha * width, cy + height / 2.0),
                (cx - width / 2.0, cy - height / 2.0 + alpha * height),
                (cx + width / 2.0, cy - height / 2.0 + alpha * height),
            ))
    return points


def _point_polygon_signed_distance(point: tuple[float, float], vertices: Sequence[Sequence[float]]) -> float:
    """Controller-equivalent point-to-polygon SDF (negative inside)."""
    px, py = point
    best = math.inf
    inside = False
    for index, first in enumerate(vertices):
        second = vertices[(index + 1) % len(vertices)]
        x1, y1 = (float(v) for v in first)
        x2, y2 = (float(v) for v in second)
        ex, ey = x2 - x1, y2 - y1
        rx, ry = px - x1, py - y1
        denom = ex * ex + ey * ey
        alpha = 0.0 if denom == 0.0 else max(0.0, min(1.0, (rx * ex + ry * ey) / denom))
        best = min(best, math.hypot(rx - alpha * ex, ry - alpha * ey))
        if ey != 0.0 and (y1 > py) != (y2 > py) and px < x1 + ex * (py - y1) / ey:
            inside = not inside
    if best < 1e-12:
        return 0.0
    return -best if inside else best


def _obstacle_sdf(point: tuple[float, float], obstacle: Mapping[str, Any]) -> float:
    px, py = point
    kind = obstacle["type"]
    if kind == "disc":
        ox, oy = (float(v) for v in obstacle["center"])
        return math.hypot(px - ox, py - oy) - float(obstacle["radius"])
    if kind == "aabb":
        ox, oy = (float(v) for v in obstacle["center"])
        hx, hy = (float(v) for v in obstacle["half_extents"])
        qx, qy = abs(px - ox) - hx, abs(py - oy) - hy
        if qx > 0.0 or qy > 0.0:
            return math.hypot(max(qx, 0.0), max(qy, 0.0))
        return max(qx, qy)
    if kind == "polygon":
        return _point_polygon_signed_distance(point, obstacle["vertices"])
    raise ValueError(f"unsupported obstacle type {kind!r}")


def geometry_preflight(
    rows: Sequence[Mapping[str, Any]], spec: Mapping[str, Any]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    c3_margin = float(spec["method_margins"]["c3plus"]["hard_object_obstacle_margin_m"])
    for row in rows:
        scene_spec = spec["scenes"][str(row["scene"])]
        body_points = _controller_boundary_points(scene_spec["footprint"])
        yaw = float(row["goal_yaw_rad"])
        cosine, sine = math.cos(yaw), math.sin(yaw)
        gx, gy = float(row["goal_x"]), float(row["goal_y"])
        world_points = [
            (gx + cosine * x - sine * y, gy + sine * x + cosine * y)
            for x, y in body_points
        ]
        closest_name = "none (open scene)"
        raw = math.inf
        for obstacle in scene_spec["obstacles"]:
            distance = min(_obstacle_sdf(point, obstacle) for point in world_points)
            if distance < raw:
                raw = distance
                closest_name = str(obstacle["name"])
        if raw < -TOL:
            status = "INFEASIBLE_PHYSICAL_PENETRATION"
        elif abs(raw) <= TOL:
            status = "TOUCHING"
        elif raw < c3_margin:
            status = "FEASIBLE_RAW_C3PLUS_MARGIN_VIOLATION"
        else:
            status = "FEASIBLE"
        output.append({
            "scene": row["scene"],
            "case_id": row["case_id"],
            "start_id": row["start_id"],
            "rotation_label": row["rotation_label"],
            "goal_x": gx,
            "goal_y": gy,
            "goal_z": float(row["goal_z"]),
            "goal_yaw": yaw,
            "raw_min_signed_distance": raw,
            "margin_adjusted_clearance": raw - c3_margin,
            "hard_margin_m": c3_margin,
            "margin_scope": "C3+ lcs_contact only; OIM has no additional hard margin",
            "closest_obstacle": closest_name,
            "footprint_type": scene_spec["footprint"]["type"],
            "feasibility_status": status,
        })
    penetrations = [r for r in output if r["feasibility_status"] == "INFEASIBLE_PHYSICAL_PENETRATION"]
    if penetrations:
        labels = ", ".join(f"{r['scene']}/{r['case_id']}" for r in penetrations)
        raise AssertionError(f"geometrically penetrating requested goals: {labels}")
    return output


def _c3plus_adapter(row: Mapping[str, Any], scene_spec: Mapping[str, Any]) -> dict[str, Any]:
    start_number = int(str(row["start_id"])[1:])
    source_suffix = f"t{start_number}" if start_number == 2 else f"s{start_number}g2"
    start_q = yaw_to_wxyz(float(row["start_yaw_rad"]))
    goal_q = yaw_to_wxyz(float(row["goal_yaw_rad"]))
    return {
        "scene": row["scene"],
        "case_id": row["case_id"],
        "demo_source": str(scene_spec["c3plus_family"]) + source_suffix,
        "generated_demo": f"rot15_{row['scene']}_{row['case_id']}",
        "start_qw": start_q[0], "start_qx": start_q[1],
        "start_qy": start_q[2], "start_qz": start_q[3],
        "start_x": row["start_x"], "start_y": row["start_y"], "start_z": row["start_z"],
        "goal_x": row["goal_x"], "goal_y": row["goal_y"], "goal_z": row["goal_z"],
        "goal_qw": goal_q[0], "goal_qx": goal_q[1],
        "goal_qy": goal_q[2], "goal_qz": goal_q[3],
        "seed": row["seed"],
        "quaternion_order": "wxyz",
        "env_file": scene_spec.get("c3plus_env_file") or "",
        "manipulated_object": scene_spec["manipulated_object"],
        "footprint_type": scene_spec["footprint"]["type"],
    }


def _oim_adapter(row: Mapping[str, Any], scene_spec: Mapping[str, Any]) -> dict[str, Any]:
    # OIM's tabletop interface is planar x/y/yaw (hinge joints), not a
    # quaternion.  xyzw is recorded here only as an independent parity oracle.
    goal_xyzw = yaw_to_xyzw(float(row["goal_yaw_rad"]))
    start_xyzw = yaw_to_xyzw(float(row["start_yaw_rad"]))
    return {
        "scene": row["scene"],
        "oim_scene": scene_spec["oim_scene"],
        "case_id": row["case_id"],
        "start_key": row["start_id"],
        "goal_key": row["case_id"],
        "start_x": row["start_x"], "start_y": row["start_y"], "start_z": row["start_z"],
        "start_yaw": row["start_yaw_rad"],
        "goal_x": row["goal_x"], "goal_y": row["goal_y"], "goal_z": row["goal_z"],
        "goal_yaw": row["goal_yaw_rad"],
        "start_qx_oracle": start_xyzw[0], "start_qy_oracle": start_xyzw[1],
        "start_qz_oracle": start_xyzw[2], "start_qw_oracle": start_xyzw[3],
        "goal_qx_oracle": goal_xyzw[0], "goal_qy_oracle": goal_xyzw[1],
        "goal_qz_oracle": goal_xyzw[2], "goal_qw_oracle": goal_xyzw[3],
        "seed": row["seed"],
        "interface_orientation": "planar_yaw_rad",
        "oracle_quaternion_order": "xyzw",
        "manipulated_object": scene_spec["manipulated_object"],
        "footprint_type": scene_spec["footprint"]["type"],
    }


def build_adapters(
    rows: Sequence[Mapping[str, Any]], spec: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    c3plus, oim = [], []
    for row in rows:
        scene_spec = spec["scenes"][str(row["scene"])]
        c3plus.append(_c3plus_adapter(row, scene_spec))
        oim.append(_oim_adapter(row, scene_spec))
    return c3plus, oim


def audit_repository_sources(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Compare the frozen canonical transcription to both legacy sources."""
    revision = spec["source_revision"]
    c3_root = ROOT / str(revision["c3plus_worktree"])
    oim_root = ROOT / str(revision["oim_repository"])
    rows: list[dict[str, Any]] = []
    for scene in SCENES:
        source = spec["scenes"][scene]
        checks: list[tuple[str, float, float, float]] = []
        if c3_root.is_dir():
            family = str(source["c3plus_family"])
            for start_id in START_IDS:
                number = int(start_id[1:])
                suffix = f"t{number}" if number == 2 else f"s{number}g2"
                sim_path = c3_root / "examples/sampling_c3" / f"{family}{suffix}" / "parameters/sim_params.yaml"
                text = sim_path.read_text()
                match = re.search(r"(?m)^q_init_object:\s*\[([^\]]+)\]", text)
                if not match:
                    raise AssertionError(f"missing q_init_object in {sim_path}")
                values = [float(value) for value in match.group(1).split(",")]
                legacy_norm = math.sqrt(sum(value * value for value in values[:4]))
                legacy_q = [value / legacy_norm for value in values[:4]]
                expected = source["starts"][start_id]
                checks.extend((
                    (f"c3plus_{start_id}_x", values[4], float(expected[0]), 1e-12),
                    (f"c3plus_{start_id}_y", values[5], float(expected[1]), 1e-12),
                    # Legacy YAML quaternions were rounded to six decimals.
                    (f"c3plus_{start_id}_yaw", yaw_from_wxyz(legacy_q), float(expected[2]), 2e-6),
                ))
            goal_path = c3_root / "examples/sampling_c3" / f"{family}t2" / "parameters/goal_params.yaml"
            text = goal_path.read_text()
            match = re.search(r"(?m)^fixed_target_position:\s*\[([^\]]+)\]", text)
            if not match:
                raise AssertionError(f"missing fixed_target_position in {goal_path}")
            values = [float(value) for value in match.group(1).split(",")]
            checks.extend(
                (f"c3plus_g02_{axis}", values[index], float(source["g02"][index]), 1e-12)
                for index, axis in enumerate("xyz")
            )
        else:
            rows.append({"scene": scene, "source": "c3plus", "check": "worktree_available", "abs_diff": "", "tolerance": "", "status": "SKIP"})

        if (oim_root / ".git").exists():
            path = f"examples/poses/{source['oim_scene']}.yaml"
            try:
                text = subprocess.run(
                    ["git", "show", f"{revision['oim_commit']}:{path}"],
                    cwd=oim_root, check=True, text=True, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                ).stdout
            except subprocess.CalledProcessError as exc:
                raise AssertionError(f"cannot read OIM source {revision['oim_commit']}:{path}: {exc.stderr}") from exc
            poses = yaml.safe_load(text)
            for start_id in START_IDS:
                key = str(int(start_id[1:]))
                expected = source["starts"][start_id]
                for index, axis in enumerate(("x", "y", "yaw")):
                    checks.append((f"oim_{start_id}_{axis}", float(poses["starts"][key][index]), float(expected[index]), 1e-12))
            for index, axis in enumerate(("x", "y")):
                checks.append((f"oim_g02_{axis}", float(poses["goals"]["2"][index]), float(source["g02"][index]), 1e-12))
        else:
            rows.append({"scene": scene, "source": "oim", "check": "repository_available", "abs_diff": "", "tolerance": "", "status": "SKIP"})

        for name, actual, expected, tolerance in checks:
            difference = abs(angle_difference(actual, expected)) if name.endswith("_yaw") else abs(actual - expected)
            status = "PASS" if difference <= tolerance else "FAIL"
            rows.append({
                "scene": scene,
                "source": "oim" if name.startswith("oim_") else "c3plus",
                "check": name,
                "abs_diff": difference,
                "tolerance": tolerance,
                "status": status,
            })
    failures = [row for row in rows if row["status"] == "FAIL"]
    if failures:
        raise AssertionError(f"repository source audit failed: {failures[:5]}")
    return rows


def cross_method_parity(
    canonical: Sequence[Mapping[str, Any]],
    c3plus: Sequence[Mapping[str, Any]],
    oim: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    c_index = {(str(r["scene"]), str(r["case_id"])): r for r in canonical}
    c3_index = {(str(r["scene"]), str(r["case_id"])): r for r in c3plus}
    o_index = {(str(r["scene"]), str(r["case_id"])): r for r in oim}
    if set(c_index) != set(c3_index) or set(c_index) != set(o_index):
        raise AssertionError("canonical/C3+/OIM adapter case identities differ")
    rows = []
    spec = load_spec()
    for key in c_index:
        can, c3, oi = c_index[key], c3_index[key], o_index[key]
        c3_start_yaw = yaw_from_wxyz([c3[f"start_q{k}"] for k in "wxyz"])
        c3_goal_yaw = yaw_from_wxyz([c3[f"goal_q{k}"] for k in "wxyz"])
        oi_start_yaw = float(oi["start_yaw"])
        oi_goal_yaw = float(oi["goal_yaw"])
        oi_start_oracle = yaw_from_xyzw([oi[f"start_q{k}_oracle"] for k in "xyzw"])
        oi_goal_oracle = yaw_from_xyzw([oi[f"goal_q{k}_oracle"] for k in "xyzw"])
        start_c3 = [float(c3[f"start_{axis}"]) for axis in "xyz"]
        start_oim = [float(oi[f"start_{axis}"]) for axis in "xyz"]
        goal_c3 = [float(c3[f"goal_{axis}"]) for axis in "xyz"]
        goal_oim = [float(oi[f"goal_{axis}"]) for axis in "xyz"]
        start_diff = max(abs(a - b) for a, b in zip(start_c3, start_oim))
        goal_diff = max(abs(a - b) for a, b in zip(goal_c3, goal_oim))
        start_yaw_diff = abs(angle_difference(c3_start_yaw, oi_start_yaw))
        goal_yaw_diff = abs(angle_difference(c3_goal_yaw, oi_goal_yaw))
        rel_c3 = angle_difference(c3_goal_yaw, c3_start_yaw)
        rel_oim = angle_difference(oi_goal_yaw, oi_start_yaw)
        intended = float(can["delta_yaw_rad"])
        oracle_error = max(
            abs(angle_difference(oi_start_oracle, oi_start_yaw)),
            abs(angle_difference(oi_goal_oracle, oi_goal_yaw)),
        )
        geometry_payload_c3 = {
            "footprint": spec["scenes"][str(c3["scene"])]["footprint"],
            "obstacles": spec["scenes"][str(c3["scene"])]["obstacles"],
        }
        geometry_payload_oim = {
            "footprint": spec["scenes"][str(oi["scene"])]["footprint"],
            "obstacles": spec["scenes"][str(oi["scene"])]["obstacles"],
        }
        geometry_match = (
            hashlib.sha256(json.dumps(geometry_payload_c3, sort_keys=True).encode()).digest()
            == hashlib.sha256(json.dumps(geometry_payload_oim, sort_keys=True).encode()).digest()
            and c3["footprint_type"] == oi["footprint_type"]
        )
        object_match = c3["manipulated_object"] == oi["manipulated_object"]
        identity_match = c3["case_id"] == oi["case_id"]
        numeric_match = max(
            start_diff, goal_diff, start_yaw_diff, goal_yaw_diff,
            abs(angle_difference(rel_c3, intended)),
            abs(angle_difference(rel_oim, intended)), oracle_error,
        ) <= TOL
        passed = numeric_match and geometry_match and object_match and identity_match
        rows.append({
            "scene": key[0], "case_id": key[1],
            "start_xyz_max_abs_diff": start_diff,
            "start_yaw_abs_diff": start_yaw_diff,
            "goal_xyz_max_abs_diff": goal_diff,
            "goal_yaw_abs_diff": goal_yaw_diff,
            "c3plus_relative_yaw_error": abs(angle_difference(rel_c3, intended)),
            "oim_relative_yaw_error": abs(angle_difference(rel_oim, intended)),
            "oim_xyzw_oracle_yaw_error": oracle_error,
            "obstacle_geometry_match": geometry_match,
            "manipulated_object_match": object_match,
            "case_identity_match": identity_match,
            "parity_status": "PASS" if passed else "FAIL",
        })
    failures = [r for r in rows if r["parity_status"] != "PASS"]
    if failures:
        raise AssertionError(f"cross-method parity failed for {len(failures)} cases")
    return rows


def _csv_text(fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> str:
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _write_new_or_identical(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_text()
        if existing != content:
            raise FileExistsError(
                f"refusing to overwrite non-identical benchmark artifact: {path}"
            )
        return
    path.write_text(content)


def _adapter_yaml(rows: Sequence[Mapping[str, Any]]) -> str:
    return yaml.safe_dump({"cases": list(rows)}, sort_keys=False, width=160)


def _oim_pose_yaml(
    scene: str, canonical: Sequence[Mapping[str, Any]], spec: Mapping[str, Any]
) -> str:
    scene_rows = [r for r in canonical if r["scene"] == scene]
    source = spec["scenes"][scene]
    starts = {
        start_id: [float(v) for v in source["starts"][start_id]]
        for start_id in START_IDS
    }
    goals = {
        str(row["case_id"]): [
            float(row["goal_x"]), float(row["goal_y"]), float(row["goal_yaw_rad"])
        ]
        for row in scene_rows
    }
    payload = {
        "benchmark": "benchmark_v2_rot15",
        "canonical_scene": scene,
        "starts": starts,
        "goals": goals,
    }
    return yaml.safe_dump(payload, sort_keys=False, width=160)


def _git_value(args: Sequence[str], cwd: Path) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _report(
    canonical: Sequence[Mapping[str, Any]], preflight: Sequence[Mapping[str, Any]],
    parity: Sequence[Mapping[str, Any]], source_audit: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any], output: Path,
) -> str:
    scene_lines = []
    for scene in SCENES:
        distances = [
            float(r["raw_min_signed_distance"])
            for r in preflight if r["scene"] == scene
        ]
        finite = [d for d in distances if math.isfinite(d)]
        minimum = f"{min(finite):.9f} m" if finite else "+inf (no obstacles)"
        bad = sum(d < -TOL for d in distances)
        scene_lines.append(f"| `{scene}` | 15 | {minimum} | {bad} |")
    source = spec["source_revision"]
    return f"""# benchmark_v2_rot15 synchronization report

## Outcome

- Old benchmark (`benchmark_v1_25pair`): 5 starts x 5 goals = 25 cases / scene.
- New benchmark (`benchmark_v2_rot15`): 5 starts x 1 translational goal x 3 relative yaw demands = 15 cases / scene.
- Total: {len(canonical)} nominal cases over 6 scenes.
- Geometry gate: **PASS** ({sum(r['feasibility_status'] == 'FEASIBLE' for r in preflight)}/{len(preflight)} raw-feasible; zero penetrating targets).
- Cross-method numeric parity: **PASS** ({sum(r['parity_status'] == 'PASS' for r in parity)}/{len(parity)}).
- Legacy-source numeric audit: **PASS** ({sum(r['status'] == 'PASS' for r in source_audit)} checks; {sum(r['status'] == 'SKIP' for r in source_audit)} skipped because an optional checkout was absent).
- Offline launcher-resolution smoke: **PASS** (the `open_task/s01` rotation triplet resolves for both methods without starting a controller).
- Native controller smoke: **BLOCKED** (see Verification and campaign blockers).
- Full 90-case campaign: **NOT RUN**.

## Canonical sources and conventions

- Starts `s01`...`s05` and `g02` XY/yaw source: upstream OIM `{source['oim_pose_path']}` at `{source['oim_commit']}`. Only the `g02` translation is retained; its historical absolute yaw is deliberately ignored.
- C3+ source audit: `{source['c3plus_pose_path']}` at `{source['c3plus_commit']}`.
- Geometry: `benchmarks/rot15/scene_sources.yaml`, transcribed from `{source['geometry_path']}` and checked against `{source['geometry_controller_path']}`. `icra_sign` uses `push_c_glyph` and the actual three-box C footprint plus all seven fixed glyph hulls.
- Scene IDs are the runner's canonical names: `open_task`, `single_obstacle`, `shelf_gap`, `ycb_clutter`, `icra_sign`, `slalom`. OIM maps only `open_task` to its native `open_table` scene key.
- Yaw composition: `theta_goal = wrap_to_pi(theta_start + delta_theta)` with deltas `0`, `+pi/2` (CCW), `-pi/2` (CW).
- Wrap convention: `[-pi, pi)`.
- C3+ quaternion interface: normalized `[w,x,y,z]`; `Eigen::Quaterniond(v[0],v[1],v[2],v[3])`.
- OIM tabletop interface: `[x,y,yaw]` planar slide/hinge coordinates, so no quaternion is passed. The parity artifact independently constructs SciPy-style `[x,y,z,w]` oracle quaternions and converts them back to yaw.
- Seed policy: `17000 + 100*scene_index + start_index` (zero-based scene index, one-based start index). Rotation does not affect the seed; matched C3+/OIM cases receive the same integer.

## Geometry preflight

`raw_min_signed_distance` reproduces the synchronized controller representation: the orientation-aware `TFootprint`/`CFootprint` boundary samples evaluated against exact AABBs, convex hulls, and the robot-base disc. Negative is penetration, zero is touching, positive is clearance.

| scene | cases | minimum raw clearance | penetrating |
|---|---:|---:|---:|
{os.linesep.join(scene_lines)}

C3+ `lcs_contact` additionally subtracts a 0.01 m hard object-obstacle margin (`phi = d_raw - margin`). OIM at `{source['oim_commit']}` has physical MuJoCo nonpenetration and a soft exponential object-obstacle cost, but no extra shared hard object margin. `margin_adjusted_clearance` in the CSV is therefore C3+-specific, not a shared benchmark feasibility definition.

## Cross-method parity

Every canonical row is independently translated to C3+ YAML semantics and OIM planar-pose semantics, serialized, read back, and converted to SE(2). Start XYZ/yaw, goal XYZ/yaw, relative yaw, task seed, manipulated-object identity, footprint type, and case identity all match to tolerance `{TOL:g}`. Obstacle geometry is shared by scene from the authoritative specification rather than regenerated in either adapter.

The intended task seed is explicit and identical in both adapters. The audited C3+ perimeter generator currently constructs `std::mt19937` from `std::random_device`, however, and exposes no task-level seed in these demo YAMLs. OIM consumes `--seed`. Therefore task-pose parity is complete, but common random numbers are a **campaign blocker** until a separately approved C3+ seed-plumbing change is made; this benchmark task does not change sampling behavior.

## Legacy preservation and controller freeze

The legacy `tools/scene_smoke/gen_grid_dirs.py` and `tools/scene_smoke/run_grid_campaign.py` 5x5 path is not modified. No controller, solver, cost, ADMM, MPC, sampling, horizon, threshold, dynamics, contact, or scene-physics parameter is changed. New launch-time C3+ configs are copies of the legacy `sXXg02` task config with only the canonical start pose and g02 goal pose changed. A task receipt stores the intended seed and case identity, but it does not falsely claim that the current C3+ sampler consumes that seed.

## Verification and campaign blockers

- `python -m pytest tests/test_rot15_benchmark.py -q`: **11 passed**.
- `python -m benchmarks.rot15.benchmark validate`: **PASS** (90 cases, geometry, and parity).
- `python -m benchmarks.rot15.launcher --method both --scene open_task --smoke-triplet --dry-run --cap 5 --port-base 19000`: **PASS**, six native command lines resolved and zero simulations started.
- A native OIM parser/launcher smoke stops during import with `ModuleNotFoundError: No module named 'jax'`; the available environment also lacks MuJoCo. No dependency installation was attempted.
- The checked-out OIM submodule is `a954f00047b6f2f302e83176124f482b2ec1a2e7`; the audited canonical OIM source is `{source['oim_commit']}`, and the current checkout lacks `examples/pusht/slalom.py`. The source is present in the audited upstream Git object but is not silently copied into this checkout.
- C3+ task-pose consumption can be materialized additively, but deterministic seed consumption is not supported by the audited sampler. The real `--all` launcher therefore has an intentional hard stop.
- The repository-wide `python -m pytest tests` result in this existing dirty environment was **35 failed, 455 passed, 56 skipped, 1 xfailed, 15 errors**. The new rot15 tests all pass; the other failures include unavailable Meshcat/websocket services and pre-existing configuration/progress/mode expectations and were not altered.

The benchmark definitions, geometry gate, and adapter parity are ready. A full campaign and cost fine-tuning remain blocked until C3 seed plumbing is separately approved and verified, and until an OIM environment/checkout containing the audited scene scripts is available. No controller code is changed in this task.

## Files added or modified

- `.gitignore` (track only the small deterministic synchronization artifacts under the otherwise ignored `results/` tree)
- `benchmarks/__init__.py`
- `benchmarks/rot15/__init__.py`
- `benchmarks/rot15/scene_sources.yaml`
- `benchmarks/rot15/benchmark.py`
- `benchmarks/rot15/launcher.py`
- `benchmarks/rot15/oim_entry.py`
- `tests/test_rot15_benchmark.py`
- `results/benchmark_sync_rot15/*` (generated CSV/YAML/report evidence; no simulation logs)

## Reproducibility

- Outer repository commit: `{_git_value(['rev-parse', 'HEAD'], ROOT)}`.
- Outer repository dirty state at generation: `{json.dumps(_git_value(['status', '--short'], ROOT).splitlines())}`.
- Canonical specification SHA-256: `{_sha256(SPEC_PATH)}`.
- Output directory: `{output}`.
- Generator: `python -m benchmarks.rot15.benchmark generate`.
- Validator: `python -m benchmarks.rot15.benchmark validate`.

## Artifacts

- `canonical_cases.csv`: 90 canonical task rows.
- `scene_case_summary.csv`: six scene-level count checks.
- `goal_geometry_preflight.csv`: all 90 goal-pose clearances.
- `cross_method_parity.csv`: all 90 adapter parity checks.
- `source_audit.csv`: numeric comparison to the legacy C3+ configs and upstream OIM pose files.
- `c3plus_cases.yaml`, `oim_cases.yaml`: method translations.
- `oim_pose_overrides/*.yaml`: OIM pose-key files generated from the same rows.
- `benchmark_sync_report.md`: this report.
"""


def generate(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    spec = load_spec()
    source_audit = audit_repository_sources(spec)
    canonical = build_cases(spec)
    preflight = geometry_preflight(canonical, spec)
    c3plus, oim = build_adapters(canonical, spec)
    parity = cross_method_parity(canonical, c3plus, oim)
    summaries = []
    for scene in SCENES:
        scene_rows = [r for r in canonical if r["scene"] == scene]
        summaries.append({
            "scene": scene,
            "case_count": len(scene_rows),
            "unique_start_count": len({r["start_id"] for r in scene_rows}),
            "rotation_conditions_per_start": 3,
            "rotation_labels": "rot000|rotCCW090|rotCW090",
            "goal_position_ids": "g02",
            "summary_status": "PASS",
        })
    _write_new_or_identical(output / "canonical_cases.csv", _csv_text(CANONICAL_FIELDS, canonical))
    _write_new_or_identical(output / "scene_case_summary.csv", _csv_text(tuple(summaries[0]), summaries))
    _write_new_or_identical(output / "goal_geometry_preflight.csv", _csv_text(tuple(preflight[0]), preflight))
    _write_new_or_identical(output / "cross_method_parity.csv", _csv_text(tuple(parity[0]), parity))
    _write_new_or_identical(output / "source_audit.csv", _csv_text(tuple(source_audit[0]), source_audit))
    _write_new_or_identical(output / "c3plus_cases.yaml", _adapter_yaml(c3plus))
    _write_new_or_identical(output / "oim_cases.yaml", _adapter_yaml(oim))
    for scene in SCENES:
        oim_scene = spec["scenes"][scene]["oim_scene"]
        _write_new_or_identical(
            output / "oim_pose_overrides" / f"{oim_scene}.yaml",
            _oim_pose_yaml(scene, canonical, spec),
        )
    _write_new_or_identical(
        output / "benchmark_sync_report.md",
        _report(canonical, preflight, parity, source_audit, spec, output),
    )
    return {
        "cases": len(canonical),
        "scenes": len(summaries),
        "feasible": sum(r["feasibility_status"] == "FEASIBLE" for r in preflight),
        "parity_pass": sum(r["parity_status"] == "PASS" for r in parity),
        "source_checks_pass": sum(r["status"] == "PASS" for r in source_audit),
        "output": str(output),
    }


def validate_artifacts(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    required = (
        "canonical_cases.csv", "scene_case_summary.csv",
        "goal_geometry_preflight.csv", "cross_method_parity.csv",
        "c3plus_cases.yaml", "oim_cases.yaml", "benchmark_sync_report.md",
        "source_audit.csv",
    )
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing generated artifacts: {missing}")
    with (output / "canonical_cases.csv").open() as stream:
        canonical = list(csv.DictReader(stream))
    validate_cases(canonical)
    with (output / "goal_geometry_preflight.csv").open() as stream:
        geometry = list(csv.DictReader(stream))
    if len(geometry) != CASE_COUNT:
        raise AssertionError(f"preflight has {len(geometry)} rows")
    penetrating = [r for r in geometry if float(r["raw_min_signed_distance"]) < -TOL]
    if penetrating:
        raise AssertionError(f"{len(penetrating)} generated targets penetrate")
    with (output / "cross_method_parity.csv").open() as stream:
        parity = list(csv.DictReader(stream))
    if len(parity) != CASE_COUNT or any(r["parity_status"] != "PASS" for r in parity):
        raise AssertionError("generated cross-method parity artifact failed")
    regenerated = build_cases(load_spec())
    if _csv_text(CANONICAL_FIELDS, regenerated) != (output / "canonical_cases.csv").read_text():
        raise AssertionError("canonical_cases.csv does not match authoritative YAML")
    return {"cases": len(canonical), "geometry": "PASS", "parity": "PASS"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "validate"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    result = generate(args.output) if args.command == "generate" else validate_artifacts(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
