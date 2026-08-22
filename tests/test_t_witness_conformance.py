"""Canonical push_t witnesses match the reference controller SDF.

The reference LCS takes an object's ground witnesses from the last three
collision geometries of its *_controller.sdf — the corner spheres emitted
by controller_sdf_generation.py get_obj_corners() (mesh-AABB bottom
corners). The port plumbs the same three centres through the task's
`ground_witness_points_body`. This test re-derives the sphere centres from
the SDF shipped in-repo and asserts the yaml literal matches, so neither
side can drift silently.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
TASKS = REPO / "config/tasks.yaml"


def _sphere_collision_poses(sdf_path):
    # Drake SDFs use the `drake:` prefix without an xmlns declaration.
    text = sdf_path.read_text().replace(
        "<sdf ", '<sdf xmlns:drake="https://drake.mit.edu" ', 1)
    root = ET.fromstring(text)
    poses = []
    for coll in root.iter("collision"):
        if coll.find("./geometry/sphere") is None:
            continue
        pose = coll.find("pose")
        poses.append([float(v) for v in pose.text.split()[:3]])
    return np.asarray(poses)


BLOCK_SDF = REPO / "sim/models/push_t/push_t_control.sdf"


def _check(task_name, sdf_path):
    sdf_centres = _sphere_collision_poses(sdf_path)
    assert sdf_centres.shape == (3, 3), (
        f"{sdf_path.name} must carry exactly 3 corner spheres")
    with open(TASKS) as f:
        tasks = yaml.safe_load(f)
    task_cfg = tasks["tasks"][task_name]
    yaml_pts = np.asarray(task_cfg["ground_witness_points_body"], dtype=float)
    # Same points, same registration order (top-left, top-right, bottom).
    np.testing.assert_allclose(yaml_pts, sdf_centres, atol=1e-12)


def test_push_t_witnesses_match_controller_sdf():
    _check("push_t", BLOCK_SDF)
