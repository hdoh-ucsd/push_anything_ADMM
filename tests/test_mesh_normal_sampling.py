"""kMeshNormal sampler conformance: face filter, area weighting, geometry.

Checks the port of the reference anything-lineage sampler
(generate_samples.cc:445-551 + face preprocessing cc:430-489) against the
actual T_shape_video.obj shipped in-repo, with the anything literals.
"""

from pathlib import Path

import numpy as np
import pytest

from control.sampling_c3.params import SamplingParams, SamplingStrategy
from control.sampling_c3.sampling import (
    generate_samples, load_mesh_faces, _random_on_mesh_normal)

REPO = Path(__file__).resolve().parents[1]
OBJ = REPO / "sim/models/T_shape_video/T_shape_video.obj"
BUFFER, CLEARANCE = 0.035, 0.027


@pytest.fixture(scope="module")
def faces():
    return load_mesh_faces(OBJ, BUFFER, CLEARANCE)


def test_face_filter_keeps_only_side_walls(faces):
    # Reference filter: normal_z^2 < buffer^2 - clearance^2 + 0.04
    thr = BUFFER ** 2 - CLEARANCE ** 2 + 0.04
    assert (faces["normals"][:, 2] ** 2 < thr).all()
    # The T has substantial vertical wall area; the filter must not
    # empty it, nor keep the (horizontal) top/bottom faces.
    assert 100 < len(faces["areas"]) < 4998


def test_bins_are_cumulative_areas(faces):
    np.testing.assert_allclose(np.diff(faces["bins"]), faces["areas"])
    assert faces["total_area"] == pytest.approx(faces["bins"][-1])


def test_samples_offset_along_normals_at_height(faces):
    params = SamplingParams()
    params.sampling_height = 0.031
    params.sample_projection_clearance = CLEARANCE
    params.buffer_distance = BUFFER
    rng = np.random.default_rng(0)
    obj_xy = np.array([0.5, 0.0])
    out = _random_on_mesh_normal(
        50, obj_xy, params, rng,
        obj_quat=np.array([1.0, 0.0, 0.0, 0.0]), obj_z=0.0243,
        projector=None, mesh_faces=faces)
    assert len(out) == 50
    P = np.array(out)
    # all snapped to sampling height (gen_planar_samples)
    np.testing.assert_allclose(P[:, 2], 0.031)
    # samples surround the object horizontally, not concentrated at a point
    spans = P[:, :2].max(0) - P[:, :2].min(0)
    assert (spans > 0.10).all()
    # each sample sits roughly buffer_distance outside the footprint slice:
    # distance from the object centre exceeds the min half-width (~stem 25mm)
    r = np.linalg.norm(P[:, :2] - obj_xy, axis=1)
    assert (r > 0.02).all()


def test_dispatch_requires_mesh_faces():
    params = SamplingParams()
    params.sampling_strategy = SamplingStrategy.kMeshNormal
    with pytest.raises(ValueError):
        generate_samples(SamplingStrategy.kMeshNormal, 2,
                         np.array([0.5, 0.0]), params)
