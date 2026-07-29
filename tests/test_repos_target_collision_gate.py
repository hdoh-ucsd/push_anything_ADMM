"""Per-tick collision check on the pursued repositioning target.

Reference (push_anything_dev@257e3ed, sampling_based_c3_controller.cc):
  cc:908-926  ComputeSignedDistanceToPoint(prev_repositioning_target_)
              against object geometries; in_collision when any distance
              <= sampling_params.sample_projection_clearance
              (push_t/parameters/sampling_params.yaml:30 -> 0.02 m)
  cc:931      pursued target re-enters the candidate set only if
              !in_collision
  cc:1205-1213 repos branch: colliding previous target is rejected,
              switch to new sample

The port skipped the whole check ("prev_repos is a previously-cleared
target so unlikely to be in collision") — false once the object rotates
under c3 pushes and the retained target ends up at/inside the moved face.
p138 forensics: T net drift (-25, -27) mm all outside c3, EE pressed at
39-46 mm from center at contact z — stale-target pressing creep.
"""
import numpy as np
import pytest

from pydrake.math import RigidTransform
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder
from pydrake.geometry import Box
from pydrake.multibody.tree import SpatialInertia, UnitInertia
from pydrake.multibody.plant import CoulombFriction

from control.sampling_c3.params import SamplingC3Params
from control.sampling_c3.sampling_based_c3_controller import SamplingC3Controller


@pytest.fixture(scope="module")
def box_world():
    """Plant with one 10 cm cube welded at the origin, collision geometry
    registered — enough for ComputeSignedDistanceToPoint."""
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    body = plant.AddRigidBody(
        "box", SpatialInertia(1.0, np.zeros(3), UnitInertia(1, 1, 1)))
    geom_id = plant.RegisterCollisionGeometry(
        body, RigidTransform(), Box(0.1, 0.1, 0.1), "box_collision",
        CoulombFriction(0.5, 0.5))
    plant.WeldFrames(plant.world_frame(), body.body_frame(), RigidTransform())
    plant.Finalize()
    diagram = builder.Build()
    diagram_ctx = diagram.CreateDefaultContext()
    plant_ctx = plant.GetMyContextFromRoot(diagram_ctx)
    return plant, plant_ctx, geom_id


def _controller(plant, geom_id, target, clearance=0.02):
    c = SamplingC3Controller.__new__(SamplingC3Controller)
    c.plant = plant
    c.params = SamplingC3Params()
    c.params.sampling_params.sample_projection_clearance = float(clearance)
    c._collision_check_geom_ids = {geom_id}
    c._current_repos_target = (None if target is None
                               else np.asarray(target, dtype=float))
    c.log_diag = False
    c._step = 1
    return c


def test_target_inside_object_is_in_collision(box_world):
    plant, ctx, gid = box_world
    c = _controller(plant, gid, target=[0.0, 0.0, 0.0])   # box center
    assert c._pursued_target_in_collision(ctx) is True


def test_target_within_clearance_is_in_collision(box_world):
    plant, ctx, gid = box_world
    # Box surface at x=0.05; point at 0.06 -> distance 0.01 < 0.02 clearance.
    c = _controller(plant, gid, target=[0.06, 0.0, 0.0])
    assert c._pursued_target_in_collision(ctx) is True


def test_target_beyond_clearance_is_clear(box_world):
    plant, ctx, gid = box_world
    # Point at x=0.09 -> distance 0.04 > 0.02 clearance.
    c = _controller(plant, gid, target=[0.09, 0.0, 0.0])
    assert c._pursued_target_in_collision(ctx) is False


def test_no_target_is_clear(box_world):
    plant, ctx, gid = box_world
    c = _controller(plant, gid, target=None)
    assert c._pursued_target_in_collision(ctx) is False
