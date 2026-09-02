import numpy as np
import pytest

from sim.oim_xarm6_tabletop import (
    OIM_ORIENTATION_SUCCESS_TOLERANCE,
    OIM_TRANSLATION_SUCCESS_TOLERANCE,
    OIM_DRAKE_TRANSLATION,
    SCENE_NAMES,
    XARM6_HOME,
    build_oim_tabletop_scene,
    load_oim_xarm6_config,
    oim_c3plus_configuration,
    oim_goal_reached,
    register_oim_pose_for_dairlab,
    set_oim_start_configuration,
    xarm_configured_actuation,
    xarm_velocity_servo_torque,
)


def test_oim_pose_success_tolerances():
    assert OIM_TRANSLATION_SUCCESS_TOLERANCE == pytest.approx(0.05)
    assert OIM_ORIENTATION_SUCCESS_TOLERANCE == pytest.approx(0.10)
    assert oim_goal_reached(0.05, 0.10)
    assert not oim_goal_reached(0.050001, 0.10)
    assert not oim_goal_reached(0.05, 0.100001)


def test_oim_pose_registration_matches_validated_dairlab_importer():
    raw = np.array([0.381, 0.400, -1.25])
    registered = register_oim_pose_for_dairlab(raw)
    np.testing.assert_allclose(registered, [0.481, 0.200, -1.25])
    np.testing.assert_allclose(raw, [0.381, 0.400, -1.25])

    with pytest.raises(ValueError, match=r"expected an \[x, y, yaw\] pose"):
        register_oim_pose_for_dairlab([0.1, 0.2])


def test_oim_robot_yaml_is_the_wrapper_source_of_truth():
    config = load_oim_xarm6_config()
    assert config["run"]["goal_pos_tol"] == pytest.approx(0.05)
    assert config["run"]["goal_theta_tol"] == pytest.approx(0.1)
    assert config["world3d"]["planning_dt"] == pytest.approx(0.05)
    assert config["run"]["steps"] == 2000
    resolved = oim_c3plus_configuration(config)
    assert resolved.execution_dt == pytest.approx(0.002)
    assert resolved.planning_dt == pytest.approx(0.05)
    assert resolved.horizon == 32
    assert resolved.admm_iterations == 4
    assert resolved.translation_tolerance == pytest.approx(0.05)
    assert resolved.orientation_tolerance == pytest.approx(0.1)
    assert "condim torsional and rolling friction" in OIM_DRAKE_TRANSLATION["approximate"]


@pytest.mark.parametrize("name", SCENE_NAMES)
def test_oim_scene_loads_with_xarm_and_planar_object(name):
    scene = build_oim_tabletop_scene(name)
    assert len(scene.arm_positions) == 6
    assert scene.plant.num_actuators() == 6
    assert scene.plant.HasBodyNamed("xarm6_stick")
    assert scene.plant.HasJointNamed("T_x")
    assert scene.plant.HasJointNamed("T_y")
    assert scene.plant.HasJointNamed("T_z")

    context = scene.diagram.CreateDefaultContext()
    set_oim_start_configuration(scene, context)
    plant_context = scene.plant.GetMyContextFromRoot(context)
    q = scene.plant.GetPositions(plant_context)
    actual = np.array([q[scene.arm_positions[f"xarm6_joint{i}"]] for i in range(1, 7)])
    np.testing.assert_allclose(actual, XARM6_HOME, atol=1e-12)
    torque = xarm_velocity_servo_torque(scene, context, np.zeros(6))
    assert torque.shape == (6,)
    assert np.all(np.abs(torque) <= np.array([50, 50, 32, 32, 32, 20]))
    configured = xarm_configured_actuation(scene, context, np.zeros(6))
    assert configured.shape == (6,)
    assert np.all(np.isfinite(configured))
    assert scene.plant.time_step() == pytest.approx(0.002)


def test_oim_scene_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown OIM scene"):
        build_oim_tabletop_scene("not_a_scene")


def test_sampling_c3plus_scene_has_exact_floating_oim_t():
    scene = build_oim_tabletop_scene(
        "open_table", sampling_c3plus_object=True
    )
    block = scene.plant.GetBodyByName("block")
    assert block.is_floating_base_body()
    assert scene.plant.num_positions() == 13
    assert scene.plant.num_velocities() == 12
    inspector = scene.scene_graph.model_inspector()
    geoms = scene.plant.GetCollisionGeometriesForBody(block)
    sizes = sorted(
        tuple(inspector.GetShape(g).size()) for g in geoms
    )
    assert sizes == pytest.approx(sorted([
        (0.0198, 0.0794, 0.0596),
        (0.0890, 0.0198, 0.0596),
    ]))
