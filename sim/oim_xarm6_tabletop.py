"""PyDrake wrapper for OIM's xArm6 tabletop MJCF scenes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import yaml
from pydrake.geometry import SceneGraph
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.multibody.tree import FixedOffsetFrame
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.systems.framework import Diagram, DiagramBuilder


SCENE_NAMES = (
    "open_table",
    "single_obstacle",
    "shelf_gap",
    "ycb_clutter",
    "icra_sign",
)

# OIM keyframes specify five joints; the canonical native xArm6 wrist roll
# starts at zero and remains available for posture/null-space regulation.
# OIM simulated tabletop family (`_tee_scene` / tee_start.xml). The
# 49.2/34.8/-80.6/0/45.9 posture belongs to the separate *_real scenes.
XARM6_HOME = np.deg2rad([0.0, -45.0, -45.0, 0.0, 90.0, 0.0])
XARM6_EFFORT_LIMITS = (50.0, 50.0, 32.0, 32.0, 32.0, 20.0)
OIM_TRANSLATION_SUCCESS_TOLERANCE = 0.05  # m
OIM_ORIENTATION_SUCCESS_TOLERANCE = 0.10  # rad
# Registration established by the DAIRLab C3+ importer comparison.  The OIM
# pose files are expressed in the source scene's planar frame; the controller
# scene uses this translated/compressed workspace frame.
OIM_DAIRLAB_X_OFFSET = 0.10  # m
OIM_DAIRLAB_Y_SCALE = 0.50


def register_oim_pose_for_dairlab(pose) -> np.ndarray:
    """Map an OIM [x, y, yaw] pose into the validated DAIRLab frame."""
    registered = np.asarray(pose, dtype=float).copy()
    if registered.shape != (3,):
        raise ValueError(f"expected an [x, y, yaw] pose, got {registered.shape}")
    registered[0] += OIM_DAIRLAB_X_OFFSET
    registered[1] *= OIM_DAIRLAB_Y_SCALE
    return registered


def oim_goal_reached(translation_error: float, orientation_error: float) -> bool:
    """Returns whether an OIM rollout satisfies its pose success criterion."""
    return (
        float(translation_error) <= OIM_TRANSLATION_SUCCESS_TOLERANCE
        and float(orientation_error) <= OIM_ORIENTATION_SUCCESS_TOLERANCE
    )


@dataclass(frozen=True)
class OimTabletopScene:
    name: str
    diagram: Diagram
    plant: MultibodyPlant
    scene_graph: SceneGraph
    model_instances: tuple
    arm_positions: Mapping[str, int]
    arm_velocities: Mapping[str, int]
    robot_config: Mapping
    render_camera: object | None = None


@dataclass(frozen=True)
class OimC3PlusConfiguration:
    """OIM YAML values consumed by a Drake-side C3+ experiment."""

    planning_dt: float
    execution_dt: float
    horizon: int
    admm_iterations: int
    rho: float
    rho_torque: float
    gamma: float
    run_steps: int
    translation_tolerance: float
    orientation_tolerance: float
    costs: Mapping


def oim_c3plus_configuration(config: Mapping) -> OimC3PlusConfiguration:
    """Resolve the C3+/run settings from the vendored xarm6.yaml."""
    world = config["world3d"]
    sampler = config["sampler"]
    admm = config["admm"]
    run = config["run"]
    return OimC3PlusConfiguration(
        planning_dt=float(world["planning_dt"]),
        execution_dt=float(world["exec_timestep"]),
        horizon=int(sampler["horizon"]),
        admm_iterations=int(admm["n_admm"]),
        rho=float(admm["rho"]),
        rho_torque=float(admm["rho_torque"]),
        gamma=float(admm["gamma"]),
        run_steps=int(run["steps"]),
        translation_tolerance=float(run["goal_pos_tol"]),
        orientation_tolerance=float(run["goal_theta_tol"]),
        costs=dict(config["costs"]),
    )


OIM_DRAKE_TRANSLATION = {
    "exact": (
        "rigid-body tree and inertias",
        "joint axes and limits",
        "six velocity-servo gains and command/effort bounds",
        "per-arm-body gravity compensation",
        "planning/run/ADMM/cost configuration values",
    ),
    "equivalent": (
        "MuJoCo collision masks and adjacent-body exclusions",
        "translational Coulomb friction",
    ),
    "approximate": (
        "solref/solimp contact compliance",
        "condim torsional and rolling friction",
        "MuJoCo implicitfast and solver iteration settings",
    ),
}


def model_root() -> Path:
    return Path(__file__).resolve().parent / "models/oim_xarm6_tabletop"


def load_oim_xarm6_config() -> Mapping:
    """Load the vendored OIM robot/run configuration without reinterpretation."""
    path = model_root() / "configs/robots/xarm6.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing imported robot configuration {path}; run "
            "scripts/import_oim_xarm6_tabletop.py"
        )
    with path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"invalid OIM xArm configuration: {path}")
    return config


def build_oim_tabletop_scene(
    scene: str,
    *,
    time_step: Optional[float] = None,
    sampling_c3plus_object: bool = False,
    add_camera: bool = False,
    camera_width: int = 960,
    camera_height: int = 720,
) -> OimTabletopScene:
    """Loads one OIM MJCF scene as a finalized PyDrake diagram.

    The MJCF supplies the complete xArm kinematic tree, joint limits, link
    masses, stick geometry, table, pushed object, and static obstacles.
    OIM's velocity actuators are imported by Drake as joint actuators; their
    controller semantics are intentionally left to the caller.
    """
    if scene not in SCENE_NAMES:
        raise ValueError(f"unknown OIM scene {scene!r}; expected one of {SCENE_NAMES}")
    filename = (
        "open_table_sampling_c3plus.xml"
        if sampling_c3plus_object and scene == "open_table"
        else f"{scene}.xml"
    )
    if sampling_c3plus_object and scene != "open_table":
        raise ValueError(
            "Sampling-C3+ floating-object scene is currently implemented "
            "only for open_table"
        )
    path = model_root() / "xarm6_pusht_tabletop" / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"missing imported scene {path}; run scripts/import_oim_xarm6_tabletop.py"
        )

    robot_config = load_oim_xarm6_config()
    resolved = oim_c3plus_configuration(robot_config)
    if time_step is None:
        time_step = resolved.execution_dt
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    models = tuple(Parser(plant).AddModels(str(path)))
    plant.AddFrame(FixedOffsetFrame(
        "oim_stick_tip",
        plant.GetBodyByName("xarm6_stick").body_frame(),
        RigidTransform([0.0, 0.0, 0.1794]),
    ))
    # Drake's MJCF parser currently ignores <velocity> actuators. Recreate
    # the controllable six-DOF arm explicitly; callers may wrap these torque
    # inputs with OIM's velocity-servo law (kv=[300,300,200,200,200,200]).
    for i, effort in enumerate(XARM6_EFFORT_LIMITS, start=1):
        joint = plant.GetJointByName(f"xarm6_joint{i}")
        actuator = plant.AddJointActuator(f"xarm6_joint{i}_actuator", joint)
        actuator.set_effort_limit(effort)
    plant.Finalize()
    arm_positions = {
        f"xarm6_joint{i}": plant.GetJointByName(f"xarm6_joint{i}").position_start()
        for i in range(1, 7)
    }
    arm_velocities = {
        f"xarm6_joint{i}": plant.GetJointByName(f"xarm6_joint{i}").velocity_start()
        for i in range(1, 7)
    }
    builder.ExportInput(plant.get_actuation_input_port(), "xarm6_actuation")
    builder.ExportOutput(plant.get_state_output_port(), "plant_state")
    builder.ExportOutput(plant.get_contact_results_output_port(), "contact_results")
    render_camera = None
    if add_camera:
        from pydrake.geometry import (
            ClippingRange, ColorRenderCamera, DepthRange, DepthRenderCamera,
            MakeRenderEngineVtk, RenderCameraCore, RenderEngineVtkParams,
        )
        from pydrake.systems.sensors import CameraInfo, RgbdSensor

        scene_graph.AddRenderer(
            "oim_vtk", MakeRenderEngineVtk(RenderEngineVtkParams())
        )
        intrinsics = CameraInfo(
            width=camera_width, height=camera_height,
            fov_y=np.deg2rad(50.0),
        )
        core = RenderCameraCore(
            "oim_vtk", intrinsics, ClippingRange(0.05, 5.0), RigidTransform()
        )
        eye = np.array([1.05, -0.72, 0.68])
        target = np.array([0.40, 0.0, 0.04])
        forward = target - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        X_WC = RigidTransform(
            RotationMatrix(np.column_stack((right, down, forward))), eye
        )
        render_camera = builder.AddSystem(RgbdSensor(
            parent_id=scene_graph.world_frame_id(), X_PB=X_WC,
            color_camera=ColorRenderCamera(core, show_window=False),
            depth_camera=DepthRenderCamera(core, DepthRange(0.1, 5.0)),
        ))
        render_camera.set_name("oim_render_camera")
        builder.Connect(
            scene_graph.get_query_output_port(),
            render_camera.query_object_input_port(),
        )
    diagram = builder.Build()
    return OimTabletopScene(
        scene, diagram, plant, scene_graph, models, arm_positions, arm_velocities,
        robot_config, render_camera,
    )


def xarm_velocity_servo_torque(
    scene: OimTabletopScene,
    context,
    desired_velocity,
) -> np.ndarray:
    """Translates OIM's velocity-actuator command into Drake joint torque.

    This reproduces the MJCF actuator gains and effort clipping. Gravity
    compensation is supplied separately by the caller's inverse-dynamics or
    gravity-compensation controller because MuJoCo's per-body ``gravcomp``
    attribute has no direct MJCF-parser equivalent in Drake.
    """
    desired = np.asarray(desired_velocity, dtype=float)
    if desired.shape != (6,):
        raise ValueError("desired_velocity must contain six xArm joint rates")
    plant_context = scene.plant.GetMyContextFromRoot(context)
    v = scene.plant.GetVelocities(plant_context)
    actual = np.array([
        v[scene.arm_velocities[f"xarm6_joint{i}"]] for i in range(1, 7)
    ])
    gains = np.array([300.0, 300.0, 200.0, 200.0, 200.0, 200.0])
    limits = np.asarray(XARM6_EFFORT_LIMITS)
    torque = gains * (np.clip(desired, -0.5, 0.5) - actual)
    return np.clip(torque, -limits, limits)


def xarm_configured_actuation(
    scene: OimTabletopScene,
    context,
    desired_velocity,
) -> np.ndarray:
    """Apply OIM's servo and ``gravcomp=1`` semantics in Drake.

    MuJoCo gravity compensation is a passive body force and therefore is not
    clipped by the velocity actuator's force range.  We reproduce that order:
    clip the velocity-servo contribution first, then add the five arm-joint
    six arm-joint gravity compensation torques.
    """
    servo = xarm_velocity_servo_torque(scene, context, desired_velocity)
    plant_context = scene.plant.GetMyContextFromRoot(context)
    gravity = scene.plant.CalcGravityGeneralizedForces(plant_context)
    compensation = np.array([
        -gravity[scene.arm_velocities[f"xarm6_joint{i}"]]
        for i in range(1, 7)
    ])
    # MuJoCo applies joint stiffness as a passive generalized force, outside
    # the velocity actuator's force clamp.  Drake's MJCF parser explicitly
    # ignores stiffness/springref, so reproduce it here in that same order.
    q = scene.plant.GetPositions(plant_context)
    spring = np.zeros(6)
    spring[3] = -175.0 * q[scene.arm_positions["xarm6_joint4"]]
    return servo + spring + compensation


def set_oim_start_configuration(scene: OimTabletopScene, context) -> None:
    """Applies OIM's shared arm home and scene-specific pushed-object start."""
    plant_context = scene.plant.GetMyMutableContextFromRoot(context)
    q = scene.plant.GetPositions(plant_context).copy()
    for i, value in enumerate(XARM6_HOME, start=1):
        q[scene.arm_positions[f"xarm6_joint{i}"]] = value

    if scene.plant.GetBodyByName("block").is_floating_base_body():
        # The Sampling-C3+ scene encodes this pose in its free-body default.
        scene.plant.SetPositions(plant_context, q)
        return

    # tee_start.xml: T at (0.381, +0.400); icra_sign: C at (0.300, +0.400).
    block_x = 0.300 if scene.name == "icra_sign" else 0.381
    for joint_name, value in (("T_x", block_x), ("T_y", 0.400), ("T_z", 0.0)):
        joint = scene.plant.GetJointByName(joint_name)
        q[joint.position_start()] = value
    scene.plant.SetPositions(plant_context, q)


def set_oim_free_body_pose(scene: OimTabletopScene, context, body_name: str,
                           pose_se2) -> None:
    """Set an imported floating object's world pose from ``[x, y, yaw]``."""
    pose = np.asarray(pose_se2, dtype=float).reshape(3)
    plant_context = scene.plant.GetMyMutableContextFromRoot(context)
    body = scene.plant.GetBodyByName(body_name)
    if not body.is_floating_base_body():
        raise ValueError(f"{body_name} is not a floating body")
    current = scene.plant.EvalBodyPoseInWorld(plant_context, body)
    scene.plant.SetFreeBodyPose(
        plant_context, body,
        RigidTransform(
            RotationMatrix.MakeZRotation(float(pose[2])),
            [float(pose[0]), float(pose[1]), current.translation()[2]],
        ),
    )
