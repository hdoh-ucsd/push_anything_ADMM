#!/usr/bin/env python3
"""Render the 2-object goal configuration: block T + block H.

The port cannot SIMULATE two manipulands (see the multi-object scoping
report), but the scene is fully determined by the reference's N=2 layout, so
it can be built and rendered statically.

Reference N=2 geometry, base_names = [push_t, H_shape_texture]:
  goal   x = 0.5, y = 0.2*i - 0.1*(N-1)          -> i0 y=-0.1, i1 y=+0.1
         quat: per-object letter_settings, else [0.707,0,0,0.707]
           push_t         : NO letter_settings entry -> default
           H_shape_texture: [0.71, 0, 0, -0.73]
  spawn  x = 0.4 + 0.02*i,  y = -0.3 + 0.2*i     -> i0 (0.40,-0.30)
         shared spawn yaw quat [0.393, 0, 0, 0.92]  (~133.8 deg)
Objects are drawn at their RESTING height (init_z); the reference's 9 mm
spawn drop is a transient and would only blur a static figure.
"""
import os
import re
import sys

import numpy as np
import pydrake.all as ad
import yaml
from PIL import Image

REPO = "/root/push_anything_ADMM/.claude/worktrees/fig8-lowcom-single-goal"
sys.path.insert(0, REPO)
from sim.env_builder import ROBOT_BASE_XYZ  # noqa: E402

OUT = "/root/push_anything_ADMM/results/fig8_blocks/render2obj"
os.makedirs(OUT, exist_ok=True)

PAIR = ["push_t", "H_shape_texture_block"]
GOAL_QUAT = {
    "push_t": [0.707, 0.0, 0.0, 0.707],                  # no letter_settings
    "H_shape_texture_block": [0.71, 0.0, 0.0, -0.73],    # letter_settings
}
SPAWN_QUAT = [0.393, 0.0, 0.0, 0.92]

# Object colours come from each block SDF's own <diffuse> (driven by the block
# task's color_rgba); only the goal ghosts are tinted here.
GHOST_RGBA = [0.10, 0.90, 0.10, 0.42]


def qn(q):
    q = np.asarray(q, float)
    return q / np.linalg.norm(q)


def rt(quat, xyz):
    w, x, y, z = qn(quat)
    return ad.RigidTransform(ad.RotationMatrix(ad.Quaternion(w, x, y, z)),
                             list(xyz))


def sdf_boxes(path):
    """(size, pose) of every <visual><box> in a block SDF."""
    txt = open(path).read()
    out = []
    for vis in re.findall(r"<visual\b.*?</visual>", txt, re.S):
        mb = re.search(r"<box>\s*<size>([^<]*)</size>", vis, re.S)
        if not mb:
            continue
        mp = re.search(r"<pose>([^<]*)</pose>", vis)
        pose = [float(v) for v in mp.group(1).split()] if mp else [0.0] * 6
        out.append(([float(v) for v in mb.group(1).split()], pose))
    return out


def main():
    cfg = yaml.safe_load(open(os.path.join(REPO, "config/tasks.yaml")))["tasks"]

    builder = ad.DiagramBuilder()
    plant, scene_graph = ad.AddMultibodyPlantSceneGraph(builder, 0.001)
    parser = ad.Parser(plant)
    parser.SetAutoRenaming(True)      # the reference's own fix for N objects

    # table (reference ground.urdf envelope)
    plant.RegisterVisualGeometry(plant.world_body(),
                                 ad.RigidTransform([0.0, 0.0, -0.05]),
                                 ad.Box(5.0, 0.91, 0.1), "table_vis",
                                 [0.55, 0.52, 0.45, 1.0])

    panda = parser.AddModelsFromUrl(
        "package://drake_models/franka_description/urdf/panda_arm.urdf")[0]
    plant.WeldFrames(plant.world_frame(),
                     plant.GetFrameByName("panda_link0", panda),
                     ad.RigidTransform(ROBOT_BASE_XYZ))

    models, spawn, goal = {}, {}, {}
    for i, task in enumerate(PAIR):
        t = cfg[task]
        sdf = os.path.join(REPO, t["object_sdf"])
        models[task] = parser.AddModels(sdf)[0]
        iz = float(t["init_xyz"][2])
        spawn[task] = (SPAWN_QUAT, [0.4 + 0.02 * i, -0.3 + 0.2 * i, iz])
        goal[task] = (GOAL_QUAT[task], [0.5, 0.2 * i - 0.1 * (len(PAIR) - 1), iz])

        # goal ghost: this object's own boxes, at the goal pose
        X_goal = rt(*goal[task])
        for k, (size, pose) in enumerate(sdf_boxes(sdf)):
            X_loc = ad.RigidTransform(
                ad.RotationMatrix(ad.RollPitchYaw(*pose[3:6])), list(pose[0:3]))
            plant.RegisterVisualGeometry(
                plant.world_body(), X_goal.multiply(X_loc), ad.Box(*size),
                f"ghost_{task}_{k}", GHOST_RGBA)

    # camera
    from pydrake.geometry import (MakeRenderEngineVtk, RenderEngineVtkParams,
                                  ClippingRange, DepthRange, RenderCameraCore,
                                  ColorRenderCamera, DepthRenderCamera)
    from pydrake.systems.sensors import CameraInfo
    scene_graph.AddRenderer("vtk", MakeRenderEngineVtk(RenderEngineVtkParams()))
    intr = CameraInfo(width=1600, height=1000, fov_y=np.radians(45.0))
    core = RenderCameraCore("vtk", intr, ClippingRange(0.05, 8.0),
                            ad.RigidTransform())
    cams = {}
    # framed on the union of the spawn row and the goal row:
    # x in [0.28,0.62], y in [-0.40,0.22]  ->  centre ~ (0.46,-0.09)
    CX, CY = 0.46, -0.09
    VIEWS = {
        "persp": ([CX + 0.62, CY - 0.58, 0.52], [CX, CY, 0.02]),
        "front": ([CX + 0.80, CY, 0.30], [CX, CY, 0.03]),
        "top":   ([CX, CY, 0.78], [CX, CY, 0.00]),
    }
    for name, (eye, tgt) in VIEWS.items():
        eye = np.asarray(eye, float)
        tgt = np.asarray(tgt, float)
        fwd = tgt - eye
        fwd /= np.linalg.norm(fwd)
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(fwd, up)) > 0.999:            # top-down: pick a stable up
            up = np.array([0.0, 1.0, 0.0])
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        R = np.column_stack([right, down, fwd])
        s = builder.AddSystem(ad.RgbdSensor(
            parent_id=scene_graph.world_frame_id(),
            X_PB=ad.RigidTransform(ad.RotationMatrix(R), eye.tolist()),
            color_camera=ColorRenderCamera(core, show_window=False),
            depth_camera=DepthRenderCamera(core, DepthRange(0.1, 8.0))))
        s.set_name(f"cam_{name}")
        builder.Connect(scene_graph.get_query_output_port(),
                        s.query_object_input_port())
        cams[name] = s

    plant.Finalize()
    diagram = builder.Build()
    ctx = diagram.CreateDefaultContext()
    pc = plant.GetMyMutableContextFromRoot(ctx)

    # Arm pose: set by joint name, which avoids depending on how the parser
    # named the model instance (SetAutoRenaming is on).
    q0 = cfg[PAIR[0]].get("q_init_franka")
    if q0:
        for j, val in zip([f"panda_joint{k}" for k in range(1, 8)], q0):
            plant.GetJointByName(j).set_angle(pc, float(val))

    for task in PAIR:
        body = plant.GetBodyByName(cfg[task]["link_name"], models[task])
        plant.SetFreeBodyPose(pc, body, rt(*spawn[task]))

    diagram.ForcedPublish(ctx)
    for name, s in cams.items():
        img = s.color_image_output_port().Eval(
            s.GetMyContextFromRoot(ctx)).data[:, :, :3]
        p = os.path.join(OUT, f"twoobj_goal_{name}.png")
        Image.fromarray(img).save(p)
        print("wrote", p)

    print("\nlayout (port frame, metres):")
    for i, task in enumerate(PAIR):
        print(f"  {task:24s} spawn={np.round(spawn[task][1],4)} "
              f"goal={np.round(goal[task][1],4)} goal_quat={GOAL_QUAT[task]}")


if __name__ == "__main__":
    main()
