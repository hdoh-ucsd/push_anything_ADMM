#!/usr/bin/env python3
"""Render a reference-C3+ ablation run from state_trace.jsonl."""
import argparse, json, math, subprocess, tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pydrake.geometry import (Box, ClippingRange, ColorRenderCamera,
    DepthRange, DepthRenderCamera, MakeRenderEngineVtk, RenderCameraCore,
    RenderEngineVtkParams)
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.sensors import CameraInfo, RgbdSensor
from pydrake.common.eigen_geometry import Quaternion

WT = Path('/root/push_anything_ADMM/external/oim_c++_anything/.claude/worktrees/c3plus-ablation')

p = argparse.ArgumentParser()
p.add_argument('--run-dir', type=Path, required=True)
p.add_argument('--object-sdf', required=True)
p.add_argument('--title', required=True)
p.add_argument('--fps', type=int, default=10)
p.add_argument('--obstacle', type=float, nargs=4, default=None, help='x y z half_size')
p.add_argument('--boxes', type=str, default=None, help='x,y,z,sx,sy,sz;... full sizes')
p.add_argument('--object-body', type=str, default='vertical_link')
args = p.parse_args()

builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)
scene_graph.AddRenderer('r', MakeRenderEngineVtk(RenderEngineVtkParams()))
parser = Parser(plant, scene_graph)
franka = parser.AddModelsFromUrl(
    'package://drake_models/franka_description/urdf/panda_arm.urdf')[0]
plant.WeldFrames(plant.world_frame(), plant.GetFrameByName('panda_link0'))
parser.AddModels(str(WT/'examples/sampling_c3/urdf/end_effector_full.urdf'))
from pydrake.math import RollPitchYaw
plant.WeldFrames(plant.GetFrameByName('panda_link7'),
                 plant.GetFrameByName('end_effector_flange'),
                 RigidTransform(RollPitchYaw(3.1415, 0, 0).ToRotationMatrix(),
                                [0, 0, 0.107]))
obj = parser.AddModels(str(WT/args.object_sdf))[0]
plant.RegisterVisualGeometry(plant.world_body(),
                             RigidTransform([0.5, 0, -0.0395]),
                             Box(1.0, 1.2, 0.05), 'platform_visual',
                             [0.92, 0.92, 0.88, 1.0])
if args.boxes:
    for bi, spec in enumerate(args.boxes.split(';')):
        bx, by, bz, sx, sy, sz = map(float, spec.split(','))
        plant.RegisterVisualGeometry(plant.world_body(),
                                     RigidTransform([bx, by, bz]),
                                     Box(sx, sy, sz), f'scene_box_{bi}',
                                     [0.85, 0.35, 0.1, 1.0])
if args.obstacle is not None:
    ox, oy, oz, oh = args.obstacle
    plant.RegisterVisualGeometry(plant.world_body(),
                                 RigidTransform([ox, oy, oz]),
                                 Box(2*oh, 2*oh, 2*oh), 'obstacle_visual',
                                 [0.85, 0.35, 0.1, 1.0])
plant.Finalize()

core = RenderCameraCore('r', CameraInfo(1280, 720, np.pi/4.5),
                        ClippingRange(0.05, 10.0), RigidTransform())
color_cam = ColorRenderCamera(core, show_window=False)
depth_cam = DepthRenderCamera(core, DepthRange(0.05, 10.0))
eye = np.array([1.5, -0.9, 0.8]); target = np.array([0.45, 0.1, 0.0])
z = (target-eye)/np.linalg.norm(target-eye)
x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x); y = np.cross(z, x)
cam = builder.AddSystem(RgbdSensor(
    scene_graph.world_frame_id(),
    RigidTransform(RotationMatrix(np.column_stack([x, y, z])), eye),
    color_cam, depth_cam))
builder.Connect(scene_graph.get_query_output_port(),
                cam.query_object_input_port())
diagram = builder.Build()
ctx = diagram.CreateDefaultContext()
pc = plant.GetMyMutableContextFromRoot(ctx)
cc = cam.GetMyContextFromRoot(ctx)
obj_body = plant.GetBodyByName(args.object_body)
font = ImageFont.truetype(
    '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 18)

recs = [json.loads(l) for l in (args.run_dir/'state_trace.jsonl').open()]
recs = [r for r in recs if r.get('q')]
stride = max(1, len(recs)//1200)
recs = recs[::stride]
out = args.run_dir/'rollout.mp4'
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    for i, r in enumerate(recs):
        for j, v in enumerate(r['q']):
            plant.GetJointByName(f'panda_joint{j+1}').set_angle(pc, v)
        q = r['obj']
        quat = np.array(q[:4]); quat /= np.linalg.norm(quat)
        plant.SetFreeBodyPose(pc, obj_body,
                              RigidTransform(Quaternion(quat), q[4:7]))
        diagram.ForcedPublish(ctx)
        rgb = cam.color_image_output_port().Eval(cc).data[:, :, :3]
        im = Image.fromarray(rgb.copy())
        d = ImageDraw.Draw(im)
        d.text((12, 8), f"{args.title}  t={r['t']:.1f}s", font=font,
               fill=(10, 10, 20))
        d.text((12, 32), f"pos_err={r['pos_err']:.3f} m  "
               f"ang_err={r['ang_err']:.3f} rad", font=font, fill=(10, 10, 20))
        im.save(tmp/f'f_{i:05d}.png')
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-framerate',
                    str(args.fps), '-i', str(tmp/'f_%05d.png'), '-c:v',
                    'libx264', '-pix_fmt', 'yuv420p', str(out)], check=True)
print('wrote', out, len(recs), 'frames')
