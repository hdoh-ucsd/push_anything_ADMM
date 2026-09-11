#!/usr/bin/env python3
"""Render an xArm6 OIM-fidelity run (white table, stick EE) from state_trace.jsonl."""
import argparse, json, subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from pydrake.geometry import (Box, ClippingRange, ColorRenderCamera,
    DepthRange, DepthRenderCamera, MakeRenderEngineVtk, RenderCameraCore,
    RenderEngineVtkParams, Rgba)
from pydrake.math import RigidTransform, RotationMatrix, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.sensors import CameraInfo, RgbdSensor
from pydrake.common.eigen_geometry import Quaternion

WT = Path('/root/push_anything_ADMM/external/oim_c++_anything/.claude/worktrees/audit-xarm6-plant')

p = argparse.ArgumentParser()
p.add_argument('--run-dir', type=Path, required=True)
p.add_argument('--object-sdf', required=True)   # relative to WT
p.add_argument('--object-body', default='vertical_link')
p.add_argument('--title', required=True)
p.add_argument('--goal', type=float, nargs=3, default=None)
p.add_argument('--fps', type=int, default=10)
p.add_argument('--boxes', type=str, default=None, help='x,y,z,sx,sy,sz;... obstacles')
p.add_argument('--out', type=Path, default=None)
args = p.parse_args()

builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)
from pydrake.geometry import LightParameter
params = RenderEngineVtkParams(
    lights=[LightParameter(type='directional', direction=[0, 0, -1],
                           intensity=1.0, frame='world'),
            LightParameter(type='directional', direction=[-0.6, 0, -0.8],
                           intensity=0.4, frame='world')])
scene_graph.AddRenderer('r', MakeRenderEngineVtk(params))
parser = Parser(plant, scene_graph)
parser.package_map().PopulateFromFolder(str(WT))
arm = parser.AddModels(str(Path('/root/push_anything_ADMM/scripts/c3ab/xarm6_render_model/xarm6_policyport.xml')))[0]
# (MJCF already welds xarm6_link_base to world.)
parser.AddModels(str(WT/'examples/sampling_c3/urdf/end_effector_xarm6_stick.urdf'))
plant.WeldFrames(plant.GetFrameByName('xarm6_link6'),
                 plant.GetFrameByName('end_effector_flange'), RigidTransform())
obj = parser.AddModels(str(WT/args.object_sdf))[0]
def object_visual_shapes(sdf_path, body_name):
    """Shapes + poses of every visual geom, expressed in body_name's frame."""
    b2 = DiagramBuilder()
    pl, sg2 = AddMultibodyPlantSceneGraph(b2, 0.0)
    m = Parser(pl, sg2).AddModels(sdf_path)[0]
    root = pl.GetBodyByName(body_name, m)
    pl.WeldFrames(pl.world_frame(), root.body_frame(), RigidTransform())
    pl.Finalize()
    ctx2 = pl.CreateDefaultContext()
    ins2 = sg2.model_inspector()
    out = []
    for bi in pl.GetBodyIndices(m):
        body2 = pl.get_body(bi)
        X_WB = pl.EvalBodyPoseInWorld(ctx2, body2)  # == X_root_body
        for gid in pl.GetVisualGeometriesForBody(body2):
            out.append((ins2.GetShape(gid), X_WB @ ins2.GetPoseInFrame(gid)))
    return out
# OIM white table: 0.80 x 1.523, top at z=-0.029 in the base frame.
plant.RegisterVisualGeometry(plant.world_body(),
                             RigidTransform([0.35, 0, -0.029-0.05]),
                             Box(0.80, 1.523, 0.1), 'table_visual',
                             [0.95, 0.95, 0.95, 1.0])
if args.boxes:
    for bi, spec in enumerate(args.boxes.split(';')):
        bx, by, bz, sx, sy, sz = map(float, spec.split(','))
        plant.RegisterVisualGeometry(plant.world_body(),
                                     RigidTransform([bx, by, bz]),
                                     Box(sx, sy, sz), f'scene_box_{bi}',
                                     [1.0, 0.647, 0.0, 1.0])
rows = [json.loads(l) for l in open(args.run_dir/'state_trace.jsonl') if l.strip()]
if args.goal is not None:
    from pydrake.multibody.tree import SpatialInertia
    gx, gy, gyaw = args.goal
    gz = rows[0]['obj'][6]  # goal ghost rests at the spawn height
    ghost_body = plant.AddRigidBody(
        'goal_ghost', plant.AddModelInstance('ghost_mi'),
        SpatialInertia.SolidSphereWithMass(0.001, 0.01))
    plant.WeldFrames(plant.world_frame(), ghost_body.body_frame(),
                     RigidTransform(RotationMatrix.MakeZRotation(gyaw),
                                    [gx, gy, gz]))
    for n, (shape, X_BG) in enumerate(
            object_visual_shapes(str(WT/args.object_sdf), args.object_body)):
        plant.RegisterVisualGeometry(ghost_body, X_BG, shape, f'ghost_{n}',
                                     np.array([0.0, 1.0, 0.0, 0.3]))
plant.Finalize()

core = RenderCameraCore('r', CameraInfo(960, 720, np.deg2rad(50)),
                        ClippingRange(0.05, 10.0), RigidTransform())
color_cam = ColorRenderCamera(core, False)
depth_cam = DepthRenderCamera(core, DepthRange(0.1, 9.0))
# OIM front-ish camera: look-at from (1.35, 0, 0.85) to the table center.
eye = np.array([1.35, 0.0, 0.85]); target = np.array([0.35, 0.0, 0.0])
zc = target - eye; zc /= np.linalg.norm(zc)              # camera +z = view dir
xc = np.cross(zc, np.array([0.0, 0.0, 1.0])); xc /= np.linalg.norm(xc)
yc = np.cross(zc, xc)                                    # +y down in image
X_WC = RigidTransform(RotationMatrix(np.column_stack([xc, yc, zc])), eye)
sensor = builder.AddSystem(RgbdSensor(scene_graph.world_frame_id(), X_WC,
                                      color_cam, depth_cam))
builder.Connect(scene_graph.get_query_output_port(),
                sensor.query_object_input_port())
diagram = builder.Build()
ctx = diagram.CreateDefaultContext()
pctx = plant.GetMyMutableContextFromRoot(ctx)
sctx = sensor.GetMyContextFromRoot(ctx)

body = plant.GetBodyByName(args.object_body, obj)
tmp = args.run_dir/'frames'; tmp.mkdir(exist_ok=True)
step = max(1, int(round(1.0/(args.fps*0.1))))
frames = rows[::step]
for i, r in enumerate(frames):
    q = np.array(r['q'][:5])
    plant.SetPositions(pctx, arm, q)
    quat = np.array(r['obj'][:4]); quat /= np.linalg.norm(quat)
    plant.SetFreeBodyPose(pctx, body,
                          RigidTransform(Quaternion(quat), r['obj'][4:7]))
    img = sensor.color_image_output_port().Eval(sctx).data[:, :, :3]
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.text((10, 10), f"{args.title}  t={r['t']:.1f}s", fill=(0, 0, 0))
    d.text((10, 28), f"pos_err={r['pos_err']:.3f}m ang_err={r['ang_err']:.3f}rad",
           fill=(0, 0, 0))
    im.save(tmp/f'f{i:05d}.png')
out = args.out or (args.run_dir/'render.mp4')
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(args.fps),
                '-i', str(tmp/'f%05d.png'), '-pix_fmt', 'yuv420p', str(out)],
               check=True)
subprocess.run(['rm', '-rf', str(tmp)])
print('wrote', out, len(frames), 'frames')
