#!/usr/bin/env python3
"""Apply one canonical OIM scenario to DAIRLab's native C++ anything demo."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import yaml


SCENES = ("open_table", "single_obstacle", "shelf_gap", "ycb_clutter", "icra_sign")
DAIRLIB_X_OFFSET = 0.10
DAIRLIB_Y_SCALE = 0.50
BOXES = {
    "open_table": [],
    "single_obstacle": [("obs_box", (0.35, 0.0, 0.05), (0.10, 0.10, 0.10))],
    "shelf_gap": [
        ("shelf_1", (0.62, 0.0, 0.07), (0.26, 0.16, 0.14)),
        ("shelf_2", (0.24, 0.0, 0.07), (0.10, 0.16, 0.14)),
    ],
    "ycb_clutter": [
        ("obs_box", (0.35, 0.0, 0.05), (0.10, 0.10, 0.10)),
        ("domino_sugar", (0.22, 0.20, 0.03), (0.175, 0.095, 0.06)),
        # Exact OIM mesh bounding extents. The native scene uses conservative
        # boxes because DAIRLab's controller does not model obstacle contacts.
        ("spam_can", (0.60, -0.16, 0.0418), (0.1021, 0.0601, 0.0835)),
        ("mustard_bottle", (0.15, -0.18, 0.0957), (0.0972, 0.0666, 0.1913)),
    ],
    # Conservative envelopes of the seven fixed glyphs in the ICRA sign.
    "icra_sign": [
        (name, (0.5, y, 0.0375), (0.103, 0.075, 0.075))
        for name, y in zip(("I", "R", "A", "2", "0", "2b", "6"),
                           (-0.55, -0.25, -0.10, 0.15, 0.30, 0.45, 0.60))
    ],
}


def replace_line(path: Path, key: str, value: str) -> None:
    text = path.read_text()
    updated, count = re.subn(rf"(?m)^{re.escape(key)}:.*$", f"{key}: {value}", text)
    if count != 1:
        raise RuntimeError(f"expected one {key!r} in {path}, found {count}")
    path.write_text(updated)


def write_scene(path: Path, scene: str) -> None:
    links = []
    for name, pos, size in BOXES[scene]:
        registered_pos = (pos[0] + DAIRLIB_X_OFFSET,
                          pos[1] * DAIRLIB_Y_SCALE, pos[2])
        pose = " ".join(map(str, (*registered_pos, 0, 0, 0)))
        dims = " ".join(map(str, size))
        links.append(f"""    <link name=\"{name}\">
      <pose>{pose}</pose>
      <collision name=\"collision\"><geometry><box><size>{dims}</size></box></geometry>
        <surface><friction><ode><mu>0.8</mu><mu2>0.8</mu2></ode></friction></surface>
      </collision>
      <visual name=\"visual\"><geometry><box><size>{dims}</size></box></geometry>
        <material><diffuse>0.45 0.45 0.48 1</diffuse></material></visual>
    </link>""")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("<?xml version=\"1.0\"?>\n<sdf version=\"1.7\">\n"
                    f"  <model name=\"oim_{scene}\"><static>true</static>\n"
                    + "\n".join(links) + "\n  </model>\n</sdf>\n")


def write_collision_enabled_object(path: Path, source: Path, scene: str) -> None:
    """Augment the free T model with obstacle links welded to world.

    Keeping the fixed links in the same model instance preserves the stock
    simulator's one-object/one-LCM-channel contract: the T remains the sole
    floating body while obstacles participate in native contact dynamics.
    """
    text = source.read_text()
    additions = []
    for name, pos, size in BOXES[scene]:
        x = pos[0] + DAIRLIB_X_OFFSET
        y = pos[1] * DAIRLIB_Y_SCALE
        z = pos[2]
        dims = " ".join(map(str, size))
        additions.append(f"""    <link name="oim_fixed_{name}">
      <pose>{x} {y} {z} 0 0 0</pose>
      <collision name="collision">
        <geometry><box><size>{dims}</size></box></geometry>
        <drake:proximity_properties xmlns:drake="uri:drake">
          <drake:mu_dynamic>0.8</drake:mu_dynamic>
          <drake:mu_static>0.8</drake:mu_static>
        </drake:proximity_properties>
      </collision>
      <visual name="visual"><geometry><box><size>{dims}</size></box></geometry>
        <material><diffuse>0.45 0.45 0.48 1</diffuse></material></visual>
    </link>
    <joint name="oim_fixed_{name}_weld" type="fixed">
      <parent>world</parent><child>oim_fixed_{name}</child>
    </joint>""")
    if "</model>" not in text:
        raise RuntimeError(f"missing </model> in {source}")
    path.write_text(text.replace("  </model>", "\n".join(additions) +
                                 "\n  </model>", 1))


def write_collision_enabled_platform(path: Path, scene: str) -> None:
    """Inject fixed OIM obstacles into the simulator-only platform URDF."""
    begin = "  <!-- OIM_IMPORTER_OBSTACLES_BEGIN -->"
    end = "  <!-- OIM_IMPORTER_OBSTACLES_END -->"
    text = re.sub(r"\n  <!-- OIM_IMPORTER_OBSTACLES_BEGIN -->.*?"
                  r"  <!-- OIM_IMPORTER_OBSTACLES_END -->\n",
                  "\n", path.read_text(), flags=re.DOTALL)
    additions = []
    # platform is welded at world z=-0.0145 m.
    for name, pos, size in BOXES[scene]:
        x = pos[0] + DAIRLIB_X_OFFSET
        y = pos[1] * DAIRLIB_Y_SCALE
        z = pos[2] + 0.0145
        dims = " ".join(map(str, size))
        link = f"oim_fixed_{name}"
        additions.append(f"""  <link name="{link}">
    <visual><geometry><box size="{dims}"/></geometry>
      <material name="oim_gray"><color rgba="0.45 0.45 0.48 1"/></material>
    </visual>
    <collision><geometry><box size="{dims}"/></geometry>
      <drake:proximity_properties>
        <drake:mu_static value="0.8"/><drake:mu_dynamic value="0.8"/>
      </drake:proximity_properties>
    </collision>
  </link>
  <joint name="{link}_weld" type="fixed">
    <parent link="platform"/><child link="{link}"/>
    <origin xyz="{x} {y} {z}" rpy="0 0 0"/>
  </joint>""")
    if "</robot>" not in text:
        raise RuntimeError(f"missing </robot> in {path}")
    if additions:
        block = begin + "\n" + "\n".join(additions) + "\n" + end
        text = text.replace("</robot>", block + "\n</robot>", 1)
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", choices=SCENES)
    parser.add_argument("--start", default="1",
                        help="pose key from the scenario's starts map")
    parser.add_argument("--goal", default="1",
                        help="pose key from the scenario's goals map")
    parser.add_argument("--reference", type=Path,
                        default=Path("/root/external/dairlib_sampling_c3"))
    args = parser.parse_args()
    root = args.reference
    pose_file = Path("external/Object-Informed-Manipulation-MJX/examples/poses") / f"{args.scene}.yaml"
    poses = yaml.safe_load(pose_file.read_text())
    if args.start not in poses["starts"]:
        parser.error(f"unknown start {args.start!r}; choices: {tuple(poses['starts'])}")
    if args.goal not in poses["goals"]:
        parser.error(f"unknown goal {args.goal!r}; choices: {tuple(poses['goals'])}")
    sx, sy, syaw = poses["starts"][args.start]
    gx, gy, gyaw = poses["goals"][args.goal]
    sx += DAIRLIB_X_OFFSET
    gx += DAIRLIB_X_OFFSET
    sy *= DAIRLIB_Y_SCALE
    gy *= DAIRLIB_Y_SCALE
    z = 0.0041315399999999995
    quat = lambda yaw: [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]

    params = root / "examples/sampling_c3/anything/parameters"
    scene_rel = f"examples/sampling_c3/urdf/oim_scenes/{args.scene}.sdf"
    write_scene(root / scene_rel, args.scene)
    base_object_rel = "examples/sampling_c3/urdf/T_shape_video/T_shape_video.sdf"
    sim_object_rel = base_object_rel
    write_collision_enabled_platform(
        root / "examples/sampling_c3/urdf/platform.urdf", args.scene)
    replace_line(params / "sim_params.yaml", "object_models",
                 repr([sim_object_rel]))
    replace_line(params / "sim_params.yaml", "q_init_objects", repr([quat(syaw) + [sx, sy, z]]))
    replace_line(params / "sampling_c3_controller_params.yaml", "base_names", "[T_shape_video]")
    replace_line(params / "sampling_c3_controller_params.yaml", "object_models",
                 "[examples/sampling_c3/urdf/T_shape_video/T_shape_video_controller.sdf]")
    replace_line(params / "vis_params.yaml", "object_vis_models",
                 "[examples/sampling_c3/urdf/T_shape_video/T_shape_video.sdf]")
    replace_line(params / "sampling_c3plus_options.yaml", "scale_lcs", "true")
    replace_line(params / "sampling_c3plus_options.yaml", "contact_model",
                 "'anitescu'")
    replace_line(params / "sampling_c3plus_options.yaml", "include_walls", "true")
    replace_line(params / "sampling_c3plus_options.yaml", "resolve_contacts_to_lists",
                 "[[0, 1, 3, 1]]")
    replace_line(params / "goal_params.yaml", "resting_object_heights", repr([z]))
    replace_line(params / "goal_params.yaml", "fixed_target_positions", repr([[gx, gy, z]]))
    replace_line(params / "goal_params.yaml", "fixed_target_orientations", repr([quat(gyaw)]))
    print(f"scene={args.scene} start_key={args.start} goal_key={args.goal} "
          f"object=T_shape_video frame_x_offset={DAIRLIB_X_OFFSET} "
          f"frame_y_scale={DAIRLIB_Y_SCALE} "
          f"start={[sx, sy, syaw]} goal={[gx, gy, gyaw]}")


if __name__ == "__main__":
    main()
