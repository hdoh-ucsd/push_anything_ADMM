#!/usr/bin/env python3
"""Import reference anything objects as port tasks (Fig 8 campaign, 2026-08-15).

For each object in OBJECTS:
  1. copy the reference asset dir -> sim/models/<name>/
  2. parse <name>.obj          -> mesh min_z / max_z
  3. parse <name>.sdf          -> sim link name + mass
  4. parse <name>_controller.sdf -> controller mass + the 3 ground-contact
     sphere positions (the object's reference ground-witness triangle)
  5. append a task block to config/tasks.yaml (idempotent: skipped if the
     task key already exists)

Per-object value derivations (multiyaml_rewrite.py, port frame = reference
frame shifted so the table top is z=0, i.e. +0.029):
  init_z            = -mesh_min_z                  (rest height)
  pwl_waypoint_height = (max_z - min_z) + 0.05     (ref: -0.029+(max-min)+0.05)
  sampling_height   = 0.031                        (ref sampling z_height 0.002)
  cost.z_ee_target  = sampling_height
  cost.z_obj_target = init_z
Cost block = the anything-N1 tables (object-independent). Goal = the
T-canonical fixed displacement for every object
(port fixed-goal deviation, documented): spawn (0.5, 0) -> goal
(0.482, 0.187), yaw -0.7379.
"""
import os
import re
import shutil
import sys

REF = "/root/reference_repos/dairlib_sampling_c3/examples/sampling_c3/urdf"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASKS_YAML = os.path.join(REPO, "config", "tasks.yaml")

# display name -> reference dir
OBJECTS = {
    "Letter I": "I_shape_texture",
    "Letter C": "C_shape_texture",
    "Letter R": "R_shape_texture",
    "Letter A": "A_shape_video",
    "Letter Y": "Y_shape_video",
    "Letter G": "G_shape_video",
    "Letter B": "B_shape_video",
    "Letter 3": "3_shape_video",
    "Letter H": "H_shape_texture",
    "Letter E": "E_shape_video",
    "Expo Box": "expo_box",
    "Lotion": "lotion",
    "Wood Block": "wood_block",
    "Tape": "tape",
    "Eraser": "eraser",
    "Milk Bottle": "milk",
    "Clamp": "clamp",
    "Chicken Broth": "chicken_broth",
    "Egg Carton": "egg_carton",
    "Book": "book",
    "Baby Toy": "baby_toy",
    "Gallon Milk": "gallon_milk",
    "Xbox": "xbox",
}


def obj_z_extents(path):
    zmin, zmax = float("inf"), float("-inf")
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                z = float(parts[3])
                zmin, zmax = min(zmin, z), max(zmax, z)
    return zmin, zmax


def parse_sdf_link_mass(path):
    s = open(path).read()
    link = re.search(r'<link\s+name="([^"]+)"', s)
    mass = re.search(r"<mass>\s*([\d.eE+-]+)\s*</mass>", s)
    return (link.group(1) if link else None,
            float(mass.group(1)) if mass else None)


def parse_controller_spheres(path):
    """Return (controller_mass, [(x,y,z)]*3) — the last 3 sphere collisions."""
    s = open(path).read()
    mass = re.search(r"<mass>\s*([\d.eE+-]+)\s*</mass>", s)
    # Collision blocks containing a sphere geometry. The <pose> element's
    # position within the block varies across reference assets (expo_box:
    # pose before geometry; letters: pose after), so scan whole blocks.
    pts = []
    for block in re.findall(r"<collision\b.*?</collision>", s, re.DOTALL):
        if "<sphere>" not in block:
            continue
        p = re.search(r"<pose>\s*([-\d.eE ]+?)\s*</pose>", block)
        if p:
            vals = [float(v) for v in p.group(1).split()]
            pts.append(tuple(vals[:3]))
    return (float(mass.group(1)) if mass else None), pts[-3:]


COST_BLOCK = """    cost:
      use_reference_q_vector: true
      w_Q: 80.0
      w_Q_position: 50.0
      q_vector_ee_pos: [0.01, 0.01, 0.01]
      q_vector_obj_quat: [0.1, 0.1, 0.1, 0.1]
      q_vector_obj_pos: [150.0, 150.0, 120.0]
      q_vector_position_obj_quat: [0.0, 0.0, 0.0, 0.0]
      q_vector_position_obj_pos: [200.0, 200.0, 120.0]
      q_vector_ee_vel: [15.0, 15.0, 10.0]
      q_vector_obj_ang_vel: [0.05, 0.05, 0.05]
      q_vector_obj_lin_vel: [0.05, 0.05, 0.05]
      w_R: 6.0
      r_vector: [0.01, 0.01, 1.0]
      w_obj_xy: 12500.0
      w_obj_z: 10.0
      w_box_z: 12490.0
      ee_target_z_offset_above_object: 0.06
      w_obj_xy_pose: 10000.0
      w_box_z_pose: 5990.0
      w_box_rp: 5.0
      w_yaw: 0.0
      use_quaternion_dependent_cost: true
      q_quaternion_dependent_weight: 1000.0
      q_quaternion_dependent_regularizer_fraction: 0.0
      w_terminal: 1.0
      z_ee_target: {z_ee:.4f}
      z_obj_target: {z_obj:.4f}
      d_push: 0.05
      w_ee_approach: 8000.0
      w_torque: 0.01
"""


def task_block(name, link, mass, cmass, zmin, zmax, witnesses):
    init_z = -zmin
    pwl = (zmax - zmin) + 0.05
    w = "".join(
        f"      - [{x:.6f}, {y:.6f}, {z:.6f}]\n" for x, y, z in witnesses)
    return (
        f"  {name}:\n"
        f"    # AUTO-IMPORTED anything object (scripts/import_anything_objects.py,\n"
        f"    # Fig 8 campaign 2026-08-15). mesh z: [{zmin:.4f}, {zmax:.4f}].\n"
        f"    object_type: tshape\n"
        f"    link_name: {link}\n"
        f"    mass: {mass}\n"
        f"    friction: 0.3\n"
        f"    pusher_friction: 1.0\n"
        f"    mu_per_pair_type: {{EE-BOX: 0.42, BOX-GND: 0.46, EE-GND: 0.823}}\n"
        f"    q_init_franka: [2.191, 1.1, -1.33, -2.22, 1.30, 2.02, 0.08]\n"
        f"    init_xyz: [0.5, 0.0, {init_z:.4f}]\n"
        f"    goal_xy: [0.482, 0.187]\n"
        f"    goal_yaw: -0.7379\n"
        f"    color_rgba: [0.75, 0.75, 0.75, 1.0]\n"
        f"    sampling_height: 0.031\n"
        f"    pwl_waypoint_height: {pwl:.4f}\n"
        f"    ground_witness_points_body:\n{w}"
        + COST_BLOCK.format(z_ee=0.031, z_obj=init_z)
        + f"    object_sdf: sim/models/{name}/{name}.sdf\n"
        f"    controller_mass: {cmass}\n"
    )


def main():
    existing = open(TASKS_YAML).read()
    blocks = []
    report = []
    for disp, name in OBJECTS.items():
        src = os.path.join(REF, name)
        dst = os.path.join(REPO, "sim", "models", name)
        if not os.path.isdir(src):
            print(f"MISSING ASSET DIR: {name}"); sys.exit(1)
        if not os.path.isdir(dst):
            shutil.copytree(src, dst)
        zmin, zmax = obj_z_extents(os.path.join(dst, f"{name}.obj"))
        link, mass = parse_sdf_link_mass(os.path.join(dst, f"{name}.sdf"))
        cmass, wit = parse_controller_spheres(
            os.path.join(dst, f"{name}_controller.sdf"))
        if link is None or mass is None or cmass is None or len(wit) != 3:
            print(f"PARSE FAILURE {name}: link={link} mass={mass} "
                  f"cmass={cmass} witnesses={len(wit)}"); sys.exit(1)
        report.append(
            f"{disp:14s} {name:18s} link={link:18s} m={mass:<7} cm={cmass:<6} "
            f"z=[{zmin:+.4f},{zmax:+.4f}] init_z={-zmin:.4f} "
            f"pwl={(zmax-zmin)+0.05:.4f} wit0={wit[0]}")
        if f"\n  {name}:\n" in existing:
            continue
        blocks.append(task_block(name, link, mass, cmass, zmin, zmax, wit))
    print("\n".join(report))
    if blocks:
        with open(TASKS_YAML, "a") as f:
            f.write("\n  # ---- AUTO-IMPORTED anything objects "
                    "(Fig 8 campaign 2026-08-15) ----\n")
            f.write("\n".join(blocks))
        print(f"\nAppended {len(blocks)} task blocks to {TASKS_YAML}")
    else:
        print("\nAll task blocks already present.")


if __name__ == "__main__":
    main()
