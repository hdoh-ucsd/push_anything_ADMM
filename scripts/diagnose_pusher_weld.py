"""Diagnostic: verify pusher is welded to panda_link8 and contact filter is tight."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import numpy as np
from pydrake.multibody.tree import BodyIndex, JointIndex
from sim.env_builder import build_environment, EE_BODY_NAME
from control.lcs_formulator import LCSFormulator

with open(Path(__file__).resolve().parent.parent / "config" / "tasks.yaml") as f:
    task_cfg = yaml.safe_load(f)["tasks"]["pushing"]

diagram, plant, _, object_model, _ = build_environment(task_cfg)
ctx = diagram.CreateDefaultContext()
plant_ctx = plant.GetMyMutableContextFromRoot(ctx)

link_name = task_cfg["link_name"]
obj_body = plant.GetBodyByName(link_name)

print("=" * 70)
print("STEP 1: Plant topology")
print("=" * 70)

print(f"\nTotal bodies: {plant.num_bodies()}")
print(f"Total joints: {plant.num_joints()}")
print(f"n_q={plant.num_positions()}  n_v={plant.num_velocities()}  n_u={plant.num_actuators()}")

print("\n--- All bodies ---")
for i in range(plant.num_bodies()):
    body = plant.get_body(BodyIndex(i))
    is_float = body.is_floating()
    model_name = plant.GetModelInstanceName(body.model_instance())
    marker = "  <-- FLOATING" if is_float else ""
    print(f"  body {i:2d}: {body.name():<25} (model: {model_name}){marker}")

print("\n--- All joints ---")
for i in range(plant.num_joints()):
    joint = plant.get_joint(JointIndex(i))
    parent = joint.parent_body().name()
    child  = joint.child_body().name()
    jtype  = type(joint).__name__
    print(f"  joint {i:2d}: {joint.name():<30}  {jtype:<20}  "
          f"parent={parent:<15} child={child}")

print("\n" + "=" * 70)
print("STEP 2: Pusher-specific check")
print("=" * 70)

try:
    pusher = plant.GetBodyByName(EE_BODY_NAME)
    print(f"\nPusher body found: {pusher.name()}")
    print(f"  is_floating: {pusher.is_floating()}")
    print(f"  model instance: {plant.GetModelInstanceName(pusher.model_instance())}")

    # World pose at default state
    pos = plant.EvalBodyPoseInWorld(plant_ctx, pusher).translation()
    print(f"  world position at default state: {pos}")

    # Find any joint connecting pusher
    pusher_joint = None
    for i in range(plant.num_joints()):
        j = plant.get_joint(JointIndex(i))
        if j.child_body().name() == EE_BODY_NAME or j.parent_body().name() == EE_BODY_NAME:
            pusher_joint = j
            break

    if pusher_joint:
        print(f"\n  Pusher joint: {pusher_joint.name()}")
        print(f"  Joint type: {type(pusher_joint).__name__}")
        print(f"  Parent: {pusher_joint.parent_body().name()}")
        print(f"  Child:  {pusher_joint.child_body().name()}")
        is_weld = "WeldJoint" in type(pusher_joint).__name__
        print(f"  Is rigid weld: {is_weld}  {'(GOOD)' if is_weld else '(BUG - should be WeldJoint)'}")
    else:
        print("\n  NO JOINT FOUND connecting pusher to anything!")
        print("  This means pusher is a disconnected floating body (BUG)")

except RuntimeError as e:
    print(f"ERROR: Could not find body named '{EE_BODY_NAME}'")
    print(f"  {e}")

print("\n" + "=" * 70)
print("STEP 3: Collision geometry inventory")
print("=" * 70)

scene_graph = None
for sys_ in diagram.GetSystems():
    if "SceneGraph" in type(sys_).__name__:
        scene_graph = sys_
        break

if scene_graph is None:
    print("Could not locate SceneGraph in diagram")
else:
    inspector = scene_graph.model_inspector()
    print(f"\nTotal collision geometries in scene: varies by query")
    print("\nBodies with collision geometries:")
    for i in range(plant.num_bodies()):
        body = plant.get_body(BodyIndex(i))
        gids = plant.GetCollisionGeometriesForBody(body)
        if len(gids) > 0:
            print(f"  {body.name():<25} has {len(gids):2d} collision geom(s): "
                  f"{[int(str(g).split('=')[-1].rstrip('>')) for g in gids]}")

print("\n" + "=" * 70)
print("STEP 4: Contact filter behavior at t=0")
print("=" * 70)

mu = task_cfg.get("friction", 0.5)
formulator = LCSFormulator(plant, mu=mu, obj_body=obj_body)

# Extract contact pairs at default state
phi, J_n, J_t, mu_out = formulator.extract_lcs_contacts(plant_ctx, distance_threshold=0.10)

print(f"\nAt default state with threshold=0.10m:")
print(f"  Number of filtered contact pairs: {len(phi)}")
print(f"  phi values: {phi}")

if hasattr(formulator, '_last_contact_info'):
    for i, info in enumerate(formulator._last_contact_info):
        print(f"\n  Contact {i}:")
        print(f"    body_A: {info['body_A']}")
        print(f"    body_B: {info['body_B']}")
        print(f"    distance: {info['distance']:.5f} m")

        valid = (info['body_A'] in ('pusher', 'box_link') and
                 info['body_B'] in ('pusher', 'box_link'))
        print(f"    Valid pusher<->box pair: {valid}  {'(GOOD)' if valid else '(BUG - arm link leaked through filter)'}")

print("\n" + "=" * 70)
print("STEP 5: Unfiltered contact check (sanity)")
print("=" * 70)

# Manually run signed-distance query WITHOUT the filter to see what pairs
# exist in the raw scene
query_obj = plant.get_geometry_query_input_port().Eval(plant_ctx)
inspector = query_obj.inspector()
raw_pairs = query_obj.ComputeSignedDistancePairwiseClosestPoints(0.10)

print(f"\nTotal raw contact pairs within 10cm (before filter): {len(raw_pairs)}")

# Count by body-pair
from collections import Counter
pair_counts = Counter()
for sdp in raw_pairs:
    body_A = plant.GetBodyFromFrameId(inspector.GetFrameId(sdp.id_A)).name()
    body_B = plant.GetBodyFromFrameId(inspector.GetFrameId(sdp.id_B)).name()
    key = tuple(sorted([body_A, body_B]))
    pair_counts[key] += 1

print("\n  Raw contact pairs by body (before filter):")
for (a, b), count in pair_counts.most_common():
    marker = ""
    if "pusher" in (a, b) and ("box_link" in (a, b) or "box" in (a, b)):
        marker = "  <-- intended contact"
    elif any("panda_link" in x for x in (a, b)) and ("box_link" in (a, b) or "box" in (a, b)):
        marker = "  <-- ARM LEAK: link touching box"
    print(f"    {a:<20} <-> {b:<20}  count={count}{marker}")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
