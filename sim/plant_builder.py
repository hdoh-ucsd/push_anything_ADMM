import os
import pydrake.all as ad

def build_pushed_world(builder, meshcat, time_step=0.001):
    plant, scene_graph = ad.AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    parser = ad.Parser(plant)
    
    current_dir = os.path.dirname(__file__)

    # --- 1. Load the Franka Arm ---
    # Register your newly downloaded local models folder
    franka_dir = os.path.abspath(os.path.join(current_dir, "../models/drake_models/franka_description"))
    parser.package_map().Add("franka_description", franka_dir)

    # Use the new package name to load the URDF
    franka_url = "package://franka_description/urdf/panda_arm.urdf"
    franka_model = parser.AddModels(url=franka_url)[0]
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("panda_link0"))

    # --- 2. Environment Setup ---
    # Ground Plane
    friction = ad.CoulombFriction(static_friction=0.5, dynamic_friction=0.5)
    plant.RegisterCollisionGeometry(
        plant.world_body(), ad.RigidTransform(), ad.HalfSpace(), "ground_col", friction)
    plant.RegisterVisualGeometry(
        plant.world_body(), ad.RigidTransform(), ad.HalfSpace(), "ground_vis", [0.4, 0.4, 0.4, 1.0])

    # --- 3. Load the Box ---
    sdf_path = os.path.abspath(os.path.join(current_dir, "../models/objects/push_box.sdf"))
    box_model = parser.AddModels(sdf_path)[0]

    # --- 4. Finalization ---
    plant.Finalize()
    ad.AddDefaultVisualization(builder=builder, meshcat=meshcat)

    return plant, box_model, franka_model