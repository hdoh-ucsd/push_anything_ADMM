import numpy as np
import pydrake.all as ad

# Import our custom controllers
from control.lcs_formulator import LCSFormulator
from control.admm_solver import ADMMSolver
from control.ci_mpc_c3 import C3MPC
from control.ci_mpc_c3plus import C3PlusMPC

def create_pushing_environment():
    """Builds the physics plant with the Franka arm, a table, and a pushable box."""
    builder = ad.DiagramBuilder()
    
    # Create the MultibodyPlant and SceneGraph
    plant, scene_graph = ad.AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = ad.Parser(plant)

    # 1. Add the Table (Ground)
    table_friction = ad.CoulombFriction(static_friction=0.5, dynamic_friction=0.5)
    table_shape = ad.Box(2.0, 2.0, 0.1)
    X_Table = ad.RigidTransform([0.0, 0.0, -0.05]) 
    
    plant.RegisterCollisionGeometry(
        plant.world_body(), X_Table, table_shape, "table_collision", table_friction)
    plant.RegisterVisualGeometry(
        plant.world_body(), X_Table, table_shape, "table_visual", [0.7, 0.7, 0.7, 1.0])

    # 2. Add the Franka Panda Arm
    panda_file = "package://drake_models/franka_description/urdf/panda_arm.urdf"
    panda_model = parser.AddModelsFromUrl(panda_file)[0]
    
    # Weld the base of the robot to the table
    X_RobotBase = ad.RigidTransform([0.0, -0.5, 0.0])
    plant.WeldFrames(
        plant.world_frame(), 
        plant.GetFrameByName("panda_link0", panda_model), 
        X_RobotBase)

    # 3. Add the Target Box
    box_sdf = """
    <sdf version="1.7">
      <model name="target_box">
        <link name="box_link">
          <pose>0 0 0.05 0 0 0</pose>
          <inertial>
            <mass>0.2</mass>
            <inertia><ixx>0.001</ixx><iyy>0.001</iyy><izz>0.001</izz></inertia>
          </inertial>
          <collision name="collision">
            <geometry><box><size>0.1 0.1 0.1</size></box></geometry>
            <surface><friction><ode><mu>0.4</mu><mu2>0.4</mu2></ode></friction></surface>
          </collision>
          <visual name="visual">
            <geometry><box><size>0.1 0.1 0.1</size></box></geometry>
            <material><ambient>0.8 0.1 0.1 1.0</ambient></material>
          </visual>
        </link>
      </model>
    </sdf>
    """
    box_model = parser.AddModelsFromString(box_sdf, "sdf")[0]

    plant.Finalize()

    # 4. Set up the Meshcat Visualizer
    meshcat = ad.StartMeshcat()
    ad.MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    diagram = builder.Build()
    return diagram, plant, panda_model, box_model, meshcat

def main():
    print("Building environment...")
    diagram, plant, panda_model, box_model, meshcat = create_pushing_environment()
    
    simulator = ad.Simulator(diagram)
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyContextFromRoot(context)

    # Find box coordinates
    box_body = plant.GetBodyByName("box_link")
    q_start = box_body.floating_positions_start()
    box_x_idx, box_y_idx = q_start + 4, q_start + 5
    print(f"Box X is at q[{box_x_idx}], Box Y is at q[{box_y_idx}]")

    # Initial Robot Pose
    initial_q_robot = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
    plant.SetPositions(plant_context, panda_model, initial_q_robot)
    
    # Place Box
    X_BoxStart = ad.RigidTransform(ad.RotationMatrix(), [0.0, 0.0, 0.05])
    plant.SetFreeBodyPose(plant_context, box_body, X_BoxStart)

    n_q = plant.num_positions()
    n_v = plant.num_velocities()
    n_u = plant.num_actuators()

    formulator = LCSFormulator(plant)
    admm_solver = ADMMSolver(n_v=n_v)

    # ---> SPEED FIX 1: Shorter horizon, fewer samples <---
    mpc = C3PlusMPC(n_u, n_v, n_q, formulator, admm_solver, horizon=5, dt=0.01, num_samples=10)

    # Target: Push +0.3m on X-axis
    target_q = np.zeros(n_q)
    target_q[box_x_idx] = 0.3
    target_q[box_y_idx] = 0.0

    # Main Simulation Loop
    sim_time = 0.0
    dt_mpc = 0.01  
    max_time = 5.0  # Increased to give it time to hit the box
    
    print(f"Simulation running at {meshcat.web_url()}")
    
    step_count = 0 
    while sim_time < max_time:
        current_q = plant.GetPositions(plant_context)
        current_v = plant.GetVelocities(plant_context)
        
        # ---> SPEED FIX 2: Gravity Compensation <---
        tau_g = plant.CalcGravityGeneralizedForces(plant_context)
        
        u_opt = mpc.compute_control(plant_context, current_q, current_v, target_q)
        
        # Monitor progress
        if step_count % 10 == 0:  
            box_x = current_q[box_x_idx]
            torque_magnitude = np.linalg.norm(u_opt)
            print(f"Time: {sim_time:.2f}s | Extra Torque: {torque_magnitude:.2f} Nm | Box X: {box_x:.4f}")

        # ---> SPEED FIX 3: Combine Gravity with MPC Torque <---
        total_torque = tau_g[:n_u] + u_opt
        plant.get_actuation_input_port().FixValue(plant_context, total_torque)
        
        sim_time += dt_mpc
        step_count += 1
        simulator.AdvanceTo(sim_time)
        
    print("Simulation finished!")

if __name__ == "__main__":
    main()