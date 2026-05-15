"""One-target diagnostic: run home_ref and print q_target vs q_now over time."""
import numpy as np
import yaml
import pydrake.all as ad

from sim.env_builder import build_environment, INITIAL_ARM_Q, EE_BODY_NAME
from control.sampling_c3.reposition_ik import RepositionIKTracker
from control.sampling_c3.params import SamplingC3Params


def main():
    with open("config/tasks.yaml") as f:
        task_cfg = yaml.safe_load(f)["tasks"]["pushing"]
    params = SamplingC3Params.from_yaml("config/sampling_c3_kik.yaml")

    diagram, plant, panda_model, _, _, _, _ = build_environment(task_cfg)
    simulator = ad.Simulator(diagram)
    context = simulator.get_mutable_context()
    plant_ctx = plant.GetMyContextFromRoot(context)

    obj_body = plant.GetBodyByName(task_cfg["link_name"])
    ee_frame = plant.GetFrameByName(EE_BODY_NAME)
    world_frame = plant.world_frame()

    # Park box, set arm
    plant.SetFreeBodyPose(plant_ctx, obj_body,
                          ad.RigidTransform(ad.RotationMatrix(), [10.0, 10.0, 10.0]))
    plant.SetPositions(plant_ctx, panda_model, INITIAL_ARM_Q)
    plant.SetVelocities(plant_ctx, np.zeros(plant.num_velocities()))

    _sgs = [s for s in diagram.GetSystems() if isinstance(s, ad.SceneGraph)]
    scene_graph = _sgs[0]

    tracker = RepositionIKTracker(
        plant=plant, ee_frame=ee_frame, obj_body=obj_body,
        n_arm_dofs=7, horizon=20, dt=0.05,
        repos_params=params.reposition_params,
        ik_params=params.repos_ik_params,
        diagram=diagram, scene_graph=scene_graph,
    )
    tracker.reset()

    # Compute target = FK at INITIAL_ARM_Q
    plant.SetPositions(plant_ctx, panda_model, INITIAL_ARM_Q)
    ee_init = plant.CalcPointsPositions(
        plant_ctx, ee_frame, np.zeros(3), world_frame
    ).flatten()
    print(f"[DIAG] initial EE = {ee_init}")
    p_target = ee_init.copy()
    print(f"[DIAG] p_target = {p_target}")

    dt_ctrl = 0.01
    simulator.Initialize()

    for step in range(401):
        current_q = plant.GetPositions(plant_ctx)
        current_v = plant.GetVelocities(plant_ctx)
        u, diag = tracker.compute_torque(
            current_q=current_q, current_v=current_v,
            plant_ctx=plant_ctx, p_target=p_target, dt_ctrl=dt_ctrl,
        )
        q_arm_now = current_q[:7]
        q_arm_target = tracker.last_q_knots[:, 0]
        ee_now = plant.CalcPointsPositions(
            plant_ctx, ee_frame, np.zeros(3), world_frame
        ).flatten()
        # FK at q_arm_target (use a temporary context to avoid disturbing sim)
        ee_at_target = tracker.last_ee_knots[:, 0]
        feasible0 = tracker.last_feasible[0] if tracker.last_feasible else None
        if step % 50 == 0:
            q_err = q_arm_target - q_arm_now
            print(f"[DIAG] step={step:3d} t={step*dt_ctrl:.2f}s "
                  f"ee=({ee_now[0]:+.3f},{ee_now[1]:+.3f},{ee_now[2]:+.3f}) "
                  f"ee@qtgt=({ee_at_target[0]:+.3f},{ee_at_target[1]:+.3f},{ee_at_target[2]:+.3f}) "
                  f"feas={feasible0} "
                  f"|u|={np.linalg.norm(u):.2f} "
                  f"|q_err|={np.linalg.norm(q_err):.3f} "
                  f"q_err_max={np.max(np.abs(q_err)):.3f} (j{int(np.argmax(np.abs(q_err)))})")
        plant.get_actuation_input_port().FixValue(plant_ctx, u)
        simulator.AdvanceTo((step + 1) * dt_ctrl)


if __name__ == "__main__":
    main()
