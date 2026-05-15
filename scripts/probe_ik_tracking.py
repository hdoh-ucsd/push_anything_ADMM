"""Dynamic tracking probe — for each static-IK-feasible target,
simulate the v3 RepositionIKTracker driving the arm from INITIAL_ARM_Q
to p_target. Box is parked far away so contact dynamics don't fire.

Runs one target per Python invocation (per stop-condition fallback),
sub-selected by the LABEL env var.
"""
import os
import sys
import numpy as np
import yaml
import pydrake.all as ad

from sim.env_builder import build_environment, INITIAL_ARM_Q, EE_BODY_NAME
from control.sampling_c3.reposition_ik import RepositionIKTracker
from control.sampling_c3.params import SamplingC3Params
from control.sampling_c3.ik import solve_ik_to_ee_pos


TARGETS = [
    ("home_ref",       np.array([ 0.000,  0.000, 0.200])),
    ("touch_back",     np.array([-0.075,  0.000, 0.050])),
    ("p_repos_5cm",    np.array([-0.130,  0.000, 0.050])),
    ("10cm_behind",    np.array([-0.175,  0.000, 0.050])),
    ("offaxis_pos_y",  np.array([-0.130, +0.100, 0.050])),
    ("offaxis_neg_y",  np.array([-0.130, -0.100, 0.050])),
    ("high_repos",     np.array([-0.130,  0.000, 0.100])),
    ("low_repos",      np.array([-0.130,  0.000, 0.025])),
]


def run_one(label, p_des, params):
    with open("config/tasks.yaml") as f:
        task_cfg = yaml.safe_load(f)["tasks"]["pushing"]

    diagram, plant, panda_model, object_model, _, _, _ = build_environment(task_cfg)
    simulator = ad.Simulator(diagram)
    context = simulator.get_mutable_context()
    plant_ctx = plant.GetMyContextFromRoot(context)

    obj_body = plant.GetBodyByName(task_cfg["link_name"])
    ee_frame = plant.GetFrameByName(EE_BODY_NAME)
    world_frame = plant.world_frame()

    n_arm = 7
    n_q = plant.num_positions()
    n_v = plant.num_velocities()

    # Park the box FAR away so EE collisions with it are impossible
    plant.SetFreeBodyPose(
        plant_ctx, obj_body,
        ad.RigidTransform(ad.RotationMatrix(), [10.0, 10.0, 10.0]),
    )
    plant.SetPositions(plant_ctx, panda_model, INITIAL_ARM_Q)
    plant.SetVelocities(plant_ctx, np.zeros(n_v))

    # Resolve scene_graph
    _sgs = [s for s in diagram.GetSystems() if isinstance(s, ad.SceneGraph)]
    scene_graph = _sgs[0]

    tracker = RepositionIKTracker(
        plant=plant, ee_frame=ee_frame, obj_body=obj_body,
        n_arm_dofs=n_arm,
        horizon=20,
        dt=0.05,
        repos_params=params.reposition_params,
        ik_params=params.repos_ik_params,
        diagram=diagram,
        scene_graph=scene_graph,
    )
    tracker.reset()

    dt_ctrl = 0.01
    sim_duration = 4.0
    simulator.Initialize()

    t = 0.0
    diverged = False
    while t < sim_duration:
        current_q = plant.GetPositions(plant_ctx)
        current_v = plant.GetVelocities(plant_ctx)
        u, _diag = tracker.compute_torque(
            current_q=current_q, current_v=current_v,
            plant_ctx=plant_ctx, p_target=p_des, dt_ctrl=dt_ctrl,
        )
        plant.get_actuation_input_port().FixValue(plant_ctx, u)
        t += dt_ctrl
        try:
            simulator.AdvanceTo(t)
        except Exception as e:
            diverged = True
            break
        # Cheap divergence guard
        ee_now = plant.CalcPointsPositions(
            plant_ctx, ee_frame, np.zeros(3), world_frame
        ).flatten()
        if np.linalg.norm(ee_now) > 1.5:
            diverged = True
            break

    if diverged:
        return label, p_des, None, None, None, "DIVERGED"

    # Final EE
    ee_now = plant.CalcPointsPositions(
        plant_ctx, ee_frame, np.zeros(3), world_frame
    ).flatten()
    ee_gap_mm = float(np.linalg.norm(ee_now - p_des) * 1000.0)
    q_settled = plant.GetPositions(plant_ctx)[:n_arm]

    # Static IK solution for comparison
    static_ctx = plant.CreateDefaultContext()
    q_init_full = np.zeros(n_q)
    q_init_full[:n_arm] = INITIAL_ARM_Q
    q_init_full[n_arm]  = 1.0
    q_init_full[n_arm+4:n_arm+7] = task_cfg["init_xyz"]
    q_ik, err, _it = solve_ik_to_ee_pos(
        plant=plant, ee_frame=ee_frame, p_target=p_des,
        q_init=q_init_full, plant_ctx=static_ctx, n_arm_dofs=n_arm,
        max_iter=60, tol=1e-4,
    )
    q_gap = float(np.linalg.norm(q_settled - q_ik[:n_arm]))

    return label, p_des, ee_now, ee_gap_mm, q_gap, "OK"


def main():
    sel = os.environ.get("LABEL", None)

    params = SamplingC3Params.from_yaml("config/sampling_c3_kik.yaml")

    if sel is None:
        # All targets — printed header
        print(f"{'label':<16} {'p_des':<32} {'EE_settled':<28} "
              f"{'EE gap (mm)':<13} {'q gap (rad)':<12} {'status'}")
        print("-" * 110)
        for label, p_des in TARGETS:
            label, p_des, ee_now, ee_gap_mm, q_gap, status = run_one(label, p_des, params)
            if status == "OK":
                ee_str = f"({ee_now[0]:+.3f},{ee_now[1]:+.3f},{ee_now[2]:+.3f})"
                print(f"{label:<16} {str(tuple(p_des)):<32} {ee_str:<28} "
                      f"{ee_gap_mm:>8.1f}     {q_gap:>7.3f}     {status}")
            else:
                print(f"{label:<16} {str(tuple(p_des)):<32} "
                      f"{'-':<28} {'-':<13} {'-':<12} {status}")
    else:
        for label, p_des in TARGETS:
            if label != sel:
                continue
            label, p_des, ee_now, ee_gap_mm, q_gap, status = run_one(label, p_des, params)
            if status == "OK":
                ee_str = f"({ee_now[0]:+.3f},{ee_now[1]:+.3f},{ee_now[2]:+.3f})"
                print(f"RESULT {label} p_des={tuple(p_des)} ee={ee_str} "
                      f"gap_mm={ee_gap_mm:.1f} q_gap={q_gap:.3f} status={status}")
            else:
                print(f"RESULT {label} p_des={tuple(p_des)} status={status}")


if __name__ == "__main__":
    main()
