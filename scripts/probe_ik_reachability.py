"""Static IK + FK characterization across a workspace sweep.

For each test target, solve IK and check FK accuracy at q_des.
Pure kinematic check; no simulation.
"""
import numpy as np
import yaml
from sim.env_builder import build_environment, INITIAL_ARM_Q
from control.sampling_c3.ik import solve_ik_to_ee_pos


def main():
    with open("config/tasks.yaml") as f:
        task_cfg = yaml.safe_load(f)["tasks"]["pushing"]

    diagram, plant, panda_model, object_model, _, _, _ = build_environment(task_cfg)
    plant_ctx = plant.CreateDefaultContext()

    pusher_frame = plant.GetBodyByName("pusher").body_frame()
    world_frame = plant.world_frame()
    n_arm = 7

    targets = [
        ("home_ref",       ( 0.000,  0.000, 0.200)),
        ("touch_back",     (-0.075,  0.000, 0.050)),
        ("p_repos_5cm",    (-0.130,  0.000, 0.050)),
        ("10cm_behind",    (-0.175,  0.000, 0.050)),
        ("offaxis_pos_y",  (-0.130, +0.100, 0.050)),
        ("offaxis_neg_y",  (-0.130, -0.100, 0.050)),
        ("high_repos",     (-0.130,  0.000, 0.100)),
        ("low_repos",      (-0.130,  0.000, 0.025)),
    ]

    # Build q_init: arm + (object's floating-body 7 DOF: qw,qx,qy,qz,x,y,z)
    n_q = plant.num_positions()
    q_init_full = np.zeros(n_q)
    q_init_full[:n_arm] = INITIAL_ARM_Q
    # Object: unit quaternion + init xyz
    q_init_full[n_arm]   = 1.0  # qw
    q_init_full[n_arm+1] = 0.0
    q_init_full[n_arm+2] = 0.0
    q_init_full[n_arm+3] = 0.0
    q_init_full[n_arm+4:n_arm+7] = task_cfg["init_xyz"]

    print(f"{'label':<16} {'p_des (m)':<32} {'IK':<5} {'iters':<6} "
          f"{'FK err (mm)':<12} {'q_des (rad)':<60}")
    print("-" * 145)

    for label, p_des in targets:
        p_des_arr = np.array(p_des)
        try:
            q_sol, err_norm, iters = solve_ik_to_ee_pos(
                plant=plant,
                ee_frame=pusher_frame,
                p_target=p_des_arr,
                q_init=q_init_full,
                plant_ctx=plant_ctx,
                n_arm_dofs=n_arm,
                max_iter=60,
                tol=1e-4,
            )
        except Exception as e:
            print(f"{label:<16} {str(p_des):<32} {'ERR':<5} {'-':<6} "
                  f"{'-':<12} (IK exception: {e})")
            continue

        feasible = (err_norm < 5e-3)  # 5mm tol for "feasible"
        # FK check at q_sol
        plant.SetPositions(plant_ctx, q_sol)
        p_at_q = plant.CalcPointsPositions(
            plant_ctx, pusher_frame, np.zeros(3), world_frame,
        ).flatten()
        fk_err_mm = float(np.linalg.norm(p_at_q - p_des_arr) * 1000.0)
        q_str = "[" + ",".join(f"{x:+.2f}" for x in q_sol[:n_arm]) + "]"
        flag = "YES" if feasible else "NO"
        print(f"{label:<16} {str(p_des):<32} {flag:<5} {iters:<6} "
              f"{fk_err_mm:>7.3f}     {q_str:<60}")


if __name__ == "__main__":
    main()
