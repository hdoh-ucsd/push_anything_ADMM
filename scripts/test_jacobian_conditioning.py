"""Compare arm-Jacobian conditioning at two configurations.

Hypothesis: at the drifted-up posture (EE z = 0.86 m), the arm
Jacobian is ill-conditioned for z motion. The EE-approach cost in
the QP augments Q[:n_u, :n_u] += 2*w*J_arm^T J_arm; the QP's
effective Cartesian-direction penalty is the diagonal of J_arm J_arm^T.
If (J J^T)[2,2] is small at the drifted config, the cost has lost
authority to pull the EE back down.

Re-uses the production J_arm calculation (matches task_costs.py:415-421
exactly): CalcJacobianTranslationalVelocity, slice [:, :n_arm].
"""
import numpy as np
import yaml
import pydrake.all as ad

from sim.env_builder import build_environment, INITIAL_ARM_Q, EE_BODY_NAME


# Configurations from results/jacond_nowrap_s0.log
Q_INITIAL = np.array([+0.8145, +1.0477, +0.8022, -2.0537,
                      -0.1557, +3.0097, +0.7850])
Q_DRIFTED = np.array([+1.0756, -0.3103, +0.4228, -0.9446,
                      +0.3892, +0.7015, +0.3819])
EE_INITIAL_LOG = np.array([-0.0760, -0.0001, +0.0499])
EE_DRIFTED_LOG = np.array([-0.1078, -0.3440, +0.8557])


def calc_J_arm(plant, plant_ctx, ee_frame, world_frame, n_arm):
    """Identical to task_costs.py:415-421."""
    J_ee = plant.CalcJacobianTranslationalVelocity(
        plant_ctx, ad.JacobianWrtVariable.kV,
        ee_frame, np.zeros(3),
        world_frame, world_frame,
    )
    return J_ee[:, : n_arm]


def fk_ee(plant, plant_ctx, ee_frame, world_frame):
    return plant.CalcPointsPositions(
        plant_ctx, ee_frame, np.zeros(3), world_frame
    ).flatten()


def report_one(label, q_arm, ee_log, plant, plant_ctx, ee_frame,
               world_frame, panda_model, n_arm):
    print(f"==== {label} ====")
    print(f"q_arm = {q_arm}")

    plant.SetPositions(plant_ctx, panda_model, q_arm)

    ee_fk = fk_ee(plant, plant_ctx, ee_frame, world_frame)
    print(f"EE FK     = ({ee_fk[0]:+.4f}, {ee_fk[1]:+.4f}, {ee_fk[2]:+.4f})")
    print(f"EE log    = ({ee_log[0]:+.4f}, {ee_log[1]:+.4f}, {ee_log[2]:+.4f})")
    print(f"FK match  = {np.allclose(ee_fk, ee_log, atol=1e-3)}")
    print()

    J = calc_J_arm(plant, plant_ctx, ee_frame, world_frame, n_arm)
    print(f"J_arm shape = {J.shape}")
    np.set_printoptions(formatter={'float': lambda x: f'{x:+.4f}'},
                        linewidth=140)
    print("J_arm =")
    print(J)
    print()

    JTJ = J.T @ J
    JJT = J @ J.T

    print("J^T J (7x7):")
    print(JTJ)
    print()
    print(f"diag(J^T J) [joint-space penalties] =")
    print(f"  {np.diag(JTJ)}")
    print()

    print(f"J J^T (3x3) [Cartesian-direction coupling]:")
    print(JJT)
    print()
    print(f"diag(J J^T) [Cartesian penalty seen by QP] =")
    print(f"  x: {JJT[0,0]:+.6f}")
    print(f"  y: {JJT[1,1]:+.6f}")
    print(f"  z: {JJT[2,2]:+.6f}")
    print()

    eig_JTJ = np.sort(np.linalg.eigvalsh(JTJ))[::-1]
    eig_JJT = np.sort(np.linalg.eigvalsh(JJT))[::-1]
    print(f"Eigenvalues J^T J (7) desc: {eig_JTJ}")
    print(f"Eigenvalues J J^T (3) desc: {eig_JJT}")
    pos_eig = eig_JTJ[eig_JTJ > 1e-12]
    if len(pos_eig) >= 2:
        print(f"Condition number J^T J (max/min nonzero) = {pos_eig[0]/pos_eig[-1]:.4e}")
    pos_eig3 = eig_JJT[eig_JJT > 1e-12]
    if len(pos_eig3) >= 2:
        print(f"Condition number J J^T (max/min nonzero) = {pos_eig3[0]/pos_eig3[-1]:.4e}")
    print()
    return JJT


def main():
    with open("config/tasks.yaml") as f:
        task_cfg = yaml.safe_load(f)["tasks"]["pushing"]

    diagram, plant, panda_model, obj_model, _, _, _ = build_environment(task_cfg)
    context = diagram.CreateDefaultContext()
    plant_ctx = plant.GetMyContextFromRoot(context)

    obj_body    = plant.GetBodyByName(task_cfg["link_name"])
    ee_frame    = plant.GetFrameByName(EE_BODY_NAME)
    world_frame = plant.world_frame()

    # Park the box far away, zero arm velocities
    plant.SetFreeBodyPose(plant_ctx, obj_body,
                          ad.RigidTransform(ad.RotationMatrix(),
                                            [10.0, 10.0, 10.0]))
    plant.SetVelocities(plant_ctx, np.zeros(plant.num_velocities()))

    n_arm = 7

    JJT_init = report_one("INITIAL  (step 1, EE z=0.05)",
                          Q_INITIAL, EE_INITIAL_LOG, plant, plant_ctx,
                          ee_frame, world_frame, panda_model, n_arm)
    JJT_drift = report_one("DRIFTED  (step 132, EE z=0.86)",
                           Q_DRIFTED, EE_DRIFTED_LOG, plant, plant_ctx,
                           ee_frame, world_frame, panda_model, n_arm)

    print("==== Cartesian penalty side-by-side ====")
    print(f"           initial      drifted     drifted/initial")
    for i, axis in enumerate("xyz"):
        ratio = JJT_drift[i,i] / JJT_init[i,i] if JJT_init[i,i] != 0 else float('nan')
        print(f"  {axis}:  {JJT_init[i,i]:+.6f}  {JJT_drift[i,i]:+.6f}   ratio={ratio:.4f}")


if __name__ == "__main__":
    main()
