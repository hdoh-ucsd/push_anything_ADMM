"""Pure-regulation probe: can joint-PD-with-gravity-comp hold INITIAL_ARM_Q?

No IK, no planner, no wrapper. Command q_target = INITIAL_ARM_Q for 5 s
and watch what the joint-PD law does. Reproduces the exact control law
from RepositionIKTracker.compute_torque (reposition_ik.py:1229-1259).

Logs per step (every PRINT_EVERY steps):
  t  ee_xyz  q_err_norm  tau_P  tau_I  tau_D  tau_g  tau_total  clipped_mask
"""
import numpy as np
import yaml
import pydrake.all as ad

from sim.env_builder import build_environment, EE_BODY_NAME
from sim.env_builder import _INITIAL_ARM_Q_SEED as INITIAL_ARM_Q  # IK seed, not the production start pose (7ff5a21)
from control.sampling_c3.params import SamplingC3Params


PRINT_EVERY = 50          # 50 × dt_ctrl(0.01s) = 0.5 s between log rows
SIM_DURATION = 5.0


def fmt_v(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:+6.2f}" for x in v) + "]"


def main():
    with open("config/tasks.yaml") as f:
        task_cfg = yaml.safe_load(f)["tasks"]["pushing"]
    params = SamplingC3Params.from_yaml("config/sampling_c3_kik.yaml")
    rp = params.reposition_params

    Kp, Kd, Ki   = rp.Kp_q, rp.Kd_q, rp.Ki_q
    I_max        = rp.I_max
    torque_limit = rp.torque_limit
    print(f"[CFG] Kp={Kp} Kd={Kd} Ki={Ki} I_max={I_max} torque_limit={torque_limit}")

    diagram, plant, panda_model, obj_model, _, _, _ = build_environment(task_cfg)
    simulator = ad.Simulator(diagram)
    context   = simulator.get_mutable_context()
    plant_ctx = plant.GetMyContextFromRoot(context)

    obj_body    = plant.GetBodyByName(task_cfg["link_name"])
    ee_frame    = plant.GetFrameByName(EE_BODY_NAME)
    world_frame = plant.world_frame()

    # Park box far away so contact dynamics don't fire
    plant.SetFreeBodyPose(plant_ctx, obj_body,
                          ad.RigidTransform(ad.RotationMatrix(), [10.0, 10.0, 10.0]))
    plant.SetPositions(plant_ctx, panda_model, INITIAL_ARM_Q)
    plant.SetVelocities(plant_ctx, np.zeros(plant.num_velocities()))

    n_arm = 7
    q_target = INITIAL_ARM_Q.copy()

    # Auxiliary context for gravity-comp evaluation at q_target (mirrors
    # reposition_ik.py:1253-1256 — separate context so we don't disturb sim).
    aux_ctx = plant.CreateDefaultContext()

    # FK at INITIAL_ARM_Q (start of run + reference for drift)
    plant.SetPositions(plant_ctx, panda_model, INITIAL_ARM_Q)
    ee_start = plant.CalcPointsPositions(
        plant_ctx, ee_frame, np.zeros(3), world_frame
    ).flatten()
    print(f"[INIT] q_target  = {fmt_v(q_target)}")
    print(f"[INIT] ee_start  = ({ee_start[0]:+.4f}, {ee_start[1]:+.4f}, {ee_start[2]:+.4f})")

    integral = np.zeros(n_arm)
    dt_ctrl  = 0.01
    n_steps  = int(round(SIM_DURATION / dt_ctrl))
    simulator.Initialize()

    print(f"\n{'t':>6} {'ee_x':>7} {'ee_y':>7} {'ee_z':>7} {'|q_err|':>8} "
          f"{'|tP|':>7} {'|tI|':>7} {'|tD|':>7} {'|tg|':>7} {'|ttot|':>7} "
          f"{'clipped':>15}")
    print("-" * 110)

    final_row = None
    for step in range(n_steps + 1):
        current_q = plant.GetPositions(plant_ctx)
        current_v = plant.GetVelocities(plant_ctx)
        q_arm = current_q[:n_arm]
        v_arm = current_v[:n_arm]

        # --- Control law (identical to reposition_ik.py:1229-1259) ---
        q_err = q_target - q_arm
        integral += q_err * dt_ctrl
        np.clip(integral, -I_max, I_max, out=integral)

        tau_P = Kp * q_err
        tau_I = Ki * integral
        tau_D = -Kd * v_arm

        # Gravity comp at q_target (Step 8 Fix 4)
        q_full_target = current_q.copy()
        q_full_target[:n_arm] = q_target
        plant.SetPositions(aux_ctx, q_full_target)
        # Negated: Drake returns generalized gravity force (the force gravity
        # exerts), we want compensation torque. See scripts/test_gravity_sign.py.
        tau_g = -plant.CalcGravityGeneralizedForces(aux_ctx)[:n_arm]

        tau_pre  = tau_P + tau_I + tau_D + tau_g
        tau_clip = np.clip(tau_pre, -torque_limit, +torque_limit)
        clipped_mask = (np.abs(tau_pre) > torque_limit + 1e-9)

        # --- Safety stops ---
        ee_pos = plant.CalcPointsPositions(
            plant_ctx, ee_frame, np.zeros(3), world_frame
        ).flatten()
        if not (-0.5 <= ee_pos[2] <= 0.5):
            print(f"[ABORT] EE z out of band at t={step*dt_ctrl:.2f}s: z={ee_pos[2]:+.3f}")
            return
        if np.any(np.abs(tau_pre) > 100.0):
            print(f"[ABORT] |tau_pre| > 100 at t={step*dt_ctrl:.2f}s")
            return

        if step % PRINT_EVERY == 0 or step == n_steps:
            print(f"{step*dt_ctrl:>6.2f} "
                  f"{ee_pos[0]:>+7.4f} {ee_pos[1]:>+7.4f} {ee_pos[2]:>+7.4f} "
                  f"{np.linalg.norm(q_err):>8.4f} "
                  f"{np.linalg.norm(tau_P):>7.2f} "
                  f"{np.linalg.norm(tau_I):>7.2f} "
                  f"{np.linalg.norm(tau_D):>7.2f} "
                  f"{np.linalg.norm(tau_g):>7.2f} "
                  f"{np.linalg.norm(tau_clip):>7.2f} "
                  f"{str(clipped_mask.astype(int).tolist()):>15}")
            final_row = {
                "t": step * dt_ctrl, "ee_pos": ee_pos.copy(),
                "q_err": q_err.copy(), "v_arm": v_arm.copy(),
                "tau_P": tau_P.copy(), "tau_I": tau_I.copy(),
                "tau_D": tau_D.copy(), "tau_g": tau_g.copy(),
                "tau_pre": tau_pre.copy(), "tau_clip": tau_clip.copy(),
                "clipped_mask": clipped_mask.copy(),
                "integral": integral.copy(),
            }

        # Apply torque and step
        plant.get_actuation_input_port().FixValue(plant_ctx, tau_clip)
        simulator.AdvanceTo((step + 1) * dt_ctrl)

    # --- Per-joint table at end of run ---
    print(f"\n[FINAL] ee_pos = ({final_row['ee_pos'][0]:+.4f}, "
          f"{final_row['ee_pos'][1]:+.4f}, {final_row['ee_pos'][2]:+.4f})")
    drift_mm = float(np.linalg.norm(final_row["ee_pos"] - ee_start) * 1000.0)
    drift_z_mm = float((final_row["ee_pos"][2] - ee_start[2]) * 1000.0)
    print(f"[FINAL] drift = {drift_mm:.2f} mm  (z component: {drift_z_mm:+.2f} mm)")
    print(f"\nPer-joint final state:")
    print(f"{'j':>2} {'q_err':>9} {'integ':>8} {'tau_P':>8} {'tau_I':>8} "
          f"{'tau_D':>8} {'tau_g':>8} {'tau_pre':>9} {'tau_clip':>9} {'clipped':>8} {'I@max':>6}")
    for j in range(n_arm):
        i_at_max = abs(final_row["integral"][j]) >= I_max - 1e-6
        print(f"{j:>2} {final_row['q_err'][j]:>+9.4f} "
              f"{final_row['integral'][j]:>+8.3f} "
              f"{final_row['tau_P'][j]:>+8.2f} "
              f"{final_row['tau_I'][j]:>+8.2f} "
              f"{final_row['tau_D'][j]:>+8.2f} "
              f"{final_row['tau_g'][j]:>+8.2f} "
              f"{final_row['tau_pre'][j]:>+9.2f} "
              f"{final_row['tau_clip'][j]:>+9.2f} "
              f"{str(bool(final_row['clipped_mask'][j])):>8} "
              f"{str(bool(i_at_max)):>6}")


if __name__ == "__main__":
    main()
