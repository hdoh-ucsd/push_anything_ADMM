"""STEP 6 — Tracking test for OperationalSpaceController.

Follow a 0.1 Hz sinusoid in +x of amplitude 5 cm. Verify peak |error|
≤ 1 cm and RMS |error| ≤ 5 mm over one full period (10 s).
"""
from __future__ import annotations

import numpy as np
import pytest

ad = pytest.importorskip("pydrake.all",
                         reason="Drake required for OSC tracking test")
import yaml

from sim.env_builder import build_environment, EE_BODY_NAME, INITIAL_ARM_Q
from control.osc.qp_builder import build_osc_qp
from pydrake.solvers import OsqpSolver


@pytest.mark.timeout(120)
def test_tracking_slow_sinusoid():
    """Track x_des(t) = x_init + [0.05 sin(2π·0.1·t), 0, 0] for 10 s."""
    with open("config/tasks.yaml") as f:
        task_cfg = yaml.safe_load(f)["tasks"]["pushing"]
    with open("config/osc_franka.yaml") as f:
        gains_cfg = yaml.safe_load(f)

    diagram, plant, panda_model, _, _, _, _ = build_environment(task_cfg)
    n_a = plant.num_actuators()
    n_q = plant.num_positions()
    n_v = plant.num_velocities()
    ee_frame = plant.GetFrameByName(EE_BODY_NAME)
    obj_body = plant.GetBodyByName(task_cfg["link_name"])

    simulator = ad.Simulator(diagram)
    context   = simulator.get_mutable_context()
    plant_ctx = plant.GetMyContextFromRoot(context)

    plant.SetPositions(plant_ctx, panda_model, INITIAL_ARM_Q)
    plant.SetVelocities(plant_ctx, np.zeros(n_v))
    plant.SetFreeBodyPose(plant_ctx, obj_body,
        ad.RigidTransform(ad.RotationMatrix(), task_cfg["init_xyz"]))

    x_initial = plant.CalcPointsPositions(
        plant_ctx, ee_frame, np.zeros(3), plant.world_frame()).flatten()

    A = 0.05    # amplitude (m)
    f = 0.1     # frequency (Hz)
    w = 2.0 * np.pi * f
    T = 10.0    # one period
    dt_ctrl = 1e-3

    def setpoint(t: float):
        x_des  = x_initial + np.array([A * np.sin(w * t), 0.0, 0.0])
        xd_des = np.array([A * w * np.cos(w * t), 0.0, 0.0])
        return x_des, xd_des

    solver = OsqpSolver()
    osc_ctx = plant.CreateDefaultContext()

    n_steps = int(T / dt_ctrl)
    err_traj = np.zeros(n_steps + 1)
    tau_max_seen = np.zeros(n_a)

    err_traj[0] = 0.0  # we start exactly at x_initial which == x_des(0)

    for step in range(n_steps):
        q = plant.GetPositions(plant_ctx)
        v = plant.GetVelocities(plant_ctx)
        plant.SetPositions(osc_ctx, q)
        plant.SetVelocities(osc_ctx, v)

        t = step * dt_ctrl
        x_des, xd_des = setpoint(t)

        prog, dv = build_osc_qp(
            plant, osc_ctx, ee_frame,
            x_des=x_des, xd_des=xd_des,
            q_home_arm=np.asarray(INITIAL_ARM_Q),
            q_now_arm=q[:n_a],
            v_now_arm=v[:n_a],
            Kp_task=np.asarray(gains_cfg["Kp_task"]),
            Kd_task=np.asarray(gains_cfg["Kd_task"]),
            Kp_posture=np.asarray(gains_cfg["Kp_posture"], dtype=float),
            Kd_posture=np.asarray(gains_cfg["Kd_posture"], dtype=float),
            W_task=float(gains_cfg["W_task"]),
            W_posture=float(gains_cfg["W_posture"]),
            torque_limits=np.asarray(gains_cfg["torque_limits"]),
            w_tau_reg=float(gains_cfg.get("w_tau_reg", 1e-4)),
            n_arm=n_a,
        )
        res = solver.Solve(prog)
        assert res.is_success(), (
            f"step {step}: tracking QP failed — {res.get_solution_result()}")

        tau = res.GetSolution(dv["tau"])
        tau_max_seen = np.maximum(tau_max_seen, np.abs(tau))
        plant.get_actuation_input_port().FixValue(plant_ctx, tau)

        simulator.AdvanceTo((step + 1) * dt_ctrl)

        x_now = plant.CalcPointsPositions(
            plant_ctx, ee_frame, np.zeros(3), plant.world_frame()).flatten()
        x_des_next, _ = setpoint((step + 1) * dt_ctrl)
        err_traj[step + 1] = float(np.linalg.norm(x_now - x_des_next))

    peak_err = float(np.max(err_traj))
    rms_err  = float(np.sqrt(np.mean(err_traj ** 2)))

    print()
    print(f"[tracking] peak |error| = {peak_err*1000:.2f} mm")
    print(f"[tracking] RMS  |error| = {rms_err*1000:.2f} mm")
    print(f"[tracking] peak |tau| per joint = "
          f"{tau_max_seen.round(2).tolist()}")

    assert peak_err <= 1e-2, f"peak |error| {peak_err*1000:.2f} mm > 10 mm limit"
    assert rms_err  <= 5e-3, f"RMS |error| {rms_err*1000:.2f} mm > 5 mm limit"
