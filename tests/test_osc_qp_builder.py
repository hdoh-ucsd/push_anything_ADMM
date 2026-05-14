"""QP-builder unit tests (Phase 2 STEP 3)."""
from __future__ import annotations

import numpy as np
import pytest

ad = pytest.importorskip("pydrake.all",
                         reason="Drake required for OSC QP tests")
import yaml
from pydrake.solvers import OsqpSolver

from sim.env_builder import build_environment, EE_BODY_NAME, INITIAL_ARM_Q
from control.osc.qp_builder import build_osc_qp


@pytest.fixture(scope="module")
def env():
    """Single Drake environment shared across QP tests."""
    with open("config/tasks.yaml") as f:
        task_cfg = yaml.safe_load(f)["tasks"]["pushing"]
    diagram, plant, panda_model, _, _, _, _ = build_environment(task_cfg)
    ctx = plant.CreateDefaultContext()
    ee_frame = plant.GetFrameByName(EE_BODY_NAME)

    n_arm = plant.num_actuators()
    q = np.zeros(plant.num_positions())
    q[:n_arm] = INITIAL_ARM_Q
    # Floating-body manipuland: unit quaternion + init xyz.
    q[n_arm]     = 1.0
    q[n_arm+1:n_arm+4] = 0.0
    q[n_arm+4:n_arm+7] = task_cfg["init_xyz"]
    plant.SetPositions(ctx, q)
    plant.SetVelocities(ctx, np.zeros(plant.num_velocities()))
    return plant, ctx, ee_frame, n_arm


def _default_gains(n_arm):
    return dict(
        Kp_task=np.array([100., 100., 100.]),
        Kd_task=np.array([ 20.,  20.,  20.]),
        Kp_posture=np.array([50, 50, 50, 50, 20, 20, 10], dtype=float),
        Kd_posture=np.array([10, 10, 10, 10,  5,  5,  3], dtype=float),
        W_task=10.0, W_posture=1.0,
        torque_limits=np.array([87., 87., 87., 87., 12., 12., 12.]),
    )


def test_static_regulation_gives_gravity_comp(env):
    """At INITIAL_ARM_Q, v=0, x_des=x_now, xd_des=0: the QP should
    return tau ≈ -tau_g[arm] (gravity comp) and qdd ≈ 0."""
    plant, ctx, ee_frame, n_arm = env
    x_now = plant.CalcPointsPositions(
        ctx, ee_frame, np.zeros(3), plant.world_frame()).flatten()

    g = _default_gains(n_arm)
    prog, dv = build_osc_qp(
        plant, ctx, ee_frame,
        x_des=x_now, xd_des=np.zeros(3),
        q_home_arm=np.asarray(INITIAL_ARM_Q),
        q_now_arm=np.asarray(INITIAL_ARM_Q),
        v_now_arm=np.zeros(n_arm),
        **g, n_arm=n_arm,
    )

    res = OsqpSolver().Solve(prog)
    assert res.is_success(), f"QP failed: {res.get_solution_result()}"
    tau = res.GetSolution(dv["tau"])
    qdd = res.GetSolution(dv["qdd"])

    # Expected: tau ≈ -tau_g[arm], qdd_arm ≈ 0.
    # NB: qdd_manipuland is not constrained — the box is unactuated, so
    # under gravity its z-acceleration sits at −9.81 m/s² in the QP
    # solution. Only the arm portion is bound to zero by the posture
    # cost + arm dynamics. Tau is the load-bearing quantity.
    tau_g_arm = dv["tau_g"][:n_arm]
    err_tau = tau - (-tau_g_arm)
    err_qdd_arm = qdd[:n_arm]

    # Tolerance: 0.5 Nm per joint on tau, 0.5 rad/s² per joint on qdd_arm.
    assert np.all(np.abs(err_tau) < 0.5), (
        f"tau != -gravity:\n  got {tau.round(3)}\n  expected {(-tau_g_arm).round(3)}\n"
        f"  err {err_tau.round(3)}")
    assert np.all(np.abs(err_qdd_arm) < 0.5), (
        f"qdd_arm should be ~0, got {qdd[:n_arm].round(3)}")


def test_step_response_pushes_in_x(env):
    """At INITIAL_ARM_Q, v=0, x_des = x_now + [0.01, 0, 0]: solution
    should produce a +x EE acceleration (J·qdd has positive x component)."""
    plant, ctx, ee_frame, n_arm = env
    x_now = plant.CalcPointsPositions(
        ctx, ee_frame, np.zeros(3), plant.world_frame()).flatten()
    x_des = x_now + np.array([0.01, 0.0, 0.0])

    g = _default_gains(n_arm)
    prog, dv = build_osc_qp(
        plant, ctx, ee_frame,
        x_des=x_des, xd_des=np.zeros(3),
        q_home_arm=np.asarray(INITIAL_ARM_Q),
        q_now_arm=np.asarray(INITIAL_ARM_Q),
        v_now_arm=np.zeros(n_arm),
        **g, n_arm=n_arm,
    )

    res = OsqpSolver().Solve(prog)
    assert res.is_success(), f"QP failed: {res.get_solution_result()}"
    qdd = res.GetSolution(dv["qdd"])
    # Task EE accel = J·qdd + J̇·v; J̇·v ≈ 0 at v=0.
    ee_accel = dv["J"] @ qdd + dv["Jdv"]
    assert ee_accel[0] > 0.05, (
        f"expected positive +x EE accel, got {ee_accel.round(3)}")


def test_torque_limit_binds_on_unreachable_target(env):
    """At INITIAL_ARM_Q, v=0, x_des = x_now + [1.0, 0, 0]: QP must
    remain feasible (task is a soft cost), and at least one joint
    torque should saturate."""
    plant, ctx, ee_frame, n_arm = env
    x_now = plant.CalcPointsPositions(
        ctx, ee_frame, np.zeros(3), plant.world_frame()).flatten()
    x_des = x_now + np.array([1.0, 0.0, 0.0])

    g = _default_gains(n_arm)
    prog, dv = build_osc_qp(
        plant, ctx, ee_frame,
        x_des=x_des, xd_des=np.zeros(3),
        q_home_arm=np.asarray(INITIAL_ARM_Q),
        q_now_arm=np.asarray(INITIAL_ARM_Q),
        v_now_arm=np.zeros(n_arm),
        **g, n_arm=n_arm,
    )

    res = OsqpSolver().Solve(prog)
    assert res.is_success(), f"QP infeasible — should be soft: {res.get_solution_result()}"
    tau = res.GetSolution(dv["tau"])
    torque_limits = g["torque_limits"]
    # Check at least one joint is within 0.5 Nm of its limit.
    at_limit = np.any(np.abs(tau) > (torque_limits - 0.5))
    assert at_limit, (
        f"expected at least one joint at torque limit for a 1m unreachable target;"
        f" tau={tau.round(2)} limits={torque_limits.tolist()}")
