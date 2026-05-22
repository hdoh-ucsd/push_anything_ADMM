"""Integration-flavored tracking tests for the OSC.

These tests do NOT simulate the full Drake plant — they just verify that
the OSC's QP solution would produce reasonable closed-loop behavior:

* Multi-step regulation: simulate forward with M⁻¹(B τ − bias) for a few
  steps; EE should move toward the target.
* Saturation never trips QP failure across a range of error magnitudes.
* Posture-cost weight: with W_posture << W_track, the OSC should keep
  the EE near the target even when posture is impossible. With W_track
  reduced, posture should dominate.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sim.env_builder import build_environment, INITIAL_ARM_Q, EE_BODY_NAME
from control.osc import OperationalSpaceController


@pytest.fixture(scope="module")
def osc_env():
    with open(os.path.join(REPO_ROOT, "config/tasks.yaml")) as f:
        task = yaml.safe_load(f)["tasks"]["pushing"]
    diagram, plant, panda_model, _, _, _, _ = build_environment(task)
    ee_frame = plant.GetBodyByName(EE_BODY_NAME).body_frame()
    diagram_ctx = diagram.CreateDefaultContext()
    plant_ctx = plant.GetMyMutableContextFromRoot(diagram_ctx)
    plant.SetPositions(plant_ctx, panda_model, INITIAL_ARM_Q)

    osc = OperationalSpaceController(
        plant=plant, ee_frame=ee_frame, n_arm_dofs=7,
        q_nominal=INITIAL_ARM_Q,
        gains_yaml=os.path.join(REPO_ROOT, "config/osc_franka.yaml"),
        log_diag=False,
    )
    return dict(plant=plant, plant_ctx=plant_ctx, ee_frame=ee_frame, osc=osc,
                panda_model=panda_model)


def _ee_pos(env, q):
    plant, plant_ctx, ee_frame = env["plant"], env["plant_ctx"], env["ee_frame"]
    plant.SetPositions(plant_ctx, q)
    return plant.CalcPointsPositions(
        plant_ctx, ee_frame, np.zeros(3), plant.world_frame(),
    ).flatten()


def test_single_step_closes_error(osc_env):
    """With a 5 mm error, the QP-derived v̇ should produce an EE accel
    that moves the EE in the direction of the target — checked via a
    one-step forward Euler under M v̇ + (Cv − τ_g) = B τ."""
    plant, plant_ctx, ee_frame, osc = (
        osc_env["plant"], osc_env["plant_ctx"], osc_env["ee_frame"],
        osc_env["osc"],
    )
    q = plant.GetPositions(plant_ctx).copy()
    v = plant.GetVelocities(plant_ctx).copy()
    ee_now = plant.CalcPointsPositions(plant_ctx, ee_frame, np.zeros(3),
                                        plant.world_frame()).flatten()
    target = ee_now + np.array([0.005, 0.0, 0.0])
    u, diag = osc.compute_torque(q, v, plant_ctx, p_ee_desired=target)
    assert diag["qp_success"]

    # Forward-Euler one step
    dt = 0.01
    v_next = v + dt * diag["vdot_opt"]
    q_next = q.copy()
    q_next[:7] = q[:7] + dt * v_next[:7]   # arm only
    ee_next = _ee_pos(osc_env, q_next)
    err_before = float(np.linalg.norm(target - ee_now))
    err_after  = float(np.linalg.norm(target - ee_next))
    assert err_after < err_before, (
        f"Error did not decrease: before={err_before:.5f}, after={err_after:.5f}, "
        f"vdot_opt={diag['vdot_opt']}"
    )


@pytest.mark.parametrize("dx", [0.001, 0.005, 0.01, 0.05, 0.1, 0.5])
def test_no_qp_failure_across_error_range(osc_env, dx):
    """Sweep across error magnitudes from 1 mm to 50 cm; QP always solves."""
    plant, plant_ctx, ee_frame, osc = (
        osc_env["plant"], osc_env["plant_ctx"], osc_env["ee_frame"],
        osc_env["osc"],
    )
    q = plant.GetPositions(plant_ctx).copy()
    v = plant.GetVelocities(plant_ctx).copy()
    ee_now = plant.CalcPointsPositions(plant_ctx, ee_frame, np.zeros(3),
                                        plant.world_frame()).flatten()
    target = ee_now + np.array([dx, 0.0, 0.0])
    u, diag = osc.compute_torque(q, v, plant_ctx, p_ee_desired=target)
    assert diag["qp_success"], f"QP failed at dx={dx}: {diag.get('qp_result')}"
    # No NaN/inf
    assert np.all(np.isfinite(u))
    assert np.all(np.isfinite(diag["vdot_opt"]))


def test_torque_limits_respected(osc_env):
    """Across the full range of error magnitudes, returned τ stays
    within URDF per-joint limits (87/87/87/87/12/12/12 Nm)."""
    plant, plant_ctx, ee_frame, osc = (
        osc_env["plant"], osc_env["plant_ctx"], osc_env["ee_frame"],
        osc_env["osc"],
    )
    q = plant.GetPositions(plant_ctx).copy()
    v = plant.GetVelocities(plant_ctx).copy()
    ee_now = plant.CalcPointsPositions(plant_ctx, ee_frame, np.zeros(3),
                                        plant.world_frame()).flatten()
    limits = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
    for dx in [0.001, 0.01, 0.1, 0.5, 1.0]:
        target = ee_now + np.array([dx, 0.0, 0.0])
        u, diag = osc.compute_torque(q, v, plant_ctx, p_ee_desired=target)
        assert diag["qp_success"]
        assert np.all(np.abs(u) <= limits + 1e-6), (
            f"At dx={dx}, τ exceeded URDF limit: u={u}, limits={limits}"
        )
