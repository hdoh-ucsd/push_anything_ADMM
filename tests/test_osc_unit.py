"""Unit tests for `OperationalSpaceController` — QP solves cleanly in
the simple cases that establish baseline correctness:

* Identity case: target = current EE, no contact → QP returns near-zero
  net torque after gravity compensation cancels out, no saturation.
* Gravity compensation: the OSC's commanded τ on a horizontal arm
  closely matches Drake's `−CalcGravityGeneralizedForces` projection
  (within the few-Nm-per-joint posture/task tie-break tolerance).
* Tracking force: with a 5 mm EE position error, the OSC commands a
  non-zero τ pointing in the direction that closes the error.
* Torque saturation graceful: with an unreachable 1 m EE position
  error, the QP saturates at URDF limits and returns a feasible u
  without crashing.
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

from sim.env_builder import build_environment, _INITIAL_ARM_Q_SEED, EE_BODY_NAME
from control.osc import OperationalSpaceController

# `sim.env_builder.INITIAL_ARM_Q` was removed in 7ff5a21 ("replace
# INITIAL_ARM_Q + --prepositioned with safe-offset IK init"): there is no
# longer one fixed start pose, the runtime derives it per task (IK safe-offset,
# or `q_init_franka` from tasks.yaml for the canonical tasks). These tests only
# ever needed *a* valid, representative arm configuration to evaluate the
# LCS/OSC at, so they pin the IK seed. Note it is NOT the production start
# pose, and its joint 2 moved 0.675 -> 1.1 rad when the arm-fly-up bug was
# fixed.
ARM_Q_FIXTURE = _INITIAL_ARM_Q_SEED


@pytest.fixture(scope="module")
def osc_env():
    """Build the Franka + box environment once for the test module."""
    with open(os.path.join(REPO_ROOT, "config/tasks.yaml")) as f:
        task = yaml.safe_load(f)["tasks"]["pushing"]
    diagram, plant, panda_model, _, _, _, _, _ = build_environment(task)
    ee_frame = plant.GetBodyByName(EE_BODY_NAME).body_frame()
    diagram_ctx = diagram.CreateDefaultContext()
    plant_ctx = plant.GetMyMutableContextFromRoot(diagram_ctx)
    plant.SetPositions(plant_ctx, panda_model, ARM_Q_FIXTURE)

    # q_nominal = ARM_Q_FIXTURE (no posture error → cleanest test).
    osc = OperationalSpaceController(
        plant=plant, ee_frame=ee_frame, n_arm_dofs=7,
        q_nominal=ARM_Q_FIXTURE,
        gains_yaml=os.path.join(REPO_ROOT, "config/osc_franka.yaml"),
        log_diag=False,
    )
    return dict(plant=plant, plant_ctx=plant_ctx, ee_frame=ee_frame, osc=osc)


def _current_state(env):
    """Return (q, v, ee_now) at the env's current plant_ctx."""
    plant = env["plant"]
    plant_ctx = env["plant_ctx"]
    q = plant.GetPositions(plant_ctx).copy()
    v = plant.GetVelocities(plant_ctx).copy()
    ee_now = plant.CalcPointsPositions(
        plant_ctx, env["ee_frame"], np.zeros(3), plant.world_frame(),
    ).flatten()
    return q, v, ee_now


def test_identity_no_contact(osc_env):
    """Target = current EE, no contact, no error → QP solves successfully
    and saturates no joint."""
    q, v, ee_now = _current_state(osc_env)
    u, diag = osc_env["osc"].compute_torque(q, v, osc_env["plant_ctx"],
                                            p_ee_desired=ee_now)
    assert diag["qp_success"]
    assert not diag["saturated"], f"Expected no saturation but got u={u}"
    assert np.all(np.isfinite(u))


def test_gravity_compensation(osc_env):
    """OSC's commanded τ should approximately match Drake's gravity
    compensation when target = current EE and posture = current q.

    Tolerance is loose (5 Nm/joint) because the task cost (W_track=100,
    Kp_cart=400) and posture cost (W_posture=1, Kp_null=10) impose small
    accel-tracking adjustments that perturb τ from pure gravity comp.
    """
    plant = osc_env["plant"]
    plant_ctx = osc_env["plant_ctx"]
    q, v, ee_now = _current_state(osc_env)

    u, diag = osc_env["osc"].compute_torque(q, v, plant_ctx,
                                            p_ee_desired=ee_now)
    g = plant.CalcGravityGeneralizedForces(plant_ctx)[:7]
    # Drake returns +g; commanded torque to hold against gravity is -g.
    tau_grav_comp = -g
    err = u - tau_grav_comp
    assert np.max(np.abs(err)) < 5.0, (
        f"OSC τ deviates from gravity comp by {np.max(np.abs(err)):.2f} Nm. "
        f"u={u}, -g={tau_grav_comp}"
    )


def test_tracking_force_direction(osc_env):
    """A 5 mm position error should produce a τ whose induced EE
    acceleration points TOWARD the target (i.e. closes the error).

    Compute the induced EE accel from M⁻¹(B τ − bias) → v̇ → J_v · v̇.
    """
    plant = osc_env["plant"]
    plant_ctx = osc_env["plant_ctx"]
    q, v, ee_now = _current_state(osc_env)
    target = ee_now + np.array([0.005, 0.0, 0.0])   # 5 mm in +x
    u, diag = osc_env["osc"].compute_torque(q, v, plant_ctx,
                                            p_ee_desired=target)
    assert diag["qp_success"]

    # The vdot the QP solved for already takes the feedforward + dynamics
    # into account; check its J_v · vdot directly.
    vdot = diag["vdot_opt"]
    import pydrake.all as ad
    J_v = plant.CalcJacobianTranslationalVelocity(
        plant_ctx, ad.JacobianWrtVariable.kV,
        osc_env["ee_frame"], np.zeros(3),
        plant.world_frame(), plant.world_frame(),
    )
    ee_accel = J_v @ vdot
    # Component along the error direction should be positive (closing).
    err_dir = (target - ee_now) / np.linalg.norm(target - ee_now)
    proj = float(np.dot(ee_accel, err_dir))
    assert proj > 0.0, (
        f"Expected EE accel along error direction > 0, got {proj:.3f} m/s². "
        f"vdot={vdot}, ee_accel={ee_accel}, err_dir={err_dir}"
    )


def test_saturation_graceful(osc_env):
    """An unreachable 1 m position error should still return a feasible
    u (not crash, not return NaN), with at least one joint saturated."""
    plant = osc_env["plant"]
    plant_ctx = osc_env["plant_ctx"]
    q, v, ee_now = _current_state(osc_env)
    target = ee_now + np.array([1.0, 0.0, 0.0])    # 1 m in +x — unreachable
    u, diag = osc_env["osc"].compute_torque(q, v, plant_ctx,
                                            p_ee_desired=target)
    assert diag["qp_success"], (
        f"QP failed on saturation test: {diag.get('qp_result')}"
    )
    assert np.all(np.isfinite(u))
    # Per-joint URDF limit
    limits = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
    # u must respect box constraints
    assert np.all(np.abs(u) <= limits + 1e-6)


def test_contact_feedforward_directions(osc_env):
    """Apply a planner-style λ_n = 5 N normal force with J_n computed at
    the current pose. The QP's solution should reflect the feedforward in
    the equality constraint — verify by checking the dynamics balance:
        M v̇ + (C v − τ_g) − B τ = J^T λ_planned  (residual should be 0)
    """
    plant = osc_env["plant"]
    plant_ctx = osc_env["plant_ctx"]
    q, v, ee_now = _current_state(osc_env)

    n_v = plant.num_velocities()
    # Synthetic contact: a 1×n_v Jacobian row pointing in +x in world.
    import pydrake.all as ad
    J_v = plant.CalcJacobianTranslationalVelocity(
        plant_ctx, ad.JacobianWrtVariable.kV,
        osc_env["ee_frame"], np.zeros(3),
        plant.world_frame(), plant.world_frame(),
    )
    J_n = J_v[0:1, :]                    # (1, n_v) — +x component
    lam_n = np.array([5.0])              # 5 N pushing along the row

    u, diag = osc_env["osc"].compute_torque(
        q, v, plant_ctx, p_ee_desired=ee_now,
        lambda_n=lam_n, J_n=J_n,
    )
    assert diag["qp_success"]

    # Verify dynamics balance.
    M = plant.CalcMassMatrix(plant_ctx)
    Cv = plant.CalcBiasTerm(plant_ctx)
    g = plant.CalcGravityGeneralizedForces(plant_ctx)
    B = plant.MakeActuationMatrix()
    vdot = diag["vdot_opt"]
    lhs = M @ vdot + (Cv - g) - B @ u
    rhs = J_n.T @ lam_n
    residual = lhs - rhs.flatten()
    assert np.max(np.abs(residual)) < 1e-4, (
        f"Dynamics residual {np.max(np.abs(residual)):.3e} exceeds 1e-4. "
        f"lhs={lhs}, rhs={rhs}"
    )
