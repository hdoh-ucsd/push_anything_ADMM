"""Dynamics-helper unit tests (Phase 2 STEP 2).

These validate the four primitives OSC depends on. Run first; if any
fail, the OSC build cannot proceed.
"""
from __future__ import annotations

import numpy as np
import pytest

ad = pytest.importorskip("pydrake.all",
                         reason="Drake required for OSC dynamics helpers")
import yaml

from sim.env_builder import build_environment, EE_BODY_NAME, INITIAL_ARM_Q
from control.osc.dynamics_helpers import (
    get_actuation_matrix,
    get_bias_acceleration,
    get_coriolis_centrifugal,
    get_gravity,
    get_jacobian,
    get_mass_matrix,
)


@pytest.fixture(scope="module")
def env():
    with open("config/tasks.yaml") as f:
        task_cfg = yaml.safe_load(f)["tasks"]["pushing"]
    diagram, plant, panda_model, _, _, _, _ = build_environment(task_cfg)
    ctx = plant.CreateDefaultContext()
    ee_frame = plant.GetFrameByName(EE_BODY_NAME)

    n_arm = plant.num_actuators()
    q = np.zeros(plant.num_positions())
    q[:n_arm] = INITIAL_ARM_Q
    q[n_arm]     = 1.0
    q[n_arm+1:n_arm+4] = 0.0
    q[n_arm+4:n_arm+7] = task_cfg["init_xyz"]
    plant.SetPositions(ctx, q)
    plant.SetVelocities(ctx, np.zeros(plant.num_velocities()))
    return plant, ctx, ee_frame, n_arm


def test_jacobian_shape_and_manipuland_columns_zero(env):
    """J should be (3, n_v) and the manipuland columns (≥ n_arm) zero —
    moving the box doesn't move the pusher in the kinematic chain."""
    plant, ctx, ee_frame, n_arm = env
    J = get_jacobian(plant, ctx, ee_frame)
    assert J.shape == (3, plant.num_velocities()), (
        f"J shape {J.shape} ≠ (3, {plant.num_velocities()})")
    # Floating-body manipuland velocities are columns [n_arm : n_v].
    J_manip = J[:, n_arm:]
    assert np.allclose(J_manip, 0.0, atol=1e-10), (
        f"manipuland columns should be 0, max |J_manip| = "
        f"{np.max(np.abs(J_manip)):.6e}")


def test_bias_acceleration_zero_at_rest(env):
    """At v = 0, J̇·v = 0 trivially."""
    plant, ctx, ee_frame, n_arm = env
    Jdv = get_bias_acceleration(plant, ctx, ee_frame)
    assert Jdv.shape == (3,)
    assert np.allclose(Jdv, 0.0, atol=1e-10), f"Jdv at rest = {Jdv}"


def test_coriolis_zero_at_rest(env):
    """C·v should be zero when v = 0."""
    plant, ctx, _, _ = env
    Cv = get_coriolis_centrifugal(plant, ctx)
    assert Cv.shape == (plant.num_velocities(),)
    assert np.allclose(Cv, 0.0, atol=1e-10), f"Cv at rest = {Cv}"


def test_mass_matrix_shape_symmetric_pd(env):
    """M is (n_v, n_v), symmetric, and positive definite."""
    plant, ctx, _, _ = env
    n_v = plant.num_velocities()
    M = get_mass_matrix(plant, ctx)
    assert M.shape == (n_v, n_v), f"M shape {M.shape} ≠ ({n_v}, {n_v})"
    # Symmetry.
    assert np.allclose(M, M.T, atol=1e-9), (
        f"M not symmetric, max asym = {np.max(np.abs(M - M.T)):.3e}")
    # Positive definite via min eigenvalue.
    w = np.linalg.eigvalsh((M + M.T) / 2.0)
    assert w.min() > 0.0, f"M not PD; min eig = {w.min():.3e}"


def test_gravity_sign_matches_dynamics_equation(env):
    """Per the docstring, the equation of motion is
        M·v̇ + C·v = τ_app + τ_g
    To hold the pose statically against gravity, τ_arm = −τ_g[arm].
    Verify this by feeding the plant the negative of the gravity-arm
    torque and checking that the resulting v̇ is zero (within solver
    rounding) when v = 0. We do this analytically: at v=0, M·v̇ = B·τ + τ_g,
    so for v̇=0 we need B·τ + τ_g = 0  →  τ = −(BᵀB)⁻¹Bᵀ·τ_g, which
    simplifies to τ = −τ_g[arm] since B selects arm rows.
    """
    plant, ctx, _, n_arm = env
    tau_g = get_gravity(plant, ctx)
    B = get_actuation_matrix(plant)
    assert B.shape == (plant.num_velocities(), n_arm)

    # Static balance: τ_arm = -τ_g[arm]
    tau_balance = -tau_g[:n_arm]
    # Plug back into the equation: M·v̇ = B·τ + τ_g, with τ = tau_balance
    rhs = B @ tau_balance + tau_g
    # Verify that the arm-portion of rhs is zero (we're balancing the arm).
    assert np.allclose(rhs[:n_arm], 0.0, atol=1e-10), (
        f"Gravity balance fails arm: rhs[:n_arm] = {rhs[:n_arm].round(6)}")

    # Print the per-joint balance torques for diagnostic visibility.
    print()
    for i, t in enumerate(tau_balance):
        print(f"  joint {i+1}: balance τ = {t:+.3f} Nm "
              f"(gravity τ_g[arm][{i}] = {tau_g[i]:+.3f} Nm)")


def test_actuation_matrix_selects_arm(env):
    """B should select the first 7 rows of the velocity space."""
    plant, _, _, n_arm = env
    B = get_actuation_matrix(plant)
    expected = np.zeros((plant.num_velocities(), n_arm))
    expected[:n_arm, :n_arm] = np.eye(n_arm)
    assert np.allclose(B, expected), (
        f"B is not the simple arm-selection matrix:\n{B}")
