"""Drake plant query wrappers for OSC.

Pure functions. Each call resets the plant context's (q, v) and queries
Drake. Caller is responsible for setting up the context once per OSC step
(the OSC class does this in compute_torque).
"""
from __future__ import annotations

import numpy as np
import pydrake.all as ad


def mass_matrix(plant, plant_ctx) -> np.ndarray:
    """Returns M(q) ∈ ℝ^{n_v × n_v}."""
    return plant.CalcMassMatrix(plant_ctx)


def bias_term(plant, plant_ctx) -> np.ndarray:
    """Returns C(q,v) v ∈ ℝ^{n_v} — Coriolis + centrifugal bundle."""
    return plant.CalcBiasTerm(plant_ctx)


def gravity_forces(plant, plant_ctx) -> np.ndarray:
    """Returns generalized gravity force ∈ ℝ^{n_v}.

    Sign convention (Drake): positive value = gravity drives the joint
    along +ve angle. For gravity compensation in commanded torque, the
    caller NEGATES.
    """
    return plant.CalcGravityGeneralizedForces(plant_ctx)


def actuation_matrix(plant) -> np.ndarray:
    """Returns B ∈ ℝ^{n_v × n_u}. Constant across configurations."""
    return plant.MakeActuationMatrix()


def ee_jacobian_translational(plant, plant_ctx, ee_frame) -> np.ndarray:
    """Returns J_v ∈ ℝ^{3 × n_v} (translational EE velocity Jacobian).

    Origin of `ee_frame` is the point being tracked.
    """
    world_frame = plant.world_frame()
    return plant.CalcJacobianTranslationalVelocity(
        plant_ctx, ad.JacobianWrtVariable.kV,
        ee_frame, np.zeros(3),
        world_frame, world_frame,
    )


def ee_jacobian_bias(plant, plant_ctx, ee_frame) -> np.ndarray:
    """Returns J̇_v · v ∈ ℝ³ (translational bias acceleration of EE).

    This is the part of EE Cartesian acceleration that does NOT depend on
    v̇ — i.e. ẍ = J_v v̇ + J̇_v v. Required for the task-tracking cost.
    """
    world_frame = plant.world_frame()
    bias_spatial = plant.CalcBiasSpatialAcceleration(
        plant_ctx, ad.JacobianWrtVariable.kV,
        ee_frame, np.zeros(3),
        world_frame, world_frame,
    )
    return bias_spatial.translational()


def ee_position(plant, plant_ctx, ee_frame) -> np.ndarray:
    """Returns the EE origin in world frame (3,)."""
    return plant.CalcPointsPositions(
        plant_ctx, ee_frame, np.zeros(3), plant.world_frame(),
    ).flatten()


def franka_effort_limits(plant) -> np.ndarray:
    """Returns the per-actuator effort limit vector (n_u,) from the URDF.

    Drake exposes this via JointActuator.effort_limit(). For Franka the
    URDF spec is 87/87/87/87/12/12/12 Nm.
    """
    idxs = plant.GetJointActuatorIndices()
    limits = np.array(
        [plant.get_joint_actuator(i).effort_limit() for i in idxs],
        dtype=float,
    )
    return limits
