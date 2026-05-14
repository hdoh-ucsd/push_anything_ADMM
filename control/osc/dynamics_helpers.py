"""Drake plant accessors used by the OSC QP builder.

Encapsulates the pydrake API quirks identified in OSC Phase 1:
  - JacobianWrtVariable.kV is the correct enum (kQDot is wrong for
    quaternion-floating bodies and unnecessary for revolute-only).
  - CalcBiasTerm returns C(q,v)·v (joint-space Coriolis + centrifugal),
    distinct from CalcBiasSpatialAcceleration which returns J̇·v
    (task-space bias for tracking).
  - The Franka has 7 actuators on a 13-DOF velocity space (7 arm + 6
    floating manipuland). The actuation matrix B selects the first
    7 rows.
"""
from __future__ import annotations

import numpy as np
import pydrake.all as ad


def get_jacobian(plant, ctx, ee_frame) -> np.ndarray:
    """Translational-velocity Jacobian (3, n_v) of ee_frame origin,
    expressed in world frame, evaluated at the current ctx state."""
    return plant.CalcJacobianTranslationalVelocity(
        ctx,
        ad.JacobianWrtVariable.kV,
        ee_frame,
        np.zeros(3),
        plant.world_frame(),
        plant.world_frame(),
    )


def get_bias_acceleration(plant, ctx, ee_frame) -> np.ndarray:
    """Translational bias acceleration J̇·v (3,) at ee_frame origin
    in world frame, evaluated at the current ctx state."""
    return plant.CalcBiasTranslationalAcceleration(
        ctx,
        ad.JacobianWrtVariable.kV,
        ee_frame,
        np.zeros(3),
        plant.world_frame(),
        plant.world_frame(),
    ).flatten()


def get_mass_matrix(plant, ctx) -> np.ndarray:
    """Generalized mass matrix M(q) ∈ ℝ^{n_v × n_v}, symmetric PSD."""
    return plant.CalcMassMatrix(ctx)


def get_coriolis_centrifugal(plant, ctx) -> np.ndarray:
    """C(q,v)·v (n_v,) — Coriolis + centrifugal bundle, *not* J̇·v."""
    return plant.CalcBiasTerm(ctx)


def get_gravity(plant, ctx) -> np.ndarray:
    """Generalized gravity force (n_v,).

    Drake convention: ``CalcGravityGeneralizedForces`` returns the
    *generalized force from gravity*, so the equation of motion is::

        M(q)·v̇ + C(q,v)·v = τ_app + τ_g

    where τ_app is applied (actuator + external) and τ_g is the
    gravity term returned here. To balance gravity statically with
    applied torque τ_arm, we need τ_arm = −τ_g[arm].
    """
    return plant.CalcGravityGeneralizedForces(ctx)


def get_actuation_matrix(plant) -> np.ndarray:
    """Actuation map B ∈ ℝ^{n_v × n_a}. For the Franka model here,
    B is the selection matrix that maps the 7 arm actuators into the
    n_v-dim velocity space (manipuland DOFs are unactuated).

    Uses ``MakeActuationMatrix`` so this works generically if the
    plant changes (e.g., a different model instance ordering).
    """
    return plant.MakeActuationMatrix()
