"""
Cartesian impedance controller — Aydinoglu 2024 §VII eq. (36).

    u = J_f^T · Λ · (K_p · x̃ + K_d · ẋ̃)        ← Cartesian impedance
        + Cf                                       ← Coriolis + gravity comp
        + J_c^T · λ_d                              ← feedforward contact force
        + N_f · (K_n · (q_d - q) + B_n · (q̇_d - q̇))   ← nullspace posture

Where:
  J_f    — (3, n_arm) translational EE Jacobian (Drake CalcJacobianTranslationalVelocity)
  Λ      — operational-space inertia (J_f · M_arm⁻¹ · J_f^T)⁻¹
  Cf     — full-order bias term: Coriolis (+) − gravity_generalized_force (i.e. +gravity comp)
  J_c    — (n_λ, n_arm) contact Jacobian at the current state, arm slice of formulator J_n/J_t
  λ_d    — first-horizon contact force from C3+'s plan; γ-slack is excluded
  N_f    — I − J_f^T · (J_f^T)⁺ — nullspace projector
  q_d, q̇_d — nominal posture and zero velocity for the nullspace term

The controller drives the 7-DoF Franka arm only — the floating-base
manipuland is not in the joint command. Drake returns full-order
(n_v, n_v) mass matrices and (n_v,) bias/gravity vectors; the
:n_arm slice gives the arm rows we apply torque to.

Notes on sign / unit conventions:
  - Drake's CalcGravityGeneralizedForces returns the generalized force
    gravity EXERTS (positive = drives joint along +ve). To compensate,
    we negate. So gravity-compensation torque = −CalcGravityGeneralizedForces.
    This matches the existing PiecewiseLinearTracker / RepositionIKTracker
    convention (tau_g_arm = −CalcGravity[:n_arm]); see
    scripts/test_gravity_sign.py for the clean-room verification.
  - λ_n ≥ 0 (Stewart-Trinkle convention); J_n is "nhat onto box".
    J_n^T λ_n applied to the FULL plant pushes the box (+ direction)
    AND reacts on the arm (− direction). Taking the [:n_arm] slice
    gives the arm reaction — applied as joint torque, it produces
    EE force INTO the box, which is what we want for τ_ff.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pydrake.all as ad


class ImpedanceController:
    """Aydinoglu §VII eq. (36) executor — replaces joint-PD-with-grav-comp."""

    def __init__(self,
                 plant,
                 ee_frame,
                 n_arm_dofs:   int,
                 Kp_cart:      np.ndarray,
                 Kd_cart:      np.ndarray,
                 Kp_null:      np.ndarray,
                 Kd_null:      np.ndarray,
                 q_nominal:    np.ndarray,
                 torque_limit: float = 30.0):
        """
        Parameters
        ----------
        plant        : Drake MultibodyPlant.
        ee_frame     : Frame whose origin defines the EE Cartesian target.
        n_arm_dofs   : Number of arm joints (7 for Franka).
        Kp_cart      : (3,) Cartesian position stiffness [N/m].
        Kd_cart      : (3,) Cartesian damping [N·s/m].
        Kp_null      : (n_arm_dofs,) nullspace posture stiffness.
        Kd_null      : (n_arm_dofs,) nullspace damping.
        q_nominal    : (n_arm_dofs,) nominal posture for nullspace term.
        torque_limit : Per-joint saturation (Nm).
        """
        self.plant         = plant
        self.ee_frame      = ee_frame
        self.world_frame   = plant.world_frame()
        self.n_arm         = int(n_arm_dofs)
        self.Kp_cart       = np.asarray(Kp_cart, dtype=float).reshape(3)
        self.Kd_cart       = np.asarray(Kd_cart, dtype=float).reshape(3)
        self.Kp_null       = np.asarray(Kp_null, dtype=float).reshape(self.n_arm)
        self.Kd_null       = np.asarray(Kd_null, dtype=float).reshape(self.n_arm)
        self.q_nominal     = np.asarray(q_nominal, dtype=float).reshape(self.n_arm)
        self.torque_limit  = float(torque_limit)
        # One-shot diagnostic print at first compute_torque call.
        self._printed_setup = False

    # ------------------------------------------------------------------
    def compute_torque(self,
                       current_q:    np.ndarray,
                       current_v:    np.ndarray,
                       plant_ctx,
                       p_ee_desired: np.ndarray,
                       v_ee_desired: Optional[np.ndarray] = None,
                       lambda_n:     Optional[np.ndarray] = None,
                       lambda_t:     Optional[np.ndarray] = None,
                       J_n:          Optional[np.ndarray] = None,
                       J_t:          Optional[np.ndarray] = None,
                       ) -> Tuple[np.ndarray, dict]:
        """Compute 7-DoF arm torque per Aydinoglu eq. (36).

        ``lambda_n`` / ``J_n`` and ``lambda_t`` / ``J_t`` come from C3+'s
        plan and the LCS formulator's most recent linearization at the
        CURRENT state. When in free mode (no contact planned), pass
        all four as None — τ_ff drops to zero.
        """
        plant = self.plant
        plant.SetPositions(plant_ctx, current_q)
        plant.SetVelocities(plant_ctx, current_v)
        n_arm = self.n_arm

        # 1. Current EE position (3,)
        p_ee_now = plant.CalcPointsPositions(
            plant_ctx, self.ee_frame, np.zeros(3), self.world_frame,
        ).flatten()

        # 2. Translational EE Jacobian (3, n_v) — arm slice (3, n_arm)
        J_full = plant.CalcJacobianTranslationalVelocity(
            plant_ctx, ad.JacobianWrtVariable.kV,
            self.ee_frame, np.zeros(3),
            self.world_frame, self.world_frame,
        )
        J_arm = J_full[:, :n_arm]

        # 3. EE velocity from arm-side joint velocities
        v_arm    = current_v[:n_arm]
        v_ee_now = J_arm @ v_arm                                # (3,)

        # 4. Cartesian error
        x_err = np.asarray(p_ee_desired, dtype=float).reshape(3) - p_ee_now
        if v_ee_desired is None:
            xdot_err = -v_ee_now
        else:
            xdot_err = np.asarray(v_ee_desired, dtype=float).reshape(3) - v_ee_now

        # 5. Operational-space inertia Λ = (J · M⁻¹ · J^T)⁻¹
        M_full = plant.CalcMassMatrix(plant_ctx)                # (n_v, n_v)
        M_arm  = M_full[:n_arm, :n_arm]
        try:
            M_arm_inv = np.linalg.inv(M_arm)
        except np.linalg.LinAlgError:
            M_arm_inv = np.linalg.pinv(M_arm)
        JMJt   = J_arm @ M_arm_inv @ J_arm.T + 1e-6 * np.eye(3)
        Lambda = np.linalg.inv(JMJt)                            # (3, 3)

        # 6. Cartesian impedance feedback → joint torque
        F_cart        = self.Kp_cart * x_err + self.Kd_cart * xdot_err  # (3,)
        tau_impedance = J_arm.T @ Lambda @ F_cart                       # (n_arm,)

        # 7. Bias term: Coriolis + gravity compensation (arm slice).
        #    Drake returns generalized gravity FORCE; we negate it to get
        #    compensation torque (matches existing tracker convention).
        bias_full = plant.CalcBiasTerm(plant_ctx)                       # (n_v,)
        g_full    = plant.CalcGravityGeneralizedForces(plant_ctx)       # (n_v,)
        tau_bias  = bias_full[:n_arm] - g_full[:n_arm]                  # (n_arm,)

        # 8. Feedforward contact-force τ_ff (arm slice).
        #
        # Sign derivation (Stewart-Trinkle / Drake nhat_BA_W convention):
        #   J_n_row = nhat_BA · (J_A − J_B). Working through both A=EE,B=box
        #   and A=box,B=EE shows
        #       (J_n^T λ_n)[:n_arm] = −J_EE_arm^T · nhat_onto_box · λ_n
        #   i.e. the joint-torque equivalent of the RECOIL force on the EE
        #   (away from box). To make the actuator APPLY the planned contact
        #   force λ_d INTO the box, we must negate. Same logic for λ_t.
        #   See scripts/test_jacobian_conditioning.py for the empirical check.
        tau_ff = np.zeros(n_arm)
        had_lam_n = (lambda_n is not None and J_n is not None
                     and J_n.size > 0 and lambda_n.size > 0)
        had_lam_t = (lambda_t is not None and J_t is not None
                     and J_t.size > 0 and lambda_t.size > 0)
        if had_lam_n:
            tau_ff += -(J_n.T @ lambda_n)[:n_arm]
        if had_lam_t:
            tau_ff += -(J_t.T @ lambda_t)[:n_arm]

        # 9. Nullspace posture: project (K_p·(q*-q) + K_d·(0-v)) onto null(J_arm^T).
        JT      = J_arm.T                                       # (n_arm, 3)
        JT_pinv = np.linalg.pinv(JT)                            # (3, n_arm)
        N_null  = np.eye(n_arm) - JT @ JT_pinv                  # (n_arm, n_arm)
        q_arm   = current_q[:n_arm]
        posture_err = self.q_nominal - q_arm
        vel_err     = -v_arm
        tau_null    = N_null @ (self.Kp_null * posture_err
                                + self.Kd_null * vel_err)

        # 10. Combine and clip
        tau_raw = tau_impedance + tau_bias + tau_ff + tau_null
        tau_out = np.clip(tau_raw, -self.torque_limit, +self.torque_limit)

        if not self._printed_setup:
            self._printed_setup = True
            print(f"[IMP-INIT] n_arm={n_arm} torque_limit={self.torque_limit}Nm")
            print(f"[IMP-INIT]   Kp_cart={self.Kp_cart.tolist()}  "
                  f"Kd_cart={self.Kd_cart.tolist()}")
            print(f"[IMP-INIT]   Kp_null={self.Kp_null.tolist()}")
            print(f"[IMP-INIT]   Kd_null={self.Kd_null.tolist()}")
            print(f"[IMP-INIT]   q_nominal={np.round(self.q_nominal, 4).tolist()}")

        diag = dict(
            p_ee_now    = p_ee_now,
            v_ee_now    = v_ee_now,
            x_err       = x_err,
            xdot_err    = xdot_err,
            F_cart      = F_cart,
            tau_imp     = tau_impedance,
            tau_bias    = tau_bias,
            tau_ff      = tau_ff,
            tau_null    = tau_null,
            tau_raw     = tau_raw,
            tau_out     = tau_out,
            saturated   = bool(np.any(np.abs(tau_raw) >= self.torque_limit - 1e-9)),
            had_lambda_n= had_lam_n,
            had_lambda_t= had_lam_t,
        )
        return tau_out, diag
