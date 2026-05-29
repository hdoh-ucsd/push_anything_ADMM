"""Operational Space Controller — QP-based executor.

Per-tick QP minimizing weighted (task tracking + posture + torque/accel
regularization) subject to the dynamics equality and per-joint URDF
effort limits. Torque saturation is handled INSIDE the QP via box
constraints, preserving task tracking quality when one joint clips.

The QP treats the planner's λ_planned as a known external force on the
RHS of the dynamics constraint (see `qp_builder.py` docstring for the
sign convention).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml

from control.osc.dynamics_helpers import (
    actuation_matrix,
    bias_term,
    ee_jacobian_bias,
    ee_jacobian_translational,
    ee_position,
    franka_effort_limits,
    gravity_forces,
    mass_matrix,
)
from control.osc.qp_builder import OscGains, OscLimits, build_and_solve_qp
from pydrake.solvers import OsqpSolver


def _load_osc_gains(yaml_path: str | Path, n_arm: int) -> tuple[OscGains, np.ndarray]:
    """Load OSC gains and torque-limit override from YAML.

    Returns (gains, tau_max_override). If `tau_max` key is missing, the
    caller falls back to URDF effort limits.
    """
    with open(yaml_path) as f:
        raw = yaml.safe_load(f) or {}
    osc = raw.get("osc", raw)   # accept either {"osc": {...}} or flat
    gains = OscGains(
        Kp_cart   = np.asarray(osc["Kp_cart"], dtype=float).reshape(3),
        Kd_cart   = np.asarray(osc["Kd_cart"], dtype=float).reshape(3),
        Kp_null   = np.asarray(osc["Kp_null"], dtype=float).reshape(n_arm),
        Kd_null   = np.asarray(osc["Kd_null"], dtype=float).reshape(n_arm),
        W_track   = float(osc["W_track"]),
        W_posture = float(osc["W_posture"]),
        W_torque  = float(osc["W_torque"]),
        W_acc     = float(osc["W_acc"]),
    )
    tau_max = osc.get("tau_max", None)
    if tau_max is not None:
        tau_max = np.asarray(tau_max, dtype=float).reshape(n_arm)
    return gains, tau_max


class OperationalSpaceController:
    """QP-based OSC executor for the Franka arm."""

    def __init__(self,
                 plant,
                 ee_frame,
                 n_arm_dofs:   int,
                 q_nominal:    np.ndarray,
                 gains_yaml:   str | Path,
                 torque_limit_override: Optional[float] = None,
                 log_diag:     bool = True,
                 use_force_tracking: bool = False,
                 W_force:      Optional[float] = None):
        """
        Parameters
        ----------
        plant         : Drake MultibodyPlant.
        ee_frame      : Frame whose origin defines the EE Cartesian target.
        n_arm_dofs    : Number of arm joints (7 for Franka).
        q_nominal     : (n_arm,) posture target for the nullspace cost.
        gains_yaml    : Path to OSC gains YAML (config/osc_franka.yaml).
        torque_limit_override : If set, overrides the URDF/yaml per-joint
                                limits with a single uniform value.
        """
        self.plant       = plant
        self.ee_frame    = ee_frame
        self.world_frame = plant.world_frame()
        self.n_arm       = int(n_arm_dofs)
        self.q_nominal   = np.asarray(q_nominal, dtype=float).reshape(self.n_arm)
        self.log_diag    = bool(log_diag)
        self.use_force_tracking = bool(use_force_tracking)

        # Load gains
        self.gains, _tau_max_yaml = _load_osc_gains(gains_yaml, self.n_arm)
        if W_force is not None:
            self.gains.W_force = float(W_force)

        # Resolve torque limits with precedence: override > yaml > URDF
        if torque_limit_override is not None:
            tau_max = np.full(self.n_arm, float(torque_limit_override))
        elif _tau_max_yaml is not None:
            tau_max = _tau_max_yaml
        else:
            tau_max = franka_effort_limits(plant)[:self.n_arm]
        self.limits = OscLimits(tau_max=tau_max)

        # Cache constant B matrix
        self._B = actuation_matrix(plant)   # (n_v, n_u)

        # Solver instance (reused across calls for warm starts if OSQP
        # internals support it — even without explicit warm-start API,
        # avoiding repeated allocation is cheaper).
        self._solver = OsqpSolver()

        # Diagnostic counters
        self._qp_failures = 0
        self._saturation_events = 0
        self._n_calls = 0
        self._total_solve_ms = 0.0
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
                       lambda_des:   Optional[np.ndarray] = None,
                       ) -> Tuple[np.ndarray, dict]:
        """Compute joint torques via QP. Returns (u ∈ ℝ⁷, diag dict)."""
        plant = self.plant
        plant.SetPositions(plant_ctx, current_q)
        plant.SetVelocities(plant_ctx, current_v)
        n_arm = self.n_arm
        n_v   = plant.num_velocities()

        # --- Drake plant queries (current state) ---
        M       = mass_matrix(plant, plant_ctx)                  # (n_v, n_v)
        Cv      = bias_term(plant, plant_ctx)                    # (n_v,)
        g       = gravity_forces(plant, plant_ctx)               # (n_v,)
        J_v     = ee_jacobian_translational(plant, plant_ctx,
                                            self.ee_frame)        # (3, n_v)
        Jdot_v_v= ee_jacobian_bias(plant, plant_ctx,
                                   self.ee_frame)                # (3,)

        # Singular gravity-comp ownership: the main loop adds tau_g to the
        # actuation port (see main.py around the FixValue call), so the QP
        # treats gravity as cancelled. bias = Cv (no -g term). Dynamics
        # constraint then becomes M v̇ + Cv = B u + F_external, and u is
        # the task-only torque (no implicit gravity-comp).
        bias = Cv

        # --- EE Cartesian state ---
        p_ee_now = ee_position(plant, plant_ctx, self.ee_frame)  # (3,)
        v_ee_now = J_v @ current_v                                # (3,)

        # --- Errors ---
        p_err = np.asarray(p_ee_desired, dtype=float).reshape(3) - p_ee_now
        if v_ee_desired is None:
            v_err = -v_ee_now
        else:
            v_err = np.asarray(v_ee_desired, dtype=float).reshape(3) - v_ee_now

        q_arm = current_q[:n_arm]
        v_arm = current_v[:n_arm]
        q_arm_err = self.q_nominal - q_arm
        v_arm_err = -v_arm

        # --- Feedforward contact force from planner ---
        # Sign convention: λ_n ≥ 0 (Stewart-Trinkle), J_n built from
        # nhat_BA · (J_A − J_B). The term +J^T λ on the RHS of dynamics
        # pushes box in goal direction (good) and reacts on the arm
        # (the QP must overcome this with τ).
        F_ff = np.zeros(n_v)
        had_lam_n = (lambda_n is not None and J_n is not None
                     and J_n.size > 0 and lambda_n.size > 0)
        had_lam_t = (lambda_t is not None and J_t is not None
                     and J_t.size > 0 and lambda_t.size > 0)
        if had_lam_n:
            F_ff += J_n.T @ lambda_n
        if had_lam_t:
            F_ff += J_t.T @ lambda_t

        # In force-tracking mode the executor commands the EE contact force
        # via the λ_ext decision variable rather than the fixed F_ff RHS.
        # Zero F_ff in that mode so the two paths don't double-count the
        # planner's contact reaction.
        if self.use_force_tracking:
            F_ff_for_qp = np.zeros(n_v)
        else:
            F_ff_for_qp = F_ff

        # --- Build & solve QP ---
        t0 = time.perf_counter()
        u_opt, vdot_opt, success, result_str, lam_ext_opt = build_and_solve_qp(
            M=M, bias=bias, B=self._B, n_arm=n_arm,
            J_v=J_v, Jdot_v_v=Jdot_v_v,
            p_err=p_err, v_err=v_err,
            q_arm_err=q_arm_err, v_arm_err=v_arm_err,
            gains=self.gains, limits=self.limits,
            F_ff_external=F_ff_for_qp, solver=self._solver,
            use_force_tracking=self.use_force_tracking,
            lambda_des=lambda_des,
        )
        solve_ms = (time.perf_counter() - t0) * 1000.0
        self._total_solve_ms += solve_ms
        self._n_calls += 1
        if not success:
            self._qp_failures += 1


        # Saturation = any joint hit its box constraint within tolerance
        saturated = bool(np.any(
            np.abs(np.abs(u_opt) - self.limits.tau_max) < 1e-6
        ))
        if saturated:
            self._saturation_events += 1

        if not self._printed_setup:
            self._printed_setup = True
            print(f"[OSC-INIT] n_arm={n_arm} n_v={n_v}")
            print(f"[OSC-INIT]   Kp_cart={self.gains.Kp_cart.tolist()}  "
                  f"Kd_cart={self.gains.Kd_cart.tolist()}")
            print(f"[OSC-INIT]   Kp_null={self.gains.Kp_null.tolist()}")
            print(f"[OSC-INIT]   Kd_null={self.gains.Kd_null.tolist()}")
            print(f"[OSC-INIT]   W_track={self.gains.W_track}  "
                  f"W_posture={self.gains.W_posture}  "
                  f"W_torque={self.gains.W_torque}  W_acc={self.gains.W_acc}")
            print(f"[OSC-INIT]   tau_max={self.limits.tau_max.tolist()}")
            print(f"[OSC-INIT]   q_nominal={np.round(self.q_nominal, 4).tolist()}")
            print(f"[OSC-INIT]   use_force_tracking={self.use_force_tracking}  "
                  f"W_force={self.gains.W_force}")

        # τ_ff diagnostic — the EE-arm slice of the feedforward force,
        # signed so it represents the joint torque needed to counter the
        # planned contact reaction. In force-tracking mode the planner's
        # reaction enters the QP as a cost on λ_ext (not as F_ff), so
        # report the *equivalent* tau_ff produced by the solved λ_ext via J_v.
        if self.use_force_tracking:
            tau_ff_equiv = -(J_v.T @ lam_ext_opt)[:n_arm]
        else:
            tau_ff_equiv = -F_ff[:n_arm] if (had_lam_n or had_lam_t) else np.zeros(n_arm)

        lam_des_arr = (np.zeros(3) if lambda_des is None
                       else np.asarray(lambda_des, dtype=float).reshape(3))

        diag = dict(
            p_ee_now    = p_ee_now,
            v_ee_now    = v_ee_now,
            x_err       = p_err,
            xdot_err    = v_err,
            tau_imp     = u_opt,      # legacy alias kept for downstream loggers
            tau_ff      = tau_ff_equiv,
            tau_out     = u_opt,
            vdot_opt    = vdot_opt,
            saturated   = saturated,
            had_lambda_n= had_lam_n,
            had_lambda_t= had_lam_t,
            qp_success  = success,
            qp_result   = result_str,
            solve_ms    = solve_ms,
            lambda_des  = lam_des_arr,
            lambda_ext  = lam_ext_opt,
            use_force_tracking = self.use_force_tracking,
        )
        return u_opt, diag

    # ------------------------------------------------------------------
    def print_summary(self) -> None:
        """Print end-of-run diagnostic line."""
        if self._n_calls == 0:
            return
        avg_ms = self._total_solve_ms / self._n_calls
        print(f"[OSC-SUMMARY] calls={self._n_calls}  "
              f"qp_failures={self._qp_failures} "
              f"({100.0*self._qp_failures/self._n_calls:.2f}%)  "
              f"saturation={self._saturation_events} "
              f"({100.0*self._saturation_events/self._n_calls:.2f}%)  "
              f"avg_solve_ms={avg_ms:.2f}")
