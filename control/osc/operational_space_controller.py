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
    ee_jacobian_angular,
    ee_jacobian_angular_bias,
    ee_jacobian_bias,
    ee_jacobian_translational,
    ee_position,
    ee_rotation,
    franka_effort_limits,
    gravity_forces,
    mass_matrix,
    rotation_error_world,
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
        Kp_joint2 = float(osc.get("Kp_joint2", 0.0)),
        Kd_joint2 = float(osc.get("Kd_joint2", 0.0)),
        W_joint2  = float(osc.get("W_joint2", 0.0)),
        joint2_target_rad = float(osc.get("joint2_target_rad", 1.1)),
        joint2_idx = int(osc.get("joint2_idx", 1)),
        Kp_rot    = (np.asarray(osc["Kp_rot"], dtype=float).reshape(3)
                     if "Kp_rot" in osc else None),
        Kd_rot    = (np.asarray(osc["Kd_rot"], dtype=float).reshape(3)
                     if "Kd_rot" in osc else None),
        W_rot     = float(osc.get("W_rot", 0.0)),
        a_ee_cap  = float(osc.get("end_effector_acceleration", 0.0)),
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

        # 2026-07-28 defaults flip: the yaml IS the reference gain set
        # (W_track=1, Kp=200, Kd=20 since fc51111; W_rot=10/800/40 since the
        # flip) — the former REFCONF_OSC_ALIGN / REFCONF_OSC_C3_MODE_GAINS /
        # REFCONF_OSC_FREE_MODE_GAINS / REFCONF_OSC_EE_ROT_TASK env gates are
        # removed. ONE gain set for all modes, exactly like the reference
        # (osc_params.yaml + franka_osc_controller.cc:171-187, which adds
        # the RotTaskSpaceTrackingData unconditionally with a constant
        # identity-quaternion target per end_effector_orientation.cc:49-57).

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
        # Rotation-hold target R_WE_target = CONSTANT IDENTITY — the exact
        # reference semantics (end_effector_orientation.cc:49-57 emits a
        # constant identity-quaternion slerp; franka_osc_controller.cc:180
        # orientation_target=(1,0,0,0)). Valid verbatim in the port frame:
        # the EE weld chain is byte-identical to the reference (link7 +
        # roll π @ kToolAttachmentFrame → flange → peg → tip), so the tip
        # frame is 0.01° from identity at working pointing-down poses
        # (measured 2026-07-28, mount-question close-out; replaces the
        # first-tick-snapshot adaptation).
        self._R_target = None  # resolved to identity at first use below

        # QP-signature dump hook. Env-gated, byte-identical when unset.
        # DIAG_QP_SIG_DUMP=1 → capture the full input tuple to the QP at
        # compute_torque call index DIAG_QP_SIG_STEP (default 60). Written
        # to DIAG_QP_SIG_DIR (default audit_output/exec_qp_sig/) as
        # dump_call{N}.{npz,txt}. Consumed by
        # scripts/_qp_sig_reference_emulator.py to produce a same-input
        # τ diff against a reference-formula Python emulator.
        import os as _os
        if _os.environ.get("DIAG_QP_SIG_DUMP", "0") == "1":
            self._sig_dump_step = int(
                _os.environ.get("DIAG_QP_SIG_STEP", "60"))
            self._sig_dump_dir = _os.environ.get(
                "DIAG_QP_SIG_DIR", "audit_output/exec_qp_sig")
            self._sig_dump_done = False
            print(f"[QP-SIG] enabled: will capture compute_torque call "
                  f"idx={self._sig_dump_step} → {self._sig_dump_dir}/",
                  flush=True)
        else:
            self._sig_dump_step = None
            self._sig_dump_done = False

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
                       a_ee_desired: Optional[np.ndarray] = None,
                       mode:         str = "free",
                       q_nominal_override: Optional[np.ndarray] = None,
                       ) -> Tuple[np.ndarray, dict]:
        """Compute joint torques via QP. Returns (u ∈ ℝ⁷, diag dict).

        `mode`: "c3" or "free". When REFCONF_OSC_C3_MODE_GAINS=1
        AND mode="c3", the QP is built with reference-aligned gains
        (Kp=200, W_track=1) instead of the port defaults (Kp=400,
        W_track=100). Free/repos calls (mode="free") always use port
        gains — this keeps §7.47's IK→c3 handoff mechanism intact.
        Default mode="free" preserves byte-identical behavior when
        callers haven't been plumbed.
        """
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

        # Tune-3: back to port's task-only bias (Cv only). main.py's
        # `tau_g[:n_u] + u_opt` universal-add owns gravity comp. Restores
        # dd2294d proven closure. 1.d divergence stays deferred.
        bias = Cv

        # --- EE Cartesian state ---
        p_ee_now = ee_position(plant, plant_ctx, self.ee_frame)  # (3,)
        v_ee_now = J_v @ current_v                                # (3,)

        # --- 2.k rotation state (only assembled when W_rot > 0) ---
        _use_rot = float(getattr(self.gains, "W_rot", 0.0)) > 0.0
        if _use_rot:
            J_w      = ee_jacobian_angular(plant, plant_ctx, self.ee_frame)   # (3, n_v)
            Jdot_w_v = ee_jacobian_angular_bias(plant, plant_ctx, self.ee_frame)
            R_now    = ee_rotation(plant, plant_ctx, self.ee_frame)
            if self._R_target is None:
                # Constant IDENTITY target — reference-exact (see the
                # constructor note; the former first-tick snapshot is
                # replaced now that the weld-chain conformance makes
                # identity ≡ pusher-down in the port frame too).
                from pydrake.math import RotationMatrix as _RotM
                self._R_target = _RotM()
            # Reference rot_space_tracking_data.cc:60-68 UpdateYError:
            # exact angle-axis (log map) of R_target · R_now⁻¹, world frame.
            w_err = rotation_error_world(self._R_target, R_now)
            w_ee_now = J_w @ current_v
        else:
            J_w = Jdot_w_v = w_err = w_ee_now = None

        # --- Errors ---
        # Non-finite desired-state sanitize (defense-in-depth; the known
        # NaN source — degenerate RepositionTrajectory legs — is fixed at
        # origin, but any NaN reaching the QP costs makes OSQP fail with
        # kIterationLimit and the executor emit zero torque).
        if v_ee_desired is not None and not np.all(
                np.isfinite(np.asarray(v_ee_desired, dtype=float))):
            v_ee_desired = None
        if a_ee_desired is not None and not np.all(
                np.isfinite(np.asarray(a_ee_desired, dtype=float))):
            a_ee_desired = None
        p_err = np.asarray(p_ee_desired, dtype=float).reshape(3) - p_ee_now
        if v_ee_desired is None:
            v_err = -v_ee_now
        else:
            v_err = np.asarray(v_ee_desired, dtype=float).reshape(3) - v_ee_now

        q_arm = current_q[:n_arm]
        v_arm = current_v[:n_arm]
        # Posture target: `q_nominal_override` (per-tick, from IK-projected
        # geometric guidance) takes precedence over the constructor's
        # `q_nominal`. Falls back to self.q_nominal when not supplied.
        if q_nominal_override is not None:
            _q_post = np.asarray(q_nominal_override, dtype=float).reshape(n_arm)
        else:
            _q_post = self.q_nominal
        q_arm_err = _q_post - q_arm
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

        # ONE gain set for all modes (reference osc_params.yaml semantics;
        # 2026-07-28 defaults flip removed the §7.70 c3/free gain-swap
        # machinery — the yaml is the reference set).
        _gains_active = self.gains

        # --- QP-signature dump hook (env-gated, byte-identical when off) ---
        _sig_do_dump = (self._sig_dump_step is not None
                        and not self._sig_dump_done
                        and self._n_calls == self._sig_dump_step)
        if _sig_do_dump:
            _sig_inputs = dict(
                n_calls_idx=int(self._n_calls),
                mode=str(mode),
                use_force_tracking=bool(self.use_force_tracking),
                c3_ref_gains_active=True,   # one reference gain set, all modes
                # State
                q=np.asarray(current_q, dtype=float).copy(),
                v=np.asarray(current_v, dtype=float).copy(),
                p_ee_now=p_ee_now.astype(float).copy(),
                v_ee_now=v_ee_now.astype(float).copy(),
                # Desired
                p_ee_desired=np.asarray(p_ee_desired, dtype=float).reshape(3).copy(),
                v_ee_desired=(np.zeros(3) if v_ee_desired is None
                              else np.asarray(v_ee_desired, dtype=float).reshape(3).copy()),
                a_ee_desired=(np.zeros(3) if a_ee_desired is None
                              else np.asarray(a_ee_desired, dtype=float).reshape(3).copy()),
                lambda_des=(np.zeros(3) if lambda_des is None
                            else np.asarray(lambda_des, dtype=float).reshape(3).copy()),
                v_ee_desired_present=(v_ee_desired is not None),
                a_ee_desired_present=(a_ee_desired is not None),
                lambda_des_present=(lambda_des is not None),
                # Errors
                p_err=p_err.astype(float).copy(),
                v_err=v_err.astype(float).copy(),
                q_arm_err=q_arm_err.astype(float).copy(),
                v_arm_err=v_arm_err.astype(float).copy(),
                q_arm=q_arm.astype(float).copy(),
                v_arm=v_arm.astype(float).copy(),
                # Dynamics tuple (Drake queries)
                M=M.astype(float).copy(),
                Cv=Cv.astype(float).copy(),
                gravity=g.astype(float).copy(),
                bias=bias.astype(float).copy(),
                B=self._B.astype(float).copy(),
                J_v=J_v.astype(float).copy(),
                Jdot_v_v=Jdot_v_v.astype(float).copy(),
                # Planner feedforward
                F_ff=F_ff.astype(float).copy(),
                F_ff_for_qp=F_ff_for_qp.astype(float).copy(),
                had_lam_n=bool(had_lam_n),
                had_lam_t=bool(had_lam_t),
                # Gains (from _gains_active — what the QP actually consumes)
                Kp_cart=np.asarray(_gains_active.Kp_cart, dtype=float).copy(),
                Kd_cart=np.asarray(_gains_active.Kd_cart, dtype=float).copy(),
                Kp_null=np.asarray(_gains_active.Kp_null, dtype=float).copy(),
                Kd_null=np.asarray(_gains_active.Kd_null, dtype=float).copy(),
                W_track=float(_gains_active.W_track),
                W_posture=float(_gains_active.W_posture),
                W_torque=float(_gains_active.W_torque),
                W_acc=float(_gains_active.W_acc),
                W_force=float(_gains_active.W_force),
                Kp_joint2=float(getattr(_gains_active, "Kp_joint2", 0.0)),
                Kd_joint2=float(getattr(_gains_active, "Kd_joint2", 0.0)),
                W_joint2=float(getattr(_gains_active, "W_joint2", 0.0)),
                joint2_target_rad=float(
                    getattr(_gains_active, "joint2_target_rad", 0.0)),
                joint2_idx=int(getattr(_gains_active, "joint2_idx", -1)),
                # Rotation tracking (2.k) — cost fires when W_rot>0 in port,
                # matches reference EndEffectorRotW·Rot{Kp,Kd}.
                W_rot=float(getattr(_gains_active, "W_rot", 0.0)),
                Kp_rot=(np.zeros(3) if getattr(_gains_active, "Kp_rot", None) is None
                        else np.asarray(_gains_active.Kp_rot, dtype=float).copy()),
                Kd_rot=(np.zeros(3) if getattr(_gains_active, "Kd_rot", None) is None
                        else np.asarray(_gains_active.Kd_rot, dtype=float).copy()),
                J_w=(np.zeros((3, n_v)) if J_w is None else J_w.astype(float).copy()),
                Jdot_w_v=(np.zeros(3) if Jdot_w_v is None else Jdot_w_v.astype(float).copy()),
                w_err=(np.zeros(3) if w_err is None else np.asarray(w_err, dtype=float).copy()),
                w_ee_now=(np.zeros(3) if w_ee_now is None else np.asarray(w_ee_now, dtype=float).copy()),
                rot_active=bool(_use_rot),
                tau_max=self.limits.tau_max.astype(float).copy(),
                # Meta
                n_arm=int(n_arm),
                n_v=int(n_v),
            )

        # --- Build & solve QP ---
        t0 = time.perf_counter()
        u_opt, vdot_opt, success, result_str, lam_ext_opt = build_and_solve_qp(
            M=M, bias=bias, B=self._B, n_arm=n_arm,
            J_v=J_v, Jdot_v_v=Jdot_v_v,
            p_err=p_err, v_err=v_err,
            q_arm_err=q_arm_err, v_arm_err=v_arm_err,
            gains=_gains_active, limits=self.limits,
            F_ff_external=F_ff_for_qp, solver=self._solver,
            use_force_tracking=self.use_force_tracking,
            lambda_des=lambda_des,
            a_ff=a_ee_desired,
            J_w=J_w, Jdot_w_v=Jdot_w_v,
            w_err=w_err, w_ee_now=w_ee_now,
            q_arm=q_arm,
            v_arm=v_arm,
        )
        solve_ms = (time.perf_counter() - t0) * 1000.0
        self._total_solve_ms += solve_ms
        self._n_calls += 1
        if not success:
            self._qp_failures += 1
            # Rate-limited failure diagnostic (first 8): result string +
            # the error magnitudes feeding the QP, so persistent-failure
            # regimes (e.g. p125 rot-task stall) are attributable from the
            # run log without a signature dump.
            if self._qp_failures <= 8:
                _we = (float(np.linalg.norm(w_err))
                       if w_err is not None else float("nan"))
                print(f"[QP-FAIL] call={self._n_calls} mode={mode} "
                      f"result={result_str} |p_err|={np.linalg.norm(p_err):.4f} "
                      f"|v_err|={np.linalg.norm(v_err):.4f} |w_err|={_we:.4f} "
                      f"W_rot={float(getattr(_gains_active, 'W_rot', 0.0)):.1f}",
                      flush=True)

        # --- QP-signature dump: outputs + write ---
        if _sig_do_dump:
            _sig_outputs = dict(
                u_opt=np.asarray(u_opt, dtype=float).copy(),
                vdot_opt=np.asarray(vdot_opt, dtype=float).copy(),
                lam_ext_opt=np.asarray(lam_ext_opt, dtype=float).copy(),
                qp_success=bool(success),
                qp_result=str(result_str),
                solve_ms=float(solve_ms),
            )
            self._write_qp_sig_dump(_sig_inputs, _sig_outputs)
            self._sig_dump_done = True

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
            print(f"[OSC-INIT]   a_ee_cap={self.gains.a_ee_cap}")

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
        a_ff_arr = (np.zeros(3) if a_ee_desired is None
                    else np.asarray(a_ee_desired, dtype=float).reshape(3))

        diag = dict(
            p_ee_now    = p_ee_now,
            v_ee_now    = v_ee_now,
            x_err       = p_err,
            xdot_err    = v_err,
            a_ff        = a_ff_arr,
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
    def compute_torque_from_trajectory(
            self,
            traj,                           # pydrake PiecewisePolynomial / Trajectory
            t_sim:        float,
            current_q:    np.ndarray,
            current_v:    np.ndarray,
            plant_ctx,
            v_ee_desired: Optional[np.ndarray] = None,
            lambda_n:     Optional[np.ndarray] = None,
            lambda_t:     Optional[np.ndarray] = None,
            J_n:          Optional[np.ndarray] = None,
            J_t:          Optional[np.ndarray] = None,
            lambda_des:   Optional[np.ndarray] = None,
            a_ee_desired: Optional[np.ndarray] = None,
            mode:         str = "free",
            q_nominal_override: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Trajectory-shaped variant of compute_torque.

        `traj` must expose `.value(t_sim) -> np.ndarray(3, 1)` — matches
        Drake's `Trajectory<double>` / `PiecewisePolynomial`. This is the
        reference-parity contract: dairlib's `franka_osc_controller.cc`
        consumes `Trajectory<double>` abstract input ports over LCM.

        For Phase 1 (reproduce-dairlib), the wrapper wraps its per-tick R³
        setpoint in a degenerate single-knot `PiecewisePolynomial.ZeroOrderHold`.
        Phase 2 flips this to the full N-knot PWL from reposition.cc without
        changing the executor's signature.
        """
        p_ee_desired = np.asarray(traj.value(t_sim)).reshape(3)
        # Reference osc_tracking_data.cc:88-108 evaluates y, ẏ, ÿ from the
        # traj every tick. Extract derivatives if the caller hasn't
        # explicitly overridden them.
        if v_ee_desired is None:
            try:
                v_ee_desired = np.asarray(
                    traj.EvalDerivative(t_sim, 1)).reshape(3)
            except Exception:
                v_ee_desired = None
        if a_ee_desired is None:
            try:
                a_ee_desired = np.asarray(
                    traj.EvalDerivative(t_sim, 2)).reshape(3)
            except Exception:
                a_ee_desired = None
        return self.compute_torque(
            current_q=current_q,
            current_v=current_v,
            plant_ctx=plant_ctx,
            p_ee_desired=p_ee_desired,
            v_ee_desired=v_ee_desired,
            lambda_n=lambda_n,
            lambda_t=lambda_t,
            J_n=J_n,
            J_t=J_t,
            lambda_des=lambda_des,
            a_ee_desired=a_ee_desired,
            mode=mode,
            q_nominal_override=q_nominal_override,
        )

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

    # ------------------------------------------------------------------
    def _write_qp_sig_dump(self, inp: dict, out: dict) -> None:
        """Write the QP input+output snapshot for offline signature diff."""
        import os as _os
        _os.makedirs(self._sig_dump_dir, exist_ok=True)
        n = int(inp["n_calls_idx"])
        npz_path = _os.path.join(self._sig_dump_dir, f"dump_call{n}.npz")
        txt_path = _os.path.join(self._sig_dump_dir, f"dump_call{n}.txt")
        payload = {}
        for k, v in inp.items():
            payload[f"in_{k}"] = v
        for k, v in out.items():
            payload[f"out_{k}"] = v
        np.savez(npz_path, **payload)
        with open(txt_path, "w") as f:
            f.write(f"# QP signature dump — compute_torque call idx={n}\n")
            f.write(f"# ee_frame body={self.ee_frame.body().name()!r} "
                    f"offset=[0,0,0]  n_arm={inp['n_arm']} n_v={inp['n_v']}\n")
            f.write(f"# mode={inp['mode']} use_force_tracking={inp['use_force_tracking']} "
                    f"c3_ref_gains_active={inp['c3_ref_gains_active']}\n")
            f.write("# --- gains (as consumed by QP) ---\n")
            for k in ("Kp_cart", "Kd_cart", "W_track", "W_posture", "W_torque",
                      "W_acc", "W_force", "Kp_null", "Kd_null",
                      "Kp_joint2", "Kd_joint2", "W_joint2",
                      "joint2_target_rad", "joint2_idx", "tau_max"):
                f.write(f"  {k} = {inp[k]!r}\n")
            f.write("# --- state ---\n")
            f.write(f"  q       = {np.round(inp['q'], 6).tolist()}\n")
            f.write(f"  v       = {np.round(inp['v'], 6).tolist()}\n")
            f.write(f"  p_ee_now= {np.round(inp['p_ee_now'], 6).tolist()}\n")
            f.write(f"  v_ee_now= {np.round(inp['v_ee_now'], 6).tolist()}\n")
            f.write("# --- desired / errors ---\n")
            f.write(f"  p_ee_desired = {np.round(inp['p_ee_desired'], 6).tolist()}  "
                    f"(present={inp['v_ee_desired_present']})\n")
            f.write(f"  v_ee_desired = {np.round(inp['v_ee_desired'], 6).tolist()}  "
                    f"(present={inp['v_ee_desired_present']})\n")
            f.write(f"  a_ee_desired = {np.round(inp['a_ee_desired'], 6).tolist()}  "
                    f"(present={inp['a_ee_desired_present']})\n")
            f.write(f"  lambda_des   = {np.round(inp['lambda_des'], 6).tolist()}  "
                    f"(present={inp['lambda_des_present']})\n")
            f.write(f"  p_err        = {np.round(inp['p_err'], 6).tolist()}\n")
            f.write(f"  v_err        = {np.round(inp['v_err'], 6).tolist()}\n")
            f.write(f"  q_arm_err    = {np.round(inp['q_arm_err'], 6).tolist()}\n")
            f.write(f"  v_arm_err    = {np.round(inp['v_arm_err'], 6).tolist()}\n")
            f.write("# --- dynamics tuple summary ---\n")
            f.write(f"  Cv[:n_arm]      = {np.round(inp['Cv'][:inp['n_arm']], 4).tolist()}\n")
            f.write(f"  gravity[:n_arm] = {np.round(inp['gravity'][:inp['n_arm']], 4).tolist()}\n")
            f.write(f"  bias[:n_arm]    = {np.round(inp['bias'][:inp['n_arm']], 4).tolist()}\n")
            f.write(f"  F_ff[:n_arm]    = {np.round(inp['F_ff'][:inp['n_arm']], 4).tolist()}\n")
            f.write(f"  F_ff_for_qp[:n_arm]={np.round(inp['F_ff_for_qp'][:inp['n_arm']], 4).tolist()}\n")
            f.write(f"  had_lam_n={inp['had_lam_n']}  had_lam_t={inp['had_lam_t']}\n")
            f.write("# --- QP outputs (port) ---\n")
            f.write(f"  u_opt       = {np.round(out['u_opt'], 4).tolist()}\n")
            f.write(f"  vdot_opt[:n_arm] = {np.round(out['vdot_opt'][:inp['n_arm']], 4).tolist()}\n")
            f.write(f"  lam_ext_opt = {np.round(out['lam_ext_opt'], 4).tolist()}\n")
            f.write(f"  qp_success  = {out['qp_success']}  ({out['qp_result']})\n")
            f.write(f"  solve_ms    = {out['solve_ms']:.3f}\n")
            f.write("# --- plant-side torque (u_opt + tau_g[:n_arm]) ---\n")
            _tau_plant = out['u_opt'] + (-inp['gravity'][:inp['n_arm']])
            f.write(f"  tau_g[:n_arm]      = {np.round(-inp['gravity'][:inp['n_arm']], 4).tolist()}\n")
            f.write(f"  tau_plant (u+tau_g)= {np.round(_tau_plant, 4).tolist()}\n")
        print(f"[QP-SIG] captured call idx={n} → {npz_path}", flush=True)
