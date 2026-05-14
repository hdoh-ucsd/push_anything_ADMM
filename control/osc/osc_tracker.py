"""Synchronous OSC tracker — matches the ``compute_torque`` API of
``PiecewiseLinearTracker`` / ``RepositionIKTracker`` so it can be a
drop-in replacement in ``SamplingC3MPC``'s free-mode path.

Distinct from ``OperationalSpaceController`` (the Drake ``LeafSystem``
variant), which exposes input/output ports for Drake-diagram use.
Both share the per-tick QP math in ``qp_builder.build_osc_qp``.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from pydrake.solvers import OsqpSolver

from control.osc.qp_builder import build_osc_qp


class OperationalSpaceTracker:
    """Drop-in replacement for the PD-based repos trackers. Each call
    rebuilds and solves the OSC QP at the current plant state."""

    def __init__(
        self,
        plant,
        ee_frame,
        n_arm_dofs: int,
        gains_cfg: dict,
    ) -> None:
        self._plant      = plant
        self._ee_frame   = ee_frame
        self._n_arm      = int(n_arm_dofs)
        self._world      = plant.world_frame()

        self._Kp_task        = np.asarray(gains_cfg["Kp_task"], dtype=float)
        self._Kd_task        = np.asarray(gains_cfg["Kd_task"], dtype=float)
        self._Kp_posture     = np.asarray(gains_cfg["Kp_posture"], dtype=float)
        self._Kd_posture     = np.asarray(gains_cfg["Kd_posture"], dtype=float)
        self._W_task         = float(gains_cfg["W_task"])
        self._W_posture      = float(gains_cfg["W_posture"])
        self._torque_limits  = np.asarray(gains_cfg["torque_limits"], dtype=float)
        self._w_tau_reg      = float(gains_cfg.get("w_tau_reg", 1e-4))
        self._q_home_arm     = np.asarray(gains_cfg["q_home_arm"], dtype=float)

        assert self._Kp_task.shape == (3,)
        assert self._Kd_task.shape == (3,)
        assert self._Kp_posture.shape == (self._n_arm,)
        assert self._Kd_posture.shape == (self._n_arm,)
        assert self._torque_limits.shape == (self._n_arm,)
        assert self._q_home_arm.shape == (self._n_arm,)

        self._solver       = OsqpSolver()
        self._last_tau     = np.zeros(self._n_arm)
        self._n_infeas     = 0
        self._step         = 0
        # Default-on for Phase 3 verdict-A diagnostics. Set False to mute.
        self._log_diag     = bool(gains_cfg.get("log_diag", True))
        # Public attribute that wrapper.py reads via getattr.
        self.last_knot0_feasible: bool = True

    # ------------------------------------------------------------------
    def compute_torque(
        self,
        *,
        current_q:  np.ndarray,
        current_v:  np.ndarray,
        plant_ctx,
        p_target:   np.ndarray,
        dt_ctrl:    float,    # noqa: ARG002 — interface compat
    ) -> Tuple[np.ndarray, dict]:
        """Solve one OSC QP and return (tau_arm, diag). ``plant_ctx`` is
        assumed to hold (current_q, current_v) — we set it explicitly to
        be defensive against callers that haven't done so."""
        n_arm = self._n_arm
        self._plant.SetPositions(plant_ctx, current_q)
        self._plant.SetVelocities(plant_ctx, current_v)

        ee_now = self._plant.CalcPointsPositions(
            plant_ctx, self._ee_frame, np.zeros(3), self._world,
        ).flatten()

        # Match the joint-PD tracker's semantics: p_target is a 3D
        # position with implicit zero velocity goal.
        x_des  = np.asarray(p_target, dtype=float).flatten()
        xd_des = np.zeros(3)

        prog, dv = build_osc_qp(
            self._plant, plant_ctx, self._ee_frame,
            x_des=x_des, xd_des=xd_des,
            q_home_arm=self._q_home_arm,
            q_now_arm=current_q[:n_arm],
            v_now_arm=current_v[:n_arm],
            Kp_task=self._Kp_task, Kd_task=self._Kd_task,
            Kp_posture=self._Kp_posture, Kd_posture=self._Kd_posture,
            W_task=self._W_task, W_posture=self._W_posture,
            torque_limits=self._torque_limits,
            w_tau_reg=self._w_tau_reg, n_arm=n_arm,
        )
        result = self._solver.Solve(prog)
        qp_feasible = bool(result.is_success())
        qdd_sol = None
        if qp_feasible:
            self._last_tau = np.asarray(
                result.GetSolution(dv["tau"]), dtype=float,
            )
            qdd_sol = np.asarray(result.GetSolution(dv["qdd"]), dtype=float)
            self.last_knot0_feasible = True
        else:
            # QP shouldn't infeasibility-fail (task is a soft cost), but
            # if OSQP returns a non-success status (e.g., iter cap on a
            # tough state), hold the previous τ.
            self._n_infeas += 1
            self.last_knot0_feasible = False
            if self._n_infeas == 1 or self._n_infeas % 100 == 0:
                print(f"[OSC-tracker] QP infeasible (count={self._n_infeas}); "
                      f"holding last τ. status={result.get_solution_result()}")

        # The wrapper reads ``finished`` to drive the
        # kToC3ReachedReposTarget mode switch; 1 cm matches the PWL
        # tracker's heuristic.
        finished = bool(np.linalg.norm(ee_now - x_des) < 0.01)

        if self._log_diag:
            tau_abs = np.abs(self._last_tau)
            tau_norm = float(np.linalg.norm(self._last_tau))
            clamp_mask = (tau_abs >= self._torque_limits - 1e-6).astype(int)
            err_xy = float(np.linalg.norm(ee_now - x_des))
            print(
                f"[OSC-tick] step={self._step} "
                f"|tau|={tau_norm:.3f} "
                f"clamp_mask={clamp_mask.tolist()} "
                f"qp_feasible={int(qp_feasible)}"
            )
            print(
                f"[OSC-target] step={self._step} "
                f"p_target=({x_des[0]:+.3f},{x_des[1]:+.3f},{x_des[2]:+.3f}) "
                f"x_now=({ee_now[0]:+.3f},{ee_now[1]:+.3f},{ee_now[2]:+.3f}) "
                f"err={err_xy:.4f}m"
            )
            if qdd_sol is not None:
                # Task residual: r = J·qdd + Jdv − xdd_cmd. QP minimizes
                # ‖r‖²·W_task, so this reads the achieved task tracking.
                J        = dv["J"]
                Jdv      = dv["Jdv"]
                xdd_cmd  = dv["xdd_cmd"]
                r_task   = J @ qdd_sol + Jdv - xdd_cmd
                r_norm   = float(np.linalg.norm(r_task))
                print(
                    f"[OSC-cost] step={self._step} "
                    f"task_residual={r_norm:.4f} "
                    f"|xdd_cmd|={float(np.linalg.norm(xdd_cmd)):.3f}"
                )

        self._step += 1

        diag = {
            "ee_now":   ee_now,
            "finished": finished,
        }
        return self._last_tau, diag

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Wipe transient state on free→c3 transitions (called by the
        wrapper). OSC has no integrator wind-up, so this is mostly a
        no-op — but clearing the held τ keeps behavior fresh on
        re-entry to free mode."""
        self._last_tau[:] = 0.0
        self.last_knot0_feasible = True
        # Keep _step monotonic across mode switches — the log timeline is
        # tied to wall-time step ordering, not tracker-residency epochs.
