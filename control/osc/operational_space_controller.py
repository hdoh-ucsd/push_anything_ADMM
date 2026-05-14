"""Drake LeafSystem wrapping the OSC QP for use in a sim diagram.

Inputs:
  - plant_state (BasicVector of n_q + n_v): current MBP state.
  - cartesian_setpoint (BasicVector of 6): x_des(3) + xd_des(3) for
    the tracked EE point. ẍ_des is not consumed by this prototype.

Output:
  - actuation (BasicVector of n_a): joint torque vector to feed the
    plant's actuation_input_port.

Periodic update period: 1 ms sim-time (1 kHz nominal).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pydrake.all as ad
from pydrake.solvers import OsqpSolver
from pydrake.systems.framework import BasicVector, LeafSystem

from control.osc.qp_builder import build_osc_qp


class OperationalSpaceController(LeafSystem):
    """OSC executor — see ``docs/osc_design.md`` for the full design."""

    def __init__(
        self,
        plant,
        ee_frame_name: str,
        gains_cfg: dict,
        *,
        plant_context=None,
        n_arm: int = 7,
        dt_osc: float = 1e-3,
    ) -> None:
        super().__init__()

        self._plant = plant
        self._n_q = plant.num_positions()
        self._n_v = plant.num_velocities()
        self._n_a = int(n_arm)
        self._ee_frame = plant.GetFrameByName(ee_frame_name)
        # Private context used for plant queries (M, J, g, ...). The
        # diagram's plant_context is read-only here; we keep our own
        # mutable scratch context.
        self._ctx = (plant_context if plant_context is not None
                     else plant.CreateDefaultContext())

        # Gains from config (asserts ensure shape parity with n_a).
        self._Kp_task = np.asarray(gains_cfg["Kp_task"], dtype=float)
        self._Kd_task = np.asarray(gains_cfg["Kd_task"], dtype=float)
        self._Kp_posture = np.asarray(gains_cfg["Kp_posture"], dtype=float)
        self._Kd_posture = np.asarray(gains_cfg["Kd_posture"], dtype=float)
        self._W_task = float(gains_cfg["W_task"])
        self._W_posture = float(gains_cfg["W_posture"])
        self._torque_limits = np.asarray(gains_cfg["torque_limits"], dtype=float)
        self._w_tau_reg = float(gains_cfg.get("w_tau_reg", 1e-4))
        assert self._Kp_task.shape == (3,)
        assert self._Kd_task.shape == (3,)
        assert self._Kp_posture.shape == (self._n_a,)
        assert self._Kd_posture.shape == (self._n_a,)
        assert self._torque_limits.shape == (self._n_a,)

        self._q_home_arm = np.asarray(gains_cfg["q_home_arm"], dtype=float)
        assert self._q_home_arm.shape == (self._n_a,)

        # Ports.
        self._state_port = self.DeclareVectorInputPort(
            "plant_state", BasicVector(self._n_q + self._n_v),
        )
        self._setpoint_port = self.DeclareVectorInputPort(
            "cartesian_setpoint", BasicVector(6),
        )
        # The output port pulls on demand; the actuation drives the
        # plant on every sim step, not just at the 1 ms tick. (Drake's
        # ZOH semantics across periodic updates apply if a discrete
        # state is added; for this prototype we use a direct calc port
        # that re-solves the QP whenever the plant pulls it. Subject to
        # revisit in Phase 3 if QP cost dominates.)
        self.DeclareVectorOutputPort(
            "actuation", BasicVector(self._n_a), self._compute_actuation,
        )

        self._solver = OsqpSolver()
        self._last_tau = np.zeros(self._n_a)
        self._dt_osc = float(dt_osc)

        # Diagnostic counters.
        self._n_solves = 0
        self._n_infeasible = 0

    # ------------------------------------------------------------------
    def _compute_actuation(self, context, output) -> None:
        # Inputs.
        x_state = self._state_port.Eval(context)
        setpoint = self._setpoint_port.Eval(context)
        q = x_state[: self._n_q]
        v = x_state[self._n_q : self._n_q + self._n_v]

        # Set scratch ctx for plant queries.
        self._plant.SetPositions(self._ctx, q)
        self._plant.SetVelocities(self._ctx, v)

        x_des  = np.asarray(setpoint[:3], dtype=float)
        xd_des = np.asarray(setpoint[3:6], dtype=float)

        prog, vars_dict = build_osc_qp(
            self._plant, self._ctx, self._ee_frame,
            x_des=x_des, xd_des=xd_des,
            q_home_arm=self._q_home_arm,
            q_now_arm=q[: self._n_a],
            v_now_arm=v[: self._n_a],
            Kp_task=self._Kp_task, Kd_task=self._Kd_task,
            Kp_posture=self._Kp_posture, Kd_posture=self._Kd_posture,
            W_task=self._W_task, W_posture=self._W_posture,
            torque_limits=self._torque_limits,
            w_tau_reg=self._w_tau_reg,
            n_arm=self._n_a,
        )

        result = self._solver.Solve(prog)
        self._n_solves += 1

        if result.is_success():
            tau = result.GetSolution(vars_dict["tau"])
            self._last_tau = np.asarray(tau, dtype=float)
        else:
            # QP infeasibility shouldn't happen given the soft task cost,
            # but if it does (numerical / OSQP iter cap), hold the last
            # output rather than crashing the sim.
            self._n_infeasible += 1
            if self._n_infeasible == 1 or self._n_infeasible % 100 == 0:
                print(f"[OSC] QP infeasible (count={self._n_infeasible}); "
                      f"holding last τ. status={result.get_solution_result()}")

        output.SetFromVector(self._last_tau)

    # ------------------------------------------------------------------
    def get_input_port_state(self):
        return self._state_port

    def get_input_port_setpoint(self):
        return self._setpoint_port
