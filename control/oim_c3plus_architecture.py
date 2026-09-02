"""Architecture boundary for running the existing C3+ baseline in OIM scenes.

OIM owns the task (scene, xArm plant, start/goal, and evaluation protocol).
The existing :class:`control.admm_solver.C3Solver` owns ADMM consensus and the
C3+ projection.  In particular, none of OIM's sampler or ADMM settings are
translated into this baseline.

The final force-to-xArm executor is intentionally an injected dependency.  A
C3+ action is a Cartesian end-effector force in Newtons, while the imported
OIM robot accepts six desired joint velocities.  Keeping those types on
opposite sides of a protocol prevents an invalid direct connection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from pydrake.multibody.tree import JacobianWrtVariable
from pydrake.systems.analysis import Simulator

from control.admm_solver import C3Solver
from control.lcs_formulator import LCSFormulator
from control.ci_mpc_c3plus import C3PlusMPC
from control.task_costs import QuadraticManipulationCost
from control.sampling_c3.params import SamplingC3Params
from control.sampling_c3.sampling_based_c3_controller import SamplingC3Controller

# The canonical xArm6 model is six actuated joints (joint 6 is the
# axisymmetric-stick wrist roll, synced from the native C++ model). The
# bridge predates that sync; every joint-count site reads this constant.
XARM_DOF = 6
from sim.oim_xarm6_tabletop import (
    OIM_ORIENTATION_SUCCESS_TOLERANCE,
    OIM_TRANSLATION_SUCCESS_TOLERANCE,
    OimTabletopScene,
    build_oim_tabletop_scene,
    load_oim_xarm6_config,
    set_oim_free_body_pose,
    set_oim_start_configuration,
    xarm_configured_actuation,
)


@dataclass(frozen=True)
class OimC3PlusBaselineConfig:
    """Parameters belonging to the existing C3+ baseline and OIM evaluator."""

    horizon: int = 7
    planning_dt: float = 0.075
    pose_planning_dt: float = 0.075
    admm_iterations: int = 3
    rho: float = 100.0
    force_limit: float = 30.0
    # Controller-only robustness model, matching SOP Push-T. The imported
    # OIM simulator retains its physical 0.10 kg T.
    controller_object_mass: float = 1.0
    ee_velocity_bounds: tuple[float, float] = (-0.14, 0.14)
    run_steps: int = 2000
    translation_tolerance: float = OIM_TRANSLATION_SUCCESS_TOLERANCE
    orientation_tolerance: float = OIM_ORIENTATION_SUCCESS_TOLERANCE


@dataclass(frozen=True)
class OimTaskState:
    """One observation in both native OIM and C3+ coordinates."""

    c3plus_state: np.ndarray
    arm_positions: np.ndarray
    arm_velocities: np.ndarray
    object_pose: np.ndarray  # [x, y, yaw]
    goal_pose: np.ndarray  # [x, y, yaw]
    tip_position: np.ndarray
    tip_velocity: np.ndarray


class CartesianForceExecutor(Protocol):
    """Required bridge from the existing C3+ action to the OIM xArm plant."""

    def step(self, ee_force_newtons: np.ndarray, duration: float) -> None:
        """Track one Cartesian EE-force command for ``duration`` seconds."""


class OimXarmDrakeAdapter:
    """State and native velocity-actuation adapter for an imported OIM scene."""

    # The MJCF stick site is ignored by Drake's parser.  Its XML position is
    # the distal end of the 179.4-mm capsule in the xarm6_stick body frame.
    STICK_TIP_IN_BODY = np.array([0.0, 0.0, 0.1794])

    def __init__(self, scene: OimTabletopScene, *, start_pose=None,
                 goal_pose=None):
        self.scene = scene
        self.simulator = Simulator(scene.diagram)
        self.context = self.simulator.get_mutable_context()
        set_oim_start_configuration(scene, self.context)
        if start_pose is not None:
            set_oim_free_body_pose(scene, self.context, "block", start_pose)
        self.goal_pose_override = (
            None if goal_pose is None
            else np.asarray(goal_pose, dtype=float).reshape(3).copy()
        )
        scene.diagram.GetInputPort("xarm6_actuation").FixValue(
            self.context, np.zeros(XARM_DOF)
        )

    @property
    def execution_dt(self) -> float:
        return float(self.scene.plant.time_step())

    def observe(self) -> OimTaskState:
        plant = self.scene.plant
        plant_context = plant.GetMyContextFromRoot(self.context)
        q = plant.GetPositions(plant_context)
        v = plant.GetVelocities(plant_context)
        arm_q = np.array([
            q[self.scene.arm_positions[f"xarm6_joint{i}"]]
            for i in range(1, XARM_DOF + 1)
        ])
        arm_v = np.array([
            v[self.scene.arm_velocities[f"xarm6_joint{i}"]]
            for i in range(1, XARM_DOF + 1)
        ])

        block = plant.GetBodyByName("block")
        goal = plant.GetBodyByName("goal")
        stick = plant.GetBodyByName("xarm6_stick")
        X_WB = plant.EvalBodyPoseInWorld(plant_context, block)
        X_WG = plant.EvalBodyPoseInWorld(plant_context, goal)
        X_WS = plant.EvalBodyPoseInWorld(plant_context, stick)
        p_tip = X_WS.multiply(self.STICK_TIP_IN_BODY)

        J_tip = plant.CalcJacobianTranslationalVelocity(
            plant_context,
            JacobianWrtVariable.kV,
            stick.body_frame(),
            self.STICK_TIP_IN_BODY,
            plant.world_frame(),
            plant.world_frame(),
        )
        v_tip = J_tip @ v
        V_WB = plant.EvalBodySpatialVelocityInWorld(plant_context, block)
        quat = X_WB.rotation().ToQuaternion().wxyz()
        object_velocity = np.concatenate(
            (V_WB.rotational(), V_WB.translational())
        )
        c3_state = np.concatenate(
            (quat, X_WB.translation(), p_tip, object_velocity, v_tip)
        )
        assert c3_state.shape == (19,)
        return OimTaskState(
            c3plus_state=c3_state,
            arm_positions=arm_q,
            arm_velocities=arm_v,
            object_pose=np.array([
                X_WB.translation()[0], X_WB.translation()[1],
                X_WB.rotation().ToRollPitchYaw().yaw_angle(),
            ]),
            goal_pose=(self.goal_pose_override.copy()
                       if self.goal_pose_override is not None else np.array([
                X_WG.translation()[0], X_WG.translation()[1],
                X_WG.rotation().ToRollPitchYaw().yaw_angle(),
            ])),
            tip_position=np.asarray(p_tip),
            tip_velocity=np.asarray(v_tip),
        )

    def step_joint_velocity(self, desired_velocity, duration: float) -> None:
        """Execute a native OIM xArm velocity command using its servo law."""
        desired = np.asarray(desired_velocity, dtype=float)
        if desired.shape != (XARM_DOF,):
            raise ValueError(
                f"desired_velocity must have shape ({XARM_DOF},)")
        if duration <= 0:
            raise ValueError("duration must be positive")
        end_time = self.context.get_time() + float(duration)
        # Refresh the nonlinear servo at every Drake execution step rather
        # than holding the initial torque over a whole planning interval.
        while self.context.get_time() < end_time - 1e-12:
            torque = xarm_configured_actuation(
                self.scene, self.context, desired
            )
            self.scene.diagram.GetInputPort("xarm6_actuation").FixValue(
                self.context, torque
            )
            self.simulator.AdvanceTo(min(end_time, self.context.get_time() + self.execution_dt))


class OimXarmCartesianForceExecutor:
    """Jacobian-transpose executor for the C3+ Cartesian-force action.

    The feed-forward term is exactly ``J_tip.T @ force``. Gravity
    compensation and joint damping make it an operational-space force
    executor suitable for the imported torque input; it does not reinterpret
    force as an OIM velocity command.
    """

    action_size = 3

    def __init__(self, adapter: OimXarmDrakeAdapter, joint_damping: float = 2.0):
        self.adapter = adapter
        self.joint_damping = float(joint_damping)

    def step(self, ee_force_newtons: np.ndarray, duration: float) -> None:
        force = np.asarray(ee_force_newtons, dtype=float)
        if force.shape != (3,) or not np.all(np.isfinite(force)):
            raise ValueError("ee_force_newtons must be a finite shape-(3,) vector")
        if duration <= 0:
            raise ValueError("duration must be positive")
        a = self.adapter
        plant = a.scene.plant
        end_time = a.context.get_time() + float(duration)
        while a.context.get_time() < end_time - 1e-12:
            pc = plant.GetMyContextFromRoot(a.context)
            stick = plant.GetBodyByName("xarm6_stick")
            J = plant.CalcJacobianTranslationalVelocity(
                pc, JacobianWrtVariable.kV, stick.body_frame(),
                a.STICK_TIP_IN_BODY, plant.world_frame(), plant.world_frame(),
            )
            v = plant.GetVelocities(pc)
            arm_indices = np.array([
                a.scene.arm_velocities[f"xarm6_joint{i}"]
                for i in range(1, XARM_DOF + 1)
            ])
            gravity = plant.CalcGravityGeneralizedForces(pc)
            tau = J[:, arm_indices].T @ force
            tau -= gravity[arm_indices]
            tau -= self.joint_damping * v[arm_indices]
            limits = np.asarray((50.0, 50.0, 32.0, 32.0, 32.0, 20.0))
            a.scene.diagram.GetInputPort("xarm6_actuation").FixValue(
                a.context, np.clip(tau, -limits, limits)
            )
            a.simulator.AdvanceTo(
                min(end_time, a.context.get_time() + a.execution_dt)
            )


class OimXarmJointTorqueExecutor:
    """Execute the production full-plant LCS's six joint torques."""

    action_size = XARM_DOF

    def __init__(self, adapter: OimXarmDrakeAdapter, *, gravity_compensation=False):
        self.adapter = adapter
        self.gravity_compensation = bool(gravity_compensation)

    def step(self, joint_torque: np.ndarray, duration: float) -> None:
        torque = np.asarray(joint_torque, dtype=float)
        if torque.shape != (XARM_DOF,) or not np.all(np.isfinite(torque)):
            raise ValueError(
                f"joint_torque must be a finite shape-({XARM_DOF},) vector")
        limits = np.asarray((50.0, 50.0, 32.0, 32.0, 32.0, 20.0))
        end = self.adapter.context.get_time() + float(duration)
        command = np.clip(torque, -limits, limits)
        if self.gravity_compensation:
            plant = self.adapter.scene.plant
            pc = plant.GetMyContextFromRoot(self.adapter.context)
            # Same contract as main.py: OSC solves with Cv-only bias and the
            # execution loop adds the negated generalized gravity force.
            command = command - np.asarray(
                plant.CalcGravityGeneralizedForces(pc)[:XARM_DOF], dtype=float
            )
            q = plant.GetPositions(pc)
            j4 = self.adapter.scene.arm_positions["xarm6_joint4"]
            command[3] += -175.0 * float(q[j4])
        self.adapter.scene.diagram.GetInputPort("xarm6_actuation").FixValue(
            self.adapter.context, command
        )
        self.adapter.simulator.AdvanceTo(end)


class OimXarmDifferentialIkExecutor:
    """Track Sampling-C3 Cartesian trajectories via OIM velocity actuators.

    OIM's native action is five desired joint velocities, not torque.  This
    adapter preserves that boundary and uses a weighted differential IK to
    track stick translation while keeping the tool upright and joint 4 near
    its spring reference.
    """

    action_size = XARM_DOF

    def __init__(self, adapter: OimXarmDrakeAdapter, sampling_controller):
        self.adapter = adapter
        self.sampling_controller = sampling_controller
        self.max_velocity = 0.5
        self.position_gain = 4.0
        self.tilt_gain = 3.0
        self.joint4_gain = 4.0
        pc = adapter.scene.plant.GetMyContextFromRoot(adapter.context)
        self.R_target = adapter.scene.plant.CalcRelativeTransform(
            pc, adapter.scene.plant.world_frame(),
            adapter.scene.plant.GetFrameByName("oim_stick_tip"),
        ).rotation()
        self.last_velocity_command = np.zeros(XARM_DOF)

    def _desired_cartesian_state(self, t_sim: float):
        call = self.sampling_controller._last_osc_call
        state = self.adapter.observe()
        if call is None:
            return state.tip_position, np.zeros(3)
        kind, kw = call
        if kind == "osc_pwl_free":
            p_des, v_des, _ = kw["pwl_traj"].eval(float(t_sim))
            return np.asarray(p_des), np.asarray(v_des)
        if kind == "osc_direct_free":
            return (np.asarray(kw["p_ee_desired"]),
                    np.zeros(3) if kw["v_ee_desired"] is None
                    else np.asarray(kw["v_ee_desired"]))
        if kind == "c3_traj":
            traj = kw["traj"]
            p_des = np.asarray(traj.value(float(t_sim))).reshape(3)
            try:
                v_des = np.asarray(traj.EvalDerivative(
                    float(t_sim), 1)).reshape(3)
            except Exception:
                v_des = np.zeros(3)
            return p_des, v_des
        return state.tip_position, np.zeros(3)

    def desired_joint_velocity(self, t_sim: float) -> np.ndarray:
        p_des, v_ff = self._desired_cartesian_state(t_sim)
        a = self.adapter
        plant = a.scene.plant
        pc = plant.GetMyContextFromRoot(a.context)
        state = a.observe()
        stick = plant.GetBodyByName("xarm6_stick")
        Jv = plant.CalcJacobianTranslationalVelocity(
            pc, JacobianWrtVariable.kV, stick.body_frame(),
            a.STICK_TIP_IN_BODY, plant.world_frame(), plant.world_frame(),
        )[:, :XARM_DOF]
        Jw = plant.CalcJacobianSpatialVelocity(
            pc, JacobianWrtVariable.kV, stick.body_frame(),
            a.STICK_TIP_IN_BODY, plant.world_frame(), plant.world_frame(),
        )[:3, :XARM_DOF]
        R_now = plant.EvalBodyPoseInWorld(pc, stick).rotation()
        aa = (self.R_target @ R_now.inverse()).ToAngleAxis()
        w_err = float(aa.angle()) * np.asarray(aa.axis())
        v_cmd = np.asarray(v_ff) + self.position_gain * (
            np.asarray(p_des) - state.tip_position)

        # Translation is primary. Roll/pitch and the spring-reference joint
        # are soft OIM shaping terms; a small Tikhonov term makes the solve
        # deterministic at singular configurations.
        A = np.vstack((Jv, 0.35 * Jw[:2, :],
                       0.8 * np.array([[0., 0., 0., 1., 0., 0.]]),
                       0.03 * np.eye(XARM_DOF)))
        b = np.concatenate((v_cmd, 0.35 * self.tilt_gain * w_err[:2],
                            [-0.8 * self.joint4_gain * state.arm_positions[3]],
                            np.zeros(XARM_DOF)))
        # Bound each command by both OIM's ±0.5 rad/s actuator range and a
        # short-horizon position-limit viability constraint.  Solve again on
        # the remaining free joints whenever a bound activates; plain
        # componentwise clipping made joint 5 sit on its lower stop while the
        # other joints never took over its Cartesian task.
        q = state.arm_positions
        q_lo = plant.GetPositionLowerLimits()[:XARM_DOF]
        q_hi = plant.GetPositionUpperLimits()[:XARM_DOF]
        margin = 0.02
        lookahead = 0.25
        lower = np.maximum(-self.max_velocity,
                           (q_lo + margin - q) / lookahead)
        upper = np.minimum(+self.max_velocity,
                           (q_hi - margin - q) / lookahead)
        lower = np.minimum(lower, upper)
        fixed = np.zeros(XARM_DOF)
        free = np.ones(XARM_DOF, dtype=bool)
        for _ in range(XARM_DOF):
            rhs = b - A[:, ~free] @ fixed[~free]
            solution, *_ = np.linalg.lstsq(A[:, free], rhs, rcond=None)
            candidate = fixed.copy()
            candidate[free] = solution
            below = free & (candidate < lower)
            above = free & (candidate > upper)
            if not np.any(below | above):
                fixed = candidate
                break
            violation = np.maximum(lower - candidate, candidate - upper)
            index = int(np.argmax(np.where(below | above, violation, -np.inf)))
            fixed[index] = lower[index] if below[index] else upper[index]
            free[index] = False
        else:
            fixed = np.clip(fixed, lower, upper)
        self.last_velocity_command = np.clip(fixed, lower, upper)
        return self.last_velocity_command

    def step_cached_trajectory(self, duration: float) -> None:
        end = self.adapter.context.get_time() + float(duration)
        while self.adapter.context.get_time() < end - 1e-12:
            dt = min(self.adapter.execution_dt,
                     end - self.adapter.context.get_time())
            command = self.desired_joint_velocity(
                self.adapter.context.get_time())
            self.adapter.step_joint_velocity(command, dt)


class OimXarmOscVelocityBridge:
    """Execute Sampling-C3+'s OSC through OIM's velocity actuators.

    Drake imports the xArm actuators as torque inputs, while the source OIM
    model exposes a velocity-servo command with
    ``tau_servo = kv * (qdot_cmd - qdot)``.  Inverting that relation lets the
    existing C3+ OSC remain the controller of record without changing the
    OIM actuator contract.  Gravity compensation and the joint-4 spring are
    still added by :func:`xarm_configured_actuation`, outside the actuator
    force clamp, matching the source MJCF ordering.
    """

    action_size = XARM_DOF
    SERVO_GAINS = np.array([300.0, 300.0, 200.0, 200.0, 200.0, 200.0])
    COMMAND_LIMIT = 0.5
    # Keep the analytic reposition path no more than this far ahead of the
    # measured tip.  The source xArm cannot track the inherited 0.18 m/s PWL
    # open-loop; allowing its clock to expire removes velocity feedforward
    # and consumes most of a short OIM rollout in an asymptotic crawl.
    REPOSITION_MAX_TRACKING_ERROR = 0.03

    def __init__(self, adapter: OimXarmDrakeAdapter, sampling_controller):
        self.adapter = adapter
        self.sampling_controller = sampling_controller
        self.last_velocity_command = np.zeros(XARM_DOF)
        self.last_osc_torque = np.zeros(XARM_DOF)
        self.velocity_saturation_events = 0
        self._reposition_trajectory = None
        self._reposition_execution_time = None
        self._reposition_last_sim_time = None

    def _osc_evaluation_time(self, t_sim: float) -> float:
        """Return a feedback-limited clock for synchronous repositioning."""
        call = self.sampling_controller._last_osc_call
        if call is None or call[0] != "osc_pwl_free":
            self._reposition_trajectory = None
            self._reposition_execution_time = None
            self._reposition_last_sim_time = None
            return float(t_sim)

        traj = call[1]["pwl_traj"]
        now = float(t_sim)
        if traj is not self._reposition_trajectory:
            self._reposition_trajectory = traj
            self._reposition_execution_time = float(traj.t_start)
            self._reposition_last_sim_time = now
            return self._reposition_execution_time

        elapsed = max(0.0, now - float(self._reposition_last_sim_time))
        self._reposition_last_sim_time = now
        p_des, _, _ = traj.eval(self._reposition_execution_time)
        tracking_error = float(np.linalg.norm(
            np.asarray(p_des) - self.adapter.observe().tip_position
        ))
        if tracking_error <= self.REPOSITION_MAX_TRACKING_ERROR:
            self._reposition_execution_time = min(
                float(traj.t_end), self._reposition_execution_time + elapsed
            )
        return float(self._reposition_execution_time)

    def desired_joint_velocity(self, t_sim: float) -> np.ndarray:
        plant = self.adapter.scene.plant
        pc = plant.GetMyContextFromRoot(self.adapter.context)
        q = plant.GetPositions(pc)
        v = plant.GetVelocities(pc)
        eval_time = self._osc_evaluation_time(float(t_sim))
        tau_osc = np.asarray(
            self.sampling_controller.compute_control_osc_only(
                q, v, pc, eval_time), dtype=float
        ).reshape(XARM_DOF)
        if not np.all(np.isfinite(tau_osc)):
            tau_osc = np.zeros(XARM_DOF)

        arm_v_indices = np.array([
            self.adapter.scene.arm_velocities[f"xarm6_joint{i}"]
            for i in range(1, XARM_DOF + 1)
        ])
        qdot = np.asarray(v[arm_v_indices], dtype=float)
        raw_command = qdot + tau_osc / self.SERVO_GAINS
        command = np.clip(raw_command, -self.COMMAND_LIMIT,
                          self.COMMAND_LIMIT)
        if np.any(np.abs(raw_command - command) > 1e-12):
            self.velocity_saturation_events += 1

        self.last_osc_torque = tau_osc.copy()
        self.last_velocity_command = command
        return command

    def step_cached_trajectory(self, duration: float) -> None:
        end = self.adapter.context.get_time() + float(duration)
        while self.adapter.context.get_time() < end - 1e-12:
            dt = min(self.adapter.execution_dt,
                     end - self.adapter.context.get_time())
            command = self.desired_joint_velocity(
                self.adapter.context.get_time())
            self.adapter.step_joint_velocity(command, dt)


class OimFullPlantC3PlusController:
    """Production LCS frontend using native OIM plant coordinates."""

    def __init__(self, scene, formulator, solver, config):
        self.scene = scene
        self.formulator = formulator
        self.solver = solver
        self.config = config
        self._last_u = np.zeros(XARM_DOF)

    def compute_action(self, adapter: OimXarmDrakeAdapter) -> np.ndarray:
        plant = self.scene.plant
        pc = plant.GetMyContextFromRoot(adapter.context)
        q = plant.GetPositions(pc)
        v = plant.GetVelocities(pc)
        x0 = np.concatenate((q, v))
        matrices = self.formulator.linearize_discrete_with_complementarity(
            pc, self.config.planning_dt, u_lin=self._last_u
        )
        A, B, D, d, E, F, H, c, J_n, J_t, phi, mu = matrices
        n_x = plant.num_positions() + plant.num_velocities()
        Q = 1e-3 * np.eye(n_x)
        for name, weight in (("T_x", 200.0), ("T_y", 200.0), ("T_z", 16.0)):
            index = plant.GetJointByName(name).position_start()
            Q[index, index] = weight
        QN = 10.0 * Q
        R = 0.05 * np.eye(XARM_DOF)
        x_ref = x0.copy()
        goal = adapter.observe().goal_pose
        x_ref[plant.GetJointByName("T_x").position_start()] = goal[0]
        x_ref[plant.GetJointByName("T_y").position_start()] = goal[1]
        x_ref[plant.GetJointByName("T_z").position_start()] = goal[2]
        u, _ = self.solver.solve(
            x0, A, B, D, d, J_n, J_t, mu, Q, R, QN, x_ref,
            N=self.config.horizon,
            admm_iter=self.config.admm_iterations,
            torque_limit=32.0, phi=phi, E=E, F=F, H=H, c_lcs=c,
        )
        self._last_u = np.asarray(u[0]).copy()
        return self._last_u


class OimSamplingC3PlusController:
    """Thin runtime bridge into the production SamplingC3Controller."""

    def __init__(self, controller, adapter):
        self.controller = controller
        self.adapter = adapter
        self._last_u = np.zeros(XARM_DOF)

    def compute_action(self, _state=None) -> np.ndarray:
        plant = self.adapter.scene.plant
        pc = plant.GetMyContextFromRoot(self.adapter.context)
        q = plant.GetPositions(pc)
        v = plant.GetVelocities(pc)
        goal = self.adapter.observe().goal_pose
        self._last_u = np.asarray(self.controller.compute_control(
            q, v, pc, goal[:2], goal[2], final_target_xy=goal[:2]
        ))
        return self._last_u

    @property
    def last_ee_force_command(self) -> np.ndarray:
        """Latest three-axis Cartesian force requested by Sampling-C3."""
        return np.asarray(
            getattr(self.controller, "last_ee_force_command", np.zeros(3)),
            dtype=float,
        ).reshape(3).copy()

    def execute_planning_interval(self, first_torque, executor, duration: float) -> None:
        """Run the cached Sampling-C3 plan through the OSC at servo rate."""
        if hasattr(executor, "step_cached_trajectory"):
            executor.step_cached_trajectory(duration)
            self._last_u = executor.last_velocity_command.copy()
            return
        remaining = float(duration)
        torque = np.asarray(first_torque, dtype=float)
        servo_dt = float(self.adapter.execution_dt)
        while remaining > 1e-12:
            dt = min(servo_dt, remaining)
            executor.step(torque, dt)
            remaining -= dt
            if remaining <= 1e-12:
                break
            plant = self.adapter.scene.plant
            pc = plant.GetMyContextFromRoot(self.adapter.context)
            torque = np.asarray(self.controller.compute_control_osc_only(
                plant.GetPositions(pc), plant.GetVelocities(pc), pc,
                self.adapter.context.get_time(),
            ))
        self._last_u = torque.copy()


class OimPlanarC3PlusController:
    """Planar OIM LCS frontend for the repository's unchanged C3+ solver."""

    def __init__(self, solver: C3Solver, config: OimC3PlusBaselineConfig):
        self.solver = solver
        self.config = config
        self._last_u = np.zeros(3)

    def _linearization(self, state: OimTaskState):
        dt = self.config.planning_dt
        x = state.c3plus_state
        A = np.eye(19)
        A[4, 13] = dt
        A[5, 14] = dt
        qw, qz = x[0], x[3]
        A[0, 12] = -0.5 * dt * qz
        A[3, 12] = 0.5 * dt * qw
        A[7:10, 16:19] = dt * np.eye(3)
        # Unit-mass EE point used by the established EE-space C3+ model.
        B = np.zeros((19, 3))
        B[7:10] = dt * dt * np.eye(3)
        B[16:19] = dt * np.eye(3)

        delta = x[7:9] - x[4:6]
        distance = float(np.linalg.norm(delta))
        normal = delta / max(distance, 1e-9)  # object -> pusher
        # Conservative circumscribed radius for the T/C manipulands plus tip.
        contact_radius = 0.081
        gap = distance - contact_radius
        n_lam = 6  # [gamma, normal, four tangent generators]
        D = np.zeros((19, n_lam))
        # Normal impulse pushes the object away from the pusher and produces
        # the equal-and-opposite EE response. Planar object mass is resolved
        # from the Drake plant by the execution model; 1 kg is the controller
        # model convention used by the existing EE-space formulator.
        D[4:6, 1] = -dt * dt * normal
        D[13:15, 1] = -dt * normal
        D[7:9, 1] = dt * dt * normal
        D[16:18, 1] = dt * normal
        d = np.zeros(19)

        # eta_n is the first-order next-knot signed separation. Other rows
        # stay zero, matching the existing C3+ v1 treatment of tangent slack.
        E = np.zeros((n_lam, 19))
        E[1, 7:9] = normal
        E[1, 4:6] = -normal
        F = np.zeros((n_lam, n_lam))
        F[1, 1] = 2.0 * dt * dt
        H = np.zeros((n_lam, 3))
        H[1, :2] = dt * dt * normal
        c = np.zeros(n_lam)
        c[1] = -contact_radius
        J_n = np.zeros((1, 9))
        J_n[0, 3:5] = -normal
        J_n[0, 6:8] = normal
        J_t = np.zeros((4, 9))
        phi = np.array([gap])
        return A, B, D, d, E, F, H, c, J_n, J_t, phi

    def compute_action(self, state: OimTaskState) -> np.ndarray:
        A, B, D, d, E, F, H, c, J_n, J_t, phi = self._linearization(state)
        x = state.c3plus_state
        goal = state.goal_pose
        Q = np.zeros((19, 19))
        Q[4, 4] = Q[5, 5] = 200.0
        Q[0, 0] = Q[3, 3] = 16.0
        Q[7, 7] = Q[8, 8] = 60.0
        Q[9, 9] = 20.0
        Q[13, 13] = Q[14, 14] = 1.0
        Q[16, 16] = Q[17, 17] = Q[18, 18] = 0.1
        QN = 10.0 * Q
        R = 0.05 * np.eye(3)
        x_ref = np.zeros(19)
        x_ref[0] = np.cos(0.5 * goal[2])
        x_ref[3] = np.sin(0.5 * goal[2])
        x_ref[4:6] = goal[:2]
        x_ref[6] = x[6]
        direction = goal[:2] - x[4:6]
        direction /= max(float(np.linalg.norm(direction)), 1e-9)
        x_ref[7:9] = x[4:6] - 0.081 * direction
        x_ref[9] = x[9]
        u_seq, _ = self.solver.solve(
            x, A, B, D, d, J_n, J_t, 0.5, Q, R, QN, x_ref,
            N=self.config.horizon,
            admm_iter=self.config.admm_iterations,
            torque_limit=self.config.force_limit,
            phi=phi, E=E, F=F, H=H, c_lcs=c,
            u_lower=np.array([-30.0, -30.0, -3.0]),
            u_upper=np.array([30.0, 30.0, 3.0]),
            ee_velocity_bounds=self.config.ee_velocity_bounds,
        )
        self._last_u = np.asarray(u_seq[0]).copy()
        return self._last_u


@dataclass
class OimC3PlusArchitecture:
    """Constructed baseline components with an explicit executor seam."""

    scene: OimTabletopScene
    adapter: OimXarmDrakeAdapter
    solver: C3Solver
    config: OimC3PlusBaselineConfig
    executor: CartesianForceExecutor | None = None
    controller: OimPlanarC3PlusController | None = None

    def execute_c3plus_action(self, action) -> None:
        if self.executor is None:
            raise RuntimeError(
                "C3+ force executor is not connected: implement the xArm OSC/IK "
                "bridge before executing planner actions"
            )
        command = np.asarray(action, dtype=float)
        expected_size = int(getattr(self.executor, "action_size", self.solver.n_u))
        if command.shape != (expected_size,):
            raise ValueError(
                f"the connected C3+ executor requires shape ({expected_size},)"
            )
        self.executor.step(command, self.config.planning_dt)

    def step(self) -> OimTaskState:
        """Run one complete observe-plan-execute C3+ control tick."""
        if self.controller is None:
            raise RuntimeError("OIM planar C3+ controller is not connected")
        before = self.adapter.observe()
        if isinstance(self.controller, OimFullPlantC3PlusController):
            action = self.controller.compute_action(self.adapter)
        elif isinstance(self.controller, OimSamplingC3PlusController):
            action = self.controller.compute_action()
        else:
            action = self.controller.compute_action(before)
        if isinstance(self.controller, OimSamplingC3PlusController):
            if self.executor is None:
                raise RuntimeError("Sampling-C3+ torque executor is not connected")
            self.controller.execute_planning_interval(
                action, self.executor, self.config.planning_dt
            )
        else:
            self.execute_c3plus_action(action)
        return self.adapter.observe()


def build_oim_c3plus_architecture(
    scene_name: str,
    *,
    config: OimC3PlusBaselineConfig | None = None,
    executor: CartesianForceExecutor | None = None,
) -> OimC3PlusArchitecture:
    """Build the OIM environment around the repository's unchanged C3+ solver."""
    baseline = config or OimC3PlusBaselineConfig()
    oim_yaml = load_oim_xarm6_config()
    scene = build_oim_tabletop_scene(
        scene_name, time_step=float(oim_yaml["world3d"]["exec_timestep"])
    )
    adapter = OimXarmDrakeAdapter(scene)
    solver = C3Solver(
        n_x=19,
        n_u=3,
        rho=baseline.rho,
        mode="c3plus",
        penalize_input_change=True,
    )
    if executor is None:
        executor = OimXarmCartesianForceExecutor(adapter)
    controller = OimPlanarC3PlusController(solver, baseline)
    return OimC3PlusArchitecture(
        scene, adapter, solver, baseline, executor, controller
    )


def build_oim_full_lcs_architecture(
    scene_name: str,
    *,
    config: OimC3PlusBaselineConfig | None = None,
) -> OimC3PlusArchitecture:
    """Build OIM scene execution around the production full-plant C3+ LCS."""
    baseline = config or OimC3PlusBaselineConfig()
    scene = build_oim_tabletop_scene(scene_name)
    inspector = scene.scene_graph.model_inspector()
    floor_ids = [
        gid for gid in scene.plant.GetCollisionGeometriesForBody(
            scene.plant.world_body()
        )
        if isinstance(inspector.GetShape(gid), HalfSpace)
    ]
    block_ids = list(scene.plant.GetCollisionGeometriesForBody(
        scene.plant.GetBodyByName("block")
    ))
    if floor_ids and block_ids:
        scene.scene_graph.collision_filter_manager().Apply(
            CollisionFilterDeclaration().ExcludeBetween(
                GeometrySet(block_ids), GeometrySet(floor_ids)
            )
        )
    adapter = OimXarmDrakeAdapter(scene)
    plant_ad = scene.plant.ToAutoDiffXd()
    context_ad = plant_ad.CreateDefaultContext()
    formulator = LCSFormulator(
        scene.plant,
        mu=0.3,
        obj_body=scene.plant.GetBodyByName("block"),
        plant_ad=plant_ad,
        context_ad=context_ad,
        object_shape="tshape",
        ee_body_name="xarm6_stick",
        controller_object_mass=baseline.controller_object_mass,
        manipuland_body_names=("block",),
    )
    # OIM's actual collision boxes supply support geometry; the imported
    # Push-Anything T witness table belongs to a different object scale.
    formulator.lcs_explicit_manipuland_ground_contacts = 0
    solver = C3Solver(
        n_x=scene.plant.num_positions() + scene.plant.num_velocities(),
        n_u=scene.plant.num_actuators(),
        rho=baseline.rho,
        mode="c3plus",
        penalize_input_change=True,
    )
    controller = OimFullPlantC3PlusController(
        scene, formulator, solver, baseline
    )
    executor = OimXarmJointTorqueExecutor(adapter)
    return OimC3PlusArchitecture(
        scene, adapter, solver, baseline, executor, controller
    )


def build_oim_sampling_c3plus_architecture(
    scene_name: str = "open_table",
    *,
    config: OimC3PlusBaselineConfig | None = None,
    start_pose=None,
    goal_pose=None,
) -> OimC3PlusArchitecture:
    """Build the production Sampling-C3+ stack around the OIM scene."""
    baseline = config or OimC3PlusBaselineConfig(horizon=10)
    scene = build_oim_tabletop_scene(
        scene_name, sampling_c3plus_object=True
    )
    # Keep the physical T-table collision active.  The LCS formulator owns
    # its synthetic three-witness support approximation, but those algebraic
    # contacts do not apply forces to Drake's simulator.  Filtering this pair
    # globally made the free body fall through the table before C3 execution.
    adapter = OimXarmDrakeAdapter(
        scene, start_pose=start_pose, goal_pose=goal_pose
    )
    plant = scene.plant
    block = plant.GetBodyByName("block")
    plant_ad = plant.ToAutoDiffXd()
    context_ad = plant_ad.CreateDefaultContext()
    formulator = LCSFormulator(
        plant, mu=0.3, obj_body=block,
        plant_ad=plant_ad, context_ad=context_ad,
        object_shape="tshape", ee_body_name="xarm6_stick",
        tshape_geometry_variant="oim_lab",
        controller_object_mass=baseline.controller_object_mass,
        ee_point_B=OimXarmDrakeAdapter.STICK_TIP_IN_BODY,
        manipuland_body_names=("block",),
    )
    solver = C3Solver(n_x=19, n_u=3, rho=baseline.rho,
                      mode="c3plus", penalize_input_change=True)
    plant_context = plant.GetMyContextFromRoot(adapter.context)
    object_rest_z = float(
        plant.EvalBodyPoseInWorld(plant_context, block).translation()[2])
    # Match the OIM static-C3 baseline (oim/algs/c3.py).  The inner QP uses
    # a local quaternion quadratic: 400*sin²(dyaw/2) ~= 100*dyaw².  Candidate
    # rollout ranking is evaluated separately with the exact wrapped-SE(2)
    # 1000/100 running and 10000/1000 terminal cost.
    oim_cost = {
        "w_obj_xy": 1000.0,
        "w_yaw": 400.0,
        "w_terminal": 10.0,
        "use_oim_se2_ranking_cost": True,
        "oim_q_pos": 1000.0,
        "oim_q_theta": 100.0,
        "oim_qf_pos": 10000.0,
        "oim_qf_theta": 1000.0,
        "z_ee_target": 0.034,
        "z_obj_target": object_rest_z,
        # OIM evaluates and optimizes the object's SE(2) pose.  Vertical and
        # roll/pitch stabilization come from contact physics, not goal costs.
        "w_obj_z": 0.0,
        "w_box_z": 0.0,
        "w_box_rp": 0.0,
        "w_ee_approach": 400.0,
        "d_push": 0.05,
        "w_torque": 0.05,
    }
    cost = QuadraticManipulationCost(
        plant, "oim_stick_tip", block, oim_cost,
        plant.num_positions() + plant.num_velocities(),
        plant.num_actuators(), object_shape="tshape",
    )
    base_mpc = C3PlusMPC(
        formulator, solver, cost,
        horizon=baseline.horizon, dt=baseline.planning_dt,
        dt_pose=baseline.pose_planning_dt,
        torque_limit=baseline.force_limit,
        admm_iter=baseline.admm_iterations,
        use_ee_space=True,
        ee_velocity_bounds=baseline.ee_velocity_bounds,
    )
    params = SamplingC3Params.from_yaml("config/sampling_c3_kik_t.yaml")
    params.sampling_params.tshape_geometry_variant = "oim_lab"
    params.sampling_params.pusher_radius_override = 0.00555
    params.sampling_params.sampling_height = 0.034
    params.sampling_params.z_height = 0.034
    params.sampling_params.use_adaptive_contact_height = False
    params.sampling_params.grid_x_limits = [-0.0445, 0.0445]
    params.sampling_params.grid_y_limits = [-0.0794, 0.0198]
    # CheckForWorkspaceLimitViolations is reference behavior, but its bounds
    # are robot/environment inputs. Replace the Franka task's x>=0.23 slab
    # with the OIM tabletop's usable region (2 cm inside each table edge).
    params.sampling_params.workspace_xy_min = [-0.03, -0.7415]
    params.sampling_params.workspace_xy_max = [0.73, 0.7415]
    params.sampling_params.workspace_z_min = 0.0
    params.sampling_params.workspace_z_max = 0.80
    # The imported xArm's valid folded configuration crosses inside the
    # Franka task's 0.12 m inner shell during repositioning.
    params.sampling_params.robot_radius_limits = [0.05, 0.90]
    params.planner_workspace_x = [-0.03, 0.73]
    params.planner_workspace_y = [-0.7415, 0.7415]
    params.planner_workspace_z = [0.0, 0.80]
    params.osc_gains_yaml = "config/osc_xarm6_oim.yaml"
    # OIM evaluates planar pushing, but this Drake wrapper uses a floating T
    # for the 19-state C3+ model.  Stop planar force execution after a 20-deg
    # tip instead of allowing the closest-pair normal to rotate toward world-z.
    # This is a wrapper safety invariant; the generic controller retains the
    # reference behavior when the parameter is None.
    params.max_planar_object_tilt_rad = float(np.deg2rad(20.0))
    # OIM's evaluator is SE(2): roll/pitch are safety state, not orientation
    # progress.  Keep its progress metric consistent with summary.json.
    params.use_planar_yaw_progress = True
    params.unsuccessful_only_on_observed_failure = True
    params.unsuccessful_body_relative = True
    # Abort a C3 contact once measured XY error has regressed 5 mm from the
    # best point in that C3 session.  The generic T config leaves this
    # off-reference guard disabled; OIM opts in because its long scenarios
    # otherwise execute a bad face for thousands of steps.
    params.progress_params.pos_regression_threshold = 0.005
    params.progress_params.yaw_regression_threshold = 0.10
    params.progress_params.regression_consecutive_steps = 3
    # Traverse above the 59.6-mm-high T plus the 5.55-mm pusher radius;
    # descending to the side-contact plane happens only at the selected
    # contact xy.  The inherited 60-mm waypoint grazed the T during free mode.
    params.reposition_params.pwl_waypoint_height = 0.075
    # The OIM xArm has only five controlled joints.  During the initial
    # vertical segment its OSC may trail the Cartesian trajectory; rebuilding
    # from the measured pose every planner tick would consequently postpone
    # the following lateral segment forever.  Keep the selected trajectory
    # alive until its target changes.  This override is local to OIM; the
    # reference Push Anything policy remains the default.
    params.rebuild_reposition_pwl_every_tick = False
    # The five-DOF xArm can asymptotically plateau just outside the generic
    # 20 mm free-mode arrival gate (the latest open_table run stopped at
    # 20.5 mm for the remainder of its budget). After two seconds with less
    # than 2 mm improvement, accept a near target within 25 mm or blacklist
    # a farther stalled target and force fresh sampling.
    params.sampling_params.reposition_stall_window_s = 2.0
    params.sampling_params.reposition_stall_min_progress = 0.002
    params.sampling_params.reposition_stall_arrival_tolerance = 0.025
    # The production YAML's nominal posture belongs to the seven-DOF Franka.
    # Use the imported xArm scene's actual configured home as the null-space
    # reference; truncating Franka coordinates produces large, saturated
    # torques even when the Cartesian target error is zero.
    plant_context = plant.GetMyContextFromRoot(adapter.context)
    params.repos_ik_params.q_nominal = list(
        np.asarray(plant.GetPositions(plant_context)[:XARM_DOF], dtype=float)
    )
    sampling = SamplingC3Controller(
        base_mpc=base_mpc, plant=plant,
        ee_frame=plant.GetFrameByName("oim_stick_tip"),
        obj_body=block, params=params,
        dt_ctrl=baseline.planning_dt, log_diag=True,
        rng=np.random.default_rng(0), diagram=scene.diagram,
        # OIM is stepped synchronously: simulation time does not advance
        # during the C3+ solve.  Therefore the reference's real-time solve
        # latency timestamp must not be added to the cached OSC trajectory.
        compensate_solve_latency=False,
    )
    # Drake's MJCF parser ignores xArm joint-4 stiffness, while the execution
    # adapter restores the source model's -175*q4 passive torque.  Tell the
    # OSC QP about that same external generalized force so it solves for the
    # actuator torque of the plant it will actually control.
    _j4_q = scene.arm_positions["xarm6_joint4"]
    _j4_v = scene.arm_velocities["xarm6_joint4"]

    def _xarm_passive_spring(osc_context):
        force = np.zeros(plant.num_velocities())
        force[_j4_v] = -175.0 * float(plant.GetPositions(osc_context)[_j4_q])
        return force

    sampling.executor.known_passive_force_fn = _xarm_passive_spring
    # The production Franka tool frame is identity when pointing down.  The
    # xArm MJCF has a different welded tool transform, so preserve its actual
    # configured down-facing orientation as the OSC rotation target.
    sampling.executor._R_target = plant.CalcRelativeTransform(
        plant_context, plant.world_frame(),
        plant.GetFrameByName("oim_stick_tip"),
    ).rotation()
    controller = OimSamplingC3PlusController(sampling, adapter)
    executor = OimXarmOscVelocityBridge(adapter, sampling)
    return OimC3PlusArchitecture(
        scene, adapter, solver, baseline, executor, controller
    )
