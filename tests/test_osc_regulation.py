"""STEP 5 — Regulation test for OperationalSpaceController.

Drive the EE 1 cm in +x from rest. Verify settling within 1 mm and no
overshoot beyond 5 mm.
"""
from __future__ import annotations

import numpy as np
import pytest

ad = pytest.importorskip("pydrake.all",
                         reason="Drake required for OSC regulation test")
import yaml

from sim.env_builder import build_environment, EE_BODY_NAME, INITIAL_ARM_Q
from control.osc.operational_space_controller import OperationalSpaceController


def _build_diagram(setpoint_offset: np.ndarray):
    with open("config/tasks.yaml") as f:
        task_cfg = yaml.safe_load(f)["tasks"]["pushing"]
    with open("config/osc_franka.yaml") as f:
        gains_cfg = yaml.safe_load(f)
    gains_cfg["q_home_arm"] = list(INITIAL_ARM_Q)

    diagram_builder = ad.DiagramBuilder()
    # Build a fresh plant+scene for this test — we don't want the meshcat
    # server and visualizer overhead of the project's build_environment.
    plant, scene_graph = ad.AddMultibodyPlantSceneGraph(diagram_builder,
                                                       time_step=1e-3)
    parser = ad.Parser(plant)
    # The project's env_builder constructs the SDF from cfg; rebuilding
    # here would duplicate work. Instead reuse build_environment and
    # pull its plant — the meshcat overhead is tolerable for a single
    # 2-s sim.
    diagram_builder = None  # discard scaffold; reuse env_builder
    diagram, plant, panda_model, _, _, _, _ = build_environment(task_cfg)
    # Wrap in a new builder to add OSC + setpoint source.
    builder = ad.DiagramBuilder()
    builder.AddNamedSystem("env", diagram)
    n_a = plant.num_actuators()

    # OSC.
    osc = OperationalSpaceController(plant, EE_BODY_NAME, gains_cfg, n_arm=n_a)
    builder.AddNamedSystem("osc", osc)

    # The original env diagram doesn't expose the plant state in its top
    # output ports; we connect through the plant's own ports by drilling
    # into the embedded plant directly. Use Drake's
    # GetSystemByName / GetInputPortByName carefully. Simpler approach:
    # construct an independent plant for the sim so we have explicit
    # ports.
    return None  # this draft is replaced below with a cleaner impl.


def _make_plant_and_diagram(task_cfg, gains_cfg):
    """Construct a minimal Drake diagram: plant + OSC + constant
    setpoint source. Returns (diagram, plant, osc, q_init)."""
    builder = ad.DiagramBuilder()
    plant, scene_graph = ad.AddMultibodyPlantSceneGraph(
        builder, time_step=1e-3,
    )
    # Load the same Franka + manipuland the production builder uses.
    # We reuse the same SDF strings via Parser. The simplest path is to
    # let the production builder construct its own diagram (which it
    # does in build_environment), but then we'd inherit meshcat etc.
    # For Phase 2 we'll just import the env builder's diagram and treat
    # it as a black box, accessing plant queries via the OSC's private
    # context. The OSC doesn't need the full sim diagram — it needs the
    # plant + ee frame to be queryable.
    raise NotImplementedError("see _build_minimal_sim below")


def _build_minimal_sim():
    """Build a sim using the project's env_builder, then add OSC as an
    external LeafSystem connected to the plant's state output and
    actuation input.

    Returns (simulator, plant, osc, q_init).
    """
    with open("config/tasks.yaml") as f:
        task_cfg = yaml.safe_load(f)["tasks"]["pushing"]
    with open("config/osc_franka.yaml") as f:
        gains_cfg = yaml.safe_load(f)
    gains_cfg["q_home_arm"] = list(INITIAL_ARM_Q)

    diagram, plant, panda_model, _, _, _, _ = build_environment(task_cfg)
    n_a = plant.num_actuators()
    n_q = plant.num_positions()
    n_v = plant.num_velocities()
    ee_frame = plant.GetFrameByName(EE_BODY_NAME)

    # Initial state.
    q_init = np.zeros(n_q)
    q_init[:n_a] = INITIAL_ARM_Q
    q_init[n_a]     = 1.0
    q_init[n_a+1:n_a+4] = 0.0
    q_init[n_a+4:n_a+7] = task_cfg["init_xyz"]

    # OSC uses its own scratch context, not the sim diagram's plant ctx.
    osc = OperationalSpaceController(plant, EE_BODY_NAME, gains_cfg, n_arm=n_a)
    return diagram, plant, panda_model, ee_frame, osc, q_init


@pytest.mark.timeout(60)
def test_regulation_static_target():
    """Hold EE 1 cm offset from initial pose; verify settling."""
    diagram, plant, panda_model, ee_frame, osc, q_init = _build_minimal_sim()

    # Use a hand-rolled stepping loop rather than wiring through Drake's
    # diagram framework — the env_builder produces a Diagram with
    # internal state-output wiring that's awkward to re-route. The OSC
    # is a pure function of (q, v, setpoint), so we can call its QP
    # builder directly in a loop and FixValue on the plant.
    from control.osc.qp_builder import build_osc_qp
    from pydrake.solvers import OsqpSolver

    simulator = ad.Simulator(diagram)
    context   = simulator.get_mutable_context()
    plant_ctx = plant.GetMyContextFromRoot(context)

    plant.SetPositions(plant_ctx, panda_model, INITIAL_ARM_Q)
    plant.SetVelocities(plant_ctx, np.zeros(n_v := plant.num_velocities()))

    # Set the manipuland pose.
    with open("config/tasks.yaml") as f:
        task_cfg = yaml.safe_load(f)["tasks"]["pushing"]
    obj_body = plant.GetBodyByName(task_cfg["link_name"])
    plant.SetFreeBodyPose(plant_ctx, obj_body,
        ad.RigidTransform(ad.RotationMatrix(), task_cfg["init_xyz"]))

    # Target = current EE + [+1cm, 0, 0].
    x_initial = plant.CalcPointsPositions(
        plant_ctx, ee_frame, np.zeros(3), plant.world_frame()).flatten()
    x_des = x_initial + np.array([0.01, 0.0, 0.0])

    n_a = plant.num_actuators()
    gains_cfg = yaml.safe_load(open("config/osc_franka.yaml"))
    solver = OsqpSolver()

    dt_ctrl = 1e-3   # sim-time step
    T = 2.0
    n_steps = int(T / dt_ctrl)
    x_traj = np.zeros((n_steps + 1, 3))
    tau_max_seen = np.zeros(n_a)

    x_traj[0] = x_initial

    # Use a private scratch context for OSC queries.
    osc_ctx = plant.CreateDefaultContext()

    for step in range(n_steps):
        q = plant.GetPositions(plant_ctx)
        v = plant.GetVelocities(plant_ctx)
        plant.SetPositions(osc_ctx, q)
        plant.SetVelocities(osc_ctx, v)

        prog, dv = build_osc_qp(
            plant, osc_ctx, ee_frame,
            x_des=x_des, xd_des=np.zeros(3),
            q_home_arm=np.asarray(INITIAL_ARM_Q),
            q_now_arm=q[:n_a],
            v_now_arm=v[:n_a],
            Kp_task=np.asarray(gains_cfg["Kp_task"]),
            Kd_task=np.asarray(gains_cfg["Kd_task"]),
            Kp_posture=np.asarray(gains_cfg["Kp_posture"], dtype=float),
            Kd_posture=np.asarray(gains_cfg["Kd_posture"], dtype=float),
            W_task=float(gains_cfg["W_task"]),
            W_posture=float(gains_cfg["W_posture"]),
            torque_limits=np.asarray(gains_cfg["torque_limits"]),
            w_tau_reg=float(gains_cfg.get("w_tau_reg", 1e-4)),
            n_arm=n_a,
        )
        res = solver.Solve(prog)
        assert res.is_success(), (
            f"step {step}: OSC QP failed — {res.get_solution_result()}")

        tau = res.GetSolution(dv["tau"])
        tau_max_seen = np.maximum(tau_max_seen, np.abs(tau))
        plant.get_actuation_input_port().FixValue(plant_ctx, tau)

        simulator.AdvanceTo((step + 1) * dt_ctrl)

        x_now = plant.CalcPointsPositions(
            plant_ctx, ee_frame, np.zeros(3), plant.world_frame()).flatten()
        x_traj[step + 1] = x_now

    # ---- Assertions ----------------------------------------------------
    final_err = float(np.linalg.norm(x_traj[-1] - x_des))
    # Overshoot in +x direction (the only commanded axis).
    x_des_x = x_des[0]
    max_overshoot = float(np.max(x_traj[:, 0] - x_des_x))
    final_v = float(np.linalg.norm(
        plant.GetVelocities(plant_ctx)[:plant.num_actuators()]))

    print(f"\n[regulation] x_initial = {x_initial.round(4)}")
    print(f"[regulation] x_des     = {x_des.round(4)}")
    print(f"[regulation] x_final   = {x_traj[-1].round(4)}")
    print(f"[regulation] final err = {final_err*1000:.2f} mm")
    print(f"[regulation] max +x overshoot = {max_overshoot*1000:+.2f} mm")
    print(f"[regulation] final joint-vel norm = {final_v:.4f} rad/s")
    print(f"[regulation] peak |tau| per joint = "
          f"{tau_max_seen.round(2).tolist()}")

    assert final_err < 1e-3, (
        f"settling error {final_err*1000:.2f} mm > 1 mm target")
    assert max_overshoot < 5e-3, (
        f"overshoot {max_overshoot*1000:.2f} mm > 5 mm allowed")
    assert final_v < 1e-2, (   # 10 mm/s on joint vel norm — generous
        f"final joint-velocity norm {final_v:.4f} indicates not settled")
