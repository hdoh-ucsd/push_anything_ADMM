import numpy as np
import pytest

from control.oim_c3plus_architecture import (
    OimC3PlusBaselineConfig,
    OimXarmOscVelocityBridge,
    build_oim_c3plus_architecture,
    build_oim_sampling_c3plus_architecture,
)
from control.sampling_c3.sampling_based_c3_controller import planar_yaw_error


def test_oim_planar_yaw_error_ignores_roll_pitch():
    # q = roll(0.6) followed by yaw(0.4). Its SE(3) geodesic is larger than
    # 0.4, but OIM's SE(2) evaluator must report only the heading error.
    cr, sr = np.cos(0.3), np.sin(0.3)
    cy, sy = np.cos(0.2), np.sin(0.2)
    q = np.array([cr * cy, sr * cy, sr * sy, cr * sy])
    assert planar_yaw_error(q, 0.0) == pytest.approx(0.4)


def test_architecture_preserves_c3plus_consensus_and_baseline_defaults():
    architecture = build_oim_c3plus_architecture("open_table")
    assert architecture.solver.mode == "c3plus"
    assert architecture.solver.n_x == 19
    assert architecture.solver.n_u == 3
    assert architecture.solver.rho == pytest.approx(100.0)
    assert architecture.config.horizon == 7
    assert architecture.config.admm_iterations == 3
    assert architecture.config.planning_dt == pytest.approx(0.075)
    # Deliberately differs from OIM's optimizer (N=32, 4 rounds, rho=2).
    assert architecture.config != OimC3PlusBaselineConfig(
        horizon=32, admm_iterations=4, rho=2.0
    )


def test_adapter_projects_planar_oim_state_into_c3plus_contract():
    architecture = build_oim_c3plus_architecture("icra_sign")
    state = architecture.adapter.observe()
    assert state.c3plus_state.shape == (19,)
    assert state.arm_positions.shape == (6,)
    assert state.arm_velocities.shape == (6,)
    assert state.object_pose.shape == (3,)
    assert state.goal_pose.shape == (3,)
    np.testing.assert_allclose(state.object_pose[:2], [0.3, 0.4], atol=1e-12)
    np.testing.assert_allclose(state.goal_pose[:2], [0.5, -0.4], atol=1e-12)
    assert np.all(np.isfinite(state.tip_position))
    assert np.all(np.isfinite(state.tip_velocity))


def test_native_velocity_execution_and_force_executor_guard():
    architecture = build_oim_c3plus_architecture("open_table")
    t0 = architecture.adapter.context.get_time()
    architecture.adapter.step_joint_velocity(np.zeros(6), 0.004)
    assert architecture.adapter.context.get_time() == pytest.approx(t0 + 0.004)
    architecture.executor = None
    with pytest.raises(RuntimeError, match="force executor is not connected"):
        architecture.execute_c3plus_action(np.zeros(3))


def test_sampling_c3plus_uses_oim_t_support_witnesses():
    architecture = build_oim_sampling_c3plus_architecture("open_table")
    formulator = architecture.controller.controller.base_mpc.formulator
    cost = architecture.controller.controller.base_mpc.quad_cost
    osc = architecture.controller.controller.executor
    assert osc.known_passive_force_fn is not None
    assert osc.gains.W_posture == pytest.approx(0.01)
    assert not architecture.controller.controller._compensate_solve_latency
    assert cost.w_obj_xy == pytest.approx(1000.0)
    # 400*sin²(dyaw/2) locally equals the reference 100*dyaw².
    assert cost.w_yaw == pytest.approx(400.0)
    assert cost.w_terminal == pytest.approx(10.0)
    assert cost.use_oim_se2_ranking_cost
    assert cost.oim_q_pos == pytest.approx(1000.0)
    assert cost.oim_q_theta == pytest.approx(100.0)
    assert cost.oim_qf_pos == pytest.approx(10000.0)
    assert cost.oim_qf_theta == pytest.approx(1000.0)
    assert cost.z_obj_target == pytest.approx(0.0298)
    assert cost.w_obj_z == pytest.approx(0.0)
    assert cost.w_box_z == pytest.approx(0.0)
    assert cost.w_box_rp == pytest.approx(0.0)
    assert (architecture.controller.controller.params.reposition_params
            .pwl_waypoint_height) == pytest.approx(0.075)
    sampling_params = architecture.controller.controller.params.sampling_params
    assert sampling_params.sampling_height == pytest.approx(0.034)
    assert sampling_params.z_height == pytest.approx(0.034)
    assert not sampling_params.use_adaptive_contact_height
    assert architecture.controller.controller.params.use_planar_yaw_progress
    assert (architecture.controller.controller.params
            .unsuccessful_only_on_observed_failure)
    assert architecture.controller.controller.params.unsuccessful_body_relative
    assert (architecture.controller.controller.params.progress_params
            .pos_regression_threshold) == pytest.approx(0.005)
    assert (architecture.controller.controller.params.progress_params
            .yaw_regression_threshold) == pytest.approx(0.10)
    assert (architecture.controller.controller.params.progress_params
            .regression_consecutive_steps) == 3
    assert formulator._tshape_geometry_variant == "oim_lab"
    assert formulator._controller_object_mass == pytest.approx(1.0)
    # The override is controller-only; simulation retains the imported
    # two-part T's physical 0.10 kg mass.
    plant = architecture.scene.plant
    plant_context = plant.GetMyContextFromRoot(architecture.adapter.context)
    block = plant.GetBodyByName("block")
    assert block.CalcSpatialInertiaInBodyFrame(
        plant_context).get_mass() == pytest.approx(0.1)
    np.testing.assert_allclose(
        formulator._tshape_vertex_set_body_frame(3),
        np.array([
            [-0.0445, +0.0445, 0.0],
            [+0.0099, +0.0099, -0.0794],
            [-0.0298, -0.0298, -0.0298],
        ]),
        atol=1e-12,
    )
    passive = osc.known_passive_force_fn(plant_context)
    assert passive.shape == (plant.num_velocities(),)
    assert np.count_nonzero(passive) == 0  # configured home has q4 = 0
    query = plant.get_geometry_query_input_port().Eval(plant_context)
    phis, J_n, J_t, info = formulator._synthesize_manipuland_ground_contacts(
        plant_context, query
    )
    np.testing.assert_allclose(phis, np.zeros(3), atol=1e-12)
    assert len(J_n) == 3
    assert len(J_t) == 12
    assert [item["tag"] for item in info] == [
        "T-VERT-0", "T-VERT-1", "T-VERT-2"
    ]
    # Composite CoM of the two equal-mass OIM boxes lies inside the support
    # triangle, on its symmetry axis.
    assert -0.0794 < -0.0149 < +0.0099


def test_oim_c3_trajectory_is_active_during_same_planning_interval():
    """Synchronous OIM must not timestamp every plan after its execution."""
    architecture = build_oim_sampling_c3plus_architecture("open_table")
    sampling = architecture.controller.controller
    sampling.base_mpc._filtered_solve_time = 0.35
    assert not sampling._compensate_solve_latency


def test_sampling_c3plus_xarm_osc_bridge_moves_toward_contact():
    architecture = build_oim_sampling_c3plus_architecture("open_table")
    adapter = architecture.adapter
    state = adapter.observe()
    target = np.array([0.397, 0.428, 0.060])
    sampling = architecture.controller.controller
    sampling._last_osc_call = ("osc_direct_free", {
        "p_ee_desired": target,
        "v_ee_desired": np.zeros(3),
        "mode": "free",
    })
    qdot = architecture.executor.desired_joint_velocity(
        adapter.context.get_time())
    assert isinstance(architecture.executor, OimXarmOscVelocityBridge)
    plant = architecture.scene.plant
    pc = plant.GetMyContextFromRoot(adapter.context)
    stick = plant.GetBodyByName("xarm6_stick")
    from pydrake.multibody.tree import JacobianWrtVariable
    J = plant.CalcJacobianTranslationalVelocity(
        pc, JacobianWrtVariable.kV, stick.body_frame(),
        adapter.STICK_TIP_IN_BODY, plant.world_frame(), plant.world_frame(),
    )[:, :6]
    predicted_tip_velocity = J @ qdot
    assert np.dot(predicted_tip_velocity, target - state.tip_position) > 0
    assert abs(qdot[3]) < abs(qdot[0])
    assert np.max(np.abs(qdot)) <= 0.5


def test_xarm_osc_bridge_inverts_source_velocity_servo():
    architecture = build_oim_sampling_c3plus_architecture("open_table")
    bridge = architecture.executor
    sampling = architecture.controller.controller
    sampling.compute_control_osc_only = lambda *_args: np.array(
        [3.0, -6.0, 4.0, -2.0, 1.0, 0.5])
    plant = architecture.scene.plant
    pc = plant.GetMyContextFromRoot(architecture.adapter.context)
    arm_indices = [architecture.scene.arm_velocities[f"xarm6_joint{i}"]
                   for i in range(1, 7)]
    actual = plant.GetVelocities(pc)[arm_indices]
    command = bridge.desired_joint_velocity(
        architecture.adapter.context.get_time())
    reproduced_torque = bridge.SERVO_GAINS * (command - actual)
    np.testing.assert_allclose(reproduced_torque, bridge.last_osc_torque,
                               atol=1e-12)


def test_xarm_reposition_clock_pauses_when_tip_falls_behind():
    architecture = build_oim_sampling_c3plus_architecture("open_table")
    bridge = architecture.executor

    class FakeTrajectory:
        t_start = 1.0
        t_end = 5.0

        def eval(self, t):
            # Deliberately place the requested point far from the real tip.
            return np.array([10.0, 10.0, 10.0]), np.ones(3), False

    traj = FakeTrajectory()
    architecture.controller.controller._last_osc_call = (
        "osc_pwl_free", {"pwl_traj": traj, "mode": "free"}
    )
    assert bridge._osc_evaluation_time(2.0) == pytest.approx(1.0)
    assert bridge._osc_evaluation_time(2.1) == pytest.approx(1.0)
