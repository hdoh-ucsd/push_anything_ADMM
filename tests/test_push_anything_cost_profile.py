import numpy as np
import pytest
import yaml

from control.push_anything_cost import (
    jacktoy_c3plus_cost_config,
    push_anything_cost_profile,
    resolve_cost_config,
)


@pytest.mark.parametrize("num_objects, expected_dim", [(1, 19), (2, 32), (3, 45), (4, 58)])
def test_multi_object_q_vector_layout(num_objects, expected_dim):
    profile = push_anything_cost_profile(num_objects)
    pose = profile.q_vector(True)
    position = profile.q_vector(False)
    assert pose.shape == position.shape == (expected_dim,)
    for i in range(num_objects):
        np.testing.assert_array_equal(pose[profile.object_quaternion_slice(i)], 0.1)
        np.testing.assert_array_equal(position[profile.object_quaternion_slice(i)], 0.0)


@pytest.mark.parametrize(
    "num_objects,horizon,threads,w_r,w_g",
    [(1, 10, 6, 6, .18), (2, 15, 5, 6, .18),
     (3, 7, 4, 10, .15), (4, 7, 5, 10, .20)],
)
def test_upstream_multi_object_scales(num_objects, horizon, threads, w_r, w_g):
    profile = push_anything_cost_profile(num_objects)
    assert (profile.horizon, profile.num_outer_threads) == (horizon, threads)
    assert (profile.w_R, profile.w_G) == (w_r, w_g)


@pytest.mark.parametrize("num_objects,contacts", [(1, 5), (2, 10), (3, 16), (4, 19)])
def test_multi_object_consensus_vector_expansion(num_objects, contacts):
    profile = push_anything_cost_profile(num_objects)
    vectors = profile.consensus_vectors()
    assert profile.num_contact_pairs == contacts
    assert vectors["g_x"].shape == (6 + 13 * num_objects,)
    assert vectors["g_lambda"].shape == vectors["u_lambda"].shape == (4 * contacts,)
    np.testing.assert_array_equal(vectors["g_lambda"], 2)
    np.testing.assert_array_equal(vectors["g_eta"], 1)
    # EE-object and every object-ground component retain the strong 1000
    # projection weight; object-object and wall components use 1.
    strong = 4 + 12 * num_objects
    np.testing.assert_array_equal(vectors["u_lambda"][:strong], 1000)
    np.testing.assert_array_equal(vectors["u_lambda"][strong:], 1)


def test_single_object_effective_costs_match_anything_n1():
    profile = push_anything_cost_profile(1)
    q_pose, r_pose = profile.tracking_matrices(True)
    q_position, r_position = profile.tracking_matrices(False)
    np.testing.assert_array_equal(np.diag(q_pose), [
        .8, .8, .8, 8, 8, 8, 8, 12000, 12000, 9600,
        1200, 1200, 800, 4, 4, 4, 4, 4, 4,
    ])
    np.testing.assert_array_equal(np.diag(q_position), [
        .5, .5, .5, 0, 0, 0, 0, 10000, 10000, 6000,
        750, 750, 500, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5,
    ])
    np.testing.assert_array_equal(np.diag(r_pose), [.06, .06, 6])
    np.testing.assert_array_equal(r_pose, r_position)


def test_profile_rejects_stale_overrides():
    with pytest.raises(ValueError, match="cannot be overridden"):
        resolve_cost_config({"profile": "push_anything", "w_Q": 45})


def test_profile_keeps_geometry_targets():
    resolved = resolve_cost_config({
        "profile": "push_anything", "z_obj_target": .02,
        "ee_target_z_offset_above_object": .06,
    })
    assert resolved["w_Q"] == 80
    assert resolved["z_obj_target"] == .02


def test_jacktoy_c3plus_profile_matches_both_reference_regimes():
    cost = jacktoy_c3plus_cost_config()
    assert (cost["w_Q"], cost["w_Q_position"], cost["w_R"]) == (45, 45, 1)
    assert cost["q_vector_obj_pos"] == [200, 200, 120]
    assert cost["q_vector_position_obj_pos"] == [1500, 1500, 1500]
    assert cost["q_vector_obj_ang_vel"] == [.05, .05, .05]
    assert cost["q_vector_position_obj_ang_vel"] == [.01, .01, .01]
    assert cost["q_vector_obj_lin_vel"] == [.05, .05, .05]
    assert cost["q_vector_position_obj_lin_vel"] == [1, 1, 1]
    assert cost["r_vector"] == [.01, .01, .01]
    assert cost["q_quaternion_dependent_weight"] == 2500


def test_push_jack_is_reference_literal_boot_and_named_profile():
    with open("config/tasks.yaml") as stream:
        jack = yaml.safe_load(stream)["tasks"]["push_jack"]
    assert jack["goal_mode"] == "kRandom"
    assert jack["goal_success_mode"] == "reference"
    assert jack["cost"]["profile"] == "jacktoy_c3plus"
    resolved = resolve_cost_config(jack["cost"])
    assert resolved["q_vector_position_obj_pos"] == [1500, 1500, 1500]


def test_jacktoy_profile_rejects_stale_overrides():
    with pytest.raises(ValueError, match="cannot be overridden"):
        resolve_cost_config({"profile": "jacktoy_c3plus", "w_Q": 80})


def test_modern_tasks_do_not_repeat_or_override_profile_values():
    with open("config/tasks.yaml") as stream:
        tasks = yaml.safe_load(stream)["tasks"]
    stale = {
        "use_reference_q_vector", "w_Q", "w_Q_position", "w_R", "r_vector",
        "q_vector_ee_pos", "q_vector_obj_quat", "q_vector_obj_pos",
        "q_vector_position_obj_quat", "q_vector_position_obj_pos",
        "q_vector_position_obj_ang_vel", "q_vector_position_obj_lin_vel",
        "q_vector_ee_vel", "q_vector_obj_ang_vel", "q_vector_obj_lin_vel",
        "w_obj_xy", "w_obj_z", "w_box_z", "w_box_rp", "w_yaw",
        "w_ee_approach", "w_torque", "w_terminal",
        "use_quaternion_dependent_cost", "q_quaternion_dependent_weight",
        "q_quaternion_dependent_regularizer_fraction",
    }
    profiled = [cfg["cost"] for cfg in tasks.values()
                if cfg.get("cost", {}).get("profile") == "push_anything"]
    assert len(profiled) == 39
    assert all(not (set(cost) & stale) for cost in profiled)
