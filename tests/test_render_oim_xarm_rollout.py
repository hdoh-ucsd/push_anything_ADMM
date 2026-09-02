import importlib.util
from pathlib import Path

from PIL import Image


def load_renderer():
    path = (Path(__file__).resolve().parents[1] /
            "scripts/render_oim_xarm_rollout.py")
    spec = importlib.util.spec_from_file_location("oim_rollout_renderer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_contact_samples_retain_nearest_state_and_instantaneous_receipt():
    renderer = load_renderer()
    records = [{"sim_time": index / 10.0, "step": index}
               for index in range(21)]
    contact = {
        "sim_time": 0.52,
        "episode": 3,
        "force_N": 4.2,
        "depth_m": 0.001,
        "point_W": [0.4, 0.2, 0.0595],
        "classification": "top_graze",
        "relative_height_m": 0.0297,
        "point_to_tip_m": 0.02,
    }
    selected = renderer.select_records(records, 10, [contact], True)
    assert [record["step"] for record in selected] == [0, 4, 5, 6, 10, 20]

    renderer.add_log_context(
        selected,
        [{"time": 0.0, "qdot": [0.0] * 6, "tau": [0.0] * 6}],
        [], [], [contact], {}, 0.052,
    )
    assert selected[2]["_contact"] == contact
    assert selected[1]["_contact"] is None


def test_planner_updates_and_terminal_outcome_are_time_aligned(tmp_path):
    renderer = load_renderer()
    planner_log = tmp_path / "planner.log"
    planner_log.write_text(
        "full_sampling_c3plus_measurement_refresh=PASS phase=contact "
        "latest_utime=1000000 elapsed_updates=0 polls=1\n"
        "full_sampling_c3plus_corrective_phase=PASS phase=contact "
        "updates=25 budget=8000\n"
        "full_sampling_c3plus_measurement_refresh=PASS phase=terminal "
        "latest_utime=2000000 elapsed_updates=0 polls=1\n"
        "full_sampling_c3plus_acceptance_gate=FAIL "
        "reason=terminal_tolerance_failed updates=8000 budget=8000\n"
    )
    updates, budget = renderer.load_planner_updates(planner_log)
    assert updates == [(1.0, 25), (2.0, 8000)]
    assert budget == 8000
    summary = renderer.load_planner_summary(planner_log)
    assert summary["outcome"]["sim_time"] == 2.0

    records = [{"sim_time": 1.5, "step": 15},
               {"sim_time": 2.0, "step": 20}]
    renderer.add_log_context(
        records,
        [{"time": 0.0, "qdot": [0.0] * 6, "tau": [0.0] * 6}],
        [], [], [], summary, 0.052, updates, budget,
    )
    # The counter interpolates between sparse receipts (it genuinely advances
    # through dwell/hold stretches that emit no updates= lines) and never
    # reaches the later observation before its own timestamp.
    assert records[0]["_controller_update"] == round(25 + 0.5 * (8000 - 25))
    assert records[0]["_controller_update"] < 8000
    assert records[1]["_controller_update"] == 8000
    assert records[1]["_controller_budget"] == 8000


def test_terminal_clip_keeps_first_sample_at_or_after_outcome():
    renderer = load_renderer()
    records = [{"sim_time": 0.1 * index, "step": index}
               for index in range(10)]
    clipped = renderer.clip_records_at_or_after(records, 0.45)
    assert [record["step"] for record in clipped] == list(range(6))
    assert renderer.clip_records_at_or_after(records, None) is records


def test_contact_overlay_projects_native_capsule_and_contact_point():
    renderer = load_renderer()
    view = Image.new("RGB", (960, 720), "white")
    record = {
        "capsule_start_W": [0.4, 0.2, 0.18],
        "capsule_end_W": [0.4, 0.2, 0.03],
        "tip_position": [0.4, 0.2, 0.03],
        "_contact": {
            "point_W": [0.4, 0.2, 0.03],
            "classification": "tip_side_candidate",
        },
    }
    rendered = renderer.add_contact_overlay(view, record)
    assert rendered.size == view.size
    assert rendered.tobytes() != view.tobytes()


def test_contact_transaction_parser_and_authoritative_join(tmp_path):
    renderer = load_renderer()
    planner_log = tmp_path / "planner.log"
    planner_log.write_text(
        "full_sampling_c3plus_contact_transaction=ACTIVE "
        "latest_utime=1000000 phase=initial_contact sample=crossbar_top_14 "
        "boundary_W_x=0.38 boundary_W_y=0.42 "
        "normal_W_x=0 normal_W_y=1 predicted_prefix_owner=1 "
        "geometric_contact=1 dwell_owner=1 dwell_steps=42 "
        "signed_gap_m=-0.0001 tip_hold_error_m=0.001\n"
    )
    transactions = renderer.load_contact_transactions(planner_log)
    assert len(transactions) == 1
    assert transactions[0]["sample"] == "crossbar_top_14"

    contact = {
        "sim_time": 1.0,
        "episode": 1,
        "force_N": 4.0,
        "point_W": [0.3805, 0.4198, 0.03],
        "force_on_object_W": [0.0, -4.0, 0.0],
        "classification": "tip_side_candidate",
    }
    # Add a second receipt so the inferred transaction cadence is explicit.
    transactions.append(dict(transactions[0], sim_time=1.02))
    renderer.join_contact_transactions(
        [contact], transactions, 0.00555, 0.003, 0.002)
    join = contact["_join"]
    assert join["accepted_contact"]
    assert join["dwell_owned_contact"]
    assert join["normal_force_on_object_N"] == 4.0
    assert join["face_point_error_m"] < join["face_limit_m"]
    summary = renderer.contact_join_summary([contact])
    assert summary["controller_dwell_claim_observations"] == 1
    assert summary["dwell_owned_contact_observations"] == 1
    assert summary["rejected_controller_dwell_claim_observations"] == 0


def test_contact_join_rejects_wrong_force_polarity_without_hiding_raw_contact():
    renderer = load_renderer()
    transactions = [{
        "sim_time": 2.0,
        "phase": "progress_dwell",
        "sample": "stem_right_2",
        "boundary_W": [0.4, 0.2],
        "normal_W": [1.0, 0.0],
        "predicted_prefix_owner": True,
        "geometric_contact": True,
        "dwell_owner": True,
        "dwell_steps": 10,
        "signed_gap_m": -0.001,
        "tip_hold_error_m": 0.001,
    }, {
        "sim_time": 2.02,
        "phase": "progress_dwell",
        "sample": "stem_right_2",
        "boundary_W": [0.4, 0.2],
        "normal_W": [1.0, 0.0],
        "predicted_prefix_owner": True,
        "geometric_contact": True,
        "dwell_owner": True,
        "dwell_steps": 11,
        "signed_gap_m": -0.001,
        "tip_hold_error_m": 0.001,
    }]
    contact = {
        "sim_time": 2.0,
        "episode": 4,
        "force_N": 3.0,
        "point_W": [0.4, 0.2, 0.03],
        "force_on_object_W": [3.0, 0.0, 0.0],
        "classification": "tip_side_candidate",
    }
    renderer.join_contact_transactions(
        [contact], transactions, 0.00555, 0.003, 0.002)
    assert contact["classification"] == "tip_side_candidate"
    assert not contact["_join"]["accepted_contact"]
    assert contact["_join"]["reason"] == "wrong_force_polarity"
    summary = renderer.contact_join_summary([contact])
    assert summary["controller_dwell_claim_observations"] == 1
    assert summary["dwell_owned_contact_observations"] == 0
    assert summary["rejected_controller_dwell_claim_observations"] == 1


def test_trial_stats_aggregate_matches_requested_schema(tmp_path):
    renderer = load_renderer()
    ledger = tmp_path / "ledger.json"
    trials = renderer.update_trial_ledger(ledger, {
        "run": "run_a", "success": False, "pos_err": 0.70,
        "theta_err": 2.8, "execution_time": 210.0, "steps_to_goal": 8000,
    })
    trials = renderer.update_trial_ledger(ledger, {
        "run": "run_b", "success": True, "pos_err": 0.04,
        "theta_err": 0.06, "execution_time": 150.0, "steps_to_goal": 5200,
    })
    stats = renderer.aggregate_trial_stats(trials)
    assert list(stats.keys()) == [
        "n_trials", "success_rate", "pos_err_mean", "pos_err_std",
        "pos_err_mean_success", "pos_err_std_success", "theta_err_mean",
        "theta_err_std", "theta_err_mean_success", "theta_err_std_success",
        "mean_execution_time", "std_execution_time", "mean_steps_to_goal",
        "std_steps_to_goal", "mean_steps_to_goal_success",
    ]
    assert stats["n_trials"] == 2
    assert stats["success_rate"] == 0.5
    assert abs(stats["pos_err_mean"] - 0.37) < 1e-12
    assert abs(stats["pos_err_mean_success"] - 0.04) < 1e-12
    assert stats["mean_steps_to_goal_success"] == 5200.0

    # Re-adding the same run replaces, not duplicates.
    trials = renderer.update_trial_ledger(ledger, {
        "run": "run_b", "success": True, "pos_err": 0.05,
        "theta_err": 0.06, "execution_time": 150.0, "steps_to_goal": 5200,
    })
    assert len(trials) == 2

    # No successes -> success-conditioned fields are None, not a crash.
    lone = renderer.aggregate_trial_stats([{
        "run": "run_c", "success": False, "pos_err": 0.7,
        "theta_err": 2.9, "execution_time": 200.0, "steps_to_goal": 8000,
    }])
    assert lone["pos_err_mean_success"] is None
    assert lone["mean_steps_to_goal_success"] is None
    assert lone["pos_err_std"] == 0.0
