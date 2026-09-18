"""Analyze passive phantom-contact observations without importing control code.

Usage: python tools/diagnostics/analyze_phantom_contact.py /tmp/REPLAY_DIR
Writes new analysis/ artifacts; refuses to overwrite an existing output directory.
Contact is the observer's independent simulator-force measurement. Target
equivalence uses object-frame coordinates frozen at each target selection, with
the last ten completed contactless no-progress attempts as the recent history.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import sys


TEXT_FIELDS = {
    "cycle_kind", "mode", "mode_switch_reason", "reason_for_c3_to_repos",
    "reason_for_repos_to_c3", "selected_candidate_id",
    "contact_acquisition_state",
}


def finite(value):
    return isinstance(value, (float, int)) and math.isfinite(value)


def median(values):
    values = [value for value in values if finite(value)]
    return statistics.median(values) if values else None


def ratio(count, denominator):
    return count / denominator if denominator else None


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def vec(row, prefix, axes="xyz"):
    return [row.get(f"{prefix}_{axis}", math.nan) for axis in axes]


def normdiff(first, second):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def angle_diff(first, second):
    return math.atan2(math.sin(first - second), math.cos(first - second))


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(clean(row) for row in rows)


def load_rows(path):
    with path.open(newline="") as stream:
        raw = list(csv.DictReader(stream))
    rows = []
    for item in raw:
        row = {}
        for key, value in item.items():
            if key in TEXT_FIELDS:
                row[key] = value
            else:
                row[key] = float(value) if value else math.nan
        row["row"] = int(row["row"])
        row["csv_line"] = row["row"] + 2
        rows.append(row)
    if not rows:
        raise ValueError("Trajectory has no observations")
    for index, row in enumerate(rows):
        if row["row"] != index:
            raise ValueError(f"Nonconsecutive row index at observation {index}")
        if index and row["sim_time"] < rows[index - 1]["sim_time"]:
            raise ValueError("Observation time decreases")
    return rows


def analyze(rows, recent_failures=10):
    # Detect target selections first. Coordinates remain fixed at selection;
    # later object movement does not rewrite a historic approach location.
    targets = []
    active_target = None
    last_mode = None
    for row in rows:
        if row["mode"] == "REPOSITION":
            world = vec(row, "reposition_target")
            changed = (active_target is None or last_mode != "REPOSITION"
                       or normdiff(world, active_target["world"]) > 1e-9)
            if changed and all(finite(value) for value in world):
                yaw = row["obj_yaw"]
                delta_x = world[0] - row["obj_x"]
                delta_y = world[1] - row["obj_y"]
                body_x = math.cos(yaw) * delta_x + math.sin(yaw) * delta_y
                body_y = -math.sin(yaw) * delta_x + math.cos(yaw) * delta_y
                active_target = {
                    "target_id": len(targets) + 1,
                    "row": row["row"], "csv_line": row["csv_line"],
                    "sim_time": row["sim_time"],
                    "selection_kind": ("reposition_entry" if last_mode != "REPOSITION"
                                       else "target_change_within_reposition"),
                    "reason": row["mode_switch_reason"],
                    "selected_candidate_id": row["selected_candidate_id"],
                    "target_matches_selected_candidate": normdiff(world, vec(row, "selected_candidate")) <= 1e-9,
                    "world": world,
                    "target_x": world[0], "target_y": world[1], "target_z": world[2],
                    "anchor_obj_x": row["obj_x"], "anchor_obj_y": row["obj_y"],
                    "anchor_obj_yaw": yaw,
                    "target_body_x": body_x, "target_body_y": body_y,
                    "target_body_angle": math.atan2(body_y, body_x),
                    "target_body_distance": math.hypot(body_x, body_y),
                    "failed_buffer_size": row["failed_buffer_size"],
                }
                targets.append(active_target)
        row["active_target_id"] = active_target["target_id"] if active_target else None
        last_mode = row["mode"]

    episodes = []
    index = 0
    while index < len(rows):
        if rows[index]["mode"] != "C3":
            index += 1
            continue
        start = index
        while index < len(rows) and rows[index]["mode"] == "C3":
            index += 1
        complete = index < len(rows)
        entry = rows[start]
        exit_row = rows[index] if complete else rows[-1]
        segment = rows[start:index]
        duration = exit_row["sim_time"] - entry["sim_time"]
        contacts = [row for row in segment if row["physical_contact"] == 1]
        contact_seconds = sum(
            max(0.0, rows[j + 1]["sim_time"] - rows[j]["sim_time"])
            for j in range(start, min(index, len(rows) - 1))
            if rows[j]["physical_contact"] == 1
        )
        yaw_change = angle_diff(exit_row["obj_yaw"], entry["obj_yaw"])
        target = (targets[entry["active_target_id"] - 1]
                  if entry["active_target_id"] else None)
        reason = exit_row["mode_switch_reason"] if complete else "END_OF_LOG_CENSORED"
        entry_delta_x = entry["ee_x"] - entry["obj_x"]
        entry_delta_y = entry["ee_y"] - entry["obj_y"]
        entry_ee_body_x = (math.cos(entry["obj_yaw"]) * entry_delta_x
                           + math.sin(entry["obj_yaw"]) * entry_delta_y)
        entry_ee_body_y = (-math.sin(entry["obj_yaw"]) * entry_delta_x
                           + math.cos(entry["obj_yaw"]) * entry_delta_y)
        no_progress_exit = (complete and reason == "kToReposUnproductive"
                            and exit_row["dispatcher_met_progress"] == 0)
        episode = {
            "episode_id": len(episodes) + 1,
            "entry_row": entry["row"], "entry_csv_line": entry["csv_line"],
            "exit_row": exit_row["row"], "exit_csv_line": exit_row["csv_line"],
            "last_c3_row": segment[-1]["row"],
            "entry_time": entry["sim_time"], "exit_time": exit_row["sim_time"],
            "duration": duration, "completed": complete,
            "left_censored": start == 0,
            "right_censored": not complete,
            "entry_reason": entry["mode_switch_reason"],
            "progress_exit_reason": reason,
            "no_progress_exit": no_progress_exit,
            "tracker_progress_window_expired_exit": (no_progress_exit
                                                      and exit_row["tracker_met_progress"] == 0),
            "contact_at_entry": bool(entry["physical_contact"]),
            "entry_gap_m": entry["ee_object_gap"],
            "entry_target_distance_m": entry["distance_to_reposition_target"],
            "first_contact_time": contacts[0]["sim_time"] if contacts else None,
            "time_to_first_contact": (contacts[0]["sim_time"] - entry["sim_time"]
                                      if contacts else None),
            "ever_contact": bool(contacts),
            "fraction_of_episode_in_contact": ratio(contact_seconds, duration),
            "contact_observation_fraction": len(contacts) / len(segment),
            "contact_seconds": contact_seconds,
            "minimum_gap_m": min(row["ee_object_gap"] for row in segment),
            "object_translation_m": normdiff(vec(exit_row, "obj", "xy"),
                                             vec(entry, "obj", "xy")),
            "object_translation_xyz_m": normdiff(vec(exit_row, "obj"), vec(entry, "obj")),
            "object_rotation_rad": abs(yaw_change),
            "object_rotation_signed_rad": yaw_change,
            "entry_obj_x": entry["obj_x"], "entry_obj_y": entry["obj_y"],
            "exit_obj_x": exit_row["obj_x"], "exit_obj_y": exit_row["obj_y"],
            "entry_obj_yaw": entry["obj_yaw"], "exit_obj_yaw": exit_row["obj_yaw"],
            "entry_ee_x": entry["ee_x"], "entry_ee_y": entry["ee_y"], "entry_ee_z": entry["ee_z"],
            "entry_ee_body_x": entry_ee_body_x, "entry_ee_body_y": entry_ee_body_y,
            "progress_updates_at_entry_decision": entry["progress_counter"],
            "progress_updates_after_entry_control": entry["progress_counter_after_control"],
            "progress_updates_at_exit_decision": exit_row["progress_counter"],
            "tracker_met_progress_at_exit": exit_row["tracker_met_progress"] if complete else None,
            "dispatcher_met_progress_at_exit": exit_row["dispatcher_met_progress"] if complete else None,
            "target_id": target["target_id"] if target else None,
            "preceding_reposition_target_x": target["target_x"] if target else None,
            "preceding_reposition_target_y": target["target_y"] if target else None,
            "preceding_reposition_target_z": target["target_z"] if target else None,
            "target_selection_row": target["row"] if target else None,
            "target_body_x_at_selection": target["target_body_x"] if target else None,
            "target_body_y_at_selection": target["target_body_y"] if target else None,
            "observations": len(segment),
        }
        episode["contactless_no_progress_exit"] = (
            episode["no_progress_exit"] and not episode["ever_contact"])
        episodes.append(episode)

    failed = [episode for episode in episodes if episode["contactless_no_progress_exit"]
              and episode["target_id"] is not None]
    for target in targets:
        recent = [episode for episode in failed if episode["exit_row"] <= target["row"]]
        recent = recent[-recent_failures:]
        distances = [
            (math.hypot(target["target_body_x"] - episode["target_body_x_at_selection"],
                        target["target_body_y"] - episode["target_body_y_at_selection"]), episode)
            for episode in recent
        ]
        distance, closest = min(distances, key=lambda item: item[0]) if distances else (None, None)
        target["recent_failed_episodes"] = len(recent)
        target["most_recent_failed_episode_id"] = recent[-1]["episode_id"] if recent else None
        target["nearest_failed_episode_id"] = closest["episode_id"] if closest else None
        target["distance_to_recent_failed_target_body_m"] = distance
        entry_distances = [
            (math.hypot(target["target_body_x"] - episode["entry_ee_body_x"],
                        target["target_body_y"] - episode["entry_ee_body_y"]), episode)
            for episode in recent
        ]
        entry_distance, nearest_entry = (min(entry_distances, key=lambda item: item[0])
                                         if entry_distances else (None, None))
        target["distance_to_recent_failed_episode_entry_ee_body_m"] = entry_distance
        target["nearest_failed_episode_entry_ee_id"] = (nearest_entry["episode_id"]
                                                       if nearest_entry else None)
        world_distances = [
            (normdiff([target["target_x"], target["target_y"], target["target_z"]],
                      [episode["entry_ee_x"], episode["entry_ee_y"], episode["entry_ee_z"]]), episode)
            for episode in recent
        ]
        world_distance, nearest_world = (min(world_distances, key=lambda item: item[0])
                                         if world_distances else (None, None))
        target["distance_to_recent_failed_episode_entry_ee_world_xyz_m"] = world_distance
        target["nearest_failed_episode_entry_ee_world_xyz_id"] = (nearest_world["episode_id"]
                                                                 if nearest_world else None)
        for millimeters in (5, 10, 20):
            target[f"repeat_within_{millimeters}mm"] = (distance is not None and distance <= millimeters / 1000)
        linked = [episode for episode in episodes if episode["target_id"] == target["target_id"]]
        target["resulting_c3_episode_id"] = linked[0]["episode_id"] if linked else None
        target["resulting_c3_contactless_no_progress"] = bool(linked and linked[0]["contactless_no_progress_exit"])

    transitions = [{
        "row": row["row"], "csv_line": row["csv_line"],
        "sim_time": row["sim_time"], "from_mode": rows[i - 1]["mode"] if i else None,
        "to_mode": row["mode"], "reason": row["mode_switch_reason"],
        "physical_contact": row["physical_contact"], "gap_m": row["ee_object_gap"],
        "target_distance_m": row["distance_to_reposition_target"],
        "progress_counter": row["progress_counter"],
        "progress_counter_after_control": row["progress_counter_after_control"],
        "tracker_met_progress": row["tracker_met_progress"],
        "dispatcher_met_progress": row["dispatcher_met_progress"],
        "progress_metric": row["progress_metric"], "progress_window_front": row["progress_window_front"],
        "finished_repos_decision": row["finished_repos_decision"],
        "failed_buffer_size": row["failed_buffer_size"],
        "obj_x": row["obj_x"], "obj_y": row["obj_y"], "obj_yaw": row["obj_yaw"],
    } for i, row in enumerate(rows) if row["mode_switch_event"] == 1]
    completed = [episode for episode in episodes if episode["completed"]]
    no_contact = [episode for episode in episodes if not episode["ever_contact"]]
    eligible_targets = [target for target in targets if target["recent_failed_episodes"]]
    selected_targets = [target for target in targets if target["target_matches_selected_candidate"]]
    eligible_selected_targets = [target for target in eligible_targets if target["target_matches_selected_candidate"]]
    immediate_after_failed_targets = [
        target for target in selected_targets
        if any(episode["exit_row"] == target["row"] for episode in failed)
    ]
    repeat_summary = {}
    for millimeters in (5, 10, 20):
        repeated = [target for target in eligible_targets if target[f"repeat_within_{millimeters}mm"]]
        repeated_failures = [target for target in repeated if target["resulting_c3_contactless_no_progress"]]
        repeated_selected = [target for target in repeated if target["target_matches_selected_candidate"]]
        immediate_repeats = [target for target in immediate_after_failed_targets
                             if target[f"repeat_within_{millimeters}mm"]]
        repeat_summary[str(millimeters)] = {
            "radius_m": millimeters / 1000,
            "repeated_failed_target_count": len(repeated),
            "eligible_target_count_after_prior_contactless_no_progress_exit": len(eligible_targets),
            "fraction_of_eligible_target_selections": ratio(len(repeated), len(eligible_targets)),
            "fraction_of_all_target_selections": ratio(len(repeated), len(targets)),
            "repeated_selected_candidate_target_count": len(repeated_selected),
            "eligible_selected_candidate_target_count": len(eligible_selected_targets),
            "fraction_of_eligible_selected_candidate_target_selections": ratio(len(repeated_selected), len(eligible_selected_targets)),
            "fraction_of_all_selected_candidate_target_selections": ratio(len(repeated_selected), len(selected_targets)),
            "repeated_immediate_after_contactless_no_progress_exit_count": len(immediate_repeats),
            "immediate_after_contactless_no_progress_exit_target_count": len(immediate_after_failed_targets),
            "fraction_immediate_after_contactless_no_progress_exit": ratio(len(immediate_repeats), len(immediate_after_failed_targets)),
            "repeated_target_followed_by_another_contactless_no_progress_exit": len(repeated_failures),
            "repeated_target_ids": [target["target_id"] for target in repeated],
            "repeated_failed_c3_episode_ids": [target["resulting_c3_episode_id"] for target in repeated_failures],
        }
    churn_links = []
    for target in eligible_targets:
        if not (target["repeat_within_10mm"] and target["resulting_c3_contactless_no_progress"]):
            continue
        first = episodes[target["nearest_failed_episode_id"] - 1]
        second = episodes[target["resulting_c3_episode_id"] - 1]
        churn_links.append({
            "prior_failed_episode_id": first["episode_id"],
            "repeated_target_id": target["target_id"],
            "repeated_failed_episode_id": second["episode_id"],
            "target_distance_body_m": target["distance_to_recent_failed_target_body_m"],
            "interval_entry_row": first["entry_row"], "interval_exit_row": second["exit_row"],
            "interval_entry_time": first["entry_time"], "interval_exit_time": second["exit_time"],
            "interval_object_translation_including_reposition_m": math.hypot(
                second["exit_obj_x"] - first["entry_obj_x"],
                second["exit_obj_y"] - first["entry_obj_y"]),
            "prior_failed_c3_translation_m": first["object_translation_m"],
            "repeated_failed_c3_translation_m": second["object_translation_m"],
        })
    planner_rows = [row for row in rows if row["cycle_kind"] == "planner"]
    dt = [rows[i + 1]["sim_time"] - row["sim_time"] for i, row in enumerate(rows[:-1])]
    summary = {
        "observation_start_time": rows[0]["sim_time"], "observation_end_time": rows[-1]["sim_time"],
        "observation_rows": len(rows), "planner_rows": len(planner_rows),
        "median_observation_interval_s": median(dt), "max_observation_interval_s": max(dt, default=0),
        "total_C3_entries": len(episodes),
        "total_C3_entries_from_reposition": sum(t["to_mode"] == "C3" for t in transitions),
        "total_repositions_including_initial_episode": sum(t["selection_kind"] == "reposition_entry" for t in targets),
        "total_C3_to_reposition_transitions": sum(t["to_mode"] == "REPOSITION" for t in transitions),
        "total_target_selections_including_retargets": len(targets),
        "total_targets_matching_selected_candidate": len(selected_targets),
        "total_override_targets_not_matching_selected_candidate": len(targets) - len(selected_targets),
        "completed_C3_episodes": len(completed), "right_censored_C3_episodes": len(episodes) - len(completed),
        "C3_entries_without_contact": sum(not episode["contact_at_entry"] for episode in episodes),
        "fraction_C3_entries_without_contact": ratio(sum(not e["contact_at_entry"] for e in episodes), len(episodes)),
        "C3_entries_with_gap_over_2mm": sum(e["entry_gap_m"] > 0.002 for e in episodes),
        "C3_episodes_never_acquiring_contact_observed": len(no_contact),
        "fraction_C3_episodes_never_acquiring_contact_observed": ratio(len(no_contact), len(episodes)),
        "completed_C3_episodes_never_acquiring_contact": sum(not e["ever_contact"] for e in completed),
        "fraction_completed_C3_episodes_never_acquiring_contact": ratio(sum(not e["ever_contact"] for e in completed), len(completed)),
        "no_progress_exits": sum(episode["no_progress_exit"] for episode in episodes),
        "tracker_progress_window_expired_exits": sum(e["tracker_progress_window_expired_exit"] for e in episodes),
        "contactless_no_progress_exits": len(failed),
        "median_time_to_contact_s_among_acquiring_episodes": median(e["time_to_first_contact"] for e in episodes),
        "median_object_translation_m_per_completed_C3_episode": median(e["object_translation_m"] for e in completed),
        "median_object_rotation_rad_per_completed_C3_episode": median(e["object_rotation_rad"] for e in completed),
        "median_contactless_no_progress_translation_m": median(e["object_translation_m"] for e in failed),
        "max_contactless_no_progress_translation_m": max((e["object_translation_m"] for e in failed), default=None),
        "median_contactless_no_progress_rotation_rad": median(e["object_rotation_rad"] for e in failed),
        "overall_object_translation_m": normdiff(vec(rows[-1], "obj", "xy"), vec(rows[0], "obj", "xy")),
        "failed_buffer_checks": rows[-1]["failed_buffer_checks_cumulative"],
        "candidate_rejected_by_failed_buffer": rows[-1]["candidate_rejected_by_failed_buffer_cumulative"],
        "max_failed_buffer_size": max(row["failed_buffer_size"] for row in rows),
        "repeat_comparison_recent_failure_capacity": recent_failures,
        "repeat_distance_sensitivity_mm": repeat_summary,
        "analysis_methodology_version": 2,
        "repeated_failed_target_count": repeat_summary["10"]["repeated_failed_target_count"],
        "repeated_failed_target_fraction": repeat_summary["10"]["fraction_of_eligible_target_selections"],
        "repeated_failed_target_fraction_denominator": "all new targets after at least one completed contactless no-progress C3 episode, including goal-retreat targets; immediate-post-failure fraction is reported separately",
        "raw_all_target_repeated_failed_target_fraction": repeat_summary["10"]["fraction_of_eligible_target_selections"],
        "immediate_after_contactless_no_progress_repeated_target_fraction": repeat_summary["10"]["fraction_immediate_after_contactless_no_progress_exit"],
        "same_target_contactless_no_progress_churn_links_10mm": churn_links,
        "method_notes": [
            "Contact uses physical_contact from independent Drake contact-force API; mode is never used to infer contact.",
            "C3 episode is [entry planner time, next REPOSITION planner time); exit object pose is included in net motion.",
            "Contact fraction integrates each sampled contact flag to the next observation; time-to-contact resolution is reported above.",
            "Last episode, if C3, is right-censored at the last observation; never-acquired counts separate completed episodes.",
            "Targets are distinct on reposition entry or world-coordinate change exceeding 1 nm; body coordinates are frozen at selection.",
            "Candidate matching does not identify approach versus retreat: the controller may insert its home retreat as the selected candidate. Headline denominator therefore includes all target selections, including retreat.",
            "Failed targets for churn are preceding targets of completed contactless kToReposUnproductive episodes, not all buffer entries.",
            "Target equivalence sensitivity is 5/10/20 mm; 10 mm matches the configured default failed-sample rejection radius.",
            "No-progress requires emitted kToReposUnproductive AND dispatcher_met_progress=0; tracker expiry is separately counted because overrides may emit the same reason.",
        ],
    }
    for target in targets:
        del target["world"]
    return clean(summary), episodes, transitions, targets


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("replay_directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--recent-failures", type=int, default=10)
    args = parser.parse_args()
    if args.recent_failures < 1:
        parser.error("--recent-failures must be positive")
    source = args.replay_directory.resolve()
    output = args.output.resolve() if args.output else source / "analysis"
    csv_path = source / "trajectory.csv"
    rows = load_rows(csv_path)
    summary, episodes, transitions, targets = analyze(rows, args.recent_failures)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    summary["replay_completed"] = manifest.get("completed")
    summary["replay_error"] = manifest.get("error")
    summary["seed"] = manifest.get("seed")
    summary["git_commit"] = manifest.get("git_commit")
    summary["source_trajectory"] = str(csv_path)
    summary["trajectory_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    summary["analysis_command"] = sys.argv
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "episodes.csv", episodes)
    write_csv(output / "transitions.csv", transitions)
    write_csv(output / "target_attempts.csv", targets)
    write_csv(output / "churn_links.csv", summary["same_target_contactless_no_progress_churn_links_10mm"])
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    shutil.copy2(__file__, output / "analyze_phantom_contact.py")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
