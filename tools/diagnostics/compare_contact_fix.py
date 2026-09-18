"""Compare immutable seed-42 baseline and patched passive replay directories.

Run: python tools/diagnostics/compare_contact_fix.py BEFORE AFTER --output NEW_DIR
The output directory must not exist. No controller code is imported or run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.diagnostics import analyze_phantom_contact as legacy


TIMEOUT_REASON = 'kToReposContactAcquisitionTimeout'


def load_rows(path):
    text_fields = legacy.TEXT_FIELDS | {'contact_acquisition_state'}
    with path.open(newline='') as stream:
        rows = [{key: (value if key in text_fields else float(value) if value else math.nan)
                 for key, value in raw.items()} for raw in csv.DictReader(stream)]
    if not rows:
        raise ValueError(f'Empty trajectory: {path}')
    for index, row in enumerate(rows):
        if row['row'] != index:
            raise ValueError(f'Nonconsecutive observation index at {index}')
        if index and row['sim_time'] < rows[index - 1]['sim_time']:
            raise ValueError('Observation time decreases')
        row['row'] = index
        row['csv_line'] = index + 2
    return rows


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def log_outcome(directory, rows):
    """Use the actual tight-goal latch, which includes full orientation error."""
    path = next((directory / name for name in ('main_log.txt', 'terminal.txt')
                 if (directory / name).exists()), None)
    text = path.read_text(errors='replace') if path is not None else ''
    result_lines = re.findall(r'^\[RESULT\].*$', text, flags=re.MULTILINE)
    result = result_lines[-1] if result_lines else ''
    success_match = re.search(r'tight_goal=(PASS|FAIL)\(([^)]*)\)', result)
    xy_match = re.search(r'translational_error=([\d.eE+-]+)m', result)
    orientation_match = re.search(r'rotational_error=([\d.eE+-]+)rad', result)
    latch = re.search(r'^\[ACHIEVED-FIXED-GOAL\] step=(\d+).*event=FIRST_LATCH',
                      text, flags=re.MULTILINE)
    latch_step = int(latch.group(1)) if latch else None
    latch_row = next((row for row in rows if row['cycle_kind'] == 'planner'
                      and row['planner_step'] == latch_step), None)
    goal_yaws = re.findall(r'goal_yaw=([\d.eE+-]+)rad', text)
    final_yaw_error = (abs(legacy.angle_diff(rows[-1]['obj_yaw'], float(goal_yaws[-1])))
                       if goal_yaws else None)
    return {
        'success': success_match.group(1) == 'PASS' if success_match else None,
        'success_source': str(path) if path is not None else None,
        'success_gate': 'existing XY < 0.02 m and full orientation < 0.1 rad; final or sticky achievement latch',
        'success_reason': success_match.group(2) if success_match else None,
        'time_to_goal_s': latch_row['sim_time'] if latch_row else None,
        'first_goal_latch_planner_step': latch_step,
        'final_xy_error_m': float(xy_match.group(1)) if xy_match else None,
        'final_orientation_error_rad': float(orientation_match.group(1)) if orientation_match else None,
        'final_yaw_error_rad': final_yaw_error,
        'final_yaw_observation_time_s': rows[-1]['sim_time'],
        'final_result_line': result,
    }


def analyze_run(directory):
    rows = load_rows(directory / 'trajectory.csv')
    old_summary, episodes, transitions, targets = legacy.analyze(rows)
    patched = 'c3_contact_acquired' in rows[0]
    for episode in episodes:
        entry = rows[episode['entry_row']]
        exit_row = rows[episode['exit_row']]
        segment = rows[episode['entry_row']:episode['last_c3_row'] + 1]
        recorded = [row for row in segment
                    if row.get('c3_contact_acquired') == 1
                    and legacy.finite(row.get('contact_acquired_time'))
                    and entry['sim_time'] <= row['contact_acquired_time'] <= episode['exit_time']]
        # A planner update can acquire contact and leave C3 on the same tick.
        # end() retains that acquisition snapshot on the exit row. Only accept
        # it when its entry timestamp proves it belongs to this episode.
        exit_entry_time = exit_row.get('c3_entry_time')
        exit_acquired_time = exit_row.get('contact_acquired_time')
        acquired_on_exit = (
            episode['completed'] and patched
            and exit_row.get('c3_contact_acquired') == 1
            and legacy.finite(exit_entry_time)
            and math.isclose(exit_entry_time, entry['sim_time'], rel_tol=0.0, abs_tol=1e-9)
            and legacy.finite(exit_acquired_time)
            and entry['sim_time'] <= exit_acquired_time <= episode['exit_time'])
        if acquired_on_exit:
            recorded.append(exit_row)
        acquired_time = (recorded[0]['contact_acquired_time'] if recorded else None) if patched \
            else episode['first_contact_time']
        acquired = acquired_time is not None
        contact_rows = segment + [exit_row] if acquired_on_exit else segment
        contact_row = min(contact_rows, key=lambda row: abs(row['sim_time'] - acquired_time)) \
            if acquired else exit_row
        acquiring = [row for row in segment if row.get('c3_contact_acquired') == 0] if patched else []
        counts = [row.get('normal_progress_counter', row['progress_counter_after_control'])
                  for row in acquiring]
        timeout = episode['completed'] and episode['progress_exit_reason'] == TIMEOUT_REASON
        deadline = entry.get('contact_acquisition_deadline')
        episode.update(
            contact_acquired=acquired,
            contact_acquired_time=acquired_time,
            time_to_contact_s=acquired_time - entry['sim_time'] if acquired else None,
            acquisition_timeout=timeout,
            acquisition_timeout_time_s=exit_row['sim_time'] if timeout else None,
            acquisition_deadline_s=deadline,
            acquisition_deadline_overrun_s=(max(0.0, exit_row['sim_time'] - deadline)
                                            if timeout and legacy.finite(deadline) else None),
            acquisition_failure=timeout,
            object_motion_before_contact_m=legacy.normdiff(legacy.vec(contact_row, 'obj', 'xy'),
                                                           legacy.vec(entry, 'obj', 'xy')),
            object_motion_after_contact_m=(legacy.normdiff(legacy.vec(exit_row, 'obj', 'xy'),
                                                           legacy.vec(contact_row, 'obj', 'xy'))
                                           if acquired else None),
            object_yaw_motion_before_contact_rad=abs(legacy.angle_diff(contact_row['obj_yaw'], entry['obj_yaw'])),
            object_yaw_motion_after_contact_rad=(abs(legacy.angle_diff(exit_row['obj_yaw'], contact_row['obj_yaw']))
                                                if acquired else None),
            max_progress_counter_while_acquiring=max(counts, default=None),
            contactless_progress_counter_violations=sum(value != 0 for value in counts),
            failed_target_memory_size_at_exit=exit_row.get('failed_target_memory_size'),
            exit_reason=episode['progress_exit_reason'],
            selected_candidate_id_at_entry=entry['selected_candidate_id'],
            controller_contact_at_entry=entry.get('controller_physical_contact'),
        )
        body_x = entry.get('attempted_contact_target_body_x')
        body_y = entry.get('attempted_contact_target_body_y')
        if legacy.finite(body_x) and legacy.finite(body_y):
            episode['target_body_x_at_selection'] = body_x
            episode['target_body_y_at_selection'] = body_y
    failures = [episode for episode in episodes if episode['acquisition_timeout']
                or episode['contactless_no_progress_exit']]
    config = read_json(directory / 'runtime_configuration.json', {})
    sampling = config.get('params', {}).get('sampling_params', {})
    radius = float(sampling.get('unsuccessful_radius', 0.01))
    capacity = int(sampling.get('N_unsuccessful_sample_buffer', 10))
    for target in targets:
        prior = [episode for episode in failures if episode['exit_row'] <= target['row']
                 and episode['target_body_x_at_selection'] is not None][-capacity:]
        distances = [math.hypot(target['target_body_x'] - episode['target_body_x_at_selection'],
                                target['target_body_y'] - episode['target_body_y_at_selection'])
                     for episode in prior]
        target['recent_verified_or_baseline_contactless_failures'] = len(prior)
        target['nearest_failed_approach_distance_m'] = min(distances, default=None)
        target['repeated_failed_approach'] = any(distance < radius for distance in distances)
    summary = {
        **log_outcome(directory, rows),
        'c3_entries': len(episodes),
        'repositions': old_summary['total_repositions_including_initial_episode'],
        'contactless_c3_entries': old_summary['C3_entries_without_contact'],
        'contactless_progress_exits': old_summary['contactless_no_progress_exits'],
        'acquisition_timeouts': sum(episode['acquisition_timeout'] for episode in episodes),
        'max_acquisition_deadline_overrun_s': max(
            (episode['acquisition_deadline_overrun_s'] for episode in episodes
             if episode['acquisition_deadline_overrun_s'] is not None), default=None),
        'median_time_to_contact_s': legacy.median(episode['time_to_contact_s'] for episode in episodes),
        'repeated_failed_targets': sum(target['repeated_failed_approach'] for target in targets),
        'failed_target_equivalence_radius_m': radius,
        'failed_target_history_capacity': capacity,
        'progress_counter_violations_while_acquiring': sum(episode['contactless_progress_counter_violations'] for episode in episodes),
        'max_failed_contact_target_memory_size': max((row.get('failed_target_memory_size', 0) for row in rows), default=0),
        'failed_buffer_predicate_rejections': rows[-1]['candidate_rejected_by_failed_buffer_cumulative'],
        'object_translation_m': old_summary['overall_object_translation_m'],
        'episode_count_with_acquired_contact': sum(episode['contact_acquired'] for episode in episodes),
        'observation_end_time_s': rows[-1]['sim_time'],
        'acquisition_state_logged': patched,
        'manifest': read_json(directory / 'manifest.json', {}),
        'trajectory_sha256': hashlib.sha256((directory / 'trajectory.csv').read_bytes()).hexdigest(),
    }
    rejection_path = directory / 'candidate_rejections.jsonl'
    rejection_events = [json.loads(line) for line in rejection_path.read_text().splitlines()] \
        if rejection_path.exists() else []
    summary['recorded_rejection_events'] = len(rejection_events)
    summary['rejection_events_with_failed_contact_memory'] = sum(
        bool(event.get('failed_contact_targets_body_xy')) for event in rejection_events)
    return legacy.clean(summary), legacy.clean(episodes), legacy.clean(transitions), legacy.clean(targets)


METRICS = [
    ('Success', 'success'), ('Time to goal (s)', 'time_to_goal_s'),
    ('C3 entries', 'c3_entries'), ('Repositions (including startup)', 'repositions'),
    ('Contactless C3 entries', 'contactless_c3_entries'),
    ('Contactless progress exits', 'contactless_progress_exits'),
    ('Acquisition timeouts', 'acquisition_timeouts'),
    ('Median time to contact (s)', 'median_time_to_contact_s'),
    ('Repeated failed targets', 'repeated_failed_targets'),
    ('Final XY error (m)', 'final_xy_error_m'), ('Final yaw error (rad)', 'final_yaw_error_rad'),
]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate-order-before', default='not supplied')
    parser.add_argument('--candidate-order-after', default='not supplied')
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        parser.error(f'Refusing to overwrite existing output: {out}')
    before = analyze_run(args.before.resolve())
    after = analyze_run(args.after.resolve())
    out.mkdir(parents=True, exist_ok=False)
    for label, result in (('before', before), ('after', after)):
        folder = out / label
        folder.mkdir()
        summary, episodes, transitions, targets = result
        (folder / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        for name, records in (('episodes', episodes), ('transitions', transitions), ('targets', targets)):
            legacy.write_csv(folder / f'{name}.csv', records)
    comparison = [{'metric': label, 'before': before[0][key], 'after': after[0][key]}
                  for label, key in METRICS]
    comparison.insert(9, {'metric': 'Candidate-order sensitivity',
                          'before': args.candidate_order_before, 'after': args.candidate_order_after})
    legacy.write_csv(out / 'comparison.csv', comparison)
    def display(value):
        if value is None:
            return 'not observed / unavailable'
        return f'{value:.6g}' if isinstance(value, float) else str(value)
    table = '| Metric | Before | After |\n|---|---:|---:|\n'
    table += ''.join(f'| {row["metric"]} | {display(row["before"])} | {display(row["after"])} |\n'
                     for row in comparison)
    table += ('\nSeed 42 is a paired behavioral check, not a success-rate estimate. '
              'The original wall-time-dependent execution behavior is preserved.\n\n'
              'Contactless-entry counts retain the original independent force threshold of 1e-8 N. '
              'Patched acquisition times come from the controller state at its configured threshold; '
              'baseline times use the first independent physical-contact observation. '
              'Motion is net XY displacement, with null post-contact motion when contact was never acquired.\n\n'
              'Repeated-target counts compare object-frame XY with the last configured-capacity failures, '
              'including historical failures after object motion; they are a descriptive recurrence count, '
              'not evidence that a currently retained failure-memory entry was bypassed. '
              'Before failures are contactless progress exits; after failures also include acquisition timeouts.\n\n'
              'Time to goal is the actual FIRST_LATCH planner timestamp; final XY and full orientation '
              'come from the main RESULT line. Final yaw is the last observer sample relative to the logged '
              'goal yaw and is separate from the full-orientation success criterion.\n')
    (out / 'comparison.md').write_text(table)
    (out / 'command.json').write_text(json.dumps(sys.argv, indent=2) + '\n')
    shutil.copy2(__file__, out / 'compare_contact_fix.py')
    shutil.copy2(Path(legacy.__file__), out / 'analyze_phantom_contact.py')
    print(table)


if __name__ == '__main__':
    main()
