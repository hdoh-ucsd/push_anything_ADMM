"""Passive observer for one main.py C3+ run (seed 42).

Run: python tools/diagnostics/audit_phantom_contact.py --output /tmp/UNIQUE_DIR
All control methods are called exactly once with unchanged arguments/returns.
Measurements and CSV I/O run after their return, outside the controller's
full-tick solve-time measurement. The progress predicate wrapper only copies
scalars, preserving its return value. No random draws or parameter edits.
The baseline itself feeds measured wall time into execution; seed 42 fixes
random draws, but does not promise bitwise replay determinism.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def jsonable(value):
    import numpy as np
    if dataclasses.is_dataclass(value):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-time', type=float, default=60.0)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    import numpy as np
    from pydrake.math import RollPitchYaw
    from pydrake.common.eigen_geometry import Quaternion
    from control.sampling_c3.progress import ProgressTracker
    from control.sampling_c3.sample_buffer import UnsuccessfulSampleBuffer
    from control.sampling_c3.sampling_based_c3_controller import SamplingC3Controller

    random.seed(42)
    np.random.seed(42)
    config = 'config/sampling_c3_kik.yaml'
    run_name = 'phantom_contact_seed42_' + out.name
    argv = ['main.py', 'pushing', '--sampling-c3', config, '--solver',
            'c3plus', '--seed', '42', '--max-time', str(args.max_time),
            '--name', run_name]
    manifest = {
        'seed': 42, 'cwd': str(ROOT), 'observer_argv': sys.argv,
        'main_argv': argv, 'output': str(out),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'python': sys.version, 'python_executable': sys.executable,
        'versions': {p: importlib.metadata.version(p) for p in
                     ['drake', 'numpy', 'scipy', 'osqp', 'PyYAML', 'pytest']},
        'control_environment': {k: v for k, v in os.environ.items()
                                if k.startswith(('PORT_', 'REFCONF_', 'DIAG_', 'OMP_', 'MKL_', 'OPENBLAS_'))},
        'physical_contact_definition': 'matching Drake point/hydroelastic EE-object contact with force norm > 1e-8 N',
        'controller_contact_definition': 'controller physical-force measurement above runtime contact_force_threshold; distinct from the unchanged raw observer threshold',
        'gap_definition': 'minimum SceneGraph signed distance over actual simulation EE-object geometry pairs, metres',
        'contact_threshold_m': 0.0,
        'proximity_threshold_m_for_comparison': 0.002,
        'timing_limitation': 'baseline measured wall-clock solve time affects execution; seed fixes RNG, not wall time',
    }
    (out / 'git_status_before.txt').write_text(subprocess.check_output(['git', 'status', '--short'], text=True))
    (out / 'git_worktrees.txt').write_text(subprocess.check_output(['git', 'worktree', 'list'], text=True))
    source_files = [Path('main.py'), Path('environment.yml'), Path('AGENTS.md')]
    source_files += list(Path('control').rglob('*.py')) + list(Path('sim').rglob('*.py'))
    source_files += list(Path('config').rglob('*.yaml'))
    manifest['sha256_before'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    shutil.copy2(config, out / 'sampling_config.yaml')
    shutil.copy2('config/tasks.yaml', out / 'tasks.yaml')
    shutil.copy2(__file__, out / 'observer.py')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (out / 'command.txt').write_text(' '.join(sys.argv) + '\nmain argv: ' + ' '.join(argv) + '\n')

    # Scalars only; no I/O or re-evaluation of a predicate inside control.
    original_met = ProgressTracker.met_progress
    def observed_met(self, near_goal):
        result = original_met(self, near_goal)
        history = self._config_cost_history
        self._audit_decision = (self._n_updates, self.steps_since_improve(),
                               history[-1] if history else math.nan,
                               history[0] if history else math.nan, bool(result))
        return result
    ProgressTracker.met_progress = observed_met
    original_avoids = UnsuccessfulSampleBuffer.sample_avoids_bad_spots
    def observed_avoids(self, *a, **kw):
        result = original_avoids(self, *a, **kw)
        self._audit_checks = getattr(self, '_audit_checks', 0) + 1
        self._audit_rejections = getattr(self, '_audit_rejections', 0) + int(not result)
        if not result:
            # Copy only values here; JSON encoding and I/O stay outside control.
            candidate = a[0] if a else kw.get('ee_candidate')
            obj_xy = a[1] if len(a) > 1 else kw.get('obj_pos_xy_now')
            obj_quat = a[2] if len(a) > 2 else kw.get('obj_quat_now')
            pending = getattr(self, '_audit_pending_rejections', [])
            pending.append(dict(
                check_id=self._audit_checks,
                candidate_world=tuple(float(v) for v in candidate),
                object_xy=None if obj_xy is None else tuple(float(v) for v in obj_xy),
                object_quaternion=None if obj_quat is None else tuple(float(v) for v in obj_quat),
                failed_contact_targets_body_xy=[
                    tuple(float(v) for v in entry.contact_target_body_xy)
                    for entry in self
                    if getattr(entry, 'contact_target_body_xy', None) is not None],
            ))
            self._audit_pending_rejections = pending
        return result
    UnsuccessfulSampleBuffer.sample_avoids_bad_spots = observed_avoids

    stream = (out / 'trajectory.csv').open('x', newline='', buffering=1024 * 1024)
    rejection_stream = (out / 'candidate_rejections.jsonl').open('x')
    writer = None
    row_count = 0
    entry_pose = None
    entry_ee_pose = None
    preceding_target = None
    decision = None
    saved_params = False

    def observe(self, q, ctx, kind, prior_mode=None, prior_target=None):
        nonlocal writer, row_count, entry_pose, entry_ee_pose, preceding_target, decision, saved_params
        sim_time = float(ctx.get_time())
        plan = self._last_plan_ctx
        mode = self.last_mode
        event = kind == 'planner' and prior_mode != mode
        form = self.base_mpc.formulator
        ee_ids = form._ee_geom_ids
        obj_ids = form._manipuland_geom_ids
        def pair_matches(a, b):
            return ((a in ee_ids and b in obj_ids) or (b in ee_ids and a in obj_ids))
        query = self.plant.get_geometry_query_input_port().Eval(ctx)
        gaps = [float(query.ComputeSignedDistancePairClosestPoints(a, b).distance)
                for a in ee_ids for b in obj_ids]
        gap = min(gaps) if gaps else math.nan
        contact = self.plant.get_contact_results_output_port().Eval(ctx)
        forces = []
        for i in range(contact.num_point_pair_contacts()):
            info = contact.point_pair_contact_info(i)
            pair = info.point_pair()
            if pair_matches(pair.id_A, pair.id_B):
                forces.append(float(np.linalg.norm(info.contact_force())))
        for i in range(contact.num_hydroelastic_contacts()):
            info = contact.hydroelastic_contact_info(i)
            surface = info.contact_surface()
            if pair_matches(surface.id_M(), surface.id_N()):
                forces.append(float(np.linalg.norm(info.F_Ac_W().translational())))
        ee = self.plant.CalcPointsPositions(ctx, self.ee_frame, np.zeros(3), self.world_frame).reshape(3)
        obj = np.array([q[self._obj_x_idx], q[self._obj_y_idx], q[self._obj_z_idx]])
        quat = np.asarray([q[self._obj_qw], q[self._obj_qx], q[self._obj_qy], q[self._obj_qz]])
        yaw = float(RollPitchYaw(Quaternion(quat / np.linalg.norm(quat))).yaw_angle())
        if kind == 'planner':
            decision = getattr(self.progress, '_audit_decision', (0, 0, math.nan, math.nan, True))
            if event and mode == 'c3':
                entry_pose = (obj.copy(), yaw)
                entry_ee_pose = ee.copy()
                preceding_target = prior_target
            if not saved_params:
                runtime = {'params': self.params, 'dt_ctrl': self._dt_ctrl,
                           'dt_mpc': self._dt_mpc, 'dt_osc': self._dt_osc,
                           'solver_penalize_input_change': self.base_mpc.solver._penalize_input_change,
                           'resolved_contact_acquisition_timeout_s': getattr(
                               getattr(self, 'contact_acquisition', None), 'timeout_s', None)}
                (out / 'runtime_configuration.json').write_text(json.dumps(jsonable(runtime), indent=2))
                saved_params = True
        target = self._current_repos_target if mode != 'c3' else preceding_target
        target = np.asarray(target) if target is not None else np.full(3, np.nan)
        index = int(self._best_sample_index) if mode != 'c3' else 0
        samples = plan['samples']
        selected = np.asarray(samples[index]) if 0 <= index < len(samples) else np.full(3, np.nan)
        label = plan['labels'][index] if 0 <= index < len(plan['labels']) else ''
        body = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]]) @ (target[:2] - obj[:2])
        displacement = float(np.linalg.norm(obj[:2] - entry_pose[0][:2])) if entry_pose is not None else math.nan
        dyaw = math.atan2(math.sin(yaw-entry_pose[1]), math.cos(yaw-entry_pose[1])) if entry_pose is not None else math.nan
        reason = self.last_switch_reason.name
        acquisition = getattr(self, 'contact_acquisition', None)
        def acquisition_scalar(name):
            value = getattr(acquisition, name, None)
            return math.nan if value is None else float(value)
        acquisition_target = getattr(acquisition, 'target_world', None)
        acquisition_target = (np.full(3, np.nan) if acquisition_target is None
                              else np.asarray(acquisition_target).reshape(3))
        acquisition_body = getattr(acquisition, 'target_body_xy', None)
        acquisition_body = (np.full(2, np.nan) if acquisition_body is None
                            else np.asarray(acquisition_body).reshape(2))
        contact_threshold = float(getattr(self.params, 'contact_force_threshold', 1e-6))
        controller_force = float(getattr(self, '_physical_contact_force_norm', math.nan))
        acquired = bool(getattr(acquisition, 'contact_acquired', False))
        active = bool(getattr(acquisition, 'active', False))
        entry_object_state = getattr(acquisition, 'entry_object_pose', None)
        entry_quat = (np.full(4, np.nan) if entry_object_state is None
                      else np.asarray(entry_object_state)[:4])
        failure_memory_size = sum(
            getattr(entry, 'contact_target_body_xy', None) is not None
            for entry in self.unsuccessful_buffer)
        row = dict(row=row_count, cycle_kind=kind, planner_step=self._step, sim_time=sim_time,
                   mode='C3' if mode == 'c3' else 'REPOSITION',
                   mode_switch_event=int(event), mode_switch_reason=reason,
                   reason_for_c3_to_repos=reason if event and prior_mode == 'c3' else '',
                   reason_for_repos_to_c3=reason if event and mode == 'c3' else '',
                   ee_x=ee[0], ee_y=ee[1], ee_z=ee[2],
                   obj_x=obj[0], obj_y=obj[1], obj_z=obj[2], obj_yaw=yaw,
                   obj_qw=quat[0], obj_qx=quat[1], obj_qy=quat[2], obj_qz=quat[3],
                   ee_object_gap=gap, physical_contact=int(max(forces, default=0.0) > 1e-8),
                   measured_contact_force=max(forces, default=0.0), measured_contact_pair_count=len(forces),
                   contact_force_norm=max(forces, default=0.0),
                   controller_contact_force_norm=controller_force,
                   controller_contact_force_threshold=contact_threshold,
                   controller_physical_contact=int(controller_force > contact_threshold),
                   contact_acquisition_state=str(getattr(acquisition, 'phase', 'legacy')),
                   contact_acquisition_active=int(active), c3_contact_acquired=int(acquired),
                   c3_entry_time=acquisition_scalar('entry_time'),
                   contact_acquired_time=acquisition_scalar('acquired_time'),
                   contact_acquisition_deadline=acquisition_scalar('timeout_time'),
                   acquisition_timeout_time=acquisition_scalar('failure_time'),
                   acquisition_failure=int(bool(getattr(acquisition, 'acquisition_failure', False))),
                   progress_evaluation_active=int(mode == 'c3' and (acquired or acquisition is None)),
                   normal_progress_counter=self.progress._n_updates,
                   c3_entry_object_x=entry_pose[0][0] if entry_pose is not None else math.nan,
                   c3_entry_object_y=entry_pose[0][1] if entry_pose is not None else math.nan,
                   c3_entry_object_z=entry_pose[0][2] if entry_pose is not None else math.nan,
                   c3_entry_object_yaw=entry_pose[1] if entry_pose is not None else math.nan,
                   c3_entry_object_qw=entry_quat[0], c3_entry_object_qx=entry_quat[1],
                   c3_entry_object_qy=entry_quat[2], c3_entry_object_qz=entry_quat[3],
                   c3_entry_ee_x=entry_ee_pose[0] if entry_ee_pose is not None else math.nan,
                   c3_entry_ee_y=entry_ee_pose[1] if entry_ee_pose is not None else math.nan,
                   c3_entry_ee_z=entry_ee_pose[2] if entry_ee_pose is not None else math.nan,
                   attempted_contact_target_x=acquisition_target[0],
                   attempted_contact_target_y=acquisition_target[1],
                   attempted_contact_target_z=acquisition_target[2],
                   attempted_contact_target_body_x=acquisition_body[0],
                   attempted_contact_target_body_y=acquisition_body[1],
                   reposition_target_x=target[0], reposition_target_y=target[1], reposition_target_z=target[2],
                   distance_to_reposition_target=float(np.linalg.norm(ee-target)),
                   target_body_x=body[0], target_body_y=body[1],
                   progress_counter=decision[0], steps_since_improve=decision[1], progress_metric=decision[2],
                   progress_window_front=decision[3], tracker_met_progress=int(decision[4]),
                   dispatcher_met_progress=int(plan['met']), progress_counter_after_control=self.progress._n_updates,
                   object_displacement_since_c3_entry=displacement, object_yaw_change_since_c3_entry=dyaw,
                   selected_candidate_id=f'{self._step}:{index}:{label}', selected_candidate_x=selected[0],
                   selected_candidate_y=selected[1], selected_candidate_z=selected[2],
                   candidate_object_relative_angle=math.atan2(body[1],body[0]),
                   candidate_object_relative_distance=float(np.linalg.norm(body)),
                   failed_buffer_size=len(self.unsuccessful_buffer), sample_buffer_size=len(self.buffer),
                   failed_target_memory_size=failure_memory_size,
                   failed_buffer_checks_cumulative=getattr(self.unsuccessful_buffer, '_audit_checks', 0),
                   candidate_rejected_by_failed_buffer_cumulative=getattr(self.unsuccessful_buffer, '_audit_rejections', 0),
                   finished_repos_decision=int(plan['finished_repos']),
                   current_cost=plan['c_samples'][0], best_other_cost=plan['best_other_cost'],
                   filtered_solve_time=float(getattr(self.base_mpc, '_filtered_solve_time', math.nan)))
        if writer is None:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
        writer.writerow(row)
        for rejected in getattr(self.unsuccessful_buffer, '_audit_pending_rejections', []):
            rejected.update(row=row_count, sim_time=sim_time, planner_step=self._step,
                            candidate_id=f'{self._step}:rejected_probe:{rejected["check_id"]}')
            rejection_stream.write(json.dumps(rejected) + '\n')
        self.unsuccessful_buffer._audit_pending_rejections = []
        row_count += 1
        if kind == 'planner':
            stream.flush()
            rejection_stream.flush()

    original_control = SamplingC3Controller.compute_control
    def observed_control(self, current_q, current_v, plant_ctx, *a, **kw):
        prior_mode = self._prev_mode
        target = self._current_repos_target
        prior_target = target.copy() if target is not None else None
        result = original_control(self, current_q, current_v, plant_ctx, *a, **kw)
        observe(self, current_q, plant_ctx, 'planner', prior_mode, prior_target)
        return result
    SamplingC3Controller.compute_control = observed_control
    original_osc = SamplingC3Controller.compute_control_osc_only
    def observed_osc(self, current_q, current_v, plant_ctx, *a, **kw):
        result = original_osc(self, current_q, current_v, plant_ctx, *a, **kw)
        observe(self, current_q, plant_ctx, 'osc')
        return result
    SamplingC3Controller.compute_control_osc_only = observed_osc
    sys.argv = argv
    started = time.time()
    try:
        import main as baseline_main
        baseline_main.main()
        manifest['completed'] = True
    except BaseException as exc:
        manifest['completed'] = False
        manifest['error'] = repr(exc)
        raise
    finally:
        stream.close()
        rejection_stream.close()
        manifest['wall_seconds'] = time.time() - started
        manifest['trajectory_rows'] = row_count
        manifest['source_changes'] = [str(p) for p in source_files
                                      if hashlib.sha256(p.read_bytes()).hexdigest() != manifest['sha256_before'][str(p)]]
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        log_path = ROOT / 'results' / (run_name + '.txt')
        if log_path.exists():
            shutil.copy2(log_path, out / 'main_log.txt')
        (out / 'git_status_after.txt').write_text(subprocess.check_output(['git', 'status', '--short'], text=True))


if __name__ == '__main__':
    main()
