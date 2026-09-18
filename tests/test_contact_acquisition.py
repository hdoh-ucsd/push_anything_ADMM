"""Seed-42 regression checks for the contact-acquisition robustness extension.

These are separate from the unchanged Push Anything progress metric.  They
exercise measured simulator contact, bounded approach time, and finite memory
of failed object-relative targets.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from control.sampling_c3.contact_acquisition import (
    ContactAcquisition,
    measured_contact_force,
    resolve_acquisition_timeout,
)
from control.sampling_c3.sample_buffer import (
    BufferedSample,
    SampleBuffer,
    UnsuccessfulSampleBuffer,
)
from control.sampling_c3.params import SamplingC3Params
from control.sampling_c3.progress import ProgressTracker, StepMetrics
from control.sampling_c3.sampling_based_c3_controller import SamplingC3Controller


@pytest.fixture(autouse=True)
def seed_42():
    state = np.random.get_state()
    np.random.seed(42)
    yield
    np.random.set_state(state)


_OBJECT_POSE = np.array([1.0, 0.0, 0.0, 0.0, 0.50, 0.0, 0.05])
_EE_POSITION = np.array([0.38, 0.002, 0.057])
_TARGET = np.array([0.40, 0.0, 0.050])
_TARGET_BODY = np.array([-0.10, 0.0])
_QUAT = _OBJECT_POSE[:4].copy()


def _episode(*, force_norm=0.0, entry_time=2.475):
    acquisition = ContactAcquisition(timeout_s=0.525, force_threshold=1e-6)
    acquisition.begin(
        sim_time=entry_time,
        object_pose=_OBJECT_POSE.copy(),
        ee_position=_EE_POSITION.copy(),
        target_world=_TARGET.copy(),
        target_body_xy=_TARGET_BODY.copy(),
        force_norm=force_norm,
    )
    return acquisition


def test_entry_snapshots_the_attempted_target_without_claiming_contact():
    object_pose = _OBJECT_POSE.copy()
    ee = _EE_POSITION.copy()
    target = _TARGET.copy()
    target_body = _TARGET_BODY.copy()
    acquisition = ContactAcquisition(timeout_s=0.525, force_threshold=1e-6)
    acquisition.begin(
        sim_time=2.475, object_pose=object_pose, ee_position=ee,
        target_world=target, target_body_xy=target_body, force_norm=0.0,
    )
    object_pose[:] = 0.0
    ee[:] = 0.0
    target[:] = 0.0
    target_body[:] = 0.0

    assert acquisition.active
    assert acquisition.phase == "ACQUIRING_CONTACT"
    assert not acquisition.contact_acquired
    assert not acquisition.acquisition_failure
    assert acquisition.entry_time == pytest.approx(2.475)
    assert acquisition.timeout_time == pytest.approx(3.0)
    assert acquisition.acquired_time is None
    np.testing.assert_array_equal(acquisition.entry_object_pose, _OBJECT_POSE)
    np.testing.assert_array_equal(acquisition.entry_ee_position, _EE_POSITION)
    np.testing.assert_array_equal(acquisition.target_world, _TARGET)
    np.testing.assert_array_equal(acquisition.target_body_xy, _TARGET_BODY)
    assert not np.array_equal(acquisition.target_world,
                              acquisition.entry_ee_position)


def test_physical_force_threshold_and_first_contact_are_latched_once():
    acquisition = _episode()
    for time, force in ((2.50, 0.0), (2.55, 0.5e-6), (2.575, 1e-6)):
        assert not acquisition.observe(time, force)
        assert not acquisition.contact_acquired

    assert acquisition.observe(2.580, 0.02)
    assert acquisition.contact_acquired
    assert acquisition.phase == "TRACKING_PROGRESS"
    assert acquisition.acquired_time == pytest.approx(2.580)
    assert not acquisition.observe(2.600, 3.0)
    assert not acquisition.observe(2.700, 0.0)
    assert acquisition.acquired_time == pytest.approx(2.580)
    assert not acquisition.expired(30.0)


def test_real_contact_already_present_at_entry_starts_acquired():
    acquisition = _episode(force_norm=0.01)
    assert acquisition.contact_acquired
    assert acquisition.acquired_time == acquisition.entry_time
    assert not acquisition.expired(acquisition.timeout_time + 10.0)


def test_no_contact_has_a_finite_separate_deadline():
    acquisition = _episode()
    # A contactless approach cannot remain active for the 2.7 s progress
    # window, nor indefinitely after progress accumulation is paused.
    for sim_time in np.linspace(2.475, 2.999999, 42):
        assert not acquisition.observe(float(sim_time), 0.0)
        assert not acquisition.expired(float(sim_time))
    assert acquisition.expired(3.0)
    acquisition.mark_failed(3.0)
    assert acquisition.acquisition_failure
    assert acquisition.phase == "ACQUISITION_TIMEOUT"
    assert acquisition.failure_time == pytest.approx(3.0)
    assert not acquisition.contact_acquired


def test_contact_at_deadline_wins_but_late_contact_does_not():
    at_deadline = _episode()
    assert at_deadline.observe(at_deadline.timeout_time, 0.02)
    assert not at_deadline.expired(at_deadline.timeout_time)

    late = _episode()
    assert not late.observe(late.timeout_time + 1e-6, 0.02)
    assert not late.contact_acquired
    assert late.expired(late.timeout_time + 1e-6)


def test_end_preserves_episode_evidence_and_reentry_resets_it():
    acquisition = _episode()
    acquisition.mark_failed(3.0)
    acquisition.end()
    assert not acquisition.active
    assert acquisition.phase == "INACTIVE"
    assert not acquisition.expired(100.0)
    assert acquisition.acquisition_failure
    assert acquisition.failure_time == pytest.approx(3.0)
    np.testing.assert_array_equal(acquisition.target_world, _TARGET)

    acquisition.begin(
        sim_time=4.0, object_pose=_OBJECT_POSE, ee_position=_EE_POSITION,
        target_world=_TARGET, target_body_xy=_TARGET_BODY, force_norm=0.02,
    )
    assert acquisition.active
    assert acquisition.contact_acquired
    assert not acquisition.acquisition_failure
    assert acquisition.failure_time is None
    assert acquisition.entry_time == pytest.approx(4.0)
    assert acquisition.acquired_time == pytest.approx(4.0)


def test_timeout_comes_from_existing_reposition_timing():
    # Canonical box runtime: main.py resolves the safe PWL height to 0.135 m.
    timeout = resolve_acquisition_timeout(
        configured=None, waypoint_height=0.135, target_height=0.05,
        speed=0.18, dt=0.075,
    )
    assert timeout == pytest.approx(0.525)
    assert timeout >= (0.135 - 0.05) / 0.18
    assert timeout - 0.075 < (0.135 - 0.05) / 0.18


def test_timeout_override_and_minimum_planner_interval():
    assert resolve_acquisition_timeout(
        configured=0.4, waypoint_height=0.135, target_height=0.05,
        speed=0.18, dt=0.075,
    ) == pytest.approx(0.4)
    assert resolve_acquisition_timeout(
        configured=None, waypoint_height=0.05, target_height=0.05,
        speed=0.18, dt=0.075,
    ) == pytest.approx(0.075)


class _ContactResults:
    def __init__(self, point=(), hydro=()):
        self.point = point
        self.hydro = hydro

    def num_point_pair_contacts(self):
        return len(self.point)

    def point_pair_contact_info(self, index):
        return self.point[index]

    def num_hydroelastic_contacts(self):
        return len(self.hydro)

    def hydroelastic_contact_info(self, index):
        return self.hydro[index]


def _point(a, b, force):
    return SimpleNamespace(
        point_pair=lambda: SimpleNamespace(id_A=a, id_B=b),
        contact_force=lambda: np.asarray(force, dtype=float),
    )


def _hydro(a, b, force):
    return SimpleNamespace(
        contact_surface=lambda: SimpleNamespace(id_M=lambda: a, id_N=lambda: b),
        F_Ac_W=lambda: SimpleNamespace(
            translational=lambda: np.asarray(force, dtype=float)),
    )


def _plant(results):
    return SimpleNamespace(
        get_contact_results_output_port=lambda: SimpleNamespace(
            Eval=lambda context: results),
    )


def test_force_measurement_uses_only_matching_real_contact_pairs():
    results = _ContactResults(
        point=[_point("object", "table", [0, 0, 100]),
               _point("table", "ee", [0, 0, 200]),
               _point("object", "ee", [3, 4, 0])],
        hydro=[_hydro("ee", "object", [0, -12, 0]),
               _hydro("other_robot_link", "object", [0, 0, 300])],
    )
    assert measured_contact_force(
        _plant(results), object(), {"ee"}, {"object"}) == pytest.approx(12.0)


def test_point_contact_alone_and_absent_object_contact():
    point_results = _ContactResults(point=[_point("ee", "object", [0, 3, 4])])
    assert measured_contact_force(
        _plant(point_results), object(), {"ee"}, {"object"}) == pytest.approx(5.0)

    unrelated = _ContactResults(point=[_point("object", "table", [0, 0, 20])])
    assert measured_contact_force(
        _plant(unrelated), object(), {"ee"}, {"object"}) == 0.0
    assert measured_contact_force(
        _plant(_ContactResults()), object(), {"ee"}, {"object"}) == 0.0


def _failure_buffer():
    return UnsuccessfulSampleBuffer(
        capacity=10, unsuccessful_radius=0.01,
        unsuccessful_pos_retention=0.006, unsuccessful_ang_retention=0.05,
    )


def _failed_target(*, target=_TARGET, body_xy=_TARGET_BODY):
    return BufferedSample(
        position=np.asarray(target, dtype=float).copy(), cost=1.0,
        obj_pos_xy=_OBJECT_POSE[4:6].copy(), obj_quat=_QUAT.copy(),
        contact_target_body_xy=np.asarray(body_xy, dtype=float).copy(),
    )


def test_failed_target_radius_is_planar_and_not_the_actual_ee_position():
    buffer = _failure_buffer()
    buffer.append(_failed_target())
    # Reproduce the audited distinction: 9.76 mm target separation remains
    # rejected even with enough vertical offset to evade a world-XYZ ball.
    near_target = _TARGET + np.array([0.00976, 0.0, 0.007])
    assert np.linalg.norm(near_target - _TARGET) > buffer.unsuccessful_radius
    assert not buffer.sample_avoids_bad_spots(
        near_target, _OBJECT_POSE[4:6], _QUAT)
    assert not buffer.sample_avoids_bad_spots(
        _TARGET, _OBJECT_POSE[4:6], _QUAT, contact_failures_only=True)
    assert buffer.sample_avoids_bad_spots(
        _TARGET + [0.01001, 0.0, 0.0], _OBJECT_POSE[4:6], _QUAT)


def test_failure_follows_small_object_pose_change_in_object_coordinates():
    buffer = _failure_buffer()
    buffer.append(_failed_target())
    yaw = 0.04
    quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    obj_xy = _OBJECT_POSE[4:6] + np.array([0.005, 0.0])
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)],
                         [np.sin(yaw), np.cos(yaw)]])
    target = np.r_[obj_xy + rotation @ _TARGET_BODY, _TARGET[2]]
    assert buffer.prune(obj_xy, quat) == 0
    assert not buffer.sample_avoids_bad_spots(target, obj_xy, quat)


@pytest.mark.parametrize("translation,yaw", [(0.00601, 0.0), (0.0, 0.05001)])
def test_failed_contact_memory_expires_after_existing_pose_thresholds(
        translation, yaw):
    buffer = _failure_buffer()
    buffer.append(_failed_target())
    obj_xy = _OBJECT_POSE[4:6] + np.array([translation, 0.0])
    quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    assert buffer.prune(obj_xy, quat) == 1
    assert len(buffer) == 0
    assert buffer.sample_avoids_bad_spots(_TARGET, obj_xy, quat)


def test_failed_contact_memory_retains_existing_fifo_capacity():
    buffer = _failure_buffer()
    entries = []
    for index in range(11):
        entry = _failed_target(body_xy=_TARGET_BODY + [0, index * 0.02])
        entries.append(entry)
        buffer.append(entry)
    assert len(buffer) == 10
    assert list(buffer)[0] is entries[1]
    assert list(buffer)[-1] is entries[-1]


def test_contact_failure_filter_and_expiry_preserve_legacy_memory_contract():
    buffer = _failure_buffer()
    legacy = BufferedSample(
        position=_TARGET.copy(), cost=1.0,
        obj_pos_xy=_OBJECT_POSE[4:6].copy(), obj_quat=_QUAT.copy(),
        position_body_xy=_TARGET_BODY.copy(),
    )
    buffer.append(legacy)
    assert not buffer.sample_avoids_bad_spots(_TARGET, _OBJECT_POSE[4:6], _QUAT)
    assert buffer.sample_avoids_bad_spots(
        _TARGET, _OBJECT_POSE[4:6], _QUAT, contact_failures_only=True)

    buffer.append(_failed_target())
    assert buffer.prune(_OBJECT_POSE[4:6] + [0.01, 0.0], _QUAT) == 1
    assert len(buffer) == 1
    assert list(buffer)[0] is legacy


@pytest.mark.parametrize("avoid_legacy_unsuccessful", [True, False])
def test_next_sample_set_rejects_failed_held_and_fresh_targets(
        monkeypatch, avoid_legacy_unsuccessful):
    """The actual dispatcher sample bank must not bypass target memory."""
    from control.sampling_c3 import sampling_based_c3_controller as controller_module

    controller = SamplingC3Controller.__new__(SamplingC3Controller)
    controller.params = SamplingC3Params()
    controller.params.sampling_params.num_additional_samples_repos = 2
    controller.unsuccessful_buffer = _failure_buffer()
    controller.unsuccessful_buffer.append(_failed_target())
    # Verified acquisition failures are mandatory regardless of whether the
    # caller elects to avoid older, broader unsuccessful-position entries.
    controller._avoid_unsuccessful = avoid_legacy_unsuccessful
    controller._current_repos_target = _TARGET.copy()
    controller._rng = np.random.default_rng(42)
    controller._mesh_faces = None
    controller.log_diag = False
    controller._step = 1
    candidate_bank = [
        _TARGET.copy(),
        _TARGET + [0.00976, 0.0, 0.007],
        _TARGET + [0.04, 0.0, 0.0],
        _TARGET + [0.0, 0.04, 0.0],
    ]
    monkeypatch.setattr(
        controller_module, "generate_samples",
        lambda **kwargs: [sample.copy() for sample in candidate_bank],
    )
    samples, labels = controller._build_samples(
        _EE_POSITION, _OBJECT_POSE[4:6], np.array([1.0, 0.0]), "free",
        obj_quat=_QUAT,
    )
    assert labels == ["current", "strat_0", "strat_1"]
    assert len(samples) == 3
    for sample in samples[1:]:
        assert np.linalg.norm(sample[:2] - _TARGET[:2]) >= 0.01

    # The finite retention policy can reconsider that same held target once
    # the object has moved beyond the unchanged 6 mm threshold.
    moved_xy = _OBJECT_POSE[4:6] + [0.00601, 0.0]
    assert controller.unsuccessful_buffer.prune(moved_xy, _QUAT) == 1
    _, labels = controller._build_samples(
        _EE_POSITION, moved_xy, np.array([1.0, 0.0]), "free", obj_quat=_QUAT,
    )
    assert "prev_repos" in labels


def _controller_with_measured_contact():
    controller = SamplingC3Controller.__new__(SamplingC3Controller)
    controller.params = SamplingC3Params()
    controller.contact_acquisition = ContactAcquisition(timeout_s=0.525)
    controller.progress = ProgressTracker(controller.params.progress_params,
                                          dt_ctrl=0.075)
    controller.unsuccessful_buffer = _failure_buffer()
    controller.buffer = SampleBuffer(capacity=10)
    controller._formulator = SimpleNamespace(
        _ee_geom_ids={"ee"}, _manipuland_geom_ids={"object"})
    contacts = _ContactResults()
    controller.plant = _plant(contacts)
    controller._obj_qw, controller._obj_qx = 0, 1
    controller._obj_qy, controller._obj_qz = 2, 3
    controller._obj_x_idx, controller._obj_y_idx = 4, 5
    controller._obj_z_idx = 6
    controller._prev_mode = "c3"
    controller._current_repos_target = _TARGET.copy()
    controller._physical_contact_force_norm = 0.0
    controller.log_diag = False
    controller._step = 1
    return controller, contacts


def _context(sim_time):
    return SimpleNamespace(get_time=lambda: sim_time)


def test_controller_starts_progress_at_first_real_contact_and_resets_only_once():
    controller, contacts = _controller_with_measured_contact()
    metrics = StepMetrics(c3_cost=10.0, config_cost=20.0, pos_error=0.15)
    controller.progress.update(metrics)
    controller._begin_contact_episode(
        _OBJECT_POSE, _EE_POSITION, _context(2.475), _TARGET)
    assert controller.progress._n_updates == 0
    assert len(controller.unsuccessful_buffer) == 0
    assert not controller._observe_physical_contact(_context(2.550))
    assert controller.progress._n_updates == 0
    assert not controller.contact_acquisition.contact_acquired

    contacts.point = [_point("object", "ee", [0.01, 0.0, 0.0])]
    assert controller._observe_physical_contact(_context(2.580))
    assert controller.progress._n_updates == 0
    assert controller.contact_acquisition.acquired_time == pytest.approx(2.580)
    controller.progress.update(metrics)
    assert not controller._observe_physical_contact(_context(2.600))
    assert controller.progress._n_updates == 1
    assert not controller._prepare_contact_cycle(_OBJECT_POSE, _context(30.0))
    assert len(controller.unsuccessful_buffer) == 0


def test_controller_timeout_records_target_once_and_purges_cached_neighbors():
    controller, _ = _controller_with_measured_contact()
    controller._begin_contact_episode(
        _OBJECT_POSE, _EE_POSITION, _context(2.475), _TARGET)
    near = BufferedSample(position=_TARGET + [0.00976, 0.0, 0.007],
                          cost=1.0, obj_pos_xy=_OBJECT_POSE[4:6].copy())
    far = BufferedSample(position=_TARGET + [0.04, 0.0, 0.0],
                         cost=2.0, obj_pos_xy=_OBJECT_POSE[4:6].copy())
    controller.buffer.append(near)
    controller.buffer.append(far)
    assert not controller._prepare_contact_cycle(_OBJECT_POSE, _context(2.999999))
    assert len(controller.unsuccessful_buffer) == 0
    assert controller._prepare_contact_cycle(_OBJECT_POSE, _context(3.0))
    assert controller.contact_acquisition.acquisition_failure
    assert len(controller.unsuccessful_buffer) == 1
    failed = list(controller.unsuccessful_buffer)[0]
    np.testing.assert_array_equal(failed.position, _TARGET)
    np.testing.assert_allclose(failed.contact_target_body_xy, _TARGET_BODY,
                               atol=1e-16, rtol=0.0)
    assert list(controller.buffer) == [far]

    assert controller._prepare_contact_cycle(_OBJECT_POSE, _context(3.075))
    assert len(controller.unsuccessful_buffer) == 1
    controller.contact_acquisition.end()
    controller._prev_mode = "free"
    moved_pose = _OBJECT_POSE.copy()
    moved_pose[4] += 0.00601
    assert not controller._prepare_contact_cycle(moved_pose, _context(3.150))
    assert len(controller.unsuccessful_buffer) == 0
