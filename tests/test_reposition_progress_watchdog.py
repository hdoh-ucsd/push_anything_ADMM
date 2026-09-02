import numpy as np

from control.sampling_c3.sampling_based_c3_controller import (
    RepositionProgressWatchdog,
    preserve_reposition_arrival,
)


def _watchdog():
    return RepositionProgressWatchdog(
        window_s=0.3, min_progress=0.002,
        arrival_tolerance=0.025, dt=0.1,
    )


def test_near_plateau_is_accepted_as_arrived():
    watchdog = _watchdog()
    target = np.array([0.4, 0.2, 0.03])
    assert watchdog.update(target, 0.0210) is None
    assert watchdog.update(target, 0.0208) is None
    assert watchdog.update(target, 0.0207) is None
    assert watchdog.update(target, 0.0206) == "arrived"


def test_far_plateau_forces_resample():
    watchdog = _watchdog()
    target = np.array([0.4, 0.2, 0.03])
    outcomes = [watchdog.update(target, d)
                for d in (0.12, 0.1198, 0.1197, 0.1196)]
    assert outcomes[-1] == "resample"


def test_real_progress_and_target_changes_reset_window():
    watchdog = _watchdog()
    first = np.array([0.4, 0.2, 0.03])
    second = np.array([0.5, 0.2, 0.03])
    for distance in (0.12, 0.117, 0.114, 0.111):
        assert watchdog.update(first, distance) is None
    assert watchdog.update(second, 0.0210) is None
    assert watchdog.update(second, 0.0209) is None


def test_watchdog_arrival_survives_next_pwl_evaluation():
    """The execution-tick latch must reach the next mode decision."""
    assert preserve_reposition_arrival(
        latched=True, build_flag=False, euclidean_arrival=False)


def test_pwl_arrival_still_works_without_watchdog_latch():
    assert preserve_reposition_arrival(
        latched=False, build_flag=True, euclidean_arrival=False)
    assert preserve_reposition_arrival(
        latched=False, build_flag=False, euclidean_arrival=True)

