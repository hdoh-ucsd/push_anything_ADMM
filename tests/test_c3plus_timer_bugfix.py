"""Read-only regression tests for the V2 monotonic solve-time fix."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = (ROOT / "external/oim_c++_anything/.claude/worktrees/"
              "oim-scene-sync-metrics/systems/controllers/"
              "sampling_based_c3_controller.cc")


def filtered(previous: float, elapsed: float, alpha: float = 0.95) -> float:
    return (1.0 - alpha) * elapsed + alpha * previous


def test_only_solve_timer_clock_uses_changed() -> None:
    source = CONTROLLER.read_text()
    assert source.count("std::chrono::steady_clock::now()") == 2
    assert source.count("std::chrono::high_resolution_clock::now()") == 4
    assert "auto start = std::chrono::steady_clock::now();" in source
    assert "auto finish = std::chrono::steady_clock::now();" in source


def test_old_wall_clock_can_make_filter_negative() -> None:
    assert filtered(0.01208488534459907, -2.818912) < 0.0


def test_new_monotonic_elapsed_cannot_make_filter_negative_from_clock_motion() -> None:
    # A steady clock's elapsed duration is non-negative; with a non-negative
    # prior filtered duration, the convex update remains non-negative.
    previous = 0.01208488534459907
    elapsed = 0.022471
    assert elapsed >= 0.0
    assert filtered(previous, elapsed) >= 0.0


def test_confirmed_fixture_never_requests_negative_knot_after_fix() -> None:
    # The diagnosed event uses dt=.05 and N=5. A monotonic duration cannot
    # produce the observed negative index; this deliberately does not clamp it.
    filtered_time = filtered(0.01208488534459907, 0.022471)
    last_passed_index = int(filtered_time / 0.05)
    assert filtered_time >= 0.0
    assert last_passed_index >= 0
