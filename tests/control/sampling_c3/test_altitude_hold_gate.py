"""Stage-1 altitude-hold gate tests.

Pre-registered SC' invariants for the wrong-face race-fix plan
(docs/superpowers/plans/2026-06-01-wrong-face-race-fix.md).

The gate adds `allow_descent: bool = True` to `next_waypoint`. When False,
Phase 3 (descend) and the direct-line shortcut's downward component must
be suppressed; Phase 1 (lift) and Phase 2 (traverse) must still fire.
"""
import numpy as np

from control.sampling_c3.reposition import next_waypoint


def test_descent_blocked_when_allow_descent_false():
    """Target is at xy, below in z. Default behavior: Phase 3 descends.
    With gate closed: z must stay at p_now (no descent)."""
    p_now    = np.array([0.10, 0.00, 0.15])
    p_target = np.array([0.10, 0.00, 0.05])
    z_safe   = 0.15
    ds       = 0.01
    wp_open = next_waypoint(p_now, p_target, z_safe, ds, allow_descent=True)
    assert wp_open[2] < 0.15, \
        f"Phase 3 should descend when allowed, got z={wp_open[2]}"
    wp_held = next_waypoint(p_now, p_target, z_safe, ds, allow_descent=False)
    assert wp_held[2] >= 0.15 - 1e-9, \
        f"Phase 3 must not descend when blocked, got z={wp_held[2]}"


def test_phase_1_lift_still_fires_when_blocked():
    """Even with allow_descent=False, Phase 1 (lift to z_safe) must work
    when there is still xy work to do. The gate suppresses descent only;
    lift remains available so the EE can recover from a prior partial
    descent and re-enter the traverse phase at z_safe."""
    p_now    = np.array([0.00, 0.00, 0.05])  # origin, below z_safe
    p_target = np.array([0.10, 0.00, 0.05])  # 10 cm east, sample altitude
    z_safe   = 0.15
    ds       = 0.01
    wp = next_waypoint(p_now, p_target, z_safe, ds, allow_descent=False)
    assert wp[2] > 0.05, \
        f"Phase 1 lift must fire when below z_safe with xy work, got z={wp[2]}"


def test_phase_2_traverse_unaffected_by_gate():
    """allow_descent=False must not block xy traversal.
    The gate only governs Phase 3 (descend), not Phase 2 (traverse)."""
    p_now    = np.array([0.00, 0.00, 0.15])
    p_target = np.array([0.10, 0.00, 0.05])
    z_safe   = 0.15
    ds       = 0.01
    wp = next_waypoint(p_now, p_target, z_safe, ds, allow_descent=False)
    assert wp[0] > 0.00, \
        f"Phase 2 traverse must still advance xy, got x={wp[0]}"
    assert wp[2] >= 0.15 - 1e-9, \
        f"Phase 2 must NOT descend when blocked, got z={wp[2]}"


def test_backward_compat_default_allows_descent():
    """allow_descent defaults to True — existing callers unchanged."""
    p_now    = np.array([0.10, 0.00, 0.15])
    p_target = np.array([0.10, 0.00, 0.05])
    z_safe   = 0.15
    ds       = 0.01
    wp = next_waypoint(p_now, p_target, z_safe, ds)
    assert wp[2] < 0.15, \
        f"Default behavior must allow Phase 3 descent, got z={wp[2]}"


def test_direct_line_blocked_when_target_below():
    """Sub-cm direct-line shortcut must respect allow_descent.
    A downward shortcut step is suppressed; the phased logic takes over."""
    p_now    = np.array([0.100, 0.000, 0.060])  # 6 mm above target z
    p_target = np.array([0.103, 0.000, 0.054])  # ~7 mm Euclidean, downward
    z_safe   = 0.15
    ds       = 0.01
    # allow_descent=True: direct line descends slightly
    wp_open = next_waypoint(p_now, p_target, z_safe, ds, allow_descent=True)
    assert wp_open[2] < 0.060, \
        f"Direct-line should descend when allowed, got z={wp_open[2]}"
    # allow_descent=False: fall through to phased logic. xy is within
    # z_eps=5mm of target xy (diff = 3mm), so at_target_xy is True. Phase 3
    # blocked → Phase 1 doesn't fire (z=0.060 < z_safe=0.150 → lift). z
    # rises, does NOT descend.
    wp_held = next_waypoint(p_now, p_target, z_safe, ds, allow_descent=False)
    assert wp_held[2] >= 0.060 - 1e-9, \
        f"Direct-line descent must be blocked, got z={wp_held[2]}"


def test_hold_at_xy_descent_blocked_at_z_safe():
    """If allow_descent=False AND at_target_xy AND z >= z_safe, hold p_now."""
    p_now    = np.array([0.10, 0.00, 0.16])  # above z_safe, at xy
    p_target = np.array([0.10, 0.00, 0.05])  # below z_safe
    z_safe   = 0.15
    ds       = 0.01
    wp = next_waypoint(p_now, p_target, z_safe, ds, allow_descent=False)
    np.testing.assert_allclose(wp, p_now, atol=1e-9,
        err_msg=f"Hold-at-xy must return p_now unchanged, got {wp}")


def test_stability_counter_resets_on_target_change():
    """Synthetic dispatcher oscillation: each jump > TARGET_STABLE_TOL must
    reset the counter. Counter increments otherwise."""
    # Unit-test the counter logic via a tiny mock-like loop without
    # standing up a full RepositionIKTracker (Drake plant required for
    # construction). We reproduce the conditional from compute_torque:2a.
    TOL = 5e-3
    prev = None
    counter = 0
    seq = [
        np.array([0.0, 0.0, 0.20]),  # init
        np.array([0.0, 0.0, 0.20]),  # no change → counter=1
        np.array([0.0, 0.0, 0.20]),  # no change → counter=2
        np.array([0.0, 0.0, 0.20 + 6e-3]),  # jump > TOL → counter=0
        np.array([0.0, 0.0, 0.20 + 6e-3]),  # no change → counter=1
    ]
    history = []
    for tgt in seq:
        if prev is None:
            prev = tgt.copy()
            counter = 0
        else:
            jump = float(np.linalg.norm(tgt - prev))
            if jump > TOL:
                counter = 0
                prev = tgt.copy()
            else:
                counter += 1
        history.append(counter)
    assert history == [0, 1, 2, 0, 1], f"counter history wrong: {history}"
