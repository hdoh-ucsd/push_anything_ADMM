"""Unit tests for Stage 2 L2 commit-face gate.

Covers the 45-degree cone behavior at the four cardinal cases the
sweep needs to handle, the boundary at exactly the threshold, the
severity tagging for refusal logging, and a cross-check that the
gate (which mutates ``finished_repos`` upstream of decide_mode) does
NOT affect the ``kToC3Cost`` cost-based engagement path.

Push-west convention used throughout: ``g_hat = (-1, 0)`` (goal lies
to the west of the box). For this g_hat:

  EE on +x face (anti-goal side of box)  ->  face_out = (+1, 0)
      face_align = (+1)(-1) + 0 = -1.0           COMMIT
  EE on -x face (goal-side of box)       ->  face_out = (-1, 0)
      face_align = (-1)(-1) + 0 = +1.0           REFUSE (anti-goal)
  EE on +/- y faces                      ->  face_out = (0, +/-1)
      face_align = 0.0                          REFUSE (perpendicular)
"""
import numpy as np
import pytest

from control.sampling_c3.commit_face_gate import (
    CommitFaceDecision,
    decide_commit_face_gate,
    NEAR_MISS_UPPER,
    PERPENDICULAR_UPPER,
)
from control.sampling_c3.mode_switch import (
    SwitchReason,
    decide_mode,
)
from control.sampling_c3.params import ProgressParams


BOX_HALF = 0.05
SETBACK = 0.030
RADIUS = BOX_HALF + SETBACK  # 0.080 — distance from box center to target


def _make_target(box_xy, face_unit_xy, radius=RADIUS):
    """Place a reposition target at ``radius`` along ``face_unit_xy``
    from the box center."""
    return np.array([
        box_xy[0] + radius * face_unit_xy[0],
        box_xy[1] + radius * face_unit_xy[1],
    ])


# ---------------------------------------------------------------------------
# 1. +x face -> face_align = -1  (push-west COMMITS)
# ---------------------------------------------------------------------------

def test_commits_on_plus_x_face_push_west():
    box_xy = np.array([0.5, 0.0])
    p_tgt = _make_target(box_xy, (+1.0, 0.0))
    g_hat = np.array([-1.0, 0.0])

    d = decide_commit_face_gate(p_tgt, box_xy, g_hat, threshold=-0.7)

    assert d.commit is True
    assert d.face_align == pytest.approx(-1.0, abs=1e-12)
    assert d.severity_tag == "commit"
    assert d.n_face_out_xy[0] == pytest.approx(+1.0, abs=1e-12)
    assert d.n_face_out_xy[1] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 2. Near-+x within cone (dot = -0.8) COMMITS
# ---------------------------------------------------------------------------

def test_commits_inside_cone_at_minus_08():
    """Face direction tilted off +x by enough to give face_align = -0.8."""
    box_xy = np.array([0.0, 0.0])
    # Pick a face direction with x-component cos(theta) such that
    # face_align = (-1) * cos(theta) = -0.8  =>  cos(theta) = 0.8.
    cx, cy = 0.8, np.sqrt(1.0 - 0.64)  # (0.8, 0.6)
    p_tgt = _make_target(box_xy, (cx, cy))
    g_hat = np.array([-1.0, 0.0])

    d = decide_commit_face_gate(p_tgt, box_xy, g_hat, threshold=-0.7)

    assert d.commit is True
    assert d.face_align == pytest.approx(-0.8, abs=1e-9)
    assert d.severity_tag == "commit"


# ---------------------------------------------------------------------------
# 3. Perpendicular +/- y (dot = 0) REFUSE
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("face_unit", [(0.0, +1.0), (0.0, -1.0)])
def test_refuses_perpendicular_y_faces(face_unit):
    box_xy = np.array([0.5, 0.2])
    p_tgt = _make_target(box_xy, face_unit)
    g_hat = np.array([-1.0, 0.0])

    d = decide_commit_face_gate(p_tgt, box_xy, g_hat, threshold=-0.7)

    assert d.commit is False
    assert d.face_align == pytest.approx(0.0, abs=1e-12)
    assert d.severity_tag == "perpendicular"


# ---------------------------------------------------------------------------
# 4. Anti-goal -x face (dot = +1) REFUSES
# ---------------------------------------------------------------------------

def test_refuses_anti_goal_minus_x_face():
    box_xy = np.array([0.0, 0.0])
    p_tgt = _make_target(box_xy, (-1.0, 0.0))
    g_hat = np.array([-1.0, 0.0])

    d = decide_commit_face_gate(p_tgt, box_xy, g_hat, threshold=-0.7)

    assert d.commit is False
    assert d.face_align == pytest.approx(+1.0, abs=1e-12)
    assert d.severity_tag == "anti-goal"


# ---------------------------------------------------------------------------
# 5. Boundary at dot = -0.7 — INCLUSIVE on commit side
#    (`face_align <= threshold` ⇒ COMMIT at exact equality)
# ---------------------------------------------------------------------------

def test_boundary_at_exact_threshold_commits():
    """At the exact threshold the inequality is ``<=``, so the gate
    COMMITS. Mirrors L1 convention (wrapper.py:928, ``<=``) modulo the
    inverted sign semantics."""
    box_xy = np.array([0.0, 0.0])
    cx, cy = 0.7, np.sqrt(1.0 - 0.49)  # (0.7, ~0.714) — face_out · g_hat = -0.7
    p_tgt = _make_target(box_xy, (cx, cy))
    g_hat = np.array([-1.0, 0.0])

    d = decide_commit_face_gate(p_tgt, box_xy, g_hat, threshold=-0.7)

    assert d.face_align == pytest.approx(-0.7, abs=1e-9)
    assert d.commit is True, (
        f"boundary face_align={d.face_align} should COMMIT under <=; "
        "if you want strict-less-than, change the function and this test."
    )
    assert d.severity_tag == "commit"


def test_just_outside_boundary_refuses_with_near_miss_tag():
    """face_align = -0.69 sits just OUTSIDE the cone; expected refuse
    with tag 'near-miss-cone'."""
    box_xy = np.array([0.0, 0.0])
    cx = 0.69
    cy = float(np.sqrt(1.0 - cx ** 2))
    p_tgt = _make_target(box_xy, (cx, cy))
    g_hat = np.array([-1.0, 0.0])

    d = decide_commit_face_gate(p_tgt, box_xy, g_hat, threshold=-0.7)

    assert d.face_align == pytest.approx(-0.69, abs=1e-9)
    assert d.commit is False
    assert d.severity_tag == "near-miss-cone"


# ---------------------------------------------------------------------------
# 5b. Boundary at dot = +0.3 — the NEW production default
#     (plan 2026-06-10, set per direct read of q1_noregress logs)
# ---------------------------------------------------------------------------

def test_boundary_at_plus_threshold_commits_production_default():
    """Pin the production-default threshold +0.3 (params.py:624).

    Picks a face direction that gives face_align = +0.3 under push-west
    (g_hat = -x): cx = -0.3, cy = sqrt(1 - 0.09). At the exact boundary
    the inequality is ``<=`` so the gate COMMITS. Tag is 'commit'
    because the commit-true branch fires first in the tag selection."""
    box_xy = np.array([0.0, 0.0])
    cx = -0.3
    cy = float(np.sqrt(1.0 - cx ** 2))
    p_tgt = _make_target(box_xy, (cx, cy))
    g_hat = np.array([-1.0, 0.0])

    d = decide_commit_face_gate(p_tgt, box_xy, g_hat, threshold=+0.3)

    assert d.face_align == pytest.approx(+0.3, abs=1e-9)
    assert d.commit is True, (
        f"boundary face_align={d.face_align} should COMMIT under <=; "
        "this pins the production default +0.3"
    )
    assert d.severity_tag == "commit"


def test_just_outside_plus_threshold_refuses_with_anti_goal_tag():
    """At threshold +0.3, face_align = +0.31 sits just OUTSIDE — refuse."""
    box_xy = np.array([0.0, 0.0])
    cx = -0.31
    cy = float(np.sqrt(1.0 - cx ** 2))
    p_tgt = _make_target(box_xy, (cx, cy))
    g_hat = np.array([-1.0, 0.0])

    d = decide_commit_face_gate(p_tgt, box_xy, g_hat, threshold=+0.3)

    assert d.face_align == pytest.approx(+0.31, abs=1e-9)
    assert d.commit is False
    # face_align +0.31 > PERPENDICULAR_UPPER=+0.3 -> 'anti-goal' band.
    assert d.severity_tag == "anti-goal"


def test_production_threshold_refuses_known_runaway_face_aligns():
    """The two empirical runaway entries from q1_noregress (seed-0 step
    434, face_align=+0.988; seed-4 step 519, face_align=+0.975) must be
    REFUSED under threshold +0.3."""
    box_xy = np.array([0.0, 0.0])
    g_hat = np.array([-1.0, 0.0])
    for fa in (0.988, 0.975):
        cx = float(-fa)
        cy = float(np.sqrt(max(0.0, 1.0 - cx ** 2)))
        p_tgt = _make_target(box_xy, (cx, cy))
        d = decide_commit_face_gate(p_tgt, box_xy, g_hat, threshold=+0.3)
        assert d.face_align == pytest.approx(fa, abs=1e-9), fa
        assert d.commit is False, f"runaway fa={fa} must refuse"
        assert d.severity_tag == "anti-goal"


def test_production_threshold_admits_known_productive_face_align():
    """The 114-tick productive session (seed-4 step 315, face_align=-0.497,
    +136 mm westward push) must be ADMITTED under threshold +0.3 with
    enormous margin (0.797 below threshold)."""
    box_xy = np.array([0.0, 0.0])
    g_hat = np.array([-1.0, 0.0])
    fa = -0.497
    cx = float(-fa)
    cy = float(np.sqrt(1.0 - cx ** 2))
    p_tgt = _make_target(box_xy, (cx, cy))
    d = decide_commit_face_gate(p_tgt, box_xy, g_hat, threshold=+0.3)
    assert d.face_align == pytest.approx(fa, abs=1e-9)
    assert d.commit is True, (
        "the q1_noregress seed-4 step-315 productive 114-tick session "
        "(face_align=-0.497) must NOT be blocked by the +0.3 threshold "
        "— blocking it would kill the only goal-reaching c3 session "
        "in the run"
    )


# ---------------------------------------------------------------------------
# 5c. Boundary at dot = 0.0 — Q7-CP2 tightened production default
#     The +0.3 default was tightened to 0.0 after the Q7 sweep showed
#     seed-2 drift-causing admitted entries at face_align +0.190 / +0.269
#     (both in the band 0.0 admits/refuses cleanly).
# ---------------------------------------------------------------------------

def test_boundary_at_zero_threshold_commits():
    """At threshold=0.0 the exact-boundary face_align=0.0 case COMMITs
    (the inequality is ``<=``). Tag is 'commit' (commit branch wins
    over the perpendicular tag)."""
    box_xy = np.array([0.0, 0.0])
    # face_align = 0.0 means n_face_out is perpendicular to g_hat.
    # Pick n_face_out = (0, +1), g_hat = (-1, 0) -> face_align = 0.
    p_tgt = _make_target(box_xy, (0.0, +1.0))
    g_hat = np.array([-1.0, 0.0])

    d = decide_commit_face_gate(p_tgt, box_xy, g_hat, threshold=0.0)

    assert d.face_align == pytest.approx(0.0, abs=1e-12)
    assert d.commit is True
    assert d.severity_tag == "commit"


def test_just_outside_zero_threshold_refuses():
    """face_align=+0.01 is just outside the 0.0 threshold -> refuse."""
    box_xy = np.array([0.0, 0.0])
    cx = -0.01
    cy = float(np.sqrt(1.0 - cx ** 2))
    p_tgt = _make_target(box_xy, (cx, cy))
    g_hat = np.array([-1.0, 0.0])

    d = decide_commit_face_gate(p_tgt, box_xy, g_hat, threshold=0.0)

    assert d.face_align == pytest.approx(+0.01, abs=1e-9)
    assert d.commit is False
    # +0.01 is in the perpendicular band (-0.3, +0.3].
    assert d.severity_tag == "perpendicular"


def test_zero_threshold_refuses_seed2_drift_entries():
    """The seed-2 drift-causing admitted entries (Q7 sweep, commit 424a6ad,
    HEAD a1677bb at sweep time) at face_align=+0.190 (step 67) and +0.269
    (step 554) must be REFUSED under threshold 0.0."""
    box_xy = np.array([0.0, 0.0])
    g_hat = np.array([-1.0, 0.0])
    for fa in (0.190, 0.269):
        cx = float(-fa)
        cy = float(np.sqrt(1.0 - cx ** 2))
        p_tgt = _make_target(box_xy, (cx, cy))
        d = decide_commit_face_gate(p_tgt, box_xy, g_hat, threshold=0.0)
        assert d.face_align == pytest.approx(fa, abs=1e-9), fa
        assert d.commit is False, f"seed-2 drift entry fa={fa} must refuse"


def test_zero_threshold_admits_productive_minus_0p497():
    """Threshold 0.0 must STILL admit the q1_noregress seed-4 step-315
    productive 114-tick session at face_align=-0.497 (margin 0.497).
    This is the noregress pin for the tightened threshold."""
    box_xy = np.array([0.0, 0.0])
    g_hat = np.array([-1.0, 0.0])
    fa = -0.497
    cx = float(-fa)
    cy = float(np.sqrt(1.0 - cx ** 2))
    p_tgt = _make_target(box_xy, (cx, cy))
    d = decide_commit_face_gate(p_tgt, box_xy, g_hat, threshold=0.0)
    assert d.face_align == pytest.approx(fa, abs=1e-9)
    assert d.commit is True, (
        "the productive -0.497 session must SURVIVE the tightened 0.0 "
        "threshold (margin 0.497)"
    )


# ---------------------------------------------------------------------------
# 6. Severity tags across the full face_align range
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("face_align_target, expected_tag", [
    (-0.9, "commit"),
    (-0.7, "commit"),               # boundary -> commit
    (-0.5, "near-miss-cone"),
    (NEAR_MISS_UPPER, "near-miss-cone"),     # -0.3 boundary -> near-miss
    (-0.1, "perpendicular"),
    (0.0, "perpendicular"),
    (+0.2, "perpendicular"),
    (PERPENDICULAR_UPPER, "perpendicular"),  # +0.3 boundary -> perpendicular
    (+0.5, "anti-goal"),
    (+1.0, "anti-goal"),
])
def test_severity_tags(face_align_target, expected_tag):
    """Sweep face_align to confirm each tag boundary. g_hat = (-1,0)
    so face_align = -cx implies cx = -face_align (clamp into [-1,1])."""
    box_xy = np.array([0.0, 0.0])
    cx = float(-face_align_target)
    cy = float(np.sqrt(max(0.0, 1.0 - cx ** 2)))
    p_tgt = _make_target(box_xy, (cx, cy))
    g_hat = np.array([-1.0, 0.0])

    d = decide_commit_face_gate(p_tgt, box_xy, g_hat, threshold=-0.7)

    assert d.severity_tag == expected_tag, (
        f"face_align={d.face_align:.3f} -> tag={d.severity_tag!r} "
        f"expected {expected_tag!r}"
    )


# ---------------------------------------------------------------------------
# 7. Push-east / push-north sanity (no-axis-bias)
# ---------------------------------------------------------------------------

def test_commits_for_push_east_on_minus_x_face():
    """Push-east: g_hat = (+1, 0). Goal-aligned EE push needs EE on
    the -x face -> face_out = (-1, 0); face_align = -1 -> COMMIT."""
    box_xy = np.array([0.4, -0.1])
    p_tgt = _make_target(box_xy, (-1.0, 0.0))
    g_hat = np.array([+1.0, 0.0])

    d = decide_commit_face_gate(p_tgt, box_xy, g_hat, threshold=-0.7)

    assert d.commit is True
    assert d.face_align == pytest.approx(-1.0, abs=1e-12)


def test_refuses_anti_goal_for_push_north():
    """Push-north: g_hat = (0, +1). The goal-side face is +y
    (face_out=(0,+1), face_align=+1) — EE placed there would push
    the box anti-goal-ward, so the gate must REFUSE."""
    box_xy = np.array([0.0, 0.0])
    p_tgt = _make_target(box_xy, (0.0, +1.0))   # face_out = (0,+1)
    g_hat = np.array([0.0, +1.0])

    d = decide_commit_face_gate(p_tgt, box_xy, g_hat, threshold=-0.7)

    assert d.commit is False
    assert d.face_align == pytest.approx(+1.0, abs=1e-12)
    assert d.severity_tag == "anti-goal"


def test_commits_for_push_north_on_minus_y_face():
    """Push-north symmetry check: EE on -y face (anti-goal side)
    pushes the box +y -> COMMIT."""
    box_xy = np.array([0.0, 0.0])
    p_tgt = _make_target(box_xy, (0.0, -1.0))   # face_out = (0,-1)
    g_hat = np.array([0.0, +1.0])

    d = decide_commit_face_gate(p_tgt, box_xy, g_hat, threshold=-0.7)

    assert d.commit is True
    assert d.face_align == pytest.approx(-1.0, abs=1e-12)
    assert d.severity_tag == "commit"


# ---------------------------------------------------------------------------
# 8. Degenerate input — target on box center
# ---------------------------------------------------------------------------

def test_degenerate_target_on_box_center_refuses():
    box_xy = np.array([0.5, 0.5])
    p_tgt = np.array([0.5, 0.5])    # coincident
    g_hat = np.array([-1.0, 0.0])

    d = decide_commit_face_gate(p_tgt, box_xy, g_hat, threshold=-0.7)

    assert d.commit is False
    assert d.face_align == 0.0
    assert d.severity_tag == "perpendicular"
    assert d.n_face_out_xy == (0.0, 0.0)


# ---------------------------------------------------------------------------
# 9. Input shape tolerance — 3D inputs (only XY used)
# ---------------------------------------------------------------------------

def test_accepts_3d_inputs_uses_xy_only():
    """The wrapper passes 3D EE targets; the helper must read only XY
    and ignore z. A target with face_out=(+1,0) at any z still
    COMMITs for push-west."""
    box_xz = np.array([0.0, 0.0, 0.05])
    p_tgt = np.array([0.08, 0.0, 0.5])   # arbitrary z, +x in XY
    g_hat = np.array([-1.0, 0.0, 99.0])  # z component must be ignored

    d = decide_commit_face_gate(p_tgt, box_xz, g_hat, threshold=-0.7)

    assert d.commit is True
    assert d.face_align == pytest.approx(-1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 10. Custom threshold — strict cone (threshold = -0.95) shrinks admit set
# ---------------------------------------------------------------------------

def test_strict_cone_threshold_refuses_dot_minus_08():
    """At threshold=-0.95, face_align=-0.8 is OUTSIDE the cone."""
    box_xy = np.array([0.0, 0.0])
    cx, cy = 0.8, 0.6
    p_tgt = _make_target(box_xy, (cx, cy))
    g_hat = np.array([-1.0, 0.0])

    d = decide_commit_face_gate(p_tgt, box_xy, g_hat, threshold=-0.95)

    assert d.face_align == pytest.approx(-0.8, abs=1e-9)
    assert d.commit is False
    assert d.severity_tag == "near-miss-cone"   # raw face_align unchanged


# ---------------------------------------------------------------------------
# 11. Decoupling check — gate refusal must NOT block the kToC3Cost path
# ---------------------------------------------------------------------------

def _progress_params():
    """Helper: build a ProgressParams with a known repos->c3 hysteresis."""
    p = ProgressParams()
    p.hyst_repos_to_c3 = 500.0
    return p


def test_kToC3Cost_unaffected_by_gate_refusal():
    """``decide_mode`` is independent of the gate by design — the cost-gap
    path (mode_switch.py:132-135) consults c3_cost vs best_other_cost,
    not finished_repos. This test pins that contract.

    Note: at the WRAPPER level (plan 2026-06-10), ``kToC3Cost`` IS gated
    on the same commit-face discriminator via a post-decide override
    site in wrapper.py — when the override refuses, mode is overridden
    to ``"free"`` with ``reason = kStayInRepos``. See
    tests/test_commit_face_gate_post_decide.py for the wrapper-level
    contract. The decide_mode contract this test pins remains correct
    and unchanged."""
    p = _progress_params()

    mode, reason = decide_mode(
        prev_mode="free",
        c3_cost=5000.0,
        best_other_cost=8000.0,    # repos cost 8000; c3 + 500 = 5500 < 8000
        current_repos_cost=8000.0,
        met_progress=True,
        near_goal=False,
        finished_repos=False,      # <-- gate just refused
        params=p,
    )

    assert mode == "c3"
    assert reason == SwitchReason.kToC3Cost


def test_kToC3ReachedReposTarget_blocked_when_gate_refuses():
    """Inverse of the above: with finished_repos forced False by the
    gate AND no cost advantage for c3, free is preserved."""
    p = _progress_params()

    mode, reason = decide_mode(
        prev_mode="free",
        c3_cost=10000.0,
        best_other_cost=8000.0,    # c3 + hyst = 10500 > 8000 -> no kToC3Cost
        current_repos_cost=8000.0,
        met_progress=True,
        near_goal=False,
        finished_repos=False,
        params=p,
    )

    assert mode != "c3"
    assert reason != SwitchReason.kToC3ReachedReposTarget
