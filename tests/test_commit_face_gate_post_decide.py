"""Tests for the post-decide commit-face override in wrapper.py.

The override (plan 2026-06-10) sits between ``decide_mode()`` and the
contact-loss disengage gate. When ``prev_mode == "free"`` and
``decide_mode`` returned ``"c3"``, the override re-evaluates the L2
commit-face gate. If the gate refuses, mode flips back to ``"free"``
with ``reason = kStayInRepos`` and a ``[GATE-COMMIT-FACE-POST]`` log
sentinel is emitted.

These tests verify the OVERRIDE LOGIC directly (no Drake plant needed).
The wrapper integration is exercised in the SC sweep
(scripts/_q7_wrong_face_reengage_sweep.sh).
"""
from __future__ import annotations

import numpy as np
import pytest

from control.sampling_c3.commit_face_gate import (
    CommitFaceDecision,
    decide_commit_face_gate,
)
from control.sampling_c3.mode_switch import SwitchReason


# Production threshold pinned by plan 2026-06-10 (params.py:624).
PROD_THRESHOLD = 0.3


def _apply_override(prev_mode, mode, reason, dec):
    """Replicate the wrapper.py post-decide override branch."""
    if prev_mode == "free" and mode == "c3" and dec is not None and not dec.commit:
        return "free", SwitchReason.kStayInRepos
    return mode, reason


# ---------------------------------------------------------------------------
# 1. The override fires on free->c3 wrong-face entries
# ---------------------------------------------------------------------------

def test_override_refuses_kToC3Cost_wrong_face_entry():
    """seed-4 step 519 type entry: kToC3Cost on a wrong (anti-goal) face.
    face_align +0.975 -> refuse -> mode flips to 'free', reason=kStayInRepos."""
    box_xy = np.array([0.50, 0.0])
    # Place target on a face that gives face_align ≈ +0.975 under push-W.
    cx = -0.975
    cy = float(np.sqrt(1.0 - cx ** 2))
    p_repos = np.array([box_xy[0] + 0.08 * cx, box_xy[1] + 0.08 * cy])
    g_hat = np.array([-1.0, 0.0])

    dec = decide_commit_face_gate(p_repos, box_xy, g_hat,
                                  threshold=PROD_THRESHOLD)
    mode, reason = _apply_override("free", "c3", SwitchReason.kToC3Cost, dec)

    assert dec.face_align == pytest.approx(+0.975, abs=1e-9)
    assert dec.commit is False
    assert mode == "free"
    assert reason == SwitchReason.kStayInRepos


def test_override_refuses_kToC3ReachedReposTarget_wrong_face_entry():
    """The post-decide override fires on EVERY free->c3 transition, not
    just kToC3Cost. seed-0 step 434 type entry."""
    box_xy = np.array([0.0, 0.0])
    cx = -0.988
    cy = float(np.sqrt(1.0 - cx ** 2))
    p_repos = np.array([0.08 * cx, 0.08 * cy])
    g_hat = np.array([-1.0, 0.0])

    dec = decide_commit_face_gate(p_repos, box_xy, g_hat,
                                  threshold=PROD_THRESHOLD)
    mode, reason = _apply_override("free", "c3",
                                   SwitchReason.kToC3ReachedReposTarget, dec)

    assert dec.commit is False
    assert mode == "free"
    assert reason == SwitchReason.kStayInRepos


# ---------------------------------------------------------------------------
# 2. The override DOES NOT fire on legit free->c3 entries
# ---------------------------------------------------------------------------

def test_override_admits_productive_session_entry():
    """seed-4 step 315 type entry: face_align=-0.497 (the 114-tick productive
    westward session). Threshold +0.3 admits with 0.797 margin. Mode stays c3,
    reason preserved."""
    box_xy = np.array([0.0, 0.0])
    cx = 0.497          # face_out · (-x) = -0.497
    cy = float(np.sqrt(1.0 - cx ** 2))
    p_repos = np.array([0.08 * cx, 0.08 * cy])
    g_hat = np.array([-1.0, 0.0])

    dec = decide_commit_face_gate(p_repos, box_xy, g_hat,
                                  threshold=PROD_THRESHOLD)
    mode, reason = _apply_override("free", "c3", SwitchReason.kToC3Cost, dec)

    assert dec.face_align == pytest.approx(-0.497, abs=1e-9)
    assert dec.commit is True
    assert mode == "c3"
    assert reason == SwitchReason.kToC3Cost


def test_override_admits_transient_entry_at_minus_0p737():
    """seed-4 step 265 type entry: 1-tick legit transient. Admits."""
    box_xy = np.array([0.0, 0.0])
    cx = 0.737
    cy = float(np.sqrt(1.0 - cx ** 2))
    p_repos = np.array([0.08 * cx, 0.08 * cy])
    g_hat = np.array([-1.0, 0.0])

    dec = decide_commit_face_gate(p_repos, box_xy, g_hat,
                                  threshold=PROD_THRESHOLD)
    mode, reason = _apply_override("free", "c3",
                                   SwitchReason.kToC3ReachedReposTarget, dec)

    assert dec.commit is True
    assert mode == "c3"
    assert reason == SwitchReason.kToC3ReachedReposTarget


# ---------------------------------------------------------------------------
# 3. The override is scoped to free->c3 transitions only
# ---------------------------------------------------------------------------

def test_override_passthrough_when_prev_was_c3():
    """prev_mode='c3' (kStayInC3 case): override must NOT fire even on a
    refused gate. Only free->c3 transitions are gated. This protects c3
    sessions in progress from being interrupted mid-push."""
    box_xy = np.array([0.50, 0.0])
    cx = -0.988
    cy = float(np.sqrt(1.0 - cx ** 2))
    p_repos = np.array([box_xy[0] + 0.08 * cx, box_xy[1] + 0.08 * cy])
    g_hat = np.array([-1.0, 0.0])

    dec = decide_commit_face_gate(p_repos, box_xy, g_hat,
                                  threshold=PROD_THRESHOLD)
    mode, reason = _apply_override("c3", "c3", SwitchReason.kStayInC3, dec)

    assert dec.commit is False  # gate WOULD refuse, but prev_mode='c3'
    assert mode == "c3"
    assert reason == SwitchReason.kStayInC3


def test_override_passthrough_when_decide_mode_returned_free():
    """prev_mode='free' and decide_mode returned 'free': nothing to override."""
    box_xy = np.array([0.0, 0.0])
    cx = -0.988
    cy = float(np.sqrt(1.0 - cx ** 2))
    p_repos = np.array([0.08 * cx, 0.08 * cy])
    g_hat = np.array([-1.0, 0.0])

    dec = decide_commit_face_gate(p_repos, box_xy, g_hat,
                                  threshold=PROD_THRESHOLD)
    mode, reason = _apply_override("free", "free",
                                   SwitchReason.kStayInRepos, dec)

    assert mode == "free"
    assert reason == SwitchReason.kStayInRepos


def test_override_passthrough_when_no_repos_target():
    """When ``self._current_repos_target is None``, the helper returns
    None and the override is a no-op even on a free->c3 transition.
    Protects against an over-eager override during initialization or
    when the dispatcher just finished a c3 session and hasn't picked
    a new repos target yet."""
    mode, reason = _apply_override("free", "c3",
                                   SwitchReason.kToC3Cost, None)

    assert mode == "c3"
    assert reason == SwitchReason.kToC3Cost


# ---------------------------------------------------------------------------
# 4. SC contract: face_align of every override refusal must be > 0.0
#    Any refusal at face_align < 0.0 means the threshold mis-calibrated
#    and the parser MUST surface it as a NOREGRESS failure.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("legit_face_align", [
    -0.999,   # seed-4 step 159 entry
    -0.961,   # seed-0 step 192
    -0.737,   # seed-4 step 265 (closest to threshold)
    -0.497,   # seed-4 step 315 (the productive 114-tick session)
])
def test_no_legit_face_align_is_refused(legit_face_align):
    """For every legit face_align observed in q1_noregress, the
    +0.3 threshold MUST admit. This is the SC-NOREGRESS pin."""
    box_xy = np.array([0.0, 0.0])
    cx = float(-legit_face_align)
    cy = float(np.sqrt(max(0.0, 1.0 - cx ** 2)))
    p_repos = np.array([0.08 * cx, 0.08 * cy])
    g_hat = np.array([-1.0, 0.0])

    dec = decide_commit_face_gate(p_repos, box_xy, g_hat,
                                  threshold=PROD_THRESHOLD)
    assert dec.commit is True, (
        f"legit face_align={legit_face_align} would be REFUSED under +0.3 "
        "— threshold is mis-calibrated"
    )


@pytest.mark.parametrize("runaway_face_align", [
    +0.988,   # seed-0 step 434 (+60 mm goal_dist regression)
    +0.975,   # seed-4 step 519 (+93 mm goal_dist regression)
])
def test_every_runaway_face_align_is_refused(runaway_face_align):
    """For every measured runaway face_align, the +0.3 threshold MUST
    refuse. This is the SC-RUNAWAY-PREVENTED pin."""
    box_xy = np.array([0.0, 0.0])
    cx = float(-runaway_face_align)
    cy = float(np.sqrt(max(0.0, 1.0 - cx ** 2)))
    p_repos = np.array([0.08 * cx, 0.08 * cy])
    g_hat = np.array([-1.0, 0.0])

    dec = decide_commit_face_gate(p_repos, box_xy, g_hat,
                                  threshold=PROD_THRESHOLD)
    assert dec.commit is False, (
        f"runaway face_align={runaway_face_align} ADMITTED under +0.3 "
        "— threshold is mis-calibrated"
    )
