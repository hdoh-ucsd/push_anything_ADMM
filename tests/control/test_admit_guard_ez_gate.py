"""Unit tests for the admit-guard EE_z gate.

The gate's job is to decide whether the admit-guard's z_safe cap should
fire. The cap is the existing behavior at reposition_ik.py:1260; the
gate adds an EE_z condition so the cap only fires at face-approach
altitude (legitimate use) and pass-throughs mid-traverse (the bump).

Threshold derivation: 2 mm above the peak observed legit-push EE_z
(0.088 m, F=5 seed3 mid-push wobble) -> 0.090 m. See
docs/superpowers/plans/2026-06-04-admit-guard-ee-z-gate.md §Threshold.
"""
import pytest

from control.sampling_c3.reposition_ik import _should_cap_z_safe


# Threshold MUST stay in sync with the constant in reposition_ik.py.
EE_Z_GATE = 0.090


def test_no_latch_never_caps():
    """Latch == 0 means no recent admit. Cap must not fire regardless of ee_z."""
    assert _should_cap_z_safe(admit_latch=0, ee_z=0.05, ee_z_gate=EE_Z_GATE) is False
    assert _should_cap_z_safe(admit_latch=0, ee_z=0.15, ee_z_gate=EE_Z_GATE) is False


def test_latched_low_ee_caps():
    """Legit face-approach: latch active, ee_z below gate. Cap fires."""
    assert _should_cap_z_safe(admit_latch=8, ee_z=0.05, ee_z_gate=EE_Z_GATE) is True
    assert _should_cap_z_safe(admit_latch=4, ee_z=0.088, ee_z_gate=EE_Z_GATE) is True
    assert _should_cap_z_safe(admit_latch=1, ee_z=0.089999, ee_z_gate=EE_Z_GATE) is True


def test_latched_high_ee_passes_through():
    """The bump case: latch active (LCS near-miss), ee_z above gate. Cap MUST NOT fire."""
    # F=5 seed3 step 521 bump-onset EE_z.
    assert _should_cap_z_safe(admit_latch=8, ee_z=0.099, ee_z_gate=EE_Z_GATE) is False
    assert _should_cap_z_safe(admit_latch=8, ee_z=0.120, ee_z_gate=EE_Z_GATE) is False
    assert _should_cap_z_safe(admit_latch=2, ee_z=0.150, ee_z_gate=EE_Z_GATE) is False


def test_gate_threshold_strict_inequality():
    """ee_z exactly at the gate: cap does NOT fire (>= passes through).

    Tie direction deliberate: false-cap is the documented bump (worse);
    false-pass at boundary is a brief sub-optimal contact altitude (milder).
    """
    assert _should_cap_z_safe(admit_latch=8, ee_z=EE_Z_GATE, ee_z_gate=EE_Z_GATE) is False


def test_gate_zero_threshold_disables_gate():
    """ee_z_gate == 0.0 disables the gate (legacy unconditional-cap fallback)."""
    assert _should_cap_z_safe(admit_latch=1, ee_z=0.50, ee_z_gate=0.0) is True
    assert _should_cap_z_safe(admit_latch=0, ee_z=0.50, ee_z_gate=0.0) is False
