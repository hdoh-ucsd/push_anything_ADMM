"""Unit tests for the lateral-alignment clamp scale parameter.

Pure-Python tests of the clamp arithmetic and the cost_cfg dict loading.
Does NOT instantiate QuadraticManipulationCost (that requires a Drake plant
and full system context). Mirrors the task_costs.py:179 c.get(...) loading
pattern and the build_ee_space():~744 / build():~438 clamp formula.

Plan: docs/superpowers/plans/2026-06-07-B-lateral-align-clamp-harden.md
"""
import pytest


def _clamp_strength(perp_magnitude: float, scale: float) -> float:
    """Pure arithmetic reference — must match the lateral-alignment block at
    task_costs.py:744 (build_ee_space) and :438 (legacy build)."""
    return min(1.0, perp_magnitude / scale) if perp_magnitude > 1e-4 else 0.0


def test_clamp_arithmetic_default_scale():
    # At scale=0.05 (legacy/default):
    #   perp=5mm  -> strength=0.10 (10% of full)
    #   perp=25mm -> strength=0.50 (half strength)
    #   perp=50mm -> strength=1.00 (saturates)
    assert _clamp_strength(0.005, 0.05) == pytest.approx(0.10)
    assert _clamp_strength(0.025, 0.05) == pytest.approx(0.50)
    assert _clamp_strength(0.050, 0.05) == pytest.approx(1.00)
    assert _clamp_strength(0.100, 0.05) == pytest.approx(1.00)   # saturated


def test_clamp_arithmetic_hardened_scale():
    # At scale=0.015 (the B-fix value):
    #   perp=5mm  -> strength=0.33 (3.3x more responsive than legacy)
    #   perp=15mm -> strength=1.00 (saturates at 15mm vs legacy 50mm)
    #   perp=25mm -> strength=1.00 (still saturated)
    assert _clamp_strength(0.005, 0.015) == pytest.approx(1 / 3, abs=1e-6)
    assert _clamp_strength(0.015, 0.015) == pytest.approx(1.00)
    assert _clamp_strength(0.025, 0.015) == pytest.approx(1.00)


def test_clamp_arithmetic_zero_perp_no_shift():
    """If EE is exactly on the push axis, no shift applies."""
    assert _clamp_strength(0.0, 0.05) == 0.0
    assert _clamp_strength(0.0, 0.015) == 0.0
    # Below the 1e-4 floor that gates the lateral block
    assert _clamp_strength(5e-5, 0.015) == 0.0


def test_cost_cfg_loads_lateral_align_full_scale_default():
    """Without the YAML key, the field defaults to 0.05 (legacy behaviour)."""
    cost_cfg = {}
    # Mirror the c.get(...) line at task_costs.py:185 directly.
    val = float(cost_cfg.get("lateral_align_full_scale", 0.05))
    assert val == 0.05


def test_cost_cfg_loads_lateral_align_full_scale_explicit():
    cost_cfg = {"lateral_align_full_scale": 0.015}
    val = float(cost_cfg.get("lateral_align_full_scale", 0.05))
    assert val == 0.015


def test_cost_cfg_loads_lateral_align_full_scale_string_coerces():
    """YAML may deliver numeric values as strings in unusual configs."""
    cost_cfg = {"lateral_align_full_scale": "0.020"}
    val = float(cost_cfg.get("lateral_align_full_scale", 0.05))
    assert val == 0.020
