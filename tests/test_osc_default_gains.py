"""Verify the default osc_franka.yaml provides joint-2 posture gains
(Task 3 — kept) and the opt-in semantics of the §7.70 reference c3-gains
flag (restored after Phase-1 re-cert on 2026-07-13 — see
memory/project_reproduce_dairlib_phase1_recert_false_positive.md).
"""
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from control.osc.operational_space_controller import (
    OperationalSpaceController,
    _load_osc_gains,
)


def test_default_yaml_provides_joint2_gains_but_weight_off():
    """Task 3 scaffolding kept, W_joint2 set to 0.0 post-recert.

    The Kp/Kd/target/idx values remain reference-aligned so a future
    coupled cost/executor re-tune can turn W back on without editing
    the YAML. Post-2026-07-13-recert default: W=0 (cost term inert).
    """
    gains, _ = _load_osc_gains(Path("config/osc_franka.yaml"), n_arm=7)
    assert gains.Kp_joint2 == 200.0
    assert gains.Kd_joint2 == 10.0
    assert gains.W_joint2 == 0.0     # default-OFF: see YAML comment for context
    assert gains.joint2_target_rad == 1.1
    assert gains.joint2_idx == 1


def _stub_plant(n_v=8):
    plant = MagicMock()
    plant.num_velocities.return_value = n_v
    plant.world_frame.return_value = MagicMock()
    return plant


def test_default_c3_mode_uses_port_gains(monkeypatch):
    """§7.70 semantics restored: reference c3-gains are OPT-IN, not default.

    Post-recert (2026-07-13): the 18498c1 flip that made reference c3-gains
    the default was reverted because it was 200× too gentle for the port's
    clean Q (66.5% closure vs the clean 75.3% baseline on the same stack).
    """
    monkeypatch.delenv("PUSHA_OSC_C3_MODE_REFERENCE_GAINS", raising=False)
    monkeypatch.delenv("PUSHA_REF_OSC_ALIGN", raising=False)
    plant = _stub_plant()
    ee_frame = MagicMock()
    osc = OperationalSpaceController(
        plant=plant, ee_frame=ee_frame, n_arm_dofs=7,
        q_nominal=np.zeros(7), gains_yaml="config/osc_franka.yaml",
        use_force_tracking=True, W_force=1.0,
    )
    # No env var → port gains everywhere.
    assert osc._c3_ref_gains_flag is False
    assert osc.gains.Kp_cart.tolist() == [400.0, 400.0, 400.0]
    assert osc.gains.W_track == 100.0


def test_opt_in_env_var_enables_reference_c3_gains(monkeypatch):
    """PUSHA_OSC_C3_MODE_REFERENCE_GAINS=1 activates reference gains in c3 mode."""
    monkeypatch.setenv("PUSHA_OSC_C3_MODE_REFERENCE_GAINS", "1")
    monkeypatch.delenv("PUSHA_REF_OSC_ALIGN", raising=False)
    plant = _stub_plant()
    ee_frame = MagicMock()
    osc = OperationalSpaceController(
        plant=plant, ee_frame=ee_frame, n_arm_dofs=7,
        q_nominal=np.zeros(7), gains_yaml="config/osc_franka.yaml",
        use_force_tracking=True, W_force=1.0,
    )
    assert osc._c3_ref_gains_flag is True
    assert osc.gains_c3.Kp_cart.tolist() == [200.0, 200.0, 200.0]
    assert osc.gains_c3.Kd_cart.tolist() == [20.0, 20.0, 20.0]
    assert osc.gains_c3.W_track == 1.0
    # Free-mode gains untouched (port defaults from YAML).
    assert osc.gains.Kp_cart.tolist() == [400.0, 400.0, 400.0]
    assert osc.gains.W_track == 100.0
