"""Verify the default osc_franka.yaml provides joint-2 posture gains
and (Task 4) that OperationalSpaceController defaults to reference
c3-mode gains.

Reproduce-dairlib Phase 1.
"""
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from control.osc.operational_space_controller import (
    OperationalSpaceController,
    _load_osc_gains,
)


def test_default_yaml_provides_joint2_gains():
    gains, _ = _load_osc_gains(Path("config/osc_franka.yaml"), n_arm=7)
    assert gains.Kp_joint2 == 200.0
    assert gains.Kd_joint2 == 10.0
    assert gains.W_joint2 == 1.0
    assert gains.joint2_target_rad == 1.1
    assert gains.joint2_idx == 1


def _stub_plant(n_v=8):
    plant = MagicMock()
    plant.num_velocities.return_value = n_v
    plant.world_frame.return_value = MagicMock()
    return plant


def test_default_c3_mode_uses_reference_gains(monkeypatch):
    """Task 4: no env vars → reference c3-gains are default."""
    monkeypatch.delenv("PUSHA_OSC_C3_MODE_LEGACY_GAINS", raising=False)
    monkeypatch.delenv("PUSHA_OSC_C3_MODE_REFERENCE_GAINS", raising=False)
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
    # Free-mode gains still port defaults from YAML.
    assert osc.gains.Kp_cart.tolist() == [400.0, 400.0, 400.0]
    assert osc.gains.W_track == 100.0


def test_legacy_env_var_disables_reference_c3_gains(monkeypatch):
    """Task 4: PUSHA_OSC_C3_MODE_LEGACY_GAINS=1 opts out."""
    monkeypatch.setenv("PUSHA_OSC_C3_MODE_LEGACY_GAINS", "1")
    plant = _stub_plant()
    ee_frame = MagicMock()
    osc = OperationalSpaceController(
        plant=plant, ee_frame=ee_frame, n_arm_dofs=7,
        q_nominal=np.zeros(7), gains_yaml="config/osc_franka.yaml",
        use_force_tracking=True, W_force=1.0,
    )
    assert osc._c3_ref_gains_flag is False
