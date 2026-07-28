"""Verify config/osc_franka.yaml carries the reference gain set as the
DEFAULT (2026-07-28 defaults flip: the former §7.43/§7.70 env gates —
REFCONF_OSC_ALIGN / REFCONF_OSC_C3_MODE_GAINS / REFCONF_OSC_FREE_MODE_GAINS
/ REFCONF_OSC_EE_ROT_TASK — were removed; ONE gain set for all modes,
matching reference osc_params.yaml + franka_osc_controller.cc:171-187).

Joint-2 posture stays Option-A (W=0, port frame decision — the reference's
raised-mount 1.1 rad elbow pin does not transfer; see
memory/project_reproduce_dairlib_restored_main_joint2_flag).
"""
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from control.osc.operational_space_controller import (
    OperationalSpaceController,
    _load_osc_gains,
)


def test_default_yaml_provides_joint2_gains_but_weight_off():
    """Joint-2 scaffolding kept reference-valued, W=0 per Option A."""
    gains, _ = _load_osc_gains(Path("config/osc_franka.yaml"), n_arm=7)
    assert gains.Kp_joint2 == 200.0
    assert gains.Kd_joint2 == 10.0
    assert gains.W_joint2 == 0.0     # Option A: port-frame decision
    assert gains.joint2_target_rad == 1.1
    assert gains.joint2_idx == 1


def _stub_plant(n_v=8):
    plant = MagicMock()
    plant.num_velocities.return_value = n_v
    plant.world_frame.return_value = MagicMock()
    return plant


def _make_osc():
    return OperationalSpaceController(
        plant=_stub_plant(), ee_frame=MagicMock(), n_arm_dofs=7,
        q_nominal=np.zeros(7), gains_yaml="config/osc_franka.yaml",
        use_force_tracking=True, W_force=1.0,
    )


def test_single_reference_gain_set_all_modes():
    """The yaml IS the reference set; no per-mode gain structs remain.

    Reference values: EndEffectorW=1, Kp=200, Kd=20 (osc_params.yaml:47-58),
    EndEffectorRotW/Kp/Kd = 10/800/40 (:59-70), LambdaEndEffectorW=1 (:74),
    end_effector_acceleration=10 (:36).
    """
    osc = _make_osc()
    assert osc.gains.Kp_cart.tolist() == [200.0, 200.0, 200.0]
    assert osc.gains.Kd_cart.tolist() == [20.0, 20.0, 20.0]
    assert osc.gains.W_track == 1.0
    assert osc.gains.W_force == 1.0
    assert osc.gains.W_rot == 10.0
    assert osc.gains.a_ee_cap == 10.0
    # The per-mode swap machinery is gone.
    assert not hasattr(osc, "gains_c3")
    assert not hasattr(osc, "_c3_ref_gains_flag")
