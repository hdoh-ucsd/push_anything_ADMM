"""Over-drive cluster step 2 — EE-orientation task conformance.

Reference (dairlib push_anything_dev @ 257e3ed):
  * franka_osc_controller.cc:171-187 — RotTaskSpaceTrackingData is added
    UNCONDITIONALLY with W=EndEffectorRotW, Kp=EndEffectorRotKp,
    Kd=EndEffectorRotKd (osc_params.yaml:59-70 → 10/800/40 diagonal).
  * osc_params.yaml:37 `track_end_effector_orientation: false` gates ONLY
    the trajectory source (end_effector_orientation.cc:41-57): when false
    the generator outputs a CONSTANT identity-quaternion slerp — the
    rotation-hold cost is always in the QP.
  * rot_space_tracking_data.cc:60-68 UpdateYError — error is the EXACT
    angle-axis of (q_des · q_now⁻¹): error_y = angle * axis (log map),
    not a small-angle skew extraction.

Port: REFCONF_OSC_EE_ROT_TASK=1 (default OFF) applies W_rot=10,
Kp_rot=800·1, Kd_rot=40·1 to BOTH gain structs (one set for all modes,
as in the reference).
"""
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydrake.math import RotationMatrix, RollPitchYaw

from control.osc.dynamics_helpers import rotation_error_world
from control.osc.operational_space_controller import OperationalSpaceController


def _make_osc():
    plant = MagicMock()
    plant.num_velocities.return_value = 8
    plant.world_frame.return_value = MagicMock()
    return OperationalSpaceController(
        plant=plant, ee_frame=MagicMock(), n_arm_dofs=7,
        q_nominal=np.zeros(7), gains_yaml="config/osc_franka.yaml",
        use_force_tracking=True, W_force=1.0,
    )


def test_rot_task_default_off(monkeypatch):
    """No env var → W_rot stays at the yaml default (0.0) in both structs."""
    monkeypatch.delenv("REFCONF_OSC_EE_ROT_TASK", raising=False)
    osc = _make_osc()
    assert osc.gains.W_rot == 0.0
    assert osc.gains_c3.W_rot == 0.0


def test_rot_task_flag_applies_reference_gains_both_modes(monkeypatch):
    """REFCONF_OSC_EE_ROT_TASK=1 → 10/800/40 on BOTH gain structs.

    Reference has ONE rotation tracking data active in every mode
    (franka_osc_controller.cc:187 AddTrackingData, unconditional).
    """
    monkeypatch.setenv("REFCONF_OSC_EE_ROT_TASK", "1")
    osc = _make_osc()
    for g in (osc.gains, osc.gains_c3):
        assert g.W_rot == 10.0
        assert np.asarray(g.Kp_rot).tolist() == [800.0, 800.0, 800.0]
        assert np.asarray(g.Kd_rot).tolist() == [40.0, 40.0, 40.0]


def test_rotation_error_exact_angle_axis_90deg():
    """90° error must give |w_err| = π/2 (exact log map).

    The pre-conformance small-angle extraction ½·[R−Rᵀ] returns
    sin(θ)·axis = 1.0 at 90° — the reference (rot_space_tracking_data.cc
    UpdateYError) returns θ·axis = π/2 ≈ 1.5708.
    """
    R_now = RotationMatrix()
    R_target = RotationMatrix(RollPitchYaw(0.0, 0.0, np.pi / 2))
    w_err = rotation_error_world(R_target, R_now)
    np.testing.assert_allclose(w_err, [0.0, 0.0, np.pi / 2], atol=1e-12)


def test_rotation_error_zero_at_identity():
    R = RotationMatrix(RollPitchYaw(0.3, -0.2, 1.1))
    w_err = rotation_error_world(R, R)
    np.testing.assert_allclose(w_err, np.zeros(3), atol=1e-12)


def test_rotation_error_large_angle_no_degeneracy():
    """170° about x: small-angle form collapses toward sin(170°)≈0.17;
    the exact form must return ≈2.967 rad."""
    R_now = RotationMatrix()
    ang = np.deg2rad(170.0)
    R_target = RotationMatrix(RollPitchYaw(ang, 0.0, 0.0))
    w_err = rotation_error_world(R_target, R_now)
    np.testing.assert_allclose(w_err, [ang, 0.0, 0.0], atol=1e-9)


def test_rotation_error_world_frame_convention():
    """Error convention matches reference: q_des · q_now⁻¹ (world frame).

    With R_now = RotZ(45°) and R_target = RotZ(90°), the world-frame
    error is RotZ(45°) → w_err = [0, 0, π/4]."""
    R_now = RotationMatrix(RollPitchYaw(0.0, 0.0, np.pi / 4))
    R_target = RotationMatrix(RollPitchYaw(0.0, 0.0, np.pi / 2))
    w_err = rotation_error_world(R_target, R_now)
    np.testing.assert_allclose(w_err, [0.0, 0.0, np.pi / 4], atol=1e-12)
