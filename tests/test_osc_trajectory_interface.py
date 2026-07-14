"""compute_torque_from_trajectory — degenerate single-knot
PiecewisePolynomial delegates to compute_torque with the evaluated
3D setpoint.

Reproduce-dairlib Phase 1, Task 5.
"""
import numpy as np
from unittest.mock import MagicMock

from pydrake.trajectories import PiecewisePolynomial

from control.osc.operational_space_controller import OperationalSpaceController


def _stub_plant(n_v=8):
    plant = MagicMock()
    plant.num_velocities.return_value = n_v
    plant.world_frame.return_value = MagicMock()
    return plant


def _single_knot_pp(point_3d, t=0.0):
    """Constant-in-time single-knot ZOH trajectory at `point_3d`."""
    p = np.asarray(point_3d, dtype=float).reshape(3, 1)
    return PiecewisePolynomial.ZeroOrderHold(
        [t, t + 1e-3], np.hstack([p, p])
    )


def test_trajectory_interface_delegates_to_compute_torque(monkeypatch):
    monkeypatch.delenv("PUSHA_OSC_C3_MODE_LEGACY_GAINS", raising=False)
    plant = _stub_plant()
    ee_frame = MagicMock()
    osc = OperationalSpaceController(
        plant=plant, ee_frame=ee_frame, n_arm_dofs=7,
        q_nominal=np.zeros(7), gains_yaml="config/osc_franka.yaml",
        use_force_tracking=True, W_force=1.0,
    )

    # Capture kwargs passed to compute_torque.
    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return np.zeros(7), {"stubbed": True}

    osc.compute_torque = _capture

    pp = _single_knot_pp(np.array([0.1, 0.2, 0.3]), t=0.0)
    u, diag = osc.compute_torque_from_trajectory(
        traj=pp, t_sim=0.0,
        current_q=np.zeros(9), current_v=np.zeros(8),
        plant_ctx=MagicMock(),
        mode="c3",
    )
    np.testing.assert_allclose(
        captured["p_ee_desired"], [0.1, 0.2, 0.3], atol=1e-12)
    assert captured["mode"] == "c3"
    assert diag["stubbed"] is True


def test_trajectory_interface_forwards_all_optional_args(monkeypatch):
    monkeypatch.delenv("PUSHA_OSC_C3_MODE_LEGACY_GAINS", raising=False)
    plant = _stub_plant()
    ee_frame = MagicMock()
    osc = OperationalSpaceController(
        plant=plant, ee_frame=ee_frame, n_arm_dofs=7,
        q_nominal=np.zeros(7), gains_yaml="config/osc_franka.yaml",
        use_force_tracking=True, W_force=1.0,
    )
    captured = {}

    def _capture(**kwargs):
        captured.update(kwargs)
        return np.zeros(7), {}

    osc.compute_torque = _capture
    pp = _single_knot_pp(np.array([0.0, 0.0, 0.05]), t=0.0)
    lam_des = np.array([1.0, 0.0, -2.0])
    lam_n = np.array([0.5])
    _ = osc.compute_torque_from_trajectory(
        traj=pp, t_sim=0.0,
        current_q=np.zeros(9), current_v=np.zeros(8),
        plant_ctx=MagicMock(),
        lambda_des=lam_des,
        lambda_n=lam_n,
        mode="free",
    )
    np.testing.assert_allclose(captured["lambda_des"], lam_des)
    np.testing.assert_allclose(captured["lambda_n"], lam_n)
    assert captured["mode"] == "free"
