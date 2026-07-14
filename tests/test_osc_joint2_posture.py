"""Joint-2 posture cost — the OSC pulls joint 2 back toward 1.1 rad.

Mirrors dairlib's `JointSpaceTrackingData("panda_joint2_target")` at
`franka_osc_controller.cc:159-165` (Kp=200, Kd=10, W=1, target=1.1 rad).
Reproduce-dairlib Phase 1, Task 1.
"""
import numpy as np

from control.osc.qp_builder import OscGains, OscLimits, build_and_solve_qp


def _minimal_dyn(n_v=8, n_u=7):
    """Small synthetic dynamics (7-DOF arm + 1 dummy DOF)."""
    M = np.eye(n_v)
    bias = np.zeros(n_v)
    B = np.vstack([np.eye(n_u), np.zeros((n_v - n_u, n_u))])
    J_v = np.zeros((3, n_v))
    J_v[0, 0] = 1.0  # trivial mapping
    return M, bias, B, J_v, np.zeros(3)


def _gains_with_joint2(W_joint2, Kp_joint2=200.0, Kd_joint2=10.0, target=1.1):
    return OscGains(
        Kp_cart   = np.array([1.0, 1.0, 1.0]),
        Kd_cart   = np.array([0.0, 0.0, 0.0]),
        Kp_null   = np.zeros(7),
        Kd_null   = np.zeros(7),
        W_track   = 0.0,
        W_posture = 0.0,
        W_torque  = 1e-6,
        W_acc     = 1e-6,
        W_force   = 0.0,
        Kp_joint2 = Kp_joint2,
        Kd_joint2 = Kd_joint2,
        W_joint2  = W_joint2,
        joint2_target_rad = target,
        joint2_idx = 1,   # Franka joint 2 → arm-index 1
    )


def test_joint2_posture_pulls_toward_target():
    """With W_joint2 > 0 and q[joint2] < target, v̇[joint2] is positive."""
    M, bias, B, J_v, Jdot_v_v = _minimal_dyn()
    q_arm = np.zeros(7)   # joint 2 at 0 → target error = 1.1
    v_arm = np.zeros(7)
    gains = _gains_with_joint2(W_joint2=1.0)
    limits = OscLimits(tau_max=np.full(7, 100.0))

    u_opt, vdot_opt, success, _, _ = build_and_solve_qp(
        M=M, bias=bias, B=B, n_arm=7,
        J_v=J_v, Jdot_v_v=Jdot_v_v,
        p_err=np.zeros(3), v_err=np.zeros(3),
        q_arm_err=np.zeros(7), v_arm_err=np.zeros(7),
        gains=gains, limits=limits,
        F_ff_external=np.zeros(8),
        q_arm=q_arm, v_arm=v_arm,
    )
    assert success
    # a_j2 = 200 * (1.1 - 0) + 10 * 0 = 220 → v̇[1] should be strongly positive.
    assert vdot_opt[1] > 0.5, f"joint-2 v̇ should be positive, got {vdot_opt[1]}"


def test_joint2_posture_off_when_weight_zero():
    """With W_joint2 = 0, the joint-2 term contributes nothing."""
    M, bias, B, J_v, Jdot_v_v = _minimal_dyn()
    q_arm = np.zeros(7)
    v_arm = np.zeros(7)
    gains_off = _gains_with_joint2(W_joint2=0.0)
    limits = OscLimits(tau_max=np.full(7, 100.0))
    u_opt, vdot_opt, success, _, _ = build_and_solve_qp(
        M=M, bias=bias, B=B, n_arm=7,
        J_v=J_v, Jdot_v_v=Jdot_v_v,
        p_err=np.zeros(3), v_err=np.zeros(3),
        q_arm_err=np.zeros(7), v_arm_err=np.zeros(7),
        gains=gains_off, limits=limits,
        F_ff_external=np.zeros(8),
        q_arm=q_arm, v_arm=v_arm,
    )
    assert success
    # With every cost weight zero except tiny torque/accel reg, vdot ≈ 0.
    assert np.max(np.abs(vdot_opt)) < 1e-3
