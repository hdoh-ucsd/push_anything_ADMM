"""Unit test: the OSC QP's λ_ext decision variable tracks λ_des.

Mirrors dairlib reference's `ExternalForceTrackingData` mechanism
(systems/controllers/osc/external_force_tracking_data.cc). Verifies
that with a high force-tracking weight and zero position error, the
QP's solved λ_ext converges to the commanded λ_des, and that the
implied joint torques are consistent with `J_v^T·λ_ext`.

This test uses a deliberately synthetic problem (M=I, B=I, J_v=eye-prefix)
so the QP's behavior depends only on the cost-vs-constraint balance —
not on a Drake plant — making the result interpretable.
"""
import numpy as np
import pytest

from control.osc.qp_builder import OscGains, OscLimits, build_and_solve_qp


def _make_synthetic_qp_inputs(n_v=7, n_u=7, n_arm=7, W_force=100.0):
    M    = np.eye(n_v)
    bias = np.zeros(n_v)
    B    = np.eye(n_v, n_u)
    J_v  = np.zeros((3, n_v))
    J_v[0, 0] = 1.0
    J_v[1, 1] = 1.0
    J_v[2, 2] = 1.0
    Jdot_v_v = np.zeros(3)
    p_err    = np.zeros(3)
    v_err    = np.zeros(3)
    q_arm_err = np.zeros(n_arm)
    v_arm_err = np.zeros(n_arm)
    gains = OscGains(
        Kp_cart   = np.array([400.0, 400.0, 400.0]),
        Kd_cart   = np.array([ 40.0,  40.0,  40.0]),
        Kp_null   = np.full(n_arm, 10.0),
        Kd_null   = np.full(n_arm,  3.0),
        W_track   = 100.0,
        W_posture =   1.0,
        W_torque  =   0.001,
        W_acc     =   0.001,
        W_force   = W_force,
    )
    limits = OscLimits(tau_max=np.full(n_arm, 87.0))
    return dict(
        M=M, bias=bias, B=B, n_arm=n_arm,
        J_v=J_v, Jdot_v_v=Jdot_v_v,
        p_err=p_err, v_err=v_err,
        q_arm_err=q_arm_err, v_arm_err=v_arm_err,
        gains=gains, limits=limits,
        F_ff_external=np.zeros(n_v),
    )


def test_lambda_ext_tracks_lambda_des_high_W_force():
    """With W_force = 100 (= W_track) and no position error, λ_ext should
    track λ_des to within 1 N (the diagnosis's acceptance threshold)."""
    inputs = _make_synthetic_qp_inputs(W_force=100.0)
    lam_des = np.array([-5.0, 0.0, 0.0])
    u_opt, vdot_opt, success, result_str, lam_ext_opt = build_and_solve_qp(
        **inputs,
        use_force_tracking=True,
        lambda_des=lam_des,
    )
    assert success, f"QP failed: {result_str}"
    err = np.linalg.norm(lam_ext_opt - lam_des)
    assert err < 1.0, (
        f"λ_ext should track λ_des within 1 N — got "
        f"λ_ext={lam_ext_opt}, λ_des={lam_des}, |err|={err:.3f} N"
    )


def test_lambda_ext_implies_consistent_joint_torque():
    """The dynamics constraint M v̇ + bias = B u + J_v^T λ_ext must hold
    exactly. Verify the equality numerically."""
    inputs = _make_synthetic_qp_inputs(W_force=100.0)
    lam_des = np.array([-5.0, 0.0, 0.0])
    u_opt, vdot_opt, success, result_str, lam_ext_opt = build_and_solve_qp(
        **inputs,
        use_force_tracking=True,
        lambda_des=lam_des,
    )
    assert success, result_str
    lhs = inputs["M"] @ vdot_opt + inputs["bias"]
    rhs = inputs["B"] @ u_opt + inputs["J_v"].T @ lam_ext_opt
    np.testing.assert_allclose(lhs, rhs, atol=1e-4)


def test_force_tracking_off_falls_back_to_legacy_path():
    """With use_force_tracking=False the QP must not declare λ_ext, and
    must accept F_ff_external as the fixed RHS feedforward."""
    inputs = _make_synthetic_qp_inputs(W_force=0.0)
    inputs["F_ff_external"] = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    u_opt, vdot_opt, success, result_str, lam_ext_opt = build_and_solve_qp(
        **inputs,
        use_force_tracking=False,
        lambda_des=None,
    )
    assert success, result_str
    np.testing.assert_allclose(lam_ext_opt, np.zeros(3), atol=1e-9)


if __name__ == "__main__":
    test_lambda_ext_tracks_lambda_des_high_W_force()
    print("PASS test_lambda_ext_tracks_lambda_des_high_W_force")
    test_lambda_ext_implies_consistent_joint_torque()
    print("PASS test_lambda_ext_implies_consistent_joint_torque")
    test_force_tracking_off_falls_back_to_legacy_path()
    print("PASS test_force_tracking_off_falls_back_to_legacy_path")
