"""Structure of the first-order linearization error in CRISP B-B dynamics.

B-B's dynamics are not generically nonlinear -- they are *bilinear*:

    f_theta = k_rot (c_x*lam_y - c_y*lam_x)     exactly bilinear in (c, lam)
    f_px    = k_t [ lam_x cos(th) - lam_y sin(th) ]   linear in lam, trig in th
    f_py    = k_t [ lam_x sin(th) + lam_y cos(th) ]

so the Taylor remainder is exactly computable rather than merely O(delta^2).
These tests pin that structure down, because it is what decides whether a
locally linearized adapter is usable at C3+ step sizes.
"""
import numpy as np
import pytest

from control.crisp.push_box import PushBoxParams, PushBoxProblem

P = PushBoxParams(a=0.5, b=0.25, mu=0.5, mass=1.0, g=9.8, c_int=0.4,
                  N=4, dt=0.02)


def _prob():
    return PushBoxProblem(P, np.zeros(3), np.array([1.0, 0.0, 0.0]))


def _lin_error(prob, q_bar, u_bar, dq, du):
    """||f_true - f_lin||, componentwise, for a perturbation (dq, du)."""
    J = prob.dynamics_jacobian(q_bar, u_bar)
    dz = np.concatenate([dq, du])
    f_true = prob.dynamics(q_bar + dq, u_bar + du)
    f_lin = prob.dynamics(q_bar, u_bar) + J @ dz
    return f_true - f_lin


NOMINAL_Q = np.array([0.3, -0.2, 0.4])
NOMINAL_U = np.array([-0.5, 0.1, 0.0, 2.0, 0.0, 0.0])


def test_zero_perturbation_has_zero_error():
    prob = _prob()
    err = _lin_error(prob, NOMINAL_Q, NOMINAL_U, np.zeros(3), np.zeros(6))
    np.testing.assert_allclose(err, 0.0, atol=1e-15)


def test_contact_only_perturbation_is_exactly_linear():
    """f is linear in c at fixed lambda, so Delta-c alone has NO error."""
    prob = _prob()
    du = np.zeros(6)
    du[:2] = [0.07, -0.04]
    err = _lin_error(prob, NOMINAL_Q, NOMINAL_U, np.zeros(3), du)
    np.testing.assert_allclose(err, 0.0, atol=1e-14)


def test_force_only_perturbation_is_exactly_linear():
    """f is linear in lambda at fixed (theta, c), so Delta-lambda alone is exact."""
    prob = _prob()
    du = np.zeros(6)
    du[2:] = [0.3, -0.5, 0.2, 0.1]
    err = _lin_error(prob, NOMINAL_Q, NOMINAL_U, np.zeros(3), du)
    np.testing.assert_allclose(err, 0.0, atol=1e-14)


def test_joint_contact_force_error_equals_the_bilinear_cross_term():
    """The whole remainder in the yaw row is k_rot(dc_x dlam_y - dc_y dlam_x)."""
    prob = _prob()
    du = np.zeros(6)
    du[:2] = [0.06, -0.03]
    du[2:] = [0.4, -0.2, 0.1, 0.05]

    err = _lin_error(prob, NOMINAL_Q, NOMINAL_U, np.zeros(3), du)

    d_lam_y = du[2] + du[4]
    d_lam_x = du[3] + du[5]
    exact = prob.k_rot * (du[0] * d_lam_y - du[1] * d_lam_x)
    assert err[2] == pytest.approx(exact, abs=1e-14)
    np.testing.assert_allclose(err[:2], 0.0, atol=1e-14)  # translation unaffected


def test_yaw_row_error_is_independent_of_pose_perturbation():
    """f_theta does not depend on q at all, so Delta-theta cannot perturb it."""
    prob = _prob()
    dq = np.array([0.1, -0.2, 0.3])
    err = _lin_error(prob, NOMINAL_Q, NOMINAL_U, dq, np.zeros(6))
    assert err[2] == pytest.approx(0.0, abs=1e-15)


def test_orientation_perturbation_error_is_second_order():
    """Translation rows carry a trig remainder; halving d(theta) quarters it."""
    prob = _prob()
    errs = []
    for d in (2e-2, 1e-2):
        dq = np.array([0.0, 0.0, d])
        errs.append(np.linalg.norm(_lin_error(prob, NOMINAL_Q, NOMINAL_U,
                                              dq, np.zeros(6))))
    ratio = errs[0] / errs[1]
    assert 3.6 < ratio < 4.4, f"expected ~4 (second order), got {ratio}"


def test_all_zero_nominal_has_a_rank_deficient_jacobian():
    """At z=0 the yaw row vanishes entirely: rank drops and rotation is blind."""
    prob = _prob()
    J = prob.dynamics_jacobian(np.zeros(3), np.zeros(6))

    np.testing.assert_allclose(J[2], 0.0, atol=1e-15)
    assert np.linalg.matrix_rank(J) == 2
