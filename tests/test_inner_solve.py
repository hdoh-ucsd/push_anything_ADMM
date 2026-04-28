"""Unit tests for the pure-numpy pieces of control.sampling_c3.inner_solve.

The Drake-using InnerSolver is exercised by the integration validation;
here we test traj_cost and traj_cost_breakdown numerically.
"""
import numpy as np
import pytest

from control.sampling_c3.inner_solve import traj_cost, traj_cost_breakdown


# ---------------------------------------------------------------------------
# traj_cost
# ---------------------------------------------------------------------------

def test_traj_cost_zero_at_reference():
    """When x_seq ≡ x_ref and u_seq ≡ 0 the cost is exactly 0."""
    n_x, n_u, N = 4, 2, 3
    x_ref = np.array([1.0, 2.0, 3.0, 4.0])
    x_seq = np.tile(x_ref, (N + 1, 1))
    u_seq = np.zeros((N, n_u))
    Q  = np.eye(n_x)
    R  = np.eye(n_u)
    QN = np.eye(n_x)
    assert traj_cost(x_seq, u_seq, Q, R, QN, x_ref) == 0.0


def test_traj_cost_quadratic_in_state_error():
    n_x, n_u, N = 2, 1, 1
    Q  = np.eye(n_x)
    R  = np.eye(n_u)
    QN = np.zeros((n_x, n_x))
    x_ref = np.zeros(n_x)
    x_seq = np.array([[1.0, 0.0], [0.0, 0.0]])   # one step at error 1
    u_seq = np.zeros((N, n_u))
    # cost = 1 * Q[0,0] * 1 + 0 (terminal)
    assert traj_cost(x_seq, u_seq, Q, R, QN, x_ref) == pytest.approx(1.0)


def test_traj_cost_includes_control_term():
    n_x, n_u, N = 1, 2, 2
    Q  = np.zeros((n_x, n_x))
    R  = 0.5 * np.eye(n_u)
    QN = np.zeros((n_x, n_x))
    x_ref = np.zeros(n_x)
    x_seq = np.zeros((N + 1, n_x))
    u_seq = np.array([[1.0, 0.0], [0.0, 2.0]])
    # cost = 0.5 * (1 + 4) = 2.5
    assert traj_cost(x_seq, u_seq, Q, R, QN, x_ref) == pytest.approx(2.5)


def test_traj_cost_includes_terminal_term():
    n_x, n_u, N = 2, 1, 1
    Q  = np.zeros((n_x, n_x))
    R  = np.zeros((n_u, n_u))
    QN = np.diag([10.0, 1.0])
    x_ref = np.zeros(n_x)
    x_seq = np.array([[0.0, 0.0], [1.0, 2.0]])
    u_seq = np.zeros((N, n_u))
    # terminal = 10*1 + 1*4 = 14
    assert traj_cost(x_seq, u_seq, Q, R, QN, x_ref) == pytest.approx(14.0)


# ---------------------------------------------------------------------------
# traj_cost_breakdown
# ---------------------------------------------------------------------------

def test_breakdown_keys_present():
    n_x, n_u, N = 27, 7, 2
    Q  = np.eye(n_x)
    R  = np.eye(n_u)
    QN = np.eye(n_x)
    x_ref = np.zeros(n_x)
    x_seq = np.zeros((N + 1, n_x))
    u_seq = np.zeros((N, n_u))
    bd = traj_cost_breakdown(
        x_seq, u_seq, Q, R, QN, x_ref,
        n_arm_dofs=7, obj_x_idx=11, obj_y_idx=12, obj_z_idx=13, obj_ps=7,
    )
    for k in ("obj_xy_term", "obj_z_term", "box_rp_term",
              "ee_approach", "torque", "terminal"):
        assert k in bd
        assert bd[k] == 0.0


def test_breakdown_isolates_obj_xy_term():
    n_x, n_u, N = 27, 7, 1
    Q  = np.zeros((n_x, n_x))
    R  = np.zeros((n_u, n_u))
    QN = np.zeros((n_x, n_x))
    x_ref = np.zeros(n_x)
    Q[11, 11] = 100.0   # obj_x weight
    Q[12, 12] = 100.0   # obj_y weight
    x_seq = np.zeros((N + 1, n_x))
    x_seq[0, 11] = 0.5   # 50 cm error in x
    x_seq[0, 12] = 0.0
    u_seq = np.zeros((N, n_u))
    bd = traj_cost_breakdown(
        x_seq, u_seq, Q, R, QN, x_ref,
        n_arm_dofs=7, obj_x_idx=11, obj_y_idx=12, obj_z_idx=13, obj_ps=7,
    )
    # only obj_xy contributes: 100 * 0.5^2 = 25
    assert bd["obj_xy_term"] == pytest.approx(25.0)
    assert bd["torque"]      == 0.0
    assert bd["terminal"]    == 0.0
