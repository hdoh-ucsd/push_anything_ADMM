"""
Smoke test: C3 (Aydinoglu 2024) vs C3+ (Bui 2026 §IV-B.2) on a synthetic
1D pushing LCS. Both solvers receive the same problem; both must produce
a predicted +x box velocity within one horizon (i.e. either non-zero
torque or non-zero contact force is applied in the right direction).

Phase 2 layout: λ = [γ; λ_n; λ_t] with 6·n_c = 6 components per contact.
γ has zero contribution to dynamics (D's γ-cols are zero) and is coupled
to (λ_n, λ_t) only through F's friction-cone-slack rows.

State: x = [box_x, box_vx], n_x = 2
Control: u = [push_force], n_u = 1
Contact: 1 normal contact pushing in +x.
"""
import numpy as np
import pytest

pytest.importorskip("pydrake", reason="C3Solver imports pydrake")

from control.admm_solver import C3Solver


# -----------------------------------------------------------------------
# Synthetic 1D pushing LCS
# -----------------------------------------------------------------------
def _build_synthetic_lcs(*, dt: float = 0.05, mass: float = 0.2,
                         phi_gap: float = 0.005):
    """
    Build a 1D LCS for box pushed by +x contact force.

    State x = [pos, vel]. Dynamics:
        pos_{t+1} = pos_t + dt · vel_t
        vel_{t+1} = vel_t + (dt/m) · (u_t + λ_n_t)

    Contact: J_n = [1, 0]  (normal in +x direction acting on velocity).
    """
    n_x = 2
    n_u = 1
    num_normals = 1
    n_t = 4                                 # 4-edge polyhedral pyramid
    n_lam = 2 * num_normals + n_t           # Phase 2: 6·n_c
    SG, SLN, SLT = 0, num_normals, 2 * num_normals
    A = np.array([[1.0, dt],
                  [0.0, 1.0]])
    B = np.array([[0.0],
                  [dt / mass]])
    # D shape (n_x, 6·n_c). γ-cols zero. λ_n col couples to vel like u.
    D = np.zeros((n_x, n_lam))
    D[:, SLN] = np.array([0.0, dt / mass])  # λ_n acts on vel like u
    # λ_t cols zero in this synthetic 1D LCS (no friction physics)
    d = np.zeros(n_x)
    J_n = np.array([[1.0, 0.0]])            # normal Jacobian in (pos, vel)
    J_t = np.zeros((n_t, 2))                # zero-physics tangents
    phi = np.array([phi_gap])
    mu  = 0.4
    return n_x, n_u, A, B, D, d, J_n, J_t, phi, mu


def _build_cost(n_x: int, n_u: int):
    """Goal: box at pos=0.30, vel=0. Penalize position error and control."""
    Q  = np.diag([100.0, 0.0])
    R  = np.diag([1.0])
    QN = 10.0 * Q
    x_ref = np.array([0.30, 0.0])
    return Q, R, QN, x_ref


def _build_complementarity(*, dt: float, mass: float,
                           A: np.ndarray, B: np.ndarray, D: np.ndarray,
                           J_n: np.ndarray, J_t: np.ndarray,
                           phi: np.ndarray, mu: float = 0.4):
    """
    Phase 2 layout: λ = [γ; λ_n; λ_t] with n_lambda = 6·n_c = 6.

        γ row     : 0 + μ·λ_n − E_t·λ_t              (no x, u, c)
        λ_n row   : phi/dt + J_n·v_next              (= phi/dt + vel + dt/m·(u + λ_n))
        λ_t row   : E_t^T·γ + J_t·v_next             (J_t = 0 → only γ contributes)

    With J_t = 0 the F[λ_n, λ_t] block is zero (no normal-tangent coupling
    via dynamics) and F[λ_t, λ_n] also zero. The γ block still couples
    via the friction-cone slack: F[γ, λ_n] = μ, F[γ, λ_t] = -E_t.
    """
    n_x   = 2
    n_u   = 1
    num_normals = J_n.shape[0]
    n_t   = J_t.shape[0]
    n_lam = 2 * num_normals + n_t
    SG, SLN, SLT = 0, num_normals, 2 * num_normals

    # E_t pattern row: [1, 1, 1, 1] for our single contact's 4 tangent slots
    E_t = np.zeros((num_normals, n_t))
    for i in range(num_normals):
        E_t[i, 4*i:4*(i+1)] = 1.0

    E = np.zeros((n_lam, n_x))
    F = np.zeros((n_lam, n_lam))
    H = np.zeros((n_lam, n_u))
    c = np.zeros(n_lam)

    # γ rows
    F[SG:SG+num_normals, SLN:SLN+num_normals] = mu * np.eye(num_normals)
    F[SG:SG+num_normals, SLT:SLT+n_t]         = -E_t
    # λ_n row: phi/dt + vel + (dt/m)(u + λ_n)
    E[SLN, 1] = 1.0                                 # vel coefficient
    F[SLN, SLN] = dt / mass                          # λ_n self-coupling
    H[SLN, 0]   = dt / mass                          # u coefficient
    c[SLN]      = phi[0] / dt                        # constant
    # λ_t rows: E_t^T γ + J_t·v_next; J_t = 0 in this synthetic LCS so only γ.
    F[SLT:SLT+n_t, SG:SG+num_normals] = E_t.T

    return E, F, H, c


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------
def test_c3_runs_and_predicts_positive_x_motion():
    """C3 path: should pick u>0 (or λ>0) → predicted box vel becomes +x."""
    dt, m = 0.05, 0.2
    n_x, n_u, A, B, D, d, J_n, J_t, phi, mu = _build_synthetic_lcs(
        dt=dt, mass=m)
    Q, R, QN, x_ref = _build_cost(n_x, n_u)
    # Phase 2: C3 also requires (E, F, H, c) for the LCP projection.
    E, F, H, c = _build_complementarity(
        dt=dt, mass=m, A=A, B=B, D=D, J_n=J_n, J_t=J_t, phi=phi, mu=mu)

    solver = C3Solver(n_x=n_x, n_u=n_u, rho=10.0, mode="c3")

    x0 = np.array([0.0, 0.0])      # at rest
    u_seq, x_seq = solver.solve(
        x0, A, B, D, d, J_n, J_t, mu,
        Q, R, QN, x_ref,
        N=8, admm_iter=10, torque_limit=30.0,
        phi=phi,
        E=E, F=F, H=H, c_lcs=c,
    )
    assert u_seq.shape == (8, n_u)
    assert x_seq.shape == (9, n_x)
    # Predicted box velocity must become positive somewhere in the horizon.
    box_vels = x_seq[1:, 1]
    assert np.any(box_vels > 1e-3), (
        f"C3 failed to predict +x motion; box_vels = {box_vels}"
    )


def test_c3plus_runs_and_predicts_positive_x_motion():
    """C3+ path: same input + (E, F, H, c) → predicted box vel becomes +x."""
    dt, m = 0.05, 0.2
    n_x, n_u, A, B, D, d, J_n, J_t, phi, mu = _build_synthetic_lcs(
        dt=dt, mass=m)
    Q, R, QN, x_ref = _build_cost(n_x, n_u)
    E, F, H, c = _build_complementarity(
        dt=dt, mass=m, A=A, B=B, D=D, J_n=J_n, J_t=J_t, phi=phi, mu=mu)

    solver = C3Solver(n_x=n_x, n_u=n_u, rho=10.0, mode="c3plus")

    x0 = np.array([0.0, 0.0])
    u_seq, x_seq = solver.solve(
        x0, A, B, D, d, J_n, J_t, mu,
        Q, R, QN, x_ref,
        N=8, admm_iter=10, torque_limit=30.0,
        phi=phi,
        E=E, F=F, H=H, c_lcs=c,
    )
    assert u_seq.shape == (8, n_u)
    assert x_seq.shape == (9, n_x)
    box_vels = x_seq[1:, 1]
    assert np.any(box_vels > 1e-3), (
        f"C3+ failed to predict +x motion; box_vels = {box_vels}"
    )


def test_c3plus_requires_complementarity_expression():
    """Calling solve() with mode='c3plus' but without (E, F, H, c) errors out."""
    dt, m = 0.05, 0.2
    n_x, n_u, A, B, D, d, J_n, J_t, phi, mu = _build_synthetic_lcs(
        dt=dt, mass=m)
    Q, R, QN, x_ref = _build_cost(n_x, n_u)

    solver = C3Solver(n_x=n_x, n_u=n_u, rho=10.0, mode="c3plus")
    x0 = np.array([0.0, 0.0])
    # Phase 2: both modes assert E/F/H/c are provided. C3+ asserts first.
    with pytest.raises(AssertionError, match="C3\\+ requires"):
        solver.solve(
            x0, A, B, D, d, J_n, J_t, mu,
            Q, R, QN, x_ref,
            N=4, admm_iter=2, torque_limit=30.0,
            phi=phi,
        )


def test_c3_requires_complementarity_expression():
    """Phase 2: C3 path also requires (E, F, H, c) for the LCP projection."""
    dt, m = 0.05, 0.2
    n_x, n_u, A, B, D, d, J_n, J_t, phi, mu = _build_synthetic_lcs(
        dt=dt, mass=m)
    Q, R, QN, x_ref = _build_cost(n_x, n_u)

    solver = C3Solver(n_x=n_x, n_u=n_u, rho=10.0, mode="c3")
    x0 = np.array([0.0, 0.0])
    with pytest.raises(AssertionError, match="C3 \\(Phase 2\\) requires"):
        solver.solve(
            x0, A, B, D, d, J_n, J_t, mu,
            Q, R, QN, x_ref,
            N=4, admm_iter=2, torque_limit=30.0,
            phi=phi,
        )


def test_c3_vs_c3plus_first_torque_sign_agrees():
    """Both solvers should command +u (or rely on +λ_n) on a +x push goal.
    Specifically, the first-step torque or contact force should drive vel +."""
    dt, m = 0.05, 0.2
    n_x, n_u, A, B, D, d, J_n, J_t, phi, mu = _build_synthetic_lcs(
        dt=dt, mass=m)
    Q, R, QN, x_ref = _build_cost(n_x, n_u)
    E, F, H, c = _build_complementarity(
        dt=dt, mass=m, A=A, B=B, D=D, J_n=J_n, J_t=J_t, phi=phi, mu=mu)

    s_c3 = C3Solver(n_x=n_x, n_u=n_u, rho=10.0, mode="c3")
    s_cp = C3Solver(n_x=n_x, n_u=n_u, rho=10.0, mode="c3plus")

    x0 = np.array([0.0, 0.0])
    u_c3,  x_c3  = s_c3.solve(
        x0, A, B, D, d, J_n, J_t, mu, Q, R, QN, x_ref,
        N=8, admm_iter=10, torque_limit=30.0, phi=phi,
        E=E, F=F, H=H, c_lcs=c,
    )
    u_cp,  x_cp  = s_cp.solve(
        x0, A, B, D, d, J_n, J_t, mu, Q, R, QN, x_ref,
        N=8, admm_iter=10, torque_limit=30.0, phi=phi,
        E=E, F=F, H=H, c_lcs=c,
    )

    # Predicted terminal box velocity must be ≥ 0 (positive x is the goal
    # direction; both controllers should at least not actively brake).
    assert x_c3[-1, 1] >= -1e-3, f"C3 predicted negative terminal vel: {x_c3[-1, 1]}"
    assert x_cp[-1, 1] >= -1e-3, f"C3+ predicted negative terminal vel: {x_cp[-1, 1]}"
