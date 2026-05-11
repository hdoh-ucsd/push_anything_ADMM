"""
Phase-1 acceptance tests (Aydinoglu 2024 eq. 8).

Verifies the first-order linearization in
LCSFormulator.extract_dynamics_with_jacobian and the matrices it feeds
into LCSFormulator.linearize_discrete.

Requires Drake (uses build_environment + the autodiff plant).
"""
import os
import sys
import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

pytest.importorskip("pydrake", reason="Drake required for J_f autodiff tests")

import pydrake.all as ad

from sim.env_builder import build_environment, INITIAL_ARM_Q
from control.lcs_formulator import LCSFormulator


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _build_env():
    """Build a minimal pushing-task environment for J_f tests."""
    cfg = {
        "object_type": "box",
        "size": [0.1, 0.1, 0.1],
        "mass": 0.2,
        "friction": 0.4,
        "color_rgba": [0.6, 0.3, 0.1, 1.0],
        "init_xyz": [0.0, 0.0, 0.05],
        "goal_xy":  [0.3, 0.0],
        "link_name": "box_link",
        "cost": {},
    }
    diagram, plant, panda_model, _obj_model, _meshcat, plant_ad, context_ad = \
        build_environment(cfg)
    diag_ctx  = diagram.CreateDefaultContext()
    plant_ctx = plant.GetMyMutableContextFromRoot(diag_ctx)
    plant.SetPositions(plant_ctx, panda_model, INITIAL_ARM_Q)
    obj_body = plant.GetBodyByName("box_link")
    plant.SetFreeBodyPose(
        plant_ctx, obj_body,
        ad.RigidTransform(ad.RotationMatrix(), [0.0, 0.0, 0.05]),
    )
    formulator = LCSFormulator(
        plant, mu=0.4, obj_body=obj_body,
        plant_ad=plant_ad, context_ad=context_ad,
    )
    return plant, plant_ctx, formulator, diag_ctx  # diag_ctx kept alive


def _f_zeroth(plant, plant_ctx, q, v, u):
    """Reference zeroth-order forward dynamics: f(q,v,u) = M^-1 (B u - Cv + τ_g)."""
    plant.SetPositions(plant_ctx, q)
    plant.SetVelocities(plant_ctx, v)
    M     = plant.CalcMassMatrixViaInverseDynamics(plant_ctx)
    Cv    = plant.CalcBiasTerm(plant_ctx)
    tau_g = plant.CalcGravityGeneralizedForces(plant_ctx)
    B     = plant.MakeActuationMatrix()
    return np.linalg.solve(M, B @ u - Cv + tau_g)


# ---------------------------------------------------------------------------
# Test 1: autodiff J_f matches central finite differences
# ---------------------------------------------------------------------------

def test_jacobian_matches_finite_differences():
    plant, plant_ctx, formulator, _diag = _build_env()
    n_q, n_v, n_u = formulator.n_q, formulator.n_v, formulator.n_u
    n_dec = n_q + n_v + n_u

    q_star = plant.GetPositions(plant_ctx).copy()
    v_star = plant.GetVelocities(plant_ctx).copy()
    rng = np.random.default_rng(7)
    u_star = rng.normal(size=n_u) * 0.3

    _M, _Cv, _tg, _B, J_f, f_eval = \
        formulator.extract_dynamics_with_jacobian(plant_ctx, u_star)

    # Central FD reference
    eps = 1e-6
    J_fd = np.zeros((n_v, n_dec))
    decvar = np.concatenate([q_star, v_star, u_star])
    for k in range(n_dec):
        e = np.zeros(n_dec); e[k] = eps
        plus  = decvar + e; minus = decvar - e
        f_p = _f_zeroth(plant, plant_ctx,
                        plus[:n_q], plus[n_q:n_q+n_v], plus[n_q+n_v:])
        f_m = _f_zeroth(plant, plant_ctx,
                        minus[:n_q], minus[n_q:n_q+n_v], minus[n_q+n_v:])
        J_fd[:, k] = (f_p - f_m) / (2 * eps)

    err = float(np.max(np.abs(J_f - J_fd)))
    print(f"max |J_f_autodiff - J_f_FD| = {err:.3e}")
    assert err < 1e-4, f"J_f deviates from central-FD by {err:.3e} > 1e-4"


# ---------------------------------------------------------------------------
# Test 2: f_eval at the linearization point matches old zeroth-order f
# ---------------------------------------------------------------------------

def test_f_eval_matches_zeroth_order():
    plant, plant_ctx, formulator, _diag = _build_env()
    n_u = formulator.n_u

    q = plant.GetPositions(plant_ctx).copy()
    v = plant.GetVelocities(plant_ctx).copy()

    for u in (np.zeros(n_u),
              np.full(n_u, 0.5),
              np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])):
        _M, _Cv, _tg, _B, _Jf, f_eval = \
            formulator.extract_dynamics_with_jacobian(plant_ctx, u)
        f_zeroth = _f_zeroth(plant, plant_ctx, q, v, u)
        err = float(np.max(np.abs(f_eval - f_zeroth)))
        assert err < 1e-10, f"f_eval mismatch at u={u}: {err:.3e}"


# ---------------------------------------------------------------------------
# Test 3: predicted next-state with new linearization is exact at lin point
# ---------------------------------------------------------------------------

def test_linearization_is_exact_at_lin_point():
    """At (q*, v*, u*) with λ=0, the discrete LCS prediction should reproduce
    the explicit forward-Euler step *exactly* — d_v_offset cancels J_f·[q*;v*;u*]
    so x_{k+1} = forward_euler(q*, v*, u*).
    """
    plant, plant_ctx, formulator, _diag = _build_env()
    n_q = formulator.n_q
    n_u = formulator.n_u

    q_star = plant.GetPositions(plant_ctx).copy()
    v_star = plant.GetVelocities(plant_ctx).copy()
    u_star = np.array([0.05, -0.1, 0.0, 0.2, -0.05, 0.0, 0.0])
    dt     = 0.05

    A, B_ctrl, D, d_vec, _E, _F, _H, _c, J_n, J_t, phi, mu = \
        formulator.linearize_discrete(plant_ctx, dt, u_lin=u_star)

    x_star = np.concatenate([q_star, v_star])
    x_next_lin = A @ x_star + B_ctrl @ u_star + d_vec  # λ = 0

    # Reference: explicit forward Euler of true f at (q*, v*, u*).
    f_ref = _f_zeroth(plant, plant_ctx, q_star, v_star, u_star)
    # N(q) at q_star
    plant.SetPositions(plant_ctx, q_star)
    plant.SetVelocities(plant_ctx, v_star)
    n_v = formulator.n_v
    N_mat = np.zeros((n_q, n_v))
    for i in range(n_v):
        e = np.zeros(n_v); e[i] = 1.0
        N_mat[:, i] = plant.MapVelocityToQDot(plant_ctx, e)
    v_next_ref = v_star + dt * f_ref
    q_next_ref = q_star + dt * N_mat @ v_next_ref
    x_next_ref = np.concatenate([q_next_ref, v_next_ref])

    err = float(np.max(np.abs(x_next_lin - x_next_ref)))
    print(f"max |x_next_lin - x_next_FE| = {err:.3e}")
    # Allow some slack: linearization is exact in the math, but float roundoff
    # in autodiff vs hand-coded f introduces ~1e-12.
    assert err < 1e-9, f"linearization at lin-point not exact: err={err:.3e}"


# ---------------------------------------------------------------------------
# Test 4: new linearization is at least as accurate as old at off-lin points
# ---------------------------------------------------------------------------

def test_linearization_better_than_zeroth_order_off_lin():
    """At a state OFF the linearization point, the first-order linearization
    should be at least as accurate as the zeroth-order linearization in
    predicting the true forward dynamics.
    """
    plant, plant_ctx, formulator, _diag = _build_env()
    n_q, n_v, n_u = formulator.n_q, formulator.n_v, formulator.n_u

    q_star = plant.GetPositions(plant_ctx).copy()
    v_star = plant.GetVelocities(plant_ctx).copy()
    u_star = np.zeros(n_u)
    dt     = 0.05

    rng = np.random.default_rng(11)
    # Perturb q (skipping floating-base quaternion at indices 7..10) and v.
    dq = np.zeros(n_q); dq[:n_u] = rng.normal(size=n_u) * 0.03
    dv = np.zeros(n_v); dv[:n_u] = rng.normal(size=n_u) * 0.05
    du = rng.normal(size=n_u) * 0.05
    q_off = q_star + dq
    v_off = v_star + dv
    u_off = u_star + du

    # New linearization at (q_star, v_star, u_star)
    A_new, B_new, D_new, d_new, *_ = \
        formulator.linearize_discrete(plant_ctx, dt, u_lin=u_star)
    # D's column count is now 6·n_c — independent of x_off, so it drops out
    # when λ = 0 (which is the case for this off-lin-point evaluation).
    x_off    = np.concatenate([q_off, v_off])
    x_next_new = A_new @ x_off + B_new @ u_off + d_new

    # Truth: explicit forward-Euler at (q_off, v_off, u_off)
    f_true = _f_zeroth(plant, plant_ctx, q_off, v_off, u_off)
    plant.SetPositions(plant_ctx, q_off)
    plant.SetVelocities(plant_ctx, v_off)
    N_mat = np.zeros((n_q, n_v))
    for i in range(n_v):
        e = np.zeros(n_v); e[i] = 1.0
        N_mat[:, i] = plant.MapVelocityToQDot(plant_ctx, e)
    v_next_true = v_off + dt * f_true
    q_next_true = q_off + dt * N_mat @ v_next_true
    x_next_true = np.concatenate([q_next_true, v_next_true])

    # Old (zeroth-order) reference: A_old @ x_off + B_old @ u_off + d_old,
    # with A_old having no J_q/J_v cross-terms and d_old using
    # f(q*, v*, 0).
    plant.SetPositions(plant_ctx, q_star)
    plant.SetVelocities(plant_ctx, v_star)
    M     = plant.CalcMassMatrixViaInverseDynamics(plant_ctx)
    Cv    = plant.CalcBiasTerm(plant_ctx)
    tau_g = plant.CalcGravityGeneralizedForces(plant_ctx)
    B     = plant.MakeActuationMatrix()
    M_inv = np.linalg.inv(M)
    N_star = np.zeros((n_q, n_v))
    for i in range(n_v):
        e = np.zeros(n_v); e[i] = 1.0
        N_star[:, i] = plant.MapVelocityToQDot(plant_ctx, e)
    A_old = np.zeros_like(A_new)
    A_old[:n_q, :n_q] = np.eye(n_q)
    A_old[:n_q, n_q:] = dt * N_star
    A_old[n_q:, n_q:] = np.eye(n_v)
    B_old = np.zeros_like(B_new)
    B_old[n_q:] = dt * (M_inv @ B)
    d_old = np.zeros_like(d_new)
    d_old[n_q:] = dt * (M_inv @ (tau_g - Cv))
    x_next_old = A_old @ x_off + B_old @ u_off + d_old

    err_old = float(np.linalg.norm(x_next_old - x_next_true))
    err_new = float(np.linalg.norm(x_next_new - x_next_true))
    print(f"err_old (zeroth) = {err_old:.6e}")
    print(f"err_new (first)  = {err_new:.6e}")
    # First-order should dominate; allow tiny slack so test isn't flaky.
    assert err_new <= err_old + 1e-12, (
        f"first-order linearization is WORSE than zeroth-order: "
        f"err_new={err_new:.3e} > err_old={err_old:.3e}"
    )
