"""
Phase-2 acceptance test for the Stewart-Trinkle (E, F, H, c) assembly.

Verifies the structural blocks of the slack expression returned by
LCSFormulator.linearize_discrete against hand-computed references for a
single-contact planar pushing scenario.

λ ordering: [γ (n_c); λ_n (n_c); λ_t (4·n_c)]   — total 6·n_c

Block structure (Aydinoglu eq. 9, with v_{k+1} substituted):

    γ rows   :  F[γ, λ_n] = μ·I,    F[γ, λ_t] = -E_t
                E[γ, :] = 0, H[γ, :] = 0, c[γ] = 0
    λ_n rows :  E[λ_n, q] = dt·J_n·J_q
                E[λ_n, v] = J_n + dt·J_n·J_v
                F[λ_n, λ_n] = dt·J_n·M⁻¹·J_n^T
                F[λ_n, λ_t] = dt·J_n·M⁻¹·J_t^T
                H[λ_n] = dt·J_n·J_u
                c[λ_n] = phi/dt + dt·J_n·d_v_offset
    λ_t rows :  E[λ_t, q] = dt·J_t·J_q,  E[λ_t, v] = J_t + dt·J_t·J_v
                F[λ_t, γ] = E_t^T
                F[λ_t, λ_n] = dt·J_t·M⁻¹·J_n^T
                F[λ_t, λ_t] = dt·J_t·M⁻¹·J_t^T
                H[λ_t] = dt·J_t·J_u
                c[λ_t] = dt·J_t·d_v_offset
"""
import os
import sys
import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

pytest.importorskip("pydrake", reason="Drake required")

import pydrake.all as ad

from sim.env_builder import build_environment, INITIAL_ARM_Q
from control.lcs_formulator import LCSFormulator


def _build():
    cfg = {
        "object_type": "box", "size": [0.1, 0.1, 0.1],
        "mass": 0.2, "friction": 0.4, "color_rgba": [0.6, 0.3, 0.1, 1.0],
        "init_xyz": [0, 0, 0.05], "goal_xy": [0.3, 0],
        "link_name": "box_link", "cost": {},
    }
    diagram, plant, panda_model, _, _, plant_ad, ctx_ad = build_environment(cfg)
    diag_ctx  = diagram.CreateDefaultContext()
    plant_ctx = plant.GetMyMutableContextFromRoot(diag_ctx)
    plant.SetPositions(plant_ctx, panda_model, INITIAL_ARM_Q)
    obj_body = plant.GetBodyByName("box_link")
    plant.SetFreeBodyPose(
        plant_ctx, obj_body,
        ad.RigidTransform(ad.RotationMatrix(), [0, 0, 0.05]),
    )
    f = LCSFormulator(plant, mu=0.4, obj_body=obj_body,
                      plant_ad=plant_ad, context_ad=ctx_ad)
    return plant, plant_ctx, f, diag_ctx


def test_efhc_shapes():
    """λ has dimension 6·n_c; E/F/H/c shapes match."""
    _, plant_ctx, f, _ = _build()
    A, B, D, d, E, F, H, c, J_n, J_t, phi, mu = \
        f.linearize_discrete(plant_ctx, dt=0.05)
    n_c = J_n.shape[0]
    n_x = f.n_q + f.n_v
    n_u = f.n_u
    n_lam = 6 * n_c
    assert E.shape == (n_lam, n_x)
    assert F.shape == (n_lam, n_lam)
    assert H.shape == (n_lam, n_u)
    assert c.shape == (n_lam,)
    assert D.shape == (n_x, n_lam)


def test_gamma_row_block_matches_hand_assembly():
    """γ rows: F[γ, λ_n] = μ·I, F[γ, λ_t] = -E_t, all other slots zero."""
    _, plant_ctx, f, _ = _build()
    _, _, _, _, E, F, H, c, J_n, J_t, phi, mu = \
        f.linearize_discrete(plant_ctx, dt=0.05)
    n_c = J_n.shape[0]
    n_t = J_t.shape[0]
    SG, SLN, SLT = 0, n_c, 2 * n_c

    # E_t reference
    E_t_ref = np.zeros((n_c, n_t))
    for i in range(n_c):
        E_t_ref[i, 4*i:4*(i+1)] = 1.0

    # E[γ,:] = 0
    assert np.allclose(E[SG:SG+n_c, :], 0.0)
    # H[γ,:] = 0
    assert np.allclose(H[SG:SG+n_c, :], 0.0)
    # c[γ] = 0
    assert np.allclose(c[SG:SG+n_c], 0.0)
    # F[γ, γ] = 0
    assert np.allclose(F[SG:SG+n_c, SG:SG+n_c], 0.0)
    # F[γ, λ_n] = μ·I
    assert np.allclose(F[SG:SG+n_c, SLN:SLN+n_c], mu * np.eye(n_c))
    # F[γ, λ_t] = -E_t
    assert np.allclose(F[SG:SG+n_c, SLT:SLT+n_t], -E_t_ref)


def test_lambda_n_rows_match_hand_assembly():
    """λ_n rows: E v-block, F λ_n / λ_t blocks, H, and c match the
    autodiff-derived reference exactly."""
    _, plant_ctx, f, _ = _build()
    A, B, D, d, E, F, H, c, J_n, J_t, phi, mu = \
        f.linearize_discrete(plant_ctx, dt=0.05)
    n_c = J_n.shape[0]
    n_t = J_t.shape[0]
    n_q, n_v, n_u = f.n_q, f.n_v, f.n_u
    SLN, SLT = n_c, 2 * n_c
    dt = 0.05

    # Re-derive J_q, J_v, J_u, d_v_offset locally.
    M, _Cv, _tg, _B, J_f, f_eval = f.extract_dynamics_with_jacobian(
        plant_ctx, np.zeros(n_u))
    M_inv = np.linalg.inv(M)
    J_q = J_f[:, :n_q]
    J_v = J_f[:, n_q:n_q + n_v]
    J_u = J_f[:, n_q + n_v:]
    q_star = f.plant.GetPositions(plant_ctx)
    v_star = f.plant.GetVelocities(plant_ctx)
    d_v_offset = f_eval - (J_q @ q_star + J_v @ v_star + J_u @ np.zeros(n_u))
    Minv_JnT = M_inv @ J_n.T
    Minv_JtT = M_inv @ J_t.T

    # E[λ_n, q] = dt·J_n·J_q
    assert np.allclose(E[SLN:SLN+n_c, :n_q], dt * (J_n @ J_q), atol=1e-12)
    # E[λ_n, v] = J_n + dt·J_n·J_v
    assert np.allclose(E[SLN:SLN+n_c, n_q:n_q+n_v],
                       J_n + dt * (J_n @ J_v), atol=1e-12)
    # F[λ_n, γ] = 0
    assert np.allclose(F[SLN:SLN+n_c, :n_c], 0.0)
    # F[λ_n, λ_n] = dt·J_n·Minv·J_n^T
    assert np.allclose(F[SLN:SLN+n_c, SLN:SLN+n_c],
                       dt * (J_n @ Minv_JnT), atol=1e-12)
    # F[λ_n, λ_t] = dt·J_n·Minv·J_t^T
    assert np.allclose(F[SLN:SLN+n_c, SLT:SLT+n_t],
                       dt * (J_n @ Minv_JtT), atol=1e-12)
    # H[λ_n] = dt·J_n·J_u
    assert np.allclose(H[SLN:SLN+n_c, :], dt * (J_n @ J_u), atol=1e-12)
    # c[λ_n] = phi/dt + dt·J_n·d_v_offset
    assert np.allclose(c[SLN:SLN+n_c],
                       phi / dt + dt * (J_n @ d_v_offset), atol=1e-12)


def test_lambda_t_rows_match_hand_assembly():
    """λ_t rows: F[λ_t, γ] = E_t^T (couples slack to tangents), and the
    F[λ_t, λ_*] / E / H / c blocks match the autodiff reference."""
    _, plant_ctx, f, _ = _build()
    A, B, D, d, E, F, H, c, J_n, J_t, phi, mu = \
        f.linearize_discrete(plant_ctx, dt=0.05)
    n_c = J_n.shape[0]
    n_t = J_t.shape[0]
    n_q, n_v, n_u = f.n_q, f.n_v, f.n_u
    SG, SLN, SLT = 0, n_c, 2 * n_c
    dt = 0.05

    M, _Cv, _tg, _B, J_f, f_eval = f.extract_dynamics_with_jacobian(
        plant_ctx, np.zeros(n_u))
    M_inv = np.linalg.inv(M)
    J_q = J_f[:, :n_q]
    J_v = J_f[:, n_q:n_q + n_v]
    J_u = J_f[:, n_q + n_v:]
    q_star = f.plant.GetPositions(plant_ctx)
    v_star = f.plant.GetVelocities(plant_ctx)
    d_v_offset = f_eval - (J_q @ q_star + J_v @ v_star + J_u @ np.zeros(n_u))
    Minv_JnT = M_inv @ J_n.T
    Minv_JtT = M_inv @ J_t.T

    E_t_ref = np.zeros((n_c, n_t))
    for i in range(n_c):
        E_t_ref[i, 4*i:4*(i+1)] = 1.0

    # E[λ_t, q] = dt·J_t·J_q
    assert np.allclose(E[SLT:SLT+n_t, :n_q], dt * (J_t @ J_q), atol=1e-12)
    # E[λ_t, v] = J_t + dt·J_t·J_v
    assert np.allclose(E[SLT:SLT+n_t, n_q:n_q+n_v],
                       J_t + dt * (J_t @ J_v), atol=1e-12)
    # F[λ_t, γ] = E_t^T
    assert np.allclose(F[SLT:SLT+n_t, SG:SG+n_c], E_t_ref.T, atol=1e-14)
    # F[λ_t, λ_n] = dt·J_t·Minv·J_n^T
    assert np.allclose(F[SLT:SLT+n_t, SLN:SLN+n_c],
                       dt * (J_t @ Minv_JnT), atol=1e-12)
    # F[λ_t, λ_t] = dt·J_t·Minv·J_t^T
    assert np.allclose(F[SLT:SLT+n_t, SLT:SLT+n_t],
                       dt * (J_t @ Minv_JtT), atol=1e-12)
    # H[λ_t] = dt·J_t·J_u
    assert np.allclose(H[SLT:SLT+n_t, :], dt * (J_t @ J_u), atol=1e-12)
    # c[λ_t] = dt·J_t·d_v_offset
    assert np.allclose(c[SLT:SLT+n_t], dt * (J_t @ d_v_offset), atol=1e-12)


def test_D_gamma_columns_zero():
    """γ does not enter dynamics — D's first n_c columns are exactly zero."""
    _, plant_ctx, f, _ = _build()
    _, _, D, _, _, _, _, _, J_n, _, _, _ = \
        f.linearize_discrete(plant_ctx, dt=0.05)
    n_c = J_n.shape[0]
    assert np.linalg.norm(D[:, :n_c]) == 0.0


def test_no_contacts_returns_zero_lcp_blocks():
    """When no contacts are detected (gap > threshold), n_λ = 0 and the
    LCP blocks have zero rows."""
    cfg = {
        "object_type": "box", "size": [0.1, 0.1, 0.1],
        "mass": 0.2, "friction": 0.4, "color_rgba": [0.6, 0.3, 0.1, 1.0],
        "init_xyz": [0, 0, 0.05], "goal_xy": [0.3, 0],
        "link_name": "box_link", "cost": {},
    }
    diagram, plant, panda_model, _, _, plant_ad, ctx_ad = build_environment(cfg)
    diag_ctx  = diagram.CreateDefaultContext()
    plant_ctx = plant.GetMyMutableContextFromRoot(diag_ctx)
    obj_body = plant.GetBodyByName("box_link")
    # Move box far from EE so contact filter finds nothing.
    plant.SetFreeBodyPose(
        plant_ctx, obj_body,
        ad.RigidTransform(ad.RotationMatrix(), [2.0, 2.0, 0.05]),
    )
    plant.SetPositions(plant_ctx, panda_model, INITIAL_ARM_Q)
    f = LCSFormulator(plant, mu=0.4, obj_body=obj_body,
                      plant_ad=plant_ad, context_ad=ctx_ad)
    A, B, D, d, E, F, H, c, J_n, J_t, phi, mu = \
        f.linearize_discrete(plant_ctx, dt=0.05)
    assert J_n.shape[0] == 0
    assert E.shape == (0, A.shape[0])
    assert F.shape == (0, 0)
    assert H.shape == (0, B.shape[1])
    assert c.shape == (0,)
    assert D.shape == (A.shape[0], 0)
