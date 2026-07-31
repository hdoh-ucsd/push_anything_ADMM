"""EE-space LCS affine consistency in u_lin — p146 walk root cause.

The complementarity constant c is assembled from gap values evaluated at
u = 0 (d_v_ee_offset = 0, "purely linear in u"), so the affine identity

    eta(x, u) = E x + H u + c

must satisfy  eta(x*, u_lin) = s * (phi/dt + J_row (A x* + B u_lin + d)_v)
for ANY u_lin.  The port subtracted H @ u_star from c (lcs_formulator
Anitescu :2158 and ST :2060) even though the value expression never
included u*'s contribution — shifting every EE-coupled row by -H@u*.
Ground rows (H ~ 0) and surrogate solves (u_lin = 0) were unaffected,
which is why sample ranking looked sane while every full c3 solve ran
with a corrupted EE-BOX gap row (p146: gap-rate inflated ~1.0 m/s,
2.5x the whole phi/dt term, fed back through _last_u every tick).
"""
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

pytest.importorskip("pydrake", reason="Drake required")

import yaml
import pydrake.all as ad

from sim.env_builder import build_environment
from control.lcs_formulator import LCSFormulator


@pytest.fixture(scope="module")
def push_t_env():
    cfg = yaml.safe_load(
        open(os.path.join(_ROOT, "config", "tasks.yaml")))["tasks"]["push_t"]
    diagram, plant, panda_model, object_model, _mc, plant_ad, ctx_ad, _vw = \
        build_environment(cfg, time_step=0.001)
    diag_ctx = diagram.CreateDefaultContext()
    plant_ctx = plant.GetMyMutableContextFromRoot(diag_ctx)
    obj_body = plant.GetBodyByName(cfg["link_name"])
    plant.SetFreeBodyPose(
        plant_ctx, obj_body,
        ad.RigidTransform(ad.RotationMatrix(), [0.5, 0.0, 0.02]))
    form = LCSFormulator(
        plant, mu=cfg["friction"], obj_body=obj_body,
        plant_ad=plant_ad, context_ad=ctx_ad,
        object_shape="tshape",
        mu_per_pair_type=cfg.get("mu_per_pair_type"))
    return plant, plant_ctx, obj_body, form


def _row_deviations(plant, plant_ctx, obj_body, form, u_lin):
    dt = 0.1
    A, B, D, d, E, F, H, c, J_n, J_t, phi, mu = \
        form.linearize_discrete_ee_space(plant_ctx, dt, u_lin=u_lin)
    n_c = phi.shape[0]
    mu_arr = np.broadcast_to(np.asarray(mu), (n_c,))
    q = plant.GetPositions(plant_ctx)
    v = plant.GetVelocities(plant_ctx)
    BOX_Q = obj_body.floating_positions_start()
    BOX_V = obj_body.floating_velocities_start_in_v()
    ee_body = plant.GetBodyByName("pusher")
    p_ee = plant.CalcPointsPositions(
        plant_ctx, ee_body.body_frame(), np.zeros((3, 1)),
        plant.world_frame()).flatten()
    Jv = plant.CalcJacobianTranslationalVelocity(
        plant_ctx, ad.JacobianWrtVariable.kV, ee_body.body_frame(),
        np.zeros(3), plant.world_frame(), plant.world_frame())
    v_ee = Jv @ v
    x_star = np.concatenate(
        [q[BOX_Q:BOX_Q + 7], p_ee, v[BOX_V:BOX_V + 6], v_ee])
    u_vec = np.zeros(3) if u_lin is None else np.asarray(u_lin, float)
    v_next = (A @ x_star + B @ u_vec + d)[10:19]
    model = E @ x_star + H @ u_vec + c
    phys = np.zeros_like(model)
    n_edges = model.shape[0] // n_c
    for p in range(n_c):
        for e in range(n_edges):
            r = n_edges * p + e
            J_row = J_n[p] + mu_arr[p] * J_t[r] if n_edges == 4 else None
            if J_row is None:
                pytest.skip("unexpected row layout")
            phys[r] = phi[p] / dt + J_row @ v_next
    # scale factor from the rows themselves (LCS scaling is uniform)
    mask = np.abs(phys) > 1e-6
    s = np.median(model[mask] / phys[mask])
    return model - s * phys, s


def test_consistent_at_zero_ulin(push_t_env):
    dev, s = _row_deviations(*push_t_env, u_lin=None)
    assert np.max(np.abs(dev)) < 1e-6 * max(1.0, abs(s)), dev


def test_consistent_at_nonzero_ulin(push_t_env):
    """The p146 defect: with u_lin != 0 the EE-BOX rows deviated by -H@u*."""
    dev, s = _row_deviations(*push_t_env, u_lin=[5.0, 10.0, -2.0])
    assert np.max(np.abs(dev)) < 1e-6 * max(1.0, abs(s)), dev
