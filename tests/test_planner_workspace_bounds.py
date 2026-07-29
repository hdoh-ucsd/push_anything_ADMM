"""Planner-side workspace state constraints — reference cc:995-1025.

The reference adds hard linear STATE constraints to EVERY per-sample C3
object (sampling_based_c3_controller.cc:995-1025): for each workspace
half-space, A·x ∈ [lb − workspace_margins, ub + workspace_margins] on the
EE position slots AND the object position slots (push_t values:
sampling_c3_options.yaml:26-30 — x [0.15,0.9], y [-0.6,0.6], z [-0.01,0.3]
their frame, margins 0.02). The port carried over only the adjacent
EE-velocity rows (cc:1027-1034, ee_velocity_bounds); the position rows were
missing — the planner was free to plan EE excursions anywhere (p140/p141
phantom stints walked the EE toward the r=0.25 workspace abort).

Port shape: C3Solver.state_position_bounds = [(state_idx, lo, hi), ...],
applied per knot + terminal in _solve_c3plus, mirroring the
ee_velocity_bounds pattern. Reuses the synthetic 1D LCS from
test_c3plus_vs_c3_smoke.py.
"""
import numpy as np
import pytest

pytest.importorskip("pydrake", reason="C3Solver imports pydrake")

from control.admm_solver import C3Solver
from tests.test_c3plus_vs_c3_smoke import (
    _build_synthetic_lcs,
    _build_cost,
    _build_complementarity,
)


def _solve(bounds):
    dt, mass = 0.05, 0.2
    n_x, n_u, A, B, D, d, J_n, J_t, phi, mu = _build_synthetic_lcs(
        dt=dt, mass=mass)
    Q, R, QN, x_ref = _build_cost(n_x, n_u)
    E, F, H, c = _build_complementarity(
        dt=dt, mass=mass, A=A, B=B, D=D, J_n=J_n, J_t=J_t, phi=phi, mu=mu)
    solver = C3Solver(n_x=n_x, n_u=n_u, rho=1.0, mode="c3plus")
    if bounds is not None:
        solver.state_position_bounds = bounds
    x0 = np.zeros(n_x)
    u_seq, x_seq = solver.solve(
        x0, A, B, D, d, J_n, J_t, mu,
        Q, R, QN, x_ref,
        N=8, admm_iter=5, torque_limit=30.0,
        phi=phi,
        E=E, F=F, H=H, c_lcs=c,
    )
    return x_seq


def test_unbounded_plan_crosses_the_limit():
    """Sanity: without bounds the plan drives pos toward the 0.30 goal,
    past 0.15 — otherwise the bounded test proves nothing."""
    x_seq = _solve(None)
    assert max(x[0] for x in x_seq) > 0.15


def test_bounded_plan_respects_position_limit():
    """With a state bound pos ∈ [-0.1, 0.15], every knot (incl. terminal)
    stays within the bound (QP-feasible tolerance)."""
    x_seq = _solve([(0, -0.1, 0.15)])
    for k, x in enumerate(x_seq):
        assert x[0] <= 0.15 + 1e-6, f"knot {k}: pos {x[0]:.4f} > bound"
        assert x[0] >= -0.1 - 1e-6, f"knot {k}: pos {x[0]:.4f} < bound"
