"""Phase 2F/2G: batched GPU primitives in CuPy.

Scope is deliberately narrow. These are the pieces of the C3+ ADMM iteration
that are (a) mathematically simple, (b) independently testable against the
CPU, and (c) parallel over the natural independent axes:

    candidate (B) x horizon knot (N) x contact slot (n_lambda)

CuPy rather than torch, per the Phase 2 directive: the surrounding code is
NumPy-shaped, CuPy arrays are drop-in, and a RawKernel can replace any of
these later without changing call sites. (An earlier feasibility arc left
torch-based modules in this package; they belong to the rejected
full-solver-on-GPU route and are not used here.)

WHICH PRIMITIVES ACTUALLY EXIST in the C3+ path -- checked, not assumed:

    complementarity projection   YES -- Bui eq.(12), the ONLY projection
    dual update                  YES -- omega += z - delta
    residual calculation         YES -- over the lambda/eta slots only
    candidate cost + argmin      YES -- but the controller's ranking cost
                                        mixes in alignment/travel terms that
                                        live outside the solver
    friction / SOC projection    NO  -- the Lorentz code belongs to the
                                        falsified mode='c3' lineage
    Drake 4D tangent-basis proj  NO  -- not in this path at all

So the Phase 2 brief's SOC and tangent-basis primitives have no C3+
counterpart to port.

All functions accept and return CuPy arrays and never synchronise. float64
throughout, matching the CPU.
"""
from __future__ import annotations

import math

import numpy as np

try:                                    # import must not hard-fail on CPU
    import cupy as cp
    HAVE_CUPY = True
except Exception:                       # pragma: no cover
    cp = None
    HAVE_CUPY = False


def _xp(a):
    """Work on CuPy or NumPy arrays interchangeably, so every primitive is
    testable on a machine without a GPU."""
    return cp if (HAVE_CUPY and isinstance(a, cp.ndarray)) else np


# ------------------------------------------------------------------ 2F.1

def project_C3Plus_batch(lam, eta, u_lambda: float = 1.0,
                         u_eta: float = 1.0):
    """Bui 2026 eq.(12) componentwise complementarity projection, batched.

    Exact transcription of `C3Solver._project_C3Plus`
    (control/admm_solver.py:1035), which is the specification:

        sqrt_ratio = sqrt(u_lambda / u_eta)
        cond1 = (eta >= 0) & (eta >= sqrt_ratio * lam)   -> (0, eta)
        cond2 = (lam >= 0) & (eta <  sqrt_ratio * lam)   -> (lam, 0)
        else                                             -> (0, 0)

    Elementwise over ANY trailing shape, so `lam`/`eta` may be
    (B, N, n_lambda) and the whole candidate x knot x slot space is covered
    by one kernel. Values are SELECTED, never recomputed, so agreement with
    the CPU is exact rather than approximate.
    """
    xp = _xp(lam)
    sqrt_ratio = math.sqrt(float(u_lambda) / float(u_eta))
    cond1 = (eta >= 0.0) & (eta >= sqrt_ratio * lam)
    cond2 = (lam >= 0.0) & (eta < sqrt_ratio * lam)
    zero = xp.zeros((), dtype=lam.dtype)
    return xp.where(cond2, lam, zero), xp.where(cond1, eta, zero)


# ------------------------------------------------------------------ 2F.2

def dual_update(omega, z, delta):
    """ADMM eq.(9): omega <- omega + (z - delta). Elementwise, batched."""
    return omega + (z - delta)


# ------------------------------------------------------------------ 2F.3

def slot_view(vec, N: int, TOT: int, SL: int, SE: int, n_lambda: int):
    """Gather the lambda and eta slots the residual is defined over.

    The CPU builds this by concatenating, per knot i,
    `[delta[i*TOT+SL : +n_lambda], delta[i*TOT+SE : +n_lambda]]`
    (admm_solver.py:2380-2392). Here it is one strided gather producing
    (..., N, 2*n_lambda) with the SAME element order.
    """
    xp = _xp(vec)
    lead = vec.shape[:-1]
    knots = vec[..., :N * TOT].reshape(*lead, N, TOT)
    return xp.concatenate([knots[..., SL:SL + n_lambda],
                           knots[..., SE:SE + n_lambda]], axis=-1)


def residuals(z, delta, delta_prev, rho, N, TOT, SL, SE, n_lambda):
    """Primal and dual residuals, per candidate.

    Mirrors admm_solver.py:2397-2398 exactly:
        pr = ||lam_vec - dlt_vec||
        dr = rho * ||dlt_vec - dlt_prev_vec||
    where the vectors span only the lambda/eta slots -- NOT the whole z.
    Returns arrays shaped like the leading (candidate) dimension.
    """
    xp = _xp(z)
    a = slot_view(z, N, TOT, SL, SE, n_lambda)
    b = slot_view(delta, N, TOT, SL, SE, n_lambda)
    c = slot_view(delta_prev, N, TOT, SL, SE, n_lambda)
    flat = (-1,) if a.ndim == 2 else (a.shape[0], -1)
    pr = xp.linalg.norm(a.reshape(*flat) - b.reshape(*flat), axis=-1)
    dr = rho * xp.linalg.norm(b.reshape(*flat) - c.reshape(*flat), axis=-1)
    return pr, dr


# ------------------------------------------------------------------ 2F.4

def candidate_effort_cost(u_seqs):
    """Solver-side surrogate objective: horizon control effort per candidate.

    NOTE this is NOT the controller's ranking cost -- that one adds
    alignment and travel terms computed outside the solver, so it cannot be
    evaluated here. This exists so the cost/argmin primitives have something
    exactly comparable on both sides.
    """
    xp = _xp(u_seqs)
    return xp.sum(u_seqs.reshape(u_seqs.shape[0], -1) ** 2, axis=-1)


def candidate_argmin(costs):
    """On-device selection. Non-finite costs must never win: an all-NaN
    cost vector once silently disabled repositioning for a whole run."""
    xp = _xp(costs)
    safe = xp.where(xp.isfinite(costs), costs, xp.inf)
    return int(xp.argmin(safe))


# ------------------------------------------------------------------ helpers

def to_device(a):
    return cp.asarray(a, dtype=cp.float64) if HAVE_CUPY else np.asarray(
        a, dtype=np.float64)


def to_host(a):
    return cp.asnumpy(a) if (HAVE_CUPY and isinstance(a, cp.ndarray)) \
        else np.asarray(a)
