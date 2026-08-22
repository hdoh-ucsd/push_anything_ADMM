"""Bui 2026 eq.(12) componentwise projection, batched (cuNRTO P5).

Exact torch transcription of C3Solver._project_C3Plus
(control/admm_solver.py:1035). That function is the specification:

    sqrt_ratio = sqrt(u_lambda / u_eta)
    cond1 = (eta >= 0) & (eta >= sqrt_ratio * lam)   -> (0, eta)
    cond2 = (lam >= 0) & (eta <  sqrt_ratio * lam)   -> (lam, 0)
    else                                             -> (0, 0)

Elementwise over any trailing shape, which is exactly what makes batching
over samples valid (cuNRTO P2). Cheaper than cuNRTO's own SOC projection:
no norms and no cone cases, just two comparisons and a select.
"""
from __future__ import annotations

import math

import torch


def project_C3Plus_batch(lam: torch.Tensor, eta: torch.Tensor,
                         u_lambda: float = 1.0, u_eta: float = 1.0):
    """Batched Bui eq.(12). Returns (delta_lam, delta_eta), same shape as
    the inputs. Values are SELECTED, never recomputed, so agreement with
    the numpy implementation is exact rather than approximate."""
    sqrt_ratio = math.sqrt(float(u_lambda) / float(u_eta))
    cond1 = (eta >= 0.0) & (eta >= sqrt_ratio * lam)
    cond2 = (lam >= 0.0) & (eta < sqrt_ratio * lam)
    zero = torch.zeros((), dtype=lam.dtype, device=lam.device)
    return torch.where(cond2, lam, zero), torch.where(cond1, eta, zero)
