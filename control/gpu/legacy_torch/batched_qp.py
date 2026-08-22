"""Batched dense OSQP iteration in torch (cuNRTO P1/P3/P6).

Solves B independent QPs that share shapes:

    min  1/2 z'Pz + q'z    s.t.   lo <= A z <= hi

using the OSQP ADMM iteration (Stellato et al., the algorithm Drake's
OsqpSolver runs on CPU today) on the reduced KKT system

    K = P + sigma*I + A' R A
    K x~ = sigma*x - q + A'(R z - y)

AMENDMENT A1 (measured 2026-08-21, see gpu-admm-baseline.txt)
------------------------------------------------------------
K is materialised as an explicit INVERSE and applied with a batched GEMV,
rather than kept as a Cholesky factor and applied with cholesky_solve.
A triangular solve is sequential back-substitution -- the most GPU-hostile
kernel in the iteration -- and measured 0.59x against the CPU at our size.
The inverse form measured 4.41x, agreeing to 1.1e-14.

This is only legitimate because K changes just once per C3+ ADMM iteration
(the rho_scale ramp perturbs its diagonal) while the RHS changes on every
inner iteration -- cuNRTO SM VIII-B's "K_KKT is constant, and only the RHS
changes, enabling factorization reuse", applied one level down. Break-even
is 10 inner iterations; OSQP ships check_termination=100, so at least 100
always run.

Use torch.cholesky_inverse (14.0 ms at batch 6, n 719), never
torch.linalg.inv (31.0 ms). Conditioning is checked against the corpus in
tests/test_gpu_batched_qp.py -- K = P + sigma*I + A'RA is SPD and well
conditioned by construction, so the usual "never invert a matrix" caution
does not bite here, but it is verified rather than assumed.

Every iterate stays on-device; solve() takes and returns device tensors so
the caller's outer loop never round-trips through the host (cuNRTO P1).
"""
from __future__ import annotations

import torch

_F64 = torch.float64


class BatchedBoxQP:
    """B QPs sharing (n, m). Equality rows must come FIRST in A, with
    lo == hi == b_eq on those rows; `n_eq_rows` says how many."""

    def __init__(self, P, A, lo, hi, n_eq_rows: int,
                 sigma: float = 1e-5, rho_bar: float = 0.1,
                 alpha: float = 1.6, device: str = "cuda"):
        self.dev = torch.device(device)
        kw = dict(dtype=_F64, device=self.dev)
        self.P = torch.as_tensor(P, **kw)
        self.A = torch.as_tensor(A, **kw)
        self.At = self.A.transpose(1, 2).contiguous()
        self.lo = torch.as_tensor(lo, **kw)
        self.hi = torch.as_tensor(hi, **kw)
        _, m, n = self.A.shape
        self.n, self.m = n, m
        self.alpha, self.sigma = float(alpha), float(sigma)
        # OSQP convention: equality rows get 1e3 * rho_bar.
        rho = torch.full((m,), float(rho_bar), **kw)
        rho[:n_eq_rows] = 1e3 * float(rho_bar)
        self.rho = rho
        self.rho_inv = 1.0 / rho
        self.eye = torch.eye(n, **kw)
        self._factor()

    # -- A1: build K^-1 once; the inner loop then only does GEMVs ----------
    def _factor(self):
        K = (self.P + self.sigma * self.eye
             + self.At @ (self.rho[:, None] * self.A))
        self.Kinv = torch.cholesky_inverse(torch.linalg.cholesky(K))

    def refactor_diag(self, diag_delta):
        """The C3+ rho_scale ramp perturbs P by a DIAGONAL each outer
        iteration (correction C2). Rebuild K^-1 for the new P."""
        self.P = self.P + torch.diag_embed(diag_delta)
        self._factor()

    def solve(self, q, max_iter: int = 2000, eps: float = 1e-5,
              x0=None, z0=None, y0=None, check_every: int = 100):
        """Returns (x, z, y) as DEVICE tensors. Pass the previous triple
        back in to warm start (cuNRTO P6)."""
        kw = dict(dtype=_F64, device=self.dev)
        q = torch.as_tensor(q, **kw)
        B = q.shape[0]
        x = torch.zeros(B, self.n, **kw) if x0 is None else x0.clone()
        z = torch.zeros(B, self.m, **kw) if z0 is None else z0.clone()
        y = torch.zeros(B, self.m, **kw) if y0 is None else y0.clone()
        a, one_minus_a = self.alpha, 1.0 - self.alpha

        for it in range(max_iter):
            rhs = self.sigma * x - q + torch.bmm(
                self.At, (self.rho * z - y).unsqueeze(-1)).squeeze(-1)
            xt = torch.bmm(self.Kinv, rhs.unsqueeze(-1)).squeeze(-1)   # A1
            zt = torch.bmm(self.A, xt.unsqueeze(-1)).squeeze(-1)
            x = a * xt + one_minus_a * x
            z_prev = z
            z = torch.clamp(a * zt + one_minus_a * z_prev + self.rho_inv * y,
                            self.lo, self.hi)
            y = y + self.rho * (a * zt + one_minus_a * z_prev - z)
            # Matches OSQP's check_termination cadence. The float() here is
            # the ONLY host sync in the loop; keeping it off the per-iter
            # path is what preserves GPU utilization.
            if (it + 1) % check_every == 0:
                Ax = torch.bmm(self.A, x.unsqueeze(-1)).squeeze(-1)
                r_p = (Ax - z).abs().amax()
                r_d = (torch.bmm(self.P, x.unsqueeze(-1)).squeeze(-1) + q
                       + torch.bmm(self.At, y.unsqueeze(-1)).squeeze(-1)
                       ).abs().amax()
                if float(torch.maximum(r_p, r_d)) < eps:
                    break
        self.last_iters = it + 1
        return x, z, y
