"""CRISP Algorithm 1 — primal-only SCP with an exact l1 penalty merit function.

arXiv:2502.01055v3 §II. Problem template (eq 7):

    min J(z)   s.t.  c_i(z) = 0, i in E;  c_i(z) >= 0, i in I

with J convex quadratic. The merit function (eq 8) is

    phi(z; mu) = J(z) + sum_E mu_i |c_i(z)| + sum_I mu_i [c_i(z)]^-

and each iteration solves the convex trust-region QP (eq 10) for a trial step,
accepting it whenever the merit does not increase.

Two details are inferred because the paper underspecifies them; both are marked
INFERRED below and neither changes the published algorithm's intent:
  * Table I prints eps_c as "1e6". The text uses it as a constraint-violation
    threshold and §IV calls a solve successful below 1e-5, so the printed value
    is 1e-6 with a lost sign.
  * Algorithm 1 never resets the trust region after a penalty escalation. Left
    collapsed, the next iterate re-triggers the same convergence branch, so the
    method escalates straight to mu_max and returns Failure. We reset to
    delta_0, the standard exact-penalty behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import osqp
import scipy.sparse as sp


class NlpProblem:
    """User-supplied problem in the eq (7) template. Subclass and override."""

    n: int = 0

    def objective(self, z: np.ndarray) -> float:
        raise NotImplementedError

    def objective_grad(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def objective_hess(self, z: np.ndarray) -> sp.spmatrix:
        raise NotImplementedError

    def eq_constraints(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def eq_jacobian(self, z: np.ndarray) -> sp.spmatrix:
        raise NotImplementedError

    def ineq_constraints(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def ineq_jacobian(self, z: np.ndarray) -> sp.spmatrix:
        raise NotImplementedError


@dataclass
class CrispParams:
    """Table I of the paper, plus inner-QP settings."""

    k_max: int = 1000
    delta_0: float = 1.0
    delta_max: float = 10.0
    mu_0: float = 10.0
    mu_max: float = 1e6
    eta_low: float = 0.25
    eta_high: float = 0.75
    gamma_shrink: float = 0.25
    gamma_expand: float = 2.0
    eps_c: float = 1e-6          # INFERRED sign, see module docstring
    eps_p: float = 1e-3
    eps_r: float = 1e-3
    second_order_correction: bool = True
    #: Reset the trust region to delta_0 after a penalty escalation. The
    #: reference ships this line COMMENTED OUT (SolverInterface.h:554), leaving
    #: the radius collapsed and only refreshing the merit/model under the new
    #: penalties. See the module docstring.
    reset_trust_region_on_escalation: bool = True
    osqp_eps_abs: float = 1e-9
    osqp_eps_rel: float = 1e-9
    osqp_max_iter: int = 20000
    verbose: bool = False


@dataclass
class CrispResult:
    z: np.ndarray
    success: bool
    status: str
    objective: float
    max_violation: float
    iterations: int
    merit: float = 0.0


def _violation(c_eq: np.ndarray, c_ineq: np.ndarray) -> np.ndarray:
    """Per-constraint violation: |c| on equalities, [c]^- on inequalities."""
    return np.concatenate([np.abs(c_eq), np.maximum(0.0, -c_ineq)])


class CrispSolver:
    def __init__(self, params: CrispParams | None = None):
        self.p = params or CrispParams()

    # ---------------------------------------------------------------- solve
    def solve(self, prob: NlpProblem, z0: np.ndarray) -> CrispResult:
        p = self.p
        z = np.asarray(z0, dtype=float).copy()
        n = prob.n

        c_eq = prob.eq_constraints(z)
        c_in = prob.ineq_constraints(z)
        m_e, m_i = c_eq.size, c_in.size
        mu = np.full(m_e + m_i, p.mu_0)
        delta = p.delta_0

        H = prob.objective_hess(z)
        status, k = "max iterations reached", 0

        for k in range(p.k_max):
            J = prob.objective(z)
            g = prob.objective_grad(z)
            c_eq = prob.eq_constraints(z)
            c_in = prob.ineq_constraints(z)
            A_eq = prob.eq_jacobian(z)
            A_in = prob.ineq_jacobian(z)

            phi_k = J + mu[:m_e] @ np.abs(c_eq) + mu[m_e:] @ np.maximum(0.0, -c_in)

            step = self._trial_step(H, g, A_eq, c_eq, A_in, c_in, mu, delta, n)
            if step is None:
                # OSQP degenerates once the trust region collapses onto an
                # optimum. Feasibility there is convergence, not failure --
                # but a numerical failure must not escalate penalties, which is
                # reserved for the merit-convergence branch below.
                if self._is_feasible(prob, z):
                    status = "optimization successful"
                    break
                delta *= p.gamma_shrink
                if delta < 1e-12:
                    status = "inner QP failed"
                    break
                continue

            pred = phi_k - self._model(J, g, H, A_eq, c_eq, A_in, c_in, mu, step, m_e)
            ared = phi_k - self._merit(prob, z + step, mu, m_e)

            abandoned = False
            if ared < 0.0 and p.second_order_correction:
                soc = self._soc_step(prob, z, step, H, g, A_eq, c_eq, A_in, c_in,
                                     mu, delta, n)
                if soc is not None:
                    step = soc
                    pred = phi_k - self._model(J, g, H, A_eq, c_eq, A_in, c_in,
                                               mu, step, m_e)
                    ared = phi_k - self._merit(prob, z + step, mu, m_e)
                if ared < 0.0:
                    delta *= p.gamma_shrink        # abandon the step (line 12)
                    abandoned = True

            if not abandoned:
                rho = ared / pred if pred > 0.0 else -np.inf
                step_inf = np.max(np.abs(step)) if n else 0.0
                if rho < p.eta_low:
                    delta *= p.gamma_shrink
                elif rho > p.eta_high and step_inf >= delta * (1.0 - 1e-9):
                    delta = min(p.gamma_expand * delta, p.delta_max)
                z = z + step

            step_inf = np.max(np.abs(step)) if n else 0.0
            if p.verbose:
                viol = _violation(prob.eq_constraints(z), prob.ineq_constraints(z))
                print(f"[crisp] k={k:4d} J={prob.objective(z):12.6f} "
                      f"viol={viol.max() if viol.size else 0.0:.3e} "
                      f"delta={delta:.3e} |p|={step_inf:.3e}")

            if delta < p.eps_r or step_inf < p.eps_p:
                done, verdict = self._check_convergence(prob, z, mu)
                if done:
                    status = verdict
                    break
                if p.reset_trust_region_on_escalation:
                    delta = p.delta_0              # see module docstring

        c_eq = prob.eq_constraints(z)
        c_in = prob.ineq_constraints(z)
        viol = _violation(c_eq, c_in)
        max_viol = float(viol.max()) if viol.size else 0.0
        return CrispResult(
            z=z,
            success=(status == "optimization successful"),
            status=status,
            objective=float(prob.objective(z)),
            max_violation=max_viol,
            iterations=k + 1,
            merit=float(self._merit(prob, z, mu, m_e)),
        )

    # ------------------------------------------------------------- internals
    def _is_feasible(self, prob, z) -> bool:
        viol = _violation(prob.eq_constraints(z), prob.ineq_constraints(z))
        return bool(viol.size == 0 or viol.max() < self.p.eps_c)

    def _check_convergence(self, prob, z, mu):
        """Algorithm 1 lines 29-45. Escalates `mu` in place when still violated.

        Returns (done, status): done=True ends the solve with that verdict,
        done=False means penalties were raised and the caller should keep going.
        """
        viol = _violation(prob.eq_constraints(z), prob.ineq_constraints(z))
        if viol.size == 0 or viol.max() < self.p.eps_c:
            return True, "optimization successful"
        hot = viol >= self.p.eps_c
        if np.all(mu[hot] >= self.p.mu_max):
            return True, "penalty max out"
        mu[hot] = np.minimum(10.0 * mu[hot], self.p.mu_max)
        return False, ""

    def _merit(self, prob, z, mu, m_e):
        c_eq = prob.eq_constraints(z)
        c_in = prob.ineq_constraints(z)
        return (prob.objective(z)
                + mu[:m_e] @ np.abs(c_eq)
                + mu[m_e:] @ np.maximum(0.0, -c_in))

    @staticmethod
    def _model(J, g, H, A_eq, c_eq, A_in, c_in, mu, step, m_e):
        """q_{mu,k}(p) of eq (9) with the l1 terms written out."""
        lin_eq = c_eq + A_eq @ step
        lin_in = c_in + A_in @ step
        return (J + g @ step + 0.5 * step @ (H @ step)
                + mu[:m_e] @ np.abs(lin_eq)
                + mu[m_e:] @ np.maximum(0.0, -lin_in))

    def _trial_step(self, H, g, A_eq, c_eq, A_in, c_in, mu, delta, n,
                    c_eq_hat=None, c_in_hat=None):
        """Solve the eq (10) trust-region QP. Returns p, or None if OSQP fails.

        Variables are y = [p, v, w, t]: v, w split the equality residual and t
        absorbs inequality violation, which turns the nonsmooth l1 objective
        into a standard convex QP.
        """
        p_ = self.p
        m_e, m_i = c_eq.size, c_in.size
        b_eq = c_eq if c_eq_hat is None else c_eq_hat
        b_in = c_in if c_in_hat is None else c_in_hat
        n_y = n + 2 * m_e + m_i

        P = sp.block_diag(
            [H.tocsc(), sp.csc_matrix((2 * m_e + m_i, 2 * m_e + m_i))], format="csc"
        )
        q = np.concatenate([g, mu[:m_e], mu[:m_e], mu[m_e:]])

        I_e = sp.identity(m_e, format="csc")
        I_i = sp.identity(m_i, format="csc")
        rows = [
            sp.hstack([A_eq, -I_e, I_e, sp.csc_matrix((m_e, m_i))], format="csc"),
            sp.hstack([A_in, sp.csc_matrix((m_i, 2 * m_e)), I_i], format="csc"),
            sp.identity(n_y, format="csc"),
        ]
        A = sp.vstack(rows, format="csc")

        lo = np.concatenate([-b_eq, -b_in,
                             np.full(n, -delta), np.zeros(2 * m_e + m_i)])
        hi = np.concatenate([-b_eq, np.full(m_i, np.inf),
                             np.full(n, delta), np.full(2 * m_e + m_i, np.inf)])

        solver = osqp.OSQP()
        try:
            solver.setup(P=P, q=q, A=A, l=lo, u=hi, verbose=False,
                         eps_abs=p_.osqp_eps_abs, eps_rel=p_.osqp_eps_rel,
                         max_iter=p_.osqp_max_iter, polishing=True)
            res = solver.solve()
        except Exception:
            return None
        if res.info.status_val not in (1, 2):   # solved / solved_inaccurate
            return None
        step = np.asarray(res.x[:n], dtype=float)
        if not np.all(np.isfinite(step)):
            return None
        return np.clip(step, -delta, delta)

    def _soc_step(self, prob, z, step, H, g, A_eq, c_eq, A_in, c_in, mu, delta, n):
        """Second-order correction, eq (13).

        Re-solve the same QP with the constraint constants replaced by the value
        actually observed at the trial point, minus its linear part. Keeps the
        subproblem convex and needs no constraint Hessians.
        """
        c_eq_hat = prob.eq_constraints(z + step) - A_eq @ step
        c_in_hat = prob.ineq_constraints(z + step) - A_in @ step
        return self._trial_step(H, g, A_eq, c_eq, A_in, c_in, mu, delta, n,
                                c_eq_hat=c_eq_hat, c_in_hat=c_in_hat)
