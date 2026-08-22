"""CRISP sequential-convex-programming solver (Li et al. 2025, Algorithm 1).

Reference: "On the Surprising Robustness of Sequential Convex Optimization for
Contact-Implicit Motion Planning", arXiv:2502.01055v3, §II + Table I.
"""
import numpy as np
import scipy.sparse as sp

from control.crisp.scp import CrispParams, CrispSolver, NlpProblem


class _Example1(NlpProblem):
    """Paper Example 1 (eq 1): min x1^2+x2^2 s.t. x1>=0, x2>=0, x1*x2=0.

    The feasible set is the union of the two positive coordinate axes and the
    optimum is the origin. Every feasible point violates LICQ/MFCQ, so this is
    the smallest problem that exercises the primal-only path.
    """

    n = 2

    def objective(self, z):
        return float(z @ z)

    def objective_grad(self, z):
        return 2.0 * z

    def objective_hess(self, z):
        return sp.csc_matrix(2.0 * np.eye(2))

    def eq_constraints(self, z):
        return np.array([z[0] * z[1]])

    def eq_jacobian(self, z):
        return sp.csc_matrix(np.array([[z[1], z[0]]]))

    def ineq_constraints(self, z):
        return z.copy()

    def ineq_jacobian(self, z):
        return sp.csc_matrix(np.eye(2))


class _EqualityQP(NlpProblem):
    """min 0.5*||z||^2 s.t. sum(z) = 1 -> z = 1/n. Smooth CQ-satisfying sanity."""

    n = 4

    def objective(self, z):
        return 0.5 * float(z @ z)

    def objective_grad(self, z):
        return z.copy()

    def objective_hess(self, z):
        return sp.csc_matrix(np.eye(self.n))

    def eq_constraints(self, z):
        return np.array([z.sum() - 1.0])

    def eq_jacobian(self, z):
        return sp.csc_matrix(np.ones((1, self.n)))

    def ineq_constraints(self, z):
        return np.zeros(0)

    def ineq_jacobian(self, z):
        return sp.csc_matrix((0, self.n))


class _Infeasible(NlpProblem):
    """z >= 1 and -z >= 0 simultaneously: no feasible point exists."""

    n = 1

    def objective(self, z):
        return 0.0

    def objective_grad(self, z):
        return np.zeros(1)

    def objective_hess(self, z):
        return sp.csc_matrix((1, 1))

    def eq_constraints(self, z):
        return np.zeros(0)

    def eq_jacobian(self, z):
        return sp.csc_matrix((0, 1))

    def ineq_constraints(self, z):
        return np.array([z[0] - 1.0, -z[0]])

    def ineq_jacobian(self, z):
        return sp.csc_matrix(np.array([[1.0], [-1.0]]))


def test_solves_paper_example1_mpcc_from_interior_start():
    """Converges to the origin from a point off both axes.

    Default tolerances stop the iteration once the complementarity residual
    drops below eps_c, which floors the iterate near sqrt(eps_c); the paper's
    own success bar is a violation below 1e-5.
    """
    res = CrispSolver(CrispParams()).solve(_Example1(), np.array([1.0, 1.0]))

    assert res.success
    assert res.max_violation < 1e-5
    np.testing.assert_allclose(res.z, [0.0, 0.0], atol=2e-3)


def test_example1_accuracy_tightens_with_tolerances():
    """The sqrt(eps_c) floor is a tolerance artefact, not a spurious stationary point."""
    tight = CrispParams(eps_c=1e-14, eps_p=1e-7, eps_r=1e-7)

    res = CrispSolver(tight).solve(_Example1(), np.array([1.0, 1.0]))

    assert res.success
    np.testing.assert_allclose(res.z, [0.0, 0.0], atol=1e-12)


def test_solves_smooth_equality_qp():
    res = CrispSolver(CrispParams()).solve(_EqualityQP(), np.zeros(4))

    assert res.success
    np.testing.assert_allclose(res.z, np.full(4, 0.25), atol=1e-6)


def test_reports_failure_on_infeasible_problem():
    res = CrispSolver(CrispParams()).solve(_Infeasible(), np.zeros(1))

    assert not res.success
    assert res.max_violation > 1e-3


def test_all_zero_initial_guess_is_accepted():
    """The paper's headline claim is convergence from all-zero guesses."""
    res = CrispSolver(CrispParams()).solve(_Example1(), np.zeros(2))

    assert res.success
    np.testing.assert_allclose(res.z, [0.0, 0.0], atol=1e-6)
