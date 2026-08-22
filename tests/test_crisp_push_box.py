"""CRISP Appendix B-B (Push Box) contact-implicit formulation.

Reference: arXiv:2502.01055v3 eqs (52)-(68). Planar quasi-static box on a table;
the body-frame contact point (c_x, c_y) is a DECISION VARIABLE and the four
face-normal forces are gated by per-face complementarity constraints, so the
solver chooses which face to push and where on it.
"""
import numpy as np
import pytest

from control.crisp.push_box import (
    PushBoxParams,
    PushBoxProblem,
    min_terminal_weight,
    to_execution_plan,
)
from control.crisp.scp import CrispParams, CrispSolver

# Our canonical box task (config/tasks.yaml: pushing).
OUR_BOX = PushBoxParams(
    a=0.05, b=0.05, mu=0.46, mass=1.0, N=40, dt=0.05, c_int=0.6
)


def _params(**kw):
    # Reference constants, SolvePushbox.cpp:9-16.
    base = dict(a=0.5, b=0.25, mu=0.5, mass=1.0, g=9.8, N=10, dt=0.02, c_int=0.4)
    base.update(kw)
    return PushBoxParams(**base)


def test_face_center_push_is_pure_translation():
    """Left-face centre push at theta=0 -> +x translation, zero yaw rate."""
    p = _params()
    prob = PushBoxProblem(p, s_init=np.zeros(3), s_goal=np.zeros(3))

    # u = [c_x, c_y, lam_1y, lam_2x, lam_3y, lam_4x]; lam_2x pushes +x_body.
    sdot = prob.dynamics(np.zeros(3), np.array([-p.a, 0.0, 0.0, 1.0, 0.0, 0.0]))

    assert sdot[0] == pytest.approx(1.0 / (p.mu * p.mass * p.g))
    assert sdot[1] == pytest.approx(0.0)
    assert sdot[2] == pytest.approx(0.0)


def test_offcenter_push_induces_clockwise_yaw_rate():
    """+x force applied above the centre line -> negative (clockwise) yaw."""
    p = _params()
    prob = PushBoxProblem(p, s_init=np.zeros(3), s_goal=np.zeros(3))

    sdot = prob.dynamics(
        np.zeros(3), np.array([-p.a, 0.5 * p.b, 0.0, 1.0, 0.0, 0.0])
    )

    assert sdot[2] < 0.0
    expected = -0.5 * p.b / (p.c_int * p.r_char * p.mu * p.mass * p.g)
    assert sdot[2] == pytest.approx(expected)


def test_body_force_is_rotated_into_the_world_frame():
    """At theta=pi/2 a body +x push moves the box along world +y."""
    p = _params()
    prob = PushBoxProblem(p, s_init=np.zeros(3), s_goal=np.zeros(3))

    sdot = prob.dynamics(
        np.array([0.0, 0.0, np.pi / 2]),
        np.array([-p.a, 0.0, 0.0, 1.0, 0.0, 0.0]),
    )

    assert sdot[0] == pytest.approx(0.0, abs=1e-12)
    assert sdot[1] == pytest.approx(1.0 / (p.mu * p.mass * p.g))


def test_handbuilt_single_face_trajectory_is_exactly_feasible():
    """Explicit-Euler rollout of a constant left-face push satisfies eqs 55-67."""
    p = _params(N=10)
    prob = PushBoxProblem(p, s_init=np.zeros(3), s_goal=np.zeros(3))

    u = np.array([-p.a, 0.0, 0.0, 1.0, 0.0, 0.0])
    controls = np.tile(u, (p.N, 1))
    states = np.zeros((p.N + 1, 3))
    for k in range(p.N):
        states[k + 1] = states[k] + p.dt * prob.dynamics(states[k], controls[k])

    z = prob.pack(states, controls)

    assert np.max(np.abs(prob.eq_constraints(z))) < 1e-12
    assert np.min(prob.ineq_constraints(z)) > -1e-12


def test_pack_unpack_roundtrip():
    p = _params(N=7)
    prob = PushBoxProblem(p, s_init=np.zeros(3), s_goal=np.zeros(3))
    rng = np.random.default_rng(0)
    states = rng.normal(size=(p.N + 1, 3))
    controls = rng.normal(size=(p.N, 6))

    s2, u2 = prob.unpack(prob.pack(states, controls))

    np.testing.assert_allclose(s2, states)
    np.testing.assert_allclose(u2, controls)


def test_analytic_jacobians_match_finite_differences():
    p = _params(N=4)
    prob = PushBoxProblem(p, s_init=np.zeros(3), s_goal=np.array([0.3, 0.1, 0.2]))
    rng = np.random.default_rng(1)
    z = rng.normal(scale=0.3, size=prob.n)

    for fn, jac in ((prob.eq_constraints, prob.eq_jacobian),
                    (prob.ineq_constraints, prob.ineq_jacobian)):
        J = np.asarray(jac(z).todense())
        Jn = np.zeros_like(J)
        for i in range(prob.n):
            dz = np.zeros(prob.n)
            dz[i] = 1e-6
            Jn[:, i] = (fn(z + dz) - fn(z - dz)) / 2e-6
        np.testing.assert_allclose(J, Jn, atol=1e-6)

    g = prob.objective_grad(z)
    gn = np.zeros_like(g)
    for i in range(prob.n):
        dz = np.zeros(prob.n)
        dz[i] = 1e-6
        gn[i] = (prob.objective(z + dz) - prob.objective(z - dz)) / 2e-6
    np.testing.assert_allclose(g, gn, atol=1e-6)


def test_solves_our_box_task_from_all_zero_initial_guess():
    """config/tasks.yaml `pushing`: translate the 0.1 m cube 0.15 m in +x."""
    prob = PushBoxProblem(
        OUR_BOX, s_init=np.zeros(3), s_goal=np.array([0.15, 0.0, 0.0])
    )

    res = CrispSolver(CrispParams()).solve(prob, np.zeros(prob.n))

    assert res.success, res.status
    assert res.max_violation < 1e-5
    states, _ = prob.unpack(res.z)
    assert np.linalg.norm(states[-1, :2] - [0.15, 0.0]) < 0.01
    assert abs(states[-1, 2]) < np.pi / 6


def test_pushes_the_minus_x_face_to_translate_along_plus_x():
    """The face choice is an output of the solve, not an input."""
    prob = PushBoxProblem(
        OUR_BOX, s_init=np.zeros(3), s_goal=np.array([0.15, 0.0, 0.0])
    )

    res = CrispSolver(CrispParams()).solve(prob, np.zeros(prob.n))

    _, controls = prob.unpack(res.z)
    faces = {PushBoxProblem.active_face(u, tol=1e-4) for u in controls} - {None}
    assert faces == {"-x"}


def test_terminal_weight_below_the_escape_threshold_parks_at_the_origin():
    """All-zero is feasible; escaping it costs mu_0*a per knot of linearised
    face-gate violation, so a weak terminal pull leaves the box unmoved."""
    goal = np.array([0.15, 0.0, 0.0])
    q_star = min_terminal_weight(
        mu_0=CrispParams().mu_0, half_extent=0.05, distance=0.15,
        dt=0.05, k_trans=1.0 / (0.46 * 1.0 * 9.81),
    )
    assert 250.0 < q_star < 350.0

    def moved(q):
        p = PushBoxParams(a=0.05, b=0.05, mu=0.46, mass=1.0, N=20, dt=0.05,
                          q_pos=q, q_yaw=q)
        prob = PushBoxProblem(p, np.zeros(3), goal)
        res = CrispSolver(CrispParams()).solve(prob, np.zeros(prob.n))
        return np.linalg.norm(prob.unpack(res.z)[0][-1, :2])

    assert moved(0.8 * q_star) < 0.02
    assert moved(1.5 * q_star) > 0.10


def test_reports_success_when_the_inner_qp_degenerates_at_the_optimum():
    """A collapsed trust region at a feasible point is convergence, not failure."""
    p = PushBoxParams(a=0.05, b=0.05, mu=0.46, mass=1.0, N=40, dt=0.05,
                      q_pos=400.0, q_yaw=400.0)
    prob = PushBoxProblem(p, np.zeros(3), np.array([0.15, 0.0, 0.0]))

    res = CrispSolver(CrispParams()).solve(prob, np.zeros(prob.n))

    assert res.max_violation < 1e-5
    assert res.success, res.status


def test_rotation_has_no_first_order_gradient_at_the_all_zero_guess():
    """Yaw enters the dynamics as c x f -- a product of two variables that both
    start at zero -- so an all-zero guess offers first-order TRANSLATION but no
    first-order ROTATION. A pure-rotation goal therefore has no descent
    direction to find, and the solve converges to a do-nothing plan while
    reporting success. Warm-starting the contact onto a face restores it.
    """
    prob = PushBoxProblem(OUR_BOX, np.zeros(3), np.array([0.0, 0.0, np.pi / 2]))

    # Finite-difference d(sdot)/du at the all-zero knot.
    grad = np.zeros((3, 6))
    for j in range(6):
        du = np.zeros(6)
        du[j] = 1e-6
        grad[:, j] = (prob.dynamics(np.zeros(3), du)
                      - prob.dynamics(np.zeros(3), -du)) / 2e-6
    assert np.abs(grad[:2]).max() > 1e-3      # translation is first-order
    np.testing.assert_allclose(grad[2], 0.0, atol=1e-12)   # rotation is not

    res = CrispSolver(CrispParams()).solve(prob, np.zeros(prob.n))

    states, controls = prob.unpack(res.z)
    assert res.success                                      # reports success...
    np.testing.assert_allclose(states[-1], 0.0, atol=1e-9)  # ...having done nothing
    assert all(PushBoxProblem.active_face(u) is None for u in controls)


def test_execution_plan_offsets_the_ee_outside_the_pushed_face():
    """The planner's contact point is ON the surface; a finite pusher must sit
    one radius outside it, the same offset our sampler calls `sampling_setback`."""
    prob = PushBoxProblem(
        OUR_BOX, s_init=np.zeros(3), s_goal=np.array([0.15, 0.0, 0.0])
    )
    res = CrispSolver(CrispParams()).solve(prob, np.zeros(prob.n))

    plan = to_execution_plan(prob, res.z, ee_height=0.05, pusher_radius=0.025)

    # Pushing the -x face: EE sits a radius further -x than the box's -x wall.
    assert np.all(plan.face == np.array(["-x"] * OUR_BOX.N))
    wall_x = plan.box_pose[:-1, 0] - OUR_BOX.a
    np.testing.assert_allclose(plan.ee_xyz[:, 0], wall_x - 0.025, atol=1e-9)
    np.testing.assert_allclose(plan.ee_xyz[:, 2], 0.05)
    # ...and pushes along +x.
    assert np.all(plan.force_world[:, 0] > 0)
    np.testing.assert_allclose(plan.force_world[:, 1], 0.0, atol=1e-9)


def test_execution_plan_reports_the_ee_speed_it_demands():
    prob = PushBoxProblem(
        OUR_BOX, s_init=np.zeros(3), s_goal=np.array([0.15, 0.0, 0.0])
    )
    res = CrispSolver(CrispParams()).solve(prob, np.zeros(prob.n))

    plan = to_execution_plan(prob, res.z, ee_height=0.05, pusher_radius=0.025)

    # A single-face straight push never asks the arm to jump.
    assert plan.max_ee_speed() < 0.5
    assert plan.face_switches() == 0


@pytest.mark.slow
def test_paper_benchmark_drives_the_box_from_an_all_zero_guess():
    """Paper IV-B2 at the reference's own constants (SolvePushbox.cpp:9-16).

    The qualitative claim reproduces: from an all-zero guess the solve finds the
    correct pushing face and carries the box essentially the whole 3 m.
    Measured: 2.9592 m travelled, terminal error 0.0408 m (1.36%), 65 iterations.
    """
    goal = np.array([3.0, 0.0, 0.0])
    prob = PushBoxProblem(_params(N=200, dt=0.02), s_init=np.zeros(3), s_goal=goal)

    res = CrispSolver(CrispParams()).solve(prob, np.zeros(prob.n))

    states, controls = prob.unpack(res.z)
    assert np.linalg.norm(states[-1, :2]) > 0.8 * 3.0
    assert np.linalg.norm(states[-1, :2] - goal[:2]) < 0.05
    assert {PushBoxProblem.active_face(u, 1e-4) for u in controls} - {None} == {"-x"}


@pytest.mark.slow
@pytest.mark.xfail(
    reason="Table II reports push box at violation 8.3e-9, and the reference's "
           "own constraintTol is 1e-6. Under the reference-faithful formulation "
           "this port lands at 7.53e-5 -- close on trajectory (1.36% terminal "
           "error) but not feasible to their bar. The inner QP is the prime "
           "suspect: the reference uses interior-point PIQP "
           "(SolverInterface.h:9) and this port uses OSQP, which under the same "
           "formulation needs ~4x the iterations and stops converging at short "
           "horizons.",
    strict=True,
)
def test_paper_benchmark_reaches_published_feasibility():
    goal = np.array([3.0, 0.0, 0.0])
    prob = PushBoxProblem(_params(N=200, dt=0.02), s_init=np.zeros(3), s_goal=goal)

    res = CrispSolver(CrispParams()).solve(prob, np.zeros(prob.n))

    assert res.max_violation < 1e-5
    assert res.success, res.status
