"""End-to-end coverage of RepositionIKTracker as a unit, isolated from
the §IV-D wrapper. Tests construct the tracker directly and call its
public surface (compute_torque, _solve_single_knot_ik via diagnose_failure_at)
with controlled inputs.

Why a separate file (not inside test_inner_solve.py): inner_solve tests
the per-sample C3+IK evaluation; this file tests the reposition tracker
that runs after a sample is committed.

Test inventory and step-5 receipts referenced in test rationale:
    - test_reachable_target_no_obstacle    : happy path
    - test_obstacle_in_path_without_dmin   : V-7 default-zero rationale
    - test_obstacle_in_path_with_dmin      : opt-in-branch coverage
    - test_joint_limit_continuity          : redundancy / centering smoothness
    - test_infeasibility_marks_target      : timeout-induced (replaces (e))
    - test_timing_p99                      : empirical wall-clock budget;
                                             gated behind PUSH_RUN_TIMING_TESTS=1

Drake required. Seed = 42, hard-coded for reproducibility.

Run:
    PYTHONPATH=. pytest tests/test_reposition_ik.py -v
    PUSH_RUN_TIMING_TESTS=1 PYTHONPATH=. pytest tests/test_reposition_ik.py -v
"""
from __future__ import annotations

import dataclasses
import os
from typing import Any, Callable, NamedTuple, Optional

import numpy as np
import pytest

# Drake is mandatory for this file. importorskip yields a clean skip
# (not an error) when running on a machine without pydrake.
ad = pytest.importorskip("pydrake.all", reason="Drake required for RepositionIKTracker tests")

import yaml

from sim.env_builder import build_environment, EE_BODY_NAME, INITIAL_ARM_Q
from control.sampling_c3.params import (
    SamplingC3Params,
    RepositionIKParams,
    RepositionParams,
    RepositioningTrajectoryType,
)
from control.sampling_c3.reposition_ik import RepositionIKTracker
from control.sampling_c3.ik import solve_ik_to_ee_pos


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

SEED = 42
DT = 0.05
DT_CTRL = 0.01
TIGHT_TIMEOUT_S = 1e-3   # used by test_infeasibility_marks_target

# Production horizon — matches main.py. Tracker stores it but the unit
# tests evaluate knot 0 only (consistent with num_full_ik_knots=1).
HORIZON = 20


# ---------------------------------------------------------------------------
# Fixture return types
# ---------------------------------------------------------------------------

class World(NamedTuple):
    """Session-scoped Drake world. Constructed once per pytest session."""
    diagram:     Any
    plant:       Any
    scene_graph: Any
    panda_model: Any
    obj_body:    Any
    ee_frame:    Any
    plant_ad:    Any
    context_ad:  Any


class RootCtx(NamedTuple):
    """Function-scoped fresh root context + plant subcontext."""
    root:     Any   # diagram root context
    plant:    Any   # plant.GetMyMutableContextFromRoot(root)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def _world() -> World:
    """Build the production environment once per pytest session.

    Construction is ~1.5 s (Drake plant Finalize + ToAutoDiffXd) and
    starts a Meshcat server on port 7000 — both are amortised across
    every test in the session. Tests must NEVER mutate fields exposed
    here; per-test mutable state lives in the function-scoped root_ctx
    fixture below.
    """
    with open("config/tasks.yaml") as f:
        task_cfg = yaml.safe_load(f)["tasks"]["pushing"]

    diagram, plant, panda_model, object_model, _meshcat, plant_ad, context_ad = \
        build_environment(task_cfg)

    obj_body = plant.GetBodyByName(task_cfg["link_name"], object_model)
    ee_frame = plant.GetFrameByName(EE_BODY_NAME)

    scene_graphs = [s for s in diagram.GetSystems() if isinstance(s, ad.SceneGraph)]
    assert len(scene_graphs) == 1, (
        f"Expected exactly 1 SceneGraph in the diagram, got {len(scene_graphs)}."
    )
    scene_graph = scene_graphs[0]

    return World(
        diagram=diagram, plant=plant, scene_graph=scene_graph,
        panda_model=panda_model, obj_body=obj_body, ee_frame=ee_frame,
        plant_ad=plant_ad, context_ad=context_ad,
    )


@pytest.fixture
def root_ctx(_world: World) -> RootCtx:
    """Fresh root + plant context per test, initialised to:
        - Panda arm at INITIAL_ARM_Q
        - manipuland at the production task's init_xyz pose

    Function-scoped: every test gets a clean context, so any mutation
    a test does (e.g., moving the manipuland) is isolated from siblings.
    """
    with open("config/tasks.yaml") as f:
        task_cfg = yaml.safe_load(f)["tasks"]["pushing"]

    simulator = ad.Simulator(_world.diagram)
    root = simulator.get_mutable_context()
    plant_ctx = _world.plant.GetMyMutableContextFromRoot(root)

    _world.plant.SetFreeBodyPose(
        plant_ctx, _world.obj_body,
        ad.RigidTransform(ad.RotationMatrix(), list(task_cfg["init_xyz"])),
    )
    _world.plant.SetPositions(plant_ctx, _world.panda_model, INITIAL_ARM_Q)

    return RootCtx(root=root, plant=plant_ctx)


@pytest.fixture
def default_params() -> SamplingC3Params:
    """Fresh SamplingC3Params() with C-2 defaults, per test.

    Function-scoped on purpose: tracker_factory applies overrides via
    dataclasses.replace (not in-place mutation), but if a test ever
    mutated this object directly, the next test would get a clean
    instance regardless. Defends against future drift if a test ever
    forgets the replace contract.
    """
    return SamplingC3Params()


@pytest.fixture
def tracker_factory(_world: World, default_params: SamplingC3Params,
                    ) -> Callable[..., RepositionIKTracker]:
    """Returns a callable that builds a RepositionIKTracker.

    Signature:
        factory(ik_overrides: Optional[dict] = None,
                repos_overrides: Optional[dict] = None,
                ) -> RepositionIKTracker

    Overrides are applied via ``dataclasses.replace`` on COPIES of the
    fixture's default_params — the fixture's instance is never mutated.
    This is the contract: every call returns a fresh tracker built on
    its own params copy. Sibling tests do not interfere.

    The tracker honors RepositionIKParams.warm_up_on_construction=True
    (the default), which costs ~15 ms once per call. Tests that care
    about timing should call the tracker once after construction to
    flush any residual cold-state effects (see test_timing_p99).

    Always uses RepositioningTrajectoryType.kIK in the inner repos_params
    — the tracker only makes sense for that traj type.
    """
    n_arm_dofs = _world.plant.num_actuators()

    def factory(ik_overrides: Optional[dict] = None,
                repos_overrides: Optional[dict] = None,
                ) -> RepositionIKTracker:
        ik_kw    = dict(ik_overrides    or {})
        repos_kw = dict(repos_overrides or {})

        new_ik = dataclasses.replace(default_params.repos_ik_params, **ik_kw)
        # traj_type is forced to kIK here (silently overriding any value
        # in repos_overrides). The factory only knows how to build a
        # RepositionIKTracker, so any other traj_type would be a test
        # bug. Forcing here keeps test bodies from accidentally building
        # the wrong tracker via a stale repos_overrides dict.
        new_repos = dataclasses.replace(
            default_params.reposition_params,
            traj_type=RepositioningTrajectoryType.kIK,
            **{k: v for k, v in repos_kw.items() if k != "traj_type"},
        )

        return RepositionIKTracker(
            plant=_world.plant,
            ee_frame=_world.ee_frame,
            obj_body=_world.obj_body,
            n_arm_dofs=n_arm_dofs,
            horizon=HORIZON,
            dt=DT,
            repos_params=new_repos,
            ik_params=new_ik,
            diagram=_world.diagram,
            scene_graph=_world.scene_graph,
        )

    return factory


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_reachable_target_no_obstacle(_world: World, root_ctx: RootCtx,
                                      tracker_factory) -> None:
    """Canary for the file's pattern: tracker constructs, solves, returns
    a finite torque, and reports knot 0 feasible. Regression protection
    for the basic happy path — if this ever breaks, every other test in
    this file is suspect.
    """
    tracker = tracker_factory()

    current_q = _world.plant.GetPositions(root_ctx.plant)
    current_v = _world.plant.GetVelocities(root_ctx.plant)

    # FK the pusher at INITIAL_ARM_Q, then offset by +5cm in y (side) and
    # +10cm in z (above). y-axis avoids resting contact with the manipuland
    # at (0, 0, 0.05); z-clearance keeps the target well above the table.
    ee_now = _world.plant.CalcPointsPositions(
        root_ctx.plant, _world.ee_frame, np.zeros(3), _world.plant.world_frame(),
    ).flatten()
    p_target = ee_now + np.array([0.0, 0.05, 0.10])

    u, _diag = tracker.compute_torque(
        current_q=current_q, current_v=current_v,
        plant_ctx=root_ctx.plant, p_target=p_target, dt_ctrl=DT_CTRL,
    )

    assert np.all(np.isfinite(u))
    assert tracker.last_knot0_feasible is True
    assert tracker.last_knot0_failure_msg is None
    # Loose pathology check; tight timing is in test_timing_p99.
    assert tracker.last_knots_solve_ms[0] < 100.0


# ---------------------------------------------------------------------------
# Module-local scenario helper — used only by the obstacle-in-path pair.
# Factored out because the two tests share an identical setup; their delta
# is the d_min override and the expected outcome. Inlining identical setup
# in both bodies would obscure that delta.
# ---------------------------------------------------------------------------

def _setup_resting_contact_scenario(world: World, root_ctx: RootCtx,
                                    ) -> tuple:
    """Drive the pusher into resting contact with the manipuland's top
    face using the unconstrained DLS IK from control/sampling_c3/ik.py.

    The DLS IK enforces no min-distance, so it can converge into the
    contact state — this is the same helper the production sample-
    evaluator uses, and using it here keeps test-setup decoupled from
    the system-under-test (the constrained Drake IK inside the tracker).

    Returns (current_q, current_v, p_target) where p_target is set to
    the warm-start EE position itself. This pins the guide path's knot 0
    to the warm-start (next_waypoint's direct-line shortcut returns
    p_target verbatim when distance is zero), so the IK's position
    constraint forces the EE to stay at the contact configuration —
    leaving no room to satisfy a positive d_min by lifting away. With
    d_min = 0.0 the warm-start IS a feasible solution (trivial); with
    d_min = 0.005 there is no feasible point (signed distance ≈ 0 < d_min,
    and position tolerance < required lift).
    """
    with open("config/tasks.yaml") as f:
        cfg = yaml.safe_load(f)["tasks"]["pushing"]

    # Box top face center: init_xyz_z + half-height of size_z.
    box_top_z = cfg["init_xyz"][2] + cfg["size"][2] / 2.0   # 0.05 + 0.05 = 0.10
    # PUSHER_RADIUS in sim/env_builder.py — sphere welded to panda_link8.
    pusher_radius = 0.025
    p_contact = np.array([
        cfg["init_xyz"][0],
        cfg["init_xyz"][1],
        box_top_z + pusher_radius,                          # 0.125
    ])

    n_arm = world.plant.num_actuators()
    q_init = world.plant.GetPositions(root_ctx.plant)
    q_solved, err_norm, _iters = solve_ik_to_ee_pos(
        plant=world.plant, ee_frame=world.ee_frame,
        p_target=p_contact, q_init=q_init,
        plant_ctx=root_ctx.plant, n_arm_dofs=n_arm,
    )
    assert err_norm < 5e-3, (
        f"DLS-IK setup failed to land EE at p_contact: err={err_norm:.4f} m"
    )
    world.plant.SetPositions(root_ctx.plant, q_solved)

    current_q = world.plant.GetPositions(root_ctx.plant)
    current_v = world.plant.GetVelocities(root_ctx.plant)
    # Pin the IK target to the warm-start EE position (FK of q_solved).
    # See helper docstring for why this is the right pin point.
    ee_now = world.plant.CalcPointsPositions(
        root_ctx.plant, world.ee_frame, np.zeros(3), world.plant.world_frame(),
    ).flatten()
    p_target = ee_now.copy()
    return current_q, current_v, p_target


def test_obstacle_in_path_without_dmin(_world: World, root_ctx: RootCtx,
                                       tracker_factory) -> None:
    """Regression protection for V-7: the production default
    (ik_min_distance_lower_bound = 0.0) must succeed in resting-contact
    scenarios. If this fails, the controller would thrash on every
    approach-to-contact loop as it did pre-V-7.
    """
    current_q, current_v, p_target = _setup_resting_contact_scenario(
        _world, root_ctx,
    )
    tracker = tracker_factory()  # defaults: ik_min_distance_lower_bound=0.0

    u, _diag = tracker.compute_torque(
        current_q=current_q, current_v=current_v,
        plant_ctx=root_ctx.plant, p_target=p_target, dt_ctrl=DT_CTRL,
    )

    assert np.all(np.isfinite(u))
    assert tracker.last_knot0_feasible is True
    assert tracker.last_knot0_failure_msg is None
    assert tracker.last_knots_solve_ms[0] < 100.0


def test_obstacle_in_path_with_dmin(_world: World, root_ctx: RootCtx,
                                    tracker_factory) -> None:
    """Regression protection for V-7: with d_min > 0 opted in, the IK
    must reject resting-contact warm-starts. This documents WHY the
    default is 0.0 — any positive d_min makes contact-rich manipulation
    infeasible. Future maintainers considering raising the default
    should read the V-7 receipt before doing so.
    """
    current_q, current_v, p_target = _setup_resting_contact_scenario(
        _world, root_ctx,
    )
    tracker = tracker_factory(ik_overrides=dict(
        ik_min_distance_lower_bound=0.005,
    ))

    tracker.compute_torque(
        current_q=current_q, current_v=current_v,
        plant_ctx=root_ctx.plant, p_target=p_target, dt_ctrl=DT_CTRL,
    )

    assert tracker.last_knot0_feasible is False
    assert tracker.last_knot0_failure_msg is not None


def test_joint_limit_continuity(_world: World, root_ctx: RootCtx,
                                tracker_factory) -> None:
    """Regression protection for the warm-start chain in _solve_chain.
    The chain seeds q_warm = q_knots[k-1] for k >= 1 (rather than
    q_initial), which prevents the IK from picking a different branch
    on each knot. This test catches a future change that breaks the
    warm-start threading by seeding all knots from a fixed initial
    guess.
    """
    K = 3
    # Override pwl_waypoint_height so the K knots traverse horizontally
    # rather than ascending to z_safe — without this, K=3 ds=0.05 covers
    # only the initial 15cm of vertical lift and the knots never reach
    # workspace-edge poses where joint-limit headroom shrinks.
    tracker = tracker_factory(
        ik_overrides=dict(num_full_ik_knots=K),
        repos_overrides=dict(pwl_waypoint_height=0.04),
    )

    # Pre-position the arm forward via the unconstrained DLS IK, so the
    # K-knot traverse lands at workspace-edge poses (0.80–0.85 m from
    # the Panda base at [0, -0.6, 0]). The DLS helper enforces no
    # min-distance — used as test scenery, not as the system under test.
    n_arm = _world.plant.num_actuators()
    p_start = np.array([0.0, 0.10, 0.05])     # 0.70 m from base
    q_init = _world.plant.GetPositions(root_ctx.plant)
    q_solved, err_norm, _iters = solve_ik_to_ee_pos(
        plant=_world.plant, ee_frame=_world.ee_frame,
        p_target=p_start, q_init=q_init,
        plant_ctx=root_ctx.plant, n_arm_dofs=n_arm,
    )
    assert err_norm < 5e-3, (
        f"Pre-positioning DLS-IK failed: err={err_norm:.4f} m"
    )
    _world.plant.SetPositions(root_ctx.plant, q_solved)

    current_q = _world.plant.GetPositions(root_ctx.plant)
    current_v = _world.plant.GetVelocities(root_ctx.plant)

    # Workspace-edge target. With K=3 and ds=0.05, knot 2 lands at
    # (0, 0.25, 0.05) ≈ 0.851 m from base — near the Panda's ~0.855 m
    # max reach. This forces the IK to extend the arm into configurations
    # where joint-limit headroom is the binding constraint, exercising
    # the IK's joint-limit branch (which is wired but rarely loaded
    # in the production trajectory shape).
    p_target = np.array([0.0, 0.25, 0.05])

    tracker.compute_torque(
        current_q=current_q, current_v=current_v,
        plant_ctx=root_ctx.plant, p_target=p_target, dt_ctrl=DT_CTRL,
    )

    # All K IK-solved knots must be feasible — q_warm holds on
    # failure, so non-feasible knots would give a meaningless
    # continuity check via warm-start carryover.
    feasible = tracker.last_feasible[:K]
    assert all(feasible), (
        f"All {K} IK knots must be feasible to test continuity; got "
        f"feasible={feasible}, msgs={tracker.last_failure_msgs[:K]}"
    )

    n_arm = _world.plant.num_actuators()
    q_knots = tracker.last_q_knots[:, :K]   # (n_arm, K)
    q_lo = _world.plant.GetPositionLowerLimits()[:n_arm]
    q_hi = _world.plant.GetPositionUpperLimits()[:n_arm]
    LIMIT_TOL = 1e-9
    for k in range(K):
        assert np.all(q_knots[:, k] >= q_lo - LIMIT_TOL), (
            f"q_knots[:, {k}] below lower limit: q-q_lo = "
            f"{q_knots[:, k] - q_lo}"
        )
        assert np.all(q_knots[:, k] <= q_hi + LIMIT_TOL), (
            f"q_knots[:, {k}] above upper limit: q_hi-q = "
            f"{q_hi - q_knots[:, k]}"
        )

    BRANCH_BOUND = np.pi / 4   # 45° — IK-branch jumps are multi-radian
    deltas = [
        float(np.max(np.abs(q_knots[:, k + 1] - q_knots[:, k])))
        for k in range(K - 1)
    ]
    max_delta = max(deltas)
    assert max_delta < BRANCH_BOUND, (
        f"IK-branch jump suspected: max ||q[k+1]-q[k]||_inf = "
        f"{max_delta:.3f} rad over knots {deltas}"
    )

    # Diagnostic for 6.4.2 critical-check: visible with `pytest -s`.
    margins = [
        float(min(
            np.min(q_knots[:, k] - q_lo),
            np.min(q_hi - q_knots[:, k]),
        ))
        for k in range(K)
    ]
    print(f"\n[continuity] max ||q[k+1]-q[k]||_inf = {max_delta:.4f} rad "
          f"(deltas = {[f'{d:.4f}' for d in deltas]})")
    print(f"[continuity] min margin to joint limits = {min(margins):.4f} rad "
          f"(per-knot margins = {[f'{m:.4f}' for m in margins]})")


def test_infeasibility_marks_target(_world: World, root_ctx: RootCtx,
                                    tracker_factory) -> None:
    """Regression protection for the wall-clock-timeout failure path.
    Step 5's (c) showed this fires naturally under real load (1 timeout
    in 200 loops at the 8 ms cap). This test deterministically
    reproduces the failure mode by tightening the cap to 1 ms, which
    guarantees overshoot on every solve. Verifies (a) the tracker
    reports the failure correctly and (b) it still returns a finite
    torque rather than NaN. Future maintainers changing the timeout-
    handling code path should run this test and visually inspect the
    failure_msg.
    """
    tracker = tracker_factory(ik_overrides=dict(
        per_knot_solve_timeout_s=TIGHT_TIMEOUT_S,
    ))

    current_q = _world.plant.GetPositions(root_ctx.plant)
    current_v = _world.plant.GetVelocities(root_ctx.plant)
    ee_now = _world.plant.CalcPointsPositions(
        root_ctx.plant, _world.ee_frame, np.zeros(3), _world.plant.world_frame(),
    ).flatten()
    # Reachable target — failure is induced by the cap, NOT the scenario.
    p_target = ee_now + np.array([0.0, 0.05, 0.10])

    u, _diag = tracker.compute_torque(
        current_q=current_q, current_v=current_v,
        plant_ctx=root_ctx.plant, p_target=p_target, dt_ctrl=DT_CTRL,
    )

    assert tracker.last_knot0_feasible is False
    assert tracker.last_knot0_failure_msg == "WALL_CLOCK_TIMEOUT"
    assert np.all(np.isfinite(u))


@pytest.mark.skipif(
    os.environ.get("PUSH_RUN_TIMING_TESTS") != "1",
    reason="Timing tests gated behind PUSH_RUN_TIMING_TESTS=1; "
           "shared CI environments produce unreliable timing.",
)
def test_timing_p99(_world: World, root_ctx: RootCtx, tracker_factory) -> None:
    """Empirical timing budget verification. The 8 ms wall-clock cap is
    load-bearing for the controller's 100 Hz loop; this test ensures IK
    solves stay well under the cap on representative reachable targets.
    Gated behind PUSH_RUN_TIMING_TESTS=1 because shared CI environments
    produce unreliable timing. Run locally for genuine signal; expect
    occasional p99 overshoot under sustained load (V-9 saw 1 timeout in
    200 loops, ~0.5%). The p99 < 9 ms bound absorbs that overshoot rate;
    tightening it would produce flaky tests.
    """
    tracker = tracker_factory()

    current_q = _world.plant.GetPositions(root_ctx.plant)
    current_v = _world.plant.GetVelocities(root_ctx.plant)
    ee_now = _world.plant.CalcPointsPositions(
        root_ctx.plant, _world.ee_frame, np.zeros(3), _world.plant.world_frame(),
    ).flatten()

    # Throwaway call to amortize any cold-state effects beyond the
    # constructor's warm-up Solve. Belt-and-suspenders against subtle
    # IPOPT pre-allocation paths the constructor's pre-pay does not hit.
    tracker.compute_torque(
        current_q=current_q, current_v=current_v,
        plant_ctx=root_ctx.plant,
        p_target=ee_now + np.array([0.01, 0.0, 0.0]),
        dt_ctrl=DT_CTRL,
    )

    N_SAMPLES = 50
    RADIUS    = 0.10        # m, around current EE
    P50_BOUND = 6.0         # ms
    P99_BOUND = 9.0         # ms

    rng = np.random.default_rng(SEED)
    timings = []
    for _ in range(N_SAMPLES):
        # Rejection-sample a uniform 3D offset within the radius ball.
        while True:
            offset = rng.uniform(-RADIUS, RADIUS, size=3)
            if float(np.linalg.norm(offset)) <= RADIUS:
                break
        p_target = ee_now + offset
        tracker.compute_torque(
            current_q=current_q, current_v=current_v,
            plant_ctx=root_ctx.plant, p_target=p_target, dt_ctrl=DT_CTRL,
        )
        # Sanity: timing distribution is meaningless if any solve failed.
        assert tracker.last_knot0_feasible is True, (
            f"Random target produced infeasible IK — test scenario bug. "
            f"target={p_target}, msg={tracker.last_knot0_failure_msg}"
        )
        timings.append(float(tracker.last_knots_solve_ms[0]))

    p50 = float(np.percentile(timings, 50))
    p99 = float(np.percentile(timings, 99))
    print(f"\n[timing] N={N_SAMPLES}  p50={p50:.2f} ms  p99={p99:.2f} ms  "
          f"(bounds: p50 < {P50_BOUND}, p99 < {P99_BOUND})")

    if p50 >= P50_BOUND or p99 >= P99_BOUND:
        # Diagnostic: full sorted list + 10-bin histogram.
        sorted_t = sorted(timings)
        print(f"[timing] sorted (ms): "
              f"{', '.join(f'{t:.2f}' for t in sorted_t)}")
        bin_edges = np.linspace(sorted_t[0], sorted_t[-1], 11)
        counts, _ = np.histogram(timings, bins=bin_edges)
        print(f"[timing] histogram (10 bins from "
              f"{bin_edges[0]:.2f} to {bin_edges[-1]:.2f} ms):")
        for i, c in enumerate(counts):
            print(f"  [{bin_edges[i]:>5.2f}, {bin_edges[i+1]:>5.2f}]  "
                  f"{'#' * int(c)} ({c})")

    assert p50 < P50_BOUND, f"p50={p50:.2f} ms >= {P50_BOUND} ms bound"
    assert p99 < P99_BOUND, f"p99={p99:.2f} ms >= {P99_BOUND} ms bound"
