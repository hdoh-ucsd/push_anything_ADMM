"""
Reposition-mode planner using Drake's ``InverseKinematics`` — sibling to
``PiecewiseLinearTracker`` (in reposition.py).

Selected by ``RepositionParams.traj_type == RepositioningTrajectoryType.kIK``.
The wrapper (sampling_c3.wrapper.SamplingC3MPC) instantiates this when the
YAML config requests it; the contact-rich (C3) path is unchanged.

Per control loop (see ``RepositionIKTracker.compute_torque``):

  1. Build a Cartesian guide path of length N from ee_now → p_target,
     reusing ``reposition.next_waypoint`` (lift → traverse → descend at
     ``repos_params.speed * dt`` per knot — one *planning* timestep). The
     guide is *only* a warm-start for IK; it does not need to be feasible.
  2. Solve K (= ``ik_params.num_full_ik_knots``) full pydrake IK problems
     with warm-start chaining. The manipuland's floating-base 7 DoFs are
     pinned via ``AddBoundingBoxConstraint`` to the values read from
     ``current_q`` so only the 7 arm DoFs are effectively free; the
     min-distance bound therefore enforces arm-vs-manipuland clearance
     against the *current* (pinned) object pose.
  3. Knots K..N-1 are filled by joint-space hold (K=1) or constant-
     velocity extrapolation (K≥2), then FK'd and checked against the
     min-distance bound via SceneGraph signed-distance queries.
  4. Execute knot 0 with the same joint-PD-with-grav-comp law as
     PiecewiseLinearTracker (parameters from ``repos_params``).

Why BoundingBoxConstraint and not Joint.Lock
--------------------------------------------
``scripts/probe_ik_lock.py`` confirmed on this build of pydrake that
``Joint.Lock(plant_ctx)`` does **not** shrink ``InverseKinematics.q()``
— the locked DoFs remain decision variables but are merely fixed during
constraint evaluation. Pinning by ``AddBoundingBoxConstraint(low, high,
q_var[obj_slice])`` with ``low == high == q_obj_value`` produces the
same downstream behavior with one fewer moving part: the contract
``q_var.size == plant.num_positions()`` is invariant, and there's no
"did Lock take?" branch to maintain. We assert the contract at every
solve.

Infeasibility feedback (the only control-flow change touching §IV-D
mode-switching)
---------------------------------------------------------------------
``last_knot0_feasible`` is False after a knot-0 IK failure or timeout.
``wrapper.py`` reads it after ``compute_torque`` and stashes the failed
``p_target``; on the *next* control loop it inflates any sample within
``ik_params.infeasibility_match_radius_m`` of that target to +∞ so the
mode-switch / cost-comparison block sees the unreachable target poisoned.
The poison memo is cleared only when the controller commits to a
different ``p_target`` (>5 mm away — handled in wrapper.py, step 5).

Real-time considerations
------------------------
- Per-knot wall-clock cap via IPOPT's ``max_cpu_time``
  (``per_knot_solve_timeout_s``, default 8 ms).
- IPOPT's structural ``max_iter`` cap (``max_ipopt_iter``, default 30) —
  IPOPT only checks ``max_cpu_time`` between iterations, so an iter cap
  prevents one slow iter from dominating.
- We measure wall-clock around ``Solve()`` ourselves and treat overshoot
  as a hard failure (``knot0_overshoot_ms`` surfaced in diag).
- ``SolverOptions`` is built once at construction; reused every solve.
- The IK ``MathematicalProgram`` is rebuilt per knot (Franka 7-DoF
  construction is sub-ms; the user OK'd this trade in the layout phase).
- The plant's ``plant_ctx`` is mutated during IK (Drake's
  ``InverseKinematics`` calls ``SetPositions`` on it during constraint
  evaluation). We snapshot the full q at entry and restore at exit so
  downstream consumers see the original state.
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np
import pydrake.all as ad

from control.sampling_c3.params import RepositionIKParams
from control.sampling_c3.reposition import next_waypoint


# ---------------------------------------------------------------------------
# Solver-options helpers
# ---------------------------------------------------------------------------

def _set_ipopt_time_limit(solver_options: "ad.SolverOptions",
                          t_seconds:      float) -> None:
    """Set IPOPT's wall-clock cap on a SolverOptions in place.

    IPOPT's ``max_cpu_time`` option is wall-clock under Drake's IpoptSolver
    (despite the name) and is checked between iterations. Solve() can
    therefore overshoot by up to one iteration's worth of work. The
    tracker pairs this with a hard ``time.perf_counter()`` check around
    Solve() and a structural ``max_iter`` cap (``max_ipopt_iter``).
    """
    solver_options.SetOption(
        ad.IpoptSolver.id(), "max_cpu_time", float(t_seconds),
    )


# ---------------------------------------------------------------------------
# Collision-filter helpers
#
# We want the AddMinimumDistanceLowerBoundConstraint inside the IK to enforce
# *arm-vs-obstacle* clearance only — not obstacle-vs-obstacle clearance.
# In this scene the manipuland is normally resting on the table (table-vs-
# manipuland gap ≈ 0), so the global min-distance constraint reads that gap
# as a violation and refuses every IK solve. The filter scopes the constraint
# correctly without us having to switch to per-pair constraints (which would
# fail to auto-include any future tool/finger geometry on the arm).
#
# Context-local: scripts/probe_collision_filter_context.py confirms that
# scene_graph.collision_filter_manager(scene_graph_context).Apply(decl)
# mutates only that context. The live sim's SceneGraph context is not
# affected, even though all contexts share the same SceneGraph instance.
# ---------------------------------------------------------------------------

def _normalize_floating_quaternions(plant, q_full: np.ndarray) -> np.ndarray:
    """Renormalize all floating-base quaternions in q_full to unit norm.

    Why: simulator integration drifts these by O(1e-7) per loop, and
    Drake's IK ``with_joint_limits=True`` implicitly enforces unit norm
    on quaternion-floating bodies. Pinning a non-unit quaternion via
    ``BoundingBoxConstraint(low=q, high=q, …)`` then creates a locally
    infeasible problem; IPOPT rejects it in <1 ms with a NaN solution.
    Diagnosed in 5f V-5 (2026-05-08): the simulator's manipuland quat
    drifted to magnitude 1.0000002004 by loop 21, which was enough.

    Iterates ``plant.GetFloatingBaseBodies()`` and renormalizes the
    quat slice for any body with quaternion DoFs. RPY-floating bodies
    (no quaternion) are skipped. The post-norm assertion catches a
    future Drake API quirk where the quat layout might shift.
    """
    q_out = q_full.copy()
    floating_quat_offsets: list = []
    for body_idx in plant.GetFloatingBaseBodies():
        body = plant.get_body(body_idx)
        if not body.has_quaternion_dofs():
            continue
        s = int(body.floating_positions_start())
        floating_quat_offsets.append(s)
        n = float(np.linalg.norm(q_out[s:s+4]))
        if n > 0.0:
            q_out[s:s+4] = q_out[s:s+4] / n
    # Catch a future bug where the offset/length arithmetic shifts
    # (e.g., new joint type with different floating-base layout).
    assert all(abs(np.linalg.norm(q_out[s:s+4]) - 1.0) < 1e-10
               for s in floating_quat_offsets), \
        "quaternion renormalization failed"
    return q_out


def _collision_geom_ids(plant, scene_graph, body) -> list:
    """Return the GeometryIds of all collision (proximity-role) geometries
    registered on ``body`` (a Drake ``Body``). Helper for resolving the
    obstacle set from user-facing body labels.
    """
    insp = scene_graph.model_inspector()
    frame_id = plant.GetBodyFrameIdOrThrow(body.index())
    return list(insp.GetGeometries(frame_id, ad.Role.kProximity))


def _build_obstacle_collision_filter(plant,
                                     scene_graph,
                                     table_body,
                                     manipuland_body
                                     ) -> Tuple["ad.GeometrySet",
                                                "ad.CollisionFilterDeclaration"]:
    """Build a ``CollisionFilterDeclaration`` that suppresses every collision
    pair *within* the obstacle set (table ∪ manipuland).

    Semantic
    --------
    For RepositionIKTracker, "obstacle" means anything the arm should
    avoid colliding with — currently the table and the floating manipuland.
    Pairs *within* this set (e.g. table-vs-manipuland, which is a
    physically realised resting contact) must NOT gate IK feasibility:
    they are not collisions the arm can avoid by re-planning, and the
    global ``AddMinimumDistanceLowerBoundConstraint`` would otherwise
    refuse every solve because the manipuland sits on the table with
    gap ≈ 0.

    We use ``ExcludeWithin(obstacle_set)`` rather than
    ``ExcludeBetween(arm_set, obstacle_set)`` deliberately:
    ``ExcludeBetween`` is the *inverse* of what we want — it would
    silently disable arm-vs-obstacle clearance, which is exactly the
    constraint we are trying to enforce. ``ExcludeWithin`` removes
    obstacle-vs-obstacle pairs only and leaves arm-vs-obstacle pairs
    intact.

    The returned declaration is *not* applied here. Apply it via
    ``scene_graph.collision_filter_manager(sg_context).Apply(decl)``
    on whichever SceneGraph context should be filtered (typically the
    tracker's private IK context — see RepositionIKTracker.__init__).

    Parameters
    ----------
    plant            : MultibodyPlant (finalised, geometry-source-registered)
    scene_graph      : SceneGraph attached to ``plant``
    table_body       : Drake Body whose collision geoms form the "table" half
                       of the obstacle set. In env_builder.py this is
                       ``plant.world_body()`` (the table is registered on
                       the world body).
    manipuland_body  : Drake Body whose collision geoms form the
                       "manipuland" half of the obstacle set.

    Returns
    -------
    obstacle_set : ``GeometrySet``  — the union of (table, manipuland)
                   collision GeometryIds. Exposed for diagnostics and the
                   post-apply assertion in __init__.
    declaration  : ``CollisionFilterDeclaration``  — pass to
                   ``CollisionFilterManager.Apply()``.

    Raises
    ------
    RuntimeError if either body has zero collision (proximity-role)
    geometries — would silently produce a no-op filter.
    """
    table_ids      = _collision_geom_ids(plant, scene_graph, table_body)
    manipuland_ids = _collision_geom_ids(plant, scene_graph, manipuland_body)
    if not table_ids:
        raise RuntimeError(
            f"_build_obstacle_collision_filter: table_body "
            f"{table_body.name()!r} has no collision geometries. "
            f"Filter would be a no-op."
        )
    if not manipuland_ids:
        raise RuntimeError(
            f"_build_obstacle_collision_filter: manipuland_body "
            f"{manipuland_body.name()!r} has no collision geometries. "
            f"Filter would be a no-op."
        )

    obstacle_set = ad.GeometrySet()
    obstacle_set.Add(table_ids + manipuland_ids)
    declaration = ad.CollisionFilterDeclaration().ExcludeWithin(obstacle_set)
    return obstacle_set, declaration


def _assert_filter_active(plant,
                          scene_graph,
                          plant_ctx,
                          table_body,
                          manipuland_body,
                          query_threshold: float = 1.0) -> None:
    """Verify the filter has taken effect on this plant_ctx's SceneGraph
    context. The dominant failure mode for collision filters is filtering
    the wrong direction (e.g. ExcludeBetween arm-vs-obstacle, which would
    silently disable the constraint we're trying to enforce). Catch it at
    construction.

    Asserts
    -------
    1. NO (table, manipuland) pair appears in
       ``ComputeSignedDistancePairwiseClosestPoints(query_threshold)``.
    2. At least one (arm-or-pusher, table-or-manipuland) pair DOES appear
       — proves arm-vs-obstacle clearance is still being enforced.

    Raises ``RuntimeError`` on either failure.
    """
    insp = scene_graph.model_inspector()
    table_ids      = _collision_geom_ids(plant, scene_graph, table_body)
    manipuland_ids = _collision_geom_ids(plant, scene_graph, manipuland_body)
    table_names      = {insp.GetName(g) for g in table_ids}
    manipuland_names = {insp.GetName(g) for g in manipuland_ids}
    obstacle_names   = table_names | manipuland_names

    qo    = plant.get_geometry_query_input_port().Eval(plant_ctx)
    pairs = qo.ComputeSignedDistancePairwiseClosestPoints(query_threshold)

    bad_table_mani = []
    arm_obstacle_count = 0
    for p in pairs:
        a = insp.GetName(p.id_A)
        b = insp.GetName(p.id_B)
        a_is_table = a in table_names
        b_is_table = b in table_names
        a_is_mani  = a in manipuland_names
        b_is_mani  = b in manipuland_names
        if (a_is_table and b_is_mani) or (a_is_mani and b_is_table):
            bad_table_mani.append((a, b))
            continue
        a_is_obs = a in obstacle_names
        b_is_obs = b in obstacle_names
        if a_is_obs ^ b_is_obs:  # exactly one side is an obstacle
            arm_obstacle_count += 1

    if bad_table_mani:
        raise RuntimeError(
            f"_assert_filter_active: (table, manipuland) pair STILL present "
            f"in {len(bad_table_mani)} entries after filter Apply. "
            f"First offender: {bad_table_mani[0]}. The CollisionFilterDeclaration "
            f"was probably applied to the wrong context, or ExcludeWithin "
            f"was inverted to ExcludeBetween. The IK will be infeasible "
            f"for any state where the manipuland rests on the table."
        )
    if arm_obstacle_count == 0:
        raise RuntimeError(
            f"_assert_filter_active: no (arm, obstacle) pair found within "
            f"distance threshold {query_threshold} m. Either the filter is "
            f"too aggressive (e.g. ExcludeBetween arm-vs-obstacle by "
            f"mistake), or the plant_ctx has the arm in a degenerate state. "
            f"The min-distance IK constraint would be a no-op."
        )


# ---------------------------------------------------------------------------
# Single-knot IK solve
# ---------------------------------------------------------------------------

def _solve_single_knot_ik(plant,
                          plant_ctx,                       # mutated during Solve; caller restores
                          ee_frame,
                          obj_floating_q_start: int,        # plant-q index of the manipuland's first floating-base position
                          q_warm_full:    np.ndarray,       # (n_q,) — supplies arm warm-start AND object pose to pin
                          p_target:       np.ndarray,       # (3,) world-frame EE target
                          ik_params:      RepositionIKParams,
                          solver_options: "ad.SolverOptions",
                          ipopt_solver:   "ad.IpoptSolver",
                          q_prev_arm:     Optional[np.ndarray] = None,  # (n_arm,) for smoothness cost; None on knot 0
                          n_arm_dofs:     int = 7,
                          ) -> Tuple[bool, np.ndarray, float, Optional[str]]:
    """Solve one knot's constrained IK on the arm only (object pose pinned
    by tight ``AddBoundingBoxConstraint``).

    Returns
    -------
    success    : True iff IPOPT reported is_success() **and** wall-clock
                 elapsed ≤ ``ik_params.per_knot_solve_timeout_s``.
    q_arm_sol  : (n_arm_dofs,) joint solution; equals
                 ``q_warm_full[:n_arm]`` on failure (caller can hold).
    elapsed_ms : wall-clock spent inside Solve(), for telemetry.
    failure_msg : None on success. On failure, one of:
                  * "WALL_CLOCK_TIMEOUT" — hard wall-clock cap exceeded
                    (regardless of IPOPT's own success flag).
                  * "SOLVER_LIMIT" — IPOPT did not converge AND
                    ``GetInfeasibleConstraintNames(prog)`` returned
                    empty (typical for max_iter / max_cpu_time hits).
                  * "<name1> | <name2> | …" — pipe-joined Drake
                    constraint names that IPOPT identified as
                    violated. This is the canonical infeasibility
                    signal for the harness's V-2.5 / V-3 read.

    The plant_ctx is left at the IK's last-evaluated configuration on
    return — the caller is responsible for restoring it.
    """
    # Renormalize floating-base quaternions before pinning; see
    # _normalize_floating_quaternions for the why. Without this, the
    # simulator's O(1e-7) integration drift makes every IK call fail
    # in <1 ms (5f V-5).
    q_warm_full = _normalize_floating_quaternions(plant, q_warm_full)
    plant.SetPositions(plant_ctx, q_warm_full)

    # Build a fresh InverseKinematics on this context. The plant_context
    # overload (with_joint_limits=True) wires in joint-limit constraints
    # and gives us SceneGraph access for the min-distance constraint.
    ik = ad.InverseKinematics(plant, plant_ctx, with_joint_limits=True)
    prog = ik.get_mutable_prog()
    q_var = ik.q()
    n_q = plant.num_positions()
    assert q_var.size == n_q, (
        f"InverseKinematics.q().size = {q_var.size}, expected n_q = {n_q}. "
        f"Contract violated — _solve_single_knot_ik assumes the full "
        f"plant-q is the decision-variable vector. See "
        f"scripts/probe_ik_lock.py."
    )

    # 1. Pin the manipuland's floating-base 7 DoFs at their current values.
    #    (low == high == current value → equality constraint expressed as a
    #    tight bounding box. The IK's constraint evaluation reads the q's
    #    set into plant_ctx via SetPositions before each eval, so the
    #    kinematics correctly use the pinned object pose.)
    obj_slice_lo = obj_floating_q_start
    obj_slice_hi = obj_floating_q_start + 7
    q_obj_pinned = q_warm_full[obj_slice_lo:obj_slice_hi]
    prog.AddBoundingBoxConstraint(
        q_obj_pinned, q_obj_pinned, q_var[obj_slice_lo:obj_slice_hi],
    )

    # 2. EE position constraint — tight axis-aligned box of half-width
    #    position_tolerance around p_target.
    eps  = float(ik_params.position_tolerance)
    p_lo = np.asarray(p_target, dtype=float) - eps
    p_hi = np.asarray(p_target, dtype=float) + eps
    ik.AddPositionConstraint(
        ee_frame, np.zeros(3),
        plant.world_frame(),
        p_lo, p_hi,
    )

    # 3. Optional orientation cone. Skipped at default (deg == 0) — see
    #    RepositionIKParams docstring on why disabling it is the right
    #    default for the push_anything pipeline.
    cone_deg = float(ik_params.orientation_cone_deg)
    if cone_deg > 0.0:
        R_raw = np.asarray(ik_params.R_des_world_to_ee, dtype=float)
        R_des = np.eye(3) if R_raw.size == 0 else R_raw.reshape(3, 3)
        ik.AddOrientationConstraint(
            plant.world_frame(), ad.RotationMatrix(R_des),
            ee_frame,            ad.RotationMatrix(),
            float(np.deg2rad(cone_deg)),
        )

    # 4. Min-distance lower bound. Skipped (the typical case) when the
    #    ik-side bound is <= 0 — see class docstring on why this is the
    #    right default for contact-rich manipulation. The FK sweep on
    #    knots K..N-1 still enforces fk_min_distance as a safety net.
    d_min_ik = float(ik_params.ik_min_distance_lower_bound)
    if d_min_ik > 0.0:
        ik.AddMinimumDistanceLowerBoundConstraint(
            d_min_ik, float(ik_params.influence_distance_offset),
        )

    # 5. Costs on the arm DoFs (first n_arm_dofs entries of q_var).
    arm_qvar = q_var[:n_arm_dofs]
    alpha = float(ik_params.joint_centering_weight)
    beta  = float(ik_params.joint_movement_weight)
    if alpha > 0.0:
        q_nom = np.asarray(ik_params.q_nominal, dtype=float)[:n_arm_dofs]
        prog.AddQuadraticErrorCost(alpha * np.eye(n_arm_dofs), q_nom, arm_qvar)
    if beta > 0.0 and q_prev_arm is not None:
        prog.AddQuadraticErrorCost(
            beta * np.eye(n_arm_dofs),
            np.asarray(q_prev_arm, dtype=float)[:n_arm_dofs],
            arm_qvar,
        )

    # 6. Warm-start with the full q (object slice already matches the
    #    BoundingBox pin).
    prog.SetInitialGuess(q_var, q_warm_full)

    # 7. Solve with hard wall-clock cap. IPOPT's max_cpu_time (set in
    #    solver_options) is a soft cap; we enforce a hard cap by clocking
    #    Solve() ourselves.
    t0 = time.perf_counter()
    result = ipopt_solver.Solve(prog, None, solver_options)
    elapsed_ms = (time.perf_counter() - t0) * 1e3
    timed_out = (elapsed_ms / 1e3) > float(ik_params.per_knot_solve_timeout_s)

    if result.is_success() and not timed_out:
        q_sol = result.GetSolution(q_var)
        return True, np.asarray(q_sol[:n_arm_dofs], dtype=float), elapsed_ms, None

    # We deliberately do NOT call result.GetInfeasibleConstraintNames(prog)
    # here. Empirical bisection (5f V-2.5, 2026-05-08) found it has a
    # state-corrupting side effect on plant context state when called
    # repeatedly in a hot loop: 90 calls per 200-loop sim drives the
    # simulator's q vector to NaN within ~100 loops, while removing the
    # call leaves the run bytewise-correct. The behaviour is reproducible
    # with the harness reset to a fixed seed; the full minimal-repro
    # script is a step-7 cleanup task, after which a drake issue can be
    # filed (TODO: insert drake#NNNNN once filed).
    # Out-of-band introspection is exposed via diagnose_failure_at()
    # which builds a fresh diagram context for a one-shot post-run query.
    if timed_out:
        failure_msg = "WALL_CLOCK_TIMEOUT"
    else:
        failure_msg = "DRAKE_INTROSPECT_DEFERRED"
    return False, q_warm_full[:n_arm_dofs].copy(), elapsed_ms, failure_msg


# ---------------------------------------------------------------------------
# RepositionIKTracker — sibling to PiecewiseLinearTracker
# ---------------------------------------------------------------------------

class RepositionIKTracker:
    """Drake-IK-based reposition planner + joint-PD tracker.

    Selected by ``RepositionParams.traj_type == kIK`` in the YAML config.
    Same ``compute_torque(...) -> (u, diag)`` surface as
    ``PiecewiseLinearTracker`` so wrapper.py needs no signature changes;
    only an ``__init__`` dispatch on ``traj_type``.

    Failure modes that surface to the wrapper:
      - ``last_knot0_feasible: bool``  — read after compute_torque(). False
        when knot-0 IK failed or timed out. wrapper.py uses it to poison
        the corresponding sample on the next control loop.
      - ``diag['feasible']``           — list[bool] length N for telemetry.
      - ``diag['any_infeasible']``     — bool, OR over feasible.
      - ``diag['knot0_solve_ms']``     — wall-clock for knot-0's Solve().
      - ``diag['knot0_overshoot_ms']`` — max(0, solve_ms − timeout_s·1000).

    IMPORTANT: simulator-integrated floating-base quaternions drift by
    O(1e-7) per integration step. Drake IK rejects pins on non-unit
    quaternions in <1 ms via implicit unit-norm constraints, causing
    every IK call to fail. We renormalize floating-base quaternions at
    solve entry (see ``_normalize_floating_quaternions`` and its call
    site at the top of ``_solve_single_knot_ik``); do not remove this
    without understanding why.
    """

    # ------------------------------------------------------------------
    # Construction / state management
    # ------------------------------------------------------------------

    def __init__(self,
                 plant,
                 ee_frame,
                 obj_body,
                 n_arm_dofs:    int,
                 horizon:       int,
                 dt:            float,
                 repos_params,                # RepositionParams
                 ik_params:     RepositionIKParams,
                 *,
                 diagram,                     # Drake Diagram (the same one main.py builds)
                 scene_graph,                 # SceneGraph attached to that diagram
                 table_body=None):            # Body whose collision geoms are "table"
                                              # (defaults to plant.world_body() — table
                                              # is registered there in env_builder.py)
        # Flag-3 guard: AddMinimumDistanceLowerBoundConstraint requires the
        # plant to be registered with a SceneGraph. Fail loud at init.
        if not plant.geometry_source_is_registered():
            raise RuntimeError(
                "RepositionIKTracker requires plant.geometry_source_is_registered()"
                " == True (use AddMultibodyPlantSceneGraph). Without"
                " SceneGraph, AddMinimumDistanceLowerBoundConstraint cannot"
                " enforce d_min and the tracker cannot detect collisions"
                " against the table or manipuland. Refusing to construct."
            )

        # ---- Private IK context (context-local collision filter) ----
        # Verified in scripts/probe_collision_filter_context.py: applying a
        # CollisionFilterDeclaration via scene_graph.collision_filter_manager
        # (sg_context).Apply(decl) mutates ONLY that context. The live sim's
        # SceneGraph context (built from the same diagram) is untouched, so
        # any other consumer of pairwise-distance queries on the live
        # plant_ctx still sees the full pair set.
        #
        # The tracker therefore holds its own diagram_ctx for IK. compute_torque
        # snapshots (q, v) from the caller's plant_ctx into _plant_ctx_ik
        # before each solve; IK and the tail-sweep distance checks both run
        # against _plant_ctx_ik so the filter actually takes effect inside
        # AddMinimumDistanceLowerBoundConstraint and inside the manual
        # ComputeSignedDistancePairwiseClosestPoints call in _fk_sweep_tail.
        self._diag_ctx_ik  = diagram.CreateDefaultContext()
        self._plant_ctx_ik = plant.GetMyMutableContextFromRoot(self._diag_ctx_ik)
        self._sg_ctx_ik    = scene_graph.GetMyMutableContextFromRoot(self._diag_ctx_ik)

        # Retain refs for ``diagnose_failure_at`` (post-run, allocates a
        # *fresh* diagram_context — does not reuse the in-run contexts).
        self._diagram      = diagram
        self._scene_graph  = scene_graph

        # Resolve the table body. env_builder.py registers the table on
        # plant.world_body() — that's the default. A subclass / future
        # builder that registers the table on its own body can pass it in.
        if table_body is None:
            table_body = plant.world_body()
        # Retain for ``diagnose_failure_at`` so the fresh context can
        # rebuild the same obstacle filter.
        self._table_body = table_body

        # Build and apply the obstacle filter on the private SG context.
        # The helper validates that both bodies actually have collision
        # geometries; the assertion below validates that the filter took
        # effect (catches the dominant ExcludeWithin/ExcludeBetween bug).
        self._obstacle_set, _decl = _build_obstacle_collision_filter(
            plant, scene_graph, table_body, obj_body,
        )
        scene_graph.collision_filter_manager(self._sg_ctx_ik).Apply(_decl)
        _assert_filter_active(
            plant, scene_graph, self._plant_ctx_ik, table_body, obj_body,
        )

        # The manipuland is a free body — its floating-base q's start at
        # this index in the full plant q. We pin them via BoundingBox in
        # _solve_single_knot_ik (see why-not-Lock note in module docstring).
        # Prefer the post-2026-06 API ``is_floating_base_body`` and fall
        # back to the deprecated ``is_floating`` for older Drake.
        is_floater = getattr(obj_body, "is_floating_base_body", None)
        is_floating = is_floater() if is_floater is not None else obj_body.is_floating()
        if not is_floating:
            raise RuntimeError(
                f"RepositionIKTracker expected obj_body {obj_body.name()!r} to"
                f" be a floating-base body. Got is_floating[_base_body]()=False."
            )
        self._obj_floating_q_start: int = int(obj_body.floating_positions_start())

        self.plant       = plant
        self.world_frame = plant.world_frame()
        self.ee_frame    = ee_frame
        self.obj_body    = obj_body
        self.n_arm_dofs  = int(n_arm_dofs)
        self.horizon     = int(horizon)
        self.dt          = float(dt)
        self.repos_params = repos_params
        self.ik_params   = ik_params

        # Solver setup — built once, reused every solve.
        self._ipopt_solver   = ad.IpoptSolver()
        self._solver_options = ad.SolverOptions()
        _set_ipopt_time_limit(
            self._solver_options, ik_params.per_knot_solve_timeout_s,
        )
        # Structural max_iter cap (the user's 4g addition) — IPOPT only
        # checks max_cpu_time between iterations, so an iter cap prevents
        # one slow iter from dominating the loop budget.
        self._solver_options.SetOption(
            ad.IpoptSolver.id(), "max_iter", int(ik_params.max_ipopt_iter),
        )
        # Quiet IPOPT — keep the [GS] log clean.
        self._solver_options.SetOption(ad.IpoptSolver.id(), "print_level", 0)
        self._solver_options.SetOption(ad.IpoptSolver.id(), "sb", "yes")

        # ---- Joint-PD state (mirrors PiecewiseLinearTracker) ----
        self._integral:        np.ndarray            = np.zeros(self.n_arm_dofs)
        self._prev_target_pos: Optional[np.ndarray]  = None

        # ---- Feasibility memo (read by wrapper.py) ----
        self._last_knot0_feasible: bool = True

        # ---- Latest-plan memos (read by diagnostics) ----
        self.last_q_knots:         Optional[np.ndarray] = None
        self.last_ee_knots:        Optional[np.ndarray] = None
        self.last_feasible:        Optional[list]       = None
        self.last_knots_solve_ms:  Optional[list]       = None
        # Failure-cause memo. Parallel to ``last_feasible`` — entry i is
        # None when knot i is feasible, or a sentinel / Drake
        # constraint-name string on failure. The
        # ``last_knot0_failure_msg`` property exposes the [0] slot
        # (the only one wrapper.py / 5c reads). Tail entries
        # (knots K..N-1) carry None — those slots come from FK + signed
        # distance, not IK.
        self.last_failure_msgs:    Optional[list]       = None
        # Failure-input memo. Parallel to ``last_failure_msgs`` — entry i
        # is None on success, or a (q_warm_full, p_target) tuple on
        # failure. Read via ``last_knot0_failure_inputs`` after the run
        # to drive ``diagnose_failure_at`` for the one-shot Drake-API
        # introspection that is unsafe to call in the hot path.
        self.last_failure_inputs:  Optional[list]       = None

        # ---- IPOPT warm-up (one-shot startup cost) ----
        # IPOPT's first Solve() call in a process pays a 15-25 ms cold-start
        # penalty (vs ~6 ms warm) — enough to overshoot the 8 ms production
        # cap on the very first compute_torque() at t=0. We pre-pay it here
        # with a trivially-feasible target (the FK of the current arm pose,
        # so q_warm_full IS itself a solution), running through the same
        # _solve_single_knot_ik and the production solver_options. The
        # result is discarded; only the side-effect on IPOPT's internal
        # warm-state matters.
        if bool(ik_params.warm_up_on_construction):
            q_warm_full = self.plant.GetPositions(self._plant_ctx_ik)
            self.plant.SetPositions(self._plant_ctx_ik, q_warm_full)
            p_warm = self.plant.CalcPointsPositions(
                self._plant_ctx_ik, self.ee_frame, np.zeros(3), self.world_frame,
            ).flatten()
            _ok, _q, warm_ms, _msg = _solve_single_knot_ik(
                plant=self.plant,
                plant_ctx=self._plant_ctx_ik,
                ee_frame=self.ee_frame,
                obj_floating_q_start=self._obj_floating_q_start,
                q_warm_full=q_warm_full,
                p_target=p_warm,
                ik_params=ik_params,
                solver_options=self._solver_options,
                ipopt_solver=self._ipopt_solver,
                q_prev_arm=None,
                n_arm_dofs=self.n_arm_dofs,
            )
            cap_ms = 1e3 * float(ik_params.per_knot_solve_timeout_s)
            if warm_ms > cap_ms:
                # Diagnostic only — cold-start overshoot is expected and the
                # whole point of running this. Don't raise.
                print(f"[RepositionIK] WARNING: IPOPT warm-up solve "
                      f"completed in {warm_ms:.1f} ms (>{cap_ms:.1f} ms cap). "
                      f"Subsequent in-loop solves should hit the warm path.")
            else:
                print(f"[RepositionIK] IPOPT warm-up solve completed in "
                      f"{warm_ms:.1f} ms")

    # ------------------------------------------------------------------
    @property
    def last_knot0_feasible(self) -> bool:
        """Single-attribute interface to wrapper.py."""
        return self._last_knot0_feasible

    @property
    def last_knot0_failure_inputs(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """The (q_warm_full, p_target) pair fed into the most recent
        knot-0 IK call when it failed. ``None`` on success or before
        any compute_torque has run. Used by ``diagnose_failure_at``
        for post-run Drake-API introspection.
        """
        if not self.last_failure_inputs:
            return None
        return self.last_failure_inputs[0]

    @property
    def last_knot0_failure_msg(self) -> Optional[str]:
        """Failure-cause string for knot 0 of the most-recent
        compute_torque call. ``None`` when knot 0 was feasible
        (paired with ``last_knot0_feasible == True``). On failure,
        one of:
          * ``"WALL_CLOCK_TIMEOUT"`` — hard cap exceeded.
          * ``"SOLVER_LIMIT"`` — IPOPT didn't converge AND
            ``GetInfeasibleConstraintNames(prog)`` returned empty.
          * Pipe-joined Drake constraint names — IPOPT identified
            specific violated constraint(s); this is the diagnostic
            signal the harness reads.
        """
        if not self.last_failure_msgs:
            return None
        return self.last_failure_msgs[0]

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Wipe joint-PD state and feasibility memos."""
        self._integral[:]          = 0.0
        self._prev_target_pos      = None
        self._last_knot0_feasible  = True
        self.last_q_knots          = None
        self.last_ee_knots         = None
        self.last_feasible         = None
        self.last_knots_solve_ms   = None
        self.last_failure_msgs     = None
        self.last_failure_inputs   = None

    # ------------------------------------------------------------------
    # Post-run failure introspection (Tightening 2)
    # ------------------------------------------------------------------

    def diagnose_failure_at(self,
                            q_full:   np.ndarray,
                            p_target: np.ndarray,
                            *,
                            with_min_distance:    bool = True,
                            position_tolerance_m: Optional[float] = None) -> dict:
        """Diagnose an IK failure POST-RUN. Allocates fresh diagram /
        plant / scene_graph contexts, applies the same obstacle filter,
        and computes constraint diagnostics AT THE WARM-START (q_full)
        instead of at the IK's solution.

        Why warm-start instead of solution: ``GetInfeasibleConstraintNames``
        calls ``SetPositions`` internally to evaluate each constraint at
        the IK solution, but on a fast-fail solve (e.g. our 0.5 ms knot-0
        bails) the solver returns ``q_solution`` with NaN values, and
        the internal SetPositions then trips ``AllFinite(q)`` and aborts
        the process. Empirically reproducible in 5f V-2.5 and not safe
        to call. Reading constraint violations at q_full is just as
        informative for V-3's tolerance-vs-distance disambiguation:
          - if position_error_at_warm_start > 1 mm tolerance, the IK
            was being asked to make a big jump, and a damped-pseudo-
            inverse would also have a hard time;
          - if min_signed_distance_at_warm_start < d_min, the warm-start
            already violates clearance — IPOPT bails before it can move.

        Returns
        -------
        dict with keys:
            position_error_at_warm_start_m         float
            min_signed_distance_at_warm_start_m    float | None
            min_distance_pair_at_warm_start        (str, str) | None
            ik_d_min_setting_m                     float
            position_tolerance_setting_m           float
            ee_pos_at_warm_start                   np.ndarray
            p_target                               np.ndarray
            q_arm_at_warm_start                    np.ndarray
        """
        # Fresh contexts — do NOT reuse self._plant_ctx_ik /
        # self._sg_ctx_ik (they carry state from the in-run solves).
        diag_ctx  = self._diagram.CreateDefaultContext()
        plant_ctx = self.plant.GetMyMutableContextFromRoot(diag_ctx)
        sg_ctx    = self._scene_graph.GetMyMutableContextFromRoot(diag_ctx)

        _obstacle_set, decl = _build_obstacle_collision_filter(
            self.plant, self._scene_graph, self._table_body, self.obj_body,
        )
        self._scene_graph.collision_filter_manager(sg_ctx).Apply(decl)

        # Mirror _solve_single_knot_ik's renormalization so the
        # diagnostic doesn't recreate the bug it's trying to diagnose.
        q_full = _normalize_floating_quaternions(self.plant, q_full)
        self.plant.SetPositions(plant_ctx, q_full)

        # FK the EE at the warm-start. p_target is the loop's IK target.
        p_ee = self.plant.CalcPointsPositions(
            plant_ctx, self.ee_frame, np.zeros(3), self.world_frame,
        ).flatten()
        position_err = float(np.linalg.norm(p_ee - np.asarray(p_target)))

        # Min signed distance over arm-vs-obstacle pairs (table-vs-mani
        # filtered out by the collision filter on sg_ctx). We sample
        # using the IK-side bound (the diagnostic mirrors what the
        # in-loop IK actually enforces); query threshold uses a generous
        # multiplier of the influence distance so distant pairs are
        # still surfaced.
        d_min = float(self.ik_params.ik_min_distance_lower_bound)
        infl  = float(self.ik_params.influence_distance_offset)
        qo = self.plant.get_geometry_query_input_port().Eval(plant_ctx)
        pairs = qo.ComputeSignedDistancePairwiseClosestPoints(d_min + 10.0 * infl)
        min_dist = None
        min_pair = None
        if pairs:
            best = min(pairs, key=lambda p: p.distance)
            insp = self._scene_graph.model_inspector()
            min_dist = float(best.distance)
            min_pair = (insp.GetName(best.id_A), insp.GetName(best.id_B))

        # Optionally also run the IK so V-4 can compare solve outcomes
        # under varying constraint configurations.
        eff_tol = (float(self.ik_params.position_tolerance)
                   if position_tolerance_m is None
                   else float(position_tolerance_m))
        ik = ad.InverseKinematics(self.plant, plant_ctx, with_joint_limits=True)
        prog = ik.get_mutable_prog()
        q_var = ik.q()
        s = self._obj_floating_q_start
        prog.AddBoundingBoxConstraint(q_full[s:s+7], q_full[s:s+7], q_var[s:s+7])
        ik.AddPositionConstraint(
            self.ee_frame, np.zeros(3), self.world_frame,
            np.asarray(p_target) - eff_tol, np.asarray(p_target) + eff_tol,
        )
        if with_min_distance and d_min > 0.0:
            ik.AddMinimumDistanceLowerBoundConstraint(d_min, infl)
        prog.SetInitialGuess(q_var, q_full)

        t0 = time.perf_counter()
        result = self._ipopt_solver.Solve(prog, None, self._solver_options)
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        is_success = bool(result.is_success())

        # Solution-side metrics. Guarded by np.isfinite — Drake's
        # SetPositions trips AllFinite if any q is NaN, and IPOPT-on-
        # fast-fail returns NaN solutions. Skip these fields entirely
        # when the solution isn't finite (we still get is_success and
        # elapsed_ms, which is enough to bisect).
        pos_err_sol  = None
        min_dist_sol = None
        min_pair_sol = None
        try:
            q_sol = np.asarray(result.GetSolution(q_var), dtype=float)
        except Exception:
            q_sol = None
        if q_sol is not None and bool(np.all(np.isfinite(q_sol))):
            self.plant.SetPositions(plant_ctx, q_sol)
            p_ee_sol = self.plant.CalcPointsPositions(
                plant_ctx, self.ee_frame, np.zeros(3), self.world_frame,
            ).flatten()
            pos_err_sol = float(np.linalg.norm(p_ee_sol - np.asarray(p_target)))
            qo_sol = self.plant.get_geometry_query_input_port().Eval(plant_ctx)
            pairs_sol = qo_sol.ComputeSignedDistancePairwiseClosestPoints(
                d_min + 10.0 * infl)
            if pairs_sol:
                best_sol = min(pairs_sol, key=lambda p: p.distance)
                insp_sol = self._scene_graph.model_inspector()
                min_dist_sol = float(best_sol.distance)
                min_pair_sol = (insp_sol.GetName(best_sol.id_A),
                                insp_sol.GetName(best_sol.id_B))

        return dict(
            # Warm-start metrics (always finite, computed before Solve).
            position_error_at_warm_start_m       = position_err,
            min_signed_distance_at_warm_start_m  = min_dist,
            min_distance_pair_at_warm_start      = min_pair,
            ik_d_min_setting_m                   = d_min,
            position_tolerance_setting_m         = float(self.ik_params.position_tolerance),
            position_tolerance_used_m            = eff_tol,
            with_min_distance                    = bool(with_min_distance),
            ee_pos_at_warm_start                 = p_ee.copy(),
            p_target                             = np.asarray(p_target).copy(),
            q_arm_at_warm_start                  = q_full[:self.n_arm_dofs].copy(),
            # Solve outcome (V-4 bisection).
            is_success                           = is_success,
            elapsed_ms                           = float(elapsed_ms),
            position_error_at_solution_m         = pos_err_sol,
            min_signed_distance_at_solution_m    = min_dist_sol,
            min_distance_pair_at_solution        = min_pair_sol,
        )

    # ------------------------------------------------------------------
    # 4c — Cartesian guide path
    # ------------------------------------------------------------------

    def _build_guide_path(self,
                          ee_now:    np.ndarray,
                          p_target:  np.ndarray) -> np.ndarray:
        """Return a (3, N) Cartesian guide. Each knot advances
        ``repos_params.speed * self.dt`` metres along the lift-traverse-
        descend PWL path (z_safe = ``repos_params.pwl_waypoint_height``)
        from ee_now toward p_target. Knots clamp at p_target once reached.

        ``self.dt`` (planning timestep) — not ``dt_ctrl`` — is the right
        per-knot stride: the upstream design treats N-knots as the C3
        horizon (1.0 s at dt=0.05 s). Each knot is therefore one *planning*
        step ahead of the previous.

        The guide is *only* a warm-start for IK; it does not need to be
        feasible or collision-free.
        """
        ds = float(self.repos_params.speed) * float(self.dt)
        N  = self.horizon
        z_safe = float(self.repos_params.pwl_waypoint_height)
        thresh = float(self.repos_params.use_straight_line_traj_under_piecewise_linear)

        p_guide = np.zeros((3, N))
        p_curr  = ee_now.copy()
        for i in range(N):
            p_next = next_waypoint(
                p_now=p_curr,
                p_target=p_target,
                z_safe=z_safe,
                ds=ds,
                straight_line_thresh=thresh,
            )
            p_guide[:, i] = p_next
            p_curr = p_next
        return p_guide

    # ------------------------------------------------------------------
    # 4d — Full-IK warm-start chain (knots 0..K-1)
    # ------------------------------------------------------------------

    def _solve_chain(self,
                     plant_ctx,
                     q_warm_full: np.ndarray,
                     p_guide:     np.ndarray
                     ) -> Tuple[np.ndarray, list, list, int, list, list]:
        """Solve K = ``ik_params.num_full_ik_knots`` IK problems with
        warm-start chaining. Knot 0 warms from ``q_warm_full[:n_arm]``;
        knot i (i > 0) warms from knot-(i-1)'s solution.

        Returns ``(q_arm_solved (n_arm, K), feasible[:K], solve_ms[:K],
                  consec_failures, failure_msgs[:K], failure_inputs[:K])``.
        ``failure_msgs[i]`` is None on success or a sentinel / Drake
        constraint-name list on failure. ``failure_inputs[i]`` is None
        on success or a ``(q_warm_full_for_solve, p_target)`` tuple on
        failure — captured so the harness can post-run introspect via
        ``diagnose_failure_at``.
        """
        K     = min(int(self.ik_params.num_full_ik_knots), p_guide.shape[1])
        n_arm = self.n_arm_dofs
        max_consec = int(self.ik_params.max_consecutive_failures_before_abort)

        q_arm_solved   = np.zeros((n_arm, K))
        feasible       = []
        solve_ms       = []
        failure_msgs:    list = []
        failure_inputs:  list = []  # parallel to failure_msgs (Tightening 1)
        consec_failures = 0

        q_warm_arm_curr     = q_warm_full[:n_arm].copy()
        q_prev_for_smooth   = None  # smoothness cost is None on knot 0

        for i in range(K):
            q_warm_for_solve = q_warm_full.copy()
            q_warm_for_solve[:n_arm] = q_warm_arm_curr

            success, q_arm_sol, elapsed_ms, failure_msg = _solve_single_knot_ik(
                plant=self.plant,
                plant_ctx=plant_ctx,
                ee_frame=self.ee_frame,
                obj_floating_q_start=self._obj_floating_q_start,
                q_warm_full=q_warm_for_solve,
                p_target=p_guide[:, i],
                ik_params=self.ik_params,
                solver_options=self._solver_options,
                ipopt_solver=self._ipopt_solver,
                q_prev_arm=q_prev_for_smooth,
                n_arm_dofs=n_arm,
            )

            q_arm_solved[:, i] = q_arm_sol
            feasible.append(bool(success))
            solve_ms.append(float(elapsed_ms))
            failure_msgs.append(failure_msg)
            failure_inputs.append(
                None if success
                else (q_warm_for_solve.copy(), np.asarray(p_guide[:, i]).copy())
            )

            if success:
                consec_failures   = 0
                q_warm_arm_curr   = q_arm_sol.copy()
                q_prev_for_smooth = q_arm_sol.copy()
            else:
                consec_failures += 1
                # q_arm_sol equals q_warm_arm_curr on failure (held).
                # The cascade-abort below activates only when K ≥ 2.
                if K >= 2 and consec_failures >= max_consec:
                    for j in range(i + 1, K):
                        q_arm_solved[:, j] = q_warm_arm_curr
                        feasible.append(False)
                        solve_ms.append(0.0)
                        failure_msgs.append("CASCADE_ABORT")
                        failure_inputs.append(
                            (q_warm_full.copy(), np.asarray(p_guide[:, j]).copy())
                        )
                    break

        return q_arm_solved, feasible, solve_ms, consec_failures, failure_msgs, failure_inputs

    # ------------------------------------------------------------------
    # 4e — Joint-space tail with FK + signed-distance check (knots K..N-1)
    # ------------------------------------------------------------------

    def _fk_sweep_tail(self,
                       plant_ctx,
                       q_arm_last:      np.ndarray,
                       q_arm_prev:      Optional[np.ndarray],
                       N_tail:          int,
                       q_full_template: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray, list]:
        """Fill knots K..N-1.

        K = 1 (q_arm_prev is None): hold q_arm_last (Override 3 default).
        K ≥ 2: extrapolate at velocity (q_arm_last − q_arm_prev) per knot.

        For each knot, FK + signed-distance check via the SceneGraph
        ``QueryObject``. Marks ``feasible[j]=False`` if any pair drops
        below ``ik_params.fk_min_distance``.

        plant_ctx is mutated during this sweep; restored by the caller.
        """
        n_arm = self.n_arm_dofs
        if q_arm_prev is None:
            v_arm = np.zeros(n_arm)
        else:
            v_arm = q_arm_last - q_arm_prev

        q_lo = self.plant.GetPositionLowerLimits()[:n_arm]
        q_hi = self.plant.GetPositionUpperLimits()[:n_arm]

        # FK-sweep safety net (separate from IK-side enforcement; see
        # RepositionIKParams docstring). Uses fk_min_distance, NOT
        # ik_min_distance_lower_bound — those are intentionally split.
        d_min   = float(self.ik_params.fk_min_distance)
        infl    = float(self.ik_params.influence_distance_offset)
        check_d = d_min > 0.0

        q_arm_tail   = np.zeros((n_arm, N_tail))
        ee_tail      = np.zeros((3, N_tail))
        feasible_tail: list = []

        for j in range(N_tail):
            q_arm_j = np.clip(q_arm_last + (j + 1) * v_arm, q_lo, q_hi)
            q_arm_tail[:, j] = q_arm_j

            if check_d:
                # FK + per-pair signed-distance check. FK is gated on
                # check_d because ee_tail is read only by the SDF block
                # below; no external consumer reads last_ee_knots' tail
                # under the C-2 default (verified via grep, May 2026).
                # Under d_min == 0.0 ee_tail entries remain at their
                # zero-init values, which is documented behavior.
                q_full = q_full_template.copy()
                q_full[:n_arm] = q_arm_j
                self.plant.SetPositions(plant_ctx, q_full)
                ee_pos = self.plant.CalcPointsPositions(
                    plant_ctx, self.ee_frame, np.zeros(3), self.world_frame,
                ).flatten()
                ee_tail[:, j] = ee_pos

                feasible_j = True
                qo = self.plant.get_geometry_query_input_port().Eval(plant_ctx)
                pairs = qo.ComputeSignedDistancePairwiseClosestPoints(d_min + infl)
                for pair in pairs:
                    if pair.distance < d_min:
                        feasible_j = False
                        break
            else:
                feasible_j = True
            feasible_tail.append(feasible_j)

        return q_arm_tail, ee_tail, feasible_tail

    # ------------------------------------------------------------------
    # 4f — Main entry: stitch 4c → 4d → 4e and drive the PD law on knot 0
    # ------------------------------------------------------------------

    def compute_torque(self,
                       current_q:  np.ndarray,
                       current_v:  np.ndarray,
                       plant_ctx,
                       p_target:   np.ndarray,
                       dt_ctrl:    float
                       ) -> Tuple[np.ndarray, dict]:
        """Plan an N-knot joint trajectory then execute knot 0 with
        joint-PD-with-grav-comp. Same return contract as
        PiecewiseLinearTracker.compute_torque.

        Side effect: leaves ``plant_ctx`` at the original ``current_q``
        on return so downstream callers see the input state.
        """
        n_arm = self.n_arm_dofs
        N     = self.horizon

        # 1. FK current EE (also seeds plant_ctx for downstream code paths).
        self.plant.SetPositions(plant_ctx, current_q)
        ee_now = self.plant.CalcPointsPositions(
            plant_ctx, self.ee_frame, np.zeros(3), self.world_frame,
        ).flatten()

        # 2. Reset PD integral on target change (matches PWL tracker).
        target_changed = (
            self._prev_target_pos is None
            or float(np.linalg.norm(p_target - self._prev_target_pos)) > 1e-3
        )
        if target_changed:
            self._integral[:]     = 0.0
            self._prev_target_pos = p_target.copy()

        # 3. Cartesian guide (warm-start only).
        p_guide = self._build_guide_path(ee_now, p_target)

        # 4. Full-IK chain on knots 0..K-1. Routed through the tracker's
        #    private _plant_ctx_ik so the context-local collision filter
        #    (applied in __init__) takes effect inside
        #    AddMinimumDistanceLowerBoundConstraint. The caller's plant_ctx
        #    is left untouched by the IK chain.
        q_arm_solved, feasible_solved, solve_ms, _consec, \
            failure_msgs_solved, failure_inputs_solved = \
            self._solve_chain(self._plant_ctx_ik, current_q, p_guide)
        K = q_arm_solved.shape[1]

        # 5. FK head knots (FK uses the solved arm q's, object pose pinned
        #    to current). Run on _plant_ctx_ik for symmetry with the IK
        #    chain — FK doesn't need the filter, but routing through the
        #    private context keeps the caller's plant_ctx clean.
        ee_head = np.zeros((3, K))
        for i in range(K):
            q_full = current_q.copy()
            q_full[:n_arm] = q_arm_solved[:, i]
            self.plant.SetPositions(self._plant_ctx_ik, q_full)
            ee_head[:, i] = self.plant.CalcPointsPositions(
                self._plant_ctx_ik, self.ee_frame, np.zeros(3), self.world_frame,
            ).flatten()

        # 6. Tail (knots K..N-1) by joint-space hold/extrapolation + FK.
        #    Same rationale as step 5 — the tail's manual
        #    ComputeSignedDistancePairwiseClosestPoints check must run on
        #    _plant_ctx_ik so the (table, manipuland) pair stays filtered.
        N_tail = N - K
        if N_tail > 0:
            q_arm_last = q_arm_solved[:, K - 1]
            q_arm_prev = q_arm_solved[:, K - 2] if K >= 2 else None
            q_arm_tail, ee_tail, feasible_tail = self._fk_sweep_tail(
                self._plant_ctx_ik, q_arm_last, q_arm_prev, N_tail, current_q,
            )
        else:
            q_arm_tail    = np.zeros((n_arm, 0))
            ee_tail       = np.zeros((3, 0))
            feasible_tail = []

        # 7. Assemble full plan.
        q_arm_knots = np.concatenate([q_arm_solved, q_arm_tail], axis=1)
        ee_knots    = np.concatenate([ee_head,      ee_tail],    axis=1)
        feasible    = list(feasible_solved) + list(feasible_tail)
        any_infeasible = (not all(feasible)) if feasible else True

        # 8. Update knot-0 feasibility (single-attribute interface).
        self._last_knot0_feasible = bool(feasible[0]) if feasible else False

        # 9. Memos for diagnostics.
        self.last_q_knots        = q_arm_knots
        self.last_ee_knots       = ee_knots
        self.last_feasible       = feasible
        self.last_knots_solve_ms = solve_ms
        # Failure-cause memo. Tail knots (K..N-1) come from FK +
        # signed-distance, not IK — pad with None for those slots so
        # the list length matches ``feasible``. ``last_knot0_failure_msg``
        # property reads the [0] slot.
        self.last_failure_msgs   = list(failure_msgs_solved) + [None] * N_tail
        # Failure-input memo (Tightening 1). Pad with None for tail slots.
        self.last_failure_inputs = list(failure_inputs_solved) + [None] * N_tail

        # 10. (formerly: restore plant_ctx to current_q.) The IK chain runs
        #     against self._plant_ctx_ik; the caller's plant_ctx is set to
        #     current_q at the top of compute_torque and not mutated
        #     thereafter, so the prior SetPositions/SetVelocities here were
        #     no-ops. CalcGravityGeneralizedForces below is position-only;
        #     no velocity write is needed.

        # 11. Joint-PD-with-grav-comp on knot 0. Identical control law to
        #     PiecewiseLinearTracker — only the q_target source differs.
        q_arm_now    = current_q[:n_arm]
        v_arm_now    = current_v[:n_arm]
        q_arm_target = q_arm_knots[:, 0]

        q_err = q_arm_target - q_arm_now
        self._integral += q_err * dt_ctrl
        np.clip(self._integral,
                -self.repos_params.I_max, self.repos_params.I_max,
                out=self._integral)

        u_p  = self.repos_params.Kp_q * q_err
        u_i  = self.repos_params.Ki_q * self._integral
        u_d  = -self.repos_params.Kd_q * v_arm_now
        u_pd = u_p + u_i + u_d

        # Step 8 Fix 4: anticipate gravity load at the IK target rather than
        # the current measured config. Counters the steady-state z-tracking
        # error characterized in 8.3 (arm settles at z≈25mm regardless of
        # the 50mm reference) by feeding the controller the gravity load it
        # SHOULD see at q_arm_target rather than the load at q_arm_now.
        # _plant_ctx_ik is the tracker's private context; nothing past this
        # point in compute_torque reads from it (verified at 8.4.1).
        q_full_target = current_q.copy()
        q_full_target[:n_arm] = q_arm_target
        self.plant.SetPositions(self._plant_ctx_ik, q_full_target)
        tau_g_arm = self.plant.CalcGravityGeneralizedForces(self._plant_ctx_ik)[:n_arm]
        u = np.clip(tau_g_arm + u_pd,
                    -self.repos_params.torque_limit,
                    +self.repos_params.torque_limit)

        # 12. Trajectory-finished signal (mirrors PWL tracker @ 2 cm).
        finished = float(np.linalg.norm(p_target - ee_now)) <= 0.02

        # 13. Diagnostics. Two distinct timing keys per the user's
        #     addition: knot0_solve_ms (raw elapsed) AND knot0_overshoot_ms
        #     (max(0, raw − cap·1000)). test_timing_p99 should assert p99
        #     of each separately — don't collapse them.
        knot0_solve_ms = float(solve_ms[0]) if solve_ms else 0.0
        timeout_ms     = 1e3 * float(self.ik_params.per_knot_solve_timeout_s)
        knot0_overshoot_ms = max(0.0, knot0_solve_ms - timeout_ms)

        p_des  = ee_knots[:, 0] if ee_knots.shape[1] > 0 else p_target.copy()
        ik_err = float(np.linalg.norm(p_des - p_guide[:, 0])) if K > 0 else 0.0

        diag = dict(
            finished           = bool(finished),
            ee_now             = ee_now,
            p_des              = p_des,
            ik_err             = ik_err,
            ik_iters           = 1 if (feasible and feasible[0]) else 0,
            q_knots            = q_arm_knots,
            ee_knots           = ee_knots,
            feasible           = feasible,
            any_infeasible     = bool(any_infeasible),
            knot0_feasible     = self._last_knot0_feasible,
            knots_solve_ms     = solve_ms,
            knot0_solve_ms     = knot0_solve_ms,
            knot0_overshoot_ms = knot0_overshoot_ms,
        )
        return u, diag
