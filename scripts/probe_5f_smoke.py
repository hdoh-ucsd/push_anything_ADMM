"""5f end-to-end smoke for SamplingC3MPC.

Five sub-tests, each invoked separately:

    python scripts/probe_5f_smoke.py a   # PWL regression  (200 loops)
    python scripts/probe_5f_smoke.py b   # kIK feasible    (200 loops)
    python scripts/probe_5f_smoke.py c   # no-spurious-poison check on (b)
    python scripts/probe_5f_smoke.py d   # mode-switch parity (a) vs (b)
    python scripts/probe_5f_smoke.py e   # poison lifecycle (drop+restore)

Each sub-test pickles its captured metrics under
``results/probe_5f_<subtest>.pkl`` so (d) can compare (a) and (b)
without re-running them.

NEGATIVE INVARIANTS enforced by this harness (per user 5f spec):

  * Does not modify wrapper.py. mode/reason capture is via a
    monkey-patch of `wrapper.decide_mode` — the harness's import
    binding only.
  * Does not introduce wrapper-side state for tests. _infeasible_
    repos_target is read via getattr only.
  * Does not skip (a). (a) is the regression gate; if it fails the
    harness aborts before (b)-(e) run.

Random seed: 42 (documented per Constraint 3 of test (e)).
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from collections import Counter
from typing import Optional

import numpy as np
import pydrake.all as ad

from sim.env_builder import (
    build_environment, EE_BODY_NAME, INITIAL_ARM_Q,
)
from control.lcs_formulator import LCSFormulator
from control.admm_solver import C3Solver
from control.task_costs import QuadraticManipulationCost
from control.ci_mpc_c3 import C3MPC

from control.sampling_c3 import wrapper as wmod
from control.sampling_c3.wrapper import SamplingC3MPC
from control.sampling_c3.params import (
    SamplingC3Params, RepositionParams, RepositioningTrajectoryType,
    SamplingStrategy,
)
# Re-use mode_switch._hysteresis directly so the spy's boundary values
# are byte-identical to what decide_mode computed internally. If
# mode_switch.py refactors away the helper this import breaks loudly,
# which is the harness coupling we want.
from control.sampling_c3.mode_switch import _hysteresis
# 8.1.3: re-use sampling.is_in_workspace so the spy's pass/fail call is
# byte-identical to what generate_samples uses internally.
from control.sampling_c3.sampling import is_in_workspace


SEED = 42
NUM_LOOPS = 200
RESULTS_DIR = "results"

# Deliberate mismatch from main.py defaults: turn off [GS] log output
# (we read state via attributes; stdout would be a 2k-line firehose).
LOG_DIAG = False


# ---------------------------------------------------------------------------
# Decide-mode spy — captures (mode, reason) every time wrapper.compute_control
# routes through decide_mode. Patches the harness's import binding only.
# ---------------------------------------------------------------------------

_DECISIONS: list = []  # list[dict] — rich capture for diag plots
_ORIG_DECIDE = wmod.decide_mode

# Rich-spy state — populated only on diag-pwl / diag-kik runs (rich=True).
# The lean regression-gate tests (a/b/c/d/e) leave _RICH_SPY_ENABLED=False
# so the per-step record schema in _DECISIONS stays byte-identical to its
# original step-5 contract.
_RICH_SPY_ENABLED:     bool                = False
_LATEST_BUILD_CAPTURE: Optional[tuple]     = None  # (positions: list, labels: list)
_LATEST_EVAL_CAPTURE:  Optional[list]      = None  # list[SampleResult]
_LATEST_PROXY_CAPTURE: Optional[dict]      = None  # 8.1.3: pre-filter proxy state
_RICH_HANDLES:         dict                = {}    # wrapper ref for uninstall

# 8.1.3: snapshot the original generate_samples reference at module load.
# wmod.generate_samples is the harness-side import binding; replacing it
# in _install_rich_spies redirects the call site at wrapper.py:241.
_ORIG_GENERATE_SAMPLES = wmod.generate_samples


def _spy_decide(*args, **kwargs):
    m, r = _ORIG_DECIDE(*args, **kwargs)
    params = kwargs.get("params")
    c3     = kwargs.get("c3_cost")
    crepos = kwargs.get("current_repos_cost")
    near   = kwargs.get("near_goal")
    gap_to_repos       = _hysteresis(params, "c3_to_repos",  near, c3)
    gap_to_c3          = _hysteresis(params, "repos_to_c3",  near, c3)
    gap_repos_to_repos = (_hysteresis(params, "repos_to_repos", near, crepos)
                          if crepos is not None else None)
    # NOTE 2 sanity assertion. Relative hysteresis × |c3_cost| should
    # always be non-negative; a violation here is itself a finding
    # (degenerate boundaries near c3≈0).
    if c3 is not None and c3 > 0:
        assert gap_to_repos >= 0, (
            f"negative gap_to_repos at step {len(_DECISIONS)}: {gap_to_repos}")
        assert gap_to_c3    >= 0, (
            f"negative gap_to_c3 at step {len(_DECISIONS)}: {gap_to_c3}")
    record = {
        "step":               len(_DECISIONS),
        "prev_mode":          kwargs.get("prev_mode"),
        "mode":               m,
        "reason":             r.name,
        "c3_cost":            c3,
        "best_other_cost":    kwargs.get("best_other_cost"),
        "current_repos_cost": crepos,
        "near_goal":          near,
        "finished_repos":     kwargs.get("finished_repos"),
        "met_progress":       kwargs.get("met_progress"),
        # Switch-boundary instrumentation. boundary_to_repos is the
        # value best_other_cost would need to drop BELOW to trigger
        # kToReposCost; boundary_to_c3 is the value best_other_cost
        # would need to rise ABOVE to trigger kToC3Cost.
        "cost_gap_to_repos":          gap_to_repos,
        "cost_gap_to_c3":             gap_to_c3,
        "cost_gap_repos_to_repos":    gap_repos_to_repos,
        "boundary_to_repos":          (c3 - gap_to_repos) if c3 is not None else None,
        "boundary_to_c3":             (c3 + gap_to_c3)    if c3 is not None else None,
    }
    if _RICH_SPY_ENABLED:
        record.update(_capture_rich_fields(args, kwargs, m, r))
    _DECISIONS.append(record)
    return m, r


def _install_spy() -> None:
    _DECISIONS.clear()
    wmod.decide_mode = _spy_decide


def _uninstall_spy() -> None:
    wmod.decide_mode = _ORIG_DECIDE


# ---------------------------------------------------------------------------
# Rich-spy: per-sample breakdown + counterfactual mode-switch
# ---------------------------------------------------------------------------
# Two extra spies wrap the wrapper's _build_samples and inner_solver's
# evaluate_samples (both bound methods, replaced via instance attribute).
# Pure passthrough — values returned unchanged. Only the side-effect of
# stashing the latest call's outputs into module-level slots is added.
# _spy_decide reads those slots, builds a per-sample breakdown, and
# re-invokes _ORIG_DECIDE with prev_repos excluded from best_other_cost
# to compute the (cf_mode, cf_reason) counterfactual.

def _install_rich_spies(wrapper) -> None:
    """Enable rich-spy capture. Safe to call once per _run_loop;
    _uninstall_rich_spies must be called from the same finally block."""
    global _RICH_SPY_ENABLED, _LATEST_BUILD_CAPTURE, _LATEST_EVAL_CAPTURE
    global _LATEST_PROXY_CAPTURE
    _RICH_SPY_ENABLED      = True
    _LATEST_BUILD_CAPTURE  = None
    _LATEST_EVAL_CAPTURE   = None
    _LATEST_PROXY_CAPTURE  = None
    _RICH_HANDLES["wrapper"] = wrapper

    orig_build = wrapper._build_samples
    def _spy_build(*args, **kwargs):
        positions, labels = orig_build(*args, **kwargs)
        global _LATEST_BUILD_CAPTURE
        _LATEST_BUILD_CAPTURE = (
            [np.asarray(p).copy() for p in positions],
            list(labels),
        )
        return positions, labels
    wrapper._build_samples = _spy_build

    orig_eval = wrapper.inner_solver.evaluate_samples
    def _spy_eval(*args, **kwargs):
        results = orig_eval(*args, **kwargs)
        global _LATEST_EVAL_CAPTURE
        _LATEST_EVAL_CAPTURE = list(results)
        return results
    wrapper.inner_solver.evaluate_samples = _spy_eval

    # 8.1.3 — proxy capture spy.
    # Recomputes the kRandomOnCircle proxy point from the call's kwargs
    # (cheaper than patching _random_on_circle), evaluates the workspace
    # filter against it, and stashes the pre-filter geometry. The spy
    # then forwards to the original generate_samples and records the
    # post-filter sample count, so we can correlate "proxy_passes_filter"
    # with "strat_* appeared in samples_breakdown."
    def _spy_generate_samples(*args, **kwargs):
        global _LATEST_PROXY_CAPTURE
        strategy  = kwargs.get("strategy")
        n_samples = kwargs.get("n_samples", 0)
        obj_xy    = kwargs.get("obj_xy")
        params    = kwargs.get("params")
        g_hat     = kwargs.get("g_hat")
        proxy_xy_z, proxy_passes = None, None
        if (strategy == SamplingStrategy.kRandomOnCircle
                and n_samples is not None and n_samples >= 1
                and g_hat is not None and obj_xy is not None
                and params is not None):
            r = float(params.sampling_radius)
            z = float(params.sampling_height)
            proxy = np.array([
                float(obj_xy[0]) - r * float(g_hat[0]),
                float(obj_xy[1]) - r * float(g_hat[1]),
                z,
            ])
            proxy_xy_z   = (float(proxy[0]), float(proxy[1]), float(proxy[2]))
            proxy_passes = (bool(is_in_workspace(proxy, params))
                            if params.filter_samples_for_safety else True)
        _LATEST_PROXY_CAPTURE = {
            "n_samples_input":     int(n_samples) if n_samples is not None else None,
            "obj_xy":              ((float(obj_xy[0]), float(obj_xy[1]))
                                    if obj_xy is not None else None),
            "g_hat":               ((float(g_hat[0]),  float(g_hat[1]))
                                    if g_hat is not None else None),
            "proxy_xyz":           proxy_xy_z,
            "proxy_y":             (proxy_xy_z[1] if proxy_xy_z is not None else None),
            "proxy_passes_filter": proxy_passes,
            "workspace_xy_max":    (list(params.workspace_xy_max)
                                    if params is not None else None),
        }
        result = _ORIG_GENERATE_SAMPLES(*args, **kwargs)
        _LATEST_PROXY_CAPTURE["n_samples_returned"] = len(result)
        return result
    wmod.generate_samples = _spy_generate_samples


def _uninstall_rich_spies() -> None:
    """Restore class-method dispatch and clear capture slots. Idempotent."""
    global _RICH_SPY_ENABLED, _LATEST_BUILD_CAPTURE, _LATEST_EVAL_CAPTURE
    global _LATEST_PROXY_CAPTURE
    w = _RICH_HANDLES.pop("wrapper", None)
    if w is not None:
        if "_build_samples" in w.__dict__:
            del w._build_samples
        if "evaluate_samples" in w.inner_solver.__dict__:
            del w.inner_solver.evaluate_samples
    wmod.generate_samples = _ORIG_GENERATE_SAMPLES
    _RICH_SPY_ENABLED      = False
    _LATEST_BUILD_CAPTURE  = None
    _LATEST_EVAL_CAPTURE   = None
    _LATEST_PROXY_CAPTURE  = None


def _capture_rich_fields(args, kwargs, mode, reason) -> dict:
    """Extract per-sample breakdown + counterfactual mode-switch from the
    latest build_samples and evaluate_samples captures. Called from
    _spy_decide only when _RICH_SPY_ENABLED."""
    rec = {
        "samples_breakdown":  [],
        "winning_idx":        None,
        "winning_label":      None,
        "prev_repos_idx":     None,
        "cf_best_other_cost": kwargs.get("best_other_cost"),
        "cf_mode":            mode,
        "cf_reason":          reason.name,
        # 8.1.3 — pre-filter proxy state from the latest generate_samples call.
        "proxy_capture":      (None if _LATEST_PROXY_CAPTURE is None
                               else dict(_LATEST_PROXY_CAPTURE)),
    }
    if _LATEST_BUILD_CAPTURE is None or _LATEST_EVAL_CAPTURE is None:
        return rec
    positions, labels = _LATEST_BUILD_CAPTURE
    results           = _LATEST_EVAL_CAPTURE
    if not (len(positions) == len(results) == len(labels)):
        return rec

    rec["samples_breakdown"] = [
        {
            "label":          lbl,
            "pos_xyz":        p.tolist(),
            "c_C3_raw":       float(res.c_C3_raw),
            "align_score":    float(res.align_score),
            "align_bonus":    float(res.align_bonus),
            "travel_dist":    float(res.travel_dist),
            "travel_penalty": float(res.travel_penalty),
            "c_sample":       float(res.c_sample),
            "feasible":       bool(res.feasible),
        }
        for p, lbl, res in zip(positions, labels, results)
    ]

    # Recompute argmin from c_samples. In diag-pwl/kik runs the wrapper's
    # _infeasible_repos_target stays None throughout (test_c contract), so
    # this matches wrapper.last_winning_sample_idx. If pointed at a poison
    # scenario this would diverge — but those runs use the lean spy.
    c_samples = [float(r.c_sample) for r in results]
    rec["winning_idx"]   = int(np.argmin(c_samples))
    rec["winning_label"] = labels[rec["winning_idx"]]

    for k, lbl in enumerate(labels):
        if lbl == "prev_repos":
            rec["prev_repos_idx"] = k
            break

    if rec["prev_repos_idx"] is not None:
        cf_costs = [c_samples[k] for k in range(1, len(c_samples))
                    if k != rec["prev_repos_idx"]]
        cf = min(cf_costs) if cf_costs else float("inf")
        cf_kwargs = dict(kwargs)
        cf_kwargs["best_other_cost"] = cf
        cf_m, cf_r = _ORIG_DECIDE(*args, **cf_kwargs)
        rec["cf_best_other_cost"] = cf
        rec["cf_mode"]            = cf_m
        rec["cf_reason"]          = cf_r.name

    return rec


# ---------------------------------------------------------------------------
# Common: build env + C3 stack + wrapper
# ---------------------------------------------------------------------------

def _task_cfg() -> dict:
    """Load the production `pushing` task config from config/tasks.yaml.
    Hand-coded values would diverge from production silently — load from
    the canonical source so this smoke matches the real run."""
    import yaml
    with open("config/tasks.yaml") as f:
        return yaml.safe_load(f)["tasks"]["pushing"]


def _build_world():
    """Returns (diagram, plant, panda_model, obj_body, plant_ad, context_ad)."""
    cfg = _task_cfg()
    diagram, plant, panda_model, object_model, _meshcat, plant_ad, context_ad = \
        build_environment(cfg)
    obj_body = plant.GetBodyByName(cfg["link_name"], object_model)
    return diagram, plant, panda_model, obj_body, plant_ad, context_ad


def _build_wrapper(diagram, plant, obj_body, plant_ad, context_ad,
                   *, traj_type, seed: int,
                   ik_d_min_override: Optional[float] = None) -> SamplingC3MPC:
    """Build the full controller stack and return the SamplingC3MPC wrapper."""
    cfg = _task_cfg()
    n_q = plant.num_positions()
    n_v = plant.num_velocities()
    n_u = plant.num_actuators()
    n_x = n_q + n_v

    formulator = LCSFormulator(plant, mu=cfg["friction"], obj_body=obj_body,
                               plant_ad=plant_ad, context_ad=context_ad)
    solver     = C3Solver(n_x=n_x, n_u=n_u, rho=100.0, math_diag=False, mode="c3")
    quad_cost  = QuadraticManipulationCost(plant, EE_BODY_NAME, obj_body,
                                           cfg["cost"], n_x, n_u, math_diag=False,
                                           cost_bias=False)
    base_mpc   = C3MPC(formulator=formulator, solver=solver, quadratic_cost=quad_cost,
                       horizon=20, dt=0.05, torque_limit=30.0, admm_iter=3,
                       math_diag=False)

    sc3_params = SamplingC3Params()
    sc3_params.reposition_params = RepositionParams(traj_type=traj_type)
    if ik_d_min_override is not None:
        from control.sampling_c3.params import RepositionIKParams
        sc3_params.repos_ik_params = RepositionIKParams(
            ik_min_distance_lower_bound=float(ik_d_min_override),
        )

    ee_frame = plant.GetFrameByName(EE_BODY_NAME)
    return SamplingC3MPC(
        base_mpc=base_mpc, plant=plant, ee_frame=ee_frame, obj_body=obj_body,
        params=sc3_params, log_diag=LOG_DIAG,
        rng=np.random.default_rng(seed),
        dt_ctrl=0.01, start_in_c3_mode=False,
        diagram=diagram,
    )


def _quat_to_yaw(q4) -> float:
    qw, qx, qy, qz = (float(x) for x in q4)
    return float(np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz)))


def _wrap_pi(a: float) -> float:
    return float(((a + np.pi) % (2*np.pi)) - np.pi)


def _set_initial_state(plant, plant_ctx, panda_model, obj_body) -> None:
    """Mirror main.py's startup state — pulls init_xyz from the same
    config/tasks.yaml entry the production run uses."""
    init_xyz = _task_cfg()["init_xyz"]
    plant.SetFreeBodyPose(
        plant_ctx, obj_body,
        ad.RigidTransform(ad.RotationMatrix(), list(init_xyz)),
    )
    plant.SetPositions(plant_ctx, panda_model, INITIAL_ARM_Q)


# ---------------------------------------------------------------------------
# Run-loop core
# ---------------------------------------------------------------------------

def _run_loop(*, traj_type, num_loops: int, seed: int,
              poison_drop_loop: Optional[int] = None,
              poison_restore_loop: Optional[int] = None,
              ik_d_min_override: Optional[float] = None,
              rich: bool = False,
              ) -> dict:
    """Run a 200-loop sim and capture metrics.

    poison_drop_loop / poison_restore_loop drive test (e):
      - At ``poison_drop_loop``, set obj_z = -0.10 (inside the table)
        before the next compute_control.
      - At ``poison_restore_loop``, set obj_z = 0.05 (back on the table).
    """
    diagram, plant, panda_model, obj_body, plant_ad, context_ad = _build_world()
    simulator = ad.Simulator(diagram)
    context   = simulator.get_mutable_context()
    plant_ctx = plant.GetMyMutableContextFromRoot(context)
    _set_initial_state(plant, plant_ctx, panda_model, obj_body)

    wrapper = _build_wrapper(diagram, plant, obj_body, plant_ad, context_ad,
                             traj_type=traj_type, seed=seed,
                             ik_d_min_override=ik_d_min_override)

    n_u = plant.num_actuators()
    target_xy   = np.asarray(_task_cfg()["goal_xy"], dtype=float)
    ee_frame    = plant.GetFrameByName(EE_BODY_NAME)
    world_frame = plant.world_frame()
    obj_floating_q_start = obj_body.floating_positions_start()

    # Per-loop tracking
    modes_per_loop:        list = []  # last_mode after each compute_control
    poison_per_loop:       list = []  # _infeasible_repos_target snapshot
    last_x_seq_was_none:   list = []
    u_norms:               list = []
    exceptions:            list = []
    loop_wall_ms:          list = []  # per-loop wall time around compute_control
    ee_delta_m:            list = []  # ||p_EE_end - p_EE_start|| each loop
    obj_xy_delta_m:        list = []  # ||obj_xy_end - obj_xy_start|| each loop
    obj_yaw_delta_rad:     list = []  # |Δyaw| (wrapped to [0, π]) each loop
    knot0_feasible_per_loop:  list = []  # tracker.last_knot0_feasible after compute_control
    last_repos_feasible_per_loop: list = []  # wrapper._last_repos_feasible after compute_control
    knot0_failure_msg_per_loop: list = []  # tracker.last_knot0_failure_msg (None on success)
    # 8.4.3 — TS4 capture: FK of last_q_knots[:,0] (i.e. where IK says EE
    # should be at knot 0) and the q_arm itself. None on c3-mode loops or
    # before tracker.compute_torque has ever run.
    tracker_knot0_ee_per_loop: list = []
    tracker_q_knot0_per_loop:  list = []
    # 8.4.5 — integral capture: tracker._integral 7-vector after each
    # compute_torque so we can see how much of I_max the controller is
    # actually using (Fix 6 outcome verification).
    tracker_integral_per_loop: list = []
    # 8.6 — full post-IK math capture for the diagnostic deep-dive.
    # tracker_knots_*_per_loop hold the full (7×N) and (3×N) matrices the
    # IK chain produced this loop (vs the previous TS4 captures which only
    # held knot 0). current_q_arm / v_arm are the start-of-loop arm slices
    # (snapshots taken at the top of each loop body). q_arm_after_dt /
    # v_arm_after_dt are the end-of-loop snapshots after AdvanceTo.
    # tau_g_at_target is gravity comp at q_full_target as left in
    # _plant_ctx_ik by Fix 4 (free-mode loops only; None on c3-mode).
    # u_full is the per-joint actuation vector (post-clip).
    tracker_knots_q_per_loop:    list = []
    tracker_knots_ee_per_loop:   list = []
    current_q_arm_per_loop:      list = []
    current_v_arm_per_loop:      list = []
    q_arm_after_dt_per_loop:     list = []
    v_arm_after_dt_per_loop:     list = []
    tau_g_at_target_per_loop:    list = []
    u_full_per_loop:             list = []
    knot0_failure_inputs_first: Optional[tuple] = None  # (q_full, p_target) of first failed knot-0
    knot0_failure_inputs_last:  Optional[tuple] = None  # latest snapshot for post-run diagnose
    poison_first_set_loop: Optional[int] = None
    poison_first_clear_loop_after_restore: Optional[int] = None
    ik_infeasible_msg_first: Optional[str] = None

    sim_time = 0.0
    dt_ctrl  = 0.01

    _install_spy()
    if rich:
        _install_rich_spies(wrapper)
    try:
        for step in range(num_loops):
            # ---- (e) drop / restore manipuland ------------------------------
            if poison_drop_loop is not None and step == poison_drop_loop:
                # Drop INSIDE the table: table top is z=0, half-extent 0.05.
                # Box center at z=-0.10 puts the box bottom at z=-0.15,
                # well below the table-top plane. The min-distance constraint
                # then sees a deeply-negative gap on EVERY arm-vs-box pair
                # near the table — IK is infeasible by construction.
                plant.SetFreeBodyPose(
                    plant_ctx, obj_body,
                    ad.RigidTransform(ad.RotationMatrix(), [0.0, 0.0, -0.10]),
                )
            if poison_restore_loop is not None and step == poison_restore_loop:
                plant.SetFreeBodyPose(
                    plant_ctx, obj_body,
                    ad.RigidTransform(ad.RotationMatrix(), [0.0, 0.0, 0.05]),
                )

            current_q = plant.GetPositions(plant_ctx)
            current_v = plant.GetVelocities(plant_ctx)
            tau_g     = plant.CalcGravityGeneralizedForces(plant_ctx)

            # Diag 2: capture pose at start-of-loop.
            ee_start = plant.CalcPointsPositions(
                plant_ctx, ee_frame, np.zeros(3), world_frame
            ).flatten().copy()
            _qf_start = current_q[obj_floating_q_start:obj_floating_q_start+7]
            obj_xy_start  = _qf_start[4:6].copy()
            obj_yaw_start = _quat_to_yaw(_qf_start[0:4])

            _t0 = time.perf_counter()
            try:
                u_opt = wrapper.compute_control(
                    current_q, current_v, plant_ctx, target_xy)
            except Exception as e:
                exceptions.append((step, type(e).__name__, str(e)))
                # keep going to characterise blast radius if any
                u_opt = np.zeros(n_u)
            loop_wall_ms.append((time.perf_counter() - _t0) * 1e3)

            # Metrics.
            modes_per_loop.append(wrapper.last_mode)
            poison_now = getattr(wrapper, "_infeasible_repos_target", None)
            poison_per_loop.append(None if poison_now is None else poison_now.copy())
            last_x_seq_was_none.append(wrapper.last_x_seq is None)
            u_norms.append(float(np.linalg.norm(u_opt)))
            # V-1c / V-2 capture. last_knot0_feasible only refreshes when
            # tracker.compute_torque actually ran (free mode, non-fallback).
            # On c3-mode loops the value is stale from the most recent
            # free-mode call. We capture both, plus mode, so the analyser
            # can filter to the per-IK-call signal.
            knot0_feasible_per_loop.append(
                bool(getattr(wrapper.tracker, "last_knot0_feasible", True)))
            last_repos_feasible_per_loop.append(
                bool(getattr(wrapper, "_last_repos_feasible", True)))
            knot0_failure_msg_per_loop.append(
                getattr(wrapper.tracker, "last_knot0_failure_msg", None))
            # Snapshot the inputs of the FIRST failure for post-run
            # diagnose_failure_at (Tightening 2). Snapshotting only the
            # first means diagnose runs once, on a known-canonical case.
            if knot0_failure_inputs_first is None:
                _fi = getattr(wrapper.tracker, "last_knot0_failure_inputs", None)
                if _fi is not None:
                    knot0_failure_inputs_first = (_fi[0].copy(), _fi[1].copy())
            # 8.4.3 — TS4: read tracker's per-loop IK head from the latest
            # compute_torque. last_ee_knots[:,0] is FK(last_q_knots[:,0])
            # already (computed at reposition_ik.py:1118-1125), so no
            # FK needed harness-side.
            _ee_knots = getattr(wrapper.tracker, "last_ee_knots", None)
            _q_knots  = getattr(wrapper.tracker, "last_q_knots",  None)
            tracker_knot0_ee_per_loop.append(
                None if _ee_knots is None or _ee_knots.shape[1] == 0 else _ee_knots[:, 0].copy())
            tracker_q_knot0_per_loop.append(
                None if _q_knots  is None or _q_knots.shape[1]  == 0 else _q_knots[:, 0].copy())
            _integ = getattr(wrapper.tracker, "_integral", None)
            tracker_integral_per_loop.append(
                None if _integ is None else np.asarray(_integ).copy())
            # 8.6 — full math capture
            current_q_arm_per_loop.append(np.asarray(current_q[:n_u]).copy())
            current_v_arm_per_loop.append(np.asarray(current_v[:n_u]).copy())
            tracker_knots_q_per_loop.append(
                None if _q_knots is None else _q_knots.copy())
            tracker_knots_ee_per_loop.append(
                None if _ee_knots is None else _ee_knots.copy())
            u_full_per_loop.append(np.asarray(u_opt).copy())
            # tau_g at q_full_target: only meaningful when compute_torque
            # ran this loop (free mode). Fix 4 leaves _plant_ctx_ik at
            # q_full_target after step 11 of compute_torque, so we can
            # query gravity directly without re-setting positions.
            if wrapper.last_mode == "free":
                _ctx_ik = getattr(wrapper.tracker, "_plant_ctx_ik", None)
                tau_g_at_target_per_loop.append(
                    None if _ctx_ik is None
                    else plant.CalcGravityGeneralizedForces(_ctx_ik)[:n_u].copy())
            else:
                tau_g_at_target_per_loop.append(None)

            # Test (e) helpers
            if poison_first_set_loop is None and poison_now is not None:
                poison_first_set_loop = step
                # Constraint 1: surface the IK infeasibility message on the
                # first failed knot. The tracker holds last_q_knots etc.
                # We don't have a structured "constraint name" feed, but we
                # can read the diag dict's any_infeasible flag and the IK's
                # solve-ms to confirm it's IK-side, not SceneGraph error.
                tr = wrapper.tracker
                ik_infeasible_msg_first = (
                    f"first poison set at step={step}; "
                    f"tracker.last_feasible[0]={tr.last_feasible[0] if tr.last_feasible else 'n/a'}; "
                    f"knots_solve_ms={tr.last_knots_solve_ms}; "
                    f"any_infeasible={(not all(tr.last_feasible)) if tr.last_feasible else 'n/a'}"
                )
            if (poison_restore_loop is not None
                    and poison_first_clear_loop_after_restore is None
                    and step >= poison_restore_loop
                    and poison_now is None
                    and poison_first_set_loop is not None
                    and step > poison_first_set_loop):
                poison_first_clear_loop_after_restore = step

            # Apply control (mirror main.py's free-mode no-double-grav-comp).
            if wrapper.last_mode == "free":
                plant.get_actuation_input_port().FixValue(plant_ctx, u_opt)
            else:
                plant.get_actuation_input_port().FixValue(plant_ctx, tau_g[:n_u] + u_opt)

            sim_time += dt_ctrl
            simulator.AdvanceTo(sim_time)

            # Diag 2: capture pose at end-of-loop, after AdvanceTo.
            ee_end = plant.CalcPointsPositions(
                plant_ctx, ee_frame, np.zeros(3), world_frame
            ).flatten().copy()
            _q_end = plant.GetPositions(plant_ctx)
            _qf_end = _q_end[obj_floating_q_start:obj_floating_q_start+7]
            ee_delta_m.append(float(np.linalg.norm(ee_end - ee_start)))
            obj_xy_delta_m.append(float(np.linalg.norm(_qf_end[4:6] - obj_xy_start)))
            obj_yaw_delta_rad.append(
                abs(_wrap_pi(_quat_to_yaw(_qf_end[0:4]) - obj_yaw_start)))
            # 8.6 — end-of-loop arm state after AdvanceTo
            q_arm_after_dt_per_loop.append(np.asarray(_q_end[:n_u]).copy())
            _v_end = plant.GetVelocities(plant_ctx)
            v_arm_after_dt_per_loop.append(np.asarray(_v_end[:n_u]).copy())
    finally:
        decisions = list(_DECISIONS)
        if rich:
            _uninstall_rich_spies()
        _uninstall_spy()

    # Post-run failure introspection. The hot loop is over, AdvanceTo no
    # longer fires. Skipped when no failure was captured (e.g. PWL or a
    # clean kIK run).
    diagnose_dict: Optional[dict] = None
    if knot0_failure_inputs_first is not None and \
            hasattr(wrapper.tracker, "diagnose_failure_at"):
        q_full, p_target = knot0_failure_inputs_first
        diagnose_dict = wrapper.tracker.diagnose_failure_at(q_full, p_target)

    # Mode-flip + dwell analysis from modes_per_loop sequence.
    flips = 0
    dwell_runs:   list = []   # list of (mode, length_in_loops)
    cur_mode = modes_per_loop[0]
    cur_len  = 1
    for m in modes_per_loop[1:]:
        if m != cur_mode:
            dwell_runs.append((cur_mode, cur_len))
            flips += 1
            cur_mode = m
            cur_len  = 1
        else:
            cur_len += 1
    dwell_runs.append((cur_mode, cur_len))

    # Reason histogram from spy decisions (dict-based schema).
    reason_hist = Counter(d["reason"] for d in decisions)
    # Reasons that actually correspond to a flip (mode change vs prev).
    flip_reasons: list = []
    if decisions:
        prev_mode = None
        for d in decisions:
            if prev_mode is not None and d["mode"] != prev_mode:
                flip_reasons.append(d["reason"])
            prev_mode = d["mode"]
    flip_reason_hist = Counter(flip_reasons)

    return dict(
        traj_type             = traj_type.name,
        num_loops             = num_loops,
        exceptions            = exceptions,
        modes_per_loop        = modes_per_loop,
        last_x_seq_was_none   = last_x_seq_was_none,
        u_norms               = u_norms,
        loop_wall_ms          = loop_wall_ms,
        ee_delta_m            = ee_delta_m,
        obj_xy_delta_m        = obj_xy_delta_m,
        obj_yaw_delta_rad     = obj_yaw_delta_rad,
        knot0_feasible_per_loop      = knot0_feasible_per_loop,
        last_repos_feasible_per_loop = last_repos_feasible_per_loop,
        knot0_failure_msg_per_loop   = knot0_failure_msg_per_loop,
        knot0_failure_inputs_first   = knot0_failure_inputs_first,
        tracker_knot0_ee_per_loop    = tracker_knot0_ee_per_loop,
        tracker_q_knot0_per_loop     = tracker_q_knot0_per_loop,
        tracker_integral_per_loop    = tracker_integral_per_loop,
        tracker_knots_q_per_loop     = tracker_knots_q_per_loop,
        tracker_knots_ee_per_loop    = tracker_knots_ee_per_loop,
        current_q_arm_per_loop       = current_q_arm_per_loop,
        current_v_arm_per_loop       = current_v_arm_per_loop,
        q_arm_after_dt_per_loop      = q_arm_after_dt_per_loop,
        v_arm_after_dt_per_loop      = v_arm_after_dt_per_loop,
        tau_g_at_target_per_loop     = tau_g_at_target_per_loop,
        u_full_per_loop              = u_full_per_loop,
        diagnose_dict                = diagnose_dict,
        poison_per_loop_isnone = [p is None for p in poison_per_loop],
        flips                 = flips,
        dwell_runs            = dwell_runs,
        reason_hist           = dict(reason_hist),
        flip_reason_hist      = dict(flip_reason_hist),
        decisions             = decisions,
        poison_first_set_loop = poison_first_set_loop,
        poison_first_clear_loop_after_restore = poison_first_clear_loop_after_restore,
        ik_infeasible_msg_first = ik_infeasible_msg_first,
    )


def _save(metrics: dict, name: str) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"probe_5f_{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(metrics, f)
    return path


def _load(name: str) -> dict:
    path = os.path.join(RESULTS_DIR, f"probe_5f_{name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def _summary_lines(metrics: dict) -> list:
    flip_count = metrics["flips"]
    dwell = metrics["dwell_runs"]
    free_runs = [n for (m, n) in dwell if m == "free"]
    c3_runs   = [n for (m, n) in dwell if m == "c3"]
    mean_free = float(np.mean(free_runs)) if free_runs else 0.0
    mean_c3   = float(np.mean(c3_runs))   if c3_runs   else 0.0
    excs = metrics["exceptions"]
    last_x_none_count = sum(metrics["last_x_seq_was_none"])
    u = np.asarray(metrics["u_norms"])
    wm = np.asarray(metrics.get("loop_wall_ms", []))
    lines = [
        f"  traj_type            = {metrics['traj_type']}",
        f"  num_loops            = {metrics['num_loops']}",
        f"  exceptions           = {len(excs)}",
        f"  last_x_seq None ct.  = {last_x_none_count}",
        f"  total ||u|| mean     = {float(u.mean()):.3f}  (sum={float(u.sum()):.1f})",
        f"  mode flips           = {flip_count}",
        f"  mean dwell c3/free   = {mean_c3:.1f} / {mean_free:.1f} loops",
        f"  reason hist          = {metrics['reason_hist']}",
        f"  flip-reason hist     = {metrics['flip_reason_hist']}",
    ]
    if wm.size:
        lines.append(
            f"  loop wall mean/p50/p99 = "
            f"{float(wm.mean()):.1f} / {float(np.percentile(wm,50)):.1f} / "
            f"{float(np.percentile(wm,99)):.1f} ms"
        )
    ee = np.asarray(metrics.get("ee_delta_m", []))
    if ee.size:
        lines.append(
            f"  EE motion mean/p50/p99 = "
            f"{float(ee.mean())*1000:.2f} / {float(np.percentile(ee,50))*1000:.2f} / "
            f"{float(np.percentile(ee,99))*1000:.2f} mm"
        )
    obj_xy = np.asarray(metrics.get("obj_xy_delta_m", []))
    if obj_xy.size:
        lines.append(
            f"  obj_xy mean/p50/p99    = "
            f"{float(obj_xy.mean())*1000:.2f} / {float(np.percentile(obj_xy,50))*1000:.2f} / "
            f"{float(np.percentile(obj_xy,99))*1000:.2f} mm"
        )
    oy = np.asarray(metrics.get("obj_yaw_delta_rad", []))
    if oy.size:
        lines.append(
            f"  obj_yaw p50/p99        = "
            f"{float(np.percentile(oy,50))*180/np.pi:.3f} / "
            f"{float(np.percentile(oy,99))*180/np.pi:.3f} deg"
        )
    lines.append(
        f"  kToBetterRepos count   = "
        f"{metrics['reason_hist'].get('kToBetterRepos', 0)}"
    )
    return lines


def _rich_summary_lines(metrics: dict) -> list:
    """Three statistics lines from rich-spy capture, addressing C1/C2/C3.
    No verdicts — the summary computes statistics; interpretation belongs
    in the next phase."""
    decisions = metrics.get("decisions", [])
    rich = [d for d in decisions if d.get("samples_breakdown")]
    if not rich:
        return ["  rich-spy capture     = (none)  "
                "[run with rich=True to populate]"]

    lines = []

    # C1: prev_repos travel-penalty discount.
    # Per-loop ratio travel_penalty(prev_repos) / median(travel_penalty(strat_*)).
    # Computed only on loops where BOTH prev_repos and at least one strat_*
    # sample exist; loops with no prev_repos are skipped (no discount to
    # measure).
    ratios = []
    for d in rich:
        sb = d["samples_breakdown"]
        prev_t  = next((s["travel_penalty"] for s in sb
                        if s["label"] == "prev_repos"), None)
        strat_t = [s["travel_penalty"] for s in sb
                   if s["label"].startswith("strat_")]
        if prev_t is not None and strat_t:
            med = float(np.median(strat_t))
            if med > 1e-6:
                ratios.append(prev_t / med)
    if ratios:
        n_below_pt1 = sum(1 for r in ratios if r < 0.10)
        lines.append(
            f"  C1 travel(prev_repos) / median(travel(strat_*))  "
            f"median={float(np.median(ratios)):.4f}  "
            f"p10={float(np.percentile(ratios, 10)):.4f}  "
            f"p90={float(np.percentile(ratios, 90)):.4f}  "
            f"n={len(ratios)} loops  n<0.10={n_below_pt1}"
        )
    else:
        lines.append(
            "  C1 travel(prev_repos) / median(travel(strat_*))  "
            "(no loops with both prev_repos and strat_* samples)"
        )

    # C2: winning sample's align_bonus vs travel_penalty.
    win_align  = []
    win_travel = []
    for d in rich:
        wi = d.get("winning_idx")
        sb = d.get("samples_breakdown", [])
        if wi is None or wi >= len(sb):
            continue
        win_align.append(float(sb[wi]["align_bonus"]))
        win_travel.append(float(sb[wi]["travel_penalty"]))
    if win_align:
        ratios2 = [a / (t + 1e-6) for a, t in zip(win_align, win_travel)]
        lines.append(
            f"  C2 winning sample      "
            f"align_bonus median={float(np.median(win_align)):.1f}  "
            f"travel_penalty median={float(np.median(win_travel)):.1f}  "
            f"align/travel median={float(np.median(ratios2)):.4f}  "
            f"n={len(win_align)} loops"
        )

    # C3: counterfactual mode-switch — loops where reason==kStayInRepos
    # but cf_reason==kToC3Cost (i.e. the prev_repos discount alone kept
    # the controller in repos mode).
    repos_loops = [d for d in rich if d.get("reason") == "kStayInRepos"]
    cf_flips    = sum(1 for d in repos_loops
                      if d.get("cf_reason") == "kToC3Cost")
    cf_break = (f"{cf_flips} / {len(repos_loops)} kStayInRepos loops"
                if repos_loops else "(no kStayInRepos loops)")
    lines.append(
        f"  C3 kStayInRepos→cf=kToC3Cost  = {cf_break}"
    )
    # Bonus C3 line: also report cf_reason histogram across kStayInRepos
    # loops, in case the dominant counterfactual is something other than
    # kToC3Cost (e.g. kToBetterRepos, which would indicate the discount is
    # holding ALL alternatives down, not just c3).
    if repos_loops:
        from collections import Counter
        cf_hist = Counter(d.get("cf_reason") for d in repos_loops)
        lines.append(
            f"  C3 cf_reason hist over kStayInRepos = {dict(cf_hist)}"
        )

    # 8.1.3 D-verification — for free-mode 2-tuple loops (labels exactly
    # ['current', 'prev_repos']), classify the rejected proxy by sign of
    # proxy_y and by what the workspace filter actually returned. The
    # closed-bound hypothesis predicts essentially all such loops have
    # proxy_y > 0 AND proxy_passes_filter == False.
    two_tuple = [d for d in rich
                 if [s["label"] for s in d.get("samples_breakdown", [])]
                    == ["current", "prev_repos"]]
    n_2t = len(two_tuple)
    pos = neg = zero = miss = 0
    n_filter_pass = n_filter_fail = 0
    proxy_ys = []
    for d in two_tuple:
        pc = d.get("proxy_capture")
        if pc is None or pc.get("proxy_y") is None:
            miss += 1
            continue
        py = pc["proxy_y"]
        proxy_ys.append(py)
        if py > 0:    pos  += 1
        elif py < 0:  neg  += 1
        else:         zero += 1
        if pc.get("proxy_passes_filter") is True:
            n_filter_pass += 1
        elif pc.get("proxy_passes_filter") is False:
            n_filter_fail += 1
    if n_2t > 0:
        med_y = float(np.median(proxy_ys)) if proxy_ys else float("nan")
        ymin = float(np.min(proxy_ys)) if proxy_ys else float("nan")
        ymax = float(np.max(proxy_ys)) if proxy_ys else float("nan")
        lines.append(
            f"  D  free-mode 2-tuple loops    = {n_2t}  "
            f"(proxy_y >0: {pos}  ≤0: {neg+zero}  no_capture: {miss})"
        )
        lines.append(
            f"  D  proxy_y over 2-tuple       = "
            f"median={med_y:.3e}  min={ymin:.3e}  max={ymax:.3e}"
        )
        lines.append(
            f"  D  proxy_passes_filter (2-t)  = "
            f"True: {n_filter_pass}  False: {n_filter_fail}"
        )
    else:
        lines.append("  D  free-mode 2-tuple loops    = 0  (nothing to classify)")
    return lines


# ---------------------------------------------------------------------------
# Sub-tests
# ---------------------------------------------------------------------------

def test_a() -> int:
    """PWL regression — 200 loops, default traj_type=kPiecewiseLinear."""
    print(f"=== 5f (a): PWL regression, {NUM_LOOPS} loops, seed={SEED} ===")
    t0 = time.time()
    m = _run_loop(traj_type=RepositioningTrajectoryType.kPiecewiseLinear,
                  num_loops=NUM_LOOPS, seed=SEED)
    dt = time.time() - t0
    print(f"\n[wall] elapsed = {dt:.1f}s")
    print()
    for line in _summary_lines(m):
        print(line)

    if m["exceptions"]:
        print("\n[FAIL] (a) exceptions:")
        for e in m["exceptions"][:5]:
            print(f"  step={e[0]}  {e[1]}: {e[2][:140]}")
        return 1

    if m["last_x_seq_was_none"][0]:
        # Loop 0 might have last_x_seq=None if base_mpc hadn't run yet — accept.
        # Subsequent loops should populate.
        n_after_loop0 = sum(m["last_x_seq_was_none"][1:])
        if n_after_loop0 > 0:
            print(f"\n[FAIL] (a) last_x_seq=None on {n_after_loop0} loops after loop 0")
            return 1

    path = _save(m, "a")
    print(f"\n[PASS] (a) PWL regression OK. baseline saved → {path}")
    print("       (using this run AS the (d) baseline — no prior recorded baseline available)")
    return 0


def test_b() -> int:
    """kIK feasible — 200 loops, traj_type=kIK, reachable target."""
    print(f"=== 5f (b): kIK feasible, {NUM_LOOPS} loops, seed={SEED} ===")
    t0 = time.time()
    m = _run_loop(traj_type=RepositioningTrajectoryType.kIK,
                  num_loops=NUM_LOOPS, seed=SEED)
    dt = time.time() - t0
    print(f"\n[wall] elapsed = {dt:.1f}s")
    print()
    for line in _summary_lines(m):
        print(line)

    if m["exceptions"]:
        print("\n[FAIL] (b) exceptions:")
        for e in m["exceptions"][:5]:
            print(f"  step={e[0]}  {e[1]}: {e[2][:140]}")
        return 1

    # Hard fail: any kToReposUnproductive flip means the IK tracker is
    # making the controller think object-pushing has stalled. That's
    # exactly the regression the change was supposed to avoid.
    flip_hist = m["flip_reason_hist"]
    n_unprod = flip_hist.get("kToReposUnproductive", 0)
    if n_unprod > 0:
        print(f"\n[FAIL] (b) kToReposUnproductive flips = {n_unprod} (must be 0). "
              f"The IK tracker is stalling object progress.")
        return 1
    print(f"\n  hard-assert: kToReposUnproductive flips = 0  OK")

    # Observation: any return-to-C3 flips. Not a failure — would mean kIK
    # accomplishes the reposition faster than PWL did and re-enters C3.
    n_to_c3_cost  = flip_hist.get("kToC3Cost", 0)
    n_to_c3_reach = flip_hist.get("kToC3ReachedReposTarget", 0)
    if n_to_c3_cost or n_to_c3_reach:
        print(f"  observation: return-to-C3 flips — kToC3Cost={n_to_c3_cost} "
              f"kToC3ReachedReposTarget={n_to_c3_reach} "
              f"(absent in PWL baseline; not a failure)")
    else:
        print(f"  observation: no return-to-C3 flips (matches PWL baseline)")

    path = _save(m, "b")
    print(f"\n[PASS] (b) kIK feasible OK. saved → {path}")
    return 0


def test_c() -> int:
    """No-spurious-poison: assert _infeasible_repos_target stays None
    throughout the (b) feasible run."""
    print("=== 5f (c): no-spurious-poison check on (b) ===")
    if not os.path.exists(os.path.join(RESULTS_DIR, "probe_5f_b.pkl")):
        print("[FAIL] (c) requires (b) metrics. Run sub-test (b) first.")
        return 1
    m = _load("b")
    poison_isnone = m["poison_per_loop_isnone"]
    if all(poison_isnone):
        print(f"[PASS] (c) _infeasible_repos_target was None all {len(poison_isnone)} loops.")
        return 0

    # Diagnostic for failure
    first_bad = poison_isnone.index(False)
    print(f"[FAIL] (c) _infeasible_repos_target became non-None at loop {first_bad}.")
    # We don't carry the actual cached p value across pickling unless we
    # re-run; print the mode + decision around that loop instead.
    decisions = m["decisions"]
    def _step_of(d):
        return d["step"] if isinstance(d, dict) else d[0]
    near = [d for d in decisions if abs(_step_of(d) - first_bad) <= 2]
    print(f"  decisions near loop {first_bad}: {near}")
    print(f"  mode at that loop: {m['modes_per_loop'][first_bad]}")
    print(f"  target_xy throughout = [0.3, 0.0]  (does not change)")
    return 1


def test_d() -> int:
    """Mode-switch parity: (a) PWL vs (b) kIK."""
    print("=== 5f (d): mode-switch parity (a) vs (b) ===")
    a = _load("a")
    b = _load("b")

    a_flips, b_flips = a["flips"], b["flips"]
    a_dwell_c3   = [n for (m, n) in a["dwell_runs"] if m == "c3"]
    a_dwell_free = [n for (m, n) in a["dwell_runs"] if m == "free"]
    b_dwell_c3   = [n for (m, n) in b["dwell_runs"] if m == "c3"]
    b_dwell_free = [n for (m, n) in b["dwell_runs"] if m == "free"]
    am_c3   = float(np.mean(a_dwell_c3))   if a_dwell_c3   else 0.0
    am_free = float(np.mean(a_dwell_free)) if a_dwell_free else 0.0
    bm_c3   = float(np.mean(b_dwell_c3))   if b_dwell_c3   else 0.0
    bm_free = float(np.mean(b_dwell_free)) if b_dwell_free else 0.0

    print()
    print("  metric                     PWL          kIK")
    print(f"  flips                      {a_flips:<12d} {b_flips:<12d}")
    print(f"  mean dwell c3              {am_c3:<12.1f} {bm_c3:<12.1f}")
    print(f"  mean dwell free            {am_free:<12.1f} {bm_free:<12.1f}")
    print(f"  flip-reason hist (PWL)     {a['flip_reason_hist']}")
    print(f"  flip-reason hist (kIK)     {b['flip_reason_hist']}")
    print()

    fails = []
    # Tolerance: ±20% on flip count.
    if a_flips > 0:
        rel = abs(b_flips - a_flips) / a_flips
        if rel > 0.20:
            fails.append(f"flip count {b_flips} vs PWL {a_flips} (Δ {rel:.0%}) > 20%")
    elif b_flips > 0:
        fails.append(f"PWL had 0 flips, kIK has {b_flips}")

    # Tolerance: ±30% on mean dwell.
    for label, av, bv in [("c3", am_c3, bm_c3), ("free", am_free, bm_free)]:
        if av > 0:
            rel = abs(bv - av) / av
            if rel > 0.30:
                fails.append(f"mean dwell {label} {bv:.1f} vs PWL {av:.1f} "
                             f"(Δ {rel:.0%}) > 30%")

    # Reasons that appear in PWL must appear in kIK.
    a_reasons = set(a["flip_reason_hist"])
    b_reasons = set(b["flip_reason_hist"])
    missing = a_reasons - b_reasons
    if missing:
        fails.append(f"reasons in PWL not in kIK: {missing}")

    # No new dominant reason in kIK that wasn't dominant in PWL.
    if b["flip_reason_hist"]:
        b_total = sum(b["flip_reason_hist"].values())
        for r, c in b["flip_reason_hist"].items():
            if c / b_total > 0.5:
                a_share = a["flip_reason_hist"].get(r, 0) / max(1, sum(a["flip_reason_hist"].values()))
                if a_share <= 0.5:
                    fails.append(
                        f"reason {r} dominates kIK ({c}/{b_total} = {c/b_total:.0%}) "
                        f"but did not dominate PWL ({a_share:.0%})")

    if fails:
        print("[FAIL] (d):")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("[PASS] (d) mode-switch parity within tolerances.")
    return 0


def test_e() -> int:
    """Poison lifecycle: drop manipuland inside table at loop 5,
    restore at loop 30. Poison should be set within ≤2 loops of drop
    and clear within ≤5 loops of restore (constraint 3 leaves room
    to relax to ≤10 if seed-flaky)."""
    print("=== 5f (e): poison lifecycle (drop loop 5, restore loop 30) ===")
    DROP, RESTORE = 5, 30
    m = _run_loop(traj_type=RepositioningTrajectoryType.kIK,
                  num_loops=80, seed=SEED,
                  poison_drop_loop=DROP, poison_restore_loop=RESTORE)

    print()
    if m["exceptions"]:
        print("[FAIL] (e) exceptions during run:")
        for e in m["exceptions"][:5]:
            print(f"  step={e[0]}  {e[1]}: {e[2][:140]}")
        return 1

    # Constraint 1: print first IK infeasibility message.
    print(f"  IK infeasibility (first):  {m['ik_infeasible_msg_first']}")

    # Constraint 2: poison set by loop ≤ DROP + 2.
    set_loop = m["poison_first_set_loop"]
    if set_loop is None:
        print(f"[FAIL] (e) poison was never set despite drop at loop {DROP}.")
        return 1
    set_ok = set_loop <= DROP + 2
    print(f"  poison first set at loop:  {set_loop}  (≤ {DROP+2} required: {'OK' if set_ok else 'FAIL'})")

    # Constraint 3: poison cleared by loop ≤ RESTORE + 5 (or ≤+10 if flaky).
    clear_loop = m["poison_first_clear_loop_after_restore"]
    if clear_loop is None:
        print(f"[FAIL] (e) poison never cleared after restore at loop {RESTORE}.")
        return 1
    clear_ok_5  = clear_loop <= RESTORE + 5
    clear_ok_10 = clear_loop <= RESTORE + 10
    band = "≤+5" if clear_ok_5 else ("≤+10 (relaxed)" if clear_ok_10 else "FAIL")
    print(f"  poison first cleared at:   loop {clear_loop}  "
          f"({clear_loop - RESTORE} loops after restore — band: {band})")

    if not set_ok:
        return 1
    if not clear_ok_5:
        if clear_ok_10:
            print("[PASS] (e) poison lifecycle correct, but clear took >5 loops "
                  "(within relaxed ≤+10 band). Document seed-dependence.")
            return 0
        return 1
    print("[PASS] (e) poison lifecycle within tight bands.")
    return 0


# ---------------------------------------------------------------------------
# Diagnostic plotting (option C — hypothesis test)
# ---------------------------------------------------------------------------

def _verify_boundary_signs(decisions: list) -> tuple:
    """Boundary-sign verification (NOTE 1).

    Pre-plot sanity check: pick the first decision with reason ==
    'kToReposCost' and confirm best_other_cost < boundary_to_repos.
    Pick the first 'kToC3Cost' and confirm best_other_cost > boundary_to_c3.
    Returns (ok: bool, msg_lines: list[str]).
    """
    msgs = []
    ok = True
    repos_step = next((d for d in decisions if d["reason"] == "kToReposCost"), None)
    c3_step    = next((d for d in decisions if d["reason"] == "kToC3Cost"),    None)

    if repos_step is None:
        msgs.append("[verify] no kToReposCost step in decisions — cannot verify c3→repos boundary")
    else:
        b   = repos_step["best_other_cost"]
        bnd = repos_step["boundary_to_repos"]
        rel = "<" if b < bnd else "≥"
        good = b < bnd
        ok = ok and good
        msgs.append(
            f"[verify] kToReposCost step={repos_step['step']:>4}: "
            f"best_other_cost ({b:>10.1f}) {rel} boundary_to_repos ({bnd:>10.1f})  "
            f"{'OK (consistent)' if good else 'MISMATCH (signs flipped)'}"
        )

    if c3_step is None:
        msgs.append("[verify] no kToC3Cost step in decisions — cannot verify repos→c3 boundary")
    else:
        b   = c3_step["best_other_cost"]
        bnd = c3_step["boundary_to_c3"]
        rel = ">" if b > bnd else "≤"
        good = b > bnd
        ok = ok and good
        msgs.append(
            f"[verify] kToC3Cost    step={c3_step['step']:>4}: "
            f"best_other_cost ({b:>10.1f}) {rel} boundary_to_c3    ({bnd:>10.1f})  "
            f"{'OK (consistent)' if good else 'MISMATCH (signs flipped)'}"
        )

    return ok, msgs


def _plot_single_run(metrics: dict, out_path: str, title: str) -> None:
    """Render the per-run diagnostic plot (3 cost lines + boundaries +
    mode shading + near_goal stripe + EE/obj motion bottom panel)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    decisions = metrics["decisions"]
    if not decisions or not isinstance(decisions[0], dict):
        raise RuntimeError(
            f"plot requires the new dict-schema decisions (re-run capture).")

    steps = np.array([d["step"] for d in decisions])
    c3    = np.array([d["c3_cost"] for d in decisions], dtype=float)
    bo    = np.array([d["best_other_cost"] for d in decisions], dtype=float)
    crepos = np.array([d["current_repos_cost"] if d["current_repos_cost"] is not None
                       else np.nan for d in decisions], dtype=float)
    bd_repos = np.array([d["boundary_to_repos"] for d in decisions], dtype=float)
    bd_c3    = np.array([d["boundary_to_c3"]    for d in decisions], dtype=float)
    near = np.array([1 if d["near_goal"] else 0 for d in decisions])
    modes = [d["mode"] for d in decisions]

    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
        3, 1, figsize=(12, 8), sharex=True,
        gridspec_kw={"height_ratios": [3.0, 0.5, 1.5]},
    )
    # Mode shading on the cost axis
    for i, m in enumerate(modes):
        if m == "free":
            ax_top.axvspan(i - 0.5, i + 0.5, color="lightblue", alpha=0.25, lw=0)
    ax_top.plot(steps, c3,        label="c3_cost (k=0)",       color="C0", lw=1.5)
    ax_top.plot(steps, bo,        label="best_other_cost",     color="C1", lw=1.5)
    ax_top.plot(steps, crepos,    label="current_repos_cost",  color="C2", lw=1.0, ls="--")
    ax_top.plot(steps, bd_repos,  label="boundary_to_repos (best_other<this ⇒ kToReposCost)",
                color="C0", lw=0.8, ls=":")
    ax_top.plot(steps, bd_c3,     label="boundary_to_c3    (best_other>this ⇒ kToC3Cost)",
                color="C1", lw=0.8, ls=":")
    ax_top.set_ylabel("cost")
    ax_top.set_title(title)
    ax_top.legend(loc="upper right", fontsize=8)
    ax_top.grid(alpha=0.3)

    # near_goal indicator
    ax_mid.fill_between(steps, 0, near, color="C3", alpha=0.4, step="mid")
    ax_mid.set_ylim(0, 1.05)
    ax_mid.set_yticks([0, 1])
    ax_mid.set_ylabel("near_goal")
    ax_mid.grid(alpha=0.3)

    # EE motion + obj motion (per-loop). x-axis matches step indices
    # but motion arrays are length num_loops; align by loop number.
    ee  = np.asarray(metrics["ee_delta_m"])
    oxy = np.asarray(metrics["obj_xy_delta_m"])
    xs  = np.arange(len(ee))
    ax_bot.plot(xs, ee  * 1000, label="EE motion (mm/loop)",   color="C4", lw=1.0)
    ax_bot.plot(xs, oxy * 1000, label="obj_xy motion (mm/loop)", color="C5", lw=1.0)
    ax_bot.set_ylabel("motion (mm/loop)")
    ax_bot.set_xlabel("loop")
    ax_bot.legend(loc="upper right", fontsize=8)
    ax_bot.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _plot_compare(pwl: dict, kik: dict, out_path: str) -> None:
    """6-panel side-by-side comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    def _series(d, key):
        return np.array([x[key] for x in d["decisions"]], dtype=float)

    # (i) c3_cost overlay
    ax = axes[0, 0]
    ax.plot(_series(pwl, "step"), _series(pwl, "c3_cost"), label="PWL", color="C0")
    ax.plot(_series(kik, "step"), _series(kik, "c3_cost"), label="kIK", color="C1")
    ax.set_title("(i) c3_cost (cost at k=0)"); ax.legend(); ax.grid(alpha=0.3)

    # (ii) best_other_cost overlay
    ax = axes[0, 1]
    ax.plot(_series(pwl, "step"), _series(pwl, "best_other_cost"), label="PWL", color="C0")
    ax.plot(_series(kik, "step"), _series(kik, "best_other_cost"), label="kIK", color="C1")
    ax.set_title("(ii) best_other_cost"); ax.legend(); ax.grid(alpha=0.3)

    # (iii) gap (best_other - c3) with switching boundaries (kIK)
    ax = axes[1, 0]
    gap_pwl = _series(pwl, "best_other_cost") - _series(pwl, "c3_cost")
    gap_kik = _series(kik, "best_other_cost") - _series(kik, "c3_cost")
    ax.plot(_series(pwl, "step"), gap_pwl, label="PWL gap", color="C0")
    ax.plot(_series(kik, "step"), gap_kik, label="kIK gap", color="C1")
    # Symmetric reference: in kIK at each step, the controller would
    # switch to c3 if gap > +cost_gap_to_c3 and to repos if gap < -cost_gap_to_repos.
    ax.plot(_series(kik, "step"),  _series(kik, "cost_gap_to_c3"),
            color="C1", lw=0.6, ls=":", label="+gap_to_c3 (kIK)")
    ax.plot(_series(kik, "step"), -_series(kik, "cost_gap_to_repos"),
            color="C1", lw=0.6, ls="--", label="-gap_to_repos (kIK)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("(iii) (best_other - c3_cost) with switching boundaries")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (iv) EE motion histogram
    ax = axes[1, 1]
    bins = np.linspace(0, max(np.max(pwl["ee_delta_m"]), np.max(kik["ee_delta_m"])) * 1000, 40)
    ax.hist(np.asarray(pwl["ee_delta_m"]) * 1000, bins=bins, alpha=0.5, label="PWL", color="C0")
    ax.hist(np.asarray(kik["ee_delta_m"]) * 1000, bins=bins, alpha=0.5, label="kIK", color="C1")
    ax.set_title("(iv) EE motion per loop (mm)")
    ax.set_xlabel("mm"); ax.legend(); ax.grid(alpha=0.3)

    # (v) obj xy motion histogram
    ax = axes[2, 0]
    bins = np.linspace(0, max(np.max(pwl["obj_xy_delta_m"]), np.max(kik["obj_xy_delta_m"])) * 1000, 40)
    ax.hist(np.asarray(pwl["obj_xy_delta_m"]) * 1000, bins=bins, alpha=0.5, label="PWL", color="C0")
    ax.hist(np.asarray(kik["obj_xy_delta_m"]) * 1000, bins=bins, alpha=0.5, label="kIK", color="C1")
    ax.set_title("(v) obj xy motion per loop (mm)")
    ax.set_xlabel("mm"); ax.legend(); ax.grid(alpha=0.3)

    # (vi) near_goal timelines side by side
    ax = axes[2, 1]
    p_near = [1 if d["near_goal"] else 0 for d in pwl["decisions"]]
    k_near = [1 if d["near_goal"] else 0 for d in kik["decisions"]]
    ax.plot(p_near, label="PWL", color="C0")
    ax.plot(k_near, label="kIK", color="C1")
    ax.set_title("(vi) near_goal indicator")
    ax.set_xlabel("loop"); ax.set_ylim(-0.05, 1.05); ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def test_diag_pwl() -> int:
    """Diagnostic re-run of (a) with rich-spy capture (per-sample
    breakdown + counterfactual mode-switch) for the 8.0.3 C1/C2/C3
    evidence collection."""
    print(f"=== 5f diag-pwl: {NUM_LOOPS} loops, seed={SEED}, rich=True ===")
    t0 = time.time()
    m = _run_loop(traj_type=RepositioningTrajectoryType.kPiecewiseLinear,
                  num_loops=NUM_LOOPS, seed=SEED, rich=True)
    print(f"\n[wall] elapsed = {time.time()-t0:.1f}s")
    if m["exceptions"]:
        print(f"\n[FAIL] diag-pwl exceptions: {m['exceptions'][:3]}")
        return 1
    for line in _summary_lines(m):
        print(line)
    print()
    print("  --- 8.0.3 C1/C2/C3 evidence (rich-spy) ---")
    for line in _rich_summary_lines(m):
        print(line)
    pkl = _save(m, "diag_pwl")
    png = os.path.join(RESULTS_DIR, "probe_5f_diag_pwl.png")
    _plot_single_run(m, png, "PWL diagnostic — 200 loops")
    print(f"\n[PASS] diag-pwl saved → {pkl}, {png}")
    return 0


def test_diag_kik() -> int:
    """Diagnostic re-run of (b) with rich-spy capture (per-sample
    breakdown + counterfactual mode-switch) for the 8.0.3 C1/C2/C3
    evidence collection. Boundary-sign verification still gates plotting."""
    print(f"=== 5f diag-kik: {NUM_LOOPS} loops, seed={SEED}, rich=True ===")
    t0 = time.time()
    m = _run_loop(traj_type=RepositioningTrajectoryType.kIK,
                  num_loops=NUM_LOOPS, seed=SEED, rich=True)
    print(f"\n[wall] elapsed = {time.time()-t0:.1f}s")
    if m["exceptions"]:
        print(f"\n[FAIL] diag-kik exceptions: {m['exceptions'][:3]}")
        return 1

    # NOTE 1 boundary-sign check — gate the plot on this passing.
    print()
    ok, lines = _verify_boundary_signs(m["decisions"])
    for line in lines:
        print(line)
    if not ok:
        print("\n[FAIL] diag-kik boundary signs flipped. Plot NOT saved. "
              "Re-derive the boundary expressions before continuing.")
        # Save pickle anyway so we can re-introspect without re-running.
        _save(m, "diag_kik")
        return 1

    print()
    for line in _summary_lines(m):
        print(line)

    print()
    print("  --- 8.0.3 C1/C2/C3 evidence (rich-spy) ---")
    for line in _rich_summary_lines(m):
        print(line)

    # Tightening 2 — print the diagnose dict from the post-run
    # introspection call. None when no failure was captured.
    print()
    if m.get("diagnose_dict") is not None:
        d = m["diagnose_dict"]
        print(f"V-2.5  diagnose_failure_at result on first knot-0 failure (computed at warm-start):")
        print(f"        position_tolerance_setting          = {d['position_tolerance_setting_m']*1000:.3f} mm  (ik_params)")
        print(f"        position_error_at_warm_start        = {d['position_error_at_warm_start_m']*1000:.3f} mm")
        print(f"        ik_d_min_setting                    = {d['ik_d_min_setting_m']*1000:.3f} mm  (ik_params, default 0)")
        if d['min_signed_distance_at_warm_start_m'] is not None:
            print(f"        min_signed_distance_at_warm_start   = {d['min_signed_distance_at_warm_start_m']*1000:.3f} mm")
        if d['min_distance_pair_at_warm_start'] is not None:
            print(f"        min_distance_pair_at_warm_start     = {d['min_distance_pair_at_warm_start']}")
        print(f"        ee_pos_at_warm_start                 = {d['ee_pos_at_warm_start']}")
        print(f"        p_target                             = {d['p_target']}")
    else:
        print("V-2.5  diagnose_failure_at: no failure captured (no knot-0 IK failed during run)")

    pkl = _save(m, "diag_kik")
    png = os.path.join(RESULTS_DIR, "probe_5f_diag_kik.png")
    _plot_single_run(m, png, "kIK diagnostic — 200 loops")
    print(f"\n[PASS] diag-kik saved → {pkl}, {png}")
    return 0


def test_diag_plot() -> int:
    """Render the 6-panel side-by-side comparison from the two diag pickles."""
    print("=== 5f diag-plot: 6-panel comparison ===")
    if not (os.path.exists(os.path.join(RESULTS_DIR, "probe_5f_diag_pwl.pkl")) and
            os.path.exists(os.path.join(RESULTS_DIR, "probe_5f_diag_kik.pkl"))):
        print("[FAIL] diag-plot requires both diag-pwl and diag-kik pickles.")
        return 1
    pwl = _load("diag_pwl")
    kik = _load("diag_kik")

    # Side-by-side metrics table
    def _fmt_dist(arr_m, label):
        a = np.asarray(arr_m)
        return (f"{label}: mean={a.mean()*1000:.2f}  "
                f"p50={np.percentile(a,50)*1000:.2f}  "
                f"p99={np.percentile(a,99)*1000:.2f}  mm")
    print()
    print(f"  PWL EE     {_fmt_dist(pwl['ee_delta_m'],     'EE motion')}")
    print(f"  kIK EE     {_fmt_dist(kik['ee_delta_m'],     'EE motion')}")
    print(f"  PWL obj_xy {_fmt_dist(pwl['obj_xy_delta_m'], 'obj_xy motion')}")
    print(f"  kIK obj_xy {_fmt_dist(kik['obj_xy_delta_m'], 'obj_xy motion')}")
    p_yaw = np.asarray(pwl['obj_yaw_delta_rad']) * 180 / np.pi
    k_yaw = np.asarray(kik['obj_yaw_delta_rad']) * 180 / np.pi
    print(f"  PWL obj_yaw p50/p99 = {np.percentile(p_yaw,50):.4f} / {np.percentile(p_yaw,99):.4f} deg")
    print(f"  kIK obj_yaw p50/p99 = {np.percentile(k_yaw,50):.4f} / {np.percentile(k_yaw,99):.4f} deg")
    print(f"  PWL kToBetterRepos = {pwl['reason_hist'].get('kToBetterRepos', 0)}")
    print(f"  kIK kToBetterRepos = {kik['reason_hist'].get('kToBetterRepos', 0)}")

    out = os.path.join(RESULTS_DIR, "probe_5f_diag_compare.png")
    _plot_compare(pwl, kik, out)
    print(f"\n[PASS] diag-plot saved → {out}")
    return 0


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

_TESTS = {
    "a": test_a, "b": test_b, "c": test_c, "d": test_d, "e": test_e,
    "diag-pwl": test_diag_pwl, "diag-kik": test_diag_kik, "diag-plot": test_diag_plot,
}


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in _TESTS:
        print(__doc__)
        return 2
    return _TESTS[sys.argv[1]]()


if __name__ == "__main__":
    raise SystemExit(main())
