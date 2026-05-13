"""
SamplingC3MPC — top-level outer controller.

Wraps an existing C3MPC and orchestrates:
  - per-step generation of K candidate EE positions (via sampling.py)
  - per-sample C3 evaluation + alignment + travel cost (inner_solve.py)
  - paper §IV-D mode-switch decision (mode_switch.py + progress.py)
  - sample buffer maintenance (sample_buffer.py)
  - reposition-mode trajectory + tracking (reposition.py)
  - rich-mode dispatch back to base_mpc.compute_control()

Public surface mirrors the legacy GlobalSamplingC3MPC so main.py's
sim loop only needs the constructor swapped. Specifically:

    last_x_seq                — for Meshcat predicted-trajectory marker
    last_winning_sample_idx   — for diagnostics
    last_mode                 — "rich" (= "c3") or "free"
    print_perf_summary()      — called once at end-of-sim
"""
from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from control.sampling_c3.inner_solve import (
    InnerSolver, SampleResult, traj_cost_breakdown,
)
from control.sampling_c3.mode_switch import SwitchReason, decide_mode
from control.sampling_c3.params import (
    SamplingC3Params, SamplingStrategy, RepositioningTrajectoryType,
)
from control.sampling_c3.progress import ProgressTracker, StepMetrics
from control.sampling_c3.reposition import PiecewiseLinearTracker
from control.sampling_c3.reposition_ik import RepositionIKTracker
from control.sampling_c3.sample_buffer import BufferedSample, SampleBuffer
from control.sampling_c3.sampling import generate_samples


class SamplingC3MPC:
    """Replaces the legacy GlobalSamplingC3MPC. Accepts a single
    SamplingC3Params object instead of a dozen individual kwargs."""

    def __init__(self,
                 base_mpc,
                 plant,
                 ee_frame,
                 obj_body,
                 params:     SamplingC3Params,
                 log_diag:   bool = True,
                 rng:        Optional[np.random.Generator] = None,
                 dt_ctrl:    float = 0.01,
                 start_in_c3_mode: bool = False,
                 *,
                 diagram=None):
        """Construct the outer sampling-C3 controller.

        Parameters
        ----------
        diagram : optional. Required ONLY when
            ``params.reposition_params.traj_type ==
            RepositioningTrajectoryType.kIK``. The IK tracker walks
            ``diagram.GetSystems()`` to find the SceneGraph for
            context-local collision filtering. PiecewiseLinearTracker
            does not use the diagram and ignores this kwarg.
        """
        self.base_mpc    = base_mpc
        self.plant       = plant
        self.ee_frame    = ee_frame
        self.world_frame = plant.world_frame()
        self.obj_body    = obj_body
        self.params      = params
        self.log_diag    = bool(log_diag)
        self._rng        = rng if rng is not None else np.random.default_rng()
        # Physical control rate (sim step). Used by the PWL tracker to
        # convert params.reposition_params.speed [m/s] to ds-per-call.
        # base_mpc.dt is the *planning* timestep (0.05s), which is 5×
        # too fast — must NOT be used here.
        self._dt_ctrl    = float(dt_ctrl)

        # Inner stack references
        self._formulator = base_mpc.formulator
        self._solver     = base_mpc.solver
        self._quad_cost  = base_mpc.quad_cost
        self._horizon    = base_mpc.horizon
        self._dt         = base_mpc.dt
        self._tlim       = base_mpc.torque_limit
        self._admm_iter  = base_mpc.admm_iter

        self.n_u = plant.num_actuators()
        self.n_q = plant.num_positions()
        self.n_v = plant.num_velocities()

        # Object-pose indices
        ps = obj_body.floating_positions_start()
        self._obj_x_idx = ps + 4
        self._obj_y_idx = ps + 5
        self._obj_z_idx = ps + 6
        self._obj_qw    = ps + 0
        self._obj_qx    = ps + 1
        self._obj_qy    = ps + 2
        self._obj_qz    = ps + 3

        # Sub-systems
        self.inner_solver = InnerSolver(
            plant=plant, ee_frame=ee_frame, obj_body=obj_body,
            formulator=self._formulator,
            solver=self._solver,
            quad_cost=self._quad_cost,
            horizon=self._horizon,
            dt=self._dt,
            torque_limit=self._tlim,
            base_admm_iter=self._admm_iter,
            params=params,
        )
        self.progress = ProgressTracker(params.progress_params)
        self.buffer   = SampleBuffer(
            capacity      = params.sampling_params.N_sample_buffer,
            pos_threshold = params.sampling_params.pos_error_sample_retention,
            ang_threshold = params.sampling_params.ang_error_sample_retention,
        )
        # Reposition-tracker dispatch on traj_type. The kIK path needs the
        # diagram so it can build its own private diag_ctx for IK and apply
        # the context-local collision filter to that context's SceneGraph
        # (see RepositionIKTracker.__init__). Other traj types use the PWL
        # tracker, which has no SceneGraph dependency.
        _traj_type = params.reposition_params.traj_type
        if _traj_type == RepositioningTrajectoryType.kIK:
            if diagram is None:
                raise ValueError(
                    "SamplingC3MPC: traj_type=kIK requires diagram=. Pass the "
                    "diagram returned by build_environment() through to the "
                    "wrapper. PiecewiseLinearTracker does not require this."
                )
            # Resolve scene_graph by walking the diagram's subsystems —
            # build_environment() does not return it, but Drake exposes it
            # as a child system of the diagram. Filter-and-assert-exactly-one
            # so a future builder that adds a second SceneGraph (e.g. for a
            # separate visualisation diagram) fails loudly instead of having
            # us pick an arbitrary one. If you genuinely want to disambiguate,
            # add a scene_graph= kwarg here and short-circuit this lookup.
            import pydrake.all as ad
            _sgs = [s for s in diagram.GetSystems() if isinstance(s, ad.SceneGraph)]
            if len(_sgs) != 1:
                raise ValueError(
                    f"SamplingC3MPC: diagram contains {len(_sgs)} SceneGraphs, "
                    f"expected exactly 1. Pass scene_graph= explicitly if you "
                    f"have multiple."
                )
            scene_graph = _sgs[0]
            self.tracker = RepositionIKTracker(
                plant=plant, ee_frame=ee_frame, obj_body=obj_body,
                n_arm_dofs=self.n_u,
                horizon=self._horizon,
                dt=self._dt,
                repos_params=params.reposition_params,
                ik_params=params.repos_ik_params,
                diagram=diagram,
                scene_graph=scene_graph,
                # table_body=None — defaults to plant.world_body() (env_builder
                # registers the table on the world body).
            )
        else:
            self.tracker = PiecewiseLinearTracker(
                plant=plant, ee_frame=ee_frame,
                n_arm_dofs=self.n_u,
                params=params.reposition_params,
            )

        # Mode state
        self.is_doing_c3 = start_in_c3_mode
        self._prev_mode:                str   = "c3" if start_in_c3_mode else "free"
        self._step:                     int   = 0
        self._n_switches:               int   = 0
        self._step_times_ms:            list  = []

        # 9.4.7 Option A — 1d watchdog re-test under F2 regime.
        # _n_watchdog_fires tracks how many times the steps_since_improve
        # threshold forced a free→c3 transition. _mode_time_{c3,free}
        # tally per-step mode residency for the end-of-run summary.
        self._n_watchdog_fires:         int   = 0
        self._mode_time_c3:             int   = 0
        self._mode_time_free:           int   = 0

        # Repos-target memo (the sample we are currently navigating toward)
        self._current_repos_target:     Optional[np.ndarray] = None
        self._current_repos_cost:       Optional[float]      = None
        self._prev_logged_repos_target: Optional[np.ndarray] = None

        # _infeasible_repos_target — the most-recent infeasible repos
        # target. Single-slot: successive failures overwrite. This is
        # intentional — the cost-inflation in compute_control (~line 329)
        # treats it as a single proximity bound, and the poison's only
        # job is to bias the NEXT loop's argmin away from a known-bad
        # point. Once the controller commits to a new target far enough
        # from the cached one (Site D, _maybe_clear_infeasible_poison),
        # or transitions back to C3 (Site B, unconditional clear in the
        # `if mode == "c3":` branch), the cache is cleared. Site C (the
        # no-candidate fallback at "target_idx is None") does NOT clear,
        # even though it triggers a "no commit" — the poison's
        # information is still useful against next loop's fresh samples.
        # PWL path leaves this at None forever (its tracker has no
        # last_knot0_feasible attr; getattr default True keeps the stash
        # site in 5c a no-op).
        self._infeasible_repos_target:  Optional[np.ndarray] = None
        self._last_repos_feasible:      bool                 = True
        # Set True by the PWL tracker when the EE has reached the repos
        # target within tolerance. Used as the primary kToC3ReachedReposTarget
        # trigger; the cost-based finished_reposition_cost is a fallback.
        self._last_repos_finished:      bool                 = False

        # Public introspection (mirrors legacy attrs)
        self.last_x_seq:               Optional[np.ndarray] = None
        self.last_winning_sample_idx:  Optional[int]        = None
        self.last_mode:                str                  = self._prev_mode

        print(f"[GS] start_mode={'c3' if start_in_c3_mode else 'free'} "
              f"(--prepositioned={start_in_c3_mode})")

    # ------------------------------------------------------------------
    # Sample generation (current EE always at index 0)
    # ------------------------------------------------------------------

    def _build_samples(self,
                       ee_pos_now:  np.ndarray,
                       obj_xy:      np.ndarray,
                       g_hat:       np.ndarray,
                       prev_mode:   str) -> tuple[list[np.ndarray], list[str]]:
        """Construct the per-loop sample list. Returns (positions, labels).

        Layout:
            k=0                             current EE
            k=1                             previous repos target (if active)
            k=2..2+N-1                      strategy samples
            (optional final)                buffer's best (when leaving C3)
        """
        sp = self.params.sampling_params

        positions: list[np.ndarray] = [ee_pos_now.copy()]
        labels:    list[str]        = ["current"]

        if self._current_repos_target is not None:
            positions.append(self._current_repos_target.copy())
            labels.append("prev_repos")

        n_strategy = (sp.num_additional_samples_c3 if prev_mode == "c3"
                      else sp.num_additional_samples_repos)
        strategy_samples = generate_samples(
            strategy  = sp.sampling_strategy,
            n_samples = n_strategy,
            obj_xy    = obj_xy,
            params    = sp,
            rng       = self._rng,
            g_hat     = g_hat,
        )
        for i, p in enumerate(strategy_samples):
            positions.append(p)
            labels.append(f"strat_{i}")

        if (sp.consider_best_buffer_sample_when_leaving_c3
                and prev_mode == "c3"
                and len(self.buffer) > 0):
            best = self.buffer.best_with_position()
            if best is not None:
                positions.append(best.position.copy())
                labels.append("buffer")

        return positions, labels

    # ------------------------------------------------------------------
    # Buffer maintenance
    # ------------------------------------------------------------------

    def _update_buffer(self,
                       results:    list[SampleResult],
                       obj_xy_now: np.ndarray,
                       obj_quat:   np.ndarray) -> None:
        # Age existing entries; prune those whose object pose has drifted
        self.buffer.tick_age()
        self.buffer.prune(obj_xy_now, obj_quat_now=obj_quat)

        # Append the best non-current feasible result so we remember it
        # across the next mode switch.
        ranked = sorted(
            ((r.c_sample, k) for k, r in enumerate(results)
             if k > 0 and r.feasible),
        )
        if ranked:
            _, best_k = ranked[0]
            r = results[best_k]
            self.buffer.append(BufferedSample(
                position   = r.sample_pos.copy(),
                cost       = r.c_sample,
                obj_pos_xy = obj_xy_now.copy(),
                obj_quat   = obj_quat.copy(),
            ))

    # ------------------------------------------------------------------
    # Infeasibility-poison cache management
    # ------------------------------------------------------------------

    def _maybe_clear_infeasible_poison(self, p_new: np.ndarray) -> None:
        """Clear the cached infeasible-repos-target IFF p_new differs
        from it by more than infeasibility_match_radius_m. Called
        whenever the controller assigns a new pursued repos target
        (mode-switch or within-repos hysteresis swap). Idempotent:
        no-op if no poison is cached.
        """
        if self._infeasible_repos_target is None:
            return
        r = self.params.repos_ik_params.infeasibility_match_radius_m
        if float(np.linalg.norm(
                p_new - self._infeasible_repos_target)) > r:
            self._infeasible_repos_target = None

    # ------------------------------------------------------------------
    # Main control entry
    # ------------------------------------------------------------------

    def compute_control(self,
                        current_q:  np.ndarray,
                        current_v:  np.ndarray,
                        plant_ctx,
                        target_xy:  np.ndarray) -> np.ndarray:
        self._step += 1
        t_step_start = time.perf_counter()

        # Restore plant_ctx (defensive — base_mpc / inner_solver may have
        # left it elsewhere)
        self.plant.SetPositions(plant_ctx,  current_q)
        self.plant.SetVelocities(plant_ctx, current_v)

        # 1. Geometry: object xy, EE position, goal direction
        obj_xy = np.array([current_q[self._obj_x_idx],
                            current_q[self._obj_y_idx]])
        obj_quat = np.array([current_q[self._obj_qw],
                              current_q[self._obj_qx],
                              current_q[self._obj_qy],
                              current_q[self._obj_qz]])
        v_goal   = target_xy - obj_xy
        goal_dist = float(np.linalg.norm(v_goal))
        g_hat   = v_goal / (goal_dist + 1e-9)
        g_hat_3d = np.array([g_hat[0], g_hat[1], 0.0])

        ee_pos_now = self.plant.CalcPointsPositions(
            plant_ctx, self.ee_frame, np.zeros(3), self.world_frame,
        ).flatten().copy()

        # 2. Build sample list (k=0 = current EE always first)
        samples, labels = self._build_samples(
            ee_pos_now, obj_xy, g_hat, self._prev_mode)

        # 3. Evaluate every sample (per-sample C3 + alignment + travel)
        results = self.inner_solver.evaluate_samples(
            samples=samples,
            current_q=current_q, current_v=current_v,
            plant_ctx=plant_ctx, target_xy=target_xy,
            ee_pos_now=ee_pos_now, g_hat_3d=g_hat_3d,
        )
        c_samples = [r.c_sample for r in results]

        # 3a. Infeasibility poison — inflate any sample within the match
        #     radius of a previously-failed kIK reposition target. The
        #     stash is set in step 8 after a non-feasible knot-0 IK and
        #     held until the controller commits to a different repos
        #     target (>= match radius away — see 5e). Fires only when the
        #     IK tracker reported infeasibility; PWL leaves the stash at
        #     None forever.
        if self._infeasible_repos_target is not None:
            _r = self.params.repos_ik_params.infeasibility_match_radius_m
            for k, _p in enumerate(samples):
                if float(np.linalg.norm(_p - self._infeasible_repos_target)) < _r:
                    c_samples[k] = float("inf")

        # 4. Pick winner (k* = argmin c_sample over all samples)
        k_star = int(np.argmin(c_samples))
        c_curr   = c_samples[0]
        best_other_idx = None
        best_other_cost = float("inf")
        for k in range(1, len(c_samples)):
            if c_samples[k] < best_other_cost:
                best_other_cost = c_samples[k]
                best_other_idx  = k

        # 5. Update progress tracker (uses k=0 cost = c_curr)
        # config_cost ≈ box-xy-error² weighted by w_obj_xy (kConfigCost
        # equivalent for our pushing task)
        w_obj_xy = self._quad_cost.w_obj_xy
        config_cost_now = w_obj_xy * (goal_dist ** 2)
        self.progress.update(StepMetrics(
            c3_cost     = c_curr,
            config_cost = config_cost_now,
            pos_error   = goal_dist,
            rot_error   = 0.0,   # no rotation goal in pushing task
        ))

        # 6. Mode-switch decision
        near_goal = goal_dist < self.params.progress_params.cost_switching_threshold_distance
        # Reposition is "finished" iff the PWL tracker reports the EE within
        # tolerance of the target on the previous control step. Trajectory-
        # based signal from reposition.py:244 (is_at_target with 2 cm tol),
        # mirroring upstream's finished_reposition_flag in reposition.h.
        #
        # The previous implementation also OR'd a cost-based fallback
        # (_current_repos_cost < finished_reposition_cost). This was
        # structurally broken: c_sample is dominated by box-xy goal tracking
        # (~80-200k for the pushing task), so no setting of
        # finished_reposition_cost cleanly distinguishes "EE reached the
        # repos target" from "EE has not reached it but cost is bounded".
        # F-cheap diagnostic with threshold=1.0 confirmed the chatter
        # disappears when Path B is disabled.
        finished_repos = self._last_repos_finished
        met = self.progress.met_progress(near_goal=near_goal)
        mode, reason = decide_mode(
            prev_mode          = self._prev_mode,
            c3_cost            = c_curr,
            best_other_cost    = best_other_cost,
            current_repos_cost = self._current_repos_cost,
            met_progress       = met,
            near_goal          = near_goal,
            finished_repos     = finished_repos,
            params             = self.params.progress_params,
        )

        # 6a. 1d watchdog override (9.4.7 Option A re-test). When the
        # configured threshold is > 0 and steps_since_improve has reached it
        # while in free mode, force c3 regardless of cost arithmetic. The
        # progress reset on the free→c3 transition (line ~430 below) zeroes
        # steps_since_improve, so the next fire is at least `threshold`
        # loops away. Disabled (default) when threshold = 0.
        _wd_thresh = self.params.progress_params.watchdog_steps_since_improve_threshold
        _wd_si     = self.progress.steps_since_improve()
        if (_wd_thresh > 0 and self._prev_mode == "free"
                and _wd_si >= _wd_thresh and mode != "c3"):
            mode = "c3"
            reason = SwitchReason.kForceC3Watchdog
            self._n_watchdog_fires += 1
            if self.log_diag:
                print(f"[GS-watchdog] step={self._step} "
                      f"steps_since_improve={_wd_si} threshold={_wd_thresh} "
                      f"FORCE c3-mode  total_fires={self._n_watchdog_fires}")

        if mode != self._prev_mode:
            self._n_switches += 1

        # Residency tally for the end-of-run summary
        if mode == "c3":
            self._mode_time_c3 += 1
        else:
            self._mode_time_free += 1

        # 7. Maintain sample buffer (independent of mode)
        self._update_buffer(results, obj_xy, obj_quat)

        # 8. Execute
        if mode == "c3":
            # Rich mode: delegate to base_mpc (it will print its standard
            # [ADMM]/[C3]/[MATH.*] diagnostics). On entry from free we wipe
            # the PI integral and reset the progress tracker so the
            # next-cycle timeout starts from scratch.
            if self._prev_mode == "free":
                self.tracker.reset()
                self.progress.reset()
            u_opt = self.base_mpc.compute_control(
                current_q, current_v, plant_ctx, target_xy,
            )
            self.last_x_seq             = self.base_mpc.last_x_seq
            self._current_repos_target  = None
            self._current_repos_cost    = None
            self._last_repos_finished   = False
            self._prev_logged_repos_target = None
            # Site B (5e): unconditional poison clear on repos→C3. Once
            # back in C3, the prior repos target's feasibility is
            # irrelevant; next free-mode entry re-evaluates from scratch.
            # Idempotent — fires every C3 step, not just transitions.
            self._infeasible_repos_target = None
            best_src = "current"

        else:
            # Free mode: pick a repos target and run the PWL tracker.
            # If k_star == 0 (current EE wins on c_sample but we're in
            # free per the mode-switch logic — typically because progress
            # timed out), use the best non-current sample as the target.
            if k_star == 0 or k_star is None:
                target_idx = best_other_idx
            else:
                target_idx = k_star

            if target_idx is None:
                # No candidates at all (only current EE). Fall back to
                # base_mpc — should be unreachable when num_additional_*≥1.
                u_opt = self.base_mpc.compute_control(
                    current_q, current_v, plant_ctx, target_xy)
                self.last_x_seq = self.base_mpc.last_x_seq
                self._current_repos_target = None
                self._current_repos_cost   = None
                best_src = "current_fallback"
            else:
                p_repos = results[target_idx].sample_pos
                # Site D (5e): controller is committing to a new pursued
                # target — clear stale poison if the new target is
                # outside the match radius of the cached failed one.
                self._maybe_clear_infeasible_poison(p_repos)
                self._current_repos_target = p_repos.copy()
                self._current_repos_cost   = c_samples[target_idx]
                best_src = labels[target_idx]

                u_opt, free_diag = self.tracker.compute_torque(
                    current_q=current_q, current_v=current_v,
                    plant_ctx=plant_ctx, p_target=p_repos,
                    dt_ctrl=self._dt_ctrl,
                )
                # Capture trajectory-finished signal for the next loop's
                # mode-switch decision (kToC3ReachedReposTarget).
                self._last_repos_finished = bool(free_diag.get("finished", False))

                # Stash the failed target if the kIK tracker reported a
                # non-feasible knot-0 IK. PiecewiseLinearTracker has no
                # last_knot0_feasible attr — getattr defaults True, so this
                # is a no-op for the PWL path. The cost-inflation block
                # in compute_control reads _infeasible_repos_target; 5e
                # clears it when the pursued target moves >= match radius.
                self._last_repos_feasible = getattr(
                    self.tracker, "last_knot0_feasible", True)
                if not self._last_repos_feasible:
                    self._infeasible_repos_target = p_repos.copy()

                if self.log_diag:
                    ee_now = free_diag.get("ee_now")
                    if ee_now is not None:
                        d = float(np.linalg.norm(ee_now - p_repos))
                        if self._prev_logged_repos_target is None:
                            tgt_changed = "Y"
                            tgt_delta = float("nan")
                        else:
                            tgt_delta = float(np.linalg.norm(
                                p_repos - self._prev_logged_repos_target))
                            tgt_changed = "Y" if tgt_delta > 1e-3 else "N"
                        print(f"[GS-tgt] step={self._step} "
                              f"ee=({ee_now[0]:+.3f},{ee_now[1]:+.3f},{ee_now[2]:+.3f}) "
                              f"p_repos=({p_repos[0]:+.3f},{p_repos[1]:+.3f},{p_repos[2]:+.3f}) "
                              f"ee_to_target={d:.3f}m "
                              f"target_label={best_src} "
                              f"target_changed={tgt_changed} delta={tgt_delta:.3f}m")
                        self._prev_logged_repos_target = p_repos.copy()

                # Predicted trajectory for Meshcat visualisation: use the
                # winning sample's plan if available
                if results[k_star].x_seq is not None:
                    self.last_x_seq = results[k_star].x_seq

        # 9. Diagnostics
        if self.log_diag:
            self._print_step_diag(
                step=self._step, mode=mode, switch_reason=reason,
                best_k=k_star, best_src=best_src,
                c_samples=c_samples,
                best_other_cost=best_other_cost,
                met_progress=met,
                steps_since_improve=self.progress.steps_since_improve(),
            )
            if self._step % 20 == 0:
                self._print_table_diag(self._step, samples, labels, results, k_star)

        # 10. Bookkeeping
        self._prev_mode              = mode
        self.last_mode               = mode
        self.last_winning_sample_idx = k_star
        self._step_times_ms.append((time.perf_counter() - t_step_start) * 1e3)

        return u_opt

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _print_step_diag(self, *, step, mode, switch_reason, best_k, best_src,
                         c_samples, best_other_cost,
                         met_progress, steps_since_improve):
        repos_cost_str = (f"{self._current_repos_cost:.2f}"
                          if self._current_repos_cost is not None else "-")
        best_other_str = (f"{best_other_cost:.2f}"
                          if best_other_cost != float("inf") else "-")
        print(f"[GS] step={step} mode={mode} switch={switch_reason.name} "
              f"best_k={best_k} best_src={best_src} "
              f"curr_cost={c_samples[0]:.2f} repos_cost={repos_cost_str} "
              f"best_other={best_other_str} "
              f"met_progress={'Y' if met_progress else 'N'} "
              f"steps_since_improve={steps_since_improve} "
              f"switches={self._n_switches}")

    def _print_table_diag(self, step, samples, labels, results, k_star):
        print(f"[GS-table] step={step}")
        for k, (p, lbl, r) in enumerate(zip(samples, labels, results)):
            win = "  ← WIN" if k == k_star else ""
            feas = "Y" if r.feasible else "N"
            print(f"  k={k} ({lbl:<10}) "
                  f"pos=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}) "
                  f"c_C3={r.c_C3_raw:10.2f} "
                  f"align={r.align_score:.4f}(bonus={r.align_bonus:8.2f}) "
                  f"travel={r.travel_dist:.3f}m(pen={r.travel_penalty:6.2f}) "
                  f"c_sample={r.c_sample:10.2f} "
                  f"feas={feas} ik_err={r.ik_err:.4f}m{win}")

    def print_perf_summary(self) -> None:
        avg_ms = (sum(self._step_times_ms) / len(self._step_times_ms)
                  if self._step_times_ms else 0.0)
        print(f"[GS-perf] avg_per_step_ms={avg_ms:.1f}  "
              f"full_solves={self.inner_solver.full_solves}  "
              f"cheap_solves={self.inner_solver.cheap_solves}  "
              f"switches={self._n_switches}")
        # 9.4.7 Option A — watchdog summary. Only printed when the
        # threshold is enabled in config (otherwise tally is 0 and the
        # line is uninformative).
        if self.params.progress_params.watchdog_steps_since_improve_threshold > 0:
            total = self._mode_time_c3 + self._mode_time_free
            frac = self._mode_time_c3 / total if total > 0 else 0.0
            print(f"[GS-watchdog-summary] n_watchdog_events={self._n_watchdog_fires}  "
                  f"mode_time_c3={self._mode_time_c3}  "
                  f"mode_time_free={self._mode_time_free}  "
                  f"c3_fraction={frac:.3f}")
