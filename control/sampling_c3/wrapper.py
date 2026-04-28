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
from control.sampling_c3.params import SamplingC3Params, SamplingStrategy
from control.sampling_c3.progress import ProgressTracker, StepMetrics
from control.sampling_c3.reposition import PiecewiseLinearTracker
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
                 start_in_c3_mode: bool = False):
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
        self.tracker  = PiecewiseLinearTracker(
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

        # Repos-target memo (the sample we are currently navigating toward)
        self._current_repos_target:     Optional[np.ndarray] = None
        self._current_repos_cost:       Optional[float]      = None
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
        # Reposition is "finished" if either:
        #   (a) the PWL tracker reached the target within tolerance LAST step
        #       (trajectory-based, primary signal — mirrors upstream's
        #        finished_reposition_flag in reposition.h)
        #   (b) the predicted cost falls below finished_reposition_cost
        #       (cost-based, kept as a fallback for cases where trajectory
        #        finish doesn't fire — e.g. retarget mid-path)
        finished_repos = self._last_repos_finished or (
            self._current_repos_cost is not None
            and self._current_repos_cost < self.params.progress_params.finished_reposition_cost
        )
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

        if mode != self._prev_mode:
            self._n_switches += 1

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
