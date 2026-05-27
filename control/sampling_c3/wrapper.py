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

import os
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
from control.osc import OperationalSpaceController
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

        # ----- Executor: OSC (QP) -----
        # OSC is the sole executor. The alternate closed-form impedance
        # executor was removed; see git history if comparison/ablation is
        # needed.  q_nominal matches the IK params' tuned posture
        # (J2=0.325) — the "comfortable" arm pose that keeps gravity-comp
        # under the 30 Nm budget.
        _q_nominal = np.asarray(params.repos_ik_params.q_nominal,
                                dtype=float)[:self.n_u]
        self.executor = OperationalSpaceController(
            plant        = plant,
            ee_frame     = ee_frame,
            n_arm_dofs   = self.n_u,
            q_nominal    = _q_nominal,
            gains_yaml   = params.osc_gains_yaml,
            log_diag     = self.log_diag,
            use_force_tracking = bool(getattr(params, "use_force_tracking", True)),
            W_force      = float(getattr(params, "W_force", 100.0)),
        )

        # Mode state
        self.is_doing_c3 = start_in_c3_mode
        self._prev_mode:                str   = "c3" if start_in_c3_mode else "free"
        self._step:                     int   = 0
        self._did_lcs_dump:             bool  = False  # one-shot trigger for [MATH.LCS-DUMP]
        self._did_cost_dump:            bool  = False  # one-shot trigger for [COST-DUMP]
        self._did_counterfactual_dump:  bool  = False  # one-shot trigger for [COUNTERFACTUAL-DUMP]
        self._did_planvsexec_dump:      bool  = False  # one-shot trigger for [PLAN-VS-EXEC]
        self._pve_pending                       = None   # carries state from record-step → dump-step
        self._n_switches:               int   = 0
        self._step_times_ms:            list  = []
        # Contact-loss disengagement gate (W13 fix). Counts consecutive
        # rich-mode steps where no EE-BOX contact pair was admitted in
        # the LCS. When the streak exceeds DISENGAGE_THRESHOLD, the
        # mode decision is overridden from kStayInC3 to a forced exit.
        self._no_ee_box_streak:         int   = 0

        # λ_planned per-step trace — writes audit_output/lambda_trace.csv
        # at the project root. Captures every rich-mode step (definitive)
        # and a sample of free-mode steps (baseline for ee_box_dist).
        _proj_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
        )
        self._lambda_log_path = os.path.join(
            _proj_root, "audit_output", "lambda_trace.csv"
        )
        os.makedirs(os.path.dirname(self._lambda_log_path), exist_ok=True)
        if (not os.path.exists(self._lambda_log_path)
                or os.path.getsize(self._lambda_log_path) == 0):
            with open(self._lambda_log_path, "w") as _f:
                _f.write(
                    "step,sim_t,mode,lambda_n_ee_box,lambda_n_idx0,"
                    "ee_box_present,n_c,lambda_t_norm,c3_cost,"
                    "ee_box_dist_mm\n"
                )

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
        # Why-replan diagnostics (one-tick lag state).
        self._last_held_existed:        bool                 = False
        self._last_held_cost_logged:    Optional[float]      = None
        self._last_retgt_payload:       Optional[dict]       = None

        self._last_repos_feasible:      bool                 = True
        # Set True by the PWL tracker when the EE has reached the repos
        # target within tolerance. Used as the primary kToC3ReachedReposTarget
        # trigger; the cost-based finished_reposition_cost is a fallback.
        self._last_repos_finished:      bool                 = False

        # ----- Sample buffer for random-ring persistence -----
        # Caches the strategy_samples list (excludes current/prev_repos)
        # across `sample_buffer_lifetime` control loops so the IK tracker
        # has a stable target to converge to. See params.SamplingParams.
        # Refresh triggers: (a) age >= lifetime, (b) finished_repos arrival,
        # (c) n_strategy changes (e.g., mode transition c3↔free).
        self._sample_buffer:            Optional[list]       = None
        self._sample_buffer_age:        int                  = 0
        self._sample_buffer_n_strategy: Optional[int]        = None

        # Public introspection (mirrors legacy attrs)
        self.last_x_seq:               Optional[np.ndarray] = None
        self.last_winning_sample_idx:  Optional[int]        = None
        self.last_mode:                str                  = self._prev_mode

        print(f"[GS] start_mode={'c3' if start_in_c3_mode else 'free'} "
              f"(--prepositioned={start_in_c3_mode})")

    # ------------------------------------------------------------------
    # Sample generation (current EE always at index 0)
    # ------------------------------------------------------------------

    def _get_persistent_samples(self,
                                *,
                                obj_quat:   Optional[np.ndarray] = None,
                                obj_xy:     np.ndarray,
                                g_hat:      np.ndarray,
                                n_strategy: int) -> list[np.ndarray]:
        """Return strategy samples, caching across loops per
        `sampling_params.sample_buffer_lifetime`.

        Refresh triggers (any of):
          * buffer empty (first call after init / arrival)
          * buffer age >= lifetime
          * cached n_strategy differs from the request (mode transition)

        With `lifetime = 0` the buffer is bypassed (re-samples every loop,
        the broken behavior we want to keep available for ablation).
        """
        sp = self.params.sampling_params
        lifetime = int(getattr(sp, "sample_buffer_lifetime", 0))

        # Ablation path: lifetime <= 0 → re-sample every loop.
        if lifetime <= 0:
            return generate_samples(
                strategy  = sp.sampling_strategy,
                n_samples = n_strategy,
                obj_xy    = obj_xy,
                params    = sp,
                rng       = self._rng,
                g_hat     = g_hat,
                obj_quat  = obj_quat,
            )

        # Force refresh on mode-transition n_strategy change so the c3-mode
        # (3 samples) and free-mode (1 sample) buffers don't get mixed.
        need_refresh = (
            self._sample_buffer is None
            or self._sample_buffer_age >= lifetime
            or self._sample_buffer_n_strategy != n_strategy
        )
        if need_refresh:
            self._sample_buffer = generate_samples(
                strategy  = sp.sampling_strategy,
                n_samples = n_strategy,
                obj_xy    = obj_xy,
                params    = sp,
                rng       = self._rng,
                g_hat     = g_hat,
                obj_quat  = obj_quat,
            )
            self._sample_buffer_n_strategy = n_strategy
            self._sample_buffer_age = 0
            if self.log_diag:
                _ages = (self._step, n_strategy,
                         [tuple(np.round(s, 4).tolist())
                          for s in self._sample_buffer])
                print(f"[PERSIST] step={_ages[0]} refresh "
                      f"n_strategy={_ages[1]} samples={_ages[2]}")
        self._sample_buffer_age += 1
        return [s.copy() for s in self._sample_buffer]

    def _refresh_buffer_on_arrival(self) -> None:
        """Force buffer refresh next loop. Called when finished_repos
        fires (EE reached the pursued repos target) so we don't keep
        proposing the already-reached target as a strategy sample."""
        sp = self.params.sampling_params
        lifetime = int(getattr(sp, "sample_buffer_lifetime", 0))
        if lifetime <= 0:
            return
        # Sentinel-trigger the refresh path: clearing the buffer is enough,
        # but setting age past lifetime makes the [PERSIST] log line at
        # the next call explicit.
        self._sample_buffer = None
        self._sample_buffer_age = lifetime + 1

    # ----------------------------------------------------------------------
    def _derive_force_command(self,
                              lambda_n: Optional[np.ndarray],
                              g_hat_3d: np.ndarray) -> np.ndarray:
        """Derive a sustained Cartesian force command for OSC λ_ext tracking.

        Mirrors the dairlib reference's `end_effector_force_target`
        (sampling_based_c3_controller.cc:1508-1515): a force command the
        executor commits to, persisting across momentary LCS contact loss.

        Convention matches the existing F_ff path
        (operational_space_controller.py:185-193 / qp_builder.py docstring):
        λ_ext is the *external force on the EE in world frame* whose
        +J_v^T·λ_ext acts on the arm — i.e., the box-reaction recoil
        direction. To make the EE press the box toward the goal, the
        recoil on the EE points in −g_hat (e.g., box goes west ⇒ EE on
        east ⇒ recoil east = −g_hat for g_hat=[−1,0,0]).

        Magnitude rule:
          * if the LCS admitted an EE-BOX pair at knot 0, use the planner's
            Σ|λ_n| as the intent magnitude (floored at ``min_push_force``).
          * else use ``nominal_push_force`` so the command does NOT collapse
            to zero on momentary contact loss.
        """
        recoil_dir = -np.asarray(g_hat_3d, dtype=float).reshape(3)
        n = float(np.linalg.norm(recoil_dir))
        if n < 1e-9:
            return np.zeros(3)
        recoil_dir = recoil_dir / n

        nominal = float(getattr(self.params, "nominal_push_force", 5.0))
        floor   = float(getattr(self.params, "min_push_force", 2.0))

        has_lam_n = (lambda_n is not None
                     and hasattr(lambda_n, "size")
                     and lambda_n.size > 0)
        if has_lam_n:
            mag = float(np.sum(np.abs(lambda_n)))
            mag = max(mag, floor)
        else:
            mag = nominal
        return mag * recoil_dir

    def _build_samples(self,
                       ee_pos_now:  np.ndarray,
                       obj_xy:      np.ndarray,
                       g_hat:       np.ndarray,
                       prev_mode:   str,
                       obj_quat:    Optional[np.ndarray] = None,
                       ) -> tuple[list[np.ndarray], list[str]]:
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
        strategy_samples = self._get_persistent_samples(
            obj_xy=obj_xy, g_hat=g_hat, n_strategy=n_strategy,
            obj_quat=obj_quat)
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
                        target_xy:  np.ndarray,
                        target_yaw: float = 0.0) -> np.ndarray:
        self._step += 1
        t_step_start = time.perf_counter()

        # Restore plant_ctx (defensive — base_mpc / inner_solver may have
        # left it elsewhere)
        self.plant.SetPositions(plant_ctx,  current_q)
        self.plant.SetVelocities(plant_ctx, current_v)

        # --- Deferred [PLAN-VS-EXEC] dump --------------------------------
        # If we recorded a planner prediction on the previous control step,
        # the actual one-step outcome is now in (current_q, current_v).
        # Print the comparison once, then clear the pending state.
        if self._pve_pending is not None:
            _p = self._pve_pending
            _ox, _oy, _oz = _p["ox"], _p["oy"], _p["oz"]
            _vy_row = _p["vy_row"]
            _x_actual = np.concatenate([current_q, current_v])
            _act_xyz  = np.array([current_q[_ox], current_q[_oy], current_q[_oz]])
            _act_dxyz_dt = (_act_xyz - _p["x0_xyz"]) / _p["dt"]
            _residual = _p["pred_xyz"] - _act_xyz
            _res_xy_norm = float(np.linalg.norm(_residual[:2]))
            # Inspect Drake's contact state at this step (post-integration result).
            try:
                _qo = self.plant.get_geometry_query_input_port().Eval(plant_ctx)
                _sdp_now = _qo.ComputeSignedDistancePairwiseClosestPoints(0.20)
                _ee_ids = self.base_mpc.formulator._ee_geom_ids
                _bx_ids = self.base_mpc.formulator._manipuland_geom_ids
                _pairs_box_ee = [s for s in _sdp_now
                                 if (s.id_A in _bx_ids and s.id_B in _ee_ids)
                                 or (s.id_B in _bx_ids and s.id_A in _ee_ids)]
                _n_active = sum(1 for s in _pairs_box_ee if s.distance < 0.0)
                _closest_d = (min(s.distance for s in _pairs_box_ee)
                              if _pairs_box_ee else float('nan'))
            except Exception as _eg:
                _n_active = -1; _closest_d = float('nan')
            # Apparent contact force on the box from actual Δv (Newton):
            #   F_y = m_box · Δv_y / dt   (ignoring table friction over one Δt)
            _m_box  = _p["m_box"]
            _dv_box = current_v[_vy_row - self.base_mpc.formulator.n_q] - _p["v0_vy"]
            _F_y_apparent = _m_box * _dv_box / _p["dt"]
            # Planner λ_n (impulse over dt): planned force ≈ λ_n
            _lam_n_planned = float(_p["lam_planned"][_p["col_ln"]])
            _ratio = (_lam_n_planned / _F_y_apparent
                      if abs(_F_y_apparent) > 1e-6 else float('nan'))
            np.set_printoptions(linewidth=200, precision=5, suppress=True)
            print(f"[PLAN-VS-EXEC] step={self._step}  (recorded at step={_p['rec_step']}, "
                  f"dt={_p['dt']})")
            print(f"  --- Planner side ---")
            print(f"  x0[box_quat]    = {np.round(_p['x0_quat'], 5)}")
            print(f"  x0[box_xyz]     = {np.round(_p['x0_xyz'],  5)}")
            print(f"  x0[arm_q]       = {np.round(_p['x0_arm_q'], 5)}")
            print(f"  x0[box_v_world] = {np.round(_p['x0_box_v'], 5)}  "
                  f"(ωx,ωy,ωz,vx,vy,vz)")
            print(f"  x0[arm_v]       = {np.round(_p['x0_arm_v'], 5)}")
            print(f"  u_opt[0]        = {np.round(_p['u_opt0'], 5)}")
            print(f"  λ_planned[0]    = {np.round(_p['lam_planned'], 6)}  "
                  f"(γ, λ_n, λ_t × 4)")
            print(f"  φ_at_x0         = {_p['phi']:.6f} m")
            print(f"  --- Predicted (LCS one step) ---")
            print(f"  predicted x1[box_xyz]      = {np.round(_p['pred_xyz'], 6)}")
            print(f"  predicted Δbox_xyz / dt    = {np.round(_p['pred_dxyz_dt'], 6)} m/s")
            print(f"  --- Simulator side (actual @ next call) ---")
            print(f"  actual x1[box_xyz]         = {np.round(_act_xyz, 6)}")
            print(f"  actual Δbox_xyz / dt       = {np.round(_act_dxyz_dt, 6)} m/s")
            print(f"  n_active_pairs (φ<0)       = {_n_active}")
            print(f"  closest box↔EE distance    = {_closest_d:+.5f} m")
            print(f"  apparent F_y_on_box (N)    = {_F_y_apparent:+.4f}   "
                  f"[= m_box·Δv_box_y/dt, m_box={_m_box:.4f}]")
            print(f"  --- Residual ---")
            print(f"  Δ(planned, actual) box_xyz = {np.round(_residual, 6)}")
            print(f"  ‖residual‖ box_xy          = {_res_xy_norm:.6f} m")
            print(f"  λ_n_planned vs F_y_apparent= {_lam_n_planned:.4f} vs "
                  f"{_F_y_apparent:.4f}   ratio={_ratio:.4f}")
            self._pve_pending = None
        # ---------- end deferred PLAN-VS-EXEC dump ----------------------

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
            ee_pos_now, obj_xy, g_hat, self._prev_mode,
            obj_quat=obj_quat)

        # 3. Evaluate every sample (per-sample C3 + alignment + travel)
        results = self.inner_solver.evaluate_samples(
            samples=samples,
            current_q=current_q, current_v=current_v,
            plant_ctx=plant_ctx, target_xy=target_xy,
            ee_pos_now=ee_pos_now, g_hat_3d=g_hat_3d,
            target_yaw=target_yaw,
        )
        c_samples = [r.c_sample for r in results]

        # 3b. Finished-reposition cost penalty.  Mirrors the reference
        #     (dairlib sampling_based_c3_controller.cc:604-608): when the
        #     IK tracker reports the EE is within tolerance of the pursued
        #     repos target, inflate that slot's c_sample by
        #     finished_reposition_cost.  The inflation feeds into both
        #     best_other_cost and the dispatcher's label discrimination
        #     (mode_switch.decide_mode), so the mode-switch trigger remains
        #     cost-based — geometry only contributes as a soft penalty.
        if (self._last_repos_finished
                and len(labels) > 1
                and labels[1] == "prev_repos"):
            c_samples[1] = (c_samples[1]
                            + self.params.progress_params.finished_reposition_cost)

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

        # Contact-proximity entry gate: don't fire kToC3ReachedReposTarget
        # just because the IK arrived at the setback target — require the
        # EE to be close enough to the box that LCS will actually admit
        # an EE-BOX pair on the first c3 step.
        #
        # Diagnosis: the IK's "finished" threshold is 20mm to a 30mm-setback
        # target (reposition_ik.py:1299), so without this gate the EE can
        # be ~35mm shy of the box surface at c3 entry; Drake's 2mm signed-
        # distance threshold (lcs_formulator.py:245) rejects the pair,
        # λ_n=0, and the disengage gate fires after 5 steps. 12/13
        # canonical c3 sessions died this way.
        #
        # Layer 2.6: prefer surface-distance metric (‖ee − box_center‖ −
        # box_half_extent) over the legacy center-distance metric. The
        # surface metric doesn't penalise tangentially-offset rotation
        # samples for being further from the CoM; threshold re-derived to
        # preserve translation engagement.
        #
        # Only the ReachedReposTarget path is affected (via finished_repos
        # -> mode_switch.py:139-140). The kToC3Cost path uses cost-gap
        # hysteresis, independent of finished_repos.
        if (finished_repos
                and getattr(self.params, "use_contact_entry_gate", True)):
            _box_xyz = np.array([
                current_q[self._obj_x_idx],
                current_q[self._obj_y_idx],
                current_q[self._obj_z_idx],
            ])
            _ee_to_box = float(np.linalg.norm(ee_pos_now - _box_xyz))
            if getattr(self.params, "use_surface_entry_gate", True):
                _box_half = float(self.params.sampling_params.box_half_extent)
                _ee_to_surf = _ee_to_box - _box_half
                _thr = float(getattr(self.params,
                                     "contact_entry_surface_threshold", 0.060))
                _block = _ee_to_surf >= _thr
                _label = f"ee_to_surf={_ee_to_surf*1000:.1f}mm"
            else:
                _ee_to_surf = None
                _thr = float(getattr(self.params, "contact_entry_threshold", 0.080))
                _block = _ee_to_box >= _thr
                _label = f"ee_to_box={_ee_to_box*1000:.1f}mm"
            if _block:
                finished_repos = False
                if self.log_diag:
                    print(f"[ENTRY-GATE] step={self._step} "
                          f"{_label} >= thr={_thr*1000:.1f}mm — block "
                          f"kToC3ReachedReposTarget", flush=True)

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

        # 6a-pre. Contact-loss disengagement (W13 fix). The kik config's
        # hyst_c3_to_repos_frac=0.95 makes the cost gate fire only when
        # best_other < 0.05·c3_cost — too sticky to react when the EE
        # has separated from the box entirely. The streak counter is
        # incremented in the [CONTACT-RUN] block below (where the data
        # is authoritative — evaluate_samples above clobbers
        # _last_contact_info with the K-1 sample's contacts). Here we
        # only consume the value: if we're proposing to stay in c3 but
        # the last DISENGAGE_THRESHOLD consecutive c3 steps had no
        # EE-BOX pair, force exit to repos.
        DISENGAGE_THRESHOLD = 5
        if (self._prev_mode == "c3"
                and mode == "c3"
                and self._no_ee_box_streak >= DISENGAGE_THRESHOLD):
            mode = "free"
            reason = SwitchReason.kToReposUnproductive
            if self.log_diag:
                print(f"[CONTACT-LOSS-EXIT] step={self._step} "
                      f"no EE-BOX for {self._no_ee_box_streak} "
                      f"steps -> exit to repos", flush=True)
            self._no_ee_box_streak = 0
        if self._prev_mode == "free":
            # Fresh start when re-entering c3 from free.
            self._no_ee_box_streak = 0

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
        # Populated by the IK tracker in the free branch when target_idx is
        # not None. Read by the impedance override below so free-mode tracks
        # the lift→traverse→descend waypoint path instead of a straight line.
        free_diag = None
        if mode == "c3":
            # [IK-LANDING] dump: every rich entry via kToC3ReachedReposTarget
            # captures p_repos vs actual EE landing vs the box face and Drake's φ.
            if reason == SwitchReason.kToC3ReachedReposTarget:
                # Signal the LCS formulator to dump its filter audit on the
                # next extract_lcs_contacts call (which happens inside the
                # base_mpc.compute_control(...) call below). The formulator
                # one-shots its own dump via _diag_dumped, so setting this
                # on every kToC3ReachedReposTarget is safe; only the first
                # one actually fires the audit.
                self._formulator._rich_mode_just_entered = True
                _p_repos = self._current_repos_target
                _ee_pos  = self.plant.CalcPointsPositions(
                    plant_ctx, self.ee_frame, np.zeros(3), self.world_frame
                ).flatten()
                _p_box   = np.array([current_q[self._obj_x_idx],
                                     current_q[self._obj_y_idx],
                                     current_q[self._obj_z_idx]])
                _g3      = np.array([g_hat[0], g_hat[1], 0.0])
                # Box half-extent along g_hat (box is axis-aligned 0.1×0.1×0.1)
                _face_pt = _p_box - 0.05 * _g3
                # Drake's signed distance (filtered to box↔EE pairs only)
                try:
                    _qo = self.plant.get_geometry_query_input_port().Eval(plant_ctx)
                    _pairs = _qo.ComputeSignedDistancePairwiseClosestPoints(0.50)
                    _ee_ids = self.base_mpc.formulator._ee_geom_ids
                    _bx_ids = self.base_mpc.formulator._manipuland_geom_ids
                    _pairs = [s for s in _pairs
                              if (s.id_A in _bx_ids and s.id_B in _ee_ids)
                              or (s.id_B in _bx_ids and s.id_A in _ee_ids)]
                    _phi = min((s.distance for s in _pairs), default=float('nan'))
                except Exception:
                    _phi = float('nan')
                _ik_err = (float(np.linalg.norm(_ee_pos - _p_repos))
                           if _p_repos is not None else float('nan'))
                _reach  = float(np.linalg.norm(_ee_pos - _face_pt))
                _admits = "Y" if (_phi == _phi and _phi < 0.020) else "N"
                print(f"[IK-LANDING] step={self._step}")
                print(f"  --- Target ---")
                if _p_repos is not None:
                    print(f"  p_repos                        = "
                          f"[{_p_repos[0]:+.4f}, {_p_repos[1]:+.4f}, {_p_repos[2]:+.4f}]")
                else:
                    print(f"  p_repos                        = None")
                print(f"  p_box                          = "
                      f"[{_p_box[0]:+.4f}, {_p_box[1]:+.4f}, {_p_box[2]:+.4f}]")
                print(f"  g_hat                          = [{g_hat[0]:+.4f}, {g_hat[1]:+.4f}]")
                print(f"  conceptual contact pt on box   = "
                      f"[{_face_pt[0]:+.4f}, {_face_pt[1]:+.4f}, {_face_pt[2]:+.4f}]")
                print(f"  --- Actual EE landing ---")
                print(f"  p_ee_actual                    = "
                      f"[{_ee_pos[0]:+.4f}, {_ee_pos[1]:+.4f}, {_ee_pos[2]:+.4f}]")
                print(f"  --- Errors ---")
                print(f"  ||p_ee_actual - p_repos||      = {1000*_ik_err:7.2f} mm  "
                      f"← IK convergence error")
                print(f"  ||p_ee_actual - face_contact|| = {1000*_reach:7.2f} mm  "
                      f"← Total reach error to box face")
                print(f"  φ (Drake signed distance)      = {1000*_phi:7.2f} mm")
                print(f"  φ < 20 mm threshold?           = {_admits}")
                # [IK-SOLVE] / [BODY-VS-CONTACT] — H1/H2/H3/H4 disambiguation.
                _qk=getattr(self.tracker,"last_q_knots",None); _ek=getattr(self.tracker,"last_ee_knots",None)
                _ft=getattr(self.tracker,"last_feasible",None); _pt=getattr(self.tracker,"_prev_target_pos",None)
                if _qk is not None and _ek is not None and _pt is not None:
                    _ee_q=_ek[:,0]; _qa=_qk[:,0]
                    _stat="success" if (_ft and _ft[0]) else "failed"
                    print(f"[IK-SOLVE] step={self._step}")
                    print(f"  IK target body name:    {self.ee_frame.body().name()}")
                    print(f"  IK target position:     [{_pt[0]:+.4f}, {_pt[1]:+.4f}, {_pt[2]:+.4f}]")
                    print(f"  IK solver status:       {_stat}")
                    print(f"  Solved q*:              {np.round(_qa,4).tolist()}")
                    print(f"  Cartesian position of target body at q*: [{_ee_q[0]:+.4f}, {_ee_q[1]:+.4f}, {_ee_q[2]:+.4f}]")
                    print(f"  ||target_body_position - p_target||: {1000*float(np.linalg.norm(_ee_q-_pt)):7.2f} mm")
                _pairs_local = locals().get("_pairs", None)
                if _pairs_local:
                    _s=_pairs_local[0]; _ee_ids_l=self.base_mpc.formulator._ee_geom_ids
                    _pl=_s.p_ACa if _s.id_A in _ee_ids_l else _s.p_BCb
                    _ww=self.plant.CalcPointsPositions(plant_ctx, self.ee_frame, _pl, self.world_frame).flatten()
                    _off=_ww-_ee_pos
                    print(f"[BODY-VS-CONTACT] step={self._step}")
                    print(f"  IK target body position (world):       [{_ee_pos[0]:+.4f}, {_ee_pos[1]:+.4f}, {_ee_pos[2]:+.4f}]")
                    print(f"  Pusher contact point (witness, world): [{_ww[0]:+.4f}, {_ww[1]:+.4f}, {_ww[2]:+.4f}]")
                    print(f"  Offset (contact - body) in world:      [{_off[0]:+.4f}, {_off[1]:+.4f}, {_off[2]:+.4f}]")
                    print(f"  Offset projected onto g_hat:           {1000*float(_off[0]*g_hat[0]+_off[1]*g_hat[1]):7.2f} mm")
            # Rich mode: delegate to base_mpc (it will print its standard
            # [ADMM]/[C3]/[MATH.*] diagnostics). On entry from free we wipe
            # the PI integral and reset the progress tracker so the
            # next-cycle timeout starts from scratch.
            if self._prev_mode == "free":
                self.tracker.reset()
                self.progress.reset()
            u_opt = self.base_mpc.compute_control(
                current_q, current_v, plant_ctx, target_xy,
                target_yaw=target_yaw,
            )
            # [CONTACT-RUN] per-step rich-mode contact diagnostic. Selects
            # the EE-BOX contact pair (not index 0, which may be BOX-GND
            # when ground friction is enabled). Mirrors the EE-BOX index
            # lookup used by the λ-trace logger below. Additive logging
            # only — no behavioral effect.
            _ci = getattr(self.base_mpc.formulator, "_last_contact_info", None)
            if _ci:
                _ee_box_idx_log = next(
                    (_i for _i, _info in enumerate(_ci)
                     if isinstance(_info, dict)
                     and _info.get("tag") == "EE-BOX"),
                    None,
                )
                if _ee_box_idx_log is not None:
                    _ci_sel = _ci[_ee_box_idx_log]
                    _n = _ci_sel["nhat_BA_W"]
                    _p = _ci_sel["p_BCb"]
                    print(f"[CONTACT-RUN] step={self._step} "
                          f"nhat_BA_W=[{_n[0]:+.3f},{_n[1]:+.3f},{_n[2]:+.3f}] "
                          f"p_BCb=[{_p[0]:+.3f},{_p[1]:+.3f},{_p[2]:+.3f}] "
                          f"distance={_ci_sel['distance']:+.5f} "
                          f"contact_type=EE-BOX", flush=True)
                    # Contact-loss disengagement (W13 fix): EE-BOX present
                    # at current EE config — reset the streak. Updated here
                    # (not at top-of-step) because base_mpc.compute_control
                    # restores plant_ctx to the current EE state, so
                    # _last_contact_info here is authoritative; up at the
                    # mode-decision gate it's stale from sample k=K-1.
                    self._no_ee_box_streak = 0
                else:
                    # No EE-BOX pair admitted this step. Emit a tagged
                    # line so the parser can distinguish "no EE-box
                    # contact" from "EE-box contact present".
                    print(f"[CONTACT-RUN] step={self._step} "
                          f"nhat_BA_W=[+0.000,+0.000,+0.000] "
                          f"p_BCb=[+0.000,+0.000,+0.000] "
                          f"distance=+1.00000 "
                          f"contact_type=NONE", flush=True)
                    # Contact-loss disengagement (W13 fix): no EE-BOX at
                    # current EE config — bump the streak. Next step's
                    # mode gate will read this and force exit if ≥ 5.
                    self._no_ee_box_streak += 1
            # One-shot full LCS matrix dump at first rich-mode entry.
            # Triggers exactly once (any step) for the LCS matrix audit.
            if not self._did_lcs_dump:
                self._did_lcs_dump = True
                _f = self.base_mpc.formulator
                _vs = self.obj_body.floating_velocities_start_in_v()
                _row_by = _f.n_q + _vs + 4            # box y-velocity row in x=[q;v]
                _col_ln = _f._last_n_c                # λ_n_first_contact col in λ
                np.set_printoptions(linewidth=200, precision=5, suppress=True)
                print(f"[MATH.LCS-DUMP] step={self._step} n_c={_f._last_n_c} "
                      f"box_y_vel_row={_row_by} lambda_n_first_col={_col_ln}")
                print(f"[MATH.LCS-DUMP] D shape={_f._last_D.shape}\n{_f._last_D}")
                print(f"[MATH.LCS-DUMP] E shape={_f._last_E.shape}\n{_f._last_E}")
                print(f"[MATH.LCS-DUMP] F shape={_f._last_F.shape}\n{_f._last_F}")
                print(f"[MATH.LCS-DUMP] H shape={_f._last_H.shape}\n{_f._last_H}")
                print(f"[MATH.LCS-DUMP] c shape={_f._last_c.shape}\n{_f._last_c}")
            # One-shot cost-decomposition dump at first rich-mode entry with
            # n_c ≥ 1 (admissible contact pairs exist; otherwise H is shape
            # (0, n_u) and the λ_n indexing below would crash).
            if not self._did_cost_dump and self.base_mpc.formulator._last_n_c > 0:
                self._did_cost_dump = True
                _f  = self.base_mpc.formulator
                _qc = self.base_mpc.quad_cost
                _A  = _f._last_A; _B = _f._last_B; _dvec = _f._last_d
                _H  = _f._last_H
                _Q  = self.base_mpc._last_Q
                _R  = self.base_mpc._last_R
                _xref = self.base_mpc._last_x_ref
                _tgt  = self.base_mpc._last_target_xy
                _n_q  = _qc.n_q; _n_u = _qc.n_u
                _ox   = _qc._obj_x_idx; _oy = _qc._obj_y_idx; _oz = _qc._obj_z_idx
                _ops  = _qc._obj_ps;    _vs2 = _qc._obj_vs
                _ovx  = _n_q + _vs2 + 3; _ovy = _n_q + _vs2 + 4
                _x0   = np.concatenate([current_q, current_v])
                _x1   = _A @ _x0 + _B @ u_opt + _dvec      # one-step predicted state
                _err  = _x1 - _xref
                _col_ln_row = _f._last_n_c                  # λ_n row in η = E·x+F·λ+H·u+c
                _Hrow = _H[_col_ln_row, :]
                # Per-term cost contributions at (x_pred=x_1, u_opt)
                _C_obj_xy = _qc.w_obj_xy * (_err[_ox]**2 + _err[_oy]**2)
                _C_obj_z  = (_qc.w_obj_z + _qc.w_box_z) * _err[_oz]**2
                _C_torque = float(u_opt @ _R @ u_opt)
                _C_ee     = float(_err[:_n_u] @ _Q[:_n_u, :_n_u] @ _err[:_n_u])
                _C_perp   = float(_err[[_ovx, _ovy]] @ _Q[np.ix_([_ovx,_ovy],[_ovx,_ovy])] @ _err[[_ovx, _ovy]])
                # Gradients wrt u_opt via x_1 = A·x0 + B·u + d  (dx_1/du = B; ignores λ)
                _g_obj_xy = 2*_qc.w_obj_xy*(_B[_ox,:]*_err[_ox] + _B[_oy,:]*_err[_oy])
                _g_obj_z  = 2*(_qc.w_obj_z + _qc.w_box_z)*_B[_oz,:]*_err[_oz]
                _g_torque = 2.0 * (_R @ u_opt)
                _g_ee     = 2.0 * _B[:_n_u,:].T @ (_Q[:_n_u,:_n_u] @ _err[:_n_u])
                _Qperp    = _Q[np.ix_([_ovx,_ovy],[_ovx,_ovy])]
                _g_perp   = 2.0 * _B[[_ovx,_ovy],:].T @ (_Qperp @ _err[[_ovx,_ovy]])
                np.set_printoptions(linewidth=200, precision=4, suppress=True)
                print(f"[COST-DUMP] step={self._step}")
                print(f"  x_des[box_xy] = [{_xref[_ox]:+.4f}, {_xref[_oy]:+.4f}]")
                print(f"  x_cur[box_xy] = [{current_q[_ox]:+.4f}, {current_q[_oy]:+.4f}]")
                print(f"  goal_xy       = [{_tgt[0]:+.4f}, {_tgt[1]:+.4f}]")
                print(f"  u_opt         = {np.round(u_opt, 4)}")
                print(f"  ||u_opt||     = {np.linalg.norm(u_opt):.4f}")
                print(f"  Q diag (state-cost weights):")
                print(f"    arm_q[0:{_n_u}]   = {np.round(np.diag(_Q)[:_n_u], 3)}")
                print(f"    box_xy            = [{_Q[_ox,_ox]:.2f}, {_Q[_oy,_oy]:.2f}]")
                print(f"    box_z             = {_Q[_oz,_oz]:.2f}")
                print(f"    box_quat[qx,qy]   = [{_Q[_ops+1,_ops+1]:.2f}, {_Q[_ops+2,_ops+2]:.2f}]")
                print(f"    box_vxy           = [{_Q[_ovx,_ovx]:.2f}, {_Q[_ovy,_ovy]:.2f}]  (perp-vel block)")
                print(f"  R diag (input-cost): {np.round(np.diag(_R), 6)}")
                print(f"  Per-term cost @ (x_pred, u_opt):")
                print(f"    w_obj_xy      contrib={_C_obj_xy:+.4e}  ||grad_u||={np.linalg.norm(_g_obj_xy):.4e}")
                print(f"    w_torque      contrib={_C_torque:+.4e}  ||grad_u||={np.linalg.norm(_g_torque):.4e}")
                print(f"    w_ee_approach contrib={_C_ee:+.4e}  ||grad_u||={np.linalg.norm(_g_ee):.4e}")
                print(f"    w_perp(box_v) contrib={_C_perp:+.4e}  ||grad_u||={np.linalg.norm(_g_perp):.4e}")
                print(f"    w_obj_z       contrib={_C_obj_z:+.4e}  ||grad_u||={np.linalg.norm(_g_obj_z):.4e}")
                print(f"  H[λ_n_first_contact={_col_ln_row}, :] = {np.round(_Hrow, 5)}")
                print(f"  ||H[λ_n, :]||  = {np.linalg.norm(_Hrow):.5f}")
                print(f"  max |H[λ_n,:]| = {np.max(np.abs(_Hrow)):.5f}")
            # One-shot counterfactual dump: re-solve C3+ with w_ee_approach=0.
            # Gated on n_c ≥ 1 so the comparison is meaningful (no contact ⇒
            # nothing to counterfactual-test against).
            if not self._did_counterfactual_dump and self.base_mpc.formulator._last_n_c > 0:
                self._did_counterfactual_dump = True
                try:
                    from control.sampling_c3.inner_solve import traj_cost
                    _f   = self.base_mpc.formulator
                    _bmp = self.base_mpc
                    _qc  = _bmp.quad_cost
                    _A, _B, _Dm, _dvec = _f._last_A, _f._last_B, _f._last_D, _f._last_d
                    _E, _F, _H, _c = _f._last_E, _f._last_F, _f._last_H, _f._last_c
                    _Jn, _Jt, _mu, _phi = _f._last_J_n, _f._last_J_t, _f._last_mu, _f._last_phi
                    _Qb   = _bmp._last_Q;  _Rb = _bmp._last_R;  _QNb = _bmp._last_QN
                    _xrb  = _bmp._last_x_ref
                    _tgt  = _bmp._last_target_xy
                    _u_seq_b = _bmp._last_u_seq
                    _x_seq_b = _bmp.last_x_seq
                    _x0   = np.concatenate([current_q, current_v])
                    _n_q  = _qc.n_q; _n_u = _qc.n_u
                    _ox, _oy = _qc._obj_x_idx, _qc._obj_y_idx
                    # Counterfactual: rebuild Q with w_ee_approach=0
                    _Q_cf, _R_cf, _QN_cf, _xr_cf = _qc.build(
                        _tgt, plant_ctx=_bmp._last_plant_ctx,
                        current_q=_bmp._last_current_q, rich_mode=True)
                    # Re-solve with identical LCS, modified cost only.
                    _u_seq_cf, _x_seq_cf = _bmp.solver.solve(
                        _x0, _A, _B, _Dm, _dvec, _Jn, _Jt, _mu,
                        _Q_cf, _R_cf, _QN_cf, _xr_cf,
                        N=_bmp.horizon, admm_iter=_bmp.admm_iter,
                        torque_limit=_bmp.torque_limit, phi=_phi,
                        E=_E, F=_F, H=_H, c_lcs=_c)
                    # Costs at u_opt (full horizon) using the ACTUAL cost of each scenario.
                    _cost_base = traj_cost(_x_seq_b, _u_seq_b, _Qb, _Rb, _QNb, _xrb)
                    _cost_cf   = traj_cost(_x_seq_cf, _u_seq_cf, _Q_cf, _R_cf, _QN_cf, _xr_cf)
                    # Cost at u=0: free-rollout (no contact) x_{t+1} = A x_t + d
                    def _rollout_u0(N):
                        xs = np.zeros((N+1, _x0.size)); xs[0] = _x0
                        for t in range(N):
                            xs[t+1] = _A @ xs[t] + _dvec
                        return xs
                    _xs_u0 = _rollout_u0(_bmp.horizon)
                    _us_u0 = np.zeros_like(_u_seq_b)
                    _cost_u0_base = traj_cost(_xs_u0, _us_u0, _Qb,   _Rb,   _QNb,  _xrb)
                    _cost_u0_cf   = traj_cost(_xs_u0, _us_u0, _Q_cf, _R_cf, _QN_cf, _xr_cf)
                    _dby_base = float(_x_seq_b[-1, _oy]  - _x0[_oy])
                    _dby_cf   = float(_x_seq_cf[-1, _oy] - _x0[_oy])
                    _u_b0  = _u_seq_b[0]; _u_cf0 = _u_seq_cf[0]
                    np.set_printoptions(linewidth=200, precision=4, suppress=True)
                    print(f"[COUNTERFACTUAL-DUMP] step={self._step}")
                    print(f"  --- Baseline (as-shipped) ---")
                    print(f"  u_opt_base    = {np.round(_u_b0, 4)}")
                    print(f"  ||u_opt_base||= {np.linalg.norm(_u_b0):.4f}")
                    print(f"  cost_at_base  = {_cost_base:.4e}")
                    print(f"  cost_at_u0    = {_cost_u0_base:.4e}")
                    print(f"  reduction_base= {(_cost_u0_base - _cost_base):.4e}")
                    print(f"  predicted x_seq[box_xy, end] = [{_x_seq_b[-1,_ox]:+.5f}, {_x_seq_b[-1,_oy]:+.5f}]")
                    print(f"  --- Counterfactual (w_ee_approach = 0) ---")
                    print(f"  u_opt_cf      = {np.round(_u_cf0, 4)}")
                    print(f"  ||u_opt_cf||  = {np.linalg.norm(_u_cf0):.4f}")
                    print(f"  cost_at_cf    = {_cost_cf:.4e}")
                    print(f"  cost_at_u0_cf = {_cost_u0_cf:.4e}")
                    print(f"  reduction_cf  = {(_cost_u0_cf - _cost_cf):.4e}")
                    print(f"  predicted x_seq[box_xy, end] = [{_x_seq_cf[-1,_ox]:+.5f}, {_x_seq_cf[-1,_oy]:+.5f}]")
                    print(f"  --- Comparison ---")
                    print(f"  ||u_opt_cf − u_opt_base|| = {np.linalg.norm(_u_cf0 - _u_b0):.4f}")
                    print(f"  predicted Δbox_y baseline       = {_dby_base:+.5f} m")
                    print(f"  predicted Δbox_y counterfactual = {_dby_cf:+.5f} m")
                    print(f"  goal direction (g_hat): [0, +1]")
                except Exception as _e:
                    import traceback as _tb
                    print(f"[COUNTERFACTUAL-DUMP] FAILED: {type(_e).__name__}: {_e}")
                    print(_tb.format_exc())
            # Record planner-side data for the [PLAN-VS-EXEC] dump fired on
            # the next call. Compares LCS one-step prediction against the
            # simulator's actual result by reading current_q on the next entry.
            if not self._did_planvsexec_dump and self.base_mpc.formulator._last_n_c > 0:
                self._did_planvsexec_dump = True
                _f  = self.base_mpc.formulator
                _qc = self.base_mpc.quad_cost
                _A, _B, _Dm, _dvec = _f._last_A, _f._last_B, _f._last_D, _f._last_d
                _x0  = np.concatenate([current_q, current_v])
                # Back out λ_planned[0] from D·λ = x_seq[1] - A·x0 - B·u_opt - d
                _xseq = self.base_mpc.last_x_seq
                _u0   = self.base_mpc._last_u_seq[0]
                _rhs  = _xseq[1] - _A @ _x0 - _B @ _u0 - _dvec
                _lam_planned, *_ = np.linalg.lstsq(_Dm, _rhs, rcond=None)
                _pred_x1 = _A @ _x0 + _B @ _u0 + _Dm @ _lam_planned + _dvec
                _ox, _oy, _oz = _qc._obj_x_idx, _qc._obj_y_idx, _qc._obj_z_idx
                _x0_xyz   = np.array([current_q[_ox], current_q[_oy], current_q[_oz]])
                _pred_xyz = np.array([_pred_x1[_ox], _pred_x1[_oy], _pred_x1[_oz]])
                _n_q  = _qc.n_q
                _vs2  = _qc._obj_vs
                _vy_row_in_x = _n_q + _vs2 + 4
                # Box mass from Drake (sum of inertias on the manipuland body).
                try:
                    _m_box = float(self.obj_body.get_default_mass())
                except Exception:
                    _m_box = 0.2  # fall back to known box mass
                self._pve_pending = dict(
                    rec_step=self._step,
                    dt=self.base_mpc.dt,
                    ox=_ox, oy=_oy, oz=_oz,
                    vy_row=_vy_row_in_x,
                    col_ln=_f._last_n_c,           # λ_n_first_contact index in λ
                    m_box=_m_box,
                    x0_quat=current_q[_qc._obj_ps:_qc._obj_ps+4].copy(),
                    x0_xyz=_x0_xyz.copy(),
                    x0_arm_q=current_q[:_qc.n_u].copy(),
                    x0_box_v=current_v[_vs2:_vs2+6].copy(),
                    x0_arm_v=current_v[:_qc.n_u].copy(),
                    v0_vy=float(current_v[_vs2 + 4]),
                    u_opt0=_u0.copy(),
                    lam_planned=_lam_planned.copy(),
                    phi=float(_f._last_phi[0]) if len(_f._last_phi) > 0 else float('nan'),
                    pred_xyz=_pred_xyz.copy(),
                    pred_dxyz_dt=((_pred_xyz - _x0_xyz) / self.base_mpc.dt).copy(),
                )
            self.last_x_seq             = self.base_mpc.last_x_seq
            self._current_repos_target  = None
            self._current_repos_cost    = None
            self._last_repos_finished   = False
            self._prev_logged_repos_target = None
            self._last_held_existed       = False
            self._last_held_cost_logged   = None
            self._last_retgt_payload      = None
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
                    current_q, current_v, plant_ctx, target_xy,
                    target_yaw=target_yaw)
                self.last_x_seq = self.base_mpc.last_x_seq
                self._current_repos_target = None
                self._current_repos_cost   = None
                best_src = "current_fallback"
            else:
                p_repos = results[target_idx].sample_pos
                self._current_repos_target = p_repos.copy()
                self._current_repos_cost   = c_samples[target_idx]
                best_src = labels[target_idx]

                # Why-replan diagnostics. Compare the won target against the
                # "held" candidate (the prev_repos slot — k=1 when present).
                # Selection is pure argmin (wrapper.py:610), so a small margin
                # over the held candidate means cost noise flipped the
                # argmin: the absent stickiness hypothesis.
                _held_idx = (1 if (len(labels) > 1 and labels[1] == "prev_repos")
                             else None)
                _held_cost = (c_samples[_held_idx]
                              if _held_idx is not None else None)
                _won_cost  = c_samples[target_idx]
                _margin    = ((_held_cost - _won_cost)
                              if _held_cost is not None else None)
                # retgt: did the dispatcher abandon the held target?
                #   Y if there was no held slot,
                #     or the won index ≠ the held index.
                if _held_idx is None:
                    _retgt = (self._last_held_existed)  # was held last tick?
                    _reason = "no_held"
                else:
                    _retgt = (target_idx != _held_idx)
                    if not _retgt:
                        _reason = "hold"
                    else:
                        # margin / held_cost — noise-flip if relative gap tiny.
                        if _held_cost > 0:
                            _rel = abs(_margin) / abs(_held_cost)
                        else:
                            _rel = float("inf")
                        if _rel < 0.01:
                            _reason = "noise_flip"  # <1% gap (likely cost noise)
                        else:
                            # Compare held_cost to prior tick's held_cost.
                            _prev_held = getattr(
                                self, "_last_held_cost_logged", None)
                            if (_prev_held is not None and _prev_held > 0
                                    and _held_cost > 1.10 * _prev_held):
                                _reason = "held_cost_rose"
                            else:
                                _reason = "new_sample_better"
                # held_still_valid: was the held sample's solve feasible? If
                # not, "held" was structurally invalid this tick.
                if _held_idx is not None:
                    _held_valid = bool(results[_held_idx].feasible)
                else:
                    _held_valid = False
                self._last_retgt_payload = dict(
                    retgt       = _retgt,
                    held_idx    = _held_idx,
                    held_cost   = _held_cost,
                    won_idx     = target_idx,
                    won_cost    = _won_cost,
                    margin      = _margin,
                    won_src     = best_src,
                    held_valid  = _held_valid,
                    reason      = _reason,
                )
                self._last_held_existed = (_held_idx is not None)
                self._last_held_cost_logged = _held_cost

                self.tracker._diag_step = self._step  # [IK-CONVERGE] plumb
                u_opt, free_diag = self.tracker.compute_torque(
                    current_q=current_q, current_v=current_v,
                    plant_ctx=plant_ctx, p_target=p_repos,
                    dt_ctrl=self._dt_ctrl,
                )
                # Capture trajectory-finished signal for the next loop's
                # mode-switch decision (kToC3ReachedReposTarget).
                self._last_repos_finished = bool(
                    free_diag.get("finished", False))
                # On arrival, force the ring-sample buffer to refresh next
                # loop. Otherwise the now-reached point persists as a
                # strategy sample and the cost gate keeps re-firing
                # kToC3ReachedReposTarget for it.
                if self._last_repos_finished:
                    self._refresh_buffer_on_arrival()

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

        # --- Executor (OSC or impedance) ---------------------------------
        # The branches above produced an informational `u_opt` (planner
        # u_seq[0] in c3, a zero placeholder from the IK tracker in free).
        # The executor below always overrides it; the tracker's job is to
        # supply the Cartesian waypoint `free_diag['p_des']`, not the
        # actuated torque.
        # Predefine c3-only locals so the unified [STEP] line below can
        # reference them unconditionally (free branch leaves them None).
        _lam_n   = None
        _lam_t   = None
        _lam_des = None
        # v_ee_desired feedforward intentionally disabled in v1: the IK
        # knot spacing produces a much larger effective velocity than the
        # task tracking can absorb without saturating every joint at
        # URDF limits. Revisit once the OSC baseline (position-only
        # tracking) is verified.
        if mode == "c3":
            # Cartesian target from C3+'s next-step state prediction.
            _x_seq = self.base_mpc.last_x_seq
            if _x_seq is not None and len(_x_seq) > 1:
                _q_full_next = current_q.copy()
                _q_full_next[:self.n_u] = _x_seq[1][:self.n_u]
                self.plant.SetPositions(plant_ctx, _q_full_next)
                _p_ee_des = self.plant.CalcPointsPositions(
                    plant_ctx, self.ee_frame, np.zeros(3), self.world_frame,
                ).flatten()
                self.plant.SetPositions(plant_ctx, current_q)
                self.plant.SetVelocities(plant_ctx, current_v)
            else:
                _p_ee_des = ee_pos_now
            _lam_n = getattr(self.base_mpc, "last_lambda_n_first", None)
            _lam_t = getattr(self.base_mpc, "last_lambda_t_first", None)
            _Jn    = self.base_mpc.formulator._last_J_n
            _Jt    = self.base_mpc.formulator._last_J_t
            _lam_des = self._derive_force_command(_lam_n, g_hat_3d)
            u_imp, imp_diag = self.executor.compute_torque(
                current_q, current_v, plant_ctx,
                p_ee_desired = _p_ee_des,
                v_ee_desired = None,
                lambda_n     = _lam_n,
                lambda_t     = _lam_t,
                J_n          = _Jn,
                J_t          = _Jt,
                lambda_des   = _lam_des,
            )
        else:
            # Free mode: follow the IK tracker's piecewise-linear waypoint
            # path (lift → traverse → descend) instead of the straight line
            # to the perpendicular-contact target. The straight line plows
            # through the box in East/South. free_diag['p_des'] is the FK
            # of IK knot 0 — the next waypoint the tracker is aiming for.
            _p_des_wp = free_diag.get("p_des") if free_diag is not None else None
            if _p_des_wp is not None:
                _p_ee_des = _p_des_wp
            elif self._current_repos_target is not None:
                _p_ee_des = self._current_repos_target
            else:
                _p_ee_des = ee_pos_now
            u_imp, imp_diag = self.executor.compute_torque(
                current_q, current_v, plant_ctx,
                p_ee_desired = _p_ee_des,
                v_ee_desired = None,
                lambda_n     = None,
                lambda_t     = None,
                J_n          = None,
                J_t          = None,
            )
        u_opt = u_imp

        # --- λ_planned per-step trace ---------------------------------
        # Pure additive logging: writes to audit_output/lambda_trace.csv.
        # Every rich-mode step is captured; free-mode is sampled every
        # 50 steps for an ee_box_dist baseline. Failures are swallowed
        # so logging cannot break a run.
        try:
            _box_xyz = np.array([
                current_q[self._obj_x_idx],
                current_q[self._obj_y_idx],
                current_q[self._obj_z_idx],
            ])
            _ee_box_dist = float(np.linalg.norm(ee_pos_now - _box_xyz))
            _sim_t = self._step * self._dt_ctrl
            if mode == "c3":
                _lam_n_first = getattr(
                    self.base_mpc, "last_lambda_n_first", None)
                _lam_t_first = getattr(
                    self.base_mpc, "last_lambda_t_first", None)
                _ci_log = getattr(
                    self.base_mpc.formulator, "_last_contact_info", None)
                _ee_box_idx = None
                if _ci_log is not None:
                    for _i, _info in enumerate(_ci_log):
                        if isinstance(_info, dict) and \
                                _info.get("tag") == "EE-BOX":
                            _ee_box_idx = _i
                            break
                if (_lam_n_first is not None
                        and hasattr(_lam_n_first, "__len__")
                        and len(_lam_n_first) > 0):
                    _lam_n_val_idx0 = float(_lam_n_first[0])
                else:
                    _lam_n_val_idx0 = float("nan")
                if (_ee_box_idx is not None
                        and _lam_n_first is not None
                        and hasattr(_lam_n_first, "__len__")
                        and len(_lam_n_first) > _ee_box_idx):
                    _lam_n_val_ee_box = float(_lam_n_first[_ee_box_idx])
                    _ee_box_present = 1
                else:
                    _lam_n_val_ee_box = 0.0
                    _ee_box_present = 0
                if (_lam_t_first is not None
                        and hasattr(_lam_t_first, "__len__")
                        and len(_lam_t_first) > 0):
                    _lam_t_norm = float(np.linalg.norm(_lam_t_first))
                else:
                    _lam_t_norm = float("nan")
                _n_c = getattr(
                    self.base_mpc.formulator, "_last_n_c", -1)
                with open(self._lambda_log_path, "a") as _f:
                    _f.write(
                        f"{self._step},{_sim_t:.3f},c3,"
                        f"{_lam_n_val_ee_box:.6f},{_lam_n_val_idx0:.6f},"
                        f"{_ee_box_present},{_n_c},{_lam_t_norm:.6f},"
                        f"NaN,{_ee_box_dist*1000:.3f}\n"
                    )
            elif self._step % 50 == 0:
                with open(self._lambda_log_path, "a") as _f:
                    _f.write(
                        f"{self._step},{_sim_t:.3f},free,"
                        f"NaN,NaN,0,NaN,NaN,NaN,{_ee_box_dist*1000:.3f}\n"
                    )
        except Exception as _exc:
            try:
                with open(self._lambda_log_path, "a") as _f:
                    _f.write(
                        f"{getattr(self, '_step', -1)},NaN,"
                        f"{mode}-ERROR,NaN,NaN,0,NaN,NaN,NaN,NaN\n"
                    )
            except Exception:
                pass

        if self.log_diag and (self._step % 10 == 0 or self._step <= 5):
            _ln_max = (float(np.max(np.abs(imp_diag["tau_ff"])))
                       if imp_diag["had_lambda_n"] or imp_diag["had_lambda_t"]
                       else 0.0)
            print(f"[IMP] step={self._step} mode={mode} "
                  f"|x_err|={np.linalg.norm(imp_diag['x_err']):.4f}m "
                  f"|tau_imp|={np.linalg.norm(imp_diag['tau_imp']):.2f}Nm "
                  f"|tau_ff|={_ln_max:.2f}Nm "
                  f"|tau_out|={np.linalg.norm(u_opt):.2f}Nm "
                  f"sat={imp_diag['saturated']} "
                  f"lam_n={imp_diag['had_lambda_n']} lam_t={imp_diag['had_lambda_t']}")

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
            # [STEP] unified per-step diagnostic: one line per control loop
            # that follows the EE through both regimes. Free mode carries
            # the reposition-tracking payload (the dominant 2% target-reach
            # failure diagnosis); c3 mode carries the contact payload.
            # Replaces nothing — [GS]/[GS-tgt] still emit for backward compat
            # with the target-chase overlay (befaed1).
            self._print_unified_step(
                step           = self._step,
                mode           = mode,
                switch_reason  = reason,
                ee_pos_now     = ee_pos_now,
                obj_xy         = obj_xy,
                current_q      = current_q,
                goal_dist      = goal_dist,
                g_hat          = g_hat,
                p_ee_des       = _p_ee_des,
                free_diag      = free_diag,
                curr_cost      = c_samples[0],
                lam_n          = _lam_n,
                lam_t          = _lam_t,
                lam_des        = _lam_des,
            )
            # [GATE-EVOLVE] — one line per loop: cost-ratio trajectory + EE
            # geometric progress toward the perpendicular-contact optimal
            # target. The cost gate fires when curr/best_other < hyst frac
            # (default 0.9); ee_to_optimal tells us whether the EE is even
            # near the productive-contact pose where that ratio could drop.
            _r_proxy = float(self.params.sampling_params.repos_target_radius)
            _optimal_xy = obj_xy - _r_proxy * g_hat
            _ee_to_optimal = float(np.linalg.norm(
                np.array([ee_pos_now[0] - _optimal_xy[0],
                          ee_pos_now[1] - _optimal_xy[1]])))
            try:
                _qo_ge = self.plant.get_geometry_query_input_port().Eval(plant_ctx)
                _pairs_ge = _qo_ge.ComputeSignedDistancePairwiseClosestPoints(0.50)
                _ee_ids_ge = self.base_mpc.formulator._ee_geom_ids
                _bx_ids_ge = self.base_mpc.formulator._manipuland_geom_ids
                _pairs_ge = [s for s in _pairs_ge
                             if (s.id_A in _bx_ids_ge and s.id_B in _ee_ids_ge)
                             or (s.id_B in _bx_ids_ge and s.id_A in _ee_ids_ge)]
                _phi_ge = min((s.distance for s in _pairs_ge), default=float('nan'))
            except Exception:
                _phi_ge = float('nan')
            _curr_ge = c_samples[0]
            _best_other_ge = best_other_cost
            if _best_other_ge != float("inf") and _best_other_ge > 0:
                _ratio_ge = _curr_ge / _best_other_ge
            else:
                _ratio_ge = float('nan')
            print(f"[GATE-EVOLVE] step={self._step} "
                  f"curr={_curr_ge:.2f} "
                  f"best_other={_best_other_ge:.2f} "
                  f"ratio={_ratio_ge:.4f} "
                  f"ee_to_optimal={_ee_to_optimal:.4f}m "
                  f"phi={_phi_ge:.4f}m")
            if self._step % 20 == 0:
                self._print_table_diag(self._step, samples, labels, results, k_star)

        # 10. Bookkeeping
        # Venkatesh 2025 §IV-D Step 4: on rich→free transition, refresh the
        # sample buffer and clear the prev_repos slot so the next loop picks
        # the lowest-cost sample from a freshly-evaluated set rather than
        # re-selecting the stale prev_repos target.
        if self._prev_mode == "c3" and mode == "free":
            self._refresh_buffer_on_arrival()
            self._current_repos_target     = None
            self._current_repos_cost       = None
            self._prev_logged_repos_target = None
            self._last_repos_finished      = False
            if self.log_diag:
                print(f"[RICH-EXIT-REFRESH] step={self._step} "
                      f"mode {self._prev_mode}->{mode} reason={reason.name} "
                      f"forcing buffer refresh + clearing prev_repos")
        self._prev_mode              = mode
        self.last_mode               = mode
        self.last_winning_sample_idx = k_star
        self._step_times_ms.append((time.perf_counter() - t_step_start) * 1e3)

        return u_opt

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _print_unified_step(self, *, step, mode, switch_reason,
                            ee_pos_now, obj_xy, current_q, goal_dist, g_hat,
                            p_ee_des, free_diag,
                            curr_cost, lam_n, lam_t, lam_des):
        """One mode-aware [STEP] line per control loop.

        Free-mode suffix: reposition-tracking payload (the 2% target-reach
        diagnostic — target / ee_to_target / landing_err / ik_ok / ik_resid /
        q_max_resid_deg / target_changed).

        C3-mode suffix: contact payload (c3_cost / lam_n / lam_t / contact /
        productive / f_cmd).
        """
        sim_t = step * self._dt_ctrl
        # Object xyz (lift z from current_q for parity with [GS-tgt]).
        obj_z = float(current_q[self._obj_z_idx])
        prefix = (
            f"[STEP] step={step} mode={mode} t={sim_t:.3f}s "
            f"ee=({ee_pos_now[0]:+.3f},{ee_pos_now[1]:+.3f},{ee_pos_now[2]:+.3f}) "
            f"obj=({obj_xy[0]:+.3f},{obj_xy[1]:+.3f},{obj_z:+.3f}) "
            f"goal_dist={goal_dist:.3f}m "
            f"switch={switch_reason.name}"
        )

        if mode != "c3":
            # Free mode: reposition payload from the tracker diag.
            # tgt = next-knot waypoint (~one planner step ahead); ee_stride is
            # the per-tick stride to that waypoint, NOT the gap to the final
            # reposition goal. ee_to_ptarget = ||ee_now - p_target|| is the
            # true goal gap (added in the horizon-coverage fix).
            tgt = (p_ee_des if p_ee_des is not None else ee_pos_now)
            ee_stride = float(np.linalg.norm(np.asarray(tgt) - ee_pos_now))
            _ptgt = self._current_repos_target
            ee_to_ptarget = (float(np.linalg.norm(np.asarray(_ptgt) - ee_pos_now))
                             if _ptgt is not None else float("nan"))
            if free_diag is not None:
                landing  = free_diag.get("landing_err", float("nan"))
                ik_ok    = "Y" if free_diag.get("knot0_feasible", False) else "N"
                ik_resid = free_diag.get("ik_err", float("nan"))
                qmax_deg = free_diag.get("q_max_resid_deg", float("nan"))
                # Executed-march three-candidate trace (instrument-only).
                _sp_pt   = free_diag.get("setpoint_to_ptarget", float("nan"))
                _fin_val = free_diag.get("finished_val", float("nan"))
                _nwp_pt  = free_diag.get("nextwp_to_ptarget", float("nan"))
                _gterm   = free_diag.get("guide_terminal_err", float("nan"))
                _pdresid = free_diag.get("pd_resid_ee", float("nan"))
            else:
                landing, ik_ok, ik_resid, qmax_deg = (
                    float("nan"), "?", float("nan"), float("nan"))
                _sp_pt = _fin_val = _nwp_pt = _gterm = _pdresid = float("nan")
            # Mirror [GS-tgt]'s target-changed logic on a separate cache so
            # the two lines stay independent.
            if not hasattr(self, "_prev_step_target") or self._prev_step_target is None:
                tgt_changed = "Y"
            else:
                tgt_changed = ("Y" if float(np.linalg.norm(
                    np.asarray(tgt) - self._prev_step_target)) > 1e-3 else "N")
            self._prev_step_target = np.asarray(tgt).copy()
            # Why-replan payload (populated in the free-mode branch above).
            rp = getattr(self, "_last_retgt_payload", None)
            if rp is not None:
                retgt_tag    = "Y" if rp["retgt"] else "N"
                held_idx_tag = (str(rp["held_idx"])
                                if rp["held_idx"] is not None else "-")
                hc = rp["held_cost"]
                wc = rp["won_cost"]
                mg = rp["margin"]
                held_cost_str = (f"{hc:.2f}" if hc is not None else "-")
                won_cost_str  = f"{wc:.2f}"
                margin_str    = (f"{mg:+.2f}" if mg is not None else "-")
                held_valid_tag = "Y" if rp["held_valid"] else "N"
                reason_tag     = rp["reason"]
                won_src_tag    = rp["won_src"]
            else:
                retgt_tag = held_idx_tag = won_cost_str = "?"
                held_cost_str = margin_str = won_src_tag = "?"
                held_valid_tag = "?"
                reason_tag = "?"
            print(
                f"{prefix} "
                f"target=({tgt[0]:+.3f},{tgt[1]:+.3f},{tgt[2]:+.3f}) "
                f"ee_stride={ee_stride:.3f}m "
                f"ee_to_ptarget={ee_to_ptarget:.3f}m "
                f"landing_err={float(landing):.4f}m "
                f"ik_ok={ik_ok} ik_resid={float(ik_resid):.4f}m "
                f"q_max_resid_deg={float(qmax_deg):.2f} "
                f"target_changed={tgt_changed} "
                f"retgt={retgt_tag} held_idx={held_idx_tag} "
                f"held_cost={held_cost_str} won_cost={won_cost_str} "
                f"margin={margin_str} won_src={won_src_tag} "
                f"held_valid={held_valid_tag} reason={reason_tag} "
                f"setpoint_to_ptarget={float(_sp_pt):.4f}m "
                f"finished_val={float(_fin_val):.4f}m "
                f"finished_thresh=0.0200m "
                f"pd_resid={float(_pdresid):.4f}m "
                f"nextwp_to_ptarget={float(_nwp_pt):.4f}m "
                f"guide_terminal_err={float(_gterm):.4f}m"
            )
            return

        # C3 mode: contact payload.
        # lam_n = max over EE-BOX pair component (n=0 fallback when present).
        ci = getattr(self.base_mpc.formulator, "_last_contact_info", None)
        ee_box_idx = None
        nhat_xy = None
        if ci:
            for i, info in enumerate(ci):
                if isinstance(info, dict) and info.get("tag") == "EE-BOX":
                    ee_box_idx = i
                    n = info.get("nhat_BA_W")
                    if n is not None and len(n) >= 2:
                        nhat_xy = (float(n[0]), float(n[1]))
                    break
        contact = "Y" if ee_box_idx is not None else "N"
        if (lam_n is not None and hasattr(lam_n, "__len__")
                and len(lam_n) > 0):
            if ee_box_idx is not None and len(lam_n) > ee_box_idx:
                lam_n_val = float(lam_n[ee_box_idx])
            else:
                lam_n_val = float(np.max(np.abs(lam_n)))
        else:
            lam_n_val = 0.0
        if lam_t is not None and hasattr(lam_t, "__len__") and len(lam_t) > 0:
            lam_t_val = float(np.linalg.norm(lam_t))
        else:
            lam_t_val = 0.0
        # Productive direction: nhat (box→EE) anti-aligned with g_hat
        # (goal direction) — i.e., nhat·g_hat < -0.3 (same threshold as
        # parser's attribution predicate at parse_log_to_jsonl.py:306).
        if nhat_xy is not None:
            dot = nhat_xy[0] * g_hat[0] + nhat_xy[1] * g_hat[1]
            productive = "Y" if dot < -0.3 else "N"
        else:
            productive = "N"
        # f_cmd: planner-derived OSC force command (force-tracking mode).
        if lam_des is not None and hasattr(lam_des, "__len__"):
            f_cmd = (float(lam_des[0]), float(lam_des[1]), float(lam_des[2]))
        else:
            f_cmd = (0.0, 0.0, 0.0)
        print(
            f"{prefix} "
            f"c3_cost={float(curr_cost):.2f} "
            f"lam_n={lam_n_val:.3f} lam_t={lam_t_val:.3f} "
            f"contact={contact} productive={productive} "
            f"f_cmd=({f_cmd[0]:+.2f},{f_cmd[1]:+.2f},{f_cmd[2]:+.2f})"
        )

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
                  f"rot={r.rot_score:+.4f}(bonus={r.rot_bonus:8.2f}) "
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
        # OSC end-of-run summary (QP failure rate, saturation rate, avg
        # solve time).
        self.executor.print_summary()
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
