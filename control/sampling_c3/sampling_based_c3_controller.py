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

import json
import os
import time
from typing import List, Optional

import numpy as np
from pydrake.trajectories import PiecewisePolynomial

from control.sampling_c3.commit_face_gate import decide_commit_face_gate
from control.sampling_c3.inner_solve import (
    InnerSolver, SampleResult, traj_cost_breakdown,
)
from control.sampling_c3.mode_switch import SwitchReason, decide_mode
from control.sampling_c3.params import (
    SamplingC3Params, SamplingStrategy, RepositioningTrajectoryType,
)
from control.sampling_c3.progress import ProgressTracker, StepMetrics
from control.osc import OperationalSpaceController
from control.osc.dynamics_helpers import ee_jacobian_translational
from control.sampling_c3.reposition import PiecewiseLinearTracker
from control.sampling_c3.reposition_trajectory import RepositionTrajectory
from control.sampling_c3.sample_buffer import BufferedSample, SampleBuffer
from control.sampling_c3.sampling import generate_samples
from sim.env_builder import PUSHER_RADIUS

# Lever 3: c3 approach-closing override constants.
# LCS_DISTANCE_THRESHOLD matches lcs_formulator.extract_lcs_contacts's
# distance_threshold (2 mm), the value at which Drake's signed-distance
# query admits an EE-box contact pair into the LCS.
# MAX_APPROACH_STEP caps the per-tick advance of the commanded EE target;
# the clamp min(MAX, surf_dist - threshold) keeps the commanded target
# >= threshold outside the contact surface, so it cannot command penetration.
LCS_DISTANCE_THRESHOLD = 0.002
MAX_APPROACH_STEP      = 0.010

# B3c-prime selection-audit lazy-init sentinel. _sel_audit_state starts at
# this value; on the FIRST hit at wrapper.py:848 the lazy-init reads env vars
# once and replaces it with either None (disabled) or a (lo, hi) int tuple.
# Hot path outside the window or when disabled is two attribute reads + an
# `is`/`None` compare + (if enabled) one int-range check — no env reads, no
# comprehensions, no f-strings.
_SEL_AUDIT_UNINIT = object()


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
        self.progress = ProgressTracker(params.progress_params,
                                        dt_ctrl=self._dt_ctrl)
        self.buffer   = SampleBuffer(
            capacity      = params.sampling_params.N_sample_buffer,
            pos_threshold = params.sampling_params.pos_error_sample_retention,
            ang_threshold = params.sampling_params.ang_error_sample_retention,
        )
        # PiecewiseLinearTracker is the only reference-conformant tracker.
        # kIK (port-only) was deleted with reposition_ik.py.
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

        # §7.31/§7.32 reference-faithful contact-establishment path (surface
        # EE desired-state + planner-tracked velocity feedforward + skip the
        # 100 mm-behind approach-cost proxy). Formerly gated by REF_RECONCILE_
        # APPROACH; now always on. Requires always-on LCS row
        # (LCS_ALWAYS_ON_EE_BOX=1) — without it the removed proxy re-opens
        # the free-mode freeze trap.
        import os as _os_rec
        # §7.35 — feedforward-accel SUB-GATE (§7.34 banked, default-OFF).
        # Independent of the reconcile path; opts in to the a_ff leg of the
        # OSC PD-plus-feedforward law once the source is clean (planner
        # converges) or a mitigation lands.
        _env_ffa = _os_rec.environ.get("REF_RECONCILE_FEEDFORWARD_ACCEL", "")
        self._reconcile_feedforward_accel = (
            bool(int(_env_ffa)) if _env_ffa else False)

        # §7.55 — PUSHA_DISABLE_CONTACT_LOSS_GATE (default-OFF) skips the
        # CONTACT-LOSS-EXIT watchdog at _solve_plan() that forces c3→repos
        # after `_no_ee_box_streak ≥ contact_loss_threshold_*_s/dt_ctrl`
        # consecutive no-contact ticks. The reference (dairlib_sampling_c3
        # @257e3ed, systems/controllers/sampling_based_c3_controller.cc:1150-
        # 1184) has NO such watchdog — it exits c3 only via cost+hysteresis
        # or the cost-based progress timeout. With PUSHA_DISABLE_C3_OVERRIDE=1
        # (§7.51), the disengage threshold drops to contact_loss_threshold_
        # default_s = 5 ticks at 100 Hz, which bounces c3 out before contact
        # admits (§7.54 root-cause). This flag is SEPARATE from PUSHA_DISABLE_
        # C3_OVERRIDE so the effect of the watchdog removal is cleanly
        # attributable. Default-OFF byte-identical preserved.
        # Default flipped to disable the port-only watchdog by default —
        # the reference dispatcher has no such gate. Explicit re-enable
        # via PUSHA_DISABLE_CONTACT_LOSS_GATE=0 for legacy runs.
        self._disable_contact_loss_gate = (
            _os_rec.environ.get("PUSHA_DISABLE_CONTACT_LOSS_GATE", "1") == "1")

        # Force-tracking follows params.use_force_tracking (True by default).
        # The reconcile path no longer silently overrides it to False —
        # historical §7.31 silent-off + §7.63 decouple-workaround retired
        # with the flag removal.
        _use_force_tracking = bool(getattr(params, "use_force_tracking", True))
        if self._reconcile_feedforward_accel:
            print("[§7.35] REF_RECONCILE_FEEDFORWARD_ACCEL=1 → feedforward-"
                  "accel ENABLED (§7.34 OVER-DRIVES regime; source-conditional)",
                  flush=True)

        # §7.42 — when PUSHA_REF_OSC_ALIGN=1, override W_force to 1.0 to match
        # the reference's `LambdaEndEffectorW = diag(1,1,1)` (osc_params.yaml:74).
        # The bundling env flag (set in main.py) wires the routing + u-bounds +
        # R-cost env vars together; W_force needs its own override here because
        # it is read at construction (not via env at use-site like the others).
        _ref_align = (os.environ.get("PUSHA_REF_OSC_ALIGN", "0") == "1")
        _W_force_val = (1.0 if _ref_align
                        else float(getattr(params, "W_force", 100.0)))
        self.executor = OperationalSpaceController(
            plant        = plant,
            ee_frame     = ee_frame,
            n_arm_dofs   = self.n_u,
            q_nominal    = _q_nominal,
            gains_yaml   = params.osc_gains_yaml,
            log_diag     = self.log_diag,
            use_force_tracking = _use_force_tracking,
            W_force      = _W_force_val,
        )

        # Stage A — Reposition PWL trajectory port (alignment plan §3).
        # Default-None until first free-mode entry. Rebuilt on (a) target
        # change > 5 mm, or (b) c3 → free transition (cleared at boundary).
        # Sampled at control rate in the free-mode OSC branch.
        self._use_pwl_traj: bool = bool(
            getattr(params, "use_reposition_pwl_trajectory", False))
        self._pwl_traj: Optional[RepositionTrajectory] = None
        self._pwl_traj_built_for_target: Optional[np.ndarray] = None
        self._pwl_traj_last_build_step: int = -1
        if self._use_pwl_traj:
            print("[STAGE-A-PWL] dispatcher: "
                  "use_reposition_pwl_trajectory=True", flush=True)

        # Mode state
        self.is_doing_c3 = start_in_c3_mode
        self._prev_mode:                str   = "c3" if start_in_c3_mode else "free"
        self._step:                     int   = 0
        # T-architecture Stage 2b: rate-decoupling state. At default rates
        # (params.dt_osc == params.dt_mpc == 0.01) N_plan=1 and the planner
        # solves every tick with elapsed=0; Stage 2c flips dt_mpc to e.g.
        # 0.025 so the planner solves every 25th OSC tick.
        self._dt_osc:                   float = float(params.dt_osc)
        self._dt_mpc:                   float = float(params.dt_mpc)
        self._last_plan_tick:           int   = -1
        self._last_plan_ctx                   = None
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
        # Override-grace flag. Set True at the end of the approach-closing
        # override emit block on each tick the override actually commands
        # an approach target; cleared otherwise. Read by the contact-loss
        # gate on the NEXT tick (1-tick lag) to pick the extended grace
        # threshold instead of the strict default.
        self._approach_override_firing: bool  = False
        # Phase the override was firing in last tick. One of:
        #   'none'         — override did not fire
        #   'A_lift_trav'  — LTD PHASE A (lift-and-traverse)
        #   'B_descend'    — LTD PHASE B (descend beside box)
        #   'C_approach'   — LTD PHASE C (push into face)
        #   'fallback_ws'  — LTD workspace fallback (legacy direct-line)
        #   'legacy'       — LTD disabled, legacy direct-line
        # PHASE A gets an extended contact-loss-exit threshold because
        # the traverse takes longer than the standard with_override grace
        # (~80-110 ticks vs 12).
        self._approach_override_phase:  str   = 'none'

        # PHASE C progress trackers (Layer 2.5/2.6 progress-gated C exit).
        # Updated in _solve_plan at the same site as the [CONTACT-RUN]
        # streak (line ~1043) so the gate at line ~849 next tick reads a
        # consistent end-of-prev-tick triple (_no_ee_box_streak,
        # _phaseC_*_streak, _approach_override_phase). Reset on
        #   (a) free→c3 edge at line 871 — UNCONDITIONAL, phase-agnostic
        #       (load-bearing for C→free→C re-entry where 1710's
        #       phase-change reset does NOT fire because _new_phase ==
        #       _approach_override_phase == 'C_approach'),
        #   (b) intra-c3 phase change at line 1710 (covers B→C, C→B/A
        #       ping-pong inside a single c3 run),
        #   (c) C gate's own fire path (line ~880 added below — bookkeeping
        #       so the trackers reach the next entry already zeroed).
        # Either (a) or (b) leaves trackers fresh on every C entry.
        self._phaseC_stall_streak:      int   = 0
        self._phaseC_active_streak:     int   = 0
        self._phaseC_surf_dist_min:     float = float('inf')
        # surf_dist cached by _run_osc's LTD override at end of each
        # PHASE C tick (None on non-C ticks and on every free tick via
        # the unconditional reset). Read by the PHASE C tracker update
        # in _solve_plan next tick — end-of-prev-tick geometry, matches
        # the gate's end-of-prev-tick read pattern, no cross-half lag.
        self._last_C_surf_dist:         "float | None" = None

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

        # B3c-prime selection audit (env-gated, lazy init at first :848 hit).
        # No env reads here — hot path stays cheap until C3_SEL_AUDIT is set.
        self._sel_audit_state:       object                = _SEL_AUDIT_UNINIT
        self._sel_audit_file                               = None
        self._sel_audit_out_dir:     str                   = ""
        self._sel_audit_pair_id:     str                   = ""
        self._sel_audit_run_id:      str                   = ""

        # Public introspection (mirrors legacy attrs)
        self.last_x_seq:               Optional[np.ndarray] = None
        self.last_winning_sample_idx:  Optional[int]        = None
        self.last_mode:                str                  = self._prev_mode

        # Cache of the last OSC/tracker call kwargs, populated at the end of
        # `compute_control` in each dispatch branch. Consumed by
        # `compute_control_osc_only` to replay the OSC at 1 kHz between
        # planner ticks (mirrors dairlib's LcmDrivenLoop where the OSC
        # subscribes to the planner's last-published trajectory).
        # Shape: (kind, kwargs) where kind ∈ {"c3_traj", "osc_direct",
        # "tracker"}.
        self._last_osc_call: Optional[tuple] = None

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
                                n_strategy: int,
                                yaw_delta:  Optional[float] = None,
                                ) -> list[np.ndarray]:
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
        # 2026-06-25 reconciliation: sim-time _s field → integer ticks.
        # At 100 Hz lifetime_s=0.30 → lifetime=30 (byte-equivalent prior int).
        # At 1 kHz lifetime_s=0.30 → lifetime=300 (300 ms wall time, same).
        _lifetime_s = float(getattr(sp, "sample_buffer_lifetime_s", 0.0))
        lifetime = int(round(_lifetime_s / self._dt_ctrl))

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
                yaw_delta = yaw_delta,
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
                yaw_delta = yaw_delta,
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
        # 2026-06-25 reconciliation: sim-time _s field → integer ticks.
        # At 100 Hz lifetime_s=0.30 → lifetime=30 (byte-equivalent prior int).
        # At 1 kHz lifetime_s=0.30 → lifetime=300 (300 ms wall time, same).
        _lifetime_s = float(getattr(sp, "sample_buffer_lifetime_s", 0.0))
        lifetime = int(round(_lifetime_s / self._dt_ctrl))
        if lifetime <= 0:
            return
        # Sentinel-trigger the refresh path: clearing the buffer is enough,
        # but setting age past lifetime makes the [PERSIST] log line at
        # the next call explicit.
        self._sample_buffer = None
        self._sample_buffer_age = lifetime + 1

    # ----------------------------------------------------------------------
    def _reconcile_surface_target(self,
                                  default_p_ee_des: np.ndarray,
                                  obj_xy: np.ndarray) -> np.ndarray:
        """§7.31 — Override the EE desired-state to the SAMPLED FACE POINT
        (~zero buffer, surface). Formerly gated by REF_RECONCILE_APPROACH;
        now always active.

        Matches the reference's `x_desired = c3_object->GetDesiredState()`
        (sampling_based_c3_controller.cc:500): the OSC tracks the sampled
        face point in BOTH modes. The port stores the sample at
        `sampling_setback` (≈30 mm) OUTSIDE the face; this method projects
        it back along the outward face normal so the target is at the
        surface itself (buffer_distance ≈ 0, matching the reference's
        push_t parameter).

        Returns the original `default_p_ee_des` unchanged when:
          * the flag is OFF, or
          * `_current_repos_target` is None (no active sample), or
          * the sample is at the box centre (degenerate normal).
        """
        sample = self._current_repos_target
        if sample is None or obj_xy is None:
            return default_p_ee_des
        sp = self.params.sampling_params
        setback = float(getattr(sp, "sampling_setback", 0.030))
        delta_xy = np.asarray(sample[:2], dtype=float) - np.asarray(obj_xy, dtype=float)
        norm = float(np.linalg.norm(delta_xy))
        if norm < 1e-6:
            return default_p_ee_des
        n_outward_xy = delta_xy / norm
        surface_xy = np.asarray(sample[:2], dtype=float) - setback * n_outward_xy
        return np.array([surface_xy[0], surface_xy[1], float(sample[2])])

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
        # Force-routing prototype (env-gated, default-inert). When
        # PUSHA_FORCE_ROUTING=u_sol AND the planner is EE-space, return the
        # planner's solved u[0] directly (with a one-shot direction-confirm
        # print so we can verify sign on first c3 entry). PUSHA_FORCE_ROUTING
        # unset or 'off' → falls through to the legacy -g_hat path below
        # (bit-identical to pre-prototype). 'neg_u_sol' returns -u_seq[0]
        # for the opposite sign convention.
        import os as _os
        _fr = _os.environ.get("PUSHA_FORCE_ROUTING", "off").lower()
        if _fr in ("u_sol", "neg_u_sol"):
            _use_ee = bool(getattr(self.base_mpc, "use_ee_space", False))
            _u_seq  = getattr(self.base_mpc, "_last_u_seq", None)
            if _use_ee and _u_seq is not None and hasattr(_u_seq, "shape") \
                    and _u_seq.ndim == 2 and _u_seq.shape[1] == 3:
                u0 = np.asarray(_u_seq[0], dtype=float).reshape(3)
                if _fr == "neg_u_sol":
                    u0 = -u0
                if not getattr(self, "_force_route_logged", False):
                    print(f"[FORCE-ROUTE] active env={_fr} u_seq[0]={u0} "
                          f"|u_seq[0]|={float(np.linalg.norm(u0)):.3f}N",
                          flush=True)
                    self._force_route_logged = True
                return u0

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
        # D2: when ADMM did not converge this tick, the exposed λ is either
        # complementarity-leaked (z_sol view) or ω-leaked (delta view).
        # In either case do NOT amplify the leakage — cap the commanded
        # magnitude at nominal_push_force so we still commit a force but
        # don't push the leakage scale.
        converged = bool(getattr(self.base_mpc, "last_converged", True))
        if has_lam_n:
            # §9 Option A: filter λ_n to EE-BOX pairs only. With N T-GND
            # synth rows in the LCS, the raw sum aggregates EE-manipuland +
            # T-ground λ_n into the EE force intent — WRONG (ground λ_n is a
            # reaction on the box, not on the EE). Use _last_contact_info tag
            # to pick out EE-BOX indices. Same scan as ci_mpc_c3plus.py:328-335.
            # §9-leak gate: this filter changed box-path mag (BOX-GND row
            # excluded) and regressed the 72 % banked closure. Restrict to
            # tshape only; box keeps the pre-§9 raw sum (b23fa82 behavior).
            _shape = getattr(self.base_mpc.formulator, "_object_shape", "box")
            _cinfo_f = getattr(self.base_mpc.formulator,
                               "_last_contact_info", None)
            if _shape == "tshape" and _cinfo_f is not None \
                    and len(_cinfo_f) == lambda_n.size:
                _ee_idxs = [i for i, info in enumerate(_cinfo_f)
                            if isinstance(info, dict)
                            and info.get("tag", "") == "EE-BOX"]
                if _ee_idxs:
                    mag = float(np.sum(np.abs(lambda_n[_ee_idxs])))
                else:
                    mag = 0.0
            else:
                # Box path or fallback: legacy raw sum (b23fa82 behavior).
                mag = float(np.sum(np.abs(lambda_n)))
            mag = max(mag, floor)
            if not converged:
                mag = min(mag, nominal)
        else:
            mag = nominal
        return mag * recoil_dir

    def _evaluate_commit_face_gate(self,
                                   current_q: np.ndarray,
                                   g_hat: np.ndarray):
        """Run the L2 commit-face gate against the active repos target.

        Pure helper. Returns the ``CommitFaceDecision`` from
        ``decide_commit_face_gate(...)`` when the gate is enabled
        (``use_commit_face_gate=True``) and a repos target is active
        (``self._current_repos_target is not None``); returns ``None``
        otherwise. No mutation of ``self`` state — caller decides how
        to act (mutate ``finished_repos`` at the pre-decide site;
        override ``mode`` at the post-decide site, plan 2026-06-10).
        """
        if not getattr(self.params, "use_commit_face_gate", False):
            return None
        if self._current_repos_target is None:
            return None
        thr = float(getattr(self.params,
                            "commit_face_gate_threshold", 0.3))
        box_xy = np.array([
            current_q[self._obj_x_idx],
            current_q[self._obj_y_idx],
        ])
        return decide_commit_face_gate(
            p_repos_target_xy=self._current_repos_target,
            box_xy=box_xy,
            g_hat_xy=g_hat,
            threshold=thr,
        )

    def _velocity_feedforward_from_xseq(self,
                                        plant_ctx,
                                        current_q: np.ndarray,
                                        current_v: np.ndarray) -> Optional[np.ndarray]:
        """Derive bounded EE velocity feedforward from planner x_seq[1].

        v_des = α · clip(J_v(q_at_1) · v_at_1, ±v_max) where (q_at_1,
        v_at_1) are the arm positions/velocities at planner knot 1
        (50 ms ahead in the default plant). The arm slice is taken from
        x_seq[1]; the floating-box slice is left at the current value,
        because the LCS drives box state via the contact solver and a
        contradictory box-pose in plant_ctx would corrupt the Jacobian
        Drake returns.

        Replaces the prior p_ee_des finite-difference path
        (commit fb1fb1c), which aliased on any p_ee_des source change
        (override↔planner seam, LTD phase transitions, c3↔free mode
        transitions) and saturated the OSC at the artifact's clip rather
        than at the planner's true commanded rate. The diagnostic that
        forced this rewrite: at the override→planner seam at step 188
        of seed4-pushing-W-α0.5, |v_des| spiked 60× from 0.015 m/s to
        0.89 m/s on a single tick and pinned at ±0.75 m/s (the post-α
        v_max clip) for the whole 6-tick contact burst, with a
        consistently-negative y-component that wasn't the planner's
        intent — it was the difference between the override's
        last-tick p_ee_des and the planner's first-tick x_seq[1] EE
        position, two heterogeneous setpoint sources.

        Returns None when:
          * use_velocity_feedforward is False (default, opt-in flag)
          * planner has no last_x_seq (cold start, or free mode where
            the IK tracker — not the C3 solver — drives p_ee_des)
        """
        # §7.32 — FAITHFUL-DESIRED-STATE: the planner's predicted velocity
        # is fed undamped (alpha = 1.0) to match the reference's
        # `ydot_des = traj.EvalDerivative(t, 1)` (osc_tracking_data.cc:87-111).
        # Honor the documented opt-in flag: when use_velocity_feedforward is
        # False (default), return None so callers pass None to the OSC and
        # v_err falls back to -v_ee_now. Diagnostic [VFF] reads the same flag
        # (line 3037-3040) — the missing gate here caused a docstring/diag/
        # behavior three-way disagreement where the flag was False, the diag
        # printed alpha=0.000, but the OSC actually received alpha·v_clipped.
        if not bool(getattr(self.params, "use_velocity_feedforward", False)):
            return None
        x_seq = getattr(self.base_mpc, "last_x_seq", None)
        if x_seq is None or x_seq.shape[0] < 2:
            return None
        # EE-space planner: v_ee is already a state slot in x_seq, read it
        # directly — no J · v computation. (Slice indices verified bit-equal
        # by scripts/verify_slice_indices.py.)
        if bool(getattr(self.base_mpc, "use_ee_space", False)):
            v_ee_raw = x_seq[1][16:19].copy()
            v_max = float(self.params.velocity_feedforward_v_max)
            # Undamped (alpha = 1.0) to match the reference's raw
            # EvalDerivative; v_max clip stays as a defensive bound
            # against numerical garbage (planner divergence, NaN, etc.).
            v_clipped = np.clip(v_ee_raw, -v_max, v_max)
            return v_clipped
        # R^7 path (legacy): finite-difference + J · v on planner knot 1.
        n_q = self.base_mpc.formulator.n_q
        q_at_1 = current_q.copy()
        v_at_1 = current_v.copy()
        q_at_1[:self.n_u] = x_seq[1][:self.n_u]
        v_at_1[:self.n_u] = x_seq[1][n_q : n_q + self.n_u]
        # Compute EE Cartesian velocity at the planner's intended (q,v)
        # for knot 1. Restoring plant_ctx is defensive: the c3 branch
        # re-sets it immediately for the executor call, but other code
        # paths in _run_osc may have read from plant_ctx between the
        # call site and the executor, and we don't want a stale knot-1
        # state to leak.
        self.plant.SetPositions(plant_ctx, q_at_1)
        self.plant.SetVelocities(plant_ctx, v_at_1)
        J_v = ee_jacobian_translational(self.plant, plant_ctx, self.ee_frame)
        v_ee_raw = J_v @ v_at_1
        self.plant.SetPositions(plant_ctx, current_q)
        self.plant.SetVelocities(plant_ctx, current_v)
        v_max = float(self.params.velocity_feedforward_v_max)
        alpha = float(self.params.velocity_feedforward_alpha)
        v_clipped = np.clip(v_ee_raw, -v_max, v_max)
        return alpha * v_clipped

    # §7.34 — FAITHFUL-DESIRED-STATE FEEDFORWARD-ACCEL
    # ------------------------------------------------------------------
    # Reference OSC PD law: yddot_command = yddot_des + Kp·error_y + Kd·error_ydot
    #                      (osc_tracking_data.cc:113-116)
    # Port OSC law (pre-§7.34): a_des = Kp_cart·p_err + Kd_cart·v_err
    #                      (qp_builder.py:140 — PD only, NO feedforward accel)
    # The §7.34 build adds the missing yddot_des leg so the port matches the
    # reference's PD + feedforward structure.
    #
    # EE-space x has NO acceleration state slot (verified
    # lcs_formulator.py:1196-1203 — x = [box_q(7), p_ee(3), box_v(6), v_ee(3)],
    # N_X_NEW = 19). a_ff is computed by SECOND-difference of the v_ee state
    # slot — (x_seq[2][16:19] − x_seq[1][16:19]) / dt_planner.
    #
    # dt_planner is base_mpc.dt (50 ms canonical) — NOT dt_ctrl. Same stride-
    # bug guard as the velocity helper.
    #
    # SOURCE/NOISE GATE (§7.34 STEP 0b): the planner is non-converged (25/25
    # per-solve, §7.33 finding), and second-differencing amplifies noise.
    # Probe on §7.32 live log (scripts/_stage_c_feedforward_noise_probe.py)
    # found median |a_ff| = 0 m/s², p90 = 16 m/s², max = 22 m/s², sign-flip
    # rate 11% — bounded enough to feed with a defensive a_max clip. Pass.
    #
    # Defensive a_max clip = 50 m/s² (2× the observed max, covers 100% of
    # the §7.32 probe sample). Bypasses garbage from planner divergence /
    # NaN. Matches the v_max=1.5 m/s defensive bound on the velocity helper.
    def _acceleration_feedforward_from_xseq(self) -> Optional[np.ndarray]:
        """Derive defensive-clipped EE acceleration feedforward from planner
        x_seq second-difference. Returns None when:
          * REF_RECONCILE_FEEDFORWARD_ACCEL is OFF (default — §7.35 sub-gate;
            the §7.34 OVER-DRIVES verdict made the feedforward source-
            conditional, so it stays OFF unless explicitly opted-in; the
            §7.33 working state (pos + vel only) is the default)
          * planner has no last_x_seq, or x_seq has fewer than 3 knots
          * not running --ee-space (the R^7 path has no analytic accel
            source; finite-differencing q-space velocity is even noisier
            and the c3-mode over-drive failure was on the EE-space path)
        """
        if not getattr(self, "_reconcile_feedforward_accel", False):
            return None
        x_seq = getattr(self.base_mpc, "last_x_seq", None)
        if x_seq is None or x_seq.shape[0] < 3:
            return None
        if not bool(getattr(self.base_mpc, "use_ee_space", False)):
            return None
        dt_planner = float(getattr(self.base_mpc, "dt", 0.05))
        if not (dt_planner > 0.0 and np.isfinite(dt_planner)):
            return None
        v_at_1 = x_seq[1][16:19]
        v_at_2 = x_seq[2][16:19]
        a_raw = (v_at_2 - v_at_1) / dt_planner
        if not np.all(np.isfinite(a_raw)):
            return None
        a_max = float(getattr(self.params,
                              "acceleration_feedforward_a_max", 50.0))
        a_clipped = np.clip(a_raw, -a_max, a_max)
        return a_clipped

    def _build_samples(self,
                       ee_pos_now:  np.ndarray,
                       obj_xy:      np.ndarray,
                       g_hat:       np.ndarray,
                       prev_mode:   str,
                       obj_quat:    Optional[np.ndarray] = None,
                       yaw_delta:   Optional[float]      = None,
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
            obj_quat=obj_quat, yaw_delta=yaw_delta)
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
        """T-architecture Stage 2b: gate the planner solve on dt_mpc
        boundaries, run the OSC every tick at dt_osc, and index the
        planner's λ-horizon by elapsed time since the last solve. At
        default rates (dt_osc == dt_mpc == 0.01s) N_plan=1, the planner
        fires every tick with elapsed=0, and the system behaves like
        pre-Stage-2b within the sim's noise floor."""
        self._step += 1
        t_step_start = time.perf_counter()

        # Gating: how many OSC ticks per planner solve?
        N_plan = max(1, int(round(self._dt_mpc / self._dt_osc)))
        should_solve = (self._last_plan_tick < 0
                        or (self._step - self._last_plan_tick) >= N_plan)

        if should_solve:
            plan_ctx = self._solve_plan(current_q, current_v, plant_ctx,
                                        target_xy, target_yaw)
            self._last_plan_tick = self._step
            self._last_plan_ctx  = plan_ctx
            elapsed = 0.0
        else:
            plan_ctx = self._last_plan_ctx
            elapsed  = (self._step - self._last_plan_tick) * self._dt_osc

        # Stage 2b integrity invariant: when the solver has populated the
        # λ-horizon, knot 0 of the horizon must equal the first-knot view
        # that pre-Stage-2b code used. A mismatch indicates a bug in the
        # Stage 1 mirror (offset / wrong solve cached / stale state). Skip
        # when either is empty (no contacts admitted) or None (cold start).
        _hn = getattr(self.base_mpc, "last_lambda_n_horizon", None)
        _fn = getattr(self.base_mpc, "last_lambda_n_first",   None)
        if (_hn is not None and _fn is not None
                and getattr(_hn, "size", 0) > 0 and getattr(_fn, "size", 0) > 0):
            assert np.allclose(_hn[0], _fn, atol=1e-9), (
                f"[Stage2b integrity] last_lambda_n_horizon[0] {_hn[0]} "
                f"!= last_lambda_n_first {_fn}"
            )

        # Inject this tick's wall-clock start so _run_osc measures the
        # OSC tick (not the cached planner-tick) for self._step_times_ms.
        plan_ctx = {**plan_ctx, "t_step_start": t_step_start}

        # B0-diagnostic: per-step planner prediction probe. Prints x_seq[1]
        # box xyz + box translational velocity so we can measure whether
        # ground-pair admission (commit e96ccdd) eliminated the prior
        # T3 "1600× predicted-vs-actual" mismatch. Compact one-line form;
        # consumed by the B0 analyzer in /tmp.
        try:
            _xseq = getattr(self.base_mpc, "last_x_seq", None)
            if _xseq is not None and _xseq.shape[0] >= 2:
                _ps = self.obj_body.floating_positions_start()
                _vs = self.obj_body.floating_velocities_start_in_v()
                _nq = self.base_mpc.formulator.n_q
                _ox, _oy, _oz = _ps + 4, _ps + 5, _ps + 6
                # In the [q;v] state layout, box translational v indices in v
                # are floating_velocities_start_in_v + [3,4,5].
                _vxr, _vyr, _vzr = _nq + _vs + 3, _nq + _vs + 4, _nq + _vs + 5
                x1 = _xseq[1]
                print(
                    f"[X-SEQ-PROBE] step={self._step} "
                    f"x0_box=({current_q[_ox]:+.5f},{current_q[_oy]:+.5f},{current_q[_oz]:+.5f}) "
                    f"x1_box=({x1[_ox]:+.5f},{x1[_oy]:+.5f},{x1[_oz]:+.5f}) "
                    f"x1_box_v=({x1[_vxr]:+.5f},{x1[_vyr]:+.5f},{x1[_vzr]:+.5f}) "
                    f"dt={self.base_mpc.dt:.3f}",
                    flush=True,
                )
        except Exception:  # noqa: BLE001 — probe must not affect control
            pass

        return self._run_osc(current_q, current_v, plant_ctx,
                             plan_ctx, elapsed=elapsed)

    # ------------------------------------------------------------------
    # B3c-prime selection audit (env-gated)
    # ------------------------------------------------------------------

    def _sel_audit_lazy_init(self):
        """One-shot env read on the first :848 hit. Returns (lo, hi) when
        enabled, None when disabled. Reads C3_SEL_AUDIT (dir; unset → no-op),
        C3_SEL_LO (default 150), C3_SEL_HI (default 170), and identity tags
        C3_SEL_PAIR_ID / C3_SEL_RUN_ID. File handle is opened lazily on the
        first in-window emit, not here."""
        out_dir = os.environ.get("C3_SEL_AUDIT", "")
        if not out_dir:
            return None
        try:
            lo = int(os.environ.get("C3_SEL_LO", "150"))
            hi = int(os.environ.get("C3_SEL_HI", "170"))
        except ValueError:
            return None
        self._sel_audit_out_dir = out_dir
        self._sel_audit_pair_id = os.environ.get("C3_SEL_PAIR_ID", "unknown")
        self._sel_audit_run_id  = os.environ.get(
            "C3_SEL_RUN_ID", f"pid_{os.getpid()}"
        )
        return (lo, hi)

    def _sel_audit_emit(self, c_samples, results, labels, k_star, ee_pos_now):
        """Append one JSONL row for this step. Opens the per-run file on
        first call. Records all-k cost components, sample positions, the
        held_idx (None when no prev_repos), ee_pos_now, and PCG64 state int
        (parity fingerprint). Cost: ~one JSON serialize + file write + flush.
        Only fires inside [C3_SEL_LO, C3_SEL_HI]."""
        if self._sel_audit_file is None:
            os.makedirs(self._sel_audit_out_dir, exist_ok=True)
            fname = f"sel_{self._sel_audit_run_id}.jsonl"
            self._sel_audit_file = open(
                os.path.join(self._sel_audit_out_dir, fname),
                "w",
            )

        # Guard: at-most-one prev_repos in labels by construction (built at
        # wrapper.py:580-582). Violations are genuine bugs, not skip cases —
        # the all-False case is legitimate (cold start / post-arrival buffer
        # refresh) and is encoded as held_idx=None.
        n_prev = sum(1 for lbl in labels if lbl == "prev_repos")
        assert n_prev <= 1, (
            f"[SEL-AUDIT] step={self._step} prev_repos count={n_prev} "
            f"(>1 violates wrapper.py:580-582 invariant)"
        )
        held_idx = labels.index("prev_repos") if "prev_repos" in labels else None

        # PCG64 state int (parity fingerprint, NOT a draw counter).
        try:
            rng_state = int(
                self._rng.bit_generator.state["state"]["state"]
            )
        except Exception:
            rng_state = -1

        per_k = []
        for k in range(len(labels)):
            r = results[k]
            lbl = labels[k]
            per_k.append({
                "k":              k,
                "label":          lbl,
                "pos":            [float(r.sample_pos[0]),
                                   float(r.sample_pos[1]),
                                   float(r.sample_pos[2])],
                "c_sample":       float(c_samples[k]),
                "c_C3_raw":       float(r.c_C3_raw),
                "align_bonus":    float(r.align_bonus),
                "rot_bonus":      float(r.rot_bonus),
                "travel_penalty": float(r.travel_penalty),
                "is_current":     (lbl == "current"),
                "is_prev_repos":  (lbl == "prev_repos"),
                "feasible":       bool(r.feasible),
            })

        row = {
            "pair_id":    self._sel_audit_pair_id,
            "run_id":     self._sel_audit_run_id,
            "step":       int(self._step),
            "k_star":     int(k_star),
            "held_idx":   held_idx,
            "ee_pos_now": [float(ee_pos_now[0]),
                           float(ee_pos_now[1]),
                           float(ee_pos_now[2])],
            "rng_state":  rng_state,
            "samples":    per_k,
        }
        self._sel_audit_file.write(json.dumps(row) + "\n")
        self._sel_audit_file.flush()

    def _solve_plan(self,
                    current_q:  np.ndarray,
                    current_v:  np.ndarray,
                    plant_ctx,
                    target_xy:  np.ndarray,
                    target_yaw: float = 0.0) -> dict:
        """Planning half of the original compute_control: sample evaluation,
        mode dispatch (c3 vs free), and the planner-side branch (c3
        invokes base_mpc.compute_control to mutate self.base_mpc.last_*;
        free runs the IK tracker for target selection). Mutates many
        self.* attributes (last_x_seq, _current_repos_*, progress, buffer,
        _no_ee_box_streak, etc.). Returns a plan_ctx dict carrying the
        locals that _run_osc needs (mode, ee_pos_now, g_hat_3d, free_diag,
        sample results, etc.). Stage 2b note: self._step increment and
        t_step_start are now owned by compute_control so they fire every
        OSC tick, not only on planner-solve ticks."""
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

        # D.3 (2026-07-13) — shortest-angle yaw delta from current to
        # target. Feeds the yaw-torque face-selection bias in the T
        # sampler. For the box (target_yaw = 0 and box stays flat),
        # yaw_delta stays near 0 and the sampler skips the bias entirely
        # (see sampling.py:_face_normal_projection D.3 block).
        _yaw_now_samp = 2.0 * float(np.arctan2(obj_quat[3], obj_quat[0]))
        _dy_samp = float(target_yaw) - _yaw_now_samp
        yaw_delta_samp = float(
            np.arctan2(np.sin(_dy_samp), np.cos(_dy_samp)))

        # 2. Build sample list (k=0 = current EE always first)
        samples, labels = self._build_samples(
            ee_pos_now, obj_xy, g_hat, self._prev_mode,
            obj_quat=obj_quat, yaw_delta=yaw_delta_samp)

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

        # === B3c-prime selection audit (env-gated, hot-path-cheap) ===========
        # Emit at the SOLE site where the full c_samples vector exists. Outside
        # the [C3_SEL_LO, C3_SEL_HI] window or when C3_SEL_AUDIT is unset, the
        # cost is two attribute reads + an `is`/`None` compare + one int-range
        # check. No env reads, no comprehensions, no f-strings on the cold
        # path. Read-only w.r.t. control flow: does NOT touch k_star, costs,
        # the buffer, or any branch downstream.
        _sa = self._sel_audit_state
        if _sa is _SEL_AUDIT_UNINIT:
            _sa = self._sel_audit_lazy_init()
            self._sel_audit_state = _sa
        if _sa is not None and _sa[0] <= self._step <= _sa[1]:
            self._sel_audit_emit(
                c_samples, results, labels, k_star, ee_pos_now,
            )
        # ====================================================================

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

        # D.2 fix (2026-07-13) — real yaw error for tasks with a rotation
        # goal (T-push). Previously hardcoded rot_error=0, which is
        # correct for the box (task has no yaw goal so goal_yaw defaults
        # to 0) but silently discards the T's dominant progress signal
        # (T cost is yaw-dominated per w_yaw=800 in the T config, while
        # c3 exits are governed by ProgressTracker's rot/pos improve
        # counters).
        #
        # Computed as the shortest angular distance from the object's
        # current z-yaw to target_yaw. For the box, target_yaw defaults
        # to 0 and the box stays flat, so rot_error stays at ~0 and the
        # ProgressTracker's rot state is functionally unchanged.
        # Additionally, box configs use track_c3_progress_via=kPosOnly
        # which does NOT read rot_error at all (progress.py:195-198),
        # so downstream met_progress is bit-identical for box. Only the
        # T config that opts into kPosOrRotCost sees behavior change.
        _yaw_now = 2.0 * float(np.arctan2(obj_quat[3], obj_quat[0]))
        _dy = _yaw_now - float(target_yaw)
        rot_error_now = float(np.abs(np.arctan2(np.sin(_dy), np.cos(_dy))))

        self.progress.update(StepMetrics(
            c3_cost     = c_curr,
            config_cost = config_cost_now,
            pos_error   = goal_dist,
            rot_error   = rot_error_now,
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
        # Stage A — when PWL trajectory is active, derive finished_repos
        # from the trajectory's is_finished (BOTH t ≥ t_end AND EE within
        # ``tol`` of p_target) instead of the legacy tracker's diag.
        #
        # BUG 1 wiring fix (2026-07-13): tolerance widened from 5 mm to
        # 20 mm to match the IK tracker's euclidean predicate at
        # reposition_ik.py:1465 (‖p_target − ee_now‖ ≤ 20 mm). The 5 mm
        # here was preventing Path 1 (kToC3ReachedReposTarget) from ever
        # firing for T: the T-shape setback is inside the vertical bar's
        # sphere-swept envelope (BUG 2), so physical descent stalls with
        # ‖p_target − ee_now‖ ≈ 20 mm rather than 5 mm — the trajectory
        # is effectively arrived but the physical residual holds above
        # 5 mm indefinitely. 20 mm matches the reference-conformant IK
        # tracker predicate and closes the mode-switch wiring gap.
        # Box path unaffected: use_reposition_pwl_trajectory defaults to
        # False (only T config sets it True), so line 1224's tracker.finished
        # remains the box's authoritative finished signal.
        if self._use_pwl_traj and self._pwl_traj is not None:
            # Align to real sim time at compute_control entry (see the fix
            # applied to _sim_t / _sim_t_c3). self._step was incremented at
            # line 850, so the real sim clock is (self._step - 1) * dt_ctrl.
            # Off-by-one here reports PWL.is_finished ONE planner tick early,
            # which can prematurely fire kToC3ReachedReposTarget.
            _sim_t_fin = float(self._step - 1) * float(self._dt_ctrl)
            try:
                _ee_now_fin = self.plant.CalcPointsPositions(
                    plant_ctx, self.ee_frame, np.zeros(3),
                    self.plant.world_frame(),
                ).flatten()
            except Exception:
                _ee_now_fin = np.zeros(3)
            finished_repos = self._pwl_traj.is_finished(
                _sim_t_fin, _ee_now_fin, tol=0.020)

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

            # Stage 2 L1 gate: goal-aligned contact-normal requirement.
            # nhat_onto_box points INTO the box (lcs_formulator.py:1090);
            # the EE-on-box force is along nhat_onto_box. Goal-aligned
            # contact ≡ nhat_onto_box · g_hat > entry_align_threshold (the
            # push direction matches the goal direction). Catches cardinal-
            # on-wrong-face (alignment ≈ 0) and off-cardinal/edge contact
            # (alignment small) with one cosine check.
            # 0.0 → identity (regression-safe default).
            _align_thr = float(getattr(self.params,
                                       "entry_align_threshold", 0.0))
            if (not _block) and _align_thr > 0.0:
                _ci = getattr(self.base_mpc.formulator,
                              "_last_contact_info", None)
                _nhat_xy = None
                if _ci:
                    for _info in _ci:
                        if (isinstance(_info, dict)
                                and _info.get("tag") == "EE-BOX"):
                            _n = _info.get("nhat_onto_box")
                            if _n is not None and len(_n) >= 2:
                                _nhat_xy = (float(_n[0]), float(_n[1]))
                                break
                if _nhat_xy is not None:
                    _align = (_nhat_xy[0] * g_hat[0]
                              + _nhat_xy[1] * g_hat[1])
                    if _align <= _align_thr:
                        finished_repos = False
                        if self.log_diag:
                            print(f"[GATE-ALIGN] step={self._step} "
                                  f"refused: align={_align:+.3f} "
                                  f"<= thr={_align_thr:.2f} "
                                  f"nhat_xy=({_nhat_xy[0]:+.3f},"
                                  f"{_nhat_xy[1]:+.3f}) "
                                  f"g_hat=({g_hat[0]:+.3f},"
                                  f"{g_hat[1]:+.3f})", flush=True)

            # Stage 2 L2 gate: commit-face requirement on the reposition
            # target's outward face. Keys on self._current_repos_target
            # (populated by definition when finished_repos==True) — works
            # 80 mm pre-contact where L1's contact_info is empty.
            #
            # Runs alongside L1 (not in series): if L1 already set
            # finished_repos=False, L2 still evaluates and logs its
            # decision so SC-L2-L1 redundancy is measurable from logs.
            #
            # Sign convention is inverted vs L1: L2's n_face_out points
            # OUTWARD (box→target), L1's nhat_onto_box points INTO box;
            # both use <= but the meanings are mirrored — see
            # commit_face_gate.py module docstring.
            if not _block:
                _l2_dec = self._evaluate_commit_face_gate(current_q, g_hat)
                if _l2_dec is not None and not _l2_dec.commit:
                    finished_repos = False
                    if self.log_diag:
                        print(f"[GATE-COMMIT-FACE] step={self._step} "
                              f"refused: face_align={_l2_dec.face_align:+.3f} "
                              f"tag={_l2_dec.severity_tag} "
                              f"n_face_out=({_l2_dec.n_face_out_xy[0]:+.3f},"
                              f"{_l2_dec.n_face_out_xy[1]:+.3f}) "
                              f"g_hat=({g_hat[0]:+.3f},"
                              f"{g_hat[1]:+.3f})", flush=True)

        met = self.progress.met_progress(near_goal=near_goal)

        # Absolute-regression early-exit. Catches runaway trajectories where
        # the box drifts AWAY from its best position by more than
        # ProgressParams.pos_regression_threshold (metres), regardless of the
        # no-improvement tick counter. Root-independent — fires on the
        # runaway signature directly, not on the wait-expiry symptom.
        # Plan: docs/superpowers/plans/2026-06-06-position-progress-fix-combined.md
        _pos_reg = self.progress.pos_regression()
        _pos_reg_thr = float(self.params.progress_params.pos_regression_threshold)
        if _pos_reg_thr > 0.0 and _pos_reg > _pos_reg_thr and met:
            met = False
            if self.log_diag:
                print(f"[POS-REGRESSION] step={self._step} "
                      f"pos_regression={_pos_reg*1000:.1f}mm "
                      f"> threshold={_pos_reg_thr*1000:.1f}mm "
                      f"— forcing met_progress=False", flush=True)

        # T1a — EE_z altitude gate (reference sampling_based_c3_controller.cc
        # :1290-1293). Block c3 entry from free while EE_z is above the
        # sampling-height ceiling (sampling_z + c3_min_clearance). Reference
        # gates the auto-entry `else if` branch as an AND-condition, covering
        # BOTH kToC3ReachedReposTarget AND kToC3Cost. Ported in-decision by
        # passing ``ee_z_gate_pass`` into decide_mode (mode_switch.py:95) —
        # decide_mode simply never returns c3 when the gate is closed, so
        # no post-decide-revert side-effect risk (no mode-switch/hysteresis
        # state latches from a c3 decision that then gets undone).
        # Complementary to (not a replacement for) the per-tick ADMIT-GUARD
        # (LCS-admission latch) and ALT-GATE (descent permission) — those
        # latch each tick; this is one-shot at mode-switch.
        #
        # wall_offset (reference lines 1246-1264, +0.01 m near workspace
        # walls) deferred as 0.0 for T1a — T-push at seed 0 doesn't push
        # into walls; workspace-margin logic lands when the T pushes near
        # a wall (T4 multi-seed).
        _ee_z_gate_pass = True
        # T1a-fix (2026-07-10) — retrofit tshape gate at the use site. The
        # original T1a landing gated only on `params.ee_z_close` (default
        # True), so the gate silently activated for the box path too and
        # regressed box closure from 71.6 % → 59.6 % (measured 2026-07-10:
        # pure-HEAD box W seed 0 vs P2 tripwire). Reference behavior for
        # T remains identical (T yaml keeps default ee_z_close=True). Box
        # yaml is byte-identical to the pre-T1a behavior — the gate is
        # skipped when object_shape != "tshape". Same §9-leak discipline as
        # T1b/T1c (both correctly gated at their use sites).
        _obj_shape = getattr(
            getattr(self.base_mpc, "formulator", None), "_object_shape", "box")
        if (self._prev_mode == "free"
                and getattr(self.params, "ee_z_close", True)
                and _obj_shape == "tshape"):
            _sampling_z = float(self.params.sampling_params.sampling_height)
            _c3_min_clearance = float(getattr(
                self.params, "c3_min_clearance", 0.01))
            _wall_offset = 0.0
            _z_ceiling = _sampling_z + _c3_min_clearance + _wall_offset
            _ee_z_now = float(ee_pos_now[2])
            _ee_z_gate_pass = _ee_z_now <= _z_ceiling
            if (not _ee_z_gate_pass) and self.log_diag:
                # Emitted whenever the altitude gate is ACTIVE (prev_mode==free
                # and ee_z above ceiling). Whether the gate actually SUPPRESSED
                # a c3 entry that would otherwise have fired is decide_mode's
                # concern — from the log it's verifiable that any [GS] mode=c3
                # switch=kToC3* line coincides with [EEZ-GATE] absent for that
                # step (dispatch-discipline effect check).
                print(f"[EEZ-GATE] step={self._step} "
                      f"ee_z={_ee_z_now*1000:.1f}mm "
                      f"> ceiling={_z_ceiling*1000:.1f}mm "
                      f"(sampling_z={_sampling_z*1000:.1f}mm + "
                      f"clearance={_c3_min_clearance*1000:.1f}mm) "
                      f"gate=BLOCK", flush=True)

        mode, reason = decide_mode(
            prev_mode          = self._prev_mode,
            c3_cost            = c_curr,
            best_other_cost    = best_other_cost,
            current_repos_cost = self._current_repos_cost,
            met_progress       = met,
            near_goal          = near_goal,
            finished_repos     = finished_repos,
            params             = self.params.progress_params,
            ee_z_gate_pass     = _ee_z_gate_pass,
        )

        # §7.56 Stage 1 — [COST-DECOMP] diagnostic. Gated by
        # PUSHA_COST_DECOMP_LOG=1 (default-OFF) so flag=0 stays byte-identical
        # to the prior behaviour, including stdout.
        import os as _os_cd
        if _os_cd.environ.get("PUSHA_COST_DECOMP_LOG", "0") == "1":
            _r0 = results[0]
            if getattr(_r0, "c_C3_raw_full", float("inf")) != float("inf"):
                _bostr = (f"{best_other_cost:.2f}"
                          if best_other_cost != float("inf") else "-")
                print(f"[COST-DECOMP] step={self._step} mode={mode} "
                      f"reason={reason.name} "
                      f"c_curr_full={_r0.c_C3_raw_full:.2f} "
                      f"c_curr_objonly={_r0.c_C3_raw_objonly:.2f} "
                      f"c_curr_used={_r0.c_C3_raw:.2f} "
                      f"best_other={_bostr}", flush=True)

        # 6a-pre0. Wrong-face re-engagement guard (post-decide L2 override).
        # The pre-decide L2 site above mutates finished_repos, which only
        # short-circuits kToC3ReachedReposTarget (mode_switch.py:123-124).
        # The cost-gap path (mode_switch.py:132-135 -> kToC3Cost) reaches
        # c3 without consulting finished_repos. Re-evaluate L2 here on
        # every free->c3 transition so the gate covers all entry paths.
        # Plan: docs/superpowers/plans/2026-06-10-wrong-face-reengage-guard.md
        #
        # Empirical threshold +0.3 (params.py:624): refuses confirmed
        # runaway entries (face_align ≈ +0.97-class, seed-0 step 434 and
        # seed-4 step 519, both producing +60-91 mm north-drift) while
        # admitting the productive seed-4 step-315 session (face_align
        # = -0.497, 114 ticks, +136 mm westward push) that the prior
        # -0.7 threshold would have false-blocked.
        if (self._prev_mode == "free"
                and mode == "c3"
                and getattr(self.params, "use_commit_face_gate", False)):
            _post_dec = self._evaluate_commit_face_gate(current_q, g_hat)
            if _post_dec is not None and not _post_dec.commit:
                _orig_reason_name = reason.name
                mode = "free"
                reason = SwitchReason.kStayInRepos
                if self.log_diag:
                    print(f"[GATE-COMMIT-FACE-POST] step={self._step} "
                          f"refused: face_align={_post_dec.face_align:+.3f} "
                          f"tag={_post_dec.severity_tag} "
                          f"n_face_out=({_post_dec.n_face_out_xy[0]:+.3f},"
                          f"{_post_dec.n_face_out_xy[1]:+.3f}) "
                          f"g_hat=({g_hat[0]:+.3f},"
                          f"{g_hat[1]:+.3f}) "
                          f"orig_reason={_orig_reason_name} "
                          f"-> override mode=free reason=kStayInRepos",
                          flush=True)

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
        # Threshold is conditioned on whether the approach-closing
        # override fired on the previous tick. When the override is
        # actively closing a sub-LCS-threshold gap, give it more time;
        # snap back to the strict default the instant the override stops
        # firing (e.g. LCS admitted a pair, or surf_dist ≤ threshold).
        # The `with_override` value is the hard cap on grace.
        # PHASE A of the LTD override holds EE.z at z_safe (above box top)
        # under active z-Kp tracking, so the earlier objection to a longer
        # grace timer ("EE has more time to fall onto the top") does not
        # apply: there's nowhere for the EE to fall in PHASE A. Use the
        # extended threshold there. Also acts as the stuck-watchdog: if
        # PHASE A can't form contact in `contact_loss_threshold_phaseA_ltd`
        # ticks, the system gives up.
        # PHASE B (descend beside the face): probe at audit_output/ltd_diag
        # shows ee.x holds 4–16 mm east of the face plane through the
        # entire descent, drifting outward toward W_side. A free-mode
        # interlude would fall east of the box, not onto its top, so the
        # fall-onto-top objection does not apply here either. Extend with
        # the same watchdog pattern as PHASE A.
        # 2026-06-25 reconciliation: thresholds live as sim-time _s fields
        # on params; convert to integer ticks via _dt_ctrl at read time so
        # the existing integer counter (`_no_ee_box_streak`) comparison
        # stays unchanged. At 100 Hz the conversion is byte-equivalent to
        # the prior int values; at 1 kHz the threshold becomes 10× larger
        # in ticks, preserving the wall-time interval.
        def _ticks(s_val: float) -> int:
            return int(round(float(s_val) / self._dt_ctrl))
        if self._approach_override_phase == 'A_lift_trav':
            disengage_threshold = _ticks(self.params.contact_loss_threshold_phaseA_ltd_s)
        elif self._approach_override_phase == 'B_descend':
            disengage_threshold = _ticks(self.params.contact_loss_threshold_phaseB_ltd_s)
        elif self._approach_override_phase == 'C_approach':
            disengage_threshold = _ticks(self.params.phaseC_hard_cap_s)
        elif self._approach_override_firing:
            disengage_threshold = _ticks(self.params.contact_loss_threshold_with_override_s)
        else:
            disengage_threshold = _ticks(self.params.contact_loss_threshold_default_s)
        if (self._prev_mode == "c3"
                and mode == "c3"
                and self._no_ee_box_streak >= disengage_threshold):
            if self._disable_contact_loss_gate:
                # §7.55 — gate skipped; c3 held by cost+progress only
                # (reference-faithful). One-shot log to confirm behavior.
                if self.log_diag and not getattr(
                        self, "_755_skip_logged", False):
                    print(f"[§7.55] PUSHA_DISABLE_CONTACT_LOSS_GATE=1 — "
                          f"CONTACT-LOSS-EXIT skipped at step={self._step} "
                          f"(streak={self._no_ee_box_streak} "
                          f"threshold={disengage_threshold} "
                          f"phase={self._approach_override_phase}); "
                          f"c3 held by cost+progress only",
                          flush=True)
                    self._755_skip_logged = True
            else:
                mode = "free"
                reason = SwitchReason.kToReposUnproductive
                if self.log_diag:
                    print(f"[CONTACT-LOSS-EXIT] step={self._step} "
                          f"no EE-BOX for {self._no_ee_box_streak} "
                          f"steps threshold={disengage_threshold} "
                          f"override_phase={self._approach_override_phase} "
                          f"-> exit to repos", flush=True)
                self._no_ee_box_streak = 0
        if self._prev_mode == "free":
            # Fresh start when re-entering c3 from free. Phase-agnostic
            # by design: zeroes BOTH the LCS contact-loss counter and
            # the PHASE C tracker triple, regardless of what
            # _approach_override_phase carried across the free
            # interlude (it's stale for the whole interlude — the
            # override block in _run_osc only runs when mode=='c3').
            # Load-bearing for C→free→C re-entry: 1710's phase-change
            # reset does NOT fire when _new_phase ('C_approach') ==
            # _approach_override_phase ('C_approach', stale from the
            # tick before the free interlude began). Without this
            # block the C trackers would carry stale active_streak /
            # surf_dist_min into the second C run and the C gate could
            # fire on the first tick.
            self._no_ee_box_streak = 0
            self._phaseC_stall_streak = 0
            self._phaseC_active_streak = 0
            self._phaseC_surf_dist_min = float('inf')
            self._last_C_surf_dist = None

        # 6a-bis. PHASE C progress-gated exit. When prev tick was in
        # PHASE C (_approach_override_phase=='C_approach', set at the
        # end of last tick's override block in _run_osc), evaluate
        # whether C is still productively closing surf_dist. Two
        # independent fire conditions:
        #   * stall: _phaseC_stall_streak ≥ phaseC_stall_threshold —
        #     no surf_dist improvement ≥ phaseC_progress_eps for that
        #     many consecutive C ticks. Catches asymptotic non-closure.
        #   * hard_cap: _phaseC_active_streak ≥ phaseC_hard_cap —
        #     absolute time budget. Bounds worst case even when
        #     surf_dist is creeping in but never quite admits.
        # Reads end-of-prev-tick (_phaseC_stall_streak,
        # _phaseC_active_streak, _approach_override_phase) — same
        # temporal pattern as the contact-loss gate above. Update
        # site is the [CONTACT-RUN] block at line ~1043 (consistent
        # within _solve_plan; no cross-half lag).
        if (self._prev_mode == "c3"
                and mode == "c3"
                and self._approach_override_phase == 'C_approach'):
            # 2026-06-25 reconciliation: sim-time _s fields → integer ticks.
            _stall_thr = int(round(self.params.phaseC_stall_threshold_s / self._dt_ctrl))
            _hard_cap  = int(round(self.params.phaseC_hard_cap_s / self._dt_ctrl))
            _stall_fire = (self._phaseC_stall_streak >= _stall_thr)
            _cap_fire = (self._phaseC_active_streak  >= _hard_cap)
            if _stall_fire or _cap_fire:
                mode = "free"
                reason = SwitchReason.kToReposUnproductive
                if self.log_diag:
                    _tag = "PHASEC-STALL" if _stall_fire else "PHASEC-HARDCAP"
                    _smin = (float('nan')
                             if not np.isfinite(self._phaseC_surf_dist_min)
                             else self._phaseC_surf_dist_min)
                    print(f"[{_tag}] step={self._step} "
                          f"stall_streak={self._phaseC_stall_streak} "
                          f"active_streak={self._phaseC_active_streak} "
                          f"stall_thr={_stall_thr} "
                          f"hard_cap={_hard_cap} "
                          f"surf_dist_min={_smin*1000:.2f}mm "
                          f"-> exit to repos", flush=True)
                self._phaseC_stall_streak = 0
                self._phaseC_active_streak = 0
                self._phaseC_surf_dist_min = float('inf')
                self._last_C_surf_dist = None

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

            # Per-contact φ / λ_n breakdown (env-gated: LCS_CONTACT_BREAKDOWN=1).
            # Logs each contact pair admitted into the LCS this c3-mode tick —
            # Drake auto-admits plus any synthesized contacts (Stage 1 12-contact
            # LCS) — with phi, lambda_n, and active/inactive state from the LCP
            # solution.
            #   active   = lambda_n > 1e-6   (force flowing through this contact)
            #   inactive = lambda_n ~= 0     (admitted but phi > 0 → LCP
            #                                 complementarity case: η_n > 0, λ_n = 0)
            # Off by default; one print per contact per c3-mode tick when enabled.
            # Preserved as its own commit so a future revert of the synthesis
            # (lcs_formulator.py) does not wipe this diagnostic.
            if os.environ.get("LCS_CONTACT_BREAKDOWN", "0") == "1" and _ci:
                _lam_n_first = getattr(self.base_mpc, "last_lambda_n_first", None)
                _n_c_log = len(_ci)
                print(f"[LCS-CONTACT-BREAKDOWN] step={self._step} n_c={_n_c_log}",
                      flush=True)
                for _i_log, _info_log in enumerate(_ci):
                    _phi_mm = float(_info_log.get("distance", 0.0)) * 1000.0
                    if (_lam_n_first is not None
                            and hasattr(_lam_n_first, "__len__")
                            and _i_log < len(_lam_n_first)):
                        _lam = float(_lam_n_first[_i_log])
                    else:
                        _lam = 0.0
                    _state = "active" if abs(_lam) > 1e-6 else "inactive"
                    _tag = _info_log.get("tag", "?")
                    print(f"[LCS-CONTACT-BREAKDOWN]   i={_i_log} "
                          f"tag={_tag:<10s} "
                          f"phi={_phi_mm:+7.2f}mm "
                          f"lam_n={_lam:+9.4f}N "
                          f"state={_state}", flush=True)

            # PHASE C progress tracker update. Co-located with the
            # contact-loss streak update so the gate next tick reads a
            # consistent end-of-prev-tick triple (no cross-half lag).
            # Conditional on _approach_override_phase=='C_approach' —
            # this is the END-OF-PREV-TICK phase (set last tick at
            # line 1711 in _run_osc), so the update only fires when
            # the previous tick was actually in C.
            # surf_dist source: self._last_C_surf_dist cached at end
            # of prev tick by _run_osc's override block (line ~1716
            # below). On the first C tick after entry, prev tick's
            # phase was not C, so this block is skipped; the cache
            # gets populated this tick in _run_osc, and the next C
            # tick begins consuming it.
            if (self._approach_override_phase == 'C_approach'
                    and self._last_C_surf_dist is not None):
                self._phaseC_active_streak += 1
                _prev_surf = float(self._last_C_surf_dist)
                if _prev_surf < (self._phaseC_surf_dist_min
                                 - self.params.phaseC_progress_eps):
                    self._phaseC_surf_dist_min = _prev_surf
                    self._phaseC_stall_streak = 0
                else:
                    self._phaseC_stall_streak += 1
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
            # EE-space planner uses 19-dim state; the R^7-sized algebra in
            # this dump block (_A @ [current_q, current_v]_27) is out-of-spec.
            # Skip the dump under --ee-space; held follow-up to re-implement
            # against [box_q, p_ee, box_v, v_ee]_19.
            _skip_r7_dump = bool(getattr(self.base_mpc, "use_ee_space", False))
            if (not self._did_cost_dump
                    and self.base_mpc.formulator._last_n_c > 0
                    and not _skip_r7_dump):
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
            if (not self._did_counterfactual_dump
                    and self.base_mpc.formulator._last_n_c > 0
                    and not _skip_r7_dump):
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
            if (not self._did_planvsexec_dump
                    and self.base_mpc.formulator._last_n_c > 0
                    and not _skip_r7_dump):
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

                # Stage C landing-storm trace — upstream argmin emit.
                # Default-OFF. Captures whether the per-tick selection is
                # flipping between cached samples (the leading hypothesis
                # for the rebuild storm at 1 kHz).
                import os as _os_lt2
                if (_os_lt2.environ.get("PUSHA_LANDING_TRACE", "0") == "1"
                        and self._step >= int(_os_lt2.environ.get(
                            "PUSHA_LANDING_TRACE_FROM", "1600"))):
                    _label = labels[target_idx]
                    _cs_str = ",".join(f"{float(c):.4f}" for c in c_samples)
                    print(f"[LANDING-SELECT] step={self._step} "
                          f"target_idx={target_idx} label={_label} "
                          f"p_repos=[{p_repos[0]:+.4f},{p_repos[1]:+.4f},{p_repos[2]:+.4f}] "
                          f"labels={labels} c_samples=[{_cs_str}]",
                          flush=True)

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
                # Contact-admit guard (Stage 2 of 2026-06-01 contact-duration fix):
                # signal the IK tracker that LCS has admitted an EE-BOX pair
                # so it can suspend its Phase 1 lift while contact is forming.
                # Debouncing happens inside the tracker (ADMIT_LATCH_TICKS).
                _ee_box_pairs = getattr(self.base_mpc.formulator,
                                        "_last_ee_box_contacts", [])
                _admit_active = bool(_ee_box_pairs)
                u_opt, free_diag = self.tracker.compute_torque(
                    current_q=current_q, current_v=current_v,
                    plant_ctx=plant_ctx, p_target=p_repos,
                    dt_osc=self._dt_osc,
                    admit_active=_admit_active,
                )
                # 1 kHz OSC decoupling: cache tracker call args for replay in
                # sub-tick. p_target is stable across the planner tick.
                self._last_osc_call = ("tracker", dict(
                    p_target=np.asarray(p_repos, dtype=float).reshape(3).copy(),
                    dt_osc=self._dt_osc,
                    admit_active=_admit_active,
                ))
                # Diagnostic: emit one-line [ADMIT-GUARD] per step the latch
                # is decrementing or active so post-fix logs can verify SC1
                # (target_z holds) and SC6 (no chatter at boundary).
                if self.log_diag:
                    # Q2c (2026-06-04): extended with ee_z + gate_cap so the
                    # parser at scripts/parse_admit_guard_gate.py can verify
                    # SC-collision-gone (pass-through at high ee_z) and the
                    # gate's per-tick decision history. ee_z comes from
                    # free_diag (the tracker already computed FK).
                    _ee_z_log = float(free_diag.get("ee_now", [0.0, 0.0, 0.0])[2])
                    _latch = int(getattr(self.tracker, "_admit_latch", 0))
                    _latch_ticks = int(getattr(self.tracker,
                                               "ADMIT_LATCH_TICKS", 0))
                    print(f"[ADMIT-GUARD] step={self._step} "
                          f"admit_active={int(_admit_active)} "
                          f"latch={_latch}/{_latch_ticks} "
                          f"ee_z={_ee_z_log:.3f} "
                          f"gate_cap={int(getattr(self.tracker, '_last_cap_z_safe', False))}",
                          flush=True)
                    # Stage 1 (2026-06-01 wrong-face race-fix): emit the
                    # descent-gate state and a one-shot [TGT-CHANGE] event
                    # when p_target jumped > TARGET_STABLE_TOL this tick.
                    # The change-interval distribution disambiguates Stage-1
                    # deadlock cause (real oscillation vs mistuned constant).
                    _stable_ticks = int(getattr(self.tracker,
                                                "_target_stable_ticks", 0))
                    _stable_req = int(getattr(self.tracker,
                                              "TARGET_STABLE_TICKS", 0))
                    _allow_desc = int(_stable_ticks >= _stable_req
                                      if _stable_req > 0 else 1)
                    print(f"[ALT-GATE] step={self._step} "
                          f"target_stable={_stable_ticks}/{_stable_req} "
                          f"allow_descent={_allow_desc}",
                          flush=True)
                    if bool(getattr(self.tracker,
                                    "_target_changed_this_tick", False)):
                        _intervals = getattr(self.tracker,
                                             "_target_change_intervals", [])
                        _last_int = (_intervals[-1] if _intervals else -1)
                        print(f"[TGT-CHANGE] step={self._step} "
                              f"interval_ticks={_last_int} "
                              f"n_changes={len(_intervals)}",
                              flush=True)
                # Capture trajectory-finished signal for the next loop's
                # mode-switch decision (kToC3ReachedReposTarget).
                self._last_repos_finished = bool(
                    free_diag.get("finished", False))
                # On arrival, force the ring-sample buffer to refresh next
                # loop. Otherwise the now-reached point persists as a
                # strategy sample and the cost gate keeps re-firing
                # kToC3ReachedReposTarget for it.
                if self._last_repos_finished:
                    # Stage C landing-storm trace: mark the call (the
                    # leading-hypothesis suspect for the per-tick rebuild
                    # storm at 1 kHz). Default-OFF.
                    import os as _os_lt0
                    if _os_lt0.environ.get("PUSHA_LANDING_TRACE", "0") == "1":
                        print(f"[LANDING-REFRESH] step={self._step} "
                              f"_refresh_buffer_on_arrival FIRED",
                              flush=True)
                    self._refresh_buffer_on_arrival()
                    self._landing_trace_refresh_fired_at = int(self._step)

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

        # End of _solve_plan. Hand off locals _run_osc needs via plan_ctx.
        return dict(
            mode            = mode,
            reason          = reason,
            ee_pos_now      = ee_pos_now,
            obj_xy          = obj_xy,
            g_hat           = g_hat,
            g_hat_3d        = g_hat_3d,
            goal_dist       = goal_dist,
            free_diag       = free_diag,
            samples         = samples,
            labels          = labels,
            results         = results,
            c_samples       = c_samples,
            k_star          = k_star,
            best_src        = best_src,
            best_other_cost = best_other_cost,
            met             = met,
            finished_repos  = finished_repos,
            # t_step_start is injected by compute_control (per OSC tick,
            # not per planner-solve tick).
        )

    def _run_osc(self,
                 current_q:  np.ndarray,
                 current_v:  np.ndarray,
                 plant_ctx,
                 plan_ctx:   dict,
                 elapsed:    float = 0.0) -> np.ndarray:
        """OSC-executor half of the original compute_control: runs the
        Operational-Space Controller (force-tracking QP), applies the
        Lever-3 approach-closing override, and emits per-step logging +
        bookkeeping. The `elapsed` argument is unused in Stage 2a (the
        OSC still reads self.base_mpc.last_lambda_n_first as before);
        Stage 2b will use it to index into self.base_mpc.last_lambda_n_horizon
        when dt_mpc > dt_osc."""
        # Unpack plan_ctx into the locals the executor body expects.
        mode            = plan_ctx["mode"]
        reason          = plan_ctx["reason"]
        ee_pos_now      = plan_ctx["ee_pos_now"]
        obj_xy          = plan_ctx["obj_xy"]
        g_hat           = plan_ctx["g_hat"]
        g_hat_3d        = plan_ctx["g_hat_3d"]
        goal_dist       = plan_ctx["goal_dist"]
        free_diag       = plan_ctx["free_diag"]
        samples         = plan_ctx["samples"]
        labels          = plan_ctx["labels"]
        results         = plan_ctx["results"]
        c_samples       = plan_ctx["c_samples"]
        k_star          = plan_ctx["k_star"]
        best_src        = plan_ctx["best_src"]
        best_other_cost = plan_ctx["best_other_cost"]
        finished_repos  = plan_ctx.get("finished_repos", False)
        met             = plan_ctx["met"]
        t_step_start    = plan_ctx["t_step_start"]

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
            # EE-space planner: p_ee is already a state slot in x_seq, read
            # it directly — no FK. (Slice indices verified bit-equal to FK
            # at the linearization point by scripts/verify_slice_indices.py;
            # max |x_seq[0][7:10] - p_ee_now| = 1.11e-15.) R^7 planner path
            # below retains the original FK extraction.
            _use_ee_space = bool(getattr(self.base_mpc, "use_ee_space", False))
            if _x_seq is not None and len(_x_seq) > 1:
                if _use_ee_space:
                    _p_ee_des = _x_seq[1][7:10].copy()
                else:
                    _q_full_next = current_q.copy()
                    _q_full_next[:self.n_u] = _x_seq[1][:self.n_u]
                    self.plant.SetPositions(plant_ctx, _q_full_next)
                    _p_ee_des = self.plant.CalcPointsPositions(
                        plant_ctx, self.ee_frame, np.zeros(3), self.world_frame,
                    ).flatten()
                    self.plant.SetPositions(plant_ctx, current_q)
                    self.plant.SetVelocities(plant_ctx, current_v)
                # IK-projected joint-space guidance for tshape c3 mode.
                # Rationale: port's C3+ ADMM is non-converged, so
                # FK(x_seq[1][:7]) is a phantom target (planner's
                # arm-state prediction is physically inconsistent with
                # its own box-motion prediction). Replace with an
                # IK-solved arm state whose FK gives the geometric
                # contact target `box_center − g_hat·(face_offset +
                # pusher_radius)`. `q_arm_ik` is passed to the OSC as
                # `q_nominal_override` so posture cost anchors null
                # space to a reachable pose.
                _shape_c3 = getattr(self.base_mpc.formulator,
                                    "_object_shape", "box")
                _q_arm_ik = None
                # Per-shape face-normal offset from CoM to contact point on
                # the pushed face. tshape: 0.13×|g_hat_x| + 0.08×|g_hat_y|.
                # box: box_half_extent uniform on both axes (0.05 default).
                _bhe = float(getattr(self.params.sampling_params,
                                     "box_half_extent", 0.05))
                if _shape_c3 == "tshape":
                    # T geometry (reference push_t.sdf + port _tshape_sdf):
                    #   vertical bar: x [-0.03, +0.13], y [-0.02, +0.02]
                    #     (center CoM+0.05 in x, 0.16 x-long, 0.04 y-wide)
                    #   horizontal bar: x [-0.07, -0.03], y [-0.08, +0.08]
                    #     (center CoM-0.05 in x, rotated 90° → 0.04 x-long,
                    #      0.16 y-wide)
                    # For pushing along +y (or -y), arm contacts the
                    # VERTICAL bar's south (or north) face at y = ±0.02.
                    # For pushing along +x (west push), arm contacts the
                    # horizontal bar's east face at x = -0.07 (or vertical
                    # bar's east face at x = +0.13). For -x push, vertical
                    # bar's west face x = -0.03 (or horizontal bar's west
                    # at x = -0.07). Old formula 0.13·|gx|+0.08·|gy| put
                    # arm at |y|=0.08 for pure-y push — outside T's actual
                    # extent (T y-half at vertical bar is 0.02, not 0.08).
                    # New: pick the bar to push based on |gx| vs |gy|.
                    if abs(g_hat_3d[1]) >= abs(g_hat_3d[0]):
                        # dominant y push: target vertical bar's y face
                        _face_offset = 0.02
                    else:
                        # dominant x push: target vertical bar's east face
                        # (x = +0.13) if pushing -x, or horizontal bar's
                        # west face (x = -0.07) if pushing +x
                        _face_offset = (0.13 if g_hat_3d[0] < 0 else 0.07)
                elif _shape_c3 == "box":
                    _face_offset = (abs(g_hat_3d[0]) + abs(g_hat_3d[1])) * _bhe
                else:
                    _face_offset = None  # sphere / unknown → skip IK proj
                if _face_offset is not None:
                    _box_xy_now = np.array([
                        current_q[self._obj_x_idx],
                        current_q[self._obj_y_idx],
                    ])
                    _contact_offset = (_face_offset
                                       + float(getattr(self, "_pusher_radius", 0.0195)))
                    # z target locked at the object CoM z (contact plane).
                    # Previously locked to ee_pos_now[2] at first c3 entry,
                    # which for box was 0.20 m (initial prepositioning) —
                    # arm would then track to z=0.20 forever, never descending
                    # to the box at z=0.05. Using obj_z anchors the target
                    # at the mid-face of the manipuland.
                    _obj_z_now = float(current_q[self._obj_z_idx])
                    if getattr(self, "_c3_geom_z_target", None) is None:
                        self._c3_geom_z_target = _obj_z_now
                    _p_ee_geom = np.array([
                        _box_xy_now[0] - g_hat_3d[0] * _contact_offset,
                        _box_xy_now[1] - g_hat_3d[1] * _contact_offset,
                        float(self._c3_geom_z_target),
                    ])
                    from control.sampling_c3.ik import solve_ik_to_ee_pos
                    _q_lo_full = self.plant.GetPositionLowerLimits()
                    _q_hi_full = self.plant.GetPositionUpperLimits()
                    _q_ik_full, _, _ = solve_ik_to_ee_pos(
                        self.plant, self.ee_frame,
                        _p_ee_geom, current_q, plant_ctx,
                        n_arm_dofs=self.n_u,
                        max_iter=10, tol=2e-3, damping=0.05,
                        q_lo=_q_lo_full, q_hi=_q_hi_full,
                    )
                    self.plant.SetPositions(plant_ctx, current_q)
                    self.plant.SetVelocities(plant_ctx, current_v)
                    _q_arm_ik = _q_ik_full[:self.n_u].copy()
                    _q_full_ik = current_q.copy()
                    _q_full_ik[:self.n_u] = _q_arm_ik
                    self.plant.SetPositions(plant_ctx, _q_full_ik)
                    _p_ee_des = self.plant.CalcPointsPositions(
                        plant_ctx, self.ee_frame, np.zeros(3),
                        self.world_frame,
                    ).flatten()
                    self.plant.SetPositions(plant_ctx, current_q)
                    self.plant.SetVelocities(plant_ctx, current_v)
            else:
                _p_ee_des = ee_pos_now
            # Stage 2b: index into the planner's λ-horizon by elapsed time
            # since the last solve. k = round(elapsed / self._dt) where
            # self._dt = base_mpc.dt = 0.05s is the planning-horizon node
            # spacing. At defaults (N_plan=1, elapsed=0) → k=0 and this
            # reduces to last_lambda_n_first (verified by the integrity
            # assert in compute_control). Falls back to first-knot if the
            # horizon mirror is unavailable (cold start or no contacts).
            _lh_n = getattr(self.base_mpc, "last_lambda_n_horizon", None)
            _lh_t = getattr(self.base_mpc, "last_lambda_t_horizon", None)
            # Read-only diagnostic (env-gated, default off): on first 50 c3
            # ticks AND every 20th c3 tick after, print the full horizon-λ
            # prediction. Used to diagnose whether the planner PREDICTS
            # sustained contact at AT-CoM or only tap-retreat.
            import os as _os
            if _os.environ.get("PUSHA_HORIZON_LAM_DUMP", "0") == "1":
                _hl_step = getattr(self, "_hl_c3_tick", 0) + 1
                self._hl_c3_tick = _hl_step
                if _hl_step <= 50 or _hl_step % 20 == 0:
                    if _lh_n is not None and getattr(_lh_n, "shape", (0,))[0] > 0:
                        _info = getattr(self.base_mpc.formulator,
                                        "_last_contact_info", [])
                        _tags = [i.get("tag", "?") for i in _info] if _info else []
                        print(f"[HORIZON-LAM] c3_tick={_hl_step} step={self._step} "
                              f"shape={_lh_n.shape} tags={_tags}", flush=True)
                        for _kk in range(_lh_n.shape[0]):
                            _row = ",".join(f"{float(v):.3f}" for v in _lh_n[_kk])
                            print(f"[HORIZON-LAM]   k={_kk}: lam_n=[{_row}]",
                                  flush=True)
                    else:
                        print(f"[HORIZON-LAM] c3_tick={_hl_step} step={self._step} "
                              f"horizon-λ unavailable", flush=True)
            if _lh_n is not None and getattr(_lh_n, "shape", (0,))[0] > 0:
                _k = min(int(round(elapsed / self._dt)), _lh_n.shape[0] - 1)
                _lam_n = _lh_n[_k]
                if _lh_t is not None and getattr(_lh_t, "shape", (0,))[0] > 0:
                    _lam_t = _lh_t[_k]
                else:
                    _lam_t = getattr(self.base_mpc, "last_lambda_t_first", None)
            else:
                _lam_n = getattr(self.base_mpc, "last_lambda_n_first", None)
                _lam_t = getattr(self.base_mpc, "last_lambda_t_first", None)
            _Jn    = self.base_mpc.formulator._last_J_n
            _Jt    = self.base_mpc.formulator._last_J_t
            # SIGN-BUG FIX: only issue a force-tracking command when the
            # planner actually predicts contact force on the EE-manipuland
            # pair. Ground-contact λ_n can be ~2 N even without EE contact,
            # which would spuriously trigger `_derive_force_command` and
            # re-open the fictional-force drift.
            #
            # Filter shape-gated to match `_derive_force_command` (line 611+):
            #   tshape → filter EE-BOX tags only
            #   box    → raw sum (byte-identical to pre-fix)
            _shape_gate = getattr(self.base_mpc.formulator,
                                   "_object_shape", "box")
            _cinfo_gate = getattr(self.base_mpc.formulator,
                                   "_last_contact_info", None)
            if (_lam_n is not None
                    and hasattr(_lam_n, "size")
                    and _lam_n.size > 0):
                if (_shape_gate == "tshape"
                        and _cinfo_gate is not None
                        and len(_cinfo_gate) == _lam_n.size):
                    _ee_idxs_gate = [i for i, info in enumerate(_cinfo_gate)
                                     if isinstance(info, dict)
                                     and info.get("tag", "") == "EE-BOX"]
                    if _ee_idxs_gate:
                        _lam_n_mag = float(np.sum(np.abs(_lam_n[_ee_idxs_gate])))
                    else:
                        _lam_n_mag = 0.0
                else:
                    _lam_n_mag = float(np.sum(np.abs(_lam_n)))
            else:
                _lam_n_mag = 0.0
            if _lam_n_mag > 0.05:
                _lam_des = self._derive_force_command(_lam_n, g_hat_3d)
            else:
                _lam_des = np.zeros(3)

            import os as _os_fr
            if _os_fr.environ.get("PUSHA_FORCE_ROUTE_TRACE", "0") == "1":
                _fr_env = _os_fr.environ.get("PUSHA_FORCE_ROUTING", "off").lower()
                _u0 = getattr(self.base_mpc, "_last_u_seq", None)
                if _u0 is not None and hasattr(_u0, "shape") and _u0.ndim == 2 and _u0.shape[1] == 3:
                    _u_seq0 = np.asarray(_u0[0], dtype=float).reshape(3)
                else:
                    _u_seq0 = np.zeros(3)
                _ld = np.asarray(_lam_des, dtype=float).reshape(3)
                _eq = bool(np.allclose(_ld, _u_seq0, atol=1e-9))
                print(f"[FORCE-ROUTE] tick={self._step} env={_fr_env} "
                      f"lambda_des=[{_ld[0]:+.4f},{_ld[1]:+.4f},{_ld[2]:+.4f}] "
                      f"u_seq0=[{_u_seq0[0]:+.4f},{_u_seq0[1]:+.4f},{_u_seq0[2]:+.4f}] "
                      f"u_z={_u_seq0[2]:+.4f} eq={_eq}",
                      flush=True)

            # Stage C post-FAIL localization probe — [SETPOINT] trace.
            # Splits the Phase 1 FAIL bottleneck across the upstream-vs-executor
            # fork: does the planner *predict* contact (x_seq EE-to-box closing
            # below 2 mm anywhere in the horizon) and ship a contact-seeking
            # p_ee_des to the OSC, or does it park p_ee_des in the [2,5 mm)
            # hover band and leave the OSC tracking a hovering setpoint?
            # Gated by PUSHA_SETPOINT_TRACE=1 (default-OFF). EE-space-only.
            if (_os_fr.environ.get("PUSHA_SETPOINT_TRACE", "0") == "1"
                    and _use_ee_space and _x_seq is not None and len(_x_seq) >= 2):
                _box_now = np.array([
                    current_q[self._obj_x_idx],
                    current_q[self._obj_y_idx],
                    current_q[self._obj_z_idx],
                ])
                _half = 0.05   # box half-extent (config/tasks.yaml pushing cube)
                _sr   = 0.025  # pusher sphere radius

                def _sphere_to_box_phi(ee_p, box_p):
                    # Axis-aligned approximation (box rotation neglected for the
                    # probe; box yaw stays small over 1 horizon × 0.05 s).
                    d = np.abs(ee_p - box_p) - _half
                    outside = float(np.linalg.norm(np.maximum(d, 0.0)))
                    inside  = float(min(0.0, np.max(d)))
                    return outside + inside - _sr

                _phi_pred_horizon = []
                for _k in range(1, _x_seq.shape[0]):
                    _ee_k  = np.asarray(_x_seq[_k][7:10], dtype=float)
                    _box_k = np.asarray(_x_seq[_k][4:7],  dtype=float)
                    _phi_pred_horizon.append(_sphere_to_box_phi(_ee_k, _box_k))
                _phi_pred1   = _phi_pred_horizon[0]
                _phi_pred_mn = float(min(_phi_pred_horizon))
                _kmin        = int(_phi_pred_horizon.index(_phi_pred_mn)) + 1
                _phi_pred_mx = float(max(_phi_pred_horizon))
                _phi_act     = _sphere_to_box_phi(ee_pos_now, _box_now)
                _setpoint_sd = _sphere_to_box_phi(_p_ee_des, _box_now)
                _N           = int(_x_seq.shape[0] - 1)
                print(
                    f"[SETPOINT] tick={self._step} N={_N} "
                    f"p_ee_des=[{_p_ee_des[0]:+.4f},{_p_ee_des[1]:+.4f},{_p_ee_des[2]:+.4f}] "
                    f"box_now=[{_box_now[0]:+.4f},{_box_now[1]:+.4f},{_box_now[2]:+.4f}] "
                    f"setpoint_sd={_setpoint_sd:+.5f} "
                    f"phi_act={_phi_act:+.5f} "
                    f"phi_pred1={_phi_pred1:+.5f} "
                    f"phi_pred_min={_phi_pred_mn:+.5f}@k={_kmin} "
                    f"phi_pred_max={_phi_pred_mx:+.5f}",
                    flush=True,
                )

            # Stage C probe B [CONSISTENCY] — ALIGNMENT-vs-RESEARCH phase
            # detector. Reads (1) u_sol direction vs toward-box & toward-goal,
            # (2) predicted EE step k=0→1 vs u_sol direction (coherence: does
            # the planner's solved force CAUSE the predicted EE motion?),
            # (3) predicted box-CoM displacement over the full horizon vs
            # goal direction (coherence: does the predicted box TRANSLATE
            # under u_sol?), (4) ADMM terminal state (pr, dr, iters/max,
            # converged?). Gated by PUSHA_CONSISTENCY_TRACE=1; default-OFF;
            # EE-space-only.
            if (_os_fr.environ.get("PUSHA_CONSISTENCY_TRACE", "0") == "1"
                    and _use_ee_space and _x_seq is not None
                    and len(_x_seq) >= 2):
                _u_seq_full = getattr(self.base_mpc, "_last_u_seq", None)
                if (_u_seq_full is not None and hasattr(_u_seq_full, "shape")
                        and _u_seq_full.ndim == 2 and _u_seq_full.shape[1] == 3):
                    _u0 = np.asarray(_u_seq_full[0], dtype=float).reshape(3)
                else:
                    _u0 = np.zeros(3)
                _u_mag = float(np.linalg.norm(_u0))
                _u_dir = (_u0 / _u_mag) if _u_mag > 1e-9 else np.zeros(3)

                # Toward-box (3D) and toward-goal (XY) direction unit vectors.
                _ee_now_v = np.asarray(ee_pos_now, dtype=float).reshape(3)
                _box_now_v = np.array([
                    current_q[self._obj_x_idx],
                    current_q[self._obj_y_idx],
                    current_q[self._obj_z_idx],
                ], dtype=float)
                _to_box = _box_now_v - _ee_now_v
                _to_box_norm = float(np.linalg.norm(_to_box))
                _to_box_dir = (_to_box / _to_box_norm) if _to_box_norm > 1e-9 else np.zeros(3)
                _goal_dir_xy = -np.asarray(g_hat_3d, dtype=float).reshape(3)[:2]
                _goal_dir_norm = float(np.linalg.norm(_goal_dir_xy))
                _goal_dir_xy = (_goal_dir_xy / _goal_dir_norm) if _goal_dir_norm > 1e-9 else np.zeros(2)
                _u_dot_box  = float(np.dot(_u_dir, _to_box_dir))
                _u_dot_goal = float(np.dot(_u_dir[:2], _goal_dir_xy))

                # Predicted EE step k=0→1 in world frame.
                _ee_pred_0 = np.asarray(_x_seq[0][7:10], dtype=float)
                _ee_pred_1 = np.asarray(_x_seq[1][7:10], dtype=float)
                _ee_step_1 = _ee_pred_1 - _ee_pred_0
                _ee_step_1_mag = float(np.linalg.norm(_ee_step_1))
                if _ee_step_1_mag > 1e-9:
                    _ee_step1_dot_box = float(np.dot(_ee_step_1 / _ee_step_1_mag,
                                                     _to_box_dir))
                    # Does the predicted EE step align with the solved u_sol?
                    _ee_step1_dot_u  = float(np.dot(_ee_step_1 / _ee_step_1_mag,
                                                     _u_dir))
                else:
                    _ee_step1_dot_box = 0.0
                    _ee_step1_dot_u  = 0.0

                # Predicted box-CoM net displacement over the full horizon.
                _box_pred_0   = np.asarray(_x_seq[0][4:7], dtype=float)
                _box_pred_end = np.asarray(_x_seq[-1][4:7], dtype=float)
                _box_total    = _box_pred_end - _box_pred_0
                _box_total_xy_mag = float(np.linalg.norm(_box_total[:2]))
                if _box_total_xy_mag > 1e-9:
                    _box_dot_goal = float(np.dot(_box_total[:2] / _box_total_xy_mag,
                                                  _goal_dir_xy))
                else:
                    _box_dot_goal = 0.0

                # ADMM terminal state (forwarded by ci_mpc_c3plus.py).
                _pr     = float(getattr(self.base_mpc, "last_pr_final",  float("nan")))
                _dr     = float(getattr(self.base_mpc, "last_dr_final",  float("nan")))
                _it_used= int(  getattr(self.base_mpc, "last_iters_used", 0))
                _tol    = float(getattr(self.base_mpc, "last_tol",       1e-3))
                _conv   = bool( getattr(self.base_mpc, "last_converged", True))

                print(
                    f"[CONSISTENCY] tick={self._step} "
                    f"u_mag={_u_mag:.4f} u_dot_box={_u_dot_box:+.3f} u_dot_goal={_u_dot_goal:+.3f} "
                    f"ee_step1_mag={_ee_step_1_mag*1000:.4f}mm "
                    f"ee_step1_dot_box={_ee_step1_dot_box:+.3f} "
                    f"ee_step1_dot_u={_ee_step1_dot_u:+.3f} "
                    f"box_total_xy={_box_total_xy_mag*1000:.4f}mm "
                    f"box_dot_goal={_box_dot_goal:+.3f} "
                    f"admm_pr={_pr:.4f} admm_dr={_dr:.4f} "
                    f"admm_iters={_it_used}/{self._admm_iter if hasattr(self,'_admm_iter') else 25} "
                    f"admm_tol={_tol:.0e} converged={_conv}",
                    flush=True,
                )

            # Lever 3: c3 approach-closing override. When LCS admits no
            # EE-BOX pair, the planner sees no contact and parks the EE in
            # place — but the typical arrival sphere-to-box gap is ~6 mm
            # while LCS admits only at <= 2 mm: chicken-and-egg. Drive the
            # EE toward the box CoM along the box-EE vector (not -g_hat:
            # off-axis arrivals would be pushed away), clamped so the
            # commanded target stays >= LCS_DISTANCE_THRESHOLD outside
            # contact (non-penetrating by construction). Self-disabling
            # the instant LCS admits an EE-BOX pair: planner authority
            # returns automatically.
            #
            # Indicator: formulator._last_ee_box_contacts (EE-BOX-only list).
            # Cannot key on _lam_n.size because lam_n includes the BOX-GND
            # contact row, which is always admitted (the box rests on the
            # ground), so lam_n.size >= 1 even with no EE-BOX pair.
            _ee_box_pairs = getattr(self.base_mpc.formulator,
                                    "_last_ee_box_contacts", [])
            # §7.51 — PUSHA_DISABLE_C3_OVERRIDE (default-OFF) skips the LTD
            # APPROACH-OVERRIDE block entirely, leaving _p_ee_des at the FK
            # source (_x_seq[1][7:10] at line 2263). Validated as load-bearing
            # for the first box closure in §7.51 (with PUSHA_EE_APPROACH_FACE_
            # TARGET=1 + w_ee_approach=8000 + W_force=1). Default-OFF
            # byte-identical preserved. One-shot log on first c3 tick.
            _disable_c3_override = (_os.environ.get(
                "PUSHA_DISABLE_C3_OVERRIDE", "0") == "1")
            if _disable_c3_override and not getattr(
                    self, "_751_banner_printed", False):
                print("[§7.51] PUSHA_DISABLE_C3_OVERRIDE=1 — LTD APPROACH-"
                      "OVERRIDE skipped in c3; p_ee_des = FK(x_seq[1][7:10])",
                      flush=True)
                self._751_banner_printed = True
            _no_admitted_pair = ((len(_ee_box_pairs) == 0)
                                 and not _disable_c3_override)
            _override_fired_this_tick = False
            if _no_admitted_pair:
                _box_xyz = np.array([
                    current_q[self._obj_x_idx],
                    current_q[self._obj_y_idx],
                    current_q[self._obj_z_idx],
                ])
                # Lever 3.1: aim the approach at the centroid of the box
                # face the EE must contact to push the box toward goal.
                # Selection rule (directional): among the four SIDE faces
                # (body axes 0, 1; both signs — exclude top/bottom z), pick
                # the face whose outward normal in world frame best aligns
                # with -g_hat. Score = sign * (R_box.T @ (-g_hat))[axis];
                # the argmax is the face the EE should push from to send
                # the box toward goal.
                #
                # Replaces argmax(|ee_in_box_local|), which picked the
                # nearest face geometrically. That logic picked face_axis=2
                # (TOP) whenever |z_local| dominated — i.e. whenever the EE
                # was above the box footprint — and aimed Lever-3 at the
                # top centroid. Result on canonical baseline (20 seeds,
                # commit 38dbf18): 192/192 top-face picks, surf_dist grew
                # 7.5 cm → 33 cm as EE chased the target upward, 0/20
                # seeds formed any EE-BOX contact.
                #
                # Box-local rotation kept (handles in-run box yaw — we saw
                # nhat tilt 1.000 → 0.986 across a contact burst).
                _qw = float(current_q[self._obj_qw])
                _qx = float(current_q[self._obj_qx])
                _qy = float(current_q[self._obj_qy])
                _qz = float(current_q[self._obj_qz])
                _R_box = np.array([
                    [1 - 2*(_qy*_qy + _qz*_qz),     2*(_qx*_qy - _qz*_qw),     2*(_qx*_qz + _qy*_qw)],
                    [    2*(_qx*_qy + _qz*_qw), 1 - 2*(_qx*_qx + _qz*_qz),     2*(_qy*_qz - _qx*_qw)],
                    [    2*(_qx*_qz - _qy*_qw),     2*(_qy*_qz + _qx*_qw), 1 - 2*(_qx*_qx + _qy*_qy)],
                ])
                _ee_from_box_W = ee_pos_now - _box_xyz
                _dist_com = float(np.linalg.norm(_ee_from_box_W))
                _push_dir_W = -np.asarray(g_hat_3d, dtype=float).reshape(3)
                _push_dir_L = _R_box.T @ _push_dir_W
                _best_score = -np.inf
                _face_axis = 0
                _face_sign = 1
                for _a in (0, 1):
                    for _s in (1, -1):
                        _sc = _s * float(_push_dir_L[_a])
                        if _sc > _best_score:
                            _best_score = _sc
                            _face_axis = _a
                            _face_sign = _s
                if _dist_com > 1e-9 and _best_score > 1e-6:
                    _box_half = float(self.params.sampling_params.box_half_extent)
                    _face_centroid_local = np.zeros(3)
                    _face_centroid_local[_face_axis] = _face_sign * _box_half
                    _face_centroid_W = _box_xyz + _R_box @ _face_centroid_local

                    _ee_to_face = _face_centroid_W - ee_pos_now
                    _dist_face = float(np.linalg.norm(_ee_to_face))
                    if _dist_face > 1e-9:
                        # The face centroid IS the surface — no box_half
                        # subtraction; only the sphere radius separates
                        # EE-center from face-plane at contact.
                        _surf_dist = _dist_face - PUSHER_RADIUS

                        # Pick the per-tick aim target: lift-traverse-descend
                        # waypoints, or legacy direct-to-face-centroid line.
                        if self.params.use_lift_traverse_descend_override:
                            # Clearance is a CORRECTNESS FLOOR, not a knob.
                            # During PHASE B descent the sphere SURFACE sits
                            # at (clearance - PUSHER_RADIUS) from the face
                            # plane. Below the floor the sphere admits
                            # contact mid-descent, re-introducing the same
                            # bypass LTD exists to fix.
                            _clearance = float(self.params.ltd_clearance)
                            _min_clearance = (PUSHER_RADIUS
                                              + LCS_DISTANCE_THRESHOLD
                                              + 0.005)
                            assert _clearance >= _min_clearance, (
                                f"ltd_clearance={_clearance:.4f} m < floor "
                                f"{_min_clearance:.4f} m (PUSHER_RADIUS + "
                                f"LCS_DISTANCE_THRESHOLD + 5 mm safety); "
                                f"PHASE B descent would admit contact "
                                f"before reaching the face."
                            )

                            # W_side: sphere center beside box at face mid-
                            # height. n̂_face is purely lateral for face_axis
                            # ∈ {0,1}, so W_side.z equals box_xyz.z for an
                            # upright box — clamp z explicitly so the spec
                            # holds independent of R_box rounding.
                            _face_centroid_z = float(_box_xyz[2])
                            _n_face_W = _R_box[:, _face_axis] * _face_sign
                            _W_side = (_box_xyz
                                       + _n_face_W * (_box_half + _clearance))
                            _W_side[2] = _face_centroid_z

                            # Workspace bounds: assert + log + fall back to
                            # legacy direct-line if W_side is outside the
                            # planner's workspace. Silent clipping would
                            # aim the EE at a geometrically wrong point.
                            _ws_min = np.asarray(
                                self.params.sampling_params.workspace_xy_min,
                                dtype=float)
                            _ws_max = np.asarray(
                                self.params.sampling_params.workspace_xy_max,
                                dtype=float)
                            _W_side_in_ws = (
                                _ws_min[0] <= _W_side[0] <= _ws_max[0]
                                and _ws_min[1] <= _W_side[1] <= _ws_max[1]
                            )

                            if not _W_side_in_ws:
                                if self.log_diag:
                                    print(
                                        f"[LTD-WORKSPACE-FALLBACK] "
                                        f"step={self._step} "
                                        f"W_side=({_W_side[0]:+.4f},"
                                        f"{_W_side[1]:+.4f}) outside "
                                        f"workspace_xy="
                                        f"[{_ws_min[0]:+.3f},"
                                        f"{_ws_max[0]:+.3f}]x"
                                        f"[{_ws_min[1]:+.3f},"
                                        f"{_ws_max[1]:+.3f}]; "
                                        f"falling back to direct-line",
                                        flush=True)
                                _target = _face_centroid_W
                                _phase = 'fallback_ws'
                            else:
                                _z_margin = float(self.params.ltd_z_margin)
                                _z_safe = (_box_xyz[2] + _box_half
                                           + PUSHER_RADIUS + _z_margin)
                                _W_lift_trav = _W_side.copy()
                                _W_lift_trav[2] = max(float(ee_pos_now[2]),
                                                      _z_safe)

                                # Stateless phase decision from EE
                                # position alone (no persisted state,
                                # no contact-pair input, no dispatcher
                                # mode coupling).
                                _xy_dist_to_W_side = float(np.linalg.norm(
                                    (ee_pos_now - _W_side)[:2]))
                                _z_above_W_side = float(
                                    ee_pos_now[2] - _W_side[2])
                                _xy_tol = float(self.params.ltd_xy_tol)
                                _z_band = float(self.params.ltd_z_band)

                                if _xy_dist_to_W_side > _xy_tol:
                                    _target = _W_lift_trav
                                    _phase = 'A_lift_trav'
                                elif _z_above_W_side > _z_band:
                                    _target = _W_side
                                    _phase = 'B_descend'
                                else:
                                    # PHASE C: rigid z clamp to face mid-
                                    # height. Per-tick command never aims
                                    # above face_centroid_z regardless of
                                    # where the sphere has drifted. The
                                    # OSC's Kp_z provides the restoring
                                    # force toward face center; if shear
                                    # creeps EE upward despite that, the
                                    # verification surfaces it cleanly.
                                    _target = np.array([
                                        float(_face_centroid_W[0]),
                                        float(_face_centroid_W[1]),
                                        _face_centroid_z,
                                    ])
                                    _phase = 'C_approach'
                        else:
                            _target = _face_centroid_W
                            _phase = 'legacy'

                        _ee_to_target = _target - ee_pos_now
                        _dist_target = float(np.linalg.norm(_ee_to_target))
                        # Cap advance to avoid (a) overshooting the chosen
                        # target and (b) penetrating LCS_DISTANCE_THRESHOLD
                        # against the chosen face plane.
                        _advance_target = max(
                            _dist_target - LCS_DISTANCE_THRESHOLD, 0.0)
                        _advance_face = max(
                            _surf_dist - LCS_DISTANCE_THRESHOLD, 0.0)
                        _advance = min(MAX_APPROACH_STEP,
                                       _advance_target, _advance_face)
                        if _advance > 0 and _dist_target > 1e-9:
                            _p_ee_des = (ee_pos_now
                                         + _advance * (_ee_to_target
                                                       / _dist_target))
                            _override_fired_this_tick = True
                            if self.log_diag:
                                print(f"[APPROACH-OVERRIDE] step={self._step} "
                                      f"phase={_phase} "
                                      f"face_axis={_face_axis} face_sign={_face_sign:+d} "
                                      f"face_score={_best_score:+.3f} "
                                      f"surf_dist={_surf_dist:.4f}m "
                                      f"advance={_advance:.4f}m "
                                      f"target=({_target[0]:+.4f},"
                                      f"{_target[1]:+.4f},"
                                      f"{_target[2]:+.4f}) "
                                      f"p_ee_des=({_p_ee_des[0]:+.4f},"
                                      f"{_p_ee_des[1]:+.4f},"
                                      f"{_p_ee_des[2]:+.4f})",
                                      flush=True)
            # Expose override firing state to the contact-loss gate (read
            # on the next tick). Phase is needed too because PHASE A gets
            # an extended grace threshold (longer traverse).
            self._approach_override_firing = _override_fired_this_tick
            _new_phase = _phase if _override_fired_this_tick else 'none'
            if _new_phase != self._approach_override_phase:
                # On any LTD phase transition (A→B, B→C, override on/off),
                # reset the contact-loss streak so each phase gets its own
                # grace budget. Without this, an inherited streak from
                # PHASE B's extended threshold (300) immediately fires the
                # gate as soon as PHASE C's stricter default (12) takes
                # effect (see audit_output/ltd_diag_phaseB).
                self._no_ee_box_streak = 0
                # PHASE C tracker triple: reset on any intra-c3 phase
                # transition (B→C entry, C→B/A ping-pong). Belt-and-braces
                # with the free→c3 reset at line 871 — that reset handles
                # C→free→C re-entry (where this block does NOT fire because
                # _new_phase == _approach_override_phase == 'C_approach'),
                # this reset handles in-c3 transitions (where the free
                # reset does NOT fire because prev_mode is still 'c3').
                self._phaseC_stall_streak = 0
                self._phaseC_active_streak = 0
                self._phaseC_surf_dist_min = float('inf')
                self._last_C_surf_dist = None
            self._approach_override_phase = _new_phase
            # Cache surf_dist for the PHASE C tracker update next tick
            # (read at line ~1043 in _solve_plan). Only set when the
            # override fired in PHASE C this tick — _surf_dist exists
            # in scope because 'C_approach' is only reachable via the
            # LTD path that sets _surf_dist at line ~1559. None on all
            # other phases / non-firing ticks so the tracker update
            # next tick skips silently when prev tick wasn't C.
            if _override_fired_this_tick and _phase == 'C_approach':
                self._last_C_surf_dist = float(_surf_dist)
            else:
                self._last_C_surf_dist = None

            _v_ee_des = self._velocity_feedforward_from_xseq(
                plant_ctx, current_q, current_v
            )
            # §7.34 — FAITHFUL-DESIRED-STATE FEEDFORWARD-ACCEL
            # Adds yddot_des leg to the OSC PD law so port matches the
            # reference's `yddot_command = yddot_des + Kp·error_y + Kd·error_ydot`.
            # Returns None when REF_RECONCILE_FEEDFORWARD_ACCEL is OFF →
            # byte-identical PD-only path.
            _a_ee_des = self._acceleration_feedforward_from_xseq()
            # Under --ee-space, the planner's J_n / J_t are in low-dim
            # velocity coords [box_v(6), v_ee(3)] — not n_v_full(13). The
            # executor uses J_n.T @ λ_n in n_v space; the planner's λ
            # scalar values map identically (one entry per contact pair),
            # only the Jacobian shape changes. linearize_discrete_ee_space
            # stashes the n_v_full Drake Jacobians; pass those instead so
            # the τ_ff = -J_n^T λ feedforward composes correctly.
            if bool(getattr(self.base_mpc, "use_ee_space", False)):
                _f = self.base_mpc.formulator
                _exec_Jn = getattr(_f, "_last_J_n_n_v_full", None)
                _exec_Jt = getattr(_f, "_last_J_t_n_v_full", None)
                _exec_lam_n = _lam_n
                _exec_lam_t = _lam_t
                # Defensive: if shapes still don't match (cold start, or
                # contact count drift between planner and executor ticks),
                # drop the feedforward rather than crash.
                if (_exec_Jn is None or _exec_Jt is None
                        or _exec_lam_n is None or _exec_lam_t is None
                        or _exec_Jn.shape[0] != len(_exec_lam_n)
                        or _exec_Jt.shape[0] != len(_exec_lam_t)):
                    _exec_lam_n, _exec_lam_t = None, None
                    _exec_Jn, _exec_Jt = None, None
            else:
                _exec_lam_n, _exec_lam_t = _lam_n, _lam_t
                _exec_Jn, _exec_Jt = _Jn, _Jt
            # §7.32 — FAITHFUL-DESIRED-STATE: the §7.31 static surface-point
            # override (a) was the over-drive failure mode (231 N, obj_z
            # → −78 m). Dropped here; _p_ee_des stays as the planner's
            # first-knot prediction (_x_seq[1][7:10] above) which IS the
            # contact-establishment plan when always-on + proxy off (the
            # plan evolves toward contact over the horizon and is
            # goal-aware). _v_ee_des already carries the planner's
            # predicted EE velocity at alpha = 1.0 under reconcile (see
            # _velocity_feedforward_from_xseq).
            # Reproduce-dairlib Phase 1: dispatch c3-mode OSC through the
            # trajectory-shaped interface. Two-knot FirstOrderHold from
            # ee_pos_now to _p_c3 over dt_ctrl so `traj.EvalDerivative(t, 1)`
            # returns the intended EE velocity (p_c3 - ee_now)/dt_ctrl rather
            # than 0. Previous single-knot ZOH caused v_ee_desired to default
            # to 0 in compute_torque_from_trajectory (line 555-560), which
            # turned OSC's v_err into pure -v_ee_now damping. When the arm
            # was already moving toward p_c3, Kd·v_err cancelled Kp·p_err
            # and drove |u| to near zero — arm coasted, never contacting
            # box. Bug surfaced after the use_velocity_feedforward gate
            # (8098e2d) closed the alternate _v_ee_des path.
            # Align to real sim time at compute_control entry (self._step
            # already incremented at line 850). Sub-tick OSC calls evaluate
            # the traj at t ∈ [t_start, t_start+dt_ctrl] → interior
            # interpolation → p_ee_desired sweeps from ee_now to p_c3, and
            # v_ee_desired = (p_c3 - ee_now)/dt_ctrl throughout the window.
            _sim_t_c3 = float(self._step - 1) * float(self._dt_ctrl)
            _p_c3_col = np.asarray(_p_ee_des, dtype=float).reshape(3, 1)
            _p_now_col = np.asarray(ee_pos_now, dtype=float).reshape(3, 1)
            _traj_c3 = PiecewisePolynomial.FirstOrderHold(
                [_sim_t_c3, _sim_t_c3 + float(self._dt_ctrl)],
                np.hstack([_p_now_col, _p_c3_col]),
            )
            u_imp, imp_diag = self.executor.compute_torque_from_trajectory(
                traj = _traj_c3, t_sim = _sim_t_c3,
                current_q = current_q, current_v = current_v,
                plant_ctx = plant_ctx,
                v_ee_desired = _v_ee_des,
                lambda_n     = _exec_lam_n,
                lambda_t     = _exec_lam_t,
                J_n          = _exec_Jn,
                J_t          = _exec_Jt,
                lambda_des   = _lam_des,
                a_ee_desired = _a_ee_des,
                mode         = "c3",  # §7.70 — reference-gain swap gate (now default)
                # IK-projected joint-space guidance for tshape c3 mode
                # (set only when PUSHA_TSHAPE_C3_GEOM=1 fires); None
                # otherwise → OSC falls back to constructor's q_nominal.
                q_nominal_override = _q_arm_ik,
            )
            # 1 kHz OSC decoupling: cache trajectory + planner-tick kwargs so
            # sub-tick `compute_control_osc_only` can re-evaluate the OSC on
            # fresh state without re-running the planner.
            self._last_osc_call = ("c3_traj", dict(
                traj=_traj_c3,
                v_ee_desired=_v_ee_des,
                lambda_n=_exec_lam_n, lambda_t=_exec_lam_t,
                J_n=_exec_Jn, J_t=_exec_Jt,
                lambda_des=_lam_des,
                a_ee_desired=_a_ee_des,
                mode="c3",
                q_nominal_override=_q_arm_ik,
            ))
            # Velocity-feedforward A/B telemetry. Emit unconditionally so
            # the alpha=0 / disabled run has parsable rows for the baseline
            # comparison (None → 0-vector for the log; the actual semantics
            # are the imp_diag.xdot_err which the executor used).
            if self.log_diag:
                if _v_ee_des is None:
                    _vff_dump = np.zeros(3)
                    _vff_mag  = 0.0
                else:
                    _vff_dump = _v_ee_des
                    _vff_mag  = float(np.linalg.norm(_v_ee_des))
                _alpha_eff = (float(self.params.velocity_feedforward_alpha)
                              if getattr(self.params,
                                         "use_velocity_feedforward", False)
                              else 0.0)
                _sat_flag = bool(imp_diag.get("saturated", False))
                _tau = imp_diag.get("tau_out")
                _tau_max = np.asarray(
                    self.executor.limits.tau_max
                    if hasattr(self.executor, "limits")
                    else [87.0] * len(_tau), dtype=float)
                # Per-joint headroom: |tau_i| / tau_max_i. Closest-to-1.0
                # joint is the bottleneck; >0.95 == near-saturation.
                if _tau is not None and len(_tau) == len(_tau_max):
                    _util = np.abs(np.asarray(_tau, dtype=float)) / _tau_max
                    _util_max = float(np.max(_util))
                    _util_argmax = int(np.argmax(_util))
                else:
                    _util_max = float('nan')
                    _util_argmax = -1
                print(f"[VFF] step={self._step} mode=c3 "
                      f"alpha={_alpha_eff:.3f} "
                      f"v_des=({_vff_dump[0]:+.4f},{_vff_dump[1]:+.4f},"
                      f"{_vff_dump[2]:+.4f}) "
                      f"|v_des|={_vff_mag:.4f} "
                      f"sat={int(_sat_flag)} "
                      f"util_max={_util_max:.3f}@j{_util_argmax}",
                      flush=True)
                # §7.34 — feedforward-accel telemetry. Emit per c3 tick.
                if _a_ee_des is None:
                    _aff_dump = np.zeros(3)
                    _aff_mag  = 0.0
                    _aff_on   = 0
                else:
                    _aff_dump = _a_ee_des
                    _aff_mag  = float(np.linalg.norm(_a_ee_des))
                    _aff_on   = 1
                print(f"[AFF] step={self._step} mode=c3 "
                      f"on={_aff_on} "
                      f"a_ff=({_aff_dump[0]:+.4f},{_aff_dump[1]:+.4f},"
                      f"{_aff_dump[2]:+.4f}) "
                      f"|a_ff|={_aff_mag:.4f}",
                      flush=True)
            # Sink 3 vs Sink 4 diagnostic: split OSC's commanded vs solved force
            _ld = imp_diag.get('lambda_des')
            _le = imp_diag.get('lambda_ext')
            _ld_mag = float(np.linalg.norm(_ld)) if _ld is not None else float('nan')
            _le_mag = float(np.linalg.norm(_le)) if _le is not None else float('nan')
            print(f"[OSC-FORCE] step={self._step} mode=c3 "
                  f"lam_des={_ld_mag:.3f} lam_ext={_le_mag:.3f}", flush=True)
        else:
            # Free-mode position target.
            #
            # Stage A — Reposition mechanism port (env-flag
            # PUSHA_REPOSITION_PWL=1, params.use_reposition_pwl_trajectory).
            # When ON: build/refresh a RepositionTrajectory at planner
            # cadence (or on target change > 5 mm), eval at current sim_t
            # to get (p_des, v_des), feed to OSC. The legacy per-tick
            # setpoint march + per-knot IK + joint-PD path is bypassed.
            # Force=0 during reposition (Stage C is separate).
            #
            # When OFF: existing free_diag['p_des'] read from legacy
            # tracker (RepositionIKTracker / PiecewiseLinearTracker).
            if self._use_pwl_traj:
                # Clear stale trajectory across c3→free transitions so the
                # rebuild below triggers fresh with the new p_start (rather
                # than inheriting a trajectory whose p_start was the
                # PREVIOUS free episode's start).
                if self._prev_mode == "c3":
                    self._pwl_traj = None
                    self._pwl_traj_built_for_target = None
                # Rebuild triggers (anti-churn — Refinement 3):
                #   (a) no trajectory yet (first free entry after a c3→free
                #       transition reset, OR sim start), OR
                #   (b) target moved > 5 mm vs build-time target.
                # NOTE: trajectory.is_finished is NOT a rebuild trigger.
                # Past t_end, RepositionTrajectory.eval returns
                # (p_target, 0, True) which holds the target — rebuilding
                # there would be the per-tick march in disguise.
                # self._step was incremented at the TOP of compute_control
                # (line 850). At that point the real sim clock is
                # (self._step - 1) * dt_ctrl (K-1 prior planner ticks have
                # elapsed). Use that so trajectories are built with
                # t_start = real sim time, letting 1 kHz OSC sub-ticks land
                # in the traj interior rather than clamping to the pre-
                # t_start start-knot branch (RepositionTrajectory.eval:102).
                _sim_t = float(self._step - 1) * float(self._dt_ctrl)
                _p_target = (
                    self._current_repos_target
                    if self._current_repos_target is not None
                    else ee_pos_now
                )
                _p_target_arr = np.asarray(_p_target, dtype=float).reshape(3)
                _need_rebuild = (
                    self._pwl_traj is None
                    or self._pwl_traj_built_for_target is None
                    or float(np.linalg.norm(
                        _p_target_arr
                        - self._pwl_traj_built_for_target)) > 5e-3
                )

                # Stage C landing-storm trace — gated, default-OFF.
                # Window: step >= PUSHA_LANDING_TRACE_FROM (default 1600 at
                # 1 kHz; ~step 160 at 100 Hz to capture the same sim-time
                # window). One consolidated emit per tick at the rebuild
                # gate carrying the full chain state. The fix is decided
                # AFTER the read — this block adds NO behavior change.
                import os as _os_lt
                _trace_on = _os_lt.environ.get("PUSHA_LANDING_TRACE", "0") == "1"
                _trace_from = int(_os_lt.environ.get("PUSHA_LANDING_TRACE_FROM", "1600"))
                if _trace_on and self._step >= _trace_from:
                    _built = self._pwl_traj_built_for_target
                    if _built is not None:
                        _delta_mm = float(np.linalg.norm(
                            _p_target_arr - _built)) * 1000.0
                    else:
                        _delta_mm = float("nan")
                    _sim_t_now = float(self._step - 1) * float(self._dt_ctrl)
                    # PWL trajectory finished-flag debug
                    if self._pwl_traj is not None:
                        _pwl_t_end = float(self._pwl_traj.t_end)
                        _pwl_p_tgt = self._pwl_traj.p_target.tolist()
                        _dist_to_pwl_target_mm = float(np.linalg.norm(
                            np.asarray(ee_pos_now) - self._pwl_traj.p_target)) * 1000.0
                        _is_finished = self._pwl_traj.is_finished(
                            _sim_t_now, ee_pos_now, tol=0.005)
                    else:
                        _pwl_t_end = float("nan")
                        _pwl_p_tgt = None
                        _dist_to_pwl_target_mm = float("nan")
                        _is_finished = False
                    _refresh_at = getattr(self, "_landing_trace_refresh_fired_at", None)
                    print(
                        f"[LANDING-TRACE] step={self._step} "
                        f"sim_t={_sim_t_now:.4f} "
                        f"mode={mode} "
                        f"need_rebuild={_need_rebuild} "
                        f"delta_mm={_delta_mm:.4f} "
                        f"curr_target=[{_p_target_arr[0]:+.4f},"
                        f"{_p_target_arr[1]:+.4f},{_p_target_arr[2]:+.4f}] "
                        f"built_for=[{(_built[0] if _built is not None else 0.0):+.4f},"
                        f"{(_built[1] if _built is not None else 0.0):+.4f},"
                        f"{(_built[2] if _built is not None else 0.0):+.4f}] "
                        f"pwl_t_end={_pwl_t_end:.4f} "
                        f"dist_ee_to_pwl_target_mm={_dist_to_pwl_target_mm:.4f} "
                        f"pwl_is_finished={_is_finished} "
                        f"ee_now=[{ee_pos_now[0]:+.4f},{ee_pos_now[1]:+.4f},{ee_pos_now[2]:+.4f}] "
                        f"refresh_last_at={_refresh_at}",
                        flush=True,
                    )
                if _need_rebuild:
                    self._pwl_traj = RepositionTrajectory(
                        p_start=ee_pos_now,
                        p_target=_p_target_arr,
                        z_safe=float(
                            self.params.reposition_params.pwl_waypoint_height),
                        speed=float(self.params.reposition_params.pwl_speed),
                        t_start=_sim_t,
                        straight_line_thresh=float(self.params
                            .reposition_params
                            .use_straight_line_traj_under_piecewise_linear),
                    )
                    self._pwl_traj_built_for_target = _p_target_arr.copy()
                    self._pwl_traj_last_build_step = int(self._step)
                    print(
                        f"[STAGE-A-PWL] step={self._step} "
                        f"sim_t={_sim_t:.3f} build "
                        f"p_start=({ee_pos_now[0]:+.4f},"
                        f"{ee_pos_now[1]:+.4f},{ee_pos_now[2]:+.4f}) "
                        f"p_target=({_p_target_arr[0]:+.4f},"
                        f"{_p_target_arr[1]:+.4f},{_p_target_arr[2]:+.4f}) "
                        f"K={self._pwl_traj.knot_positions.shape[1]} "
                        f"t_end={self._pwl_traj.t_end:.3f}",
                        flush=True,
                    )
                _p_des, _v_des, _done = self._pwl_traj.eval(_sim_t)
                _p_ee_des = _p_des
                _v_ee_des = _v_des
                # §7.32 — FAITHFUL-DESIRED-STATE: static surface override
                # dropped (free / PWL path); the PWL trajectory's own
                # (_p_des, _v_des) is the free-mode analog of the
                # reference's repos_execution_lcm_traj_ (analytic
                # piecewise-linear path with its own velocity).
                u_imp, imp_diag = self.executor.compute_torque(
                    current_q, current_v, plant_ctx,
                    p_ee_desired = _p_ee_des,
                    v_ee_desired = _v_ee_des,
                    lambda_n     = None,
                    lambda_t     = None,
                    J_n          = None,
                    J_t          = None,
                    lambda_des   = None,
                    mode         = "free",  # §7.70 — keep port Kp/W_track
                )
                # 1 kHz OSC decoupling: cache PWL traj object so sub-tick can
                # re-evaluate the SAME piecewise-linear path at fresh t_sim.
                self._last_osc_call = ("osc_pwl_free", dict(
                    pwl_traj=self._pwl_traj,
                    mode="free",
                ))
            else:
                # Legacy free-mode path (unchanged).
                _p_des_wp = (free_diag.get("p_des")
                             if free_diag is not None else None)
                if _p_des_wp is not None:
                    _p_ee_des = _p_des_wp
                elif self._current_repos_target is not None:
                    _p_ee_des = self._current_repos_target
                else:
                    _p_ee_des = ee_pos_now
                # §7.32 — FAITHFUL-DESIRED-STATE: static surface override
                # dropped (free / legacy path); _p_ee_des stays as the IK
                # tracker's waypoint (or ee_pos_now if no waypoint). The
                # legacy free path passes v_ee_desired = None (target
                # velocity 0) — acceptable per §7.32 spec (free-mode
                # over-drive is not the failure mode; c3-mode penetration
                # is).
                u_imp, imp_diag = self.executor.compute_torque(
                    current_q, current_v, plant_ctx,
                    p_ee_desired = _p_ee_des,
                    v_ee_desired = None,
                    lambda_n     = None,
                    lambda_t     = None,
                    J_n          = None,
                    J_t          = None,
                    mode         = "free",  # §7.70 — keep port Kp/W_track
                )
                # 1 kHz OSC decoupling: legacy path has a static p_ee_des set
                # by the wrapper (last IK-tracker waypoint or ee_pos_now).
                # Cached verbatim; sub-tick replays with fresh state.
                self._last_osc_call = ("osc_direct_free", dict(
                    p_ee_desired=np.asarray(_p_ee_des, dtype=float).reshape(3).copy(),
                    v_ee_desired=None,
                    mode="free",
                ))
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
            _sim_t = (self._step - 1) * self._dt_ctrl
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
            # [STAGE-A-TRACE] purpose-built per-tick trace for the Stage A
            # bar parser (alignment plan §3 Stage A). One line per control
            # tick carrying all parser inputs:
            #   phi             — Drake signed-distance EE→box surface (m).
            #                     nan when no EE-BOX pair within 0.50 m
            #                     (typical of free mode away from contact).
            #   box_xy          — for goal_motion (informational, A→E
            #                     cumulative).
            #   lam_n_ee_box    — admitted EE-BOX normal force (nan in
            #                     free mode).
            #   qy, qz          — box-quaternion components (orientation
            #                     guard).
            #   finished_repos  — entry-gate candidate accounting.
            # Reads only — no behavior change.
            _lam_n_ee_box_trace = float("nan")
            if mode == "c3":
                _ci_tr = getattr(
                    self.base_mpc.formulator, "_last_contact_info", None)
                if (_ci_tr is not None and _lam_n is not None
                        and hasattr(_lam_n, "__len__") and len(_lam_n) > 0):
                    for _i_tr, _info_tr in enumerate(_ci_tr):
                        if (isinstance(_info_tr, dict)
                                and _info_tr.get("tag") == "EE-BOX"):
                            if len(_lam_n) > _i_tr:
                                _lam_n_ee_box_trace = float(_lam_n[_i_tr])
                            break
            print(
                f"[STAGE-A-TRACE] step={self._step} "
                f"sim_t={(self._step - 1) * self._dt_ctrl:.3f} "
                f"mode={mode} "
                f"phi={_phi_ge:.5f} "
                f"box_xy={float(current_q[self._obj_x_idx]):+.5f},"
                f"{float(current_q[self._obj_y_idx]):+.5f} "
                f"lam_n_ee_box={_lam_n_ee_box_trace:.4f} "
                f"qy={float(current_q[self._obj_qy]):+.5f} "
                f"qz={float(current_q[self._obj_qz]):+.5f} "
                f"finished_repos={int(bool(finished_repos))}",
                flush=True,
            )
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
            # Reset _c3_geom_z_target so the next c3 entry re-captures obj_z.
            # Prevents stale z-target if the object shifted vertically during
            # a c3 stint (e.g., box tilted then relaxed).
            self._c3_geom_z_target = None
            if self.log_diag:
                print(f"[RICH-EXIT-REFRESH] step={self._step} "
                      f"mode {self._prev_mode}->{mode} reason={reason.name} "
                      f"forcing buffer refresh + clearing prev_repos")
        self._prev_mode              = mode
        self.last_mode               = mode
        self.last_switch_reason      = reason
        self.last_winning_sample_idx = k_star
        self._step_times_ms.append((time.perf_counter() - t_step_start) * 1e3)

        return u_opt

    # ------------------------------------------------------------------
    # 1 kHz OSC decoupling — replay the executor on fresh state without
    # re-running the planner. Called from main.py between planner ticks.
    # Mirrors dairlib's LcmDrivenLoop where the OSC ticks at 1 kHz on the
    # latest planner-published trajectory. If the wrapper hasn't cached a
    # planner output yet (cold start), returns zeros.
    # ------------------------------------------------------------------
    def compute_control_osc_only(self,
                                 current_q: np.ndarray,
                                 current_v: np.ndarray,
                                 plant_ctx,
                                 t_sim: float) -> np.ndarray:
        if self._last_osc_call is None:
            n_u = int(getattr(self.executor, "n_arm", 7))
            return np.zeros(n_u)
        kind, kw = self._last_osc_call
        if kind == "c3_traj":
            u_opt, _ = self.executor.compute_torque_from_trajectory(
                traj=kw["traj"], t_sim=float(t_sim),
                current_q=current_q, current_v=current_v,
                plant_ctx=plant_ctx,
                v_ee_desired=kw["v_ee_desired"],
                lambda_n=kw["lambda_n"], lambda_t=kw["lambda_t"],
                J_n=kw["J_n"], J_t=kw["J_t"],
                lambda_des=kw["lambda_des"],
                a_ee_desired=kw["a_ee_desired"],
                mode=kw["mode"],
                q_nominal_override=kw["q_nominal_override"],
            )
            return u_opt
        if kind == "osc_pwl_free":
            _pwl = kw["pwl_traj"]
            if _pwl is None:
                n_u = int(getattr(self.executor, "n_arm", 7))
                return np.zeros(n_u)
            _p_des, _v_des, _done = _pwl.eval(float(t_sim))
            u_opt, _ = self.executor.compute_torque(
                current_q, current_v, plant_ctx,
                p_ee_desired=_p_des,
                v_ee_desired=_v_des,
                lambda_n=None, lambda_t=None, J_n=None, J_t=None,
                lambda_des=None,
                mode=kw["mode"],
            )
            return u_opt
        if kind == "osc_direct_free":
            u_opt, _ = self.executor.compute_torque(
                current_q, current_v, plant_ctx,
                p_ee_desired=kw["p_ee_desired"],
                v_ee_desired=kw["v_ee_desired"],
                lambda_n=None, lambda_t=None, J_n=None, J_t=None,
                lambda_des=None,
                mode=kw["mode"],
            )
            return u_opt
        if kind == "tracker":
            u_opt, _ = self.tracker.compute_torque(
                current_q=current_q, current_v=current_v,
                plant_ctx=plant_ctx,
                p_target=kw["p_target"],
                dt_osc=kw["dt_osc"],
                admit_active=kw["admit_active"],
            )
            return u_opt
        n_u = int(getattr(self.executor, "n_arm", 7))
        return np.zeros(n_u)

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
        sim_t = (step - 1) * self._dt_ctrl
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
