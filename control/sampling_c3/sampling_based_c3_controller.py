"""
SamplingC3Controller — top-level outer controller.

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
from control.sampling_c3.sample_buffer import (
    BufferedSample, SampleBuffer, UnsuccessfulSampleBuffer)
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


class SamplingC3Controller:
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
            # Pass dt_pose from base_mpc so per-sample cost-LCS builds use
            # the reference-conformant fine dt near-goal (ci_mpc_c3plus.py:76
            # holds dt_pose separately; wrapper syncs the crossed flag onto
            # quad_cost each tick).
            dt_pose=getattr(base_mpc, "dt_pose", self._dt),
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
        # Unsuccessful-sample buffer — reference generate_samples.cc:181-205
        # SampleAvoidsBadSpots + cc:2161-2205 AddToUnsuccessfulBuffer.
        # Populated at retarget events (reference: free→c3; port also:
        # kToBetterRepos).  When populated, filters generated samples to
        # avoid EE positions within `unsuccessful_radius` of any stored
        # bad spot.  See TODO #6 in docs/port-todo.md.
        _sp = params.sampling_params
        self.unsuccessful_buffer = UnsuccessfulSampleBuffer(
            capacity                    = int(getattr(
                _sp, "N_unsuccessful_sample_buffer", 20)),
            unsuccessful_radius         = float(getattr(
                _sp, "unsuccessful_radius", 0.05)),
            unsuccessful_pos_retention  = float(getattr(
                _sp, "unsuccessful_pos_error_sample_retention", 0.10)),
            unsuccessful_ang_retention  = float(getattr(
                _sp, "unsuccessful_ang_error_sample_retention", 0.60)),
        )
        self._avoid_unsuccessful = bool(getattr(
            _sp, "avoid_choosing_unsuccessful_samples", True))
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
        # (PORT_LCS_ALWAYS_ON_EE_BOX=1) — without it the removed proxy re-opens
        # the free-mode freeze trap.
        import os as _os_rec
        # §7.35 — feedforward-accel SUB-GATE (§7.34 banked, default-OFF).
        # Independent of the reconcile path; opts in to the a_ff leg of the
        # OSC PD-plus-feedforward law once the source is clean (planner
        # converges) or a mitigation lands.
        _env_ffa = _os_rec.environ.get("REF_RECONCILE_FEEDFORWARD_ACCEL", "")
        self._reconcile_feedforward_accel = (
            bool(int(_env_ffa)) if _env_ffa else False)

        # §7.55 — PORT_DISABLE_CONTACT_LOSS_GATE (default-OFF) skips the
        # CONTACT-LOSS-EXIT watchdog at _solve_plan() that forces c3→repos
        # after `_no_ee_box_streak ≥ contact_loss_threshold_*_s/dt_ctrl`
        # consecutive no-contact ticks. The reference (dairlib_sampling_c3
        # @257e3ed, systems/controllers/sampling_based_c3_controller.cc:1150-
        # 1184) has NO such watchdog — it exits c3 only via cost+hysteresis
        # or the cost-based progress timeout. With PORT_DISABLE_C3_OVERRIDE=1
        # (§7.51), the disengage threshold drops to contact_loss_threshold_
        # default_s = 5 ticks at 100 Hz, which bounces c3 out before contact
        # admits (§7.54 root-cause). This flag is SEPARATE from
        # PORT_DISABLE_C3_OVERRIDE so the effect of the watchdog removal is cleanly
        # attributable. Default-OFF byte-identical preserved.
        # Default flipped to disable the port-only watchdog by default —
        # the reference dispatcher has no such gate. Explicit re-enable
        # via PORT_DISABLE_CONTACT_LOSS_GATE=0 for legacy runs.
        self._disable_contact_loss_gate = (
            _os_rec.environ.get("PORT_DISABLE_CONTACT_LOSS_GATE", "1") == "1")

        # Force-tracking follows params.use_force_tracking (True by default).
        # The reconcile path no longer silently overrides it to False —
        # historical §7.31 silent-off + §7.63 decouple-workaround retired
        # with the flag removal.
        _use_force_tracking = bool(getattr(params, "use_force_tracking", True))
        if self._reconcile_feedforward_accel:
            print("[§7.35] REF_RECONCILE_FEEDFORWARD_ACCEL=1 → feedforward-"
                  "accel ENABLED (§7.34 OVER-DRIVES regime; source-conditional)",
                  flush=True)

        # §7.42 — when REFCONF_OSC_ALIGN=1, override W_force to 1.0 to match
        # the reference's `LambdaEndEffectorW = diag(1,1,1)` (osc_params.yaml:74).
        # The bundling env flag (set in main.py) wires the routing + u-bounds +
        # R-cost env vars together; W_force needs its own override here because
        # it is read at construction (not via env at use-site like the others).
        _ref_align = (os.environ.get("REFCONF_OSC_ALIGN", "0") == "1")
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
        # Reference cc:1858-1863 x_pred_curr_plan_ for repos mode. Holds
        # the planned EE position (row 0..2 of x_lcs) at the previous
        # planner tick's forward-simulated point in the trajectory.
        # Used by ClampEndEffectorAcceleration analog in the p_start
        # computation below to march the reposition trajectory forward
        # in prediction-space instead of restarting at the arm's actual
        # (lagging) position each tick.
        self._x_pred_repos_plan: Optional[np.ndarray] = None
        if self._use_pwl_traj:
            print("[STAGE-A-PWL] dispatcher: "
                  "use_reposition_pwl_trajectory=True", flush=True)

        # Mode state
        self.is_doing_c3 = start_in_c3_mode
        self._prev_mode:                str   = "c3" if start_in_c3_mode else "free"
        self._step:                     int   = 0
        # Reference `pursued_target_source_` state
        # (dairlib_sampling_c3/systems/controllers/sampling_based_c3_controller.h:505).
        # Purely telemetry — derived from the mode + best-sample-source label at
        # each mode-switch tick via `mode_switch.pursued_from_label`. Emitted in
        # the [GS] line.
        from control.sampling_c3.mode_switch import PursuedTargetSource as _PTS
        self._pursued_target_source: _PTS = _PTS.kNoTarget
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

        # Reference-conformant sample bookkeeping. Mirrors dairlib
        # `all_sample_locations_` / `all_sample_costs_` / `best_sample_index_`
        # (systems/controllers/sampling_based_c3_controller.cc:216, :957, :1842).
        # Rebuilt each tick after evaluate_samples. Index convention matches
        # _build_samples layout:
        #   [0]        = current EE
        #   [1]        = prev repos target (only when _current_repos_target set)
        #   [2..N+1]   = strategy samples
        #   back()     = buffered best when leaving c3 (conditional)
        # `labels` (parallel string vector) remains the source of truth for
        # slot identity; these members exist for reference-compat inspection.
        self._all_sample_locations:     list[np.ndarray]     = []
        self._all_sample_costs:         list[float]          = []
        self._best_sample_index:        Optional[int]        = None
        # Parallel labels list for the [EE-SAMPLES] diagnostic tag.
        self._last_sample_labels:       list[str]            = []

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
        """Generate fresh strategy samples every tick — reference conformant.

        Reference `GenerateSampleStates` (sampling_based_c3_controller.cc:900)
        fires every control tick with no caching. The port previously cached
        samples for `sample_buffer_lifetime_s` seconds as a stability
        workaround, but that produced sample-slot duplicates (chosen sample
        stayed in cache AND appeared as prev_repos next tick).

        2026-07-19 refactor: match reference. Cache-related state
        (_sample_buffer, _sample_buffer_age, _sample_buffer_n_strategy,
        _refresh_buffer_on_arrival) is now inert — kept for API surface
        stability but no longer read.
        """
        sp = self.params.sampling_params
        # Draw generously — up to 3× requested — so downstream filtering
        # against the unsuccessful-buffer can reject bad-spot samples
        # without leaving us short.  Reference generate_samples.cc:154 has
        # a retry loop; port draws once and filters at the end.
        _draw_n = int(n_strategy) * 3 if (
            self._avoid_unsuccessful and len(self.unsuccessful_buffer) > 0
        ) else int(n_strategy)
        _raw_samples = generate_samples(
            strategy  = sp.sampling_strategy,
            n_samples = _draw_n,
            obj_xy    = obj_xy,
            params    = sp,
            rng       = self._rng,
            g_hat     = g_hat,
            obj_quat  = obj_quat,
            yaw_delta = yaw_delta,
        )
        # Log ALL raw samples emitted by generate_samples before any
        # filtering — deep-log addition for direction-commit debugging.
        # Each sample gets a keep/reject tag so we can reconstruct why
        # certain faces of the object dominate the persisted target.
        if self.log_diag:
            _raw_str = ";".join(
                f"({p[0]:+.4f},{p[1]:+.4f},{p[2]:+.4f})"
                for p in _raw_samples)
            print(f"[SAMPLE-GEN] step={self._step} "
                  f"n_requested={int(n_strategy)} "
                  f"n_drawn={len(_raw_samples)} "
                  f"raw=[{_raw_str}]",
                  flush=True)
        # Apply the unsuccessful-buffer filter — reference
        # generate_samples.cc:181-205 SampleAvoidsBadSpots.  Only fires
        # when `avoid_choosing_unsuccessful_samples` is on AND the buffer
        # has entries.
        if self._avoid_unsuccessful and len(self.unsuccessful_buffer) > 0:
            _samples = []
            _rejected = []
            for s in _raw_samples:
                if self.unsuccessful_buffer.sample_avoids_bad_spots(s):
                    _samples.append(s)
                    if len(_samples) >= int(n_strategy):
                        break
                else:
                    _rejected.append(s)
            if self.log_diag and len(_rejected) > 0:
                _rej_str = ";".join(
                    f"({p[0]:+.4f},{p[1]:+.4f},{p[2]:+.4f})"
                    for p in _rejected)
                print(f"[UNSUCC-FILTER] step={self._step} "
                      f"rejected={len(_rejected)} kept={len(_samples)} "
                      f"buffer_size={len(self.unsuccessful_buffer)} "
                      f"rejected_pos=[{_rej_str}]",
                      flush=True)
        else:
            _samples = _raw_samples[:int(n_strategy)]
        if self.log_diag:
            _tup = [tuple(np.round(s, 4).tolist()) for s in _samples]
            print(f"[PERSIST] step={self._step} refresh "
                  f"n_strategy={n_strategy} samples={_tup}")
        return _samples

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
                              g_hat_3d: np.ndarray,
                              plant_ctx=None) -> np.ndarray:
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
        # PORT_FORCE_ROUTING=u_sol AND the planner is EE-space, return the
        # planner's solved u[0] directly (with a one-shot direction-confirm
        # print so we can verify sign on first c3 entry). PORT_FORCE_ROUTING
        # unset or 'off' → falls through to the legacy -g_hat path below
        # (bit-identical to pre-prototype). 'neg_u_sol' returns -u_seq[0]
        # for the opposite sign convention.
        # Reference c3-mode force target = u_sol (planner's raw output),
        # QP-bounded by u_horizontal_limits=[-10,10] N, u_vertical=[-3,3] N.
        # See sampling_based_c3_controller.cc:1820-1832 (reference).
        #
        # EE-space path (n_u=3): use u_sol[0] directly via
        # PORT_FORCE_ROUTING=u_sol. R^7 path: recovery via pinv(J^T)
        # doesn't yield a clean Cartesian direction (verified — box
        # tumbled 180°), so R^7 falls through to the fabricated -g_hat
        # path below. Default OFF (fabricated path bit-identical to
        # pre-prototype).
        import os as _os
        _fr = _os.environ.get("PORT_FORCE_ROUTING", "off").lower()
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
            # R^7 path: recover Cartesian force from joint torque via
            # pinv(J^T). Requires current plant_ctx which we access via
            # the base_mpc formulator's cached last-tick state.
            if _u_seq is not None and hasattr(_u_seq, "shape") \
                    and _u_seq.ndim == 2 and _u_seq.shape[1] == 7 \
                    and plant_ctx is not None:
                try:
                    from pydrake.multibody.tree import JacobianWrtVariable as _JWV
                    _J = self.plant.CalcJacobianTranslationalVelocity(
                        plant_ctx, _JWV.kV, self.ee_frame,
                        np.zeros(3), self.world_frame, self.world_frame,
                    )   # (3, n_v)
                    _n_u_arm = self.n_u
                    _J_arm = _J[:, :_n_u_arm]        # (3, n_arm)
                    _u7 = np.asarray(_u_seq[0], dtype=float).reshape(7)
                    # The R^7 planner u-box is GRAVITY-CENTERED (37f7573):
                    # u_sol contains the gravity-holding torque
                    # u_hold = −τ_g_arm. The reference's u is a pure EE
                    # push force (gravity absent by construction), so the
                    # Cartesian push intent is u_sol − u_hold; mapping the
                    # raw u_sol recovers the ~34 Nm gravity hold as a
                    # spurious Cartesian force on top of the push.
                    _tau_g = np.asarray(
                        self.plant.CalcGravityGeneralizedForces(plant_ctx),
                        dtype=float)
                    _u_push = _u7 - (-_tau_g[:_n_u_arm])
                    # min ‖J^T f − u‖ → f = (J J^T)^-1 J u
                    _JJT = _J_arm @ _J_arm.T          # (3, 3)
                    _f_ee = np.linalg.solve(
                        _JJT + 1e-6*np.eye(3), _J_arm @ _u_push)  # (3,)
                    if _fr == "neg_u_sol":
                        _f_ee = -_f_ee
                    # Reference u_horizontal_limits=[-10,10], u_vertical=[-3,3]
                    # (sampling_c3plus_options.yaml lines 34-35). Reference
                    # planner enforces these as QP bounds so u_sol respects
                    # them naturally. Port R^7 planner has separate joint
                    # torque limits, so we re-apply the reference EE-force
                    # bounds on the recovered f_ee.
                    _f_ee[0] = float(np.clip(_f_ee[0], -10.0, 10.0))
                    _f_ee[1] = float(np.clip(_f_ee[1], -10.0, 10.0))
                    _f_ee[2] = float(np.clip(_f_ee[2],  -3.0,  3.0))
                    if not getattr(self, "_force_route_logged", False):
                        _nm = max(1e-9, float(np.linalg.norm(_f_ee)))
                        print(f"[FORCE-ROUTE] R^7 recovery env={_fr} "
                              f"|f_ee|={float(np.linalg.norm(_f_ee)):.3f}N "
                              f"dir=({_f_ee[0]/_nm:.2f},"
                              f"{_f_ee[1]/_nm:.2f},{_f_ee[2]/_nm:.2f}) "
                              f"[clamped ±10/±10/±3 N]",
                              flush=True)
                        self._force_route_logged = True
                    return _f_ee
                except Exception:
                    pass  # fall through to fabricated path

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

        # 2026-07-19 reference-conformant refactor
        # (sampling_based_c3_controller.cc:898-938):
        #   candidate_states = GenerateSampleStates(...)         # fresh
        #   if (!is_doing_c3_ && !in_collision):
        #       candidate_states.insert(begin, prev_repos)
        #   candidate_states.insert(begin, current)
        # The buffer slot is added SEPARATELY by AugmentSamplesWithBuffer
        # (cc:2106) only when currently in c3 mode; that's now handled by
        # _augment_samples_with_buffer, invoked by the caller AFTER cost
        # evaluation. This method returns only the pre-cost sample bank.
        positions: list[np.ndarray] = []
        labels:    list[str]        = []

        # prev_repos only in REPOS mode (matches reference's !is_doing_c3_).
        # Reference also checks !in_collision; port skips that here — the
        # workspace filter in generate_samples handles collision-avoidance for
        # freshly generated samples, and prev_repos is a previously-cleared
        # target so unlikely to be in collision.
        # Achieved-fixed-goal exception: skip prev_repos add. Reference
        # (cc:928-936) still inserts prev_repos, but its cost fn is
        # c3_cost + travel_cost_per_meter × xy_travel only, so a far retreat
        # eventually wins on structural grounds. The port's cost fn adds a
        # large `align_bonus = w_align × align_score` term (inner_solve.py:762,
        # w_align default 30000) that biases heavily toward near-box samples;
        # if prev_repos (near-object, high align) stays in the candidate set
        # once retreat samples appear, best_other selects prev_repos and the
        # arm never migrates. Skipping prev_repos here + replacing strategy
        # samples with retreat copies below makes the sample list
        # [current, retreat, retreat, ...] so best_other = retreat.
        if (prev_mode == "free" and self._current_repos_target is not None
                and not getattr(self, "_achieved_fixed_goal", False)):
            positions.append(self._current_repos_target.copy())
            labels.append("prev_repos")

        # Strategy samples — always fresh, never cached (matches reference).
        n_strategy = (sp.num_additional_samples_c3 if prev_mode == "c3"
                      else sp.num_additional_samples_repos)
        # Reference sampling_based_c3_controller.cc:887-897: when
        # achieved_fixed_goal_ is set, candidate_states is populated with
        # N identical copies of a fixed retreat position (reference literal
        # `head(3) << 0.3, 0.4, 0.1`) INSTEAD of calling GenerateSampleStates,
        # so the reposition tracker drives the arm off the object.
        # Port previously called `_get_persistent_samples` unconditionally,
        # so post-latch samples remained clustered around the object; the
        # reposition IK kept its target adjacent to the object and micro-
        # motions nudged it back out of the tight-goal thresholds
        # (observed in results/tight_goal_p8_dtpose_180s.txt:
        # latch at step 1238 with dist=0.0198m rot_err=0.0103rad;
        # drift to dist=0.035m rot_err=0.206rad by step 1800).
        if getattr(self, "_achieved_fixed_goal", False):
            retreat = np.asarray(sp.retreat_ee_position, dtype=float)
            for i in range(n_strategy):
                positions.append(retreat.copy())
                labels.append(f"strat_{i}")
        else:
            strategy_samples = self._get_persistent_samples(
                obj_xy=obj_xy, g_hat=g_hat, n_strategy=n_strategy,
                obj_quat=obj_quat, yaw_delta=yaw_delta)
            for i, p in enumerate(strategy_samples):
                positions.append(p)
                labels.append(f"strat_{i}")

        # Insert current EE at the front (reference cc:938).
        positions.insert(0, ee_pos_now.copy())
        labels.insert(0, "current")

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

        # 2026-07-19: reference-conformant bulk-add.  Reference
        # MaintainSampleBuffers (sampling_based_c3_controller.cc:2050-
        # 2075) appends ALL non-current, non-repos-target samples per
        # tick.  Prior single-best-append (d453709) was a workaround
        # for the port's earlier broken buffer params (50 mm radius,
        # 100 mm retention); those made bulk-add pollute the buffer
        # with samples that persisted across huge object motions.  With
        # the corrected reference params (10 mm radius, 6 mm retention
        # — see e406072), the buffer prunes old entries as the object
        # moves, so bulk-add gives dense per-pose sampling coverage
        # matching reference.
        #
        # Reference skips index 0 (current) always, plus index 1
        # (prev_repos_target) if in repos mode and the size check
        # confirms it's the pursued target.  Port maps index 0 = current
        # and index 1 = prev_repos when present (see _build_samples).
        _sp = self.params.sampling_params
        _skip_prev_repos = (
            self._prev_mode == "free"
            and len(results) == int(_sp.num_additional_samples_repos) + 2)
        for k, r in enumerate(results):
            if k == 0:                       # always skip current
                continue
            if k == 1 and _skip_prev_repos:  # skip prev_repos in repos mode
                continue
            if not r.feasible:
                continue
            self.buffer.append(BufferedSample(
                position   = r.sample_pos.copy(),
                cost       = r.c_sample,
                obj_pos_xy = obj_xy_now.copy(),
                obj_quat   = obj_quat.copy(),
                result     = r,   # carry the SampleResult so
                                  # AugmentSamplesWithBuffer can re-inject
                                  # this sample with its full plan.
            ))
        # Deep-log: periodic dump of both buffers (successful sample-buffer
        # + unsuccessful bad-spots buffer) so post-hoc analysis can trace
        # when entries accumulate/prune vs when the dispatcher's choices
        # flip direction.
        if self.log_diag and self._step <= 30:
            _sb_entries = ";".join(
                f"({s.position[0]:+.4f},{s.position[1]:+.4f},"
                f"{s.position[2]:+.4f}|c={s.cost:.1f}|age={s.age_steps})"
                for s in self.buffer)
            _ub_entries = ";".join(
                f"({s.position[0]:+.4f},{s.position[1]:+.4f},"
                f"{s.position[2]:+.4f})"
                for s in self.unsuccessful_buffer)
            print(f"[BUFFER-STATE] step={self._step} "
                  f"sample_buffer={len(self.buffer)}=[{_sb_entries}] "
                  f"unsucc_buffer={len(self.unsuccessful_buffer)}="
                  f"[{_ub_entries}]",
                  flush=True)

    # ------------------------------------------------------------------
    # Main control entry
    # ------------------------------------------------------------------

    def compute_control(self,
                        current_q:  np.ndarray,
                        current_v:  np.ndarray,
                        plant_ctx,
                        target_xy:  np.ndarray,
                        target_yaw: float = 0.0,
                        final_target_xy: Optional[np.ndarray] = None) -> np.ndarray:
        """T-architecture Stage 2b: gate the planner solve on dt_mpc
        boundaries, run the OSC every tick at dt_osc, and index the
        planner's λ-horizon by elapsed time since the last solve. At
        default rates (dt_osc == dt_mpc == 0.01s) N_plan=1, the planner
        fires every tick with elapsed=0, and the system behaves like
        pre-Stage-2b within the sim's noise floor."""
        self._step += 1
        t_step_start = time.perf_counter()

        # Gating: how many OSC ticks per planner solve?
        # main.py decouples OSC via compute_control_osc_only (1 kHz sub-tick
        # loop), so compute_control fires ONLY at planner rate. Each call
        # should trigger a fresh solve — force N_plan=1 regardless of the
        # dt_mpc/dt_osc YAML values (which correctly describe the rates but
        # would confuse this counter, since self._step counts compute_control
        # calls = planner ticks, NOT OSC ticks).
        # Pre-fix: N_plan=75 (dt_mpc=0.075 / dt_osc=0.001) made
        # should_solve fire every 75 planner ticks = every 5.6 s wall; only
        # 2 solves per 8 s run.
        N_plan = 1
        should_solve = (self._last_plan_tick < 0
                        or (self._step - self._last_plan_tick) >= N_plan)

        if should_solve:
            plan_ctx = self._solve_plan(current_q, current_v, plant_ctx,
                                        target_xy, target_yaw,
                                        final_target_xy=final_target_xy)
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
                # Twist-prediction probe (p113 rot diagnosis): planner-
                # predicted box yaw at knot 1 and terminal knot vs current.
                # Answers whether the LCS predicts the CCW twist the real
                # push produces (model gap) or predicts it and accepts it
                # (cost balance). qz≈sin(yaw/2) for upright box.
                _qzc = float(current_q[_ps + 3])
                _qz1 = float(x1[_ps + 3])
                _qzN = float(_xseq[-1][_ps + 3])
                print(
                    f"[X-SEQ-PROBE] step={self._step} "
                    f"x0_box=({current_q[_ox]:+.5f},{current_q[_oy]:+.5f},{current_q[_oz]:+.5f}) "
                    f"x1_box=({x1[_ox]:+.5f},{x1[_oy]:+.5f},{x1[_oz]:+.5f}) "
                    f"x1_box_v=({x1[_vxr]:+.5f},{x1[_vyr]:+.5f},{x1[_vzr]:+.5f}) "
                    f"qz_now={_qzc:+.5f} qz_pred1={_qz1:+.5f} "
                    f"qz_predN={_qzN:+.5f} "
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
                    target_yaw: float = 0.0,
                    final_target_xy: Optional[np.ndarray] = None) -> dict:
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

        # Reference-conformance: distance to FINAL goal for progress tracking
        # and achieved_fixed_goal check. target_xy here is the LOOKAHEAD
        # sub-goal (main.py:746), which spikes back to 0.15 m each time box
        # reaches the sub-goal. Progress tracker fed with sub-goal distance
        # sees false regression. Reference uses x_lcs_final_des (absolute
        # goal). Fall back to target_xy if final not passed (legacy callers).
        if final_target_xy is not None:
            _final_target = np.asarray(final_target_xy, dtype=float).reshape(2)
            _v_final = _final_target - obj_xy
            _final_goal_dist = float(np.linalg.norm(_v_final))
        else:
            _final_goal_dist = goal_dist

        ee_pos_now = self.plant.CalcPointsPositions(
            plant_ctx, self.ee_frame, np.zeros(3), self.world_frame,
        ).flatten().copy()

        # D.3 (2026-07-13) — shortest-angle yaw delta from current to
        # target. Feeds the yaw-torque face-selection bias in the T
        # sampler. For the box (target_yaw = 0 and box stays flat),
        # yaw_delta stays near 0 and the sampler skips the bias entirely
        # (see sampling.py:_face_normal_projection D.3 block).
        # Was `2.0 * atan2(qz, qw)` — a small-tilt approximation that
        # only equals the world-z yaw when the box is axis-aligned. For
        # a tipped box (qy≠0), it can be off by ~30-90°, feeding a wrong
        # yaw_delta to the sampler's D.3 face-selection bias. Use the
        # standard quaternion-to-yaw extraction instead:
        #   yaw = atan2(2·(qw·qz + qx·qy), 1 - 2·(qy² + qz²))
        _qw, _qx, _qy, _qz = (float(obj_quat[0]), float(obj_quat[1]),
                              float(obj_quat[2]), float(obj_quat[3]))
        _yaw_now_samp = float(np.arctan2(
            2.0 * (_qw * _qz + _qx * _qy),
            1.0 - 2.0 * (_qy * _qy + _qz * _qz),
        ))
        _dy_samp = float(target_yaw) - _yaw_now_samp
        yaw_delta_samp = float(
            np.arctan2(np.sin(_dy_samp), np.cos(_dy_samp)))

        # 2. Build sample list (k=0 = current EE always first)
        samples, labels = self._build_samples(
            ee_pos_now, obj_xy, g_hat, self._prev_mode,
            obj_quat=obj_quat, yaw_delta=yaw_delta_samp)

        # Reference-conformant bookkeeping (dairlib
        # sampling_based_c3_controller.cc:945-949). `samples` == candidate EE
        # 3-vectors == `all_sample_locations_` in the reference.
        self._all_sample_locations = samples
        # Parallel labels for the [EE-SAMPLES] diagnostic tag.
        self._last_sample_labels = labels

        # E2 surrogate-side x_pred clamp — pass base_mpc's cached predicted
        # EE state (x_pred_curr_plan from prev tick's primary solve) so the
        # k=0 surrogate can clamp its x0.p_ee too. Default OFF (env-gated
        # on inner_solve side via REFCONF_C3_SURROGATE_XPRED_CLAMP); this
        # setter always fires so enabling the env var is enough.
        _xpred_plan = getattr(self.base_mpc, "_x_pred_curr_plan", None)
        _nom_accel  = float(getattr(self.base_mpc, "nominal_ee_accel", 0.0))
        _fst        = float(getattr(self.base_mpc, "_filtered_solve_time",
                                    getattr(self.base_mpc, "dt", 0.05)))
        if _xpred_plan is not None and _nom_accel > 0.0:
            _dt_c = min(0.1, _fst)
            _delta_pos_e2 = _nom_accel * _dt_c * _dt_c
            _EE_POS_SLICE = getattr(self.base_mpc, "_EE_POS_SLICE",
                                    slice(7, 10))
            self.inner_solver.set_ee_pos_clamp(
                _xpred_plan[_EE_POS_SLICE], _delta_pos_e2)
        else:
            self.inner_solver.set_ee_pos_clamp(None, 0.0)

        # 3. Evaluate every sample (per-sample C3 + alignment + travel)
        results = self.inner_solver.evaluate_samples(
            samples=samples,
            current_q=current_q, current_v=current_v,
            plant_ctx=plant_ctx, target_xy=target_xy,
            ee_pos_now=ee_pos_now, g_hat_3d=g_hat_3d,
            target_yaw=target_yaw,
        )
        # 2026-07-26 arc-2 E3 fix (unify progress + mode-switch cost signal).
        # Reference `KeepTrackOfC3ModeProgress` (cc:2208-2305) reads
        # `all_sample_costs_[kCurrentLocation]` which is the same signal that
        # drives the mode gate. Port previously fed the tracker `c_C3_raw`
        # (see later comment for rationale: c_sample can go negative with
        # w_align>0) but drove the gate with `c_samples[k]` (bonus-adjusted).
        # Unify by seeding c_samples from `c_C3_raw` when the env-gate is on.
        # Downstream inflations (finished_reposition_cost, buffer append) then
        # apply UNIFORMLY to both progress and gate signals — no split.
        # Env: REFCONF_MODE_SWITCH_USE_C3_RAW=1 (default ON = reference-conformant).
        # In current push_t config (w_align=w_travel=w_rot=0) c_C3_raw equals
        # r.c_sample so this is a no-op; the change is defensive.
        import os as _os_e3
        _use_c3_raw_for_gate = (
            _os_e3.environ.get("REFCONF_MODE_SWITCH_USE_C3_RAW", "1") == "1")
        if _use_c3_raw_for_gate:
            c_samples = [float(r.c_C3_raw) for r in results]
        else:
            c_samples = [r.c_sample for r in results]

        # 3b. Finished-reposition cost penalty.  Mirrors the reference
        #     (sampling_based_c3_controller.cc:1081-1084):
        #         if (i == kCurrentReposTarget && finished_reposition_flag_) {
        #             all_sample_costs_[i] += finished_reposition_cost;
        #             finished_reposition_flag_ = false;   // RESET after use
        #         }
        #     The flag is RESET after being applied — inflation fires ONCE
        #     per finished-reposition event. Port previously left the flag
        #     True, so cost was inflated EVERY tick — kToC3ReachedReposTarget
        #     kept re-firing (or blocking) as long as the arm hovered near
        #     the target.
        if (self._last_repos_finished
                and len(labels) > 1
                and labels[1] == "prev_repos"):
            c_samples[1] = (c_samples[1]
                            + self.params.progress_params.finished_reposition_cost)
            self._last_repos_finished = False   # match reference reset

        # 3c. AugmentSamplesWithBuffer — reference cc:2106-2158.
        # Re-enabled 2026-07-19 after BufferedSample now carries the full
        # SampleResult. Only fires when currently in c3 mode. Adds the
        # best buffer sample unless it's a cost+position exact match
        # (1e-5 tolerance) with the current best — matches reference's
        # cc:2141-2146 check.
        sp = self.params.sampling_params
        if (self._prev_mode == "c3"
                and sp.consider_best_buffer_sample_when_leaving_c3
                and len(self.buffer) > 0):
            _best_buf = self.buffer.best_with_position()
            if _best_buf is not None and _best_buf.result is not None:
                _lowest_new_cost = float(min(c_samples))
                _best_new_idx = int(np.argmin(c_samples))
                _best_new_pos = np.asarray(samples[_best_new_idx])
                _buf_pos = np.asarray(_best_buf.position)
                _cost_match = abs(_best_buf.cost - _lowest_new_cost) < 1e-5
                _pos_match  = np.linalg.norm(_buf_pos - _best_new_pos) < 1e-5
                if not (_cost_match and _pos_match):
                    samples.append(_buf_pos)
                    labels.append("buffer")
                    c_samples.append(float(_best_buf.cost))
                    results.append(_best_buf.result)
            # Refresh mirrored bookkeeping.
            self._all_sample_locations = samples
            self._last_sample_labels = labels

        # 4. Pick winner (k* = argmin c_sample over all samples)
        k_star = int(np.argmin(c_samples))

        # Reference-conformant bookkeeping (dairlib
        # sampling_based_c3_controller.cc:957, :1842). `c_samples` is the raw
        # argmin winner; the c3/free mode gate below may promote a different
        # slot to `target_idx`, at which point `_best_sample_index` is
        # overwritten to match (see :2343-nearby edit).
        self._all_sample_costs  = c_samples
        self._best_sample_index = k_star

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

        # E3 unification above (line ~1416) already seeds c_samples from
        # c_C3_raw when enabled, so downstream mode-switch reads the same
        # signal as the progress tracker.
        c_curr   = c_samples[0]
        best_other_idx = None
        best_other_cost = float("inf")
        for k in range(1, len(c_samples)):
            if c_samples[k] < best_other_cost:
                best_other_cost = c_samples[k]
                best_other_idx  = k

        # 5. Update progress tracker (uses k=0 cost = c_curr)
        # Reference sampling_based_c3_controller.cc:2228-2232:
        #   curr_pos_and_rot_cost = err^T · Q_block(7x7) · err
        # where err spans [quat(4), pos(3)] for the object.
        # Port computed only xy pos component. For T (yaw-goal task with
        # w_yaw=800 dominant), yaw progress was invisible to
        # ProgressTracker's kConfigCostDrop metric. Now includes yaw
        # cost so kConfigCostDrop tracks the reference's full pos+rot.
        w_obj_xy = self._quad_cost.w_obj_xy
        w_yaw    = float(getattr(self._quad_cost, "w_yaw", 0.0))
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

        # Include yaw component in config_cost (reference full Q_block covers
        # quat too). Uses the half-angle metric that w_yaw multiplies:
        # sin((ψ-α)/2). For box, w_yaw=0 in tasks.yaml so contribution=0.
        # NOTE: Uses _final_goal_dist (distance to FINAL goal), NOT goal_dist
        # (which is distance to LOOKAHEAD sub-goal — spikes as sub-goal
        # advances). Reference progress tracker feeds x_lcs_final_des cost.
        _half_yaw_err = np.sin(0.5 * (rot_error_now))   # >= 0
        config_cost_now = (w_obj_xy * (_final_goal_dist ** 2)
                           + w_yaw * (_half_yaw_err ** 2))

        # Reference sampling_based_c3_controller.cc:1150-1157 calls
        # KeepTrackOfC3ModeProgress ONLY when is_doing_c3_ == true.
        # Port previously updated progress tracker every tick (both modes),
        # polluting the history with free-mode ticks. The metric-drop
        # test (kConfigCostDrop) is designed to detect C3-mode stalls;
        # free-mode ticks skew the front/back window.
        if self._prev_mode == "c3":
            # Feed progress tracker the pure C3 quadratic cost, not the
            # ranking score. Reference sampling_based_c3_controller.cc:2236-2240
            # uses all_sample_costs_[kCurrentLocation] which for the reference
            # is `c3_cost + travel_cost_per_meter * travel_dist(=0 for current)`
            # = pure c3_cost, always ≥ 0. Port previously used c_curr (=
            # c_samples[0] = c_C3_raw − w_align·align_score − w_rot·rot_score +
            # w_travel·travel), which can go NEGATIVE (see w_align=30 000 in
            # config/sampling_c3_kik.yaml — load-bearing for argmin ranking but
            # pollutes progress tracking). Bonus fluctuation from align_score
            # kept resetting steps_since_c3_improve so met_progress=Y fired
            # every tick regardless of physical box motion — the 30 s/60 s
            # stall analysis showed 17/57 mode transitions with the box stuck
            # at x=0.083. Use c_C3_raw (pure quadratic, always ≥ 0) here; the
            # bonus survives everywhere it matters for sample selection.
            self.progress.update(StepMetrics(
                c3_cost     = results[0].c_C3_raw,
                config_cost = config_cost_now,
                pos_error   = _final_goal_dist,
                rot_error   = rot_error_now,
            ))

        # Reference achieved_fixed_goal_ (sampling_based_c3_controller.cc:887-897):
        # when the object is at goal (position_success_threshold=0.02 m
        # per goal_params.yaml:10), the controller pins samples at a
        # "get out of the way" position and skips c3 mode.
        # Port had no equivalent — arm kept pushing past goal and box
        # overshot by 260mm (seed=0 60s: box (+0.30, 0) → (+0.561, +0.029)).
        # Simple port version: track a sticky _achieved_fixed_goal flag
        # that latches when goal_dist and rot_error both under thresholds.
        # When set, force mode=free (arm won't push more).
        if not hasattr(self, "_achieved_fixed_goal"):
            self._achieved_fixed_goal = False
        _pos_thr = 0.02  # reference position_success_threshold
        _rot_thr = 0.10  # reference orientation_success_threshold
        if not self._achieved_fixed_goal:
            # Reference cc:876-881 object_on_target:
            #   crossed_ mode:   pos < pos_thr AND rot < rot_thr
            #   !crossed_ mode:  pos < pos_thr only  (rot ignored)
            # Port previously required both always, which prevented latch
            # for a tumbled box that reached position but had large rot_err.
            _on_target = (_final_goal_dist < _pos_thr
                          and (not self._crossed_switching_threshold
                               or rot_error_now < _rot_thr))
            if _on_target:
                self._achieved_fixed_goal = True
                if self.log_diag:
                    print(f"[ACHIEVED-FIXED-GOAL] step={self._step} "
                          f"final_goal_dist={_final_goal_dist:.4f}m rot_err={rot_error_now:.4f}rad "
                          f"crossed={self._crossed_switching_threshold} "
                          f"— pinning free mode to prevent overshoot",
                          flush=True)

        # 6. Mode-switch decision
        # Reference sampling_based_c3_controller.cc:800,821-830:
        # crossed_cost_switching_threshold_ is STICKY — set True once
        # pose_diff < threshold, and stays True until goal changes.
        # Port previously re-evaluated near_goal each tick, causing
        # hysteresis-variant thrashing near the threshold. Now tracked as
        # persistent state that latches on first crossing.
        if not hasattr(self, "_crossed_switching_threshold"):
            self._crossed_switching_threshold = False
        _cost_switch_thr = self.params.progress_params.cost_switching_threshold_distance
        # Reference cc:821-830 uses `x_lcs_final_des` (FINAL goal), not the
        # per-tick lookahead sub-goal. Port previously used goal_dist here
        # (lookahead sub-goal distance) — same bug as the progress-tracker
        # feed I fixed in bf2828b. Use _final_goal_dist for the crossed check.
        if not self._crossed_switching_threshold and _final_goal_dist < _cost_switch_thr:
            self._crossed_switching_threshold = True
            # Reference cc:845-847: reset progress metrics when threshold
            # is crossed while doing c3. Pose regime uses different cost
            # scaling than position regime, so stale position-regime
            # progress history would trigger spurious kToReposUnproductive.
            if self._prev_mode == "c3":
                self.progress.reset()
            if self.log_diag:
                print(f"[CROSSED-COST-THRESHOLD] step={self._step} "
                      f"final_goal_dist={_final_goal_dist:.4f}m < {_cost_switch_thr:.4f}m — "
                      f"switching to pose-tracking (non-_position) hyst variants"
                      f"{' + progress reset' if self._prev_mode == 'c3' else ''}",
                      flush=True)
        near_goal = self._crossed_switching_threshold
        # Sync near-goal flag to quad_cost so build_ee_space activates the
        # reference near-goal hessian-quaternion Q_block (task_costs.py
        # `use_quaternion_dependent_cost` path, mirroring reference
        # sampling_based_c3_controller.cc:1517-1570). The base_mpc call
        # below re-reads this attribute per tick.
        try:
            self._quad_cost._crossed_switching_threshold = bool(near_goal)
        except AttributeError:
            pass
        # Also sync to the base MPC so ci_mpc_c3plus.compute_control can
        # swap planning_dt (dt_pose) and per-axis u-limits (POSE env
        # vars) when the box is inside cost_switching_threshold_distance.
        try:
            self.base_mpc._crossed_switching_threshold = bool(near_goal)
        except AttributeError:
            pass
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
            # 2026-07-19: reference-conformant finished_reposition_flag.
            # Reference reposition.cc sets the flag at BUILD time inside the
            # walking-loop: fires when total travel time <= 1 planner tick
            # (single step_size covers the whole path). Port previously used
            # an arm-distance predicate `is_finished(sim_t, ee_now, tol)`
            # which held FALSE indefinitely when the arm hovered a few mm
            # outside 20-mm tol — kToC3ReachedReposTarget never fired.
            # The new attribute reads the reference build-time flag.
            # Requires _need_rebuild=True every tick so the flag reflects
            # the CURRENT p_start=ee_now (see also the rebuild-every-tick
            # fix at the RepositionTrajectory build site).
            _flag_build = bool(getattr(
                self._pwl_traj, "finished_reposition_flag", False))
            # 2026-07-19 (T-push tight-goal debug): T stall analysis
            # (results/push_t_near_goal_Q_180s.txt, 0/1800 c3-mode steps)
            # showed the build-time flag NEVER fires in 180 s. Root cause:
            # rebuild every tick with p_start=ee_now toward a same-tick
            # sample target ~5-15 cm away, so `(t_end-t_start) > dt_plan`
            # always — flag semantically requires arm to be within
            # `pwl_speed*dt_plan` ≈ 1.35 cm at BUILD time.
            #
            # Comment above already flags the underlying issue: "the T-shape
            # setback is inside the vertical bar's sphere-swept envelope
            # (BUG 2), so physical descent stalls with ‖p_target − ee_now‖
            # ≈ 20 mm rather than 5 mm — the trajectory is effectively
            # arrived but the physical residual holds above 5 mm
            # indefinitely." Restore the euclidean predicate as an OR-fallback
            # at the same 20 mm tolerance the IK tracker uses at
            # reposition_ik.py:1299 (`finished` = ‖p_target − ee_now‖ ≤ 0.02).
            # OR'd, not replacing: preserves reference build-time semantics
            # when the arm genuinely finishes the path fast, adds a physical-
            # arrival trigger for the T-collision-stall case.
            _tol_arr    = 0.020
            _p_target   = np.asarray(self._pwl_traj.p_target)
            _euclid_arr = (float(np.linalg.norm(_p_target - ee_pos_now))
                           <= _tol_arr)
            finished_repos = _flag_build or _euclid_arr
            if self.log_diag and _euclid_arr and not _flag_build:
                _d_now = float(np.linalg.norm(_p_target - ee_pos_now))
                print(f"[REPOS-ARR-EUCLID] step={self._step} "
                      f"‖p_target−ee‖={_d_now*1000:.1f}mm "
                      f"≤ tol={_tol_arr*1000:.1f}mm — "
                      f"finished_repos=True via euclidean fallback "
                      f"(build_flag={_flag_build})", flush=True)

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
        # Reference sampling_based_c3_controller.cc:1290-1293 applies this
        # gate for ALL objects. Retested twice for box (once earlier, once
        # after achieved_fixed_goal + cost-drop-fraction + hyst-inversion +
        # W_posture + a_ee_cap fixes) — box still regresses when gate is
        # enabled (goal_dist 0.109→0.374 → 0.602 m, orient 92°→180° tumble).
        # Kept tshape-only. Port divergence documented.
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

        # Reference cc:1148-1149: repos_target_cost = all_sample_costs_[
        # kCurrentReposTarget=1]. Cost is FRESHLY evaluated on THIS tick
        # for the slot 1 = prev-repos-target position. Port previously
        # passed `self._current_repos_cost` which was stored from the
        # PREVIOUS tick's target selection — stale relative to current
        # arm position. Fresh c_samples[1] matches reference semantics.
        if len(labels) > 1 and labels[1] == "prev_repos":
            _fresh_repos_cost = float(c_samples[1])
        else:
            _fresh_repos_cost = self._current_repos_cost  # fallback
        mode, reason = decide_mode(
            prev_mode          = self._prev_mode,
            c3_cost            = c_curr,
            best_other_cost    = best_other_cost,
            current_repos_cost = _fresh_repos_cost,
            met_progress       = met,
            near_goal          = near_goal,
            finished_repos     = finished_repos,
            params             = self.params.progress_params,
            ee_z_gate_pass     = _ee_z_gate_pass,
        )
        # 2026-07-22: expose the applied hysteresis margin so [GS] can
        # print the decision arithmetic (best_other + gap vs curr_cost)
        # rather than just the raw costs. Answers "why is a cheaper raw
        # sample rejected" without requiring re-derivation from yaml.
        from control.sampling_c3.mode_switch import _hysteresis as _hyst_fn
        if self._prev_mode == "c3":
            self._last_hyst_kind = "c3_to_repos"
            self._last_hyst_ref  = float(c_curr)
        else:
            self._last_hyst_kind = "repos_to_c3"
            self._last_hyst_ref  = (float(best_other_cost)
                                     if best_other_cost != float("inf")
                                     else float(c_curr))
        self._last_hyst_gap  = float(_hyst_fn(
            self.params.progress_params, self._last_hyst_kind,
            near_goal, self._last_hyst_ref))
        self._last_near_goal = bool(near_goal)

        # Reference achieved_fixed_goal_ override (cc:1160-1164, 1279-1283):
        # if goal met, force mode=free so arm stops pushing.
        if self._achieved_fixed_goal and mode == "c3":
            mode = "free"
            reason = SwitchReason.kToReposUnproductive

        # Unsuccessful-buffer maintenance — reference cc:1276 & 1308 add
        # arm's current EE-pos to the unsuccessful buffer at the free→c3
        # transition (kToC3Cost / kToC3ReachedReposTarget).  Port also
        # fires on kToBetterRepos: when the dispatcher abandons the
        # previous repos target for a new one, the previous target is
        # marked as "arm went here but it didn't produce progress" — the
        # sample generator will avoid re-picking within
        # `unsuccessful_radius` next tick.  This is off-reference in the
        # trigger (reference only fires at c3 entry) but the buffer
        # mechanism itself is byte-conformant with cc:2161-2205.
        if self._avoid_unsuccessful and reason in (
                SwitchReason.kToC3Cost,
                SwitchReason.kToC3ReachedReposTarget,
                SwitchReason.kToBetterRepos):
            _obj_xy_now = np.array([
                float(current_q[self._obj_x_idx]),
                float(current_q[self._obj_y_idx]),
            ])
            _obj_quat_now = np.array([
                float(current_q[self._obj_x_idx - 4]),
                float(current_q[self._obj_x_idx - 3]),
                float(current_q[self._obj_x_idx - 2]),
                float(current_q[self._obj_x_idx - 1]),
            ])
            # Prune stale entries first (object may have drifted).
            self.unsuccessful_buffer.prune(_obj_xy_now, _obj_quat_now)
            # Add the arm's CURRENT EE position (reference cc:2177 uses
            # candidate_states[0] which represents current arm state).
            self.unsuccessful_buffer.append(BufferedSample(
                position   = np.asarray(ee_pos_now, dtype=float).copy(),
                cost       = float(c_curr),
                obj_pos_xy = _obj_xy_now.copy(),
                obj_quat   = _obj_quat_now.copy(),
            ))
            if self.log_diag:
                print(f"[UNSUCC-ADD] step={self._step} "
                      f"reason={reason.name} "
                      f"ee_pos=({float(ee_pos_now[0]):+.4f},"
                      f"{float(ee_pos_now[1]):+.4f},"
                      f"{float(ee_pos_now[2]):+.4f}) "
                      f"buffer_size={len(self.unsuccessful_buffer)}",
                      flush=True)

        # Per-tick sample-selection trace. Env-gated (PORT_ALL_SAMP=1,
        # default OFF). Answers "why doesn't the controller try other
        # samples?" by dumping, for every tick: mode, switch reason, raw
        # argmin (k_star), best-non-current index/cost, effective target
        # under the c3/free mode gate, all labels, all c_samples. Emit AFTER
        # the achieved_fixed_goal override so mode reflects the final
        # decision. `target` here is the slot that the free-mode branch
        # would reposition to (:2354-2357); in c3 mode the plan runs off
        # base_mpc at k=0 rather than a sample, so `target` is informational.
        import os as _os_asamp
        if _os_asamp.environ.get("PORT_ALL_SAMP", "0") == "1":
            _short = {"current": "cur", "prev_repos": "prv", "buffer": "buf"}
            _lstr = ",".join(_short.get(lbl, lbl.replace("strat_", "s"))
                             for lbl in labels)
            _cstr = ",".join(f"{float(c):.3f}" for c in c_samples)
            _tgt  = k_star if k_star != 0 else best_other_idx
            _bo   = f"{best_other_idx}" if best_other_idx is not None else "-"
            print(f"[ALL-SAMP] step={self._step} mode={mode} "
                  f"reason={reason.name} k_star={k_star} best_other={_bo} "
                  f"target={_tgt} labels=[{_lstr}] costs=[{_cstr}]",
                  flush=True)

        # §7.56 Stage 1 — [COST-DECOMP] diagnostic. Gated by
        # DIAG_COST_DECOMP_LOG=1 (default-OFF) so flag=0 stays byte-identical
        # to the prior behaviour, including stdout.
        import os as _os_cd
        if _os_cd.environ.get("DIAG_COST_DECOMP_LOG", "0") == "1":
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
                    print(f"[§7.55] PORT_DISABLE_CONTACT_LOSS_GATE=1 — "
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
            # the IK tracker's PI integral. Progress-metric reset is NOT
            # done here — reference sampling_based_c3_controller.cc:1189
            # resets ResetProgressMetrics() on entry to REPOSITIONING
            # (c3->free), not on entry to c3. The port's c3->free block
            # (line ~4361) now mirrors that reference behavior.
            if self._prev_mode == "free":
                self.tracker.reset()
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
                    # PHANTOM-aware (2026-07-27): with the always-on EE-BOX
                    # row (PORT_LCS_ALWAYS_ON_EE_BOX=1 default) a pair is
                    # admitted at ANY distance, so the plain presence test
                    # kept the streak permanently 0 and the contact-loss
                    # gate — even when re-enabled — could never fire during
                    # the phantom retreats (p111 wall crash, p115/p116
                    # 43 s stalls, p117 ceiling crash all sat at
                    # φ = 0.10-0.30 m with 'contact'). Count an admitted
                    # pair beyond PORT_CONTACT_LOSS_PHI_THRESH (default
                    # 0.04 m) as contact LOSS. Inert unless the gate is
                    # re-enabled via PORT_DISABLE_CONTACT_LOSS_GATE=0.
                    _phi_loss_thr = float(os.environ.get(
                        "PORT_CONTACT_LOSS_PHI_THRESH", "0.04"))
                    if float(_ci_sel["distance"]) > _phi_loss_thr:
                        self._no_ee_box_streak += 1
                    else:
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

            # [CONTACT-CHECK] — verify _last_contact_info reflects the CURRENT
            # plant_ctx (not a stale value from a surrogate solve at a
            # non-current sample position). Compares LCS-cached signed
            # distance for the EE-BOX pair against a live Drake query at the
            # current plant_ctx. Env-gated (PORT_CONTACT_CHECK=1, default
            # OFF). If consistent=N frequently, _last_contact_info is stale
            # and any decision downstream keyed on `contact=Y` (sample
            # ranking, `productive` classification, k*=0 preference) is
            # being fed stale contact geometry.
            import os as _os_cc
            if _os_cc.environ.get("PORT_CONTACT_CHECK", "0") == "1":
                try:
                    _lcs_ee_box_info = None
                    if _ci:
                        for _info in _ci:
                            if (isinstance(_info, dict)
                                    and _info.get("tag") == "EE-BOX"):
                                _lcs_ee_box_info = _info
                                break
                    _query_obj_live = (
                        self.plant.get_geometry_query_input_port().Eval(plant_ctx))
                    _sd_live = (
                        _query_obj_live.ComputeSignedDistancePairwiseClosestPoints(
                            max_distance=0.5))
                    _ee_ids  = self.base_mpc.formulator._ee_geom_ids
                    _box_ids = self.base_mpc.formulator._manipuland_geom_ids
                    _drake_ee_box_dist = None
                    for _sd in _sd_live:
                        _has_ee  = (_sd.id_A in _ee_ids)  or (_sd.id_B in _ee_ids)
                        _has_box = (_sd.id_A in _box_ids) or (_sd.id_B in _box_ids)
                        if _has_ee and _has_box:
                            _drake_ee_box_dist = float(_sd.distance)
                            break
                    _ee_pos_live = self.plant.CalcPointsPositions(
                        plant_ctx, self.ee_frame, np.zeros(3), self.world_frame
                    ).flatten()
                    _box_pos_live = self.plant.EvalBodyPoseInWorld(
                        plant_ctx, self.obj_body).translation()
                    _lcs_val = (float(_lcs_ee_box_info["distance"])
                                if _lcs_ee_box_info is not None else float("nan"))
                    _drake_val = (float(_drake_ee_box_dist)
                                  if _drake_ee_box_dist is not None
                                  else float("nan"))
                    _lcs_str = (f"{_lcs_val:+.5f}"
                                if _lcs_ee_box_info is not None else "     nan")
                    _drake_str = (f"{_drake_val:+.5f}"
                                  if _drake_ee_box_dist is not None else "     nan")
                    if (_lcs_ee_box_info is not None
                            and _drake_ee_box_dist is not None):
                        _delta_mm = 1000.0 * (_drake_val - _lcs_val)
                        _consistent = "Y" if abs(_delta_mm) < 1.0 else "N"
                    else:
                        _delta_mm = float("nan")
                        _consistent = "-"
                    print(f"[CONTACT-CHECK] step={self._step} "
                          f"lcs_dist={_lcs_str} drake_dist={_drake_str} "
                          f"delta_mm={_delta_mm:+9.3f} consistent={_consistent} "
                          f"ee_p=({_ee_pos_live[0]:+.4f},"
                          f"{_ee_pos_live[1]:+.4f},{_ee_pos_live[2]:+.4f}) "
                          f"box_p=({_box_pos_live[0]:+.4f},"
                          f"{_box_pos_live[1]:+.4f},{_box_pos_live[2]:+.4f})",
                          flush=True)
                except Exception as _cc_exc:
                    print(f"[CONTACT-CHECK] step={self._step} "
                          f"ERROR={type(_cc_exc).__name__}: {_cc_exc}",
                          flush=True)

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
                # Reference-conformant: read the winning EE 3-vector out of
                # `_all_sample_locations` (mirrors dairlib
                # sampling_based_c3_controller.cc:1842-1843
                #   all_sample_locations_[best_sample_index_]
                # ). Value is identical to `results[target_idx].sample_pos`.
                self._best_sample_index = target_idx
                p_repos = self._all_sample_locations[target_idx]
                self._current_repos_target = p_repos.copy()
                self._current_repos_cost   = c_samples[target_idx]
                best_src = labels[target_idx]

                # Stage C landing-storm trace — upstream argmin emit.
                # Default-OFF. Captures whether the per-tick selection is
                # flipping between cached samples (the leading hypothesis
                # for the rebuild storm at 1 kHz).
                import os as _os_lt2
                if (_os_lt2.environ.get("DIAG_LANDING_TRACE", "0") == "1"
                        and self._step >= int(_os_lt2.environ.get(
                            "DIAG_LANDING_TRACE_FROM", "1600"))):
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
                    if _os_lt0.environ.get("DIAG_LANDING_TRACE", "0") == "1":
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

    def _compute_r7_c3_direct_torque(self, current_q, current_v, plant_ctx):
        """R^7 full-plant c3-mode executor: apply planner's u_seq[0] directly
        + gravity comp + joint-PD for stability. Bypasses the EE-space OSC.

        Under R^7 (use_ee_space=False), the C3 planner's `u_seq[k] ∈ ℝ⁷` is
        joint torque directly. The port's OSC/impedance stack was designed
        to CONVERT a Cartesian force intent (EE-space) into joint torques
        via J_ee^T — under R^7 that conversion is redundant and mis-steers
        (p102 evidence: rot 0.708 rad, planner engaged 90% of ticks with
        λ_n=28 but f_cmd pointed AWAY from goal).

        Structure:
            τ_out = clip( u_seq[0] + τ_g_arm + Kp·(q_des − q) + Kd·(v_des − v) )
        where q_des, v_des come from x_seq[1][:n_q_arm] / x_seq[1][n_q:n_q+n_v_arm].

        Gravity contract: main.py universally applies −τ_g on top of this
        executor's output (main.py "universal-add", both at the planner tick
        and at every 1 kHz sub-tick). The R^7 LCS models gravity (rhs =
        B u − Cv + τ_g), so the planner's u_seq is meant for a gravity-FULL
        plant. Adding +τ_g here nets out main.py's −τ_g, so the plant the
        planner's torque acts on matches the plant the LCS modeled:
            applied_total = (−τ_g) + (τ_ff + τ_g + τ_pd) = τ_ff + τ_pd.
        (Previous form subtracted τ_g here, which — combined with main.py's
        −τ_g and the anti-gravity content already inside u_seq — compensated
        gravity twice over and biased the arm with a spurious ≈−2τ_g.)

        Returns (τ (n_arm,), diag dict).
        """
        n_arm = int(self.n_u)
        _u_seq = getattr(self.base_mpc, "_last_u_seq", None)
        if _u_seq is not None and hasattr(_u_seq, "shape") \
                and _u_seq.ndim == 2 and _u_seq.shape[1] >= n_arm:
            _tau_ff = np.asarray(_u_seq[0][:n_arm], dtype=float)
        else:
            _tau_ff = np.zeros(n_arm)

        # Gravity term — +tau_g to net out main.py's universal −tau_g add
        # (see docstring "Gravity contract"). The LCS includes gravity, so
        # u_seq expects a gravity-full plant; main.py's add would otherwise
        # cancel gravity out from under the planner.
        _tau_g_full = self.plant.CalcGravityGeneralizedForces(plant_ctx)
        _tau_g_arm = np.asarray(_tau_g_full[:n_arm], dtype=float)

        # Joint-space PD toward planner's next-tick predicted arm state.
        _x_seq = getattr(self.base_mpc, "last_x_seq", None)
        n_q_full = int(self.n_q)
        if _x_seq is not None and len(_x_seq) > 1 \
                and _x_seq[1].shape[0] >= n_q_full + n_arm:
            _q_des = np.asarray(_x_seq[1][:n_arm], dtype=float)
            _v_des = np.asarray(_x_seq[1][n_q_full:n_q_full + n_arm], dtype=float)
        else:
            _q_des = np.asarray(current_q[:n_arm], dtype=float)
            _v_des = np.zeros(n_arm)

        _q_err = _q_des - np.asarray(current_q[:n_arm], dtype=float)
        _v_err = _v_des - np.asarray(current_v[:n_arm], dtype=float)

        # Modest joint-PD gains. Env-tunable for future study.
        import os as _os_r7pd
        _kp = float(_os_r7pd.environ.get("PORT_R7_JOINT_KP", "40.0"))
        _kd = float(_os_r7pd.environ.get("PORT_R7_JOINT_KD", "4.0"))
        _tau_pd = _kp * _q_err + _kd * _v_err

        # Total: feedforward + gravity-restore (nets main.py's −tau_g) + PD
        _tau_out = _tau_ff + _tau_g_arm + _tau_pd

        # Clip to Franka joint torque limit.
        _tau_max = 87.0
        _sat_before = np.any(np.abs(_tau_out) > _tau_max)
        _tau_out = np.clip(_tau_out, -_tau_max, +_tau_max)

        _diag = dict(
            tau_ff=_tau_ff.copy(),
            tau_g=_tau_g_arm.copy(),
            tau_pd=_tau_pd.copy(),
            tau_out=_tau_out.copy(),
            tau_imp=_tau_pd.copy(),   # alias — impedance term = joint-PD here
            q_err_norm=float(np.linalg.norm(_q_err)),
            v_err_norm=float(np.linalg.norm(_v_err)),
            x_err=_q_err.copy(),      # joint-space error stands in for Cartesian
            saturated=bool(_sat_before),
            qp_success=True,
            # Downstream telemetry compatibility (EE-space imp_diag shape).
            # Under R^7 direct-torque, there's no external λ QP; report False.
            had_lambda_n=False,
            had_lambda_t=False,
            solve_ms=0.0,
            lambda_ext=np.zeros(3),
            lambda_des=np.zeros(3),
        )
        return _tau_out, _diag

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
            # 2026-07-27 R^7 full-plant direct-torque path. When use_ee_space=False,
            # planner produces joint torques directly (u_seq[0] ∈ ℝ⁷). The port's
            # OSC/impedance stack downstream is EE-space-only — it converts a
            # Cartesian force intent into joint torques via J_ee^T. Under R^7
            # that conversion is redundant and mis-steers (verified by p102
            # rot regression). Bypass entirely: apply u_seq[0] as feedforward
            # arm torque + gravity comp + small joint-PD for stability. Env-gate
            # REFCONF_R7_DIRECT_TORQUE=1 (default ON when R^7 is active).
            _use_ee_space = bool(getattr(self.base_mpc, "use_ee_space", False))
            import os as _os_r7dt
            _r7_direct = (
                not _use_ee_space
                and _os_r7dt.environ.get("REFCONF_R7_DIRECT_TORQUE", "1") == "1")
            if _r7_direct:
                u_imp, imp_diag = self._compute_r7_c3_direct_torque(
                    current_q, current_v, plant_ctx)
                # Cache for sub-tick decoupling; sub-tick will re-invoke this
                # path via `compute_control_osc_only`.
                self._last_osc_call = ("c3_r7_direct", dict())
                # Skip all the EE-space downstream. Jump past the OSC dispatch.
                _v_ee_des = None; _lam_des = None; _a_ee_des = None
                _exec_lam_n = None; _exec_lam_t = None; _exec_Jn = None; _exec_Jt = None
                _q_arm_ik = None; _traj_c3 = None
                # No further executor call for this branch; mark flag for skip.
                self._r7_direct_this_tick = True
                # Preserve the informational-u path used by caller.
                if self.log_diag:
                    print(f"[R7-DIRECT] step={self._step} u_arm_norm="
                          f"{float(np.linalg.norm(u_imp)):.3f} "
                          f"tau_max={float(np.max(np.abs(u_imp))):.3f}",
                          flush=True)
            else:
                self._r7_direct_this_tick = False
            # Cartesian target from C3+'s next-step state prediction.
            _x_seq = self.base_mpc.last_x_seq
            # EE-space planner: p_ee is already a state slot in x_seq, read
            # it directly — no FK. (Slice indices verified bit-equal to FK
            # at the linearization point by scripts/verify_slice_indices.py;
            # max |x_seq[0][7:10] - p_ee_now| = 1.11e-15.) R^7 planner path
            # below retains the original FK extraction.
            if _x_seq is not None and len(_x_seq) > 1:
                if _use_ee_space:
                    _p_ee_des = _x_seq[1][7:10].copy()
                    # Bug 3 guard (2026-07-22): if LCS predicted unphysical
                    # one-step EE displacement, clip to a plausible target.
                    # Franka arm max EE Cartesian velocity is ~1 m/s; with
                    # planning dt=0.1 s the max plausible one-step move is
                    # ~0.1 m. Use 0.15 m as the safety threshold (50 % margin).
                    # p41 showed the LCS can produce >>1 m predictions after
                    # _EE_MASS was miscalibrated — the OSC then chases an
                    # impossible target and the T becomes uncontrollable.
                    # Clip, log once per instance.
                    _plan_move = np.asarray(_p_ee_des, dtype=float) - np.asarray(
                        ee_pos_now, dtype=float)
                    _plan_mag  = float(np.linalg.norm(_plan_move))
                    if _plan_mag > 0.15:
                        _dir = _plan_move / _plan_mag
                        _p_ee_des = np.asarray(ee_pos_now, dtype=float) + 0.15 * _dir
                        if not getattr(self, "_xseq_plausibility_warned", False):
                            print(
                                f"[X-SEQ-GUARD] step={self._step} planner "
                                f"predicted |Δp_ee|={_plan_mag:.3f}m > 0.15m — "
                                f"clipped to 0.15m. Likely LCS mis-calibration "
                                f"(e.g., _EE_MASS or dt). See p41 forensic.",
                                flush=True,
                            )
                            self._xseq_plausibility_warned = True
                    # Reference `sampling_based_c3_controller.cc:1757-1759`
                    # freezes the c3-mode EE-position target Z at
                    # sampling_height + wall_offset on every horizon knot:
                    #   knots(2, i) = sampling_params_.z_height + wall_offset
                    # Prior port took Z from x_seq[1][9] (planner's next-tick
                    # predicted EE z). Once real contact fired, the planner's
                    # LCS included the contact pair; x_seq[1][9] retreated;
                    # arm lost contact. Freeze Z at sampling_height to hold
                    # contact height stable — matches reference OSC track_z.
                    _sh = float(self.params.sampling_params.sampling_height)
                    _p_ee_des[2] = _sh
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
                    # Reference c3-mode UpdateC3ExecutionTrajectory
                    # (cc:1757-1761) OVERRIDES the c3 planner's z at each
                    # knot to sampling_params_.z_height:
                    #   knots(2, i) = sampling_z_height;
                    # → arm z is PINNED at the sampling plane throughout c3.
                    # Port previously used obj_z (object CoM z), which for
                    # T (obj_z=0.020) is below reference's z_height=0.034
                    # equivalent. Aligning to sampling_height matches
                    # reference's constant-plane push posture.
                    _sampling_z = float(
                        self.params.sampling_params.sampling_height)
                    if getattr(self, "_c3_geom_z_target", None) is None:
                        self._c3_geom_z_target = _sampling_z
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
            if _os.environ.get("DIAG_HORIZON_LAM_DUMP", "0") == "1":
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
            _ee_pair_admitted = False
            if (_lam_n is not None
                    and hasattr(_lam_n, "size")
                    and _lam_n.size > 0):
                if (_shape_gate == "tshape"
                        and _cinfo_gate is not None
                        and len(_cinfo_gate) == _lam_n.size):
                    _ee_idxs_gate = [i for i, info in enumerate(_cinfo_gate)
                                     if isinstance(info, dict)
                                     and info.get("tag", "") == "EE-BOX"]
                    _ee_pair_admitted = bool(_ee_idxs_gate)
                    if _ee_idxs_gate:
                        _lam_n_mag = float(np.sum(np.abs(_lam_n[_ee_idxs_gate])))
                    else:
                        _lam_n_mag = 0.0
                else:
                    _lam_n_mag = float(np.sum(np.abs(_lam_n)))
                    # Box path / fallback: no per-pair tag scan available,
                    # so admission is inferred from the fact that _lam_n has
                    # nonzero size AND cinfo indicates at least one pair.
                    _ee_pair_admitted = (
                        _cinfo_gate is not None and len(_cinfo_gate) > 0)
            else:
                _lam_n_mag = 0.0
            # Bug 1 fix (2026-07-22, gated 2026-07-22 v2): the p41 pattern
            # was "LCS admits EE-BOX pair AND planner says λ_n=0" for many
            # consecutive c3 steps while Drake reality remains in contact.
            # Root cause: LCS conditioning (17.5x-amplified D rows) caused
            # planner to disengage. Fallback via nominal_push_force is
            # WRONG on healthy runs — during legitimate approach/transit
            # the planner correctly commands small/zero λ. Gate the
            # fallback on the same "suspicious conditioning" flag Bug 2
            # emits (dt/m_ee > 0.5). Under normal port config
            # (m_ee=1.0, dt=0.1 → ratio=0.1), the gate is False and this
            # block is byte-identical to pre-fix behavior. Under future
            # qvector-migration attempts where m_ee is rescaled, the
            # fallback kicks in as intended.
            _suspicious_cond = (
                getattr(self.base_mpc.formulator, "_dt_mee_warned", False))
            if _lam_n_mag > 0.05 or (_ee_pair_admitted and _suspicious_cond):
                _lam_des = self._derive_force_command(
                    _lam_n, g_hat_3d, plant_ctx=plant_ctx)
            else:
                _lam_des = np.zeros(3)

            import os as _os_fr
            if _os_fr.environ.get("DIAG_FORCE_ROUTE_TRACE", "0") == "1":
                _fr_env = _os_fr.environ.get("PORT_FORCE_ROUTING", "off").lower()
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
            # Gated by DIAG_SETPOINT_TRACE=1 (default-OFF). EE-space-only.
            if (_os_fr.environ.get("DIAG_SETPOINT_TRACE", "0") == "1"
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
            # converged?). Gated by DIAG_CONSISTENCY_TRACE=1; default-OFF;
            # EE-space-only.
            if (_os_fr.environ.get("DIAG_CONSISTENCY_TRACE", "0") == "1"
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
            # §7.51 — PORT_DISABLE_C3_OVERRIDE (default-OFF) skips the LTD
            # APPROACH-OVERRIDE block entirely, leaving _p_ee_des at the FK
            # source (_x_seq[1][7:10] at line 2263). Validated as load-bearing
            # for the first box closure in §7.51 (with PORT_EE_APPROACH_FACE_TARGET=1
            # + w_ee_approach=8000 + W_force=1). Default-OFF
            # byte-identical preserved. One-shot log on first c3 tick.
            _disable_c3_override = (_os.environ.get(
                "PORT_DISABLE_C3_OVERRIDE", "0") == "1")
            if _disable_c3_override and not getattr(
                    self, "_751_banner_printed", False):
                print("[§7.51] PORT_DISABLE_C3_OVERRIDE=1 — LTD APPROACH-"
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
            # Bugfix: use Drake's actual plant clock (get_time()) instead of
            # the step-counter estimate (self._step - 1) * dt_ctrl. The
            # step-counter has ~1-tick offset from the plant clock due to
            # the sub-tick OSC decoupling in main.py:846-857 — main advances
            # plant time with _DT_OSC (1 ms) sub-steps while self._step
            # increments per outer tick. Prior N-knot trajectory used the
            # step-counter as start time; OSC evaluated at plant clock →
            # trajectory time-range didn't include the plant-clock query →
            # PiecewisePolynomial returned edge (last knot) values →
            # OSC saw an incorrect target (previous run box_dual_095654:
            # box moved 1 mm instead of ~485 mm).
            _sim_t_c3 = float(plant_ctx.get_time()) if plant_ctx is not None \
                else float(self._step - 1) * float(self._dt_ctrl)
            # Full-horizon c3-mode position trajectory (matches reference
            # sampling_based_c3_controller.cc:1757-1770 publishing N-knot
            # LcmTrajectory to OSC). Prior port built a 2-knot trajectory
            # spanning dt_ctrl only (~10 ms) — OSC saw ~1 point of look-ahead.
            # Reference publishes N knots at dt_planner spacing (0.075 s × 5
            # = 0.375 s of look-ahead), so OSC has continuous forward-motion
            # info as it evaluates at increasing t_sim between planner solves.
            _use_ee_space = bool(getattr(self.base_mpc, "use_ee_space", False))
            _x_seq_full = getattr(self.base_mpc, "last_x_seq", None)
            _dt_plan = float(getattr(self.base_mpc, "dt", 0.075))
            # Near-goal (pose regime): planner uses dt_pose. Trajectory
            # knot spacing must match so the OSC eval times land on the
            # correct planner-predicted states.
            if getattr(self.base_mpc, "_crossed_switching_threshold", False):
                _dt_plan = float(
                    getattr(self.base_mpc, "dt_pose", _dt_plan))
            _N_plan = int(getattr(self.base_mpc, "horizon", 5))
            # 2026-07-26 arc-2 phantom-contact override:
            # PORT_LCS_ALWAYS_ON_EE_BOX=1 injects an EE-BOX pair into the LCS even
            # when arm is physically far from box (bypasses ref's 2mm signed-
            # distance threshold). Under low-force regimes (u<friction breakout)
            # this creates a feedback loop:
            #   1. Planner sees "contact", plans push along phantom n_hat
            #   2. x_seq predicts arm+box motion that can't happen physically
            #   3. OSC blindly tracks x_seq[i][7:10], pulling arm away from box
            #   4. Arm drifts, box unmoved, LCS's phantom contact worsens
            # p92 crashed via workspace violation at step=202 with arm at
            # (0.64, -0.42) while box was at (0.48, 0.00) — 45cm separation.
            # This override replaces the planner-trajectory with a 2-knot
            # linear ramp toward the IK-projected geometric contact target
            # (_p_ee_des from :3167) when EE-to-box distance exceeds
            # phantom_dist_thresh. Reference-directional (moves arm TOWARD
            # box) rather than reference-conformant (which would just drop
            # the phantom-contact via signed-distance threshold, but that's
            # PORT_LCS_ALWAYS_ON_EE_BOX=0). Env-gated PORT_C3_PHANTOM_TRAJ_OVERRIDE=1
            # (default OFF, byte-identical when disabled).
            _box_xy_now = np.array([
                current_q[self._obj_x_idx],
                current_q[self._obj_y_idx],
            ])
            _ee_box_xy_dist = float(np.linalg.norm(
                np.asarray(ee_pos_now[:2], dtype=float) - _box_xy_now))
            _phantom_dist_thresh = float(_os.environ.get(
                "PORT_C3_PHANTOM_DIST_THRESH", "0.10"))
            _use_phantom_override = (
                _os.environ.get("PORT_C3_PHANTOM_TRAJ_OVERRIDE", "0") == "1"
                and _ee_box_xy_dist > _phantom_dist_thresh
                and _p_ee_des is not None)
            if _use_phantom_override:
                # 2-knot FirstOrderHold from ee_now → IK-projected box-face target.
                # Bypasses planner's phantom x_seq entirely.
                _p_ik_col  = np.asarray(_p_ee_des, dtype=float).reshape(3, 1)
                _p_now_col = np.asarray(ee_pos_now, dtype=float).reshape(3, 1)
                _traj_c3 = PiecewisePolynomial.FirstOrderHold(
                    [_sim_t_c3, _sim_t_c3 + float(self._dt_ctrl)],
                    np.hstack([_p_now_col, _p_ik_col]),
                )
                # Null accel feedforward — we're NOT following planner's x_seq,
                # so its predicted accel is irrelevant.
                _a_ee_des = None
                if not getattr(self, "_c3_phantom_override_banner", False):
                    self._c3_phantom_override_banner = True
                    print(f"[C3-PHANTOM-OVERRIDE] first fire step={self._step} "
                          f"ee_box_xy_dist={_ee_box_xy_dist*1000:.1f}mm > "
                          f"thresh={_phantom_dist_thresh*1000:.1f}mm — OSC follows "
                          f"IK-projected target instead of planner x_seq",
                          flush=True)
            elif (_use_ee_space and _x_seq_full is not None
                    and hasattr(_x_seq_full, "shape")
                    and _x_seq_full.ndim == 2
                    and _x_seq_full.shape[0] >= _N_plan + 1
                    and _x_seq_full.shape[1] >= 10
                    and _N_plan >= 2):
                # Match reference (cc:1757-1761): after building the raw
                # knots from x_sol_[k], overwrite knots(2, i) = z_height +
                # wall_offset on every horizon knot so the OSC z-target is
                # constant. Reference keeps EE height stable while it
                # traverses the perimeter of the object; without this
                # freeze, the C3+ planner's raw x_seq z-slice (which is
                # solving against an LCS with a phantom always-on EE-BOX
                # row) drags the OSC setpoint upward, producing the c3-
                # mode arm slam (max ee_z 0.86 m observed pre-fix).
                # 2026-07-19 revision: prior port used raw z (comment on
                # line 3479 misread reference); reference cc:1757-1761
                # clearly z-freezes every knot.
                _knots = np.array([
                    _x_seq_full[i][7:10] for i in range(1, _N_plan + 1)
                ], dtype=float).T
                _sh_c3 = float(self.params.sampling_params.sampling_height)
                _knots[2, :] = _sh_c3   # z-freeze every knot (ref cc:1757)
                _ts = [_sim_t_c3 + i * _dt_plan for i in range(_N_plan)]
                _traj_c3 = PiecewisePolynomial.FirstOrderHold(_ts, _knots)
                # 2026-07-19 trajectory-log for descent audit (c3 mode).
                # x_seq layout for EE-space: [box_q(7), p_ee(3), box_v(6),
                # v_ee(3)] → box_p at 4:7, ee_p at 7:10. Log the planner's
                # predicted BOX trajectory and EE trajectory over the
                # full horizon so we can see if the planner is intending
                # to lift the T (or the arm) somewhere unexpected.
                if self.log_diag:
                    _ee_zs = [float(_x_seq_full[i][9])
                              for i in range(1, _N_plan + 1)]
                    _box_zs = [float(_x_seq_full[i][6])
                               for i in range(1, _N_plan + 1)]
                    _box_ys = [float(_x_seq_full[i][5])
                               for i in range(1, _N_plan + 1)]
                    _ee_zs_str  = ",".join(f"{z:+.4f}" for z in _ee_zs)
                    _box_zs_str = ",".join(f"{z:+.4f}" for z in _box_zs)
                    _box_ys_str = ",".join(f"{y:+.4f}" for y in _box_ys)
                    print(f"[C3-TRAJ] step={self._step} "
                          f"ee_now_z={float(ee_pos_now[2]):+.4f} "
                          f"planned_ee_z=[{_ee_zs_str}] "
                          f"planned_box_z=[{_box_zs_str}] "
                          f"planned_box_y=[{_box_ys_str}] "
                          f"z_frozen_to={_sh_c3:+.4f}",
                          flush=True)
            else:
                # Legacy 2-knot fallback for R^7 path or missing x_seq.
                _p_c3_col = np.asarray(_p_ee_des, dtype=float).reshape(3, 1)
                _p_now_col = np.asarray(ee_pos_now, dtype=float).reshape(3, 1)
                _traj_c3 = PiecewisePolynomial.FirstOrderHold(
                    [_sim_t_c3, _sim_t_c3 + float(self._dt_ctrl)],
                    np.hstack([_p_now_col, _p_c3_col]),
                )
            if getattr(self, "_r7_direct_this_tick", False):
                # R^7 direct-torque path already produced u_imp/imp_diag at
                # the top of this branch. Skip the EE-space OSC executor.
                pass
            else:
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
                    # (set only when PORT_TSHAPE_C3_GEOM=1 fires); None
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
            # REFCONF_REPOSITION_PWL=1, params.use_reposition_pwl_trajectory).
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
                    # Reset the predicted p_start on c3→free transition so
                    # the first repos build after leaving c3 starts from
                    # the arm's actual (post-c3) position rather than an
                    # outdated prediction from an earlier repos episode.
                    self._x_pred_repos_plan = None
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
                # 2026-07-19: rebuild EVERY planner tick, matching reference
                # sampling_based_c3_controller.cc:1330 which calls
                # UpdateRepositioningExecutionTrajectory unconditionally.
                # Prior port only rebuilt on target-change (>5 mm) — with a
                # stale trajectory, `finished_reposition_flag` (which is set
                # at build time from `t_end - t_start <= dt_plan`) also went
                # stale and never fired mid-run.  Ref cc:1330 does not gate.
                _need_rebuild = True

                # Stage C landing-storm trace — gated, default-OFF.
                # Window: step >= DIAG_LANDING_TRACE_FROM (default 1600 at
                # 1 kHz; ~step 160 at 100 Hz to capture the same sim-time
                # window). One consolidated emit per tick at the rebuild
                # gate carrying the full chain state. The fix is decided
                # AFTER the read — this block adds NO behavior change.
                import os as _os_lt
                _trace_on = _os_lt.environ.get("DIAG_LANDING_TRACE", "0") == "1"
                _trace_from = int(_os_lt.environ.get("DIAG_LANDING_TRACE_FROM", "1600"))
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
                    # 2026-07-19 reference-conformant p_start.
                    # Reference sampling_based_c3_controller.cc:1442 calls
                    # ClampEndEffectorAcceleration BEFORE Reposition when
                    # use_predicted_x0_repos is true.  That replaces the
                    # actual EE position with the PREDICTED position from
                    # the previous tick's trajectory (clamped to be within
                    # nominal_ee_accel × dt² of actual).  Effect: p_start
                    # marches forward in prediction-space at trajectory
                    # speed even if the physical arm lags — descends past
                    # the re-lift trap that keeps `p_start = ee_pos_now`
                    # stuck at z ≈ z_safe.
                    _nominal_ee_accel = float(getattr(
                        self.base_mpc, "nominal_ee_accel", 2.0))
                    if (self._x_pred_repos_plan is not None
                            and _nominal_ee_accel > 0.0):
                        _dt_c = min(0.1, float(self._dt_ctrl))
                        _delta_pos = _nominal_ee_accel * _dt_c * _dt_c
                        _p_start_ref = np.clip(
                            np.asarray(self._x_pred_repos_plan, dtype=float),
                            ee_pos_now - _delta_pos,
                            ee_pos_now + _delta_pos,
                        )
                    else:
                        _p_start_ref = np.asarray(ee_pos_now, dtype=float)
                    self._pwl_traj = RepositionTrajectory(
                        p_start=_p_start_ref,
                        p_target=_p_target_arr,
                        z_safe=float(
                            self.params.reposition_params.pwl_waypoint_height),
                        speed=float(self.params.reposition_params.pwl_speed),
                        t_start=_sim_t,
                        straight_line_thresh=float(self.params
                            .reposition_params
                            .use_straight_line_traj_under_piecewise_linear),
                        # dt_plan feeds RepositionTrajectory's ref-conformant
                        # finished_reposition_flag (t_end-t_start ≤ dt_plan).
                        dt_plan=float(self._dt_ctrl),
                    )
                    # Update the predicted state for NEXT tick's clamp.
                    # Reference cc:1858-1863: sample the just-built trajectory
                    # at `filtered_solve_time_` (average planner-solve time),
                    # save as `x_pred_curr_plan_`. Port uses `_dt_ctrl` as the
                    # analog since the port doesn't measure solve time.
                    _p_pred_next, _, _ = self._pwl_traj.eval(
                        float(_sim_t) + float(self._dt_ctrl))
                    self._x_pred_repos_plan = np.asarray(
                        _p_pred_next, dtype=float).copy()
                    self._pwl_traj_built_for_target = _p_target_arr.copy()
                    self._pwl_traj_last_build_step = int(self._step)
                    print(
                        f"[STAGE-A-PWL] step={self._step} "
                        f"sim_t={_sim_t:.3f} build "
                        f"p_start=({_p_start_ref[0]:+.4f},"
                        f"{_p_start_ref[1]:+.4f},{_p_start_ref[2]:+.4f}) "
                        f"p_target=({_p_target_arr[0]:+.4f},"
                        f"{_p_target_arr[1]:+.4f},{_p_target_arr[2]:+.4f}) "
                        f"K={self._pwl_traj.knot_positions.shape[1]} "
                        f"t_end={self._pwl_traj.t_end:.3f}",
                        flush=True,
                    )
                _p_des, _v_des, _done = self._pwl_traj.eval(_sim_t)
                _p_ee_des = _p_des
                _v_ee_des = _v_des
                # PORT_ACHIEVED_VERTICAL_RETRACT (default OFF) — p121
                # overshoot bug: after the ACHIEVED-FIXED-GOAL latch, the
                # get-out-of-the-way retraction cut the corner — ~30 mm of
                # lateral creep while the EE was still at contact height
                # (z 0.022→0.075) plowed through the T and shoved it
                # +90 mm past the goal (tight latch at step 334, final
                # loose FAIL). Geometry-dependent: p118/r1/r3 retracted
                # clear. Guard: while achieved and below the clear height
                # (T top 0.04 + tip radius 0.0195 + margin), command a
                # STRICTLY VERTICAL hover ascent at the current xy; lateral
                # motion resumes only above the clear height.
                _avr = (getattr(self, "_achieved_fixed_goal", False)
                        and os.environ.get(
                            "PORT_ACHIEVED_VERTICAL_RETRACT", "0") == "1"
                        and float(ee_pos_now[2]) < 0.10)
                if _avr:
                    _p_ee_des = np.array([float(ee_pos_now[0]),
                                          float(ee_pos_now[1]), 0.12])
                    _v_ee_des = None
                    if self.log_diag and not getattr(
                            self, "_avr_logged", False):
                        self._avr_logged = True
                        print(f"[ACHIEVED-VERT-RETRACT] step={self._step} "
                              f"ee_z={float(ee_pos_now[2]):+.4f} — vertical "
                              f"ascent before lateral retract", flush=True)
                # 2026-07-19 trajectory-log for descent audit.  Emits the
                # full knot z-profile at the planner tick so we can trace
                # WHERE the arm is being commanded to go vs where it
                # actually is.  Free-mode only (c3-mode uses x_seq via
                # base_mpc — logged separately below).
                if self.log_diag:
                    _kz = self._pwl_traj.knot_positions[2, :].tolist()
                    _kt = (self._pwl_traj.knot_times
                           - self._pwl_traj.t_start).tolist()
                    _kz_str = ",".join(f"{z:+.4f}" for z in _kz)
                    _kt_str = ",".join(f"{t:.4f}" for t in _kt)
                    print(f"[PWL-TRAJ] step={self._step} "
                          f"ee_now_z={float(ee_pos_now[2]):+.4f} "
                          f"p_des_now=({_p_des[0]:+.4f},"
                          f"{_p_des[1]:+.4f},{_p_des[2]:+.4f}) "
                          f"v_des_now=({_v_des[0]:+.4f},"
                          f"{_v_des[1]:+.4f},{_v_des[2]:+.4f}) "
                          f"knot_z=[{_kz_str}] "
                          f"knot_t_rel=[{_kt_str}]",
                          flush=True)
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
                # Under the vertical-retract override, cache the STATIC
                # vertical target instead — sub-tick PWL re-eval would
                # reintroduce the lateral corner-cut at 1 kHz.
                if _avr:
                    self._last_osc_call = ("osc_direct_free", dict(
                        p_ee_desired=np.asarray(
                            _p_ee_des, dtype=float).reshape(3).copy(),
                        v_ee_desired=None,
                        mode="free",
                    ))
                else:
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

        # --- Free-mode Cartesian-trap stall recovery (PORT-only, gated) ---
        # p115 anatomy: after the phantom-λ retreat parked the arm at a
        # stretched high posture, the free-mode OSC sat 43 s producing a
        # healthy ~15 Nm toward a 2 cm-away reference with ~zero motion —
        # a Cartesian local trap. The per-tick IK meanwhile found a valid
        # in-limits solution ~30° away in joint space (a configuration-
        # branch jump the task-space OSC cannot take). The reposition
        # machinery itself is reference-conformant (rebuild-per-tick +
        # nominal_ee_accel=2 leash match cc:1457-1474), so the conformant
        # stack cannot escape a posture the reference never reaches; this
        # recovery is a PORT_* workaround for the port-specific upstream
        # (phantom retreat) until that cluster is fixed.
        # Detection: EE displacement < 1 mm per tick for ≥10 consecutive
        # free ticks while > 5 cm from the tracker target. Action: IK the
        # CURRENT free-mode Cartesian target, drive joint-PD toward that
        # solution (sub-tick replayed via 'free_joint_recovery').
        if (mode != "c3"
                and os.environ.get(
                    "PORT_FREE_STALL_JOINT_RECOVERY", "0") == "1"):
            _ee_prev = getattr(self, "_free_stall_ee_prev", None)
            _stride = (float(np.linalg.norm(ee_pos_now - _ee_prev))
                       if _ee_prev is not None else 1.0)
            self._free_stall_ee_prev = np.asarray(ee_pos_now, float).copy()
            _tgt_now = np.asarray(_p_ee_des, dtype=float).reshape(3)
            _far = float(np.linalg.norm(_tgt_now - ee_pos_now)) > 0.05 \
                or (self._pwl_traj is not None and float(np.linalg.norm(
                    np.asarray(self._pwl_traj.p_target) - ee_pos_now)) > 0.05)
            if _stride < 0.001 and _far:
                self._free_stall_streak = getattr(
                    self, "_free_stall_streak", 0) + 1
            else:
                self._free_stall_streak = 0
            if self._free_stall_streak >= 10:
                from control.sampling_c3.ik import solve_ik_to_ee_pos
                # Escape target: the ACTUAL repos sample (PWL p_target),
                # NOT the leashed per-tick reference _p_ee_des — the leash
                # keeps that within ~2 cm of the stuck arm, so its IK
                # solution is degrees away and the PD torque is negligible
                # (p116: engaged at q_err=1.2°, no escape). The branch
                # jump needs the far target's IK solution.
                _tgt_rec = (np.asarray(self._pwl_traj.p_target, float)
                            if self._pwl_traj is not None else _tgt_now)
                _q_saved_rec = self.plant.GetPositions(plant_ctx).copy()
                try:
                    _q_rec, _rec_err, _rec_it = solve_ik_to_ee_pos(
                        self.plant, self.ee_frame,
                        p_target=_tgt_rec, q_init=current_q.copy(),
                        plant_ctx=plant_ctx, n_arm_dofs=self.n_u)
                finally:
                    self.plant.SetPositions(plant_ctx, _q_saved_rec)
                if _rec_err < 0.02:
                    _kp_r = float(os.environ.get(
                        "PORT_FREE_RECOVERY_KP", "40.0"))
                    _kd_r = float(os.environ.get(
                        "PORT_FREE_RECOVERY_KD", "4.0"))
                    _qd_rec = np.asarray(_q_rec[: self.n_u], float)
                    _qe_rec = _qd_rec - np.asarray(
                        current_q[: self.n_u], float)
                    u_imp = np.clip(
                        _kp_r * _qe_rec
                        - _kd_r * np.asarray(current_v[: self.n_u], float),
                        -87.0, +87.0)
                    self._last_osc_call = ("free_joint_recovery", dict(
                        q_des=_qd_rec.copy(), kp=_kp_r, kd=_kd_r))
                    if self.log_diag and (self._free_stall_streak == 10
                                          or self._step % 20 == 0):
                        print(f"[FREE-RECOVERY] step={self._step} "
                              f"streak={self._free_stall_streak} "
                              f"q_err_max_deg="
                              f"{np.degrees(np.max(np.abs(_qe_rec))):.1f} "
                              f"ik_err={_rec_err:.4f} "
                              f"tgt=({_tgt_rec[0]:+.3f},{_tgt_rec[1]:+.3f},"
                              f"{_tgt_rec[2]:+.3f})", flush=True)
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
            # [C3-VIZ] — port of reference c3_mode_visualizer.cc:69-108.
            # Reference publishes an LCM traj that shows a pink dot at the EE
            # when in c3 mode (and (0,0,0) when not). Port equivalent is a
            # per-tick log line: is_c3_mode (1/0) + EE xyz. Downstream tools
            # can grep [C3-VIZ] to reconstruct the mode-vs-EE trajectory
            # without the full [STEP] parsing.
            _is_c3 = 1 if mode == "c3" else 0
            print(f"[C3-VIZ] step={self._step} t={(self._step - 1) * self._dt_ctrl:.3f}s "
                  f"is_c3={_is_c3} "
                  f"ee=({ee_pos_now[0]:+.4f},{ee_pos_now[1]:+.4f},{ee_pos_now[2]:+.4f})",
                  flush=True)
            # [EE-SAMPLES] — port of reference `all_sample_locations_`
            # (systems/controllers/sampling_based_c3_controller.h:186 + cc:216).
            # Reference publishes the full sample set each tick as an LCM
            # trajectory of xyz points; downstream tools plot them as dots
            # around the object. Port equivalent: per-tick list of candidate
            # EE positions with slot labels (parallel to `labels` list).
            # Slot layout matches _build_samples: [current, prev_repos?,
            # strat_0..strat_N-1, (buffer)?]. Populated in _sample_and_select
            # right after _build_samples runs.
            _samples = getattr(self, "_all_sample_locations", None) or []
            if _samples:
                _sample_labels = getattr(self, "_last_sample_labels", None) or [
                    f"k{k}" for k in range(len(_samples))
                ]
                _parts = []
                for _k, (_lbl, _p) in enumerate(zip(_sample_labels, _samples)):
                    _parts.append(f"{_k}:{_lbl}=({float(_p[0]):+.4f},"
                                  f"{float(_p[1]):+.4f},{float(_p[2]):+.4f})")
                print(f"[EE-SAMPLES] step={self._step} n={len(_samples)} "
                      + " ".join(_parts), flush=True)
            # [WORKSPACE-VIOLATION] — port of reference cc:1476-1494
            # CheckForWorkspaceLimitViolations. Reference DRAKE_DEMANDs (aborts)
            # when EE exits its axis-aligned bounds or radius shell. Port
            # logs a warning by default; when
            # `params.sampling_params.strict_workspace=True`, raises
            # RuntimeError after logging (reference-conformant hard abort).
            _sp = self.params.sampling_params
            _wsp_xmin, _wsp_ymin = _sp.workspace_xy_min
            _wsp_xmax, _wsp_ymax = _sp.workspace_xy_max
            _wsp_zmin = _sp.workspace_z_min
            _wsp_zmax = _sp.workspace_z_max
            _r_min, _r_max = _sp.robot_radius_limits
            _viol = []
            if ee_pos_now[0] < _wsp_xmin: _viol.append(f"x<{_wsp_xmin:.3f}")
            if ee_pos_now[0] > _wsp_xmax: _viol.append(f"x>{_wsp_xmax:.3f}")
            if ee_pos_now[1] < _wsp_ymin: _viol.append(f"y<{_wsp_ymin:.3f}")
            if ee_pos_now[1] > _wsp_ymax: _viol.append(f"y>{_wsp_ymax:.3f}")
            if ee_pos_now[2] < _wsp_zmin: _viol.append(f"z<{_wsp_zmin:.3f}")
            if ee_pos_now[2] > _wsp_zmax: _viol.append(f"z>{_wsp_zmax:.3f}")
            # Radius shell on sqrt(x²+y²) — reference
            # cc:1488-1493 checks x²+y² against r_min² and r_max².
            _r_ee = float((ee_pos_now[0] * ee_pos_now[0]
                           + ee_pos_now[1] * ee_pos_now[1]) ** 0.5)
            if _r_ee < _r_min: _viol.append(f"r<{_r_min:.3f}")
            if _r_ee > _r_max: _viol.append(f"r>{_r_max:.3f}")
            if _viol:
                _msg = (f"[WORKSPACE-VIOLATION] step={self._step} "
                        f"ee=({ee_pos_now[0]:+.4f},{ee_pos_now[1]:+.4f},"
                        f"{ee_pos_now[2]:+.4f}) r={_r_ee:.4f} "
                        f"violations=[{','.join(_viol)}]")
                print(_msg, flush=True)
                if _sp.strict_workspace:
                    raise RuntimeError(
                        "CheckForWorkspaceLimitViolations aborted run "
                        "(reference DRAKE_DEMAND intent); " + _msg)
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
            # First 30 ticks: emit every step for direction-commit debugging.
            # After that, throttle to every 20 as before.
            if self._step <= 30 or self._step % 20 == 0:
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
            # Reference sampling_based_c3_controller.cc:1189 resets progress
            # metrics on entry to repositioning. Without this, a freshly-entered
            # free stint carries the c3-stint's accumulated steps_since_improve,
            # so any brief free->c3 re-entry immediately re-triggers
            # kToReposUnproductive instead of getting a fresh grinding budget.
            self.progress.reset()
            if self.log_diag:
                print(f"[RICH-EXIT-REFRESH] step={self._step} "
                      f"mode {self._prev_mode}->{mode} reason={reason.name} "
                      f"forcing buffer refresh + clearing prev_repos + progress reset")
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
        if kind == "free_joint_recovery":
            # Cartesian-trap escape: joint-PD toward the cached IK branch
            # solution on fresh state (see PORT_FREE_STALL_JOINT_RECOVERY).
            _qe = kw["q_des"] - np.asarray(current_q[: self.n_u], float)
            return np.clip(
                kw["kp"] * _qe
                - kw["kd"] * np.asarray(current_v[: self.n_u], float),
                -87.0, +87.0)
        if kind == "c3_r7_direct":
            # R^7 full-plant direct-torque: re-evaluate feedforward + joint-PD
            # on fresh state at 1 kHz. base_mpc's cached _last_u_seq / last_x_seq
            # (this planner tick's solution) supply tau_ff and (q_des, v_des).
            u_opt, _ = self._compute_r7_c3_direct_torque(
                current_q, current_v, plant_ctx)
            return u_opt
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
        # Derive reference `pursued_target_source_` from the port's existing
        # mode + best_src labels and update the state field. Emitted below.
        from control.sampling_c3.mode_switch import pursued_from_label
        self._pursued_target_source = pursued_from_label(mode, best_src)
        # 2026-07-22: decision arithmetic so the reader can see WHY a
        # cheaper raw sample is rejected. `hyst_kind` labels which of the
        # {c3_to_repos, repos_to_c3, repos_to_repos} margin was applied;
        # `gap` is the value from _hysteresis(); `lhs vs rhs` is the
        # exact comparison the mode-switch used.
        _hyst_kind = getattr(self, "_last_hyst_kind", "?")
        _hyst_gap  = getattr(self, "_last_hyst_gap", float("nan"))
        _near      = getattr(self, "_last_near_goal", False)
        if mode == "c3" and self._prev_mode == "c3":
            # stay-in-c3 arithmetic: best_other + gap < c3 was FALSE
            _lhs = float(best_other_cost) if best_other_cost != float("inf") else float("nan")
            _rhs = float(c_samples[0])
            _cmp = f"best_other({_lhs:.2f})+gap({_hyst_gap:.2f})={_lhs+_hyst_gap:.2f} vs c3({_rhs:.2f})"
        elif mode == "free" and self._prev_mode == "free":
            # stay-in-free arithmetic: best_other > curr + gap was FALSE
            _lhs = float(best_other_cost) if best_other_cost != float("inf") else float("nan")
            _rhs = float(c_samples[0])
            _cmp = f"best_other({_lhs:.2f}) vs curr({_rhs:.2f})+gap({_hyst_gap:.2f})={_rhs+_hyst_gap:.2f}"
        else:
            _cmp = "transition"
        print(f"[GS] step={step} mode={mode} switch={switch_reason.name} "
              f"best_k={best_k} best_src={best_src} "
              f"pursued={self._pursued_target_source.name} "
              f"curr_cost={c_samples[0]:.2f} repos_cost={repos_cost_str} "
              f"best_other={best_other_str} "
              f"met_progress={'Y' if met_progress else 'N'} "
              f"steps_since_improve={steps_since_improve} "
              f"switches={self._n_switches} "
              f"hyst[{_hyst_kind}]={_hyst_gap:.2f}({'pose' if _near else 'pos'}) "
              f"decision: {_cmp}")

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
