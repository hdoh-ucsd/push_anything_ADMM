"""
C3+ MPC Controller — Bui 2026 ICRA §IV-B.2 slack-variable variant.

Adds a slack variable η to the LCS so that the complementarity condition
0 ≤ λ ⊥ η ≥ 0 is expressed as a hard equality
    η_t = E x_t + F λ_t + H u_t + c
inside the QP, rather than the soft penalty C3 uses. The δ-update becomes
the closed-form Bui 2026 eq (12) C3+ projection on (λ, η)
pairs (see C3Solver._project_C3Plus) instead of C3's Lorentz cone
projection on (λ_n, λ_t).

This file owns the C3+-specific control loop:
  1. Linearise plant via LCSFormulator.linearize_discrete_with_complementarity
     (returns A, B, D, d, E, F, H, c, J_n, J_t, φ, μ).
  2. Build cost (same as C3 — Q, R, QN, x_ref from QuadraticManipulationCost).
  3. Forward the (E, F, H, c) tuple to C3Solver.solve, which dispatches
     to its mode='c3plus' branch.
  4. Apply u_opt[0] (receding horizon).

For the baseline C3 controller see control/ci_mpc_c3.py. The two share
the same C3Solver class (selected by mode), the same QuadraticManipulationCost,
and the same [MATH.cost] / [FORCE] diagnostic helpers (imported below).

v1 caveat: the formulator populates only the *normal* rows of E, F, H, c.
The four polyhedral tangent rows are zero, which means η_t ≡ 0 is forced
by the QP and friction is unenforced in η. See
milestones/4_c3plus_math/MATH_C3PLUS.md §1.2 for the closed-form
derivation and §5 for the implications on box motion.
"""
import os
import numpy as np

from control.ci_mpc_c3 import log_math_cost, log_force_diag_once


class C3PlusMPC:
    """
    Parameters
    ----------
    formulator    : LCSFormulator  — must support
                                     linearize_discrete_with_complementarity()
    solver        : C3Solver       — must be configured with mode='c3plus'
    quadratic_cost: QuadraticManipulationCost — builds Q, R, QN, x_ref
    horizon       : int    planning horizon (steps)
    dt            : float  planning timestep (s)
    torque_limit  : float  joint torque clamp (Nm)
    admm_iter     : int    ADMM iterations per control step
    """

    def __init__(self,
                 formulator,
                 solver,
                 quadratic_cost,
                 horizon:      int   = 8,
                 dt:           float = 0.03,
                 dt_pose:      float = None,
                 torque_limit: float = 30.0,
                 admm_iter:    int   = 10,
                 math_diag:    bool  = False,
                 use_ee_space: bool  = False,
                 ee_velocity_bounds: tuple | None = None):
        assert getattr(solver, 'mode', None) == 'c3plus', (
            "C3PlusMPC requires a C3Solver with mode='c3plus'. "
            "Use control.ci_mpc_c3.C3MPC for the baseline C3 path."
        )
        self.formulator    = formulator
        self.solver        = solver
        self.quad_cost     = quadratic_cost
        self.horizon       = horizon
        self.dt            = dt
        # Near-goal (pose regime) planning dt. Mirrors reference push_t
        # `planning_dt_pose: 0.05` — finer temporal resolution when box is
        # inside cost_switching_threshold_distance, so the planner can
        # brake a fast-moving box in time. None → falls back to dt (=
        # bit-identical to prior behavior when the caller doesn't set it).
        self.dt_pose       = float(dt_pose) if dt_pose is not None else dt
        # Mutable per-tick flag written by the wrapper (SamplingC3Controller).
        # When True, compute_control uses dt_pose + POSE u-limits.
        self._crossed_switching_threshold = False
        # When use_ee_space=True, `torque_limit` is reinterpreted as the
        # EE-force limit (Newtons), since u is now R^3 EE Cartesian force.
        # The downstream OSC realizes the joint torques.
        self.torque_limit  = torque_limit
        self.admm_iter     = admm_iter
        # Stage D: when True, the planner runs in the EE-space LCS
        # (Push-Anything §IV-A) with x ∈ ℝ^19 and u ∈ ℝ^3.
        # When False, the legacy R^7 joint-torque path remains.
        self.use_ee_space  = bool(use_ee_space)
        # Banner-vs-reality guard: assert solver dims match the active
        # formulation in BOTH directions. A label that fools the reviewer
        # is a correctness hazard; the only thing that can prevent a future
        # "ran R^7 but banner said EE-space" confound is a hard-fail here.
        if self.use_ee_space:
            assert solver.n_x == 19 and solver.n_u == 3, (
                f"use_ee_space=True requires C3Solver(n_x=19, n_u=3); "
                f"got n_x={solver.n_x}, n_u={solver.n_u}. "
                f"The active LCS path is linearize_discrete_ee_space, which "
                f"produces 19-dim state and 3-dim input. Mismatch would "
                f"silently run the wrong dimensions."
            )
        else:
            # Under R^7, expected n_x = formulator.n_q + formulator.n_v and
            # n_u = formulator.n_u (Drake plant DOFs). The expected values are
            # DERIVED from the formulator instance, not hardcoded — so any
            # future plant change (e.g. different arm) propagates here too.
            _expected_n_x = formulator.n_q + formulator.n_v
            _expected_n_u = formulator.n_u
            assert solver.n_x == _expected_n_x and solver.n_u == _expected_n_u, (
                f"R^7 (use_ee_space=False) requires "
                f"C3Solver(n_x={_expected_n_x}, n_u={_expected_n_u}) "
                f"derived from formulator's plant DOFs; got "
                f"n_x={solver.n_x}, n_u={solver.n_u}."
            )
        # Print the verified dims with their formulation label so a log
        # reader sees both at once and can't misread one for the other.
        print(f"[C3+] planner construction verified: "
              f"use_ee_space={self.use_ee_space} "
              f"solver.n_x={solver.n_x} solver.n_u={solver.n_u}  "
              f"({'EE force (Newtons)' if self.use_ee_space else 'joint torque (Nm)'})",
              flush=True)
        self._math_diag        = math_diag
        self._mpc_step         = 0
        self._math_setup_done  = False
        self._printed_force_diag = False
        # Reference SamplingC3Controller::ClampEndEffectorAcceleration
        # (sampling_based_c3_controller.cc:1457-1472). Each tick, the LCS x0's
        # EE-position and EE-velocity slots are clamped to
        # x_pred_curr_plan_[i] ± nominal_ee_accel · dt² (position) or dt
        # (velocity). Prevents the LCS from seeing large EE-state jumps
        # (which manifested in results/push_t_iter9_orient_20260718_145158
        # as arm flying 1.5 m from T while LCS still reported phantom
        # contact λ_n=1.7).
        #
        # Enabled by default; disable via nominal_ee_accel=0 or explicit flag.
        # Reference push_t/parameters/sampling_c3plus_options.yaml:66
        # nominal_ee_accel=2 (inherited from anything). No clamp applies on
        # the first tick (no previous plan).
        self.nominal_ee_accel  = 2.0
        self._x_pred_curr_plan = None
        # Reference sampling_c3plus_options.yaml:14 solve_time_filter_alpha,
        # cc:1394-1397 `filtered_solve_time_ = (1-alpha)·solve_time +
        # alpha·prev`. Reference feeds this filtered wall-time into
        # ClampEndEffectorAcceleration (cc:1460 `approx_loop_dt =
        # min(0.1, filtered_solve_time_)`). Port previously used the
        # fixed planning_dt as `_dt_c`, which gave delta_pos = 5 mm for
        # push_t (planning_dt_pose=0.05) vs reference's ~18 mm at the
        # port's actual ~96 ms wall time. Result: port CLIPS x0 to
        # current more often; reference lets x_pred_curr_plan_ drive
        # x0 more (more anticipatory).
        # Initial value = planning_dt so the first solve behaves like
        # the prior (planning_dt-based) clamp until wall-time samples
        # arrive.
        self._solve_time_filter_alpha = 0.95
        self._filtered_solve_time     = float(self.dt)
        # Reference sampling_c3plus_options.yaml:36 ee_velocity_limits.
        # Reference cc:1027-1034 applies as AddLinearConstraint(..., STATE)
        # on state slots (n_q_+0, n_q_+1, n_q_+2) at each knot. In the port's
        # EE-space layout, these are the EE velocity slots at indices
        # (16, 17, 18). Anything default: [-0.14, 0.14] m/s.
        # None → constraint disabled (byte-identical to prior behavior).
        self.ee_velocity_bounds = ee_velocity_bounds
        # EE-space state layout (must match ci_mpc_c3plus.py:320 concatenation):
        #   [box_q(7), p_ee(3), box_v(6), v_ee(3)]  →  slices below.
        self._EE_POS_SLICE = slice(7, 10)
        self._EE_VEL_SLICE = slice(16, 19)

        # Last predicted trajectory — set after every solve, used for Meshcat viz
        self.last_x_seq: np.ndarray | None = None   # (N+1, n_x)
        # Final-QP x copy (reference GetStateSolution/x_sol_) — the copy the
        # reference's UpdateC3ExecutionTrajectory consumers track with the
        # OSC (cc:1701-1732). Published x_seq stays the z copy (CalcCost
        # source). REFCONF_OSC_TARGET_QP_COPY=0 reverts consumers to x_seq.
        self.last_x_qp_seq: np.ndarray | None = None   # (N+1, n_x)
        self._osc_target_qp_copy = (
            os.environ.get("REFCONF_OSC_TARGET_QP_COPY", "1") == "1")
        # Previous solve's u_seq[0] for the next-step linearization (Aydinoglu eq. 8).
        self._last_u: np.ndarray = np.zeros(solver.n_u)
        # First-horizon planned contact force λ_d (Aydinoglu eq. 36 feedforward).
        # Mirrored from solver._last_lambda_{n,t}_first after each solve.
        self.last_lambda_n_first: np.ndarray | None = None
        self.last_lambda_t_first: np.ndarray | None = None
        # T-architecture Stage 1: full λ horizon mirrored from solver. Shape
        # (N, num_normals) and (N, 4*num_normals). Set by every solve; Stage 2
        # will let the wrapper's OSC index into these between MPC re-solves.
        self.last_lambda_n_horizon: np.ndarray | None = None
        self.last_lambda_t_horizon: np.ndarray | None = None
        # D2: convergence flag for the wrapper's force-derive degrade.
        # True until the first solve (so a missing flag doesn't trip the cap).
        self.last_converged: bool = True

    def _emit_reference_plan_diag(self, x_seq, u_seq) -> None:
        """Mirror the reference's verbose plan diagnostics.

        sampling_based_c3_controller.cc:1344-1378 prints, per knot:
          - "Right side of complementarity": E x + F λ + H u + c
          - "Complementarity violation":     λ · (E x + F λ + H u + c)
          - "Dynamically feasible ee current plan":     x[0:3]
          - "Dynamically feasible object current plan": x[n_q-7 : n_q]
        The port's EE-space layout is x = [box_q(7), p_ee(3), box_v(6),
        v_ee(3)], so the EE slice is [7:10] and the object pose slice is
        [0:7] — the reference's own indices are NOT copied verbatim (they
        assume its actor-first layout); the SLOTS are matched instead.
        """
        try:
            _f = self.formulator
            E = getattr(_f, "_last_E", None)
            F = getattr(_f, "_last_F", None)
            H = getattr(_f, "_last_H", None)
            c = getattr(_f, "_last_c", None)
            lam_h = getattr(self.solver, "_last_lambda_anitescu_horizon", None)
            N = len(u_seq)
            print(f"[C3-PLAN] step={self._mpc_step} "
                  f"--- reference verbose plan dump (cc:1344-1378) ---",
                  flush=True)
            if E is not None and lam_h is not None and lam_h.size:
                print("[C3-PLAN] right side of complementarity "
                      "(E x + F lam + H u + c), per knot:", flush=True)
                for i in range(min(N, len(lam_h))):
                    _eta = (E @ x_seq[i] + F @ lam_h[i]
                            + H @ u_seq[i] + c)
                    _viol = float(np.dot(lam_h[i], _eta))
                    print(f"[C3-PLAN]   k={i} eta_min={_eta.min():+.5f} "
                          f"eta_max={_eta.max():+.5f} "
                          f"|lam|={np.linalg.norm(lam_h[i]):.4f} "
                          f"violation=lam.eta={_viol:+.5f}", flush=True)
            # Plan knots: EE (slot 7:10) and object pose (slot 0:7).
            for _tag, _sl in (("ee", slice(7, 10)), ("object", slice(0, 7))):
                print(f"[C3-PLAN] {_tag} current plan, per knot:", flush=True)
                for i in range(len(x_seq)):
                    _v = np.asarray(x_seq[i][_sl], dtype=float)
                    print("[C3-PLAN]   k=%d %s" % (
                        i, " ".join(f"{q:+.5f}" for q in _v)), flush=True)
        except Exception as _e:      # diagnostic must never break a run
            print(f"[C3-PLAN] diag error {type(_e).__name__}: {_e}",
                  flush=True)

    def compute_control(self,
                        current_q:  np.ndarray,
                        current_v:  np.ndarray,
                        plant_ctx,
                        target_xy:  np.ndarray,
                        target_yaw: float = 0.0) -> np.ndarray:
        """
        Compute one torque command via C3+ trajectory optimisation.
        Same signature as C3MPC.compute_control — interchangeable to the caller.
        """
        plant = self.formulator.plant
        plant.SetPositions(plant_ctx, current_q)
        plant.SetVelocities(plant_ctx, current_v)

        self._mpc_step += 1

        # Near-goal (pose regime) dt swap. Mirrors reference push_t
        # `GetC3Options(crossed_cost_switching_threshold_)` returning
        # a C3Options with a different `planning_dt`. Finer resolution
        # near goal lets the planner react to fast box velocity in time.
        _dt = self.dt_pose if self._crossed_switching_threshold else self.dt
        # One-shot pose regime activation diagnostic. Fires the first
        # tick after crossed_switching_threshold latches.
        if (self._crossed_switching_threshold
                and not getattr(self, "_pose_regime_logged", False)):
            self._pose_regime_logged = True
            print(f"[POSE-REGIME] step={self._mpc_step} activated: "
                  f"dt {self.dt:.3f}→{self.dt_pose:.3f}s", flush=True)

        # 1. Linearise Drake plant into discrete LCS + slack expression, around
        # the previous solve's u[0] (Aydinoglu 2024 eq. 8 linearization point).
        if self.use_ee_space:
            (A, B_ctrl, D, d,
             E_lcs, F_lcs, H_lcs, c_lcs,
             J_n, J_t, phi, mu) = \
                self.formulator.linearize_discrete_ee_space(
                    plant_ctx, _dt, u_lin=self._last_u)
        else:
            (A, B_ctrl, D, d,
             E_lcs, F_lcs, H_lcs, c_lcs,
             J_n, J_t, phi, mu) = \
                self.formulator.linearize_discrete_with_complementarity(
                    plant_ctx, _dt, u_lin=self._last_u)

        # ---- [MATH.setup] fires ONCE on first MPC step ----------------------
        if self._math_diag and not self._math_setup_done:
            self._math_setup_done = True
            # Stewart-Trinkle: λ = [γ; λ_n; λ_t] = 2·n_c + 4·n_c = 6·n_c.
            # Earlier this print computed n_c + 4·n_c = 5·n_c, missing γ; the
            # solver itself (admm_solver.py:752) uses the correct 6·n_c sizing.
            n_lambda = 2 * J_n.shape[0] + J_t.shape[0]
            # C3+ doubles the per-step block to carry η alongside λ.
            TOT      = self.solver.n_x + n_lambda + self.solver.n_u + n_lambda
            total    = self.horizon * TOT + self.solver.n_x
            qc       = self.quad_cost
            print(f"[MATH.setup] mode=c3plus  Horizon N={self.horizon}, "
                  f"dt={self.dt}s ({self.horizon * self.dt:.1f}s lookahead)")
            print(f"[MATH.setup] State dim n_x={self.solver.n_x}, "
                  f"control dim n_u={self.solver.n_u}, "
                  f"contact dim n_λ={n_lambda}, slack dim n_η={n_lambda}")
            print(f"[MATH.setup] Total QP variable dim: z ∈ R^{total}"
                  f"  (= N·TOT+n_x = {self.horizon}·{TOT}"
                  f"+{self.solver.n_x})")
            print(f"[MATH.setup] Cost weights (from tasks.yaml):")
            print(f"[MATH.setup]   w_obj_xy={qc.w_obj_xy:.1f}  "
                  f"w_obj_z={qc.w_obj_z:.1f}  "
                  f"w_box_z={qc.w_box_z:.1f}  "
                  f"w_box_rp={qc.w_box_rp:.1f}  "
                  f"w_torque={qc.w_torque}  "
                  f"w_terminal={qc.w_terminal:.1f}")
            print(f"[MATH.setup] ADMM: rho_0={self.solver.rho:.1f}, "
                  f"rho_clamp=[0.1, 1000.0], "
                  f"max_iter={self.admm_iter}, tol=1e-3")
            print(f"[MATH.setup] Soft complementarity: w_comp={self.solver._w_comp:.1f}"
                  f"  (disabled in C3+ — replaced by η equality)")
            print(f"[MATH.setup-C3+] u_λ={self.solver._u_lambda:.3f}  "
                  f"u_η={self.solver._u_eta:.3f}  "
                  f"√(u_λ/u_η)={np.sqrt(self.solver._u_lambda / self.solver._u_eta):.3f}  "
                  f"w_G_ee_contact={self.solver._w_G_ee_contact:.1f} "
                  f"(declared, NOT applied in v1)")
            # μ may be scalar or ndarray (per-pair, ref
            # sampling_c3plus_options.yaml:44 mu_per_pair_type). Format
            # generically.
            _mu_str = (f"{float(mu):.4f}" if np.isscalar(mu)
                       or (hasattr(mu, "ndim") and mu.ndim == 0)
                       else np.array2string(np.asarray(mu),
                                            precision=4, suppress_small=True))
            print(f"[MATH.setup] Friction coefficient μ={_mu_str}")
            print(f"[MATH.setup] Torque limit: ±{self.torque_limit:.1f} Nm")

        # ---- [MATH.LCS] step 1 (once-per-solve seed) + every 50 MPC steps ----
        if self._math_diag and (self._mpc_step == 1 or self._mpc_step % 50 == 0):
            nc_now     = J_n.shape[0] + J_t.shape[0]
            contact_on = J_n.shape[0] > 0
            phi_str    = ("  ".join(f"{v:.5f}" for v in phi)
                          if len(phi) > 0 else "(none)")
            print(f"[MATH.LCS] step={self._mpc_step}, "
                  f"contact active: {'Y' if contact_on else 'N'}, "
                  f"n_c={nc_now}")
            print(f"[MATH.LCS] A  shape={A.shape}, "
                  f"norm(F)={np.linalg.norm(A):.4f}")
            print(f"[MATH.LCS] B  shape={B_ctrl.shape}, "
                  f"norm(F)={np.linalg.norm(B_ctrl):.4f}")
            print(f"[MATH.LCS] D  shape={D.shape}, "
                  f"norm(F)={np.linalg.norm(D):.4f}"
                  f"  ← couples contact force to state")
            print(f"[MATH.LCS] d  shape={d.shape}, "
                  f"norm(F)={np.linalg.norm(d):.4f}")
            if J_n.shape[0] > 0:
                print(f"[MATH.LCS] J_n shape={J_n.shape}, "
                      f"J_n[0,:5]={np.round(J_n[0, :min(5, J_n.shape[1])], 5).tolist()}"
                      f"  ← normal contact Jacobian")
                print(f"[MATH.LCS] J_t shape={J_t.shape}, "
                      f"J_t[0,:5]={np.round(J_t[0, :min(5, J_t.shape[1])], 5).tolist()}"
                      f"  ← tangent contact Jacobian")
            else:
                print(f"[MATH.LCS] J_n/J_t: empty (no contacts within 0.10m threshold)")
            print(f"[MATH.LCS] φ (SDF gap): [{phi_str}] m")
            # Slack-equality matrices (E, F, H, c) — the C3+ delta over C3.
            # zero-rows count is the v1 "deferred friction" indicator: the four
            # polyhedral tangent rows of E, F, H, c are populated as zero, so
            # η_t is forced to 0 by the QP and friction is unenforced.
            if E_lcs is not None and E_lcs.size > 0:
                row_norms = np.linalg.norm(E_lcs, axis=1)
                zero_rows = int(np.sum(row_norms < 1e-12))
                frac_zero = zero_rows / E_lcs.shape[0]
                print(f"[MATH.LCS] E   shape={E_lcs.shape}, "
                      f"norm(F)={np.linalg.norm(E_lcs):.4f}, "
                      f"zero-rows={zero_rows}/{E_lcs.shape[0]} "
                      f"({100*frac_zero:.1f}%)  "
                      f"← v1: tangent rows zeroed → friction unenforced in η")
                print(f"[MATH.LCS] F   shape={F_lcs.shape}, "
                      f"norm(F)={np.linalg.norm(F_lcs):.4f}")
                print(f"[MATH.LCS] H   shape={H_lcs.shape}, "
                      f"norm(F)={np.linalg.norm(H_lcs):.4f}")
                print(f"[MATH.LCS] c   shape={c_lcs.shape}, "
                      f"norm={np.linalg.norm(c_lcs):.4f}")

        # Per-(solve, iter, k) λ probe — first 5 C3+ solves with contact tags
        # from the formulator. Diagnoses whether λ_n_gnd reaches the gravity-
        # support level m·g across the horizon (regime A vs B).
        if (self._math_diag
                and getattr(self.solver, "_lprobe_path", None) is None
                and J_n.shape[0] > 0):
            _tags = [info.get("tag", "?")
                     for info in self.formulator._last_contact_info]
            self.solver.enable_lambda_horizon_probe(
                path="audit_output/lambda_horizon_trace.csv",
                contact_tags=_tags,
                max_solves=5,
            )
            print(f"[LAMBDA-PROBE] enabled; tags={_tags}  "
                  f"path=audit_output/lambda_horizon_trace.csv  max_solves=5")

        # 2. Quadratic cost and reference state (with linearised EE approach)
        if self.use_ee_space:
            Q, R, QN, x_ref = self.quad_cost.build_ee_space(
                target_xy, plant_ctx=plant_ctx, current_q=current_q,
                target_yaw=target_yaw,
            )
        else:
            Q, R, QN, x_ref = self.quad_cost.build(
                target_xy, plant_ctx=plant_ctx, current_q=current_q,
                rich_mode=True, target_yaw=target_yaw,
            )

        # 3. Current state x0 — in EE-space mode build [box_q, p_ee, box_v, v_ee].
        if self.use_ee_space:
            BOX_Q_START = self.formulator._obj_body.floating_positions_start()
            BOX_V_START = self.formulator._obj_body.floating_velocities_start_in_v()
            box_q = current_q[BOX_Q_START : BOX_Q_START + 7]
            box_v = current_v[BOX_V_START : BOX_V_START + 6]
            ee_body  = plant.GetBodyByName('pusher')
            p_ee_now = plant.CalcPointsPositions(
                plant_ctx, ee_body.body_frame(), np.zeros((3, 1)),
                plant.world_frame(),
            ).flatten()
            J_ee_full = plant.CalcJacobianTranslationalVelocity(
                plant_ctx,
                __import__('pydrake.all', fromlist=['JacobianWrtVariable'])
                .JacobianWrtVariable.kV,
                ee_body.body_frame(), np.zeros(3),
                plant.world_frame(), plant.world_frame(),
            )
            v_ee_now = J_ee_full @ current_v
            x0 = np.concatenate([box_q, p_ee_now, box_v, v_ee_now])
            # ClampEndEffectorAcceleration: keep x0's EE slots within a
            # bounded band around the previously-planned trajectory.
            # Mirrors reference cc:1457-1472. Only fires when a previous plan
            # exists AND clamping is enabled (nominal_ee_accel > 0).
            # Reference cc:1460 `approx_loop_dt = min(0.1,
            # filtered_solve_time_)` — use actual wall-clock filtered
            # solve time, NOT the planning discretization dt. Prior port
            # used planning_dt (0.05 s for pose regime) giving
            # delta_pos = 5 mm; reference at port's ~96 ms wall time
            # would give delta_pos ~ 18 mm — reference allows the
            # prediction to drive x0 more often (more anticipatory).
            if (self._x_pred_curr_plan is not None
                    and self.nominal_ee_accel > 0.0):
                _dt_c = min(0.1, self._filtered_solve_time)
                _delta_pos = self.nominal_ee_accel * _dt_c * _dt_c
                _delta_vel = self.nominal_ee_accel * _dt_c
                _ee_pos_plan = self._x_pred_curr_plan[self._EE_POS_SLICE]
                _ee_vel_plan = self._x_pred_curr_plan[self._EE_VEL_SLICE]
                x0[self._EE_POS_SLICE] = np.clip(
                    _ee_pos_plan,
                    x0[self._EE_POS_SLICE] - _delta_pos,
                    x0[self._EE_POS_SLICE] + _delta_pos,
                )
                x0[self._EE_VEL_SLICE] = np.clip(
                    _ee_vel_plan,
                    x0[self._EE_VEL_SLICE] - _delta_vel,
                    x0[self._EE_VEL_SLICE] + _delta_vel,
                )
        else:
            x0 = np.concatenate([current_q, current_v])

        # Stage 5 per-axis u bounds (env-gated, default-inert). When the
        # EE-space planner is active and the env vars are set, override the
        # symmetric scalar torque_limit with per-axis bounds:
        #   u = [Fx, Fy, Fz];  Fx,Fy ∈ ±PORT_U_HORIZONTAL;
        #                      Fz   ∈ ±PORT_U_VERTICAL.
        # When EITHER env var is unset the scalar torque_limit path is used
        # unchanged (bit-identical to pre-Stage-5).
        _u_lo = None
        _u_hi = None
        if self.use_ee_space and self.solver.n_u == 3:
            # Near-goal (pose regime) u-limit swap. Mirrors reference
            # push_t vs anything: push_t uses ±50 N horizontal, anything
            # uses ±10 N. When box is near goal, higher force limits let
            # the planner brake a fast-moving box in time. When
            # POSE env vars are unset OR the near-goal flag hasn't
            # latched, the base (position-regime) values apply.
            if (self._crossed_switching_threshold
                    and os.environ.get("PORT_U_HORIZONTAL_POSE", "")
                    and os.environ.get("PORT_U_VERTICAL_POSE", "")):
                _uh_s = os.environ.get("PORT_U_HORIZONTAL_POSE", "")
                _uv_s = os.environ.get("PORT_U_VERTICAL_POSE", "")
            else:
                _uh_s = os.environ.get("PORT_U_HORIZONTAL", "")
                _uv_s = os.environ.get("PORT_U_VERTICAL", "")
            if _uh_s and _uv_s:
                try:
                    _uh = float(_uh_s)
                    _uv = float(_uv_s)
                    _u_lo = np.array([-_uh, -_uh, -_uv])
                    _u_hi = np.array([+_uh, +_uh, +_uv])
                    _u_src = "env"
                except ValueError:
                    _u_lo = None
                    _u_hi = None
            # 2026-08-04 u-limit conformance: flagless runs previously fell
            # through to the scalar torque_limit (±87 N/axis on a Cartesian
            # force!) — off-reference for BOTH tasks (push_t ±50/±50,
            # anything ±10/±3 per sampling_c3plus_options.yaml:34-35). The
            # per-task yaml values (plumbed via params →
            # _u_horizontal_cfg/_u_vertical_cfg) are now the flagless
            # default; env vars above remain a falsification override.
            if _u_lo is None:
                _uh_c = getattr(self, "_u_horizontal_cfg", None)
                _uv_c = getattr(self, "_u_vertical_cfg", None)
                if _uh_c is not None and _uv_c is not None:
                    _u_lo = np.array([-_uh_c, -_uh_c, -_uv_c])
                    _u_hi = np.array([+_uh_c, +_uh_c, +_uv_c])
                    _u_src = "yaml"
            if _u_lo is not None and not getattr(
                    self, "_u_limits_banner", False):
                self._u_limits_banner = True
                print(f"[U-LIMITS] per-axis u-box active "
                      f"(source={_u_src}): Fx,Fy ∈ ±{_u_hi[0]:.0f}N "
                      f"Fz ∈ ±{_u_hi[2]:.0f}N (reference "
                      f"u_horizontal/vertical_limits)", flush=True)
        elif not self.use_ee_space and self.solver.n_u > 3:
            # 2026-07-28 defaults flip: gravity-centered u-box unconditional
            # (was REFCONF_R7_U_GRAVITY_CENTERED, default ON).
            # R^7 gravity-centered u-box (reference u-limit conformance).
            # The reference bounds PUSH EFFORT: u is a ~N-scale EE force with
            # u_horizontal/vertical_limits = ±50 N (push_t
            # sampling_c3plus_options.yaml:34-35); gravity is not in its u.
            # The port's R^7 u is joint torque against a gravity-included
            # LCS, so u must CONTAIN the gravity-holding torque −τ_g — which
            # at the working posture is −34.1 Nm on joint 2, OUTSIDE the
            # symmetric ±30 box entirely: the planner could not even hold
            # the arm within its own bound, saturating joint 2 on every
            # solve and leaving ~zero push headroom (p110 diagnosis: 232
            # correct-face c3 steps produced +19 mm of the +187 mm needed).
            # Conformant translation: center the box on the gravity-holding
            # torque and give it the torque image of the reference force
            # limits through the arm Jacobian at the linearization point:
            #   u ∈ [u_hold − Δ, u_hold + Δ],  u_hold = −τ_g_arm(q*),
            #   Δ_j = (|J_arm|ᵀ · F_ref)_j,   F_ref = [50, 50, 50] N,
            # floored at 1 Nm per joint (wrist rows of J are ~0 — a
            # zero-width box would hard-pin those joints) and clipped to
            # the ±87 Nm Franka effort limit.
            _n_arm = int(self.solver.n_u)
            _tau_g_full = self.quad_cost.plant.CalcGravityGeneralizedForces(
                plant_ctx)
            _u_hold = -np.asarray(_tau_g_full[:_n_arm], dtype=float)
            _J_ee_u = self.quad_cost.plant.CalcJacobianTranslationalVelocity(
                plant_ctx, self.quad_cost._ad.JacobianWrtVariable.kV,
                self.quad_cost.ee_frame, np.zeros(3),
                self.quad_cost.world_frame, self.quad_cost.world_frame,
            )
            _F_ref = np.array([
                float(os.environ.get("PORT_U_HORIZONTAL", "50.0")),
                float(os.environ.get("PORT_U_HORIZONTAL", "50.0")),
                float(os.environ.get("PORT_U_VERTICAL",   "50.0")),
            ])
            _delta_u = np.abs(_J_ee_u[:, :_n_arm]).T @ _F_ref
            _delta_u = np.maximum(_delta_u, 1.0)
            _u_lo = np.maximum(_u_hold - _delta_u, -87.0)
            _u_hi = np.minimum(_u_hold + _delta_u, +87.0)
            if not getattr(self, "_r7_ubox_logged", False):
                self._r7_ubox_logged = True
                print(f"[REFCONF_R7_U_GRAVITY_CENTERED] u-box centered on "
                      f"u_hold={np.round(_u_hold, 1)} with "
                      f"Δ=|J|ᵀ·{_F_ref}={np.round(_delta_u, 1)} "
                      f"(replaces symmetric ±torque_limit)", flush=True)

        # §7.67 — B1-A plumbing: pass the EE-BOX contact index to the solver
        # (read from formulator's _last_contact_info tag list) so
        # _solve_c3plus can apply Bui §IV-B.2's final-iter G-weighting on
        # the load-bearing pair. Idx = position in the admitted-pair order;
        # None if no EE-BOX pair was admitted this tick (solver skips B1-A).
        _ee_box_idx = None
        _cinfo = getattr(self.formulator, "_last_contact_info", None)
        if _cinfo:
            for _i, _info in enumerate(_cinfo):
                if _info.get("tag", "") == "EE-BOX":
                    _ee_box_idx = _i
                    break
        self.solver._ee_box_pair_idx = _ee_box_idx
        # §7.67 — one-shot diagnostic: which tags did we see + n_c
        if not getattr(self, "_b1a_tag_dump_done", False):
            self._b1a_tag_dump_done = True
            _tags = ([_i.get("tag", "?") for _i in _cinfo]
                     if _cinfo else "(_cinfo None or empty)")
            _n_cinfo = len(_cinfo) if _cinfo else 0
            _n_jn = int(J_n.shape[0]) if J_n is not None else -1
            print(f"[§7.67 B1-A PLUMB] first-solve mpc_step={self._mpc_step} "
                  f"n_contact_info={_n_cinfo}  n_J_n={_n_jn}  "
                  f"tags={_tags}  ee_box_idx={_ee_box_idx}", flush=True)

        # 4. Full-horizon C3+ ADMM solve — forwards slack expression (E, F, H, c)
        import time as _time_solve
        _t0_solve = _time_solve.perf_counter()
        u_seq, x_seq = self.solver.solve(
            x0, A, B_ctrl, D, d, J_n, J_t, mu,
            Q, R, QN, x_ref,
            N=self.horizon,
            admm_iter=self.admm_iter,
            torque_limit=self.torque_limit,
            phi=phi,
            E=E_lcs, F=F_lcs, H=H_lcs, c_lcs=c_lcs,
            u_lower=_u_lo, u_upper=_u_hi,
            ee_velocity_bounds=self.ee_velocity_bounds,
        )
        _solve_wall_s = _time_solve.perf_counter() - _t0_solve
        # Reference sampling_based_c3_controller.cc:1408-1418 filter.
        # solve_time_filter_alpha = 0.95 (sampling_c3plus_options.yaml:14).
        # 2026-08-15 leg 5: the reference filters the FULL control-loop
        # wall time, not just this committed solve — when the wrapper
        # (SamplingC3MPC) drives the EMA with its full-tick wall at the
        # end of each tick, skip the committed-only update here (it
        # under-measures by the sample-evaluation cost and parked the
        # consumption depth exactly on the final-QP-pinned first knot).
        # Standalone base-MPC use (no wrapper) keeps the local update.
        if not getattr(self, "_fst_source_full_tick", False):
            _alpha = self._solve_time_filter_alpha
            self._filtered_solve_time = (
                (1.0 - _alpha) * _solve_wall_s
                + _alpha * self._filtered_solve_time
            )
        # First-few-solves diagnostic so the parameter is visible in logs.
        if not hasattr(self, "_filtered_solve_time_logged"):
            self._filtered_solve_time_logged = 0
        if self._filtered_solve_time_logged < 3:
            _dt_c_next = min(0.1, self._filtered_solve_time)
            _delta_pos_next = self.nominal_ee_accel * _dt_c_next * _dt_c_next
            print(f"[C3+] filtered_solve_time={self._filtered_solve_time*1000:.2f}ms "
                  f"(this solve={_solve_wall_s*1000:.2f}ms alpha={_alpha:.2f}) "
                  f"next-tick clamp: _dt_c={_dt_c_next*1000:.2f}ms "
                  f"delta_pos={_delta_pos_next*1000:.2f}mm  "
                  f"(ref cc:1394,1460)", flush=True)
            self._filtered_solve_time_logged += 1

        # ---- Reference verbose diagnostics (sampling_based_c3_controller.cc
        # :1344-1378). Mirrors, in the reference's own order and formulas:
        #   "Right side of complementarity:"  E x + F λ + H u + c   per knot
        #   "Complementarity violation:"      λ · (E x + F λ + H u + c)
        #   "Dynamically feasible ee/object current plan:"  per-knot EE and
        #                                                   object plan
        # Enable with DIAG_C3_PLAN=1 (optionally DIAG_C3_PLAN_AT_STEP=N to
        # emit only at one tick). Off = zero cost.
        if os.environ.get("DIAG_C3_PLAN", "0") == "1":
            _at = os.environ.get("DIAG_C3_PLAN_AT_STEP", "")
            if (not _at) or int(_at) == int(self._mpc_step):
                self._emit_reference_plan_diag(x_seq, u_seq)

        # 5. Store predicted trajectory + u[0] for next-step linearization
        self.last_x_seq = x_seq        # (N+1, n_x)
        # QP-copy mirror (4.t stash): reference x_sol_ for the OSC-target /
        # x_pred consumers. None on solver paths that don't stash (Lorentz).
        _x_qp_h = getattr(self.solver, "_last_x_qp_horizon", None)
        self.last_x_qp_seq = _x_qp_h.copy() if _x_qp_h is not None else None
        self._last_u    = u_seq[0].copy()
        # Save x_pred_curr_plan_ for next tick's ClampEndEffectorAcceleration.
        # Reference sampling_based_c3_controller.cc:1723-1732 sets this by
        # linear interpolation into the plan trajectory at fraction
        # (filtered_solve_time / dt):
        #     last_idx = filtered_solve_time / dt
        #     frac     = (filtered_solve_time / dt) - last_idx
        #     x_pred_curr_plan = knots[last_idx] + frac·(knots[last_idx+1] - knots[last_idx])
        # This anticipates where the arm SHOULD be by the time the next tick
        # fires (accounting for solve wall time). Prior port cached x_seq[1]
        # directly — TIME-BEHIND reality when wall time > dt, causing the
        # clamp to fight nondeterminism instead of damping it.
        # For EE-space runs only; joint-torque path skips clamp entirely.
        if self.use_ee_space and x_seq is not None and len(x_seq) > 1:
            # Reference cc:1723-1732 interpolates into the SAME knots the
            # OSC tracks — UpdateC3ExecutionTrajectory's x_sol (final-QP)
            # copy, not the published z copy. Fall back to x_seq when the
            # QP copy is unavailable or the wiring is switched off.
            _x_knots = (self.last_x_qp_seq
                        if (self._osc_target_qp_copy
                            and self.last_x_qp_seq is not None
                            and len(self.last_x_qp_seq) > 1)
                        else x_seq)
            _dt_pred = self.dt_pose if self._crossed_switching_threshold else self.dt
            _idx_f = float(self._filtered_solve_time) / _dt_pred
            _last_idx = int(_idx_f)
            _frac = _idx_f - _last_idx
            _N_knots = len(_x_knots)
            if _last_idx < _N_knots - 1:
                _pred = (np.asarray(_x_knots[_last_idx], dtype=float)
                         + _frac * (np.asarray(_x_knots[_last_idx + 1], dtype=float)
                                    - np.asarray(_x_knots[_last_idx], dtype=float)))
            else:
                _pred = np.asarray(_x_knots[_N_knots - 1], dtype=float)
            self._x_pred_curr_plan = _pred.copy()
        # Plumb first-horizon λ for the impedance controller's feedforward
        # contact-force term. None until the first solve produces them.
        self.last_lambda_n_first = (
            self.solver._last_lambda_n_first.copy()
            if self.solver._last_lambda_n_first is not None else None
        )
        self.last_lambda_t_first = (
            self.solver._last_lambda_t_first.copy()
            if self.solver._last_lambda_t_first is not None else None
        )
        # T-architecture Stage 1: mirror the full λ horizons too. None until
        # the first solve produces them.
        self.last_lambda_n_horizon = (
            self.solver._last_lambda_n_horizon.copy()
            if self.solver._last_lambda_n_horizon is not None else None
        )
        self.last_lambda_t_horizon = (
            self.solver._last_lambda_t_horizon.copy()
            if self.solver._last_lambda_t_horizon is not None else None
        )
        # D2: surface ADMM convergence to the wrapper's force-derive logic.
        # When False, wrapper caps the OSC λ_des magnitude at nominal so we
        # don't amplify ω-leakage (delta view) or complementarity-leakage
        # (z_sol view). True by default — only set False on actual divergence.
        self.last_converged = bool(getattr(self.solver, "_last_converged", True))
        # Stage C probe B [CONSISTENCY] — forward ADMM terminal-state fields.
        self.last_pr_final   = float(getattr(self.solver, "_last_pr_final",  float("nan")))
        self.last_dr_final   = float(getattr(self.solver, "_last_dr_final",  float("nan")))
        self.last_iters_used = int(  getattr(self.solver, "_last_iters_used", 0))
        self.last_tol        = float(getattr(self.solver, "_last_tol",       1e-3))
        # Stash cost-build outputs for the wrapper's [COST-DUMP] diagnostic
        # (purely additive — read only by one-shot logging).
        self._last_Q          = Q
        self._last_R          = R
        self._last_QN         = QN
        self._last_x_ref      = x_ref
        self._last_target_xy  = np.asarray(target_xy, dtype=float).copy()
        self._last_u_seq      = u_seq
        self._last_plant_ctx  = plant_ctx
        self._last_current_q  = np.asarray(current_q, dtype=float).copy()

        # ---- [MATH.cost] every 50 MPC steps ----------------------------------
        if self._math_diag and self._mpc_step % 50 == 0:
            log_math_cost(self.quad_cost, self.formulator, self.solver,
                          self.horizon, x_seq, x_ref, u_seq, Q, R, QN,
                          self._mpc_step)

        # 6. Planned contact force diagnostic (one-time, at first contact)
        if log_force_diag_once(self._printed_force_diag,
                               self.formulator, self.quad_cost,
                               current_q, target_xy,
                               x_seq, A, B_ctrl, d, u_seq, D, J_n):
            self._printed_force_diag = True

        # 7. Receding horizon: return u[0], clipped to torque limit
        return np.clip(u_seq[0], -self.torque_limit, self.torque_limit)
