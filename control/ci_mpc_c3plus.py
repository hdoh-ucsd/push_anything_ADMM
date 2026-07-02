"""
C3+ MPC Controller — Bui 2026 ICRA §IV-B.2 slack-variable variant.

Adds a slack variable η to the LCS so that the complementarity condition
0 ≤ λ ⊥ η ≥ 0 is expressed as a hard equality
    η_t = E x_t + F λ_t + H u_t + c
inside the QP, rather than the soft penalty C3 uses. The δ-update becomes
the closed-form Bui 2026 eq (12) componentwise projection on (λ, η)
pairs (see C3Solver._project_componentwise) instead of C3's Lorentz cone
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
                 torque_limit: float = 30.0,
                 admm_iter:    int   = 10,
                 math_diag:    bool  = False,
                 use_ee_space: bool  = False):
        assert getattr(solver, 'mode', None) == 'c3plus', (
            "C3PlusMPC requires a C3Solver with mode='c3plus'. "
            "Use control.ci_mpc_c3.C3MPC for the baseline C3 path."
        )
        self.formulator    = formulator
        self.solver        = solver
        self.quad_cost     = quadratic_cost
        self.horizon       = horizon
        self.dt            = dt
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

        # Last predicted trajectory — set after every solve, used for Meshcat viz
        self.last_x_seq: np.ndarray | None = None   # (N+1, n_x)
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

        # 1. Linearise Drake plant into discrete LCS + slack expression, around
        # the previous solve's u[0] (Aydinoglu 2024 eq. 8 linearization point).
        if self.use_ee_space:
            (A, B_ctrl, D, d,
             E_lcs, F_lcs, H_lcs, c_lcs,
             J_n, J_t, phi, mu) = \
                self.formulator.linearize_discrete_ee_space(
                    plant_ctx, self.dt, u_lin=self._last_u)
        else:
            (A, B_ctrl, D, d,
             E_lcs, F_lcs, H_lcs, c_lcs,
             J_n, J_t, phi, mu) = \
                self.formulator.linearize_discrete_with_complementarity(
                    plant_ctx, self.dt, u_lin=self._last_u)

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
            print(f"[MATH.setup] Friction coefficient μ={mu:.4f}")
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
        else:
            x0 = np.concatenate([current_q, current_v])

        # Stage 5 per-axis u bounds (env-gated, default-inert). When the
        # EE-space planner is active and the env vars are set, override the
        # symmetric scalar torque_limit with per-axis bounds:
        #   u = [Fx, Fy, Fz];  Fx,Fy ∈ ±PUSHA_STAGE5_U_HORIZONTAL;
        #                      Fz   ∈ ±PUSHA_STAGE5_U_VERTICAL.
        # When EITHER env var is unset the scalar torque_limit path is used
        # unchanged (bit-identical to pre-Stage-5).
        _u_lo = None
        _u_hi = None
        if self.use_ee_space and self.solver.n_u == 3:
            _uh_s = os.environ.get("PUSHA_STAGE5_U_HORIZONTAL", "")
            _uv_s = os.environ.get("PUSHA_STAGE5_U_VERTICAL", "")
            if _uh_s and _uv_s:
                try:
                    _uh = float(_uh_s)
                    _uv = float(_uv_s)
                    _u_lo = np.array([-_uh, -_uh, -_uv])
                    _u_hi = np.array([+_uh, +_uh, +_uv])
                except ValueError:
                    _u_lo = None
                    _u_hi = None

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
        u_seq, x_seq = self.solver.solve(
            x0, A, B_ctrl, D, d, J_n, J_t, mu,
            Q, R, QN, x_ref,
            N=self.horizon,
            admm_iter=self.admm_iter,
            torque_limit=self.torque_limit,
            phi=phi,
            E=E_lcs, F=F_lcs, H=H_lcs, c_lcs=c_lcs,
            u_lower=_u_lo, u_upper=_u_hi,
        )

        # 5. Store predicted trajectory + u[0] for next-step linearization
        self.last_x_seq = x_seq        # (N+1, n_x)
        self._last_u    = u_seq[0].copy()
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
