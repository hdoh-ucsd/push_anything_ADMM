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
                 math_diag:    bool  = False):
        assert getattr(solver, 'mode', None) == 'c3plus', (
            "C3PlusMPC requires a C3Solver with mode='c3plus'. "
            "Use control.ci_mpc_c3.C3MPC for the baseline C3 path."
        )
        self.formulator    = formulator
        self.solver        = solver
        self.quad_cost     = quadratic_cost
        self.horizon       = horizon
        self.dt            = dt
        self.torque_limit  = torque_limit
        self.admm_iter     = admm_iter
        self._math_diag        = math_diag
        self._mpc_step         = 0
        self._math_setup_done  = False
        self._printed_force_diag = False

        # Last predicted trajectory — set after every solve, used for Meshcat viz
        self.last_x_seq: np.ndarray | None = None   # (N+1, n_x)
        # Previous solve's u_seq[0] for the next-step linearization (Aydinoglu eq. 8).
        self._last_u: np.ndarray = np.zeros(solver.n_u)

    def compute_control(self,
                        current_q:  np.ndarray,
                        current_v:  np.ndarray,
                        plant_ctx,
                        target_xy:  np.ndarray) -> np.ndarray:
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
        (A, B_ctrl, D, d,
         E_lcs, F_lcs, H_lcs, c_lcs,
         J_n, J_t, phi, mu) = \
            self.formulator.linearize_discrete_with_complementarity(
                plant_ctx, self.dt, u_lin=self._last_u)

        # ---- [MATH.setup] fires ONCE on first MPC step ----------------------
        if self._math_diag and not self._math_setup_done:
            self._math_setup_done = True
            n_lambda = J_n.shape[0] + J_t.shape[0]
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

        # 2. Quadratic cost and reference state (with linearised EE approach)
        Q, R, QN, x_ref = self.quad_cost.build(
            target_xy, plant_ctx=plant_ctx, current_q=current_q
        )

        # 3. Current full state x0 = [q; v]
        x0 = np.concatenate([current_q, current_v])

        # 4. Full-horizon C3+ ADMM solve — forwards slack expression (E, F, H, c)
        u_seq, x_seq = self.solver.solve(
            x0, A, B_ctrl, D, d, J_n, J_t, mu,
            Q, R, QN, x_ref,
            N=self.horizon,
            admm_iter=self.admm_iter,
            torque_limit=self.torque_limit,
            phi=phi,
            E=E_lcs, F=F_lcs, H=H_lcs, c_lcs=c_lcs,
        )

        # 5. Store predicted trajectory + u[0] for next-step linearization
        self.last_x_seq = x_seq        # (N+1, n_x)
        self._last_u    = u_seq[0].copy()

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
