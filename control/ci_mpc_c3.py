"""
C3 MPC Controller for contact-implicit manipulation (Aydinoglu 2024).

Linearises the Drake plant into a discrete LCS at the current state,
then solves the full N-step trajectory with C3Solver (stacked QP + ADMM
with Lorentz cone projection on contact forces).

For the Bui 2026 §IV-B.2 slack-variable variant, see
control/ci_mpc_c3plus.py — that controller wraps the same C3Solver
class but configured with mode='c3plus' and forwards the additional
(E, F, H, c) complementarity-slack expression.

Control loop per timestep:
  1. Linearise Drake plant → A, B_ctrl, D, d (held constant over horizon)
  2. Build quadratic cost Q, R, QN and reference state x_ref from goal
  3. Run full-horizon ADMM (admm_iter iters, all N steps in one QP)
  4. Apply u_opt[0] (receding horizon); shift nominal plan forward
"""
import numpy as np


# ---------------------------------------------------------------------------
# Shared diagnostic helpers — used by both C3MPC and C3PlusMPC
# ---------------------------------------------------------------------------

def log_math_cost(quad_cost, formulator, solver, horizon,
                  x_seq, x_ref, u_seq, Q, R, QN, mpc_step):
    """
    Print [MATH.cost] per-term cost breakdown over the predicted horizon.
    Caller decides when to fire (typically every 50 MPC steps).
    """
    def _mfmt(v):
        av = abs(v) if v != 0 else 0.0
        return f"{v:.4f}" if (av == 0.0 or 1e-3 <= av <= 1e3) else f"{v:.4e}"

    qc  = quad_cost
    n_u = solver.n_u
    n_q = formulator.n_q
    arm_q = arm_v = obj_xy = obj_z = obj_rp = 0.0

    for si in range(1, horizon):
        e = x_seq[si] - x_ref
        eq = e[:n_u]
        arm_q += float(eq @ Q[:n_u, :n_u] @ eq)
        ev = e[n_q : n_q + n_u]
        arm_v += float(ev @ Q[n_q:n_q+n_u, n_q:n_q+n_u] @ ev)
        ex = e[qc._obj_x_idx]
        ey = e[qc._obj_y_idx]
        obj_xy += (ex**2 * Q[qc._obj_x_idx, qc._obj_x_idx]
                   + ey**2 * Q[qc._obj_y_idx, qc._obj_y_idx])
        ez = e[qc._obj_z_idx]
        obj_z += float(ez**2 * Q[qc._obj_z_idx, qc._obj_z_idx])
        erx = e[qc._obj_ps + 1]
        ery = e[qc._obj_ps + 2]
        obj_rp += (erx**2 * Q[qc._obj_ps+1, qc._obj_ps+1]
                   + ery**2 * Q[qc._obj_ps+2, qc._obj_ps+2])

    ctrl  = sum(float(u_seq[i] @ R @ u_seq[i]) for i in range(horizon))
    e_N   = x_seq[horizon] - x_ref
    term  = float(e_N @ QN @ e_N)
    run   = arm_q + arm_v + obj_xy + obj_z + obj_rp
    total = run + ctrl + term

    print(f"[MATH.cost] Cost breakdown at step={mpc_step}:")
    print(f"[MATH.cost]   joint pos error cost  = {_mfmt(arm_q)}")
    print(f"[MATH.cost]   joint vel error cost  = {_mfmt(arm_v)}")
    print(f"[MATH.cost]   box XY error cost     = {_mfmt(obj_xy)}")
    print(f"[MATH.cost]   box Z error cost      = {_mfmt(obj_z)}")
    print(f"[MATH.cost]   box rotation error    = {_mfmt(obj_rp)}")
    print(f"[MATH.cost]   Sum running state cost (horizon): {_mfmt(run)}")
    print(f"[MATH.cost]   Terminal cost x_N^T QN x_N: {_mfmt(term)}")
    print(f"[MATH.cost]   Control cost Σ u^T R u: {_mfmt(ctrl)}")
    print(f"[MATH.cost]   Total predicted cost: {_mfmt(total)}")


def log_force_diag_once(already_printed, formulator, quad_cost,
                        current_q, target_xy, x_seq, A, B_ctrl, d,
                        u_seq, D, J_n) -> bool:
    """
    Print [FORCE] diagnostic on the first MPC step that produces a nonzero
    contact normal force. Returns True if it printed (caller should set
    its `_printed_force_diag` flag accordingly). No-op when already
    printed, when no contact is detected, or when the inferred λ_n_max
    is below 0.01 N.
    """
    if already_printed:
        return False
    num_normals = J_n.shape[0]
    if num_normals == 0 or D.shape[1] == 0:
        return False
    # Infer λ[0] from LCS dynamics: D λ[0] = x[1] − A x[0] − B u[0] − d
    residual = x_seq[1] - A @ x_seq[0] - B_ctrl @ u_seq[0] - d
    if np.linalg.norm(residual) <= 1e-6:
        return False
    lam_inferred, *_ = np.linalg.lstsq(D, residual, rcond=None)
    lam_n_0 = lam_inferred[:num_normals]
    if not np.any(lam_n_0 > 0.01):
        return False

    nhats = getattr(formulator, '_last_nhats', [])
    qc = quad_cost
    obj_xy_now = np.array([
        current_q[qc._obj_x_idx],
        current_q[qc._obj_y_idx],
    ])
    v_goal = target_xy - obj_xy_now
    g_hat  = v_goal / (np.linalg.norm(v_goal) + 1e-9)

    print(f"[FORCE] First nonzero contact at step 0:")
    for i in range(num_normals):
        if i < len(nhats):
            n = nhats[i]
            F_mag = float(lam_n_0[i])
            F_world = F_mag * n
            proj = float(np.dot(F_world[:2], g_hat))
            print(f"  contact {i}: λ_n={F_mag:.4f}  "
                  f"nhat_onto_box={np.round(n, 3)}  "
                  f"F_world={np.round(F_world, 3)}  "
                  f"F·g_hat={proj:.4f} "
                  f"({'→goal ✓' if proj > 0 else '←WRONG ✗'})")
    print(f"  g_hat (goal direction)={np.round(g_hat, 3)}")
    print(f"  lam_t[0]={np.round(lam_inferred[num_normals:num_normals+4], 4)}")
    return True


# ---------------------------------------------------------------------------
# C3MPC — Aydinoglu 2024 contact-implicit MPC (Lorentz cone projection)
# ---------------------------------------------------------------------------

class C3MPC:
    """
    Parameters
    ----------
    formulator    : LCSFormulator  — extracts and linearises Drake dynamics
    solver        : C3Solver       — full-horizon ADMM optimiser (mode='c3')
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
        # Previous solve's u_seq[0], used as the linearization input u* for
        # the next step (Aydinoglu 2024 eq. 8). First step linearizes around u=0.
        self._last_u: np.ndarray = np.zeros(solver.n_u)

    def compute_control(self,
                        current_q:  np.ndarray,
                        current_v:  np.ndarray,
                        plant_ctx,
                        target_xy:  np.ndarray) -> np.ndarray:
        """
        Compute one torque command via C3 trajectory optimisation.

        Parameters
        ----------
        current_q : (n_q,)  real plant positions (from GetPositions)
        current_v : (n_v,)  real plant velocities (from GetVelocities)
        plant_ctx :         Drake plant context (must be a sub-context of a
                            diagram context so geometry queries are connected)
        target_xy : (2,)    goal [x, y] in world frame

        Returns
        -------
        u_opt : (n_u,)  joint torque command for this timestep
        """
        plant = self.formulator.plant
        plant.SetPositions(plant_ctx, current_q)
        plant.SetVelocities(plant_ctx, current_v)

        self._mpc_step += 1

        # 1. Linearise Drake plant into discrete LCS at current state, around
        # the previous solve's u[0] (Aydinoglu 2024 eq. 8 linearization point).
        # Phase 2: linearize_discrete now also returns the Stewart-Trinkle
        # LCP slack expression (E, F, H, c) for use by the LCP projection.
        (A, B_ctrl, D, d,
         E_lcs, F_lcs, H_lcs, c_lcs,
         J_n, J_t, phi, mu) = \
            self.formulator.linearize_discrete(plant_ctx, self.dt,
                                               u_lin=self._last_u)

        # ---- [MATH.setup] fires ONCE on first MPC step ----------------------
        if self._math_diag and not self._math_setup_done:
            self._math_setup_done = True
            n_lambda = J_n.shape[0] + J_t.shape[0]
            TOT      = self.solver.n_x + n_lambda + self.solver.n_u
            total    = self.horizon * TOT + self.solver.n_x
            qc       = self.quad_cost
            print(f"[MATH.setup] mode=c3  Horizon N={self.horizon}, "
                  f"dt={self.dt}s ({self.horizon * self.dt:.1f}s lookahead)")
            print(f"[MATH.setup] State dim n_x={self.solver.n_x}, "
                  f"control dim n_u={self.solver.n_u}, "
                  f"contact dim n_λ={n_lambda}")
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
            print(f"[MATH.setup] Soft complementarity: w_comp={self.solver._w_comp:.1f}")
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
            # Phase 2: Stewart-Trinkle LCP slack expression (shared with C3+)
            if E_lcs is not None and E_lcs.size > 0:
                print(f"[MATH.LCS] E   shape={E_lcs.shape}, "
                      f"norm(F)={np.linalg.norm(E_lcs):.4f}")
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

        # 4. Full-horizon C3 ADMM solve. Phase 2: forwards (E, F, H, c) for
        # the LCP projection in the δ-step.
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
