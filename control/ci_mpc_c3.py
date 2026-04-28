"""
C3 MPC Controller for contact-implicit manipulation.

Linearises the Drake plant into a discrete LCS at the current state,
then solves the full N-step trajectory with C3Solver (stacked QP + ADMM).
No MPPI sampling — single deterministic trajectory optimisation.

Control loop per timestep:
  1. Linearise Drake plant → A, B_ctrl, D, d (held constant over horizon)
  2. Build quadratic cost Q, R, QN and reference state x_ref from goal
  3. Run full-horizon ADMM (admm_iter iters, all N steps in one QP)
  4. Apply u_opt[0] (receding horizon); shift nominal plan forward
"""
import numpy as np


class C3MPC:
    """
    Parameters
    ----------
    formulator    : LCSFormulator  — extracts and linearises Drake dynamics
    solver        : C3Solver       — full-horizon ADMM optimiser
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

        # Last predicted trajectory — set after every solve, used for Meshcat viz
        self.last_x_seq: np.ndarray | None = None   # (N+1, n_x)

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

        # 1. Linearise Drake plant into discrete LCS at current state
        A, B_ctrl, D, d, J_n, J_t, phi, mu = \
            self.formulator.linearize_discrete(plant_ctx, self.dt)

        # ---- [MATH.setup] fires ONCE on first MPC step ----------------------
        if self._math_diag and not self._math_setup_done:
            self._math_setup_done = True
            _n_lambda = J_n.shape[0] + J_t.shape[0]
            _TOT      = self.solver.n_x + _n_lambda + self.solver.n_u
            _total    = self.horizon * _TOT + self.solver.n_x
            _qc       = self.quad_cost
            print(f"[MATH.setup] Horizon N={self.horizon}, dt={self.dt}s "
                  f"({self.horizon * self.dt:.1f}s lookahead)")
            print(f"[MATH.setup] State dim n_x={self.solver.n_x}, "
                  f"control dim n_u={self.solver.n_u}, "
                  f"contact dim n_λ={_n_lambda}")
            print(f"[MATH.setup] Total QP variable dim: z ∈ R^{_total}"
                  f"  (= N·TOT+n_x = {self.horizon}·{_TOT}"
                  f"+{self.solver.n_x})")
            print(f"[MATH.setup] Cost weights (from tasks.yaml):")
            print(f"[MATH.setup]   w_obj_xy={_qc.w_obj_xy:.1f}  "
                  f"w_obj_z={_qc.w_obj_z:.1f}  "
                  f"w_box_z={_qc.w_box_z:.1f}  "
                  f"w_box_rp={_qc.w_box_rp:.1f}  "
                  f"w_torque={_qc.w_torque}  "
                  f"w_terminal={_qc.w_terminal:.1f}")
            print(f"[MATH.setup] ADMM: rho_0={self.solver.rho:.1f}, "
                  f"rho_clamp=[0.1, 1000.0], "
                  f"max_iter={self.admm_iter}, tol=1e-3")
            print(f"[MATH.setup] Soft complementarity: "
                  f"w_comp={self.solver._w_comp:.1f}")
            print(f"[MATH.setup] Friction coefficient μ={mu:.4f}")
            print(f"[MATH.setup] Torque limit: ±{self.torque_limit:.1f} Nm")

        # ---- [MATH.LCS] every 50 MPC steps -----------------------------------
        if self._math_diag and self._mpc_step % 50 == 0:
            _nc_now     = J_n.shape[0] + J_t.shape[0]
            _contact_on = J_n.shape[0] > 0
            _phi_str    = ("  ".join(f"{v:.5f}" for v in phi)
                           if len(phi) > 0 else "(none)")
            print(f"[MATH.LCS] step={self._mpc_step}, "
                  f"contact active: {'Y' if _contact_on else 'N'}, "
                  f"n_c={_nc_now}")
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
            print(f"[MATH.LCS] φ (SDF gap): [{_phi_str}] m")

        # 2. Quadratic cost and reference state (with linearised EE approach)
        Q, R, QN, x_ref = self.quad_cost.build(
            target_xy, plant_ctx=plant_ctx, current_q=current_q
        )

        # 3. Current full state x0 = [q; v]
        x0 = np.concatenate([current_q, current_v])

        # 4. Full-horizon C3 ADMM solve (phi forwarded for soft complementarity)
        u_seq, x_seq = self.solver.solve(
            x0, A, B_ctrl, D, d, J_n, J_t, mu,
            Q, R, QN, x_ref,
            N=self.horizon,
            admm_iter=self.admm_iter,
            torque_limit=self.torque_limit,
            phi=phi,
        )

        # 5. Store predicted trajectory for external visualisation
        self.last_x_seq = x_seq   # (N+1, n_x)

        # ---- [MATH.cost] every 50 MPC steps ----------------------------------
        if self._math_diag and self._mpc_step % 50 == 0:
            def _mfmt(v):
                av = abs(v) if v != 0 else 0.0
                return f"{v:.4f}" if (av == 0.0 or 1e-3 <= av <= 1e3) else f"{v:.4e}"

            _qc  = self.quad_cost
            _n_u = self.solver.n_u
            _n_q = self.formulator.n_q
            _arm_q  = _arm_v  = _obj_xy = _obj_z = _obj_rp = 0.0

            for _si in range(1, self.horizon):
                _xi = x_seq[_si]
                _e  = _xi - x_ref
                # arm joint positions (0:n_u in x)
                _eq  = _e[:_n_u]
                _arm_q += float(_eq @ Q[:_n_u, :_n_u] @ _eq)
                # arm joint velocities (n_q:n_q+n_u in x)
                _ev  = _e[_n_q : _n_q + _n_u]
                _arm_v += float(_ev @ Q[_n_q:_n_q+_n_u, _n_q:_n_q+_n_u] @ _ev)
                # box XY
                _ex  = _e[_qc._obj_x_idx]
                _ey  = _e[_qc._obj_y_idx]
                _obj_xy += (_ex**2 * Q[_qc._obj_x_idx, _qc._obj_x_idx]
                            + _ey**2 * Q[_qc._obj_y_idx, _qc._obj_y_idx])
                # box Z
                _ez  = _e[_qc._obj_z_idx]
                _obj_z  += float(_ez**2 * Q[_qc._obj_z_idx, _qc._obj_z_idx])
                # box rotation (qx, qy)
                _erx = _e[_qc._obj_ps + 1]
                _ery = _e[_qc._obj_ps + 2]
                _obj_rp += (_erx**2 * Q[_qc._obj_ps+1, _qc._obj_ps+1]
                            + _ery**2 * Q[_qc._obj_ps+2, _qc._obj_ps+2])

            _ctrl = sum(float(u_seq[_i] @ R @ u_seq[_i])
                        for _i in range(self.horizon))
            _e_N   = x_seq[self.horizon] - x_ref
            _term  = float(_e_N @ QN @ _e_N)
            _run   = _arm_q + _arm_v + _obj_xy + _obj_z + _obj_rp
            _total = _run + _ctrl + _term

            print(f"[MATH.cost] Cost breakdown at step={self._mpc_step}:")
            print(f"[MATH.cost]   joint pos error cost  = {_mfmt(_arm_q)}")
            print(f"[MATH.cost]   joint vel error cost  = {_mfmt(_arm_v)}")
            print(f"[MATH.cost]   box XY error cost     = {_mfmt(_obj_xy)}")
            print(f"[MATH.cost]   box Z error cost      = {_mfmt(_obj_z)}")
            print(f"[MATH.cost]   box rotation error    = {_mfmt(_obj_rp)}")
            print(f"[MATH.cost]   Sum running state cost (horizon): {_mfmt(_run)}")
            print(f"[MATH.cost]   Terminal cost x_N^T QN x_N: {_mfmt(_term)}")
            print(f"[MATH.cost]   Control cost Σ u^T R u: {_mfmt(_ctrl)}")
            print(f"[MATH.cost]   Total predicted cost: {_mfmt(_total)}")

        # 6. Planned contact force diagnostic (one-time, at first contact)
        num_normals = J_n.shape[0]
        if num_normals > 0 and not getattr(self, '_printed_force_diag', False):
            # Infer λ[0] from LCS dynamics: D λ[0] = x[1] - A x[0] - B u[0] - d
            residual = x_seq[1] - A @ x_seq[0] - B_ctrl @ u_seq[0] - d
            n_c = D.shape[1]
            if n_c > 0 and np.linalg.norm(residual) > 1e-6:
                # Least-squares solve for lambda
                lam_inferred, *_ = np.linalg.lstsq(D, residual, rcond=None)
                lam_n_0 = lam_inferred[:num_normals]   # normal forces at step 0

                if np.any(lam_n_0 > 0.01):
                    self._printed_force_diag = True
                    nhats = getattr(self.formulator, '_last_nhats', [])
                    # g_hat from current box pos to goal (use current_q box xy)
                    n_q = self.formulator.n_q
                    # quad_cost stores obj position indices
                    qc = self.quad_cost
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

        # 7. Receding horizon: return u[0], clipped to torque limit
        return np.clip(u_seq[0], -self.torque_limit, self.torque_limit)
