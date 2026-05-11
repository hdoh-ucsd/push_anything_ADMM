"""
C3 ADMM Solver — full-horizon stacked trajectory optimisation.

ADMM consensus split over the complete N-step LCS trajectory:

  Decision variable:  z = [x₀, λ₀, u₀,  x₁, λ₁, u₁,  …,  x_N]
  Per-step block size TOT = n_x + n_c + n_u

  z-update  (stacked QP, built once per control step):
      min  0.5 z^T P z + q^T z
      s.t. x_0            = x_current          (initial state)
           x_{t+1} = A x_t + D λ_t + B u_t + d  (LCS dynamics, t=0..N-1)
           λ_n ≥ 0                              (normal forces repulsive)
           |u_t| ≤ torque_limit

  δ-update  (per-step Lorentz cone projection):
      δ[λ_t] = proj_C( z[λ_t] + ω[λ_t] )   C = {λ_n ≥ 0, ‖λ_t‖₂ ≤ μ·λ_n}
      δ[x_t] = z[x_t] + ω[x_t]              (unconstrained)
      δ[u_t] = z[u_t] + ω[u_t]              (unconstrained)

  ω-update  (dual ascent):
      ω += z − δ

Cost:
    Σ_{t=1}^{N-1} (x_t−x_ref)^T Q (x_t−x_ref)  +  Σ_{t=0}^{N-1} u_t^T R u_t
    + (x_N−x_ref)^T QN (x_N−x_ref)
    + (ρ/2) ‖z − δ + ω‖²

Speed notes
-----------
 - MathematicalProgram built ONCE per control step (contact geometry fixed).
 - Only the linear cost term q is refreshed each ADMM iteration via
   UpdateCoefficients — avoids repeated program allocation.
"""
import numpy as np
import pydrake.all as ad

try:
    from profiling.section_timer import timed
except ImportError:
    from contextlib import contextmanager
    @contextmanager
    def timed(_name):   # noqa: E306
        yield


def _fmt(v: float) -> str:
    """Format: 4 d.p. if |v| in [1e-3, 1e3], else 4-digit scientific."""
    av = abs(v) if v != 0 else 0.0
    if av == 0.0 or 1e-3 <= av <= 1e3:
        return f"{v:.4f}"
    return f"{v:.4e}"


class C3Solver:
    """
    Full-horizon C3 ADMM solver.

    Parameters
    ----------
    n_x : int    State dimension (n_q + n_v).
    n_u : int    Control input dimension.
    rho : float  ADMM penalty parameter (default 1.0).
    """

    def __init__(self, n_x: int, n_u: int, rho: float = 1.0,
                 math_diag: bool = False, mode: str = "c3"):
        assert mode in ("c3", "c3plus"), f"unknown solver mode: {mode}"
        self.n_x        = n_x
        self.n_u        = n_u
        self.rho        = rho
        self._math_diag = math_diag
        self.mode       = mode                          # ← C3+ NEW
        # Soft-complementarity penalty: disabled in BOTH modes from Phase 2
        # onward.  C3 now uses LCP projection (Aydinoglu §V-B.3.b) which
        # produces feasible λ by construction; C3+ uses the slack-equality
        # η = E x + F λ + H u + c.  In either case the linear penalty
        # `w_comp · phi · λ_n` is structurally redundant and would bias
        # λ_n toward zero on contact.
        self._w_comp    = 0.0
        self._solver    = ad.OsqpSolver()
        self._diag_step = 0
        # Pre-allocated identity matrices — n_x is fixed; total_dim is cached on first use
        self._eye_nx         = np.eye(n_x)
        self._eye_total_dim  = -1       # sentinel: rebuild when total_dim changes
        self._eye_total      = None
        # ===== C3+ specific (Bui 2026 §IV-B.2) =====               ← C3+ NEW
        # u_λ, u_η are the per-component G weights in eq (12). Default 1
        # so √(u_λ/u_η) = 1 — the cleanest projection.
        self._u_lambda           = 1.0
        self._u_eta              = 1.0
        # Bui §IV-B.2 final paragraph: large G-weight on EE-object contact
        # components in the final QP step. NOT applied here yet (would
        # require knowing which contact is EE↔box; see TODO in _solve_c3plus).
        self._w_G_ee_contact     = 1000.0
        self._eye_total_c3p_dim  = -1
        self._eye_total_c3p      = None

    # ------------------------------------------------------------------
    # Lorentz cone projection (per-contact scalar implementation)
    # ------------------------------------------------------------------
    @staticmethod
    def _project_single_contact(lam_n: float,
                            lam_t: np.ndarray,
                            mu: float) -> tuple[float, np.ndarray]:
        """
        Project (λ_n, λ_t) onto the friction cone, handling both the 1D scalar
        tangent (sandbox) and the 4D polyhedral-pyramid tangent (Drake).

        Drake's LCS formulator builds J_t with stacked rows [t1, -t1, t2, -t2]
        and associates each row with a non-negative force-magnitude component in
        λ_t. The physical tangent force in the contact plane is:
            F_t_world = (λ_t^[0] - λ_t^[1]) t1 + (λ_t^[2] - λ_t^[3]) t2
        whose Euclidean magnitude is:
            ‖F_t‖₂ = sqrt((λ_t^[0]-λ_t^[1])² + (λ_t^[2]-λ_t^[3])²)

        The friction cone constraint is ‖F_t‖₂ ≤ μ λ_n (Lorentz in the
        contact-plane 2D force vector). We project into that cone in 2D
        Cartesian coordinates, then split back into non-negative polyhedral
        components.

        For the sandbox's 1D scalar tangent this reduces to the standard
        Lorentz projection.
        """
        k = lam_t.shape[0]

        # Convert polyhedral λ_t → Cartesian F_t in the contact plane
        if k == 4:
            F_t = np.array([
                lam_t[0] - lam_t[1],   # component along t1
                lam_t[2] - lam_t[3],   # component along t2
            ])
        elif k == 1 or k == 2:
            # Sandbox case: already Cartesian
            F_t = lam_t.copy()
        else:
            raise ValueError(f"Unsupported tangent dimension k={k}")

        b_norm = float(np.linalg.norm(F_t))

        # Three cases — standard Lorentz cone projection on (λ_n, F_t)
        if b_norm <= mu * lam_n + 1e-12:                    # Case 1: inside
            n_new = float(lam_n)
            F_t_new = F_t.copy()
        elif mu * b_norm <= -lam_n + 1e-12:                  # Case 2: polar → apex
            n_new = 0.0
            F_t_new = np.zeros_like(F_t)
        else:                                                # Case 3: surface
            n_new = (lam_n + mu * b_norm) / (1.0 + mu * mu)
            F_t_new = (mu * n_new / b_norm) * F_t

        # Cartesian → polyhedral (split back into non-negative components)
        if k == 4:
            t_new = np.array([
                max(F_t_new[0],  0.0),   # λ_t^[0] (along +t1)
                max(-F_t_new[0], 0.0),   # λ_t^[1] (along -t1)
                max(F_t_new[1],  0.0),   # λ_t^[2] (along +t2)
                max(-F_t_new[1], 0.0),   # λ_t^[3] (along -t2)
            ])
        else:
            t_new = F_t_new

        # Sanity check (on the physical cone, not the polyhedral components)
        F_t_new_norm = float(np.linalg.norm(F_t_new))
        if F_t_new_norm > mu * n_new + 1e-8:
            raise AssertionError(
                f"Projection incorrect: ‖F_t*‖={F_t_new_norm:.8f} > "
                f"μ·λ_n*={mu*n_new:.8f}"
            )

        return n_new, t_new

    @staticmethod
    def _lorentz_project(lam: np.ndarray,
                         num_normals: int,
                         mu: float) -> np.ndarray:
        """
        Apply _project_single_contact to each contact in lam.

        lam layout: [λ_n_0, …, λ_n_{K-1},
                     λ_t_0 (4 vals), λ_t_1 (4 vals), …, λ_t_{K-1} (4 vals)]
        where K = num_normals.  Slicing: λ_n_i → lam[i],
        λ_t_i → lam[K + 4i : K + 4(i+1)].
        """
        if num_normals == 0:
            return lam.copy()

        result = lam.copy()
        for i in range(num_normals):
            n_new, t_new = C3Solver._project_single_contact(
                float(lam[i]),
                lam[num_normals + 4 * i : num_normals + 4 * (i + 1)],
                mu,
            )
            result[i] = n_new
            result[num_normals + 4 * i : num_normals + 4 * (i + 1)] = t_new
        return result

    # ------------------------------------------------------------------
    # Main ADMM solve
    # ------------------------------------------------------------------
    def solve(self,
              x0:      np.ndarray,
              A:       np.ndarray,
              B_ctrl:  np.ndarray,
              D:       np.ndarray,
              d:       np.ndarray,
              J_n:     np.ndarray,
              J_t:     np.ndarray,
              mu:      float,
              Q:       np.ndarray,
              R:       np.ndarray,
              QN:      np.ndarray,
              x_ref:   np.ndarray,
              N:       int   = 8,
              admm_iter: int = 10,
              torque_limit: float = 30.0,
              phi:     np.ndarray | None = None,
              # ===== C3+ NEW: complementarity slack expression =====
              E:       np.ndarray | None = None,
              F:       np.ndarray | None = None,
              H:       np.ndarray | None = None,
              c_lcs:   np.ndarray | None = None,
              ) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the C3 full-horizon trajectory optimisation.

        Parameters
        ----------
        x0       : (n_x,)        current state [q; v]
        A        : (n_x, n_x)    discrete state transition
        B_ctrl   : (n_x, n_u)    discrete control input matrix
        D        : (n_x, n_c)    discrete contact force matrix (may be (n_x,0))
        d        : (n_x,)        constant LCS offset
        J_n      : (nc, n_v)     normal contact Jacobians
        J_t      : (4nc, n_v)    tangential contact Jacobians
        mu       : float         friction coefficient
        Q        : (n_x, n_x)    running state cost
        R        : (n_u, n_u)    control cost
        QN       : (n_x, n_x)    terminal state cost
        x_ref    : (n_x,)        reference state (goal)
        N        : int            planning horizon
        admm_iter: int            ADMM iterations per control step
        torque_limit: float       joint torque clamp (Nm)

        Returns
        -------
        u_seq : (N, n_u)    optimal torque sequence
        x_seq : (N+1, n_x)  predicted state trajectory
        """
        # ===== C3+ NEW: dispatch to alternate algorithm path =====
        if self.mode == "c3plus":
            assert E is not None and F is not None and H is not None and c_lcs is not None, (
                "C3+ requires the complementarity expression (E, F, H, c). "
                "Call LCSFormulator.linearize_discrete_with_complementarity() "
                "and forward all four matrices to solve()."
            )
            return self._solve_c3plus(
                x0=x0, A=A, B_ctrl=B_ctrl, D=D, d=d,
                E=E, F=F, H=H, c_lcs=c_lcs,
                J_n=J_n, J_t=J_t, mu=mu,
                Q=Q, R=R, QN=QN, x_ref=x_ref,
                N=N, admm_iter=admm_iter, torque_limit=torque_limit,
                phi=phi,
            )

        # ===== C3 (Phase 2 — paper-exact LCP projection) =====
        # λ ordering changed from [λ_n; λ_t] (5n_c) to [γ; λ_n; λ_t] (6n_c)
        # to carry the Stewart-Trinkle friction-cone slack γ. The LCP
        # projection (Aydinoglu §V-B.3.b) replaces the old Lorentz-cone
        # projection on (λ_n, λ_t).
        assert E is not None and F is not None and H is not None and c_lcs is not None, (
            "C3 (Phase 2) requires the LCP slack expression (E, F, H, c). "
            "Pass them through from LCSFormulator.linearize_discrete."
        )

        from control.lcp_solver import solve_lcp  # local import — Drake-dependent

        n_x = self.n_x
        n_u = self.n_u
        rho = self.rho

        num_normals = J_n.shape[0]
        n_t         = J_t.shape[0]                  # 4·num_normals
        n_lam       = 2 * num_normals + n_t         # 6·num_normals
        TOT         = n_x + n_lam + n_u
        total_dim   = N * TOT + n_x

        # Per-step λ slot offsets (within an n_lam-sized block).
        SG  = 0
        SLN = num_normals
        SLT = 2 * num_normals

        # Reuse cached identity; rebuild only when total_dim changes (rare)
        if total_dim != self._eye_total_dim:
            self._eye_total     = np.eye(total_dim)
            self._eye_total_dim = total_dim
        _eye_total = self._eye_total

        # One-time debug print to verify contact slicing
        if not getattr(self, '_debug_printed', False):
            self._debug_printed = True
            print(f"[DEBUG-C3] n_contacts={num_normals}, n_lambda={n_lam} "
                  f"(γ:{num_normals} + λ_n:{num_normals} + λ_t:{n_t})")
            print(f"[DEBUG-C3] per-step λ slots:  γ=[{SG}:{SLN})  "
                  f"λ_n=[{SLN}:{SLT})  λ_t=[{SLT}:{n_lam})")
            print(f"[DEBUG-C3] TOT={TOT}, total_dim={total_dim}")

        # ---- Cost matrix P (quadratic, static within ADMM loop) ----------
        # OSQP: min 0.5 z^T P z + q^T z
        # Tracking cost x_t^T Q x_t → P block += 2*Q; x_N^T QN x_N → 2*QN
        # Control cost  u_t^T R u_t → P block += 2*R
        # ADMM augment  (rho/2)||z||^2 → P += rho*I
        with timed("admm.qp_build"):
            P      = np.zeros((total_dim, total_dim))
            q_ref  = np.zeros(total_dim)

            for i in range(1, N):
                xi = i * TOT
                P[xi:xi+n_x, xi:xi+n_x] += 2.0 * Q
                q_ref[xi:xi+n_x]          = -2.0 * (Q @ x_ref)

            xN = N * TOT
            P[xN:xN+n_x, xN:xN+n_x] += 2.0 * QN
            q_ref[xN:xN+n_x]          = -2.0 * (QN @ x_ref)

            for i in range(N):
                ui = i * TOT + n_x + n_lam
                P[ui:ui+n_u, ui:ui+n_u] += 2.0 * R

            # Phase 2: soft complementarity penalty disabled (w_comp = 0).
            # The δ-update's LCP projection enforces complementarity exactly,
            # so the linear penalty would only bias λ_n toward zero on contact.
            w_comp = self._w_comp   # 0.0 in Phase 2

            P_total = P + rho * _eye_total
            # Symmetrise + small diagonal regularisation for OSQP
            P_sym   = 0.5 * (P_total + P_total.T) + 1e-8 * _eye_total

            # ---- Equality constraint matrix (initial state + dynamics) ---
            # Rows 0..n_x-1       : x_0 = x0
            # Rows n_x + i*n_x    : A x_i + D λ_i + B u_i − x_{i+1} = −d
            # D's first num_normals cols (γ slot) are zero — γ does not enter
            # dynamics in the Stewart-Trinkle formulation.
            n_eq = (N + 1) * n_x
            C_eq = np.zeros((n_eq, total_dim))
            b_eq = np.zeros(n_eq)

            C_eq[:n_x, :n_x] = self._eye_nx
            b_eq[:n_x]        = x0

            for i in range(N):
                row  = n_x + i * n_x
                xi   = i * TOT
                li   = xi + n_x
                ui   = li + n_lam
                xnxt = (i + 1) * TOT if i < N - 1 else N * TOT

                C_eq[row:row+n_x, xi:xi+n_x]      = A
                if n_lam > 0:
                    C_eq[row:row+n_x, li:li+n_lam] = D
                C_eq[row:row+n_x, ui:ui+n_u]      = B_ctrl
                C_eq[row:row+n_x, xnxt:xnxt+n_x]  = -self._eye_nx
                b_eq[row:row+n_x]                  = -d

            # ---- Build MathematicalProgram ONCE per control step ---------
            prog  = ad.MathematicalProgram()
            z_var = prog.NewContinuousVariables(total_dim, "z")

            prog.AddLinearEqualityConstraint(C_eq, b_eq, z_var)

            # All of λ = [γ; λ_n; λ_t] are nonnegative per Stewart-Trinkle
            # complementarity. The LCP projection in the δ-update enforces
            # the slack equality (η = E x + F λ + H u + c) and the
            # complementarity λ ⊥ η.
            if n_lam > 0:
                for i in range(N):
                    prog.AddBoundingBoxConstraint(
                        np.zeros(n_lam),
                        np.full(n_lam, np.inf),
                        z_var[i*TOT + n_x : i*TOT + n_x + n_lam],
                    )

            # Torque bounds per horizon step
            for i in range(N):
                ui = i * TOT + n_x + n_lam
                prog.AddBoundingBoxConstraint(
                    np.full(n_u, -torque_limit),
                    np.full(n_u,  torque_limit),
                    z_var[ui : ui + n_u],
                )

            cost_bd = prog.AddQuadraticCost(P_sym, np.zeros(total_dim), z_var)

        # ---- ADMM iterations (only q refreshed, no reallocations) -------
        delta      = np.zeros(total_dim)
        omega      = np.zeros(total_dim)
        delta_prev = np.zeros(total_dim)

        # Warm-start z: fill every x_i block with x0
        z_sol = np.zeros(total_dim)
        for i in range(N):
            z_sol[i * TOT : i * TOT + n_x] = x0
        z_sol[N * TOT : N * TOT + n_x] = x0

        primal_hist = []   # ||z - δ|| per iteration (contact vars only)
        dual_hist   = []   # ρ·||δ - δ_prev|| per iteration
        tol         = 1e-3
        actual_iters = admm_iter

        for it in range(admm_iter):
            delta_prev = delta.copy()

            with timed("admm.qp_build"):
                q_total = q_ref - rho * (delta - omega)
                cost_bd.evaluator().UpdateCoefficients(P_sym, q_total)

            with timed("admm.osqp_solve"):
                res = self._solver.Solve(prog)

            if res.is_success():
                z_sol = res.GetSolution(z_var)

            with timed("admm.z_update"):
                # δ-update: x and u pass through; λ blocks projected onto
                # the Stewart-Trinkle LCP set H_k via the LCP-projection
                # method (Aydinoglu §V-B.3.b):
                #   q_lcp_k = E·δ_x_k + H·δ_u_k + c
                #   δ_λ_k    = solve_lcp(F, q_lcp_k)
                # x and u in δ are taken from the QP solution z (they pass
                # through the projection unchanged).
                delta = z_sol + omega
                lcp_residuals: list[float] = []
                if n_lam > 0:
                    for i in range(N):
                        li = i * TOT + n_x
                        ui = li + n_lam
                        delta_x = z_sol[i * TOT : i * TOT + n_x] \
                                  + omega[i * TOT : i * TOT + n_x]
                        delta_u = z_sol[ui : ui + n_u] + omega[ui : ui + n_u]
                        q_lcp   = E @ delta_x + H @ delta_u + c_lcs
                        d_lam, lcp_res = solve_lcp(F, q_lcp)
                        lcp_residuals.append(lcp_res)
                        delta[i * TOT          : i * TOT + n_x]            = delta_x
                        delta[li                : li + n_lam]              = d_lam
                        delta[ui                : ui + n_u]                = delta_u

            omega = omega + z_sol - delta

            # Track contact-variable residuals only (x/u are unconstrained)
            if n_lam > 0:
                lam_vec = np.concatenate([
                    z_sol[i * TOT + n_x : i * TOT + n_x + n_lam]
                    for i in range(N)
                ])
                dlt_vec = np.concatenate([
                    delta[i * TOT + n_x : i * TOT + n_x + n_lam]
                    for i in range(N)
                ])
                dlt_prev_vec = np.concatenate([
                    delta_prev[i * TOT + n_x : i * TOT + n_x + n_lam]
                    for i in range(N)
                ])
                pr = float(np.linalg.norm(lam_vec - dlt_vec))
                dr = float(rho * np.linalg.norm(dlt_vec - dlt_prev_vec))
                primal_hist.append(pr)
                dual_hist.append(dr)

                # Adaptive ρ (Boyd §3.4.1) — every 10 iterations
                if (it + 1) % 10 == 0:
                    if pr > 10.0 * dr and rho < 1000.0:
                        rho   *= 2.0
                        omega /= 2.0
                        P_total2 = P + rho * _eye_total
                        P_sym    = 0.5 * (P_total2 + P_total2.T) + 1e-8 * _eye_total
                        cost_bd.evaluator().UpdateCoefficients(P_sym, q_total)
                    elif dr > 10.0 * pr and rho > 0.1:
                        rho   /= 2.0
                        omega *= 2.0
                        P_total2 = P + rho * _eye_total
                        P_sym    = 0.5 * (P_total2 + P_total2.T) + 1e-8 * _eye_total
                        cost_bd.evaluator().UpdateCoefficients(P_sym, q_total)

                # Early termination
                if pr < tol and dr < tol:
                    actual_iters = it + 1
                    break
            else:
                # No contacts: nothing to track, run all iters
                pass

        # Print residual summary
        if n_lam > 0 and primal_hist:
            mono = all(primal_hist[i] >= primal_hist[i+1]
                       for i in range(len(primal_hist)-1))
            print(f"[ADMM] primal: {primal_hist[0]:.4f}->{primal_hist[-1]:.4f}  "
                  f"dual: {dual_hist[0]:.4f}->{dual_hist[-1]:.4f}  "
                  f"mono={mono}  iters={actual_iters}/{admm_iter}  rho={rho:.1f}")

        # ---- Extract outputs ---------------------------------------------
        u_seq = np.zeros((N, n_u))
        x_seq = np.zeros((N + 1, n_x))
        for i in range(N):
            x_seq[i] = z_sol[i * TOT : i * TOT + n_x]
            u_seq[i] = z_sol[i * TOT + n_x + n_lam : i * TOT + n_x + n_lam + n_u]
        x_seq[N] = z_sol[N * TOT : N * TOT + n_x]

        # ---- Contact diagnostics (Phase 2: LCP projection) --------------
        self._diag_step += 1

        if n_lam > 0:
            lam_n_all = np.concatenate([
                z_sol[i * TOT + n_x + SLN : i * TOT + n_x + SLN + num_normals]
                for i in range(N)
            ]) if num_normals else np.zeros(0)
            lam_t_all = np.concatenate([
                z_sol[i * TOT + n_x + SLT : i * TOT + n_x + SLT + n_t]
                for i in range(N)
            ]) if n_t else np.zeros(0)
            lam_n_max = float(lam_n_all.max()) if lam_n_all.size else 0.0
            lt_max    = float(lam_t_all.max()) if lam_t_all.size else 0.0
            pr_last   = primal_hist[-1] if primal_hist else float('nan')
            lcp_max_res = max(lcp_residuals) if lcp_residuals else 0.0
            print(f"[C3] step={self._diag_step} "
                  f"|u[0]|={np.linalg.norm(u_seq[0]):.2f}Nm "
                  f"λ_n_max={lam_n_max:.3f} λ_t_max={lt_max:.3f} "
                  f"lcp_res_max={lcp_max_res:.2e} "
                  f"primal={pr_last:.3f} iters={actual_iters}/{admm_iter}")
        else:
            print(f"[C3] step={self._diag_step} n_λ=0  "
                  f"|u[0]|={np.linalg.norm(u_seq[0]):.3f} Nm")

        # ---- [MATH.QP] every 10th control step --------------------------------
        if self._math_diag and self._diag_step % 10 == 0:
            dim = P_sym.shape[0]
            is_sym = bool(np.allclose(P_sym, P_sym.T, atol=1e-8))
            if dim <= 1000:
                eigs    = np.linalg.eigvalsh(P_sym)
                min_eig = float(eigs.min())
                max_eig = float(eigs.max())
                pos_sd  = min_eig >= -1e-8
                cond_val = max_eig / max(abs(min_eig), 1e-30)
                cond_str = _fmt(cond_val)
            else:
                pos_sd   = "?"
                cond_str = f"skipped (dim={dim}>1000)"
            q_norm = float(np.linalg.norm(q_total))
            osqp_ok = res.is_success()
            osqp_status = "solved" if osqp_ok else "failed/infeasible"
            try:
                _det = res.get_solver_details()
                osqp_iters_val = int(getattr(_det, 'iters',
                                     getattr(_det, 'iter', -1)))
                osqp_time_ms   = float(getattr(_det, 'run_time',
                                       float('nan'))) * 1000.0
            except Exception:
                osqp_iters_val = -1
                osqp_time_ms   = float('nan')
            n_eq = (N + 1) * n_x
            print(f"[MATH.QP] Minimizing: (1/2) z^T P z + q^T z")
            print(f"[MATH.QP]   s.t. A_eq z = b_eq  "
                  f"({n_eq} rows — x_0 fixation + {N} LCS steps)")
            print(f"[MATH.QP]        bbox: γ ≥ 0, λ_n ≥ 0, λ_t ≥ 0, "
                  f"|u| ≤ {torque_limit:.1f} Nm")
            print(f"[MATH.QP] P shape=({dim},{dim}), symmetric={is_sym}, "
                  f"pos-semidef={pos_sd}, cond(P)={cond_str}")
            print(f"[MATH.QP] q norm={_fmt(q_norm)}")
            print(f"[MATH.QP] Augmented term: (ρ/2) Σ ||λ-δ+ω||^2  ρ={_fmt(rho)}")
            print(f"[MATH.QP] Phase 2: soft-complementarity disabled "
                  f"(w_comp=0); LCP projection handles complementarity exactly.")
            _time_str = (f"{osqp_time_ms:.2f}ms"
                         if not (isinstance(osqp_time_ms, float)
                                 and np.isnan(osqp_time_ms))
                         else "?ms")
            print(f"[MATH.QP] OSQP status: {osqp_status}, "
                  f"iters={osqp_iters_val}, solve time={_time_str}")

        # ---- [MATH.δ] LCP projection summary ---------------------------------
        if self._math_diag and n_lam > 0:
            # Aydinoglu §V-B.3.b: solve LCP(F, q_lcp) per timestep, where
            # q_lcp = E δ_x + H δ_u + c. The projection produces feasible
            # (λ ≥ 0, F λ + q ≥ 0, λ^T(Fλ+q)=0) by construction.
            res_arr = np.asarray(lcp_residuals)
            res_max  = float(res_arr.max()) if res_arr.size else 0.0
            res_mean = float(res_arr.mean()) if res_arr.size else 0.0
            n_failed = int(np.sum(np.isinf(res_arr))) if res_arr.size else 0
            # Contact-mode counts: how many of N timesteps have any nonzero
            # δ_λ (i.e. the LCP picked up an active contact).
            active_steps = 0
            for _i in range(N):
                _li = _i * TOT + n_x
                if float(np.max(delta[_li : _li + n_lam])) > 1e-6:
                    active_steps += 1
            print(f"[MATH.δ] LCP projection (Aydinoglu §V-B.3.b):")
            print(f"[MATH.δ]   N={N} step-LCPs solved (n_λ={n_lam} per step)")
            print(f"[MATH.δ]   max |λ^T(Fλ+q)| = {_fmt(res_max)}, "
                  f"mean = {_fmt(res_mean)}  (≈0 ⇒ feasible)")
            if n_failed > 0:
                print(f"[MATH.δ]   ⚠ {n_failed}/{N} LCPs failed Lemke pivot — "
                      f"check F regularisation (eps_reg)")
            print(f"[MATH.δ]   active contact steps (max δ_λ > 1e-6): "
                  f"{active_steps}/{N}")
            # First-step glance at γ, λ_n, λ_t magnitudes
            _li0 = 0 * TOT + n_x
            _g  = float(np.max(delta[_li0 : _li0 + num_normals])) if num_normals else 0.0
            _ln = float(np.max(delta[_li0 + SLN : _li0 + SLN + num_normals])) if num_normals else 0.0
            _lt = float(np.max(delta[_li0 + SLT : _li0 + SLT + n_t])) if n_t else 0.0
            print(f"[MATH.δ]   step k=0 max δ values:  "
                  f"γ={_fmt(_g)}  λ_n={_fmt(_ln)}  λ_t={_fmt(_lt)}")

        # ---- [MATH.ω] every control step --------------------------------------
        if self._math_diag:
            _omega_norm = float(np.linalg.norm(omega))
            if n_lam > 0 and primal_hist:
                _pr = primal_hist[-1]
                _dr = dual_hist[-1]
                _ratio = _pr / (_dr + 1e-30)
                _lam_f = np.concatenate([
                    z_sol[_i*TOT+n_x : _i*TOT+n_x+n_lam] for _i in range(N)
                ])
                _dlt_f = np.concatenate([
                    delta[_i*TOT+n_x : _i*TOT+n_x+n_lam] for _i in range(N)
                ])
                _ld_max = float(np.max(np.abs(_lam_f - _dlt_f)))
            else:
                _pr = _dr = _ratio = _ld_max = 0.0
            print(f"[MATH.ω] ω update: ω += (λ-δ), over {actual_iters} ADMM iters")
            print(f"[MATH.ω] ||ω||={_fmt(_omega_norm)}, "
                  f"||λ-δ||_max={_fmt(_ld_max)}")
            if n_lam > 0:
                if _ratio > 10.0:
                    _rho_note = (f"ratio={_fmt(_ratio)} > 10 "
                                 f"→ would double ρ to {_fmt(rho*2)}")
                elif _ratio < 0.1:
                    _rho_note = (f"ratio={_fmt(_ratio)} < 0.1 "
                                 f"→ would halve ρ to {_fmt(rho/2)}")
                else:
                    _rho_note = f"ratio={_fmt(_ratio)} → ρ unchanged"
                print(f"[MATH.ω] ρ decision: primal={_fmt(_pr)}, "
                      f"dual={_fmt(_dr)}, {_rho_note}")
            else:
                print(f"[MATH.ω] ρ decision: n/a (n_λ=0, no contact variables)")
            _never = " ← never triggers!" if admm_iter < 10 else ""
            print(f"[MATH.ω] Note: adaptive-ρ fires every 10 iters; "
                  f"current max_iter={admm_iter}{_never}")

        return u_seq, x_seq

    # ==================================================================
    # C3+  (Bui 2026 ICRA §IV-B.2 — slack-variable reformulation)
    # ==================================================================
    @staticmethod
    def _project_componentwise(lam: np.ndarray,
                               eta: np.ndarray,
                               u_lambda: float = 1.0,
                               u_eta:    float = 1.0
                               ) -> tuple[np.ndarray, np.ndarray]:
        """
        Bui 2026 eq (12) componentwise complementarity projection.    ← C3+ NEW

        For each scalar pair (λ°, η°) chosen from (z + ω):
            (δ_λ, δ_η) =
              (0,  η°)   if  η° ≥ 0  AND  η° ≥ √(u_λ/u_η) · λ°       (case 1)
              (λ°, 0 )   if  λ° ≥ 0  AND  η° <  √(u_λ/u_η) · λ°       (case 2)
              (0,  0 )   otherwise                                     (case 3)

        REPLACES C3's _lorentz_project, which projected per-contact-pair
        onto the friction cone. C3+ does not project onto the friction
        cone in the δ-update at all — friction is enforced through the
        LCS structure of (E, F, H, c). ← That is the entire point of the
        slack variable.
        """
        sqrt_ratio = float(np.sqrt(u_lambda / u_eta))
        cond1 = (eta >= 0.0) & (eta >= sqrt_ratio * lam)
        cond2 = (lam >= 0.0) & (eta <  sqrt_ratio * lam)

        delta_lam = np.where(cond2, lam, 0.0)
        delta_eta = np.where(cond1, eta, 0.0)
        return delta_lam, delta_eta

    def _solve_c3plus(self,
                      x0:     np.ndarray,
                      A:      np.ndarray,
                      B_ctrl: np.ndarray,
                      D:      np.ndarray,
                      d:      np.ndarray,
                      E:      np.ndarray,
                      F:      np.ndarray,
                      H:      np.ndarray,
                      c_lcs:  np.ndarray,
                      J_n:    np.ndarray,
                      J_t:    np.ndarray,
                      mu:     float,
                      Q:      np.ndarray,
                      R:      np.ndarray,
                      QN:     np.ndarray,
                      x_ref:  np.ndarray,
                      N:      int   = 8,
                      admm_iter: int = 10,
                      torque_limit: float = 30.0,
                      phi:    np.ndarray | None = None,
                      ) -> tuple[np.ndarray, np.ndarray]:
        """
        C3+ ADMM solve (Bui 2026 ICRA §IV-B.2).                      ← C3+ NEW

        Differences from C3 (the "← NEW" tags below mark each delta):
          1. Decision variable z is augmented with slack η:           ← NEW
                z_k = [x_k, λ_k, u_k, η_k]   per step (was [x_k, λ_k, u_k])
          2. QP includes equality constraint η_k = E x_k + F λ_k +    ← NEW
             H u_k + c (one block of n_λ rows per timestep).
          3. Soft-complementarity penalty (w_comp · phi · λ_n) in the ← NEW
             linear cost q is REMOVED — η is the hard expression.
          4. δ-update on (λ_k, η_k) uses Bui eq (12) componentwise    ← NEW
             — not C3's per-contact Lorentz projection.

        Identical to C3 (kept):
          - x and u blocks of δ-update pass through (no projection)
          - ω-update: ω += (z − δ)
          - x_0 fixation, dynamics equality, λ_n ≥ 0, torque bbox
          - Adaptive ρ schedule (Boyd §3.4.1) every 10 iters
        """
        n_x = self.n_x
        n_u = self.n_u
        rho = self.rho

        num_normals = J_n.shape[0]
        # Phase 2: λ now includes Stewart-Trinkle's friction-cone slack γ
        # so n_lambda = 6·num_normals (= 2·n_c + 4·n_c). The Bui eq. (12)
        # componentwise projection still operates pair-by-pair, so this
        # change is transparent to the projection logic itself.
        n_lambda    = 2 * num_normals + J_t.shape[0]          # = 6·num_normals
        # ===== C3+ NEW: per-step block size doubles the contact slot =====
        TOT       = n_x + n_lambda + n_u + n_lambda
        total_dim = N * TOT + n_x

        # Per-step slot offsets (within a single TOT-sized block)
        SX  = 0
        SL  = SX + n_x                                        # λ slot start
        SU  = SL + n_lambda                                   # u slot start
        SE  = SU + n_u                                        # η slot start

        # Cached identity for full z-vector regularization
        if total_dim != self._eye_total_c3p_dim:
            self._eye_total_c3p     = np.eye(total_dim)
            self._eye_total_c3p_dim = total_dim
        _eye_total = self._eye_total_c3p

        # One-time debug print
        if not getattr(self, '_debug_printed_c3p', False):
            self._debug_printed_c3p = True
            print(f"[DEBUG-C3+] n_contacts={num_normals}, n_lambda={n_lambda}, "
                  f"TOT={TOT}, total_dim={total_dim}")
            print(f"[DEBUG-C3+] per-step slots:  x=[{SX}:{SL})  "
                  f"λ=[{SL}:{SU})  u=[{SU}:{SE})  η=[{SE}:{TOT})")

        # ---------------------------------------------------------------
        # QP cost: P = 2·diag(Q,_,_,_,_, R block, _,_,...)·etc + ρ·I
        # ---------------------------------------------------------------
        with timed("admm.qp_build"):
            P     = np.zeros((total_dim, total_dim))
            q_ref = np.zeros(total_dim)

            for i in range(1, N):
                xi = i * TOT
                P[xi:xi+n_x, xi:xi+n_x] += 2.0 * Q
                q_ref[xi:xi+n_x]          = -2.0 * (Q @ x_ref)

            xN = N * TOT
            P[xN:xN+n_x, xN:xN+n_x] += 2.0 * QN
            q_ref[xN:xN+n_x]          = -2.0 * (QN @ x_ref)

            for i in range(N):
                ui = i * TOT + SU
                P[ui:ui+n_u, ui:ui+n_u] += 2.0 * R

            # ===== C3+ NEW: NO soft-complementarity penalty here =====
            # The η = E x + F λ + H u + c equality below replaces the
            # `q_ref[λ_n] += w_comp · phi_gap` hack used by C3.

            P_total = P + rho * _eye_total
            P_sym   = 0.5 * (P_total + P_total.T) + 1e-8 * _eye_total

            # ---------------------------------------------------------------
            # Equality constraints: x_0 fixation + N dynamics + N η-slack
            # ---------------------------------------------------------------
            n_eq_state = n_x + N * n_x                        # x_0 + N dynamics
            n_eq_eta   = N * n_lambda                         # ← C3+ NEW
            n_eq       = n_eq_state + n_eq_eta
            C_eq = np.zeros((n_eq, total_dim))
            b_eq = np.zeros(n_eq)

            # Row block 1: x_0 = x0
            C_eq[:n_x, :n_x] = self._eye_nx
            b_eq[:n_x]        = x0

            # Row block 2: A x_i + D λ_i + B u_i − x_{i+1} = −d
            for i in range(N):
                row  = n_x + i * n_x
                xi   = i * TOT
                li   = xi + SL
                ui   = xi + SU
                xnxt = (i + 1) * TOT if i < N - 1 else N * TOT

                C_eq[row:row+n_x, xi:xi+n_x]                 = A
                if n_lambda > 0:
                    C_eq[row:row+n_x, li:li+n_lambda]        = D
                C_eq[row:row+n_x, ui:ui+n_u]                 = B_ctrl
                C_eq[row:row+n_x, xnxt:xnxt+n_x]             = -self._eye_nx
                b_eq[row:row+n_x]                             = -d

            # ===== C3+ NEW: Row block 3 — slack equality =====
            # η_i − E x_i − F λ_i − H u_i = c
            if n_lambda > 0:
                for i in range(N):
                    row  = n_eq_state + i * n_lambda
                    xi   = i * TOT
                    li   = xi + SL
                    ui   = xi + SU
                    ei   = xi + SE

                    C_eq[row:row+n_lambda, xi:xi+n_x]        = -E
                    C_eq[row:row+n_lambda, li:li+n_lambda]   = -F
                    C_eq[row:row+n_lambda, ui:ui+n_u]        = -H
                    C_eq[row:row+n_lambda, ei:ei+n_lambda]   = np.eye(n_lambda)
                    b_eq[row:row+n_lambda]                    = c_lcs

            # ---------------------------------------------------------------
            # Build MathematicalProgram once
            # ---------------------------------------------------------------
            prog  = ad.MathematicalProgram()
            z_var = prog.NewContinuousVariables(total_dim, "z")

            prog.AddLinearEqualityConstraint(C_eq, b_eq, z_var)

            # All of λ = [γ; λ_n; λ_t] are non-negative under Stewart-Trinkle
            # complementarity. Phase 2: bound the full 6·n_c λ slot ≥ 0.
            # The Bui eq. (12) projection on (λ_j, η_j) pairs handles the
            # signs structurally, but enforcing ≥ 0 in the QP keeps OSQP
            # from drifting into infeasible regions.
            if n_lambda > 0:
                for i in range(N):
                    prog.AddBoundingBoxConstraint(
                        np.zeros(n_lambda),
                        np.full(n_lambda, np.inf),
                        z_var[i*TOT + SL : i*TOT + SL + n_lambda],
                    )

            # Torque bounds per step
            for i in range(N):
                ui = i * TOT + SU
                prog.AddBoundingBoxConstraint(
                    np.full(n_u, -torque_limit),
                    np.full(n_u,  torque_limit),
                    z_var[ui : ui + n_u],
                )

            cost_bd = prog.AddQuadraticCost(
                P_sym, np.zeros(total_dim), z_var)

        # ---------------------------------------------------------------
        # ADMM iterations
        # ---------------------------------------------------------------
        delta      = np.zeros(total_dim)
        omega      = np.zeros(total_dim)
        delta_prev = np.zeros(total_dim)

        z_sol = np.zeros(total_dim)
        for i in range(N):
            z_sol[i * TOT : i * TOT + n_x] = x0
        z_sol[N * TOT : N * TOT + n_x] = x0

        primal_hist = []
        dual_hist   = []
        tol         = 1e-3
        actual_iters = admm_iter
        u_lam_w = self._u_lambda
        u_eta_w = self._u_eta

        for it in range(admm_iter):
            delta_prev = delta.copy()

            with timed("admm.qp_build"):
                q_total = q_ref - rho * (delta - omega)
                cost_bd.evaluator().UpdateCoefficients(P_sym, q_total)

            with timed("admm.osqp_solve"):
                res = self._solver.Solve(prog)

            if res.is_success():
                z_sol = res.GetSolution(z_var)

            with timed("admm.z_update"):
                # ===== δ-update (C3+ NEW): x and u pass through =====
                delta = z_sol + omega

                # ===== δ-update (C3+ NEW): per-component eq (12) on (λ, η) =====
                if n_lambda > 0:
                    for i in range(N):
                        li = i * TOT + SL
                        ei = i * TOT + SE
                        lam_blk = z_sol[li:li+n_lambda] + omega[li:li+n_lambda]
                        eta_blk = z_sol[ei:ei+n_lambda] + omega[ei:ei+n_lambda]
                        d_lam, d_eta = self._project_componentwise(
                            lam_blk, eta_blk, u_lam_w, u_eta_w)
                        delta[li:li+n_lambda] = d_lam
                        delta[ei:ei+n_lambda] = d_eta

            omega = omega + z_sol - delta

            # Residuals over (λ, η) blocks
            if n_lambda > 0:
                lam_vec = np.concatenate([
                    np.concatenate([
                        z_sol[i*TOT + SL : i*TOT + SL + n_lambda],
                        z_sol[i*TOT + SE : i*TOT + SE + n_lambda],
                    ])
                    for i in range(N)
                ])
                dlt_vec = np.concatenate([
                    np.concatenate([
                        delta[i*TOT + SL : i*TOT + SL + n_lambda],
                        delta[i*TOT + SE : i*TOT + SE + n_lambda],
                    ])
                    for i in range(N)
                ])
                dlt_prev_vec = np.concatenate([
                    np.concatenate([
                        delta_prev[i*TOT + SL : i*TOT + SL + n_lambda],
                        delta_prev[i*TOT + SE : i*TOT + SE + n_lambda],
                    ])
                    for i in range(N)
                ])
                pr = float(np.linalg.norm(lam_vec - dlt_vec))
                dr = float(rho * np.linalg.norm(dlt_vec - dlt_prev_vec))
                primal_hist.append(pr)
                dual_hist.append(dr)

                if (it + 1) % 10 == 0:
                    if pr > 10.0 * dr and rho < 1000.0:
                        rho   *= 2.0
                        omega /= 2.0
                        P_total2 = P + rho * _eye_total
                        P_sym    = 0.5 * (P_total2 + P_total2.T) + 1e-8 * _eye_total
                        cost_bd.evaluator().UpdateCoefficients(P_sym, q_total)
                    elif dr > 10.0 * pr and rho > 0.1:
                        rho   /= 2.0
                        omega *= 2.0
                        P_total2 = P + rho * _eye_total
                        P_sym    = 0.5 * (P_total2 + P_total2.T) + 1e-8 * _eye_total
                        cost_bd.evaluator().UpdateCoefficients(P_sym, q_total)

                if pr < tol and dr < tol:
                    actual_iters = it + 1
                    break

        if n_lambda > 0 and primal_hist:
            mono = all(primal_hist[i] >= primal_hist[i+1]
                       for i in range(len(primal_hist)-1))
            print(f"[ADMM-C3+] primal: {primal_hist[0]:.4f}->{primal_hist[-1]:.4f}  "
                  f"dual: {dual_hist[0]:.4f}->{dual_hist[-1]:.4f}  "
                  f"mono={mono}  iters={actual_iters}/{admm_iter}  rho={rho:.1f}")

        # ---------------------------------------------------------------
        # Extract outputs
        # ---------------------------------------------------------------
        u_seq = np.zeros((N, n_u))
        x_seq = np.zeros((N + 1, n_x))
        for i in range(N):
            x_seq[i] = z_sol[i * TOT : i * TOT + n_x]
            u_seq[i] = z_sol[i * TOT + SU : i * TOT + SU + n_u]
        x_seq[N] = z_sol[N * TOT : N * TOT + n_x]

        # ---------------------------------------------------------------
        # Diagnostics — mirror C3's [MATH.QP], [MATH.δ], [MATH.ω] blocks.
        # ---------------------------------------------------------------
        self._diag_step += 1

        # ---- [MATH.QP-C3+] every 10th control step ---------------------
        if self._math_diag and self._diag_step % 10 == 0:
            dim = P_sym.shape[0]
            is_sym = bool(np.allclose(P_sym, P_sym.T, atol=1e-8))
            if dim <= 1000:
                eigs    = np.linalg.eigvalsh(P_sym)
                min_eig = float(eigs.min())
                max_eig = float(eigs.max())
                pos_sd  = min_eig >= -1e-8
                cond_val = max_eig / max(abs(min_eig), 1e-30)
                cond_str = _fmt(cond_val)
            else:
                pos_sd   = "?"
                cond_str = f"skipped (dim={dim}>1000)"
            q_norm = float(np.linalg.norm(q_total))
            osqp_ok = res.is_success()
            osqp_status = "solved" if osqp_ok else "failed/infeasible"
            try:
                _det = res.get_solver_details()
                osqp_iters_val = int(getattr(_det, 'iters',
                                     getattr(_det, 'iter', -1)))
                osqp_time_ms   = float(getattr(_det, 'run_time',
                                       float('nan'))) * 1000.0
            except Exception:
                osqp_iters_val = -1
                osqp_time_ms   = float('nan')
            print(f"[MATH.QP-C3+] Minimizing: (1/2) z^T P z + q^T z  "
                  f"(z augmented with η)")
            print(f"[MATH.QP-C3+]   s.t. A_eq z = b_eq  "
                  f"({n_eq} rows = x_0 fixation + {N} dynamics + "
                  f"{n_eq_eta} η-slack rows; slack-equality block ADDED)")
            print(f"[MATH.QP-C3+]        bbox: λ_n ≥ 0, |u| ≤ {torque_limit:.1f} Nm  "
                  f"(η is unbounded — sign enforced via projection eq 12)")
            print(f"[MATH.QP-C3+] P shape=({dim},{dim}), symmetric={is_sym}, "
                  f"pos-semidef={pos_sd}, cond(P)={cond_str}")
            print(f"[MATH.QP-C3+] q norm={_fmt(q_norm)}")
            print(f"[MATH.QP-C3+] Augmented term: (ρ/2) Σ ||z-δ+ω||^2_G  "
                  f"ρ={_fmt(rho)}  (G=I in v1: u_λ=u_η=1)")
            print(f"[MATH.QP-C3+] Soft complementarity: w_comp={_fmt(self._w_comp)}  "
                  f"(C3+ disables it — η equality replaces the linear penalty)")
            _time_str = (f"{osqp_time_ms:.2f}ms"
                         if not (isinstance(osqp_time_ms, float)
                                 and np.isnan(osqp_time_ms))
                         else "?ms")
            print(f"[MATH.QP-C3+] OSQP status: {osqp_status}, "
                  f"iters={osqp_iters_val}, solve time={_time_str}")

        if self._math_diag and n_lambda > 0:
            sqrt_ratio = float(np.sqrt(u_lam_w / u_eta_w))
            _c1 = _c2 = _c3 = 0
            _pre_lam = _pre_eta = 0.0
            _post_dlam = _post_deta = 0.0
            for _i in range(N):
                _li = _i * TOT + SL
                _ei = _i * TOT + SE
                lam_p = z_sol[_li:_li+n_lambda] + omega[_li:_li+n_lambda]
                eta_p = z_sol[_ei:_ei+n_lambda] + omega[_ei:_ei+n_lambda]
                for j in range(n_lambda):
                    _l, _e = float(lam_p[j]), float(eta_p[j])
                    if _e >= 0.0 and _e >= sqrt_ratio * _l:
                        _c1 += 1
                    elif _l >= 0.0 and _e <  sqrt_ratio * _l:
                        _c2 += 1
                    else:
                        _c3 += 1
                if _i == 0:
                    _pre_lam   = float(lam_p[0])
                    _pre_eta   = float(eta_p[0])
                    _post_dlam = float(delta[_li])
                    _post_deta = float(delta[_ei])
            print(f"[MATH.δ-C3+] Bui 2026 eq (12) projection results "
                  f"(N={N} × {n_lambda} λ-components):")
            print(f"[MATH.δ-C3+]   case 1 (η wins, λ→0):     {_c1}")
            print(f"[MATH.δ-C3+]   case 2 (λ wins, η→0):     {_c2}")
            print(f"[MATH.δ-C3+]   case 3 (both zero):       {_c3}")
            print(f"[MATH.δ-C3+] First step k=0, component 0:")
            print(f"[MATH.δ-C3+]   pre:  λ°={_fmt(_pre_lam)}, η°={_fmt(_pre_eta)}")
            print(f"[MATH.δ-C3+]   post: δ_λ={_fmt(_post_dlam)}, "
                  f"δ_η={_fmt(_post_deta)}")

        # ---- [MATH.ω-C3+] every control step ---------------------------
        if self._math_diag:
            _omega_norm = float(np.linalg.norm(omega))
            if n_lambda > 0 and primal_hist:
                _pr = primal_hist[-1]
                _dr = dual_hist[-1]
                _ratio = _pr / (_dr + 1e-30)
                _lam_f = np.concatenate([
                    np.concatenate([
                        z_sol[_i*TOT + SL : _i*TOT + SL + n_lambda],
                        z_sol[_i*TOT + SE : _i*TOT + SE + n_lambda],
                    ])
                    for _i in range(N)
                ])
                _dlt_f = np.concatenate([
                    np.concatenate([
                        delta[_i*TOT + SL : _i*TOT + SL + n_lambda],
                        delta[_i*TOT + SE : _i*TOT + SE + n_lambda],
                    ])
                    for _i in range(N)
                ])
                _ld_max = float(np.max(np.abs(_lam_f - _dlt_f)))
            else:
                _pr = _dr = _ratio = _ld_max = 0.0
            print(f"[MATH.ω-C3+] ω update: ω += (z-δ), over {actual_iters} ADMM iters")
            print(f"[MATH.ω-C3+] ||ω||={_fmt(_omega_norm)}, "
                  f"||z-δ||_max={_fmt(_ld_max)}  "
                  f"(scale differs from C3 — ω carries η-block residuals too)")
            if n_lambda > 0:
                if _ratio > 10.0:
                    _rho_note = (f"ratio={_fmt(_ratio)} > 10 "
                                 f"→ would double ρ to {_fmt(rho*2)}")
                elif _ratio < 0.1:
                    _rho_note = (f"ratio={_fmt(_ratio)} < 0.1 "
                                 f"→ would halve ρ to {_fmt(rho/2)}")
                else:
                    _rho_note = f"ratio={_fmt(_ratio)} → ρ unchanged"
                print(f"[MATH.ω-C3+] ρ decision: primal={_fmt(_pr)}, "
                      f"dual={_fmt(_dr)}, {_rho_note}")
            else:
                print(f"[MATH.ω-C3+] ρ decision: n/a (n_λ=0, no contact variables)")
            _never = " ← never triggers!" if admm_iter < 10 else ""
            print(f"[MATH.ω-C3+] Note: adaptive-ρ fires every 10 iters; "
                  f"current max_iter={admm_iter}{_never}")

        # ---- [MATH.STATE] one-shot dump at step 1 (C3+ only) -----------
        # Captures the exact inputs and ADMM trace for closed-form
        # cross-checking in MATH_C3PLUS.md. Three guards (math_diag,
        # mode=c3plus by virtue of being inside _solve_c3plus, step==1).
        if self._math_diag and self._diag_step == 1:
            import json
            import os
            _dump_dir  = "results/math_state_dumps"
            _dump_path = os.path.join(_dump_dir, "c3plus_step1.json")
            os.makedirs(_dump_dir, exist_ok=True)
            _dump = {
                "step": int(self._diag_step),
                "mode": "c3plus",
                "N": int(N),
                "admm_iter": int(admm_iter),
                "actual_iters": int(actual_iters),
                "rho_final": float(rho),
                "n_x": int(n_x),
                "n_u": int(n_u),
                "n_lambda": int(n_lambda),
                "num_normals": int(num_normals),
                "u_lambda": float(self._u_lambda),
                "u_eta":    float(self._u_eta),
                "mu":       float(mu),
                "torque_limit": float(torque_limit),
                "x0":       x0.tolist(),
                "x_ref":    x_ref.tolist(),
                "phi":      (phi.tolist() if phi is not None else None),
                "A_diag":   np.diag(A).tolist(),
                "A_top_right_norm": float(np.linalg.norm(A[:n_x//2, n_x//2:])
                                          if n_x > 1 else 0.0),
                "B_ctrl_norm": float(np.linalg.norm(B_ctrl)),
                "D_norm":   float(np.linalg.norm(D)),
                "D_first_row": (D[0, :].tolist() if D.size else None),
                "d":        d.tolist(),
                "J_n_row0": (J_n[0, :].tolist() if J_n.shape[0] > 0 else None),
                "J_t_row0": (J_t[0, :].tolist() if J_t.shape[0] > 0 else None),
                "E_row0":   (E[0, :].tolist() if E.shape[0] > 0 else None),
                "F_diag":   (np.diag(F).tolist()
                             if F.shape[0] == F.shape[1] and F.size > 0
                             else None),
                "F_norm":   float(np.linalg.norm(F)),
                "H_row0":   (H[0, :].tolist() if H.shape[0] > 0 else None),
                "H_norm":   float(np.linalg.norm(H)),
                "c_lcs":    c_lcs.tolist(),
                # Quick zero-row indicator for v1's deferred friction
                "E_zero_row_count": int(np.sum(
                    np.linalg.norm(E, axis=1) < 1e-12)) if E.size else 0,
                "E_total_rows":     int(E.shape[0]),
                # ADMM trace
                "z_sol":        z_sol.tolist(),
                "delta_final":  delta.tolist(),
                "omega_final":  omega.tolist(),
                "primal_hist":  [float(v) for v in primal_hist],
                "dual_hist":    [float(v) for v in dual_hist],
                # Slot offsets within a single per-step block (TOT-sized)
                "TOT":  int(TOT),
                "SX":   int(SX),
                "SL":   int(SL),
                "SU":   int(SU),
                "SE":   int(SE),
                # First u and λ_n value extracted from z_sol
                "u_step0":   z_sol[SU : SU + n_u].tolist(),
                "lambda_step0": z_sol[SL : SL + n_lambda].tolist(),
                "eta_step0":    z_sol[SE : SE + n_lambda].tolist(),
            }
            with open(_dump_path, "w") as _f:
                json.dump(_dump, _f, indent=2)
            print(f"[MATH.STATE] step 1 dump written to {_dump_path}  "
                  f"(E_zero_rows={_dump['E_zero_row_count']}/"
                  f"{_dump['E_total_rows']})")

        # Single-line summary every step (mirrors C3's [C3] line).
        # Phase 2: λ = [γ; λ_n; λ_t]; the λ_n block now starts at SL+num_normals.
        if n_lambda > 0:
            lam_n_all = np.concatenate([
                z_sol[i * TOT + SL + num_normals
                      : i * TOT + SL + 2 * num_normals]
                for i in range(N)
            ]) if num_normals else np.zeros(0)
            eta_n_all = np.concatenate([
                z_sol[i * TOT + SE + num_normals
                      : i * TOT + SE + 2 * num_normals]
                for i in range(N)
            ]) if num_normals else np.zeros(0)
            lam_n_max = float(lam_n_all.max()) if lam_n_all.size else 0.0
            eta_n_max = float(eta_n_all.max()) if eta_n_all.size else 0.0
            pr_last   = primal_hist[-1] if primal_hist else float('nan')
            print(f"[C3+] step={self._diag_step} "
                  f"|u[0]|={np.linalg.norm(u_seq[0]):.2f}Nm "
                  f"λ_n_max={lam_n_max:.3f} η_n_max={eta_n_max:.3f} "
                  f"primal={pr_last:.3f} iters={actual_iters}/{admm_iter}")
        else:
            print(f"[C3+] step={self._diag_step} n_λ=0  "
                  f"|u[0]|={np.linalg.norm(u_seq[0]):.3f} Nm")

        return u_seq, x_seq


# Module-level alias so unit tests can import without instantiating C3Solver
def project_lorentz(lam_n: float,
                    lam_t: "np.ndarray",
                    mu: float) -> "tuple[float, np.ndarray]":
    """Public wrapper around C3Solver._project_single_contact for testing."""
    return C3Solver._project_single_contact(lam_n, lam_t, mu)


def project_componentwise_eq12(lam: "np.ndarray",
                               eta: "np.ndarray",
                               u_lambda: float = 1.0,
                               u_eta:    float = 1.0
                               ) -> "tuple[np.ndarray, np.ndarray]":
    """Public wrapper around C3Solver._project_componentwise for testing.   ← C3+ NEW

    Implements Bui 2026 ICRA eq (12) — the C3+ δ-update closed form.
    """
    return C3Solver._project_componentwise(lam, eta, u_lambda, u_eta)
