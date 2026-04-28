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
                 math_diag: bool = False):
        self.n_x        = n_x
        self.n_u        = n_u
        self.rho        = rho
        self._math_diag = math_diag
        self._w_comp    = 100.0   # exposed so C3MPC can read it for [MATH.setup]
        self._solver    = ad.OsqpSolver()
        self._diag_step = 0
        # Pre-allocated identity matrices — n_x is fixed; total_dim is cached on first use
        self._eye_nx         = np.eye(n_x)
        self._eye_total_dim  = -1       # sentinel: rebuild when total_dim changes
        self._eye_total      = None

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
        n_x = self.n_x
        n_u = self.n_u
        rho = self.rho

        num_normals = J_n.shape[0]
        n_c         = num_normals + J_t.shape[0]
        TOT         = n_x + n_c + n_u
        total_dim   = N * TOT + n_x

        # Reuse cached identity; rebuild only when total_dim changes (rare)
        if total_dim != self._eye_total_dim:
            self._eye_total     = np.eye(total_dim)
            self._eye_total_dim = total_dim
        _eye_total = self._eye_total

        # One-time debug print to verify contact slicing
        if not getattr(self, '_debug_printed', False):
            self._debug_printed = True
            lam_n_start = n_x
            lam_n_end   = n_x + num_normals
            lam_t_start = lam_n_end
            lam_t_end   = n_x + n_c
            print(f"[DEBUG] n_contacts={num_normals}, n_c={n_c}, "
                  f"TOT={TOT}, total_dim={total_dim}")
            print(f"[DEBUG] per-step λ_n slice: [{lam_n_start}:{lam_n_end}]  "
                  f"λ_t slice: [{lam_t_start}:{lam_t_end}]")
            print(f"[DEBUG] J_n shape: {J_n.shape}, J_t shape: {J_t.shape}")
            print(f"[DEBUG] λ_t per contact: 4 components each → "
                  f"contact i at [{lam_t_start}+4i:{lam_t_start}+4(i+1)]")

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
                ui = i * TOT + n_x + n_c
                P[ui:ui+n_u, ui:ui+n_u] += 2.0 * R

            # Soft complementarity: penalise λ_n · φ for each contact at every
            # horizon step.  When φ > 0 (gap, no contact) the QP pays w_comp
            # per unit of planned normal force, biasing the solution toward
            # zero λ_n until the arm actually closes the gap.  This is the
            # linear-approximation smooth-complementarity penalty used in C3+.
            # OSQP minimises 0.5 z^T P z + q^T z, so λ_n cost is in q.
            w_comp = self._w_comp
            if num_normals > 0 and phi is not None and len(phi) == num_normals:
                phi_gap = np.clip(phi, 0.0, None)   # penalise gap only
                for i in range(N):
                    li = i * TOT + n_x
                    q_ref[li : li + num_normals] += w_comp * phi_gap

            P_total = P + rho * _eye_total
            # Symmetrise + small diagonal regularisation for OSQP
            P_sym   = 0.5 * (P_total + P_total.T) + 1e-8 * _eye_total

            # ---- Equality constraint matrix (initial state + dynamics) ---
            # Rows 0..n_x-1       : x_0 = x0
            # Rows n_x + i*n_x    : A x_i + D λ_i + B u_i − x_{i+1} = −d
            n_eq = (N + 1) * n_x
            C_eq = np.zeros((n_eq, total_dim))
            b_eq = np.zeros(n_eq)

            C_eq[:n_x, :n_x] = self._eye_nx
            b_eq[:n_x]        = x0

            for i in range(N):
                row  = n_x + i * n_x
                xi   = i * TOT
                li   = xi + n_x
                ui   = li + n_c
                xnxt = (i + 1) * TOT if i < N - 1 else N * TOT

                C_eq[row:row+n_x, xi:xi+n_x]     = A
                if n_c > 0:
                    C_eq[row:row+n_x, li:li+n_c] = D
                C_eq[row:row+n_x, ui:ui+n_u]     = B_ctrl
                C_eq[row:row+n_x, xnxt:xnxt+n_x] = -self._eye_nx
                b_eq[row:row+n_x]                 = -d

            # ---- Build MathematicalProgram ONCE per control step ---------
            prog  = ad.MathematicalProgram()
            z_var = prog.NewContinuousVariables(total_dim, "z")

            prog.AddLinearEqualityConstraint(C_eq, b_eq, z_var)

            # λ_n ≥ 0 per horizon step
            if num_normals > 0:
                for i in range(N):
                    prog.AddBoundingBoxConstraint(
                        np.zeros(num_normals),
                        np.full(num_normals, np.inf),
                        z_var[i*TOT + n_x : i*TOT + n_x + num_normals],
                    )

            # Torque bounds per horizon step
            for i in range(N):
                ui = i * TOT + n_x + n_c
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
                # δ-update: λ blocks → Lorentz cone; x and u pass through
                delta = z_sol + omega
                if n_c > 0:
                    for i in range(N):
                        li = i * TOT + n_x
                        delta[li : li + n_c] = self._lorentz_project(
                            z_sol[li : li + n_c] + omega[li : li + n_c],
                            num_normals, mu,
                        )

            omega = omega + z_sol - delta

            # Track contact-variable residuals only (x/u are unconstrained)
            if n_c > 0:
                lam_vec = np.concatenate([
                    z_sol[i * TOT + n_x : i * TOT + n_x + n_c]
                    for i in range(N)
                ])
                dlt_vec = np.concatenate([
                    delta[i * TOT + n_x : i * TOT + n_x + n_c]
                    for i in range(N)
                ])
                dlt_prev_vec = np.concatenate([
                    delta_prev[i * TOT + n_x : i * TOT + n_x + n_c]
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
        if n_c > 0 and primal_hist:
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
            u_seq[i] = z_sol[i * TOT + n_x + n_c : i * TOT + n_x + n_c + n_u]
        x_seq[N] = z_sol[N * TOT : N * TOT + n_x]

        # ---- Contact diagnostics ----------------------------------------
        self._diag_step += 1

        if n_c > 0:
            # Always extract summary scalars for the single-line print
            lam_n_all = np.concatenate([
                z_sol[i * TOT + n_x : i * TOT + n_x + num_normals]
                for i in range(N)
            ]) if num_normals else np.zeros(0)
            delta_t_all = np.concatenate([
                delta[i * TOT + n_x + num_normals : i * TOT + n_x + n_c]
                for i in range(N)
            ]) if n_c > num_normals else np.zeros(0)
            lam_n_max = float(lam_n_all.max()) if lam_n_all.size else 0.0
            dt_max    = float(delta_t_all.max()) if delta_t_all.size else 0.0
            pr_last   = primal_hist[-1] if primal_hist else float('nan')

            if self._diag_step % 20 == 0:
                # Full table every 20 MPC steps
                lam_n_seq   = np.zeros((N, num_normals))
                dlt_n_seq   = np.zeros((N, num_normals))
                lam_t_norms = np.zeros(N)
                dlt_t_norms = np.zeros(N)
                for i in range(N):
                    li = i * TOT + n_x
                    lam_n_seq[i]   = z_sol[li : li + num_normals]
                    dlt_n_seq[i]   = delta[li : li + num_normals]
                    lt_start       = li + num_normals
                    lam_t_norms[i] = np.linalg.norm(z_sol[lt_start : lt_start + n_c - num_normals])
                    dlt_t_norms[i] = np.linalg.norm(delta[lt_start : lt_start + n_c - num_normals])

                print(f"[C3diag] step={self._diag_step} nc={num_normals}  "
                      f"step | λ_n(z)     δ_n        |δ_t|      cone(δ)?  |λ_t|(z)")
                for i in range(N):
                    ln  = lam_n_seq[i].max() if num_normals else 0.0
                    dn  = dlt_n_seq[i].max() if num_normals else 0.0
                    lt  = lam_t_norms[i]
                    dt  = dlt_t_norms[i]
                    cone_ok = dt <= mu * dn + 1e-4
                    tag = ""
                    if dn < 1e-6 and ln < 1e-6:
                        tag = "  ← both zero"
                    elif not cone_ok:
                        tag = "  ← PROJECTION BUG"
                    elif lt > mu * ln + 1e-4:
                        tag = "  (z violates, δ OK — ADMM converging)"
                    print(f"  [{i:02d}]  {ln:10.4f}  {dn:10.4f}  {dt:10.4f}  "
                          f"{'OK' if cone_ok else 'FAIL'}    {lt:10.4f}{tag}")
                any_contact = dlt_n_seq.max() > 1e-6
                print(f"  → δ contact: {any_contact}  "
                      f"|u[0]|={np.linalg.norm(u_seq[0]):.3f} Nm")
            else:
                # Single-line summary every other step
                print(f"[C3] step={self._diag_step} "
                      f"|u[0]|={np.linalg.norm(u_seq[0]):.2f}Nm "
                      f"λ_n_max={lam_n_max:.3f} |δ_t|_max={dt_max:.3f} "
                      f"primal={pr_last:.3f} iters={actual_iters}/{admm_iter}")
        else:
            print(f"[C3] step={self._diag_step} n_c=0  "
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
            # Soft complementarity penalty VALUE at the current solution
            comp_penalty = 0.0
            if num_normals > 0 and phi is not None and len(phi) == num_normals:
                phi_gap = np.clip(phi, 0.0, None)
                for _i in range(N):
                    _li = _i * TOT + n_x
                    comp_penalty += float(
                        w_comp * (phi_gap @ z_sol[_li : _li + num_normals])
                    )
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
            print(f"[MATH.QP]        bbox: λ_n ≥ 0, |u| ≤ {torque_limit:.1f} Nm  "
                  f"(Drake BoundingBox — no explicit G/h matrices)")
            print(f"[MATH.QP] P shape=({dim},{dim}), symmetric={is_sym}, "
                  f"pos-semidef={pos_sd}, cond(P)={cond_str}")
            print(f"[MATH.QP] q norm={_fmt(q_norm)}")
            print(f"[MATH.QP] Augmented term: (ρ/2) Σ ||λ-δ+ω||^2  ρ={_fmt(rho)}")
            print(f"[MATH.QP] Soft complementarity: w_comp={_fmt(w_comp)} · "
                  f"Σ λ_n·max(φ,0) = {_fmt(comp_penalty)}")
            _time_str = (f"{osqp_time_ms:.2f}ms"
                         if not (isinstance(osqp_time_ms, float)
                                 and np.isnan(osqp_time_ms))
                         else "?ms")
            print(f"[MATH.QP] OSQP status: {osqp_status}, "
                  f"iters={osqp_iters_val}, solve time={_time_str}")

        # ---- [MATH.δ] every control step (when contacts exist) ----------------
        if self._math_diag and n_c > 0:
            _c1 = _c2 = _c3 = 0
            _pre_ln = _pre_lt = _pre_mu = 0.0
            _post_dn = _post_dt = _dphi = 0.0
            for _i in range(N):
                _li = _i * TOT + n_x
                for _ci in range(num_normals):
                    _ln  = float(z_sol[_li + _ci])
                    _lt4 = z_sol[_li + num_normals + 4*_ci
                                 : _li + num_normals + 4*(_ci+1)]
                    # Polyhedral → Cartesian for case detection
                    _Ft  = np.array([_lt4[0]-_lt4[1], _lt4[2]-_lt4[3]])
                    _bn  = float(np.linalg.norm(_Ft))
                    if _bn <= mu * _ln + 1e-12:
                        _c1 += 1
                    elif mu * _bn <= -_ln + 1e-12:
                        _c2 += 1
                    else:
                        _c3 += 1
                    if _i == 0 and _ci == 0:
                        _pre_ln  = _ln
                        _pre_lt  = float(np.linalg.norm(_lt4))
                        _pre_mu  = mu * _ln
                        _dn      = float(delta[_li + _ci])
                        _dt4     = delta[_li + num_normals
                                         : _li + num_normals + 4]
                        _phi0    = (float(phi[0])
                                    if phi is not None and len(phi) > 0
                                    else float('nan'))
                        _post_dn = _dn
                        _post_dt = float(np.linalg.norm(_dt4))
                        _dphi    = _dn * _phi0
            print(f"[MATH.δ] δ-projection results "
                  f"(N={N} horizon × {num_normals} contacts):")
            print(f"[MATH.δ]   case 1 (inside cone, ||λ_t|| ≤ μ·λ_n): {_c1}")
            print(f"[MATH.δ]   case 2 (polar cone, → apex):            {_c2}")
            print(f"[MATH.δ]   case 3 (surface projection):            {_c3}")
            print(f"[MATH.δ] First contact k=0, contact 0:")
            print(f"[MATH.δ]   pre:  λ_n={_fmt(_pre_ln)}, "
                  f"||λ_t||={_fmt(_pre_lt)}, μ·λ_n={_fmt(_pre_mu)}")
            print(f"[MATH.δ]   post: δ_n={_fmt(_post_dn)}, "
                  f"||δ_t||={_fmt(_post_dt)}, "
                  f"δ_n·φ={_fmt(_dphi)} (≈0 if complementarity holds)")

        # ---- [MATH.ω] every control step --------------------------------------
        if self._math_diag:
            _omega_norm = float(np.linalg.norm(omega))
            if n_c > 0 and primal_hist:
                _pr = primal_hist[-1]
                _dr = dual_hist[-1]
                _ratio = _pr / (_dr + 1e-30)
                _lam_f = np.concatenate([
                    z_sol[_i*TOT+n_x : _i*TOT+n_x+n_c] for _i in range(N)
                ])
                _dlt_f = np.concatenate([
                    delta[_i*TOT+n_x : _i*TOT+n_x+n_c] for _i in range(N)
                ])
                _ld_max = float(np.max(np.abs(_lam_f - _dlt_f)))
            else:
                _pr = _dr = _ratio = _ld_max = 0.0
            print(f"[MATH.ω] ω update: ω += (λ-δ), over {actual_iters} ADMM iters")
            print(f"[MATH.ω] ||ω||={_fmt(_omega_norm)}, "
                  f"||λ-δ||_max={_fmt(_ld_max)}")
            if n_c > 0:
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
                print(f"[MATH.ω] ρ decision: n/a (n_c=0, no contact variables)")
            _never = " ← never triggers!" if admm_iter < 10 else ""
            print(f"[MATH.ω] Note: adaptive-ρ fires every 10 iters; "
                  f"current max_iter={admm_iter}{_never}")

        return u_seq, x_seq


# Module-level alias so unit tests can import without instantiating C3Solver
def project_lorentz(lam_n: float,
                    lam_t: "np.ndarray",
                    mu: float) -> "tuple[float, np.ndarray]":
    """Public wrapper around C3Solver._project_single_contact for testing."""
    return C3Solver._project_single_contact(lam_n, lam_t, mu)
