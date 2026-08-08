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
                 math_diag: bool = False, mode: str = "c3",
                 penalize_input_change: bool = True):
        assert mode in ("c3", "c3plus"), f"unknown solver mode: {mode}"
        # Workspace state constraints (reference cc:995-1025): list of
        # (state_idx, lo, hi) applied per-knot in _solve_c3plus. Set by
        # main.py from planner_workspace_* yaml keys; None → no constraint.
        self.state_position_bounds: list | None = None
        # Lifetime count of QP-step failures (OSQP infeasible / not solved).
        # On failure the ADMM iter reuses the stale z_sol, which degenerates
        # to an all-zero plan when every solve fails (run p142) — so failures
        # must be counted and surfaced, never silent.
        self.qp_failures: int = 0
        # C3+ δ-projection is the paper's `componentwise` (Bui 2026
        # eq 12) per-scalar-pair test — matches reference
        # sampling_c3plus_options.yaml projection_type: 'C3+'. The
        # port previously carried an alternate LCP-projection variant
        # (Aydinoglu §V-B.3.b retrofit); it was reference-nonconformant
        # and empirically convergence-limited at admm_iter=3, so it was
        # removed. If a formal C3+ vs C3-classic study is needed later,
        # use --solver c3 (which still runs an LCP-based δ-step from
        # control/lcp_solver.py, matching the classic paper exactly).
        self.n_x        = n_x
        self.n_u        = n_u
        self.rho        = rho
        self._math_diag = math_diag
        self.mode       = mode                          # ← C3+ NEW
        # Unit string for u, derived from n_u (NOT hardcoded):
        #   n_u == 3 → EE Cartesian force in Newtons (Push-Anything §IV-A).
        #   else      → joint torque in Nm.
        # Used in [C3]/[C3+] step printouts and the MATH.QP torque-limit
        # line. A hardcoded "Nm" misled the reviewer this round (a 3-vector
        # of Newtons under --ee-space was printed with the Nm label); the
        # unit must be a function of the active formulation.
        self.u_unit_str = "N" if n_u == 3 else "Nm"
        self.u_unit_kind = "EE force" if n_u == 3 else "joint torque"
        # Soft-complementarity penalty: disabled in BOTH modes from Phase 2
        # onward.  C3 now uses LCP projection (Aydinoglu §V-B.3.b) which
        # produces feasible λ by construction; C3+ uses the slack-equality
        # η = E x + F λ + H u + c.  In either case the linear penalty
        # `w_comp · phi · λ_n` is structurally redundant and would bias
        # λ_n toward zero on contact.
        self._w_comp    = 0.0
        self._solver    = ad.OsqpSolver()
        # Reference solver_options_default.yaml settings — DEFAULT
        # (2026-07-28 reference-conformance defaults flip; formerly gated
        # on REFCONF_OSQP_OPTS).
        self._osqp_refopts = True
        self._osqp_solver_options = None
        if self._osqp_refopts:
            _so = ad.SolverOptions()
            _osqp_id = ad.OsqpSolver().solver_id()
            _so.SetOption(_osqp_id, "polishing", 1)
            _so.SetOption(_osqp_id, "polish_refine_iter", 3)
            _so.SetOption(_osqp_id, "warm_starting", 1)
            _so.SetOption(_osqp_id, "scaled_termination", 1)
            _so.SetOption(_osqp_id, "scaling", 10)
            _so.SetOption(_osqp_id, "adaptive_rho", 1)
            _so.SetOption(_osqp_id, "adaptive_rho_interval", 0)
            _so.SetOption(_osqp_id, "eps_abs", 1e-5)
            _so.SetOption(_osqp_id, "eps_rel", 1e-5)
            _so.SetOption(_osqp_id, "max_iter", 1000)
            self._osqp_solver_options = _so
            print("[OSQP-REFOPTS] active: polishing=1 adaptive_rho=1 "
                  "warm_starting=1 scaling=10 eps=1e-5 max_iter=1000",
                  flush=True)
        self._diag_step = 0
        # Pre-allocated identity matrices — n_x is fixed; total_dim is cached on first use
        self._eye_nx         = np.eye(n_x)
        self._eye_total_dim  = -1       # sentinel: rebuild when total_dim changes
        self._eye_total      = None
        # ===== C3+ specific (Bui 2026 §IV-B.2) =====               ← C3+ NEW
        # u_λ, u_η are the per-component G weights in eq (12). Reference
        # push_t/sampling_c3plus_options.yaml:120-125 sets u_lambda_list=20,
        # u_eta_list=1 (ratio 20:1). Port previously hardcoded 1:1 — this
        # matters for which projection case (λ→0 vs η→0) wins per component.
        # Reference's 20:1 penalizes λ mismatches 20× more strongly than
        # η — projection prefers case 1 (η wins, λ→0) more often, matching
        # reference C3+ convergence behavior. Fixed 2026-07-18 iter9.
        self._u_lambda           = 20.0
        self._u_eta              = 1.0
        # 4.j — reference penalize_changes_in_u_across_solves. When True,
        # the R-block cost becomes ‖u_k − u_prev_k‖²_R (matches c3.cc:302-310).
        # Reference `anything/sampling_c3plus_options.yaml`: true;
        # reference `push_t/sampling_c3plus_options.yaml`: false. Task-driven
        # per caller (see main.py).
        self._penalize_input_change = penalize_input_change
        # Cache the previous solve's u_seq for the delta penalty. Shape (N, n_u)
        # after the first solve; None (or first-call sentinel) means "use u=0
        # as u_prev on the first call".
        self._u_prev_solve = None
        # Reference rho_scale (c3.cc:389-390):
        #   w = w / rho_scale;  G = G * rho_scale;  (per ADMM iter)
        # push_t/parameters/sampling_c3plus_options.yaml sets rho_scale: 3.
        # Enabled 2026-07-17 after commit 4c3bad5 conformed admm_iter=25→3;
        # the port now runs the reference's 3-iter geometric ρ ramp instead
        # of the dead "adaptive-ρ every 10 iters" branch below (which the
        # solve loop even self-flags as unreachable at admm_iter=3).
        self._rho_scale = 3.0
        # 4.g — reference sampling_c3plus_options.yaml end_on_qp_step: false
        # → do an LCS rollout of x from x0 using solved (u, λ) after the
        # ADMM loop, so the returned x_seq is LCS-feasible even under
        # non-convergence. Port True = skip rollout (opt-out from reference).
        # Default now False = reference-conformant.
        self._end_on_qp_step = False
        # Bui §IV-B.2 final paragraph: large G-weight on EE-object contact
        # components in the final QP step. NOT applied here yet (would
        # require knowing which contact is EE↔box; see TODO in _solve_c3plus).
        self._w_G_ee_contact     = 1000.0
        self._eye_total_c3p_dim  = -1
        self._eye_total_c3p      = None
        # Reference-conformant per-slot G matrix for ADMM augmentation.
        # Reference c3_options.h:189 G = w_G · diag(g_vector). Reference
        # push_t/parameters/sampling_c3plus_options.yaml:70-101:
        #   w_G: 0.01
        #   g_x = 0 (all state slots)  → no state augmentation
        #   g_u = 0 (all input slots)  → no input augmentation
        #   g_lambda = 2 (all λ slots)
        #   g_eta    = 1 (all η slots)
        # Port previously applied uniform ρ (scalar) to ALL slots, meaning
        # state/input got 100× more augmentation than reference (rho=100 vs
        # ref 0) and λ got 50× more (100 vs 2). This ill-conditioned the
        # ADMM. When enabled, replaces `rho * I` with `rho * diag(G_diag)`
        # in the P and q_total updates. Default True for c3+ mode.
        # Env-gate: REFCONF_USE_G_MATRIX=1 to enable (default OFF).
        # 2026-07-22 test (p68/p69): full ref G with g_x=g_u=0 destabilizes
        # port ADMM — T physically tipped in p69, trans regressed in p68.
        # Reference-conformance for G matrix requires additional structural
        # changes (possibly per-slot ρ scaling, dt-dependent G tuning, or
        # different QP formulation) that were not in scope this session.
        # 2026-07-26 arc-2 resolved the destabilization (D1 rho-override +
        # factor-of-2 + 7 sibling fixes); 2026-07-28 defaults flip makes
        # G-on the DEFAULT (formerly REFCONF_USE_G_MATRIX=1, canonical
        # since p112).
        self._use_g_matrix = True
        self._w_G          = 0.01
        self._g_lambda     = 2.0
        self._g_eta        = 1.0
        self._g_x          = 0.0   # reference has zero state augmentation
        self._g_u          = 0.0   # reference has zero input augmentation
        # PORT_G_X override (diagnostic): per-slot x augmentation weight.
        # Set to e.g. "1" to add uniform g_x=1 for state-memory regularization
        # under G-ON — falsification test for the "sparse-Q + zero-g_x kills
        # ADMM convergence" hypothesis.
        import os as _os_g
        _pusha_gx = _os_g.environ.get("PORT_G_X", "")
        if _pusha_gx:
            try:
                self._g_x = float(_pusha_gx)
                print(f"[PORT_G_X] override: g_x={self._g_x}", flush=True)
            except ValueError:
                pass
        self._g_diag_c3p_cache: np.ndarray | None = None
        self._g_diag_c3p_shape: tuple | None = None
        # First-horizon contact force from the most recent solve. Read by
        # the impedance controller (Aydinoglu eq. 36 τ_ff = J_c^T λ_d).
        # _last_lambda_n_first : (num_normals,) — non-negative normal forces.
        # _last_lambda_t_first : (4*num_normals,) — non-negative polyhedral
        #                       tangent components [t1+, t1−, t2+, t2−].
        # γ slack is intentionally excluded — only the physical force
        # components contribute to τ_ff.
        self._last_lambda_n_first: np.ndarray | None = None
        self._last_lambda_t_first: np.ndarray | None = None
        # T-architecture Stage 1 substrate: full λ_n / λ_t over the planning
        # horizon. Populated alongside the first-knot views. Shape (N, num_normals)
        # for λ_n and (N, 4*num_normals) for λ_t (or empty if no contacts).
        # Stage 2 will let the OSC index into these between MPC re-solves.
        self._last_lambda_n_horizon: np.ndarray | None = None
        self._last_lambda_t_horizon: np.ndarray | None = None

        # D1+D2: dual-view (z_sol vs delta) exposure + convergence flag.
        # When ADMM converges (pr<tol AND dr<tol), z_sol and delta agree
        # on the λ block. When non-converged, z_sol is QP-optimal but
        # complementarity-leaked; delta is complementarity-feasible but
        # ω-leaked. Expose BOTH so consumer A/B is one bool flip.
        # Default consumers see delta (the GATE verdict's safer choice
        # under non-convergence).
        self.expose_zsol:        bool = False
        self._last_converged:    bool = True
        self._last_lambda_n_first_zsol:    np.ndarray | None = None
        self._last_lambda_n_first_delta:   np.ndarray | None = None
        self._last_lambda_t_first_zsol:    np.ndarray | None = None
        self._last_lambda_t_first_delta:   np.ndarray | None = None
        self._last_lambda_n_horizon_zsol:  np.ndarray | None = None
        self._last_lambda_n_horizon_delta: np.ndarray | None = None
        self._last_lambda_t_horizon_zsol:  np.ndarray | None = None
        self._last_lambda_t_horizon_delta: np.ndarray | None = None

        # Per-(mpc_step, admm_iter, horizon_k) λ probe. Disabled by default.
        # Enabled via enable_lambda_horizon_probe(); writes one CSV row per
        # (solve, iter, k, contact). Used to diagnose whether λ_n_gnd is
        # being driven up to the gravity-support level m·g across the
        # horizon or pinned to ~0 by the ADMM componentwise projection.
        self._lprobe_path:        str | None  = None
        self._lprobe_tags:        list[str]   = []   # contact tags per row of J_n
        self._lprobe_max_solves:  int         = 5
        self._lprobe_n_solves:    int         = 0
        self._lprobe_mpc_step:    int         = 0

    # ------------------------------------------------------------------
    def enable_lambda_horizon_probe(self,
                                    path: str,
                                    contact_tags: list[str],
                                    max_solves: int = 5) -> None:
        """Enable per-(solve, iter, k) λ trace dump. Caller passes contact
        tags (e.g. ['BOX-GND', 'EE-BOX']) ordered to match J_n rows."""
        import os
        self._lprobe_path       = path
        self._lprobe_tags       = list(contact_tags)
        self._lprobe_max_solves = int(max_solves)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Write header — λ_n per contact, ||λ_t|| per contact, γ per contact.
        with open(path, "w") as f:
            tag_cols = ",".join(
                f"lam_n_{t},lam_t_norm_{t},gamma_{t}"
                for t in self._lprobe_tags
            )
            f.write(f"mpc_step,admm_iter,k,n_c,{tag_cols}\n")

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
              u_lower: np.ndarray | None = None,
              u_upper: np.ndarray | None = None,
              ee_velocity_bounds: tuple | None = None,
              ee_vel_state_indices: tuple = (16, 17, 18),
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
                u_lower=u_lower, u_upper=u_upper,
                ee_velocity_bounds=ee_velocity_bounds,
                ee_vel_state_indices=ee_vel_state_indices,
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
        # §7.36 — dimension n_lam from the LCS DIRECTLY (E.shape[0]), not
        # from J_n/J_t shapes. Under Stewart-Trinkle E.shape[0] = 6·num_normals
        # (γ + λ_n + λ_t); under Anitescu E.shape[0] = 4·num_normals (single
        # folded λ block, no γ slack). Reading from E honours whatever LCS
        # the formulator produced — no assumption about contact model.
        n_lam       = int(E.shape[0]) if E is not None else (2 * num_normals + n_t)
        TOT         = n_x + n_lam + n_u
        total_dim   = N * TOT + n_x

        # LCS scaling (reference C3::ScaleLCS, c3.cc:203-212 — a BASE-class
        # step, so it applies to every projection variant, Lorentz included).
        # scale = ||A||/||D||; D *= s; E,c,H /= s. Internal λ = λ_phys / s;
        # un-scaled back to Newtons at the end of this method (c3.cc:350-353).
        # Added 2026-08-08 when the duplicate formulator-side scaling was
        # removed, so this path keeps its conditioning.
        _lcs_scale_c3 = 1.0
        if (getattr(self, "_scale_lcs_in_solver", True)
                and E is not None and D is not None
                and np.linalg.norm(D) > 1e-12):
            _lcs_scale_c3 = float(np.linalg.norm(A) / np.linalg.norm(D))
            D     = D     * _lcs_scale_c3
            E     = E     / _lcs_scale_c3
            H     = (H / _lcs_scale_c3) if H is not None else H
            c_lcs = (c_lcs / _lcs_scale_c3) if c_lcs is not None else c_lcs

        # Per-step λ slot offsets (within an n_lam-sized block).
        # ST layout: γ=[0:n_c), λ_n=[n_c:2n_c), λ_t=[2n_c:6n_c).
        # Anitescu layout: single λ block [0:4n_c) — SLN/SLT are not
        # meaningful under Anitescu; their only consumers below are
        # diagnostic prints and the ST-keyed first-knot extraction, both
        # of which are guarded by the (n_lam == 2·num_normals + n_t)
        # check before use.
        SG  = 0
        SLN = num_normals
        SLT = 2 * num_normals
        _is_st_layout = (n_lam == 2 * num_normals + n_t)

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

            # λ ≥ 0 is enforced by the componentwise δ-projection (Lorentz for
            # C3, Bui eq (12) for C3+). Reference c3.cc has no λ bounding-box.
            # Removed 2026-07-26 arc-2 D3 — port-only defensive constraint
            # binds under G-on and OSQP polishing off, zeroing λ.

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

        # 4.f — reference delta_option=1 (c3.cc:311-316; yaml delta_option:
        # 1 for both push_t and anything): bias-initialize delta.head=x0
        # for every knot's state slot. The c3plus path has had this since
        # fc51111; reference C3::Solve applies it regardless of projection
        # variant, so the Lorentz path gets the same init.
        for i in range(N + 1):
            base = i * TOT if i < N else N * TOT
            delta[base : base + n_x] = x0

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
                res = self._solver.Solve(prog, None, self._osqp_solver_options)

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

        # First-horizon contact force for Aydinoglu eq. 36 τ_ff. λ slot
        # layout matches C3+: [γ; λ_n; λ_t] under Stewart-Trinkle.
        # §7.36 — under Anitescu the layout collapses to a single λ block
        # (no γ, no λ_n/λ_t split). Per-component normal/tangent extraction
        # is ST-specific; under Anitescu we stash the raw folded λ block in
        # self._last_lambda_anitescu_first and leave the ST-keyed views as
        # zero placeholders (the executor pipeline is the separate next
        # block — this guard keeps the solver shape-correct for the smoke).
        if num_normals > 0 and _is_st_layout:
            _SLN0 = n_x + SLN
            _SLT0 = n_x + SLT
            self._last_lambda_n_first = z_sol[_SLN0 : _SLN0 + num_normals].copy()
            self._last_lambda_t_first = (z_sol[_SLT0 : _SLT0 + n_t].copy()
                                         if n_t > 0 else np.zeros(0))
            # T-architecture Stage 1: full λ horizon, shaped (N, ·).
            _ln_h = np.zeros((N, num_normals))
            _lt_h = np.zeros((N, n_t)) if n_t > 0 else np.zeros((N, 0))
            for _k in range(N):
                _base = _k * TOT + n_x
                _ln_h[_k] = z_sol[_base + SLN : _base + SLN + num_normals]
                if n_t > 0:
                    _lt_h[_k] = z_sol[_base + SLT : _base + SLT + n_t]
            self._last_lambda_n_horizon = _ln_h
            self._last_lambda_t_horizon = _lt_h
            self._last_lambda_anitescu_first = np.zeros(0)
            self._last_lambda_anitescu_horizon = np.zeros((N, 0))
        elif num_normals > 0:
            # Anitescu single-block layout — n_lam = 4·num_normals.
            _LAN0 = n_x
            self._last_lambda_anitescu_first = z_sol[_LAN0 : _LAN0 + n_lam].copy()
            _la_h = np.zeros((N, n_lam))
            for _k in range(N):
                _base = _k * TOT + n_x
                _la_h[_k] = z_sol[_base : _base + n_lam]
            self._last_lambda_anitescu_horizon = _la_h
            # ST views left as placeholders — downstream executor pipeline
            # under Anitescu is the SEPARATE next block (out of scope here).
            self._last_lambda_n_first = np.zeros(num_normals)
            self._last_lambda_t_first = np.zeros(n_t)
            self._last_lambda_n_horizon = np.zeros((N, num_normals))
            self._last_lambda_t_horizon = np.zeros((N, n_t))
        else:
            self._last_lambda_n_first = np.zeros(0)
            self._last_lambda_t_first = np.zeros(0)
            self._last_lambda_n_horizon = np.zeros((N, 0))
            self._last_lambda_t_horizon = np.zeros((N, 0))
            self._last_lambda_anitescu_first = np.zeros(0)
            self._last_lambda_anitescu_horizon = np.zeros((N, 0))

        # LCS-scaling un-scale (reference c3.cc:350-353 `lambda_sol_ *=
        # AnDn_`): the ADMM solved with λ_internal = λ_phys / scale, so
        # multiply the published views back into Newtons. Unconditional on
        # the scale value — matching the reference, which has no guard.
        if _lcs_scale_c3 != 1.0:
            for _attr_c3 in ("_last_lambda_n_first", "_last_lambda_t_first",
                             "_last_lambda_n_horizon", "_last_lambda_t_horizon",
                             "_last_lambda_anitescu_first",
                             "_last_lambda_anitescu_horizon"):
                _v_c3 = getattr(self, _attr_c3, None)
                if _v_c3 is not None and hasattr(_v_c3, "size") and _v_c3.size:
                    setattr(self, _attr_c3, _v_c3 * _lcs_scale_c3)

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
                  f"|u[0]|={np.linalg.norm(u_seq[0]):.2f}{self.u_unit_str} "
                  f"λ_n_max={lam_n_max:.3f} λ_t_max={lt_max:.3f} "
                  f"lcp_res_max={lcp_max_res:.2e} "
                  f"primal={pr_last:.3f} iters={actual_iters}/{admm_iter}")
        else:
            print(f"[C3] step={self._diag_step} n_λ=0  "
                  f"|u[0]|={np.linalg.norm(u_seq[0]):.3f} {self.u_unit_str}")

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
                  f"|u| ≤ {torque_limit:.1f} {self.u_unit_str} "
                  f"({self.u_unit_kind})")
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
    def _project_C3Plus(lam: np.ndarray,
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
                      u_lower: np.ndarray | None = None,
                      u_upper: np.ndarray | None = None,
                      # 2026-07-18: reference sampling_c3plus_options.yaml:36
                      # `ee_velocity_limits: [-0.14, 0.14]` applied as
                      # AddLinearConstraint(..., STATE) at cc:1027-1034.
                      # When passed, adds per-knot BoundingBoxConstraint on
                      # state slots [ee_vel_state_indices] ∈ [lo, hi]. None →
                      # no constraint (byte-identical to prior behavior).
                      ee_velocity_bounds: tuple | None = None,
                      ee_vel_state_indices: tuple = (16, 17, 18),
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

        # Stage C disambiguation probe — one-shot ADMM-instance dump.
        # Gated by DIAG_ADMM_DUMP=PATH; on the c3-tick whose `_solve_c3plus`
        # call number equals DIAG_ADMM_DUMP_AT (default 50, mid-c3 by the
        # time the planner is solidly in the predicting-retreat regime),
        # write all inputs to a .npz at PATH and continue. Subsequent calls
        # do nothing. The harness `scripts/_stage_c_admm_harness.py` reads
        # this .npz to (i) replay with iter×ρ sweeps, (ii) call the direct
        # LCP/MIQP existence check, (iii) inspect E-matrix structure.
        import os as _os_d
        _dump_path = _os_d.environ.get("DIAG_ADMM_DUMP", "")
        _dump_at   = int(_os_d.environ.get("DIAG_ADMM_DUMP_AT", "50"))
        _dump_min_iter = int(_os_d.environ.get("DIAG_ADMM_DUMP_MIN_ITER", "20"))
        # Filter: skip surrogate sample-eval calls (admm_iter < 20); count
        # only full c3-mode solves. The disambiguation needs the FULL path.
        if (_dump_path
                and not getattr(self, "_admm_dump_done", False)
                and admm_iter >= _dump_min_iter):
            self._admm_dump_call = getattr(self, "_admm_dump_call", 0) + 1
            if self._admm_dump_call == _dump_at:
                np.savez(
                    _dump_path,
                    x0=x0, A=A, B_ctrl=B_ctrl, D=D, d=d,
                    E=E, F=F, H=H, c_lcs=c_lcs,
                    J_n=J_n, J_t=J_t,
                    mu=np.asarray(mu, dtype=float),
                    Q=Q, R=R, QN=QN, x_ref=x_ref,
                    N=np.int32(N), admm_iter=np.int32(admm_iter),
                    torque_limit=np.asarray(torque_limit, dtype=float),
                    phi=(phi if phi is not None else np.zeros(0)),
                    u_lower=(u_lower if u_lower is not None else np.zeros(0)),
                    u_upper=(u_upper if u_upper is not None else np.zeros(0)),
                    rho_initial=np.asarray(rho, dtype=float),
                    n_x=np.int32(n_x), n_u=np.int32(n_u),
                    solver_rho_attr=np.asarray(self.rho, dtype=float),
                    solver_mode=np.array(["c3plus"], dtype=object),
                )
                self._admm_dump_done = True
                print(f"[ADMM-DUMP] wrote c3-tick {self._admm_dump_call} "
                      f"to {_dump_path} "
                      f"(n_x={n_x} n_u={n_u} N={N} n_lambda will be derived)",
                      flush=True)

        num_normals = J_n.shape[0]
        # Phase 2: λ now includes Stewart-Trinkle's friction-cone slack γ
        # so n_lambda = 6·num_normals (= 2·n_c + 4·n_c). The Bui eq. (12)
        # componentwise projection still operates pair-by-pair, so this
        # change is transparent to the projection logic itself.
        # §7.36 — dimension n_lambda from the LCS DIRECTLY (E.shape[0]) so
        # the consensus scaffolding honours whatever LCS the formulator
        # produced — ST: 6·num_normals, Anitescu: 4·num_normals (folded).
        # The Bui eq.(12) projection is element-wise (no slot keying) so
        # the change is transparent to it.
        n_lambda    = int(E.shape[0]) if E is not None else (2 * num_normals + J_t.shape[0])
        _is_st_c3p  = (n_lambda == 2 * num_normals + J_t.shape[0])
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
        # Reference-conformant G matrix (c3_options.h:189, applied every
        # iter). Precompute the per-slot G_diag vector for the current
        # problem shape; cache to avoid rebuilding each solve.
        _shape_key = (total_dim, N, TOT, n_x, n_lambda, n_u, SL, SU, SE)
        if (self._use_g_matrix
                and self._g_diag_c3p_shape != _shape_key):
            _gd = np.zeros(total_dim)
            _wG = float(self._w_G)
            _gx = float(self._g_x) * _wG
            _gL = float(self._g_lambda) * _wG
            _gu = float(self._g_u) * _wG
            _gE = float(self._g_eta) * _wG
            # Per-knot layout: [x, λ, u, η] at offsets [0, SL, SU, SE).
            for _k_kn in range(N):
                _base = _k_kn * TOT
                _gd[_base + 0 : _base + n_x]                    = _gx
                _gd[_base + SL : _base + SL + n_lambda]         = _gL
                _gd[_base + SU : _base + SU + n_u]              = _gu
                _gd[_base + SE : _base + SE + n_lambda]         = _gE
            # Terminal x_N at end of vector.
            _gd[N * TOT : N * TOT + n_x] = _gx
            self._g_diag_c3p_cache = _gd
            self._g_diag_c3p_shape = _shape_key
            if not getattr(self, "_g_matrix_banner", False):
                self._g_matrix_banner = True
                print(f"[G-MATRIX] active: w_G={_wG} g_x={self._g_x} "
                      f"g_λ={self._g_lambda} g_u={self._g_u} g_η={self._g_eta}  "
                      f"→ per-slot: x={_gx} λ={_gL} u={_gu} η={_gE}  "
                      f"(effective at rho=100: x=0 λ=2.0 η=1.0)",
                      flush=True)
        _use_g = self._use_g_matrix and self._g_diag_c3p_cache is not None

        # 2026-07-26 arc-2 D1 fix: under G-on the reference bakes the full
        # augmentation weight into G = w_G · diag(g_vec) with yaml rho ≈ 0
        # (c3/core/configs/*.yaml:16 sets rho=0.0001). Port's outer scalar
        # rho=100 multiplied against G gives ~100× the reference aug at
        # iter 0 and blows up under the rho_scale=3 per-iter ramp. Override
        # rho to 1.0 under G-on (env: REFCONF_ADMM_RHO_UNDER_G) so the port's
        # `rho · G` matches the reference's `G`. G-off path is untouched
        # so the port's tuned rho=100 baseline for the uniform ρ·I aug
        # keeps its calibration.
        if _use_g:
            _rho_under_g = 1.0   # reference w_G·diag(g_vec) scale (D1 fix);
                                 # 2026-07-28 defaults flip removed the
                                 # REFCONF_ADMM_RHO_UNDER_G env override.
            if abs(rho - _rho_under_g) > 1e-9:
                if not getattr(self, "_rho_under_g_banner", False):
                    self._rho_under_g_banner = True
                    print(f"[RHO-UNDER-G] override rho: {rho} → {_rho_under_g} "
                          f"(G-on active; matches reference w_G·diag(g_vec) scale)",
                          flush=True)
                rho = _rho_under_g

        # ---------------------------------------------------------------
        # LCS scaling — reference c3.cc:81, 204-212 + lcs.cc:46-58.
        # ScaleComplementarityDynamics: scale = ||A[0]|| / ||D[0]||;
        # D *= scale, E /= scale, c /= scale, H /= scale. Renormalizes
        # the LCS so ADMM's OSQP sees comparable magnitudes for state
        # propagation (A) and contact impulse (D). Without this, small-D
        # contact rows (T-push EE-BOX: 4cm bar geometry → small D) get
        # numerically zeroed by the ADMM projection.
        # Physical λ recovered at end via lambda_sol *= _lcs_scale
        # (reference c3.cc:349-354).
        _lcs_scale = 1.0
        if n_lambda > 0 and D is not None and np.linalg.norm(D) > 0:
            _lcs_scale = float(np.linalg.norm(A) / np.linalg.norm(D))
            D     = D     * _lcs_scale
            E     = E     / _lcs_scale
            c_lcs = c_lcs / _lcs_scale
            H     = H     / _lcs_scale
            # 2026-08-08: this is now the SOLE scaling site (the duplicate
            # in lcs_formulator was removed — it made this recompute to
            # exactly 1.0 and thereby disabled the un-scale below, leaving
            # every published λ at λ_phys/scale = 5.05× too large).
            if not getattr(self, "_lcs_scale_banner", False):
                self._lcs_scale_banner = True
                print(f"[LCS-SCALE] solver-side (sole site, ref "
                      f"C3::ScaleLCS): scale={_lcs_scale:.4f} — λ published "
                      f"in Newtons via ×{_lcs_scale:.4f} un-scale "
                      f"(c3.cc:350-353)", flush=True)

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

            # 4.j — penalize_input_change: cost becomes ‖u_k − u_prev_k‖²_R
            # = u_k^T R u_k − 2·u_prev_k^T R u_k + const. Add the linear term
            # (−2·R·u_prev_k) to q_ref for each u slot when a previous solve
            # exists. The Hessian block above is unchanged (still 2·R).
            if self._penalize_input_change and self._u_prev_solve is not None:
                _u_prev = self._u_prev_solve      # shape (N, n_u)
                for i in range(N):
                    ui = i * TOT + SU
                    q_ref[ui:ui+n_u] += -2.0 * (R @ _u_prev[i])

            # ===== C3+ NEW: NO soft-complementarity penalty here =====
            # The η = E x + F λ + H u + c equality below replaces the
            # `q_ref[λ_n] += w_comp · phi_gap` hack used by C3.

            # Reference-conformant G-matrix augmentation: replaces uniform
            # rho*I with rho*diag(G) where G has per-slot weights (0 on
            # state/input, w_G·g_λ on λ, w_G·g_η on η). Matches c3.cc AD
            # step exactly. Falls back to uniform ρ if _use_g False.
            # 2026-07-26 factor-of-2 fix (G-on path only): reference
            # c3_plus.cc:157 uses AddQuadraticCost(2G, -2G·WD, ...) →
            # Drake `0.5·H·z²+b·z` convention gives `G·z² - 2G·WD·z`,
            # i.e., ADMM augmentation `(ρ/2)||z - WD||²_G` with effective
            # penalty 2ρ. Previous port passed `rho·g_diag` — half the
            # reference scale — so under G-on the port ran at half the
            # intended penalty. G-off path preserved (port's tuned ρ=100
            # baseline is calibrated against the half-scale convention;
            # doubling would silently regress every G-off run).
            if _use_g:
                _P_aug = 2.0 * rho * np.diag(self._g_diag_c3p_cache)
            else:
                _P_aug = rho * _eye_total
            P_total = P + _P_aug
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

            # λ ≥ 0 is enforced by the Bui eq. (12) componentwise projection
            # on (λ_j, η_j) pairs. Reference c3_plus.cc has no λ bounding-box.
            # Removed 2026-07-26 arc-2 D3 — port-only defensive constraint
            # binds under G-on + rho=1, OSQP polishes off by default so the
            # active bound zeroes λ in z_sol and the projection over-corrects.

            # Torque bounds per step (per-axis vectors when supplied;
            # default-inert scalar torque_limit path when u_lower/u_upper None).
            _u_lo = (np.full(n_u, -torque_limit)
                     if u_lower is None else np.asarray(u_lower, dtype=float))
            _u_hi = (np.full(n_u,  torque_limit)
                     if u_upper is None else np.asarray(u_upper, dtype=float))
            assert _u_lo.shape == (n_u,) and _u_hi.shape == (n_u,), (
                f"u_lower/u_upper must be shape ({n_u},); got "
                f"{_u_lo.shape}/{_u_hi.shape}"
            )
            for i in range(N):
                ui = i * TOT + SU
                prog.AddBoundingBoxConstraint(
                    _u_lo, _u_hi, z_var[ui : ui + n_u],
                )

            # Workspace state constraints — reference cc:995-1025 adds, to
            # EVERY per-sample C3 object, AddLinearConstraint(A·x ∈
            # [lb − workspace_margins, ub + workspace_margins], STATE) rows
            # selecting the EE position AND object position slots (push_t
            # values sampling_c3_options.yaml:26-30). The port had carried
            # over only the adjacent EE-velocity rows (cc:1027-1034, below);
            # without the position rows the planner was free to plan EE
            # excursions anywhere in R^3 (p140/p141: phantom stints walked
            # the EE into the r=0.25 workspace abort). Instance attribute
            # `state_position_bounds` = [(state_idx, lo, hi), ...] so the
            # surrogate per-sample solves (inner_solve → this same solver)
            # inherit the constraint exactly like the reference's per-sample
            # C3 objects. None/empty → byte-identical legacy behavior.
            _spb = getattr(self, "state_position_bounds", None)
            if _spb:
                _spb_idx = np.array([int(b[0]) for b in _spb])
                _spb_lo  = np.array([float(b[1]) for b in _spb])
                _spb_hi  = np.array([float(b[2]) for b in _spb])
                for i in range(N):
                    _base = i * TOT + SX
                    prog.AddBoundingBoxConstraint(
                        _spb_lo, _spb_hi, z_var[_spb_idx + _base])
                prog.AddBoundingBoxConstraint(
                    _spb_lo, _spb_hi, z_var[_spb_idx + N * TOT])

            # State velocity bounds (ee_velocity_limits). Reference cc:1027-1034
            # applies at each knot: A · x_k ∈ [lo, hi] where A selects the EE
            # velocity slots. Port equivalent: BoundingBoxConstraint on the
            # same state indices at each knot plus the terminal state.
            if ee_velocity_bounds is not None:
                _ev_lo, _ev_hi = float(ee_velocity_bounds[0]), float(ee_velocity_bounds[1])
                _ev_idx = list(ee_vel_state_indices)
                _n_ev = len(_ev_idx)
                _ev_lo_vec = np.full(_n_ev, _ev_lo)
                _ev_hi_vec = np.full(_n_ev, _ev_hi)
                for i in range(N):
                    _base = i * TOT + SX
                    prog.AddBoundingBoxConstraint(
                        _ev_lo_vec, _ev_hi_vec,
                        z_var[np.array(_ev_idx) + _base],
                    )
                # Terminal state (position N*TOT + n_x slots)
                _base_terminal = N * TOT
                prog.AddBoundingBoxConstraint(
                    _ev_lo_vec, _ev_hi_vec,
                    z_var[np.array(_ev_idx) + _base_terminal],
                )

            cost_bd = prog.AddQuadraticCost(
                P_sym, np.zeros(total_dim), z_var)

        # ---------------------------------------------------------------
        # ADMM iterations
        # ---------------------------------------------------------------
        # 2026-07-22: paper-notation [CONSENSUS] instrumentation
        # (Bui et al. 2026, arXiv:2510.19974v2, eq (6)+(9)+(10)).
        # DEFAULT ON — one-shot symbol-binding and definition emission
        # at first C3+ solve of the run, plus per-iter [CONSENSUS]
        # blocks below (iters 0/1/last) that decompose r_prim per
        # sub-block [x, lam, u, eta]. ~50 lines total per run — not
        # spam.  Override the target solve with DIAG_CONSENSUS_DUMP_SOLVE_N.
        import os as _os_consensus
        if not getattr(self, "_consensus_bind_printed", False):
            self._consensus_bind_printed = True
            print("[CONSENSUS-BIND] x=z_sol[i*TOT+SX:i*TOT+SL]  "
                  "lam=z_sol[i*TOT+SL:i*TOT+SU]  "
                  "u=z_sol[i*TOT+SU:i*TOT+SE]  "
                  "eta=z_sol[i*TOT+SE:(i+1)*TOT]  "
                  "delta_*=delta[same slots]  w=omega[same slots]",
                  flush=True)
            print("[CONSENSUS-DEF] eq(6): z_k = delta_k, for all k=0..N-1 ; "
                  "z_k = [x, lam, u, eta] , delta_k = [dx, dlam, du, deta]",
                  flush=True)
            print("[CONSENSUS-DEF] eq(9): w_k^{i+1} = w_k^i + "
                  "( z_k^{i+1} - delta_k^{i+1} )",
                  flush=True)
            print(f"[CONSENSUS-DEF] eq(10) penalty term: rho * "
                  f"|| z_k - delta_k^i + w_k^i ||^2_G ,  "
                  f"G(ee-obj lam,eta)={self._w_G_ee_contact} , else=1  "
                  f"[current rho={rho:.1f} N={N} n_x={n_x} n_lambda="
                  f"{n_lambda} n_u={n_u}]",
                  flush=True)
            self._consensus_solve_number = 0
        else:
            self._consensus_solve_number = getattr(
                self, "_consensus_solve_number", 0) + 1
        # Emit per-iter [CONSENSUS] blocks on the FIRST c3+ solve of the
        # run (avoids log spam). Override with DIAG_CONSENSUS_DUMP_SOLVE_N.
        _consensus_target_solve = int(_os_consensus.environ.get(
            "DIAG_CONSENSUS_DUMP_SOLVE_N", "0"))
        _emit_consensus_this_solve = (
            getattr(self, "_consensus_solve_number", 0)
            == _consensus_target_solve)

        delta      = np.zeros(total_dim)
        omega      = np.zeros(total_dim)
        delta_prev = np.zeros(total_dim)

        # Reference delta_option=1 (c3.cc:312-316) — bias-initialize
        # delta.head=x0 for every knot's state slot.
        for i in range(N + 1):
            base = i * TOT if i < N else N * TOT
            delta[base : base + n_x] = x0

        z_sol = np.zeros(total_dim)
        for i in range(N):
            z_sol[i * TOT : i * TOT + n_x] = x0
        z_sol[N * TOT : N * TOT + n_x] = x0

        primal_hist = []
        dual_hist   = []
        rho_hist    = []
        # Per-iter (raw QP cost, augmented penalty, augmented Lagrangian,
        # horizon-max η_n, horizon-max λ_n) — populated when either
        # DIAG_MATH_ITER_LOG or DIAG_ADMM_RESID_CSV would consume them.
        cost_hist   = []
        pen_hist    = []
        lrho_hist   = []
        eta_max_hist = []
        lam_max_hist = []
        tol         = 1e-3
        actual_iters = admm_iter
        u_lam_w = self._u_lambda
        u_eta_w = self._u_eta

        # §7.67 — B1-A: Bui §IV-B.2 final-paragraph mechanism.
        # Wire the `_w_G_ee_contact=1000.0` scaffolded at line 122 (TODO).
        # On the FINAL ADMM iter, override the augmented-cost weight matrix
        # G (eq 10) to `W` (=1000) on the EE-BOX λ_n + 4 λ_t + η_n + 4 η_t
        # slots per knot, forcing z^λ_EE-BOX → δ^λ_EE-BOX on the load-
        # bearing pair. BOX-GND and other pairs keep G=1 → their pr can
        # stay high (per paper). Default-OFF; byte-identical when
        # `PORT_G_WEIGHT_EE_BOX_FINAL` unset. Requires ST layout AND
        # caller (ci_mpc_c3plus) to have set `self._ee_box_pair_idx`.
        import os as _os_g
        _g_ee_box_final_flag = (_os_g.environ.get(
            "PORT_G_WEIGHT_EE_BOX_FINAL", "0") == "1")
        _ee_box_pair_idx = getattr(self, "_ee_box_pair_idx", None)
        _b1a_active = (_g_ee_box_final_flag
                       and _is_st_c3p
                       and n_lambda > 0
                       and num_normals > 0
                       and _ee_box_pair_idx is not None)
        if (_g_ee_box_final_flag and not _b1a_active
                and not getattr(self, "_b1a_skip_banner", False)):
            self._b1a_skip_banner = True
            print(f"[§7.67 B1-A] flag ON but SKIPPED — "
                  f"_is_st_c3p={_is_st_c3p} n_lambda={n_lambda} "
                  f"num_normals={num_normals} "
                  f"ee_box_pair_idx={_ee_box_pair_idx}", flush=True)

        # DIAG_MATH_ITER_LOG — per-ADMM-iter equation trace for teaching /
        # writeup use. When set, emit the Bui 2026 LCS equations (5b, 5c,
        # complementarity) and the Aydinoglu ADMM steps (7, 8, 9) at every
        # iter of every solve. Per-knot values shown at k=0 (first
        # "contact-rich stage") to keep line count manageable. Default OFF.
        _math_iter_log = (_os_g.environ.get(
            "DIAG_MATH_ITER_LOG", "0") == "1")
        # Pull the resid-CSV gate up so the per-iter cost/slack computation
        # (populating cost_hist/eta_max_hist/lam_max_hist) can be skipped
        # entirely when nothing consumes it — keeps the surrogate solves in
        # sample-eval hot loops free of extra O(total_dim²) matvecs.
        _resid_csv_hoisted     = _os_g.environ.get("DIAG_ADMM_RESID_CSV", "")
        _resid_min_iter_hoisted = int(_os_g.environ.get(
            "DIAG_ADMM_RESID_MIN_ITER", "20"))
        _need_cost = bool(
            _math_iter_log
            or (_resid_csv_hoisted and admm_iter >= _resid_min_iter_hoisted)
        )
        # One-shot equation-form header on the first solve after enabling.
        if _math_iter_log and not getattr(self, "_math_iter_header_done", False):
            self._math_iter_header_done = True
            print("[MATH.LCS] Bui 2026 discrete LCS (eq 5b, 5c, complementarity):",
                  flush=True)
            print("[MATH.LCS]   (5b)  x_{k+1} = A·x_k + B·u_k + D·λ_k + d",
                  flush=True)
            print("[MATH.LCS]   (5c)  η_k     = E·x_k + F·λ_k + H·u_k + c",
                  flush=True)
            print("[MATH.LCS]         0 ≤ λ_k ⊥ η_k ≥ 0",
                  flush=True)
            print("[MATH.ADMM] Aydinoglu 2024 ADMM steps (eq 7, 8, 9):",
                  flush=True)
            print("[MATH.ADMM]   (7)  z^{i+1}   = argmin_z Lρ(z, δ^i, ω^i)   — quadratic step (z-update)",
                  flush=True)
            print("[MATH.ADMM]   (8)  δ^{i+1}_k = argmin_δ Lρ(z^{i+1}, δ_k, ω^i_k), ∀k — projection (δ-update)",
                  flush=True)
            print("[MATH.ADMM]   (9)  ω^{i+1}_k = ω^i_k + z^{i+1}_k − δ^{i+1}_k, ∀k — dual (ω-update)",
                  flush=True)

        for it in range(admm_iter):
            delta_prev = delta.copy()
            # Snapshot ω before the ω-update at end-of-iter so the cost
            # log can evaluate `pen = (ρ/2)||z − δ_prev + ω_pre||²` — the
            # augmented penalty at the point OSQP actually minimized.
            omega_pre  = omega.copy() if _need_cost else omega
            rho_iter   = rho  # ρ used to build this iter's QP (pre-scale)

            with timed("admm.qp_build"):
                # Ref-conformant G-matrix: element-wise per-slot weighting.
                # 2026-07-26 factor-of-2 fix (G-on path only): reference
                # gives `-2·G·WD·z` linear term. G-off preserved at
                # `-rho·(δ-ω)` to keep the tuned ρ=100 baseline stable.
                if _use_g:
                    q_total = q_ref - 2.0 * rho * self._g_diag_c3p_cache * (delta - omega)
                else:
                    q_total = q_ref - rho * (delta - omega)

                # §7.67 — final-iter G-weighting override.
                if _b1a_active and it == admm_iter - 1:
                    _W = float(self._w_G_ee_contact)
                    _SLN_c = num_normals              # λ_n offset in λ block
                    _SLT_c = 2 * num_normals          # λ_t offset in λ block
                    _idx   = int(_ee_box_pair_idx)
                    _G_diag = np.ones(total_dim)
                    for _k_kn in range(N):
                        _base = _k_kn * TOT
                        # EE-BOX λ_n (1 scalar per pair)
                        _G_diag[_base + SL + _SLN_c + _idx] = _W
                        # EE-BOX λ_t (4-edge polyhedron per pair)
                        _lt_s = _base + SL + _SLT_c + 4 * _idx
                        _G_diag[_lt_s : _lt_s + 4]           = _W
                        # EE-BOX η_n
                        _G_diag[_base + SE + _SLN_c + _idx] = _W
                        # EE-BOX η_t
                        _et_s = _base + SE + _SLT_c + 4 * _idx
                        _G_diag[_et_s : _et_s + 4]           = _W
                    _P_total_final = P + rho * np.diag(_G_diag)
                    _P_sym_final = (0.5 * (_P_total_final + _P_total_final.T)
                                    + 1e-8 * _eye_total)
                    # Augmented linear term picks up G element-wise. Under
                    # a converged ADMM, ω is near 0 and the pull is toward
                    # δ. In a NON-converged ADMM, ω has drifted large (e.g.
                    # +124 at slot 22 empirically) and δ−ω is far from δ —
                    # even flipped in sign, forcing z to the λ≥0 bound at
                    # 0. Since the paper's mechanism assumes the pull is
                    # toward the previous projection δ (which IS LCS-
                    # feasible), drop the ω term on the final iter for
                    # the EE-BOX slots only. Other slots keep the standard
                    # augmented form.
                    _q_total_final = q_ref - rho * (delta - omega)
                    # For each EE-BOX slot, overwrite: pull = -ρ·G·δ (no ω).
                    for _k_kn in range(N):
                        _base = _k_kn * TOT
                        _sn  = _base + SL + _SLN_c + _idx        # λ_n
                        _q_total_final[_sn] = -rho * _W * delta[_sn]
                        _lt_s = _base + SL + _SLT_c + 4 * _idx   # λ_t
                        _q_total_final[_lt_s : _lt_s + 4] = (
                            -rho * _W * delta[_lt_s : _lt_s + 4])
                        _en  = _base + SE + _SLN_c + _idx        # η_n
                        _q_total_final[_en] = -rho * _W * delta[_en]
                        _et_s = _base + SE + _SLT_c + 4 * _idx   # η_t
                        _q_total_final[_et_s : _et_s + 4] = (
                            -rho * _W * delta[_et_s : _et_s + 4])
                    cost_bd.evaluator().UpdateCoefficients(
                        _P_sym_final, _q_total_final)
                    if not getattr(self, "_b1a_banner", False):
                        self._b1a_banner = True
                        _n_slots = 10 * N   # (λ_n + 4 λ_t + η_n + 4 η_t) · N
                        # One-shot debug: state of the slot 22 (EE-BOX λ_n k=0)
                        # right before the solve on first fire.
                        _dbg_slot = 0 + SL + _SLN_c + _idx
                        _dbg_Psym = float(_P_sym_final[_dbg_slot, _dbg_slot])
                        _dbg_q    = float(_q_total_final[_dbg_slot])
                        _dbg_del  = float(delta[_dbg_slot])
                        _dbg_om   = float(omega[_dbg_slot])
                        _dbg_iso  = -_dbg_q / _dbg_Psym  # isolated optimum
                        print(f"[§7.67 B1-A] PORT_G_WEIGHT_EE_BOX_FINAL=1 "
                              f"FIRST FIRE — final iter it={it}/"
                              f"{admm_iter-1}. EE-BOX idx={_idx} "
                              f"W_G={_W:.1f} slots={_n_slots} "
                              f"(1 λ_n + 4 λ_t + 1 η_n + 4 η_t per knot "
                              f"× N={N})", flush=True)
                        print(f"[§7.67 B1-A DBG] slot={_dbg_slot} "
                              f"(k=0 λ_n EE-BOX) "
                              f"P_sym[slot,slot]={_dbg_Psym:.2e} "
                              f"q[slot]={_dbg_q:.2e} "
                              f"δ[slot]={_dbg_del:.5f} "
                              f"ω[slot]={_dbg_om:.5f} "
                              f"δ-ω={_dbg_del-_dbg_om:.5f} "
                              f"isolated_QP_min=-q/P={_dbg_iso:.5f} "
                              f"(pre-solve prediction)", flush=True)
                else:
                    cost_bd.evaluator().UpdateCoefficients(P_sym, q_total)

            with timed("admm.osqp_solve"):
                res = self._solver.Solve(prog, None, self._osqp_solver_options)

            if res.is_success():
                z_sol = res.GetSolution(z_var)
            else:
                self.qp_failures += 1
                if self.qp_failures == 1 or self.qp_failures % 100 == 0:
                    print(f"[QP-INFEASIBLE] count={self.qp_failures} it={it} "
                          f"status={res.get_solution_result()} — QP step "
                          f"failed; ADMM reuses stale z_sol", flush=True)

            # §7.67 — B1-A: on the FINAL iter with G-weighting active,
            # SKIP the projection + ω-update. Paper §IV-B.2 says
            # "we terminate after the quadratic step" — the large G-pull on
            # EE-BOX components in the QP already forces z^λ_EE-BOX →
            # δ_prev^λ_EE-BOX; running another projection would recompute
            # δ against the shifted z, defeating the consensus. δ stays at
            # its pre-iter value (`delta_prev` == `delta` before this
            # block), so the convergence log below reads
            # |z_sol − δ_prev| = post-QP-pull gap.
            _skip_projection = (_b1a_active and it == admm_iter - 1)
            if not _skip_projection:
                with timed("admm.z_update"):
                    # ===== δ-update (C3+ NEW): x and u pass through =====
                    delta = z_sol + omega

                    # ===== δ-update on (λ, η): Bui 2026 eq (12) =====
                    # Componentwise per-scalar-pair projection. Matches
                    # reference sampling_c3plus_options.yaml
                    # projection_type: 'C3+'. See _project_C3Plus.
                    if n_lambda > 0:
                        # Per-iter case histogram for [CONSENSUS] view.
                        # Split into λ_n (N) and λ_t (T) sub-slots (γ slots
                        # bucketed separately, not rendered).
                        _proj_N = [0, 0, 0]
                        _proj_T = [0, 0, 0]
                        _proj_G = [0, 0, 0]
                        _proj_case1 = 0
                        _proj_case2 = 0
                        _proj_case3 = 0
                        _N_lo_off = num_normals
                        _N_hi_off = 2 * num_normals
                        for i in range(N):
                            li = i * TOT + SL
                            ei = i * TOT + SE
                            lam_blk = z_sol[li:li+n_lambda] + omega[li:li+n_lambda]
                            eta_blk = z_sol[ei:ei+n_lambda] + omega[ei:ei+n_lambda]
                            d_lam, d_eta = self._project_C3Plus(
                                lam_blk, eta_blk, u_lam_w, u_eta_w)
                            _sqrt_ratio = float(np.sqrt(
                                (u_lam_w if np.isscalar(u_lam_w) else float(u_lam_w))
                                / (u_eta_w if np.isscalar(u_eta_w) else float(u_eta_w))))
                            for _j in range(n_lambda):
                                _lo = float(lam_blk[_j])
                                _eo = float(eta_blk[_j])
                                _c1 = (_eo >= 0.0) and (_eo >= _sqrt_ratio * _lo)
                                _c2 = (_lo >= 0.0) and (_eo <  _sqrt_ratio * _lo)
                                if _c1:
                                    _case_idx = 0
                                elif _c2:
                                    _case_idx = 1
                                else:
                                    _case_idx = 2
                                if _j < _N_lo_off:
                                    _proj_G[_case_idx] += 1
                                elif _j < _N_hi_off:
                                    _proj_N[_case_idx] += 1
                                else:
                                    _proj_T[_case_idx] += 1
                            _proj_case1 = _proj_N[0]+_proj_T[0]+_proj_G[0]
                            _proj_case2 = _proj_N[1]+_proj_T[1]+_proj_G[1]
                            _proj_case3 = _proj_N[2]+_proj_T[2]+_proj_G[2]
                            delta[li:li+n_lambda] = d_lam
                            delta[ei:ei+n_lambda] = d_eta
                        # Expose case counts + N/T split for the [CONSENSUS]
                        # emission below.
                        self._last_proj_case_hist = (
                            _proj_case1, _proj_case2, _proj_case3)
                        self._last_proj_case_N = tuple(_proj_N)
                        self._last_proj_case_T = tuple(_proj_T)
                        self._last_proj_n_slots = int(N * n_lambda)

                # Capture omega BEFORE the dual update so [CONSENSUS] can
                # print w_before / delta_w / w_after per eq (9).
                _omega_before_dual = (omega.copy() if _emit_consensus_this_solve
                                       else None)
                omega = omega + z_sol - delta

            # ---- [CONSENSUS] per-iter, per-knot block-decomposed view ------
            # Emits iters 0, 1, and last (actual_iters-1). Substitutes real
            # values into eq (6) agreement and eq (9) dual update, per
            # sub-block [x, lam, u, eta]. Rules from the plan:
            #   - x, u blocks project through identity → gaps and w must be ~0
            #   - G-weight applies to ee-obj lam,eta components; unweighted
            #     gap is what maps to the paper's r_prim.
            #   - r_prim_k here must equal Tier-1 r_prim in the [ADMM-C3+]
            #     line at the same iter (self-check).
            if _emit_consensus_this_solve and (
                    it == 0 or it == 1 or it == admm_iter - 1):
                _tol = float('nan')
                _pr_stack_sq = 0.0
                for k_out in range(N):
                    _base = k_out * TOT
                    _x_z    = z_sol[_base + SX : _base + SL]
                    _x_d    = delta[_base + SX : _base + SL]
                    _lam_z  = z_sol[_base + SL : _base + SU]
                    _lam_d  = delta[_base + SL : _base + SU]
                    _u_z    = z_sol[_base + SU : _base + SE]
                    _u_d    = delta[_base + SU : _base + SE]
                    _eta_z  = z_sol[_base + SE : _base + TOT]
                    _eta_d  = delta[_base + SE : _base + TOT]

                    _gap_x   = float(np.linalg.norm(_x_z   - _x_d))
                    _gap_lam = float(np.linalg.norm(_lam_z - _lam_d))
                    _gap_u   = float(np.linalg.norm(_u_z   - _u_d))
                    _gap_eta = float(np.linalg.norm(_eta_z - _eta_d))
                    _rprim_k_sq = (_gap_x**2 + _gap_lam**2
                                   + _gap_u**2 + _gap_eta**2)
                    _rprim_k = float(np.sqrt(_rprim_k_sq))
                    _pr_stack_sq += _rprim_k_sq

                    # eq (9) substituted: w_before + (z - delta) = w_after
                    _wx_before   = _omega_before_dual[_base + SX : _base + SL]
                    _wlam_before = _omega_before_dual[_base + SL : _base + SU]
                    _wu_before   = _omega_before_dual[_base + SU : _base + SE]
                    _weta_before = _omega_before_dual[_base + SE : _base + TOT]
                    _dw_x    = _x_z   - _x_d
                    _dw_lam  = _lam_z - _lam_d
                    _dw_u    = _u_z   - _u_d
                    _dw_eta  = _eta_z - _eta_d
                    _wx_after   = _wx_before   + _dw_x
                    _wlam_after = _wlam_before + _dw_lam
                    _wu_after   = _wu_before   + _dw_u
                    _weta_after = _weta_before + _dw_eta

                    print(f"[CONSENSUS] i={it} k={k_out}", flush=True)
                    print(f"  # eq(6) agreement, per sub-block:", flush=True)
                    print(f"  gap_x   = || x   - dx   || = {_gap_x:.6e}",
                          flush=True)
                    print(f"  gap_lam = || lam - dlam || = {_gap_lam:.6e}",
                          flush=True)
                    print(f"  gap_u   = || u   - du   || = {_gap_u:.6e}",
                          flush=True)
                    print(f"  gap_eta = || eta - deta || = {_gap_eta:.6e}",
                          flush=True)
                    print(f"  r_prim_k = || z_k - delta_k || = {_rprim_k:.6e} "
                          f"# stacked; sums to r_prim in Tier-1 [ADMM-C3+]",
                          flush=True)
                    print(f"  # eq(9) dual update, substituted:", flush=True)
                    print(f"  w_before = [wx={float(np.linalg.norm(_wx_before)):.3e} "
                          f"wlam={float(np.linalg.norm(_wlam_before)):.3e} "
                          f"wu={float(np.linalg.norm(_wu_before)):.3e} "
                          f"weta={float(np.linalg.norm(_weta_before)):.3e}]",
                          flush=True)
                    print(f"  delta_w  = ( z_k - delta_k ) = "
                          f"[{float(np.linalg.norm(_dw_x)):.3e} "
                          f"{float(np.linalg.norm(_dw_lam)):.3e} "
                          f"{float(np.linalg.norm(_dw_u)):.3e} "
                          f"{float(np.linalg.norm(_dw_eta)):.3e}]",
                          flush=True)
                    print(f"  w_after  = w_before + delta_w = "
                          f"[{float(np.linalg.norm(_wx_after)):.3e} "
                          f"{float(np.linalg.norm(_wlam_after)):.3e} "
                          f"{float(np.linalg.norm(_wu_after)):.3e} "
                          f"{float(np.linalg.norm(_weta_after)):.3e}]",
                          flush=True)
                # Self-check: sqrt of Σ r_prim_k² should equal r_prim in the
                # Tier-1 [ADMM-C3+] line at this iter. Plus r_dual (paper's
                # dual residual = rho·||delta - delta_prev||) and the
                # per-case histogram from Bui eq (12).
                _r_prim_stacked = float(np.sqrt(_pr_stack_sq))
                # r_dual over lam+eta slots (matches Tier-1 dual exactly;
                # x and u δs also change across iters but are not part of
                # the paper's consensus dual residual).
                _lam_eta_slots = []
                for _k in range(N):
                    _base = _k * TOT
                    _lam_eta_slots.append(delta[_base + SL:_base + SU])
                    _lam_eta_slots.append(delta[_base + SE:_base + TOT])
                _delta_le = np.concatenate(_lam_eta_slots)
                _lam_eta_slots_prev = []
                for _k in range(N):
                    _base = _k * TOT
                    _lam_eta_slots_prev.append(
                        delta_prev[_base + SL:_base + SU])
                    _lam_eta_slots_prev.append(
                        delta_prev[_base + SE:_base + TOT])
                _delta_le_prev = np.concatenate(_lam_eta_slots_prev)
                _r_dual_stacked = float(
                    rho * np.linalg.norm(_delta_le - _delta_le_prev))
                _case_hist = getattr(self, "_last_proj_case_hist", (0, 0, 0))
                _n_slots = getattr(self, "_last_proj_n_slots", 0)
                print(f"[CONSENSUS] i={it} mode=c3plus proj=C3+ SUM:  "
                      f"r_prim_stacked = sqrt(Σ r_prim_k²) = "
                      f"{_r_prim_stacked:.6e}  "
                      f"r_dual = rho·||δ-δ_prev|| = {_r_dual_stacked:.6e}  "
                      f"proj_case_hist(1,2,3)=({_case_hist[0]},"
                      f"{_case_hist[1]},{_case_hist[2]})/{_n_slots}  "
                      f"# r_prim must equal Tier-1 [ADMM-C3+] primal for "
                      f"this iter",
                      flush=True)

            # ---- Per-iter cost + η-slack stats -----------------------------
            # Populated when either DIAG_MATH_ITER_LOG or DIAG_ADMM_RESID_CSV
            # would consume them. `f(z)` is the raw planning cost at the new
            # z; `pen` is the ADMM augmented penalty evaluated at OSQP's
            # minimizer (δ_prev and ω_pre are what q_total was built from);
            # `L_ρ = f + pen` is the full augmented Lagrangian, the object
            # ADMM is descending. η stats read the just-projected delta.
            _have_zsol = ('z_sol' in dir())
            if _need_cost and _have_zsol:
                _f_raw   = (0.5 * float(z_sol @ P @ z_sol)
                            + float(q_ref @ z_sol))
                _z_shift = z_sol - delta_prev + omega_pre
                _pen     = 0.5 * rho_iter * float(_z_shift @ _z_shift)
                _L_rho   = _f_raw + _pen
                cost_hist.append(_f_raw)
                pen_hist.append(_pen)
                lrho_hist.append(_L_rho)
                if n_lambda > 0 and num_normals > 0:
                    _n_t_iter = n_lambda - 2 * num_normals   # 4·n_c under ST
                    _en_max_h = 0.0
                    _et_max_h = 0.0
                    _en_min_h = float('inf')
                    _ln_max_h = 0.0
                    _active_h = 0    # knots with any λ_n > 1e-6 (engaged)
                    for _k_kn in range(N):
                        _base_e = _k_kn * TOT + SE
                        _base_l = _k_kn * TOT + SL
                        _en_k = delta[_base_e + num_normals
                                      : _base_e + 2 * num_normals]
                        _et_k = delta[_base_e + 2 * num_normals
                                      : _base_e + 2 * num_normals + _n_t_iter]
                        _ln_k = delta[_base_l + num_normals
                                      : _base_l + 2 * num_normals]
                        _en_max_h = max(_en_max_h, float(_en_k.max()))
                        _et_max_h = max(_et_max_h, float(_et_k.max()))
                        _en_min_h = min(_en_min_h, float(_en_k.min()))
                        _ln_max_k = float(_ln_k.max())
                        _ln_max_h = max(_ln_max_h, _ln_max_k)
                        if _ln_max_k > 1e-6:
                            _active_h += 1
                    if not np.isfinite(_en_min_h):
                        _en_min_h = 0.0
                else:
                    _en_max_h = _et_max_h = _en_min_h = _ln_max_h = 0.0
                    _active_h = 0
                eta_max_hist.append(_en_max_h)
                lam_max_hist.append(_ln_max_h)
                if _math_iter_log:
                    _dcost = (cost_hist[-1] - cost_hist[-2]
                              if len(cost_hist) >= 2 else 0.0)
                    print(f"[MATH.COST] step={self._diag_step} it={it}: "
                          f"f(z)={_f_raw:+.4e} "
                          f"pen=(ρ/2)‖z−δ+ω‖²={_pen:.4e} "
                          f"L_ρ={_L_rho:+.4e} "
                          f"Δf={_dcost:+.3e} ρ={rho_iter:.2f}",
                          flush=True)
                    print(f"[MATH.η] step={self._diag_step} it={it} "
                          f"horizon (post-δ-update): "
                          f"η_n_max={_en_max_h:.3e} "
                          f"η_t_max={_et_max_h:.3e} "
                          f"η_min={_en_min_h:+.3e} "
                          f"λ_n_max={_ln_max_h:.3e} "
                          f"engaged_knots(λ_n>1e-6)={_active_h}/{N}",
                          flush=True)
            # ----------------------------------------------------------------

            # DIAG_MATH_ITER_LOG — emit Bui LCS (5b/5c/comp) + Aydinoglu
            # ADMM (7/8/9) numeric values at k=0 for this iter. Gated to
            # keep prod runs quiet. Header printed once outside the loop.
            if _math_iter_log and 'z_sol' in dir():
                _x_k    = z_sol[0:n_x]
                _lam_k  = z_sol[n_x:n_x + n_lambda]
                _u_k    = z_sol[n_x + n_lambda:n_x + n_lambda + n_u]
                _eta_k  = z_sol[n_x + n_lambda + n_u
                                :n_x + n_lambda + n_u + n_lambda]
                _x_next = z_sol[TOT:TOT + n_x] if N > 1 else _x_k
                # LCS (5b) breakdown: ||A·x||, ||B·u||, ||D·λ||, ||d||, ||x_{k+1}||
                _Ax = float(np.linalg.norm(A @ _x_k))
                _Bu = float(np.linalg.norm(B_ctrl @ _u_k))
                _Dl = float(np.linalg.norm(D @ _lam_k)) if n_lambda > 0 else 0.0
                _nd = float(np.linalg.norm(d))
                _xn = float(np.linalg.norm(_x_next))
                # LCS (5c) breakdown: ||E·x||, ||F·λ||, ||H·u||, ||c||, ||η||
                if n_lambda > 0:
                    _Ex = float(np.linalg.norm(E @ _x_k))
                    _Fl = float(np.linalg.norm(F @ _lam_k))
                    _Hu = float(np.linalg.norm(H @ _u_k))
                    _nc = float(np.linalg.norm(c_lcs))
                    _ne = float(np.linalg.norm(_eta_k))
                    _comp = float(np.max(np.abs(_lam_k * _eta_k)))
                    _min_l = float(np.min(_lam_k))
                    _min_e = float(np.min(_eta_k))
                else:
                    _Ex = _Fl = _Hu = _nc = _ne = _comp = 0.0
                    _min_l = _min_e = 0.0
                # ADMM (7/8/9) norms
                _z_norm = float(np.linalg.norm(z_sol))
                _d_norm = float(np.linalg.norm(delta))
                _w_norm = float(np.linalg.norm(omega))
                _dw     = float(np.linalg.norm(z_sol - delta))
                print(f"[MATH.LCS-5b] step={self._diag_step} it={it} k=0: "
                      f"||A·x||={_Ax:.3e} ||B·u||={_Bu:.3e} "
                      f"||D·λ||={_Dl:.3e} ||d||={_nd:.3e} "
                      f"→ ||x_{{k+1}}||={_xn:.3e}",
                      flush=True)
                print(f"[MATH.LCS-5c] step={self._diag_step} it={it} k=0: "
                      f"||E·x||={_Ex:.3e} ||F·λ||={_Fl:.3e} "
                      f"||H·u||={_Hu:.3e} ||c||={_nc:.3e} "
                      f"→ ||η_k||={_ne:.3e}",
                      flush=True)
                print(f"[MATH.LCS-COMP] step={self._diag_step} it={it} k=0: "
                      f"max|λ·η|={_comp:.3e} min(λ)={_min_l:+.3e} "
                      f"min(η)={_min_e:+.3e}",
                      flush=True)
                print(f"[MATH.ADMM-7] step={self._diag_step} it={it}: "
                      f"z-update (quadratic) ||z||={_z_norm:.3e}",
                      flush=True)
                print(f"[MATH.ADMM-8] step={self._diag_step} it={it}: "
                      f"δ-update (projection) ||δ||={_d_norm:.3e} "
                      f"||z−δ||={_dw:.3e}",
                      flush=True)
                print(f"[MATH.ADMM-9] step={self._diag_step} it={it}: "
                      f"ω-update (dual) ||ω||={_w_norm:.3e} "
                      f"Δω=||z−δ||={_dw:.3e}",
                      flush=True)

            # §7.67 — B1-A convergence probe: on the final iter, log
            # |z^λ_EE-BOX − δ^λ_EE-BOX| per component (λ_n + 4 λ_t) at k=0
            # and horizon-max. One-shot proof on first fire, per-solve
            # summary thereafter.
            if _b1a_active and it == admm_iter - 1:
                _idx = int(_ee_box_pair_idx)
                _SLN_c = num_normals
                _SLT_c = 2 * num_normals
                # First-knot (k=0)
                _zn0 = float(z_sol[SL + _SLN_c + _idx])
                _dn0 = float(delta[SL + _SLN_c + _idx])
                _lt_zs_0 = z_sol[SL + _SLT_c + 4*_idx : SL + _SLT_c + 4*(_idx+1)]
                _lt_dl_0 = delta[SL + _SLT_c + 4*_idx : SL + _SLT_c + 4*(_idx+1)]
                _gap_n0 = abs(_zn0 - _dn0)
                _gap_t0 = float(np.max(np.abs(_lt_zs_0 - _lt_dl_0)))
                # Horizon-max diff
                _gap_n_hmax = _gap_n0
                _gap_t_hmax = _gap_t0
                for _k_kn in range(1, N):
                    _base = _k_kn * TOT
                    _zn_k = float(z_sol[_base + SL + _SLN_c + _idx])
                    _dn_k = float(delta[_base + SL + _SLN_c + _idx])
                    _gap_n_hmax = max(_gap_n_hmax, abs(_zn_k - _dn_k))
                    _lt_zs_k = z_sol[_base + SL + _SLT_c + 4*_idx :
                                     _base + SL + _SLT_c + 4*(_idx+1)]
                    _lt_dl_k = delta[_base + SL + _SLT_c + 4*_idx :
                                     _base + SL + _SLT_c + 4*(_idx+1)]
                    _gap_t_hmax = max(_gap_t_hmax,
                                      float(np.max(np.abs(_lt_zs_k - _lt_dl_k))))
                # Check-1 verdict: EE-BOX pr below tol?
                _b1a_conv = (_gap_n_hmax < tol and _gap_t_hmax < tol)
                if not getattr(self, "_b1a_conv_dump_done", False):
                    self._b1a_conv_dump_done = True
                    print(f"[§7.67 B1-A CONV] first-solve final-iter "
                          f"convergence proof: "
                          f"λ_n_k0: zsol={_zn0:+.5f} δ={_dn0:+.5f} "
                          f"|Δ|={_gap_n0:.2e}  "
                          f"λ_t_k0 max|Δ|={_gap_t0:.2e}  "
                          f"λ_n horizon-max|Δ|={_gap_n_hmax:.2e}  "
                          f"λ_t horizon-max|Δ|={_gap_t_hmax:.2e}  "
                          f"tol={tol:.0e}  EE-BOX-converged={_b1a_conv}",
                          flush=True)
                self._b1a_last_gap_n_k0    = _gap_n0
                self._b1a_last_gap_t_k0    = _gap_t0
                self._b1a_last_gap_n_hmax  = _gap_n_hmax
                self._b1a_last_gap_t_hmax  = _gap_t_hmax
                self._b1a_last_ee_conv     = _b1a_conv

            # ===== λ-horizon probe (writes per-iter, per-k) =====
            if (self._lprobe_path is not None
                    and n_lambda > 0
                    and self._lprobe_n_solves < self._lprobe_max_solves):
                n_c_probe = num_normals
                with open(self._lprobe_path, "a") as _pf:
                    for k_probe in range(N):
                        z_lam = z_sol[k_probe*TOT + SL : k_probe*TOT + SL + n_lambda]
                        cols = []
                        for ci in range(n_c_probe):
                            lam_n_ci = float(z_lam[n_c_probe + ci])
                            lam_t_ci = z_lam[2*n_c_probe + 4*ci : 2*n_c_probe + 4*(ci+1)]
                            lam_t_norm = float(np.linalg.norm(lam_t_ci))
                            gamma_ci = float(z_lam[ci])
                            cols.extend([f"{lam_n_ci:.6f}",
                                         f"{lam_t_norm:.6f}",
                                         f"{gamma_ci:.6f}"])
                        # Pad if probe was configured for more contacts than this
                        # solve has (n_c can vary by step).
                        while len(cols) < 3 * len(self._lprobe_tags):
                            cols.append("nan")
                        _pf.write(f"{self._lprobe_mpc_step},{it},{k_probe},"
                                  f"{n_c_probe},{','.join(cols)}\n")

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
                rho_hist.append(float(rho))

                # 4.c — reference rho_scale (c3.cc:389-390): multiply ρ each
                # iter and shrink ω to keep the dual gradient ρ·ω consistent.
                # Update P_sym in place by scaling the ρ·I diagonal delta
                # (adds ~n operations instead of ~n² for a full rebuild).
                # 2026-07-26 factor-of-2 fix (G-on only): aug is
                # `2·rho·g_diag` under G-on, so per-iter delta is
                # `2·(_delta_rho)·g_diag`. G-off preserves `_delta_rho`.
                _rs = float(self._rho_scale)
                if _rs > 1.0 and admm_iter > 0 and rho * _rs < 1e6:
                    _delta_rho = rho * (_rs - 1.0)
                    rho   *= _rs
                    omega /= _rs
                    if _use_g:
                        # Ref-conformant: scale by per-slot G_diag.
                        _delta_diag = 2.0 * _delta_rho * self._g_diag_c3p_cache
                        np.fill_diagonal(P_sym, P_sym.diagonal() + _delta_diag)
                    else:
                        np.fill_diagonal(P_sym, P_sym.diagonal() + _delta_rho)
                    cost_bd.evaluator().UpdateCoefficients(P_sym, q_total)
                elif (it + 1) % 10 == 0:
                    # Legacy Boyd §3.4.1 primal/dual balance step (rho_scale
                    # disabled → fall back to this).
                    if pr > 10.0 * dr and rho < 1000.0:
                        rho   *= 2.0
                        omega /= 2.0
                        _aug = (2.0 * rho * np.diag(self._g_diag_c3p_cache)
                                if _use_g else rho * _eye_total)
                        P_total2 = P + _aug
                        P_sym    = 0.5 * (P_total2 + P_total2.T) + 1e-8 * _eye_total
                        cost_bd.evaluator().UpdateCoefficients(P_sym, q_total)
                    elif dr > 10.0 * pr and rho > 0.1:
                        rho   /= 2.0
                        omega *= 2.0
                        _aug = (2.0 * rho * np.diag(self._g_diag_c3p_cache)
                                if _use_g else rho * _eye_total)
                        P_total2 = P + _aug
                        P_sym    = 0.5 * (P_total2 + P_total2.T) + 1e-8 * _eye_total
                        cost_bd.evaluator().UpdateCoefficients(P_sym, q_total)

                if pr < tol and dr < tol:
                    actual_iters = it + 1
                    break

        if n_lambda > 0 and primal_hist:
            mono = all(primal_hist[i] >= primal_hist[i+1]
                       for i in range(len(primal_hist)-1))
            # Optional cost + slack tail (only populated when _need_cost was on).
            _cost_tail = ""
            if cost_hist:
                _cost_tail = (f"  f: {cost_hist[0]:+.3e}->{cost_hist[-1]:+.3e}"
                              f"  L_ρ: {lrho_hist[0]:+.3e}->{lrho_hist[-1]:+.3e}"
                              f"  η_max_h={max(eta_max_hist):.2e}"
                              f"  λ_n_max_h={max(lam_max_hist):.2e}")
            print(f"[ADMM-C3+] step={self._diag_step} "
                  f"primal: {primal_hist[0]:.4f}->{primal_hist[-1]:.4f}  "
                  f"dual: {dual_hist[0]:.4f}->{dual_hist[-1]:.4f}  "
                  f"mono={mono}  iters={actual_iters}/{admm_iter}  "
                  f"rho_start={rho_hist[0]:.1f} rho_end={rho:.1f}"
                  f"{_cost_tail}")

            # ---- Per-step [CONSENSUS-STEP] record (panel-consumable) ----
            # One compact line per solve carrying every field the paper-
            # notation dashboard needs to render the CONSENSUS region.
            # Values are the LAST-iter values (what the ADMM actually
            # settled on).
            _gap_x_sq = 0.0
            _gap_lam_sq = 0.0
            _gap_u_sq = 0.0
            _gap_eta_sq = 0.0
            for _k_out in range(N):
                _base = _k_out * TOT
                _gap_x_sq   += float(np.sum(
                    (z_sol[_base + SX:_base + SL]
                     - delta[_base + SX:_base + SL])**2))
                _gap_lam_sq += float(np.sum(
                    (z_sol[_base + SL:_base + SU]
                     - delta[_base + SL:_base + SU])**2))
                _gap_u_sq   += float(np.sum(
                    (z_sol[_base + SU:_base + SE]
                     - delta[_base + SU:_base + SE])**2))
                _gap_eta_sq += float(np.sum(
                    (z_sol[_base + SE:_base + TOT]
                     - delta[_base + SE:_base + TOT])**2))
            _gap_x   = float(np.sqrt(_gap_x_sq))
            _gap_lam = float(np.sqrt(_gap_lam_sq))
            _gap_u   = float(np.sqrt(_gap_u_sq))
            _gap_eta = float(np.sqrt(_gap_eta_sq))
            _case_N = getattr(self, "_last_proj_case_N", (0, 0, 0))
            _case_T = getattr(self, "_last_proj_case_T", (0, 0, 0))
            print(f"[CONSENSUS-STEP] step={self._diag_step} "
                  f"mode=c3plus proj=C3+ "
                  f"rho_start={rho_hist[0]:.1f} rho_end={rho:.1f} "
                  f"iters={actual_iters}/{admm_iter} "
                  f"primal={primal_hist[0]:.4e}->{primal_hist[-1]:.4e} "
                  f"dual={dual_hist[0]:.4e}->{dual_hist[-1]:.4e} "
                  f"mono={mono} "
                  f"gap=[x={_gap_x:.2e} lam={_gap_lam:.2e} "
                  f"u={_gap_u:.2e} eta={_gap_eta:.2e}] "
                  f"proj_case_N=[{_case_N[0]},{_case_N[1]},{_case_N[2]}] "
                  f"proj_case_T=[{_case_T[0]},{_case_T[1]},{_case_T[2]}]",
                  flush=True)

        # §7.37 measurement scaffold (default-OFF). When
        # DIAG_ADMM_RESID_CSV=PATH is set, append per-iter (pr, dr, rho,
        # f_raw, pen, L_rho, eta_max_h, lam_n_max_h) for each rich-mode
        # solve (admm_iter >= DIAG_ADMM_RESID_MIN_ITER, default 20) to a
        # CSV. Surrogate sample-eval solves are skipped. No behaviour
        # change when the env var is unset — env is re-read here for
        # backwards compat with any external monitor that inspects
        # `_resid_csv` / `_resid_min_iter` names.
        import os as _os_r
        _resid_csv = _os_r.environ.get("DIAG_ADMM_RESID_CSV", "")
        _resid_min_iter = int(_os_r.environ.get("DIAG_ADMM_RESID_MIN_ITER", "20"))
        if _resid_csv and n_lambda > 0 and primal_hist and admm_iter >= _resid_min_iter:
            if not getattr(self, "_resid_csv_inited", False):
                self._resid_csv_inited = True
                self._resid_csv_solve_idx = 0
                _need_header = not _os_r.path.exists(_resid_csv)
                if _need_header:
                    with open(_resid_csv, "w") as _f:
                        _f.write("solve_idx,n_lambda,iter,pr,dr,rho,"
                                 "f_raw,pen,L_rho,eta_max_h,lam_n_max_h,"
                                 "converged,iters_used,admm_iter\n")
            self._resid_csv_solve_idx += 1
            _solve_idx = self._resid_csv_solve_idx
            _conv_flag = int(actual_iters < admm_iter)
            # cost_hist/eta_max_hist/lam_max_hist should always be populated
            # when this branch fires (_need_cost was true given admm_iter
            # ≥ _resid_min_iter_hoisted matches _resid_min_iter here). Fall
            # back to NaN if a caller ever races the env var mid-solve.
            def _hget(_h, _i):
                return _h[_i] if _i < len(_h) else float('nan')
            with open(_resid_csv, "a") as _f:
                for _i in range(len(primal_hist)):
                    _f.write(f"{_solve_idx},{n_lambda},{_i+1},"
                             f"{primal_hist[_i]:.6e},{dual_hist[_i]:.6e},"
                             f"{rho_hist[_i]:.6f},"
                             f"{_hget(cost_hist,_i):.6e},"
                             f"{_hget(pen_hist,_i):.6e},"
                             f"{_hget(lrho_hist,_i):.6e},"
                             f"{_hget(eta_max_hist,_i):.6e},"
                             f"{_hget(lam_max_hist,_i):.6e},"
                             f"{_conv_flag},"
                             f"{actual_iters},{admm_iter}\n")

        # D2: surface ADMM convergence + non-converged warning. Consumed
        # by C3PlusMPC.last_converged → wrapper._derive_force_command,
        # which caps the OSC λ_des magnitude at nominal_push_force when
        # the planner did not converge (avoids amplifying ω-leakage on
        # the delta view, or complementarity-leakage on the z_sol view).
        _pr_final = primal_hist[-1] if primal_hist else 0.0
        _dr_final = dual_hist[-1]   if dual_hist   else 0.0
        self._last_converged = bool(
            n_lambda == 0 or (_pr_final < tol and _dr_final < tol)
        )
        # Stage C probe B [CONSISTENCY] — expose the ADMM terminal state on
        # self so the dispatcher's per-tick trace can read pr/dr/iters/tol
        # without re-running the solve or parsing the [ADMM-C3+] log line.
        self._last_pr_final  = float(_pr_final)
        self._last_dr_final  = float(_dr_final)
        self._last_iters_used = int(actual_iters)
        self._last_tol       = float(tol)
        if not self._last_converged:
            print(f"[ADMM-C3+] WARNING non-converged: "
                  f"pr={_pr_final:.4f} dr={_dr_final:.4f} tol={tol:.0e}")

        # ---------------------------------------------------------------
        # Extract outputs
        # ---------------------------------------------------------------
        # 4.t — reference FINAL QP (c3.cc:332). After the ADMM loop the
        # reference solves ONE more QP at the ramped G: ADMMStep scales
        # G·rho_scale at the end of EVERY iter (c3.cc:389-390), so this
        # solve runs at rho_scale^admm_iter (27× for the push_t
        # 3-iter/rho_scale=3 regime). C3Plus overrides the final-solve
        # augmentation (c3_plus.cc:117-172): pull toward δ ALONE (ω
        # dropped) + optional per-task EE-contact GScaling — see the
        # block inside `timed("admm.final_qp")`.
        # StoreQPResults(..., is_final_solve=true) publishes THIS solve as
        # x_sol_/λ_sol_/u_sol_ — the copies UpdateC3ExecutionTrajectory
        # tracks with the OSC (sampling_based_c3_controller.cc:1703-1704).
        # The port loop ends on a projection, so pre-fix its published QP
        # copy was one consensus solve behind the reference's.
        # REFCONF_FINAL_QP_STEP=0 restores the whole pre-fix tail
        # (including the recursive 4.g re-roll below). Mutually exclusive
        # with B1-A, which terminates on its own boosted quadratic step.
        _final_qp_on = ((_os_g.environ.get("REFCONF_FINAL_QP_STEP", "1")
                         == "1") and not _b1a_active)
        if _final_qp_on:
            if not getattr(self, "_final_qp_banner", False):
                self._final_qp_banner = True
                print(f"[FINAL-QP] active (c3.cc:332 conformance): "
                      f"post-loop QP at rho={rho:.1f} "
                      f"(= rho_start·rho_scale^{admm_iter}); published "
                      f"x_seq/u_seq switch to the reference z_sol_ copy "
                      f"(c3.cc:336-347 half-step + CalcCost x_N append)",
                      flush=True)
            with timed("admm.final_qp"):
                # C3Plus::AddAugmentedCost final-solve semantics
                # (c3_plus.cc:117-172, paper 2510.19974 §IV-B.2 final ¶):
                #   1. WD_i = δ — the final pull targets the PROJECTED copy
                #      alone; ω is dropped (in-loop iterations keep δ−ω).
                #   2. If the task config sets
                #      final_augmented_cost_contact_scaling (reference
                #      anything yaml: 1000; push_t: absent → None), scale
                #      the EE-object pair's chosen complementarity slot per
                #      component: the λ slot when δ_λ == 0 (pin no-force),
                #      else the η slot (pin the gap slack) — hard-enforcing
                #      the projection's branch on the load-bearing pair.
                #      Exact ==0 test matches the reference (projection
                #      writes exact zeros).
                _fs = getattr(self, "_final_aug_contact_scaling", None)
                _fp_idx = getattr(self, "_ee_box_pair_idx", None)
                _boost_slots = []
                if _fs is not None and n_lambda > 0 and _fp_idx is not None:
                    _i_p = int(_fp_idx)
                    if _is_st_c3p and num_normals > 0:
                        # ST layout: pair's λ_n + 4 λ_t components (γ slots
                        # are an ST artifact absent in the reference — not
                        # boosted, matching B1-A precedent).
                        _lam_comps = ([num_normals + _i_p]
                                      + list(range(2 * num_normals + 4 * _i_p,
                                                   2 * num_normals + 4 * _i_p + 4)))
                    elif num_normals > 0:
                        # Anitescu layout: pair's 4-component block.
                        _lam_comps = list(range(4 * _i_p, 4 * _i_p + 4))
                    else:
                        _lam_comps = []
                    for _c_f in _lam_comps:
                        for _k_f in range(N):
                            _b_f = _k_f * TOT
                            if delta[_b_f + SL + _c_f] == 0.0:
                                _boost_slots.append(_b_f + SL + _c_f)
                            else:
                                _boost_slots.append(_b_f + SE + _c_f)
                    if _boost_slots and not getattr(
                            self, "_final_boost_banner", False):
                        self._final_boost_banner = True
                        print(f"[FINAL-QP-BOOST] contact scaling "
                              f"{_fs:.0f}× active (c3_plus.cc:131-145): "
                              f"pair_idx={_i_p} slots/solve="
                              f"{len(_boost_slots)} (δ-conditional λ/η)",
                              flush=True)
                _aug_vec = (2.0 * rho * self._g_diag_c3p_cache
                            if _use_g else np.full(total_dim, rho))
                if _boost_slots:
                    _gsc = np.ones(total_dim)
                    _gsc[_boost_slots] = float(_fs)
                    _P_fin = P_sym.copy()
                    np.fill_diagonal(
                        _P_fin, _P_fin.diagonal() + (_gsc - 1.0) * _aug_vec)
                    q_total = q_ref - _gsc * _aug_vec * delta
                else:
                    _P_fin = P_sym
                    q_total = q_ref - _aug_vec * delta
                cost_bd.evaluator().UpdateCoefficients(_P_fin, q_total)
                res = self._solver.Solve(prog, None,
                                         self._osqp_solver_options)
            if res.is_success():
                z_sol = res.GetSolution(z_var)
            else:
                # Reference SetFallbackSolution: hold current state, zero
                # inputs and forces.
                self.qp_failures += 1
                print(f"[FINAL-QP] INFEASIBLE "
                      f"status={res.get_solution_result()} — reference "
                      f"fallback x=x0, u=0, λ=0", flush=True)
                z_sol = np.zeros_like(z_sol)
                for i in range(N):
                    z_sol[i * TOT : i * TOT + n_x] = x0
                z_sol[N * TOT : N * TOT + n_x] = x0

        # ---------------------------------------------------------------
        u_seq = np.zeros((N, n_u))
        x_seq = np.zeros((N + 1, n_x))
        for i in range(N):
            x_seq[i] = z_sol[i * TOT : i * TOT + n_x]
            u_seq[i] = z_sol[i * TOT + SU : i * TOT + SU + n_u]
        x_seq[N] = z_sol[N * TOT : N * TOT + n_x]

        # QP-copy stash (reference x_sol_/u_sol_, final-QP under 4.h) for
        # consumers that track the QP trajectory — the reference OSC
        # target comes from these, NOT from the published z copy.
        self._last_x_qp_horizon = x_seq.copy()
        self._last_u_qp_horizon = u_seq.copy()

        # 4.g — end_on_qp_step=False (reference default): publish the
        # reference z_sol_ copy. Under 4.t this is c3.cc:336-347 +
        # CalcCost:501-524 verbatim: u and λ slots from δ; x slots the
        # HALF-STEP A·x_qp[i−1] + B·u_qp[i−1] + D·λ_qp[i−1] + d from the
        # final-QP copies (NOT a recursive rollout); x_N appended
        # CalcCost-style from δ (u, λ). With 4.t off, the pre-fix
        # recursive re-roll from the last in-loop z is preserved for A/B.
        if _final_qp_on and not self._end_on_qp_step and n_lambda > 0:
            _x_z = np.zeros((N + 1, n_x))
            _x_z[0] = x0
            for i in range(1, N):
                _p = (i - 1) * TOT
                _x_z[i] = (A @ z_sol[_p : _p + n_x]
                           + B_ctrl @ z_sol[_p + SU : _p + SU + n_u]
                           + D @ z_sol[_p + SL : _p + SL + n_lambda]
                           + d)
            _u_z = np.zeros((N, n_u))
            for i in range(N):
                _u_z[i] = delta[i * TOT + SU : i * TOT + SU + n_u]
            _lam_dN = delta[(N - 1) * TOT + SL
                            : (N - 1) * TOT + SL + n_lambda]
            _x_z[N] = (A @ _x_z[N - 1] + B_ctrl @ _u_z[N - 1]
                       + D @ _lam_dN + d)
            x_seq = _x_z
            u_seq = _u_z
        elif not self._end_on_qp_step and n_lambda > 0:
            _x_roll = np.zeros((N + 1, n_x))
            _x_roll[0] = x0
            for i in range(N):
                lam_i = z_sol[i * TOT + SL : i * TOT + SL + n_lambda]
                _x_roll[i + 1] = (A @ _x_roll[i]
                                  + B_ctrl @ u_seq[i]
                                  + D @ lam_i
                                  + d)
            x_seq = _x_roll

        # 4.j — cache u_seq for the next solve's `‖u − u_prev‖²_R` cost.
        if self._penalize_input_change:
            self._u_prev_solve = u_seq.copy()

        # First-horizon contact force for Aydinoglu eq. 36 τ_ff feedforward.
        # λ = [γ; λ_n; λ_t] under Stewart-Trinkle; we expose only the physical
        # components.
        # D1: expose BOTH z_sol and delta views. Default consumers see
        # delta (the complementarity-feasible projection — see GATE
        # verdict + investigation T2). Flip via `solver.expose_zsol = True`
        # for A/B comparison.
        # §7.36 — under Anitescu the layout collapses to a single λ block
        # (no γ, no λ_n/λ_t split). Slot-keyed extraction below is
        # ST-specific; we stash the raw Anitescu λ block in
        # _last_lambda_anitescu_first / _horizon and leave the ST-keyed
        # views as zero placeholders (executor pipeline under Anitescu is
        # the separate next block — this guard keeps the solver shape-
        # correct for the smoke).
        if num_normals > 0 and _is_st_c3p:
            _SLN0 = SL + num_normals                   # λ_n offset in step 0's λ slot
            _SLT0 = SL + 2 * num_normals               # λ_t offset
            _n_t  = J_t.shape[0]
            # First-knot dual views
            self._last_lambda_n_first_zsol  = z_sol[_SLN0 : _SLN0 + num_normals].copy()
            self._last_lambda_n_first_delta = delta[_SLN0 : _SLN0 + num_normals].copy()
            self._last_lambda_t_first_zsol  = (z_sol[_SLT0 : _SLT0 + _n_t].copy()
                                               if _n_t > 0 else np.zeros(0))
            self._last_lambda_t_first_delta = (delta[_SLT0 : _SLT0 + _n_t].copy()
                                               if _n_t > 0 else np.zeros(0))
            # Horizon dual views, shape (N, ·).
            _ln_h_z = np.zeros((N, num_normals))
            _ln_h_d = np.zeros((N, num_normals))
            _lt_h_z = np.zeros((N, _n_t)) if _n_t > 0 else np.zeros((N, 0))
            _lt_h_d = np.zeros((N, _n_t)) if _n_t > 0 else np.zeros((N, 0))
            for _k in range(N):
                _base = _k * TOT
                _ln_h_z[_k] = z_sol[_base + _SLN0 : _base + _SLN0 + num_normals]
                _ln_h_d[_k] = delta[_base + _SLN0 : _base + _SLN0 + num_normals]
                if _n_t > 0:
                    _lt_h_z[_k] = z_sol[_base + _SLT0 : _base + _SLT0 + _n_t]
                    _lt_h_d[_k] = delta[_base + _SLT0 : _base + _SLT0 + _n_t]
            self._last_lambda_n_horizon_zsol  = _ln_h_z
            self._last_lambda_n_horizon_delta = _ln_h_d
            self._last_lambda_t_horizon_zsol  = _lt_h_z
            self._last_lambda_t_horizon_delta = _lt_h_d

            # Default-consumer aliases (selectable via expose_zsol).
            if self.expose_zsol:
                self._last_lambda_n_first   = self._last_lambda_n_first_zsol
                self._last_lambda_t_first   = self._last_lambda_t_first_zsol
                self._last_lambda_n_horizon = _ln_h_z
                self._last_lambda_t_horizon = _lt_h_z
            else:
                self._last_lambda_n_first   = self._last_lambda_n_first_delta
                self._last_lambda_t_first   = self._last_lambda_t_first_delta
                self._last_lambda_n_horizon = _ln_h_d
                self._last_lambda_t_horizon = _lt_h_d
            # ST path: no Anitescu λ stash
            self._last_lambda_anitescu_first   = np.zeros(0)
            self._last_lambda_anitescu_horizon = np.zeros((N, 0))
        elif num_normals > 0:
            # §7.36 — Anitescu single-block layout: stash raw folded λ.
            # 4.g conformance (2026-07-27): reference push_t runs
            # end_on_qp_step=false (c3.cc:336-347), so its published
            # z_sol λ is the PROJECTED δ copy — complementarity-feasible,
            # phantom-zeroed — while x_sol_/u_sol_ stay QP. The ST branch
            # above already defaults consumers to δ (expose_zsol=False);
            # this Anitescu branch was still reporting the raw QP λ —
            # the source of the phantom λ=0.6-19 readings on far rows
            # (p111/p115-p117 telemetry). Mirror the dual-view pattern:
            # default = δ, expose_zsol=True flips back to QP for A/B.
            _LAN0 = SL
            _lam_src = z_sol if self.expose_zsol else delta
            self._last_lambda_anitescu_first = \
                _lam_src[_LAN0 : _LAN0 + n_lambda].copy()
            _la_h = np.zeros((N, n_lambda))
            for _k in range(N):
                _base = _k * TOT
                _la_h[_k] = _lam_src[_base + SL : _base + SL + n_lambda]
            self._last_lambda_anitescu_horizon = _la_h
            # Anitescu → per-pair λ_n recovery. Under Anitescu, each contact
            # has 4 folded λ components (lcs_formulator.py:1765-1798,
            # NUM_FRICTION_DIRECTIONS=2 → 4 dirs/contact). Reduce to per-pair
            # activation magnitude by summing the 4 components per contact
            # (mirrors E_t_an at lcs_formulator.py:1771). This populates
            # `_last_lambda_n_first` (n_c,) so the downstream tag-filter at
            # sampling_based_c3_controller.py:2762-2779 can identify active
            # EE-BOX pairs — without this, `_lam_n` was zeros(n_c) and the
            # `_lam_n_mag > 0.05` gate blocked u_sol force routing even when
            # PORT_FORCE_ROUTING=u_sol was set. Root cause of T-push
            # λ_EE-BOX=0 executor symptom.
            _dirs_per_contact = int(n_lambda // num_normals) if num_normals > 0 else 4
            _lam_an0 = _lam_src[_LAN0 : _LAN0 + n_lambda]
            _lam_n_per_pair = np.array([
                float(np.sum(np.abs(
                    _lam_an0[i * _dirs_per_contact : (i + 1) * _dirs_per_contact]
                ))) for i in range(num_normals)
            ])
            _lam_n_horizon = np.zeros((N, num_normals))
            for _k in range(N):
                _base = _k * TOT + SL
                _la_k = _lam_src[_base : _base + n_lambda]
                for _i in range(num_normals):
                    _lam_n_horizon[_k, _i] = float(np.sum(np.abs(
                        _la_k[_i * _dirs_per_contact : (_i + 1) * _dirs_per_contact]
                    )))
            self._last_lambda_n_first        = _lam_n_per_pair
            self._last_lambda_t_first        = np.zeros(J_t.shape[0])
            self._last_lambda_n_horizon      = _lam_n_horizon
            self._last_lambda_t_horizon      = np.zeros((N, J_t.shape[0]))
            self._last_lambda_n_first_zsol   = None
            self._last_lambda_n_first_delta  = None
            self._last_lambda_t_first_zsol   = None
            self._last_lambda_t_first_delta  = None
            self._last_lambda_n_horizon_zsol  = None
            self._last_lambda_n_horizon_delta = None
            self._last_lambda_t_horizon_zsol  = None
            self._last_lambda_t_horizon_delta = None
        else:
            self._last_lambda_n_first        = np.zeros(0)
            self._last_lambda_t_first        = np.zeros(0)
            self._last_lambda_n_horizon      = np.zeros((N, 0))
            self._last_lambda_t_horizon      = np.zeros((N, 0))
            self._last_lambda_n_first_zsol   = None
            self._last_lambda_n_first_delta  = None
            self._last_lambda_t_first_zsol   = None
            self._last_lambda_t_first_delta  = None
            self._last_lambda_n_horizon_zsol  = None
            self._last_lambda_n_horizon_delta = None
            self._last_lambda_t_horizon_zsol  = None
            self._last_lambda_t_horizon_delta = None
            self._last_lambda_anitescu_first  = np.zeros(0)
            self._last_lambda_anitescu_horizon = np.zeros((N, 0))

        # LCS-scaling unscale — reference c3.cc:349-354. The ADMM solved in
        # scaled space (D *= scale, so internal λ_scaled = λ_physical/scale).
        # Recover physical λ magnitudes by multiplying all cached views by
        # _lcs_scale. Executor/downstream code sees physical Newtons.
        if _lcs_scale != 1.0 and num_normals > 0:
            for _attr in (
                "_last_lambda_n_first",
                "_last_lambda_t_first",
                "_last_lambda_n_horizon",
                "_last_lambda_t_horizon",
                "_last_lambda_n_first_zsol",
                "_last_lambda_n_first_delta",
                "_last_lambda_t_first_zsol",
                "_last_lambda_t_first_delta",
                "_last_lambda_n_horizon_zsol",
                "_last_lambda_n_horizon_delta",
                "_last_lambda_t_horizon_zsol",
                "_last_lambda_t_horizon_delta",
                "_last_lambda_anitescu_first",
                "_last_lambda_anitescu_horizon",
            ):
                _v = getattr(self, _attr, None)
                if _v is not None and hasattr(_v, "shape") and _v.size > 0:
                    setattr(self, _attr, _v * _lcs_scale)

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
            print(f"[MATH.QP-C3+]        bbox: λ_n ≥ 0, "
                  f"|u| ≤ {torque_limit:.1f} {self.u_unit_str} "
                  f"({self.u_unit_kind})  "
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
            # |u[0]| is the L2 norm of the 3-vector [Fx,Fy,Fz] (EE-space mode)
            # or a 1-vector norm (arm-torque mode). u_axis lists the components
            # per axis for per-axis-cap diagnostics — the box constraint is
            # per-axis, so L2 alone can be misleading (see path A diagnostic).
            _u0 = np.atleast_1d(u_seq[0])
            _axis_str = ",".join(f"{v:+.2f}" for v in _u0)
            print(f"[C3+] step={self._diag_step} "
                  f"|u[0]|={np.linalg.norm(u_seq[0]):.2f}{self.u_unit_str} "
                  f"u_axis=({_axis_str}){self.u_unit_str} "
                  f"λ_n_max={lam_n_max:.3f} η_n_max={eta_n_max:.3f} "
                  f"primal={pr_last:.3f} iters={actual_iters}/{admm_iter}")
        else:
            print(f"[C3+] step={self._diag_step} n_λ=0  "
                  f"|u[0]|={np.linalg.norm(u_seq[0]):.3f} {self.u_unit_str}")

        if self._lprobe_path is not None:
            self._lprobe_n_solves += 1
            self._lprobe_mpc_step  = self._diag_step

        return u_seq, x_seq


# Module-level alias so unit tests can import without instantiating C3Solver
def project_lorentz(lam_n: float,
                    lam_t: "np.ndarray",
                    mu: float) -> "tuple[float, np.ndarray]":
    """Public wrapper around C3Solver._project_single_contact for testing."""
    return C3Solver._project_single_contact(lam_n, lam_t, mu)


def project_C3Plus_eq12(lam: "np.ndarray",
                               eta: "np.ndarray",
                               u_lambda: float = 1.0,
                               u_eta:    float = 1.0
                               ) -> "tuple[np.ndarray, np.ndarray]":
    """Public wrapper around C3Solver._project_C3Plus for testing.   ← C3+ NEW

    Implements Bui 2026 ICRA eq (12) — the C3+ δ-update closed form.
    """
    return C3Solver._project_C3Plus(lam, eta, u_lambda, u_eta)
