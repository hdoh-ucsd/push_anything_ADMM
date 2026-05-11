"""
LCS (Linearized Complementarity System) Formulator.

Extracts, at each timestep, the dynamics matrices and contact geometry
needed by the ADMM solver from the Drake MultibodyPlant.

Dynamics (continuous-time Newton-Euler):
    M(q) v_dot + C(q,v) v = tau_g(q) + B u + J_n^T lambda_n + J_t^T lambda_t

Phase 1 — first-order linearization (Aydinoglu 2024 eq. 8):
    f(q, v, u) = M(q)^-1 (B u - C(q,v) v + tau_g(q))
    v_{k+1} = v_k + Δt · (J_f · [q;v;u] + d_v_offset + M^-1 J_c^T λ)
    where J_f = ∂f/∂(q,v,u) is computed via Drake autodiff and
          d_v_offset = f(q*,v*,u*) - J_f · [q*;v*;u*] is the constant
                       offset that makes the linearization exact at
                       the linearization point.

Contact geometry:
    phi : (n_c,)       signed gap distances (negative = penetrating)
    J_n : (n_c, n_v)   normal contact Jacobians
    J_t : (4*n_c, n_v) tangential Jacobians (4-edge quadhedron per contact)
    mu  : float        uniform friction coefficient from task config
"""
import numpy as np
import pydrake.all as ad
from pydrake.autodiffutils import (
    InitializeAutoDiff, ExtractValue, ExtractGradient,
)

try:
    from profiling.section_timer import timed
except ImportError:
    from contextlib import contextmanager
    @contextmanager
    def timed(_name):   # noqa: E306
        yield

# Single authoritative EE body name — must match EE_BODY_NAME in env_builder.py.
_EE_BODY_NAME = "pusher"


class LCSFormulator:
    """
    Parameters
    ----------
    plant    : Drake MultibodyPlant (must be Finalized, inside a Diagram)
    mu       : float  Friction coefficient from task config.
    obj_body : Drake Body  The manipuland (box_link / ball_link).
               When supplied, contact pairs are filtered to only those
               between the manipuland and the pusher sphere.
               Without filtering, nc=32-59 phantom pairs corrupt the QP.
    """

    def __init__(self, plant, mu: float = 0.5, obj_body=None,
                 plant_ad=None, context_ad=None):
        self.plant = plant
        self.mu    = float(mu)

        self.n_q = plant.num_positions()
        self.n_v = plant.num_velocities()
        self.n_u = plant.num_actuators()

        # Autodiff plant — required for Phase 1 first-order linearization
        # (Aydinoglu eq. 8). build_environment() now returns these alongside
        # the float plant; construct LCSFormulator with both.
        assert plant_ad is not None and context_ad is not None, (
            "LCSFormulator requires plant_ad and context_ad for Aydinoglu 2024 "
            "eq. (8) first-order linearization. Update build_environment() "
            "callers to receive (diagram, plant, panda_model, object_model, "
            "meshcat, plant_ad, context_ad)."
        )
        self.plant_ad   = plant_ad
        self.context_ad = context_ad

        # Geometry ID sets for contact-pair filtering.
        self._manipuland_geom_ids: set = set()
        self._ee_geom_ids: set = set()

        if obj_body is not None:
            for gid in plant.GetCollisionGeometriesForBody(obj_body):
                self._manipuland_geom_ids.add(gid)

        # EE contact filter: dedicated spherical pusher only — no fallbacks.
        print("[FILTER INIT] Building EE geometry ID set:")
        ee_body = plant.GetBodyByName(_EE_BODY_NAME)
        gids    = list(plant.GetCollisionGeometriesForBody(ee_body))
        for gid in gids:
            self._ee_geom_ids.add(gid)
        print(f"  {_EE_BODY_NAME}: {len(gids)} collision geom(s)")
        assert self._ee_geom_ids, (
            f"No collision geometry on '{_EE_BODY_NAME}' — "
            "check build_environment() registers pusher_collision before Finalize()"
        )
        print(f"[FILTER INIT] EE body: {_EE_BODY_NAME}  "
              f"geom IDs: {list(self._ee_geom_ids)}")
        print(f"[FILTER INIT] Manipuland geom IDs : {len(self._manipuland_geom_ids)}")

    # ------------------------------------------------------------------
    def extract_dynamics(self, context):
        """
        Return M, Cv, tau_g, B at the state encoded in context.

        Returns
        -------
        M     : (n_v, n_v)  mass / inertia matrix
        Cv    : (n_v,)      Coriolis + centripetal bias
        tau_g : (n_v,)      gravity generalised forces
        B     : (n_v, n_u)  actuation matrix
        """
        with timed("lcs.extract_dynamics"):
            M     = self.plant.CalcMassMatrixViaInverseDynamics(context)
            Cv    = self.plant.CalcBiasTerm(context)
            tau_g = self.plant.CalcGravityGeneralizedForces(context)
            B     = self.plant.MakeActuationMatrix()
        return M, Cv, tau_g, B

    # ------------------------------------------------------------------
    def extract_dynamics_with_jacobian(self, context, u_lin):
        """
        Compute M, Cv, tau_g, B AND J_f = ∂f/∂(q,v,u) at (q*, v*, u*),
        plus the value f_eval = f(q*, v*, u*), where
            f(q, v, u) = M(q)^-1 (B u - C(q, v) v + tau_g(q)).

        np.linalg.solve doesn't accept AutoDiffXd dtype-object arrays,
        so we autodiff (M, Cv, tau_g) separately and apply the chain rule
        for M^-1 by hand:
            df/dx = M^-1 [drhs/dx - (dM/dx) f]

        Parameters
        ----------
        context : Drake plant context (the float plant) at (q*, v*).
        u_lin   : (n_u,) linearization input u*.

        Returns
        -------
        M, Cv, tau_g, B   : same as extract_dynamics, evaluated at (q*, v*).
        J_f               : (n_v, n_q + n_v + n_u) Jacobian.
        f_eval            : (n_v,) value of f(q*, v*, u*).
        """
        n_q, n_v, n_u = self.n_q, self.n_v, self.n_u
        n_dec = n_q + n_v + n_u

        # 1. Float values at the linearization point
        M, Cv, tau_g, B = self.extract_dynamics(context)
        rhs_d = B @ u_lin - Cv + tau_g
        M_inv = np.linalg.inv(M)
        f_eval = M_inv @ rhs_d

        # 2. Seed AD on (q, v, u_lin) and evaluate dynamics on the AD plant.
        with timed("lcs.extract_dynamics"):
            q_star = self.plant.GetPositions(context)
            v_star = self.plant.GetVelocities(context)
            decvar = np.concatenate([q_star, v_star, u_lin])
            decvar_ad = InitializeAutoDiff(decvar)
            decvar_ad = decvar_ad.flatten() if decvar_ad.ndim > 1 else decvar_ad
            q_ad = decvar_ad[:n_q]
            v_ad = decvar_ad[n_q : n_q + n_v]
            u_ad = decvar_ad[n_q + n_v :]

            self.plant_ad.SetPositions(self.context_ad, q_ad)
            self.plant_ad.SetVelocities(self.context_ad, v_ad)

            M_ad     = self.plant_ad.CalcMassMatrixViaInverseDynamics(self.context_ad)
            Cv_ad    = self.plant_ad.CalcBiasTerm(self.context_ad)
            tau_g_ad = self.plant_ad.CalcGravityGeneralizedForces(self.context_ad)
            B_ad     = self.plant_ad.MakeActuationMatrix()
            rhs_ad   = B_ad @ u_ad - Cv_ad + tau_g_ad

        # 3. Chain rule for f = M^-1 rhs:
        #    df/dx = M^-1 [drhs/dx - (dM/dx) f]
        J_M   = ExtractGradient(M_ad).reshape(n_v, n_v, n_dec)
        J_rhs = ExtractGradient(rhs_ad)                # (n_v, n_dec)
        J_f   = np.empty((n_v, n_dec))
        for k in range(n_dec):
            J_f[:, k] = M_inv @ (J_rhs[:, k] - J_M[:, :, k] @ f_eval)

        return M, Cv, tau_g, B, J_f, f_eval

    # ------------------------------------------------------------------
    def extract_lcs_contacts(self, context,
                             distance_threshold: float = 0.10):
        """
        Find all geometry pairs within distance_threshold and compute
        gap, normal Jacobian, and quadhedron tangential Jacobians.

        The context must come from a diagram context (not a standalone
        plant context) so the geometry query port is connected to SceneGraph.

        Returns
        -------
        phi : (n_c,)        signed distances
        J_n : (n_c, n_v)    normal Jacobians
        J_t : (4*n_c, n_v)  tangential Jacobians (4 per contact)
        mu  : float         friction coefficient
        """
        with timed("lcs.geometry_query"):
            query_obj = self.plant.get_geometry_query_input_port().Eval(context)
            inspector = query_obj.inspector()
            sd_pairs  = query_obj.ComputeSignedDistancePairwiseClosestPoints(
                distance_threshold
            )

        # Keep only pusher-to-object pairs; discard arm self-collision,
        # arm-table, arm-base, and object-ground pairs.  Without this filter
        # nc=32-59, producing phantom λ_n up to 33 N that saturate QP torques.
        if self._manipuland_geom_ids and self._ee_geom_ids:
            sd_pairs = [
                sdp for sdp in sd_pairs
                if (sdp.id_A in self._manipuland_geom_ids and
                    sdp.id_B in self._ee_geom_ids)
                or (sdp.id_B in self._manipuland_geom_ids and
                    sdp.id_A in self._ee_geom_ids)
            ]

        n_filtered = len(sd_pairs)
        if n_filtered > 10:
            print(f"[LCS] WARNING: {n_filtered} contact pairs after filtering "
                  f"(expected ≤10) — check EE/object geometry IDs")

        W = self.plant.world_frame()
        phis, J_n_rows, J_t_rows = [], [], []

        # Stored for diagnostic access by the MPC controller
        self._last_nhats: list  = []   # world-frame normals (force-on-box direction)
        self._last_contact_info: list = []  # dicts for one-time geometry print

        for sdp in sd_pairs:
            phis.append(sdp.distance)

            body_A = self.plant.GetBodyFromFrameId(
                inspector.GetFrameId(sdp.id_A))
            body_B = self.plant.GetBodyFromFrameId(
                inspector.GetFrameId(sdp.id_B))

            # Translational velocity Jacobians at the contact witness points
            with timed("lcs.calc_jacobians"):
                J_A = self.plant.CalcJacobianTranslationalVelocity(
                    context, ad.JacobianWrtVariable.kV,
                    body_A.body_frame(), sdp.p_ACa, W, W,
                )  # (3, n_v)
                J_B = self.plant.CalcJacobianTranslationalVelocity(
                    context, ad.JacobianWrtVariable.kV,
                    body_B.body_frame(), sdp.p_BCb, W, W,
                )  # (3, n_v)

            J_rel = J_A - J_B       # relative velocity Jacobian (3, n_v)
            nhat  = sdp.nhat_BA_W   # contact normal (unit, 3,) — from B to A

            # Determine which body is the manipuland (box) so we can report
            # the direction of force ON the box.
            # Convention: J_n^T λ_n applies generalized force (J_A - J_B)^T nhat λ_n.
            # Force on box = J_box^T * nhat_onto_box * λ_n where:
            #   A=box → nhat_onto_box = nhat_BA_W (away from EE toward box)
            #   A=EE  → nhat_onto_box = -nhat_BA_W (same direction, different sign)
            a_is_box = (sdp.id_A in self._manipuland_geom_ids)
            nhat_onto_box = np.array(nhat) if a_is_box else -np.array(nhat)
            self._last_nhats.append(nhat_onto_box)
            self._last_contact_info.append({
                "body_A": body_A.name(), "body_B": body_B.name(),
                "a_is_box": a_is_box,
                "nhat_BA_W": np.array(nhat),
                "nhat_onto_box": nhat_onto_box,
                "p_ACa": np.array(sdp.p_ACa),
                "p_BCb": np.array(sdp.p_BCb),
                "distance": float(sdp.distance),
            })

            # Normal Jacobian row
            J_n_rows.append(nhat @ J_rel)   # (n_v,)

            # Tangential Jacobians: 4-edge quadhedron {t1, -t1, t2, -t2}
            ref = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(nhat, ref))) > 0.99:
                ref = np.array([0.0, 1.0, 0.0])
            t1 = np.cross(nhat, ref)
            t1 = t1 / np.linalg.norm(t1)
            t2 = np.cross(nhat, t1)   # unit (nhat ⊥ t1, both unit)

            for d in (t1, -t1, t2, -t2):
                J_t_rows.append(d @ J_rel)  # (n_v,)

        if not phis:
            return (
                np.zeros(0),
                np.zeros((0, self.n_v)),
                np.zeros((0, self.n_v)),
                self.mu,
            )

        return (
            np.array(phis),
            np.vstack(J_n_rows),    # (n_c, n_v)
            np.vstack(J_t_rows),    # (4*n_c, n_v)
            self.mu,
        )

    # ------------------------------------------------------------------
    def linearize_discrete(self, context, dt: float, u_lin=None):
        """
        Linearize the Drake plant into a discrete-time LCS at (q*, v*, u*).

        Phase 1 (Aydinoglu eq. 8) gives the first-order dynamics linearization
        with autodiff Jacobian J_f.  Phase 2 (Aydinoglu eq. 9) extends the
        return tuple with the Stewart-Trinkle complementarity slack expression
            η = E·x + F·λ + H·u + c,        0 ≤ λ ⊥ η ≥ 0
        with λ = [γ; λ_n; λ_t] of dimension 6·n_c (γ is the friction-cone
        slack).  This shape is shared by both the C3 and C3+ paths from
        Phase 2 onward; the prior `linearize_discrete_with_complementarity`
        method is now a thin alias that returns the same tuple.

        Dynamics (state x = [q; v]):
            x[t+1] = A x[t] + B_ctrl u[t] + D λ[t] + d
        where (with J_q, J_v, J_u = decompositions of J_f, and Δt = dt):
            A[:n_q, :n_q] = I + dt² · N · J_q
            A[:n_q, n_q:] = dt · N · (I + dt · J_v)
            A[n_q:, :n_q] = dt · J_q
            A[n_q:, n_q:] = I + dt · J_v
            B_ctrl[:n_q]  = dt² · N · J_u
            B_ctrl[n_q:]  = dt · J_u
            D has zero columns in the γ slot (γ does not enter dynamics);
              for λ_n / λ_t cols: D[:n_q]=dt²·N·M⁻¹·J_*^T, D[n_q:]=dt·M⁻¹·J_*^T
            d[:n_q]       = dt² · N · d_v_offset
            d[n_q:]       = dt · d_v_offset
            d_v_offset    = f(q*, v*, u*) − J_f · [q*; v*; u*]

        Stewart-Trinkle LCP rows (Aydinoglu eq. 9, with v_{k+1} substituted):
            γ row    : 0 ≤ γ   ⊥  μ·λ_n − E_t·λ_t                  ≥ 0
            λ_n row  : 0 ≤ λ_n ⊥  φ/dt + (1/dt)·J_n·(q−q*) + J_n·v_{k+1}  ≥ 0
            λ_t row  : 0 ≤ λ_t ⊥  E_t^T·γ + J_t·v_{k+1}            ≥ 0
        where E_t ∈ ℝ^{n_c×4n_c} has e=[1,1,1,1] on the 4 tangent slots of
        each contact.  After substituting v_{k+1} = v + dt·v_dot_lin we get
        E, F, H, c populated as documented in test_lcs_efhc.py.

        Parameters
        ----------
        context : Drake plant context at (q*, v*).
        dt      : planning timestep (s).
        u_lin   : (n_u,) linearization input u*. None → zeros.

        Returns
        -------
        A      : (n_x, n_x)
        B_ctrl : (n_x, n_u)
        D      : (n_x, n_λ)   n_λ = 6·n_c; γ-cols are zero
        d      : (n_x,)
        E      : (n_λ, n_x)
        F      : (n_λ, n_λ)
        H      : (n_λ, n_u)
        c_vec  : (n_λ,)
        J_n    : (n_c, n_v)
        J_t    : (4·n_c, n_v)
        phi    : (n_c,)
        mu     : float
        """
        if u_lin is None:
            u_lin = np.zeros(self.n_u)
        else:
            u_lin = np.asarray(u_lin, dtype=float).reshape(self.n_u)

        # Phase 1 — autodiff Jacobian of f at (q*, v*, u_lin).
        M, Cv, tau_g, B, J_f, f_eval = self.extract_dynamics_with_jacobian(
            context, u_lin)
        phi, J_n, J_t, mu = self.extract_lcs_contacts(context)

        n_q, n_v, n_u = self.n_q, self.n_v, self.n_u
        n_x = n_q + n_v

        # N(q) matrix: q_dot = N(q) @ v, built column-by-column via Drake API
        with timed("lcs.extract_dynamics"):
            N_mat = np.zeros((n_q, n_v))
            for i in range(n_v):
                e = np.zeros(n_v)
                e[i] = 1.0
                N_mat[:, i] = self.plant.MapVelocityToQDot(context, e)

        M_inv = np.linalg.inv(M)
        n_c   = J_n.shape[0]               # number of contacts
        n_t   = J_t.shape[0]               # 4·n_c (polyhedral pyramid)
        n_lam = 2 * n_c + n_t              # 6·n_c — [γ; λ_n; λ_t]
        # Slot offsets within the per-step λ block.
        SG    = 0
        SLN   = n_c
        SLT   = 2 * n_c

        # Decompose J_f into J_q, J_v, J_u blocks.
        J_q = J_f[:, :n_q]                          # ∂f/∂q  (n_v, n_q)
        J_v = J_f[:, n_q : n_q + n_v]               # ∂f/∂v  (n_v, n_v)
        J_u = J_f[:, n_q + n_v :]                   # ∂f/∂u  (n_v, n_u)
                                                    # (== M⁻¹·B at lin point)

        # d_v_offset = f(q*, v*, u*) - J_f · [q*; v*; u*]  (Aydinoglu eq. 8)
        q_star = self.plant.GetPositions(context)
        v_star = self.plant.GetVelocities(context)
        d_v_offset = f_eval - (J_q @ q_star + J_v @ v_star + J_u @ u_lin)

        # A — substituting v_{k+1} into q_{k+1} = q + dt·N·v_{k+1}:
        #   q_{k+1} = (I + dt²·N·J_q) q + dt·N·(I + dt·J_v) v + ...
        #   v_{k+1} = dt·J_q q + (I + dt·J_v) v + ...
        A = np.zeros((n_x, n_x))
        A[:n_q, :n_q] = np.eye(n_q) + (dt * dt) * (N_mat @ J_q)
        A[:n_q, n_q:] = dt * N_mat @ (np.eye(n_v) + dt * J_v)
        A[n_q:, :n_q] = dt * J_q
        A[n_q:, n_q:] = np.eye(n_v) + dt * J_v

        # B_ctrl picks up the same N·dt cross-term in the q-block.
        B_ctrl = np.zeros((n_x, n_u))
        B_ctrl[:n_q] = (dt * dt) * (N_mat @ J_u)
        B_ctrl[n_q:] = dt * J_u

        # D — λ ordering is [γ; λ_n; λ_t]; γ-cols zero (no dynamics coupling).
        D = np.zeros((n_x, n_lam))
        if n_c > 0:
            Minv_JnT = M_inv @ J_n.T               # (n_v, n_c)
            Minv_JtT = M_inv @ J_t.T               # (n_v, 4·n_c)
            D[:n_q,  SLN:SLN + n_c]      = (dt * dt) * (N_mat @ Minv_JnT)
            D[n_q:,  SLN:SLN + n_c]      = dt * Minv_JnT
            D[:n_q,  SLT:SLT + n_t]      = (dt * dt) * (N_mat @ Minv_JtT)
            D[n_q:,  SLT:SLT + n_t]      = dt * Minv_JtT

        # d — uses d_v_offset; same q-block cross-term as A.
        d_vec = np.zeros(n_x)
        d_vec[:n_q] = (dt * dt) * (N_mat @ d_v_offset)
        d_vec[n_q:] = dt * d_v_offset

        # ---- Stewart-Trinkle LCP slack expression (Aydinoglu eq. 9) -------
        E_lcs = np.zeros((n_lam, n_x))
        F_lcs = np.zeros((n_lam, n_lam))
        H_lcs = np.zeros((n_lam, n_u))
        c_lcs = np.zeros(n_lam)

        if n_c > 0:
            # E_t: n_c × 4n_c with e = [1,1,1,1] on the 4 tangent slots of
            # each contact. The friction-cone slack row reads
            #     μ·λ_n − E_t·λ_t = γ.
            E_t = np.zeros((n_c, n_t))
            for i in range(n_c):
                E_t[i, 4 * i : 4 * (i + 1)] = 1.0

            # γ rows (slot SG : SG+n_c) — no x, u, c dependence.
            F_lcs[SG:SG + n_c,  SLN:SLN + n_c]  = mu * np.eye(n_c)
            F_lcs[SG:SG + n_c,  SLT:SLT + n_t]  = -E_t

            # λ_n rows (slot SLN : SLN+n_c).
            # We use the simpler gap discretization (matches Bui v1 and
            # avoids the floating-base n_q ≠ n_v mismatch from Aydinoglu's
            # explicit (1/dt)·J_n·(q − q*) term):
            #   η_n = phi(q*)/dt + J_n · v_{k+1}     (gap prediction)
            # Substitute v_{k+1} = v + dt·(J_q q + J_v v + J_u u + d_v + Minv·J_c^T λ_phys):
            E_lcs[SLN:SLN + n_c, :n_q]            = dt * (J_n @ J_q)
            E_lcs[SLN:SLN + n_c, n_q:n_q + n_v]   = J_n + dt * (J_n @ J_v)
            F_lcs[SLN:SLN + n_c, SLN:SLN + n_c]   = dt * (J_n @ Minv_JnT)
            F_lcs[SLN:SLN + n_c, SLT:SLT + n_t]   = dt * (J_n @ Minv_JtT)
            H_lcs[SLN:SLN + n_c, :]               = dt * (J_n @ J_u)
            c_lcs[SLN:SLN + n_c]                  = phi / dt + dt * (J_n @ d_v_offset)

            # λ_t rows (slot SLT : SLT+4n_c).
            #   η_t = E_t^T·γ + J_t·v_{k+1}
            E_lcs[SLT:SLT + n_t, :n_q]            = dt * (J_t @ J_q)
            E_lcs[SLT:SLT + n_t, n_q:n_q + n_v]   = J_t + dt * (J_t @ J_v)
            F_lcs[SLT:SLT + n_t, SG:SG + n_c]     = E_t.T
            F_lcs[SLT:SLT + n_t, SLN:SLN + n_c]   = dt * (J_t @ Minv_JnT)
            F_lcs[SLT:SLT + n_t, SLT:SLT + n_t]   = dt * (J_t @ Minv_JtT)
            H_lcs[SLT:SLT + n_t, :]               = dt * (J_t @ J_u)
            c_lcs[SLT:SLT + n_t]                  = dt * (J_t @ d_v_offset)

        if not getattr(self, '_printed_contact_frames', False) and J_n.shape[0] > 0:
            self._printed_contact_frames = True
            nc = J_n.shape[0]
            n_tangent_per_contact = J_t.shape[0] // nc
            print(f"[SANITY] nc={nc}  n_tangent_per_contact={n_tangent_per_contact}  "
                  f"J_n={J_n.shape}  J_t={J_t.shape}")
            print(f"[SANITY] Tangent interpretation: "
                  f"{'4 → polyhedral pyramid' if n_tangent_per_contact == 4 else '2 → Lorentz'}")

            for i, info in enumerate(self._last_contact_info):
                print(f"[CONTACT {i}]")
                print(f"  body_A={info['body_A']}  body_B={info['body_B']}  "
                      f"a_is_box={info['a_is_box']}")
                print(f"  nhat_BA_W (B→A): {np.round(info['nhat_BA_W'], 4)}")
                print(f"  nhat_onto_box  : {np.round(info['nhat_onto_box'], 4)}")
                print(f"  p_ACa (on A)   : {np.round(info['p_ACa'], 4)}")
                print(f"  p_BCb (on B)   : {np.round(info['p_BCb'], 4)}")
                print(f"  distance       : {info['distance']:.5f} m")

            # J_n sign test: J_n[i, box_vx_dof] should be positive when nhat_onto_box
            # is in the +x direction (EE to west of box, pushing box east).
            # A positive value means λ_n > 0 accelerates box in the correct direction.
            print(f"[SIGN] J_n[0] (first contact normal row):")
            print(f"       {np.round(J_n[0], 5)}")
            print(f"  nhat_onto_box = {np.round(self._last_nhats[0], 4)}")
            print(f"  → if nhat_onto_box·[1,0,0] > 0, box should accelerate eastward "
                  f"from λ_n")

        return (
            A, B_ctrl, D, d_vec,
            E_lcs, F_lcs, H_lcs, c_lcs,
            J_n, J_t, phi, mu,
        )

    # ------------------------------------------------------------------
    def linearize_discrete_with_complementarity(self, context, dt: float,
                                                u_lin=None):
        """
        Phase 2 alias — `linearize_discrete` now returns the complementarity
        slack expression (E, F, H, c) directly. This wrapper exists only for
        backward compatibility with C3+ call sites and forwards verbatim.
        Phase 3 will deprecate it in favour of `linearize_discrete_anitescu`
        for the C3+ path.

        Returns
        -------
        A, B_ctrl, D, d_vec, E, F, H, c_vec, J_n, J_t, phi, mu
        """
        return self.linearize_discrete(context, dt, u_lin=u_lin)


