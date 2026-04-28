"""
LCS (Linearized Complementarity System) Formulator.

Extracts, at each timestep, the dynamics matrices and contact geometry
needed by the ADMM solver from the Drake MultibodyPlant.

Dynamics (continuous-time Newton-Euler):
    M(q) v_dot + C(q,v) v = tau_g(q) + B u + J_n^T lambda_n + J_t^T lambda_t

Contact geometry:
    phi : (n_c,)       signed gap distances (negative = penetrating)
    J_n : (n_c, n_v)   normal contact Jacobians
    J_t : (4*n_c, n_v) tangential Jacobians (4-edge quadhedron per contact)
    mu  : float        uniform friction coefficient from task config
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

    def __init__(self, plant, mu: float = 0.5, obj_body=None):
        self.plant = plant
        self.mu    = float(mu)

        self.n_q = plant.num_positions()
        self.n_v = plant.num_velocities()
        self.n_u = plant.num_actuators()

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
    def linearize_discrete(self, context, dt: float):
        """
        Linearize the Drake plant into a discrete-time LCS at the current state.

        State x = [q; v], dimension n_x = n_q + n_v.
        N(q), M(q), and Cv(q,v) are evaluated at the current (q, v) and held
        constant over the horizon (first-order LCS approximation).
        Coriolis bias is folded into the constant offset term d.

        Discrete-time model:
            x[t+1] = A x[t] + B_ctrl u[t] + D λ[t] + d
        where:
            A      = [[I,     dt·N(q)],
                      [0,     I      ]]
            B_ctrl = [[0             ],
                      [dt·M⁻¹·B     ]]
            D      = [[0             ],
                      [dt·M⁻¹·Jc^T  ]]
            d      = [[0             ],
                      [dt·M⁻¹·(τ_g − Cv)]]

        Returns
        -------
        A      : (n_x, n_x)
        B_ctrl : (n_x, n_u)
        D      : (n_x, n_c)   zeros matrix if no contacts
        d      : (n_x,)
        J_n    : (num_normals, n_v)
        J_t    : (4*num_normals, n_v)
        phi    : (num_normals,)
        mu     : float
        """
        with timed("lcs.extract_dynamics"):
            M, Cv, tau_g, B = self.extract_dynamics(context)
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
        num_normals = J_n.shape[0]
        n_c = num_normals + J_t.shape[0]
        J_c = np.vstack([J_n, J_t]) if num_normals > 0 else np.zeros((0, n_v))

        # A = [[I_nq, dt*N]; [0, I_nv]]
        A = np.zeros((n_x, n_x))
        A[:n_q, :n_q] = np.eye(n_q)
        A[:n_q, n_q:] = dt * N_mat
        A[n_q:, n_q:] = np.eye(n_v)

        # B_ctrl = [[0]; [dt * M_inv @ B]]
        B_ctrl = np.zeros((n_x, n_u))
        B_ctrl[n_q:] = dt * (M_inv @ B)

        # D = [[0]; [dt * M_inv @ J_c^T]]
        D = np.zeros((n_x, n_c)) if n_c > 0 else np.zeros((n_x, 0))
        if n_c > 0:
            D[n_q:] = dt * (M_inv @ J_c.T)

        # d = [0; dt * M_inv @ (tau_g - Cv)]
        d_vec = np.zeros(n_x)
        d_vec[n_q:] = dt * (M_inv @ (tau_g - Cv))

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

        return A, B_ctrl, D, d_vec, J_n, J_t, phi, mu
