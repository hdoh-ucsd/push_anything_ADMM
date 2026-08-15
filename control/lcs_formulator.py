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
import os
import contextlib
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
                 plant_ad=None, context_ad=None,
                 object_shape: str = "box",
                 mu_per_pair_type: dict | None = None,
                 controller_object_mass: float | None = None,
                 tshape_mesh_witnesses: bool = False):
        """
        mu_per_pair_type : optional dict mapping contact-pair tag
            ("EE-BOX", "BOX-GND", "EE-GND") to a per-pair friction
            coefficient. When set, overrides the scalar `mu` on a
            per-pair basis in `extract_lcs_contacts`. Reference:
            sampling_c3plus_options.yaml:44 `mu_per_pair_type`.
        """
        self.plant = plant
        self.mu    = float(mu)
        # Mesh-T migration (2026-08-11): ground-witness table comes from the
        # reference T_shape_video mesh footprint instead of the box-T table.
        self._tshape_mesh_witnesses = bool(tshape_mesh_witnesses)
        # Reference ships a SEPARATE, heavier object model for the CONTROLLER
        # than for the sim, and the difference is task-specific:
        #     push_t   push_t.sdf 1.0 kg  == push_t_control.sdf 1.0 kg   (1x)
        #     jacktoy  jack.sdf 0.156 kg  vs jack_control.sdf 0.99 kg    (6.35x)
        #     anything H_shape_texture.sdf 0.05 kg vs
        #              H_shape_texture_controller.sdf 1.0 kg             (20x)
        # i.e. the planning model is always ~1 kg regardless of the true mass.
        # The port runs ONE plant for sim and control, so without this it plans
        # at the SIM mass -- fine for push_t (they coincide), but 6.35x too
        # light for the jack, where it makes the LCS predict a phantom LAUNCH:
        # at 0.156 kg, u=1 N across a 16 mm gap lifts the object 19.45 mm in
        # the model (physics: 0.00 mm) while a real topple needs only 5.11 mm,
        # so the planner never plans the roll. At 0.99 kg the launch is 0.00 mm.
        # None => use the plant's own mass (the historical behaviour).
        self._controller_object_mass = (
            None if controller_object_mass is None
            else float(controller_object_mass))
        self._obj_body = obj_body
        self._object_shape = str(object_shape)
        self._mu_per_pair_type = (
            {str(k): float(v) for k, v in mu_per_pair_type.items()}
            if mu_per_pair_type else None
        )
        if self._mu_per_pair_type is not None:
            print(f"[LCS-MU-PER-PAIR] loaded overrides: "
                  f"{self._mu_per_pair_type}  (scalar fallback mu={self.mu})",
                  flush=True)

        # Reference dairlib push_anything_dev@257e3ed:
        #   contact_model = 'anitescu' (sampling_c3_options.yaml:9)
        #   always-on pair admission (LCSFactory::LinearizePlantToLCS, no
        #     distance threshold)
        #   3 T-ground / 4 box-vertex-ground witnesses (per shape)
        #   no box_ground_drag (Anitescu holds λ_n_gnd on its own)
        #   no normal-row patches (Stewart-Trinkle-specific, dead here)
        self._contact_model = "anitescu"
        # Reference c3/core/lcs.cc:46 ScaleComplementarityDynamics.
        # Reference push_t/parameters/sampling_c3plus_options.yaml:11
        # `scale_lcs: true`. Rescales the LCS complementarity block so
        # ||D|| == ||A|| — improves ADMM conditioning by keeping state and
        # λ-direction gradients balanced. Without it, port's 3-iter ADMM
        # converges only on state, leaves λ under-updated → planner |u|
        # stuck at ~1N regardless of goal_dist (p43 diagnosis). Default
        # True matches reference push_t.
        self._scale_lcs = True   # 2026-07-28 defaults flip (was REFCONF_SCALE_LCS)
        # Closest EE-box pair injection. CORRECTED 2026-07-28: this is
        # REFERENCE-CONFORMANT, not port-only — the reference has NO
        # distance threshold at all; it resolves each contact-pair group
        # to its N closest pairs unconditionally
        # (sampling_based_c3_controller.cc:1595-1614 ResolveContactPairs →
        # LCSFactory::GetNClosestContactPairs, counts from
        # resolve_contacts_to_lists, push_t planner [0,1,3] = 0 EE-ground /
        # 1 EE-T / 3 T-ground). The port's 2 mm signed-distance admission
        # (extract_lcs_contacts) is the port-side divergence; this fallback
        # approximates N-closest=1 whenever the threshold admits no EE-BOX
        # pair. Hard-wired ON (was PORT_LCS_ALWAYS_ON_EE_BOX, defaults
        # flip). The tshape path supersedes it with exact top-K admission
        # (planner k=1, cost k=2 — matches reference [0,1,3]/[0,2,3]).
        self._always_on_ee_box = True
        self._ref_pair_admission_planner_lcs = True
        self._box_drag_c = 0.0
        # 2026-08-15 conformance step 3: the box joins the tshape at the
        # reference count of 3 ground witnesses (reference resolves the
        # ground-object group to 3 sphere contacts for EVERY anything
        # object — resolve_contacts_to_lists [[0,1,3,1]]). Other shapes
        # (hshape, jack) keep their existing defaults/config overrides.
        self.lcs_explicit_manipuland_ground_contacts = (
            3 if object_shape in ("tshape", "box") else 4)

        # Legacy Stewart-Trinkle normal-row patches: all no-ops under
        # Anitescu (no separate normal row). Kept as dead defaults so
        # downstream code doesn't need conditional guards.
        self._normal_compliance_k = 0.0
        self._normal_velocity_level = False
        self._normal_phi_clamp_v_cap = None

        # LCS EE inertia: the reference's 0.057 kg free-floating point mass
        # (end_effector_simple_model.urdf). The arc-2 "A1" arm operational-
        # space inertia was REMOVED 2026-08-08 — its premise ("the same 3x3
        # the reference full-plant LCS produces") was false: the reference's
        # LCS plant has no arm at all (AddLCSModelsToPlant loads EE + ground
        # + objects only). See the LCS-EE-MASS block in
        # linearize_discrete_ee_space for the measured 152x divergence and
        # its mechanism. Flag retained (False) only so the removal is
        # explicit to readers; nothing reads it.
        self._use_arm_cartesian_inertia = False
        self._arm_cart_inertia_banner_done = False

        # Lazy-initialized box half-extents (queried from geometry inspector
        # on the first synthesis call; needs a context, not available here).
        self._box_half_extents = None
        self._ground_z = 0.0

        # Cache of the most recent planner-dt seen by linearize_discrete*.
        # Used by _maybe_dump_filter_audit to report sim_t correctly; was
        # previously hardcoded as step*0.01 assuming 100 Hz.
        self._last_planner_dt: float = 0.01

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
        # Ground/table geometry IDs — collected from the plant's
        # world_body, which is where env_builder.py:143-149 registers
        # "table_collision" as a static 2 m × 2 m × 0.1 m box. Admitting
        # the (manipuland, ground) pair into the LCS gives the planner a
        # physical model for box-ground friction; without it the planner
        # predicts the box coasts undisturbed after a single tap (see
        # the LCS-PREDICT diagnostic dump).
        self._ground_geom_ids: set = set()

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

        # Ground filter: collision geometries on plant.world_body() (the
        # static table). Cleanest unambiguous identification — avoids
        # name-string matching.
        for gid in plant.GetCollisionGeometriesForBody(plant.world_body()):
            self._ground_geom_ids.add(gid)

        print(f"[FILTER INIT] EE body: {_EE_BODY_NAME}  "
              f"geom IDs: {list(self._ee_geom_ids)}")
        print(f"[FILTER INIT] Manipuland geom IDs : {len(self._manipuland_geom_ids)}")
        print(f"[FILTER INIT] Ground geom IDs     : {len(self._ground_geom_ids)}  "
              f"(world_body collision geoms)")

        # Contact-pair diagnostic (off by default; enabled via --contacts-diag)
        self._diag_contacts      = False
        self._diag_contacts_step = 0

        # One-shot filter audit (fires at first rich-mode entry / near contact).
        # Read-only — prints what Drake returns at wide threshold, what the
        # 2 mm narrow threshold keeps, what the admission filter admits, and
        # the resulting J_n row content. Used to diagnose why EE-box pair is
        # missing from the LCS at φ ≈ 0.34 mm.
        self._diag_dumped     = False
        self._diag_step_count = 0
        # Set externally by the wrapper at the first kToC3ReachedReposTarget
        # event (control/sampling_c3/wrapper.py). The audit dumps on the
        # NEXT extract_lcs_contacts call after this flips True, which is
        # the LCS-formulation moment for the first rich-mode plan.
        self._rich_mode_just_entered = False
        # Write the audit to a project-local file so it survives even when
        # the parent process's stdout capture (e.g. Claude Code's
        # /tmp/claude-0 staging) fails with EIO.
        self._diag_output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "audit_output",
            "lcs_filter_audit.txt",
        )
        os.makedirs(os.path.dirname(self._diag_output_path), exist_ok=True)

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
    def _maybe_init_box_half_extents(self, query_obj):
        """Lazy-init self._box_half_extents from the manipuland's collision
        geometry. Fires once on the first synthesis call. Returns the
        cached (3,) np.ndarray on subsequent calls."""
        if self._box_half_extents is not None:
            return self._box_half_extents
        inspector = query_obj.inspector()
        for gid in self._manipuland_geom_ids:
            shape = inspector.GetShape(gid)
            if isinstance(shape, ad.Box):
                # Drake Box.size() returns (3,) full edge lengths.
                self._box_half_extents = np.asarray(shape.size()) / 2.0
                return self._box_half_extents
        # Fallback for non-Box manipulands (sphere etc.): no synthesis possible.
        self._box_half_extents = np.zeros(3)
        return self._box_half_extents

    def _box_vertex_set_body_frame(self, n_synth: int) -> np.ndarray:
        """Return (3, n_synth) array of body-frame contact points to enumerate
        for box-ground synthesis. Choices:
          3  → REFERENCE support triangle (2026-08-15 conformance, step 3).
               Every reference anything object carries exactly 3 ground-
               contact spheres (r=1 mm, centers on the base plane) in its
               *controller* model — two near the corners of one base edge
               plus one at the midpoint of the opposite edge (e.g.
               expo_box_controller.sdf:283-311: pair at (-0.0637, ±~0.07),
               single at (+0.0605, ~0)), resolved to all 3 by
               resolve_contacts_to_lists [[0,1,3,1]]. Port analog for the
               cube footprint: pair at (-0.048, ±0.048), single at
               (+0.048, 0), 2 mm inset, base plane. Same surface-point
               convention as the mesh-T witness table (sphere centers,
               r_tip=0).
          4  → 4 bottom corners only       (pre-conformance port default)
          8  → all 8 cube vertices         (corners; top set inactive at rest)
          12 → 8 vertices + 4 bottom-face edge midpoints
        """
        hx, hy, hz = self._box_half_extents
        if n_synth == 3:
            _ix, _iy = hx - 0.002, hy - 0.002
            pts = np.array([[-_ix, +_iy, -hz],
                            [-_ix, -_iy, -hz],
                            [+_ix,  0.0, -hz]]).T
        elif n_synth == 4:
            pts = np.array([[+hx, +hy, -hz],
                            [+hx, -hy, -hz],
                            [-hx, +hy, -hz],
                            [-hx, -hy, -hz]]).T
        elif n_synth == 8:
            pts = np.array([[sx, sy, sz]
                            for sx in (+hx, -hx)
                            for sy in (+hy, -hy)
                            for sz in (+hz, -hz)]).T
        elif n_synth == 12:
            # 8 corners + 4 bottom-face edge midpoints
            corners = [[sx, sy, sz]
                       for sx in (+hx, -hx)
                       for sy in (+hy, -hy)
                       for sz in (+hz, -hz)]
            edge_mids = [[+hx,   0.0, -hz],
                         [-hx,   0.0, -hz],
                         [  0.0, +hy, -hz],
                         [  0.0, -hy, -hz]]
            pts = np.array(corners + edge_mids).T
        else:
            raise ValueError(
                f"box vertex-set n_synth must be 4, 8, or 12 (got {n_synth})"
            )
        return pts

    def _tshape_vertex_set_body_frame(self, n_synth: int) -> np.ndarray:
        """Return T-shape bottom-face witness points in the link frame.

        Reference push_t geometry (see sim/env_builder.py:_tshape_sdf):
          - vertical_bar (crossbar): pose (+0.05, 0, 0), size 0.16 × 0.04 × 0.04
            → bottom face at link-z = -0.02, spans link-x ∈ [-0.03, +0.13],
              link-y ∈ [-0.02, +0.02].
          - horizontal_bar (stem): pose (-0.05, 0, 0, 0, 0, π/2), size 0.16 ×
            0.04 × 0.04 → after 90° z-rot, bottom face at link-z = -0.02,
            spans link-x ∈ [-0.07, -0.03], link-y ∈ [-0.08, +0.08].

        n_synth = 3 (reference push_t resolve_contacts_to=[0,1,3]): a
        triangular support spanning the T footprint —
          W1 = crossbar +x tip     (+0.13,  0.00, -0.02)
          W2 = stem +y tip         (-0.05, +0.08, -0.02)
          W3 = stem -y tip         (-0.05, -0.08, -0.02)
        This layout captures torsional friction resistance (three-point
        contact vs Drake auto-admit's single point).
        """
        if n_synth != 3:
            raise ValueError(
                f"T-shape vertex-set n_synth must be 3 (matches reference "
                f"push_t resolve_contacts_to=[0,1,3]); got {n_synth}."
            )
        if self._tshape_mesh_witnesses:
            # Reference T_shape_video mesh bottom-face support extremities
            # (computed from T_shape_video.obj: bottom ring at z=-0.0243).
            return np.array([
                [+0.1168, +0.0069, -0.0243],   # crossbar +x tip
                [-0.0620, +0.0691, -0.0243],   # arm +y tip
                [-0.0548, -0.0789, -0.0243],   # arm -y tip
            ]).T   # (3, 3)
        return np.array([
            [+0.13,  0.00, -0.02],   # crossbar +x tip
            [-0.05, +0.08, -0.02],   # stem +y tip
            [-0.05, -0.08, -0.02],   # stem -y tip
        ]).T   # (3, 3)

    # Reference jack_control.sdf gives each tip sphere radius 0.025 with its
    # CENTRE at +/-0.0625 along the capsule axis. The witness table below holds
    # centres, so the ground witness sits one radius below -- unlike the box/T/H
    # tables, whose entries are true corner points already on the surface.
    _JACK_TIP_RADIUS = 0.025

    def _jack_vertex_set_body_frame(self, n_synth: int) -> np.ndarray:
        """Return the jack's SIX tip-sphere centres in the link frame.

        Reference: examples/sampling_c3/urdf/jack_control.sdf declares
        capsule_{1,2,3}_sphere_{1,2} at +/-0.0625 along each capsule's axis,
        and franka_sampling_c3_controller.cc:193-204 pairs all six against
        GROUND. `resolve_contacts_to = [0, 1, 3]` then keeps the THREE closest
        (GetNClosestContactPairs) -- i.e. the reference does not hard-code
        which tips are resting, it enumerates every candidate and lets the
        signed distance rank them each tick.

        This is why the jack needs no special case beyond the table: unlike the
        T and H, whose bottom face is fixed in the body frame, the jack rolls
        between tripods and its resting triple changes with pose. The caller
        performs the closest-3 selection.
        """
        h = 0.0625
        return np.array([
            [0.0, 0.0, +h], [0.0, 0.0, -h],   # capsule_1 (body +z)
            [+h, 0.0, 0.0], [-h, 0.0, 0.0],   # capsule_2 (body +x)
            [0.0, +h, 0.0], [0.0, -h, 0.0],   # capsule_3 (body +y)
        ]).T   # (3, 6)

    def _hshape_vertex_set_body_frame(self, n_synth: int) -> np.ndarray:
        """Return H-shape bottom-face witness points in the link frame.

        Geometry (see sim/env_builder.py:_hshape_sdf) — bottom face at
        link-z = -0.016:
          left  bar  x ∈ [-0.056, -0.032], y ∈ [-0.064, +0.064]
          right bar  x ∈ [+0.032, +0.056], y ∈ [-0.064, +0.064]
          crossbar   x ∈ [-0.032, +0.032], y ∈ [-0.012, +0.012]

        n_synth = 4 (DEFAULT for the H): one witness per bar end, at the bar
        centre-lines. A symmetric rectangular support 0.088 × 0.128 whose
        interior contains the CoM projection with a wide margin — the natural
        analogue of the box path's 4-vertex set, and the right choice for a
        shape with two-fold symmetry.

        n_synth = 3 is also accepted (matching the T's triangular support, for
        A/B against push_t): the two bottom bar tips plus the crossbar's
        top-centre. Verified to contain the CoM projection with barycentric
        margin 0.079. Note the "three bar corners" triangle was REJECTED —
        it puts the CoM exactly on an edge (margin 0.0), a degenerate support
        that would let the LCS tip the object about that edge for free.
        """
        z = -0.016
        if n_synth == 12:
            # Reference `anything` resolve_contacts_to_lists starts [0, 1, 12,
            # ...]: 0 EE-ground, 1 EE-object, 12 object-ground. Twelve maps
            # exactly onto the H's 3-box decomposition as the 4 bottom corners
            # of each box, which is also the densest support the geometry
            # admits without duplicating points.
            v = []
            for (x0, x1, y0, y1) in ((-0.056, -0.032, -0.064, +0.064),
                                     (+0.032, +0.056, -0.064, +0.064),
                                     (-0.032, +0.032, -0.012, +0.012)):
                v += [[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]]
            return np.array(v).T   # (3, 12)
        if n_synth == 4:
            return np.array([
                [-0.044, -0.064, z],   # left bar, -y end
                [+0.044, -0.064, z],   # right bar, -y end
                [+0.044, +0.064, z],   # right bar, +y end
                [-0.044, +0.064, z],   # left bar, +y end
            ]).T   # (3, 4)
        if n_synth == 3:
            return np.array([
                [-0.044, -0.064, z],   # left bar, -y end
                [+0.044, -0.064, z],   # right bar, -y end
                [ 0.000, +0.012, z],   # crossbar, +y centre
            ]).T   # (3, 3)
        raise ValueError(
            f"H-shape vertex-set n_synth must be 3, 4 or 12; got {n_synth}."
        )

    def _synthesize_manipuland_ground_contacts(self, context, query_obj):
        """Synthesize N manipuland-vertex ↔ ground contact rows. Dispatches
        on self._object_shape:
          - "box"    → _box_vertex_set_body_frame(n_synth) [n∈{3,4,8,12}].
          - "tshape" → _tshape_vertex_set_body_frame(n_synth) [n=3].
          - "hshape" → _hshape_vertex_set_body_frame(n_synth) [n∈{3,4,12}].
          - "jack"   → _jack_vertex_set_body_frame(n_synth) [6 candidates,
            closest n_synth kept per tick — the rolling tripod].

        Returns four parallel lists in the same format as Drake-admitted
        contacts:
          phis, J_n_rows, J_t_rows, ci

        On no-op (knob=0, unsupported shape, or missing obj_body): empty lists.
        """
        if self.lcs_explicit_manipuland_ground_contacts == 0 or self._obj_body is None:
            return [], [], [], []

        n_synth = self.lcs_explicit_manipuland_ground_contacts
        if self._object_shape == "tshape":
            verts_body = self._tshape_vertex_set_body_frame(n_synth)
        elif self._object_shape == "hshape":
            verts_body = self._hshape_vertex_set_body_frame(n_synth)
        elif self._object_shape == "jack":
            verts_body = self._jack_vertex_set_body_frame(n_synth)
        elif self._object_shape == "box":
            half_extents = self._maybe_init_box_half_extents(query_obj)
            if not np.all(half_extents > 0):
                return [], [], [], []
            verts_body = self._box_vertex_set_body_frame(n_synth)
        else:
            # Unsupported shape (e.g. "sphere"): no vertex enumeration.
            return [], [], [], []

        box_frame    = self._obj_body.body_frame()
        W            = self.plant.world_frame()
        nhat_ground  = np.array([0.0, 0.0, 1.0])    # ground normal: force on box +z

        # Witness offset below the tabled point. Zero for the box/T/H, whose
        # table entries are true surface corners; one sphere radius for the
        # jack, whose entries are tip-sphere centres.
        r_tip = self._JACK_TIP_RADIUS if self._object_shape == "jack" else 0.0

        # Reference GetNClosestContactPairs (sampling_based_c3_controller.cc:
        # 1605-1614): when a contact group offers more candidates than the
        # group's resolve_contacts_to count, keep the CLOSEST n. The box/T/H
        # tables are built exactly n_synth long so this is a no-op for them;
        # the jack supplies 6 tips and keeps the resting 3, re-selected every
        # tick as the object rolls.
        if verts_body.shape[1] > n_synth:
            R_WB = self.plant.CalcRelativeRotationMatrix(
                context, W, box_frame).matrix()
            _z = np.array([
                float(self.plant.CalcPointsPositions(
                    context, box_frame, verts_body[:, j:j+1], W).flatten()[2])
                for j in range(verts_body.shape[1])
            ])
            keep = np.argsort(_z, kind="stable")[:n_synth]
            verts_body = verts_body[:, np.sort(keep)]

        phis_s:     list = []
        J_n_rows_s: list = []
        J_t_rows_s: list = []
        ci_s:       list = []

        # Body-frame offset that carries a tip-sphere centre down to the
        # sphere's lowest surface point (world -z). Identity when r_tip == 0.
        if r_tip > 0.0:
            R_WB_now = self.plant.CalcRelativeRotationMatrix(
                context, W, box_frame).matrix()
            _drop_body = R_WB_now.T @ np.array([0.0, 0.0, -r_tip])
        else:
            _drop_body = np.zeros(3)

        for i in range(n_synth):
            # The witness — and the point the Jacobian must be taken at — is
            # the surface point, not the tabled centre.
            pt_body = (verts_body[:, i] + _drop_body).reshape(3, 1)
            # World position of this vertex
            pt_world = self.plant.CalcPointsPositions(
                context, box_frame, pt_body, W,
            ).flatten()
            phi_i = float(pt_world[2] - self._ground_z)
            phis_s.append(phi_i)

            # Translational Jacobian at this body-frame point (3, n_v).
            # Ground is welded (world_body); relative Jacobian == box Jacobian.
            J_box = self.plant.CalcJacobianTranslationalVelocity(
                context, ad.JacobianWrtVariable.kV,
                box_frame, pt_body, W, W,
            )  # (3, n_v)

            # Normal Jacobian row: nhat-projected (force on box upward).
            J_n_rows_s.append(nhat_ground @ J_box)

            # Tangent Jacobians: 4-edge polyhedral pyramid in xy plane,
            # matching the structure used for Drake-admitted pairs (line 393).
            for t_dir in (np.array([1.0, 0.0, 0.0]),
                          np.array([-1.0, 0.0, 0.0]),
                          np.array([0.0, 1.0, 0.0]),
                          np.array([0.0, -1.0, 0.0])):
                J_t_rows_s.append(t_dir @ J_box)

            # Contact-info dict in the same shape as Drake's (line 354-363).
            # Tagged shape-specific (BOX-VERT-i / T-VERT-i) so the diagnostic
            # distinguishes synthesized rows from Drake's "EE-BOX" / "BOX-GND".
            # Downstream tag consumers (_derive_force_command EE-BOX filter,
            # B1-A pair-index scan) match on "EE-BOX" prefix only, so these
            # synthesized rows are correctly excluded from EE-force intent.
            _tag_prefix = self._SYNTH_GND_TAG_PREFIX_BY_SHAPE.get(
                self._object_shape, "BOX-VERT")
            ci_s.append({
                "body_A":       self._obj_body.name(),
                "body_B":       "ground (world_body)",
                "a_is_box":     True,
                "tag":          f"{_tag_prefix}-{i}",
                "nhat_BA_W":    nhat_ground.copy(),
                "nhat_onto_box": nhat_ground.copy(),
                "p_ACa":        pt_body.flatten().copy(),
                "p_BCb":        np.array([pt_world[0], pt_world[1], self._ground_z]),
                "distance":     phi_i,
            })

        return phis_s, J_n_rows_s, J_t_rows_s, ci_s

    # ------------------------------------------------------------------
    # Tag prefix used for each shape's SYNTHESIZED manipuland-ground rows.
    # Single source of truth: _synthesize_manipuland_ground_contacts stamps
    # the tag from here, and _mu_for_tag collapses anything in here onto
    # "BOX-GND" for the mu_per_pair_type lookup. Keeping both sides on one
    # table is what stops a new shape from silently getting scalar-fallback
    # friction (which is exactly what happened to the H).
    _SYNTH_GND_TAG_PREFIX_BY_SHAPE = {
        "tshape": "T-VERT",
        "hshape": "H-VERT",
        "jack":   "J-TIP",
        "box":    "BOX-VERT",
    }
    _SYNTH_GND_TAG_PREFIXES = tuple(
        set(_SYNTH_GND_TAG_PREFIX_BY_SHAPE.values()))

    def _mu_for_tag(self, tag: str) -> float:
        """Return per-pair-type μ using self._mu_per_pair_type override
        if present, otherwise fall back to the scalar self.mu.

        Tag normalization: synthesized manipuland-ground rows use
        shape-prefixed tags (see _synthesize_manipuland_ground_contacts):
        "BOX-VERT-{i}", "T-VERT-{i}", "H-VERT-{i}", "J-TIP-{i}". All are
        collapsed onto "BOX-GND" for lookup so the yaml `mu_per_pair_type`
        map only needs the three canonical keys
        (EE-BOX / BOX-GND / EE-GND).

        2026-08-10: this list was BOX-VERT/T-VERT only, so the H's twelve
        synthesized ground rows (added later, tagged H-VERT) silently fell
        through to the scalar `self.mu` -- 0.3 for push_h instead of the
        configured BOX-GND 0.4615, a 35% under-estimate of ground friction
        in every H run to date. The jack's J-TIP rows would have inherited
        the same hole. Driven off the shared prefix set below so a new
        shape cannot reintroduce it by adding a tag and forgetting this
        function.
        """
        if self._mu_per_pair_type is not None:
            _lookup = tag
            if any(tag.startswith(pfx) for pfx in self._SYNTH_GND_TAG_PREFIXES):
                _lookup = "BOX-GND"
            v = self._mu_per_pair_type.get(_lookup)
            if v is not None:
                return float(v)
        return float(self.mu)

    # ------------------------------------------------------------------
    def extract_lcs_contacts(self, context,
                             # 0.002 m is the validated Pareto-optimal value.
                             # An ablation at 0.040 m (results/thresh40_west_*)
                             # raised EE-box admission from 40% → 87% but
                             # regressed West box motion from 29 mm → 10 mm:
                             # the looser threshold admits LCS rows at φ > 0
                             # which must satisfy complementarity (λ_n = 0)
                             # and collapses the dispatcher cost-gap that
                             # triggers kToC3Cost entries. 2 mm slack covers
                             # Drake signed-distance discretization noise.
                             distance_threshold: float = 0.002,
                             # §9 Option B (5-pair cost-LCS): top-N-by-phi
                             # EE-manipuland admission. Reference push_t
                             # resolve_contacts_to_lists=[[0,1,3],[0,2,3]] →
                             # planner LCS uses top-1 (n_ee_top_k=1),
                             # cost-LCS uses top-2 (n_ee_top_k=2). Default
                             # n_ee_top_k=1 preserves planner behavior.
                             # Only applies to the always-on injection path
                             # (fires when no EE-manipuland pair admits at
                             # distance_threshold). When Drake auto-admits
                             # ≥1 EE-manipuland pair at 2 mm, all admitted
                             # pairs pass through regardless of n_ee_top_k.
                             n_ee_top_k: int = 1,
                             # §9 Option B faithful cost-LCS: when True,
                             # unconditionally REPLACE the EE-manipuland
                             # slice with the top-K (by phi) candidate
                             # pairs — bypasses both the 2 mm auto-admit
                             # and _always_on_ee_box gates. Mirrors the
                             # reference's GetResolvedContactPairs
                             # (sampling_based_c3_controller.cc:1582-1615):
                             # each contact-group is resolved to its top-N
                             # closest pairs. Used by inner_solve.py to
                             # build the cost-LCS with EXACTLY n_ee_top_k
                             # EE-manipuland rows regardless of setback
                             # distance — the load-bearing piece for the
                             # productive-face distinction (east vs north
                             # on the T). Default False preserves the
                             # planner LCS build byte-identically.
                             force_top_k_ee_box: bool = False):
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

        # Snapshot the pre-admit list so the audit diagnostic can compare
        # what Drake returned (with the 2 mm threshold) against what the
        # admission filter kept.
        sd_pairs_pre_admit = list(sd_pairs)

        # Keep two kinds of pairs:
        #   (1) EE↔manipuland — the active push contact.
        #   (2) manipuland↔ground — provides physical box-ground friction
        #       in the LCS so the planner doesn't predict the box coasts
        #       undisturbed (see the LCS-PREDICT diagnostic). Originally
        #       excluded because "phantom λ_n up to 33 N saturated QP
        #       torques" — but the OSC executor's per-joint URDF box
        #       constraints handle torque saturation natively, so the
        #       prior rationale no longer applies.
        # All other pairs (arm self-collision, arm-table, arm-base) stay
        # excluded.
        if self._manipuland_geom_ids and self._ee_geom_ids:
            # When Stage 1 / §9 Option A synthesis is active (knob > 0),
            # suppress Drake's auto-admitted BOX-GND pair: the synthesized
            # manipuland-vertex contacts replace it (otherwise the single
            # Drake pair would be double-counted alongside the synth rows).
            _synth_active = self.lcs_explicit_manipuland_ground_contacts > 0
            def _admit(sdp):
                ee_box = ((sdp.id_A in self._manipuland_geom_ids and
                           sdp.id_B in self._ee_geom_ids)
                       or (sdp.id_B in self._manipuland_geom_ids and
                           sdp.id_A in self._ee_geom_ids))
                box_ground = ((sdp.id_A in self._manipuland_geom_ids and
                               sdp.id_B in self._ground_geom_ids)
                           or (sdp.id_B in self._manipuland_geom_ids and
                               sdp.id_A in self._ground_geom_ids))
                if _synth_active and box_ground:
                    return False    # de-dup: synthesis owns this contact
                return ee_box or box_ground
            sd_pairs = [sdp for sdp in sd_pairs if _admit(sdp)]

            # §7.30 — Always-on EE-BOX admission. If the flag is set and
            # the 2 mm threshold did NOT admit an EE-BOX pair this step,
            # inject the EE-manipuland pair explicitly (regardless of phi)
            # via the pair-specific Drake call which does NOT apply the
            # threshold. Mirrors lcs_factory.cc:31-105 (every contact_geom
            # iterated unconditionally) for the EE-manipuland pair only.
            #
            # Multi-collision-element bodies (e.g. T-shape: vertical_bar +
            # horizontal_bar): admit the CLOSEST manipuland collision by phi.
            # The prior version broke after the first iteration → for a T,
            # picked whichever element came first in set order regardless of
            # geometric proximity, producing a non-closing LCS row and
            # λ_n = 0/NaN downstream. This iterates all pairs and picks the
            # smallest-phi one.
            #
            # Open-gap fidelity note: the faithful reference fix is
            # GetNClosestContactPairs (top-N by phi so the planner has
            # multiple candidate contact modes). This single-closest patch
            # unblocks the T; the N-closest port is the open item.
            # §9 Option B (5-pair cost-LCS): faithful GetResolvedContactPairs
            # for the EE-manipuland group — REPLACE all Drake-auto-admitted
            # EE-manipuland pairs with the top-K closest candidates. This
            # decouples the cost-LCS from the setback distance so the
            # forward-sim sees EE-T contact rows for east-face samples
            # that sit 30 mm outside 2 mm auto-admit, giving the productive-
            # face distinction (east < north) the LCP needs.
            if force_top_k_ee_box:
                _n_to_admit = max(1, int(n_ee_top_k))
                # Drop any auto-admitted EE-manipuland pairs (top-K will
                # supersede them; keep BOX-GND and synthesized rows).
                sd_pairs = [
                    sdp for sdp in sd_pairs
                    if not (
                        (sdp.id_A in self._manipuland_geom_ids
                         and sdp.id_B in self._ee_geom_ids)
                        or (sdp.id_B in self._manipuland_geom_ids
                            and sdp.id_A in self._ee_geom_ids))
                ]
                _all_candidates = []
                for gid_ee in self._ee_geom_ids:
                    for gid_box in self._manipuland_geom_ids:
                        _all_candidates.append(
                            query_obj.ComputeSignedDistancePairClosestPoints(
                                gid_ee, gid_box))
                _all_candidates.sort(key=lambda s: s.distance)
                for _sdp_i in _all_candidates[:_n_to_admit]:
                    sd_pairs.append(_sdp_i)
            elif self._always_on_ee_box:
                _has_ee_box = any(
                    ((sdp.id_A in self._manipuland_geom_ids
                      and sdp.id_B in self._ee_geom_ids)
                     or (sdp.id_B in self._manipuland_geom_ids
                         and sdp.id_A in self._ee_geom_ids))
                    for sdp in sd_pairs
                )
                if not _has_ee_box:
                    # Gather all EE↔manipuland candidate pairs.
                    _all_candidates = []
                    for gid_ee in self._ee_geom_ids:
                        for gid_box in self._manipuland_geom_ids:
                            sdp_candidate = (
                                query_obj.ComputeSignedDistancePairClosestPoints(
                                    gid_ee, gid_box))
                            _all_candidates.append(sdp_candidate)
                    # Sort by signed distance; admit top-N (N=n_ee_top_k).
                    # §9 Option B: cost-LCS uses top-2 (reference
                    # resolve_contacts_to_for_cost=[0,2,3] for push_t).
                    _all_candidates.sort(key=lambda s: s.distance)
                    _n_to_admit = max(1, int(n_ee_top_k))
                    for _sdp_i in _all_candidates[:_n_to_admit]:
                        sd_pairs.append(_sdp_i)

        n_filtered = len(sd_pairs)
        if n_filtered > 10:
            print(f"[LCS] WARNING: {n_filtered} contact pairs after filtering "
                  f"(expected ≤10) — check EE/object geometry IDs")

        W = self.plant.world_frame()
        phis, J_n_rows, J_t_rows = [], [], []
        # Per-pair μ vector — aligned with `phis` (one entry per admitted
        # pair). Populated from `self._mu_per_pair_type` (if set) using
        # the pair's tag ("EE-BOX" | "BOX-GND" | "EE-GND" | "OTHER").
        # Falls back to the scalar `self.mu` when the map is unset or
        # the tag is not present. Reference: sampling_c3plus_options.yaml:44.
        mus: list = []

        # Stored for diagnostic access by the MPC controller
        self._last_nhats: list  = []   # world-frame normals (force-on-box direction)
        self._last_contact_info: list = []  # dicts for one-time geometry print
        # (p_contact_W, nhat_onto_box) tuples — EE-BOX pairs only. Used by
        # the rotation-bonus sample scorer (inner_solve.py).
        self._last_ee_box_contacts: list = []

        for sdp in sd_pairs:
            phis.append(sdp.distance)

            body_A = self.plant.GetBodyFromFrameId(
                inspector.GetFrameId(sdp.id_A))
            body_B = self.plant.GetBodyFromFrameId(
                inspector.GetFrameId(sdp.id_B))

            # Drake's SignedDistancePair witness points p_ACa/p_BCb are
            # expressed in the GEOMETRY frames of A/B — NOT the body
            # frames. For geometries registered at a non-identity pose in
            # their body (the T's two collision bars at ±0.05 m with a 90°
            # yaw; the pusher tip sphere at its z-offset) using them
            # directly as body-frame offsets mis-places the contact lever
            # arm: on the T crossbar the witness r_x flips sign
            # (+0.03 → −0.02), inverting the predicted twist direction —
            # the p112-p114 wrong-way-rotation root cause (see
            # scripts/test_r7_twist_sign.py). Identity-posed single
            # geometries (cube manipuland, ground plane) are unaffected,
            # which is why the box task never exposed this. Map to body
            # frame via each geometry's registered pose X_BG.
            _X_BG_A = inspector.GetPoseInFrame(sdp.id_A)
            _X_BG_B = inspector.GetPoseInFrame(sdp.id_B)
            p_ACa_B = np.asarray(
                _X_BG_A.multiply(np.asarray(sdp.p_ACa)), dtype=float)
            p_BCb_B = np.asarray(
                _X_BG_B.multiply(np.asarray(sdp.p_BCb)), dtype=float)

            # Translational velocity Jacobians at the contact witness points
            with timed("lcs.calc_jacobians"):
                J_A = self.plant.CalcJacobianTranslationalVelocity(
                    context, ad.JacobianWrtVariable.kV,
                    body_A.body_frame(), p_ACa_B, W, W,
                )  # (3, n_v)
                J_B = self.plant.CalcJacobianTranslationalVelocity(
                    context, ad.JacobianWrtVariable.kV,
                    body_B.body_frame(), p_BCb_B, W, W,
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
            if (sdp.id_A in self._ee_geom_ids and
                    sdp.id_B in self._manipuland_geom_ids) or \
               (sdp.id_B in self._ee_geom_ids and
                    sdp.id_A in self._manipuland_geom_ids):
                _tag = "EE-BOX"
            elif (sdp.id_A in self._manipuland_geom_ids and
                    sdp.id_B in self._ground_geom_ids) or \
                 (sdp.id_B in self._manipuland_geom_ids and
                    sdp.id_A in self._ground_geom_ids):
                _tag = "BOX-GND"
            else:
                _tag = "OTHER"
            # Geometry-element name identifies WHICH collision element on the
            # manipuland was contacted (matters for multi-collision-element
            # bodies like the T — 'vertical_bar' vs 'horizontal_bar'). For
            # single-element bodies (box, sphere) it's the sole collision name.
            _elem_A = inspector.GetName(sdp.id_A)
            _elem_B = inspector.GetName(sdp.id_B)
            _manip_elem = _elem_A if a_is_box else _elem_B
            self._last_contact_info.append({
                "body_A": body_A.name(), "body_B": body_B.name(),
                "elem_A": _elem_A, "elem_B": _elem_B,
                "manipuland_element": _manip_elem,
                "a_is_box": a_is_box,
                "tag": _tag,
                "nhat_BA_W": np.array(nhat),
                "nhat_onto_box": nhat_onto_box,
                # BODY-frame witness points (converted from Drake's
                # geometry-frame p_ACa/p_BCb via X_BG — see above).
                "p_ACa": np.array(p_ACa_B),
                "p_BCb": np.array(p_BCb_B),
                "distance": float(sdp.distance),
            })
            mus.append(self._mu_for_tag(_tag))
            # One-line contact-mode log for the T port validation. Emits per
            # EE-manipuland admission: which element was contacted + φ. Muted
            # for non-EE-BOX pairs to avoid ground-contact spam.
            if _tag == "EE-BOX":
                print(f"[CONTACT-ELEM] step={self._diag_step_count} "
                      f"element={_manip_elem} phi={float(sdp.distance):+.4f}m")

            # Rotation-bonus scorer needs (p_contact_W, nhat_onto_box) for
            # EE-BOX pairs only. Use the contact witness on the box body,
            # transformed to world via the current plant context.
            if _tag == "EE-BOX":
                if a_is_box:
                    body_box = body_A
                    p_BoCo = np.asarray(p_ACa_B).reshape(3, 1)
                else:
                    body_box = body_B
                    p_BoCo = np.asarray(p_BCb_B).reshape(3, 1)
                p_contact_W = self.plant.CalcPointsPositions(
                    context, body_box.body_frame(), p_BoCo, W,
                ).flatten()
                self._last_ee_box_contacts.append(
                    (np.array(p_contact_W), np.array(nhat_onto_box))
                )

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

        # === §9 Option A / Stage 1: append synthesized manipuland ↔ ground ===
        # When self.lcs_explicit_manipuland_ground_contacts > 0, append
        # explicit vertex-ground contact rows after the Drake-admitted ones.
        # Drake's BOX-GND pair was already de-duplicated above. The synthesized
        # rows share the polyhedral-pyramid layout (4 tangent dirs per contact)
        # so downstream n_t = 4·n_c arithmetic holds without modification.
        if self.lcs_explicit_manipuland_ground_contacts > 0:
            (phis_s, J_n_rows_s,
             J_t_rows_s, ci_s) = self._synthesize_manipuland_ground_contacts(
                context, query_obj,
            )
            phis.extend(phis_s)
            J_n_rows.extend(J_n_rows_s)
            J_t_rows.extend(J_t_rows_s)
            self._last_contact_info.extend(ci_s)
            for ci in ci_s:
                self._last_nhats.append(ci["nhat_onto_box"])
                # Synthesized rows are all manipuland ↔ ground; use the
                # BOX-GND per-pair μ (or scalar fallback).
                mus.append(self._mu_for_tag(ci.get("tag", "BOX-GND")))
            # ee_box_contacts list is for the rotation-bonus scorer (EE-BOX
            # pairs only) — synthesized BOX-VERT contacts do not contribute.

        # === [LCS-FILTER-AUDIT] one-shot diagnostic ============================
        # Fires once, at the first extract_lcs_contacts call after the
        # wrapper signals "first rich-mode entry" via
        # self._rich_mode_just_entered. Replaces the earlier
        # closest-EE-box-distance trigger, which fired at MPC step 2 on
        # initial-configuration penetration rather than at the moment that
        # actually matters for the r075 paradox (LCS construction during
        # the first rich-mode plan). Disables itself after one fire via
        # self._diag_dumped.
        self._diag_step_count += 1
        if not self._diag_dumped and self._rich_mode_just_entered:
            try:
                self._maybe_dump_filter_audit(
                    context, query_obj, inspector,
                    sd_pairs_pre_admit, sd_pairs,
                    J_n_rows, distance_threshold,
                )
            except Exception as _audit_err:  # noqa: BLE001
                print(f"[LCS-FILTER-AUDIT] diagnostic raised: "
                      f"{type(_audit_err).__name__}: {_audit_err}", flush=True)
            finally:
                # Consume the flag whether or not the dump succeeded so we
                # don't re-attempt every step.
                self._rich_mode_just_entered = False

        if self._diag_contacts:
            self._diag_contacts_step += 1
            n_c = len(self._last_contact_info)
            line = f"[CONTACTS] step={self._diag_contacts_step} n_c={n_c}"
            for i, info in enumerate(self._last_contact_info):
                nh = info["nhat_onto_box"]
                line += (f" |pair{i}: sd={info['distance']*1000:+.1f}mm "
                         f"nhat=({nh[0]:+.4f},{nh[1]:+.4f},{nh[2]:+.4f}) "
                         f"A={info['body_A']} B={info['body_B']} "
                         f"a_is_box={info['a_is_box']}")
            print(line, flush=True)

        if not phis:
            return (
                np.zeros(0),
                np.zeros((0, self.n_v)),
                np.zeros((0, self.n_v)),
                np.zeros(0),
            )

        if not getattr(self, "_mu_per_pair_first_logged", False):
            _tags = [info.get("tag", "?") for info in self._last_contact_info]
            print(f"[LCS-MU-PER-PAIR] first admission: tags={_tags} "
                  f"mus={mus}  scalar_fallback_mu={self.mu}", flush=True)
            self._mu_per_pair_first_logged = True

        return (
            np.array(phis),
            np.vstack(J_n_rows),    # (n_c, n_v)
            np.vstack(J_t_rows),    # (4*n_c, n_v)
            np.asarray(mus, dtype=float),   # (n_c,) per-pair μ
        )

    # ------------------------------------------------------------------
    def _geom_label(self, inspector, gid):
        """Best-effort human label for a Drake GeometryId."""
        try:
            name = inspector.GetName(gid)
        except Exception:  # noqa: BLE001
            name = "<no-name>"
        kind = "?"
        try:
            if gid in self._ee_geom_ids:
                kind = "EE"
            elif gid in self._manipuland_geom_ids:
                kind = "BOX"
            elif gid in self._ground_geom_ids:
                kind = "GND"
            else:
                kind = "other"
        except Exception:  # noqa: BLE001
            pass
        return f"{int(gid.get_value())}/{kind}/{name}"

    def _maybe_dump_filter_audit(self, context, query_obj, inspector,
                                 sd_pairs_pre_admit, sd_pairs_post_admit,
                                 J_n_rows, narrow_threshold):
        """
        One-shot read-only audit of LCS contact-pair selection.

        Trigger: first rich-mode entry (set by the wrapper via
        self._rich_mode_just_entered on kToC3ReachedReposTarget).
        """
        # --- Wide-net signed-distance query is still useful for section (1)
        # (the all-pairs listing) and for computing the EE-box distance
        # reported in the new section (0). The trigger gate has already
        # been applied upstream.
        WIDE = 0.5  # 0.5 m wide net to surface even far-apart pairs
        try:
            sd_pairs_wide = query_obj.ComputeSignedDistancePairwiseClosestPoints(
                WIDE
            )
        except Exception:  # noqa: BLE001
            sd_pairs_wide = []

        min_eb = float('inf')
        eb_pair = None
        for sdp in sd_pairs_wide:
            ee_box = ((sdp.id_A in self._manipuland_geom_ids and
                       sdp.id_B in self._ee_geom_ids)
                   or (sdp.id_B in self._manipuland_geom_ids and
                       sdp.id_A in self._ee_geom_ids))
            if ee_box and sdp.distance < min_eb:
                min_eb = sdp.distance
                eb_pair = sdp

        # --- Mark fired BEFORE printing so a print exception doesn't loop us.
        self._diag_dumped = True
        step = self._diag_step_count
        # Use last-seen planner dt cached at linearize_discrete{,_ee_space}
        # entry (was hardcoded 0.01 assuming 100 Hz; after freq-match to
        # dt=0.075, that under-reported sim_t by ~7.5×).
        sim_t = step * self._last_planner_dt

        # Write through a single open file handle. Stdout still gets a brief
        # tee'd header so live observers see that the audit fired, but every
        # detail line lands in the project-local file (audit_output/
        # lcs_filter_audit.txt) so it survives /tmp output-capture failure.
        f = open(self._diag_output_path, "a")
        def _w(msg):
            f.write(msg + "\n")
        try:
            header = (f"\n=== [LCS-FILTER-AUDIT] step={step} "
                      f"trigger=first-rich-mode-entry "
                      f"sim_t={sim_t:.2f}s "
                      f"narrow_threshold={narrow_threshold*1000:.1f}mm "
                      f"wide_threshold={WIDE*1000:.0f}mm ===")
            print(header, flush=True)
            print(f"[LCS-FILTER-AUDIT] writing detail to "
                  f"{self._diag_output_path}", flush=True)
            _w(header)

            # (0) The most diagnostic single number — EE-box distance now.
            if eb_pair is not None:
                _w(f"[LCS-FILTER-AUDIT] (0) EE-box distance at audit "
                   f"moment: {min_eb*1000:+.3f} mm")
            else:
                _w(f"[LCS-FILTER-AUDIT] (0) EE-box distance at audit "
                   f"moment: >500 mm (no EE-box pair within wide net)")
            _w(f"[LCS-FILTER-AUDIT]     (expected ~0 mm if EE actually "
               f"touched the box face; ~45 mm hyst80-style means EE "
               f"never made it)\n")

            # (1) Drake's raw output at WIDE threshold (all nearby pairs).
            _w(f"[LCS-FILTER-AUDIT] (1) Drake wide-threshold output "
               f"({len(sd_pairs_wide)} pairs at threshold={WIDE} m):")
            sd_pairs_wide_sorted = sorted(sd_pairs_wide,
                                          key=lambda p: p.distance)
            for i, sdp in enumerate(sd_pairs_wide_sorted[:30]):
                la = self._geom_label(inspector, sdp.id_A)
                lb = self._geom_label(inspector, sdp.id_B)
                nh = np.array(sdp.nhat_BA_W)
                pA = np.array(sdp.p_ACa)
                pB = np.array(sdp.p_BCb)
                _w(f"[LCS-FILTER-AUDIT]   pair {i:2d}: "
                   f"A={la}  B={lb}  "
                   f"dist={sdp.distance*1000:+8.3f}mm  "
                   f"nhat_BA_W=({nh[0]:+.3f},{nh[1]:+.3f},{nh[2]:+.3f})  "
                   f"p_ACa=({pA[0]:+.3f},{pA[1]:+.3f},{pA[2]:+.3f})  "
                   f"p_BCb=({pB[0]:+.3f},{pB[1]:+.3f},{pB[2]:+.3f})")
            if len(sd_pairs_wide_sorted) > 30:
                _w(f"[LCS-FILTER-AUDIT]   ... "
                   f"({len(sd_pairs_wide_sorted)-30} more omitted)")

            # (2) Known geometry IDs in this formulator.
            def _ids_with_names(id_set):
                out = []
                for gid in id_set:
                    try:
                        nm = inspector.GetName(gid)
                    except Exception:  # noqa: BLE001
                        nm = "<no-name>"
                    out.append(f"{int(gid.get_value())}({nm})")
                return out

            _w(f"[LCS-FILTER-AUDIT] (2) Known geom ID sets in formulator:")
            _w(f"[LCS-FILTER-AUDIT]   EE     "
               f"({len(self._ee_geom_ids):2d}): "
               f"{_ids_with_names(self._ee_geom_ids)}")
            _w(f"[LCS-FILTER-AUDIT]   BOX    "
               f"({len(self._manipuland_geom_ids):2d}): "
               f"{_ids_with_names(self._manipuland_geom_ids)}")
            _w(f"[LCS-FILTER-AUDIT]   GROUND "
               f"({len(self._ground_geom_ids):2d}): "
               f"{_ids_with_names(self._ground_geom_ids)}")

            # (3) Filter evaluation for every pair returned at the NARROW
            # (2 mm) threshold — these are the only ones eligible for
            # admission.
            _w(f"[LCS-FILTER-AUDIT] (3) Admission-filter evaluation on the "
               f"{len(sd_pairs_pre_admit)} pairs Drake returned at "
               f"narrow={narrow_threshold*1000:.1f}mm:")
            for i, sdp in enumerate(sd_pairs_pre_admit):
                ee_box = ((sdp.id_A in self._manipuland_geom_ids and
                           sdp.id_B in self._ee_geom_ids)
                       or (sdp.id_B in self._manipuland_geom_ids and
                           sdp.id_A in self._ee_geom_ids))
                box_gnd = ((sdp.id_A in self._manipuland_geom_ids and
                            sdp.id_B in self._ground_geom_ids)
                        or (sdp.id_B in self._manipuland_geom_ids and
                            sdp.id_A in self._ground_geom_ids))
                admitted = ee_box or box_gnd
                tag = "EE-BOX" if ee_box else ("BOX-GND" if box_gnd
                                                          else "OTHER")
                la = self._geom_label(inspector, sdp.id_A)
                lb = self._geom_label(inspector, sdp.id_B)
                _w(f"[LCS-FILTER-AUDIT]   pair {i:2d}: "
                   f"A={la}  B={lb}  "
                   f"dist={sdp.distance*1000:+7.3f}mm  "
                   f"tag={tag:7s}  admitted={'Y' if admitted else 'N'}")

            # Cross-check: was the EE-box pair seen at WIDE but absent from
            # NARROW?
            if eb_pair is not None:
                in_narrow = any(
                    ((sdp.id_A == eb_pair.id_A and sdp.id_B == eb_pair.id_B)
                     or (sdp.id_A == eb_pair.id_B
                         and sdp.id_B == eb_pair.id_A))
                    for sdp in sd_pairs_pre_admit
                )
                _w(f"[LCS-FILTER-AUDIT]   EE-box pair "
                   f"(dist={eb_pair.distance*1000:+.3f}mm) "
                   f"present_at_narrow={'Y' if in_narrow else 'N'}")
            else:
                _w(f"[LCS-FILTER-AUDIT]   EE-box pair: NOT in "
                   f"wide-threshold output at all — geom ID sets may "
                   f"be misregistered.")

            # (4) Final admitted pair list (these become LCS rows).
            _w(f"[LCS-FILTER-AUDIT] (4) Final admitted pairs "
               f"({len(sd_pairs_post_admit)} → LCS rows):")
            for i, sdp in enumerate(sd_pairs_post_admit):
                ee_box = ((sdp.id_A in self._manipuland_geom_ids and
                           sdp.id_B in self._ee_geom_ids)
                       or (sdp.id_B in self._manipuland_geom_ids and
                           sdp.id_A in self._ee_geom_ids))
                ptype = "EE-BOX" if ee_box else "BOX-GND"
                nh = np.array(sdp.nhat_BA_W)
                la = self._geom_label(inspector, sdp.id_A)
                lb = self._geom_label(inspector, sdp.id_B)
                _w(f"[LCS-FILTER-AUDIT]   contact {i:2d}: type={ptype:7s}  "
                   f"A={la}  B={lb}  "
                   f"dist={sdp.distance*1000:+7.3f}mm  "
                   f"nhat_BA_W=({nh[0]:+.3f},{nh[1]:+.3f},{nh[2]:+.3f})")

            # (5) Resulting J_n row structure.
            _w(f"[LCS-FILTER-AUDIT] (5) Resulting J_n row structure:")
            if not J_n_rows:
                _w(f"[LCS-FILTER-AUDIT]   J_n is empty (n_c=0)")
            else:
                try:
                    J_n_mat = np.vstack(J_n_rows)
                    _w(f"[LCS-FILTER-AUDIT]   n_c={len(J_n_rows)}  "
                       f"J_n.shape={J_n_mat.shape}")
                    with np.printoptions(precision=4, suppress=True,
                                         linewidth=200):
                        for i, row in enumerate(J_n_rows):
                            nonzero = int(np.count_nonzero(
                                np.abs(row) > 1e-9))
                            _w(f"[LCS-FILTER-AUDIT]   J_n[{i}] "
                               f"(nonzero_dofs={nonzero}/{len(row)}): "
                               f"{np.array(row)}")
                except Exception as e:  # noqa: BLE001
                    _w(f"[LCS-FILTER-AUDIT]   J_n vstack failed: {e}")

            _w(f"=== [LCS-FILTER-AUDIT] end ===\n")
        finally:
            f.close()

    # ------------------------------------------------------------------
    def linearize_discrete(self, context, *a, **kw):
        """Reference-conformant controller mass applied around the build.
        See _controller_inertia_scope."""
        with self._controller_inertia_scope(context):
            return self._linearize_discrete_impl(context, *a, **kw)

    def _linearize_discrete_impl(self, context, dt: float, u_lin=None):
        self._last_planner_dt = float(dt)
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
        # T-only: force top-K EE-manipuland admission for the R^7 planner LCS
        # (same gate that already exists in linearize_discrete_ee_space at
        # line 1405-1409). Bypasses the 2 mm distance threshold so the LCS
        # keeps an EE-T pair even during the arm's lift-traverse-descend,
        # preventing c3-chatter. Box path unchanged (gate requires tshape).
        if (self._object_shape in ("tshape", "hshape")
                and getattr(self, "_ref_pair_admission_planner_lcs", False)):
            phi, J_n, J_t, mu = self.extract_lcs_contacts(
                context, force_top_k_ee_box=True, n_ee_top_k=1)
        else:
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

        # Box-ground Coulomb drag (approximation). Subtract c·dt from the
        # diagonal of A's box translational-velocity block so v_box_xy decays
        # at v_dot = -c·v per step. The LCS already admits the BOX-GND pair
        # when φ_gnd ≤ 2 mm, but the ADMM componentwise projection cannot
        # sustain λ_n_gnd at the m·g level needed to apply μ·λ_n_gnd friction
        # over the prediction horizon; without this damping the predicted
        # box trajectory coasts at the post-impact velocity for the full
        # horizon (observed: -388 mm in 1 s vs executed ~-4 mm). Affects the
        # LCS prediction model only — the real Drake sim is untouched.
        if self._box_drag_c > 0.0 and self._obj_body is not None:
            try:
                v_start = self._obj_body.floating_velocities_start_in_v()
                # Free-floating body v layout: [ωx, ωy, ωz, vx, vy, vz]
                for k in (3, 4, 5):
                    idx = n_q + v_start + k
                    A[idx, idx] -= self._box_drag_c * dt
                if not getattr(self, "_drag_printed", False):
                    self._drag_printed = True
                    print(f"[LCS-DRAG] box translational v indices in v: "
                          f"[{v_start+3},{v_start+4},{v_start+5}]  "
                          f"c={self._box_drag_c:.2f}/s  "
                          f"per-step multiplier=(1 - c·dt)="
                          f"{1.0 - self._box_drag_c * dt:.4f}")
            except RuntimeError:
                # Body is not floating (welded); skip damping silently.
                pass

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
            # μ can be a scalar (uniform) OR an ndarray of shape (n_c,)
            # containing per-pair coefficients (ref
            # sampling_c3plus_options.yaml:44 mu_per_pair_type). Use
            # np.diag(mu) which handles both — scalar broadcasts to
            # μ·I, array uses each entry on the diagonal.
            F_lcs[SG:SG + n_c,  SLN:SLN + n_c]  = np.diag(
                np.broadcast_to(np.asarray(mu), (n_c,)))
            F_lcs[SG:SG + n_c,  SLT:SLT + n_t]  = -E_t

            # 2026-07-26 arc-2 A2 fix ported to R^7 full-plant path:
            # Reference lcs_factory.cc:465 (Stewart-Trinkle) / :533 (Anitescu)
            # includes `Jn · vNqdot / dt` in E's q-column (position-forcing
            # gradient of phi wrt q). Port R^7 previously had only `dt·(J_n@J_q)`
            # — missing the pure phi(q_{k+1}) gradient. Same bug that was
            # fixed for EE-space at A2 (commit 5e5ec10). Env-gated
            # REFCONF_E_BLOCK_SPLIT=1 (default ON = reference-conformant).
            _use_e_block_split_r7 = True   # 2026-07-28 defaults flip
            vNqdot_full = None
            if _use_e_block_split_r7:
                NqI = self.plant.MakeQDotToVelocityMap(context)  # sparse (n_v, n_q)
                vNqdot_full = np.asarray(NqI.todense())          # (n_v, n_q)
                if not getattr(self, "_e_block_split_r7_banner", False):
                    self._e_block_split_r7_banner = True
                    print(f"[E-BLOCK-SPLIT-R7] active: vNqdot_full shape="
                          f"{vNqdot_full.shape} (full-plant MakeQDotToVelocityMap)  "
                          f"adds Jn·vNqdot/dt to E[q] in R^7 LCS", flush=True)

            # λ_n rows (slot SLN : SLN+n_c).
            # We use the simpler gap discretization (matches Bui v1 and
            # avoids the floating-base n_q ≠ n_v mismatch from Aydinoglu's
            # explicit (1/dt)·J_n·(q − q*) term):
            #   η_n = phi(q*)/dt + J_n · v_{k+1}     (gap prediction)
            # Substitute v_{k+1} = v + dt·(J_q q + J_v v + J_u u + d_v + Minv·J_c^T λ_phys):
            E_lcs[SLN:SLN + n_c, :n_q]            = dt * (J_n @ J_q)
            E_lcs[SLN:SLN + n_c, n_q:n_q + n_v]   = J_n + dt * (J_n @ J_v)
            if _use_e_block_split_r7:
                # A2 fix: add phi-gradient wrt q. Reference lcs_factory.cc:465
                E_lcs[SLN:SLN + n_c, :n_q] += (J_n @ vNqdot_full) / dt
            F_lcs[SLN:SLN + n_c, SLN:SLN + n_c]   = dt * (J_n @ Minv_JnT)
            F_lcs[SLN:SLN + n_c, SLT:SLT + n_t]   = dt * (J_n @ Minv_JtT)
            H_lcs[SLN:SLN + n_c, :]               = dt * (J_n @ J_u)
            c_lcs[SLN:SLN + n_c]                  = phi / dt + dt * (J_n @ d_v_offset)

            # λ_t rows (slot SLT : SLT+4n_c).
            #   η_t = E_t^T·γ + J_t·v_{k+1}
            E_lcs[SLT:SLT + n_t, :n_q]            = dt * (J_t @ J_q)
            E_lcs[SLT:SLT + n_t, n_q:n_q + n_v]   = J_t + dt * (J_t @ J_v)
            if _use_e_block_split_r7:
                # A2 fix: same phi-gradient addition for tangent row block.
                E_lcs[SLT:SLT + n_t, :n_q] += (J_t @ vNqdot_full) / dt
            F_lcs[SLT:SLT + n_t, SG:SG + n_c]     = E_t.T
            F_lcs[SLT:SLT + n_t, SLN:SLN + n_c]   = dt * (J_t @ Minv_JnT)
            F_lcs[SLT:SLT + n_t, SLT:SLT + n_t]   = dt * (J_t @ Minv_JtT)
            H_lcs[SLT:SLT + n_t, :]               = dt * (J_t @ J_u)
            c_lcs[SLT:SLT + n_t]                  = dt * (J_t @ d_v_offset)

        # =================================================================
        # 2026-07-26 arc-2 R^7 Anitescu overwrite. Reference default for
        # push_t is Anitescu (`self._contact_model = "anitescu"` at :86).
        # Prior port only had Anitescu in EE-space; R^7 was ST-only.
        # Mirrors reference `c3/multibody/lcs_factory.cc:496-545
        # FormulateAnitescuContactDynamics`. Overwrites D/E/F/H/c with
        # friction-folded (J_c = E_tᵀ·Jn + diag(μ)·Jt) formulation:
        #   n_λ_an = 2·n_c·num_friction_directions = 4·n_c (dirs=2)
        #   D = [dt² · qdotNv · M⁻¹·J_cᵀ ; dt · M⁻¹·J_cᵀ]
        #   E = [dt·J_c·Jf_q + E_tᵀ·Jn·vNqdot/dt ; J_c + dt·J_c·Jf_v]
        #   F = dt·J_c·M⁻¹·J_cᵀ
        #   H = dt·J_c·J_u
        #   c = E_tᵀ·phi/dt + dt·J_c·d_v - E_tᵀ·Jn·vNqdot·q/dt
        # Default-active when _contact_model == "anitescu" (matches EE-space
        # behavior). ST fallback preserved when _contact_model == "stewart_trinkle".
        if self._contact_model == "anitescu" and n_c > 0:
            NUM_FRICTION_DIRECTIONS = 2   # 2·dirs = 4 tangent dirs / contact
            n_lam_an = 2 * n_c * NUM_FRICTION_DIRECTIONS    # = 4·n_c

            # E_t: (n_c, 4·n_c) — sums tangent-direction λ's per contact
            E_t_an = np.zeros((n_c, n_lam_an))
            for i in range(n_c):
                E_t_an[i, 4*i : 4*(i+1)] = 1.0

            # μ replicated 2·dirs per contact (scalar-or-array μ handling)
            _mu_arr = np.broadcast_to(np.asarray(mu), (n_c,))
            anitescu_mu_vec = np.zeros(n_lam_an)
            for i in range(n_c):
                anitescu_mu_vec[
                    2 * NUM_FRICTION_DIRECTIONS * i :
                    2 * NUM_FRICTION_DIRECTIONS * (i + 1)
                ] = float(_mu_arr[i])
            anitescu_mu_diag = np.diag(anitescu_mu_vec)

            # J_c = E_tᵀ·J_n + diag(μ)·J_t  (reference cc:522-523)
            J_c = E_t_an.T @ J_n + anitescu_mu_diag @ J_t     # (4n_c, n_v)
            Minv_JcT = M_inv @ J_c.T                          # (n_v, 4n_c)

            # D (reference cc:528-529)
            D_an = np.zeros((n_x, n_lam_an))
            D_an[:n_q, :] = (dt * dt) * (N_mat @ Minv_JcT)
            D_an[n_q:, :] = dt * Minv_JcT

            # E (reference cc:531-534, includes A2 position-forcing split
            # via the `E_tᵀ·Jn·vNqdot/dt` term; port needs the same when
            # REFCONF_E_BLOCK_SPLIT=1).
            E_an = np.zeros((n_lam_an, n_x))
            E_an[:, :n_q]            = dt * (J_c @ J_q)
            E_an[:, n_q:n_q + n_v]   = J_c + dt * (J_c @ J_v)
            if _use_e_block_split_r7 and vNqdot_full is not None:
                E_an[:, :n_q] += (E_t_an.T @ J_n @ vNqdot_full) / dt

            # F (reference cc:537) — single PSD block
            F_an = dt * (J_c @ Minv_JcT)

            # H (reference cc:540)
            H_an = dt * (J_c @ J_u)

            # c (reference cc:543-544)
            c_an = E_t_an.T @ phi / dt + dt * (J_c @ d_v_offset)
            if _use_e_block_split_r7 and vNqdot_full is not None:
                # Position-forcing subtraction with current-q value
                q_curr = self.plant.GetPositions(context)
                c_an -= (E_t_an.T @ J_n @ vNqdot_full @ q_curr) / dt

            # Overwrite the ST outputs with the Anitescu LCS.
            n_lam = n_lam_an
            D = D_an
            E_lcs = E_an
            F_lcs = F_an
            H_lcs = H_an
            c_lcs = c_an

            if not getattr(self, '_anitescu_r7_banner', False):
                self._anitescu_r7_banner = True
                print(f"[ANITESCU-R7] active: n_lam={n_lam_an} (=4·n_c) "
                      f"J_c shape={J_c.shape} F PSD block only "
                      f"(replaces Stewart-Trinkle 6·n_c LCS in R^7 path)",
                      flush=True)

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

        # Stash the most recent LCS matrices so external diagnostics (e.g.,
        # the wrapper's [MATH.LCS-DUMP] at the first rich-mode entry) can
        # read them without a second linearization. Purely additive.
        self._last_A, self._last_B, self._last_D, self._last_d = A, B_ctrl, D, d_vec
        self._last_E, self._last_F, self._last_H, self._last_c = E_lcs, F_lcs, H_lcs, c_lcs
        self._last_n_c = n_c
        self._last_J_n, self._last_J_t = J_n, J_t
        self._last_phi, self._last_mu  = phi, mu

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

    # ==================================================================
    # EE-space LCS (Push-Anything §IV-A reference architecture).
    # ==================================================================
    # State (n_x_new = 19):
    #     x = [ box_q (7=quat+pos), p_ee (3), box_v (6=omega+lin), v_ee (3) ]
    # Slot indices in x:
    BOX_Q_SLOT = slice(0,  7)    # length 7
    P_EE_SLOT  = slice(7, 10)    # length 3
    BOX_V_SLOT = slice(10,16)    # length 6
    V_EE_SLOT  = slice(16,19)    # length 3
    N_X_NEW    = 19
    N_U_NEW    = 3               # u = EE Cartesian force ∈ ℝ^3

    # EE point-mass for the planner. Configuration-INDEPENDENT by construction.
    # Numerical value chosen so B_ctrl entries are O(1) at dt=0.05: dt²/m_ee
    # ≈ 2.5e-3 for m_ee=1.0. Paper-typical effective EE mass; the downstream
    # OSC is what realizes the actual joint torque from u, so this is a
    # planner-internal scaling, not a physical claim.
    # 2026-07-22 qvector migration attempts (p41, p44-p46) all diverged.
    # Coordinated tuple (m_ee=0.057 + u=50 + qvec=True + w_Q=50) produces
    # ill-conditioned ADMM: p46 at admm_iter=25 showed dual residual
    # exploding 1000× per solve. Root cause is structural: port uses
    # scalar ρ ADMM augmentation, reference uses per-slot G matrix.
    # Without matching G, no combination of parameters yields convergent
    # ADMM under the reference qvector. Reverted to 1.0.
    # Reference literal: examples/sampling_c3/urdf/end_effector_simple_model
    # .urdf `<mass value="0.057"/>` — the EE body in the reference's LCS
    # plant (a free-floating point mass on 3 prismatic joints; the arm is
    # NOT in that plant). Was 1.0 kg = 17.5x too heavy; corrected 2026-08-08
    # together with the removal of the arm operational-space inertia path.
    _EE_MASS = 0.057   # kg

    @contextlib.contextmanager
    def _controller_inertia_scope(self, context):
        """Temporarily give the manipuland its CONTROLLER-model mass.

        Drake carries mass/inertia as context PARAMETERS, so the swap is
        per-context: the simulator's own context keeps the true sim mass and
        the physics is untouched. Applied to both the double and autodiff
        plants, since the LCS reads M/Cv/tau_g from one and the linearization
        Jacobian from the other. Restores on exit even if the build throws.
        """
        if self._controller_object_mass is None or self._obj_body is None:
            yield
            return
        pairs = []
        try:
            for pl, cx in ((self.plant, context),
                           (self.plant_ad, self.context_ad)):
                if pl is None or cx is None:
                    continue
                body = pl.GetBodyByName(self._obj_body.name())
                M0 = body.CalcSpatialInertiaInBodyFrame(cx)
                m0 = float(ad.ExtractValue(np.array([[M0.get_mass()]]))[0, 0]
                           if hasattr(M0.get_mass(), "value") else M0.get_mass())
                if m0 <= 0.0:
                    continue
                scale = self._controller_object_mass / m0
                # Scaling the SpatialInertia's mass while holding com and unit
                # inertia fixed scales the rotational inertia by the same
                # factor, which is what a denser copy of the same shape gives.
                Mnew = M0.__class__(M0.get_mass() * scale, M0.get_com(),
                                    M0.get_unit_inertia())
                body.SetSpatialInertiaInBodyFrame(cx, Mnew)
                pairs.append((body, cx, M0))
            yield
        finally:
            for body, cx, M0 in pairs:
                body.SetSpatialInertiaInBodyFrame(cx, M0)

    def linearize_discrete_ee_space(self, context, *a, **kw):
        """Reference-conformant controller mass applied around the build.
        See _controller_inertia_scope."""
        with self._controller_inertia_scope(context):
            return self._linearize_discrete_ee_space_impl(context, *a, **kw)

    def _linearize_discrete_ee_space_impl(self, context, dt: float, u_lin=None,
                                    n_ee_top_k: int = 1,
                                    force_top_k_ee_box: bool = False):
        self._last_planner_dt = float(dt)
        # d.1 — reference-conforming pair admission for the planner LCS.
        # When the caller didn't already opt in (force_top_k_ee_box=False,
        # the ci_mpc_c3plus.py planner default) AND the object is a tshape
        # AND the class-level flag is on, promote to force_top_k=True with
        # n_ee_top_k=1 (matches reference push_t planner's 1 EE-manipuland
        # pair). Bypasses the 2 mm auto-discovery, keeping the pair in the
        # LCS across the arm's off-face rise — the T-c3-chatter fix.
        # Cost-LCS caller (inner_solve.py:593) sets force_top_k=True
        # explicitly, so this override is a no-op there.
        # Gated to tshape so the box planner path is byte-identical when the
        # class flag is True.
        if (not force_top_k_ee_box
                and self._object_shape in ("tshape", "hshape")
                and getattr(self, "_ref_pair_admission_planner_lcs", False)):
            force_top_k_ee_box = True
            n_ee_top_k = 1
        """
        Paper-aligned low-dim LCS at (q*, v*, u*).

            x = [box_q, p_ee, box_v, v_ee]  ∈ ℝ^19
            u = F_ee (EE Cartesian force)   ∈ ℝ^3
            λ = [γ, λ_n, λ_t]                ∈ ℝ^(6 n_c)

        Continuous dynamics:
            d(box_q)/dt = N_box(box_q) · box_v
            d(p_ee)/dt  = v_ee
            M_box · d(box_v)/dt = -Cv_box + tau_g_box
                                 + J_n_box^T λ_n + J_t_box^T λ_t  + drag
            m_ee · d(v_ee)/dt   = u  +  J_n_ee^T λ_n + J_t_ee^T λ_t

        Where:
            M_box ∈ ℝ^{6×6}   sliced from CalcMassMatrix at box v indices.
                              For a free-floating box + fixed-base arm
                              (no kinematic coupling), this block is
                              INDEPENDENT of arm q. (verified Stage A.)
            J_n_box, J_t_box  box-velocity columns of Drake's contact J
                              (geometric, depends on box pose + contact
                              witness, NOT on arm q).
            J_n_ee = ±nhat    EE-position gradient of φ (just the unit
                              normal; depends on contact geometry, NOT
                              on arm Jacobian).
            m_ee = 1.0 kg     planner-internal EE point-mass.

        B_ctrl is CONFIGURATION-INDEPENDENT (only entries are dt²/m_ee · I
        and dt/m_ee · I on the EE rows). H_lcs depends on contact geometry
        (nhat, M_box) but NOT on arm. Arm Jacobian and arm M^{-1} do NOT
        appear anywhere in this construction. The downstream OSC remains
        responsible for mapping the planner's u (R^3 EE force) into joint
        torque — paper §IV-A reference architecture.

        Returns the same 12-tuple shape as linearize_discrete (R^7 path)
        so Stage B/C/D can plug it in with parallel sizing:
            A      : (19, 19)
            B_ctrl : (19,  3)
            D      : (19, 6 n_c)
            d      : (19,)
            E      : (6 n_c, 19)
            F      : (6 n_c, 6 n_c)   (identical to R^7 path)
            H      : (6 n_c, 3)
            c      : (6 n_c,)
            J_n    : (n_c, 9)   in new velocity coords [box_v(6), v_ee(3)]
            J_t    : (4n_c, 9)
            phi    : (n_c,)
            mu     : float
        """
        if u_lin is None:
            u_lin = np.zeros(self.N_U_NEW)
        else:
            u_lin = np.asarray(u_lin, dtype=float).reshape(self.N_U_NEW)

        # Plant index layout (verified at runtime; cached on first call).
        # Box is the free-floating manipuland; arm is the fixed-base Franka.
        # Drake puts the arm q first (q[0:7]), then box q (q[7:14]).
        # In v: arm v first (v[0:7]), then box v (v[7:13]).
        if self._obj_body is None:
            raise RuntimeError(
                "linearize_discrete_ee_space requires obj_body to be set "
                "so we can locate the box's floating velocity slot."
            )
        BOX_Q_START = self._obj_body.floating_positions_start()        # 7
        BOX_V_START = self._obj_body.floating_velocities_start_in_v()  # 7
        BOX_N_Q     = 7   # quat (4) + pos (3)
        BOX_N_V     = 6   # omega (3) + lin (3)
        N_X = self.N_X_NEW
        N_U = self.N_U_NEW

        # -----------------------------------------------------------------
        # 1. Read current state (for the linearization point).
        # -----------------------------------------------------------------
        q_full = self.plant.GetPositions(context)
        v_full = self.plant.GetVelocities(context)
        box_q  = q_full[BOX_Q_START : BOX_Q_START + BOX_N_Q]
        box_v  = v_full[BOX_V_START : BOX_V_START + BOX_N_V]
        # EE position from forward kinematics on the arm. NOTE: the arm
        # Jacobian / FK is used HERE only to read the current EE position
        # (a state-space coordinate). It does NOT enter B_ctrl, H_lcs, or
        # the LCS dynamics. After Stage D, the planner's solved x_seq will
        # carry p_ee directly; the OSC will track that.
        ee_body  = self.plant.GetBodyByName(_EE_BODY_NAME)
        ee_frame = ee_body.body_frame()
        W        = self.plant.world_frame()
        p_ee = self.plant.CalcPointsPositions(
            context, ee_frame, np.zeros((3, 1)), W
        ).flatten()
        # EE velocity from arm: J_arm · v_arm. Same caveat — used only to
        # set the linearization point. Not folded into B/H.
        J_ee_full = self.plant.CalcJacobianTranslationalVelocity(
            context, ad.JacobianWrtVariable.kV,
            ee_frame, np.zeros(3), W, W,
        )  # (3, n_v_full)
        v_ee = J_ee_full @ v_full

        x_star = np.zeros(N_X)
        x_star[self.BOX_Q_SLOT] = box_q
        x_star[self.P_EE_SLOT]  = p_ee
        x_star[self.BOX_V_SLOT] = box_v
        x_star[self.V_EE_SLOT]  = v_ee
        u_star = u_lin

        # -----------------------------------------------------------------
        # 2. Extract box-only dynamics by SLICING the full plant. For a
        #    fixed-base arm + free-floating box (independent kinematic
        #    trees), M is block-diagonal between arm and box, so the box
        #    slice is INDEPENDENT of arm q. Verified by Stage A test.
        # -----------------------------------------------------------------
        with timed("lcs.extract_dynamics"):
            M_full     = self.plant.CalcMassMatrixViaInverseDynamics(context)
            Cv_full    = self.plant.CalcBiasTerm(context)
            tau_g_full = self.plant.CalcGravityGeneralizedForces(context)
        BS = BOX_V_START
        BE = BS + BOX_N_V
        M_box     = M_full[BS:BE, BS:BE]                  # (6, 6)
        Cv_box    = Cv_full[BS:BE]                        # (6,)
        tau_g_box = tau_g_full[BS:BE]                     # (6,)
        M_box_inv = np.linalg.inv(M_box)

        # EE inertia in the LCS = the REFERENCE's free-floating point mass.
        # 2026-08-08: the arc-2 "A1 fix" (arm Cartesian operational-space
        # inverse inertia, M_ee_op_inv = J_arm M_arm^-1 J_arm^T) is REMOVED —
        # it was a port-only addition, not a conformance fix. The reference
        # LCS plant contains NO ARM: `AddLCSModelsToPlant`
        # (sampling_c3_utils.cc:175-190, called from
        # franka_sampling_c3_controller.cc:99-108) loads exactly
        # end_effector_simple_model.urdf + ground + object models, and that
        # URDF is a point mass on 3 prismatic joints with mass 0.057 kg.
        # Measured divergence at the canonical seed pose: the arm operational
        # -space effective EE mass eigenvalues were 1.79-21.14 kg (mean ~8.7)
        # = 152x the reference; the prior isotropic _EE_MASS=1.0 was 17.5x.
        # Consequence of the too-heavy EE: with u bounded at +/-50 N over
        # N=5 x 0.1 s the planner could barely translate the EE, so it bought
        # object progress with phantom lambda instead (object cost weight 200
        # vs EE 0.01) -> plans with no approach phase. The same term enters
        # F = dt*(J_box M_box^-1 J_box^T + J_ee M_ee_op_inv J_ee^T), where a
        # 152x-small EE contribution suppressed eta's response to lambda and
        # pinned the eq-12 projection in case-2 (lambda wins) = the phantom
        # endorsement. Isotropic by construction: the reference's 3 prismatic
        # joints give an isotropic 0.057 kg point mass with no arm coupling.
        _m_ee_iso = float(self._EE_MASS)
        M_ee_op_inv = (1.0 / _m_ee_iso) * np.eye(3)        # (3, 3) isotropic
        if not self._arm_cart_inertia_banner_done:
            self._arm_cart_inertia_banner_done = True
            print(f"[LCS-EE-MASS] reference point-mass EE: m_ee="
                  f"{_m_ee_iso:.4f} kg isotropic "
                  f"(end_effector_simple_model.urdf; arm operational-space "
                  f"inertia removed 2026-08-08)", flush=True)

        # Box's N(q) sub-block: q_dot_box = N_box · box_v.
        N_box = np.zeros((BOX_N_Q, BOX_N_V))
        for i in range(BOX_N_V):
            e_full = np.zeros(self.n_v)
            e_full[BS + i] = 1.0
            qdot_full = self.plant.MapVelocityToQDot(context, e_full)
            N_box[:, i] = qdot_full[BOX_Q_START : BOX_Q_START + BOX_N_Q]

        # 2026-07-26 arc-2 A2 fix: box's N⁺(q) sub-block (qdot→v map, reverse
        # of N_box). Reference `lcs_factory.cc:387` builds full-plant vNqdot
        # via `plant_.MakeQDotToVelocityMap()`. Used below to add the missing
        # position-forcing gradient `E_tᵀ·Jn·vNqdot/dt` to E's q-column
        # (`lcs_factory.cc:533` in Anitescu, `:465` in Stewart-Trinkle).
        # Env-gate REFCONF_E_BLOCK_SPLIT=1 (default ON = reference-conformant).
        # G-off calibration was previously validated without this gradient,
        # so the OFF path preserves p73 arc-1 baseline byte-identical.
        self._use_e_block_split = True   # 2026-07-28 defaults flip
        if self._use_e_block_split:
            # For a floating base with quaternion, N⁺ (v_box = N⁺·qdot_box)
            # is a 6×7 matrix. Extract via Drake API.
            NqI = self.plant.MakeQDotToVelocityMap(context)   # (n_v, n_q) sparse
            vNqdot_full = np.asarray(NqI.todense())
            vNqdot_box  = vNqdot_full[BS:BE,
                                       BOX_Q_START:BOX_Q_START + BOX_N_Q]  # (6, 7)
            if not getattr(self, "_e_block_split_banner", False):
                self._e_block_split_banner = True
                print(f"[E-BLOCK-SPLIT] active: vNqdot_box shape={vNqdot_box.shape} "
                      f"(box-slice of MakeQDotToVelocityMap)  "
                      f"adds Jn·vNqdot/dt to E[q_box], Jn_ee/dt to E[p_ee]",
                      flush=True)
        else:
            vNqdot_box = None

        # Box accel at linearization point (continuous): f_box = M_box^-1
        # (-Cv_box + tau_g_box).  λ and u contributions are added through
        # the LCS structure (D and H), so f_box here is the AUTONOMOUS part.
        f_box = M_box_inv @ (-Cv_box + tau_g_box)

        # EE accel at linearization point (continuous): f_ee = M_ee_op_inv · u + λ-term.
        # A1 fix: M_ee_op_inv replaces scalar 1/m_ee (isotropic when flag off,
        # arm-Cartesian operational-space when flag on). Legacy m_ee scalar
        # retained for the [LCS-COND-WARN] diagnostic and comments only.
        m_ee = float(self._EE_MASS)
        f_ee = M_ee_op_inv @ u_star   # autonomous (no λ) part

        # -----------------------------------------------------------------
        # 3. Contacts: phi, Drake's J_n (n_c, n_v_full), nhat list. We then
        #    PROJECT to the new low-dim velocity space [box_v(6), v_ee(3)].
        # -----------------------------------------------------------------
        phi, J_n_drake, J_t_drake, mu = self.extract_lcs_contacts(
            context, n_ee_top_k=n_ee_top_k,
            force_top_k_ee_box=force_top_k_ee_box)
        n_c = J_n_drake.shape[0]
        n_t = J_t_drake.shape[0]               # 4·n_c
        n_lam = 2 * n_c + n_t                  # 6·n_c — [γ; λ_n; λ_t]
        SG  = 0
        SLN = n_c
        SLT = 2 * n_c

        # Project J_n, J_t to (n_c, 9) and (4n_c, 9).
        # New velocity coords: u_vel = [box_v (6), v_ee (3)].
        # Box columns: take Drake J_n[:, box_v_start : box_v_start + 6].
        # EE columns:
        #   - EE-BOX contact: J_n_ee = -nhat_onto_box (3-vec).
        #     Sign reasoning: nhat_onto_box points INTO box (from EE side);
        #     when EE moves OUTWARD (-nhat_onto_box direction), φ grows,
        #     so dφ/dv_ee = -nhat_onto_box.
        #   - BOX-GND  contact: EE not involved → J_n_ee = 0.
        # J_t_ee analogous: t1, -t1, t2, -t2 in EE coords for EE-BOX pairs,
        #   zeros for BOX-GND pairs.
        J_n_new = np.zeros((n_c, BOX_N_V + 3))
        J_t_new = np.zeros((n_t, BOX_N_V + 3))
        if n_c > 0:
            J_n_new[:, :BOX_N_V] = J_n_drake[:, BS:BE]
            J_t_new[:, :BOX_N_V] = J_t_drake[:, BS:BE]
            for i, info in enumerate(self._last_contact_info):
                tag = info["tag"]
                nhat_onto_box = info["nhat_onto_box"]  # into box
                if tag == "EE-BOX":
                    nhat_box_to_ee = -np.asarray(nhat_onto_box, dtype=float)
                    J_n_new[i, BOX_N_V:BOX_N_V + 3] = nhat_box_to_ee
                    # Tangent basis (same as extract_lcs_contacts):
                    nhat = info["nhat_BA_W"]
                    ref = np.array([1.0, 0.0, 0.0])
                    if abs(float(np.dot(nhat, ref))) > 0.99:
                        ref = np.array([0.0, 1.0, 0.0])
                    t1 = np.cross(nhat, ref); t1 /= np.linalg.norm(t1)
                    t2 = np.cross(nhat, t1)
                    # EE-contribution to dφ_t/dt = (±t) · v_ee.
                    # Sign: same as J_n_ee — when EE moves in +t direction,
                    # tangential slip grows in that t direction.
                    a_is_box = info["a_is_box"]
                    sign_ee = +1.0 if (not a_is_box) else -1.0
                    for d, drow in zip((t1, -t1, t2, -t2),
                                       (4*i, 4*i+1, 4*i+2, 4*i+3)):
                        J_t_new[drow, BOX_N_V:BOX_N_V + 3] = sign_ee * d

        # Convenience aliases for box and EE columns in J_n_new / J_t_new.
        J_n_box = J_n_new[:, :BOX_N_V]            # (n_c, 6)
        J_n_ee  = J_n_new[:, BOX_N_V:]            # (n_c, 3)
        J_t_box = J_t_new[:, :BOX_N_V]            # (4n_c, 6)
        J_t_ee  = J_t_new[:, BOX_N_V:]            # (4n_c, 3)

        # -----------------------------------------------------------------
        # 4. Build A, B_ctrl, D, d (state-step matrices).
        #
        # State stride (Euler step):
        #   box_q_{k+1} = box_q + dt · N_box · box_v_{k+1}
        #               = box_q + dt · N_box · (box_v + dt · f_box_full)
        #     where f_box_full = f_box + M_box^-1 · J_n_box^T λ_n
        #                              + M_box^-1 · J_t_box^T λ_t
        #   p_ee_{k+1} = p_ee  + dt · v_ee_{k+1}
        #              = p_ee  + dt · (v_ee + dt · f_ee_full)
        #     where f_ee_full = u/m_ee + J_n_ee^T λ_n / m_ee
        #                             + J_t_ee^T λ_t / m_ee
        #   box_v_{k+1} = box_v + dt · f_box_full
        #   v_ee_{k+1}  = v_ee  + dt · f_ee_full
        #
        # Linearization (∂/∂x and ∂/∂u). For B_ctrl / D / H_lcs we
        # short-circuit ∂f/∂q ≈ 0 for the box dynamics state-coupling — the
        # exact linearization of f_box wrt box_q is computed below via the
        # full-plant autodiff, restricted to box-velocity rows and box-q
        # columns (still arm-independent because of block diagonality).
        # -----------------------------------------------------------------

        # 4a. Exact linearization of box autonomous dynamics ∂f_box/∂(box_q,
        # box_v) via Drake autodiff. f_box = M^-1 (−Cv + tau_g) is computed
        # on the full plant; we read out only the box rows and box columns.
        with timed("lcs.extract_dynamics"):
            q_star_full = q_full.copy()
            v_star_full = v_full.copy()
            decvar = np.concatenate([q_star_full, v_star_full])
            decvar_ad = InitializeAutoDiff(decvar)
            decvar_ad = (decvar_ad.flatten()
                         if decvar_ad.ndim > 1 else decvar_ad)
            q_ad = decvar_ad[:self.n_q]
            v_ad = decvar_ad[self.n_q:self.n_q + self.n_v]
            self.plant_ad.SetPositions(self.context_ad, q_ad)
            self.plant_ad.SetVelocities(self.context_ad, v_ad)
            M_full_ad     = self.plant_ad.CalcMassMatrixViaInverseDynamics(
                self.context_ad)
            Cv_full_ad    = self.plant_ad.CalcBiasTerm(self.context_ad)
            tau_g_full_ad = self.plant_ad.CalcGravityGeneralizedForces(
                self.context_ad)
            rhs_box_ad    = (- Cv_full_ad[BS:BE] + tau_g_full_ad[BS:BE])
        # Extract box-block of M and gradient.
        # df_box/d(q, v) = M_box^-1 [ d(rhs_box)/d(q,v) - (dM_box/d(q,v)) f_box ]
        J_rhs_box_full = ExtractGradient(rhs_box_ad)     # (6, n_q + n_v)
        J_M_full = ExtractGradient(M_full_ad).reshape(
            self.n_v, self.n_v, self.n_q + self.n_v)
        # Slice box-block: M_box is M_full[BS:BE, BS:BE]; its gradient wrt
        # each (q,v) variable is J_M_full[BS:BE, BS:BE, k].
        df_box_dxfull = np.zeros((BOX_N_V, self.n_q + self.n_v))
        for k in range(self.n_q + self.n_v):
            dM_box_k = J_M_full[BS:BE, BS:BE, k]
            df_box_dxfull[:, k] = M_box_inv @ (
                J_rhs_box_full[:, k] - dM_box_k @ f_box
            )
        # Project to new state coords. Only the box-q columns of df_box/dq
        # and box-v columns of df_box/dv are non-trivial (in principle the
        # arm columns could be nonzero from CalcMassMatrix's full-plant
        # numerics, but for an independent kinematic tree they're zero
        # within autodiff noise; Stage A verifies this empirically).
        df_box_dboxq = df_box_dxfull[:, BOX_Q_START : BOX_Q_START + BOX_N_Q]
        df_box_dboxv = df_box_dxfull[:, self.n_q + BS : self.n_q + BE]

        # 4b. Assemble A.
        A = np.zeros((N_X, N_X))
        # box_q rows.
        # A[box_q, box_q] = I + dt² · N_box · df_box_dboxq
        A[self.BOX_Q_SLOT, self.BOX_Q_SLOT] = (
            np.eye(BOX_N_Q) + (dt * dt) * (N_box @ df_box_dboxq)
        )
        # A[box_q, box_v] = dt · N_box · (I + dt · df_box_dboxv)
        A[self.BOX_Q_SLOT, self.BOX_V_SLOT] = (
            dt * N_box @ (np.eye(BOX_N_V) + dt * df_box_dboxv)
        )
        # p_ee rows.
        # A[p_ee, p_ee] = I; A[p_ee, v_ee] = dt · I (since v_ee_{k+1} = v_ee + ...).
        A[self.P_EE_SLOT, self.P_EE_SLOT] = np.eye(3)
        A[self.P_EE_SLOT, self.V_EE_SLOT] = dt * np.eye(3)
        # box_v rows.
        A[self.BOX_V_SLOT, self.BOX_Q_SLOT] = dt * df_box_dboxq
        A[self.BOX_V_SLOT, self.BOX_V_SLOT] = np.eye(BOX_N_V) + dt * df_box_dboxv
        # v_ee rows.
        A[self.V_EE_SLOT, self.V_EE_SLOT] = np.eye(3)  # no damping

        # 4c. Box-ground drag — mirrors R^7 path (lcs_formulator.py:778-794).
        if self._box_drag_c > 0.0:
            # The box's translational velocity is at box_v slots 3..5
            # (after omega at 0..2). In the NEW state, those are at
            # BOX_V_SLOT.start + 3 .. BOX_V_SLOT.start + 5.
            base = self.BOX_V_SLOT.start
            for k in (3, 4, 5):
                A[base + k, base + k] -= self._box_drag_c * dt

        # 4d. B_ctrl — arm-config-DEPENDENT under REFCONF_ARM_CART_INERTIA=1;
        # config-independent otherwise (isotropic 1/m_ee · I fallback).
        B_ctrl = np.zeros((N_X, N_U))
        # p_ee rows: ∂p_ee_{k+1}/∂u = dt · ∂v_ee_{k+1}/∂u = dt² · M_ee_op_inv.
        B_ctrl[self.P_EE_SLOT, :] = (dt * dt) * M_ee_op_inv
        # v_ee rows: ∂v_ee_{k+1}/∂u = dt · M_ee_op_inv.
        B_ctrl[self.V_EE_SLOT, :] = dt * M_ee_op_inv
        # box_q rows and box_v rows: zero (u doesn't enter box dynamics).

        # Bug 2 safety: LCS conditioning warning (2026-07-22). Under the
        # matrix formulation, watch max diagonal of dt · M_ee_op_inv (the
        # V_EE_SLOT row scale in B_ctrl). Same threshold semantics as the
        # legacy dt/m_ee check; scalar-mode falls back to dt/m_ee exactly.
        _dt_M_ee_diag_max = float(np.max(dt * np.abs(M_ee_op_inv.diagonal())))
        if (_dt_M_ee_diag_max > 0.5
                and not getattr(self, "_dt_mee_warned", False)):
            print(
                f"[LCS-COND-WARN] dt·max(diag M_ee_op_inv) = {_dt_M_ee_diag_max:.3f} "
                f"(dt={dt:.3f}s, m_ee_scalar={m_ee:.4f}kg, arm_cart={self._use_arm_cartesian_inertia}) "
                f"— B_ctrl[V_EE] and D[V_EE,λ] rows amplified; planner may "
                f"disengage due to velocity-cost inflation. See p41 forensic report.",
                flush=True,
            )
            self._dt_mee_warned = True

        # 4e. D — contact force coupling on state.
        # D has zero columns in γ slot.
        # For λ_n: ∂(box_v_{k+1})/∂λ_n = dt · M_box^-1 · J_n_box^T  (6, n_c)
        #         ∂(v_ee_{k+1})/∂λ_n  = dt · J_n_ee^T / m_ee         (3, n_c)
        #         ∂(box_q_{k+1})/∂λ_n = dt² · N_box · M_box^-1 · J_n_box^T
        #         ∂(p_ee_{k+1})/∂λ_n  = dt² · J_n_ee^T / m_ee
        # Same for λ_t.
        D = np.zeros((N_X, n_lam))
        if n_c > 0:
            Minv_JnT_box = M_box_inv @ J_n_box.T               # (6, n_c)
            Minv_JtT_box = M_box_inv @ J_t_box.T               # (6, 4n_c)
            # box_q rows
            D[self.BOX_Q_SLOT, SLN:SLN + n_c]   = (dt*dt) * (N_box @ Minv_JnT_box)
            D[self.BOX_Q_SLOT, SLT:SLT + n_t]   = (dt*dt) * (N_box @ Minv_JtT_box)
            # p_ee rows: dt² · M_ee_op_inv · J_n_ee^T (and J_t_ee^T)
            D[self.P_EE_SLOT, SLN:SLN + n_c]    = (dt*dt) * (M_ee_op_inv @ J_n_ee.T)
            D[self.P_EE_SLOT, SLT:SLT + n_t]    = (dt*dt) * (M_ee_op_inv @ J_t_ee.T)
            # box_v rows
            D[self.BOX_V_SLOT, SLN:SLN + n_c]   = dt * Minv_JnT_box
            D[self.BOX_V_SLOT, SLT:SLT + n_t]   = dt * Minv_JtT_box
            # v_ee rows
            D[self.V_EE_SLOT, SLN:SLN + n_c]    = dt * (M_ee_op_inv @ J_n_ee.T)
            D[self.V_EE_SLOT, SLT:SLT + n_t]    = dt * (M_ee_op_inv @ J_t_ee.T)

        # 4f. d_vec — affine offset (constant term in x_{k+1} after linearization).
        # d_box_v_offset = f_box(box_q*, box_v*) − df_box_dboxq · box_q*
        #                  − df_box_dboxv · box_v*
        # d_v_ee_offset  = f_ee(u*) − (∂f_ee/∂u) · u*  = u*/m_ee − u*/m_ee = 0
        d_box_v_offset = f_box - df_box_dboxq @ box_q - df_box_dboxv @ box_v
        d_v_ee_offset  = np.zeros(3)   # purely linear in u (no constant offset)
        d_vec = np.zeros(N_X)
        d_vec[self.BOX_Q_SLOT] = (dt * dt) * (N_box @ d_box_v_offset)
        d_vec[self.P_EE_SLOT]  = (dt * dt) * d_v_ee_offset    # zero
        d_vec[self.BOX_V_SLOT] = dt * d_box_v_offset
        d_vec[self.V_EE_SLOT]  = dt * d_v_ee_offset           # zero

        # -----------------------------------------------------------------
        # 5. Stewart-Trinkle LCP slack:
        #      η = E·x + F·λ + H·u + c,    0 ≤ λ ⊥ η ≥ 0
        # -----------------------------------------------------------------
        E_lcs = np.zeros((n_lam, N_X))
        F_lcs = np.zeros((n_lam, n_lam))
        H_lcs = np.zeros((n_lam, N_U))
        c_lcs = np.zeros(n_lam)

        if n_c > 0:
            E_t = np.zeros((n_c, n_t))
            for i in range(n_c):
                E_t[i, 4 * i : 4 * (i + 1)] = 1.0

            # γ rows (state-independent, identical to R^7 path).
            # μ can be a scalar (uniform) OR an ndarray of shape (n_c,)
            # containing per-pair coefficients (ref
            # sampling_c3plus_options.yaml:44 mu_per_pair_type). Use
            # np.diag(mu) which handles both — scalar broadcasts to
            # μ·I, array uses each entry on the diagonal.
            F_lcs[SG:SG + n_c,  SLN:SLN + n_c]  = np.diag(
                np.broadcast_to(np.asarray(mu), (n_c,)))
            F_lcs[SG:SG + n_c,  SLT:SLT + n_t]  = -E_t

            # λ_n rows: η_n = φ/dt + J_n · v_{k+1}
            # In new coords J_n · v = J_n_box · box_v + J_n_ee · v_ee.
            # v_{k+1} = current_v + dt · (autonomous f + λ-coupling + u-coupling).
            # The E-row picks up the v_{k+1}-vs-x dependence:
            #   ∂(J_n_box · box_v_{k+1}) / ∂box_q = dt · J_n_box · df_box_dboxq
            #   ∂(J_n_box · box_v_{k+1}) / ∂box_v = J_n_box + dt · J_n_box · df_box_dboxv
            #   ∂(J_n_ee  · v_ee_{k+1})  / ∂v_ee  = J_n_ee
            #   ∂(J_n_ee  · v_ee_{k+1})  / ∂u     = dt · J_n_ee / m_ee
            E_lcs[SLN:SLN + n_c, self.BOX_Q_SLOT] = dt * (J_n_box @ df_box_dboxq)
            E_lcs[SLN:SLN + n_c, self.BOX_V_SLOT] = J_n_box + dt * (J_n_box @ df_box_dboxv)
            E_lcs[SLN:SLN + n_c, self.V_EE_SLOT]  = J_n_ee
            # 2026-07-26 arc-2 A2 fix: position-forcing gradient of phi wrt q.
            # Reference `lcs_factory.cc:465` adds `Jn·vNqdot` to E's q-column.
            # Port previously baked position-forcing into c via J_c·v (velocity
            # term) but missed the pure phi(q_{k+1}) gradient. Adds:
            #   - Box q slot: Jn_box · vNqdot_box / dt   (Reference symbol split)
            #   - P_ee slot : Jn_ee / dt                  (EE Cartesian, N⁺ = I)
            # c's `-= E·x* + H·u*` at end auto-adjusts the constant.
            if self._use_e_block_split:
                E_lcs[SLN:SLN + n_c, self.BOX_Q_SLOT] += (J_n_box @ vNqdot_box) / dt
                E_lcs[SLN:SLN + n_c, self.P_EE_SLOT]  += J_n_ee / dt
            # F: λ-coupling — ∂v_{k+1}/∂λ via D-style entries.
            F_lcs[SLN:SLN + n_c, SLN:SLN + n_c] = (
                dt * (J_n_box @ Minv_JnT_box)
                + dt * (J_n_ee @ M_ee_op_inv @ J_n_ee.T)
            )
            F_lcs[SLN:SLN + n_c, SLT:SLT + n_t] = (
                dt * (J_n_box @ Minv_JtT_box)
                + dt * (J_n_ee @ M_ee_op_inv @ J_t_ee.T)
            )

            # §7.24 Candidate A — soft-LCP compliance on EE-BOX-only normal
            # contacts. Adds k·1 to the F-diagonal of each EE-BOX λ_n slot:
            # 0 ≤ λ_n ⊥ (η_n_rigid + k·λ_n) ≥ 0. Bounds λ_n at deep depth
            # by allowing partial penetration to persist. NOT applied to
            # BOX-VERT (floor) contacts — softening those re-introduces floor
            # penetration and risks re-breaking the §7.9-§7.12 vertical fix.
            # Default k = 0.0 (OFF, byte-identical pre-§7.24 behaviour).
            if self._normal_compliance_k > 0.0:
                for i_c, info in enumerate(self._last_contact_info[:n_c]):
                    if info.get('tag', '') == 'EE-BOX':
                        F_lcs[SLN + i_c, SLN + i_c] += self._normal_compliance_k
            # H: u-coupling — only the v_ee path contributes, M_ee_op_inv scaling.
            H_lcs[SLN:SLN + n_c, :] = dt * (J_n_ee @ M_ee_op_inv)
            # c: const offset: φ/dt + J_n · (current_v + dt · d_offset)
            #    − E_lcs · x* − H_lcs · u*   (linearization residual)
            c_const_v_box  = J_n_box @ (box_v + dt * d_box_v_offset)
            c_const_v_ee   = J_n_ee  @ (v_ee  + dt * d_v_ee_offset)
            c_lcs[SLN:SLN + n_c] = phi / dt + c_const_v_box + c_const_v_ee
            # DIAG_ZVEE build-time row decomposition (2026-08-11): what is
            # ACTUALLY in each lambda_n row vs the contact metadata.
            import os as _os_rd
            if _os_rd.environ.get("DIAG_ZVEE", ""):
                _tags = [i.get('tag', '?')
                         for i in self._last_contact_info[:n_c]]
                print(f"[ROWBUILD] p_ee_star={np.array2string(p_ee, precision=4)} "
                      f"tags={_tags} "
                      f"phi/dt={np.array2string(phi/dt, precision=3)} "
                      f"cv_box={np.array2string(c_const_v_box, precision=3)} "
                      f"cv_ee={np.array2string(c_const_v_ee, precision=3)} "
                      f"Jnb_z={np.array2string(J_n_box[:, 5], precision=3)} "
                      f"vrow={np.array2string(phi/dt + c_const_v_box + c_const_v_ee, precision=3)}",
                      flush=True)
            # NOTE: c_lcs absorbs the "constant" of η linearized at (x*, u*);
            # E and H carry the gradient parts. We subtract E·x* + H·u* below.

            # §7.27 Candidate E — clamped-φ/dt saturating-stiffness on EE-BOX
            # only. Replace phi[i_c]/dt with max(phi[i_c]/dt, -v_cap) — i.e.
            # cap |phi|/dt at v_cap. Shallow contacts (|phi|/dt ≤ v_cap) are
            # unchanged (rigid); deep contacts get a softened drive so the
            # LCP no longer demands an unbounded next-step separation
            # velocity. depth-ASYMMETRIC by construction (rigid below cap,
            # saturated above) — NOT a β-scaling. BOX-VERT/floor contacts
            # untouched (preserves the §7.9-§7.12 vertical fix). Default
            # OFF (byte-identical pre-§7.27 behaviour).
            if self._normal_phi_clamp_v_cap is not None:
                v_cap = self._normal_phi_clamp_v_cap
                for i_c, info in enumerate(self._last_contact_info[:n_c]):
                    if info.get('tag', '') == 'EE-BOX':
                        phi_over_dt = phi[i_c] / dt
                        clamped = max(phi_over_dt, -v_cap)
                        c_lcs[SLN + i_c] += (clamped - phi_over_dt)

            # §7.26 Candidate C — velocity-level normal on EE-BOX only:
            # subtract phi/dt for each EE-BOX-tagged contact (drop the
            # position-forcing term, keep the velocity contributions). This
            # is the Anitescu-Potra velocity-level normal formulation with
            # v_target = 0; the rigid Stewart-Trinkle behaviour is restored
            # when the flag is OFF (default). BOX-VERT/floor contacts keep
            # their phi/dt (altering them re-breaks the §7.9-§7.12 vertical
            # fix — see §7.24 (5)).
            if self._normal_velocity_level:
                for i_c, info in enumerate(self._last_contact_info[:n_c]):
                    if info.get('tag', '') == 'EE-BOX':
                        c_lcs[SLN + i_c] -= phi[i_c] / dt

            # λ_t rows analogously.
            E_lcs[SLT:SLT + n_t, self.BOX_Q_SLOT] = dt * (J_t_box @ df_box_dboxq)
            E_lcs[SLT:SLT + n_t, self.BOX_V_SLOT] = J_t_box + dt * (J_t_box @ df_box_dboxv)
            E_lcs[SLT:SLT + n_t, self.V_EE_SLOT]  = J_t_ee
            # A2 fix: same phi-gradient addition for tangent row block.
            if self._use_e_block_split:
                E_lcs[SLT:SLT + n_t, self.BOX_Q_SLOT] += (J_t_box @ vNqdot_box) / dt
                E_lcs[SLT:SLT + n_t, self.P_EE_SLOT]  += J_t_ee / dt
            F_lcs[SLT:SLT + n_t, SG:SG + n_c]     = E_t.T
            F_lcs[SLT:SLT + n_t, SLN:SLN + n_c]   = (
                dt * (J_t_box @ Minv_JnT_box)
                + dt * (J_t_ee @ M_ee_op_inv @ J_n_ee.T)
            )
            F_lcs[SLT:SLT + n_t, SLT:SLT + n_t]   = (
                dt * (J_t_box @ Minv_JtT_box)
                + dt * (J_t_ee @ M_ee_op_inv @ J_t_ee.T)
            )
            H_lcs[SLT:SLT + n_t, :] = dt * (J_t_ee @ M_ee_op_inv)
            c_const_v_box_t = J_t_box @ (box_v + dt * d_box_v_offset)
            c_const_v_ee_t  = J_t_ee  @ (v_ee  + dt * d_v_ee_offset)
            c_lcs[SLT:SLT + n_t] = c_const_v_box_t + c_const_v_ee_t

            # Subtract E·x* so c carries η's affine offset. The value
            # expression above is evaluated at u = 0 (d_v_ee_offset = 0 —
            # "purely linear in u"), so H·u* must NOT be subtracted: doing so
            # shifted every EE-coupled row by −H·u* whenever the full solve
            # linearized at u_lin = _last_u ≠ 0 (p146 walk root cause; ground
            # rows have H ≈ 0 and surrogates pass u_lin = 0, which is why
            # only the committed plan's EE-BOX gap row was corrupted).
            c_lcs -= E_lcs @ x_star

        # =================================================================
        # §7.36 ANITESCU FRICTION-FOLDED LCS — opt-in OVERWRITE of D/E/F/H/c
        # ----------------------------------------------------------------
        # Default OFF (LCS_CONTACT_MODEL unset → "stewart_trinkle" → byte-
        # identical pre-§7.36 path). When set to "anitescu", REPLACES the
        # ST [γ, λ_n, λ_t] (6·n_c slots) LCS with the friction-folded
        # Anitescu LCS (4·n_c slots) at the same linearization point.
        #
        # Mirrors lcs_factory.cc:235-275 (kAnitescu branch):
        #   J_c   = E_t^T · J_n + diag(μ) · J_t            (n_λ, n_v_lowdim)
        #   n_λ   = 2 · n_c · num_friction_directions = 4·n_c (dirs=2)
        #   D, E, H follow with J_c folded in
        #   F = dt · J_c · M^{-1} · J_c^T  (single PSD block — no γ-γ rank
        #     deficiency, the structural difference vs ST's 3-block F)
        #
        # The §7.24 / §7.26 / §7.27 patches are STEWART-TRINKLE specific —
        # they touch the λ_n row, which does NOT exist as a separate row
        # under Anitescu. They are NO-OPs in Anitescu mode by construction;
        # their flags stay banked as ST-diagnostics, orthogonal to this one.
        # =================================================================
        if self._contact_model == "anitescu":
            NUM_FRICTION_DIRECTIONS = 2   # 2·dirs = 4 tangent dirs / contact
            n_lam_an = 2 * n_c * NUM_FRICTION_DIRECTIONS    # = 4·n_c
            if n_c > 0:
                # E_t: (n_c, 4·n_c) — sums tangent-direction λ's per contact
                E_t_an = np.zeros((n_c, n_lam_an))
                for i in range(n_c):
                    E_t_an[i, 4*i : 4*(i+1)] = 1.0
                # anitescu_mu_vec: μ replicated 2·dirs per contact.
                # μ can be a scalar (uniform) or an ndarray of shape (n_c,)
                # (per-pair, ref sampling_c3plus_options.yaml:44
                # mu_per_pair_type). Indexed lookup handles both.
                _mu_arr = np.broadcast_to(np.asarray(mu), (n_c,))
                anitescu_mu_vec = np.zeros(n_lam_an)
                for i in range(n_c):
                    anitescu_mu_vec[
                        2 * NUM_FRICTION_DIRECTIONS * i :
                        2 * NUM_FRICTION_DIRECTIONS * (i + 1)
                    ] = float(_mu_arr[i])
                anitescu_mu_diag = np.diag(anitescu_mu_vec)
                # J_c folding (analog of lcs_factory.cc:246):
                #   J_c = E_t^T·J_n + diag(μ)·J_t
                # In EE-space coords (box_v[6], v_ee[3]):
                J_c_box = (E_t_an.T @ J_n_box
                           + anitescu_mu_diag @ J_t_box)         # (4n_c, 6)
                J_c_ee  = (E_t_an.T @ J_n_ee
                           + anitescu_mu_diag @ J_t_ee)          # (4n_c, 3)
                Minv_JcT_box = M_box_inv @ J_c_box.T              # (6, 4n_c)

                # D — single block, no γ/λ_n/λ_t partition.
                # Box rows: dt² · N_box · M_box^{-1} · J_c_box^T (box_q)
                #          dt   · M_box^{-1} · J_c_box^T          (box_v)
                # EE rows:  dt² · M_ee_op_inv · J_c_ee^T          (p_ee)
                #          dt   · M_ee_op_inv · J_c_ee^T          (v_ee)
                D_an = np.zeros((N_X, n_lam_an))
                D_an[self.BOX_Q_SLOT, :] = (dt * dt) * (N_box @ Minv_JcT_box)
                D_an[self.P_EE_SLOT,  :] = (dt * dt) * (M_ee_op_inv @ J_c_ee.T)
                D_an[self.BOX_V_SLOT, :] = dt * Minv_JcT_box
                D_an[self.V_EE_SLOT,  :] = dt * (M_ee_op_inv @ J_c_ee.T)

                # E — gradient of η wrt x (single 4n_c × N_X block).
                # η = E_t^T·phi/dt + J_c·v_{k+1}, with v_{k+1} = v + dt·f(x,u,λ).
                #   ∂/∂box_q: dt · J_c_box · df_box_dboxq + J_c_box·vNqdot_box/dt (A2 fix)
                #   ∂/∂box_v: J_c_box + dt · J_c_box · df_box_dboxv
                #   ∂/∂p_ee : J_c_ee / dt                                    (A2 fix)
                #   ∂/∂v_ee : J_c_ee
                # 2026-07-26 arc-2 A2 fix: reference `lcs_factory.cc:533`
                # includes `E_tᵀ·Jn·vNqdot/dt` in E's q-column (phi-gradient
                # wrt q). Port previously baked the position term into c via
                # J_c·v (velocity term) but missed the pure phi(q_{k+1})
                # gradient. This affects η prediction accuracy away from x*.
                E_an = np.zeros((n_lam_an, N_X))
                E_an[:, self.BOX_Q_SLOT] = dt * (J_c_box @ df_box_dboxq)
                E_an[:, self.BOX_V_SLOT] = (J_c_box
                                            + dt * (J_c_box @ df_box_dboxv))
                E_an[:, self.V_EE_SLOT]  = J_c_ee
                if self._use_e_block_split:
                    E_an[:, self.BOX_Q_SLOT] += (J_c_box @ vNqdot_box) / dt
                    E_an[:, self.P_EE_SLOT]  += J_c_ee / dt

                # F — single PSD block (analog of lcs_factory.cc:259):
                #   F = dt · J_c · M^{-1} · J_c^T
                # With M block-diag between box and EE operational-space:
                #   F = dt · J_c_box · M_box^{-1} · J_c_box^T
                #     + dt · J_c_ee · M_ee_op_inv · J_c_ee^T
                F_an = (dt * (J_c_box @ Minv_JcT_box)
                        + dt * (J_c_ee @ M_ee_op_inv @ J_c_ee.T))

                # H — u-coupling: only v_ee path contributes, M_ee_op_inv scaling.
                H_an = dt * (J_c_ee @ M_ee_op_inv)                # (4n_c, 3)

                # c — linearization-point value of η at u = 0 (d_v_ee_offset
                # = 0); subtraction converts to the affine offset. H·u* must
                # NOT be subtracted — the value expression excludes u, so
                # subtracting H·u* shifted the EE-coupled rows by −H·u*
                # whenever u_lin ≠ 0 (p146 walk root cause; see ST-path note).
                c_an = (E_t_an.T @ phi / dt
                        + J_c_box @ (box_v + dt * d_box_v_offset)
                        + J_c_ee  @ (v_ee  + dt * d_v_ee_offset))
                c_an -= E_an @ x_star
            else:
                # No contacts: trivial dimensions (n_lam_an = 0)
                D_an = np.zeros((N_X, n_lam_an))
                E_an = np.zeros((n_lam_an, N_X))
                F_an = np.zeros((n_lam_an, n_lam_an))
                H_an = np.zeros((n_lam_an, N_U))
                c_an = np.zeros(n_lam_an)

            # Overwrite the ST output with the Anitescu LCS.
            n_lam = n_lam_an
            D     = D_an
            E_lcs = E_an
            F_lcs = F_an
            H_lcs = H_an
            c_lcs = c_an

        # -----------------------------------------------------------------
        # 5.5. LCS scaling REMOVED here 2026-08-08 — moved to the solver.
        # -----------------------------------------------------------------
        # Reference structure: LCSFactory produces a RAW (physical) LCS;
        # `C3::ScaleLCS()` (c3.cc:203-212) scales it inside the SOLVER and
        # `C3::Solve` un-scales λ back to physical units before publishing
        # (c3.cc:350-353 `lambda_sol_ *= AnDn_`). So every LCS consumer
        # outside the solver — cost simulation, diagnostics, the surrogate
        # evaluators — sees physical matrices, and every λ consumer sees
        # Newtons.
        #
        # The port previously scaled HERE as well as in the solver. The
        # double application was idempotent for the dynamics (the solver's
        # recomputed scale came out exactly 1.0 because ||D_scaled|| ==
        # ||A|| by construction) but it silently DISABLED the solver's
        # un-scale, which is guarded by `_lcs_scale != 1.0`. Net effect:
        # every λ reported to the executor, the force-command thresholds
        # and the logs was the internal value λ_phys/scale — 5.05× too
        # large at the measured push_t scale of 0.198. The prior comment
        # here ("reference downstream also sees the scaled λ") was wrong:
        # c3.cc:350-353 un-scales unconditionally.
        # `self._scale_lcs` is retained as the reference `scale_lcs` option
        # and is now read by the solver.

        # -----------------------------------------------------------------
        # 6. Stash for diagnostics + return (mirror R^7 path's API).
        # -----------------------------------------------------------------
        # Mirror the R^7 path's _last_* stash so downstream diagnostics in
        # wrapper.py (e.g. the [COST-DUMP] / [PLAN-VS-EXEC] dumps at
        # wrapper.py:1240, 1266, 1325, 1388) find the attributes they
        # expect. Values are EE-space-sized; diagnostic code that
        # multiplies by current_q+current_v (R^7 layout) would still
        # silently fail via its own try/except, but it won't AttributeError
        # the run.
        self._last_A, self._last_B, self._last_D, self._last_d = A, B_ctrl, D, d_vec
        self._last_E, self._last_F, self._last_H, self._last_c = E_lcs, F_lcs, H_lcs, c_lcs
        self._last_n_c = n_c
        self._last_J_n, self._last_J_t = J_n_new, J_t_new
        self._last_phi, self._last_mu  = phi, mu
        # Separate stash for tests that want to distinguish R^7 vs EE-space.
        self._last_ee_space_A = A
        self._last_ee_space_B = B_ctrl
        self._last_ee_space_D = D
        self._last_ee_space_d = d_vec
        self._last_ee_space_E = E_lcs
        self._last_ee_space_F = F_lcs
        self._last_ee_space_H = H_lcs
        self._last_ee_space_c = c_lcs
        # Also stash the Drake-side (n_c, n_v_full) Jacobians so the
        # downstream OSC can compose τ_ff = -J_n^T λ in n_v space (the
        # planner's scalar λ values map identically; only the Jacobian
        # changes shape between the EE-space LCS coords and Drake n_v).
        self._last_J_n_n_v_full = J_n_drake
        self._last_J_t_n_v_full = J_t_drake
        return (
            A, B_ctrl, D, d_vec,
            E_lcs, F_lcs, H_lcs, c_lcs,
            J_n_new, J_t_new, phi, mu,
        )


