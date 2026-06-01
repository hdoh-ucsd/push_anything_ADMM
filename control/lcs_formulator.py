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
                 box_ground_drag: float = 10.0):
        self.plant = plant
        self.mu    = float(mu)
        # Effective viscous damping coefficient applied to the manipuland's
        # translational velocity in the LCS-prediction A matrix. Approximates
        # box-ground Coulomb drag, which the LCS complementarity machinery
        # cannot reliably enforce over the horizon (λ_n_gnd collapses to ~0
        # in the ADMM projection, so μ·λ_n_gnd ≈ 0 and the box is predicted
        # to coast frictionlessly even with BOX-GND admitted). Tuned so that
        # at sliding speed v ≈ 0.4 m/s, c·v ≈ μ·g ≈ 3.9 m/s² (μ=0.4, g=9.81).
        # Set 0 to disable.
        self._box_drag_c = float(box_ground_drag)
        self._obj_body   = obj_body

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
                             distance_threshold: float = 0.002):
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
            def _admit(sdp):
                ee_box = ((sdp.id_A in self._manipuland_geom_ids and
                           sdp.id_B in self._ee_geom_ids)
                       or (sdp.id_B in self._manipuland_geom_ids and
                           sdp.id_A in self._ee_geom_ids))
                box_ground = ((sdp.id_A in self._manipuland_geom_ids and
                               sdp.id_B in self._ground_geom_ids)
                           or (sdp.id_B in self._manipuland_geom_ids and
                               sdp.id_A in self._ground_geom_ids))
                return ee_box or box_ground
            sd_pairs = [sdp for sdp in sd_pairs if _admit(sdp)]

        n_filtered = len(sd_pairs)
        if n_filtered > 10:
            print(f"[LCS] WARNING: {n_filtered} contact pairs after filtering "
                  f"(expected ≤10) — check EE/object geometry IDs")

        W = self.plant.world_frame()
        phis, J_n_rows, J_t_rows = [], [], []

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
            self._last_contact_info.append({
                "body_A": body_A.name(), "body_B": body_B.name(),
                "a_is_box": a_is_box,
                "tag": _tag,
                "nhat_BA_W": np.array(nhat),
                "nhat_onto_box": nhat_onto_box,
                "p_ACa": np.array(sdp.p_ACa),
                "p_BCb": np.array(sdp.p_BCb),
                "distance": float(sdp.distance),
            })

            # Rotation-bonus scorer needs (p_contact_W, nhat_onto_box) for
            # EE-BOX pairs only. Use the contact witness on the box body,
            # transformed to world via the current plant context.
            if _tag == "EE-BOX":
                if a_is_box:
                    body_box = body_A
                    p_BoCo = np.asarray(sdp.p_ACa).reshape(3, 1)
                else:
                    body_box = body_B
                    p_BoCo = np.asarray(sdp.p_BCb).reshape(3, 1)
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
                self.mu,
            )

        return (
            np.array(phis),
            np.vstack(J_n_rows),    # (n_c, n_v)
            np.vstack(J_t_rows),    # (4*n_c, n_v)
            self.mu,
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
        sim_t = step * 0.01  # MPC dt is 10 ms

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
    _EE_MASS = 1.0   # kg

    def linearize_discrete_ee_space(self, context, dt: float, u_lin=None):
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

        # Box's N(q) sub-block: q_dot_box = N_box · box_v.
        N_box = np.zeros((BOX_N_Q, BOX_N_V))
        for i in range(BOX_N_V):
            e_full = np.zeros(self.n_v)
            e_full[BS + i] = 1.0
            qdot_full = self.plant.MapVelocityToQDot(context, e_full)
            N_box[:, i] = qdot_full[BOX_Q_START : BOX_Q_START + BOX_N_Q]

        # Box accel at linearization point (continuous): f_box = M_box^-1
        # (-Cv_box + tau_g_box).  λ and u contributions are added through
        # the LCS structure (D and H), so f_box here is the AUTONOMOUS part.
        f_box = M_box_inv @ (-Cv_box + tau_g_box)

        # EE accel at linearization point (continuous): f_ee = u/m_ee + λ-term.
        m_ee = float(self._EE_MASS)
        f_ee = u_star / m_ee   # autonomous (no λ) part

        # -----------------------------------------------------------------
        # 3. Contacts: phi, Drake's J_n (n_c, n_v_full), nhat list. We then
        #    PROJECT to the new low-dim velocity space [box_v(6), v_ee(3)].
        # -----------------------------------------------------------------
        phi, J_n_drake, J_t_drake, mu = self.extract_lcs_contacts(context)
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

        # 4d. B_ctrl — CONFIGURATION-INDEPENDENT (Stage A crux).
        B_ctrl = np.zeros((N_X, N_U))
        # p_ee rows: ∂p_ee_{k+1}/∂u = dt · ∂v_ee_{k+1}/∂u = dt² / m_ee · I.
        B_ctrl[self.P_EE_SLOT, :] = (dt * dt / m_ee) * np.eye(3)
        # v_ee rows: ∂v_ee_{k+1}/∂u = dt / m_ee · I.
        B_ctrl[self.V_EE_SLOT, :] = (dt / m_ee) * np.eye(3)
        # box_q rows and box_v rows: zero (u doesn't enter box dynamics).

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
            # p_ee rows: dt² / m_ee · J_n_ee^T (and J_t_ee^T)
            D[self.P_EE_SLOT, SLN:SLN + n_c]    = (dt*dt / m_ee) * J_n_ee.T
            D[self.P_EE_SLOT, SLT:SLT + n_t]    = (dt*dt / m_ee) * J_t_ee.T
            # box_v rows
            D[self.BOX_V_SLOT, SLN:SLN + n_c]   = dt * Minv_JnT_box
            D[self.BOX_V_SLOT, SLT:SLT + n_t]   = dt * Minv_JtT_box
            # v_ee rows
            D[self.V_EE_SLOT, SLN:SLN + n_c]    = (dt / m_ee) * J_n_ee.T
            D[self.V_EE_SLOT, SLT:SLT + n_t]    = (dt / m_ee) * J_t_ee.T

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
            F_lcs[SG:SG + n_c,  SLN:SLN + n_c]  = mu * np.eye(n_c)
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
            # F: λ-coupling — ∂v_{k+1}/∂λ via D-style entries.
            F_lcs[SLN:SLN + n_c, SLN:SLN + n_c] = (
                dt * (J_n_box @ Minv_JnT_box) + (dt / m_ee) * (J_n_ee @ J_n_ee.T)
            )
            F_lcs[SLN:SLN + n_c, SLT:SLT + n_t] = (
                dt * (J_n_box @ Minv_JtT_box) + (dt / m_ee) * (J_n_ee @ J_t_ee.T)
            )
            # H: u-coupling — only the v_ee path contributes, m_ee scaling.
            H_lcs[SLN:SLN + n_c, :] = (dt / m_ee) * J_n_ee
            # c: const offset: φ/dt + J_n · (current_v + dt · d_offset)
            #    − E_lcs · x* − H_lcs · u*   (linearization residual)
            c_const_v_box  = J_n_box @ (box_v + dt * d_box_v_offset)
            c_const_v_ee   = J_n_ee  @ (v_ee  + dt * d_v_ee_offset)
            c_lcs[SLN:SLN + n_c] = phi / dt + c_const_v_box + c_const_v_ee
            # NOTE: c_lcs absorbs the "constant" of η linearized at (x*, u*);
            # E and H carry the gradient parts. We subtract E·x* + H·u* below.

            # λ_t rows analogously.
            E_lcs[SLT:SLT + n_t, self.BOX_Q_SLOT] = dt * (J_t_box @ df_box_dboxq)
            E_lcs[SLT:SLT + n_t, self.BOX_V_SLOT] = J_t_box + dt * (J_t_box @ df_box_dboxv)
            E_lcs[SLT:SLT + n_t, self.V_EE_SLOT]  = J_t_ee
            F_lcs[SLT:SLT + n_t, SG:SG + n_c]     = E_t.T
            F_lcs[SLT:SLT + n_t, SLN:SLN + n_c]   = (
                dt * (J_t_box @ Minv_JnT_box) + (dt / m_ee) * (J_t_ee @ J_n_ee.T)
            )
            F_lcs[SLT:SLT + n_t, SLT:SLT + n_t]   = (
                dt * (J_t_box @ Minv_JtT_box) + (dt / m_ee) * (J_t_ee @ J_t_ee.T)
            )
            H_lcs[SLT:SLT + n_t, :] = (dt / m_ee) * J_t_ee
            c_const_v_box_t = J_t_box @ (box_v + dt * d_box_v_offset)
            c_const_v_ee_t  = J_t_ee  @ (v_ee  + dt * d_v_ee_offset)
            c_lcs[SLT:SLT + n_t] = c_const_v_box_t + c_const_v_ee_t

            # Subtract E·x* and H·u* so c carries the "value of η at (x*, u*)".
            # This makes η(x*, u*, λ=?) = E·x* + F·λ + H·u* + c equal to the
            # linearization value at the operating point — same convention as
            # the R^7 path.
            c_lcs -= E_lcs @ x_star + H_lcs @ u_star

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


