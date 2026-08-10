"""
Per-task cost functions for C3+ MPC.

All three tasks (pushing, hard_pushing, shepherding) share the same geometric
cost structure — they differ only in weights loaded from config/tasks.yaml.

Geometry (2D top-down view):
  - g_hat    : unit vector from object to goal
  - y_ref    : proxy target = obj - d_push * g_hat  (spot behind the object)
  - s        : signed distance of EE along g_hat from object centre
              (s > 0 means EE is on goal-side = wrong side)

Cost terms:
  1. progress     : w_progress  * ||obj - goal||          (move object toward goal)
  2. proxy        : w_proxy     * ||ee - y_ref||           (approach from correct side)
  3. behind       : w_behind    * max(0, s + margin)^2     (quadratic wrong-side penalty)
  4. interaction  : w_interaction * exp(-||ee-obj||^2/sigma^2)  (< 0 = attractive reward)
"""
import numpy as np


def _goal_quaternion(target_yaw: float, target_quat=None) -> np.ndarray:
    """Goal quaternion [w, x, y, z] for the object-orientation cost.

    Legacy/planar tasks specify a scalar `goal_yaw` and the goal is the
    Z-rotation [cos(a/2), 0, 0, sin(a/2)]. That is sufficient for every task
    whose object rests on a fixed face (box, T, H) -- the reachable set of
    resting orientations is a one-parameter yaw family.

    The jack breaks that assumption: it rests on a tripod of tip spheres and
    reorients by ROLLING onto a different tripod, so its goal
    (jacktoy/parameters/goal_params.yaml fixed_target_orientation) tilts out
    of the plane and cannot be written as any yaw. Such tasks pass the full
    quaternion via `goal_quat` in tasks.yaml.

    With target_quat=None this returns exactly the previous expression, so
    all yaw-specified tasks are bit-identical.
    """
    if target_quat is None:
        a_half = 0.5 * float(target_yaw)
        return np.array([np.cos(a_half), 0.0, 0.0, np.sin(a_half)])
    q = np.asarray(target_quat, dtype=float).flatten()
    if q.shape[0] != 4:
        raise ValueError(f"goal_quat must have 4 entries [w,x,y,z]; got {q.shape[0]}")
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        raise ValueError("goal_quat has zero norm")
    return q / n


class ManipulationCost:
    """
    Callable cost function shared by all three manipulation tasks.
    Instantiated once per run and injected into C3MPC.

    Parameters
    ----------
    plant         : Drake MultibodyPlant
    ee_frame_name : name of the end-effector frame (e.g. 'panda_link8')
    obj_body      : Drake Body for the manipulated object
    cost_cfg      : dict with keys d_push, margin, sigma,
                    w_progress, w_proxy, w_behind, w_interaction
    """

    def __init__(self, plant, ee_frame_name: str, obj_body, cost_cfg: dict):
        self.plant       = plant
        self.world_frame = plant.world_frame()
        self.ee_frame    = plant.GetFrameByName(ee_frame_name)

        # Pre-compute object position indices once.
        # Drake floating-body layout: [qw, qx, qy, qz, x, y, z]
        ps = obj_body.floating_positions_start()
        self._obj_x_idx = ps + 4
        self._obj_y_idx = ps + 5

        # Cost weights
        c = cost_cfg
        self.d_push        = float(c["d_push"])
        self.margin        = float(c["margin"])
        self.sigma         = float(c["sigma"])
        self.w_progress    = float(c["w_progress"])
        self.w_proxy       = float(c["w_proxy"])
        self.w_behind      = float(c["w_behind"])
        self.w_interaction = float(c["w_interaction"])
        self.z_ee_target   = float(c.get("z_ee_target", 0.05))

    def __call__(self, plant_ctx, q_sim: np.ndarray,
                 target_xy: np.ndarray) -> float:
        """
        Evaluate scalar cost at a single rollout timestep.

        Parameters
        ----------
        plant_ctx  : Drake plant context (already seeded with q_sim/v_sim).
        q_sim      : (n_q,) generalized positions at this rollout step.
        target_xy  : (2,)   goal [x, y] in world frame.

        Returns
        -------
        cost : float
        """
        # --- Object position (XY from q_sim; z from config target height) ---
        obj_xy  = np.array([q_sim[self._obj_x_idx], q_sim[self._obj_y_idx]])
        obj_3d  = np.array([obj_xy[0], obj_xy[1], self.z_ee_target])

        # --- End-effector 3D position via Drake kinematics ---
        ee_3d = self.plant.CalcPointsPositions(
            plant_ctx, self.ee_frame, np.zeros(3), self.world_frame
        ).flatten()
        ee_xy = ee_3d[:2]

        # --- Goal direction (2D, horizontal) ---
        v_goal = target_xy - obj_xy
        dist   = np.linalg.norm(v_goal)
        if dist < 1e-6:
            return 0.0
        g_hat = v_goal / dist

        # 1. Progress: object distance to goal
        progress = self.w_progress * dist

        # 2. Proxy target: 3D point behind object at contact height
        y_ref_3d = np.array([obj_xy[0] - self.d_push * g_hat[0],
                              obj_xy[1] - self.d_push * g_hat[1],
                              self.z_ee_target])
        proxy = self.w_proxy * np.linalg.norm(ee_3d - y_ref_3d)

        # 3. Behind constraint: arm must stay on the opposite side from goal (XY only)
        s = float(np.dot(ee_xy - obj_xy, g_hat))
        behind = self.w_behind * max(0.0, s + self.margin) ** 2

        # 4. Interaction: 3D Gaussian reward — pulls EE to correct height AND XY
        r_sq      = float(np.dot(ee_3d - obj_3d, ee_3d - obj_3d))
        interaction = self.w_interaction * np.exp(-r_sq / (self.sigma ** 2))

        return progress + proxy + behind + interaction


# ---------------------------------------------------------------------------
# Quadratic cost for C3 MPC (replaces geometric heuristic)
# ---------------------------------------------------------------------------

class QuadraticManipulationCost:
    """
    Builds Q, R, QN cost matrices and x_ref for C3 MPC's LQR-style tracking.

    Two cost components:
      1. Object XY goal cost  — drives the object to target_xy (always active)
      2. Linearised EE approach cost — drives the arm EE toward a proxy contact
         point behind the object, computed from the arm Jacobian at each step.
         Critical when there is no contact (D≈0): without this the QP has no
         incentive to move the arm and just minimises u^T R u → arm freezes.

    Weights from tasks.yaml cost section.

    Parameters
    ----------
    plant         : Drake MultibodyPlant
    ee_frame_name : end-effector frame (e.g. 'panda_link8')
    obj_body      : Drake Body for the manipulated object
    cost_cfg      : dict — keys: w_obj_xy, w_obj_z, w_torque, w_terminal,
                    z_ee_target, d_push, w_ee_approach
    n_x           : int  state dim n_q + n_v
    n_u           : int  control dim (= number of arm joints)
    """

    def __init__(self, plant, ee_frame_name: str, obj_body, cost_cfg: dict,
                 n_x: int, n_u: int, math_diag: bool = False,
                 object_shape: str = None,
                 object_half_extent: float = None,
                 pusher_radius: float = None):
        import pydrake.all as ad
        self.plant       = plant
        self.ee_frame    = plant.GetFrameByName(ee_frame_name)
        self.world_frame = plant.world_frame()
        self._ad         = ad
        self.n_x         = n_x
        self.n_u         = n_u   # arm DOF (first n_u velocities = arm joint vels)

        # Object position and orientation indices in q
        # Drake floating-body layout: [qw, qx, qy, qz, x, y, z]
        ps = obj_body.floating_positions_start()
        vs = obj_body.floating_velocities_start_in_v()
        self.n_q        = plant.num_positions()
        self._obj_ps    = ps          # quaternion base index in q
        self._obj_vs    = vs          # velocity base index in v (ωx,ωy,ωz,vx,vy,vz)
        self._obj_x_idx = ps + 4
        self._obj_y_idx = ps + 5
        self._obj_z_idx = ps + 6

        c = cost_cfg
        self.w_obj_xy      = float(c.get("w_obj_xy",      1000.0))
        # Path-C Pareto probe (2026-07-09): PORT_W_OBJ_XY_MULT scales w_obj_xy
        # in-place after yaml load. Default 1.0 = byte-identical. Used to sweep
        # closure recovery at N=5 without touching tasks.yaml.
        import os as _os_woxm
        _w_obj_xy_mult = float(_os_woxm.environ.get("PORT_W_OBJ_XY_MULT", "1.0"))
        if _w_obj_xy_mult != 1.0:
            _orig = self.w_obj_xy
            self.w_obj_xy *= _w_obj_xy_mult
            print(f"[PORT_W_OBJ_XY_MULT] w_obj_xy {_orig:.1f} → "
                  f"{self.w_obj_xy:.1f} (×{_w_obj_xy_mult:.2f})", flush=True)
        self.w_obj_z       = float(c.get("w_obj_z",         10.0))
        self.w_box_z       = float(c.get("w_box_z",        100.0))
        self.w_box_rp      = float(c.get("w_box_rp",        50.0))
        # Near-goal (pose regime) weight overrides. Mirrors reference
        # push_t sampling_c3plus_options.yaml: q_vector_position obj_pos
        # 250,250,250 → q_vector obj_pos 200,200,120 (20% xy reduction,
        # 52% z reduction). When _crossed_switching_threshold is True,
        # build() scales down w_obj_xy and w_box_z accordingly. Defaults
        # fall back to the far-goal values → byte-identical to prior
        # behavior when the pose override is not configured.
        self.w_obj_xy_pose = float(c.get("w_obj_xy_pose", self.w_obj_xy))
        self.w_box_z_pose  = float(c.get("w_box_z_pose",  self.w_box_z))
        # Yaw-error penalty (Jin & Posa eq. 40 analog). w_yaw=0 → fully inert.
        # Residual e_z = c_yaw · q_box where c_yaw = [-sin(α/2),0,0,cos(α/2)]
        # is EXACTLY linear in the quaternion (not a small-angle approximation):
        # for an upright yaw-only configuration q_box = [cos(ψ/2),0,0,sin(ψ/2)],
        # e_z = sin((ψ-α)/2) — the standard quaternion half-angle metric,
        # globally monotonic on ψ ∈ (α-π, α+π). NOT a raw qz penalty, which is
        # invalid for a unit quaternion.
        self.w_yaw         = float(c.get("w_yaw",            0.0))
        self._target_yaw   = 0.0   # updated each build() call via target_yaw kwarg
        # Optional full goal quaternion (tasks.yaml `goal_quat`). None => the
        # goal is the yaw-only rotation built from target_yaw, which is what
        # every flat-resting task (box, T, H) wants. Set for the jack, whose
        # goal tilts out of the plane. See _goal_quaternion.
        #
        # This is STATIC: unlike target_yaw it is not re-clipped per tick by
        # the angular lookahead. main.py asserts at startup that the task's
        # total reorientation demand is under goal_params lookahead_angle, so
        # the clip would never fire anyway; a task exceeding that needs the
        # per-tick geodesic SLERP (reference goal_generator.cc:410-434).
        self._goal_quat    = None

        # Near-goal quaternion-dependent Q_block (reference
        # sampling_based_c3_controller.cc:1517-1570). When enabled AND the
        # wrapper's `crossed_switching_threshold` is True, `build_ee_space`
        # replaces the linear w_yaw quat block with the Hessian-of-
        # squared-quaternion-angle-difference cost. Guides yaw via a proper
        # 3D rotation metric near goal; the linear w_yaw form saturates
        # near the goal (its residual is sin((ψ-α)/2), curvature → 0).
        # Reference push_t: use_quaternion_dependent_cost=true, weight=1000,
        # regularizer_fraction=0.
        self.use_quaternion_dependent_cost = bool(
            c.get("use_quaternion_dependent_cost", False))
        self.q_quaternion_dependent_weight = float(
            c.get("q_quaternion_dependent_weight", 1000.0))
        self.q_quaternion_dependent_regularizer_fraction = float(
            c.get("q_quaternion_dependent_regularizer_fraction", 0.0))

        # Reference-conforming Q construction — Q = w_Q · diag(q_vector).
        # When `use_reference_q_vector: true` in cost config, build_ee_space
        # replaces the per-weight base Q (w_obj_xy, w_obj_z, w_box_rp, w_yaw
        # rank-1 outer product) with a diagonal built from the reference
        # push_t/parameters/sampling_c3_options.yaml:65-75 layout, remapped
        # from the reference's [ee_pos, quat, obj_pos, ee_vel, obj_ang_vel,
        # obj_lin_vel] order into the port's [quat, obj_pos, ee_pos,
        # obj_ang_vel, obj_lin_vel, ee_vel] order. Near-goal quaternion
        # Hessian override still fires on top of this base; EE-approach and
        # yaw rank-1 outer product are skipped when this flag is set
        # (reference has neither).
        self.use_reference_q_vector = bool(
            c.get("use_reference_q_vector", False))
        self.w_Q = float(c.get("w_Q", 1.0))
        # Per-group q_vector lists in reference layout order.
        self._q_vec_ee_pos      = list(c.get(
            "q_vector_ee_pos",      [0.01, 0.01, 0.01]))
        self._q_vec_obj_quat    = list(c.get(
            "q_vector_obj_quat",    [0.1, 0.1, 0.1, 0.1]))
        self._q_vec_obj_pos     = list(c.get(
            "q_vector_obj_pos",     [200.0, 200.0, 120.0]))
        self._q_vec_ee_vel      = list(c.get(
            "q_vector_ee_vel",      [5.0, 5.0, 5.0]))
        self._q_vec_obj_ang_vel = list(c.get(
            "q_vector_obj_ang_vel", [0.05, 0.05, 0.05]))
        self._q_vec_obj_lin_vel = list(c.get(
            "q_vector_obj_lin_vel", [0.05, 0.05, 0.05]))
        # Mutable flag set by wrapper per tick. Enables the near-goal
        # quat-dep cost override. Default False → no divergence for
        # existing tasks / pre-crossed-threshold behavior.
        self._crossed_switching_threshold = False
        self.w_torque      = float(c.get("w_torque",         0.01))
        self.w_terminal    = float(c.get("w_terminal",        5.0))
        self.z_ref         = float(c.get("z_ee_target",      0.05))
        # Separate obj z target from EE z target. Reference push_t goal target
        # keeps T at its init z (does NOT lift). Prior x_ref[obj_z]=z_ee_target
        # forced planner to lift push_t 14mm (z_ee=0.034 vs T_z=0.020).
        # Default: obj target z equals init obj z if provided, else z_ee_target
        # (backward-compat for tasks where they coincide — box, sphere).
        self.z_obj_target  = float(c.get("z_obj_target",    self.z_ref))
        self.d_push        = float(c.get("d_push",           0.10))
        self.w_ee_approach = float(c.get("w_ee_approach",   800.0))
        # Reference `goal_generator.cc:113-114` semantics: EE-position LCS
        # reference tracks the CURRENT object position (not the goal) with a
        # fixed z-offset above the object center. Reference push_t
        # `goal_params.yaml:14`: ee_target_z_offset_above_object = 0.06.
        # When > 0, `build_ee_space` writes:
        #   x_ref[7:10] = [obj_curr_x, obj_curr_y, obj_curr_z + z_offset]
        #   Q[7:10, 7:10] += w_Q * q_vec_ee_pos (small; matches ref weight)
        # so the planner gently pulls the arm to hover directly above the
        # T center as the T moves — the missing counterpart to `2.k`
        # (which pins EE orientation via OSC).
        # Default 0.0 preserves the port's prior behavior (§7.31 note:
        # "EE-approach cost SKIPPED"). Set >0 in tasks.yaml push_t block
        # to activate.
        self.ee_target_z_offset = float(c.get(
            "ee_target_z_offset_above_object", 0.0))
        # Lateral-alignment clamp scale (metres). The proxy shift at
        # build_ee_space()::~739 (and the legacy build()::~438) saturates at
        # `extra_shift = -perp_vec * min(1.0, perp_magnitude / scale)`. Smaller
        # scale = more responsive correction (full strength at smaller
        # off-equator distance). Default 0.05 preserves legacy behavior;
        # pushing/tasks.yaml pins the operational value. See plan
        # docs/superpowers/plans/2026-06-07-B-lateral-align-clamp-harden.md.
        self.lateral_align_full_scale = float(
            c.get("lateral_align_full_scale", 0.05))

        # Geometry (used by PORT_BOX_DPUSH_FIX for the box face-target proxy).
        # object_shape ∈ {"box", "sphere", "tshape", None}. Only "box" activates
        # the geometry-derived d_push override. object_half_extent is the box's
        # half-size along the push axis (task_cfg["size"][0]/2 for cubes).
        # pusher_radius mirrors sim.env_builder.PUSHER_RADIUS (honors the
        # PORT_PUSHER_RADIUS env override).
        self._object_shape       = object_shape
        self._object_half_extent = object_half_extent
        self._pusher_radius      = pusher_radius

        self._math_diag = math_diag
        self._q_printed = False

        # (Cost-bias state removed in the 2026-08-05 CLI prune.)

        # EE-approach cost diagnostic (off by default; enabled via --ee-cost-diag)
        self._diag_ee_cost      = False

        # Static parts of the base object-goal cost
        self._Q_obj = self._make_Q_obj()
        self._R     = self.w_torque * np.eye(n_u)

    def _make_Q_obj(self) -> np.ndarray:
        Q = np.zeros((self.n_x, self.n_x))
        # XY position → goal
        Q[self._obj_x_idx, self._obj_x_idx] = self.w_obj_xy
        Q[self._obj_y_idx, self._obj_y_idx] = self.w_obj_xy
        # Z height — base penalty from YAML, hard floor from w_box_z
        Q[self._obj_z_idx, self._obj_z_idx] = self.w_obj_z + self.w_box_z
        # Roll / pitch quaternion components (qx = ps+1, qy = ps+2).
        # Penalise deviation from zero so the solver keeps the box upright.
        # qz (yaw, ps+3) is left free — the box may rotate horizontally.
        Q[self._obj_ps + 1, self._obj_ps + 1] = self.w_box_rp   # qx (roll)
        Q[self._obj_ps + 2, self._obj_ps + 2] = self.w_box_rp   # qy (pitch)
        return Q

    def set_goal_quat(self, q):
        """Install a full goal quaternion [w, x, y, z] (or None to clear).

        Called once at startup from main.py when tasks.yaml supplies
        `goal_quat`. Every build() thereafter uses it in place of the
        yaw-derived goal.
        """
        self._goal_quat = None if q is None else _goal_quaternion(0.0, q)

    def build(self, target_xy: np.ndarray,
              plant_ctx=None, current_q: np.ndarray = None,
              rich_mode: bool = False,
              target_yaw: float = 0.0,
              target_quat=None):
        """
        Return (Q, R, QN, x_ref) for one MPC step.

        If plant_ctx and current_q are provided, augments Q and x_ref with a
        linearised EE approach cost via the arm Jacobian.

        rich_mode=True disables the EE-approach proxy gradient (see
        counterfactual audit, results/counterfactual_north.log). Both the
        J_arm^T J_arm cost block and the arm x_ref shift are skipped; the
        perpendicular-box-velocity block (task-tracking) is retained.

        target_yaw=0.0 + w_yaw=0.0 → no yaw cost contribution (inert path for
        existing tasks). With w_yaw>0, the quaternion block of Q gets
        w_yaw · c_yaw c_yawᵀ added on the [qw,qz] slots and x_ref[qw,qz]
        gets set to the goal quaternion [cos(α/2), sin(α/2)].
        """
        # --- Base object-goal cost ---
        Q     = self._Q_obj.copy()
        if self.use_reference_q_vector:
            # Reference-conforming base Q for the R^7 full-plant layout —
            # mirror of build_ee_space:785-802. Q_base object block =
            # w_Q · diag(q_vector) with the reference per-group vectors
            # placed at the R^7 state indices:
            #   obj quat  → q slots ps..ps+3   (ALL FOUR components,
            #               incl. qw/qz — the base yaw weight the legacy
            #               _Q_obj lacks; w_box_rp roll/pitch is replaced)
            #   obj pos   → q slots ps+4..ps+6
            #   obj ω/v   → v slots n_q+obj_vs .. +6
            # The reference ee_pos / ee_vel groups have no R^7 counterpart
            # (EE pose is FK of arm q, not a state slot); the port's
            # linearised J_arm EE-approach block below remains the R^7
            # analog of that steering. As in build_ee_space, the base
            # assignment is regime-independent (no w_obj_xy_pose swap —
            # q_vector_obj_pos already carries the reference pose-regime
            # values).
            ps = self._obj_ps
            for _i in range(4):
                Q[ps + _i, ps + _i] = self.w_Q * self._q_vec_obj_quat[_i]
            Q[self._obj_x_idx, self._obj_x_idx] = \
                self.w_Q * self._q_vec_obj_pos[0]
            Q[self._obj_y_idx, self._obj_y_idx] = \
                self.w_Q * self._q_vec_obj_pos[1]
            Q[self._obj_z_idx, self._obj_z_idx] = \
                self.w_Q * self._q_vec_obj_pos[2]
            _vb = self.n_q + self._obj_vs
            for _i in range(3):
                Q[_vb + _i, _vb + _i] = \
                    self.w_Q * self._q_vec_obj_ang_vel[_i]
                Q[_vb + 3 + _i, _vb + 3 + _i] = \
                    self.w_Q * self._q_vec_obj_lin_vel[_i]
            if not getattr(self, "_ref_qvec_r7_logged", False):
                self._ref_qvec_r7_logged = True
                print("[REF-QVEC-R7] build(): reference q_vector base Q "
                      f"active (quat={self.w_Q * np.asarray(self._q_vec_obj_quat)}, "
                      f"pos={self.w_Q * np.asarray(self._q_vec_obj_pos)}, "
                      f"ω/v={self.w_Q * np.asarray(self._q_vec_obj_ang_vel)}/"
                      f"{self.w_Q * np.asarray(self._q_vec_obj_lin_vel)}); "
                      "ee_pos group has no R^7 state slot — J_arm "
                      "EE-approach block is the R^7 analog; ee_vel group "
                      "mapped via J_arm below", flush=True)
        # Near-goal (pose regime) cost swap. Mirrors reference push_t
        # `sampling_c3_options_.GetC3Options(crossed_cost_switching_threshold_)`
        # — swap object-position weights when box is inside 5 cm of goal.
        # Rewrites the Q_obj_pos slots in-place; leaves w_box_rp and other
        # blocks alone (analogous to reference push_t regime deltas).
        # Skipped under use_reference_q_vector (parity with build_ee_space,
        # whose ref branch is likewise regime-independent).
        elif self._crossed_switching_threshold:
            Q[self._obj_x_idx, self._obj_x_idx] = self.w_obj_xy_pose
            Q[self._obj_y_idx, self._obj_y_idx] = self.w_obj_xy_pose
            Q[self._obj_z_idx, self._obj_z_idx] = (self.w_obj_z
                                                   + self.w_box_z_pose)

        # --- Reference q_vector_ee_vel group, mapped through J_arm ---
        # The reference EE-space state carries v_ee as a slot with weight
        # w_Q · q_vector_ee_vel (push_t: 50 · 5.0 = 250/axis, target 0).
        # R^7 has no v_ee slot — v_ee = J_arm(q) · v_arm — so the exact
        # quadratic image on the arm-vel block is
        #   ‖v_ee‖²_W = v_armᵀ (J_armᵀ W J_arm) v_arm,  W = diag(w_Q·q_vec).
        # x_ref arm-vel slots stay 0 (reference ee_vel target is zero).
        # J is evaluated at the current q (same linearization convention as
        # the EE-approach J_arm block below).
        if self.use_reference_q_vector and plant_ctx is not None:
            _J_ee_v = self.plant.CalcJacobianTranslationalVelocity(
                plant_ctx, self._ad.JacobianWrtVariable.kV,
                self.ee_frame, np.zeros(3),
                self.world_frame, self.world_frame,
            )
            _J_arm_v = _J_ee_v[:, : self.n_u]   # (3, n_u)
            _W_vee = self.w_Q * np.asarray(self._q_vec_ee_vel, dtype=float)
            _vb_arm = self.n_q   # arm-vel slots x[n_q : n_q+n_u]
            Q[_vb_arm:_vb_arm + self.n_u, _vb_arm:_vb_arm + self.n_u] += \
                _J_arm_v.T @ np.diag(_W_vee) @ _J_arm_v
            if not getattr(self, "_ref_eevel_r7_logged", False):
                self._ref_eevel_r7_logged = True
                print(f"[REF-QVEC-R7] build(): ee_vel group mapped onto "
                      f"arm-vel block via J_armᵀ·diag({_W_vee})·J_arm "
                      f"(x[{_vb_arm}:{_vb_arm + self.n_u}])", flush=True)
        x_ref = np.zeros(self.n_x)
        x_ref[self._obj_x_idx] = target_xy[0]
        x_ref[self._obj_y_idx] = target_xy[1]
        x_ref[self._obj_z_idx] = self.z_obj_target

        # --- Yaw-target cost (Jin & Posa eq. 40 analog) ---
        # Add w_yaw · (c_yaw · (q_box - q_goal))² on the box quaternion slots.
        # c_yaw = [-sin(α/2), 0, 0, cos(α/2)] is linear in [qw,qx,qy,qz] and
        # equals the z-component of q_goal⁻¹ ⊗ q_box (vector part of the
        # relative quaternion). For upright box: c_yaw · q_box = sin((ψ-α)/2).
        # The xy components of c_yaw are zero, so this is orthogonal to the
        # existing w_box_rp roll/pitch regularization.
        self._target_yaw = float(target_yaw)
        _q_goal = _goal_quaternion(
            target_yaw, target_quat if target_quat is not None else self._goal_quat)
        a_half = 0.5 * self._target_yaw
        ps = self._obj_ps
        # Always seed the goal quaternion reference on qw/qz (parity with
        # build_ee_space:868-871): any quat-slot weight — the w_yaw rank-1
        # form, the reference q_vector base diag, or the near-goal Hessian —
        # only drives x toward q_goal if x_ref IS q_goal. With x_ref[quat]=0
        # a diagonal qw/qz weight would pull the quaternion toward the zero
        # vector instead. qx,qy stay 0 (upright), preserving roll/pitch
        # behavior. No-op for cost purposes when all quat weights are zero.
        x_ref[ps + 0] = np.cos(a_half)   # qw
        x_ref[ps + 3] = np.sin(a_half)   # qz
        # Optional linear rank-1 form — port-only far-goal add-on; skipped
        # under use_reference_q_vector (reference has no linear yaw cost;
        # parity with build_ee_space:874).
        if self.w_yaw > 0.0 and not self.use_reference_q_vector:
            cy = np.array([-np.sin(a_half), 0.0, 0.0, np.cos(a_half)])
            # Outer product on the (qw..qz) 4×4 sub-block of Q
            Q[ps:ps+4, ps:ps+4] += self.w_yaw * np.outer(cy, cy)

        # --- Near-goal quaternion-dependent Q_block override ---
        # Reference sampling_based_c3_controller.cc:1517-1570; mirror of
        # build_ee_space:887-914 on the R^7 quat slots (ps..ps+3). When
        # `use_quaternion_dependent_cost` is set AND the wrapper's
        # crossed_switching_threshold has latched (obj-to-goal < 5 cm),
        # REPLACE the quat block with the Hessian-of-squared-quaternion-
        # angle-difference cost (weight 1000) — the reference's sole yaw
        # driver (w_yaw=0 per reference). Without this, the R^7 planner's
        # only yaw authority is the base q_vector quat diag (5.0), which
        # is regularization-scale, not goal-driving.
        if (self.use_quaternion_dependent_cost
                and self._crossed_switching_threshold
                and plant_ctx is not None
                and current_q is not None):
            from control.quaternion_hessian import (
                build_regularized_quaternion_cost,
            )
            q_now = np.array([
                current_q[ps], current_q[ps + 1],
                current_q[ps + 2], current_q[ps + 3],
            ])
            q_goal = np.array([np.cos(a_half), 0.0, 0.0, np.sin(a_half)])
            Q_quat = build_regularized_quaternion_cost(
                q_now, q_goal,
                weight=self.q_quaternion_dependent_weight,
                regularizer_fraction=
                    self.q_quaternion_dependent_regularizer_fraction,
            )
            # Reference replaces (=) rather than adds — same as
            # build_ee_space, avoids double-counting base-diag + Hessian.
            Q[ps:ps + 4, ps:ps + 4] = Q_quat
            # x_ref quat already seeded to q_goal above.

        # --- Linearised EE approach cost (arm joints only) ---
        if plant_ctx is not None and current_q is not None:
            obj_xy  = np.array([current_q[self._obj_x_idx],
                                 current_q[self._obj_y_idx]])
            v_goal  = target_xy - obj_xy
            dist    = np.linalg.norm(v_goal)

            if self.use_reference_q_vector:
                # Reference ee_pos analog — REPLACES the port-only
                # w_ee_approach proxy block below (parity with
                # build_ee_space, whose direct EE-approach cost is always
                # skipped per §7.31: the reference has no backward-pull
                # approach cost). The reference's only EE-position cost is
                # hover tracking (goal_generator.cc:113-114):
                #   x_des[ee] = obj_curr + [0, 0, z_offset],
                #   weight w_Q · q_vector_ee_pos (push_t: 50·0.01 = 0.5/ax)
                # R^7 image via the arm Jacobian, same damped-least-squares
                # x_ref construction as the legacy block:
                #   Q[arm_q] += J_armᵀ · diag(w_Q·q_vec_ee_pos) · J_arm
                #   x_ref[arm_q] = q_arm + J⁺·(p_hover − p_ee)
                # This retires the off-reference 8000-weight 3-stage proxy
                # under the flag (port pos:vel authority was 8000:250 vs
                # reference 0.5:250 — p108 diagnosis).
                if self.ee_target_z_offset > 0.0 and not rich_mode:
                    ee_pos = self.plant.CalcPointsPositions(
                        plant_ctx, self.ee_frame, np.zeros(3),
                        self.world_frame
                    ).flatten()
                    _obj_z_now = float(current_q[self._obj_z_idx])
                    p_hover = np.array([
                        obj_xy[0], obj_xy[1],
                        _obj_z_now + self.ee_target_z_offset,
                    ])
                    J_ee = self.plant.CalcJacobianTranslationalVelocity(
                        plant_ctx, self._ad.JacobianWrtVariable.kV,
                        self.ee_frame, np.zeros(3),
                        self.world_frame, self.world_frame,
                    )
                    J_arm = J_ee[:, : self.n_u]
                    _W_pee = self.w_Q * np.asarray(
                        self._q_vec_ee_pos, dtype=float)
                    # §7.76 z-hold, R^7 analog (PORT_EE_Z_HOLD=1, default
                    # OFF). EE-space applies Q[9,9] += w_zh with
                    # x_ref[9] = z_tgt — a hard cost-side z anchor that
                    # breaks the §7.75 sphere-climb loop (p112: EE rode
                    # z 0.019→0.077 during the push burst, tipping the T
                    # and amplifying the wrong-way twist). R^7 has no EE-z
                    # state slot, so fold the same override into this
                    # hover cost: retarget the z-axis to z_tgt and boost
                    # the z-axis weight before the J-mapping below —
                    # structurally identical to the EE-space form.
                    import os as _os_zh_r7
                    if _os_zh_r7.environ.get("PORT_EE_Z_HOLD", "0") == "1":
                        _w_zh = float(_os_zh_r7.environ.get(
                            "PORT_EE_Z_HOLD_W", "1000.0"))
                        _z_tgt = float(_os_zh_r7.environ.get(
                            "PORT_EE_Z_HOLD_TARGET", str(self.z_ref)))
                        p_hover[2] = _z_tgt
                        _W_pee = _W_pee.copy()
                        _W_pee[2] += _w_zh
                        if not getattr(self, "_z_hold_r7_logged", False):
                            self._z_hold_r7_logged = True
                            print(f"[§7.76-Z-HOLD-R7] PORT_EE_Z_HOLD=1 "
                                  f"w_zh={_w_zh} z_tgt={_z_tgt:.4f}m "
                                  f"(J-mapped onto arm-q block)",
                                  flush=True)
                    Q[: self.n_u, : self.n_u] += \
                        J_arm.T @ np.diag(_W_pee) @ J_arm
                    ee_err = p_hover - ee_pos
                    lam = 0.001
                    JJT = J_arm @ J_arm.T + lam * np.eye(3)
                    dq = J_arm.T @ np.linalg.solve(JJT, ee_err)
                    x_ref[: self.n_u] = current_q[: self.n_u] + dq
                    if not getattr(self, "_ref_eepos_r7_logged", False):
                        self._ref_eepos_r7_logged = True
                        print(f"[REF-QVEC-R7] build(): ee_pos hover analog "
                              f"active (w={_W_pee}/axis via J_arm, "
                              f"z_offset={self.ee_target_z_offset:.3f}m) — "
                              f"w_ee_approach={self.w_ee_approach:.0f} "
                              f"proxy block RETIRED under "
                              f"use_reference_q_vector", flush=True)
            elif dist > 1e-3:
                g_hat = v_goal / dist
                # Contact-face proxy: d_push behind object at contact height
                proxy_3d = np.array([
                    obj_xy[0] - self.d_push * g_hat[0],
                    obj_xy[1] - self.d_push * g_hat[1],
                    self.z_ref,
                ])

                # Current EE position
                ee_pos = self.plant.CalcPointsPositions(
                    plant_ctx, self.ee_frame, np.zeros(3), self.world_frame
                ).flatten()

                # Three-stage approach: forces pusher to reach the push-axis
                # BEHIND the box before closing in, preventing corner contact.
                # Stage 1 (dist > 0.25m): target pre-approach, 0.30m behind box
                # Stage 2 (0.10–0.25m):   blend pre_approach → approach_waypoint
                # Stage 3 (< 0.10m):      blend approach_waypoint → contact_proxy
                ee_xy          = ee_pos[:2]
                ee_to_box_dist = float(np.linalg.norm(ee_xy - obj_xy))

                pre_approach_3d = np.array([
                    obj_xy[0] - 0.16 * g_hat[0],
                    obj_xy[1] - 0.16 * g_hat[1],
                    self.z_ref,
                ])
                approach_3d = np.array([
                    obj_xy[0] - (self.d_push + 0.15) * g_hat[0],
                    obj_xy[1] - (self.d_push + 0.15) * g_hat[1],
                    self.z_ref,
                ])

                if ee_to_box_dist > 0.25:
                    effective_proxy = pre_approach_3d.copy()
                    stage = 1
                elif ee_to_box_dist > 0.10:
                    t = (ee_to_box_dist - 0.10) / 0.15
                    effective_proxy = t * pre_approach_3d + (1.0 - t) * approach_3d
                    stage = 2
                else:
                    t = ee_to_box_dist / 0.10
                    effective_proxy = t * approach_3d + (1.0 - t) * proxy_3d
                    stage = 3

                # Close-range lateral alignment: when the pusher is within 0.15m
                # of the box but laterally offset from the push axis, shift the
                # effective proxy toward the axis to prevent corner contact.
                rel_vec        = ee_xy - obj_xy
                along_push     = float(np.dot(rel_vec, g_hat))
                perp_vec       = rel_vec - along_push * g_hat
                perp_magnitude = float(np.linalg.norm(perp_vec))

                # (Cost-bias face-transition machinery removed in the
                #  2026-08-05 CLI prune — cube-era port-only heuristic,
                #  byte-identical OFF since it shipped.)

                if ee_to_box_dist < 0.15 and perp_magnitude > 1e-4:
                    extra_shift = -perp_vec * min(1.0,
                        perp_magnitude / self.lateral_align_full_scale)
                    effective_proxy = effective_proxy.copy()
                    effective_proxy[:2] += extra_shift

                # --- Sanity check: approach waypoint must be BEHIND the box ---
                approach_proj = float(np.dot(approach_3d[:2] - obj_xy, g_hat))
                if approach_proj > 0:
                    print(f"[BUG] Approach waypoint is on the WRONG SIDE of the box!")
                    print(f"      obj_xy={obj_xy}, g_hat={g_hat}, waypoint={approach_3d[:2]}")
                    print(f"      projection onto g_hat = {approach_proj:.4f} (should be negative)")

                # --- Sanity check: contact proxy must also be BEHIND the box ---
                proxy_proj = float(np.dot(proxy_3d[:2] - obj_xy, g_hat))
                if proxy_proj > 0:
                    print(f"[BUG] Proxy is on the WRONG SIDE of the box!")
                    print(f"      proxy={proxy_3d[:2]}, g_hat={g_hat}, projection={proxy_proj:.4f}")

                # --- EE diagnostic ---
                print(f"[EErel] along_push={along_push:+.3f}m (neg=correct)  "
                      f"perp={perp_magnitude:.3f}m  ee_to_box={ee_to_box_dist:.3f}m  "
                      f"stage={stage}  obj={obj_xy.round(3)}  g_hat={g_hat.round(3)}")

                # EE translational velocity Jacobian (3 × n_v)
                J_ee = self.plant.CalcJacobianTranslationalVelocity(
                    plant_ctx, self._ad.JacobianWrtVariable.kV,
                    self.ee_frame, np.zeros(3),
                    self.world_frame, self.world_frame,
                )
                # Arm joints: first n_u columns (revolute → q̇ = v for arm)
                J_arm = J_ee[:, : self.n_u]   # (3, n_u)

                ee_err = effective_proxy - ee_pos  # (3,) desired EE displacement
                print(f"[proxy] err={np.linalg.norm(ee_err):.3f}m  "
                      f"effective={effective_proxy.round(3)}")

                # Damped pseudoinverse: dq_arm = J^T (J J^T + λI)^{-1} ee_err
                lam   = 0.001
                JJT   = J_arm @ J_arm.T + lam * np.eye(3)
                dq    = J_arm.T @ np.linalg.solve(JJT, ee_err)  # (n_u,)

                # Add J^T J block to Q (arm q indices 0..n_u-1)
                w = self.w_ee_approach if not rich_mode else 0.0
                Q[: self.n_u, : self.n_u] += 2.0 * w * (J_arm.T @ J_arm)

                # Shift arm reference toward effective proxy
                if not rich_mode:
                    x_ref[: self.n_u] = current_q[: self.n_u] + dq

                if self._diag_ee_cost:
                    ee2proxy = effective_proxy - ee_pos
                    q_arm = current_q[: self.n_u]
                    q_arm_str = "[" + ",".join(f"{v:+.4f}" for v in q_arm) + "]"
                    print(f"[EE-COST] "
                          f"obj=({obj_xy[0]:+.4f},{obj_xy[1]:+.4f}) "
                          f"ghat=({g_hat[0]:+.4f},{g_hat[1]:+.4f}) "
                          f"ee=({ee_pos[0]:+.4f},{ee_pos[1]:+.4f},{ee_pos[2]:+.4f}) "
                          f"ee_to_box={ee_to_box_dist*1000:.1f}mm "
                          f"stage={stage} "
                          f"proxy=({effective_proxy[0]:+.4f},{effective_proxy[1]:+.4f},{effective_proxy[2]:+.4f}) "
                          f"ee2proxy=({ee2proxy[0]:+.4f},{ee2proxy[1]:+.4f},{ee2proxy[2]:+.4f}) "
                          f"w_ee={w:.0f} "
                          f"q_arm={q_arm_str}",
                          flush=True)

            elif self.w_yaw > 0.0:
                # --- Rotation-task EE-approach (Jin & Posa eq. 40 w1 analog) ---
                # No translation goal exists (dist=0 → g_hat undefined). Pull
                # the EE toward the box CoM at contact height so the arm can
                # reach a torquing contact. Reuses the J_arm-pseudoinverse
                # mechanism from the translation branch verbatim; only the
                # proxy point differs (box CoM, no directional setback) and
                # the perpendicular-velocity penalty is dropped (defined
                # relative to the push axis g_hat, which has no meaning here).
                proxy_3d = np.array([obj_xy[0], obj_xy[1], self.z_ref])

                ee_pos = self.plant.CalcPointsPositions(
                    plant_ctx, self.ee_frame, np.zeros(3), self.world_frame
                ).flatten()

                J_ee = self.plant.CalcJacobianTranslationalVelocity(
                    plant_ctx, self._ad.JacobianWrtVariable.kV,
                    self.ee_frame, np.zeros(3),
                    self.world_frame, self.world_frame,
                )
                J_arm = J_ee[:, : self.n_u]

                ee_err = proxy_3d - ee_pos
                lam   = 0.001
                JJT   = J_arm @ J_arm.T + lam * np.eye(3)
                dq    = J_arm.T @ np.linalg.solve(JJT, ee_err)

                w = self.w_ee_approach if not rich_mode else 0.0
                Q[: self.n_u, : self.n_u] += 2.0 * w * (J_arm.T @ J_arm)

                if not rich_mode:
                    x_ref[: self.n_u] = current_q[: self.n_u] + dq

            # --- Perpendicular box velocity penalty ---
            # Penalise object velocity components orthogonal to the goal direction.
            # Drake floating-body vel layout: [ωx, ωy, ωz, vx, vy, vz]; vx at +3.
            # Skipped for tshape — parity with build_ee_space: the reference
            # sampling_c3plus_options.yaml has no analog; box linear velocity
            # is weighted at 0.05·w_Q = 2.5 in the base q_vector, not
            # amplified by a 10·w_obj_xy heuristic. Retained for box/sphere
            # paths where the heuristic is load-bearing.
            if dist > 1e-3 and self._object_shape != "tshape":
                obj_vx_idx = self.n_q + self._obj_vs + 3   # vx in world frame
                obj_vy_idx = self.n_q + self._obj_vs + 4   # vy in world frame

                # g_hat locally derived — the legacy approach branch that
                # used to define it is skipped under use_reference_q_vector.
                g_hat  = v_goal / dist
                g_perp = np.array([-g_hat[1], g_hat[0]])   # 90° CCW of g_hat
                w_perp = 10.0 * self.w_obj_xy

                # Penalise (v_box · g_perp)^2 = (g_perp[0]*vx + g_perp[1]*vy)^2
                Q[obj_vx_idx, obj_vx_idx] += w_perp * g_perp[0] ** 2
                Q[obj_vy_idx, obj_vy_idx] += w_perp * g_perp[1] ** 2
                Q[obj_vx_idx, obj_vy_idx] += w_perp * g_perp[0] * g_perp[1]
                Q[obj_vy_idx, obj_vx_idx] += w_perp * g_perp[0] * g_perp[1]

        QN = self.w_terminal * Q

        # ---- [MATH.Q] fires ONCE after first full Q/R/QN assembly -----------
        if self._math_diag and not self._q_printed:
            self._q_printed = True

            def _qfmt(v):
                av = abs(v) if v != 0 else 0.0
                return f"{v:.4f}" if (av == 0.0 or 1e-3 <= av <= 1e3) else f"{v:.4e}"

            Q_diag = np.diag(Q)
            R_diag = np.diag(self._R)
            _n_u   = self.n_u
            _n_q   = self.n_q
            _n_x   = self.n_x
            _ps    = self._obj_ps
            _off   = int(np.count_nonzero(Q - np.diag(Q_diag)))
            _is_diag = _off == 0

            print(f"[MATH.Q] Q shape=({_n_x},{_n_x}), "
                  f"{'diagonal' if _is_diag else f'NOT diagonal — {_off} off-diag nonzeros'}:")
            print(f"[MATH.Q]   Q[0:{_n_u}]  = arm joint pos weights "
                  f"(EE-approach augmented  J^T J block)")
            print(f"[MATH.Q]   values: "
                  f"{[_qfmt(v) for v in Q_diag[:_n_u]]}")
            # Object quaternion + position block
            _obj_labels = [
                f"{_n_u}(qw)=0",
                f"{_n_u+1}(qx/roll)={_qfmt(Q_diag[_n_u+1])}",
                f"{_n_u+2}(qy/pitch)={_qfmt(Q_diag[_n_u+2])}",
                f"{_n_u+3}(qz/yaw)={_qfmt(Q_diag[_n_u+3])}",
                f"{_n_u+4}(obj_x)={_qfmt(Q_diag[_n_u+4])}",
                f"{_n_u+5}(obj_y)={_qfmt(Q_diag[_n_u+5])}",
                f"{_n_u+6}(obj_z)={_qfmt(Q_diag[_n_u+6])}",
            ]
            print(f"[MATH.Q]   Q[{_n_u}:{_n_q}] = object quaternion+pos:  "
                  + "  ".join(_obj_labels))
            print(f"[MATH.Q]   Q[{_n_q}:{_n_q+_n_u}] = arm joint vel weights: "
                  f"{[_qfmt(v) for v in Q_diag[_n_q:_n_q+_n_u]]}")
            _obj_v_base = _n_q + _n_u
            print(f"[MATH.Q]   Q[{_obj_v_base}:{_n_x}] = object vel weights "
                  f"(perp-vel penalty on vx/vy): "
                  f"{[_qfmt(v) for v in Q_diag[_obj_v_base:]]}")
            print(f"[MATH.Q] R diagonal (n_u={_n_u}): "
                  f"{[_qfmt(v) for v in R_diag]}")
            print(f"[MATH.Q] QN = w_terminal·Q  "
                  f"(w_terminal={self.w_terminal:.1f}), "
                  f"||QN||_F={np.linalg.norm(QN):.4e}")
            if not _is_diag:
                print(f"[MATH.Q] Off-diagonal note: EE-approach adds J^T J "
                      f"block to Q[0:{_n_u},0:{_n_u}]; "
                      f"perp-vel adds cross-terms to Q[vx,vy]")

        return Q, self._R, QN, x_ref

    # ==================================================================
    # EE-space cost (paper-aligned, Stage C of the EE-space rewrite).
    # ==================================================================
    # State layout in the new LCS (must match LCSFormulator.*_SLOT):
    #     x = [box_q (7=quat+pos), p_ee (3), box_v (6=omega+lin), v_ee (3)]
    # Indices (absolute):
    #     box_q: 0..6   (qw=0, qx=1, qy=2, qz=3, x=4, y=5, z=6)
    #     p_ee : 7..9
    #     box_v: 10..15 (ωx=10, ωy=11, ωz=12, vx=13, vy=14, vz=15)
    #     v_ee : 16..18
    #
    # Key simplification vs R^7 path: p_ee is a STATE variable. The
    # EE-approach cost is therefore a direct quadratic on the p_ee slot
    # (Q[7:10, 7:10] = w_ee_approach · I_3, x_ref[7:10] = effective_proxy).
    # No arm Jacobian needed — that mapping happens downstream in the OSC.

    N_X_EE_SPACE = 19
    N_U_EE_SPACE = 3
    _NEW_OBJ_QW   = 0
    _NEW_OBJ_QX   = 1
    _NEW_OBJ_QY   = 2
    _NEW_OBJ_QZ   = 3
    _NEW_OBJ_X    = 4
    _NEW_OBJ_Y    = 5
    _NEW_OBJ_Z    = 6
    _NEW_PEE_SLOT = slice(7, 10)
    _NEW_VBOX_OMEGA = slice(10, 13)
    _NEW_VBOX_LIN_X = 13
    _NEW_VBOX_LIN_Y = 14
    _NEW_VBOX_LIN_Z = 15
    _NEW_VEE_SLOT   = slice(16, 19)

    def build_ee_space(self, target_xy: np.ndarray,
                       plant_ctx=None, current_q: np.ndarray = None,
                       target_yaw: float = 0.0,
                       target_quat=None):
        """
        Return (Q, R, QN, x_ref) for the EE-space LCS.

        Same cost components as build() but in low-dim coords:
          - object xy goal     : Q[4,4] = Q[5,5] = w_obj_xy
          - object z upright   : Q[6,6] = w_obj_z + w_box_z
          - box roll/pitch     : Q[1,1] = Q[2,2] = w_box_rp
          - yaw target         : w_yaw outer product on Q[0:4, 0:4]
          - EE-approach        : Q[7:10, 7:10] = w_ee_approach · I_3,
                                 x_ref[7:10] = effective_proxy (3-stage)
          - perp box velocity  : penalize box-linear-vx/vy components
                                 orthogonal to g_hat (Q[13:15, 13:15])

        Returns
        -------
        Q       : (19, 19)
        R       : (3, 3)  =  w_torque · I_3   (now a force-cost, not torque)
        QN      : (19, 19)
        x_ref   : (19,)
        """
        n_x = self.N_X_EE_SPACE
        n_u = self.N_U_EE_SPACE

        # --- Base Q ---
        Q = np.zeros((n_x, n_x))
        if self.use_reference_q_vector:
            # Reference-conforming path: Q_base = w_Q · diag(q_vector) with the
            # per-group vectors placed at the port's state indices. Reference
            # layout is [ee_pos(3), quat(4), obj_pos(3), ee_vel(3),
            # obj_ang_vel(3), obj_lin_vel(3)]; port layout is [quat(4),
            # obj_pos(3), ee_pos(3), obj_ang_vel(3), obj_lin_vel(3), ee_vel(3)].
            q_diag = np.zeros(n_x)
            q_diag[self._NEW_OBJ_QW:self._NEW_OBJ_QZ + 1] = self._q_vec_obj_quat
            q_diag[self._NEW_OBJ_X]  = self._q_vec_obj_pos[0]
            q_diag[self._NEW_OBJ_Y]  = self._q_vec_obj_pos[1]
            q_diag[self._NEW_OBJ_Z]  = self._q_vec_obj_pos[2]
            q_diag[self._NEW_PEE_SLOT]   = self._q_vec_ee_pos
            q_diag[self._NEW_VBOX_OMEGA] = self._q_vec_obj_ang_vel
            q_diag[self._NEW_VBOX_LIN_X] = self._q_vec_obj_lin_vel[0]
            q_diag[self._NEW_VBOX_LIN_Y] = self._q_vec_obj_lin_vel[1]
            q_diag[self._NEW_VBOX_LIN_Z] = self._q_vec_obj_lin_vel[2]
            q_diag[self._NEW_VEE_SLOT]   = self._q_vec_ee_vel
            Q[np.arange(n_x), np.arange(n_x)] = self.w_Q * q_diag
        else:
            # Near-goal (pose regime) obj-position weight swap — parity with
            # legacy build() at :336-340. Reference push_t
            # sampling_c3plus_options.yaml has q_vector_position (far) and
            # q_vector (near) with obj_pos slots dropping 250→200 (×0.80 for
            # xy) and 250→120 (×0.48 for z) when
            # crossed_cost_switching_threshold_ latches. The port's
            # build_ee_space previously fired the quaternion-Hessian override
            # but left obj_pos weights untouched, so the yaw Hessian had to
            # compete against a still-huge xy penalty. Wiring the swap here
            # matches the reference mechanism; the swap is a no-op when
            # cost config sets w_obj_xy_pose == w_obj_xy (backward-compat).
            if self._crossed_switching_threshold:
                Q[self._NEW_OBJ_X, self._NEW_OBJ_X] = self.w_obj_xy_pose
                Q[self._NEW_OBJ_Y, self._NEW_OBJ_Y] = self.w_obj_xy_pose
                Q[self._NEW_OBJ_Z, self._NEW_OBJ_Z] = (self.w_obj_z
                                                       + self.w_box_z_pose)
            else:
                Q[self._NEW_OBJ_X, self._NEW_OBJ_X] = self.w_obj_xy
                Q[self._NEW_OBJ_Y, self._NEW_OBJ_Y] = self.w_obj_xy
                Q[self._NEW_OBJ_Z, self._NEW_OBJ_Z] = (self.w_obj_z
                                                       + self.w_box_z)
            Q[self._NEW_OBJ_QX, self._NEW_OBJ_QX] = self.w_box_rp   # roll
            Q[self._NEW_OBJ_QY, self._NEW_OBJ_QY] = self.w_box_rp   # pitch

        # --- x_ref base ---
        x_ref = np.zeros(n_x)
        x_ref[self._NEW_OBJ_X] = target_xy[0]
        x_ref[self._NEW_OBJ_Y] = target_xy[1]
        x_ref[self._NEW_OBJ_Z] = self.z_obj_target

        # --- Reference EE-position tracking (goal_generator.cc:113-114) ---
        # Reference build: x_lcs_des[0:3] = obj_curr_xyz + [0, 0, z_offset]
        # (see franka_sampling_c3_controller.cc:440 target_state_mux
        # port 0 → ee_target from CalcEndEffectorTarget).
        # Port state layout has EE-pos at slots 7:10 (reference layout has
        # it at 0:3). Weight matches reference q_vector[0:3] * w_Q
        # (default 0.01 * w_Q). Only activates when
        # `ee_target_z_offset_above_object > 0` in cost config.
        if (self.ee_target_z_offset > 0.0
                and plant_ctx is not None
                and current_q is not None):
            _obj_curr_z = float(current_q[self._obj_z_idx])
            x_ref[self._NEW_PEE_SLOT] = np.array([
                float(current_q[self._obj_x_idx]),
                float(current_q[self._obj_y_idx]),
                _obj_curr_z + self.ee_target_z_offset,
            ])
            # Reference weight: w_Q * q_vector[0:3] = 50 * 0.01 = 0.5 per axis.
            # Use the port's `q_vec_ee_pos * w_Q` (defaults [0.01,0.01,0.01]*45=0.45)
            # to match reference structure. Skip when `use_reference_q_vector`
            # is set — the base Q assignment at line 796 already writes this
            # exact weight into Q[7:10], so adding again would double-count.
            if not self.use_reference_q_vector:
                _w_ee = float(self.w_Q) * np.asarray(self._q_vec_ee_pos, dtype=float)
                for _i, _slot in enumerate(range(7, 10)):
                    Q[_slot, _slot] += _w_ee[_i]

        # --- Yaw target ---
        # Reference has no rank-1 outer-product yaw form; yaw is entirely
        # handled by the near-goal quaternion Hessian override (line ~886).
        # Always seed x_ref quaternion from target_yaw so BOTH the (optional)
        # port-only linear form and the reference Hessian form have a proper
        # goal-quat reference — the Hessian cost `(x - x_ref)^T Q_quat (x - x_ref)`
        # only drives x toward q_goal if x_ref = q_goal.
        self._target_yaw = float(target_yaw)
        a_half = 0.5 * self._target_yaw
        # Full goal quaternion: seeds ALL FOUR slots, not just qw/qz. A tilted
        # goal (the jack) has nonzero qx/qy; leaving those at 0 would make the
        # quaternion Hessian pull toward a non-unit, non-reachable reference.
        _q_goal = _goal_quaternion(
            target_yaw, target_quat if target_quat is not None else self._goal_quat)
        x_ref[self._NEW_OBJ_QW:self._NEW_OBJ_QW + 4] = _q_goal
        # Optional linear rank-1 form — port-only far-goal add-on. Set
        # w_yaw:0 in tasks.yaml to match reference (Hessian-only) behavior.
        if self.w_yaw > 0.0 and not self.use_reference_q_vector:
            cy = np.array([-np.sin(a_half), 0.0, 0.0, np.cos(a_half)])
            Q[0:4, 0:4] += self.w_yaw * np.outer(cy, cy)

        # --- Near-goal quaternion-dependent Q_block override ---
        # Reference sampling_based_c3_controller.cc:1517-1570.
        # When (a) `use_quaternion_dependent_cost` is set in cost config AND
        # (b) the wrapper's `crossed_switching_threshold` is True (object is
        # inside `cost_switching_threshold_distance` of the goal), REPLACE
        # (not add to) the linear w_yaw quat block with the Hessian-of-
        # squared-quaternion-angle-difference cost. Guides yaw via the
        # proper 3D rotation metric near goal, where the linear residual
        # saturates.
        if (self.use_quaternion_dependent_cost
                and self._crossed_switching_threshold
                and plant_ctx is not None
                and current_q is not None):
            # Import lazily to keep this optional path zero-cost when disabled.
            from control.quaternion_hessian import (
                build_regularized_quaternion_cost,
            )
            # Current object quaternion (positions ps..ps+3 in q).
            ps = self._obj_ps
            q_now = np.array([
                current_q[ps], current_q[ps + 1],
                current_q[ps + 2], current_q[ps + 3],
            ])
            # Goal quaternion: yaw-only for planar tasks, full quaternion
            # when the task supplies goal_quat (see _goal_quaternion).
            q_goal = _goal_quaternion(
                target_yaw,
                target_quat if target_quat is not None else self._goal_quat)
            Q_quat = build_regularized_quaternion_cost(
                q_now, q_goal,
                weight=self.q_quaternion_dependent_weight,
                regularizer_fraction=
                    self.q_quaternion_dependent_regularizer_fraction,
            )
            # Reference replaces (=) rather than adds. Do the same to avoid
            # double-counting yaw cost (linear + hessian).
            Q[0:4, 0:4] = Q_quat
            # Keep x_ref quat set to goal (both cost forms reference the same
            # goal quaternion in slot 0/3).

        # --- EE-approach cost (DIRECT — no arm Jacobian) ---
        if plant_ctx is not None and current_q is not None:
            obj_xy = np.array([current_q[self._obj_x_idx],
                               current_q[self._obj_y_idx]])
            v_goal = target_xy - obj_xy
            dist   = float(np.linalg.norm(v_goal))

            if dist > 1e-3:
                g_hat = v_goal / dist
                # Current EE position (used to pick approach stage; in the new
                # LCS this is also a state variable, but for cost-building we
                # read it from the plant context).
                ee_pos = self.plant.CalcPointsPositions(
                    plant_ctx, self.ee_frame, np.zeros(3), self.world_frame
                ).flatten()
                ee_xy          = ee_pos[:2]
                ee_to_box_dist = float(np.linalg.norm(ee_xy - obj_xy))

                # PORT_BOX_DPUSH_FIX (2026-07-09) — d_push is a SPHERE-CENTER
                # target, but the yaml value (0.05 for box tasks) equals the
                # box half-extent, placing the sphere-CENTER on the box face
                # plane → sphere-SURFACE penetrates INTO the box by the tip
                # radius (25 mm for the box canonical) → Drake resists with a
                # huge impulse → §7.31 hammer-blow tumble. Geometry-derived
                # override: sphere-CENTER at box_half + pusher_radius + 0.5 mm
                # so sphere-SURFACE sits 0.5 mm off the box face — mirroring
                # the reference sample_projection_clearance=0.02 semantics
                # (20 mm center offset − 19.5 mm tip radius = 0.5 mm surface
                # gap). Default-OFF. Applies to object_shape="box" only —
                # T-shape samples through _TSHAPE_FACE_TABLE + setback=0.020
                # (already 0.5 mm gap on 19.5 mm interpretation) and is
                # separately validated at 75.5 % closure.
                import os as _os_dpush
                _use_dpush_fix = (
                    _os_dpush.environ.get("PORT_BOX_DPUSH_FIX", "0") == "1"
                    and self._object_shape == "box"
                    and self._object_half_extent is not None
                    and self._pusher_radius is not None
                )
                if _use_dpush_fix:
                    _d_push_eff = (float(self._object_half_extent)
                                   + float(self._pusher_radius)
                                   + 0.0005)
                    if not getattr(self, "_dpush_fix_logged", False):
                        print(f"[PORT_BOX_DPUSH_FIX] enabled: "
                              f"d_push={self.d_push:.4f} → {_d_push_eff:.4f} m "
                              f"(box_half={self._object_half_extent:.4f}, "
                              f"pusher_r={self._pusher_radius:.4f}, "
                              f"+0.5mm clearance)", flush=True)
                        self._dpush_fix_logged = True
                else:
                    _d_push_eff = self.d_push

                # Three-stage approach proxy (same as R^7 build()).
                proxy_3d = np.array([
                    obj_xy[0] - _d_push_eff * g_hat[0],
                    obj_xy[1] - _d_push_eff * g_hat[1],
                    self.z_ref,
                ])
                pre_approach_3d = np.array([
                    obj_xy[0] - 0.16 * g_hat[0],
                    obj_xy[1] - 0.16 * g_hat[1],
                    self.z_ref,
                ])
                approach_3d = np.array([
                    obj_xy[0] - (self.d_push + 0.15) * g_hat[0],
                    obj_xy[1] - (self.d_push + 0.15) * g_hat[1],
                    self.z_ref,
                ])
                # §7.46 — when PORT_EE_APPROACH_FACE_TARGET=1, re-target the
                # planner-cost proxy to the contact-face point (proxy_3d =
                # obj_xy − d_push·g_hat) directly, bypassing the 3-stage
                # pre_approach→approach→proxy blend. §7.45 confirmed the
                # blend's backed-off staging puts effective_proxy ~90mm EAST
                # of the EE at first c3 entry (West push) → planner u_x
                # +x = wrong sign. proxy_3d sits +d_push WEST of the box,
                # which is +0.03m WEST of the EE at the §7.42/45 geometry
                # → predicted u_x flips negative.
                # Default-OFF byte-identical (else-branch keeps the 3-stage
                # blend). Affects ONLY the planner cost (build_ee_space);
                # IK/reposition/APPROACH-OVERRIDE path is independent.
                import os as _os_face
                if (_os_face.environ.get(
                        "PORT_EE_APPROACH_FACE_TARGET", "0") == "1"):
                    effective_proxy = proxy_3d.copy()
                    if not getattr(self, "_face_target_logged", False):
                        print(f"[§7.46] PORT_EE_APPROACH_FACE_TARGET=1 "
                              f"effective_proxy=proxy_3d="
                              f"({proxy_3d[0]:+.3f},{proxy_3d[1]:+.3f},"
                              f"{proxy_3d[2]:+.3f}) "
                              f"ee_to_box={ee_to_box_dist:.3f}m "
                              f"(staging blend bypassed)", flush=True)
                        self._face_target_logged = True
                elif ee_to_box_dist > 0.25:
                    effective_proxy = pre_approach_3d.copy()
                elif ee_to_box_dist > 0.10:
                    t = (ee_to_box_dist - 0.10) / 0.15
                    effective_proxy = t * pre_approach_3d + (1.0 - t) * approach_3d
                else:
                    t = ee_to_box_dist / 0.10
                    effective_proxy = t * approach_3d + (1.0 - t) * proxy_3d

                # Close-range lateral alignment toward push axis.
                rel_vec        = ee_xy - obj_xy
                along_push     = float(np.dot(rel_vec, g_hat))
                perp_vec       = rel_vec - along_push * g_hat
                perp_magnitude = float(np.linalg.norm(perp_vec))
                if ee_to_box_dist < 0.15 and perp_magnitude > 1e-4:
                    _scale = self.lateral_align_full_scale
                    _strength = min(1.0, perp_magnitude / _scale)
                    extra_shift = -perp_vec * _strength
                    effective_proxy = effective_proxy.copy()
                    effective_proxy[:2] += extra_shift
                    # B-fix diagnostic: only emit when correction is non-trivial
                    # (>10% of full) to keep log volume bounded.
                    if _strength > 0.10:
                        print(f"[LATERAL] perp={perp_magnitude*1000:.1f}mm "
                              f"scale={_scale*1000:.1f}mm "
                              f"strength={_strength:.2f} "
                              f"shift_mm=({extra_shift[0]*1000:+.1f},"
                              f"{extra_shift[1]*1000:+.1f})", flush=True)

                # DIRECT EE-approach cost on the p_ee state slot — SKIPPED.
                # §7.31: with the always-on EE-BOX row (PORT_LCS_ALWAYS_ON_EE_BOX=1)
                # keeping D ≠ 0, the proxy's anti-freeze role is unnecessary
                # and the reference (sampling_based_c3_controller.cc:500,
                # x_desired = GetDesiredState — the sampled face point in
                # both modes) has no equivalent backward-pull cost.
                # Formerly gated by REF_RECONCILE_APPROACH; now always off.
                import os as _os_rec

                # §7.76 (ii) — cost-side z-hold penalty on the plan's
                # EE-z, default-OFF. Breaks the §7.75 tip→plan-z-rise→
                # sphere-climb→more-tip loop plan-consistently: the plan
                # naturally prefers a lower x_seq[k][9], so ADMM's inner
                # solve discovers a lower-z push trajectory and the OSC
                # receives a low z-target (no plan/OSC tension). Design
                # in experiments/§7.75c_repro/REPORT.md:117-133.
                # Target = sampling_height by default (contact-plane EE z).
                _z_hold_on = _os_rec.environ.get("PORT_EE_Z_HOLD", "0") == "1"
                if _z_hold_on:
                    _w_zh = float(_os_rec.environ.get(
                        "PORT_EE_Z_HOLD_W", "1000.0"))
                    # Default target = self.z_ref (cost_cfg["z_ee_target"],
                    # 0.05 m for cube; matches sampling_height contact-plane).
                    _z_tgt = float(_os_rec.environ.get(
                        "PORT_EE_Z_HOLD_TARGET", str(self.z_ref)))
                    Q[9, 9] += _w_zh
                    x_ref[9] = _z_tgt
                    if not getattr(self, "_z_hold_logged", False):
                        print(
                            f"[§7.76-Z-HOLD] PORT_EE_Z_HOLD=1  "
                            f"w_zh={_w_zh}  z_tgt={_z_tgt:.4f}m",
                            flush=True,
                        )
                        self._z_hold_logged = True

            # --- Perpendicular box-velocity penalty ---
            # Penalize box linear velocity components orthogonal to g_hat.
            # Skipped for tshape (push_t): reference sampling_c3plus_options.yaml
            # has no analog — box linear velocity is weighted at 0.05·w_Q=50=2.5
            # in the base q_vector, not amplified by a 10·w_obj_xy=1M term.
            # Retained for box/sphere paths where the heuristic is load-bearing.
            if dist > 1e-3 and self._object_shape != "tshape":
                g_hat = v_goal / dist
                g_perp = np.array([-g_hat[1], g_hat[0]])
                w_perp = 10.0 * self.w_obj_xy
                ix = self._NEW_VBOX_LIN_X
                iy = self._NEW_VBOX_LIN_Y
                Q[ix, ix] += w_perp * g_perp[0] ** 2
                Q[iy, iy] += w_perp * g_perp[1] ** 2
                Q[ix, iy] += w_perp * g_perp[0] * g_perp[1]
                Q[iy, ix] += w_perp * g_perp[0] * g_perp[1]

        # --- R: torque cost is now an EE-force cost. Same scalar weight,
        #     applied to R^3 input. ---
        # Stage 5 per-axis R override (env-gated, default-inert). When
        # PORT_R_VECTOR is set as "rx,ry,rz" (e.g., "0.01,0.01,1"),
        # use np.diag([rx,ry,rz]); else scalar w_torque*I (bit-identical to
        # pre-Stage-5).
        import os as _os
        _r_s = _os.environ.get("PORT_R_VECTOR", "")
        if _r_s and n_u == 3:
            try:
                _rv = [float(x) for x in _r_s.split(",")]
                if len(_rv) == 3:
                    R = np.diag(_rv)
                else:
                    R = self.w_torque * np.eye(n_u)
            except ValueError:
                R = self.w_torque * np.eye(n_u)
        else:
            R = self.w_torque * np.eye(n_u)

        # Terminal weight: QN = w_terminal · Q, always. Caller drives shape
        # via config/tasks.yaml.
        #   w_terminal=1.0 → QN=Q → matches reference behavior for tshape
        #     (sampling_based_c3_controller.cc:1507-1515 UpdateCostMatrices,
        #     gamma=1.0 → Q_[N] = Q_[N-1]).
        #   w_terminal>1.0 → terminal-knot amplifier for tight-goal tuning;
        #     off-reference. Explicit opt-in via yaml keeps the reference
        #     conformance testable — set w_terminal:1.0 to recover reference.
        # 2026-07-20: former `if _object_shape == 'tshape': QN=Q` hardcode
        # removed; the T-push YAML w_terminal (was dead) is now live. Kept
        # the box/sphere/cube path effectively unchanged since their YAMLs
        # keep the pre-existing w_terminal scaling.
        QN = self.w_terminal * Q
        return Q, R, QN, x_ref