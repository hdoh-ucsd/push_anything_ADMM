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
                 cost_bias: bool = False):
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
        self.w_obj_z       = float(c.get("w_obj_z",         10.0))
        self.w_box_z       = float(c.get("w_box_z",        100.0))
        self.w_box_rp      = float(c.get("w_box_rp",        50.0))
        # Yaw-error penalty (Jin & Posa eq. 40 analog). w_yaw=0 → fully inert.
        # Residual e_z = c_yaw · q_box where c_yaw = [-sin(α/2),0,0,cos(α/2)]
        # is EXACTLY linear in the quaternion (not a small-angle approximation):
        # for an upright yaw-only configuration q_box = [cos(ψ/2),0,0,sin(ψ/2)],
        # e_z = sin((ψ-α)/2) — the standard quaternion half-angle metric,
        # globally monotonic on ψ ∈ (α-π, α+π). NOT a raw qz penalty, which is
        # invalid for a unit quaternion.
        self.w_yaw         = float(c.get("w_yaw",            0.0))
        self._target_yaw   = 0.0   # updated each build() call via target_yaw kwarg
        self.w_torque      = float(c.get("w_torque",         0.01))
        self.w_terminal    = float(c.get("w_terminal",        5.0))
        self.z_ref         = float(c.get("z_ee_target",      0.05))
        self.d_push        = float(c.get("d_push",           0.10))
        self.w_ee_approach = float(c.get("w_ee_approach",   800.0))
        # Lateral-alignment clamp scale (metres). The proxy shift at
        # build_ee_space()::~739 (and the legacy build()::~438) saturates at
        # `extra_shift = -perp_vec * min(1.0, perp_magnitude / scale)`. Smaller
        # scale = more responsive correction (full strength at smaller
        # off-equator distance). Default 0.05 preserves legacy behavior;
        # pushing/tasks.yaml pins the operational value. See plan
        # docs/superpowers/plans/2026-06-07-B-lateral-align-clamp-harden.md.
        self.lateral_align_full_scale = float(
            c.get("lateral_align_full_scale", 0.05))

        self._math_diag = math_diag
        self._q_printed = False

        # Cost-bias state (all variables inactive when cost_bias=False)
        self._cost_bias         = cost_bias
        self._bias_phase        = 'PUSH'  # 'PUSH' | 'LIFT' | 'APPROACH'
        self._bias_counter      = 0       # steps elapsed in current timed phase
        self._bias_step         = 0       # total build() calls (for [BIAS] diagnostic)
        self._bias_prev_obj_xy  = None    # obj_xy at previous step for delta progress
        self._bias_progress_buf = []      # per-step goal-aligned box progress samples
        self._bias_last_face    = None    # last correct face label ('E','W','N','S')
        self._bias_face_init    = False   # one-time initial wrong-face check done

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

    def build(self, target_xy: np.ndarray,
              plant_ctx=None, current_q: np.ndarray = None,
              rich_mode: bool = False,
              target_yaw: float = 0.0):
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
        x_ref = np.zeros(self.n_x)
        x_ref[self._obj_x_idx] = target_xy[0]
        x_ref[self._obj_y_idx] = target_xy[1]
        x_ref[self._obj_z_idx] = self.z_ref

        # --- Yaw-target cost (Jin & Posa eq. 40 analog) ---
        # Add w_yaw · (c_yaw · (q_box - q_goal))² on the box quaternion slots.
        # c_yaw = [-sin(α/2), 0, 0, cos(α/2)] is linear in [qw,qx,qy,qz] and
        # equals the z-component of q_goal⁻¹ ⊗ q_box (vector part of the
        # relative quaternion). For upright box: c_yaw · q_box = sin((ψ-α)/2).
        # The xy components of c_yaw are zero, so this is orthogonal to the
        # existing w_box_rp roll/pitch regularization.
        self._target_yaw = float(target_yaw)
        if self.w_yaw > 0.0:
            a_half = 0.5 * self._target_yaw
            cy = np.array([-np.sin(a_half), 0.0, 0.0, np.cos(a_half)])
            ps = self._obj_ps
            # Outer product on the (qw..qz) 4×4 sub-block of Q
            Q[ps:ps+4, ps:ps+4] += self.w_yaw * np.outer(cy, cy)
            # Set the goal quaternion reference on the qw and qz slots.
            # qx,qy stay at 0 (upright), preserving w_box_rp behavior since
            # x_ref[qx]=x_ref[qy]=0 makes that block unchanged.
            x_ref[ps + 0] = np.cos(a_half)   # qw
            x_ref[ps + 3] = np.sin(a_half)   # qz

        # --- Linearised EE approach cost (arm joints only) ---
        if plant_ctx is not None and current_q is not None:
            obj_xy  = np.array([current_q[self._obj_x_idx],
                                 current_q[self._obj_y_idx]])
            v_goal  = target_xy - obj_xy
            dist    = np.linalg.norm(v_goal)

            if dist > 1e-3:
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

                # --- Cost-bias heuristic (face-transition, Phase 2 design) --------
                # Guarded by self._cost_bias; when False, output is byte-identical to
                # the baseline (no state is read or written from this block).
                # Overrides effective_proxy in LIFT/APPROACH phases; PUSH is unchanged.
                # Inserted BEFORE lateral alignment so alignment always applies.
                # Out of scope: shepherding (sphere has no faces — bias is a no-op).
                if self._cost_bias:
                    self._bias_step += 1

                    # --- Constants (all derived from Phase 1 data; see design memo) ---
                    # Z_LIFT: box_top(0.10m) + pusher_radius(0.025m) + distance_threshold(0.10m)
                    # Must exceed distance_threshold above box_top so no phantom contacts fire.
                    _Z_LIFT       = 0.225
                    # N_I: E1 showed ~50 steps to lift clear; 60 = 1.2× observed margin
                    _N_I          = 60
                    # N_II: E1 far-approach ~100 steps before contact; matched here
                    _N_II         = 100
                    # N_STALL: 100-step window (1.0 s at dt_osc=0.01s)
                    _N_STALL      = 100
                    # STALL_THRESH: 0.003m cumulative goal progress in N_STALL steps;
                    # E1 active push ≈106mm in 100 steps >> 3mm; plateau ≈1mm < 3mm
                    _STALL_THRESH = 0.003
                    # CONTACT_PROX: 3D dist threshold to consider face check relevant
                    _CONTACT_PROX = 0.12
                    # FACE_ALIGN: cos(60°); EE must be within 60° of correct face normal
                    _FACE_ALIGN   = 0.5
                    # Box geometry for 3D distance (pushes task is axis-aligned box)
                    _BOX_HALF_Z   = 0.05
                    _PUSHER_R     = 0.025

                    # Correct face: axis-aligned face most anti-aligned to g_hat.
                    # The EE should push against this face to move the box toward goal.
                    _face_candidates = [
                        ('E', np.array([1., 0.])),
                        ('W', np.array([-1., 0.])),
                        ('N', np.array([0., 1.])),
                        ('S', np.array([0., -1.])),
                    ]
                    _best = min(_face_candidates,
                                key=lambda fn: float(np.dot(fn[1], g_hat)))
                    _correct_face, _n_correct = _best[0], _best[1]

                    # 3D EE-to-box distance: captures the phantom-contact scenario
                    # (E1 plateau) where EE is above box top but 2D ee_to_box ≈ 0.04m.
                    _obj_z      = current_q[self._obj_z_idx]
                    _box_top    = _obj_z + _BOX_HALF_Z
                    _ee_z_above = max(0.0, ee_pos[2] - (_box_top + _PUSHER_R))
                    _ee_box_3d  = float(np.sqrt(ee_to_box_dist**2 + _ee_z_above**2))

                    # Goal-aligned progress tracking (raw Δobj_xy·g_hat per step).
                    # Using |Δobj_xy| not F·g_hat — avoids the zero-threshold bug
                    # where F·g_hat=0.000 is printed as "→goal ✓" (see Phase 1).
                    if self._bias_prev_obj_xy is not None:
                        _delta = float(np.dot(obj_xy - self._bias_prev_obj_xy, g_hat))
                        self._bias_progress_buf.append(max(0.0, _delta))
                        if len(self._bias_progress_buf) > _N_STALL:
                            self._bias_progress_buf.pop(0)
                    self._bias_prev_obj_xy = obj_xy.copy()
                    _buf_full        = len(self._bias_progress_buf) >= _N_STALL
                    _progress_sum    = sum(self._bias_progress_buf)
                    _progress_recent = _buf_full and (_progress_sum >= _STALL_THRESH)

                    # Face-change detection: fires when the correct face changes
                    # (goal direction rotated — e.g., box overshot or displaced).
                    _face_changed = (self._bias_last_face is not None and
                                     self._bias_last_face != _correct_face)
                    self._bias_last_face = _correct_face

                    # Initial wrong-face check (fires once at first build() call).
                    # If EE is already in contact-range of a wrong face, start LIFT
                    # immediately rather than waiting N_STALL steps for stall detection.
                    if not self._bias_face_init:
                        self._bias_face_init = True
                        if (self._bias_phase == 'PUSH' and
                                _ee_box_3d < _CONTACT_PROX and
                                ee_to_box_dist > 1e-4):
                            _rel_hat = (ee_xy - obj_xy) / (ee_to_box_dist + 1e-9)
                            if float(np.dot(_rel_hat, _n_correct)) < _FACE_ALIGN:
                                self._bias_phase   = 'LIFT'
                                self._bias_counter = 0

                    # Advance counter for timed phases (counter ticks on entry step).
                    if self._bias_phase in ('LIFT', 'APPROACH'):
                        self._bias_counter += 1

                    # State transitions
                    if self._bias_phase == 'PUSH':
                        if _face_changed or (_buf_full and not _progress_recent):
                            self._bias_phase   = 'LIFT'
                            self._bias_counter = 0
                    elif self._bias_phase == 'LIFT':
                        if self._bias_counter >= _N_I:
                            self._bias_phase   = 'APPROACH'
                            self._bias_counter = 0
                    else:  # APPROACH
                        if self._bias_counter >= _N_II:
                            self._bias_phase        = 'PUSH'
                            self._bias_counter      = 0
                            self._bias_progress_buf = []  # flush stale samples so stall detection needs a full N_STALL window

                    # Override effective_proxy (PUSH: unchanged from three-stage logic)
                    if self._bias_phase == 'LIFT':
                        effective_proxy = np.array([obj_xy[0], obj_xy[1], _Z_LIFT])
                    elif self._bias_phase == 'APPROACH':
                        effective_proxy = approach_3d.copy()

                    print(f"[BIAS] step={self._bias_step} "
                          f"phase={self._bias_phase} "
                          f"face={_correct_face} "
                          f"target={np.round(effective_proxy, 3).tolist()} "
                          f"progress_recent={_progress_recent} "
                          f"face_changed={_face_changed}")
                # --- end cost-bias block ---

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
            if dist > 1e-3:
                obj_vx_idx = self.n_q + self._obj_vs + 3   # vx in world frame
                obj_vy_idx = self.n_q + self._obj_vs + 4   # vy in world frame

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
                       target_yaw: float = 0.0):
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

        # --- Base Q (object xy/z, roll/pitch) ---
        Q = np.zeros((n_x, n_x))
        Q[self._NEW_OBJ_X, self._NEW_OBJ_X] = self.w_obj_xy
        Q[self._NEW_OBJ_Y, self._NEW_OBJ_Y] = self.w_obj_xy
        Q[self._NEW_OBJ_Z, self._NEW_OBJ_Z] = self.w_obj_z + self.w_box_z
        Q[self._NEW_OBJ_QX, self._NEW_OBJ_QX] = self.w_box_rp   # roll
        Q[self._NEW_OBJ_QY, self._NEW_OBJ_QY] = self.w_box_rp   # pitch

        # --- x_ref base ---
        x_ref = np.zeros(n_x)
        x_ref[self._NEW_OBJ_X] = target_xy[0]
        x_ref[self._NEW_OBJ_Y] = target_xy[1]
        x_ref[self._NEW_OBJ_Z] = self.z_ref

        # --- Yaw target (linear-in-quaternion residual) ---
        self._target_yaw = float(target_yaw)
        if self.w_yaw > 0.0:
            a_half = 0.5 * self._target_yaw
            cy = np.array([-np.sin(a_half), 0.0, 0.0, np.cos(a_half)])
            Q[0:4, 0:4] += self.w_yaw * np.outer(cy, cy)
            x_ref[self._NEW_OBJ_QW] = np.cos(a_half)
            x_ref[self._NEW_OBJ_QZ] = np.sin(a_half)

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

                # Three-stage approach proxy (same as R^7 build()).
                proxy_3d = np.array([
                    obj_xy[0] - self.d_push * g_hat[0],
                    obj_xy[1] - self.d_push * g_hat[1],
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
                # §7.46 — when PUSHA_EE_APPROACH_FACE_TARGET=1, re-target the
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
                        "PUSHA_EE_APPROACH_FACE_TARGET", "0") == "1"):
                    effective_proxy = proxy_3d.copy()
                    if not getattr(self, "_face_target_logged", False):
                        print(f"[§7.46] PUSHA_EE_APPROACH_FACE_TARGET=1 "
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

                # DIRECT EE-approach cost on the p_ee state slot.
                # No arm Jacobian, no J^T J block — paper-aligned.
                # §7.31 — proxy off: when REF_RECONCILE_APPROACH is set
                # AND the always-on EE-BOX row is enabled (LCS row keeps
                # D ≠ 0 so the proxy's anti-freeze role is unnecessary),
                # skip this block. The reference has no equivalent
                # backward-pull cost (sampling_based_c3_controller.cc:500
                # x_desired = GetDesiredState — the sampled face point in
                # both modes, NO 100 mm-behind term).
                import os as _os_rec
                _skip_proxy = bool(int(_os_rec.environ.get(
                    "REF_RECONCILE_APPROACH", "0") or "0"))
                if not _skip_proxy:
                    Q[self._NEW_PEE_SLOT, self._NEW_PEE_SLOT] = (
                        self.w_ee_approach * np.eye(3)
                    )
                    x_ref[self._NEW_PEE_SLOT] = effective_proxy

            # --- Perpendicular box-velocity penalty ---
            # Penalize box linear velocity components orthogonal to g_hat.
            if dist > 1e-3:
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
        # PUSHA_STAGE5_R_VECTOR is set as "rx,ry,rz" (e.g., "0.01,0.01,1"),
        # use np.diag([rx,ry,rz]); else scalar w_torque*I (bit-identical to
        # pre-Stage-5).
        import os as _os
        _r_s = _os.environ.get("PUSHA_STAGE5_R_VECTOR", "")
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

        QN = self.w_terminal * Q
        return Q, R, QN, x_ref