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
    Instantiated once per run and injected into BaseMPC.

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
                 n_x: int, n_u: int):
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
        self.w_torque      = float(c.get("w_torque",         0.01))
        self.w_terminal    = float(c.get("w_terminal",        5.0))
        self.z_ref         = float(c.get("z_ee_target",      0.05))
        self.d_push        = float(c.get("d_push",           0.10))
        self.w_ee_approach = float(c.get("w_ee_approach",   800.0))

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
              plant_ctx=None, current_q: np.ndarray = None):
        """
        Return (Q, R, QN, x_ref) for one MPC step.

        If plant_ctx and current_q are provided, augments Q and x_ref with a
        linearised EE approach cost via the arm Jacobian.
        """
        # --- Base object-goal cost ---
        Q     = self._Q_obj.copy()
        x_ref = np.zeros(self.n_x)
        x_ref[self._obj_x_idx] = target_xy[0]
        x_ref[self._obj_y_idx] = target_xy[1]
        x_ref[self._obj_z_idx] = self.z_ref

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
                    obj_xy[0] - 0.18 * g_hat[0],
                    obj_xy[1] - 0.18 * g_hat[1],
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
                if ee_to_box_dist < 0.15 and perp_magnitude > 1e-4:
                    extra_shift = -perp_vec * min(1.0, perp_magnitude / 0.05)
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
                w = self.w_ee_approach
                Q[: self.n_u, : self.n_u] += 2.0 * w * (J_arm.T @ J_arm)

                # Shift arm reference toward effective proxy
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
        return Q, self._R, QN, x_ref
