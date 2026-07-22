"""
Reposition — collision-free path planner + joint-space PD tracker.

Implements traj_type=kPiecewiseLinear from the dairlib reference: lift
the EE straight up to a safe height, traverse horizontally above the
target xy, then descend to the target z. Replanned each control loop.

The Cartesian planner (`next_waypoint`) is pure numpy and unit-testable.
The torque controller (`PiecewiseLinearTracker.compute_torque`) wraps
IK + joint-PD-with-grav-comp and needs a Drake plant.

Why this trajectory shape? The workspace is a Franka pushing a single
box on a table. With z_safe well above (box top + contact-detection
threshold), the lift+traverse+descend path is collision-safe by
construction with no need for an RRT planner. This replaces the
joint-space straight-line free-mode tracker in the legacy wrapper which
clipped the box face on diagonal swings.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from control.sampling_c3.params import RepositionParams
from control.sampling_c3.ik import solve_ik_to_ee_pos


# ---------------------------------------------------------------------------
# Pure trajectory generator (no Drake)
# ---------------------------------------------------------------------------

def next_waypoint(p_now:                np.ndarray,
                  p_target:             np.ndarray,
                  z_safe:               float,
                  ds:                   float,
                  straight_line_thresh: float = 0.008,
                  z_eps:                float = 5e-3,
                  allow_descent:        bool = True) -> np.ndarray:
    """Advance one control-step worth (`ds` metres) along the
    piecewise-linear path from p_now to p_target via z=z_safe.

    Three phases (selected based on p_now):

      Phase 1 (lift):    p_now.z < z_safe   → climb straight up
      Phase 2 (traverse): at z_safe         → translate xy toward p_target xy
      Phase 3 (descend):  at p_target.xy    → descend to p_target.z

    If the total Cartesian distance from p_now to p_target is below
    `straight_line_thresh` the lift+descend phases are skipped and the
    next waypoint is `ds` along the direct line — avoids over-shoot for
    sub-cm corrections.

    Stage 1 of the wrong-face race-fix plan (2026-06-01): `allow_descent`
    gates Phase 3 and the direct-line shortcut's downward component. When
    False, the path holds altitude (lifting to z_safe if below, otherwise
    staying put). Used by `RepositionIKTracker` to suspend descent until
    the dispatcher's `p_target` has been stable for N consecutive ticks,
    preventing the EE from being dropped onto a wrong face during
    dispatcher oscillation. Default True preserves pre-Stage-1 behavior
    for callers that don't opt in (e.g. PiecewiseLinearTracker standalone).

    Note on the two 5mm constants in this file's neighborhood:
      - `z_eps` (default 5e-3) = "xy within 5mm of target → at_target_xy".
        Used by Phase 3 to declare xy convergence.
      - `TARGET_STABLE_TOL` (5e-3) in `reposition_ik.RepositionIKTracker` =
        "p_target jumped > 5mm → not the same target anymore, reset
        stability counter". Used by the Stage-1 gate to gate `allow_descent`.
      They share a numeric value but serve distinct jobs (xy-convergence
      vs. target-identity). Tune independently if needed.

    Pure numpy. Returns a (3,) ndarray.
    """
    p_now    = np.asarray(p_now,    dtype=float)
    p_target = np.asarray(p_target, dtype=float)
    if p_now.shape != (3,) or p_target.shape != (3,):
        raise ValueError(
            f"p_now/p_target must be (3,), got {p_now.shape}/{p_target.shape}")
    if ds <= 0:
        raise ValueError(f"ds must be positive, got {ds}")

    # Direct-line shortcut for sub-cm corrections. When descent is blocked
    # and the shortcut would step the EE downward, fall through to the
    # phased logic which holds altitude.
    direct = float(np.linalg.norm(p_target - p_now))
    if direct < straight_line_thresh:
        if direct == 0.0:
            return p_target.copy()
        if allow_descent or p_target[2] >= p_now[2] - z_eps:
            step = min(ds, direct)
            return p_now + step * (p_target - p_now) / direct
        # else: fall through to phased logic with descent blocked

    xy_dist = float(np.linalg.norm(p_target[:2] - p_now[:2]))
    at_target_xy = xy_dist <= z_eps

    # Phase 3 takes priority once xy is at the target AND descent is allowed.
    # This prevents the phase-1/phase-3 oscillation that occurs when the
    # descent drops z below z_safe-eps and the unconditional Phase 1 check
    # would otherwise snap back up.
    if at_target_xy and allow_descent:
        if abs(p_now[2] - p_target[2]) <= z_eps:
            return p_target.copy()
        if p_now[2] > p_target[2]:
            next_z = max(p_target[2], p_now[2] - ds)
        else:
            # Target above current — climb (degenerate case for the
            # pushing task but harmless to support).
            next_z = min(p_target[2], p_now[2] + ds)
        return np.array([p_target[0], p_target[1], next_z])

    # Phase 1: lift to z_safe (when there is still work to do — either xy
    # is not at target, or descent is blocked and we have not yet reached
    # z_safe). In the descent-blocked at-target case this re-lifts after a
    # prior trajectory had partially descended.
    if p_now[2] < z_safe - z_eps:
        next_z = min(z_safe, p_now[2] + ds)
        return np.array([p_now[0], p_now[1], next_z])

    # Hold-at-xy: descent blocked, xy already at target, z at or above
    # z_safe. Stay put — Stage-1 descent-blocked analogue of Phase 3.
    if at_target_xy:
        return p_now.copy()

    # Phase 2: traverse horizontally at current z (which is ≥ z_safe).
    direction = (p_target[:2] - p_now[:2]) / xy_dist
    step = min(ds, xy_dist)
    return np.array([
        p_now[0] + step * direction[0],
        p_now[1] + step * direction[1],
        p_now[2],
    ])


def is_at_target(p_now: np.ndarray,
                 p_target: np.ndarray,
                 tol: float = 1e-3) -> bool:
    """True iff p_now is within `tol` metres of p_target (3D Euclidean)."""
    return float(np.linalg.norm(p_target - p_now)) <= tol


# ---------------------------------------------------------------------------
# Drake-using PD tracker
# ---------------------------------------------------------------------------

class PiecewiseLinearTracker:
    """One per outer-controller — owns the joint-space integral state and
    the previous-target memo (so the integrator is reset on retarget).

    Per control step:

        ee_now = FK(q_now)
        p_des  = next_waypoint(ee_now, p_target, z_safe, ds)
        q_des  = solve_ik_to_ee_pos(plant, ee_frame, p_des, q_seed=q_now, ctx)
        u      = clip(τ_g + Kp(q_des-q_now) + Ki·∫err - Kd v_now, ±tlim)

    `compute_torque` returns the joint torque to apply at this step.
    """

    def __init__(self,
                 plant,
                 ee_frame,
                 n_arm_dofs:  int,
                 params:      RepositionParams):
        self.plant       = plant
        self.ee_frame    = ee_frame
        self.world_frame = plant.world_frame()
        self.n_arm_dofs  = int(n_arm_dofs)
        self.params      = params

        self._integral:        np.ndarray            = np.zeros(self.n_arm_dofs)
        self._prev_target_pos: Optional[np.ndarray]  = None
        # Setpoint position: marches along the PWL path independent of
        # how fast the physical arm tracks. Reset whenever the target
        # changes (or on explicit reset()). If we used FK(q_now) as the
        # next-waypoint start, a lagging arm would cause the setpoint to
        # crawl forward only as fast as the arm — which is zero at
        # initialisation. Tracking it separately fixes that.
        self._setpoint_pos:    Optional[np.ndarray]  = None

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Wipe integral state and the previous-target memo. Call when
        leaving repos mode so the next entry starts fresh."""
        self._integral[:]     = 0.0
        self._prev_target_pos = None
        self._setpoint_pos    = None

    # ------------------------------------------------------------------
    # Per-step torque
    # ------------------------------------------------------------------

    def compute_torque(self,
                       current_q:  np.ndarray,
                       current_v:  np.ndarray,
                       plant_ctx,
                       p_target:   np.ndarray,
                       dt_osc:     float,
                       admit_active: bool = False,
                       **_kwargs) -> tuple[np.ndarray, dict]:
        """Compute one control-step's joint torque toward p_target.

        Returns (u, diag) where diag is a dict of per-component torque
        norms suitable for the [GS-free] diagnostic line.

        ── TEMPORARY MARKED NO-OP SHIM (2026-06-17 LPF validation) ──
        ``admit_active`` is consumed-and-IGNORED here. This is a temporary
        plumbing shim so that wrapper.py:1815's unconditional
        ``admit_active=...`` kwarg does not crash when the YAML selects
        the PWL tracker. This is NOT an implementation of the debounced
        admit-active latch — see RepositionIKTracker for that behavior
        (reposition_ik.py:693, 1320-1327: 8-tick latch that suspends
        Phase-1 lift to keep the EE pressed against the box face during
        admitted contact, robust to 1-tick admit drops).

        The PWL-latch question is OPEN: the canonical kPWL config is
        running WITHOUT a contact-persistence mechanism while kIK has
        one. Adding the equivalent latch to PWL is a SEPARATE
        evaluation, deferred until after the LPF validation produces
        its verdict. Do not interpret this shim as resolving that
        question — it just unblocks the LPF run.
        """
        _ = admit_active  # consumed-and-ignored; see docstring
        # 1. FK current EE
        self.plant.SetPositions(plant_ctx, current_q)
        ee_now = self.plant.CalcPointsPositions(
            plant_ctx, self.ee_frame, np.zeros(3), self.world_frame,
        ).flatten()

        # Reset integral + restart setpoint at current EE if the target
        # has shifted (prevents stale windup; restarts the PWL path from
        # a sensible starting point).
        target_changed = (
            self._prev_target_pos is None
            or float(np.linalg.norm(p_target - self._prev_target_pos)) > 1e-3
        )
        if target_changed:
            self._integral[:]     = 0.0
            self._setpoint_pos    = ee_now.copy()
            self._prev_target_pos = p_target.copy()

        # 2. Advance the SETPOINT (not the physical EE) along the PWL path.
        # This makes the desired Cartesian position march at params.speed
        # m/s regardless of how well the arm tracks — the arm chases a
        # moving setpoint, building up integrator authority if it lags.
        ds = self.params.speed * dt_osc
        p_des = next_waypoint(
            self._setpoint_pos, p_target,
            z_safe=self.params.pwl_waypoint_height,
            ds=ds,
            straight_line_thresh=self.params.use_straight_line_traj_under_piecewise_linear,
        )
        self._setpoint_pos = p_des.copy()

        # 3. IK that waypoint to a joint config
        q_des, ik_err, ik_iters = solve_ik_to_ee_pos(
            self.plant, self.ee_frame,
            p_target=p_des, q_init=current_q,
            plant_ctx=plant_ctx, n_arm_dofs=self.n_arm_dofs,
        )

        # IK left plant_ctx at q_des — restore to current_q so caller's
        # downstream FK calls reflect the actual state.
        self.plant.SetPositions(plant_ctx, current_q)

        # 4. Joint-space PD with grav-comp
        q_arm_now    = current_q[: self.n_arm_dofs]
        v_arm_now    = current_v[: self.n_arm_dofs]
        q_arm_target = q_des[: self.n_arm_dofs]

        q_err = q_arm_target - q_arm_now
        self._integral += q_err * dt_osc
        np.clip(self._integral, -self.params.I_max, self.params.I_max,
                out=self._integral)

        u_p = self.params.Kp_q * q_err
        u_i = self.params.Ki_q * self._integral
        u_d = -self.params.Kd_q * v_arm_now
        u_pd = u_p + u_i + u_d

        # Gravity-comp is owned by the main loop, not the tracker; emit a
        # task-only torque. (Note: under the SamplingC3Controller wrapper this
        # tracker's `u` is overwritten by the OSC executor at wrapper.py
        # line ~1608, so this only matters when PWL is used standalone.)
        _u_raw = u_pd
        u = np.clip(_u_raw,
                    -self.params.torque_limit, self.params.torque_limit)
        _pd_sat = bool(np.any(
            np.abs(_u_raw) >= self.params.torque_limit - 1e-9))
        _q_max_resid_deg = float(np.degrees(np.max(np.abs(q_err))))
        # PWL tracker doesn't plan a knot trajectory — landing_err is the
        # IK solution endpoint distance to p_target (i.e., FK(q_des) ↔
        # p_target), which equals ‖p_des − p_target‖ since IK constrains FK
        # to p_des. Free in compute time.
        _landing_err = float(np.linalg.norm(p_des - p_target))

        # Trajectory-finished signal: EE physically at the repos target.
        # Was z-only (`|ee_z - p_target_z| ≤ 5 mm`). After fc51111 flipped
        # use_reposition_pwl_trajectory to True by default, the wrapper's
        # line 1269 overrides finished_repos with the PWL trajectory's own
        # is_finished (20 mm 3D tol) for the mode-switch decision — but
        # `_last_repos_finished` (written from `finished` below at wrapper
        # line 2259) is ALSO consumed at wrapper line 1155 to inflate the
        # prev_repos sample cost by finished_reposition_cost. The z-only
        # check fired false-positives whenever the arm's z happened to
        # pass through the repos-target z on descent (e.g. box run: EE
        # z=0.20 → 0.030 sweeps through), spuriously inflating prev_repos
        # cost mid-descent → target-flip instability. Use 3 D euclidean
        # with 20 mm tolerance to match the PWL is_finished predicate.
        finished_val = float(np.linalg.norm(p_target - ee_now))
        finished     = finished_val <= 0.020

        diag = dict(
            up_norm         = float(np.linalg.norm(u_p)),
            ui_norm         = float(np.linalg.norm(u_i)),
            ud_norm         = float(np.linalg.norm(u_d)),
            uclip_norm      = float(np.linalg.norm(u)),
            qerr_norm       = float(np.linalg.norm(q_err)),
            ik_err          = float(ik_err),
            ik_iters        = int(ik_iters),
            p_des           = p_des,
            ee_now          = ee_now,
            finished        = bool(finished),
            finished_val    = finished_val,
            landing_err     = _landing_err,
            pd_sat          = _pd_sat,
            q_max_resid_deg = _q_max_resid_deg,
            knot0_feasible  = bool(ik_iters > 0),
        )
        return u, diag
