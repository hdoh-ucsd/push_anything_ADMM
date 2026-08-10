"""Cartesian PWL reposition trajectory — Stage A port of the reference's
Reposition(...) + LcmTrajectoryReceiver mechanism.

Pure-numpy. Builds a 3-leg (lift / traverse / descend) Cartesian PWL
trajectory between two world-frame points, parameterized by absolute
sim time. Replaces the per-tick setpoint march + per-knot IK + joint-PD
path in the legacy free-mode trackers. Fed to the existing
OperationalSpaceController via (p_des, v_des) = eval(sim_t) at each
control tick.

Spec: docs/superpowers/plans/2026-06-23-alignment-phase-plan.md §3
Stage A. The reference dairlib code is reposition.cc (Reposition(...))
+ sampling_based_c3_controller.cc:1839-1928 (UpdateRepositioningExecution-
Trajectory) + franka_osc_controller.cc:101-103, 149-158
(LcmTrajectoryReceiver + TransTaskSpaceTrackingData).
"""
from __future__ import annotations

import numpy as np


def spherical_arc_knots(p_start: np.ndarray,
                        p_target: np.ndarray,
                        obj_xyz: np.ndarray,
                        sphere_radius: float,
                        z_min: float,
                        n_arc: int = 24) -> np.ndarray:
    """kSpherical knots — port of reference reposition.cc RepositionSpherical.

    The EE leaves its current position, joins a sphere of radius
    `sphere_radius` centred on the OBJECT, arcs around the object to the
    bearing of the target, then leaves the sphere for the target:

        v1 = unit(p_start  - obj)        v2 = unit(p_target - obj)
        wp1 = obj + r*v1                 wp2 = obj + r*v2
        arc(theta) = obj + r*(cos(theta)*v1 + sin(theta)*v4),  theta in [0, ang]
        v3 = unit(v1 x v2)               v4 = unit(v3 x v1)

    This is what makes the jack tractable. A piecewise-linear reposition
    traverses at a fixed `pwl_waypoint_height`, which for a flat object can
    always be chosen above the object -- but the jack is 0.122 m tall and
    jacktoy's waypoint height is 0.06, exactly its mid-height, so a PWL
    traverse drives the EE THROUGH the object. The arc never enters the
    sphere, so it cannot.

    Two guards, both from the reference:
      - if v4 points downward and the arc is long, flip v4 and go the long
        way round, so the arc does not dive at the ground;
      - if the target is nearly antipodal (> 0.9*pi) the v1 x v2 cross
        product is ill-conditioned, so pick the arc plane using world +z.

    Returns a (3, K) knot array: p_start, arc samples, p_target. Knot TIMING
    is left to RepositionTrajectory's cumulative-arclength parameterisation,
    which reproduces the reference's constant `speed*dt` spacing.
    """
    p_start  = np.asarray(p_start,  dtype=float).reshape(3)
    p_target = np.asarray(p_target, dtype=float).reshape(3)
    obj_xyz  = np.asarray(obj_xyz,  dtype=float).reshape(3)
    r = float(sphere_radius)

    d1 = p_start  - obj_xyz
    d2 = p_target - obj_xyz
    n1, n2 = float(np.linalg.norm(d1)), float(np.linalg.norm(d2))
    if n1 < 1e-9 or n2 < 1e-9:
        # Degenerate: EE or target sits on the object centre. No arc plane.
        return np.stack([p_start, p_target], axis=1)
    v1, v2 = d1 / n1, d2 / n2

    # Reference clamps to (-0.9999, 0.9999) to avoid acos NaNs.
    travel_angle = float(np.arccos(np.clip(float(v1 @ v2), -0.9999, 0.9999)))

    wp1 = obj_xyz + r * v1
    wp2 = obj_xyz + r * v2

    v3 = np.cross(v1, v2)
    n3 = float(np.linalg.norm(v3))
    if n3 < 1e-9:
        v3 = np.cross(v1, np.array([0.0, 0.0, 1.0]))
        n3 = float(np.linalg.norm(v3))
        if n3 < 1e-9:
            v3 = np.cross(v1, np.array([1.0, 0.0, 0.0]))
            n3 = float(np.linalg.norm(v3))
    v3 = v3 / n3
    v4 = np.cross(v3, v1)
    v4 = v4 / float(np.linalg.norm(v4))

    # Keep the traversed arc above the ground.
    if v4[2] < -0.5 and travel_angle > 0.5 * np.pi:
        v4 = -v4
        travel_angle = 2.0 * np.pi - travel_angle

    # Near-antipodal: v1 x v2 is ill-conditioned; pick the plane via world +z.
    if travel_angle > 0.9 * np.pi:
        almost_v4 = v1.copy()
        almost_v4[2] += 1.0
        na = float(np.linalg.norm(almost_v4))
        if na > 1e-9:
            v3 = np.cross(v1, almost_v4 / na)
            n3 = float(np.linalg.norm(v3))
            if n3 > 1e-9:
                v3 = v3 / n3
                v4 = np.cross(v3, v1)
                v4 = v4 / float(np.linalg.norm(v4))

    k = max(2, int(n_arc))
    thetas = np.linspace(0.0, travel_angle, k)
    arc = (obj_xyz[:, None]
           + r * (np.cos(thetas)[None, :] * v1[:, None]
                  + np.sin(thetas)[None, :] * v4[:, None]))

    knots = np.concatenate(
        [p_start.reshape(3, 1), wp1.reshape(3, 1), arc,
         wp2.reshape(3, 1), p_target.reshape(3, 1)], axis=1)

    # Reference enforces the workspace z floor on every knot.
    knots[2, :] = np.maximum(knots[2, :], float(z_min))

    # Drop consecutive duplicates so the arclength parameterisation cannot
    # produce a zero-length leg (which would give NaN velocity).
    keep = [0]
    for j in range(1, knots.shape[1]):
        if float(np.linalg.norm(knots[:, j] - knots[:, keep[-1]])) > 1e-9:
            keep.append(j)
    return knots[:, keep] if len(keep) >= 2 else np.stack([p_start, p_target], axis=1)


class RepositionTrajectory:
    """3-leg Cartesian PWL trajectory parameterized by sim time.

    Knots (typical):
        knot 0 = p_start
        knot 1 = (p_start.xy, z_safe)     [lift-end]
        knot 2 = (p_target.xy, z_safe)    [traverse-end]
        knot 3 = p_target                 [descend-end]

    Zero-length legs are pruned (no NaN velocity). If
    ``||p_start - p_target|| < straight_line_thresh``, the trajectory
    collapses to a single leg p_start → p_target (no z_safe transit).

    Knot times are computed from cumulative Euclidean leg lengths
    divided by ``speed`` (constant speed per leg).

    ``eval(sim_t)`` returns ``(p_des, v_des, done)``:
        - p_des: linearly interpolated position on the active leg.
        - v_des: constant per leg, = (knot[i+1] - knot[i]) / (t[i+1] - t[i]).
                 Zero when sim_t >= final knot time.
        - done: True iff sim_t >= final knot time.

    ``is_finished(sim_t, ee_now, tol)`` is the reference's
    ``finished_reposition_flag`` analog: BOTH the trajectory time has
    elapsed AND the EE is physically within ``tol`` of p_target.
    """

    def __init__(self,
                 p_start: np.ndarray,
                 p_target: np.ndarray,
                 z_safe: float,
                 speed: float,
                 t_start: float,
                 straight_line_thresh: float = 0.008,
                 dt_plan: float = 0.0,
                 traj_type=None,
                 obj_xyz=None,
                 sphere_radius: float = 0.12,
                 straight_line_within_angle: float = 0.30,
                 z_min: float = 0.0):
        p_start  = np.asarray(p_start,  dtype=float).reshape(3)
        p_target = np.asarray(p_target, dtype=float).reshape(3)
        if speed <= 0.0:
            raise ValueError(f"speed must be positive, got {speed}")

        self.p_start  = p_start.copy()
        self.p_target = p_target.copy()
        self.z_safe   = float(z_safe)
        self.speed    = float(speed)
        self.t_start  = float(t_start)
        self.dt_plan  = float(dt_plan)

        # 2026-07-19 reference-conformant straight-line dispatch.
        # Reference `reposition.cc:44-56` routes to `RepositionStraightLine`
        # when `xy_travel_distance < use_straight_line_traj_under_piecewise_
        # linear` (0.008 m for push_t). Port previously used 3-D distance
        # (`np.linalg.norm(p_target - p_start)`), so a sample below a
        # near-target arm (arm at (target_xy, 0.06) → target (target_xy,
        # 0.005): xy=0, 3D=55 mm > 8 mm) failed the check and dispatched
        # to the lift-traverse-descend PWL branch.  The subsequent
        # re-lift zig-zag (leg 1 pulls arm back to z_safe=0.06 before
        # leg 3 descends) blocked physical descent.  Matching reference's
        # xy-only test lets the arm proceed straight to target once xy
        # is close.
        # ---- kSpherical dispatch (reference reposition.cc:48-50, 63-67) --
        # For the spherical/circular types the reference's straight-line
        # shortcut is an ANGLE test, not the PWL types' xy-distance test.
        _is_spherical = (traj_type is not None
                         and int(traj_type) == 1        # kSpherical
                         and obj_xyz is not None)
        if _is_spherical:
            _d1 = np.asarray(obj_xyz, float).reshape(3)
            _a = p_start - _d1
            _b = p_target - _d1
            _na, _nb = float(np.linalg.norm(_a)), float(np.linalg.norm(_b))
            _ang = (float(np.arccos(np.clip(
                        float((_a / _na) @ (_b / _nb)), -0.9999, 0.9999)))
                    if (_na > 1e-9 and _nb > 1e-9) else 0.0)
            if _ang < float(straight_line_within_angle):
                self.knot_positions = np.stack([p_start, p_target], axis=1)
            else:
                self.knot_positions = spherical_arc_knots(
                    p_start, p_target, _d1, sphere_radius, z_min)
            self._finalise_timing()
            return

        xy_dist = float(np.linalg.norm(p_target[:2] - p_start[:2]))
        if xy_dist < float(straight_line_thresh):
            # Near-target: direct 2-knot straight line from p_start to
            # p_target. No lift, no traverse, no zig-zag.
            self.knot_positions = np.stack([p_start, p_target], axis=1)  # (3, 2)
        else:
            lift_end     = np.array([p_start[0],  p_start[1],  z_safe])
            traverse_end = np.array([p_target[0], p_target[1], z_safe])
            # 2026-07-19 wasteful-lift skip.  Reference reposition.cc
            # unconditionally adds the lift-to-waypoint knot whenever
            # p_start.z != z_safe.  For push_t (arm frequently at
            # z=0.02..0.06 near a T sample at z=0.005 with waypoint at
            # 0.075) the port's rebuild-every-tick + 8-mm xy dispatch
            # threshold caused a K=2⇔K=4 flip-flop: any small
            # perturbation that pushed xy_dist across 8 mm re-added the
            # lift knot, dragging arm from z=0.02 all the way up to
            # z=0.075 before descending again.  54 such lifts over 300
            # planner ticks in the heights run (traj-log
            # push_t_show_traj_20260719_121953).
            #
            # Skip the lift knot when the arm is already below z_safe
            # AND the target is below the arm (arm descending, no
            # obstacle-clearance need).  In that case do a direct
            # straight line from p_start to p_target — same shape the
            # `xy_dist < straight_line_thresh` branch produces.  For
            # arm-above-z_safe or target-above-arm cases, keep the
            # reference-conformant lift-traverse-descend structure.
            _arm_below_safe   = p_start[2] < z_safe - 1e-3
            _target_below_arm = p_target[2] < p_start[2] - 1e-3
            # 2026-07-19: additional xy-proximity constraint on skip-lift.
            # The original skip-lift condition (arm_below_safe AND
            # target_below_arm) fires whenever arm is at any low z, but
            # when xy_dist is LARGE, the direct diagonal from p_start to
            # p_target passes THROUGH the object at low z — the arm
            # collides with the object during transit (observed in
            # push_t_show_run_20260719_134637 steps 240-254: arm crossed
            # T's east face at z=0.02, pushing T further NE and blocking
            # the corrective SW push).  Add an xy-proximity gate so we
            # only skip lift when arm has already committed to descending
            # near the target (xy within a T-half-extent so any transit
            # stays within an already-committed contact region).
            _xy_dist = float(np.linalg.norm(p_target[:2] - p_start[:2]))
            _skip_lift_xy_thresh = 0.05   # 50 mm — half a T stem width
            if (_arm_below_safe and _target_below_arm
                    and _xy_dist < _skip_lift_xy_thresh):
                self.knot_positions = np.stack(
                    [p_start, p_target], axis=1)  # (3, 2)
            else:
                # Build knots, pruning zero-length legs.
                knots = [p_start.copy()]
                if abs(p_start[2] - z_safe) > 1e-3:
                    knots.append(lift_end)
                if not np.allclose(
                        traverse_end[:2], knots[-1][:2], atol=1e-9):
                    knots.append(traverse_end)
                if abs(p_target[2] - z_safe) > 1e-3:
                    knots.append(p_target.copy())
                else:
                    # Target z == z_safe: last knot IS the target (no descend).
                    knots[-1] = p_target.copy()
                self.knot_positions = np.stack(knots, axis=1)  # (3, K)

        self._finalise_timing()

    def _finalise_timing(self):
        """Parameterise self.knot_positions by arclength / speed.

        Factored out so the kSpherical branch, which returns early with its
        own knot set, shares the identical timing and finished-flag logic.
        """
        # Knot times = t_start + cumulative leg-length / speed.
        seg_lengths = np.linalg.norm(
            np.diff(self.knot_positions, axis=1), axis=0
        )                                              # (K-1,)
        seg_durations = seg_lengths / self.speed       # (K-1,)
        cum = np.concatenate([[0.0], np.cumsum(seg_durations)])  # (K,)
        self.knot_times = self.t_start + cum

        # 2026-07-19 reference-conformant finished_reposition_flag.
        # Reference reposition.cc:462-465 (PiecewiseLinear branch) and
        # :111 (StraightLine branch) both set the flag when the fill/
        # trajectory-walking loop places its first knot at index 1 while
        # NOT already at the goal — i.e., when total travel time fits in
        # a single planner tick.  Effective condition (unified across
        # both branches): total 3-leg (or direct-line) path length is
        # covered by one step_size = speed * dt_plan.  Equivalently:
        # (t_end - t_start) <= dt_plan.  Reference sets this per call to
        # Reposition(...), which runs once per planner tick with
        # p_start = current EE and p_target = best sample location.
        self.finished_reposition_flag = bool(
            self.dt_plan > 0.0
            and float(self.knot_times[-1] - self.t_start)
                <= float(self.dt_plan))

    @property
    def t_end(self) -> float:
        return float(self.knot_times[-1])

    def eval(self, sim_t: float) -> tuple[np.ndarray, np.ndarray, bool]:
        """Returns ``(p_des, v_des, done)``."""
        t = float(sim_t)
        if t <= self.t_start:
            # Before start: hold start position, command initial leg velocity.
            # seg_dt guarded like the interior branch: a degenerate leg
            # (p_start == p_target through the straight-line / skip-lift
            # branches) has zero length AND zero duration, so the guard
            # yields v = 0 exactly — the unguarded 0/0 here produced NaN
            # v_des on every build tick (eval at t == t_start), poisoning
            # the OSC QP (kIterationLimit → zero torque; see
            # tests/test_reposition_trajectory_degenerate.py).
            p = self.knot_positions[:, 0].copy()
            if self.knot_positions.shape[1] >= 2:
                seg_dt = self.knot_times[1] - self.knot_times[0]
                v = (self.knot_positions[:, 1]
                     - self.knot_positions[:, 0]) / max(seg_dt, 1e-12)
            else:
                v = np.zeros(3)
            return p, v, False

        if t >= self.t_end:
            # Past end: hold target, zero velocity, done.
            return self.knot_positions[:, -1].copy(), np.zeros(3), True

        # Interior: find segment [i, i+1] s.t. t in [t_i, t_{i+1}).
        i = int(np.searchsorted(self.knot_times, t, side="right")) - 1
        i = max(0, min(i, self.knot_positions.shape[1] - 2))
        t_i  = self.knot_times[i]
        t_ip = self.knot_times[i + 1]
        p_i  = self.knot_positions[:, i]
        p_ip = self.knot_positions[:, i + 1]
        alpha = (t - t_i) / max(t_ip - t_i, 1e-12)
        p = p_i + alpha * (p_ip - p_i)
        v = (p_ip - p_i) / max(t_ip - t_i, 1e-12)
        return p, v, False

    def is_finished(self, sim_t: float, ee_now: np.ndarray,
                    tol: float = 0.005) -> bool:
        """Reference-style finished_reposition_flag analog.

        True iff trajectory time has elapsed AND EE is within ``tol``
        of p_target.
        """
        if sim_t < self.t_end:
            return False
        return float(np.linalg.norm(np.asarray(ee_now).reshape(3)
                                    - self.p_target)) <= float(tol)
