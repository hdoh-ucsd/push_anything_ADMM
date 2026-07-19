"""
Sample generation strategies — Venkatesh et al. RA-L 2025 §IV-B.

Mirrors examples/sampling_c3/generate_samples.{h,cc} from dairlib upstream.
First-cut implementation supports kRandomOnCircle (the strategy this
project's WEST validation requires); the other six SamplingStrategy enum
values raise NotImplementedError so adding them is a one-spot change.

Each strategy returns 3D Cartesian EE target positions (world frame).
The wrapper is responsible for solving IK to convert each target to a
joint-space q seed.

Pure-numpy; no Drake dependency. Workspace bounds are enforced here
(samples that fall outside the rectangular xy slab or z range are
re-drawn up to a small budget; failures are logged but not raised).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from control.sampling_c3.params import SamplingParams, SamplingStrategy


# Sampler-centering on goal-aligned faces. When _face_normal_projection
# picks a face whose contact would push the box toward the goal (cosine of
# alignment between -n_outward and g_hat exceeds the threshold), the
# tangential jitter range is reduced to keep the sample near the face
# center. Other faces keep the existing uniform jitter.
GOAL_ALIGN_THRESHOLD = 0.7          # cos > 0.7 → ~45° cone around goal
CENTERED_JITTER_FRACTION = 0.2      # 2026-07-19: restored 0.0 → 0.2. With 0.0, multiple strategy samples on the goal-aligned face all landed at exact face center → identical positions. Traced in results/push_t_liftfix_20260719_003029: n=2 strategy samples both at (-0.0503, -0.1044, +0.005) at step 68. Dispatcher had no meaningful "best_other" alternative, arm hesitation observed. 0.2 restores tangent jitter within ±0.2·half_len (≈±16mm on crossbar's 0.08 half-len).


# ---------------------------------------------------------------------------
# T-shape face table (single-body-collapsed T; see env_builder._tshape_sdf).
# T LINK FRAME origin at the T's combined CoM (both bars 0.5 kg).
# Vertical bar (crossbar) at pose (+0.05, 0, 0), size 0.16×0.04×0.04.
# Horizontal bar (stem)      at pose (-0.05, 0, 0, 0, 0, π/2), size 0.16×0.04×0.04.
#
# 8 outward-facing side face patches, at 4 distinct body-frame outward normals.
# Column order: (center_x, center_y, normal_x, normal_y, half_length_along_tangent).
# The tangent direction is CCW-perpendicular to (normal_x, normal_y).
# ---------------------------------------------------------------------------
_TSHAPE_FACE_TABLE = np.array([
    # (cx,    cy,     nx,   ny,   half_len)
    ( +0.13,  0.00,  +1.0,  0.0,  0.02),   # face 1: crossbar tip (+x)
    (  0.05, -0.02,   0.0, -1.0,  0.08),   # face 2: crossbar bottom (-y)
    ( -0.03, -0.05,  +1.0,  0.0,  0.03),   # face 3: bottom shoulder (+x)
    ( -0.05, -0.08,   0.0, -1.0,  0.02),   # face 4: stem bottom (-y)
    ( -0.07,  0.00,  -1.0,  0.0,  0.08),   # face 5: stem back (-x)
    ( -0.05, +0.08,   0.0, +1.0,  0.02),   # face 6: stem top (+y)
    ( -0.03, +0.05,  +1.0,  0.0,  0.03),   # face 7: top shoulder (+x)
    (  0.05, +0.02,   0.0, +1.0,  0.08),   # face 8: crossbar top (+y)
], dtype=float)

# §9 Option B (video-verified): TOP faces of the T for down-press samples.
# The reference video (https://youtu.be/rv9n8Uyvoh0 t=155s "20 Goals") shows the
# arm pressing DOWN on the T's top surface — a mode the port previously missed
# because sampling_height=0.034 kept the sphere below the T top at z=0.040. Two
# top patches match the T's two collision-body extents:
#   crossbar_top: link_x ∈ [-0.03, +0.13], link_y ∈ [-0.02, +0.02] → center
#                 (+0.05, 0), tangent half-extents (0.08, 0.02).
#   stem_top:     link_x ∈ [-0.07, -0.03], link_y ∈ [-0.08, +0.08] → center
#                 (-0.05, 0), tangent half-extents (0.02, 0.08).
# Normal for both is +z (world frame; T only rotates about z, so top-face
# world normal stays (0,0,+1) regardless of obj_quat). Column format:
# (cx, cy, half_len_x, half_len_y) in T link frame.
_TSHAPE_TOP_TABLE = np.array([
    # (cx,    cy,    half_len_x, half_len_y)
    (  0.05,  0.00,  0.08,       0.02),   # crossbar top face (spans crossbar)
    ( -0.05,  0.00,  0.02,       0.08),   # stem top face (spans stem)
], dtype=float)

# T half-height (world-z half of the 0.04 m bar height). T sits on ground with
# link origin at world-z = init_xyz[2] (0.020 by default), so T top is at
# init_xyz[2] + _TSHAPE_HALF_HEIGHT = 0.040 m in the port frame.
_TSHAPE_HALF_HEIGHT = 0.020
# Vertical setback ABOVE the top face — sphere-center world-z. Analogous to
# the horizontal sampling_setback (0.020 m) used for side faces. With
# PUSHER_RADIUS=0.0195 the sphere lowest sits 0.5 mm above the top surface
# at the setback point, mirroring the 0.5 mm side-face gap.
_TSHAPE_TOP_SETBACK = 0.020


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------

def generate_samples(strategy:    SamplingStrategy,
                     n_samples:   int,
                     obj_xy:      np.ndarray,
                     params:      SamplingParams,
                     rng:         Optional[np.random.Generator] = None,
                     g_hat:       Optional[np.ndarray] = None,
                     obj_quat:    Optional[np.ndarray] = None,
                     yaw_delta:   Optional[float] = None,
                     ) -> list[np.ndarray]:
    """
    Generate n_samples 3D EE target positions.

    Parameters
    ----------
    strategy   : sampling strategy enum
    n_samples  : number of samples to return (caller computes from
                 sampling_params.num_additional_samples_*)
    obj_xy     : (2,) object xy position (m, world frame)
    params     : SamplingParams (radius, height, workspace bounds, ...)
    rng        : numpy Generator (pass one for deterministic tests)
    g_hat      : (2,) unit goal direction; used for kRandomOnCircle to bias
                 the angular distribution toward the push axis (optional)

    Returns
    -------
    samples : list of (3,) np.ndarray, length == n_samples
              (may be shorter if workspace filtering rejected some)
    """
    if rng is None:
        rng = np.random.default_rng()

    if n_samples <= 0:
        return []

    if strategy == SamplingStrategy.kRandomOnCircle:
        raw = _random_on_circle(n_samples, obj_xy, params, rng, g_hat)
    elif strategy == SamplingStrategy.kRadiallySymmetric:
        raw = _radially_symmetric(n_samples, obj_xy, params, g_hat)
    elif strategy == SamplingStrategy.kFaceNormal:
        raw = _face_normal_projection(
            n_samples, obj_xy, params, rng, g_hat, obj_quat,
            yaw_delta=yaw_delta)
    elif strategy == SamplingStrategy.kFixed:
        raw = _fixed_samples(n_samples, params)
    else:
        raise NotImplementedError(
            f"Sampling strategy {strategy.name} not yet implemented; "
            f"only kRandomOnCircle, kRadiallySymmetric, kFaceNormal, kFixed "
            f"are supported.")

    if not params.filter_samples_for_safety:
        return raw
    return [p for p in raw if is_in_workspace(p, params)]


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def _random_on_circle(n_samples:    int,
                      obj_xy:       np.ndarray,
                      params:       SamplingParams,
                      rng:          np.random.Generator,
                      g_hat:        Optional[np.ndarray]) -> list[np.ndarray]:
    """Paper-faithful random ring sampler — Venkatesh §IV-B.

    Draws n_samples independent uniform angles θ_i ~ U[0, 2π) and places
    each sample at
        (obj_xy + sampling_radius · [cos θ_i, sin θ_i], sampling_height)

    Restoring the paper's sampler intent — `_random_on_circle` previously
    returned N copies of a perpendicular task-biased point. With OSC in
    place the executor no longer needs the kinematic-plowing assist that
    motivated the task-biased variant, and the dispatcher's cost gate
    relies on per-sample cost diversity to discriminate worthwhile
    candidates from kStayInRepos churn.

    `g_hat` is unused now (the sampler is direction-agnostic). Kept on
    the signature for backward compatibility with the public dispatcher.
    """
    del g_hat  # paper-faithful sampler is direction-agnostic
    r = float(params.sampling_radius)
    z = float(params.sampling_height)
    thetas = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
    return [
        np.array([obj_xy[0] + r * np.cos(t),
                  obj_xy[1] + r * np.sin(t),
                  z])
        for t in thetas
    ]


def _radially_symmetric(n_samples:  int,
                        obj_xy:     np.ndarray,
                        params:     SamplingParams,
                        g_hat:      Optional[np.ndarray]) -> list[np.ndarray]:
    """Evenly spaced points on the circle. When g_hat is given, sample 0
    is the proxy and the remaining n-1 are spaced 2π/(n-1) apart starting
    from the proxy angle."""
    samples: list[np.ndarray] = []
    r = float(params.sampling_radius)
    z = float(params.sampling_height)

    if g_hat is not None:
        proxy_angle = float(np.arctan2(-g_hat[1], -g_hat[0]))
    else:
        proxy_angle = 0.0

    for i in range(n_samples):
        theta = proxy_angle + 2.0 * np.pi * i / max(1, n_samples)
        samples.append(np.array([
            obj_xy[0] + r * np.cos(theta),
            obj_xy[1] + r * np.sin(theta),
            z,
        ]))
    return samples


def _face_normal_projection(n_samples:    int,
                            obj_xy:       np.ndarray,
                            params:       SamplingParams,
                            rng:          np.random.Generator,
                            g_hat:        Optional[np.ndarray],
                            obj_quat:     Optional[np.ndarray],
                            yaw_delta:    Optional[float] = None,
                            ) -> list[np.ndarray]:
    """Paper-faithful face-normal sampler — Push-Anything §IV-B1.

    For each sample:
      1. Pick one of the box's side faces uniformly (equal area for a cube).
      2. Sample a uniform point on that face (tangential jitter along the
         face width).
      3. Project the point outward along the face's world-frame normal by
         `sampling_setback`.
      4. Project to fixed world height `sampling_height`.
      5. Reject (resample) if the post-projection xy is still within
         `sample_reject_clearance` of the box surface.

    Compared to `_random_on_circle` (which places samples on a ring of
    radius `sampling_radius` around the box center), this sampler
    guarantees a constant outward clearance from each face — equivalent
    at the four cardinal angles but strictly larger at off-axis angles,
    where the ring sampler's geometry sends samples inside the box.

    `obj_quat=None` → axis-aligned fallback (treats body frame == world
    frame). Sufficient for our cube under translation-only pushing; if
    the box rotates during the run, pass the world-frame quaternion so
    the face normals are rotated correctly.

    `g_hat` (2D, points box→goal) conditions the tangential jitter on the
    sampled face's goal-alignment: samples on a face whose contact would
    push the box toward the goal cluster near face center (small jitter),
    while non-goal faces keep full uniform jitter to preserve sample
    diversity. If `g_hat` is None or zero-norm (e.g. pure-rotation task
    with no translation goal), all faces fall through to uniform jitter.
    """
    box_half = float(params.box_half_extent)
    setback  = float(params.sampling_setback)
    z        = float(params.sampling_height)
    reject_clearance = float(params.sample_reject_clearance)
    shape    = str(getattr(params, "object_shape", "box"))

    # BUG 2 fix (2026-07-13) — horizontal-setback compensation for the
    # port's fatter pusher radius. This is a COMPENSATION, not a
    # correction of a diverged setback value:
    #
    #   Reference push_t runs `end_effector_simple_model.urdf` sphere
    #   r = 0.0195 m with sample_projection_clearance = 0.020 m; sphere
    #   surface sits +0.5 mm OUTSIDE the object face at the setback
    #   position (clean side-face contact).
    #
    #   The port carries a GLOBAL pusher radius of 0.025 m (see
    #   sim/env_builder.py:PUSHER_RADIUS_DEFAULT — a known divergence
    #   from the reference that also surfaced in the box arc). Inherit
    #   the reference's 20 mm setback verbatim and the sphere surface
    #   sits 5 mm INSIDE the T bar face — the vertical_bar phi≈−0.005 m
    #   penetration observed at the SIDE-face setback target.
    #
    #   The fix: raise the effective setback to at least the port's
    #   pusher radius + clearance so the SIDE-face intent works with
    #   the port's sphere. This ADAPTS the setback to the port's
    #   radius; it does NOT correct the sampling-setback yaml value.
    #
    # Bit-identical special cases:
    #   - Box config (sampling_c3_kik.yaml:203) already ships
    #     sampling_setback: 0.030 (pusher_radius + 5 mm), so the max()
    #     is a no-op — box degenerates to its shipped setback and is
    #     bit-identical.
    #   - Any future config that already ships setback ≥ (r_pusher +
    #     clearance) is bit-identical for the same reason.
    #
    # Clearance floor (5 mm): matches the port's box config's already-
    # shipped 5 mm margin (sampling_c3_kik.yaml:203, setback = 0.030 =
    # pusher_radius 0.025 + 0.005). So the port's SIDE-face intent for
    # ANY object gets the same sphere-surface margin the box already
    # relies on.
    #
    # Float-comparison tolerance (1 µm): the box's yaml `0.030` floats to
    # exactly 0.03, but `PUSHER_RADIUS + 0.005` = 0.025 + 0.005 evaluates
    # to 0.030000000000000002 in binary float. A naive `<` fires on that
    # ULP gap and perturbs the box's downstream sample positions, IK, and
    # Drake dynamics — the box tripwire broke on this on 2026-07-13. Only
    # bump the setback when it's strictly less than min_setback minus 1 µm,
    # so any yaml value at or above (pusher_r + clearance) is a strict
    # no-op regardless of float representation.
    from sim.env_builder import PUSHER_RADIUS as _PUSHER_RADIUS
    _MIN_CLEARANCE_ABOVE_R = 0.005
    _min_setback = float(_PUSHER_RADIUS) + _MIN_CLEARANCE_ABOVE_R
    if setback + 1e-6 < _min_setback:
        setback = _min_setback

    # Goal-alignment conditional jitter setup. Only meaningful when g_hat
    # has a real direction; rotation-only tasks fall through to uniform.
    if g_hat is not None:
        g2 = np.asarray(g_hat, dtype=float).reshape(-1)[:2]
        _g_norm = float(np.linalg.norm(g2))
    else:
        g2 = None
        _g_norm = 0.0
    _use_goal_align = (_g_norm > 0.5)   # unit-vector check (excl. degenerate)
    if _use_goal_align:
        g2 = g2 / _g_norm

    if shape == "tshape":
        # Per-face table for the reference push_t geometry (single-body-collapsed).
        # T LINK FRAME face patches → rotate through obj_quat to get world frame.
        # Each row: (cx, cy, nx, ny, half_len). Center-x/y and normals rotate;
        # half_len is invariant under rigid rotation.
        _table = _TSHAPE_FACE_TABLE
        body_centers = np.column_stack([_table[:, 0], _table[:, 1],
                                        np.zeros(_table.shape[0])])   # (N,3)
        body_normals = np.column_stack([_table[:, 2], _table[:, 3],
                                        np.zeros(_table.shape[0])])   # (N,3)
        half_lens    = _table[:, 4]                                    # (N,)
        if obj_quat is not None:
            R = _quat_to_rot(obj_quat)
            world_centers = (R @ body_centers.T).T
            world_normals = (R @ body_normals.T).T
        else:
            world_centers = body_centers
            world_normals = body_normals
        n_side_faces = _table.shape[0]
        # §9 Option B: TOP faces (video-verified missing contact mode). Two
        # planar top patches; normal is +z regardless of obj_quat (T rotates
        # only about z). Body-frame centers rotate through R for world-frame
        # placement.
        _top_table = _TSHAPE_TOP_TABLE
        _top_body_centers = np.column_stack([
            _top_table[:, 0], _top_table[:, 1],
            np.zeros(_top_table.shape[0])])
        if obj_quat is not None:
            top_world_centers = (R @ _top_body_centers.T).T
        else:
            top_world_centers = _top_body_centers
        top_half_lens_xy = _top_table[:, 2:4]           # (N_top, 2)
        n_top_faces = _top_table.shape[0]
        n_faces = n_side_faces + n_top_faces
    else:
        # Box path (regression-safe): 4 cardinal ±x/±y body-frame normals.
        body_normals = np.array([
            [ 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [ 0.0, 1.0, 0.0],
            [ 0.0,-1.0, 0.0],
        ])
        if obj_quat is not None:
            R = _quat_to_rot(obj_quat)
            world_normals = (R @ body_normals.T).T
        else:
            world_normals = body_normals
        world_centers = None    # box uses obj_xy + box_half*normal (below)
        half_lens     = None
        n_faces = 4

    obj_xy_2 = np.asarray(obj_xy, dtype=float).reshape(2)

    # T1c — kMeshNormal area-weighted face pick (reference
    # generate_samples.cc:454, barycentric_bias=1). Overrides uniform /
    # goal-align face selection with a categorical distribution proportional
    # to face area. Uniform-within-face is unchanged (already true for the
    # port's rectangular jitter, matches barycentric_bias=1 semantics).
    # See SamplingParams.use_mesh_normal_area_weighting docstring for
    # per-face expected fractions.
    _T_BAR_HEIGHT = 0.04  # matches env_builder._tshape_sdf T bar cross-section
    _use_mesh_normal = bool(getattr(
        params, "use_mesh_normal_area_weighting", False))
    _beta = float(getattr(params, "face_bias_strength", 0.0))
    _n_horizontal = n_side_faces if shape == "tshape" else n_faces
    if _use_mesh_normal and shape == "tshape":
        # Side face area: 2 * half_len (tangent) * T_bar_height (fixed 0.04 m).
        _side_areas = 2.0 * half_lens * _T_BAR_HEIGHT           # (n_side,)
        # Top face area: 2 * hx * 2 * hy (rectangle).
        _top_areas = 4.0 * (top_half_lens_xy[:, 0]
                            * top_half_lens_xy[:, 1])            # (n_top,)
        _face_areas = np.concatenate([_side_areas, _top_areas])
        _face_probs = _face_areas / _face_areas.sum()
    elif _use_mesh_normal and shape == "box":
        # 4 identical square faces -> area-weighted == uniform. No-op.
        _face_probs = None
    elif _use_goal_align and _beta > 0.0:
        # Stage 2B sampler bias: weight the face draw so the face whose
        # outward normal points opposite g_hat (contact would push box
        # toward goal) is over-represented. β = 0 -> uniform 1-of-4
        # (regression-safe identity). §9 Option B: for the tshape, only
        # SIDE faces have a horizontal normal for g_hat alignment; TOP
        # faces (nz=+1) do not project onto the planar g_hat and get
        # uniform weight.
        # max(0, -n_world . g_hat) per SIDE face -> 0 on anti-goal &
        # perpendicular, 1 on the perfectly goal-aligned face.
        _aligns_side = np.maximum(
            0.0, -(world_normals[:_n_horizontal, :2] @ g2))
        _weights_side = 1.0 + _beta * _aligns_side
        if shape == "tshape":
            # TOP faces: uniform weight (goal_align doesn't discriminate them).
            _weights_top = np.ones(n_top_faces, dtype=float)
            _face_weights = np.concatenate([_weights_side, _weights_top])
        else:
            _face_weights = _weights_side
        _face_probs = _face_weights / _face_weights.sum()
    else:
        _face_probs = None

    # D.3 (2026-07-13) — yaw-torque × translation-alignment face bias.
    #
    # For tasks with a rotation goal (T-push: goal_yaw = −0.7379 rad),
    # the sphere's contact torque about world-z depends on WHICH side
    # face it approaches. Two side faces with the same outward-normal
    # direction can produce yaw torques of OPPOSITE sign if their body-
    # frame centres sit on opposite sides of the object's z-axis.
    #
    # For a unit push force F = −n̂ (into the object) applied at
    # world-frame face-centre offset r_w = (rx, ry, 0) from the CoM:
    #     τ_z = (r_w × F)_z  =  rx·(−ny) − ry·(−nx)
    #                       =  ry·nx − rx·ny        (per unit force)
    #
    # Combined bonus = translation-align × yaw-align:
    #     trans_align = max(0, −n̂·g_hat)                (0..1, +ve when
    #                                                     push is goal-aligned)
    #     yaw_align   = max(0, sign(yaw_delta) · τ_z)   (goal-aligned yaw)
    #     bonus       = trans_align · yaw_align         (nonzero only when
    #                                                    BOTH match)
    #
    # This composition is critical: yaw-alone would reward faces whose
    # normal points AWAY from the goal (e.g. crossbar TOP for the T
    # seed 0 case with goal +y) if they happen to produce the right
    # torque sign, even though a push there would translate the T away
    # from the goal. Multiplying trans_align in eliminates that: face
    # 8 (crossbar top) has −n̂·g_hat = −0.995 → trans_align = 0 →
    # bonus = 0 regardless of yaw sign.
    #
    # Sanity check with T seed 0 (goal_yaw = −0.7379, goal_xy mostly
    # +y so g_hat ≈ (0, +1)):
    #   face 2 (crossbar bottom, r=(+0.05,−0.02), n=(0,−1)):
    #     τ_z   = (−0.02)·0 − 0.05·(−1) = +0.05         → CCW (wrong yaw)
    #     trans = max(0, −n̂·g_hat) = max(0, +0.995) ≈ 1.0
    #     bonus = 1.0 · max(0, −1·0.05) = 0
    #   face 4 (stem bottom,      r=(−0.05,−0.08), n=(0,−1)):
    #     τ_z   = (−0.08)·0 − (−0.05)·(−1) = −0.05      → CW  (correct)
    #     trans = 1.0
    #     bonus = 1.0 · max(0, −1·(−0.05)) = 0.05        (rewarded)
    #   face 8 (crossbar top,     r=(+0.05,+0.02), n=(0,+1)):
    #     τ_z   = +0.02·0 − 0.05·1 = −0.05              → CW (correct)
    #     trans = max(0, −n̂·g_hat) = max(0, −0.995) = 0  → zero-out
    #     bonus = 0                                      (wrong translation)
    # Face 4 alone gets the bonus.
    #
    # Universal safety:
    #   - yaw_delta = None or |yaw_delta| < eps → skip the multiplier
    #     entirely. Box has goal_yaw = 0 → yaw_delta stays near 0.
    #   - g_hat = None or zero-norm → skip. Rotation-only tasks with no
    #     translation goal don't benefit from this bias.
    #   - Top faces (T only): n_z = +1, so trans_align uses only n_xy
    #     component (0 for top faces) → bonus = 0. Top faces preserve
    #     their current area-weighted probability.
    #   - Box: for 4 cardinal side faces, (rx,ry) = box_half·(nx,ny),
    #     so ry·nx − rx·ny = 0 identically. A single-face push cannot
    #     yaw a cube. The bonus is null for the box shape regardless
    #     of yaml value. Bit-identical for box.
    _alpha = float(getattr(params, "yaw_face_bias_strength", 0.0))
    if (yaw_delta is not None
            and abs(float(yaw_delta)) > 1e-6
            and _alpha > 0.0
            and _use_goal_align
            and shape == "tshape"):
        # Box path never enters this block regardless of yaml value:
        # box has 4 cardinal side faces with (rx,ry) = box_half·(nx,ny),
        # so τ_z = ry·nx − rx·ny = 0 identically for every face — the
        # bonus is null and the multiplier would be all-ones, but the
        # renormalisation `_biased / _biased.sum()` still introduces a
        # 1-ULP float perturbation on _face_probs that propagates through
        # IK and Drake dynamics into macroscopic box divergence over
        # ~300 steps (same failure mode caught by the BUG-2 tolerance
        # guard). Gate on shape == "tshape" to keep box strictly
        # bit-identical.
        _sign = 1.0 if float(yaw_delta) > 0.0 else -1.0
        _rx = world_centers[:_n_horizontal, 0]
        _ry = world_centers[:_n_horizontal, 1]
        _nx = world_normals[:_n_horizontal, 0]
        _ny = world_normals[:_n_horizontal, 1]
        _tau_z_side = _ry * _nx - _rx * _ny
        _yaw_align_side = np.maximum(0.0, _sign * _tau_z_side)
        _trans_align_side = np.maximum(
            0.0, -(world_normals[:_n_horizontal, :2] @ g2))
        _bonus_side = _trans_align_side * _yaw_align_side
        _yaw_mult_side = 1.0 + _alpha * _bonus_side
        _yaw_mult_top = np.ones(n_top_faces, dtype=float)
        _yaw_mult = np.concatenate([_yaw_mult_side, _yaw_mult_top])

        if _face_probs is None:
            _face_probs = _yaw_mult / _yaw_mult.sum()
        else:
            _biased = _face_probs * _yaw_mult
            _face_probs = _biased / _biased.sum()

    # T1c effect-check hook: track face-pick histogram per _face_normal_projection
    # call. Emitted at end of loop as a single [SAMPLE-FACE-HIST] line — used by
    # scripts/_t1c_effect_check.sh to verify the observed distribution matches
    # area weights (not uniform). Cheap: ~1 line per call, ~800 calls per T sim.
    _face_hist = np.zeros(n_faces, dtype=int) if shape == "tshape" else None

    def _face_center_xy(face_idx: int) -> np.ndarray:
        if shape == "tshape":
            if face_idx < n_side_faces:
                # SIDE face: obj_xy + world_centers[face_idx] (world-frame
                # face-center offset from the T's link origin, already rotated).
                return obj_xy_2 + world_centers[face_idx, :2]
            # TOP face: obj_xy + top_world_centers (rotated body-frame center).
            _t_idx = face_idx - n_side_faces
            return obj_xy_2 + top_world_centers[_t_idx, :2]
        n_world = world_normals[face_idx]
        # Box: obj_xy + box_half * outward_normal (single scalar half-extent).
        return obj_xy_2 + box_half * n_world[:2]

    def _face_half_len(face_idx: int) -> float:
        return float(half_lens[face_idx]) if shape == "tshape" else box_half

    def _is_top_face(face_idx: int) -> bool:
        return shape == "tshape" and face_idx >= n_side_faces

    samples: list[np.ndarray] = []
    max_tries = n_samples * 20
    tries = 0
    while len(samples) < n_samples and tries < max_tries:
        tries += 1
        if _face_probs is None:
            face_idx = int(rng.integers(0, n_faces))
        else:
            face_idx = int(rng.choice(n_faces, p=_face_probs))
        if _face_hist is not None:
            _face_hist[face_idx] += 1

        # §9 Option B: TOP-face branch. Sphere descends onto the T's top
        # surface (world +z normal). Sample xy inside the top patch and set
        # sphere-center z = obj_z + T_half_height + top_setback (matches the
        # 0.5 mm sphere-lowest-to-top-surface geometry of the side-face
        # setback). obj_z here is assumed to be at the resting height
        # (T half_height above ground) since only obj_xy is passed in; the
        # setback margin absorbs small vertical drift.
        if _is_top_face(face_idx):
            _t_idx = face_idx - n_side_faces
            hx, hy = float(top_half_lens_xy[_t_idx, 0]), \
                     float(top_half_lens_xy[_t_idx, 1])
            # Uniform jitter within the (2·hx × 2·hy) top patch, in world
            # frame. Body-frame axes rotate through obj_quat; for the T's
            # z-only rotation the top-patch tangents rotate as
            # (cos θ, sin θ), (-sin θ, cos θ).
            j_body = np.array([
                float(rng.uniform(-hx, hx)),
                float(rng.uniform(-hy, hy)),
                0.0,
            ])
            if obj_quat is not None:
                j_world = R @ j_body
            else:
                j_world = j_body
            face_center_xy = _face_center_xy(face_idx)
            _sample_xy = face_center_xy + j_world[:2]
            _sample_z = (_TSHAPE_HALF_HEIGHT +          # obj_z at rest
                         _TSHAPE_HALF_HEIGHT +          # T top offset from center
                         _TSHAPE_TOP_SETBACK)           # +z setback above top
            samples.append(
                np.array([_sample_xy[0], _sample_xy[1], _sample_z]))
            continue

        n_world = world_normals[face_idx]
        face_center_xy = _face_center_xy(face_idx)
        _half_len = _face_half_len(face_idx)
        tang = np.array([-n_world[1], n_world[0]])
        tn = float(np.linalg.norm(tang))
        if tn > 1e-9:
            tang = tang / tn
        # Conditional jitter: on a goal-aligned face (contact would push obj
        # toward goal), tighten the range so the sample lands near the face
        # center. Off-center landings on the goal-aligned face produce a
        # moment arm → yaw → friction-coupled lateral drift; centered
        # landings push cleanly. Non-goal faces keep full jitter so rotation
        # tasks and multi-object scenarios retain sample diversity.
        if _use_goal_align:
            _goal_align = float(-n_world[0]*g2[0] - n_world[1]*g2[1])
            if _goal_align > GOAL_ALIGN_THRESHOLD:
                _jitter_range = _half_len * CENTERED_JITTER_FRACTION
            else:
                _jitter_range = _half_len
        else:
            _jitter_range = _half_len
        jitter = float(rng.uniform(-_jitter_range, _jitter_range))
        point_on_face_xy = face_center_xy + jitter * tang
        proj_xy = point_on_face_xy + setback * n_world[:2]
        # Rejection: for the box, an axis-aligned bbox check. For the T,
        # skip the concave-corner-aware check and just accept — the T's
        # setback-projected samples are always safe (each face's outward
        # normal doesn't re-enter the T's footprint at the setback distance
        # for setback ≥ pusher_radius; violations would require intra-T
        # geometry re-entry, ruled out by the per-face outward-normal choice).
        if shape == "box":
            dx = max(abs(proj_xy[0] - obj_xy_2[0]) - box_half, 0.0)
            dy = max(abs(proj_xy[1] - obj_xy_2[1]) - box_half, 0.0)
            surf_dist = float(np.hypot(dx, dy))
            if surf_dist < reject_clearance:
                continue
        samples.append(np.array([proj_xy[0], proj_xy[1], z]))

    # Pad with deterministic setback samples if the rejection loop failed.
    # (Side faces only; padding falls back to face 0 tangent.)
    while len(samples) < n_samples:
        _pad_side_idx = len(samples) % n_side_faces if shape == "tshape" \
                        else len(samples) % n_faces
        n_world = world_normals[_pad_side_idx]
        face_center_xy = _face_center_xy(_pad_side_idx)
        proj_xy = face_center_xy + setback * n_world[:2]
        samples.append(np.array([proj_xy[0], proj_xy[1], z]))

    # T1c — emit per-call face-pick histogram for the effect check. Only
    # for tshape (box's uniform-across-4-faces gives a boring histogram).
    if _face_hist is not None and int(_face_hist.sum()) > 0:
        print(f"[SAMPLE-FACE-HIST] n={int(_face_hist.sum())} "
              f"hist={_face_hist.tolist()} shape={shape} "
              f"mesh_normal={_use_mesh_normal}", flush=True)

    return samples


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Convert [w,x,y,z] quaternion to 3x3 rotation matrix. Returns
    identity for a zero-norm quaternion."""
    q = np.asarray(q, dtype=float).reshape(4)
    w, x, y, z = q
    n = w*w + x*x + y*y + z*z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1.0 - s*(y*y + z*z), s*(x*y - w*z),       s*(x*z + w*y)      ],
        [s*(x*y + w*z),       1.0 - s*(x*x + z*z), s*(y*z - w*x)      ],
        [s*(x*z - w*y),       s*(y*z + w*x),       1.0 - s*(x*x + y*y)],
    ])


def _fixed_samples(n_samples: int,
                   params:    SamplingParams) -> list[np.ndarray]:
    """Reserved for kFixed strategy — currently raises since the project's
    YAML doesn't yet expose a fixed_sample_locations field on SamplingParams.
    Add the field to params.py and remove this stub when needed."""
    raise NotImplementedError(
        "kFixed requires a fixed_sample_locations list on SamplingParams "
        "(matching upstream's Eigen::MatrixXd field). Not exposed yet.")


# ---------------------------------------------------------------------------
# Workspace filter
# ---------------------------------------------------------------------------

# Tolerance protects against float-arithmetic ε on bound-equal samples.
# Without this, samples within ε of an axis-aligned closed bound (e.g.,
# workspace_xy_max[1] = 0.0) are systematically rejected — see step 8
# receipts for the diag-kik 176/200-loop trace where every behind-box
# proxy sample was rejected at proxy_y ≈ +2.4e-6 m.
_WORKSPACE_BOUND_TOL: float = 1e-3  # 1 mm


def is_in_workspace(p: np.ndarray, params: SamplingParams) -> bool:
    """True iff sample p satisfies the workspace_xy / workspace_z bounds
    (within _WORKSPACE_BOUND_TOL)."""
    if p.shape != (3,):
        raise ValueError(f"sample must be (3,), got {p.shape}")
    tol = _WORKSPACE_BOUND_TOL
    if not (params.workspace_xy_min[0] - tol <= p[0] <= params.workspace_xy_max[0] + tol):
        return False
    if not (params.workspace_xy_min[1] - tol <= p[1] <= params.workspace_xy_max[1] + tol):
        return False
    if not (params.workspace_z_min   - tol <= p[2] <= params.workspace_z_max   + tol):
        return False
    return True
