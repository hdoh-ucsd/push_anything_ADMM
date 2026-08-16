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

import math
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


# H-shape side faces, body frame. Geometry from sim/env_builder.py
# `_hshape_sdf` (envelope 0.112 x 0.128 x 0.032, centred on the link origin):
#     left  bar  x[-0.056,-0.032]  y[-0.064,+0.064]
#     right bar  x[+0.032,+0.056]  y[-0.064,+0.064]
#     crossbar   x[-0.032,+0.032]  y[-0.012,+0.012]
# Twelve faces trace the outline. Validated: the face half-lengths sum to
# 0.688 m, exactly the analytic H perimeter, so the decomposition is complete
# and non-overlapping. The four notch-inner faces are reachable — projecting
# a sample outward by (pusher_r + clearance) = 39.5 mm leaves 24.5 mm of
# clearance to the opposing bar, above the 19.5 mm sphere radius.
_HSHAPE_FACE_TABLE = np.array([
    # (cx,     cy,     nx,   ny,   half_len)
    (-0.056,  0.000, -1.0,  0.0,  0.064),   # left bar, outer (-x)
    (-0.044, -0.064,  0.0, -1.0,  0.012),   # left bar, bottom (-y)
    (-0.044, +0.064,  0.0, +1.0,  0.012),   # left bar, top (+y)
    (-0.032, -0.038, +1.0,  0.0,  0.026),   # left bar, inner lower (+x)
    (-0.032, +0.038, +1.0,  0.0,  0.026),   # left bar, inner upper (+x)
    ( 0.000, -0.012,  0.0, -1.0,  0.032),   # crossbar, bottom (-y)
    ( 0.000, +0.012,  0.0, +1.0,  0.032),   # crossbar, top (+y)
    (+0.032, -0.038, -1.0,  0.0,  0.026),   # right bar, inner lower (-x)
    (+0.032, +0.038, -1.0,  0.0,  0.026),   # right bar, inner upper (-x)
    (+0.044, -0.064,  0.0, -1.0,  0.012),   # right bar, bottom (-y)
    (+0.044, +0.064,  0.0, +1.0,  0.012),   # right bar, top (+y)
    (+0.056,  0.000, +1.0,  0.0,  0.064),   # right bar, outer (+x)
], dtype=float)

_HSHAPE_HALF_HEIGHT = 0.016


# Registry of non-box ("polygonal") manipuland shapes. Each entry supplies the
# three things the perimeter sampler needs: the outline face table, the
# footprint as a union of body-frame AABBs, and the tight bounding box used to
# seed the uniform draw. Adding a shape here is what makes the reference
# kRandomOnPerimeter sampler work for it — the alternative is another
# `shape == "..."` branch at each of the four sites below.
_POLY_SHAPES = {
    "tshape": {
        "face_table": _TSHAPE_FACE_TABLE,
        # (x_lo, x_hi, y_lo, y_hi) per rectangle
        "rects": ((-0.03, +0.13, -0.02, +0.02),    # crossbar
                  (-0.07, -0.03, -0.08, +0.08)),   # stem
        "bbox":  (-0.07, +0.13, -0.08, +0.08),
        "half_height": _TSHAPE_HALF_HEIGHT,
    },
    "hshape": {
        "face_table": _HSHAPE_FACE_TABLE,
        "rects": ((-0.056, -0.032, -0.064, +0.064),   # left bar
                  (+0.032, +0.056, -0.064, +0.064),   # right bar
                  (-0.032, +0.032, -0.012, +0.012)),  # crossbar
        "bbox":  (-0.056, +0.056, -0.064, +0.064),
        "half_height": _HSHAPE_HALF_HEIGHT,
    },
}


def _poly_spec(shape: str):
    """Return the _POLY_SHAPES entry for `shape`, or None for box/sphere."""
    return _POLY_SHAPES.get(str(shape))


def _surface_clearance(x: float, y: float, rects) -> float:
    """Body-frame distance from (x, y) to the union of AABBs `rects`.

    Zero inside the footprint. Used to re-check projected samples: a sample is
    the EE SPHERE CENTRE, so anything closer than the pusher radius is a
    position the sphere cannot occupy.
    """
    best = float("inf")
    for x0, x1, y0, y1 in rects:
        dx = max(x0 - x, 0.0, x - x1)
        dy = max(y0 - y, 0.0, y - y1)
        d = math.hypot(dx, dy)
        if d < best:
            best = d
    return best


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
                     obj_z:       Optional[float] = None,
                     projector=None,
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
    elif strategy == SamplingStrategy.kRandomOnPerimeter:
        if projector is not None:
            # Geometry-generic path (2026-08-15 Fig 8 sampler fix): the
            # reference PerimeterSampling is face-table-free — it draws
            # interior points on the object's own footprint and projects
            # them off the ACTUAL collision geometry via signed-distance
            # queries (generate_samples.cc:270-360). The analytic
            # _POLY_SHAPES tables below remain only for the analytic-SDF
            # legacy tasks (push_t box-T, push_h), whose sim geometry IS
            # the table. Imported mesh objects sampled on the box-T table
            # produced targets at/inside their real surfaces → the
            # pursued-target collision gate dropped them every few ticks
            # (no_held retarget storm, chicken_broth 64%) → repositions
            # never completed → c3 never engaged (memory:
            # fig8-sampler-hardcoded-facetable-bug).
            raw = _random_on_perimeter_projected(
                n_samples, obj_xy, params, rng, obj_quat, obj_z, projector)
        else:
            raw = _random_on_perimeter(
                n_samples, obj_xy, params, rng, g_hat, obj_quat)
    elif strategy == SamplingStrategy.kRandomOnSphere:
        raw = _random_on_sphere(n_samples, obj_xy, params, rng, obj_z)
    else:
        raise NotImplementedError(
            f"Sampling strategy {strategy.name} not yet implemented; "
            f"only kRandomOnCircle, kRadiallySymmetric, kFaceNormal, "
            f"kRandomOnPerimeter, kFixed are supported.")

    if not params.filter_samples_for_safety:
        return raw
    return [p for p in raw if is_in_workspace(p, params)]


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def _random_on_perimeter_projected(n_samples: int,
                                   obj_xy:    np.ndarray,
                                   params:    SamplingParams,
                                   rng:       np.random.Generator,
                                   obj_quat:  Optional[np.ndarray],
                                   obj_z:     Optional[float],
                                   projector) -> list[np.ndarray]:
    """Geometry-generic kRandomOnPerimeter for arbitrary mesh objects.

    Port of the reference mechanism (generate_samples.cc:270-360,
    PerimeterSampling + ProjectSampleOutsideObject), which needs no
    per-shape face table:

      1. Draw (x, y) uniformly in the object's BODY frame on its z=0
         plane; rotate/translate into world; snap world z to
         sampling_height.
      2. Keep the draw only if it is interior (signed distance to the
         object's surface <= 0 at that height) — this makes the draw an
         interior-point sampler over the true footprint slice.
      3. Project the point outward past the surface along the
         signed-distance gradient to sample_projection_clearance.
      4. Re-query: reject if the achieved clearance fell short (witness
         changed during projection) or the point drifted more than 1 mm
         off sampling_height (non-vertical wall). Retry otherwise.

    `projector` is a callable p(3,) -> (min_signed_distance, grad_W) over
    the manipuland's collision geometries in the CURRENT plant context
    (built per tick by the dispatcher; same query the pursued-target
    collision gate uses, so sampler and gate agree by construction).

    The reference draws from per-task grid_x/y_limits; the port draws
    from a generous fixed body-frame box (covers every imported object;
    exterior draws are rejected by step 2, so the grid affects only
    draw efficiency, not the distribution over the footprint slice).
    Bounded retries replace the reference's unbounded loop so a
    degenerate tick cannot hang the controller.
    """
    half = 0.20
    # Projection standoff (2026-08-15, third and reference-exact iteration).
    # ProjectSampleOutsideObject (generate_samples.cc:839-868) projects the
    # sample from the OBJECT WITNESS POINT outward by
    #     ee_radius + sample_projection_clearance
    # (GetEERadiusFromPlant — the EE radius is ADDED, not floored), so the
    # reference sphere's SURFACE clears the object by the full 2 cm yaml
    # clearance. Iteration history: v1 used the raw 0.02 as an EE-CENTER
    # clearance (sphere penetrated ~5 mm; mesh-T spun to 1.86 rad,
    # meshT_projfix_seed0.log); v2 floored at PUSHER_RADIUS+5 mm (sphere
    # cleared only 5 mm; T STILL spun to 1.80, meshT_clrfix_seed0.log —
    # c3 engaged from near-touching starts and scraped the T around in a
    # ~90° contact orbit, steps 150-200). The additive reference formula
    # gives c3 a genuinely clear starting standoff.
    from sim.env_builder import PUSHER_RADIUS as _PUSHER_R
    clearance = (float(_PUSHER_R)
                 + float(getattr(params, "sample_projection_clearance", 0.02)))
    h = float(params.sampling_height)
    q = (np.asarray(obj_quat, dtype=float)
         if obj_quat is not None else np.array([1.0, 0.0, 0.0, 0.0]))
    R = _quat_to_rot(q)
    z_obj = float(obj_z) if obj_z is not None else h
    p_obj = np.array([float(obj_xy[0]), float(obj_xy[1]), z_obj])

    out: list[np.ndarray] = []
    attempts = 0
    max_attempts = 400 * max(int(n_samples), 1)
    while len(out) < n_samples and attempts < max_attempts:
        attempts += 1
        xb, yb = rng.uniform(-half, half, size=2)
        p = R @ np.array([xb, yb, 0.0]) + p_obj
        p[2] = h
        d, grad = projector(p)
        if d is None or d > 0.0:
            continue                      # not interior at the slice
        gn = float(np.linalg.norm(grad))
        if gn < 1e-9:
            continue
        p_proj = p + (clearance - d) * (grad / gn)
        d2, _ = projector(p_proj)
        if d2 is None or d2 < clearance - 1e-6:
            continue                      # projection under-cleared
        if abs(float(p_proj[2]) - h) > 0.001:
            continue                      # wall not vertical here
        out.append(np.asarray(p_proj, dtype=float))
    return out


def _random_on_sphere(n_samples: int,
                      obj_xy:    np.ndarray,
                      params:    SamplingParams,
                      rng:       np.random.Generator,
                      obj_z:     Optional[float] = None) -> list[np.ndarray]:
    """kRandomOnSphere — reference generate_samples.cc RandomOnSphereSampling.

    Reference body, verbatim in structure:

        theta           = U(0, 2*pi)
        elevation_theta = U(min_angle_from_vertical, max_angle_from_vertical)
        sample.x = obj.x + r * cos(theta) * sin(elevation_theta)
        sample.y = obj.y + r * sin(theta) * sin(elevation_theta)
        sample.z = obj.z + r * cos(elevation_theta)

    Note this is NOT a uniform distribution over the spherical cap — the
    reference draws the elevation angle uniformly, which concentrates samples
    near the pole. Reproduced as-is rather than "corrected": matching the
    reference's sample density matters more than statistical uniformity.

    Unlike the planar samplers this needs the object's full 3D position, since
    the sphere is centred on the object rather than projected to a fixed world
    height. `obj_z` falls back to the configured sampling_height when the
    caller has not supplied it.

    Used by the reference `jacktoy` task (sampling_strategy: 2), whose jack is
    a 3-capsule caltrop that rolls between tripods — it has no flat resting
    face, so the perimeter/face-table samplers do not apply to it.
    """
    r = float(params.sampling_radius)
    lo = float(getattr(params, "min_angle_from_vertical", 0.0))
    hi = float(getattr(params, "max_angle_from_vertical", math.pi))
    cz = float(obj_z) if obj_z is not None else float(params.sampling_height)
    cx, cy = float(obj_xy[0]), float(obj_xy[1])

    out: list[np.ndarray] = []
    tries = 0
    max_tries = max(1, n_samples) * 40
    while len(out) < n_samples and tries < max_tries:
        tries += 1
        theta = float(rng.uniform(0.0, 2.0 * math.pi))
        elev = float(rng.uniform(lo, hi))
        p = np.array([cx + r * math.cos(theta) * math.sin(elev),
                      cy + r * math.sin(theta) * math.sin(elev),
                      cz + r * math.cos(elev)])
        # Reference wraps the draw in a do/while on SampleIsAcceptable; the
        # port's equivalent acceptance test is the workspace filter.
        if is_in_workspace(p, params):
            out.append(p)
    return out


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
        #
        # 2026-07-26 investigation of rot ceiling (rot stuck at ~0.4 rad
        # through all arc-2 experiments): the centering logic works AGAINST
        # yaw-goal tasks like push_t (goal_yaw = -0.7379 rad, w_yaw = 800).
        # When yaw needs correction, we WANT off-center samples to explore
        # lever-arm contacts. Env-gated inversion: under strong yaw_delta,
        # scale jitter UP toward full half_len based on urgency. Off by
        # default (PORT_SAMPLER_YAW_JITTER_MAX=0.0). Only fires when goal-
        # aligned AND yaw_delta > threshold.
        import os as _os_yj
        _yaw_jitter_max = float(_os_yj.environ.get(
            "PORT_SAMPLER_YAW_JITTER_MAX", "0.0"))
        _yaw_jitter_scale = float(_os_yj.environ.get(
            "PORT_SAMPLER_YAW_URGENCY_SCALE", "0.5"))
        if _use_goal_align:
            _goal_align = float(-n_world[0]*g2[0] - n_world[1]*g2[1])
            if _goal_align > GOAL_ALIGN_THRESHOLD:
                # Yaw-aware jitter expansion: max(centered_default, urgency*scale)
                _yaw_urgency = min(
                    1.0, abs(float(yaw_delta or 0.0)) / max(_yaw_jitter_scale, 1e-6))
                _yaw_frac = _yaw_jitter_max * _yaw_urgency
                _jitter_frac = max(CENTERED_JITTER_FRACTION, _yaw_frac)
                _jitter_range = _half_len * _jitter_frac
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


def _random_on_perimeter(n_samples: int,
                         obj_xy:    np.ndarray,
                         params:    SamplingParams,
                         rng:       np.random.Generator,
                         g_hat:     Optional[np.ndarray],
                         obj_quat:  Optional[np.ndarray]) -> list[np.ndarray]:
    """Reference-conformant kRandomOnPerimeter sampler.

    Mirrors dairlib_sampling_c3 generate_samples.cc:270-362
    (`PerimeterSampling`).  Algorithm (per sample):

      1. Draw uniform random (x_body, y_body) inside the object's body-
         frame bounding box (grid_x_limits × grid_y_limits).  Set
         z_body = 0.
      2. Rotate to world frame via obj_quat + obj_xy, snap z to
         sampling_height.
      3. Reject if the (x_body, y_body) is OUTSIDE the object's
         footprint — reference uses collision-query penetration test
         `IsSampleWithinDistanceOfSurface(clearance=0)`.  Port uses the
         geometry-table AABB test (T = union of two rectangles; box =
         single rectangle) — same effect, no Drake dependency.
      4. Find the nearest object edge (`_TSHAPE_FACE_TABLE` for tshape,
         4-face box table for box).  Project the sample OUTWARD along
         that edge's world-frame normal by `sample_projection_clearance`.
         Reference uses `ProjectSampleOutsideObject(...)` with actual
         witness points from `GeomGeomCollider::CalcWitnessPoints`; the
         port approximates using the nearest face-line foot-of-
         perpendicular.
      5. Filter by workspace bounds + z tolerance (reference rejects
         when projected z drifts more than 1 mm from `sampling_height`).

    Distributional equivalence: uniform-in-bounding-box + reject-if-
    outside gives a distribution proportional to *perimeter length* on
    the boundary — same as reference's post-projection distribution.

    Not passed through Drake: this is a pure-numpy geometric port.
    Concave regions (e.g. T's shoulder inside corner) may produce
    small artifacts vs the reference's true witness-point projection.
    """
    obj_xy_2 = np.asarray(obj_xy, dtype=float).flatten()[:2]
    shape = getattr(params, "object_shape", "box")
    z_target = float(params.sampling_height)
    # Outward projection distance — reference generate_samples.cc:854-866
    # (ProjectSampleOutsideObject): sample point = witness point +
    # (ee_radius + sample_projection_clearance) · n̂, then re-checked and
    # retried until strictly clear (cc:340-348). The point-to-surface
    # distance is therefore ee_radius + clearance — clearing the per-tick
    # pursued-target gate (cc:908-926, distance <= clearance) with an
    # ee_radius margin. The port previously projected by sampling_setback
    # alone (kik_t yaml 0.020, misread as the reference clearance): samples
    # landed exactly AT the gate boundary and p139 rejected 698 of the
    # sampler's own fresh outputs. sampling_setback remains the kFaceNormal
    # knob; it is not used here.
    from sim.env_builder import PUSHER_RADIUS as _PUSHER_R
    clearance = float(_PUSHER_R) + float(getattr(
        params, "sample_projection_clearance", 0.02))

    R = _quat_to_rot(obj_quat) if obj_quat is not None else np.eye(3)

    # Bounding-box (body-frame) for the uniform draw.  Prefer explicit
    # grid_x_limits / grid_y_limits if present (matches reference YAML
    # names); else derive from the shape's face-table extents.
    _has_gx = hasattr(params, "grid_x_limits") \
        and params.grid_x_limits is not None
    _has_gy = hasattr(params, "grid_y_limits") \
        and params.grid_y_limits is not None
    if _has_gx and _has_gy:
        gx = np.asarray(params.grid_x_limits, dtype=float).flatten()
        gy = np.asarray(params.grid_y_limits, dtype=float).flatten()
        x_lo, x_hi = float(gx[0]), float(gx[1])
        y_lo, y_hi = float(gy[0]), float(gy[1])
    elif _poly_spec(shape) is not None:
        # Tight bounding box from the shape's face-table extents.
        x_lo, x_hi, y_lo, y_hi = _poly_spec(shape)["bbox"]
    else:
        h = float(params.box_half_extent)
        x_lo, x_hi = -h, +h
        y_lo, y_hi = -h, +h

    # Face table (body-frame).  Each row: (cx, cy, nx, ny, half_len).
    _spec = _poly_spec(shape)
    if _spec is not None:
        face_table = _spec["face_table"]
    else:
        h = float(params.box_half_extent)
        face_table = np.array([
            (+h, 0.0, +1.0, 0.0, h),
            (-h, 0.0, -1.0, 0.0, h),
            (0.0, +h, 0.0, +1.0, h),
            (0.0, -h, 0.0, -1.0, h),
        ], dtype=float)

    def _inside_footprint(x: float, y: float) -> bool:
        # Polygonal shapes are a union of body-frame AABBs (T = crossbar +
        # stem; H = left bar + right bar + crossbar).
        if _spec is not None:
            for x0, x1, y0, y1 in _spec["rects"]:
                if (x0 <= x <= x1) and (y0 <= y <= y1):
                    return True
            return False
        h = float(params.box_half_extent)
        return (abs(x) <= h) and (abs(y) <= h)

    def _project_to_nearest_face(x: float, y: float) -> np.ndarray:
        """Return the projected body-frame point AFTER pushing outward by
        `clearance` along the nearest face's outward normal.  Ref
        equivalent of ProjectSampleOutsideObject via face-table lookup."""
        best_face = None
        best_d2   = np.inf
        for i in range(face_table.shape[0]):
            cx, cy, nx, ny, hl = face_table[i]
            # Tangent = 90° CCW of normal.
            tx, ty = -ny, nx
            # Foot of perpendicular from (x, y) onto face segment center-line.
            dpx, dpy = x - cx, y - cy
            dt = dpx * tx + dpy * ty       # along-tangent projection
            dt = np.clip(dt, -hl, +hl)
            foot_x = cx + dt * tx
            foot_y = cy + dt * ty
            d2 = (x - foot_x) ** 2 + (y - foot_y) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_face = (foot_x, foot_y, nx, ny)
        assert best_face is not None
        foot_x, foot_y, nx, ny = best_face
        return np.array([foot_x + clearance * nx,
                         foot_y + clearance * ny])

    samples: list[np.ndarray] = []
    max_tries = n_samples * 40
    tries = 0
    while len(samples) < n_samples and tries < max_tries:
        tries += 1
        x_body = rng.uniform(x_lo, x_hi)
        y_body = rng.uniform(y_lo, y_hi)
        # Step 3: reject if outside the T/box footprint (must originally
        # penetrate; reference uses phi ≤ -1 mm).
        if not _inside_footprint(x_body, y_body):
            continue
        # Step 4: project outward to nearest face + clearance.
        proj_body = _project_to_nearest_face(x_body, y_body)
        # Step 4b: RE-CHECK that the projected point is actually clear of the
        # whole object, not just of the face it was projected from. Reference
        # ProjectSampleOutsideObject re-checks and retries until strictly clear
        # (generate_samples.cc:340-348); the port previously projected once and
        # accepted. On a CONCAVE outline the nearest-face projection can land
        # beside a different part of the body: measured 21% of T samples and
        # 32% of H samples closer than the pusher radius, i.e. positions the EE
        # sphere cannot physically occupy. Retry (the caller's while-loop draws
        # a fresh point) rather than accept an unreachable target.
        if _spec is not None:
            if _surface_clearance(proj_body[0], proj_body[1],
                                  _spec["rects"]) < _PUSHER_R:
                tries += 1
                continue
        # Step 2/5: rotate to world, snap z, filter workspace.
        p_body_3d = np.array([proj_body[0], proj_body[1], 0.0])
        p_world = R @ p_body_3d + np.array([obj_xy_2[0], obj_xy_2[1], 0.0])
        p_world[2] = z_target
        samples.append(p_world)

    return samples


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
def is_in_workspace(p: np.ndarray, params: SamplingParams) -> bool:
    """True iff sample p satisfies the workspace_xy / workspace_z bounds and
    the robot_radius_limits shell, with x/y and the radial shell inset by
    `workspace_margins`.

    Reference IsSampleInWorkspace (generate_samples.cc:760-775) rejects
    samples within workspace_margins of the x/y bounds and the radial
    shell; z bounds carry no margin. The live-EE DRAKE_DEMAND check
    (sampling_based_c3_controller.cc:1476-1494) has NO margin — this inset
    is what keeps commanded targets clear of the abort boundary."""
    if p.shape != (3,):
        raise ValueError(f"sample must be (3,), got {p.shape}")
    m = float(params.workspace_margins)
    if not (params.workspace_xy_min[0] + m <= p[0] <= params.workspace_xy_max[0] - m):
        return False
    if not (params.workspace_xy_min[1] + m <= p[1] <= params.workspace_xy_max[1] - m):
        return False
    if not (params.workspace_z_min <= p[2] <= params.workspace_z_max):
        return False
    r = float(np.hypot(p[0], p[1]))
    r_min, r_max = params.robot_radius_limits
    if not (r_min + m <= r <= r_max - m):
        return False
    return True
