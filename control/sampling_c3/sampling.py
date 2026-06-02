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
CENTERED_JITTER_FRACTION = 0.2      # 0.2 × box_half = ±10 mm on a 100 mm box


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
            n_samples, obj_xy, params, rng, g_hat, obj_quat)
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
                            obj_quat:     Optional[np.ndarray]) -> list[np.ndarray]:
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

    # Body-frame outward normals of the 4 side faces (+x, -x, +y, -y).
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

    obj_xy_2 = np.asarray(obj_xy, dtype=float).reshape(2)

    # Stage 2B sampler bias: weight the face draw so the face whose outward
    # normal points opposite g_hat (contact would push box toward goal) is
    # over-represented. β = 0 -> uniform 1-of-4 (regression-safe identity).
    _beta = float(getattr(params, "face_bias_strength", 0.0))
    if _use_goal_align and _beta > 0.0:
        # max(0, -n_world . g_hat) per face -> 0 on anti-goal & perpendicular,
        # 1 on the perfectly goal-aligned face (downhill bias).
        _aligns = np.maximum(0.0, -(world_normals[:, :2] @ g2))
        _face_weights = 1.0 + _beta * _aligns
        _face_probs = _face_weights / _face_weights.sum()
    else:
        _face_probs = None

    samples: list[np.ndarray] = []
    max_tries = n_samples * 20
    tries = 0
    while len(samples) < n_samples and tries < max_tries:
        tries += 1
        if _face_probs is None:
            face_idx = int(rng.integers(0, 4))
        else:
            face_idx = int(rng.choice(4, p=_face_probs))
        n_world = world_normals[face_idx]
        face_center_xy = obj_xy_2 + box_half * n_world[:2]
        tang = np.array([-n_world[1], n_world[0]])
        tn = float(np.linalg.norm(tang))
        if tn > 1e-9:
            tang = tang / tn
        # Conditional jitter: on a goal-aligned face (contact would push box
        # toward goal), tighten the range so the sample lands near the face
        # center. Off-center landings on the goal-aligned face produce a
        # moment arm → yaw → friction-coupled lateral drift; centered
        # landings push cleanly. Non-goal faces keep full jitter so rotation
        # tasks and multi-object scenarios retain sample diversity.
        # Force on box from a sample-side approach is along -n_world; align
        # that with g_hat (box→goal) to score the face.
        if _use_goal_align:
            _goal_align = float(-n_world[0]*g2[0] - n_world[1]*g2[1])
            if _goal_align > GOAL_ALIGN_THRESHOLD:
                _jitter_range = box_half * CENTERED_JITTER_FRACTION
            else:
                _jitter_range = box_half
        else:
            _jitter_range = box_half
        jitter = float(rng.uniform(-_jitter_range, _jitter_range))
        point_on_face_xy = face_center_xy + jitter * tang
        proj_xy = point_on_face_xy + setback * n_world[:2]
        # Rejection: in body-frame (post-projection), the |max(x,y)| relative
        # to the box center must exceed box_half + reject_clearance. We
        # approximate in world frame by axis-aligned bbox check, which is
        # exact when obj_quat is None and an upper bound on penetration in
        # the rotated case.
        dx = max(abs(proj_xy[0] - obj_xy_2[0]) - box_half, 0.0)
        dy = max(abs(proj_xy[1] - obj_xy_2[1]) - box_half, 0.0)
        surf_dist = float(np.hypot(dx, dy))
        if surf_dist < reject_clearance:
            continue
        samples.append(np.array([proj_xy[0], proj_xy[1], z]))

    # Pad with deterministic cardinal-direction setback samples if the
    # rejection loop failed (shouldn't happen for a free-standing cube).
    while len(samples) < n_samples:
        idx = len(samples) % 4
        n_world = world_normals[idx]
        face_center_xy = obj_xy_2 + box_half * n_world[:2]
        proj_xy = face_center_xy + setback * n_world[:2]
        samples.append(np.array([proj_xy[0], proj_xy[1], z]))

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
