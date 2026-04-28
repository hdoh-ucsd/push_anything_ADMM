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


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------

def generate_samples(strategy:    SamplingStrategy,
                     n_samples:   int,
                     obj_xy:      np.ndarray,
                     params:      SamplingParams,
                     rng:         Optional[np.random.Generator] = None,
                     g_hat:       Optional[np.ndarray] = None,
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
    elif strategy == SamplingStrategy.kFixed:
        raw = _fixed_samples(n_samples, params)
    else:
        raise NotImplementedError(
            f"Sampling strategy {strategy.name} not yet implemented; "
            f"only kRandomOnCircle, kRadiallySymmetric, kFixed are supported.")

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
    """Random points uniformly on the circle of radius params.sampling_radius
    around obj_xy at z = params.sampling_height.

    When g_hat is provided, the proxy point (behind the box on the push
    axis) is forced to be sample 0 so the controller always considers it;
    remaining samples are uniform random on the circle. This matches the
    behavior of the legacy global_sampling_c3.py wrapper that always
    seeded k=1=proxy.
    """
    samples: list[np.ndarray] = []
    r = float(params.sampling_radius)
    z = float(params.sampling_height)

    if g_hat is not None and n_samples >= 1:
        # Mandatory proxy point: behind the box on the push axis
        proxy = np.array([
            obj_xy[0] - r * float(g_hat[0]),
            obj_xy[1] - r * float(g_hat[1]),
            z,
        ])
        samples.append(proxy)
        n_remain = n_samples - 1
    else:
        n_remain = n_samples

    for _ in range(n_remain):
        theta = rng.uniform(0.0, 2.0 * np.pi)
        samples.append(np.array([
            obj_xy[0] + r * np.cos(theta),
            obj_xy[1] + r * np.sin(theta),
            z,
        ]))
    return samples


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

def is_in_workspace(p: np.ndarray, params: SamplingParams) -> bool:
    """True iff sample p satisfies the workspace_xy / workspace_z bounds."""
    if p.shape != (3,):
        raise ValueError(f"sample must be (3,), got {p.shape}")
    if not (params.workspace_xy_min[0] <= p[0] <= params.workspace_xy_max[0]):
        return False
    if not (params.workspace_xy_min[1] <= p[1] <= params.workspace_xy_max[1]):
        return False
    if not (params.workspace_z_min <= p[2] <= params.workspace_z_max):
        return False
    return True
