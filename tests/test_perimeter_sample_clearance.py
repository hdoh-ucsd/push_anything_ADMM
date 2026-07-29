"""kRandomOnPerimeter samples must clear the pursued-target collision gate.

Reference generate_samples.cc:854-866 (ProjectSampleOutsideObject): the
sample point is placed at

    witness_point + (ee_radius + sample_projection_clearance) * n_hat

i.e. point-to-surface distance = ee_radius + clearance (= 0.0195 + 0.02 =
0.0395 for push_t), then re-checked and retried until strictly clear
(generate_samples.cc:340-348). The per-tick pursued-target gate
(cc:908-926) measures POINT distance <= sample_projection_clearance, so a
reference sample always clears it with an ee_radius margin.

The port projected by `sampling_setback` alone (kik_t.yaml had 0.020,
believing it equaled the reference clearance) — samples landed exactly AT
the 0.02 gate boundary: p139 logged 698 [REPOS-COLLIDE] rejections of the
sampler's own fresh output, c3 entries collapsed 111 -> 50, rotation
stalled (0.6917 vs 0.7379 initial).
"""
import numpy as np

from control.sampling_c3.params import SamplingC3Params, SamplingStrategy
from control.sampling_c3.sampling import generate_samples
from sim.env_builder import PUSHER_RADIUS


def _params():
    sp = SamplingC3Params().sampling_params
    sp.sampling_strategy = SamplingStrategy.kRandomOnPerimeter
    sp.object_shape = "box"
    sp.box_half_extent = 0.05
    sp.sampling_height = 0.034
    sp.sampling_setback = 0.020            # the old (wrong) projection dist
    sp.sample_projection_clearance = 0.02
    sp.grid_x_limits = None
    sp.grid_y_limits = None
    # Generous workspace so the filter never bites.
    sp.workspace_xy_min = [-2.0, -2.0]
    sp.workspace_xy_max = [2.0, 2.0]
    sp.workspace_z_min = 0.0
    sp.workspace_z_max = 1.0
    return sp


def _rect_surface_distance(p_xy, h):
    """Exact 2D distance from point to the boundary of the square [-h,h]^2
    (positive outside)."""
    dx = max(abs(p_xy[0]) - h, 0.0)
    dy = max(abs(p_xy[1]) - h, 0.0)
    return float(np.hypot(dx, dy))


def test_perimeter_samples_clear_projection_gate():
    sp = _params()
    rng = np.random.default_rng(0)
    obj_xy = np.array([0.5, 0.0])
    samples = generate_samples(
        strategy=SamplingStrategy.kRandomOnPerimeter,
        n_samples=40, obj_xy=obj_xy, params=sp, rng=rng,
        g_hat=None, obj_quat=None)
    assert len(samples) > 0
    gate = float(sp.sample_projection_clearance)
    expected = PUSHER_RADIUS + gate       # reference offset, cc:861-864
    for s in samples:
        d = _rect_surface_distance(
            (s[0] - obj_xy[0], s[1] - obj_xy[1]), sp.box_half_extent)
        # Must clear the per-tick gate strictly, with the ee_radius margin
        # the reference guarantees (small tolerance for corner projection).
        assert d > gate + 1e-9, f"sample {s} at surface dist {d:.4f}"
        assert d >= expected - 1e-6, f"sample {s} at surface dist {d:.4f}"
