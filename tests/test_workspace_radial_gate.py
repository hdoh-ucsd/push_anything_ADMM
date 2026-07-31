"""is_in_workspace must reject candidates outside the robot radius shell.

Reference IsSampleInWorkspace (generate_samples.cc:364-378) tests the
candidate against the axis-aligned workspace_limits AND

    candidate_radius = sqrt(x^2 + y^2)
    candidate_radius < robot_radius_limits[0]  -> reject
    candidate_radius > robot_radius_limits[1]  -> reject

The port's is_in_workspace (sampling.py) checked only the AABB, even though
SamplingParams.robot_radius_limits has existed since the runtime
CheckForWorkspaceLimitViolations port (commit 6563e37). A candidate inside
the AABB but inside r_min (or beyond r_max) could therefore be proposed as
a repos target that the reference would never generate.
"""
import numpy as np

from control.sampling_c3.params import SamplingC3Params
from control.sampling_c3.sampling import is_in_workspace


def _params(radius_limits):
    sp = SamplingC3Params().sampling_params
    # Generous AABB so only the radial term can reject.
    sp.workspace_xy_min = [-2.0, -2.0]
    sp.workspace_xy_max = [2.0, 2.0]
    sp.workspace_z_min = 0.0
    sp.workspace_z_max = 1.0
    sp.robot_radius_limits = list(radius_limits)
    return sp


def test_rejects_inside_r_min():
    sp = _params([0.25, 0.75])
    # r = 0.20 < 0.25, well inside the AABB.
    assert not is_in_workspace(np.array([0.20, 0.0, 0.1]), sp)


def test_rejects_beyond_r_max():
    sp = _params([0.25, 0.75])
    # r = sqrt(0.6^2 + 0.6^2) = 0.849 > 0.75, inside the AABB.
    assert not is_in_workspace(np.array([0.60, 0.60, 0.1]), sp)


def test_accepts_inside_shell():
    sp = _params([0.25, 0.75])
    # r = 0.50, mid-shell.
    assert is_in_workspace(np.array([0.50, 0.0, 0.1]), sp)


def test_default_limits_change_nothing():
    # Default [0.0, 100.0] is effectively unbounded: legacy callers see no
    # behaviour change.
    sp = _params([0.0, 100.0])
    assert is_in_workspace(np.array([0.01, 0.0, 0.1]), sp)
    assert is_in_workspace(np.array([1.9, 0.0, 0.1]), sp)
