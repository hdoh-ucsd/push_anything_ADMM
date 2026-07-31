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


def test_default_limits_reject_only_margin_shell():
    # Default [0.0, 100.0] is effectively unbounded; only the
    # workspace_margins inset (0.02) around r=0 can reject.
    sp = _params([0.0, 100.0])
    assert is_in_workspace(np.array([0.03, 0.0, 0.1]), sp)
    assert is_in_workspace(np.array([1.9, 0.0, 0.1]), sp)


# ---------------------------------------------------------------------------
# workspace_margins inset — reference IsSampleInWorkspace
# (generate_samples.cc:760-775) rejects samples within `workspace_margins`
# of the x/y bounds and the radial shell; z bounds carry no margin. The
# live-EE DRAKE_DEMAND check (sampling_based_c3_controller.cc:1476-1494)
# has NO margin, so this inset is what keeps commanded targets clear of
# the crash boundary.
# ---------------------------------------------------------------------------

def test_margin_rejects_just_inside_r_min():
    sp = _params([0.25, 0.75])          # workspace_margins default 0.02
    # r = 0.26 is inside the shell but within the 0.02 margin of r_min.
    assert not is_in_workspace(np.array([0.26, 0.0, 0.1]), sp)
    # r = 0.28 clears the margin.
    assert is_in_workspace(np.array([0.28, 0.0, 0.1]), sp)


def test_margin_rejects_just_inside_r_max():
    sp = _params([0.25, 0.75])
    assert not is_in_workspace(np.array([0.74, 0.0, 0.1]), sp)
    assert is_in_workspace(np.array([0.72, 0.0, 0.1]), sp)


def test_margin_insets_xy_bounds_not_z():
    sp = _params([0.0, 100.0])
    sp.workspace_xy_min = [-0.5, -0.5]
    sp.workspace_xy_max = [0.5, 0.5]
    # x within 0.02 of x_max → reject; z exactly at its bound → accept
    # (reference applies no margin on z).
    assert not is_in_workspace(np.array([0.49, 0.0, 0.1]), sp)
    assert is_in_workspace(np.array([0.47, 0.0, sp.workspace_z_min]), sp)
