"""Stage 2 L2 commit-face gate (pure helper).

Refuses ``kToC3ReachedReposTarget`` when the active reposition target's
outward face is not goal-aligned within a configurable cone. Imported
by ``wrapper.py`` at the ``finished_repos`` evaluation site (Task 4).

Geometry. Let ``n_face_out`` be the unit vector from the box center to
the reposition target's XY projection. The EE will push the box
along ``-n_face_out``. When ``face_align = n_face_out · g_hat == -1``,
the EE sits on the anti-goal face and its push is perfectly
goal-aligned — the desired contact geometry. Therefore COMMIT iff
``face_align <= threshold`` (more negative = more anti-goal face =
more goal-aligned push). Default ``threshold = -0.7`` is a 45-degree
cone: ``-cos(45°) ≈ -0.707``.

Relationship to L1 (``entry_align_threshold`` in wrapper.py:903-937).
L1 keys on ``formulator._last_contact_info`` (``nhat_onto_box``, into
the box) and is structurally empty 80 mm pre-contact. L2 keys on
``_current_repos_target`` (the IK setback target) which is populated
by definition when ``finished_repos == True``. Sign conventions are
inverted: L1 refuses when ``align <= thr`` (uses nhat_INTO_box);
L2 commits when ``align <= thr`` (uses n_face_OUT_of_box). The
inequality is inclusive on the commit side at the boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


NEAR_MISS_UPPER = -0.3
PERPENDICULAR_UPPER = 0.3


@dataclass(frozen=True)
class CommitFaceDecision:
    commit: bool
    face_align: float
    severity_tag: str
    n_face_out_xy: Tuple[float, float]


def decide_commit_face_gate(
    p_repos_target_xy: np.ndarray,
    box_xy: np.ndarray,
    g_hat_xy: np.ndarray,
    threshold: float = -0.7,
) -> CommitFaceDecision:
    """Decide commit/refuse for the L2 commit-face gate.

    Returns ``commit=True`` iff ``face_align <= threshold`` (boundary
    inclusive on the commit side). Always returns a non-None decision
    with diagnostics — degenerate target (target on box center)
    REFUSES with tag ``'perpendicular'`` and face_align=0.

    Parameters
    ----------
    p_repos_target_xy
        XY of the EE reposition setback target. Shape (2,) or (3,);
        only the first two components are used.
    box_xy
        XY of the box center. Shape (2,) or (3,); only the first two
        components are used.
    g_hat_xy
        Unit goal direction in the XY plane. Shape (2,) or (3,).
    threshold
        Commit-face cosine threshold. Default -0.7.

    Severity tags (refusal categories, derived from raw ``face_align``):
        ``'commit'``         when ``face_align <= threshold``
        ``'near-miss-cone'`` when ``threshold < face_align <= -0.3``
        ``'perpendicular'``  when ``-0.3 < face_align <= +0.3``
        ``'anti-goal'``      when ``+0.3 < face_align``
    """
    p_xy = np.asarray(p_repos_target_xy, dtype=float).flatten()[:2]
    b_xy = np.asarray(box_xy, dtype=float).flatten()[:2]
    g_xy = np.asarray(g_hat_xy, dtype=float).flatten()[:2]

    delta = p_xy - b_xy
    norm = float(np.linalg.norm(delta))
    if norm < 1e-9:
        return CommitFaceDecision(
            commit=False,
            face_align=0.0,
            severity_tag="perpendicular",
            n_face_out_xy=(0.0, 0.0),
        )

    n_face = delta / norm
    face_align = float(n_face[0] * g_xy[0] + n_face[1] * g_xy[1])

    commit = face_align <= threshold
    if commit:
        tag = "commit"
    elif face_align <= NEAR_MISS_UPPER:
        tag = "near-miss-cone"
    elif face_align <= PERPENDICULAR_UPPER:
        tag = "perpendicular"
    else:
        tag = "anti-goal"

    return CommitFaceDecision(
        commit=commit,
        face_align=face_align,
        severity_tag=tag,
        n_face_out_xy=(float(n_face[0]), float(n_face[1])),
    )
