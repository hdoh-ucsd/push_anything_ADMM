#!/usr/bin/env python3
"""Do the generated block OBJs have OUTWARD face normals?

kMeshNormal offsets a face point by +buffer_distance along the face normal and
rejects the sample if the signed distance to the object is <= clearance. If the
normals point INWARD the offset moves the point INTO the body, every draw is
rejected, and the sampler returns only the EE's current position forever.
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/root/push_anything_ADMM/.claude/worktrees/fig8-lowcom-single-goal")
from control.sampling_c3.sampling import load_mesh_faces  # noqa: E402

REPO = "/root/push_anything_ADMM/.claude/worktrees/fig8-lowcom-single-goal"
CASES = [
    ("BLOCK  push_t_white_block", "sim/models/push_t_white_block/push_t_white.obj"),
    ("BLOCK  H_shape_texture_block", "sim/models/H_shape_texture_block/H_shape_texture.obj"),
    ("BLOCK  book_block", "sim/models/book_block/book.obj"),
    ("MESH   push_t_white", "sim/models/push_t_white/push_t_white.obj"),
    ("MESH   H_shape_texture", "sim/models/H_shape_texture/H_shape_texture.obj"),
]

print(f"{'asset':32s} {'faces':>6s} {'outward':>8s} {'inward':>7s}  verdict")
for label, rel in CASES:
    mf = load_mesh_faces(os.path.join(REPO, rel), 0.035, 0.027)
    tri, nrm = mf["tri"], mf["normals"]
    centroid = tri.reshape(-1, 3).mean(axis=0)
    fc = tri.mean(axis=1)                       # face centres
    radial = fc - centroid                      # points away from body centre
    radial[:, 2] = 0.0
    n = nrm.copy()
    n[:, 2] = 0.0
    dot = np.einsum("ij,ij->i", n, radial)
    out_n = int((dot > 0).sum())
    in_n = int((dot < 0).sum())
    verdict = ("OUTWARD ok" if in_n == 0 else
               ("INWARD - sampler will reject every draw" if out_n == 0
                else "MIXED"))
    print(f"{label:32s} {len(nrm):6d} {out_n:8d} {in_n:7d}  {verdict}")
