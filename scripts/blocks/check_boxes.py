#!/usr/bin/env python3
"""Score a hand-authored box decomposition against the object's true silhouette.

Usage:  python3 check_boxes.py spec.json [name ...]

spec.json:  {"<task_name>": {"boxes": [{"c":[x,y,z], "s":[sx,sy,sz], "yaw":0.0}, ...]}, ...}
All values in the object's LINK frame, metres / radians.

Prints, per object:
  IoU        intersection-over-union of the box footprint vs the mesh footprint
  missed     % of the mesh footprint NOT covered by any box  (under-coverage)
  spill      % of the box footprint outside the mesh         (over-coverage)
  z check    box z span vs mesh z span
plus an ASCII overlay:  # both   o box-only(spill)   x mesh-only(missed)
"""
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from occupancy import load_obj, silhouette, RES, REPO  # noqa: E402


def box_footprint(boxes, xmin, ymin, shape):
    ny, nx = shape
    cx = xmin + (np.arange(nx) + 0.5) * RES
    cy = ymin + (np.arange(ny) + 0.5) * RES
    GX, GY = np.meshgrid(cx, cy)
    occ = np.zeros(shape, dtype=bool)
    for b in boxes:
        c = b["c"]
        s = b["s"]
        yaw = float(b.get("yaw", 0.0))
        dx = GX - c[0]
        dy = GY - c[1]
        ca, sa = np.cos(-yaw), np.sin(-yaw)
        lx = ca * dx - sa * dy
        ly = sa * dx + ca * dy
        occ |= (np.abs(lx) <= s[0] / 2) & (np.abs(ly) <= s[1] / 2)
    return occ


def main():
    spec = json.load(open(sys.argv[1]))
    cfg = yaml.safe_load(open(os.path.join(REPO, "config/tasks.yaml")))["tasks"]
    names = sys.argv[2:] or list(spec.keys())
    worst = 1.0
    for name in names:
        if name not in spec:
            print(f"{name}: NOT IN SPEC")
            worst = 0.0
            continue
        t = cfg[name]
        objp = os.path.join(REPO, os.path.dirname(t["object_sdf"]),
                            f"{t['link_name']}.obj")
        V, F = load_obj(objp)
        mesh, xmin, ymin, xmax, ymax = silhouette(V, F)
        boxes = spec[name]["boxes"]
        bocc = box_footprint(boxes, xmin, ymin, mesh.shape)

        inter = (mesh & bocc).sum()
        union = (mesh | bocc).sum()
        iou = inter / union if union else 0.0
        missed = (mesh & ~bocc).sum() / max(mesh.sum(), 1) * 100
        spill = (bocc & ~mesh).sum() / max(bocc.sum(), 1) * 100
        worst = min(worst, iou)

        bz0 = min(b["c"][2] - b["s"][2] / 2 for b in boxes)
        bz1 = max(b["c"][2] + b["s"][2] / 2 for b in boxes)
        mz0, mz1 = V[:, 2].min(), V[:, 2].max()

        print("=" * 78)
        print(f"{name}: {len(boxes)} box(es)  IoU={iou:.3f}  "
              f"missed={missed:.1f}%  spill={spill:.1f}%")
        print(f"   z: boxes [{bz0:+.4f},{bz1:+.4f}]  mesh [{mz0:+.4f},{mz1:+.4f}]"
              f"   dz_err=({bz0-mz0:+.4f},{bz1-mz1:+.4f})")
        ny, nx = mesh.shape
        for r in range(ny - 1, -1, -1):
            row = ""
            for c in range(nx):
                m, b = mesh[r, c], bocc[r, c]
                row += "#" if (m and b) else ("o" if b else ("x" if m else "."))
            print("   " + row)
    print(f"\nWORST IoU = {worst:.3f}")


main()
