#!/usr/bin/env python3
"""Top-down silhouette of each object, rasterised from the FULL-RES source mesh.

The VHACD collision pieces are decimated and leave gaps, so they read as noise.
The source <name>.obj is the true shape (and is what the kMeshNormal sampler
uses), so its triangles projected to XY give an exact silhouette to fit boxes to.

Rows are +y (up), columns are +x (right); '+' marks the link-frame origin.
"""
import os
import sys

import numpy as np
import yaml

REPO = "/root/push_anything_ADMM/.claude/worktrees/fig8-lowcom-single-goal"
RES = float(__import__("os").environ.get("BLOCK_RES", "0.004"))

LOWCOM = ["book", "lotion", "H_shape_texture", "Y_shape_video", "3_shape_video",
          "clamp", "I_shape_texture", "B_shape_video", "G_shape_video",
          "R_shape_texture", "A_shape_video", "C_shape_texture", "baby_toy",
          "E_shape_video"]


def load_obj(path):
    V, F = [], []
    for ln in open(path):
        if ln.startswith("v "):
            V.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("f "):
            idx = [int(p.split("/")[0]) for p in ln.split()[1:]]
            idx = [i - 1 if i > 0 else len(V) + i for i in idx]
            for k in range(1, len(idx) - 1):
                F.append([idx[0], idx[k], idx[k + 1]])
    return np.array(V), np.array(F)


def silhouette(V, F, res=RES, bounds=None):
    if bounds is None:
        xmin, ymin = V[:, 0].min(), V[:, 1].min()
        xmax, ymax = V[:, 0].max(), V[:, 1].max()
    else:
        # Explicit bounds let a caller pad the grid to cover box material that
        # sticks out PAST the mesh AABB. Without this the grid is sized from the
        # mesh alone and any overshoot is silently clipped -- it never appears
        # as spill, which hid a 58 mm overshoot on lotion_block.
        xmin, ymin, xmax, ymax = bounds
    nx = int(np.ceil((xmax - xmin) / res)) + 1
    ny = int(np.ceil((ymax - ymin) / res)) + 1
    occ = np.zeros((ny, nx), dtype=bool)

    for tri in F:
        p = V[tri][:, :2]
        c0 = np.floor((p[:, 0].min() - xmin) / res).astype(int)
        c1 = np.ceil((p[:, 0].max() - xmin) / res).astype(int)
        r0 = np.floor((p[:, 1].min() - ymin) / res).astype(int)
        r1 = np.ceil((p[:, 1].max() - ymin) / res).astype(int)
        c0, r0 = max(c0, 0), max(r0, 0)
        c1, r1 = min(c1, nx - 1), min(r1, ny - 1)
        if c1 < c0 or r1 < r0:
            continue
        cc = xmin + (np.arange(c0, c1 + 1) + 0.5) * res
        rr = ymin + (np.arange(r0, r1 + 1) + 0.5) * res
        GX, GY = np.meshgrid(cc, rr)
        # barycentric sign test
        (x1, y1), (x2, y2), (x3, y3) = p
        d = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if abs(d) < 1e-14:
            continue
        a = ((y2 - y3) * (GX - x3) + (x3 - x2) * (GY - y3)) / d
        b = ((y3 - y1) * (GX - x3) + (x1 - x3) * (GY - y3)) / d
        c = 1.0 - a - b
        inside = (a >= -1e-9) & (b >= -1e-9) & (c >= -1e-9)
        occ[r0:r1 + 1, c0:c1 + 1] |= inside
    return occ, xmin, ymin, xmax, ymax


def main():
    cfg = yaml.safe_load(open(os.path.join(REPO, "config/tasks.yaml")))["tasks"]
    for name in (sys.argv[1:] or LOWCOM):
        t = cfg[name]
        objp = os.path.join(REPO, os.path.dirname(t["object_sdf"]),
                            f"{t['link_name']}.obj")
        V, F = load_obj(objp)
        occ, xmin, ymin, xmax, ymax = silhouette(V, F)
        ny, nx = occ.shape
        ox = int((0 - xmin) / RES)
        oy = int((0 - ymin) / RES)
        print("=" * 78)
        print(f"{name}   x[{xmin:+.4f},{xmax:+.4f}] y[{ymin:+.4f},{ymax:+.4f}] "
              f"z[{V[:,2].min():+.4f},{V[:,2].max():+.4f}]")
        print(f"  size {xmax-xmin:.4f} x {ymax-ymin:.4f} x "
              f"{V[:,2].max()-V[:,2].min():.4f} m   {RES*1000:.0f}mm grid "
              f"fill={occ.mean()*100:.0f}%  tris={len(F)}")
        for r in range(ny - 1, -1, -1):
            print("   " + "".join(
                "#" if occ[r, c] else
                ("+" if (r == oy and c == ox) else
                 ("|" if c == ox else ("-" if r == oy else ".")))
                for c in range(nx)))


if __name__ == "__main__":
    main()
