#!/usr/bin/env python3
"""Full spawn/goal overlap matrix for the T+H 2-object layout."""
import numpy as np

rT, rH = 0.1442, 0.1000     # yaw-invariant max radius from link origin
need = rT + rH
pts = {"T spawn": (0.40, -0.30), "H spawn": (0.42, -0.10),
       "T goal": (0.50, -0.10), "H goal": (0.50, 0.10)}

print(f"T R_xy = {rT:.4f}   H R_xy = {rH:.4f}   clearance needed = {need:.4f} m\n")
for a in ("T spawn", "T goal"):
    for b in ("H spawn", "H goal"):
        d = float(np.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1]))
        print(f"  {a:8s} <-> {b:8s} = {d:.4f} m   "
              f"{'OVERLAP by %.4f' % (need - d) if d < need else 'clear'}")
print()
d = float(np.hypot(pts['T spawn'][0] - pts['H spawn'][0],
                   pts['T spawn'][1] - pts['H spawn'][1]))
print(f"  T spawn  <-> H spawn  = {d:.4f} m   "
      f"{'OVERLAP' if d < need else 'clear'}")
