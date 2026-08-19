#!/usr/bin/env python3
"""Regression check: EE z must not sag below the c3 frozen height.

Root cause this guards (2026-08-10): with use_velocity_feedforward=True the
OSC receives the planner's EE velocity STATE slot as ydot_des, whose z carries
the planner's intended descent -- while c3 mode freezes the position track's z
to z_height. The two targets contradict, and the OSC settles at

    sag = (Kd_cart / Kp_cart) * |v_des_z|

below the frozen height. On push_jack that was 3.21 mm against a 5.4 mm margin
to the workspace floor, tripping CheckForWorkspaceLimitViolations at t=8.2 s.

Usage:  python3 scripts/check_vff_z_consistency.py results/<run>.txt [max_mm]
Exit 0 if the mean settled sag is under the threshold (default 1.0 mm).
"""
import re
import statistics as st
import sys


def analyse(path):
    sags, vz = [], []
    for line in open(path, errors="ignore"):
        if line.startswith("[C3-TRAJ]"):
            a = re.search(r"ee_now_z=([-+\d.]+)", line)
            b = re.search(r"z_frozen_to=([-+\d.]+)", line)
            if a and b:
                s = float(b.group(1)) - float(a.group(1))
                # Exclude c3-entry transients: the arm is still descending from
                # a reposition and is legitimately far above the frozen height.
                if abs(s) < 0.020:
                    sags.append(s)
        elif line.startswith("[VFF]") and "mode=c3" in line:
            m = re.search(r"v_des=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\)", line)
            if m:
                vz.append(float(m.group(3)))
    return sags, vz


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    thresh_mm = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    sags, vz = analyse(path)
    if not sags:
        print(f"{path}: no settled [C3-TRAJ] rows — nothing to check.")
        return 0
    mean_mm = st.mean(sags) * 1000.0
    max_mm = max(sags) * 1000.0
    vz_mean = st.mean(vz) if vz else 0.0
    print(f"{path}")
    print(f"  settled c3 steps    : {len(sags)}")
    print(f"  mean sag below freeze: {mean_mm:+.2f} mm   (threshold {thresh_mm:.2f})")
    print(f"  max  sag below freeze: {max_mm:+.2f} mm")
    print(f"  mean v_des_z         : {vz_mean:+.5f} m/s")
    if mean_mm > thresh_mm:
        print(f"  FAIL: EE sits {mean_mm:.2f} mm below the frozen c3 height. "
              f"Expected ~(Kd/Kp)*|v_des_z| = {0.1 * abs(vz_mean) * 1000:.2f} mm "
              f"-- check use_velocity_feedforward.")
        return 1
    print("  OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
