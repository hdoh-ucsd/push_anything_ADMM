#!/usr/bin/env python3
"""Convergence summary for a push_jack run.

The jack is the port's only SE(3) task, so the usual planar readouts do not
apply: its rotation error is a GEODESIC angle to a full goal quaternion, and
its object z legitimately rises while it rolls between tripods. This reports
the quantities that actually characterise the task.

Usage: python3 scripts/analyze_jack_run.py results/<run>.txt
"""
import re
import statistics as st
import sys

GOAL_XY = (0.450, 0.200)
REST_Z = 0.061084          # h/sqrt(3) + r, the tripod rest height
TIGHT_POS, TIGHT_ROT = 0.02, 0.10     # reference goal_params thresholds
LOOSE_POS, LOOSE_ROT = 0.05, 0.40


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    steps = []
    modes = {}
    result = None
    for line in open(path, errors="ignore"):
        if line.startswith("[STEP]"):
            t = re.search(r"t=([\d.]+)s", line)
            o = re.search(r"obj=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\)", line)
            m = re.search(r"mode=(\w+)", line)
            if t and o:
                steps.append((float(t.group(1)),
                              float(o.group(1)), float(o.group(2)), float(o.group(3))))
            if m:
                modes[m.group(1)] = modes.get(m.group(1), 0) + 1
        elif line.startswith("[STAGE-A-TRACE]"):
            pass
        elif line.startswith("[RESULT]"):
            result = line.strip()
    if not steps:
        print(f"{path}: no [STEP] rows found.")
        return 1

    t0, t1 = steps[0][0], steps[-1][0]
    print(f"{path}")
    print(f"  sim duration       : {t0:.1f} -> {t1:.1f} s   ({len(steps)} planner steps)")
    tot = sum(modes.values()) or 1
    print(f"  mode split         : " +
          "  ".join(f"{k}={v} ({100*v/tot:.0f}%)" for k, v in sorted(modes.items())))

    def dist(s):
        return ((s[1] - GOAL_XY[0]) ** 2 + (s[2] - GOAL_XY[1]) ** 2) ** 0.5
    d0, dN = dist(steps[0]), dist(steps[-1])
    dmin = min(dist(s) for s in steps)
    print(f"\n  translation to goal xy")
    print(f"    start {d0:.4f} m -> end {dN:.4f} m   (best {dmin:.4f} m)")
    print(f"    net progress     : {d0 - dN:+.4f} m")

    zs = [s[3] for s in steps]
    print(f"\n  object z (rest height {REST_Z:.4f})")
    print(f"    min {min(zs):+.4f}  max {max(zs):+.4f}  mean {st.mean(zs):+.4f}")
    rolled = sum(1 for z in zs if z > REST_Z + 0.005)
    print(f"    steps with z > rest+5mm (mid-roll / lifted): {rolled} "
          f"({100*rolled/len(zs):.0f}%)")
    if min(zs) < 0.0:
        print(f"    !! object went BELOW ground (min {min(zs):+.4f}) — fell off the table")

    if result:
        print(f"\n  {result}")
        rot = re.search(r"rotational_error=([\d.]+)rad", result)
        pos = re.search(r"translational_error=([\d.]+)m", result)
        if rot and pos:
            r, p = float(rot.group(1)), float(pos.group(1))
            print(f"    initial reorientation demand was 1.6926 rad (97.0 deg)")
            print(f"    rot now {r:.4f} rad ({r*57.2958:.1f} deg) -> "
                  f"{'TOWARD goal' if r < 1.6926 else 'AWAY from goal'}")
            print(f"    tight gate (<{TIGHT_POS} m, <{TIGHT_ROT} rad): "
                  f"{'PASS' if (p < TIGHT_POS and r < TIGHT_ROT) else 'FAIL'}")
            print(f"    loose gate (<{LOOSE_POS} m, <{LOOSE_ROT} rad): "
                  f"{'PASS' if (p < LOOSE_POS and r < LOOSE_ROT) else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
