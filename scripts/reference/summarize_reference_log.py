#!/usr/bin/env python3
"""Summarize a reference-stack run directory produced by run_reference.sh.

Reads c3controller.log for [TPOSE] (c3-mode cost + object pose) and [WSCHK]
(one triplet per plan loop) lines, and scans all three logs for crash
markers. Python line-walk, not grep (long-line logs defeat grep here).

Usage: summarize_reference_log.py RUN_DIR
"""
import math
import sys
from pathlib import Path

CRASH_MARKERS = (
    "DRAKE_DEMAND",
    "Failure at",
    "abort (core dumped)",
    "terminate called",
    "what():",
    "Segmentation fault",
)


def parse_tpose(line):
    # [TPOSE] cost=C obj_pos=x y z obj_quat=w x y z
    try:
        head, rest = line.split("obj_pos=", 1)
        cost = float(head.split("cost=", 1)[1])
        pos_str, quat_str = rest.split("obj_quat=", 1)
        pos = [float(v) for v in pos_str.split()]
        quat = [float(v) for v in quat_str.split()]
        if len(pos) == 3 and len(quat) == 4:
            return cost, pos, quat
    except (ValueError, IndexError):
        pass
    return None


def quat_angle(qa, qb):
    dot = abs(sum(a * b for a, b in zip(qa, qb)))
    return 2.0 * math.acos(min(1.0, dot))


def main():
    run_dir = Path(sys.argv[1])
    tposes = []
    n_wschk = 0
    crashes = []

    for name in ("sim.log", "osc.log", "c3controller.log"):
        path = run_dir / name
        if not path.exists():
            print(f"[summary] {name}: MISSING")
            continue
        with open(path, errors="replace") as f:
            for line in f:
                if name == "c3controller.log":
                    if "[WSCHK]" in line:
                        n_wschk += 1
                    elif "[TPOSE]" in line:
                        parsed = parse_tpose(line)
                        if parsed:
                            tposes.append(parsed)
                for marker in CRASH_MARKERS:
                    if marker in line:
                        crashes.append(f"{name}: {line.strip()[:200]}")
                        break

    print(f"[summary] run_dir={run_dir}")
    print(f"[summary] plan_loops={n_wschk // 3} (WSCHK triplets)  "
          f"c3_mode_loops={len(tposes)} (TPOSE lines)")

    if tposes:
        costs = [t[0] for t in tposes]
        c0, p0, q0 = tposes[0]
        c1, p1, q1 = tposes[-1]
        disp_xy = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        rot = quat_angle(q0, q1)
        print(f"[summary] cost first={c0:.2f} last={c1:.2f} min={min(costs):.2f}")
        print(f"[summary] obj first=({p0[0]:.4f}, {p0[1]:.4f}, {p0[2]:.4f}) "
              f"last=({p1[0]:.4f}, {p1[1]:.4f}, {p1[2]:.4f})")
        print(f"[summary] obj xy displacement={disp_xy:.4f} m  "
              f"net rotation={rot:.4f} rad")
    else:
        print("[summary] no TPOSE lines — controller never entered c3 mode")

    if crashes:
        print(f"[summary] CRASH MARKERS ({len(crashes)}):")
        for c in crashes[:10]:
            print(f"[summary]   {c}")
    else:
        print("[summary] no crash markers in any log")


if __name__ == "__main__":
    main()
