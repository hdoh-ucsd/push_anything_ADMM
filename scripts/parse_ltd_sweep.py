#!/usr/bin/env python3
"""Parse LTD sweep results.

For each seed log: extract first EE-BOX contact (face classification), full
contact-face distribution, box motion in goal-ward direction, and override
phase breakdown. Prints a per-seed table and an aggregate summary against
the pre-registered bars.
"""
import argparse
import re
import sys
from pathlib import Path

# Goal direction for task-id 4 (west) is -x in world; box motion goal-ward
# = -(final.x - initial.x). For other tasks adjust _goal_dir().
TASK_GOAL_DIR = {1: (0.0,  1.0),   # N
                 2: (1.0,  0.0),   # E
                 3: (0.0, -1.0),   # S
                 4: (-1.0, 0.0)}   # W

RE_CONTACT = re.compile(
    r"\[CONTACT-RUN\] step=(\d+) "
    r"nhat_BA_W=\[([+-]?\d+\.\d+),([+-]?\d+\.\d+),([+-]?\d+\.\d+)\] "
    r"p_BCb=\[([+-]?\d+\.\d+),([+-]?\d+\.\d+),([+-]?\d+\.\d+)\] "
    r"distance=([+-]?\d+\.\d+) contact_type=(\S+)"
)
RE_XBOX = re.compile(
    r"\[X-SEQ-PROBE\] step=(\d+) x0_box=\(([+-]?\d+\.\d+),"
    r"([+-]?\d+\.\d+),([+-]?\d+\.\d+)\)"
)
RE_PHASE = re.compile(r"\[APPROACH-OVERRIDE\].*phase=(\S+)")
RE_GSPERF = re.compile(
    r"\[GS-perf\] avg_per_step_ms=([\d.]+).*full_solves=(\d+)"
)
RE_LTD_FALLBACK = re.compile(r"\[LTD-WORKSPACE-FALLBACK\]")


def classify_face(nx, ny, nz):
    """Return 'side', 'top', 'bottom', or 'corner' based on nhat (onto-box)."""
    az = abs(nz)
    axy = max(abs(nx), abs(ny))
    if az > 0.7 and axy < 0.5:
        return "top" if nz > 0 else "bottom"
    if axy > 0.7 and az < 0.5:
        return "side"
    return "corner"


def parse_seed(log_path):
    """Parse one seed log. Returns dict with seed-level metrics."""
    first_ee_box = None
    contact_face_counts = {"top": 0, "bottom": 0, "side": 0, "corner": 0}
    n_ee_box = 0
    n_none = 0
    x0_first = None
    x0_last = None
    phase_counts = {}
    perf = None
    workspace_fallbacks = 0

    with open(log_path) as f:
        for line in f:
            m = RE_CONTACT.match(line)
            if m:
                ctype = m.group(9)
                if ctype == "EE-BOX":
                    n_ee_box += 1
                    nx, ny, nz = (float(m.group(2)),
                                  float(m.group(3)),
                                  float(m.group(4)))
                    face = classify_face(nx, ny, nz)
                    contact_face_counts[face] += 1
                    if first_ee_box is None:
                        first_ee_box = {
                            "step": int(m.group(1)),
                            "nhat": (nx, ny, nz),
                            "p_BCb": (float(m.group(5)),
                                      float(m.group(6)),
                                      float(m.group(7))),
                            "distance": float(m.group(8)),
                            "face": face,
                        }
                elif ctype == "NONE":
                    n_none += 1
                continue
            m = RE_XBOX.match(line)
            if m:
                xyz = (float(m.group(2)),
                       float(m.group(3)),
                       float(m.group(4)))
                if x0_first is None:
                    x0_first = xyz
                x0_last = xyz
                continue
            m = RE_PHASE.search(line)
            if m:
                ph = m.group(1)
                phase_counts[ph] = phase_counts.get(ph, 0) + 1
                continue
            m = RE_GSPERF.search(line)
            if m:
                perf = {"ms_per_step": float(m.group(1)),
                        "full_solves": int(m.group(2))}
                continue
            if RE_LTD_FALLBACK.search(line):
                workspace_fallbacks += 1

    return {
        "log": str(log_path),
        "first_ee_box": first_ee_box,
        "contact_faces": contact_face_counts,
        "n_ee_box": n_ee_box,
        "n_none": n_none,
        "x0_first": x0_first,
        "x0_last": x0_last,
        "phase_counts": phase_counts,
        "perf": perf,
        "workspace_fallbacks": workspace_fallbacks,
    }


def goal_progress(x0_first, x0_last, task_id):
    """Project box translation onto goal direction. Positive = goal-ward."""
    if x0_first is None or x0_last is None:
        return None
    dx = x0_last[0] - x0_first[0]
    dy = x0_last[1] - x0_first[1]
    gx, gy = TASK_GOAL_DIR[task_id]
    return dx * gx + dy * gy


def motion_bin(progress_m):
    """Pre-registered motion bins. Returns label or None."""
    if progress_m is None:
        return None
    pm = progress_m * 1000  # mm
    if pm >= 50:
        return ">50mm AMP_WAS_WALL"
    if pm >= 5:
        return "5-50mm PARTIAL"
    return "<5mm OTHER_DOMINATES"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep_dir", help="Directory with seed*.log files")
    ap.add_argument("--task-id", type=int, default=4)
    args = ap.parse_args()

    sweep = Path(args.sweep_dir)
    seeds = sorted(sweep.glob("seed*.log"),
                   key=lambda p: int(re.search(r"seed(\d+)", p.name).group(1)))
    if not seeds:
        print(f"no seed*.log in {sweep}", file=sys.stderr)
        sys.exit(1)

    print(f"# LTD sweep parse — {sweep} (task-id={args.task_id})\n")
    print(f"{'seed':>4} {'firstEE':>8} {'face':>7} {'nhat':>22} "
          f"{'EE-BOX':>7} {'NONE':>5} {'motion_mm':>10} {'bin':>20} "
          f"{'ms/step':>8} {'phaseA':>7} {'phaseB':>7} {'phaseC':>7} "
          f"{'wsFB':>5}")

    contact_face_first_dist = {"side": 0, "top": 0, "bottom": 0,
                               "corner": 0, "NO_CONTACT": 0}
    motion_bin_dist = {">50mm AMP_WAS_WALL": 0, "5-50mm PARTIAL": 0,
                       "<5mm OTHER_DOMINATES": 0, "NO_MOTION_DATA": 0}
    n_seeds = 0
    n_with_contact = 0

    for p in seeds:
        seed_id = int(re.search(r"seed(\d+)", p.name).group(1))
        s = parse_seed(p)
        n_seeds += 1
        if s["first_ee_box"]:
            n_with_contact += 1
            face = s["first_ee_box"]["face"]
            contact_face_first_dist[face] += 1
            firstEE = s["first_ee_box"]["step"]
            nhat = s["first_ee_box"]["nhat"]
            nhat_s = f"[{nhat[0]:+.2f},{nhat[1]:+.2f},{nhat[2]:+.2f}]"
        else:
            contact_face_first_dist["NO_CONTACT"] += 1
            face = "-"
            firstEE = "-"
            nhat_s = "-"

        prog = goal_progress(s["x0_first"], s["x0_last"], args.task_id)
        if prog is None:
            mbin = "NO_MOTION_DATA"
            prog_mm = "-"
        else:
            mbin = motion_bin(prog)
            prog_mm = f"{prog*1000:+.1f}"
        motion_bin_dist[mbin] = motion_bin_dist.get(mbin, 0) + 1

        ms = (f"{s['perf']['ms_per_step']:.0f}" if s['perf'] else "-")
        pcs = s["phase_counts"]
        pa = pcs.get("A_lift_trav", 0)
        pb = pcs.get("B_descend", 0)
        pc = pcs.get("C_approach", 0)
        ws = s["workspace_fallbacks"]
        print(f"{seed_id:>4} {str(firstEE):>8} {face:>7} {nhat_s:>22} "
              f"{s['n_ee_box']:>7} {s['n_none']:>5} {prog_mm:>10} "
              f"{mbin:>20} {ms:>8} {pa:>7} {pb:>7} {pc:>7} {ws:>5}")

    print()
    print(f"## Aggregate ({n_seeds} seeds)")
    print()
    print("### Contact face on FIRST EE-BOX event (primary bar):")
    for k in ["side", "top", "bottom", "corner", "NO_CONTACT"]:
        pct = 100.0 * contact_face_first_dist[k] / n_seeds if n_seeds else 0
        print(f"  {k:>12}: {contact_face_first_dist[k]:>3} / {n_seeds} "
              f"({pct:.0f}%)")
    side_frac = (contact_face_first_dist["side"] / n_with_contact
                 if n_with_contact else 0)
    print(f"\n  Pre-registered bar: ≥70% of contacting seeds show side face")
    print(f"  Actual: {contact_face_first_dist['side']}/{n_with_contact} "
          f"({100*side_frac:.0f}%) of contacting seeds had side-face first")
    print()
    print("### Box motion bins (secondary bar):")
    for k, v in motion_bin_dist.items():
        pct = 100.0 * v / n_seeds if n_seeds else 0
        print(f"  {k:>22}: {v:>3} / {n_seeds} ({pct:.0f}%)")


if __name__ == "__main__":
    main()
