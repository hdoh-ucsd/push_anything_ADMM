#!/usr/bin/env python3
"""Compare a reference-stack run against a port run at the same goal.

Reference side: parses `x_lcs_curr:` verbose lines (one per plan loop) from
c3controller.log[.gz] — obj quat in slots 3-6, obj pos in slots 7-9. The
reference log has no absolute clock, so loop index is mapped linearly onto
[0, duration] (sim runs at realtime_rate=1; loop rate is quasi-constant).

Port side: parses `[STEP]` lines (sim t + obj xy) and `[GATE-CONTACT]`
lines (box quaternion, joined on step, forward-filled).

Usage:
  compare_port_vs_reference.py REF_RUN_DIR REF_DURATION_S PORT_LOG [--plot OUT.png]
"""
import gzip
import math
import sys
from pathlib import Path

GOAL_XY = (0.481992, 0.187454)
GOAL_QUAT = (-0.932773, 0.0, 0.0, 0.360464)  # yaw -0.7379 rad


def quat_angle(qa, qb):
    dot = abs(sum(a * b for a, b in zip(qa, qb)))
    return 2.0 * math.acos(min(1.0, dot))


def errors(pos_xy, quat):
    perr = math.hypot(pos_xy[0] - GOAL_XY[0], pos_xy[1] - GOAL_XY[1])
    rerr = quat_angle(quat, GOAL_QUAT)
    return perr, rerr


def load_reference(run_dir, duration_s):
    """→ list of (t, pos_err, rot_err) from per-loop x_lcs_curr lines."""
    path = Path(run_dir) / "c3controller.log.gz"
    if not path.exists():
        path = Path(run_dir) / "c3controller.log"
    opener = gzip.open if path.suffix == ".gz" else open
    raw = []
    with opener(path, "rt", errors="replace") as f:
        for line in f:
            if "x_lcs_curr:" in line:
                try:
                    vals = [float(v) for v in
                            line.split("x_lcs_curr:", 1)[1].split()]
                except ValueError:
                    continue
                if len(vals) >= 10:
                    raw.append((vals[3:7], vals[7:10]))
    n = len(raw)
    out = []
    for i, (quat, pos) in enumerate(raw):
        t = duration_s * i / max(1, n - 1)
        perr, rerr = errors(pos[:2], quat)
        out.append((t, perr, rerr))
    return out


def load_port(log_path):
    """→ list of (t, pos_err, rot_err) from [STEP] + [GATE-CONTACT] lines."""
    step_t_obj = {}   # step -> (t, obj_xy)
    step_quat = {}    # step -> quat
    with open(log_path, errors="replace") as f:
        for line in f:
            if line.startswith("[STEP]"):
                try:
                    toks = dict(
                        kv.split("=", 1) for kv in line.split() if "=" in kv)
                    step = int(toks["step"])
                    t = float(toks["t"].rstrip("s"))
                    ox, oy = [
                        float(v) for v in
                        toks["obj"].strip("()").split(",")[:2]]
                    step_t_obj[step] = (t, (ox, oy))
                except (KeyError, ValueError):
                    continue
            elif line.startswith("[GATE-CONTACT]"):
                try:
                    step = int(line.split("step=", 1)[1].split()[0])
                    q = [float(v) for v in
                         line.split("box_q=(", 1)[1].split(")", 1)[0]
                         .split(",")]
                    if len(q) == 4:
                        step_quat[step] = q
                except (ValueError, IndexError):
                    continue
    out = []
    last_q = (1.0, 0.0, 0.0, 0.0)
    for step in sorted(step_t_obj):
        t, obj_xy = step_t_obj[step]
        last_q = step_quat.get(step, last_q)
        perr, rerr = errors(obj_xy, last_q)
        out.append((t, perr, rerr))
    return out


def milestones(traj, label):
    print(f"--- {label}: {len(traj)} samples, "
          f"t=[{traj[0][0]:.0f}, {traj[-1][0]:.0f}]s")
    t0, p0, r0 = traj[0]
    tN, pN, rN = traj[-1]
    print(f"    pos_err {p0:.4f} -> {pN:.4f} m   "
          f"rot_err {r0:.4f} -> {rN:.4f} rad")
    best_p = min(tr[1] for tr in traj)
    best_r = min(tr[2] for tr in traj)
    print(f"    best pos_err={best_p:.4f} m  best rot_err={best_r:.4f} rad")
    for thr in (0.15, 0.10, 0.05, 0.02):
        hit = next((tr[0] for tr in traj if tr[1] < thr), None)
        print(f"    pos_err<{thr:.2f} m: "
              f"{'t=%.0fs' % hit if hit is not None else 'never'}")
    for thr in (0.5, 0.3, 0.1):
        hit = next((tr[0] for tr in traj if tr[2] < thr), None)
        print(f"    rot_err<{thr:.1f} rad: "
              f"{'t=%.0fs' % hit if hit is not None else 'never'}")
    tight = next(
        (tr[0] for tr in traj if tr[1] < 0.02 and tr[2] < 0.1), None)
    print(f"    tight (<0.02m & <0.1rad): "
          f"{'t=%.0fs' % tight if tight is not None else 'never'}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ref_dir, dur, port_log = args[0], float(args[1]), args[2]
    plot_out = None
    if "--plot" in sys.argv:
        plot_out = sys.argv[sys.argv.index("--plot") + 1]

    ref = load_reference(ref_dir, dur)
    port = load_port(port_log)
    print(f"goal: xy=({GOAL_XY[0]:.4f}, {GOAL_XY[1]:.4f}) "
          f"quat=({GOAL_QUAT[0]:.4f}, 0, 0, {GOAL_QUAT[3]:.4f})")
    milestones(ref, f"REFERENCE ({ref_dir})")
    milestones(port, f"PORT ({port_log})")

    if plot_out:
        from plot_compare import render  # noqa: local sibling module
        render(ref, port, plot_out)
        print(f"[compare] plot written to {plot_out}")


if __name__ == "__main__":
    main()
