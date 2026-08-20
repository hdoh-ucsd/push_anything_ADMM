"""Convert a reference jacktoy c3controller capture into a port-format log
that render_log_drake_scene.py + paint_log_sidepanel.py can consume.

The capture (results/reference/jacktoy_eezoff1) carries:
  [TRAJ] t=<lcm-ts> wall=<steady-clock s> obj_pos=<x y z> obj_quat=<w x y z>
      -- emitted per CalcCost call, C3 loops only (~3/loop, identical pose);
      wall spans the full run. The lcm t= field repeats/jumps: unusable.
  "Switching to C3 because reached repositioning target" /
  "Repositioning after not making progress in C3" /
  "Repositioning because found good sample" -- mode markers, line-ordered,
      no timestamps (assigned the wall of the last TRAJ line seen).
  NO EE/arm stream (the capture never printed x_lcs EE slots) -- the arm is
      rendered PARKED at the port spawn EE and the panel banner says so.
  NO object stream during repos gaps (up to 10 s) -- pose holds last value.

Frame shift: port z = reference z + 0.029 (ground-top offset). Quats wxyz
in both. Goal = the patched single fixed target (0.45, 0.2, AllUp quat).

Synthesized for the panel: [FLIP] lines via the port's tripod_id with the
port's 3-tick persistence; a latching full-gate (pos<0.02 & rot<0.1)
"[GOAL-GEN] goal #1 REACHED (full gate)" milestone (worded so it does NOT
match RE_REGOAL -- one ghost segment, no env rebuild); [RUN-META] with
solver=c3plus so the panel labels mode C3+.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from control.sampling_c3.goal_generator import (  # noqa: E402
    TRIPOD_NAMES, geodesic_angle, tripod_id)

Z_OFFSET = 0.029
GOAL_XY = (0.45, 0.2)
GOAL_QUAT = np.array([0.8804762392171493, 0.27984814233312133,
                      -0.3647051996310009, -0.11591689595929514])
# Parked EE: mirror of the port jack spawn EE on the base's -y side —
# IK-reachable with the vertical tool (the +y spawn is the proven point)
# and out of the standard jack camera's foreground (the +y original
# loomed over the scene; (0.25,-0.40,0.30) was unreachable and fell back
# to per-frame best-effort home poses).
PARKED_EE = (0.285, -0.530, 0.172)
TICK = 0.1
FLIP_PERSIST = 3                    # ticks, mirrors the port's [FLIP] logger

RE_TRAJ = re.compile(
    r"^\[TRAJ\] t=[\d.eE+-]+ wall=([\d.eE+-]+) obj_pos=\s*([-\d.eE]+)\s+"
    r"([-\d.eE]+)\s+([-\d.eE]+) obj_quat=\s*([-\d.eE]+)\s+([-\d.eE]+)\s+"
    r"([-\d.eE]+)\s+([-\d.eE]+)")
MARKERS = [
    ("Switching to C3 because reached repositioning target",
     "c3", "kToC3ReachedReposTarget"),
    ("Repositioning after not making progress in C3",
     "free", "kToReposUnproductive"),
    ("Repositioning because found good sample", "free", "kToReposCost"),
    ("All objects on target, switching to repositioning",
     "free", "kToReposUnproductive"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_log", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stem", default="REF_jacktoy_eezoff1")
    args = ap.parse_args()

    poses = []       # (wall, p(3) port frame, q(4) normalized)
    switches = []    # (wall, mode, reason)
    last_wall = 0.0
    with open(args.ref_log, errors="replace") as f:
        for line in f:
            m = RE_TRAJ.match(line)
            if m:
                last_wall = float(m.group(1))
                p = np.array([float(m.group(2)), float(m.group(3)),
                              float(m.group(4)) + Z_OFFSET])
                q = np.array([float(m.group(i)) for i in (5, 6, 7, 8)])
                n = np.linalg.norm(q)
                if n > 1e-9:
                    poses.append((last_wall, p, q / n))
                continue
            for text, mode, reason in MARKERS:
                if text in line:
                    switches.append((last_wall, mode, reason))
                    break
    if not poses:
        sys.exit("no [TRAJ] poses found")
    t_end = poses[-1][0]
    print(f"[convert-ref] {len(poses)} poses, {len(switches)} switches, "
          f"wall span {t_end:.1f}s")

    lines = [
        f"[RUN-META] git=ref-257e3ed seed=- task=push_jack "
        f"stem={args.stem} flags=[solver=c3plus ee_space=True admm_iter=- "
        f"max_time={t_end:.1f}]",
        f"[ENV]  Goal coords: [{GOAL_XY[0]}, {GOAL_XY[1]}]",
        "[GOAL-QUAT] goal_quat=[{:.10f} {:.10f} {:.10f} {:.10f}]".format(
            *GOAL_QUAT),
        "[TASK] REFERENCE CAPTURE jacktoy_eezoff1 @ push_anything_dev "
        "257e3ed (C3+, entry ceiling off) — rendered from [TRAJ] telemetry",
        "[TASK] capture limits: object pose logged on C3+ loops only "
        "(repos gaps hold last pose); ARM NOT RECORDED — shown parked",
    ]

    # Full-gate latch on the RAW pose stream (the tick sampling below can
    # step over a brief joint-gate window — measured: the eezoff1 capture
    # meets pos<0.02 & rot<0.1 momentarily around wall 170.0 s).
    reach_wall = None
    for w, p, q in poses:
        gd = float(np.hypot(p[0] - GOAL_XY[0], p[1] - GOAL_XY[1]))
        if gd < 0.02 and geodesic_angle(q, GOAL_QUAT) < 0.1:
            reach_wall = w
            reach_vals = (gd, geodesic_angle(q, GOAL_QUAT))
            break

    # uniform ticks, hold-last pose, interval mode
    n_ticks = int(t_end / TICK) + 1
    pi = 0
    si = 0
    mode, reason = "c3", "kStayInC3"   # run opens in C3 mode (TRAJ from 0s)
    prev_tripod = None
    tripod_hold = None
    hold_n = 0
    flip_no = 0
    reached = False
    p = poses[0][1]
    q = poses[0][2]
    for k in range(1, n_ticks + 1):
        t = (k - 1) * TICK
        while pi < len(poses) and poses[pi][0] <= t:
            p, q = poses[pi][1], poses[pi][2]
            pi += 1
        tick_reason = None
        while si < len(switches) and switches[si][0] <= t:
            _, mode, tick_reason = switches[si]
            si += 1
        if tick_reason is None:
            tick_reason = "kStayInC3" if mode == "c3" else "kStayInRepos"
        reason = tick_reason

        gd = float(np.hypot(p[0] - GOAL_XY[0], p[1] - GOAL_XY[1]))
        rot = geodesic_angle(q, GOAL_QUAT)

        # [FLIP] synthesis with persistence (port semantics)
        trip = tripod_id(q)
        if prev_tripod is None:
            prev_tripod = trip
        if trip != prev_tripod:
            if tripod_hold == trip:
                hold_n += 1
            else:
                tripod_hold, hold_n = trip, 1
            if hold_n >= FLIP_PERSIST:
                flip_no += 1
                lines.append(
                    f"[FLIP] #{flip_no} at t={t:.3f}s "
                    f"{TRIPOD_NAMES.get(prev_tripod, prev_tripod)} -> "
                    f"{TRIPOD_NAMES.get(trip, trip)}")
                prev_tripod = trip
                tripod_hold, hold_n = None, 0
        else:
            tripod_hold, hold_n = None, 0

        if not reached and reach_wall is not None and t >= reach_wall:
            reached = True
            lines.append(
                f"[GOAL-GEN] goal #1 REACHED (full gate) at t={t:.3f}s "
                f"pos={reach_vals[0]:.4f}m rot={reach_vals[1]:.4f}rad — "
                f"reference redraws to the same fixed target "
                f"(ghost unchanged)")

        lines.append(
            f"[STEP] step={k} mode={mode} t={t:.3f}s "
            f"ee=({PARKED_EE[0]:+.3f},{PARKED_EE[1]:+.3f},{PARKED_EE[2]:+.3f}) "
            f"obj=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}) "
            f"goal_dist={min(gd, 0.15):.3f}m switch={reason} rot_err={rot:.4f}")
        lines.append(
            f"[GATE-CONTACT] step={k} F_W=(+0.0000,+0.0000,+0.0000) "
            f"F_on_box=(+0.0000,+0.0000,+0.0000) "
            f"n_face_out=(+0.0000,+0.0000,+1.0000) A_is_ee=0 "
            f"box_q=({q[0]:+.5f},{q[1]:+.5f},{q[2]:+.5f},{q[3]:+.5f}) "
            f"box_p=({p[0]:+.5f},{p[1]:+.5f},{p[2]:+.5f}) "
            f"ee_p=({PARKED_EE[0]:+.5f},{PARKED_EE[1]:+.5f},"
            f"{PARKED_EE[2]:+.5f})")

    args.out.write_text("\n".join(lines) + "\n")
    print(f"[convert-ref] wrote {args.out} ({n_ticks} ticks, "
          f"{flip_no} flips, reached={reached})")


if __name__ == "__main__":
    main()
