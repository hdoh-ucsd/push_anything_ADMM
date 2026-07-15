"""Extract SC metrics from a sampling_c3 run.log for the Q2c admit-guard
EE_z gate.

SCs:
  - SC-collision-gone: no [DRAKE-CONTACT] ee_box_normal>0 while mode=free
                       AND switch in {kStayInRepos,kToBetterRepos}
                       AND ee_z > EE_Z_GATE.
  - SC-legit-approach-works: at least one sustained mode=c3 interval >=20 ticks
                             with ee_box_normal>0 AND ee_z<EE_Z_GATE.
  - SC-noregress-working: goal_dist@8s, first_c3, trajectory xy@{2,4,6,8}s.

Constants:
  EE_Z_GATE = 0.090  # MUST stay in sync with reposition_ik.py
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

EE_Z_GATE = 0.090

STEP_RE = re.compile(
    r"\[STEP\] step=(\d+) mode=(\w+) t=([0-9.]+)s.*?"
    r"ee=\(([+-][0-9.]+),([+-][0-9.]+),([+-][0-9.]+)\).*?"
    r"obj=\(([+-][0-9.]+),([+-][0-9.]+),([+-][0-9.]+)\).*?"
    r"goal_dist=([0-9.]+)m.*?"
    r"switch=(\w+)"
)
DRAKE_RE = re.compile(
    r"\[DRAKE-CONTACT\] step=(\d+) n_pairs=(\d+) ee_box_normal=([0-9.]+)"
)
GUARD_RE = re.compile(
    r"\[ADMIT-GUARD\] step=(\d+) admit_active=(\d+) latch=(\d+)/(\d+)"
)


def parse(log_path: Path) -> dict:
    steps: dict[int, dict] = {}
    drake: dict[int, tuple[int, float]] = {}
    guard: dict[int, tuple[int, int, int]] = {}

    for line in log_path.read_text(errors="replace").splitlines():
        m = STEP_RE.search(line)
        if m:
            s = int(m.group(1))
            steps[s] = dict(
                mode=m.group(2),
                t=float(m.group(3)),
                ee=(float(m.group(4)), float(m.group(5)), float(m.group(6))),
                obj=(float(m.group(7)), float(m.group(8)), float(m.group(9))),
                goal_dist=float(m.group(10)),
                switch=m.group(11),
            )
            continue
        m = DRAKE_RE.search(line)
        if m:
            drake[int(m.group(1))] = (int(m.group(2)), float(m.group(3)))
            continue
        m = GUARD_RE.search(line)
        if m:
            guard[int(m.group(1))] = (int(m.group(2)), int(m.group(3)), int(m.group(4)))

    if not steps:
        return dict(log=str(log_path), n_steps=0, error="no_step_lines")

    # SC-collision-gone events
    swing_collision_events = []
    for s in sorted(steps):
        d = steps[s]
        if d["mode"] != "free":
            continue
        if d["switch"] not in ("kStayInRepos", "kToBetterRepos"):
            continue
        if d["ee"][2] <= EE_Z_GATE:
            continue
        normal = drake.get(s, (0, 0.0))[1]
        if normal > 0.0:
            swing_collision_events.append(
                dict(step=s, t=d["t"], ee_z=d["ee"][2], normal=normal)
            )

    # SC-legit-approach: sustained mode=c3 + drake>0 + ee_z < gate
    in_legit = 0
    legit_intervals = []
    for s in sorted(steps):
        d = steps[s]
        normal = drake.get(s, (0, 0.0))[1]
        cond = d["mode"] == "c3" and normal > 0.0 and d["ee"][2] < EE_Z_GATE
        if cond:
            in_legit += 1
        else:
            if in_legit >= 20:
                legit_intervals.append(in_legit)
            in_legit = 0
    if in_legit >= 20:
        legit_intervals.append(in_legit)

    # noregress metrics
    last_step = max(steps)
    goal_at_end = steps[last_step]["goal_dist"]
    first_c3 = next((s for s in sorted(steps) if steps[s]["mode"] == "c3"), None)
    traj_samples = {}
    for tick_target in (200, 400, 600, 800):
        if tick_target in steps:
            traj_samples[f"t{tick_target * 0.01:.1f}"] = list(steps[tick_target]["obj"][:2])

    # swing-peak EE_z
    swing_peaks = []
    cur_max = 0.0
    in_swing = False
    for s in sorted(steps):
        d = steps[s]
        if d["mode"] == "free" and d["switch"] in ("kStayInRepos", "kToBetterRepos"):
            if not in_swing:
                cur_max = 0.0
                in_swing = True
            cur_max = max(cur_max, d["ee"][2])
        else:
            if in_swing and cur_max > 0:
                swing_peaks.append(cur_max)
            in_swing = False
    if in_swing and cur_max > 0:
        swing_peaks.append(cur_max)

    # ADMIT-GUARD pass-through vs cap (post-change only; baseline runs lack gate_cap field)
    n_admit_active = sum(1 for s, g in guard.items() if g[0] == 1)

    return dict(
        log=str(log_path),
        n_steps=len(steps),
        last_step=last_step,
        sim_t_end=steps[last_step]["t"],
        sc_collision_gone=dict(
            n_events=len(swing_collision_events),
            first_5=swing_collision_events[:5],
            passes=(len(swing_collision_events) == 0),
        ),
        sc_legit_approach=dict(
            intervals=legit_intervals,
            max_interval=max(legit_intervals) if legit_intervals else 0,
            passes=(max(legit_intervals) >= 20 if legit_intervals else False),
        ),
        noregress=dict(
            goal_dist_end=goal_at_end,
            first_c3_step=first_c3,
            trajectory_xy=traj_samples,
        ),
        swing_peak_ee_z=max(swing_peaks) if swing_peaks else None,
        n_admit_active_ticks=n_admit_active,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    results = {p.stem: parse(p) for p in args.logs}
    args.out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
