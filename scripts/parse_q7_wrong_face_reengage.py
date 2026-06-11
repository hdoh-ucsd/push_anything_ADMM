#!/usr/bin/env python3
"""Parse a q7_wrong_face_reengage sweep output directory and emit per-seed
columns plus the SC routing verdict.

Per-seed columns:
  seed | gate_pre | gate_post | unique_refused | n_watchdog |
  c3_entries | ping_pong_cycles | box_y_initial | box_y_final | box_y_drift |
  legit_refused_min_fa | result

SCs (folded in per plan 2026-06-10):
  - SC-RUNAWAY-PREVENTED: abs(box_y_drift) <= 0.16 AND n_watchdog == 0
    for all 3 seeds.
  - SC-NOREGRESS: every [GATE-COMMIT-FACE-POST] refusal has face_align > 0.0.
    Any refusal at face_align < 0.0 means a legit-class entry was blocked
    (the seed-4 step-315 productive session lived at face_align=-0.497) ->
    FAIL.
  - SC-PING-PONG-DISTINCTION: cycles<=3 RECOVERED (win), >3 LOOPS (gate
    works but sampler face-selection is next layer).
  - SC-NOT-NECESSARILY-REACH-GOAL: informational, no BAR.

Reads logs from <base>/seed{0,2,4}/run.log. Writes a single SUMMARY block
to stdout for the launcher script to capture.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Per-tick state regex: [GS] step=N mode=c3 switch=...
RE_GS = re.compile(
    r'^\[GS\] step=(\d+) mode=(c3|free) switch=(\S+)'
)
# Pre-decide refusal: existing [GATE-COMMIT-FACE].
RE_PRE = re.compile(
    r'^\[GATE-COMMIT-FACE\] step=(\d+) refused: face_align=([+-]?[0-9.]+)'
)
# Post-decide override: new [GATE-COMMIT-FACE-POST].
RE_POST = re.compile(
    r'^\[GATE-COMMIT-FACE-POST\] step=(\d+) refused: face_align=([+-]?[0-9.]+).*orig_reason=(\S+)'
)
# Watchdog firing.
RE_WATCHDOG = re.compile(r'^\[GS-watchdog\]')
# [STEP] line gives box pose per tick: obj=(x,y,z).
RE_STEP_OBJ = re.compile(
    r'^\[STEP\] step=(\d+) .* obj=\(([+-]?[0-9.]+),([+-]?[0-9.]+),([+-]?[0-9.]+)\)'
)
# Final result line.
RE_RESULT = re.compile(
    r'^\[RESULT\] .* final_obj_xy=\(([+-]?[0-9.]+),\s*([+-]?[0-9.]+)\).* goal_dist=([0-9.]+)m\s+success=(\S+)'
)

# Plan 2026-06-10 production threshold.
THRESHOLD = 0.3
# Ping-pong window in ticks (refusal -> re-attempt must happen within this).
PING_PONG_WINDOW = 60
PING_PONG_RECOVER_MAX = 3   # <=3 cycles = RECOVERED, >3 = LOOPS


def parse_run(seed_dir: Path):
    log = seed_dir / "run.log"
    if not log.exists():
        return None
    lines = log.read_text(errors="ignore").splitlines()

    n_pre = 0
    n_post = 0
    n_watchdog = 0
    pre_steps = set()
    post_steps = set()
    n_c3_entries = 0
    prev_mode = "free"
    last_post_refusal_step = None
    ping_pong_cycles = 0
    initial_obj_y = None
    final_obj_y = None
    final_obj_x = None
    final_step = 0
    result_box_x = None
    result_box_y = None
    result_goal_dist = None
    result_success = None
    # Track post-decide refusal face_aligns; any < 0.0 means we blocked a
    # legit-class entry (SC-NOREGRESS FAIL).
    legit_refused_face_aligns = []

    for line in lines:
        m = RE_GS.match(line)
        if m:
            step = int(m.group(1))
            mode = m.group(2)
            if mode == "c3" and prev_mode == "free":
                n_c3_entries += 1
                # Ping-pong: refusal in the last PING_PONG_WINDOW ticks
                # followed by a SUCCESSFUL c3-entry counts as one
                # refuse->retry->re-enter cycle.
                if (last_post_refusal_step is not None
                        and step - last_post_refusal_step <= PING_PONG_WINDOW):
                    ping_pong_cycles += 1
                last_post_refusal_step = None
            prev_mode = mode
            continue
        m = RE_PRE.match(line)
        if m:
            n_pre += 1
            pre_steps.add(int(m.group(1)))
            continue
        m = RE_POST.match(line)
        if m:
            n_post += 1
            step = int(m.group(1))
            fa = float(m.group(2))
            post_steps.add(step)
            last_post_refusal_step = step
            # SC-NOREGRESS: refusal at face_align < 0.0 means a legit-class
            # entry was blocked. Capture for the SUMMARY.
            if fa < 0.0:
                legit_refused_face_aligns.append((step, fa))
            continue
        if RE_WATCHDOG.match(line):
            n_watchdog += 1
            continue
        m = RE_STEP_OBJ.match(line)
        if m:
            step = int(m.group(1))
            ox = float(m.group(2))
            oy = float(m.group(3))
            if initial_obj_y is None:
                initial_obj_y = oy
            final_obj_x = ox
            final_obj_y = oy
            final_step = step
            continue
        m = RE_RESULT.match(line)
        if m:
            result_box_x = float(m.group(1))
            result_box_y = float(m.group(2))
            result_goal_dist = float(m.group(3))
            result_success = m.group(4)

    # Prefer [RESULT] for final pose (it is the authoritative post-sim
    # measurement) and [STEP] only as fallback.
    box_y_final = result_box_y if result_box_y is not None else final_obj_y
    box_x_final = result_box_x if result_box_x is not None else final_obj_x

    box_y_drift = (
        (box_y_final - initial_obj_y)
        if (box_y_final is not None and initial_obj_y is not None)
        else float("nan")
    )

    return dict(
        n_pre=n_pre,
        n_post=n_post,
        unique_refused=len(pre_steps | post_steps),
        n_watchdog=n_watchdog,
        n_c3_entries=n_c3_entries,
        ping_pong_cycles=ping_pong_cycles,
        initial_obj_y=initial_obj_y,
        box_y_final=box_y_final,
        box_x_final=box_x_final,
        box_y_drift=box_y_drift,
        legit_refused_face_aligns=legit_refused_face_aligns,
        final_step=final_step,
        result_goal_dist=result_goal_dist,
        result_success=result_success,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", type=Path)
    args = ap.parse_args()

    rows = []
    for seed in [0, 2, 4]:
        sd = args.base / f"seed{seed}"
        r = parse_run(sd)
        if r is None:
            rows.append((seed, None))
            continue
        rows.append((seed, r))

    print("=" * 110)
    print(
        f"{'seed':>4} {'gate_pre':>8} {'gate_post':>9} {'uniq_ref':>8} "
        f"{'watchdog':>8} {'c3_ent':>6} {'ping_pong':>9} "
        f"{'oy_init':>8} {'oy_final':>9} {'oy_drift':>9} {'goal':>6} {'ok':>4}"
    )
    print("-" * 110)
    for seed, r in rows:
        if r is None:
            print(f"{seed:>4}   NO LOG")
            continue
        print(
            f"{seed:>4} {r['n_pre']:>8} {r['n_post']:>9} "
            f"{r['unique_refused']:>8} {r['n_watchdog']:>8} "
            f"{r['n_c3_entries']:>6} {r['ping_pong_cycles']:>9} "
            f"{(r['initial_obj_y'] or 0.0):>8.4f} "
            f"{(r['box_y_final'] or 0.0):>9.4f} "
            f"{r['box_y_drift']:>9.4f} "
            f"{(r['result_goal_dist'] or 0.0):>6.3f} "
            f"{(r['result_success'] or '-'):>4}"
        )
    print("=" * 110)

    valid_rows = [r for _, r in rows if r is not None]

    # SC-RUNAWAY-PREVENTED: |box_y_drift| <= 0.16 AND n_watchdog == 0 for all.
    if not valid_rows:
        runaway_verdict = "INCONCLUSIVE (no runs)"
    else:
        runaway_pass = all(
            abs(r["box_y_drift"]) <= 0.16 and r["n_watchdog"] == 0
            for r in valid_rows
        )
        runaway_verdict = "PASS" if runaway_pass else "FAIL"

    # SC-NOREGRESS: no [GATE-COMMIT-FACE-POST] refusal with face_align < 0.0
    # on any seed (would mean we blocked a legit-class entry).
    legit_blocks_total = sum(
        len(r["legit_refused_face_aligns"]) for r in valid_rows
    )
    noregress_verdict = "PASS" if legit_blocks_total == 0 else "FAIL"

    # SC-PING-PONG-DISTINCTION: per-seed status when post-decide refusals
    # occurred. If 0 post-refusals, status N/A (gate didn't need to fire).
    pp_status_strs = []
    for seed, r in rows:
        if r is None:
            continue
        if r["n_post"] == 0:
            pp_status_strs.append(f"seed{seed}=N/A(no_refusal)")
        elif r["ping_pong_cycles"] <= PING_PONG_RECOVER_MAX:
            pp_status_strs.append(
                f"seed{seed}=RECOVERED({r['ping_pong_cycles']})"
            )
        else:
            pp_status_strs.append(
                f"seed{seed}=LOOPS({r['ping_pong_cycles']})"
            )

    print(f"SC-RUNAWAY-PREVENTED  {runaway_verdict}  "
          f"(BAR: abs(box_y_drift)<=0.16 AND n_watchdog==0 for all seeds)")
    print(f"SC-NOREGRESS          {noregress_verdict}  "
          f"(BAR: no [GATE-COMMIT-FACE-POST] refusal with face_align<0; "
          f"observed legit_blocks={legit_blocks_total})")
    if legit_blocks_total > 0:
        for r in valid_rows:
            for step, fa in r["legit_refused_face_aligns"]:
                print(f"  NOREGRESS-VIOLATION step={step} face_align={fa:+.3f}")
    print(f"SC-PING-PONG          " + "  ".join(pp_status_strs))
    print(f"SC-NOT-NECESSARILY-REACH-GOAL  informational (no BAR)")


if __name__ == "__main__":
    main()
