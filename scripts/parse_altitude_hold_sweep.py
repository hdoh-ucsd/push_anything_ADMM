"""Stage-1 altitude-hold sweep verdict parser.

Reads altitude_hold_sweep/seed{0..4}_altitude_hold.log + sweep.summary
and evaluates against the pre-registered SCs from
docs/superpowers/plans/2026-06-01-wrong-face-race-fix.md:
  SC'             : <20 ticks per seed in box-z + non-goal-face region
  SC-goal         : >=3/5 seeds first-contact +x AND box motion -x AND goal_dist<0.15
  SC-noregress    : seed 0 success=YES, lambda_n_max ≥5.0 count = 0 across all seeds
  SC-distributional: 5-seed protocol enforced by script
  Deadlock        : ANY seed n_aee1=0 OR final goal_dist > 0.5m

Plus reports the target-change interval distribution per seed (the
diagnostic the user pre-registered for routing Stage 2 if Stage 1 fails).
"""
import re
import pathlib
import sys
from collections import Counter

import numpy as np

OUTDIR = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "altitude_hold_sweep")

RGX_GATE = re.compile(
    r"\[GATE-CONTACT\] step=(\d+) F_W=\(([+-][\d.]+),([+-][\d.]+),([+-][\d.]+)\)"
    r" F_on_box=\(([+-][\d.]+),([+-][\d.]+),([+-][\d.]+)\)"
    r" n_face_out=\(([+-][\d.]+),([+-][\d.]+),([+-][\d.]+)\)"
    r" A_is_ee=(\d).*?box_p=\(([+-][\d.]+),([+-][\d.]+),([+-][\d.]+)\)"
    r" ee_p=\(([+-][\d.]+),([+-][\d.]+),([+-][\d.]+)\)"
)
RGX_TGTCHG = re.compile(r"\[TGT-CHANGE\] step=(\d+) interval_ticks=(-?\d+)")
RGX_RESULT = re.compile(
    r"\[RESULT\] method=sampling-c3\s+final_obj_xy=\(([+-]?[\d.]+),\s*([+-]?[\d.]+)\)"
    r"\s+goal_dist=([\d.]+)m\s+success=(YES|NO)"
)
RGX_LAMBDA = re.compile(r"\[C3\+\] step=\d+ \|u\[0\]\|=[\d.]+N λ_n_max=([\d.]+)")


def parse_seed(seed: int):
    p = OUTDIR / f"seed{seed}_altitude_hold.log"
    rows = []
    intervals = []
    result = None
    lambda_max_over_threshold = 0  # count of ticks with λ_n_max >= 5.0
    with open(p) as fh:
        for ln in fh:
            m = RGX_GATE.match(ln)
            if m:
                rows.append(dict(
                    step=int(m[1]),
                    F_W=np.array([float(m[2]), float(m[3]), float(m[4])]),
                    n_face_out=np.array([float(m[8]), float(m[9]), float(m[10])]),
                    A_is_ee=int(m[11]),
                    box_p=np.array([float(m[12]), float(m[13]), float(m[14])]),
                    ee_p=np.array([float(m[15]), float(m[16]), float(m[17])]),
                ))
                continue
            m = RGX_TGTCHG.search(ln)
            if m:
                ivl = int(m[2])
                if ivl > 0:
                    intervals.append(ivl)
                continue
            m = RGX_RESULT.search(ln)
            if m:
                result = dict(
                    final_obj_x=float(m[1]),
                    final_obj_y=float(m[2]),
                    goal_dist=float(m[3]),
                    success=(m[4] == 'YES'),
                )
                continue
            m = RGX_LAMBDA.search(ln)
            if m and float(m[1]) >= 5.0:
                lambda_max_over_threshold += 1
    return rows, intervals, result, lambda_max_over_threshold


def axis_label(v: np.ndarray) -> str:
    if np.linalg.norm(v) < 1e-6:
        return 'zero'
    a = np.abs(v)
    i = int(np.argmax(a))
    if a[i] / np.linalg.norm(v) < 0.85:
        return f"mixed({v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f})"
    return f"{'+' if v[i] >= 0 else '-'}{'xyz'[i]}"


def evaluate(seed: int, rows, intervals, result, lambda_violations):
    drake_rows = [r for r in rows if r['A_is_ee'] == 1]
    n_aee1 = len(drake_rows)
    # SC' — ticks where EE was in "box-z + non-goal-face region":
    # box_top z ~ 0.10 (box at z=0.05, half-extent 0.05). Non-goal face
    # means |y| > 0.06 (south side primarily). Goal is -x (push west).
    # The pre-registered "non-goal" region: ee z below 0.10 AND ee y outside [-0.06, +0.06].
    sc_prime_ticks = sum(
        1 for r in rows
        if r['ee_p'][2] < 0.10 and abs(r['ee_p'][1]) > 0.06
    )
    # SC-goal — first Drake contact face / dominant face / box motion direction
    if drake_rows:
        first = drake_rows[0]
        first_face = axis_label(first['n_face_out'])
        # Total Δbox over contact window
        bp_first = drake_rows[0]['box_p']
        bp_last = drake_rows[-1]['box_p']
        dbox = bp_last - bp_first
        box_motion_axis = axis_label(dbox)
    else:
        first_face = '<no_drake_contact>'
        box_motion_axis = '<no_drake_contact>'
        dbox = np.zeros(3)
    # SC-goal pass: first_face is +x or +x-dominant mixed AND box motion is -x
    sc_goal_face_pass = first_face.startswith('+x') or ('+x' in first_face and 'mixed' in first_face and abs(first['n_face_out'][0]) > 0.85)
    sc_goal_motion_pass = bool(dbox[0] < -0.05)  # >5cm westward
    sc_goal_dist_pass = bool(result and result['goal_dist'] < 0.15)
    sc_goal_pass = sc_goal_face_pass and sc_goal_motion_pass and sc_goal_dist_pass
    # Deadlock
    deadlock = (n_aee1 == 0) or (result and result['goal_dist'] > 0.50)
    return dict(
        n_aee1=n_aee1,
        sc_prime_ticks=sc_prime_ticks,
        first_face=first_face,
        box_motion_axis=box_motion_axis,
        dbox=dbox,
        sc_goal_face_pass=sc_goal_face_pass,
        sc_goal_motion_pass=sc_goal_motion_pass,
        sc_goal_dist_pass=sc_goal_dist_pass,
        sc_goal_pass=sc_goal_pass,
        deadlock=deadlock,
        result=result,
        lambda_violations=lambda_violations,
        n_intervals=len(intervals),
        interval_min=(min(intervals) if intervals else None),
        interval_max=(max(intervals) if intervals else None),
        interval_mean=(sum(intervals) / len(intervals) if intervals else None),
        interval_distribution=Counter(intervals).most_common(5),
    )


def main():
    print(f"=== Stage 1 altitude-hold sweep verdict (read from {OUTDIR}) ===")
    print()
    results = {}
    for seed in range(5):
        p = OUTDIR / f"seed{seed}_altitude_hold.log"
        if not p.exists():
            print(f"seed {seed}: log not found ({p})")
            continue
        rows, intervals, result, lambda_v = parse_seed(seed)
        ev = evaluate(seed, rows, intervals, result, lambda_v)
        results[seed] = ev
        print(f"--- seed {seed} ---")
        print(f"  result: {result}")
        print(f"  n_aee1 (Drake-contact ticks): {ev['n_aee1']}")
        print(f"  first-contact face   : {ev['first_face']}")
        print(f"  box motion (Δbox)    : {ev['dbox']} axis={ev['box_motion_axis']}")
        print(f"  SC' wrong-face ticks : {ev['sc_prime_ticks']}  (bar < 20)")
        print(f"  SC-goal face pass    : {ev['sc_goal_face_pass']}  motion pass : {ev['sc_goal_motion_pass']}  dist pass : {ev['sc_goal_dist_pass']}  TOTAL : {ev['sc_goal_pass']}")
        print(f"  Deadlock             : {ev['deadlock']}")
        print(f"  λ_n_max ≥ 5.0 count  : {ev['lambda_violations']}  (bar = 0)")
        _mean_str = f"{ev['interval_mean']:.1f}" if ev['interval_mean'] is not None else "n/a"
        print(f"  TGT-CHANGE intervals : n={ev['n_intervals']}  min={ev['interval_min']}  max={ev['interval_max']}  mean={_mean_str}  top5={ev['interval_distribution']}")
        print()

    # Aggregate SC verdict
    print("=== AGGREGATE VERDICT ===")
    n_goal = sum(1 for ev in results.values() if ev['sc_goal_pass'])
    n_seeds = len(results)
    print(f"SC-goal       : {n_goal}/{n_seeds} seeds goal-directed (bar ≥ 3/5)  → {'PASS' if n_goal >= 3 else 'FAIL'}")
    print(f"SC-noregress  : seed 0 success = {results.get(0, {}).get('result', {}).get('success', '?')}  → {'PASS' if results.get(0, {}).get('result', {}).get('success') else 'FAIL'}")
    total_violations = sum(ev['lambda_violations'] for ev in results.values())
    print(f"SC-noregress  : λ violations total = {total_violations}  → {'PASS' if total_violations == 0 else 'FAIL'}")
    any_deadlock = any(ev['deadlock'] for ev in results.values())
    print(f"Deadlock      : any seed deadlocked = {any_deadlock}  → routes to {'Stage 2A' if any_deadlock else 'continue'}")
    # SC' aggregate
    sc_prime_seeds_pass = sum(1 for ev in results.values() if ev['sc_prime_ticks'] < 20)
    print(f"SC'           : {sc_prime_seeds_pass}/{n_seeds} seeds with <20 wrong-face ticks")

    # Overall verdict
    print()
    if n_goal >= 3 and results.get(0, {}).get('result', {}).get('success') and total_violations == 0 and not any_deadlock:
        print(">>> OUTCOME: Stage 1 ALONE CLEARS SC-goal. STOP. Stages 2A/2B unneeded.")
    elif any_deadlock:
        print(">>> OUTCOME: Stage 1 DEADLOCKED. Read TGT-CHANGE intervals to disambiguate:")
        print("    - interval_mean < TARGET_STABLE_TICKS (=5): contributor-2 oscillation real → Stage 2A")
        print("    - interval_mean >> 5: gate constant mistuned → retune, not Stage 2A")
    else:
        print(f">>> OUTCOME: Stage 1 INSUFFICIENT ({n_goal}/{n_seeds} goal-directed). Classify failure mode:")
        print("    - clean descent (SC' high pass-rate) but wrong-face commitment without oscillation → Stage 2B (sampling bias)")
        print("    - oscillation evident in TGT-CHANGE intervals → Stage 2A (dispatcher commit)")

if __name__ == "__main__":
    main()
