#!/usr/bin/env python3
"""Phase A2 analyzer — post-entry phantom-contact trace.

Reads run.log files from a setarch-probe-style directory and emits a CSV
with per-tick {ee, box, drake_admit, planner_lam_n, planner_f_cmd, drake_distance}
for steps in a window around the c3-entry point.

Output: CSV + a short markdown summary cross-tabulating phantom-DURATION
(consecutive c3 ticks with no Drake EE-BOX admit) against basin outcome.

Usage:
    python scripts/_brittleness_phase_a_postentry_trace.py \\
        --run-dir nondet_seed0_setarch_hashseed \\
        --start-step 105 --end-step 200 \\
        --out-csv audit_output/brittleness_phase_a2_phantom.csv \\
        --out-md audit_output/brittleness_phase_a2_phantom.md
"""
import argparse
import csv
import pathlib
import re
import sys


STEP_RE = re.compile(
    r'^\[STEP\] step=(?P<step>\d+) mode=(?P<mode>\w+) .*'
    r'ee=\((?P<ee>[^)]+)\) obj=\((?P<obj>[^)]+)\) .*'
    r'lam_n=(?P<lam>[0-9.]+).* contact=(?P<contact>[NY]).*'
    r'f_cmd=\((?P<fcmd>[^)]+)\)'
)
CONTACT_RUN_RE = re.compile(
    r'^\[CONTACT-RUN\] step=(?P<step>\d+) nhat_BA_W=\[(?P<nhat>[^\]]+)\] '
    r'p_BCb=\[[^\]]+\] distance=(?P<dist>[+\-0-9.]+) contact_type=(?P<ctype>\w+)'
)
RESULT_RE = re.compile(r'\[RESULT\].*goal_dist=([-0-9.]+)m')
GS_C3_RE = re.compile(r'^\[GS\] step=(\d+) mode=c3')


def parse_run(log_path: pathlib.Path, start: int, end: int):
    """Per-tick rows for a single run log + summary metrics."""
    rows = {}  # step -> dict
    contacts = {}  # step -> list of (ctype, distance, nhat)
    first_c3_step = None
    goal_dist = None

    with log_path.open() as f:
        for line in f:
            m = STEP_RE.match(line)
            if m:
                step = int(m['step'])
                if start <= step <= end:
                    ee = tuple(float(x) for x in m['ee'].split(','))
                    obj = tuple(float(x) for x in m['obj'].split(','))
                    fcmd = tuple(float(x) for x in m['fcmd'].split(','))
                    rows[step] = dict(
                        mode=m['mode'], ee=ee, obj=obj,
                        lam_n=float(m['lam']),
                        contact_admit=(m['contact'] == 'Y'),
                        f_cmd=fcmd,
                    )
                continue
            m = CONTACT_RUN_RE.match(line)
            if m:
                step = int(m['step'])
                if start <= step <= end:
                    contacts.setdefault(step, []).append(
                        (m['ctype'], float(m['dist']),
                         tuple(float(x) for x in m['nhat'].split(','))))
                continue
            if first_c3_step is None:
                m = GS_C3_RE.match(line)
                if m:
                    first_c3_step = int(m[1])
            m = RESULT_RE.search(line)
            if m:
                goal_dist = float(m[1])

    # Derive drake_admit + min_distance per step.
    for step, row in rows.items():
        ee_box_pairs = [c for c in contacts.get(step, [])
                        if c[0] == 'EE-BOX' and c[1] < 1.0]
        row['drake_admit'] = bool(ee_box_pairs)
        row['drake_distance'] = (min(c[1] for c in ee_box_pairs)
                                 if ee_box_pairs else float('nan'))
        # Phantom = c3 mode AND planner says contact (lam_n > 0.1)
        # AND Drake says NO real EE-BOX pair admitted.
        row['phantom'] = (row['mode'] == 'c3'
                          and row['lam_n'] > 0.1
                          and not row['drake_admit'])

    # Find first real Drake EE-BOX admit.
    first_admit = None
    for step in sorted(rows):
        if rows[step]['drake_admit']:
            first_admit = step
            break

    # Phantom duration = ticks from first_c3 to first_admit, ALL phantom.
    phantom_duration = None
    if first_c3_step and first_admit:
        phantom_duration = first_admit - first_c3_step
    elif first_c3_step:
        # Never admitted in window — duration is the window tail.
        phantom_duration = end - first_c3_step

    return dict(
        rows=rows, first_c3=first_c3_step, first_admit=first_admit,
        phantom_duration=phantom_duration, goal_dist=goal_dist,
    )


def classify_basin(goal_dist):
    if goal_dist is None:
        return 'UNKNOWN'
    if goal_dist < 0.10:
        return 'GOOD'
    if goal_dist < 0.15:
        return 'borderline'
    return 'BAD'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--start-step', type=int, default=105)
    ap.add_argument('--end-step', type=int, default=200)
    ap.add_argument('--out-csv', required=True)
    ap.add_argument('--out-md', required=True)
    args = ap.parse_args()

    run_dir = pathlib.Path(args.run_dir)
    runs = {}
    for sub in sorted(run_dir.glob('run*')):
        log = sub / 'run.log'
        if not log.is_file():
            continue
        idx = int(re.search(r'run(\d+)', sub.name)[1])
        runs[idx] = parse_run(log, args.start_step, args.end_step)

    if not runs:
        sys.exit(f'no run logs in {run_dir}')

    pathlib.Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'step', 'mode', 'ee_x', 'ee_y', 'ee_z',
                    'obj_x', 'obj_y', 'obj_z',
                    'planner_lam_n', 'planner_contact_admit',
                    'drake_admit', 'drake_distance',
                    'phantom', 'fcmd_x', 'fcmd_y', 'fcmd_z'])
        for idx in sorted(runs):
            for step in sorted(runs[idx]['rows']):
                r = runs[idx]['rows'][step]
                w.writerow([idx, step, r['mode'], *r['ee'], *r['obj'],
                            f"{r['lam_n']:.4f}",
                            int(r['contact_admit']),
                            int(r['drake_admit']),
                            f"{r['drake_distance']:.5f}",
                            int(r['phantom']), *r['f_cmd']])

    # Build cross-tab markdown.
    lines = []
    lines.append('# Phase A2 — phantom-contact cross-tab\n')
    lines.append(f'**Source:** `{args.run_dir}` | **window:** steps '
                 f'{args.start_step}–{args.end_step}\n')
    lines.append('## Per-run phantom duration vs basin outcome\n')
    lines.append('| run | first_c3 | first_real_admit | phantom_duration '
                 '(ticks/ms) | goal_dist | basin |')
    lines.append('|---|---|---|---|---|---|')
    for idx in sorted(runs):
        r = runs[idx]
        admit = r['first_admit'] if r['first_admit'] else '— (none in window)'
        dur = r['phantom_duration']
        dur_str = f'**{dur}** / {dur * 10}ms' if dur is not None else 'n/a'
        gd_str = f'{r["goal_dist"]:.4f}m' if r['goal_dist'] else '?'
        lines.append(f'| {idx} | {r["first_c3"]} | {admit} | {dur_str} | '
                     f'{gd_str} | {classify_basin(r["goal_dist"])} |')
    lines.append('')

    # Phantom-fraction in first 10 c3 ticks.
    lines.append('## Phantom-fraction in first 10 c3 ticks per run\n')
    lines.append('| run | c3 ticks in window | phantom ticks | phantom fraction |')
    lines.append('|---|---|---|---|')
    for idx in sorted(runs):
        r = runs[idx]
        if not r['first_c3']:
            continue
        c3_ticks = [s for s, row in r['rows'].items()
                    if row['mode'] == 'c3'
                    and r['first_c3'] <= s < r['first_c3'] + 10]
        phantoms = [s for s in c3_ticks if r['rows'][s]['phantom']]
        frac = len(phantoms) / max(len(c3_ticks), 1)
        lines.append(f'| {idx} | {len(c3_ticks)} | {len(phantoms)} | '
                     f'{frac * 100:.0f}% |')
    lines.append('')

    # Discriminator analysis.
    lines.append('## Discriminator analysis\n')
    good = [r for r in runs.values()
            if r['goal_dist'] and r['goal_dist'] < 0.15
            and r['phantom_duration'] is not None]
    bad = [r for r in runs.values()
           if r['goal_dist'] and r['goal_dist'] >= 0.15
           and r['phantom_duration'] is not None]
    if good and bad:
        g_max = max(r['phantom_duration'] for r in good)
        b_min = min(r['phantom_duration'] for r in bad)
        gap = b_min - g_max
        lines.append(f'- GOOD runs phantom_duration: max = {g_max} ticks')
        lines.append(f'- BAD runs phantom_duration: min = {b_min} ticks')
        lines.append(f'- **Gap: {gap} ticks** ({"CLEAN — discriminator works" if gap > 0 else "OVERLAP"})')
    lines.append('')

    pathlib.Path(args.out_md).write_text('\n'.join(lines))
    print(f'wrote {args.out_csv} ({sum(len(r["rows"]) for r in runs.values())} rows)')
    print(f'wrote {args.out_md}')


if __name__ == '__main__':
    main()
