#!/usr/bin/env python3
"""Collect all c3ab run videos into one folder with feature-encoded names."""
import json, shutil
from pathlib import Path

R = Path('/root/push_anything_ADMM/results')
OUT = R / 'c3ab_video_collection_20260903'
OUT.mkdir(exist_ok=True)

LEDGERS = {
    'c3ab_exp1_trials':      'nativeT-1kg_oimgoal_nativeQ',
    'c3ab_exp3_trials':      'lightT-0.05kg_oimgoal',
    'c3ab_expTx_trials':     'Tx-scaled-0.371_1kg',
    'c3ab_expTy_trials':     'Ty-scaled-0.620_1kg',
    'c3ab_expTz_trials':     'Tz-scaled-1.49_1kg',
    'c3ab_exp2_ctrl_trials': 'oimdimsT-1kg_nativeQ-control',
    'c3ab_exp4_ruleQ_trials':'oimdimsT-1kg_ruleQ-quat260',
}
count = 0
for led, tag in LEDGERS.items():
    lp = R / f'{led}.jsonl'
    if not lp.exists():
        continue
    for t in map(json.loads, lp.open()):
        src = Path(t['run_dir']) / 'rollout.mp4'
        if not src.exists():
            continue
        if t['success']:
            feat = f"SUCCESS-{t['success_t']:.0f}s"
        else:
            feat = f"FAIL_pos{t['pos_err']:.2f}m_th{t['theta_err']:.2f}rad"
        name = f"{tag}_trial{t['trial']}_{feat}.mp4"
        shutil.copy2(src, OUT / name)
        count += 1

# Screening finals (already-rendered single runs).
FINALS = [
    ('c3ab_exp0_final', 'nativeT-1kg_stockgoal_control-screening_FAIL_pos0.16m_th0.01rad'),
    ('c3ab_exp1_final', 'nativeT-1kg_oimgoal_screening_FAIL_pos0.36m_th0.27rad'),
    ('c3ab_exp2_final', 'oimdimsT-1kg_oimgoal_screening_FAIL_pos0.53m_th2.22rad'),
    ('c3ab_exp3_final', 'lightT-0.05kg_oimgoal_screening_SUCCESS-69s'),
]
for d, name in FINALS:
    src = R / d / 'rollout.mp4'
    if src.exists():
        shutil.copy2(src, OUT / f'{name}.mp4')
        count += 1
print(f'collected {count} videos into {OUT}')
for f in sorted(OUT.iterdir()):
    print(' ', f.name)
