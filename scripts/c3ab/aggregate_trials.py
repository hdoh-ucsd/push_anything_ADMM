#!/usr/bin/env python3
"""Aggregate a c3ab trial ledger into the requested result schema.

Definitions: execution_time = sim seconds to first success (censored at the
trial end for failures); steps_to_goal = 10 Hz object-state samples to first
success (10 x execution_time), censored likewise; errors are the trial-final
measured pose errors (for successes, the post-success settled values).
"""
import json, sys, statistics as st

def moments(vals):
    if not vals: return {'mean': None, 'std': None}
    return {'mean': st.fmean(vals),
            'std': st.pstdev(vals) if len(vals) > 1 else 0.0}

trials = [json.loads(l) for l in open(sys.argv[1])]
succ = [t for t in trials if t['success']]
exec_t = [(t['success_t'] if t['success'] else t['end_t']) for t in trials]
steps = [10.0*x for x in exec_t]
steps_s = [10.0*t['success_t'] for t in succ]
pos = moments([t['pos_err'] for t in trials])
pos_s = moments([t['pos_err'] for t in succ])
th = moments([t['theta_err'] for t in trials])
th_s = moments([t['theta_err'] for t in succ])
et = moments(exec_t); sg = moments(steps); sg_s = moments(steps_s)
result = {
    'n_trials': len(trials),
    'success_rate': len(succ)/len(trials) if trials else None,
    'pos_err_mean': pos['mean'], 'pos_err_std': pos['std'],
    'pos_err_mean_success': pos_s['mean'], 'pos_err_std_success': pos_s['std'],
    'theta_err_mean': th['mean'], 'theta_err_std': th['std'],
    'theta_err_mean_success': th_s['mean'],
    'theta_err_std_success': th_s['std'],
    'mean_execution_time': et['mean'], 'std_execution_time': et['std'],
    'mean_steps_to_goal': sg['mean'], 'std_steps_to_goal': sg['std'],
    'mean_steps_to_goal_success': sg_s['mean'],
}
print(json.dumps(result, indent=2))
if len(sys.argv) > 2:
    open(sys.argv[2], 'w').write(json.dumps(result, indent=2))
