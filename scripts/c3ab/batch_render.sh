#!/usr/bin/env bash
set -u
P=/root/miniconda3/envs/push_anything_ADMM/bin/python3
R=/root/push_anything_ADMM/results
render() { # dir sdf title
  [ -f "$R/$1/rollout.mp4" ] && { echo "skip $1"; return; }
  [ -f "$R/$1/state_trace.jsonl" ] || { echo "no trace $1"; return; }
  $P /root/.claude/jobs/cfe8fda1/tmp/render_c3plus_run.py --run-dir "$R/$1" \
    --object-sdf "examples/sampling_c3/urdf/$2" --title "$3" 2>&1 | tail -1
}
for i in 1 2 3 4 5; do render c3ab_exp1_trials_trial$i push_t.sdf "native T 1kg | OIM goal | native Q | trial $i"; done
for i in 1 2 3 4 5; do render c3ab_exp3_trials_trial$i push_t_m05.sdf "native shape 0.05kg | OIM goal | trial $i"; done
for i in 1 2 3; do render c3ab_expTx_trials_trial$i push_t_Tx.sdf "T_x scaled 0.371 | 1kg | trial $i"; done
for i in 1 2 3; do render c3ab_expTy_trials_trial$i push_t_Ty.sdf "T_y scaled 0.620 | 1kg | trial $i"; done
for i in 1 2 3; do render c3ab_expTz_trials_trial$i push_t_Tz.sdf "T_z scaled 1.49 | 1kg | trial $i"; done
for i in 1 2 3; do render c3ab_exp2_ctrl_trials_trial$i push_t_oimscale.sdf "OIM-dims T | native Q (control) | trial $i"; done
for i in 1 2 3; do render c3ab_exp4_ruleQ_trials_trial$i push_t_oimscale.sdf "OIM-dims T | rule Q (quat 260) | trial $i"; done
echo BATCH_DONE
