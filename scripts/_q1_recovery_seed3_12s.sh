#!/usr/bin/env bash
# Q1 SC-Q1-recovery probe: single seed-3 12s run at the post-Q1-fix HEAD.
# CRITICAL: --save-video OFF (it OOM'd the 16s run at sim_t=9.6s).
# Plan: docs/superpowers/plans/2026-06-03-q1-dispatcher-reentry-guard.md
# Detached, manual reads only, NO Monitor/loop/wakeup.
set -eo pipefail
OUTDIR=${OUTDIR:-q1_recovery_seed3_12s}
PY=${PY:-/root/miniconda3/envs/push_anything_ADMM/bin/python}
mkdir -p "$OUTDIR"
git rev-parse HEAD > "$OUTDIR/HEAD.txt"
TMP_PCT=$(df --output=pcent /tmp 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
if [ "${TMP_PCT:-0}" -ge 85 ]; then
  echo "ABORT: /tmp at ${TMP_PCT}% — STOP, do NOT rm -rf /tmp/claude-0/*" \
    | tee -a "$OUTDIR/run.log"
  exit 1
fi
echo "=== start $(date +%s) HEAD=$(cat $OUTDIR/HEAD.txt) ===" | tee -a "$OUTDIR/run.log"
START=$(date +%s)
"$PY" -u main.py pushing \
    --task-id 4 \
    --solver c3plus \
    --c3plus-projection lcp \
    --ee-space \
    --sampling-c3 config/sampling_c3_kik.yaml \
    --admm-iter 25 \
    --max-time 12 \
    --name seed3_12s_q1fix \
    --seed 3 \
    >> "$OUTDIR/run.log" 2>&1 || true
RC=$?
ELAPSED=$(( $(date +%s) - START ))
LOG="$OUTDIR/run.log"
N_LCS=$(grep -c '\[CONTACT-RUN\] step=.* contact_type=EE-BOX' "$LOG" || true)
N_REENTRY_INF=$(grep -cE '^\[GS\] .* switch=kToC3Cost .* best_other=- ' "$LOG" || true)
N_REPOS_UNPROD=$(grep -cE '^\[GS\] .* switch=kToReposUnproductive' "$LOG" || true)
N_STAY_REPOS=$(grep -cE '^\[GS\] .* switch=kStayInRepos' "$LOG" || true)
RESULT_LINE=$(grep '^\[RESULT\]' "$LOG" | tail -1 || true)
echo "DONE rc=$RC ${ELAPSED}s n_lcs=$N_LCS n_reentry_inf=$N_REENTRY_INF n_unprod=$N_REPOS_UNPROD n_stay_repos=$N_STAY_REPOS | $RESULT_LINE" \
  | tee -a "$OUTDIR/run.log"
echo "FINISHED $(date +%s)" | tee -a "$OUTDIR/run.log"
