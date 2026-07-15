#!/usr/bin/env bash
# Single seed-3 run at L2 baseline 02eca7f, max-time=16s, Drake video ON.
# Stall-permanence probe — does the extra 8s let seed 3 re-acquire contact?
# Detached, manual reads only, NO Monitor/loop/wakeup.
set -eo pipefail
OUTDIR=${OUTDIR:-stage2_L2_seed3_16s}
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
    --max-time 16 \
    --name seed3_16s_02eca7f \
    --seed 3 \
    >> "$OUTDIR/run.log" 2>&1 || true
RC=$?
ELAPSED=$(( $(date +%s) - START ))
LOG="$OUTDIR/run.log"
N_LCS=$(grep -c '\[CONTACT-RUN\] step=.* contact_type=EE-BOX' "$LOG" || true)
N_GATE_L1=$(grep -c '^\[GATE-ALIGN\]' "$LOG" || true)
N_GATE_L2=$(grep -c '^\[GATE-COMMIT-FACE\]' "$LOG" || true)
RESULT_LINE=$(grep '^\[RESULT\]' "$LOG" | tail -1 || true)
echo "DONE rc=$RC ${ELAPSED}s n_lcs=$N_LCS gate_L1=$N_GATE_L1 gate_L2=$N_GATE_L2 | $RESULT_LINE" \
  | tee -a "$OUTDIR/run.log"
echo "FINISHED $(date +%s)" | tee -a "$OUTDIR/run.log"
