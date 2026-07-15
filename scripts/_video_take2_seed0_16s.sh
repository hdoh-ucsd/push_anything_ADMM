#!/usr/bin/env bash
# Video record-pass take 2: seed=0, 16s, SERIAL, HEAD 7479985.
# Identical protocol to take 1; relies on Ipopt FP-divergence after ~step 103
# (project_b3b_refuted_paired_bit_identical) to produce a different trajectory.
# Take 1 landed intermediate (0.150); take 2 looks for deep good-basin (<=0.12).
set -uo pipefail

OUTBASE=/root/push_anything_ADMM/video_take2
PY=${PY:-/root/miniconda3/envs/push_anything_ADMM/bin/python}
LABEL=take2_seed0_16s

TMP_PCT=$(df --output=pcent /tmp 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
if [ "${TMP_PCT:-0}" -ge 85 ]; then
  echo "ABORT: /tmp at ${TMP_PCT}% -- STOP, do NOT rm -rf /tmp/claude-0/*"
  exit 1
fi

mkdir -p "$OUTBASE"
START=$(date +%s)
echo "=== ${LABEL} start $START seed=0 16s serial HEAD=$(git rev-parse --short HEAD) ===" \
  >> "$OUTBASE/run.log"
cd /root/push_anything_ADMM
"$PY" -u main.py pushing \
    --task-id 4 \
    --solver c3plus --c3plus-projection lcp --ee-space \
    --sampling-c3 config/sampling_c3_kik.yaml \
    --admm-iter 25 \
    --max-time 16.0 \
    --no-record \
    --name "${LABEL}" \
    --seed 0 \
    >> "$OUTBASE/run.log" 2>&1 || true
ELAPSED=$(( $(date +%s) - START ))
echo "DONE ${LABEL} ${ELAPSED}s" >> "$OUTBASE/run.log"
tail -3 "$OUTBASE/run.log"
