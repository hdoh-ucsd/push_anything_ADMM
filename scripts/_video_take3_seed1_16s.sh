#!/usr/bin/env bash
# Video record-pass take 3: FRESH SEED (seed=1, NOT seed=0 re-roll), 16s,
# SERIAL, HEAD 7479985. seed=0 missed twice differently (take 1 stall 15cm
# short, take 2 overshoot 28cm past) — both reached goal vicinity, neither
# settled. seed=1 is a deterministically-different trajectory, evidence-gathering
# take: land-it or learn-it.
# REASSESS TRIGGER: if seed-1 also misses, clean-<=0.12 settle is genuinely
# rare across seeds (3 misses across 2 seeds = reconsider-deck-needs).
set -uo pipefail

OUTBASE=/root/push_anything_ADMM/video_take3
PY=${PY:-/root/miniconda3/envs/push_anything_ADMM/bin/python}
LABEL=take3_seed1_16s

TMP_PCT=$(df --output=pcent /tmp 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
if [ "${TMP_PCT:-0}" -ge 85 ]; then
  echo "ABORT: /tmp at ${TMP_PCT}% -- STOP, do NOT rm -rf /tmp/claude-0/*"
  exit 1
fi

mkdir -p "$OUTBASE"
START=$(date +%s)
echo "=== ${LABEL} start $START seed=1 16s serial HEAD=$(git rev-parse --short HEAD) ===" \
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
    --seed 1 \
    >> "$OUTBASE/run.log" 2>&1 || true
ELAPSED=$(( $(date +%s) - START ))
echo "DONE ${LABEL} ${ELAPSED}s" >> "$OUTBASE/run.log"
tail -3 "$OUTBASE/run.log"
