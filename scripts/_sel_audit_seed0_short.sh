#!/usr/bin/env bash
# B3c-prime selection audit — N=6 SHORT instrumented paired sweep at seed=0.
#
# Purpose: emit the c_samples vector + 4 cost components per-k at the argmin
# site (wrapper.py:957-975) across steps [150,170] in 6 identical-seed runs,
# so the reducer can find the FLIP TICK (where paired k_star differs) and pin
# SELECT_HYSTERESIS against the at-that-tick gap (not a window average).
#
# Protocol matches scripts/_nondet_seed0_518bcfa.sh in everything but
# --max-time (16s -> 2.5s) and the C3_SEL_* env-gate. ~2.5s reaches step
# ~159 (at dt_ctrl=0.01s) and finishes well before the EIO window that
# killed runs 3/4/5/6 in the long-run sweep.
#
# Each run writes sel_0_${I}.jsonl into $OUTBASE/sel/. Per-row flush =
# EIO-durable to the last partial line. Serial (parallelism 1), single-try
# per run, no retry, no auto-resume.
set -uo pipefail

OUTBASE=/root/push_anything_ADMM/sel_audit_seed0_short
PY=${PY:-/root/miniconda3/envs/push_anything_ADMM/bin/python}
YAML=/tmp/sampling_c3_kik_518bcfa.yaml

# /tmp abort guard FIRST. NEVER sudo rm -rf /tmp/claude-0/*.
TMP_PCT=$(df --output=pcent /tmp 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
if [ "${TMP_PCT:-0}" -ge 85 ]; then
  echo "ABORT: /tmp at ${TMP_PCT}% -- STOP, do NOT rm -rf /tmp/claude-0/*"
  exit 1
fi
[ -r "$YAML" ] || { echo "ABORT: YAML missing at $YAML"; exit 1; }

mkdir -p "$OUTBASE/sel"
cp "$YAML" "$OUTBASE/effective_config.yaml"
git -C /root/push_anything_ADMM rev-parse HEAD > "$OUTBASE/main_tree_HEAD.txt"

run_one() {
  local I="$1"
  local OUT="$OUTBASE/run${I}"
  mkdir -p "$OUT"
  local START
  START=$(date +%s)
  echo "=== run $I start $START seed=0 short=2.5s sel_audit=on ===" \
    >> "$OUT/run.log"
  cd /root/push_anything_ADMM
  C3_SEL_AUDIT="$OUTBASE/sel" \
  C3_SEL_PAIR_ID="0" \
  C3_SEL_RUN_ID="0_${I}" \
  C3_SEL_LO="150" \
  C3_SEL_HI="170" \
  "$PY" -u main.py pushing \
      --task-id 4 \
      --solver c3plus --c3plus-projection lcp --ee-space \
      --sampling-c3 "$YAML" \
      --admm-iter 25 \
      --max-time 2.5 \
      --no-record \
      --name "sel_audit_run${I}_seed0" \
      --seed 0 \
      >> "$OUT/run.log" 2>&1 || true
  local ELAPSED=$(( $(date +%s) - START ))
  local N_ROWS
  N_ROWS=$(wc -l < "$OUTBASE/sel/sel_0_${I}.jsonl" 2>/dev/null || echo 0)
  echo "DONE run=$I ${ELAPSED}s rows=$N_ROWS" >> "$OUT/run.log"
}

# Serial execution (parallelism 1). One run at a time → no concurrent FP
# noise from sharing cores / cache, and EIO on one run doesn't stomp another.
for I in 1 2 3 4 5 6; do
  # /tmp guard between runs too, in case /tmp filled during the sweep.
  TMP_PCT=$(df --output=pcent /tmp 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
  if [ "${TMP_PCT:-0}" -ge 85 ]; then
    echo "ABORT mid-sweep: /tmp at ${TMP_PCT}% before run ${I}"
    exit 1
  fi
  run_one "$I"
done

# Summary line per run.
for I in 1 2 3 4 5 6; do
  tail -1 "$OUTBASE/run${I}/run.log" 2>/dev/null
done
