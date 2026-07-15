#!/usr/bin/env bash
# Early-window entry_align_threshold ablation pair: 0.7 (current/baseline) vs
# 0.0 (disabled/identity) at HEAD 7479985, seed=0, SERIAL, <=1.5s sim,
# --no-record.
#
# Third principled component in the conformance ablation. Toggle is at
# config/sampling_c3_kik.yaml:62 (params.py:618 — L1 goal-aligned
# contact-normal requirement at c3 admission). Same protocol as
# _early_window_pair_wAlign_ablation.sh.
#
# GAMBLE: this gate fires at c3 admission. Prior runs show first c3 entry
# at ~step 110 (1.1s sim) — entry_align_threshold MAY only fire near
# the end of [0, 1.5s] window or not at all if alignment exceeds 0.7 in
# both legs. If no early-window delta surfaces, conclude "early-window
# method doesn't cover entry_align — decide treatment separately."
set -uo pipefail

OUTBASE=/root/push_anything_ADMM/early_window_pair_entry_align_ablation
PY=${PY:-/root/miniconda3/envs/push_anything_ADMM/bin/python}

TMP_PCT=$(df --output=pcent /tmp 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
if [ "${TMP_PCT:-0}" -ge 85 ]; then
  echo "ABORT: /tmp at ${TMP_PCT}% -- STOP, do NOT rm -rf /tmp/claude-0/*"
  exit 1
fi
[ -r "$OUTBASE/entry_align_0p7.yaml" ] || { echo "ABORT: 0p7 YAML missing"; exit 1; }
[ -r "$OUTBASE/entry_align_0p0.yaml" ] || { echo "ABORT: 0p0 YAML missing"; exit 1; }

run_one() {
  local LABEL="$1"
  local YAML="$2"
  local OUT="$OUTBASE/${LABEL}"
  mkdir -p "$OUT"
  local START
  START=$(date +%s)
  echo "=== ${LABEL} start $START seed=0 1.5s serial yaml=$YAML ===" \
    >> "$OUT/run.log"
  cd /root/push_anything_ADMM
  "$PY" -u main.py pushing \
      --task-id 4 \
      --solver c3plus --c3plus-projection lcp --ee-space \
      --sampling-c3 "$YAML" \
      --admm-iter 25 \
      --max-time 1.5 \
      --no-record \
      --name "early_window_${LABEL}" \
      --seed 0 \
      >> "$OUT/run.log" 2>&1 || true
  local ELAPSED=$(( $(date +%s) - START ))
  echo "DONE ${LABEL} ${ELAPSED}s" >> "$OUT/run.log"
}

run_one "entry_align_0p7" "$OUTBASE/entry_align_0p7.yaml"

TMP_PCT=$(df --output=pcent /tmp 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
if [ "${TMP_PCT:-0}" -ge 85 ]; then
  echo "ABORT mid-sweep: /tmp at ${TMP_PCT}% before entry_align_0p0"
  exit 1
fi

run_one "entry_align_0p0" "$OUTBASE/entry_align_0p0.yaml"

for L in entry_align_0p7 entry_align_0p0; do
  tail -2 "$OUTBASE/${L}/run.log" 2>/dev/null
done
