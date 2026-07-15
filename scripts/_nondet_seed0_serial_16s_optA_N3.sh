#!/usr/bin/env bash
# N=3 SERIAL 16s seed=0 characterization of Option A noise-floor reducer.
#
# Goal: measure the goal_dist spread WITH Option A
# (hold_last_good_p_des_on_failure=true). Expect spread MUCH smaller than
# the 52mm pre-A IK-failure-cascade floor — just the silent FP-drift left
# after the 35cm bifurcations are removed. That residual spread IS the
# noise floor the downstream ablation will measure against.
#
# Protocol: serial (parallelism=1), 16s, --no-record, seed=0, MANUAL read
# (no Monitor/loop/wakeup). /tmp abort-guard FIRST. Never sudo rm -rf
# /tmp/claude-0/*. Each run writes its own [RESULT] line so partial-completion
# (WSL OOM, as happened to OptB run2) is recoverable per-run.
set -uo pipefail

OUTBASE=/root/push_anything_ADMM/nondet_seed0_serial_16s_optA_N3
PY=${PY:-/root/miniconda3/envs/push_anything_ADMM/bin/python}
YAML=/tmp/sampling_c3_kik_optA.yaml

TMP_PCT=$(df --output=pcent /tmp 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
if [ "${TMP_PCT:-0}" -ge 85 ]; then
  echo "ABORT: /tmp at ${TMP_PCT}% -- STOP, do NOT rm -rf /tmp/claude-0/*"
  exit 1
fi
[ -r "$YAML" ] || { echo "ABORT: YAML missing at $YAML"; exit 1; }

mkdir -p "$OUTBASE"
cp "$YAML" "$OUTBASE/effective_config.yaml"
git -C /root/push_anything_ADMM rev-parse HEAD > "$OUTBASE/main_tree_HEAD.txt"
git -C /root/push_anything_ADMM diff HEAD > "$OUTBASE/working_tree_diff.patch"

run_one() {
  local I="$1"
  local OUT="$OUTBASE/run${I}"
  mkdir -p "$OUT"
  local START
  START=$(date +%s)
  echo "=== run $I start $START seed=0 16s serial optA(hold_last_good_p_des) ===" \
    >> "$OUT/run.log"
  cd /root/push_anything_ADMM
  "$PY" -u main.py pushing \
      --task-id 4 \
      --solver c3plus --c3plus-projection lcp --ee-space \
      --sampling-c3 "$YAML" \
      --admm-iter 25 \
      --max-time 16.0 \
      --no-record \
      --name "serial_16s_optA_run${I}_seed0" \
      --seed 0 \
      >> "$OUT/run.log" 2>&1 || true
  local ELAPSED=$(( $(date +%s) - START ))
  echo "DONE run=$I ${ELAPSED}s" >> "$OUT/run.log"
}

for I in 1 2 3; do
  TMP_PCT=$(df --output=pcent /tmp 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
  if [ "${TMP_PCT:-0}" -ge 85 ]; then
    echo "ABORT mid-sweep: /tmp at ${TMP_PCT}% before run ${I}"
    exit 1
  fi
  run_one "$I"
done

for I in 1 2 3; do
  tail -2 "$OUTBASE/run${I}/run.log" 2>/dev/null
done
