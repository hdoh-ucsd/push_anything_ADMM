#!/usr/bin/env bash
# run_until_complete.sh — run one long sim to completion across VM reboots.
#
# The WSL VM on this box restarts on its own (5 reboots on 2026-08-09, two of
# them 6 min apart, one with nothing running). A 1200 s T-push needs ~25 min of
# wall time, which exceeds the mean time between reboots, so a plain launch
# keeps dying partway. This script is idempotent and is driven by a systemd
# unit with Restart=always + WantedBy=multi-user.target, so after each reboot
# it retries automatically until one attempt gets a long enough window.
#
# The sim cannot resume mid-trajectory, so each attempt restarts from t=0.
# Output only lands at RESULT_PATH when a run actually finishes, so a partial
# attempt can never be mistaken for a completed one.
#
# Usage: run_until_complete.sh RUN_NAME MAX_TIME [TASK] [CONFIG]

set -uo pipefail

RUN_NAME="${1:?usage: run_until_complete.sh RUN_NAME MAX_TIME [TASK] [CONFIG]}"
MAX_TIME="${2:?missing MAX_TIME}"
TASK="${3:-push_t}"
CONFIG="${4:-config/sampling_c3_kik_t.yaml}"

cd /root/push_anything_ADMM

PY=/root/miniconda3/envs/push_anything_ADMM/bin/python3
# main.py ALREADY tees its stdout to results/<name>.txt (main.py:362-364), so
# it owns this path. Do NOT also redirect the shell's stdout here: two handles
# at independent offsets overwrite each other and shred the log (that is what
# produced the interleaved lines, the vanished [POSE-REGIME] banner and the
# invalid UTF-8 byte that crashed the frame renderer). Shell stdout goes to a
# separate diagnostic file instead.
RESULT_PATH="results/${RUN_NAME}.txt"
SHELL_LOG="results/.${RUN_NAME}.shell"
ATTEMPT_LOG="results/.${RUN_NAME}.attempts"

# Already finished? Nothing to do — exit 0 so systemd stops retrying.
if [ -f "$RESULT_PATH" ] && grep -aq "Simulation complete" "$RESULT_PATH"; then
  echo "[run-until-complete] $RUN_NAME already complete; nothing to do"
  exit 0
fi

echo "[run-until-complete] attempt at $(date '+%F %T'), boot age $(cut -d. -f1 /proc/uptime)s" \
  >> "$ATTEMPT_LOG"

"$PY" main.py "$TASK" --sampling-c3 "$CONFIG" --max-time "$MAX_TIME" \
  --name "$RUN_NAME" > "$SHELL_LOG" 2>&1
rc=$?

# main.py truncates RESULT_PATH ("w") at startup, so a killed attempt leaves a
# partial there. Completion is decided ONLY by the end-of-run marker, never by
# the file existing.
if grep -aq "Simulation complete" "$RESULT_PATH" 2>/dev/null; then
  echo "[run-until-complete] COMPLETE (rc=$rc) -> $RESULT_PATH" >> "$ATTEMPT_LOG"
  exit 0
fi

LAST_STEP=$(grep -ao "step=[0-9]*" "$RESULT_PATH" 2>/dev/null | tail -1)
echo "[run-until-complete] INCOMPLETE (rc=$rc, $LAST_STEP) — will retry" >> "$ATTEMPT_LOG"
exit 1   # non-zero so systemd Restart=always fires again
