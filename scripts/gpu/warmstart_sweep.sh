#!/usr/bin/env bash
# Candidate warm-start semantics sweep (measurement only; defaults unchanged).
#
#   A "ordered"     current behaviour, candidate k warm-starts k+1
#   B "independent" every candidate sees the tick's entry u_prev
#   C "reset"       every candidate starts from u_prev = None
#
# Runs are SEQUENTIAL by construction -- evals are never parallelized.
# Usage: scripts/gpu/warmstart_sweep.sh <box|t> <seconds> <outdir>
set -euo pipefail
TASK=${1:-box}
LIMIT=${2:-60}
OUT=${3:-/tmp}

if [ "$TASK" = "box" ]; then
  CMD=(python3 main.py pushing --task-id 4 --sampling-c3 config/sampling_c3_kik.yaml)
else
  CMD=(python3 main.py push_t_mesh --sampling-c3 config/sampling_c3_kik_t.yaml)
fi

for m in ordered independent reset; do
  echo "=== ${TASK} ${LIMIT}s : ${m} ==="
  PORT_CANDIDATE_WARMSTART="$m" timeout 3000 \
    "${CMD[@]}" --max-time "$LIMIT" --seed 0 \
    > "${OUT}/ws_${TASK}_${m}.log" 2>&1 || echo "  (exit $?)"
done
echo "ALLDONE ${TASK}"
