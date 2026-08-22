#!/usr/bin/env bash
# Canonical evaluation run for the GPU-ADMM plan.
#
# ONE seed (0) is held fixed for every run in this plan while the
# CONFIGURATION varies (threads=1 / threads=4 / PORT_GPU_ADMM=1). This is
# NOT a seed sweep -- see memory/feedback_no_statistical_evaluation.md --
# it is the same-protocol baseline discipline required by
# memory/feedback_baseline_provenance.md.
#
# Usage: scripts/gpu/run_gate.sh <max_time_s> <out_log>
set -euo pipefail
LIMIT=${1:-60}
OUT=${2:-/tmp/gate.log}
timeout 1800 python3 main.py pushing --task-id 4 --max-time "$LIMIT" \
  --sampling-c3 config/sampling_c3_kik.yaml --seed 0 2>&1 | tee "$OUT"
