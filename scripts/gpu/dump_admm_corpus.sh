#!/usr/bin/env bash
# Capture a corpus of C3+ ADMM instances (inputs + CPU golden outputs) for
# GPU replay validation. GPU-ADMM plan Task 4.
set -euo pipefail
OUT=${1:-audit_output/admm_corpus}
mkdir -p "$OUT"
DIAG_ADMM_DUMP_DIR="$OUT" DIAG_ADMM_DUMP_EVERY=${EVERY:-20} \
  DIAG_ADMM_DUMP_MAX=${MAXN:-60} \
  scripts/gpu/run_gate.sh "${LIMIT:-60}" /tmp/corpus_run.log
echo "instances: $(ls "$OUT"/inst_*[0-9].npz 2>/dev/null | wc -l)"
echo "paired   : $(ls "$OUT"/inst_*_out.npz 2>/dev/null | wc -l)"
