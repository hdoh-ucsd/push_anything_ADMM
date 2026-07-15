#!/usr/bin/env bash
# Canonical baseline sweep on pristine HEAD (38dbf18).
# 20 seeds, 8-way parallel, --admm-iter 25, max-time 2.5s, task-id 4.
# Output: $OUT/seed{N}.log per seed.
set -uo pipefail

OUT="${1:?usage: run_canonical_baseline_sweep.sh OUT_DIR}"
mkdir -p "$OUT"

PARALLEL=8
N_SEEDS=20
MAX_TIME=2.5
ADMM_ITER=25
PER_RUN_TIMEOUT=900

run_one() {
    local seed="$1"
    local out="$OUT/seed${seed}.log"
    local t0=$(date +%s)
    timeout "$PER_RUN_TIMEOUT" python main.py pushing \
        --task-id 4 \
        --solver c3plus \
        --sampling-c3 \
        --admm-iter "$ADMM_ITER" \
        --max-time "$MAX_TIME" \
        --no-record \
        --seed "$seed" \
        --name "canon_seed${seed}" \
        > "$out" 2>&1
    local rc=$?
    local dt=$(($(date +%s) - t0))
    local n_step=$(grep -c '^\[STEP\]' "$out" 2>/dev/null || echo 0)
    local n_contact=$(grep -c 'contact=Y' "$out" 2>/dev/null || echo 0)
    local res=$(grep '^\[RESULT\]' "$out" 2>/dev/null | head -1)
    if [ $rc -eq 0 ]; then
        echo "seed=$seed OK ${dt}s steps=$n_step contact=$n_contact"
    elif [ $rc -eq 124 ]; then
        echo "seed=$seed TIMEOUT ${dt}s"
    else
        echo "seed=$seed FAIL rc=$rc ${dt}s"
    fi
}
export -f run_one
export OUT PER_RUN_TIMEOUT MAX_TIME ADMM_ITER

# Use xargs -P for 8-way parallelism. Sequential within the same shell so
# stdout lines from run_one are line-buffered.
seq 0 $((N_SEEDS - 1)) | xargs -n 1 -P "$PARALLEL" -I{} bash -c 'run_one "$@"' _ {}
echo "=== sweep complete: $OUT ==="
