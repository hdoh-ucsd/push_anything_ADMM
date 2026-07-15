#!/usr/bin/env bash
# Paired seed sweep for β (Phase-3 descent stickiness).
#
# STEPC at task-id 4 (west), z=0.03 (per-task sampling_height), 2.5s sim,
# c3plus solver with --admm-iter 25. 5 seeds × {β on, β off} = 10 runs.
#
# Each --seed N is used twice (β on and β off) so the only nondeterminism
# difference is the toggle.  Runs sequentially — parallel CPU contention
# would skew per-step timing and make wall-clock comparisons noisy.
#
# After all runs complete, per-log analysis via
# scripts/parse_beta_contact_events.py and a side-by-side aggregate.
set -uo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT="results/beta_seed_sweep_${TIMESTAMP}"
mkdir -p "$OUT"

MAX_TIME=2.5
ADMM_ITER=25
SEEDS=(0 1 2 3 4)
PER_RUN_TIMEOUT=600        # wall-clock cap per run, seconds (10 min)

run_one() {
    local seed="$1"
    local label="$2"        # "beta_on" or "beta_off"
    local extra="$3"        # "" or "--no-beta"
    local out="$OUT/stepc_seed${seed}_${label}.log"

    echo "--- seed=$seed $label ---"
    local t0=$(date +%s)
    timeout "$PER_RUN_TIMEOUT" python main.py pushing \
        --task-id 4 \
        --solver c3plus \
        --sampling-c3 \
        --admm-iter "$ADMM_ITER" \
        --max-time "$MAX_TIME" \
        --no-record \
        --seed "$seed" \
        --name "stepc_seed${seed}_${label}" \
        $extra \
        > "$out" 2>&1
    local rc=$?
    local dt=$(($(date +%s) - t0))
    if [ $rc -eq 124 ]; then
        echo "  TIMEOUT after ${dt}s"
    elif [ $rc -ne 0 ]; then
        echo "  FAILED rc=$rc (${dt}s)"
    else
        local n_step=$(grep -c '^\[STEP\]' "$out" || true)
        local n_contact=$(grep -c 'contact=Y' "$out" || true)
        local res=$(grep '^\[RESULT\]' "$out" | head -1 || true)
        echo "  OK ${dt}s steps=$n_step contact=$n_contact"
        [ -n "$res" ] && echo "  $res"
    fi
}

for seed in "${SEEDS[@]}"; do
    run_one "$seed" "beta_on"  ""
    run_one "$seed" "beta_off" "--no-beta"
done

echo
echo "=== Sweep complete: $OUT ==="
echo
echo "Run aggregation:"
echo "  for f in $OUT/*.log; do python scripts/parse_beta_contact_events.py \"\$f\"; done"
