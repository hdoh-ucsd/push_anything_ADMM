#!/usr/bin/env bash
# Extend the 5-seed β-on sweep to 20 seeds.
# Canonical baseline = HEAD (38dbf18) + uncommitted β working tree.
# Already collected: seeds 0..4 in results/beta_seed_sweep_20260528_164722/.
# This script runs seeds 5..19 with the same flags.
set -uo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT="results/baseline_ext_${TIMESTAMP}"
mkdir -p "$OUT"

MAX_TIME=2.5
ADMM_ITER=25
SEEDS=(5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)
PER_RUN_TIMEOUT=600

run_one() {
    local seed="$1"
    local out="$OUT/stepc_seed${seed}_beta_on.log"
    local t0=$(date +%s)
    timeout "$PER_RUN_TIMEOUT" python main.py pushing \
        --task-id 4 \
        --solver c3plus \
        --sampling-c3 \
        --admm-iter "$ADMM_ITER" \
        --max-time "$MAX_TIME" \
        --no-record \
        --seed "$seed" \
        --name "stepc_seed${seed}_beta_on" \
        > "$out" 2>&1
    local rc=$?
    local dt=$(($(date +%s) - t0))
    local n_contact=$(grep -c 'contact=Y' "$out" || true)
    echo "seed=$seed rc=$rc ${dt}s contact_steps=$n_contact"
}

for seed in "${SEEDS[@]}"; do
    run_one "$seed"
done

echo
echo "=== Extension complete: $OUT ==="
