#!/usr/bin/env bash
# Multi-seed sweep: c3plus + ee-space with --c3plus-projection in
# {componentwise, lcp}. 5 seeds per arm = 10 runs total. 2-parallel.
#
# Primary signal: λ_n_max distribution per solve (PRIMARY discriminator
# of whether the feasibility-guaranteed projection bounds λ across seeds,
# not just on seed-4).
# Secondary: contact engagement + per-solve wallclock (the speed cost).
set -uo pipefail
REPO=/root/push_anything_ADMM
cd "$REPO"
OUT=${PROBE_OUT:?PROBE_OUT must be set}
mkdir -p "$OUT"
PY=/root/miniconda3/envs/push_anything_ADMM/bin/python
[ -x "$PY" ] || { echo "missing python interpreter: $PY"; exit 4; }

MAX_TIME=3.5
ADMM_ITER=25
PER_RUN_TIMEOUT=1200
PARALLEL=2

git -C "$REPO" rev-parse HEAD  > "$OUT/HEAD.txt"
git -C "$REPO" status --short >> "$OUT/HEAD.txt"

run_one() {
    local seed="$1" proj="$2"
    local out="$OUT/seed${seed}_${proj}.log"
    local t0=$(date +%s)
    timeout "$PER_RUN_TIMEOUT" "$PY" main.py pushing \
        --task-id 4 \
        --solver c3plus \
        --c3plus-projection "$proj" \
        --ee-space \
        --sampling-c3 config/sampling_c3_kik.yaml \
        --admm-iter "$ADMM_ITER" \
        --max-time "$MAX_TIME" \
        --no-record \
        --seed "$seed" \
        --name "c3plus_${proj}_seed${seed}_off" \
        > "$out" 2>&1
    local rc=$?
    local dt=$(($(date +%s) - t0))
    local n_aee1=$(grep -c 'A_is_ee=1' "$out" 2>/dev/null || true)
    local n_lcs=$(grep -c 'CONTACT-RUN.*EE-BOX' "$out" 2>/dev/null || true)
    local n_solves=$(grep -c '^\[C3+\] step=' "$out" 2>/dev/null || true)
    local final=$(grep -m1 '\[RESULT\]' "$out" 2>/dev/null || echo 'no-result')
    echo "seed=$seed proj=$proj rc=$rc ${dt}s n_solves=$n_solves n_lcs=$n_lcs n_aee1=$n_aee1 | $final"
}
export -f run_one
export OUT PY MAX_TIME ADMM_ITER PER_RUN_TIMEOUT

JOBS="$OUT/jobs.txt"
: > "$JOBS"
for seed in 0 1 2 3 4; do
    for proj in componentwise lcp; do
        printf "%s %s\n" "$seed" "$proj" >> "$JOBS"
    done
done

echo "=== launching ${PARALLEL}-parallel sweep: $(wc -l < "$JOBS") runs ==="
xargs -P "$PARALLEL" -L 1 -a "$JOBS" bash -c 'run_one "$@"' _ \
    | tee "$OUT/sweep.summary"

echo
echo "=== sweep complete: $OUT ==="
ls "$OUT" | sort | head -25
