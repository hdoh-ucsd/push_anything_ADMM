#!/usr/bin/env bash
# w_align A/B (resume): only the wa30k arm.  wa0 reuses prior
# overshoot_causal_pwl_seed{0,1,2}.txt (same YAML — landed conformance).
# Both arms share traj_type=kPiecewiseLinear, entry_align_threshold=0.0.
# Only w_align differs (isolated test of the alignment bonus).
# Timeout per run bumped to 2400s (wa0_seed0 hit 1820s in the prior pass;
# wa30k can be heavier due to extra cost-term term + ranking flips).
set -u
OUTDIR=/root/push_anything_ADMM/results
STATUS=/root/push_anything_ADMM/scripts/walign_ab_status.log
: > "$STATUS"

run_one() {
  local arm="$1" seed="$2" yaml="$3"
  local stem="walign_ab_${arm}_seed${seed}"
  local start=$(date +%s)
  echo "[$(date)] BEGIN arm=${arm} seed=${seed} yaml=${yaml}" >> "$STATUS"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  timeout 2400 python3 -u main.py pushing \
      --task-id 4 \
      --solver c3plus \
      --c3plus-projection lcp \
      --ee-space \
      --sampling-c3 "$yaml" \
      --admm-iter 25 \
      --max-time 12 \
      --seed "$seed" \
      --no-record \
      --name "$stem" \
      > "${OUTDIR}/${stem}.stdout" 2>&1
  local rc=$?
  local elapsed=$(( $(date +%s) - start ))
  local result=$(grep -E '^\[RESULT\]' "${OUTDIR}/${stem}.txt" 2>/dev/null | tail -1)
  echo "[$(date)] END   arm=${arm} seed=${seed} rc=${rc} elapsed=${elapsed}s ${result}" >> "$STATUS"
}

for SEED in 0 1 2; do
  run_one wa30k "$SEED" config/sampling_c3_walign30k.yaml
done

echo "[$(date)] ALL DONE" >> "$STATUS"
cat "$STATUS"
