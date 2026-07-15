#!/usr/bin/env bash
# Overshoot-causation pass: 3 seeds x 2 trackers, --max-time 12, SERIAL.
# kIK uses config/sampling_c3_causal_kik.yaml; kPWL uses the landed
# config/sampling_c3_kik.yaml.  Output stems land in results/.
set -u
OUTDIR=/root/push_anything_ADMM/results
STATUS=/root/push_anything_ADMM/scripts/overshoot_causal_status.log
: > "$STATUS"

run_one() {
  local tracker="$1" seed="$2" yaml="$3"
  local stem="overshoot_causal_${tracker}_seed${seed}"
  local start=$(date +%s)
  echo "[$(date)] BEGIN tracker=${tracker} seed=${seed} yaml=${yaml}" >> "$STATUS"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  timeout 1500 python3 -u main.py pushing \
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
  echo "[$(date)] END   tracker=${tracker} seed=${seed} rc=${rc} elapsed=${elapsed}s ${result}" >> "$STATUS"
  return $rc
}

# kPWL first (faster), then kIK
for SEED in 0 1 2; do
  run_one pwl "$SEED" config/sampling_c3_kik.yaml || true
done
for SEED in 0 1 2; do
  run_one kik "$SEED" config/sampling_c3_causal_kik.yaml || true
done

echo "[$(date)] ALL DONE" >> "$STATUS"
echo "--- final status ---"
cat "$STATUS"
