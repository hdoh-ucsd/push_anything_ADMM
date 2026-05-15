#!/usr/bin/env bash
# Run the contact-free joint-PD operating-envelope sweep.
# 5 distances x 3 seeds = 15 runs, sequential.
set -u
cd "$(dirname "$0")/.."

OUT_DIR="results/contact_free_sweep/logs"
mkdir -p "$OUT_DIR"

DISTANCES_CM=(5 10 15 20 30)
SEEDS=(0 1 2)

TOTAL=15
i=0
sweep_start=$(date +%s)

for d in "${DISTANCES_CM[@]}"; do
  # zero-pad
  dd=$(printf '%02d' "$d")
  goal_x=$(python3 -c "print(round(${d}/100.0, 6))")
  for s in "${SEEDS[@]}"; do
    i=$((i+1))
    name="sweep_d${dd}cm_s${s}"
    log="${OUT_DIR}/${name}.log"
    t0=$(date +%s)
    echo "[SWEEP] ${i}/${TOTAL}: d=${dd}cm s=${s} (goal=${goal_x},0.0) → running..."

    timeout 360 python main.py pushing \
        --sampling-c3 config/sampling_c3_kik.yaml \
        --prepositioned \
        --no-record \
        --goal-xy "${goal_x},0.0" \
        --seed "${s}" \
        --name "${name}" \
        > "${log}" 2>&1
    rc=$?

    t1=$(date +%s)
    wall=$((t1 - t0))

    if [[ $rc -eq 124 ]]; then
      status="timeout"
    elif [[ $rc -ne 0 ]]; then
      status="crashed(rc=${rc})"
    else
      status=$(grep -m1 "^\[RESULT\]" "$log" | sed -E 's/.*success=([A-Z]+).*/result=\1/')
      [[ -z "$status" ]] && status="no_result_line"
    fi
    echo "[SWEEP] ${i}/${TOTAL}: d=${dd}cm s=${s} → ${status} (${wall}s)"
  done
done

sweep_end=$(date +%s)
echo "[SWEEP] DONE. Total wall-clock: $((sweep_end - sweep_start)) s"
