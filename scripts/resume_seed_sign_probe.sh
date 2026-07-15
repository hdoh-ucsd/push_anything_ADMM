#!/usr/bin/env bash
# Resume the seed-sign probe for seeds 6..9 (both arms).
# Reuses the layout of scripts/run_seed_sign_probe.sh; APPENDS to sweep.summary.
set -uo pipefail

REPO=/root/push_anything_ADMM
cd "$REPO"

if [ -z "${PROBE_OUT:-}" ]; then
    echo "ERROR: PROBE_OUT not set" >&2
    exit 2
fi
OUT="$PROBE_OUT"
[ -d "$OUT" ] || { echo "missing $OUT"; exit 3; }

MAX_TIME=3.5
ADMM_ITER=25
PER_RUN_TIMEOUT=900
PARALLEL=2
PY=/root/miniconda3/envs/push_anything_ADMM/bin/python
[ -x "$PY" ] || { echo "missing python interpreter: $PY"; exit 4; }

YAML_OFF=config/sampling_c3_kik.yaml
YAML_ON=audit_output/phaseC_gate_runs/vff_ab/vff_alpha_0.5.yaml

[ -f "$YAML_OFF" ] || { echo "missing $YAML_OFF"; exit 3; }
[ -f "$YAML_ON"  ] || { echo "missing $YAML_ON";  exit 3; }
HEAD=$(git -C "$REPO" rev-parse HEAD)
SHORT=$(git -C "$REPO" rev-parse --short HEAD)
EXPECT=37418f67d296c9f4fba5e91222a871196ff24e56
if [ "$HEAD" != "$EXPECT" ]; then
    echo "WARN: HEAD=$HEAD (short=$SHORT) is NOT the v_des structural fix ($EXPECT)" >&2
fi
echo "RESUME HEAD=$HEAD (short=$SHORT)" | tee -a "$OUT/HEAD.txt"

run_one() {
    local seed="$1"
    local arm="$2"
    local yaml
    if   [ "$arm" = "off"      ]; then yaml="$YAML_OFF"
    elif [ "$arm" = "alpha05"  ]; then yaml="$YAML_ON"
    else echo "bad arm: $arm" >&2; return 2; fi
    local out="$OUT/seed${seed}_${arm}.log"
    local t0=$(date +%s)
    timeout "$PER_RUN_TIMEOUT" "$PY" main.py pushing \
        --task-id 4 \
        --solver c3plus \
        --sampling-c3 "$yaml" \
        --admm-iter "$ADMM_ITER" \
        --max-time "$MAX_TIME" \
        --no-record \
        --seed "$seed" \
        --name "seedsign_seed${seed}_${arm}" \
        > "$out" 2>&1
    local rc=$?
    local dt=$(($(date +%s) - t0))
    local n_contact=$(grep -c 'A_is_ee=1' "$out" 2>/dev/null || true)
    local final=$(grep -m1 '\[RESULT\]' "$out" 2>/dev/null || echo 'no-result')
    echo "seed=$seed arm=$arm rc=$rc ${dt}s n_contact=$n_contact | $final"
}
export -f run_one
export OUT YAML_OFF YAML_ON MAX_TIME ADMM_ITER PER_RUN_TIMEOUT PY

JOBS_FILE="$OUT/jobs_resume.txt"
: > "$JOBS_FILE"
for seed in 6 7 8 9; do
    for arm in off alpha05; do
        printf "%s %s\n" "$seed" "$arm" >> "$JOBS_FILE"
    done
done

echo
echo "=== resume sweep: $(wc -l < "$JOBS_FILE") runs, ${PARALLEL}-parallel ==="
xargs -P "$PARALLEL" -L 1 -a "$JOBS_FILE" bash -c 'run_one "$@"' _ \
    | tee -a "$OUT/sweep.summary"

echo
echo "=== resume complete: $OUT ==="
ls "$OUT" | sort | head -60
