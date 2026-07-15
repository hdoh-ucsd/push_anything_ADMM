#!/usr/bin/env bash
# Seed-sign probe (pushing-W) — 10 seeds × 2 v_ff arms = 20 runs, 2-parallel.
#
# Purpose (per probe-2 verdict, 2026-05-31):
#   Falsification of "ADMM warm-start carryforward" mechanism for the
#   planner south-bias. ADMM resets delta/omega each tick (admm_solver.py
#   :951-953), so the bias must propagate via a NON-ADMM channel. Probe:
#   does cross-track slide direction CORRELATE with the seed-perturbed
#   initial randomness (first-strategy-sample y-sign), or is it
#   always-south regardless?
#
# Three possible verdicts (decided post-hoc from cross-tab IC vs outcome):
#   (i)  cross-track sign CORRELATES with first-sample y-sign
#        -> IC-driven symmetry-breaking; fix = make planner robust to IC sign
#   (ii) cross-track ALWAYS south regardless of IC sign
#        -> hidden asymmetry; find the always-south channel
#        (sign convention, argmax tie-breaking, t2 = nhat x t1 orientation)
#   (iii) cross-track sign RANDOM, uncorrelated with IC
#        -> neither — solver-nondeterminism / non-convergence symptom;
#        connects to the unresolved ADMM 25/25 pr~4.9 dr~80 wall
#
# Binary checked:
#   HEAD = 37418f6 — v_ff structural fix committed (v_des from x_seq[1])
#   This is the FIX (not the buggy p_ee_des differencing path);
#   tree clean, no uncommitted modifications to swept code.
#
# Runtime:
#   max_time 3.5s — slide settles AFTER push event; 2.5s cuts settle
#   N=10 seeds — power for binary falsification at p<0.01
#   2-parallel — 4-wide previously OOM'd tmpfs
set -uo pipefail

REPO=/root/push_anything_ADMM
cd "$REPO"

if [ -z "${PROBE_OUT:-}" ]; then
    echo "ERROR: PROBE_OUT not set" >&2
    exit 2
fi
OUT="$PROBE_OUT"
mkdir -p "$OUT"

MAX_TIME=3.5
ADMM_ITER=25
PER_RUN_TIMEOUT=900
PARALLEL=2
PY=/root/miniconda3/envs/push_anything_ADMM/bin/python
[ -x "$PY" ] || { echo "missing python interpreter: $PY"; exit 4; }

YAML_OFF=config/sampling_c3_kik.yaml
YAML_ON=audit_output/phaseC_gate_runs/vff_ab/vff_alpha_0.5.yaml

# Pre-flight: confirm yamls exist and HEAD is the structural fix.
[ -f "$YAML_OFF" ] || { echo "missing $YAML_OFF"; exit 3; }
[ -f "$YAML_ON"  ] || { echo "missing $YAML_ON";  exit 3; }
HEAD=$(git -C "$REPO" rev-parse HEAD)
SHORT=$(git -C "$REPO" rev-parse --short HEAD)
EXPECT=37418f67d296c9f4fba5e91222a871196ff24e56
if [ "$HEAD" != "$EXPECT" ]; then
    echo "WARN: HEAD=$HEAD (short=$SHORT) is NOT the v_des structural fix ($EXPECT)" >&2
    echo "      Probe sign may be the aliasing artifact, not planner intent." >&2
fi
echo "HEAD=$HEAD (short=$SHORT)"   | tee "$OUT/HEAD.txt"
echo "YAML_OFF=$YAML_OFF"           | tee -a "$OUT/HEAD.txt"
echo "YAML_ON =$YAML_ON"            | tee -a "$OUT/HEAD.txt"
git -C "$REPO" status --short        | tee -a "$OUT/HEAD.txt"

run_one() {
    local seed="$1"
    local arm="$2"        # off | alpha05
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

JOBS_FILE="$OUT/jobs.txt"
: > "$JOBS_FILE"
for seed in 0 1 2 3 4 5 6 7 8 9; do
    for arm in off alpha05; do
        printf "%s %s\n" "$seed" "$arm" >> "$JOBS_FILE"
    done
done

echo
echo "=== launching ${PARALLEL}-parallel sweep: $(wc -l < "$JOBS_FILE") runs ==="
xargs -P "$PARALLEL" -L 1 -a "$JOBS_FILE" bash -c 'run_one "$@"' _ \
    | tee "$OUT/sweep.summary"

echo
echo "=== sweep complete: $OUT ==="
ls "$OUT" | sort | head -40
