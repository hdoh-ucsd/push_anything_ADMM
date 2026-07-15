#!/usr/bin/env bash
# Non-determinism FIX trial: seed-0 @ 518bcfa-effective + Ipopt determinism pins,
# N=4 reruns. Same protocol as nondet_seed0_518bcfa.
#
# THREE PINS APPLIED:
#  (1) OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1 — single
#      thread BLAS so FP reductions cannot reorder across threads.
#  (2) IPOPT linear_solver=spral (pinned at reposition_ik.py:671; spral is
#      the only solver Drake's Ipopt build ships with).
#  (3) IPOPT acceptable_tol=tol=1e-8 + bound_relax_factor=0 (reposition_ik.py:673-677)
#      so early-acceptance branch and bound-relaxation FP-noise are removed.
#
# Working tree HEAD = 7479985 (gate disabled). The YAML override pins the
# *runtime* effective config to 518bcfa (gate enabled at +0.3 via params.py
# defaults). The ONLY code-level diff vs nondet_seed0_518bcfa is the IK Ipopt
# determinism patch at reposition_ik.py:656-677.
#
# Test: does the 52.6mm same-seed spread COLLAPSE to <5mm?
set -uo pipefail
OUTBASE=/root/push_anything_ADMM/nondet_seed0_ipopt_det
PY=${PY:-/root/miniconda3/envs/push_anything_ADMM/bin/python}
YAML=/tmp/sampling_c3_kik_518bcfa.yaml

# /tmp abort guard FIRST.
TMP_PCT=$(df --output=pcent /tmp 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
if [ "${TMP_PCT:-0}" -ge 85 ]; then
  echo "ABORT: /tmp at ${TMP_PCT}% -- STOP, do NOT rm -rf /tmp/claude-0/*"
  exit 1
fi
[ -r "$YAML" ] || { echo "ABORT: YAML missing at $YAML"; exit 1; }

mkdir -p "$OUTBASE"
cp "$YAML" "$OUTBASE/effective_config.yaml"
git -C /root/push_anything_ADMM rev-parse HEAD > "$OUTBASE/main_tree_HEAD.txt"
echo "518bcfab6520a6387712dfa383e7117b2cb3a845" > "$OUTBASE/effective_HEAD.txt"
echo "IK Ipopt determinism pins: OMP/MKL/OPENBLAS=1 + linear_solver=spral + acceptable_tol=tol=1e-8 + bound_relax_factor=0" \
  > "$OUTBASE/determinism_knobs.txt"

# Single-thread BLAS for all child processes.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores

run_one() {
  local I="$1"
  local OUT="$OUTBASE/run${I}"
  mkdir -p "$OUT"
  local START
  START=$(date +%s)
  echo "=== run $I start $START effective_HEAD=518bcfa + ipopt_det seed=0 ===" \
    >> "$OUT/run.log"
  echo "ENV: OMP_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$MKL_NUM_THREADS OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS" \
    >> "$OUT/run.log"
  cd /root/push_anything_ADMM
  "$PY" -u main.py pushing \
      --task-id 4 \
      --solver c3plus --c3plus-projection lcp --ee-space \
      --sampling-c3 "$YAML" \
      --admm-iter 25 \
      --max-time 16 \
      --no-record \
      --name "nondet_ipoptdet_run${I}_seed0" \
      --seed 0 \
      >> "$OUT/run.log" 2>&1 || true
  local ELAPSED=$(( $(date +%s) - START ))
  local N_GATE_PRE N_GATE_POST N_DRAKE_PAIRS RESULT
  N_GATE_PRE=$(grep -c '^\[GATE-COMMIT-FACE\] ' "$OUT/run.log" 2>/dev/null || true)
  N_GATE_POST=$(grep -c '^\[GATE-COMMIT-FACE-POST\] ' "$OUT/run.log" 2>/dev/null || true)
  N_DRAKE_PAIRS=$(grep -c '^\[DRAKE-CONTACT\] ' "$OUT/run.log" 2>/dev/null || true)
  RESULT=$(grep '^\[RESULT\]' "$OUT/run.log" | tail -1 || true)
  echo "DONE run=$I ${ELAPSED}s gate_pre=$N_GATE_PRE gate_post=$N_GATE_POST drake_pairs_lines=$N_DRAKE_PAIRS | $RESULT" \
    >> "$OUT/run.log"
}

# Batches of 2 (parallelism <= 2).
run_one 1 & P1=$!
run_one 2 & P2=$!
wait "$P1" "$P2"

run_one 3 & P3=$!
run_one 4 & P4=$!
wait "$P3" "$P4"

# Aggregate.
{
  echo "=== nondet_seed0_ipopt_det N=4 ==="
  echo "main_tree_HEAD=$(cat $OUTBASE/main_tree_HEAD.txt)"
  echo "effective_HEAD=518bcfa + ipopt_det (YAML: $YAML)"
  echo "knobs: $(cat $OUTBASE/determinism_knobs.txt)"
  echo ""
  echo "run  final_obj_xy                 goal_dist  oy_drift_signed   gate_pre  gate_post  drake_pairs_emits"
  echo "---  ----------                   ---------  ----------------  --------  ---------  -----------------"
  for I in 1 2 3 4; do
    LOG="$OUTBASE/run${I}/run.log"
    [ -f "$LOG" ] || { echo "$I  MISSING"; continue; }
    R=$(grep -E '^\[RESULT\]' "$LOG" | tail -1)
    if [ -z "$R" ]; then
      printf '%-3s  TRUNCATED (no [RESULT])\n' "$I"
      continue
    fi
    OXY=$(echo "$R" | grep -oE 'final_obj_xy=\([^)]+\)' | tr -d '()' | sed 's/final_obj_xy=//')
    GD=$(echo "$R" | grep -oE 'goal_dist=[-0-9.]+m' | tr -d 'm' | sed 's/goal_dist=//')
    OY=$(echo "$OXY" | awk -F, '{print $2}' | tr -d ' ')
    NGP=$(grep -c '^\[GATE-COMMIT-FACE\] ' "$LOG" 2>/dev/null || echo 0)
    NGPOST=$(grep -c '^\[GATE-COMMIT-FACE-POST\] ' "$LOG" 2>/dev/null || echo 0)
    NDR=$(grep -c '^\[DRAKE-CONTACT\] ' "$LOG" 2>/dev/null || echo 0)
    printf '%-3s  %-26s  %-9s  %-16s  %-8s  %-9s  %-17s\n' \
      "$I" "$OXY" "$GD" "$OY" "$NGP" "$NGPOST" "$NDR"
  done
  echo ""
  # Spread + verdict.
  GDS=$(for I in 1 2 3 4; do
    grep -E '^\[RESULT\]' "$OUTBASE/run${I}/run.log" 2>/dev/null | tail -1 \
      | grep -oE 'goal_dist=[-0-9.]+m' | tr -d 'm' | sed 's/goal_dist=//'
  done)
  echo "goal_dists: $GDS"
  python3 - <<PY
gds = [float(x) for x in """$GDS""".split() if x]
if gds:
    spread = max(gds) - min(gds)
    print(f"N={len(gds)}  min={min(gds):.4f}  max={max(gds):.4f}  spread={spread*1000:.1f}mm")
    prior = 0.0526
    print(f"prior_spread (nondet_seed0_518bcfa N=4): {prior*1000:.1f}mm")
    if spread < 0.005:
        verdict = "COLLAPSED (<5mm) -> root fixed, single-seed comparison VALID -> re-validate prior claims"
    elif spread < 0.020:
        verdict = "REDUCED (<20mm) -> partial fix, route (b') tighter knobs or accept narrower floor"
    elif spread < 0.030:
        verdict = "MILD REDUCTION (>=20mm, <30mm) -> root not fully addressed"
    else:
        verdict = "PERSISTS (>=30mm) -> Ipopt FP-non-determinism irreducible here -> route (c) >=3-seed/>60mm-floor"
    print(f"verdict: {verdict}")
PY
} | tee "$OUTBASE/SUMMARY.txt"
echo "FINISHED $(date +%s)" >> "$OUTBASE/launch.log"
