#!/usr/bin/env bash
# Probe: PYTHONHASHSEED=0 + same Ipopt det knobs as nondet_seed0_ipopt_det.
# Decisive about source #2 (cross-batch 2-3mm IK-settle offset at step 105):
#   - All 4 byte-identical -> source #2 WAS Python dict-iteration order
#     feeding Ipopt constraint construction. Done.
#   - Still split (within-batch identical, cross-batch differing) -> source #2
#     is unseeded RNG (audit reposition_ik.py:769 warm-up call) or ASLR.
# ALSO reports WHICH BASIN: 0.091 (good, pins-to-best, replaces filmed
# 0.177 lucky draw) vs 0.22 (deterministic-but-bad, basin-preference is
# a separate question).
#
# Single back-to-back batched launch (parallelism 2, no spawn-time delays).
set -uo pipefail
OUTBASE=/root/push_anything_ADMM/nondet_seed0_ipopt_det_hashseed
PY=${PY:-/root/miniconda3/envs/push_anything_ADMM/bin/python}
YAML=/tmp/sampling_c3_kik_518bcfa.yaml

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
echo "PYTHONHASHSEED=0 + IK Ipopt det: OMP/MKL/OPENBLAS=1 + linear_solver=spral + acceptable_tol=tol=1e-8 + bound_relax_factor=0" \
  > "$OUTBASE/determinism_knobs.txt"

# Same Ipopt-det env vars + NEW: pin Python hash seed.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export PYTHONHASHSEED=0

run_one() {
  local I="$1"
  local OUT="$OUTBASE/run${I}"
  mkdir -p "$OUT"
  local START
  START=$(date +%s)
  echo "=== run $I start $START effective_HEAD=518bcfa + ipopt_det + PYTHONHASHSEED=0 seed=0 ===" \
    >> "$OUT/run.log"
  echo "ENV: OMP_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$MKL_NUM_THREADS OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS PYTHONHASHSEED=$PYTHONHASHSEED" \
    >> "$OUT/run.log"
  cd /root/push_anything_ADMM
  "$PY" -u main.py pushing \
      --task-id 4 \
      --solver c3plus --c3plus-projection lcp --ee-space \
      --sampling-c3 "$YAML" \
      --admm-iter 25 \
      --max-time 16 \
      --no-record \
      --name "nondet_ipoptdet_hashseed_run${I}_seed0" \
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

# Two waited pairs, back-to-back (no delay between batches).
run_one 1 & P1=$!
run_one 2 & P2=$!
wait "$P1" "$P2"

run_one 3 & P3=$!
run_one 4 & P4=$!
wait "$P3" "$P4"

# Aggregate + decisive verdict.
{
  echo "=== nondet_seed0_ipopt_det_hashseed N=4 ==="
  echo "main_tree_HEAD=$(cat $OUTBASE/main_tree_HEAD.txt)"
  echo "effective_HEAD=518bcfa + ipopt_det + PYTHONHASHSEED=0 (YAML: $YAML)"
  echo "knobs: $(cat $OUTBASE/determinism_knobs.txt)"
  echo ""
  echo "run  final_obj_xy                 goal_dist  oy_drift_signed   gate_pre  gate_post  first_c3  fv@105    fv@108"
  echo "---  ----------                   ---------  ----------------  --------  ---------  --------  --------  --------"
  for I in 1 2 3 4; do
    LOG="$OUTBASE/run${I}/run.log"
    [ -f "$LOG" ] || { echo "$I  MISSING"; continue; }
    R=$(grep -E '^\[RESULT\]' "$LOG" | tail -1)
    if [ -z "$R" ]; then printf '%-3s  TRUNCATED (no [RESULT])\n' "$I"; continue; fi
    OXY=$(echo "$R" | grep -oE 'final_obj_xy=\([^)]+\)' | tr -d '()' | sed 's/final_obj_xy=//')
    GD=$(echo "$R" | grep -oE 'goal_dist=[-0-9.]+m' | tr -d 'm' | sed 's/goal_dist=//')
    OY=$(echo "$OXY" | awk -F, '{print $2}' | tr -d ' ')
    NGP=$(grep -c '^\[GATE-COMMIT-FACE\] ' "$LOG" 2>/dev/null || echo 0)
    NGPOST=$(grep -c '^\[GATE-COMMIT-FACE-POST\] ' "$LOG" 2>/dev/null || echo 0)
    FC3=$(grep -oE '^\[GS\] step=[0-9]+ mode=c3' "$LOG" | head -1 | grep -oE 'step=[0-9]+')
    FV105=$(grep -E "^\[STEP\] step=105 " "$LOG" | head -1 | grep -oE 'finished_val=[0-9.]+m' | tr -d 'm' | sed 's/finished_val=//')
    FV108=$(grep -E "^\[STEP\] step=108 " "$LOG" | head -1 | grep -oE 'finished_val=[0-9.]+m' | tr -d 'm' | sed 's/finished_val=//')
    printf '%-3s  %-26s  %-9s  %-16s  %-8s  %-9s  %-8s  %-8s  %-8s\n' \
      "$I" "$OXY" "$GD" "$OY" "$NGP" "$NGPOST" "${FC3:--}" "${FV105:--}" "${FV108:--}"
  done

  echo ""
  echo "=== Pairwise byte-identity on [STEP] lines (source-#2 decisive test) ==="
  for PAIR in "1 2" "1 3" "1 4" "2 3" "2 4" "3 4"; do
    A=$(echo $PAIR | awk '{print $1}'); B=$(echo $PAIR | awk '{print $2}')
    LA=$OUTBASE/run${A}/run.log; LB=$OUTBASE/run${B}/run.log
    [ -f "$LA" ] && [ -f "$LB" ] || continue
    NDIFF=$(diff <(grep -E '^\[STEP\] step=' "$LA") <(grep -E '^\[STEP\] step=' "$LB") | wc -l)
    echo "  run${A} vs run${B}: ${NDIFF} differing [STEP] lines"
  done
  echo ""
  python3 - <<PY
import re, pathlib
gds, fcs, fv105s = [], [], []
for I in (1,2,3,4):
    log = pathlib.Path(f"$OUTBASE/run{I}/run.log")
    if not log.exists(): continue
    text = log.read_text(errors='replace')
    m = re.search(r'\[RESULT\].*goal_dist=([-0-9.]+)m', text)
    if m: gds.append(float(m.group(1)))
    m = re.search(r'\[GS\] step=(\d+) mode=c3', text)
    if m: fcs.append(int(m.group(1)))
    m = re.search(r'\[STEP\] step=105 .*finished_val=([0-9.]+)m', text)
    if m: fv105s.append(float(m.group(1)))

print("=== verdict ===")
if gds:
    spread = max(gds) - min(gds)
    print(f"N={len(gds)}  goal_dists={[f'{g:.4f}' for g in gds]}")
    print(f"spread = {spread*1000:.1f}mm  (prior ipopt_det: 149.6mm  prior baseline: 52.6mm)")
if fv105s:
    fv_spread = max(fv105s) - min(fv105s)
    print(f"finished_val@step105 = {[f'{1000*v:.2f}mm' for v in fv105s]}")
    print(f"fv@105 spread = {fv_spread*1000:.2f}mm  (prior ipopt_det: ~2.4mm)")
print(f"first_c3 = {fcs}")

# Decisive verdict on source #2.
if gds and len(gds) == 4 and len(set(f'{g:.4f}' for g in gds)) == 1:
    print("SOURCE_#2: CLOSED — all 4 runs settle at the same goal_dist.")
    if gds[0] < 0.15:
        print("BASIN: GOOD (~0.09) — controller pinned to BEST seed-0 result deterministically.")
        print("       The filmed 0.177 lucky-draw is REPLACED by a deterministic better outcome.")
    else:
        print("BASIN: BAD (~0.22) — determinism achieved but locked into the bad basin.")
        print("       Basin-preference is a separate question (c3-entry timing sensitivity).")
elif gds:
    print("SOURCE_#2: STILL SPLIT — cross-pair divergence persists with PYTHONHASHSEED=0.")
    print("           PYTHONHASHSEED was NOT the source; remaining candidates:")
    print("           (a) unseeded RNG in IK warm-up (reposition_ik.py:769)")
    print("           (b) ASLR / memory-layout-dependent FP path")
    print("           Next probe: audit :769 for unseeded std::mt19937 / random / np global state.")
PY
} | tee "$OUTBASE/SUMMARY.txt"
echo "FINISHED $(date +%s)" >> "$OUTBASE/launch.log"
