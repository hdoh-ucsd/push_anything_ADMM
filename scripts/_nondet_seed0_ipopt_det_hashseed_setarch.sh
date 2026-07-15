#!/usr/bin/env bash
# Probe: setarch -R (ADDR_NO_RANDOMIZE) + PYTHONHASHSEED=0 + Ipopt det knobs.
# DECISIVE about source #2 residual (cross-batch 1.20mm fv@105 spread after
# PYTHONHASHSEED+ipopt_det). One-line change: prefix `setarch -R` on the
# python call to disable ASLR for that process.
#   - All 6 pairs == 0 differing [STEP] lines AND fv@105 all-identical
#     -> C++ ASLR (libstdc++ hash-container address-derived seeding, or
#        Drake/Ipopt std::random_device-based init) was source #2. CLOSED.
#   - Any nonzero pair OR any fv@105 split
#     -> PARTIAL. Report the new fv@105 spread (1.20mm -> ?). DO NOT
#        celebrate. Next probe: grep Drake C++ binding tree for
#        RandomGenerator / std::random_device / std::mt19937 in IK / SPRAL.
#
# ALSO reports WHICH BASIN: 0.063 (GOOD, pins-to-best — replaces filmed
# 0.177 lucky draw) vs ~0.22 (BAD basin = wrong-face runaway, 246+ gate
# firings on run3 prior). Even if determinism fully closes, basin matters:
# closed+GOOD = the dream outcome; closed+BAD = c3-entry-timing-bias is
# the separate fix (the deeper hypothesis — IK wobble tipping c3-entry
# one tick early/late is THE lever between clean push vs wrong-face
# runaway).
#
# Single back-to-back batched launch (parallelism 2, no spawn-time delays).
set -uo pipefail
OUTBASE=/root/push_anything_ADMM/nondet_seed0_setarch_hashseed
PY=${PY:-/root/miniconda3/envs/push_anything_ADMM/bin/python}
YAML=/tmp/sampling_c3_kik_518bcfa.yaml

TMP_PCT=$(df --output=pcent /tmp 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
if [ "${TMP_PCT:-0}" -ge 85 ]; then
  echo "ABORT: /tmp at ${TMP_PCT}% -- STOP, do NOT rm -rf /tmp/claude-0/*"
  exit 1
fi
[ -r "$YAML" ] || { echo "ABORT: YAML missing at $YAML"; exit 1; }
command -v setarch >/dev/null 2>&1 || { echo "ABORT: setarch not on PATH"; exit 1; }

mkdir -p "$OUTBASE"
cp "$YAML" "$OUTBASE/effective_config.yaml"
git -C /root/push_anything_ADMM rev-parse HEAD > "$OUTBASE/main_tree_HEAD.txt"
echo "518bcfab6520a6387712dfa383e7117b2cb3a845" > "$OUTBASE/effective_HEAD.txt"
echo "setarch -R (ADDR_NO_RANDOMIZE) + PYTHONHASHSEED=0 + IK Ipopt det: OMP/MKL/OPENBLAS=1 + linear_solver=spral + acceptable_tol=tol=1e-8 + bound_relax_factor=0" \
  > "$OUTBASE/determinism_knobs.txt"

# Same env vars (Ipopt-det + PYTHONHASHSEED) PLUS the new setarch -R prefix on the python call.
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
  echo "=== run $I start $START effective_HEAD=518bcfa + ipopt_det + PYTHONHASHSEED=0 + setarch -R seed=0 ===" \
    >> "$OUT/run.log"
  echo "ENV: OMP_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$MKL_NUM_THREADS OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS PYTHONHASHSEED=$PYTHONHASHSEED" \
    >> "$OUT/run.log"
  echo "WRAP: setarch -R $PY ..." >> "$OUT/run.log"
  cd /root/push_anything_ADMM
  setarch -R "$PY" -u main.py pushing \
      --task-id 4 \
      --solver c3plus --c3plus-projection lcp --ee-space \
      --sampling-c3 "$YAML" \
      --admm-iter 25 \
      --max-time 16 \
      --no-record \
      --name "nondet_setarch_hashseed_run${I}_seed0" \
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

# Aggregate + decisive verdict (with PARTIAL-IS-NOT-CLOSURE bar).
{
  echo "=== nondet_seed0_setarch_hashseed N=4 ==="
  echo "main_tree_HEAD=$(cat $OUTBASE/main_tree_HEAD.txt)"
  echo "effective_HEAD=518bcfa + ipopt_det + PYTHONHASHSEED=0 + setarch -R (YAML: $YAML)"
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
  echo "=== Pairwise byte-identity on [STEP] lines (source-#2 decisive test, FULL 6-pair matrix) ==="
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

print("=== verdict (PARTIAL-IS-NOT-CLOSURE bar) ===")
print("Prior baselines: fv@105 spread  baseline=N/A  ipopt_det=2.40mm  +hashseed=1.20mm")
if gds:
    spread = max(gds) - min(gds)
    print(f"N={len(gds)}  goal_dists={[f'{g:.4f}' for g in gds]}")
    print(f"goal_dist spread = {spread*1000:.1f}mm")
if fv105s:
    fv_spread = max(fv105s) - min(fv105s)
    print(f"finished_val@step105 = {[f'{1000*v:.2f}mm' for v in fv105s]}")
    print(f"fv@105 spread = {fv_spread*1000:.2f}mm  (prior +hashseed: 1.20mm)")
print(f"first_c3 = {fcs}")

# Decisive verdict.
all_fv_equal = fv105s and len(set(f'{v:.6f}' for v in fv105s)) == 1
all_gd_equal = gds and len(set(f'{g:.4f}' for g in gds)) == 1
if all_gd_equal and all_fv_equal:
    print("")
    print("SOURCE_#2: CLOSED — all 4 runs identical at fv@105 AND goal_dist.")
    print("           C++ ASLR (libstdc++ hash-container address-derived seeding")
    print("           OR Drake/Ipopt std::random_device-based init) WAS source #2.")
    if gds[0] < 0.15:
        print("BASIN: GOOD (~0.06) — controller pinned to BEST seed-0 result deterministically.")
        print("       Dream outcome: filmed 0.177 lucky-draw is REPLACED by ~0.06 reproducible.")
        print("       Next: re-validate + pin-best across remaining seeds.")
    else:
        print("BASIN: BAD (~0.22) — determinism achieved BUT locked into the wrong-face runaway.")
        print("       Determinism + basin are SEPARATE problems. Next: c3-entry-timing-bias")
        print("       investigation (the deeper hypothesis — IK wobble tipping entry tick")
        print("       early/late was the good/bad lever; deterministic bad basin means we")
        print("       now need to deliberately bias entry toward the GOOD basin).")
elif gds:
    print("")
    print("SOURCE_#2: STILL SPLIT — PARTIAL collapse only. DO NOT CELEBRATE.")
    print(f"           fv@105 spread: prior 1.20mm -> now {(max(fv105s)-min(fv105s))*1000:.2f}mm" if fv105s else "")
    print("           setarch -R was NOT sufficient alone. Next probe:")
    print("             grep Drake C++ binding tree for explicit RNG sources:")
    print("               RandomGenerator / std::random_device / std::mt19937")
    print("             in IK / SPRAL / collision-query layers.")
    print("           Pattern to avoid (bit twice): over-reading a PARTIAL as closure.")
    # Report basin distribution even on partial.
    n_good = sum(1 for g in gds if g < 0.15)
    n_bad  = sum(1 for g in gds if g >= 0.15)
    print(f"BASIN distribution: good (<0.15) = {n_good}/4   bad (>=0.15) = {n_bad}/4")
PY
} | tee "$OUTBASE/SUMMARY.txt"
echo "FINISHED $(date +%s)" >> "$OUTBASE/launch.log"
