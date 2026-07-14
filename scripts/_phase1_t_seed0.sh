#!/usr/bin/env bash
# Reproduce-dairlib Phase 1 — T seed=0 diagnostic.
#
# Advisory only per scope §3 Phase 1: "expect partial — the T's
# reposition/admission still diverge — but the *executor* behavior
# should be reference-shaped." Success gate for the T translation
# (position<0.02m AND orient<0.1rad) is a Phase-3 target, not Phase-1.
#
# Phase-0 anchor (partial, commit de14138):
#   469/800 steps at internal timeout=1800s. T never translated
#   (obj drift < 6mm). Full run at this stack was ~53 min wall.
#
# Phase-1 expectation:
#   OSC qp_failures == 0 (must hold).
#   Some c3-mode contact (Phase-0 saw 9 c3 steps).
#   T likely does not translate (goal_dist ≥ 0.15m). Diagnostic only.
set -uo pipefail
cd "$(dirname "$0")/.."

OUT="results/_phase1_baseline/t_seed0"
mkdir -p "$OUT"
PY="${PY:-/root/miniconda3/envs/push_anything_ADMM/bin/python}"

{
  echo "HEAD=$(git rev-parse HEAD)"
  echo "branch=$(git branch --show-current)"
  echo "date=$(date -Iseconds)"
  echo "invocation=Phase-1 T seed=0 (Cartesian-force OSC default; phase-2 sweep bundle minus stale flag; math-diag OFF for speed)"
} > "$OUT/manifest.txt"

unset PUSHA_OSC_C3_MODE_REFERENCE_GAINS PUSHA_OSC_C3_MODE_LEGACY_GAINS

T0=$(date +%s)
PUSHA_G_WEIGHT_EE_BOX_FINAL=1 \
PUSHA_STAGE5_U_HORIZONTAL=50 \
PUSHA_STAGE5_U_VERTICAL=3 \
PUSHA_STAGE5_R_VECTOR=0.1,0.1,10 \
LCS_ALWAYS_ON_EE_BOX=1 \
PUSHA_FORCE_ROUTING=u_sol \
PUSHA_EE_APPROACH_FACE_TARGET=1 \
PUSHA_DISABLE_C3_OVERRIDE=1 \
timeout 3900 "$PY" -u main.py push_t \
    --solver c3plus --c3plus-projection lcp \
    --ee-space \
    --sampling-c3 config/sampling_c3_kik_t.yaml \
    --admm-iter 25 \
    --max-time 8 \
    --seed 0 \
    --no-record \
    > "$OUT/run.log" 2>&1
RC=$?
T1=$(date +%s)
WALL=$((T1 - T0))
echo "rc=$RC wall=${WALL}s"

grep -E '^\[RESULT\]' "$OUT/run.log" | tail -1 > "$OUT/result.txt" || true
grep -E '^\[OSC-SUMMARY\]' "$OUT/run.log" | tail -1 > "$OUT/osc_summary.txt" || true
tail -30 "$OUT/run.log" > "$OUT/tail.txt"
echo "---result---"
cat "$OUT/result.txt" "$OUT/osc_summary.txt"
