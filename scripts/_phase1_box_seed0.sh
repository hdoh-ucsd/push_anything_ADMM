#!/usr/bin/env bash
# Reproduce-dairlib Phase 1 — box seed=0 go/no-go gate.
#
# Validates that the Cartesian-force OSC swap (reference c3-gains default +
# joint-2 posture + trajectory-shaped input) does not regress the Phase-0
# box baseline.
#
# Phase-0 anchor (commit de14138):
#   goal_dist=0.1289m (57% closure of 0.3m init), orient_err=1.68 rad,
#   OSC 0 QP failures / 6.16% saturation.
#
# Phase-1 gate:
#   goal_dist <= 0.135m  AND  qp_failures == 0  AND  saturation <= 8%.
# NO-GO on any failure per scope §3 go/no-go; STOP and reassess.
set -uo pipefail
cd "$(dirname "$0")/.."

OUT="results/_phase1_baseline/box_seed0"
mkdir -p "$OUT"
PY="${PY:-/root/miniconda3/envs/push_anything_ADMM/bin/python}"

{
  echo "HEAD=$(git rev-parse HEAD)"
  echo "branch=$(git branch --show-current)"
  echo "date=$(date -Iseconds)"
  echo "invocation=Phase-1 box seed=0 (Cartesian-force OSC default; §7.51 chain minus stale c3-gains flag)"
} > "$OUT/manifest.txt"

unset LCS_NORMAL_PHI_CLAMP PUSHA_REF_OSC_ALIGN
unset PUSHA_OSC_C3_MODE_REFERENCE_GAINS PUSHA_OSC_C3_MODE_LEGACY_GAINS

T0=$(date +%s)
PUSHA_FORCE_ROUTING=u_sol \
PUSHA_STAGE5_U_HORIZONTAL=10 \
PUSHA_STAGE5_U_VERTICAL=3 \
PUSHA_STAGE5_R_VECTOR=0.1,0.1,10 \
PUSHA_EE_APPROACH_FACE_TARGET=1 \
PUSHA_DISABLE_C3_OVERRIDE=1 \
LCS_ALWAYS_ON_EE_BOX=1 \
REF_RECONCILE_APPROACH=1 \
LCS_NORMAL_VELOCITY_LEVEL=0 \
LCS_NORMAL_COMPLIANCE_K=0.0 \
timeout 1800 "$PY" -u main.py pushing \
    --task-id 4 \
    --solver c3plus --c3plus-projection lcp --ee-space \
    --sampling-c3 config/sampling_c3_kik.yaml \
    --admm-iter 25 \
    --max-time 6 \
    --name "phase1_box_seed0" \
    --seed 0 \
    --no-record \
    > "$OUT/run.log" 2>&1
RC=$?
T1=$(date +%s)
WALL=$((T1 - T0))
echo "rc=$RC wall=${WALL}s"

grep -E '^\[RESULT\]' "$OUT/run.log" | tail -1 > "$OUT/result.txt" || true
grep -E '^\[OSC-SUMMARY\]' "$OUT/run.log" | tail -1 > "$OUT/osc_summary.txt" || true
echo "---result---"
cat "$OUT/result.txt" "$OUT/osc_summary.txt"
