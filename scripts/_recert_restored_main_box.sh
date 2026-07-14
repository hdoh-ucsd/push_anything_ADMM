#!/usr/bin/env bash
# Re-certification of restored main: after reverting BOTH ec7fc4a
# (full-stack cost/dynamics contamination) and 18498c1 (gain flip),
# does the box return to the clean ~75.3% baseline?
#
# Env bundle mirrors _p2_tripwire.sh EXACTLY (port canonical values):
# U_HORIZONTAL=50, U_VERTICAL=3, R_VECTOR=0.1,0.1,10, and the reference
# c3-gains flag OPT-IN (matches how the 75.3% baseline was captured on
# 687b8a6). --max-time 8 to hit 75.3% (vs 71.6% at max-time 6).
#
# Kept from Phase 1 (verified as byte-inert additions if the box hits
# 75.3%): joint-2 posture pull (YAML weight 1.0), trajectory-shaped
# input contract (single-knot ZOH PP for c3-mode dispatch).
#
# If box < 75%: one of the kept additions is NOT harmless — flag.
set -uo pipefail
cd "$(dirname "$0")/.."

OUT="results/_recert_restored_main_box_seed0"
mkdir -p "$OUT"
PY="${PY:-/root/miniconda3/envs/push_anything_ADMM/bin/python}"

{
  echo "HEAD=$(git rev-parse HEAD)"
  echo "branch=$(git branch --show-current)"
  echo "date=$(date -Iseconds)"
  echo "invocation=Restored-main box seed=0, p2_tripwire port env bundle, --max-time 8"
  echo "expected_target=~75.3% closure (matches clean 687b8a6 baseline)"
} > "$OUT/manifest.txt"

unset LCS_NORMAL_PHI_CLAMP PUSHA_REF_OSC_ALIGN

T0=$(date +%s)
PUSHA_G_WEIGHT_EE_BOX_FINAL=1 \
PUSHA_OSC_C3_MODE_REFERENCE_GAINS=1 \
PUSHA_STAGE5_U_HORIZONTAL=50 \
PUSHA_STAGE5_U_VERTICAL=3 \
PUSHA_STAGE5_R_VECTOR=0.1,0.1,10 \
LCS_ALWAYS_ON_EE_BOX=1 \
PUSHA_FORCE_ROUTING=u_sol \
PUSHA_EE_APPROACH_FACE_TARGET=1 \
PUSHA_DISABLE_C3_OVERRIDE=1 \
timeout 1800 "$PY" -u main.py pushing \
    --task-id 4 \
    --solver c3plus --c3plus-projection lcp \
    --ee-space \
    --sampling-c3 config/sampling_c3_kik.yaml \
    --admm-iter 25 \
    --max-time 8 \
    --seed 0 \
    --early-exit-goal-d 0.085 \
    --goal-settle-time 0.5 \
    --no-record \
    > "$OUT/run.log" 2>&1
RC=$?
T1=$(date +%s)
WALL=$((T1 - T0))
echo "rc=$RC wall=${WALL}s"

grep -E '^\[RESULT\]' "$OUT/run.log" | tail -1 > "$OUT/result.txt" || true
grep -E '^\[OSC-SUMMARY\]' "$OUT/run.log" | tail -1 > "$OUT/osc_summary.txt" || true
grep -E '^\[STEP\] step=' "$OUT/run.log" | tail -1 > "$OUT/last_step.txt" || true
echo "---result---"
cat "$OUT/result.txt" "$OUT/osc_summary.txt"
