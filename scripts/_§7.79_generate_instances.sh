#!/usr/bin/env bash
# §7.79 instance generation — grows contacts via LCS_EXPLICIT_BOX_GND to
# capture real-solve LCS dumps at higher n_λ.
#
# Port constraint: LCS_EXPLICIT_BOX_GND accepts only {0, 4, 8, 12}
# (lcs_formulator.py:397). Combined with LCS_ALWAYS_ON_EE_BOX=1, achievable
# n_λ = 6 (no explicit) → 30 (=4) → 54 (=8) → 78 (=12). Six-slots-per-contact
# means the user's originally proposed n_λ ∈ {6, 12, 18} is not reachable
# without a control/ change (LCS_EXPLICIT_BOX_GND doesn't accept 1 or 2).
# This script uses the port's actual grid.
#
# PUSHA_ADMM_DUMP_AT=1 triggers dump at the FIRST c3-mode full solve
# (admm_iter >= 20). --max-time 3 keeps each run short — the dump fires
# around c3-tick 135 in the canonical push (t=1.35s).
#
# Usage: bash scripts/_§7.79_generate_instances.sh <n_grid>
#        where <n_grid> ∈ {4, 8, 12}
# Output: §7.79_instances/n_lambda_<n_λ>_<contacts>contact.npz

set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -ne 1 ]; then
    echo "Usage: $0 <LCS_EXPLICIT_BOX_GND value: 4, 8, or 12>"
    exit 2
fi
NGRID=$1
case "$NGRID" in
    4|8|12) : ;;
    *) echo "Error: LCS_EXPLICIT_BOX_GND must be 4, 8, or 12."; exit 2 ;;
esac

# Predicted n_λ: 6 (EE-BOX) + 6*NGRID (box-gnd contacts)
NLAMBDA=$((6 + 6 * NGRID))
NCONTACTS=$((1 + NGRID))
OUTNPZ="§7.79_instances/n_lambda_${NLAMBDA}_${NCONTACTS}contact.npz"
LOG="§7.79_instances/gen_${NLAMBDA}.log"

echo "[GEN] Target n_λ=${NLAMBDA} ($NCONTACTS contacts)  output: $OUTNPZ"
echo "[GEN] Log: $LOG"

# Match run_canonical.sh env flags + LCS_EXPLICIT_BOX_GND + PUSHA_ADMM_DUMP
PUSHA_G_WEIGHT_EE_BOX_FINAL=1 \
PUSHA_OSC_C3_MODE_REFERENCE_GAINS=1 \
PUSHA_DECOUPLE_RECONCILE_FORCE_TRACKING=1 \
PUSHA_STAGE5_U_HORIZONTAL=50 \
PUSHA_STAGE5_U_VERTICAL=3 \
PUSHA_STAGE5_R_VECTOR=0.1,0.1,10 \
REF_RECONCILE_APPROACH=1 \
LCS_ALWAYS_ON_EE_BOX=1 \
LCS_EXPLICIT_BOX_GND="$NGRID" \
PUSHA_FORCE_ROUTING=u_sol \
PUSHA_EE_APPROACH_FACE_TARGET=1 \
PUSHA_DISABLE_C3_OVERRIDE=1 \
PUSHA_ADMM_DUMP="$OUTNPZ" \
PUSHA_ADMM_DUMP_AT=1 \
PUSHA_ADMM_DUMP_MIN_ITER=20 \
python main.py pushing \
    --task-id 4 \
    --solver c3plus --c3plus-projection lcp \
    --ee-space \
    --sampling-c3 config/sampling_c3_kik.yaml \
    --admm-iter 25 \
    --max-time 3 \
    --seed 0 \
    --no-record 2>&1 | tee "$LOG"

if [ -f "$OUTNPZ" ]; then
    echo "[GEN] SUCCESS: dump written to $OUTNPZ"
    python -c "
import numpy as np
d = np.load('$OUTNPZ', allow_pickle=True)
F = d['F']
Jn = d['J_n']
print(f'  actual n_λ = {F.shape[0]}  (predicted {${NLAMBDA}})')
print(f'  actual num_normals = {Jn.shape[0]}')
print(f'  F symmetric = {np.allclose(F, F.T)}')
"
else
    echo "[GEN] FAILURE: dump NOT written. Likely c3-mode did not fire within max-time=3s."
    echo "[GEN] Grep the log for c3-mode entry or errors."
    exit 3
fi
