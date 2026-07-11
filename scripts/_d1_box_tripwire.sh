#!/usr/bin/env bash
# =============================================================================
# d.1 box tripwire — planner-LCS reference pair-admission (box no-regression)
#
# The d.1 change (T-only flag `use_reference_pair_admission_planner_lcs=true`
# in the T yaml + tshape gate at the use site in lcs_formulator.py's
# linearize_discrete_ee_space) opts the T into the pre-specified-pairs
# admission mechanism. The box yaml doesn't set the flag → default False →
# planner still uses the 2 mm auto-discovery.
#
# Tripwire: `run_box.sh W` seed 0 --max-time 6 early-exit-0.085.
# Bar: closure 68-75% (matches P2 tripwire measurement of 71.6% ± 3%).
#   GREEN → box behavior neutral → commit is safe.
#   RED   → box regressed → do NOT commit, investigate leak.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="results/_d1_box_tripwire_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/d1_box_W_seed0.txt"

echo "[d.1-BOX-TRIPWIRE] out_dir=$OUT_DIR"
echo "[d.1-BOX-TRIPWIRE] HEAD=$(git rev-parse HEAD)"
echo "[d.1-BOX-TRIPWIRE] tree_dirty=$(git diff --stat HEAD | tail -1)"
echo "[d.1-BOX-TRIPWIRE] target closure: 68-75% (P2 measured 71.6%)"

echo
echo "[d.1-BOX-TRIPWIRE] === BOX seed 0 W ==="

PUSHA_G_WEIGHT_EE_BOX_FINAL=1 \
PUSHA_OSC_C3_MODE_REFERENCE_GAINS=1 \
PUSHA_STAGE5_U_HORIZONTAL=50 \
PUSHA_STAGE5_U_VERTICAL=3 \
PUSHA_STAGE5_R_VECTOR=0.1,0.1,10 \
LCS_ALWAYS_ON_EE_BOX=1 \
PUSHA_FORCE_ROUTING=u_sol \
PUSHA_EE_APPROACH_FACE_TARGET=1 \
PUSHA_DISABLE_C3_OVERRIDE=1 \
python main.py pushing \
    --task-id 4 \
    --solver c3plus --c3plus-projection lcp \
    --ee-space \
    --sampling-c3 config/sampling_c3_kik.yaml \
    --admm-iter 25 \
    --max-time 6 \
    --seed 0 \
    --early-exit-goal-d 0.085 \
    --goal-settle-time 0.5 \
    --no-record \
    --math-diag 2>&1 | tee "$LOG"

echo
echo "[d.1-BOX-TRIPWIRE] === EXTRACT ==="
python3 - <<PY
import re
from pathlib import Path
log = Path("$LOG").read_text()
m = re.search(r"\[RESULT\].*?goal_dist=([\d.]+)m", log)
if m:
    gd = float(m.group(1))
    closure = (0.30 - gd) / 0.30 * 100
    print(f"box_goal_dist: {gd:.4f} m")
    print(f"box_closure:   {closure:.1f} %  (target 68-75%, P2 71.6%)")
    if 68 <= closure <= 75:
        print("VERDICT: GREEN (box behavior neutral) → commit is safe")
    elif closure < 68:
        print(f"VERDICT: RED (regression {closure:.1f}% < 68%) → do NOT commit, investigate")
    else:
        print(f"VERDICT: SUSPICIOUS (improvement {closure:.1f}% > 75%) → flag, likely still commit but note")
else:
    print("VERDICT: FAIL — no [RESULT] line found. Sim crashed.")
PY

echo
echo "[d.1-BOX-TRIPWIRE] Log: $LOG"
