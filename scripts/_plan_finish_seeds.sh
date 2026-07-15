#!/usr/bin/env bash
# =============================================================================
# Plan-finish T4 multi-seed runner (sequential per §8).
# Arg: SEED number (1, 2, or 4).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
SEED="${1:?seed argument required}"
OUT_DIR="results/_conformance_plan_finish"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/T4_seed${SEED}.log"

echo "[T4-seed$SEED] HEAD=$(git rev-parse HEAD)  started=$(date -Is)"

PUSHA_G_WEIGHT_EE_BOX_FINAL=1 \
PUSHA_OSC_C3_MODE_REFERENCE_GAINS=1 \
PUSHA_STAGE5_U_HORIZONTAL=50 \
PUSHA_STAGE5_U_VERTICAL=3 \
PUSHA_STAGE5_R_VECTOR=0.1,0.1,10 \
LCS_ALWAYS_ON_EE_BOX=1 \
PUSHA_FORCE_ROUTING=u_sol \
PUSHA_EE_APPROACH_FACE_TARGET=1 \
PUSHA_DISABLE_C3_OVERRIDE=1 \
PUSHA_C3PLUS_N=5 \
python main.py push_t \
    --solver c3plus --c3plus-projection lcp --ee-space \
    --sampling-c3 config/sampling_c3_kik_t.yaml \
    --admm-iter 25 --max-time 8 --seed "$SEED" \
    --no-record --math-diag 2>&1 | tee "$LOG"

python3 - "$LOG" "$SEED" <<'PY'
import re, sys
from pathlib import Path
log = Path(sys.argv[1]).read_text()
seed = sys.argv[2]
gd = re.search(r"\[RESULT\].*?goal_dist=([\d.]+)m", log)
if gd:
    v = float(gd.group(1))
    passing = v < 0.02
    print(f"[T4-EXTRACT seed={seed}] goal_dist={v:.4f}m (bar <0.02m) → "
          f"{'PASS' if passing else f'FAIL gap={v-0.02:+.4f}m'}")
else:
    print(f"[T4-EXTRACT seed={seed}] no [RESULT] line — sim did not complete")
# Count c3 events
gs = list(re.finditer(r"\[GS\] step=(\d+) mode=(\w+) switch=(\w+)", log))
c3_entries = 0; c3_ticks = 0; prev = "free"
c3_windows = []
in_c3_start = None
for m in gs:
    step = int(m.group(1)); mode = m.group(2)
    if mode == "c3":
        c3_ticks += 1
        if prev == "free":
            c3_entries += 1
            in_c3_start = step
    elif mode == "free" and prev == "c3" and in_c3_start is not None:
        c3_windows.append((in_c3_start, step - 1))
        in_c3_start = None
    prev = mode
if in_c3_start is not None and gs:
    c3_windows.append((in_c3_start, int(gs[-1].group(1))))
longest = max((b-a+1 for a,b in c3_windows), default=0)
print(f"[T4-EXTRACT seed={seed}] c3 events={c3_entries}  total_ticks={c3_ticks}  "
      f"longest_run={longest} ticks ({longest*10}ms)")
PY

echo "[T4-seed$SEED] finished=$(date -Is)"
