#!/usr/bin/env bash
# =============================================================================
# P2 tripwire — reconcile-default WIP no-op check
#
# Gate for the reconcile-default commit (~318+/142- across 8 files).
# The refactor collapses REF_RECONCILE_APPROACH env-gated behavior into the
# default path. Tripwire = "does this silently change default behavior?"
#
# Runs box W seed 0 + T seed 0 on the CURRENT WIP TREE and reports:
#   Box:  goal_dist -> closure %.  Target ~75.5% (b23fa82-bit-identical per
#         memory).  HARD FAIL if <40% (silent regression).
#   T:    F_on_box max + nonzero-frac + box_z max.  Target: same tossing
#         signature (F_on_box mostly 0, obj tosses).  HARD FAIL if T
#         accidentally starts working (F_on_box sustained AND box_z quiet).
#
# Post-run: prints TRIPWIRE_VERDICT line; the human decides GREEN/RED.
#
# Usage: ./scripts/_p2_tripwire.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="results/_p2_tripwire_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

echo "[P2-TRIPWIRE] out_dir=$OUT_DIR"
echo "[P2-TRIPWIRE] HEAD=$(git rev-parse HEAD)"
echo "[P2-TRIPWIRE] tree_dirty=$(git diff --stat HEAD | tail -1)"

# ---- Box: seed 0 direction W, ~30 s wall (max-time 6, at ~5x RT) ----
echo
echo "[P2-TRIPWIRE] === BOX seed 0 W ==="
BOX_STEM="p2_tripwire_box"
BOX_LOG="$OUT_DIR/${BOX_STEM}.txt"

# Use env bundle mirroring run_box.sh (the new WIP canonical). --no-record
# added to skip video for tripwire speed.
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
    --math-diag 2>&1 | tee "$BOX_LOG"

# ---- T: seed 0, ~40 s wall (max-time 8) ----
echo
echo "[P2-TRIPWIRE] === T (push_t) seed 0 ==="
T_STEM="p2_tripwire_t"
T_LOG="$OUT_DIR/${T_STEM}.txt"

PUSHA_G_WEIGHT_EE_BOX_FINAL=1 \
PUSHA_OSC_C3_MODE_REFERENCE_GAINS=1 \
PUSHA_STAGE5_U_HORIZONTAL=50 \
PUSHA_STAGE5_U_VERTICAL=3 \
PUSHA_STAGE5_R_VECTOR=0.1,0.1,10 \
LCS_ALWAYS_ON_EE_BOX=1 \
PUSHA_FORCE_ROUTING=u_sol \
PUSHA_EE_APPROACH_FACE_TARGET=1 \
PUSHA_DISABLE_C3_OVERRIDE=1 \
python main.py push_t \
    --solver c3plus --c3plus-projection lcp \
    --ee-space \
    --sampling-c3 config/sampling_c3_kik_t.yaml \
    --admm-iter 25 \
    --max-time 8 \
    --seed 0 \
    --no-record \
    --math-diag 2>&1 | tee "$T_LOG"

# ---- Extract stats ----
echo
echo "[P2-TRIPWIRE] === EXTRACT ==="

python3 - <<PY
import re, math, sys
from pathlib import Path

box_log = Path("$BOX_LOG")
t_log   = Path("$T_LOG")

def extract_result(log_path):
    """Return final goal_dist (m) from the [RESULT] line, or None."""
    m = re.search(r"\[RESULT\].*?goal_dist=([\d.]+)m", log_path.read_text(), re.M)
    return float(m.group(1)) if m else None

def extract_f_on_box(log_path):
    """Return (max_mag, nonzero_frac, n_ticks) across all [GATE-CONTACT] lines."""
    txt = log_path.read_text()
    mags = []
    for m in re.finditer(
        r"\[GATE-CONTACT\].*?F_on_box=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\)", txt):
        fx, fy, fz = float(m.group(1)), float(m.group(2)), float(m.group(3))
        mags.append(math.sqrt(fx*fx + fy*fy + fz*fz))
    if not mags:
        return None
    max_mag = max(mags)
    nz = sum(1 for v in mags if v > 0.1)  # >0.1 N as "nonzero"
    return max_mag, nz / len(mags), len(mags)

def extract_box_z_max(log_path):
    """Max box z from [GATE-CONTACT] box_p= lines (toss detector)."""
    txt = log_path.read_text()
    zs = []
    for m in re.finditer(
        r"\[GATE-CONTACT\].*?box_p=\([-+\d.]+,[-+\d.]+,([-+\d.]+)\)", txt):
        zs.append(float(m.group(1)))
    return max(zs) if zs else None

# --- BOX ---
print()
print("--- BOX (seed 0 W) ---")
box_gd = extract_result(box_log)
if box_gd is None:
    print("BOX_RESULT: MISSING [RESULT] line — run may have crashed. Inspect log.")
    box_closure = None
else:
    box_closure = (0.30 - box_gd) / 0.30 * 100  # initial ~0.30 m
    print(f"BOX_goal_dist: {box_gd:.4f} m")
    print(f"BOX_closure:   {box_closure:.1f} %  (target ~75.5%, HARD FAIL <40%)")

box_f = extract_f_on_box(box_log)
if box_f:
    fmax, fnz, n = box_f
    print(f"BOX_F_on_box:  max={fmax:.2f} N  nonzero_frac={fnz*100:.1f}%  n_ticks={n}")

# --- T ---
print()
print("--- T (push_t seed 0) ---")
t_gd = extract_result(t_log)
if t_gd is not None:
    print(f"T_goal_dist:   {t_gd:.4f} m  (T is yaw-target; goal_dist secondary)")

t_f = extract_f_on_box(t_log)
if t_f:
    fmax, fnz, n = t_f
    print(f"T_F_on_box:    max={fmax:.2f} N  nonzero_frac={fnz*100:.1f}%  n_ticks={n}")
    T_TOSS = (fnz < 0.05 or fmax < 1.0)  # "arm tosses" ~ near-zero contact
    print(f"T_toss_signature_preserved: {'YES' if T_TOSS else 'NO — check for accidental improvement'}")

t_zmax = extract_box_z_max(t_log)
if t_zmax is not None:
    print(f"T_box_z_max:   {t_zmax:.4f} m  (rest ~0.020; toss => rise)")

# --- Verdict ---
print()
print("--- TRIPWIRE_VERDICT ---")
verdict_notes = []
if box_closure is None:
    verdict_notes.append("BOX: RUN FAILED (no [RESULT])")
elif box_closure < 40:
    verdict_notes.append(f"BOX: HARD FAIL closure={box_closure:.1f}% (<40%)")
elif box_closure < 65:
    verdict_notes.append(f"BOX: SOFT FAIL closure={box_closure:.1f}% (<65%, sub-baseline)")
else:
    verdict_notes.append(f"BOX: OK closure={box_closure:.1f}%")

if t_f:
    fmax, fnz, n = t_f
    if fnz > 0.30 and fmax > 5.0:
        verdict_notes.append(f"T: ACCIDENTAL IMPROVEMENT? F_on_box nz={fnz*100:.0f}% max={fmax:.1f}N — investigate")
    else:
        verdict_notes.append(f"T: toss signature preserved (F max={fmax:.1f}N nz={fnz*100:.1f}%)")

print("TRIPWIRE_VERDICT: " + " | ".join(verdict_notes))
PY

echo
echo "[P2-TRIPWIRE] Logs: $OUT_DIR"
echo "[P2-TRIPWIRE] Report the TRIPWIRE_VERDICT line above to the user."
