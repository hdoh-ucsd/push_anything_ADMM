#!/usr/bin/env bash
# =============================================================================
# T1a effect check — EE_z altitude gate dispatch-discipline
#
# Pass bar: NO c3 dispatch from free occurs with ee_z > sampling_z +
# c3_min_clearance. For T config (sampling_c3_kik_t.yaml): sampling_z=0.034,
# c3_min_clearance=0.01 → ceiling 44 mm.
#
# Effect check has TWO parts:
#   (a) [EEZ-GATE] block-log fires at least once — proves gate is exercised.
#   (b) Every [GS] switch=kToC3ReachedReposTarget or switch=kToC3Cost from
#       mode=free is co-emitted with ee_z <= 44 mm — proves discipline.
#
# Runs push_t seed 0 --max-time 8 (same as P2 tripwire T leg). ~1 hour wall.
#
# Usage: ./scripts/_t1a_effect_check.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="results/_t1a_effect_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/t1a_t_seed0.txt"

echo "[T1A-EFFECT] out_dir=$OUT_DIR"
echo "[T1A-EFFECT] HEAD=$(git rev-parse HEAD)"
echo "[T1A-EFFECT] tree_dirty=$(git diff --stat HEAD | tail -1)"

echo
echo "[T1A-EFFECT] === T (push_t) seed 0 with EE_z gate ==="

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
    --math-diag 2>&1 | tee "$LOG"

echo
echo "[T1A-EFFECT] === EXTRACT ==="

python3 - <<PY
import re
from pathlib import Path

log = Path("$LOG").read_text()

# Config: sampling_z from T yaml (0.034), c3_min_clearance from params (0.01)
SAMPLING_Z = 0.034
CLEARANCE  = 0.01
CEILING    = SAMPLING_Z + CLEARANCE   # 0.044 m

# (a) EEZ-GATE block-log fires
eez_lines = re.findall(r"\[EEZ-GATE\] step=(\d+) .*?ee_z=([\d.]+)mm", log)
print(f"[EEZ-GATE] block-log fires: {len(eez_lines)}")
if eez_lines:
    first_step = eez_lines[0][0]
    max_ee_z = max(float(z) for _, z in eez_lines)
    print(f"  first fire: step={first_step}, max blocked ee_z: {max_ee_z:.1f}mm "
          f"(ceiling {CEILING*1000:.1f}mm)")

# (b) Cross-check: any c3 dispatch from free with ee_z > ceiling?
# Match [GS] lines with switch=kToC3* AND read the co-emitted ee_z from
# the corresponding [STEP] line at the same step.
gs_c3_switches = []
for m in re.finditer(
    r"\[GS\] step=(\d+) mode=c3 switch=(kToC3[A-Za-z]+)", log):
    gs_c3_switches.append((int(m.group(1)), m.group(2)))

# For each c3-entry event, find the [STEP] line at that step and read ee_z.
step_ee_z = {}
for m in re.finditer(
    r"\[STEP\] step=(\d+) mode=\w+ t=[\d.]+s ee=\([^)]*,([^)]*)\)", log):
    # ee=(x, y, z) — grab z (last of the 3 numbers)
    step = int(m.group(1))
    ee_z_str = m.group(2).strip().split(",")[-1] if "," in m.group(2) else m.group(2)
    try:
        step_ee_z[step] = float(ee_z_str.strip())
    except ValueError:
        pass

# Simpler regex: grab ee=(A,B,C) directly
step_ee_z = {}
for m in re.finditer(
    r"\[STEP\] step=(\d+) .*?ee=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\)", log):
    step_ee_z[int(m.group(1))] = float(m.group(4))

violations = []
for step, reason in gs_c3_switches:
    ee_z = step_ee_z.get(step)
    if ee_z is not None and ee_z > CEILING:
        violations.append((step, reason, ee_z))

print(f"\nc3 dispatch events from free (kToC3*): {len(gs_c3_switches)}")
print(f"  → with ee_z > ceiling ({CEILING*1000:.1f}mm): {len(violations)}")
if violations:
    print("VIOLATIONS (gate discipline BROKEN):")
    for step, reason, ee_z in violations[:10]:
        print(f"    step={step} reason={reason} ee_z={ee_z*1000:.1f}mm")

# --- Contact stats (for context; not the pass bar) ---
import math
mags, ticks = [], 0
for m in re.finditer(
    r"\[GATE-CONTACT\].*?F_on_box=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\)", log):
    fx, fy, fz = float(m.group(1)), float(m.group(2)), float(m.group(3))
    mags.append(math.sqrt(fx*fx + fy*fy + fz*fz))
    ticks += 1
if mags:
    fmax = max(mags)
    fnz = sum(1 for v in mags if v > 0.1) / len(mags)
    print(f"\nContext: F_on_box max={fmax:.2f}N nonzero_frac={fnz*100:.1f}% n_ticks={ticks}")

# --- Verdict ---
gate_exercised = len(eez_lines) > 0
gate_disciplined = len(violations) == 0
print("\n--- T1A_EFFECT_VERDICT ---")
if gate_exercised and gate_disciplined:
    print(f"PASS: gate fired {len(eez_lines)}x, discipline holds "
          f"({len(gs_c3_switches)} c3 dispatches, 0 above-ceiling)")
elif not gate_exercised and gate_disciplined:
    print(f"UNCLEAR: gate NEVER fired ({len(gs_c3_switches)} c3 dispatches, "
          f"all below ceiling) — either the arm stayed low (no need to fire) "
          f"or dead code. Inspect ee_z altitude trace.")
else:
    print(f"FAIL: violations={len(violations)} (gate discipline BROKEN)")
PY

echo
echo "[T1A-EFFECT] Log: $LOG"
