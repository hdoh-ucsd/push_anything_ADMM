#!/usr/bin/env bash
# =============================================================================
# d.1 T effect check — planner-LCS reference pair-admission
#
# 3-part mechanism-tied pass bar:
#   (i)   EE-manipuland pair PERSISTS in the planner LCS across the arm's
#         off-face rise (φ = 3, 5, 20 mm — pair still admitted, no drop).
#   (ii)  c3 SUSTAINS (does not exit at 2 mm φ-crossing). Longest c3-mode
#         run ≫ T1c's 180 ms.
#   (iii) T TRANSLATES (goal_dist improves materially).
#
# Fold-in measurement (NOT a pass condition — data for the next fork):
#   Does u_z still SATURATE at +3 N (STAGE5_U_VERTICAL cap)?
#   - If yes → vertical cap is the next conformance gap (reference is ±50 N).
#   - If no  → threshold fix alone sufficed.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="results/_d1_effect_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/d1_t_seed0.txt"

echo "[d.1-EFFECT] out_dir=$OUT_DIR"
echo "[d.1-EFFECT] HEAD=$(git rev-parse HEAD)"
echo "[d.1-EFFECT] tree_dirty=$(git diff --stat HEAD | tail -1)"

echo
echo "[d.1-EFFECT] === T (push_t) seed 0 with planner-LCS ref-pair-admission ==="

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
echo "[d.1-EFFECT] === EXTRACT ==="
python3 - <<PY
import re, math
import numpy as np
from pathlib import Path
log = Path("$LOG").read_text()

# --- c3-mode timeline + windows ---
step_mode = {}
step_reason = {}
for m in re.finditer(r"\[GS\] step=(\d+) mode=(\w+) switch=(\w+)", log):
    step_mode[int(m.group(1))] = m.group(2)
    step_reason[int(m.group(1))] = m.group(3)

steps_c3 = sorted(s for s, m in step_mode.items() if m == "c3")
windows = []
if steps_c3:
    start = prev = steps_c3[0]
    for s in steps_c3[1:]:
        if s == prev + 1:
            prev = s
        else:
            windows.append((start, prev))
            start = prev = s
    windows.append((start, prev))

# --- LCS admission trace: planner λ_n_max nonzero rate ---
# When force_top_k_ee_box takes over, the EE-BOX pair should ALWAYS be
# admitted. Watch [CONTACT-ELEM] for manipulated_object phi values (Drake
# contact scan; independent of the 2 mm auto-admit).
# Also [MATH.δ-C3+] projection counts show N λ-components active.

# --- Pair persistence proxy: [DRAKE-CONTACT] n_pairs across ticks + λ_n ---
step_pairs = {}
for m in re.finditer(r"\[DRAKE-CONTACT\] step=(\d+) n_pairs=(\d+)", log):
    step_pairs[int(m.group(1))] = int(m.group(2))

step_lam_n = {}
for m in re.finditer(r"\[C3\+\] step=(\d+) .*?λ_n_max=([\d.]+)", log):
    step_lam_n[int(m.group(1))] = float(m.group(2))

# --- F on box ---
step_fxyz = {}
for m in re.finditer(
    r"\[GATE-CONTACT\] step=(\d+) .*?F_on_box=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\)", log):
    step_fxyz[int(m.group(1))] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))

# --- box position for goal_dist / translation ---
step_boxxy = {}
for m in re.finditer(
    r"\[GATE-CONTACT\] step=(\d+) .*?box_p=\(([-+\d.]+),([-+\d.]+),[-+\d.]+\)", log):
    step_boxxy[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))

# --- u_z from [C3+] first-knot u_axis ---
step_u_axis = {}
for m in re.finditer(
    r"\[C3\+\] step=(\d+) .*?u_axis=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\)", log):
    step_u_axis[int(m.group(1))] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))

# ---- (i) pair persistence ----
print("\n--- (i) EE-manipuland pair persistence in planner LCS ---")
# Planner λ_n_max nonzero AT EVERY plan solve step means the LCS has EE-manipuland row
lam_ticks = list(step_lam_n.values())
if lam_ticks:
    n_nonzero = sum(1 for l in lam_ticks if l > 0.01)
    print(f"  planner λ_n_max samples: {len(lam_ticks)}")
    print(f"    non-zero (>0.01): {n_nonzero} = {n_nonzero/len(lam_ticks)*100:.1f}%")
    print(f"    T1c baseline non-zero rate proxy: comparable to c3-mode duty (~4-5%)")

# ---- (ii) c3 sustains ----
print("\n--- (ii) c3-mode sustain ---")
if windows:
    print(f"  c3 events: {len(windows)}")
    for i, (s0, s1) in enumerate(windows[:6], 1):
        dur_ticks = s1 - s0 + 1
        exit_reason = "?"
        for k in range(1, 10):
            sk = s1 + k
            if sk in step_mode and step_mode[sk] == "free":
                exit_reason = step_reason.get(sk, "?")
                break
        print(f"    event {i}: steps {s0}..{s1}  ({dur_ticks} ticks, {dur_ticks*10} ms)  exit={exit_reason}")
    if len(windows) > 6:
        print(f"    ... {len(windows)-6} more")
    longest = max(s1-s0+1 for s0, s1 in windows)
    total_c3 = sum(s1-s0+1 for s0, s1 in windows)
    print(f"  longest c3 run: {longest} ticks ({longest*10} ms)  vs T1c 180ms")
    print(f"  total c3 ticks: {total_c3} vs T1c 34 ticks")
else:
    print("  NO c3 events (VERY BAD — investigate)")

# ---- (iii) T translates ----
print("\n--- (iii) T translation ---")
gd = re.search(r"\[RESULT\].*?goal_dist=([\d.]+)m", log)
if gd:
    print(f"  final goal_dist: {gd.group(1)} m  vs T1c 0.1834 m, baseline 0.19 m")

if step_boxxy:
    steps_sorted = sorted(step_boxxy)
    x0, y0 = step_boxxy[steps_sorted[0]]
    xf, yf = step_boxxy[steps_sorted[-1]]
    dxy = math.sqrt((xf-x0)**2 + (yf-y0)**2) * 1000
    print(f"  T net xy displacement: {dxy:.2f} mm  (T1c ~1 mm)")

# ---- Fold-in: u_z saturation ----
print("\n--- FOLD-IN: u_z saturation (vertical-cap coupling) ---")
if step_u_axis:
    uz = np.array([step_u_axis[s][2] for s in sorted(step_u_axis)])
    at_cap_pos = (uz >= 2.95).sum()  # near +3N cap
    at_cap_neg = (uz <= -2.95).sum()  # near -3N cap
    total = len(uz)
    print(f"  u_z samples: {total}")
    print(f"    at +3N cap (>=+2.95): {at_cap_pos} ({at_cap_pos/total*100:.1f}%)")
    print(f"    at -3N cap (<=-2.95): {at_cap_neg} ({at_cap_neg/total*100:.1f}%)")
    print(f"    u_z max: {uz.max():+.2f}N  min: {uz.min():+.2f}N")
    print(f"    Reference u_vertical_limits=[-50,+50] (17x wider)")

# ---- F decomposition for context ----
if step_fxyz:
    fs = list(step_fxyz.values())
    lat_max = max(math.sqrt(f[0]**2 + f[1]**2) for f in fs)
    vert_max = max(abs(f[2]) for f in fs)
    mag_max = max(math.sqrt(f[0]**2 + f[1]**2 + f[2]**2) for f in fs)
    print(f"\n  F_on_box: |F| max {mag_max:.1f}N (F_lat max {lat_max:.1f}N, F_vert max {vert_max:.1f}N)")

# ---- Verdict ----
print("\n--- d.1 VERDICT ---")
notes = []
pair_ok = lam_ticks and (sum(1 for l in lam_ticks if l > 0.01) / len(lam_ticks) > 0.20)
sustain_ok = windows and (max(s1-s0+1 for s0, s1 in windows) > 40)  # >400ms = big improvement over 180ms
translate_ok = False
if gd:
    gd_val = float(gd.group(1))
    translate_ok = gd_val < 0.16  # meaningful translation
notes.append(f"pair persists (planner λ_n nonzero rate>20%): {'YES' if pair_ok else 'NO'}")
notes.append(f"c3 sustains (longest > 400ms): {'YES' if sustain_ok else 'NO'}")
notes.append(f"T translates (goal_dist < 0.16): {'YES' if translate_ok else 'NO'}")

if step_u_axis:
    uz = np.array([step_u_axis[s][2] for s in sorted(step_u_axis)])
    sat_pos_rate = (uz >= 2.95).sum() / len(uz)
    notes.append(f"u_z sat +3N rate: {sat_pos_rate*100:.1f}% (vs T1c ~50%)")

print("D1_VERDICT: " + " | ".join(notes))
PY

echo
echo "[d.1-EFFECT] Log: $LOG"
