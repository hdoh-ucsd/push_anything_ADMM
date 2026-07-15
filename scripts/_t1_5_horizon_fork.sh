#!/usr/bin/env bash
# =============================================================================
# T1.5 horizon fork — planner-authority diagnostic (N=20→5)
#
# NOT A COMMIT — this is the fork test. Runs the composed T1a+b+c stack
# (as landed at 3aa29ef) with the cube-arc's proven planner-authority
# knob (PUSHA_C3PLUS_N=5), decomposes the T's contact into lateral vs
# vertical components, and reports whether:
#   (a) N=5 gentles the first-knot drive AND the 403N top-press softens
#       AND contact sustains better → planner-authority; adopt N=5 as T
#       canonical → T2.
#   (b) N=5 doesn't help / the 403N survives → descent-geometry problem
#       (PWL 0.18 m/s constant vertical velocity), not horizon; needs
#       a different fix (executor restructure or descent-velocity profile).
#
# Measurements (with contact-survival guard):
#   - first-knot |u[0]| AND u_z SEPARATELY (upward/vertical drive)
#   - peak F AND peak F_z SEPARATELY (lateral push vs vertical top-press)
#   - does the 403N top-press SOFTEN under N=5, or survive?
#   - duty %, longest F>0.5N run, longest c3-mode run
#   - |qy|/|qz| (does the T tip? cube arc's tumble metric)
#   - goal_dist
#   - press-vs-slam: duration of the F_z peak (slam = brief impulse;
#     press = sustained contact)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="results/_t1_5_horizon_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/t1_5_t_seed0_N5.txt"

echo "[T1.5-FORK] out_dir=$OUT_DIR"
echo "[T1.5-FORK] HEAD=$(git rev-parse HEAD)"
echo "[T1.5-FORK] tree_dirty=$(git diff --stat HEAD | tail -1)"
echo "[T1.5-FORK] PUSHA_C3PLUS_N=5 (down from default 20)"

echo
echo "[T1.5-FORK] === T (push_t) seed 0, composed T1a+b+c + N=5 ==="

PUSHA_C3PLUS_N=5 \
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
echo "[T1.5-FORK] === EXTRACT ==="

python3 - <<PY
import re, math
import numpy as np
from pathlib import Path

log = Path("$LOG").read_text()

# --- First-knot |u[0]| and axis components (from [C3+] lines) ---
u0_all, uz_all = [], []
for m in re.finditer(
    r"\[C3\+\] step=(\d+) \|u\[0\]\|=([\d.]+)N u_axis=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\)N", log):
    u0_all.append(float(m.group(2)))
    uz_all.append(float(m.group(5)))
u0_arr = np.array(u0_all)
uz_arr = np.array(uz_all)
print(f"\n--- first-knot |u[0]| (planner authority proxy) ---")
if u0_arr.size:
    print(f"  |u[0]|  max: {u0_arr.max():.2f}N  mean: {u0_arr.mean():.2f}N  n={u0_arr.size}")
    print(f"  u_z     max: {uz_arr.max():+.2f}N  min: {uz_arr.min():+.2f}N  "
          f"|u_z| max: {np.abs(uz_arr).max():.2f}N")
    print(f"  compare cube arc N=20→5 result: peak |u[0]| dropped 67% (~50N→~16N)")

# --- F decomposition ---
lat_mags, vert_mags = [], []
steps, mags_all = [], []
fz_all = []
for m in re.finditer(
    r"\[GATE-CONTACT\] step=(\d+) .*?F_on_box=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\)", log):
    step = int(m.group(1))
    fx, fy, fz = float(m.group(2)), float(m.group(3)), float(m.group(4))
    lat = math.sqrt(fx*fx + fy*fy)
    vert = abs(fz)
    mag = math.sqrt(fx*fx + fy*fy + fz*fz)
    lat_mags.append(lat)
    vert_mags.append(vert)
    steps.append(step)
    mags_all.append(mag)
    fz_all.append(fz)

print(f"\n--- F decomposition (lateral push vs vertical top-press) ---")
if lat_mags:
    lat_arr = np.array(lat_mags)
    vert_arr = np.array(vert_mags)
    mag_arr = np.array(mags_all)
    fz_arr = np.array(fz_all)
    print(f"  |F| total   max: {mag_arr.max():.2f}N  nz frac: {(mag_arr > 0.1).mean()*100:.1f}%")
    print(f"  |F_lateral| max: {lat_arr.max():.2f}N  (translates T toward goal)")
    print(f"  |F_vert|    max: {vert_arr.max():.2f}N  F_z max: {fz_arr.max():+.2f}N  F_z min: {fz_arr.min():+.2f}N  (top-press or slam)")

    # Compare vs T1c baseline (403N total, F_z=-389N)
    print(f"\n  vs T1c (composed, N=20): |F| max 403N, F_z min -389N")
    if vert_arr.max() < 350:
        soft = "SOFTENED"
    else:
        soft = "SURVIVED"
    print(f"  → top-press (|F_vert|) {soft} under N=5")

    # Press vs slam: consecutive-tick run at peak F_z
    peak_step = steps[int(np.argmax(vert_arr))]
    peak_step_idx = steps.index(peak_step)
    # Count |F_z|>10N run around peak (window ±100 steps)
    window_size = 50
    lo, hi = max(0, peak_step_idx - window_size), min(len(steps), peak_step_idx + window_size)
    window_vert = vert_arr[lo:hi]
    at_peak_run = 0
    max_run = 0
    for v in window_vert:
        if v > 10:
            at_peak_run += 1
            max_run = max(max_run, at_peak_run)
        else:
            at_peak_run = 0
    print(f"\n  press-vs-slam: at F_vert peak (step={peak_step}), "
          f"consecutive |F_z|>10N ticks (in ±{window_size}-step window): "
          f"max_run={max_run} ({max_run*10}ms)")
    print(f"    slam   ≈ 1-3 ticks (brief impulse)")
    print(f"    press  ≥ 20 ticks (200ms sustained)")

# --- Contact duty + streaks ---
print(f"\n--- duty + sustain streaks ---")
if lat_mags:
    runs, run = [], 0
    for m in mag_arr:
        if m > 0.5:
            run += 1
        else:
            if run > 0:
                runs.append(run)
            run = 0
    if run > 0:
        runs.append(run)
    if runs:
        print(f"  duty (|F|>0.1N): {(mag_arr > 0.1).mean()*100:.1f}%   "
              f"(baseline 5.1%, T1a 4.6%, T1b 10.7%, T1c 14.4%)")
        print(f"  longest |F|>0.5N run: {max(runs)} ticks ({max(runs)*10}ms)   "
              f"(T1c: 52 ticks / 520ms)")
        print(f"  runs ≥ 20 ticks (200ms sustained): {sum(1 for r in runs if r >= 20)}")

# --- c3 mode presence ---
c3_lines = re.findall(r"\[GS\] step=(\d+) mode=c3 switch=(\w+)", log)
c3_steps_set = sorted({int(s) for s, _ in c3_lines})
if c3_steps_set:
    max_c3_run = cur = 0
    prev = None
    for s in c3_steps_set:
        if prev is not None and s == prev + 1:
            cur += 1
        else:
            cur = 1
        max_c3_run = max(max_c3_run, cur)
        prev = s
    print(f"  c3 mode ticks total: {len(c3_lines)}  longest c3-mode run: "
          f"{max_c3_run} ticks ({max_c3_run*10}ms)   (T1c: 34 / 18-tick / 180ms)")
    from collections import Counter
    for reason, cnt in Counter(r for _, r in c3_lines if r.startswith("kToC3")).most_common():
        print(f"    dispatch reason {reason}: {cnt}")

# --- T tip metrics |qy|/|qz| ---
qys, qzs = [], []
for m in re.finditer(
    r"\[GATE-CONTACT\] .*?box_q=\(([-+\d.]+),([-+\d.]+),([-+\d.]+),([-+\d.]+)\)", log):
    _, _, qy, qz = m.group(1), m.group(2), m.group(3), m.group(4)
    qys.append(abs(float(qy)))
    qzs.append(abs(float(qz)))
if qys:
    print(f"\n--- T tip (does the T tumble?) ---")
    print(f"  |qy| max: {max(qys):.4f}  mean: {sum(qys)/len(qys):.4f}   "
          f"(cube arc |qy|=0.70-0.73 pinned across probes = tumbled)")
    print(f"  |qz| max: {max(qzs):.4f}  mean: {sum(qzs)/len(qzs):.4f}   "
          f"(yaw — the T's task target axis)")

# --- box_z rise (no-launch) ---
zs = []
for m in re.finditer(r"\[GATE-CONTACT\] .*?box_p=\([-+\d.]+,[-+\d.]+,([-+\d.]+)\)", log):
    zs.append(float(m.group(1)))
if zs:
    print(f"\n  box_z rise: {(max(zs) - zs[0])*1000:.2f}mm  "
          f"(T1c: 0.41mm; bar <5mm = no launch)")

# --- goal_dist ---
gd = re.search(r"\[RESULT\].*?goal_dist=([\d.]+)m", log)
if gd:
    print(f"  goal_dist: {gd.group(1)}m  (T1c: 0.1834m; baseline ~0.19m)")

# --- Fork verdict ---
print(f"\n--- T1.5 FORK VERDICT ---")
notes = []
# Softening?
if lat_mags:
    if vert_arr.max() < 250:
        notes.append(f"top-press SOFTENED (|F_vert| {vert_arr.max():.0f}N vs T1c 389N)")
    elif vert_arr.max() < 350:
        notes.append(f"top-press partially softened (|F_vert| {vert_arr.max():.0f}N vs T1c 389N)")
    else:
        notes.append(f"top-press SURVIVED (|F_vert| {vert_arr.max():.0f}N vs T1c 389N)")

    # Sustain?
    if runs and max(runs) > 100:
        notes.append(f"sustained IMPROVED (longest {max(runs)*10}ms vs T1c 520ms)")
    elif runs and max(runs) > 50:
        notes.append(f"sustained similar (longest {max(runs)*10}ms vs T1c 520ms)")
    else:
        notes.append(f"sustained WORSE or similar (longest {max(runs)*10 if runs else 0}ms vs T1c 520ms)")

    # Planner-authority?
    if u0_arr.size and u0_arr.max() < 40:
        notes.append(f"|u[0]| max {u0_arr.max():.0f}N (softened from N=20 regime — planner-authority hypothesis SUPPORTED)")
    elif u0_arr.size:
        notes.append(f"|u[0]| max {u0_arr.max():.0f}N")

print("T1_5_FORK_VERDICT: " + " | ".join(notes))
PY

echo
echo "[T1.5-FORK] Log: $LOG"
