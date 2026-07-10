#!/usr/bin/env bash
# =============================================================================
# T1b effect check — 3-leg high-waypoint reposition
#
# T1b config: use_reposition_pwl_trajectory=true, pwl_waypoint_height=0.0774
# (reference push_t value). Activates the Stage A RepositionTrajectory:
# lift → traverse (at z_safe=0.0774) → descend onto p_target.
#
# Effect checks:
#   (a) PRIMARY — 3-leg fires. [STAGE-A-PWL] build lines appear in the log;
#       at least one build event with K knots >= 3 (i.e. actual 3-leg, not
#       collapsed single-leg for short hops).
#   (b) up→across→down trace. Over the traj-active window, max ee_z reached
#       must exceed the ~z_safe ceiling (>=0.070 m — allowing tracking lag)
#       AND min ee_z reached must be below the T1a c3-ceiling (< 0.044 m)
#       or at least materially lower than baseline.
#   (c) NEW — c3 gate release. Does min ee_z drop below 44 mm so gated c3
#       can dispatch? Count [GS] mode=c3 switch=kToC3* events.
#   (d) SECONDARY — F_on_box max drops from T1a's 314 N (traverse-through
#       becomes traverse-over).
#
# Comparison points:
#   P2 T baseline (58694eb):  F max 96.7 N nz 5.1% (no gate, dive-whack)
#   T1a (ae5b429):            F max 314.0 N nz 4.6% (gated, reposition brush)
#   T1b (this run):           expect F max << 314 (traverse OVER, not brush),
#                              expect min_ee_z < 44 mm (descent leg completes),
#                              expect [GS] c3 dispatch events > 0
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="results/_t1b_effect_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/t1b_t_seed0.txt"

echo "[T1B-EFFECT] out_dir=$OUT_DIR"
echo "[T1B-EFFECT] HEAD=$(git rev-parse HEAD)"
echo "[T1B-EFFECT] tree_dirty=$(git diff --stat HEAD | tail -1)"

echo
echo "[T1B-EFFECT] === T (push_t) seed 0 with EE_z gate + 3-leg reposition ==="

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
echo "[T1B-EFFECT] === EXTRACT ==="

python3 - <<PY
import re, math
from pathlib import Path

log = Path("$LOG").read_text()

# --- (a) 3-leg PWL fires ---
pwl_builds = re.findall(
    r"\[STAGE-A-PWL\] step=(\d+) sim_t=([\d.]+) build .*?K=(\d+) t_end=([\d.]+)",
    log)
print(f"\n--- (a) 3-leg PWL builds ---")
print(f"total [STAGE-A-PWL] build events: {len(pwl_builds)}")
three_leg = [b for b in pwl_builds if int(b[2]) >= 3]
print(f"  of which 3-leg (K>=3, actual lift+traverse+descend): {len(three_leg)}")
if pwl_builds:
    print(f"  first build:  step={pwl_builds[0][0]}, sim_t={pwl_builds[0][1]}, "
          f"K={pwl_builds[0][2]}, t_end={pwl_builds[0][3]}")
    print(f"  last build:   step={pwl_builds[-1][0]}, sim_t={pwl_builds[-1][1]}, "
          f"K={pwl_builds[-1][2]}, t_end={pwl_builds[-1][3]}")

# --- (b, c) EE_z trace ---
# Extract (step, ee_z) from [STEP] lines.
step_ee_z = {}
for m in re.finditer(
    r"\[STEP\] step=(\d+) .*?ee=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\)", log):
    step_ee_z[int(m.group(1))] = float(m.group(4))
if not step_ee_z:
    for m in re.finditer(
        r"\[GATE-CONTACT\] step=(\d+) .*?ee_p=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\)", log):
        step_ee_z[int(m.group(1))] = float(m.group(4))

print(f"\n--- (b, c) EE_z trace ---")
if step_ee_z:
    all_z = list(step_ee_z.values())
    print(f"  EE_z overall: max={max(all_z)*1000:.1f}mm min={min(all_z)*1000:.1f}mm "
          f"over {len(step_ee_z)} ticks")
    # If a 3-leg build fired, look at the WINDOW starting from that build
    # up to build_step + duration ≈ t_end - t_start seconds worth of ticks.
    if three_leg:
        b_step = int(three_leg[0][0])
        b_sim_t = float(three_leg[0][1])
        b_t_end = float(three_leg[0][3])
        b_dur_ticks = int((b_t_end - b_sim_t) / 0.01)
        window = [(s, z) for s, z in step_ee_z.items()
                  if b_step <= s <= b_step + b_dur_ticks + 20]
        if window:
            w_z_max = max(z for _, z in window)
            w_z_min = min(z for _, z in window)
            print(f"  window after first 3-leg build (step={b_step} → "
                  f"step≈{b_step + b_dur_ticks}, dur≈{b_dur_ticks} ticks):")
            print(f"    max ee_z: {w_z_max*1000:.1f}mm "
                  f"(z_safe=77.4mm target; ≥70mm = lift reached)")
            print(f"    min ee_z: {w_z_min*1000:.1f}mm "
                  f"(T1a ceiling=44.0mm; <44 = c3 gate can release)")
            lift_reached = w_z_max >= 0.070
            descent_reached = w_z_min < 0.044
            print(f"    lift-reached: {'YES' if lift_reached else 'NO (traverse-only?)'}")
            print(f"    descent-below-ceiling: "
                  f"{'YES' if descent_reached else 'NO (c3 dispatch will be blocked)'}")

# --- c3 dispatch events (gate release) ---
c3_switches = re.findall(
    r"\[GS\] step=\d+ mode=c3 switch=(kToC3[A-Za-z]+)", log)
print(f"\nc3 dispatch events (kToC3*): {len(c3_switches)}")
if c3_switches:
    from collections import Counter
    for reason, cnt in Counter(c3_switches).most_common():
        print(f"  {reason}: {cnt}")

# --- (d) F_on_box peak ---
mags = []
for m in re.finditer(
    r"\[GATE-CONTACT\].*?F_on_box=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\)", log):
    fx, fy, fz = float(m.group(1)), float(m.group(2)), float(m.group(3))
    mags.append(math.sqrt(fx*fx + fy*fy + fz*fz))
if mags:
    fmax = max(mags)
    fnz = sum(1 for v in mags if v > 0.1) / len(mags)
    n = len(mags)
    print(f"\n--- (d) F_on_box (secondary) ---")
    print(f"  max={fmax:.2f}N nonzero_frac={fnz*100:.1f}% n_ticks={n}")
    print(f"  compare: P2 baseline 96.7N nz 5.1% (dive-whack)")
    print(f"           T1a       314.0N nz 4.6% (gated, reposition brush)")

# --- Goal_dist context ---
m = re.search(r"\[RESULT\].*?goal_dist=([\d.]+)m", log)
if m:
    print(f"\n[RESULT] goal_dist={m.group(1)}m")

# --- Verdict ---
print(f"\n--- T1B_EFFECT_VERDICT ---")
notes = []
if three_leg:
    notes.append(f"3-leg fired ({len(three_leg)} builds with K>=3)")
else:
    notes.append("3-leg DID NOT FIRE — investigate")
if step_ee_z and three_leg:
    b_step = int(three_leg[0][0])
    b_sim_t = float(three_leg[0][1])
    b_t_end = float(three_leg[0][3])
    b_dur_ticks = int((b_t_end - b_sim_t) / 0.01)
    window = [(s, z) for s, z in step_ee_z.items()
              if b_step <= s <= b_step + b_dur_ticks + 20]
    if window:
        w_z_max = max(z for _, z in window)
        w_z_min = min(z for _, z in window)
        if w_z_max >= 0.070 and w_z_min < 0.044:
            notes.append(f"up→across→down trace VERIFIED (max {w_z_max*1000:.0f}mm, min {w_z_min*1000:.0f}mm)")
        elif w_z_max >= 0.070:
            notes.append(f"up+across OK but descent SHORT (min ee_z {w_z_min*1000:.0f}mm still > 44mm ceiling)")
        else:
            notes.append(f"up→across→down NOT trace-verified (max {w_z_max*1000:.0f}mm)")
notes.append(f"c3 dispatch events: {len(c3_switches)}")
if mags:
    fmax = max(mags)
    notes.append(f"F_on_box max {fmax:.1f}N (vs T1a 314N)")
print("T1B_EFFECT_VERDICT: " + " | ".join(notes))
PY

echo
echo "[T1B-EFFECT] Log: $LOG"
