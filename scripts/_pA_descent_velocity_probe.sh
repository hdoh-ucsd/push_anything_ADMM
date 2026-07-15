#!/usr/bin/env bash
# =============================================================================
# P-A — descent-velocity probe (diagnostic, NOT a commit)
#
# Falsifiable question: is the T's 308-403 N F_z peak the PWL descent leg's
# rigid impact, i.e., F_z ∝ descent kinetic energy (∝ v²)?
#
# Test: pwl_speed 0.18 (T1c baseline) → 0.09 (halved, expected F_z ~1/4)
#                                     → 0.045 (quartered, expected F_z ~1/16)
# Note: pwl_speed applies to ALL legs (lift + traverse + descend). Traverse
# also slows proportionally — accepted for a diagnostic; the falsifiable
# ratio still holds if the impact hypothesis is correct.
#
# Secondary check (the gap from T1.5): F_lateral peak — does the T actually
# translate? If F_z drops but F_lateral stays weak and goal_dist doesn't
# improve → the slam is fixed but the PUSH problem remains → executor
# restructure (P-C) is unavoidable regardless.
#
# Runs sequential (§8 substrate discipline; parallel Drake sims are an OOM
# trigger).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="results/_pA_descent_probe_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

echo "[P-A] out_dir=$OUT_DIR"
echo "[P-A] HEAD=$(git rev-parse HEAD)"
echo "[P-A] baseline (T1c): pwl_speed=0.18, F_z peak -389N (2-tick slam)"
echo "[P-A] baseline (T1.5 N=5): pwl_speed=0.18, F_z peak -308N (2-tick slam)"

run_one() {
    local label=$1
    local yaml=$2
    local log="$OUT_DIR/${label}.txt"
    echo
    echo "[P-A] === RUN $label (yaml=$yaml) ==="
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
        --sampling-c3 "$yaml" \
        --admm-iter 25 \
        --max-time 8 \
        --seed 0 \
        --no-record \
        --math-diag 2>&1 | tee "$log"
    echo "$log"
}

LOG_090=$(run_one "pA_v090" "config/_pA_v090.yaml")
LOG_045=$(run_one "pA_v045" "config/_pA_v045.yaml")

echo
echo "[P-A] === COMPARE ==="

python3 - <<PY
import re, math
import numpy as np
from pathlib import Path

def analyze(label, path, v):
    log = Path(path).read_text()
    # F decomposition
    lat_mags, vert_mags, mags_all = [], [], []
    fz_all, steps = [], []
    for m in re.finditer(
        r"\[GATE-CONTACT\] step=(\d+) .*?F_on_box=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\)", log):
        step = int(m.group(1))
        fx, fy, fz = float(m.group(2)), float(m.group(3)), float(m.group(4))
        lat = math.sqrt(fx*fx + fy*fy)
        vert = abs(fz)
        mag = math.sqrt(fx*fx + fy*fy + fz*fz)
        lat_mags.append(lat)
        vert_mags.append(vert)
        mags_all.append(mag)
        fz_all.append(fz)
        steps.append(step)
    if not lat_mags:
        return None
    lat_a = np.array(lat_mags); vert_a = np.array(vert_mags); mag_a = np.array(mags_all)
    # Streaks
    runs, run = [], 0
    for m_val in mag_a:
        if m_val > 0.5:
            run += 1
        else:
            if run > 0:
                runs.append(run)
            run = 0
    if run > 0:
        runs.append(run)
    duty = (mag_a > 0.1).mean() * 100
    longest_run = max(runs) if runs else 0

    # box_z rise
    zs = []
    for m in re.finditer(r"\[GATE-CONTACT\] .*?box_p=\([-+\d.]+,[-+\d.]+,([-+\d.]+)\)", log):
        zs.append(float(m.group(1)))
    zrise = (max(zs) - zs[0]) * 1000 if zs else float("nan")

    gd = re.search(r"\[RESULT\].*?goal_dist=([\d.]+)m", log)
    gdv = float(gd.group(1)) if gd else float("nan")
    c3s = re.findall(r"\[GS\] step=\d+ mode=c3", log)

    print(f"\n--- {label} (pwl_speed={v} m/s) ---")
    print(f"  |F| total  max: {mag_a.max():.2f}N  duty (>0.1N): {duty:.1f}%")
    print(f"  |F_lat|    max: {lat_a.max():.2f}N   (LATERAL push — translates T)")
    print(f"  |F_vert|   max: {vert_a.max():.2f}N   (VERTICAL slam — descent geometry)")
    print(f"  longest F>0.5N run: {longest_run} ticks ({longest_run*10}ms)")
    print(f"  c3 mode ticks: {len(c3s)}")
    print(f"  box_z rise: {zrise:.2f}mm  goal_dist: {gdv:.4f}m")
    return {
        "v": v,
        "fmax_total": float(mag_a.max()),
        "fmax_lat": float(lat_a.max()),
        "fmax_vert": float(vert_a.max()),
        "duty": duty,
        "longest_run_ticks": longest_run,
        "c3_ticks": len(c3s),
        "zrise": zrise,
        "gdist": gdv,
    }

r090 = analyze("v090", "$LOG_090", 0.09)
r045 = analyze("v045", "$LOG_045", 0.045)

# Baselines
t1c = {"v": 0.18, "fmax_total": 403, "fmax_vert": 389, "duty": 14.4,
       "longest_run_ticks": 52, "gdist": 0.1834, "fmax_lat": None}
t15 = {"v": 0.18, "fmax_total": 367, "fmax_vert": 308, "duty": 1.2,
       "longest_run_ticks": 2, "gdist": 0.1866, "fmax_lat": 199}

print("\n=== FALSIFIABLE TEST: F_z ∝ v²? ===")
print(f"  Hypothesis (rigid impact): F_z(v) / F_z(0.18) ≈ (v/0.18)²")
if r090 and r045:
    ratio_expected_090 = (0.09/0.18)**2   # 0.25
    ratio_expected_045 = (0.045/0.18)**2  # 0.0625
    ratio_actual_090 = r090["fmax_vert"] / t15["fmax_vert"]
    ratio_actual_045 = r045["fmax_vert"] / t15["fmax_vert"]
    print(f"  v=0.09:  expected F_z/389 ≈ {ratio_expected_090:.3f}  actual {r090['fmax_vert']:.0f}/389 = {ratio_actual_090:.3f}")
    print(f"  v=0.045: expected F_z/389 ≈ {ratio_expected_045:.3f}  actual {r045['fmax_vert']:.0f}/389 = {ratio_actual_045:.3f}")
    # Also compare linear (F ∝ v)
    lin_actual_090 = r090["fmax_vert"] / t15["fmax_vert"]
    lin_actual_045 = r045["fmax_vert"] / t15["fmax_vert"]
    print(f"  (linear alt: F ∝ v, expected {0.09/0.18:.3f} / {0.045/0.18:.3f})")

    # Verdict
    fits_quadratic = (abs(ratio_actual_090 - ratio_expected_090) < 0.15 and
                       abs(ratio_actual_045 - ratio_expected_045) < 0.10)
    scales_at_all = (r090["fmax_vert"] < 0.7 * t15["fmax_vert"] and
                      r045["fmax_vert"] < 0.4 * t15["fmax_vert"])

    print(f"\n=== SECONDARY: does the T translate? ===")
    print(f"  T1c baseline (v=0.18): goal_dist=0.1834m, F_lat max unknown")
    print(f"  T1.5 N=5   (v=0.18): goal_dist=0.1866m, F_lat max=199N")
    print(f"  v=0.09     goal_dist={r090['gdist']:.4f}m, F_lat max={r090['fmax_lat']:.0f}N")
    print(f"  v=0.045    goal_dist={r045['gdist']:.4f}m, F_lat max={r045['fmax_lat']:.0f}N")

    print("\n=== P-A VERDICT ===")
    verdict = []
    if fits_quadratic:
        verdict.append("F_z scales QUADRATICALLY with descent velocity — descent-geometry CONFIRMED")
    elif scales_at_all:
        verdict.append("F_z scales WITH velocity (not perfectly ∝ v²) — descent-geometry LIKELY")
    else:
        verdict.append("F_z does NOT scale with descent velocity — hypothesis REFUTED, F_z is something else")

    # Does T translate?
    best_gd = min(r090["gdist"], r045["gdist"])
    if best_gd < 0.16:
        verdict.append(f"T translates: best goal_dist {best_gd:.3f}m (from ~0.19 baseline)")
    else:
        verdict.append(f"T does NOT translate: best goal_dist {best_gd:.3f}m (baseline ~0.19)")

    # sustain?
    best_run = max(r090["longest_run_ticks"], r045["longest_run_ticks"])
    if best_run > 50:
        verdict.append(f"contact sustains at slower descent (longest {best_run*10}ms)")
    else:
        verdict.append(f"contact does NOT sustain at slower descent (longest {best_run*10}ms)")

    print("PA_VERDICT: " + " | ".join(verdict))

print(f"\n[P-A] Logs: $LOG_090  $LOG_045")
PY
