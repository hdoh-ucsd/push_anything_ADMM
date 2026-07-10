#!/usr/bin/env bash
# =============================================================================
# T1c effect check — kMeshNormal area-weighted face pick
#
# T1c config: use_mesh_normal_area_weighting=true (T yaml sampling_params).
# Replaces uniform-across-faces face selection with categorical distribution
# proportional to face area. Uniform-within-face is unchanged.
#
# Effect checks (BOTH must pass):
#   (i)  Top faces (indices 8, 9) drawn at NON-ZERO rate. The TOP-face WIP
#        is landed with T1c; if this fires, both top-face inclusion (the
#        "which faces" axis) AND area-weighting (the "how sampled" axis)
#        are exercised.
#   (ii) Observed face-pick distribution matches AREA-WEIGHTED (not uniform).
#        Expected fractions (T's 10 patches, indices 0..9):
#          side small  (idx 0, 3, 5, half_len 0.02):  3.85% each
#          side medium (idx 2, 6,    half_len 0.03):  5.77% each
#          side large  (idx 1, 4, 7, half_len 0.08): 15.38% each
#          top         (idx 8, 9):                   15.38% each
#        Uniform baseline: 10% per face. Sample size ~800 gens × ~5 samples
#        = ~4000; strong statistical power.
#
# Pass bar: (i) AND (ii). If area-weighting proves IRRELEVANT (no measurable
# change in T behavior — F peak, duty, closure), record as FINDING, do NOT
# declare substitution complete.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR="results/_t1c_effect_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/t1c_t_seed0.txt"

echo "[T1C-EFFECT] out_dir=$OUT_DIR"
echo "[T1C-EFFECT] HEAD=$(git rev-parse HEAD)"
echo "[T1C-EFFECT] tree_dirty=$(git diff --stat HEAD | tail -1)"

echo
echo "[T1C-EFFECT] === T (push_t) seed 0 with kMeshNormal + 3-leg + EE_z gate ==="

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
echo "[T1C-EFFECT] === EXTRACT ==="

python3 - <<PY
import re, math
import numpy as np
from pathlib import Path

log = Path("$LOG").read_text()

# --- (i, ii) Face-pick histogram ---
hists = []
for m in re.finditer(
    r"\[SAMPLE-FACE-HIST\] n=\d+ hist=\[([\d, ]+)\] shape=(\w+) mesh_normal=(\w+)",
    log):
    counts = [int(x) for x in m.group(1).split(",")]
    hists.append((counts, m.group(2), m.group(3)))

print(f"\n--- (i, ii) [SAMPLE-FACE-HIST] emissions ---")
print(f"total emissions: {len(hists)}")
if not hists:
    print("  NO histograms emitted — sampling.py hook not firing. Investigate.")
else:
    total = np.zeros(len(hists[0][0]), dtype=int)
    for counts, _, _ in hists:
        total += np.array(counts)
    n_faces = len(total)
    n_side = 8  # T has 8 side faces + 2 top faces
    total_samples = int(total.sum())
    fracs = total / max(total_samples, 1)
    print(f"total samples counted: {total_samples}")
    print(f"observed per-face counts:  {total.tolist()}")
    print(f"observed per-face percent: [" + ", ".join(f"{f*100:.2f}%" for f in fracs) + "]")

    # Expected area-weighted (T geometry):
    side_hl = np.array([0.02, 0.08, 0.03, 0.02, 0.08, 0.02, 0.03, 0.08])
    T_BAR_H = 0.04
    side_areas = 2 * side_hl * T_BAR_H
    top_hxy = np.array([[0.08, 0.02], [0.02, 0.08]])
    top_areas = 4 * top_hxy[:,0] * top_hxy[:,1]
    exp_probs = np.concatenate([side_areas, top_areas])
    exp_probs = exp_probs / exp_probs.sum()
    unif_probs = np.ones(n_faces) / n_faces
    print(f"expected area-weighted:    [" + ", ".join(f"{p*100:.2f}%" for p in exp_probs) + "]")
    print(f"uniform baseline:          [" + ", ".join(f"{p*100:.2f}%" for p in unif_probs) + "]")

    # RMSE-style fit
    rmse_area = float(np.sqrt(np.mean((fracs - exp_probs)**2)) * 100)
    rmse_unif = float(np.sqrt(np.mean((fracs - unif_probs)**2)) * 100)
    print(f"\nRMSE vs area-weighted expected: {rmse_area:.2f}pp")
    print(f"RMSE vs uniform baseline:       {rmse_unif:.2f}pp")

    # Top-face rate
    top_count = int(total[n_side:].sum())
    top_frac = top_count / max(total_samples, 1)
    print(f"\nTop-face draws (indices 8, 9): {top_count} / {total_samples} = {top_frac*100:.2f}% "
          f"(expected area-weighted: {(exp_probs[n_side:].sum()*100):.2f}%)")

    # Pass criteria
    top_nonzero = top_count > 0
    area_fits_better = rmse_area < rmse_unif
    print(f"\n(i)  top faces drawn: {'YES' if top_nonzero else 'NO — FAIL'}")
    print(f"(ii) area-weighted fits better than uniform: "
          f"{'YES' if area_fits_better else 'NO — FAIL'}")

# --- Contact stats (for context — is this pass useful?) ---
mags = []
for m in re.finditer(
    r"\[GATE-CONTACT\].*?F_on_box=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\)", log):
    fx, fy, fz = float(m.group(1)), float(m.group(2)), float(m.group(3))
    mags.append(math.sqrt(fx*fx + fy*fy + fz*fz))

# c3 dispatch events
c3_switches = re.findall(
    r"\[GS\] step=\d+ mode=c3 switch=(kToC3[A-Za-z]+)", log)

# EEZ-GATE fires
eez_fires = len(re.findall(r"\[EEZ-GATE\]", log))

# 3-leg PWL builds
pwl_builds = re.findall(r"\[STAGE-A-PWL\] step=\d+ .*?K=(\d+)", log)
pwl_3leg = sum(1 for k in pwl_builds if int(k) >= 3)

# [RESULT]
gd = re.search(r"\[RESULT\].*?goal_dist=([\d.]+)m", log)

print(f"\n--- Context (T1 integration bar preview) ---")
print(f"EE_z gate fires (T1a):        {eez_fires}")
print(f"3-leg PWL builds K>=3 (T1b):  {pwl_3leg} (of {len(pwl_builds)})")
print(f"c3 dispatch events (kToC3*):  {len(c3_switches)}")
if c3_switches:
    from collections import Counter
    for reason, cnt in Counter(c3_switches).most_common():
        print(f"  {reason}: {cnt}")
if mags:
    fmax = max(mags)
    fnz = sum(1 for v in mags if v > 0.1) / len(mags)
    print(f"F_on_box: max={fmax:.2f}N nonzero_frac={fnz*100:.1f}% n_ticks={len(mags)}")
    print(f"  compare P2 baseline:   96.7N nz 5.1% (dive-whack)")
    print(f"  compare T1a:          314.0N nz 4.6% (repos brush, no 3-leg)")
    print(f"  compare T1b:           65.0N nz 10.7% (3-leg, no area-weighting)")
if gd:
    print(f"[RESULT] goal_dist={gd.group(1)}m")

# --- Verdict ---
print(f"\n--- T1C_EFFECT_VERDICT ---")
notes = []
if hists:
    if top_count > 0 and rmse_area < rmse_unif:
        notes.append(f"AREA-WEIGHTED CONFIRMED (RMSE {rmse_area:.1f} vs uniform {rmse_unif:.1f})")
    elif top_count > 0:
        notes.append(f"TOP FACES DRAWN but distribution doesn't clearly area-fit (RMSE area {rmse_area:.1f} vs unif {rmse_unif:.1f})")
    else:
        notes.append("TOP FACES NOT DRAWN — FAIL")
else:
    notes.append("NO [SAMPLE-FACE-HIST] EMITTED — HOOK BROKEN")
if mags:
    fmax_val = max(mags)
    fnz_val = sum(1 for v in mags if v > 0.1) / len(mags) * 100
    notes.append(f"F max {fmax_val:.1f}N nz {fnz_val:.1f}%")
notes.append(f"c3 dispatches: {len(c3_switches)}")
print("T1C_EFFECT_VERDICT: " + " | ".join(notes))
PY

echo
echo "[T1C-EFFECT] Log: $LOG"
