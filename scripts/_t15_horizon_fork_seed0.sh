#!/usr/bin/env bash
# =============================================================================
# T-Push Completion Plan — Stage T1.5 diagnostic fork
# Same env as the Phase 2 sweep + PUSHA_C3PLUS_N=5 (horizon 20→5).
# Single seed (0) — this is a mechanism probe, not a sweep.
#
# Measures (from plan §4 Stage T1.5):
#   - first-knot |u[0]| and u_z
#   - peak Drake force (F_on_box) and duty
#   - |qy| / |qz|
#   - does whack convert to push? (contact sustains, duty rises)
#   - does the c3-entry stall clear? (seed 0 in the sweep stayed mode=free 8s)
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="results/_t15_horizon_fork_${STAMP}"
LOG="$OUT_DIR/seed0_N5.log"
mkdir -p "$OUT_DIR"

echo "[T1.5] out_dir=$OUT_DIR"
echo "[T1.5] HEAD=$(git rev-parse HEAD)"
echo "[T1.5] tree_dirty=$(git diff --stat HEAD | tail -1)"

T0=$(date +%s)

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
    --solver c3plus --c3plus-projection lcp \
    --ee-space \
    --sampling-c3 config/sampling_c3_kik_t.yaml \
    --admm-iter 25 \
    --max-time 8 \
    --seed 0 \
    --no-record \
    --math-diag > "$LOG" 2>&1
RC=$?

T1=$(date +%s)
WALL=$((T1 - T0))

if grep -qE "Input/output error|OSError:\s*\[Errno 5\]|EIO|Bus error|SIGBUS" "$LOG"; then
  echo "[T1.5] HALT: EIO/SIGBUS in seed0 log"
  exit 2
fi

echo
echo "[T1.5] === wall=${WALL}s rc=$RC ==="
echo

# Verify N-override actually fired.
grep -E "^\[HORIZON-PROBE\]" "$LOG" | head -1

# Mode timeline: how many ticks in c3 vs free?
N_TICKS=$(grep -cE '^\[STEP\] step=' "$LOG" || true)
N_C3=$(grep -E '^\[STEP\] step=' "$LOG" | grep -c 'mode=c3' || true)
N_FREE=$(grep -E '^\[STEP\] step=' "$LOG" | grep -c 'mode=free' || true)
echo "[T1.5] tick counts: total=$N_TICKS  c3=$N_C3  free=$N_FREE"

# First-knot drive (u[0]) — peak and mean of u_z from [C3+] lines.
echo "[T1.5] u[0] first-knot drive summary:"
grep -E '^\[C3\+\] step=' "$LOG" | tail -1
grep -oE '\|u\[0\]\|=[0-9.]+N' "$LOG" | tr -dc 'N0-9.\n' | \
  awk '{gsub("N","",$0); if($0=="") next; if(!min||$0<min)min=$0; if(!max||$0>max)max=$0; s+=$0; n++} END{if(n)printf "        |u[0]|:  min=%.2fN  max=%.2fN  mean=%.2fN  n=%d\n", min, max, s/n, n}'

# Drake-contact force on box
echo "[T1.5] F_on_box magnitude summary (from [GATE-CONTACT]):"
grep -oE 'F_on_box=\([^)]+\)' "$LOG" | \
  awk -F'[(,)]' '{
    fx=$2; fy=$3; fz=$4;
    m=sqrt(fx*fx+fy*fy+fz*fz);
    if(m>peak) peak=m;
    if(m>0){s+=m; n++;}
  }
  END {
    printf "        F_on_box: peak=%.2fN  mean_nonzero=%.2fN  contact_ticks=%d\n",
      peak, (n?s/n:0), n
  }'

# Object pitch / roll  (qy, qz from STAGE-A-TRACE)
echo "[T1.5] object attitude summary:"
grep -oE 'qy=[-0-9.]+ qz=[-0-9.]+' "$LOG" | \
  awk '{gsub(/qy=|qz=/,"",$0); split($0,a," "); qy=a[1]; qz=a[2];
        aqy=(qy<0?-qy:qy); aqz=(qz<0?-qz:qz);
        if(aqy>mqy)mqy=aqy; if(aqz>mqz)mqz=aqz}
  END{printf "        |qy| peak=%.4f   |qz| peak=%.4f\n", mqy, mqz}'

# [RESULT] line
echo
echo "[T1.5] [RESULT] line:"
grep -E '^\[RESULT\]' "$LOG" | tail -1
echo
echo "[T1.5] log: $LOG"
