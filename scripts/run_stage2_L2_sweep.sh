#!/usr/bin/env bash
# Stage 2 L2 commit-face gate sweep — 5 seeds.
# Reads against pre-registered SCs from the rotation roadmap:
#   SC-L2-convert (seeds 1,2): final.x < -0.05 AND first-face=+x
#   SC-L2-noregress-working (seeds 0,3,4): final.x < {-0.45,-0.50,-0.26}
#                                          AND first-face=+x (must not delay legitimate +x commits)
#   SC-L2-deadlock (1,2): contact_ticks==0 -> routes to sampler bias (motivated, not pre-emptive)
#   SC-L2-lambda: SC-L1-lambda holds 5/5 (gate doesn't touch face distribution)
#   SC-L2-L1-redundancy: total [GATE-ALIGN] fires across all 5 seeds
#       0   -> L1 confirmed redundant (Task 8 removes it)
#       >0  -> L2 has a gap; L1-only / L2-only / both attribution
#
# Discipline: STRICTLY SERIAL — one seed's process fully exits before the
# next launches. NEVER run two sims concurrently here (EIO wedge bit this
# project twice).
# Threshold = -0.7 (default), L1 entry_align_threshold = 0.7 from kik YAML
# (L1 + L2 alongside so SC-L2-L1-redundancy is measurable from logs).
set -eo pipefail
OUTDIR=${OUTDIR:-stage2_L2_sweep}
PY=${PY:-/root/miniconda3/envs/push_anything_ADMM/bin/python}
mkdir -p "$OUTDIR"
git rev-parse HEAD > "$OUTDIR/HEAD.txt"
: > "$OUTDIR/sweep.summary"
for SEED in 0 1 2 3 4; do
  echo "=== seed=$SEED start $(date +%s) ===" | tee -a "$OUTDIR/sweep.summary"
  # tmpfs guard — abort if /tmp climbs into the danger zone.
  TMP_PCT=$(df --output=pcent /tmp 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
  if [ "${TMP_PCT:-0}" -ge 85 ]; then
    echo "ABORT: /tmp at ${TMP_PCT}% before seed $SEED — STOP, do NOT rm -rf /tmp/claude-0/*" \
      | tee -a "$OUTDIR/sweep.summary"
    exit 1
  fi
  START=$(date +%s)
  "$PY" -u main.py pushing \
      --task-id 4 \
      --solver c3plus \
      --c3plus-projection lcp \
      --ee-space \
      --sampling-c3 config/sampling_c3_kik.yaml \
      --admm-iter 25 \
      --max-time 8 \
      --no-record \
      --seed "$SEED" \
      > "$OUTDIR/seed${SEED}_stage2_L2.log" 2>&1 || true
  RC=$?
  ELAPSED=$(( $(date +%s) - START ))
  LOG="$OUTDIR/seed${SEED}_stage2_L2.log"
  N_LCS=$(grep -c '\[CONTACT-RUN\] step=.* contact_type=EE-BOX' "$LOG" || true)
  N_GATE_L1=$(grep -c '^\[GATE-ALIGN\]' "$LOG" || true)
  N_GATE_L2=$(grep -c '^\[GATE-COMMIT-FACE\]' "$LOG" || true)
  N_TAG_NM=$(grep -c '^\[GATE-COMMIT-FACE\].* tag=near-miss-cone' "$LOG" || true)
  N_TAG_PP=$(grep -c '^\[GATE-COMMIT-FACE\].* tag=perpendicular' "$LOG" || true)
  N_TAG_AG=$(grep -c '^\[GATE-COMMIT-FACE\].* tag=anti-goal' "$LOG" || true)
  RESULT_LINE=$(grep '^\[RESULT\]' "$LOG" | tail -1 || true)
  echo "seed=$SEED rc=$RC ${ELAPSED}s n_lcs=$N_LCS gate_L1=$N_GATE_L1 gate_L2=$N_GATE_L2 (nm=$N_TAG_NM perp=$N_TAG_PP anti=$N_TAG_AG) | $RESULT_LINE" \
    | tee -a "$OUTDIR/sweep.summary"
done
echo "DONE; see $OUTDIR/sweep.summary"
