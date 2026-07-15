#!/usr/bin/env bash
# Task 4: F=5 seed-3 post-change at HEAD c8402b3 (Q2c gate active).
# Mirrors scripts/_q2_force_sweep_seed3.sh:50-51 nominal=5.0/min=5.0 protocol
# (the only config that produced the 35-tick mid-traverse bump at steps 521-557).
# Edits params.py BOTH knobs to 5.0 via sed (BOTH places — dataclass + from_dict
# fallback per CLAUDE.md params.py:533 trap), runs seed 3, RESTORES params.py.
# Detached. Manual read of OUT/seed3.log when done.
set -uo pipefail
ROOT=/root/push_anything_ADMM
PARAMS=$ROOT/control/sampling_c3/params.py
OUT=$ROOT/audit_output/admit_guard_gate/postchange_f5_seed3
mkdir -p "$OUT"

restore_params() {
  sed -i "s/nominal_push_force: float = [0-9.]*\$/nominal_push_force: float = 5.0/" "$PARAMS" || true
  sed -i "s/nominal_push_force   = float(raw.get(\"nominal_push_force\", [0-9.]*))/nominal_push_force   = float(raw.get(\"nominal_push_force\", 5.0))/" "$PARAMS" || true
  sed -i "s/min_push_force: float = [0-9.]*\$/min_push_force: float = 2.0/" "$PARAMS" || true
  sed -i "s/min_push_force       = float(raw.get(\"min_push_force\", [0-9.]*))/min_push_force       = float(raw.get(\"min_push_force\", 2.0))/" "$PARAMS" || true
  echo "[task4] params.py after restore (trap):" >> "$OUT/HEAD.txt"
  grep -nE "min_push_force|nominal_push_force" "$PARAMS" >> "$OUT/HEAD.txt" 2>/dev/null || true
}
trap 'restore_params' EXIT INT TERM HUP

cd "$ROOT"
echo "[task4] HEAD=$(git rev-parse HEAD)" | tee "$OUT/HEAD.txt"
echo "[task4] params.py before sed:" >> "$OUT/HEAD.txt"
grep -nE "min_push_force|nominal_push_force" "$PARAMS" >> "$OUT/HEAD.txt"

# Set BOTH knobs (nominal + min) to 5.0 — matches F=5 sweep run.log protocol.
sed -i "s/nominal_push_force: float = [0-9.]*\$/nominal_push_force: float = 5.0/" "$PARAMS"
sed -i "s/nominal_push_force   = float(raw.get(\"nominal_push_force\", [0-9.]*))/nominal_push_force   = float(raw.get(\"nominal_push_force\", 5.0))/" "$PARAMS"
sed -i "s/min_push_force: float = [0-9.]*\$/min_push_force: float = 5.0/" "$PARAMS"
sed -i "s/min_push_force       = float(raw.get(\"min_push_force\", [0-9.]*))/min_push_force       = float(raw.get(\"min_push_force\", 5.0))/" "$PARAMS"

# Verify all 4 knob edits succeeded.
NOMINAL_OK=$(grep -c "nominal_push_force: float = 5.0\$" "$PARAMS")
MIN_OK=$(grep -c "min_push_force: float = 5.0\$" "$PARAMS")
NOMINAL_FB_OK=$(grep -c "nominal_push_force\", 5.0)" "$PARAMS")
MIN_FB_OK=$(grep -c "min_push_force\", 5.0)" "$PARAMS")
echo "[task4] sed-verify: nominal=$NOMINAL_OK min=$MIN_OK nominalfb=$NOMINAL_FB_OK minfb=$MIN_FB_OK" >> "$OUT/HEAD.txt"
if [ "$NOMINAL_OK" != "1" ] || [ "$MIN_OK" != "1" ] || [ "$NOMINAL_FB_OK" != "1" ] || [ "$MIN_FB_OK" != "1" ]; then
  echo "[task4] ABORT: sed verify failed; restoring." >> "$OUT/HEAD.txt"
  sed -i "s/min_push_force: float = [0-9.]*\$/min_push_force: float = 2.0/" "$PARAMS"
  sed -i "s/min_push_force       = float(raw.get(\"min_push_force\", [0-9.]*))/min_push_force       = float(raw.get(\"min_push_force\", 2.0))/" "$PARAMS"
  exit 2
fi

grep -nE "min_push_force|nominal_push_force" "$PARAMS" > "$OUT/params_state.txt"

# Run sim (synchronous; --no-record to skip video).
LOG="$OUT/seed3.log"
echo "[task4] seed=3 F=5 → $LOG"
timeout 480 python -u main.py pushing \
  --task-id 4 \
  --seed 3 \
  --max-time 8 \
  --admm-iter 3 \
  --solver c3plus \
  --ee-space \
  --sampling-c3 config/sampling_c3_kik.yaml \
  --no-record \
  > "$LOG" 2>&1 || echo "[task4] seed=3 exit=$?"

# Trap will restore on EXIT — no inline restore needed.
echo "[task4] complete."
