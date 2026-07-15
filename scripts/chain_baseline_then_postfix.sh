#!/usr/bin/env bash
# Driver: wait for baseline-ext (seeds 5..19) to finish, then apply the
# face-picker patch and launch the post-fix 20-seed sweep (seeds 0..19).
# Final step: aggregate baseline contact-rate vs post-fix contact-rate.
set -uo pipefail

BASELINE_EXT_DIR="$(ls -dt results/baseline_ext_* 2>/dev/null | head -1)"
if [ -z "$BASELINE_EXT_DIR" ]; then
    echo "ERROR: no baseline_ext_* dir; baseline must already be running"
    exit 1
fi
echo "watching baseline at $BASELINE_EXT_DIR"

# 1. Wait for baseline-ext driver to finish (15 seeds).
while pgrep -f "run_baseline_seed_extension.sh" >/dev/null; do
    sleep 60
    n_done=$(ls "$BASELINE_EXT_DIR" 2>/dev/null \
             | xargs -I{} sh -c 'grep -l "Simulation complete" "'"$BASELINE_EXT_DIR"'/{}" 2>/dev/null' \
             | wc -l)
    echo "$(date +%H:%M:%S) baseline_ext done=$n_done/15"
done
echo "baseline-ext driver exited"

# 2. Apply the face-picker patch.
echo "applying face-picker patch"
cd /root/push_anything_ADMM
git apply /tmp/face_picker_fix.patch
echo "wrapper.py now has the fix:"
sed -n '1500,1525p' control/sampling_c3/wrapper.py

# 3. Launch post-fix 20-seed sweep (seeds 0..19).
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
POSTFIX_DIR="results/postfix_${TIMESTAMP}"
mkdir -p "$POSTFIX_DIR"
echo "post-fix sweep → $POSTFIX_DIR"

for seed in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19; do
    out="$POSTFIX_DIR/stepc_seed${seed}_postfix.log"
    t0=$(date +%s)
    timeout 600 python main.py pushing \
        --task-id 4 --solver c3plus --sampling-c3 \
        --admm-iter 25 --max-time 2.5 --no-record \
        --seed "$seed" \
        --name "stepc_seed${seed}_postfix" \
        > "$out" 2>&1
    rc=$?
    dt=$(($(date +%s) - t0))
    nc=$(grep -c "contact=Y" "$out" || true)
    echo "POSTFIX seed=$seed rc=$rc ${dt}s contact_steps=$nc"
done

# 4. Aggregate.
echo
echo "=== BASELINE (HEAD + β, pre-fix) ==="
baseline_existing="$(ls -dt results/beta_seed_sweep_* | head -1)"
n_seeds=0
n_contact=0
for f in "$baseline_existing"/stepc_seed*_beta_on.log "$BASELINE_EXT_DIR"/stepc_seed*_beta_on.log; do
    [ -f "$f" ] || continue
    grep -q "Simulation complete" "$f" || continue
    n_seeds=$((n_seeds + 1))
    if grep -q "contact=Y" "$f"; then
        n_contact=$((n_contact + 1))
    fi
done
echo "baseline N=$n_seeds, contact-formed=$n_contact"

echo
echo "=== POST-FIX (HEAD + β + face-picker directional fix) ==="
n_seeds=0
n_contact=0
for f in "$POSTFIX_DIR"/stepc_seed*_postfix.log; do
    [ -f "$f" ] || continue
    grep -q "Simulation complete" "$f" || continue
    n_seeds=$((n_seeds + 1))
    if grep -q "contact=Y" "$f"; then
        n_contact=$((n_contact + 1))
    fi
done
echo "post-fix N=$n_seeds, contact-formed=$n_contact"

echo
echo "=== DONE ==="
