#!/usr/bin/env bash
# N success-terminated trials of one demo; per-trial JSON appended to ledger.
# Usage: run_c3plus_trials.sh LEDGER_NAME DEMO GX GY GYAW N_TRIALS WALL_CAP PORT
set -uo pipefail
LEDGER="${1:?}"; DEMO="${2:?}"; GX="$3"; GY="$4"; GYAW="$5"; N="${6:-5}"; CAP="${7:-1200}"; PORT="${8:-7991}"
WT=/root/push_anything_ADMM/external/oim_c++_anything/.claude/worktrees/c3plus-ablation
BIN=$WT/bazel-bin/examples/sampling_c3
URL="udpm://239.255.76.67:${PORT}?ttl=0"
LED=/root/push_anything_ADMM/results/${LEDGER}.jsonl
cd "$WT"
for i in $(seq 1 "$N"); do
  OUT=/root/push_anything_ADMM/results/${LEDGER}_trial${i}
  mkdir -p "$OUT"
  "$BIN/franka_osc_controller" --demo_name="$DEMO" --lcm_url="$URL" > "$OUT/osc.log" 2>&1 & OSC=$!
  "$BIN/franka_sampling_c3_controller" --demo_name="$DEMO" --lcm_url="$URL" > "$OUT/planner.log" 2>&1 & PLAN=$!
  /root/miniconda3/envs/push_anything_ADMM/bin/python3 \
    /root/push_anything_ADMM/scripts/c3ab/record_full_state.py \
    --goal "$GX" "$GY" "$GYAW" --out "$OUT/state_trace.jsonl" \
    --url "$URL" --duration "$CAP" --exit-on-success > "$OUT/success.log" 2>&1 & REC=$!
  sleep 3
  "$BIN/franka_sim" --demo_name="$DEMO" --lcm_url="$URL" > "$OUT/sim.log" 2>&1 & SIM=$!
  wait $REC
  kill $SIM $OSC $PLAN 2>/dev/null; wait $SIM $OSC $PLAN 2>/dev/null
  /root/miniconda3/envs/push_anything_ADMM/bin/python3 - "$OUT" "$LED" "$i" <<'PYEOF'
import json, sys
out, led, idx = sys.argv[1], sys.argv[2], int(sys.argv[3])
lines = open(f'{out}/success.log').read().splitlines()
fin = json.loads([l for l in lines if l.startswith('FINAL')][-1][6:])
trial = {'trial': idx, 'run_dir': out,
         'success': fin['first_success_t'] is not None,
         'success_t': fin['first_success_t'],
         'end_t': fin.get('t'), 'samples': fin.get('samples'),
         'pos_err': fin.get('pos_err'), 'theta_err': fin.get('ang_err')}
open(led, 'a').write(json.dumps(trial) + '\n')
print('TRIAL', json.dumps(trial))
PYEOF
  sleep 2
done
echo "LEDGER_DONE $LED"
