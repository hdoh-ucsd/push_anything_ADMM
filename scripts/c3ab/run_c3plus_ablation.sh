#!/usr/bin/env bash
# One reference-C3+ ablation run: franka_sim + OSC + sampling controller +
# success recorder. Usage: run_c3plus_ablation.sh RUN_NAME DEMO GOAL_X GOAL_Y GOAL_YAW [DURATION]
set -uo pipefail
RUN_NAME="${1:?}"; DEMO="${2:?}"; GX="${3:?}"; GY="${4:?}"; GYAW="${5:?}"; DUR="${6:-180}"
WT=/root/push_anything_ADMM/external/oim_c++_anything/.claude/worktrees/c3plus-ablation
BIN=$WT/bazel-bin/examples/sampling_c3
URL="udpm://239.255.76.67:${LCM_PORT:-7985}?ttl=0"
OUT=/root/push_anything_ADMM/results/$RUN_NAME
mkdir -p "$OUT"; cd "$WT"
"$BIN/franka_osc_controller" --demo_name="$DEMO" --lcm_url="$URL" > "$OUT/osc.log" 2>&1 & OSC=$!
"$BIN/franka_sampling_c3_controller" --demo_name="$DEMO" --lcm_url="$URL" > "$OUT/planner.log" 2>&1 & PLAN=$!
/root/miniconda3/envs/push_anything_ADMM/bin/python3 \
  /root/.claude/jobs/cfe8fda1/tmp/record_full_state.py \
  --goal "$GX" "$GY" "$GYAW" --out "$OUT/state_trace.jsonl" \
  --url "$URL" --duration "$DUR" > "$OUT/success.log" 2>&1 & REC=$!
sleep 3
timeout "$DUR" "$BIN/franka_sim" --demo_name="$DEMO" --lcm_url="$URL" > "$OUT/sim.log" 2>&1
SIM=$?
wait $REC 2>/dev/null
kill $OSC $PLAN 2>/dev/null; wait $OSC $PLAN 2>/dev/null
printf "demo=%s\nsim_exit=%s\nduration_s=%s\n" "$DEMO" "$SIM" "$DUR" > "$OUT/process_status.txt"
echo "done $RUN_NAME"; tail -1 "$OUT/success.log"
