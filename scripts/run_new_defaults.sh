#!/usr/bin/env bash
# run_new_defaults.sh
# First sweep after CLI cleanup: verify the new defaults (C3+ inner, Sampling-C3 outer)
# and probe three orthogonal axes — prepositioned regression, sampling ablation, alt task.

set -uo pipefail   # not -e: we want to continue past a single scenario failure

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/new_defaults_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

MAX_TIME=6.0           # sim seconds — matches ik_planner_demo_s0
PER_SCENARIO_TIMEOUT=420   # wall-clock seconds (sim ~4 min at 372 ms/step × ~600 steps)

echo "================================================================"
echo "  New-defaults sweep — ${TIMESTAMP}"
echo "  Results: $RESULTS_DIR"
echo "  Per-scenario sim time: ${MAX_TIME}s"
echo "  Per-scenario wall timeout: ${PER_SCENARIO_TIMEOUT}s"
echo "  Expected total wall time: ~15-25 minutes"
echo "================================================================"
echo

# ----------------------------------------------------------------------
# Run one scenario. Args: label, extra_main_args
# ----------------------------------------------------------------------
run_scenario() {
    local label="$1"
    local args="$2"
    local log="$RESULTS_DIR/${label}.log"
    local video="$RESULTS_DIR/${label}.mp4"
    local html="$RESULTS_DIR/${label}.html"

    echo "--- [$label] ---"
    echo "    python main.py pushing $args --max-time $MAX_TIME"

    local start=$(date +%s)
    timeout "$PER_SCENARIO_TIMEOUT" python main.py pushing $args \
        --max-time "$MAX_TIME" \
        --save-video "$video" \
        --video-path "$html" \
        > "$log" 2>&1
    local rc=$?
    local elapsed=$(($(date +%s) - start))

    if [ $rc -eq 124 ]; then
        echo "    ✗ TIMEOUT after ${elapsed}s"
    elif [ $rc -ne 0 ]; then
        echo "    ✗ Exit code $rc after ${elapsed}s"
    else
        echo "    ✓ Completed in ${elapsed}s"
    fi

    # Quick inline preview of the result line
    grep -E "^\[RESULT\]|^\[GS-perf\]" "$log" | sed 's/^/    /' || echo "    (no [RESULT] line)"
    echo
}

# ----------------------------------------------------------------------
# Scenarios
# ----------------------------------------------------------------------

# 1. The headline: what users get out of the box now.
#    C3+ inner + Sampling-C3 outer + IK reposition + default task (push east 0.30m)
run_scenario "01_default" ""

# 2. Regression diagnostic. Hidden flag, productive contact start.
#    This is the only configuration that worked in the April-30 deck —
#    if C3+ here doesn't move the box correctly, something regressed.
run_scenario "02_prepositioned_diag" "--prepositioned"

# 3. Outer-wrapper ablation. Confirms --no-sampling-c3 escape hatch wired correctly
#    AND tells us how much of today's behavior is the dispatcher vs. the inner solver.
run_scenario "03_no_sampling_ablation" "--no-sampling-c3"

# 4. Generalization probe. Alternate task direction (verify task-id 1 from README).
#    Same defaults, different goal — tests whether IK reposition handles non-east targets.
run_scenario "04_default_alt_task" "--task-id 1"

# ----------------------------------------------------------------------
# Summary table
# ----------------------------------------------------------------------
echo
echo "================================================================"
echo "  Summary"
echo "================================================================"
echo
printf "%-28s %-22s %-14s %-10s %s\n" \
    "Scenario" "final_obj_xy" "goal_dist" "success" "switches"
printf "%-28s %-22s %-14s %-10s %s\n" \
    "--------" "------------" "---------" "-------" "--------"

for log in "$RESULTS_DIR"/*.log; do
    label=$(basename "$log" .log)
    result_line=$(grep "^\[RESULT\]" "$log" | tail -1)
    perf_line=$(grep "^\[GS-perf\]" "$log" | tail -1)

    if [ -z "$result_line" ]; then
        printf "%-28s %s\n" "$label" "(no [RESULT] — crashed or timeout, check log)"
        continue
    fi

    box=$(echo "$result_line"   | grep -oE 'final_obj_xy=\([^)]+\)' | sed 's/final_obj_xy=//')
    dist=$(echo "$result_line"  | grep -oE 'goal_dist=[0-9.]+m'    | sed 's/goal_dist=//')
    succ=$(echo "$result_line"  | grep -oE 'success=[A-Z]+'        | sed 's/success=//')
    sw=$(echo "$perf_line"      | grep -oE 'switches=[0-9]+'       | sed 's/switches=//')

    printf "%-28s %-22s %-14s %-10s %s\n" "$label" "${box:-?}" "${dist:-?}" "${succ:-?}" "${sw:-?}"
done

echo
echo "Logs:    $RESULTS_DIR/*.log"
echo "Videos:  $RESULTS_DIR/*.mp4"
echo "Replays: $RESULTS_DIR/*.html"
echo
echo "Inspect first:"
echo "  head -40 $RESULTS_DIR/01_default.log              # confirm C3+ and sampling-C3 on"
echo "  grep '\[GS\] step=' $RESULTS_DIR/01_default.log | tail -5   # final mode state"
echo "  grep 'switches=' $RESULTS_DIR/01_default.log | tail -1      # did the FSM ever commit?"
