#!/bin/bash
# Force sweep via nominal_push_force ∈ {1, 2, 3, 5, 8} on seed 3, push-W, 8s, jitter=0
# Tracks min_push_force with nominal so each value is a HARD effective force
# (avoids D2-cap neutralization documented in project_d2_cap_neutralizes_min_push.md).
# SERIAL: one run fully exits before next. Video off. No commits.

ROOT=/root/push_anything_ADMM
SWEEP_DIR=$ROOT/q2_force_sweep_seed3
PARAMS=$ROOT/control/sampling_c3/params.py
SWEEPLOG=$SWEEP_DIR/sweep_progress.log

mkdir -p $SWEEP_DIR
cd $ROOT

echo "=== sweep start $(date) ===" >> $SWEEPLOG
echo "HEAD: $(git rev-parse HEAD)" >> $SWEEPLOG
echo "Sweep values: 1.0 2.0 3.0 5.0 8.0 N" >> $SWEEPLOG
echo "Knob tracking: nominal_push_force == min_push_force == F (per run)" >> $SWEEPLOG

for force in 1.0 2.0 3.0 5.0 8.0; do
    RUN_DIR=$SWEEP_DIR/nominal${force}N
    mkdir -p $RUN_DIR

    echo "" >> $SWEEPLOG
    echo "=== force=${force} START $(date) ===" >> $SWEEPLOG

    # Edit params.py: BOTH knobs, BOTH places (dataclass default + from_dict fallback)
    sed -i "s/nominal_push_force: float = [0-9.]*\$/nominal_push_force: float = ${force}/" $PARAMS
    sed -i "s/nominal_push_force   = float(raw.get(\"nominal_push_force\", [0-9.]*))/nominal_push_force   = float(raw.get(\"nominal_push_force\", ${force}))/" $PARAMS
    sed -i "s/min_push_force: float = [0-9.]*\$/min_push_force: float = ${force}/" $PARAMS
    sed -i "s/min_push_force       = float(raw.get(\"min_push_force\", [0-9.]*))/min_push_force       = float(raw.get(\"min_push_force\", ${force}))/" $PARAMS

    # Verify both knobs match force value
    NOMINAL_OK=$(grep -c "nominal_push_force: float = ${force}\$" $PARAMS)
    MIN_OK=$(grep -c "min_push_force: float = ${force}\$" $PARAMS)
    NOMINAL_FB_OK=$(grep -c "nominal_push_force\", ${force})" $PARAMS)
    MIN_FB_OK=$(grep -c "min_push_force\", ${force})" $PARAMS)
    if [ "$NOMINAL_OK" != "1" ] || [ "$MIN_OK" != "1" ] || [ "$NOMINAL_FB_OK" != "1" ] || [ "$MIN_FB_OK" != "1" ]; then
        echo "ABORT_EDIT_VERIFY force=${force} nominal=${NOMINAL_OK} min=${MIN_OK} nominalfb=${NOMINAL_FB_OK} minfb=${MIN_FB_OK}" >> $SWEEPLOG
        grep -nE "min_push_force|nominal_push_force" $PARAMS >> $SWEEPLOG
        continue
    fi

    # Record state
    grep -nE "min_push_force|nominal_push_force" $PARAMS > $RUN_DIR/params_state.txt
    git rev-parse HEAD > $RUN_DIR/HEAD.txt
    git diff HEAD -- $PARAMS >> $RUN_DIR/HEAD.txt
    echo "F=${force}" > $RUN_DIR/force.txt

    # Run sim (synchronous; this loop iteration blocks until exit)
    python main.py pushing --task-id 4 --seed 3 --max-time 8 --admm-iter 3 --solver c3plus --ee-space --sampling-c3 config/sampling_c3_kik.yaml --no-record > $RUN_DIR/run.log 2>&1
    EXIT=$?

    # Verify the right wrapper + planner engaged
    if ! grep -q "use_ee_space=True solver.n_x=19 solver.n_u=3" $RUN_DIR/run.log; then
        echo "FLAG_FAIL force=${force} (use_ee_space) -- archiving _BAD" >> $SWEEPLOG
        mv $RUN_DIR ${RUN_DIR}_BAD
        continue
    fi
    if ! grep -q "SamplingC3MPC enabled" $RUN_DIR/run.log; then
        echo "FLAG_FAIL force=${force} (SamplingC3MPC) -- archiving _BAD" >> $SWEEPLOG
        mv $RUN_DIR ${RUN_DIR}_BAD
        continue
    fi

    # Quick row for the sweep log
    RESULT=$(grep "method=sampling-c3" $RUN_DIR/run.log | tail -1)
    MAXFCMD=$(grep -oE "f_cmd=\([0-9.+-]+,[0-9.+-]+,[0-9.+-]+" $RUN_DIR/run.log | awk -F'[(,]' '{m=sqrt($2*$2+$3*$3+$4*$4); if (m>max) max=m} END {printf "%.3f",max}')
    CONTACT_Y=$(grep -cE "^\[STEP\].*contact=Y" $RUN_DIR/run.log)
    echo "force=${force} exit=${EXIT} max_f_cmd=${MAXFCMD} contact_Y=${CONTACT_Y} ${RESULT}" >> $SWEEPLOG
    echo "=== force=${force} DONE $(date) ===" >> $SWEEPLOG
done

# Restore params.py to defaults (5.0 nominal, 2.0 min)
sed -i "s/nominal_push_force: float = [0-9.]*\$/nominal_push_force: float = 5.0/" $PARAMS
sed -i "s/nominal_push_force   = float(raw.get(\"nominal_push_force\", [0-9.]*))/nominal_push_force   = float(raw.get(\"nominal_push_force\", 5.0))/" $PARAMS
sed -i "s/min_push_force: float = [0-9.]*\$/min_push_force: float = 2.0/" $PARAMS
sed -i "s/min_push_force       = float(raw.get(\"min_push_force\", [0-9.]*))/min_push_force       = float(raw.get(\"min_push_force\", 2.0))/" $PARAMS

echo "" >> $SWEEPLOG
echo "=== sweep END $(date) ===" >> $SWEEPLOG
echo "params.py restored: $(grep -E 'min_push_force|nominal_push_force' $PARAMS | grep -v 'raw.get' | head -2)" >> $SWEEPLOG
