# G2 — Test the ADMM-iteration hypothesis

Working hypothesis from the without-prepositioned-with-F1 run analysis:

> The c3 inner solve plans `||λ_t||` at ~4× the friction cone limit
> on every horizon step (`[MATH.δ] case 3 surface: 20`). With
> `admm_iter=3`, the dual variable ω never accumulates enough to
> penalise infeasible λ in the QP, so the QP keeps re-asking for
> the same impossible friction; the Lorentz projection inflates
> `λ_n` to comply, and the planned trajectory commands forces the
> physical EE can't deliver. Result: EE drifts off-axis instead of
> pushing the box.
>
> Adaptive ρ is in the code but fires every 10 iters — the
> hard-coded `admm_iter=3` means it never triggers. Raising
> `admm_iter` should both let ω accumulate and let ρ adapt.

## Change

`main.py` — three small hunks:

1. New CLI flag `--admm-iter N` (default 3, matching old behaviour).
2. Constructor wires `admm_iter=args.admm_iter` instead of hardcoded 3.
3. Top-of-file docstring updated to reflect the actual MPC parameters
   (horizon=20 not 8, dt=0.05 not 0.03, etc. — the docstring was wildly
   stale for unrelated reasons; fixed while we were nearby).

No other file changed. ADMM solver itself, sampling-C3 wrapper,
env_builder, IK — all untouched.

## Apply

```bash
cd push_anything_ADMM/
cp <thisdir>/main.py main.py
python -m pytest tests/      # all 111 should still pass
python main.py --help        # confirm --admm-iter shows up
```

## Test plan

Sweep four ADMM-iter values across the without-prepositioned baseline,
using the new `--name` CLI for clean output naming:

```bash
# Control: current behaviour (admm_iter=3). Same as the run that just failed.
python main.py pushing --sampling-c3 --math-diag --admm-iter 3 \
    --name g2_admm_03iter

# Just-below-rho-adaptation threshold. Tests whether ω accumulation
# alone (without rho adaptation) is enough.
python main.py pushing --sampling-c3 --math-diag --admm-iter 9 \
    --name g2_admm_09iter

# At the rho-adaptation threshold. ω accumulates AND rho adapts once.
python main.py pushing --sampling-c3 --math-diag --admm-iter 10 \
    --name g2_admm_10iter

# Comfortable budget: rho adapts twice, ω fully settled.
python main.py pushing --sampling-c3 --math-diag --admm-iter 25 \
    --name g2_admm_25iter
```

Each writes `results/g2_admm_NNiter.{txt,mp4,html}`. The 25-iter run
will be the slowest — expect roughly 8× the wall-clock of the 3-iter
run since per-step solve cost scales nearly linearly with admm_iter.
On the 8 s sim with ~200 ms/step at 3 iters, that's ~25 minutes for
the 25-iter run. If you want to sanity-check the hypothesis cheaply
first, run only 3-iter (control) and 10-iter; if 10-iter shows
qualitative improvement, run 25 to confirm the trend; if 10 looks
identical to 3, skip 25.

## What to grep for in each log

Three diagnostics tell us whether G2 is working:

```bash
for f in results/g2_admm_{03,09,10,25}iter.txt; do
    echo "=== $f ==="

    # 1. Final outcome
    grep -E '^\[RESULT\]|^\[GS-perf\]' $f | head -2

    # 2. Friction cone violation rate (case 3 surface count)
    # If G2 works, case 1 count goes UP and case 3 count goes DOWN.
    echo "[MATH.δ] cone case histogram (last few steps):"
    grep -E 'case [123]' $f | tail -12

    # 3. Did rho adapt? Search for the "would halve" / "would double"
    # log lines AND the actual rho value in [ADMM] header. With
    # --admm-iter ≥10 we expect rho to drift from 100.
    echo "rho values seen:"
    grep -E '^\[ADMM\]' $f | awk '{for(i=1;i<=NF;i++) if($i~/^rho=/) print $i}' \
        | sort -u

    # 4. Did the box ever move?
    grep -E '^\s+t=8\.00s' $f
done
```

## How to read the result

| outcome | interpretation |
|---|---|
| 03-iter still fails (control), 25-iter succeeds | G2 confirmed. ADMM iter was the binding constraint. Pick a sensible default (probably 10 or 15). |
| 03-iter fails, 25-iter also fails the same way | G2 ruled out. Friction-cone violations are inherent to the QP setup, not the iter count. Move to G3 (add quadratic penalty on λ_t). |
| 03-iter fails, 10-iter succeeds, 25-iter regresses | Sweet spot exists. Probably worth checking whether rho adaptation is overshooting at 25 iters. Set default to 10. |
| 03-iter succeeds (!) | The previous run was non-deterministic (random sampling) and we got lucky/unlucky. Re-run several times, look at variance. |

The most diagnostic single number across the runs: the
`case 3 surface` count on `[MATH.δ]`. Currently 20/20 — every
horizon step has friction-cone violations. If G2 works, this
should drop substantially (maybe to 0-5 out of 20 once ω
penalises infeasible λ).

The second-most diagnostic number: the final `obj_xy`. The previous
run parked at `(-0.005, +0.002)` — 5 mm from origin, basically
unmoved over 8 s. If even one of the iter values produces
`obj_xy.x > +0.05` (50 mm progress toward the goal), the
hypothesis is partially confirmed.

## Notes

- `--admm-iter` doesn't affect the sampling-C3 outer loop's
  cheap-sample evaluations; those use `surrogate_admm_iters` from
  the YAML (default 1) and aren't touched here. Only the k=0 full
  solve uses the new value.
- If the 25-iter run takes too long (>30 min), you can drop the
  horizon length on a one-off via the C3MPC constructor — but for
  this diagnostic we want to keep the rest of the controller
  identical.
- This is purely a diagnostic test. If G2 works, the next question
  is "what's the right default" and "should we expose it in the
  YAML" — both for a follow-up.
