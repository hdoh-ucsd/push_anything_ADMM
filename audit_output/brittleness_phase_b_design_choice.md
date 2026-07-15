# Phase A3 → Phase B design choice — B3 (sample-selection stabilization)

**Decision date:** 2026-06-13
**Inputs:** `audit_output/brittleness_phase_a1_postentry.md` (mechanism + lever localization), `audit_output/brittleness_phase_a2_phantom.csv` (per-tick trace).

---

## Decision rule application

The Phase A3 decision rule (plan §Phase A Task A3):

| Finding | Candidate |
|---|---|
| Divergence step at entry tick (110) and ≥ 50% phantom in ticks 110–119 | B1 entry-stabilization |
| Divergence step post-entry (≥ step 115), phantom-fraction similar | B2 in-c3 ADMM stability |
| Divergence at entry AND post-entry phantom high | sequential B1 + B2 |

**Measured findings:**
- Phantom-fraction in ticks 110–119: **100 % for all 4 runs** — does not discriminate good vs bad. The "≥ 50 % phantom" condition for B1 is met homogeneously, but the second condition for B1 (entry-distance separation) is FALSIFIED — all runs have `ee_to_surf = 4 mm` at c3-entry with zero good-vs-bad gap.
- Initial phantom-c3 burst is bounded at ~18–19 ticks for ALL runs by the existing 5-tick disengage gate (`wrapper.py:622`, `DISENGAGE_THRESHOLD=5`) — B2's "exit c3 if Drake-admit drops" is structurally REDUNDANT.
- Actual divergence step (≥ 1 mm box-xy split): **step 170–180**, well after the phantom-c3 burst ended.
- Actual lever step (root cause): **step ~159**, when the sample-buffer / IK-target selection produces a > 30 mm target-position difference from < 1 mm EE input difference (30:1 amplification).
- State at first real-contact c3 admit: good runs at `nhat_y ≈ +0.30` (upper-+y lever), bad runs at `nhat_y ≈ −0.22` (lower-−y lever). Same +x face, mirror-image lever arms.

**Therefore the decision-rule output is NEW route — B3 (not in the original plan).**

## Chosen candidate: **B3 — sample-buffer / sample-selection stabilization**

The root lever is sample-selection logic amplifying sub-mm box-position noise into > 30 mm sample-target differences. The fix attacks that surface directly. Three sub-candidates ordered by cost:

### B3a (cheap — start here)

**Audit `sampling.py:30 generate_samples` and `sampling.py:146 _face_normal_projection` for any reliance on the global numpy RNG state** (rather than the explicitly-passed `rng` parameter). If `np.random.*` calls bypass the user's seeded RNG, sub-mm box-position noise can flip which sample is drawn even though the explicit RNG seed is constant.

**Expected effect:** seeds become reproducible; sample selection becomes deterministic for fixed box position. Doesn't fix the sub-mm-noise amplification — but if amplification was via RNG draws, this *is* the fix.

**Test (Phase C C1):** unit test that for the same box position and seed, `generate_samples` returns identical sample positions across two separate calls.

### B3b (mechanism investigation)

**Read `_face_normal_projection` (`sampling.py:146`) line-by-line + `SampleBuffer.best_with_position()` (`sample_buffer.py:136`) to find the line that flips sample order under sub-mm noise.**

Suspected culprits:
- A face-selection branch (4 faces; if the face is chosen by `argmax(cost)` where cost has < mm sensitivity to box quat, then sub-mm quat changes flip face choice → 30 mm target shift).
- A tangent-jitter draw that uses the RNG advanced by the face-selection branch (different face → different RNG state → different jitter).
- A buffer-pruning step that flips an entry's eligibility based on sub-mm pose threshold (`sample_buffer.py:95` uses `pos_threshold=0.05` and `ang_threshold=0.30`; these should be coarse, but a corner-case at exactly the threshold can flip).

**Expected effect:** identifies the specific line; B3c can then quantize the input to that line.

**Test (Phase C C2):** unit test that for box positions differing by < 1 mm, sample selection differs by < 1 mm (currently it can differ by > 30 mm).

### B3c (alternate — if B3b can't find a single line)

**Quantize sample positions to a stable lattice.** E.g., snap tangent offsets in `_face_normal_projection` to a 1 cm grid (`offset = round(offset_continuous / 0.01) * 0.01`). Sub-mm input noise produces the same snapped offset.

**Risk:** may reduce sample diversity; needs the Q5-noregress guard.

## Pre-registered success bar (carried from Phase B§4 of the plan, unchanged)

- PRIMARY: GOOD-basin occupancy **≥ 5/6** on the SERIAL N=6 sweep (baseline pre-fix: 2/4).
- SECONDARY: best ≤ 0.10 m (preserves the lead).
- GUARD: Q5-noregress within ±10 % vs c8402b3 on seeds 0/2/4.
- PARTIAL: 4/6 ≤ occupancy < 5/6 → iterate.
- FAIL: < 4/6 OR Q5 regression > 10 % → rollback.
- Median (NOT best) GOOD-basin run = canonical run for video re-render + pruning ablation.

## What this changes vs the original plan

- Phase B (design doc) → re-scoped to design B3a/b/c, NOT B1.
- Phase C (TDD) → new params under `SamplingC3Params` (e.g., `use_sample_rng_audit`, `use_sample_quantization`, `sample_quantization_grid_m`) instead of `use_entry_stabilization`. Identity defaults preserved.
- Phase D harness unchanged (SERIAL N=6, per-run `[RESULT]`, EIO recovery).

## What stays the same

- All EIO-avoidance structure (top of plan).
- Two checkpoints (CP1 here, CP2 after Phase C iteration).
- Pre-registered success bar.
- Median-not-best canonical-run pick.
- Both goals (pruning ablation + video) resume on the fix.

## File:line citations for Phase B's exact code-site additions

- `sampling.py:30` — `generate_samples` (B3a audit entry point).
- `sampling.py:146` — `_face_normal_projection` (B3b mechanism + B3c quantization site).
- `sample_buffer.py:95` — `SampleBuffer.prune` (B3b alternate culprit).
- `sample_buffer.py:136` — `SampleBuffer.best_with_position` (B3b cost-ordering culprit).
- `control/sampling_c3/params.py:222-260` — `SamplingParams` (where new opt-in fields land).
- `config/sampling_c3_kik.yaml` — YAML enable line.

---

## STOP — awaiting user go for Phase A4 / Phase B

The original B1/B2 candidates are falsified or redundant. Plan now routes to B3 with B3a → B3b → B3c as a fall-through chain. Phase A4 (audit sampling.py + sample_buffer.py line-by-line) is the next step before Phase B locks the exact fix.
