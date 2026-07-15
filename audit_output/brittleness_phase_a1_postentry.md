# Phase A1 — Post-entry divergence trace (setarch-probe N=4)

**Source data:** `nondet_seed0_setarch_hashseed/run{1,2,3,4}/run.log` (existing logs; no new sims).
**Substrate:** 518bcfa effective HEAD + ipopt-det + PYTHONHASHSEED=0 + setarch -R, seed=0, 16s sim.
**Tool:** awk on `[STEP]` / `[CONTACT-RUN]` / mode-change traces; cross-checked against direct log inspection.

> **REVISION NOTE (2026-06-13):** An earlier version of this document claimed the brittleness was a "50–80 tick phantom-c3" amplifier with phantom-c3 running anti-productively via the recoil convention. **That was wrong** — the existing 5-tick disengage gate (`wrapper.py:622`) caps phantom-c3 at ~18 ticks across ALL runs (good and bad). The actual divergence is in the FREE-mode (reposition) trajectory **after** the initial phantom-c3 burst, driven by sample-buffer / IK target selection. Corrected mechanism below.

---

## 1. Mode timeline per run

```
run1 (GOOD 0.0985):    free→c3@110  c3→free@128  free→c3@161  c3→free@162  ...
run2 (GOOD 0.0779):    free→c3@111  c3→free@130  free→c3@161  c3→free@162  ...
run3 (BORDER 0.1510):  free→c3@110  c3→free@129  free→c3@194  c3→free@195  ...
run4 (BAD 0.2145):     free→c3@110  c3→free@128  free→c3@192  c3→free@193  ...
```

- **Initial phantom-c3 burst: ~18–19 ticks for ALL runs.** The 5-tick disengage gate (`wrapper.py:622`, `DISENGAGE_THRESHOLD=5`) caps it. Box does NOT move during this phase (obj_xy stays at (0,0) for all runs through step 130).
- **First real-contact c3 re-entry**: step 161 for GOOD runs, step 192–194 for BAD runs — **~31-tick gap**.

## 2. Box trajectory — converges through step ~165, diverges in step ~170–180

| step | run1 obj_xy | run2 obj_xy | run3 obj_xy | run4 obj_xy |
|---|---|---|---|---|
| 130 | (0, 0) | (0, 0) | (0, 0) | (0, 0) |
| 150 | (−0.045, +0.010) | (−0.040, +0.008) | (−0.042, +0.008) | (−0.046, +0.008) |
| 160 | (−0.075, +0.009) | (−0.071, +0.007) | (−0.074, +0.008) | (−0.075, +0.008) |
| 170 | (−0.092, +0.006) | (−0.089, +0.007) | (−0.091, **−0.000**) | (−0.092, **−0.003**) |
| 180 | (−0.113, +0.002) | (−0.112, −0.000) | (−0.103, **−0.010**) | (−0.101, **−0.012**) |
| 190 | (−0.134, −0.006) | (−0.138, −0.006) | (−0.100, −0.017) | (−0.096, −0.019) |
| 200 | **(−0.155, −0.016)** | **(−0.162, −0.015)** | (−0.095, −0.021) | (−0.090, −0.023) |

**At step 160 all 4 runs are at obj≈(−0.074, +0.008) — within 4 mm.**
**At step 200 GOOD runs are at −0.16 m; BAD runs have STALLED and even REVERSED (run3: −0.097 at step 194 → −0.090 at step 200, moving back toward +x).**

Divergence point: between step 160 and 180. The bad runs lose productive push around step 170–180.

## 3. The actual mechanism — sample-buffer / IK target divergence at step ~159

Up to step ~158, runs 1 and 3 have near-identical EE positions AND nearly-identical IK targets. At step 159 the IK targets DIVERGE significantly:

```
run1@159  ee=(+0.005,+0.034,+0.058)  target=(-0.024,+0.005,+0.056)
run3@159  ee=(+0.006,+0.033,+0.055)  target=(-0.046,+0.036,+0.051)
                                              ^^^^^^^^^^^^^^^^^^^^
                                              22 mm Δx, 31 mm Δy  TARGET DIVERGENCE
```

ee_xy differs by < 1 mm across the runs, but the IK target shifted by 22 mm in x and 31 mm in y. **The sample buffer / sample selection logic amplified sub-mm input noise into a >30 mm target choice difference.** That's the brittleness lever.

The 30 mm target shift sends the EE on DIFFERENT trajectories around the box:
- Good runs (target ~(−0.024, +0.005)): EE continues approaching from +y, lands real contact at step 161 with `nhat_BA_W=(+0.892, +0.286, +0.350)` (upper-+y lever arm) → productive yaw.
- Bad runs (target ~(−0.046, +0.036)): EE traverses farther, ends up approaching from −y, lands real contact at step 192–194 with `nhat_BA_W=(+0.961, −0.221, +0.166)` (upper-−y lever arm) → counter-productive yaw → box stalls and recoils.

## 4. State at first real-contact c3 admit per run

| run | step | ee | obj | nhat_BA_W | basin |
|---|---|---|---|---|---|
| 1 | **161** | (+0.000, +0.034, +0.059) | (−0.078, +0.009, +0.064) | (+0.892, **+0.286**, +0.350) | GOOD |
| 2 | **161** | (+0.002, +0.034, +0.054) | (−0.074, +0.007, +0.062) | (+0.904, **+0.309**, +0.295) | GOOD (best) |
| 3 | **194** | (−0.024, **−0.056**, +0.046) | (−0.097, −0.020, +0.059) | (+0.955, **−0.215**, +0.203) | BORDER |
| 4 | **192** | (−0.023, **−0.057**, +0.045) | (−0.095, −0.020, +0.058) | (+0.961, **−0.221**, +0.166) | BAD |

All runs hit the **same +x face** (correct face for pushing West). The discriminator is the **y-offset** on that face: good runs land contact at +y_offset (+0.286/+0.309); bad runs at −y_offset (−0.215/−0.221). Mirror-image lever arms → opposite box yaw rotation → opposite basin.

## 5. Phantom-c3 is NOT the root cause

The initial 18-tick phantom-c3 burst (steps 110–128) is bounded by the existing 5-tick disengage gate. Box stays static through step 130. ALL runs go through the same phantom-c3 burst — it does not discriminate good vs bad.

What this means for the candidates:
- **B1 (entry-side gate that delays c3 entry until real contact)**: would save the 18-tick phantom-c3 burst, but the burst itself does NOT cause the basin split. The lever happens 30 ticks later, in free-mode IK target selection. So B1 attacks a non-load-bearing surface.
- **B2 (in-c3 watchdog: exit c3 if drake-admit drops for K ticks)**: redundant with the existing `DISENGAGE_THRESHOLD=5` gate.
- **B3 (NEW candidate — sample-buffer / IK target stabilization)**: the root surface. Make the sample selection robust to sub-mm input noise so all runs pursue the SAME target at step 159, landing the same lever-arm on the +x face.

## 6. Threshold-pin findings (Phase A2 Step 3a)

`ee_to_surf` (= ‖ee − box_center‖_xy − box_half_extent_xy − pusher_radius) per run, at key ticks:

| run | at c3-entry (step 110/111) | at first real admit | gap |
|---|---|---|---|
| 1 | 4 mm | ≈ 0 mm | 4 mm |
| 2 | 4 mm | ≈ 0 mm | 4 mm |
| 3 | 4 mm | ≈ 0 mm | 4 mm |
| 4 | 4 mm | ≈ 0 mm | 4 mm |

**ee_to_surf is IDENTICAL across all 4 runs at c3-entry (4 mm).** There is NO separation in entry-distance between good and bad basin runs. The B1 candidate's placeholder `surface_eps_m = 5 mm` does NOT discriminate. The PIN-AGAINST-DATA discipline says: **B1 is not a viable candidate based on entry-distance — there is no gap to sit in.**

## 7. CORRECTED A3 routing recommendation — B3 (NEW), not B1 / B2

Both the originally-planned B1 (entry-side ee_to_surf gate) and B2 (in-c3 watchdog) attack surfaces that do not discriminate good vs bad runs in the setarch data. The actual discriminator is the IK target chosen at step ~159 in free (reposition) mode. The fix surface is **sample-buffer / sample-selection stabilization**:

- **B3a (cheap)**: deterministic seeding of the sample generator (`sampling.py:30 generate_samples`). The RNG passed in is currently re-seeded per-controller-instance with the user `--seed`, but may use the global numpy RNG state internally. Audit + force a per-tick deterministic seed derived from step number + user seed.
- **B3b (mechanism)**: investigate why `_face_normal_projection` (`sampling.py:146`) selects different y-offset samples for sub-mm-different box positions. The face selection logic + tangent jitter draws may amplify sub-mm noise into 30 mm target changes.
- **B3c (alternate)**: quantize sample positions to a stable lattice (e.g., snap tangent offsets to 1 cm grid) so sub-mm input noise produces the same sample, not different samples.

A4 (proposed extension): the audit at `sampling.py:146` may surface a direct cause (e.g., a sort by a noise-sensitive cost score that flips order when costs differ by less than the noise).

## 8. CRITICAL CAVEAT — this analysis falsifies the original plan's B1 candidate

The PIN-AGAINST-DATA discipline that the user flagged caught the original candidate. **B1's threshold cannot be pinned against measured good-vs-bad separation because there is no separation at the entry-distance metric.** Routing to B1 with a placeholder 5 mm value would have shipped a fix that doesn't address the actual lever. The mechanism investigation in this phase identified the correct surface (sample-buffer / IK target stabilization) — Phase B should be re-scoped to design candidate B3, with its own pre-registered success criteria.

---

## File:line citations

- `reposition_ik.py:1455` — `finished = finished_val <= 0.02` (IK 20 mm gate to setback target). **Not the lever** (timing of c3 entry doesn't discriminate).
- `lcs_formulator.py:245` — Drake's 2 mm `phi <= 0.002` admit gate. (Confirms phi=4 mm at entry → phantom prediction.)
- `wrapper.py:622-674` — existing 5-tick disengage gate. **Already handles** the phantom-c3 cap.
- `wrapper.py:365` — recoil-convention `_derive_force_command`. (Mechanism context, not the lever — box doesn't move during phantom-c3.)
- **`sampling.py:30`** — `generate_samples` dispatcher. **Suspected lever surface** (sample selection brittleness).
- **`sampling.py:146`** — `_face_normal_projection`. **Suspected lever surface** (tangent-offset choice on +x face).
- **`sample_buffer.py:73`** — `SampleBuffer` FIFO + pose-pruning. **Suspected lever surface** (buffer selection logic).
- `wrapper.py:884-988` — existing entry-gate stack. **Not the lever** for this brittleness.

## What's NOT yet measured

The full audit of `sampling.py:30-150` + `sample_buffer.py:73-150` is the next investigation step. The hypothesis to test in Phase A4 (extension): which line of code amplifies sub-mm box-position noise into a >30 mm sample-target shift?
