# Investigation: c3-exit dispatch defect pair (p133 → p136)

**Date:** 2026-07-28 (evening arc)
**Commits:** `f781a4d` (height frame-shift, context), `6a9a241` (defect 1), `b9218ab` (defect 2)
**Runs:** p133 (baseline), p134 (unmasking), p135 (falsification of fix 1), p136 (verification)
**Reference:** dairlib `push_anything_dev@257e3ed`,
`systems/controllers/sampling_based_c3_controller.cc` (cited below as `cc:`)
**Method:** systematic-debugging (root cause before fix; each fix TDD'd; each
claim log- or source-verified)

---

## 0. TL;DR

The sampling-height frame-shift fix (`f781a4d`, push_t 0.005 → 0.034) made
reposition arrivals reliable for the first time — and thereby unmasked two
stacked port-only defects in the c3→free exit path that together produced a
1-tick c3-dwell limit cycle (T pushed once per ~16-step repos cycle, goal_dist
frozen for 180 s). Both defects contradict the reference implementation
line-for-line and are now fixed and unit-guarded. p136 restores healthy
dispatch structure (1294 stay-ticks / 68 entries ≈ 19-tick dwells); end
metrics return to the accepted conformant band (trans 0.1986 / rot 0.4847).

---

## 1. Timeline of evidence

| run | HEAD | trans (m) | rot (rad) | c3 entries | kStayInC3 | signature |
|---|---|---|---|---|---|---|
| p133 | pre-height | 0.1843 | 0.3788 | 56 | 998 | arrival plateau 0.0204–0.0207 (325 steps) |
| p134 | `f781a4d` | 0.2050 | 0.6493 | 114 | 9 | 111/114 exits cite buffer promise; goal_dist frozen 0.150 |
| p135 | `6a9a241` | 0.1985 | 0.6255 | 121 | 0 | ALL 121 exits cite the SAME promise c=1454.81 |
| p136 | `b9218ab` | 0.1986 | 0.4847 | 68 | 1294 | 41 distinct promises; watchdog fires again; band metrics |

(`kStayInC3` counted on `[GS]` lines only. All runs: 180 s canonical flagless
protocol `scripts/run_t_r7_canonical.sh`, 0 crashes / 0 QP failures /
0 saturation.)

---

## 2. Context: what the height fix unmasked (p133 → p134)

p133's dominant phenotype was an **arrival plateau**: `tasks.yaml` still
forced `sampling_height: 0.005` (the reference value verbatim, but the
reference measures from its ground at z=−0.029 → 34 mm above ground; port
ground is z=0). The EE-center floor (tip r 0.0195) made the priced target
physically unreachable — 325 steps parked at `finished_val` 0.0204–0.0207
against the 0.020 threshold. Arrivals mostly failed, so the arrive → enter-c3
→ exit cycle rarely completed.

`f781a4d` set the per-task height to 0.034 (port frame) + separate
`z_height: 0.025` c3-track plane (`SamplingParams.z_height`, consumed via
`_c3_track_z()`). Arrivals became reliable: sub-threshold arrival steps
143 vs 78, c3 entries 114 vs 56. **The dispatcher's exit path now executed at
full frequency for the first time — and it was broken.**

## 3. Defect 1 — every c3 exit discards the target it exited for

**Symptom (p134):** step-32/33 window:
- step 32: enter c3 (`kToC3ReachedReposTarget`), real contact (`lam_n=32.8`),
  curr_cost 6685.
- step 33: exit `kToReposCost` — buffer sample promises 1326.49 vs live 6666
  minus hysteresis 2666. Free branch commits: `_current_repos_target = p_repos`
  (`sampling_based_c3_controller.py:2828`, `best_src=buffer`).
- step 34: `held_valid=N reason=no_held` — the committed target is GONE; a
  fresh strategy sample at ~7095 is picked somewhere else; EE lifts to 0.075
  and traverses away.

**Root cause:** the end-of-tick bookkeeping on every c3→free transition
(`sampling_based_c3_controller.py:4769-4788`, comment citing "Venkatesh 2025
§IV-D Step 4") ran AFTER the free-branch target selection in the same tick and:
1. `self._current_repos_target = None` — discarded the just-committed pursuit
   target (so no `prev_repos` slot next tick — `:956` requires it — and no
   repos_to_repos hysteresis anchor);
2. `self._refresh_buffer_on_arrival()` — cleared the strategy-sample
   persistence cache, forcing full re-sampling.

**Reference behavior (contradicts both):**
- cc:1845 `prev_repositioning_target_ = best_sample_location;` — the pursued
  target (incl. buffer-sourced) is RETAINED every repositioning tick;
- cc:914/932 — it re-enters the candidate set as `kCurrentReposTarget` next
  loop, freshly priced (cc:1148), protected by repos_to_repos hysteresis
  (cc:1204-1230), so the arm actually travels to the promise;
- cc:1186-1201 — on exit (all reasons): `finished_reposition_flag_ = false`,
  `ResetProgressMetrics()`, and if the pursued target is the buffer slot,
  remove THAT ONE sample from the buffer (cc:1196-1198). Nothing else cleared.

**Fix (`6a9a241`):** epilogue retains `_current_repos_target`/`_cost`; removes
only the pursued buffer entry (`best_src == "buffer"`) via new
`SampleBuffer.remove(entry)` (identity match; `sample_buffer.py`); keeps the
two conformant lines (`progress.reset()`, `_last_repos_finished = False` =
cc:1188-1189). `_refresh_buffer_on_arrival` keeps its arrival-site call
(`:2988`) only. The augmented entry is captured by identity at the
augmentation site (`_augmented_buffer_entry`) because `_update_buffer` runs
between augmentation and epilogue. New `[RICH-EXIT]` log line reports
`pursued_src` / `buf_removed`.

**Tests:** `tests/test_sample_buffer.py` — 3 new `remove()` tests
(failing-first).

## 4. Defect 2 — the augmented buffer promise self-replicates (immortal)

**Falsification (p135):** fix 1 verified active (121 `[RICH-EXIT]
target_retained=Y buf_removed=Y`) — yet ALL 121 exits across 180 s cited the
**exact same promise: c=1454.81**, the entry appended at step 2 at
(+0.470, +0.040, 0.034) (visible in the `[BUFFER-STATE]` dump). Removal fired
every time; the entry always returned. kStayInC3 = 0.

**Root cause:** feedback loop between augmentation and maintenance:
1. On every c3 tick, `AugmentSamplesWithBuffer`
   (`sampling_based_c3_controller.py:1509-1531`) appends the stored
   `BufferedSample.result` — the ORIGINAL `SampleResult` with the ORIGINAL
   cost — into the candidate `results` list (label `"buffer"`).
2. `_update_buffer` (`:2310`) runs AFTER augmentation and bulk-appends all
   non-current feasible results (`:1034-1042`, `cost = r.c_sample`) — including
   the stale re-injection. A clone enters the buffer the same tick.
3. Exit-side removal deletes one copy; the clone survives. The stale promise
   is immortal and ejects every subsequent c3 stint.

(Scale-mismatch hypothesis checked and rejected: push_t runs
`w_align = w_travel = w_rot = 0`, so `c_sample = c_C3_raw` — the stored costs
are honest for their append tick; staleness, not scale, is the poison.)

**Reference behavior:** call order makes this impossible —
cc:1094 `MaintainSampleBuffers` (prune + append, fresh samples only) runs
BEFORE cc:1097 `AugmentSamplesWithBuffer`. The maintenance loop never sees the
augmented entry; combined with cc:1196-1198 removal, a cached promise is
citable at most once.

**Fix (`b9218ab`):** `_update_buffer` takes `labels` and skips
`label == "buffer"` entries — behaviorally identical to the reference call
order without restructuring the tick pipeline.

**Tests:** `tests/test_update_buffer_no_reappend.py` (2 tests;
bare-controller harness via `SamplingC3Controller.__new__`).

## 5. Verification (p136)

- **Dwell restored:** 1294 kStayInC3 / 68 entries ≈ 19 ticks per stint
  (p135: 1). 66 cost exits + 2 `kToReposUnproductive` — the progress watchdog
  can fire again because stints now live long enough.
- **Promise economy fixed:** 41 distinct exit-promise values (p135: one);
  `buf_removed` mixed Y(39)/N(29) — a cited promise is consumed; later exits
  need fresh evidence.
- **Clean run:** 0 crashes / QP failures / saturation; avg 138 ms/step.
- **Metrics:** trans 0.1986 / rot 0.4847 — inside the accepted conformant band
  (p132: 0.186/0.709 @60 s). Rotation recovered most of the p134 regression
  (0.649 → 0.485). p133's 0.184/0.379 is not a valid target: its dispatch load
  was suppressed by the arrival-plateau bug (no-revert rule applies).

## 6. What remains (not defects)

1. **Optimistic cached promises still trigger exits** — reference-conformant
   by design (the reference pays the same tax; see accepted-band memory /
   phantom-park B-investigation). Exits are now *productive* (pursued, priced
   fresh, hysteresis-protected) rather than eliminated.
2. **Single-run Ipopt FP spread** moves results within the band
   (smoke @60 s vs p135 @180 s divergence pre-fix-2 was the documented
   nondeterminism cascading through sample jitter).
3. **Secondary observation from p134** (mostly dissolved by the fixes, worth a
   glance in future logs): stale-buffer targets whose xy falls inside the
   rotated T footprint park the EE on the T top (z = 0.040 + 0.0195 = 0.0595
   signature). Buffer pose-prune thresholds allow up to 5 cm / 0.30 rad drift.
4. **Unpinned minor:** the transient 0.039 waypoint z seen in p134 free-mode
   logs (5 mm above sampling_height) was never traced to a source; it did not
   recur as a park site in p136.

## 7. Guards against regression

- `tests/test_sample_buffer.py::test_remove_*` — one-entry identity removal.
- `tests/test_update_buffer_no_reappend.py` — augmented entry never
  re-appended; fresh strategy samples still appended.
- Memory: `dispatch-retention-fixes-p135-p136` — "never re-add
  transition-time buffer refresh/target clears."
- Pre-existing unrelated failures (10, in `test_mode_switch.py` /
  `test_commit_face_gate.py`) reproduce at parent HEAD — not introduced here.
