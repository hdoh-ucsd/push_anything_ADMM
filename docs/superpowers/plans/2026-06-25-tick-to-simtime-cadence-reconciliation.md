# Tick-to-Sim-Time Cadence Reconciliation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the port's tick-counted cadence constants in the dispatcher / sample-buffer / contact-loss disengage / reposition path to sim-time durations driven by elapsed sim_t, so reposition + dispatch behavior is rate-independent (the cadence discriminator's reposition-isolation guard holds at 1 kHz) AND the port's cadence SEMANTICS reconcile to the reference's sim-time-cadenced design.

**Architecture:** Two concrete classes of tick-coupled constants are surfaced by the audit. (Class A) constants that exist in BOTH port and reference but encode different sim-time intervals at different rates — reconcile by converting to a duration field driven by elapsed sim_t. (Class B) port-only counters (no reference equivalent — contact-loss disengage, sample-buffer lifetime, target-stable-ticks) — reconcile to a `duration_seconds` field whose default at 100 Hz equals the current `tick_count × 0.01 s`, so the 100 Hz baseline behavior is byte-equivalent (no-op smoke passes) and the same duration applies at 1 kHz (preserves the reposition-isolation guard). The shared mechanism is a single helper on the wrapper that converts each tick-counted age increment into a sim_t-based elapsed comparison (`elapsed_s ≥ duration_s` rather than `_age++ ≥ lifetime`). The `PUSHA_CONTROL_HZ` gate (commit `02abed9`) stays untouched as the rate driver.

**Tech Stack:** Python (no new dependencies). Existing dataclasses (`SamplingC3Params`, `SamplingParams`, `ProgressParams`, `RepositionIKParams`), the sample buffer, the wrapper dispatcher, the IK reposition tracker, the YAML loader.

**Spec source:** `docs/superpowers/plans/2026-06-23-alignment-phase-plan.md` §1 row 8 (Entry & cadence ACTIVE FRONT, sub-divergence (a) tick-vs-sim-time semantics) + §7.3 (Cadence discriminator STOP-AT-SCOPE: tick-vs-sim-time semantics is a separate alignment target). This plan operationalizes the §7.3 direction decision (reconcile properly, NOT a scaled-gate hack).

**Branch / HEAD at plan time:** `reference-conformance` @ `308f188` (cadence smoke STOP-AT-SCOPE record, with `PUSHA_CONTROL_HZ` gate at `02abed9`).

**REVIEW LABEL ADJUSTMENTS (2026-06-25, see canonical plan §7.4):** Class B (B1-B11) is RATE-INDEPENDENCE-ONLY (no reference analog; alignment-status-OPEN), NOT "reconciled." Class A constants are PRESERVED at the port's 100 Hz sim-time-equivalent on this pass (reference's 5/5/16 ms values are a separate later alignment decision). A3 LOCATED on the reference at `anything/parameters/progress_params_c3plus.yaml:45 = 16 ticks = 16 ms`; port preserves 300 ms. §1 row 8 sub-(a) flip narrows to "tick→sim-time semantics RATE-INDEPENDENT" — explicitly NOT "Class B reconciled" or "Class A dispatch-timing reconciled."

**Carry-forward (binding on this plan and the implementation block that follows):**
- `REFCONF_REPOSITION_PWL=1`, `pwl_speed=0.18` held.
- `PORT_FORCE_ROUTING=u_sol`, `W_force=100` held.
- `PUSHA_CONTROL_HZ` env gate stays (default 100 → no behavior change; 1000 → 1 kHz for the verification smoke once reconciliation lands).

---

## 0. PART 1 — AUDIT (output of the read-only enumeration)

The full set of tick-coupled constants found in the dispatcher / buffer / disengage / reposition path, grouped by class.

### Class A — tick-counted in BOTH port and reference; sim-time interval differs

| # | Constant | Port file:line | Port default | Port sim-t @ 100 Hz | Reference file:line | Reference default | Reference sim-t @ 1 kHz |
|---|---|---|---|---|---|---|---|
| A1 | `num_control_loops_to_wait` | `params.py:150` + `progress.py:167` | 60 | **600 ms** | `progress_params.h:53` + `anything/parameters/progress_params_c3plus.yaml:40` | 5 (anything/c3plus); 14 (jacktoy) | **5 ms** (anything); 14 ms (jacktoy) |
| A2 | `num_control_loops_to_wait_position` | `params.py:151` + `progress.py:167` | 30 | 300 ms | `progress_params.h:54` + same YAML | 5 (anything/c3plus); 20 (jacktoy) | 5 ms (anything); 20 ms (jacktoy) |
| A3 | `progress_enforced_over_n_loops` | `params.py:165` | 30 | 300 ms | `progress_params.h` (same struct family) | (lookup pending — same yaml family) | (same family) |
| A4 | `best_progress_steps_ago_` counter | port: derived inside `progress.py` (`steps_since_improve`); reference: `sampling_based_c3_controller.cc:2286-2313 best_progress_steps_ago_++` | per-call ++ | (depends on A1) | per-call ++ (cc:2288) | (depends on A1) | (depends on A1) |

**Class-A reconciliation target choice (FLAG):** the reference's `num_control_loops_to_wait = 5 @ 1 kHz = 5 ms` for the `anything/c3plus` task is **much shorter in sim-time** than the port's 600 ms. Adopting the reference value at the port's 100 Hz would give 0.5 ticks — semantically nonsense and a huge behavioral change for the port (cuts the progress-timeout window from 600 ms to 5 ms). **The plan elects the rate-independent-from-100Hz preservation** for Class A on the first reconciliation pass: keep the current sim-time interval at 100 Hz (`A1 = 600 ms`, `A2 = 300 ms`, `A3 = 300 ms`), making the 100 Hz baseline byte-equivalent. A SEPARATE follow-on alignment block can choose to additionally align to the reference's sim-time value once we've verified the rate-independent rewrite works in both directions.

### Class B — port-only counters (no reference equivalent)

| # | Constant | Port file:line | Port default (ticks) | Port sim-t @ 100 Hz | Reference equivalent | Reconciliation target |
|---|---|---|---|---|---|---|
| B1 | `sample_buffer_lifetime` | `params.py:291` + `config/sampling_c3_kik.yaml:188` | 30 | 300 ms | none (reference uses event-driven sample buffer via output ports `sample_buffer_configurations_port_`, `sampling_based_c3_controller.cc:358-388`) | preserve 300 ms |
| B2 | `_sample_buffer_age` (counter) | `sampling_based_c3_controller.py:343, 423` (++) | per-call ++ | (depends on B1) | none | replace with `_sample_buffer_built_sim_t` + `elapsed_s` check |
| B3 | `contact_loss_threshold_default` (`DISENGAGE_THRESHOLD` family) | `params.py:693`, used at `sampling_based_c3_controller.py:1286, 1289` | 5 | **50 ms** | none (reference has no disengage counter — grep `disengage|streak|consecutive|no_contact` in `sampling_based_c3_controller.cc` returned 0 hits) | preserve 50 ms |
| B4 | `contact_loss_threshold_with_override` | `params.py:694`, used at `sampling_based_c3_controller.py:1284` | 12 | 120 ms | none | preserve 120 ms |
| B5 | `contact_loss_threshold_phaseA_ltd` | `params.py:705`, used at `sampling_based_c3_controller.py:1269` | 120 | 1.2 s | none | preserve 1.2 s |
| B6 | `contact_loss_threshold_phaseB_ltd` | `params.py:715`, used at `sampling_based_c3_controller.py:1271` | 300 | 3.0 s | none | preserve 3.0 s |
| B7 | `phaseC_hard_cap` | `params.py:736`, used at `sampling_based_c3_controller.py:1282` | 100 | 1.0 s | none | preserve 1.0 s |
| B8 | `phaseC_stall_threshold` | `params.py:735, 859` | 30 | 300 ms | none | preserve 300 ms |
| B9 | `_no_ee_box_streak` (counter) | `sampling_based_c3_controller.py:251 (init), :1289 (read), :1298/:1313/:1519 (reset to 0), :1532 (++), :2499 (reset to 0)` | per-call ++ | (depends on B3-B7) | none | replace with `_no_ee_box_streak_started_sim_t` + `elapsed_s` check |
| B10 | `TARGET_STABLE_TICKS` | `reposition_ik.py:709 self.TARGET_STABLE_TICKS=5` | 5 | 50 ms | none | preserve 50 ms |
| B11 | `_target_stable_ticks` (counter) | `reposition_ik.py:711, :1269, :1282, :1286, :1354` | per-call ++ | (depends on B10) | none | replace with `_target_stable_started_sim_t` + `elapsed_s` check |

### Class C — out-of-scope for this reconciliation (NOT touched)

| # | Constant | Why excluded |
|---|---|---|
| C1 | `watchdog_steps_since_improve_threshold` | port-only, default 0 (DISABLED). Reconcile when/if enabled. |
| C2 | `dt_ctrl/dt_osc` coupling at `main.py:511` | the rate driver itself; PUSHA_CONTROL_HZ gate (commit 02abed9) already controls this. Not a behavior counter. |
| C3 | `arm_obstacle_count` (`reposition_ik.py:313, 327`) | single-call counter; loop variable inside IK helper; not a cross-call dispatch counter. |

### Audit verdict on plan §7.3's "known three" expansion

Plan §7.3 named three known tick-coupled constants: (i) dt_ctrl/dt_osc coupling, (ii) sample_buffer_lifetime, (iii) the 5-tick contact-loss disengage. The audit **EXPANDED** the scope from three to **fifteen tick-counted constants in two classes** (Class A: 4; Class B: 11). This is the §7.3-mandated FLAG: the scope is wider than the initial three. The wider scope is contained — all fifteen reside in 4 files (`params.py`, `sample_buffer.py`, `sampling_based_c3_controller.py`, `reposition_ik.py`) + one YAML (`config/sampling_c3_kik.yaml`).

---

## 1. PART 2 — REFERENCE-READ (per-constant sim-time cadence target)

For each constant in Class A, the reference's sim-time cadence target is set from the reference YAML (the `anything/c3plus` task — the closest analog to the port's task) at the reference's 1 kHz cadence. For each constant in Class B, the reference has no equivalent → the plan adopts the rate-independent-from-100Hz fallback per the §7.3 direction (preserve the current 100 Hz sim-time interval; this is also the second-best fallback the §7.3 prompt named "old_tick_count × 10 ms").

### Class A targets

- **A1 `num_control_loops_to_wait`** — reference value `5 @ 1 kHz = 5 ms` (FLAGGED 100×-shorter than the port's 600 ms; the plan PRESERVES the port's 600 ms on this pass to keep the 100 Hz baseline byte-equivalent; a separate alignment block can choose to additionally adopt the 5 ms reference value after the rate-independent rewrite is verified).
- **A2 `num_control_loops_to_wait_position`** — reference value `5 @ 1 kHz = 5 ms` (same FLAG; PRESERVE 300 ms).
- **A3 `progress_enforced_over_n_loops`** — reference target NOT directly grepped under this name (FLAG: lookup may require a deeper trace into the reference's progress-cost variant). PRESERVE 300 ms.
- **A4 `best_progress_steps_ago_` counter** — the SHAPE of the counter (per-call ++) is shared with the reference; only its tick→sim-time semantics change. PRESERVE 100 Hz sim-time interval.

### Class B targets

All Class B entries — `sample_buffer_lifetime` (300 ms), the contact-loss disengage family (50 ms / 120 ms / 1.2 s / 3.0 s / 1.0 s / 300 ms), and `TARGET_STABLE_TICKS` (50 ms) — adopt the 100 Hz sim-time-equivalent as the duration default. This makes the 100 Hz baseline byte-equivalent (the no-op smoke passes) and gives the same duration at 1 kHz (so the reposition-isolation guard holds: a 1.4 s PWL trajectory can land before a 300 ms sample-buffer refresh fires at 1 kHz, exactly as it does at 100 Hz).

### Reference-mechanism note (informational, not actioned this pass)

The reference's sample buffer is event-driven via output ports (`sample_buffer_configurations_port_` etc.) in a multi-process LCM pipeline; the port's single-loop architecture cannot adopt that mechanism without the Stage F multi-process port. The plan's Class B reconciliation keeps the port's existing single-loop refresh trigger (sample-buffer-lifetime as a duration); the multi-process port is row-8 sub-divergence (b), deferred per §7.3.

---

## 2. PART 3 — CONVERSION SPEC (file:line edits — AUTHORED, not applied)

### 2.1 Sample buffer (B1 + B2)

**File: `control/sampling_c3/params.py`**

- `params.py:291` — change the field type and rename:
  ```python
  # BEFORE:
  sample_buffer_lifetime:              int   = 30
  # AFTER:
  sample_buffer_lifetime_s:            float = 0.30   # seconds (was 30 ticks @ 100 Hz)
  ```
- `params.py:from_dict` for `SamplingParams` (find the matching parser block — `_filter_kwargs` should pick up the rename if YAML is updated; otherwise add an explicit alias-handling line). Add a YAML-back-compat shim: if YAML has `sample_buffer_lifetime` (int), warn and convert to `sample_buffer_lifetime_s = value * 0.01`; otherwise read `sample_buffer_lifetime_s` directly.

**File: `config/sampling_c3_kik.yaml`**

- Line 188 — replace:
  ```yaml
  # BEFORE:
  sample_buffer_lifetime:      30
  # AFTER:
  sample_buffer_lifetime_s:    0.30
  ```

**File: `control/sampling_c3/sampling_based_c3_controller.py`**

- `:343` — change init:
  ```python
  # BEFORE:
  self._sample_buffer_age:        int                  = 0
  # AFTER:
  self._sample_buffer_built_sim_t: Optional[float]     = None
  ```
- `:384` — read lifetime from new field:
  ```python
  # BEFORE:
  lifetime = int(getattr(sp, "sample_buffer_lifetime", 0))
  # AFTER:
  lifetime_s = float(getattr(sp, "sample_buffer_lifetime_s", 0.0))
  ```
- `:402` — change the refresh trigger:
  ```python
  # BEFORE:
  or self._sample_buffer_age >= lifetime
  # AFTER:
  or (self._sample_buffer_built_sim_t is None
      or (self._step * self._dt_ctrl) - self._sample_buffer_built_sim_t >= lifetime_s)
  ```
- `:416` — change the reset on refresh:
  ```python
  # BEFORE:
  self._sample_buffer_age = 0
  # AFTER:
  self._sample_buffer_built_sim_t = float(self._step * self._dt_ctrl)
  ```
- `:423` — REMOVE the per-call increment line (no longer needed; the elapsed_s comparison handles it):
  ```python
  # REMOVE:
  self._sample_buffer_age += 1
  ```
- `:438` — change the force-refresh marker:
  ```python
  # BEFORE:
  self._sample_buffer_age = lifetime + 1
  # AFTER:
  # Force refresh on next call: leave built_sim_t as is; subtract a marker.
  self._sample_buffer_built_sim_t = -float("inf")  # any past sim_t > lifetime_s ago
  ```

**File: `control/sampling_c3/sample_buffer.py`**

- `:43, :50, :120-123` — the `age_steps` field + `tick_age()` method are now dead code (the wrapper's refresh trigger drives lifetime). DELETE `age_steps` field; DELETE `tick_age()` method; remove the doc references at `:2-4` to "control loops".

### 2.2 Contact-loss disengage (B3-B7 + B9)

**File: `control/sampling_c3/params.py`**

- `:693-694, 705, 715, 736` — convert all five contact-loss-threshold fields to seconds:
  ```python
  # BEFORE:
  contact_loss_threshold_default:           int = 5
  contact_loss_threshold_with_override:     int = 12
  contact_loss_threshold_phaseA_ltd:        int = 120
  contact_loss_threshold_phaseB_ltd:        int = 300
  phaseC_hard_cap:                          int = 100
  # AFTER:
  contact_loss_threshold_default_s:         float = 0.05   # was 5  ticks  @ 100 Hz
  contact_loss_threshold_with_override_s:   float = 0.12   # was 12 ticks  @ 100 Hz
  contact_loss_threshold_phaseA_ltd_s:      float = 1.20   # was 120 ticks @ 100 Hz
  contact_loss_threshold_phaseB_ltd_s:      float = 3.00   # was 300 ticks @ 100 Hz
  phaseC_hard_cap_s:                        float = 1.00   # was 100 ticks @ 100 Hz
  ```
- `:855-860 from_dict` — update raw.get keys and types accordingly; add YAML back-compat shim (warn and convert if the int form is found).

**File: `control/sampling_c3/sampling_based_c3_controller.py`**

- `:251` — change the init:
  ```python
  # BEFORE:
  self._no_ee_box_streak:         int   = 0
  # AFTER:
  self._no_ee_box_streak_started_sim_t: Optional[float] = None
  # (None = not currently streaking)
  ```
- `:1269-1286` — replace tick-threshold reads:
  ```python
  # BEFORE: read tick threshold
  disengage_threshold = self.params.contact_loss_threshold_phaseA_ltd
  # AFTER: read seconds threshold
  disengage_threshold_s = self.params.contact_loss_threshold_phaseA_ltd_s
  ```
  (apply to all five variants on those lines)
- `:1289` — replace the comparison:
  ```python
  # BEFORE:
  and self._no_ee_box_streak >= disengage_threshold
  # AFTER:
  and self._no_ee_box_streak_started_sim_t is not None
  and (self._step * self._dt_ctrl) - self._no_ee_box_streak_started_sim_t >= disengage_threshold_s
  ```
- `:1294` — update the log message format to print elapsed seconds instead of ticks.
- `:1298, 1313, 1519, 2499` — change the reset-to-0 to reset-to-None:
  ```python
  # BEFORE:
  self._no_ee_box_streak = 0
  # AFTER:
  self._no_ee_box_streak_started_sim_t = None
  ```
- `:1532` — change the per-call ++:
  ```python
  # BEFORE:
  self._no_ee_box_streak += 1
  # AFTER:
  if self._no_ee_box_streak_started_sim_t is None:
      self._no_ee_box_streak_started_sim_t = float(self._step * self._dt_ctrl)
  ```

### 2.3 `phaseC_stall_threshold` (B8)

**File: `control/sampling_c3/params.py`**

- `:735` and `:859` — convert:
  ```python
  # BEFORE:
  phaseC_stall_threshold: int = 30
  ...
  phaseC_stall_threshold = int(raw.get("phaseC_stall_threshold", 30)),
  # AFTER:
  phaseC_stall_threshold_s: float = 0.30
  ...
  phaseC_stall_threshold_s = float(raw.get("phaseC_stall_threshold_s", 0.30)),
  ```
- Update each call site of `phaseC_stall_threshold` (TBD — find via `grep -n phaseC_stall_threshold control/sampling_c3/sampling_based_c3_controller.py`; same pattern as B9: comparator-and-counter pair).

### 2.4 Reposition IK `TARGET_STABLE_TICKS` (B10 + B11)

**File: `control/sampling_c3/reposition_ik.py`**

- `:709` — change the constant:
  ```python
  # BEFORE:
  self.TARGET_STABLE_TICKS: int   = 5
  # AFTER:
  self.TARGET_STABLE_DURATION_S: float = 0.05   # was 5 ticks @ 100 Hz
  ```
- `:711` — change the counter init:
  ```python
  # BEFORE:
  self._target_stable_ticks: int  = 0
  # AFTER:
  self._target_stable_started_sim_t: Optional[float] = None
  ```
- `:1269, :1282` — reset-to-None instead of reset-to-0.
- `:1286` — set-started-sim-t instead of ++.
- `:1354` — change the comparison from ticks-vs-threshold to elapsed-vs-duration_s. **NOTE:** `compute_torque` does NOT currently receive sim_t; either pass it via the method signature OR accumulate `dt_ctrl` per call into `self._sim_t_accumulator`. Either approach is contained in this file. PLAN ELECTS the dt_ctrl-accumulator (no signature change required).

### 2.5 Progress-loops fields (A1-A4)

**File: `control/sampling_c3/params.py`**

- `:150-151` — convert:
  ```python
  # BEFORE:
  num_control_loops_to_wait:           int   = 60
  num_control_loops_to_wait_position:  int   = 30
  # AFTER:
  num_control_loops_to_wait_s:         float = 0.60   # was 60 ticks @ 100 Hz
  num_control_loops_to_wait_position_s: float = 0.30  # was 30 @ 100 Hz
  ```
- `:165` — same for `progress_enforced_over_n_loops` → `progress_enforced_over_duration_s = 0.30`.
- `from_dict` — same YAML back-compat shim pattern.

**File: `config/sampling_c3_kik.yaml`**

- `:82, :83, :97` — convert to the seconds form.

**File: `control/sampling_c3/progress.py`**

- `:120-121` — `cap` uses the two `num_control_loops_to_wait*` values; convert to seconds (cap = `max(s, s_position)`).
- `:167-168` — change comparator to elapsed-vs-duration_s.
- `:203 steps_since_improve()` returns a tick count today. **Add** `elapsed_since_improve_s()` returning seconds. Update the `met_progress` call site (`sampling_based_c3_controller.py:NNN` — find via grep; same dispatcher path) to use the seconds variant.

### 2.6 YAML back-compat shim (applies to all `int → float` renamed fields)

In every `from_dict` block that reads a renamed field, add a one-line back-compat path:

```python
# At the top of from_dict (illustrative pattern; apply per field):
if "sample_buffer_lifetime" in raw and "sample_buffer_lifetime_s" not in raw:
    print(f"[YAML-COMPAT] sample_buffer_lifetime={raw['sample_buffer_lifetime']} "
          f"(ticks) → sample_buffer_lifetime_s={raw['sample_buffer_lifetime'] * 0.01:.4f} s",
          flush=True)
    raw["sample_buffer_lifetime_s"] = float(raw["sample_buffer_lifetime"]) * 0.01
    del raw["sample_buffer_lifetime"]
```

Apply this pattern for every constant in §2.1-§2.5 that was renamed.

### 2.7 What this plan does NOT change

- `main.py:557 dt_ctrl=0.01` (already gated by `PUSHA_CONTROL_HZ`, commit 02abed9). NOT touched.
- `watchdog_steps_since_improve_threshold` (default 0 = disabled). NOT touched. Audit Class C1.
- The `--c3plus-projection {componentwise, lcp}` flag. NOT touched. Cadence reconciliation is orthogonal to the projection question.
- Any §1 row status outside row 8. Row 8 sub-divergence (a) flips RECONCILED on Task 4's verification pass; rows 2/3/4/5/Stage E remain blocked on the cadence discriminator (sub-divergence (b)) until that runs on the reconciled base.

### 2.8 Edits surface summary

| Class | Files modified |
|---|---|
| A | `control/sampling_c3/params.py`, `control/sampling_c3/progress.py`, `control/sampling_c3/sampling_based_c3_controller.py`, `config/sampling_c3_kik.yaml` |
| B | `control/sampling_c3/params.py`, `control/sampling_c3/sample_buffer.py`, `control/sampling_c3/sampling_based_c3_controller.py`, `control/sampling_c3/reposition_ik.py`, `config/sampling_c3_kik.yaml` |
| Total unique files | 5 (4 source + 1 YAML) |

---

## 3. PART 4 — VERIFICATION DESIGN (smokes — DESIGNED, not run)

### 3.1 100 Hz no-op smoke (the byte-or-distribution-equivalence guard)

**Premise.** Every Class B duration_s default is set to the current `tick_count × 0.01 s`, so at 100 Hz the new elapsed-s comparator fires at the exact same control tick as the old tick-counter comparator. The 100 Hz baseline behavior should be byte-equivalent OR distribution-equivalent (modulo any FP rounding at the comparator boundary).

**Smoke run:** `seed 0, 12 s, --c3plus-projection=componentwise (default), PUSHA_CONTROL_HZ unset (= 100), REFCONF_REPOSITION_PWL=1, PORT_FORCE_ROUTING=u_sol, --ee-space --solver c3plus --admm-iter 25 --sampling-c3 config/sampling_c3_kik.yaml`.

**Compare against:** `stage_a_speed018/seed0/run.log` (the existing 100 Hz componentwise baseline).

**Pass bar (HARD):**
1. `[STAGE-A-TRACE] step=N` timing of first c3 entry within ±2 ticks of baseline's `step=173`.
2. First c3 entry's `phi` within ±0.1 mm of baseline's `+4.94 mm` (the reposition lands at the same place).
3. `[GS-perf] switches=N` within ±2 of baseline's `switches=16`.
4. `[STAGE-A-TRACE] mode` per-step distribution: chi-square or KS test against baseline (or simpler: per-tick mode-match rate ≥ 95 %).
5. Final `[RESULT] final_obj_xy` within ±0.5 mm of baseline `(-0.0007, 0.0102)`.

**Wired-signal check (anti-no-op-bypass):** at least one of `[STAGE-A-PWL] step=N build` log lines must fire at a step count consistent with the OLD 300 ms sample-buffer-lifetime interval (every ~30 ticks), proving the refresh path actually traversed the new elapsed-s comparator (not a side-bypass).

### 3.2 1 kHz reconciliation smoke (the cadence-discriminator-unblocker)

**Premise.** With every Class B comparator now duration-based, at 1 kHz the sample-buffer-refresh interval should be 300 ms (300 ticks), not 30 ms (30 ticks), preserving the PWL trajectory's ability to land. `[GS-perf] switches > 0` (c3 mode engaged) and the actual landing φ should match the 100 Hz baseline.

**Smoke run:** same as 3.1 but with `PUSHA_CONTROL_HZ=1000`.

**Pass bar (HARD):**
1. `[GS-perf] switches > 0` (c3 mode engaged — THE thing that failed in the STOP-AT-SCOPE smoke).
2. First c3 entry's `phi` within ±0.2 mm of baseline's `+4.94 mm` (the reposition lands the SAME PLACE — the isolation guard).
3. `[STAGE-A-PWL] build` log lines fire every ~300 ticks (sim-time-consistent 300 ms refresh, NOT the 30 ms scope-stop pattern).
4. `[STAGE-A-TRACE] finished_repos=1` reached within 5 % of the sim-time at which the 100 Hz baseline first reaches `finished_repos=1` (~1.7 s). At 1 kHz that's step ~1700 ± 5 %.

**Wired-signal check (anti-discriminator-confound):** the comparator firing time is sim-time-based, not tick-based. Verify by examining the `[STAGE-A-PWL] step=N build` step-count delta: at 1 kHz between consecutive rebuilds the delta must be ~300 ticks (within ±10 %), not 30 ticks (the scope-stop pattern) AND not 3000 ticks (the rate-ignoring pattern that would mean the comparator didn't see dt_ctrl).

### 3.3 What both smokes prove together

| Smoke | What it proves | Why it matters |
|---|---|---|
| 100 Hz no-op | The reconciliation is byte/distribution equivalent to the working baseline → no regression in the working path. | Guards the Stage A flag-ON validated behavior. |
| 1 kHz reconciliation | The reposition-isolation guard now holds at 1 kHz → c3 mode engages → the cadence discriminator can run cleanly. | Unblocks the next gate (cadence discriminator on the reconciled base). |

**BOTH passing ⇔ sub-divergence (a) tick-vs-sim-time semantics is RECONCILED.** §1 row 8 sub-divergence (a) flips RECONCILED on this pass. Sub-divergence (b) rate/architecture stays ACTIVE FRONT (the discriminator + Stage F multi-process are the next gates).

### 3.4 Test-design tooling (existing, REUSED)

- The `[STAGE-A-TRACE]` per-step emit at `sampling_based_c3_controller.py:2737` is the substrate for §3.1 pass-bar items 1, 2, 4 and §3.2 items 1, 2.
- The `[STAGE-A-PWL] build` emit at `sampling_based_c3_controller.py` PWL-branch is the substrate for §3.1 wired-signal and §3.2 items 3, wired-signal.
- The `[GS-perf]` summary at run end is the substrate for §3.1 item 3 and §3.2 item 1.
- The `[RESULT]` summary at run end is the substrate for §3.1 item 5.
- A new tiny Python script `scripts/_stage_c_reconciliation_smoke_compare.py` (~80 lines) parses these and computes the pass-bar deltas; written in §3.5.

### 3.5 New comparator script (file structure — full code in implementation block)

**Create:** `scripts/_stage_c_reconciliation_smoke_compare.py`

**Inputs:** two log paths (100 Hz baseline + 100 Hz no-op smoke) OR (100 Hz baseline + 1 kHz reconciliation smoke).

**Outputs:** a JSON verdict with per-pass-bar pass/fail. Exit 0 if all pass, non-zero otherwise.

**Behavior:**
- Parse `[STAGE-A-TRACE]` step + mode + phi for both logs.
- Parse `[STAGE-A-PWL] step=N build` step counts for both logs.
- Parse `[GS-perf] switches=N` from both logs.
- Parse `[RESULT] final_obj_xy=(X, Y)` from both logs.
- Compute the 5 pass-bar items per §3.1 OR §3.2 (selected by a `--mode {100hz-noop, 1khz-recon}` flag).
- Print a JSON dict + summary line + exit code.

(Full source code is written in the implementation block's Task 5 Step 2.)

---

## 4. PART 5 — DOWNSTREAM (stated, NOT actioned in this plan)

Once Tasks 1-7 below land + both smokes pass:

1. **§1 row 8 sub-divergence (a) tick-vs-sim-time semantics → RECONCILED.** The cell text is updated; sub-divergence (b) stays ACTIVE FRONT. Anti-stale per §7 — the row update is part of the verification commit's definition of done.
2. **The cadence discriminator (componentwise @ 1 kHz)** runs on the reconciled base in the FOLLOWING block. That run finally answers the projection-vs-cadence question §7.2 left open.
3. **`--c3plus-projection=lcp` default flip, {0, 4} validation, §1 rows 4/5 RECONCILED flip, Stage D advance / N-closest port** — still HELD per §7.2/§7.3 holds; await the cadence discriminator outcome.

**NOT actioned this block.** Implementation Tasks 1-7 below are authored; the implementation BLOCK runs after the user reviews this plan.

---

## 5. File structure

**Created (1 source + 1 test):**
- `scripts/_stage_c_reconciliation_smoke_compare.py` — the verification smoke comparator (Part 4 §3.5).
- `tests/test_tick_to_simtime_yaml_compat.py` — back-compat shim unit test (Task 6).

**Modified (5 files):**
- `control/sampling_c3/params.py` — 7 dataclass fields renamed; `from_dict` shims added (3 dataclasses: `SamplingParams`, `ProgressParams`, `RepositionIKParams` — also a 4th if `phaseC_stall_threshold_s` is in a separate dataclass).
- `control/sampling_c3/progress.py` — 3 internal field reads converted from ticks to seconds; add `elapsed_since_improve_s()` and switch the wrapper call site.
- `control/sampling_c3/sample_buffer.py` — delete dead `age_steps` + `tick_age()`; update class docstring to remove "control loops" language.
- `control/sampling_c3/sampling_based_c3_controller.py` — sample-buffer refresh trigger + contact-loss disengage + watchdog_? counter ports.
- `control/sampling_c3/reposition_ik.py` — `TARGET_STABLE_TICKS` → `TARGET_STABLE_DURATION_S`; `_target_stable_ticks` counter → `_target_stable_started_sim_t` + dt_ctrl accumulator.
- `config/sampling_c3_kik.yaml` — 4 fields renamed (sample_buffer_lifetime, num_control_loops_to_wait, num_control_loops_to_wait_position, progress_enforced_over_n_loops); units suffix added.
- `docs/superpowers/plans/2026-06-23-alignment-phase-plan.md` — §1 row 8 + §7.3 status updates on pass.

---

## 6. Tasks (the implementation drill-down — to execute in the FOLLOW-ON block after this plan is reviewed)

### Task 1: Sample buffer reconciliation (B1 + B2)

**Files:**
- Modify: `control/sampling_c3/params.py:291` + `from_dict` for `SamplingParams`
- Modify: `control/sampling_c3/sampling_based_c3_controller.py:343, 384, 402, 416, 423, 438`
- Modify: `control/sampling_c3/sample_buffer.py:43, 50, 120-123, docstring`
- Modify: `config/sampling_c3_kik.yaml:188`

- [ ] **Step 1: Rename the dataclass field + add back-compat shim.**
- [ ] **Step 2: Update the YAML.**
- [ ] **Step 3: Update the wrapper's refresh trigger to elapsed-s semantics.**
- [ ] **Step 4: Delete dead `age_steps` field + `tick_age()` method in `sample_buffer.py`.**
- [ ] **Step 5: Update docstring references to "control loops".**
- [ ] **Step 6: Run AST syntax check + python -c imports for all four files.**
- [ ] **Step 7: Commit.**

```bash
git add control/sampling_c3/params.py control/sampling_c3/sampling_based_c3_controller.py control/sampling_c3/sample_buffer.py config/sampling_c3_kik.yaml
git commit -m "tick→sim-t reconciliation: sample buffer lifetime (B1+B2)"
```

(Full code blocks — including the exact before/after snippets — are in §2.1 above.)

### Task 2: Contact-loss disengage reconciliation (B3-B7 + B9)

**Files:**
- Modify: `control/sampling_c3/params.py:693-694, 705, 715, 736, 855-860`
- Modify: `control/sampling_c3/sampling_based_c3_controller.py:251, 1269-1286, 1289, 1294, 1298, 1313, 1519, 1532, 2499`

Apply §2.2's edits step-by-step, then commit.

### Task 3: `phaseC_stall_threshold` reconciliation (B8)

**Files:**
- Modify: `control/sampling_c3/params.py:735, 859`
- Modify: `control/sampling_c3/sampling_based_c3_controller.py` (call sites — find via grep)

Apply §2.3's edits.

### Task 4: Reposition IK `TARGET_STABLE_TICKS` (B10 + B11)

**Files:**
- Modify: `control/sampling_c3/reposition_ik.py:709, 711, 1269, 1282, 1286, 1354`

Apply §2.4's edits, including the dt_ctrl accumulator strategy. Run `python -c 'import control.sampling_c3.reposition_ik'` after to verify no syntax errors.

### Task 5: Progress-loops fields + comparator script (A1-A4 + Part 4)

**Files:**
- Modify: `control/sampling_c3/params.py:150, 151, 165, from_dict`
- Modify: `control/sampling_c3/progress.py:120, 121, 167, 168, 203`
- Modify: `config/sampling_c3_kik.yaml:82, 83, 97`
- Modify: `control/sampling_c3/sampling_based_c3_controller.py` (the progress call site — find via grep `steps_since_improve\|met_progress`)
- Create: `scripts/_stage_c_reconciliation_smoke_compare.py`

Apply §2.5's edits. Then author the comparator script per §3.5.

### Task 6: YAML back-compat unit test

**Files:**
- Create: `tests/test_tick_to_simtime_yaml_compat.py`

- [ ] **Step 1: Write tests that load OLD-style YAML (sample_buffer_lifetime: 30) and verify it parses to the new field `sample_buffer_lifetime_s = 0.30` with a `[YAML-COMPAT]` log line.**

```python
import pytest, yaml, io, sys
from control.sampling_c3.params import SamplingC3Params

def test_old_sample_buffer_lifetime_int_converts_to_seconds(capsys):
    yaml_text = """
sampling_params:
  sample_buffer_lifetime: 30
reposition_params: {}
"""
    cfg = SamplingC3Params.from_dict(yaml.safe_load(yaml_text))
    assert cfg.sampling_params.sample_buffer_lifetime_s == pytest.approx(0.30)
    captured = capsys.readouterr()
    assert "YAML-COMPAT" in captured.out

def test_new_sample_buffer_lifetime_s_float_loads_directly():
    yaml_text = """
sampling_params:
  sample_buffer_lifetime_s: 0.45
reposition_params: {}
"""
    cfg = SamplingC3Params.from_dict(yaml.safe_load(yaml_text))
    assert cfg.sampling_params.sample_buffer_lifetime_s == pytest.approx(0.45)

def test_old_contact_loss_int_converts_to_seconds(capsys):
    yaml_text = """
sampling_params: {}
reposition_params: {}
contact_loss_threshold_default: 5
"""
    cfg = SamplingC3Params.from_dict(yaml.safe_load(yaml_text))
    assert cfg.contact_loss_threshold_default_s == pytest.approx(0.05)
```

- [ ] **Step 2: Run the test, watch it fail before the implementation changes land.**

Run: `pytest tests/test_tick_to_simtime_yaml_compat.py -v`
Expected: 3 FAILED (the new field names don't exist yet).

- [ ] **Step 3: After Tasks 1-5 land, re-run.**

Expected: 3 PASSED.

- [ ] **Step 4: Commit.**

```bash
git add tests/test_tick_to_simtime_yaml_compat.py
git commit -m "tick→sim-t reconciliation: YAML back-compat shim unit test"
```

### Task 7: Verification smokes + plan-doc update

**Files:**
- Read-only: existing baseline `stage_a_speed018/seed0/run.log`.
- Create: `stage_c/cadence_reconciliation/seed0_100hz_noop.log`, `stage_c/cadence_reconciliation/seed0_1khz_recon.log`, `stage_c/cadence_reconciliation/verdict.json`.
- Modify: `docs/superpowers/plans/2026-06-23-alignment-phase-plan.md` §1 row 8 + §7.3 status on pass.

- [ ] **Step 1: Run the 100 Hz no-op smoke.**

```bash
mkdir -p stage_c/cadence_reconciliation
REFCONF_REPOSITION_PWL=1 PORT_FORCE_ROUTING=u_sol \
  python -u main.py pushing --task-id 4 --max-time 12.0 --admm-iter 25 \
  --solver c3plus --ee-space --sampling-c3 config/sampling_c3_kik.yaml \
  --seed 0 --no-record \
  > stage_c/cadence_reconciliation/seed0_100hz_noop.log 2>&1
```

- [ ] **Step 2: Compare to baseline; assert §3.1 pass-bar items.**

```bash
python scripts/_stage_c_reconciliation_smoke_compare.py \
    --mode 100hz-noop \
    --baseline stage_a_speed018/seed0/run.log \
    --candidate stage_c/cadence_reconciliation/seed0_100hz_noop.log \
    > stage_c/cadence_reconciliation/100hz_verdict.json
```

Expected: all 5 §3.1 pass-bar items PASS; script exit 0.

- [ ] **Step 3: Run the 1 kHz reconciliation smoke.**

```bash
REFCONF_REPOSITION_PWL=1 PORT_FORCE_ROUTING=u_sol PUSHA_CONTROL_HZ=1000 \
  python -u main.py pushing --task-id 4 --max-time 12.0 --admm-iter 25 \
  --solver c3plus --ee-space --sampling-c3 config/sampling_c3_kik.yaml \
  --seed 0 --no-record \
  > stage_c/cadence_reconciliation/seed0_1khz_recon.log 2>&1
```

(Wall-clock will be ~10× longer than the 100 Hz smoke — FLAG and use `run_in_background` if exceeding the local Bash timeout cap.)

- [ ] **Step 4: Assert §3.2 pass-bar items.**

```bash
python scripts/_stage_c_reconciliation_smoke_compare.py \
    --mode 1khz-recon \
    --baseline stage_a_speed018/seed0/run.log \
    --candidate stage_c/cadence_reconciliation/seed0_1khz_recon.log \
    > stage_c/cadence_reconciliation/1khz_verdict.json
```

Expected: all 4 §3.2 pass-bar items PASS; script exit 0.

- [ ] **Step 5: Update the canonical alignment plan §1 row 8 + §7.3 on pass.**

Apply the cell-text update: §1 row 8's sub-divergence (a) flips RECONCILED; sub-divergence (b) stays ACTIVE FRONT. §7.3 gains an "Actual outcome (date): reconciliation landed; both smokes pass; sub-(a) RECONCILED; sub-(b) discriminator runs next" subsection.

- [ ] **Step 6: Commit the verification artifacts + plan update.**

```bash
git add stage_c/cadence_reconciliation/ docs/superpowers/plans/2026-06-23-alignment-phase-plan.md
git commit -m "tick→sim-t reconciliation: verification smokes pass + row-8 (a) flip"
```

---

## 7. Self-review

**Spec coverage (the PART 1-5 requirements from the prompt):**
- PART 1 AUDIT (table, full enumeration, FLAG if more than known three): §0 above. 4 Class A + 11 Class B + 3 Class C exclusions. Scope expansion (3 → 15) explicitly FLAGGED in §0 audit verdict. ✓
- PART 2 REFERENCE-READ (per-constant sim-time cadence target, citation, or flagged fallback): §1 above. A1-A4 reference values cited (`progress_params_c3plus.yaml:40-41`); fallback adopted on this pass with FLAG. Class B fallback adopted (no reference equivalent) — explicit per-constant. ✓
- PART 3 CONVERSION SPEC (file:line edits, AUTHORED): §2 above. Concrete before/after code blocks per constant. Surface counted: 5 files. ✓
- PART 4 VERIFICATION DESIGN (smokes, DESIGNED): §3 above. 100 Hz no-op smoke + 1 kHz reconciliation smoke; explicit pass-bars; wired-signal checks; comparator script structure. ✓
- PART 5 DOWNSTREAM (state, NOT action): §4 above. ✓

**Placeholder scan:** the plan mentions "TBD — find via grep" for `phaseC_stall_threshold` call sites in §2.3, AND for the progress call site in §2.5. These are FLAGS to be resolved during implementation, NOT plan placeholders — the implementation block's first step under those tasks is the grep. Acceptable per the read-only audit constraint (grepping every dispatcher site for every constant inflates the audit scope beyond what's needed to bound the work). NO `TODO`, `add appropriate error handling`, or similar.

**Type consistency:** field rename pattern `<name>_s` is consistent across all 15 conversions. Counter rename pattern `_<thing>_started_sim_t` is consistent across B2/B9/B11. Comparator pattern `(self._step * self._dt_ctrl) - self._<thing>_started_sim_t >= duration_s` is consistent across the three counter-converting Tasks (1, 2, 4). YAML back-compat shim pattern is identical for all renamed fields.

**Cross-section invariants:** the audit (§0) drives the reference read (§1); §1 drives the conversion spec (§2); §2 drives Task 1-5's file modifications. §3 verification design directly tests §2's behavior (the 100 Hz no-op smoke proves byte-equivalence on §2's `× 0.01 s` defaults; the 1 kHz smoke proves the elapsed-s comparator from §2 actually fires sim-time-consistently). §4 downstream gates Tasks 6 + Task 7 Step 5's plan-doc update on Task 7 Steps 1-4 verification passing.

---

## 8. Execution authorization gate

**Per the user's prompt: PLAN ONLY (+ the read-only AUDIT + REFERENCE-READ that filled in PARTS 1, 2). STOP for review. No implementation, no smokes, no commits beyond the plan doc itself.**

Tasks 1-7 above are AUTHORED, NOT actioned. The implementation block (Tasks 1-5 land the conversion; Task 6 lands the unit test; Task 7 lands the verification + plan update) executes ONLY after the user reviews this plan.

In particular, this plan-authoring turn:
- Did NOT modify `control/sampling_c3/params.py` or any of the 5 source files / 1 YAML in §2.8.
- Did NOT modify the canonical alignment plan (`2026-06-23-alignment-phase-plan.md`) row 8 or §7.3 (those updates are bundled into Task 7 Step 5 of the implementation block).
- Did NOT run the smokes (Task 7).
- Did NOT advance to the cadence discriminator (the post-reconciliation gate).
- Did NOT flip `main.py:248 --c3plus-projection` default.
- Did NOT advance to Stage D or the multi-process LCM port.
