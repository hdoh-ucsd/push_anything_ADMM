# Contact-Duration Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Suspend the tracker target_z climb during admitted-contact ticks so 1.1N realized contact force has time to accumulate box motion (achieve first goal-directed push).

**Architecture:** Reposition IK tracker (`reposition_ik.py:_build_guide_path`) and the LTD A_lift_trav override (`wrapper.py`) BOTH command the EE to climb during their normal lift-traverse-descend cycle. Probes confirmed that during the only Drake-contact window in the 5 LCP seeds (seed 2, steps 85–88), `target_z` climbs +4 mm/tick (0.095 → 0.107) regardless of whether contact is admitted — and the EE follows, losing contact at step 89. The fix: a contact-admission guard that suspends the climb (holds or sinks `target_z`) while LCS-admit or Drake A_is_ee=1 is true.

**Tech Stack:** Python, PyDrake, NumPy, pytest. Modifies the dispatcher/executor layer of the bilevel control stack.

**Held context:**
- Force is REFUTED as the bottleneck (executor commands 2 N floor; Drake realizes 1.1 N > 0.785 N static-friction breakaway).
- Walls-collapsed-under-LCP hypothesis REFUTED — descend-brush-lift persists, same ~100 mm magnitude, only slower rate.
- Root located at line-level: a tracker writing climbing `target_z` during the contact window. The OWNER (IK-tracker vs LTD-override) MUST be confirmed by Stage 1 before any fix lands.

---

## Pre-Registered Success Criteria (CHECK each before next stage)

These were pre-registered by the operator on 2026-06-01. The plan is judged by them, not by impressions:

- **SC1 — target_z hold during contact (mechanism)**: On a seed that reaches Drake contact (seed 2 baseline; ideally a second seed too), `target_z` does NOT climb during admitted-contact ticks. Specifically: while `A_is_ee=1` OR `CONTACT-RUN contact_type=EE-BOX`, the per-tick Δtarget_z ≤ 0 (hold or sink). Verifiable by tick-aligned re-parse of the LCP run log post-fix.
- **SC2 — Drake-contact duration extends materially**: Drake A_is_ee=1 contiguous run length increases beyond the current 4-tick ceiling (target: ≥ 15 ticks of sustained contact in at least one seed). Verifiable by `grep -c A_is_ee=1` + contiguity check.
- **SC3 — goal-directed box motion (the prize)**: Box moves in the GOAL direction (for pushing-W, −x direction; goal_xy = [-0.30, 0.0] in seed-2 log) by a non-trivial amount on at least one seed. **NOT-A-FAILURE if it falls short of full 300 mm — the over-admission fidelity tax is a separate downstream wall (per project_lcs_input_rep_verdict.md). Partial goal-directed motion is the win here.** Verifiable by `final_obj_xy` − `init_obj_xy` projected onto g_hat = (−1, 0).
- **SC4 — no regression (LCP projection + dispatcher)**: λ_n_max distribution stays bounded at machine precision (max ≤ 3 across all solves, like the c3plus_projection_sweep result), and dispatcher mode-switches stay in the 3–10 range per run (no return to v6-componentwise's 11.6). Verifiable by re-running the multi-seed sweep and comparing to `c3plus_projection_sweep/sweep.summary`.
- **SC5 — distributional verification (≥ 5 seeds)**: SC1, SC2, SC4 hold across ≥ 5 seeds, not just seed-2. Per the standing discipline (canonical seeded baseline, no lucky-draw conclusions).
- **SC6 — no chatter at the admit boundary (debounce works)**: post-fix, the EE z-trajectory during the contact window is monotone-hold or press-in, NOT oscillating up-down tracking admit toggles. Guard against the failure mode of an instantaneous-admit gate (which would re-create chatter through the debounce-less guard).
- **SC7 — dispatcher mode during contact (record-only, not pass/fail)**: report whether the dispatcher flips to free during admit (handing control to the lifting tracker). If yes, flag as held second-contributor for a follow-up commit; do not block this PR.

### Stage-2 fix-design refinements (incorporated)

- **Debounce the admit gate**: do NOT gate on instantaneous admit. Latch admit-active for N ticks (suggest N=8 ≈ 400 ms at dt_ctrl=50ms) so the guard survives 1-tick admit drops. An instantaneous gate would trade the lift for a guard-driven chatter (SC6 catches this).
- **Hold-at-z may need press-in**: if SC2 (duration) clears but Drake force still doesn't sustain, the next refinement is press-in (target_z slightly below the box face), NOT a refutation. Pre-registered: partial-SC2 = refine to press-in, not "fix failed."
- **SC7 dispatcher-free-flip-during-contact is held**: seed 2's contact happened while mode=free (kStayInRepos); seeds 0/1/3/4 held c3-mode (override) but never formed Drake contact. The architecturally-correct expectation is c3-mode-during-contact. If post-fix the dispatcher still flips to free during admit, that's a second contributor recorded but not in this commit's scope.

---

## File Structure (file-by-file responsibility map)

Files to be read in Stage 1 (no modifications):
- `control/sampling_c3/reposition_ik.py` — `RepositionIKTracker.compute_torque` + `_build_guide_path` + `_solve_chain`. Owns the IK tracker's per-tick target waypoint generation. Suspect #1 (free-mode target_z).
- `control/sampling_c3/wrapper.py` — `_run_osc` c3-mode branch (around line 1621+), the APPROACH-OVERRIDE A_lift_trav phase machine (around line 1559+), and the free-mode `_p_ee_des = _p_des_wp` path (around line 1980). Suspect #2 (c3-mode target_z via override).
- `postfix_ee_space_v6/seed4_off.log`, `c3plus_projection_sweep/seed2_lcp.log` — read-only evidence for which path was active during the admitted-contact window.

Files potentially modified in Stage 3 (depends on Stage 1 finding):
- One of `reposition_ik.py` OR `wrapper.py` (whichever owns the climb) — add the admit-suspend guard.
- `tests/test_reposition.py` OR `tests/test_reposition_ik.py` OR a new test file — unit test asserting target_z does not climb when an "admitted-contact" flag is set.

Files NOT touched (per pre-registration):
- `control/sampling_c3/wrapper.py:_derive_force_command` — force magnitude is refuted as bottleneck. Do NOT raise the floor.
- `control/admm_solver.py` LCP projection — must not regress (SC4).

---

## Stage 1 — Investigate which path owns the target_z climb (read-only)

This stage MUST complete and report a confirmed OWNER before Stage 2 starts. No code modifications in Stage 1.

### Task 1.1: Identify what mode and what code path was active at seed-2 steps 85–88

**Files:**
- Read: `c3plus_projection_sweep/seed2_lcp.log` (lines around dispatcher ticks 85–88)

- [ ] **Step 1: Extract per-tick mode, override-phase, target source for steps 80–95**

```bash
grep -E '\[(STEP|APPROACH-OVERRIDE|GS-tgt|GATE-CONTACT|CONTACT-RUN|IMP)\] step=(8[0-9]|9[0-5])\b' \
  /root/push_anything_ADMM/c3plus_projection_sweep/seed2_lcp.log
```

Expected: for each step in [80, 95], show `mode={c3|free}`, override phase (if APPROACH-OVERRIDE fires), and `target=(x,y,z)` from [STEP] line. Capture which mode is active during steps 85–88 (the Drake-contact window).

- [ ] **Step 2: Confirm mode at admitted-contact ticks**

The earlier probe established: at steps 85–88, mode=free, kStayInRepos, won_src=prev_repos, target_z climbing +4 mm/tick. APPROACH-OVERRIDE is c3-mode-only. Verify: if mode=free at steps 85–88, the LTD A_lift_trav override is INACTIVE; the target comes from the IK tracker waypoint (`free_diag['p_des']` in wrapper.py:1980).

- [ ] **Step 3: Read `wrapper.py:_run_osc` free-mode target dispatch (line ~1972–1998)**

```bash
sed -n '1970,2005p' /root/push_anything_ADMM/control/sampling_c3/wrapper.py
```

Expected to find: `_p_des_wp = free_diag.get("p_des") if free_diag is not None else None; _p_ee_des = _p_des_wp`. This means in free mode, the target is whatever the IK tracker computes as the next waypoint.

- [ ] **Step 4: Trace `free_diag['p_des']` to its source**

The IK tracker is `RepositionIKTracker.compute_torque` returning `diag` dict containing `p_des`. Read:

```bash
grep -n 'p_des\|free_diag' /root/push_anything_ADMM/control/sampling_c3/reposition_ik.py | head -10
sed -n '1240,1310p' /root/push_anything_ADMM/control/sampling_c3/reposition_ik.py
```

Expected: `p_des` is one of the IK knots from the guide path built by `_build_guide_path`.

- [ ] **Step 5: Read `_build_guide_path` to see the lift logic**

```bash
grep -n 'def _build_guide_path\|lift_height\|pwl_waypoint_height' /root/push_anything_ADMM/control/sampling_c3/reposition_ik.py
sed -n '894,955p' /root/push_anything_ADMM/control/sampling_c3/reposition_ik.py
```

Expected: the guide path has a LIFT phase that commands the EE to climb to a configured `pwl_waypoint_height` (per CLAUDE.md, `lift_height` parameter in `RepositionParams`). This is the source of the +4 mm/tick climb.

### Task 1.2: Confirm the same mechanism applies on other seeds (generalization)

**Files:**
- Read: `c3plus_projection_sweep/seed{0,1,3,4}_lcp.log` (the runs WITHOUT Drake contact but WITH lifts)

- [ ] **Step 1: For each of seeds 0, 1, 3, 4 LCP, check whether mode was free + IK-tracker-driven during the lift portion of the trajectory**

```bash
for seed in 0 1 3 4; do
  echo "=== seed $seed ==="
  grep -E '\[STEP\] step=(29[0-9]|30[0-9]|31[0-9]|32[0-9])\b' \
    /root/push_anything_ADMM/c3plus_projection_sweep/seed${seed}_lcp.log \
    | head -10
done
```

Expected: for the steps where z_min was reached and the climb began (per the lift-survival check probe — seeds 0/2/4 had z_min around steps 253–319), mode=free and best_src=prev_repos or strat_X (sample-pursuit). Same IK-tracker mechanism.

- [ ] **Step 2: Confirm finding**

If all 5 seeds' lift trajectories were free-mode IK-tracker-driven, the IK tracker is THE OWNER and the c3-mode override is NOT involved in v6 / c3plus_projection_sweep. Stage 2 designs the fix at the IK tracker.

If some seeds had c3-mode lifts (APPROACH-OVERRIDE A_lift_trav firing during the lift), both paths contribute and Stage 2 must address both.

### Task 1.3: Report Stage 1 verdict — STOP here for go-decision on Stage 2

- [ ] **Step 1: Report OWNER + generalization + file:line evidence**

Required content:
- OWNER: IK tracker (reposition_ik.py:_build_guide_path) or LTD A_lift_trav override (wrapper.py) or BOTH.
- File:line where target_z is written during admitted-contact ticks.
- Generalization: same mechanism on ≥ 4 of 5 seeds, or per-seed variance.
- Stop. Do not proceed to Stage 2 without the operator's go on the confirmed-owner fix scope.

---

## Stage 2 — Design the contact-admission guard (PAPER ONLY, no code, gated on Stage 1)

After Stage 1 confirms the owner, design (don't implement) the fix. Discussion items for the plan reviewer:

### Task 2.1: Design the suspension predicate

- [ ] **Step 1: Decide what counts as "admitted contact"**

Candidates:
- LCS-admit (CONTACT-RUN contact_type=EE-BOX) — narrow, fires when planner sees contact
- Drake-realized (A_is_ee=1) — narrower, fires only when Drake resolves force
- Geometric proximity (φ ≤ some threshold) — broadest

Recommendation: LCS-admit (already plumbed; `formulator._last_ee_box_contacts` or the per-tick CONTACT-RUN inspection). Drake-realized is the gold standard but only available post-hoc (after plant integration).

- [ ] **Step 2: Decide what the suspended target_z should be**

Candidates:
- Hold (target_z = current ee_z) — minimal change, EE doesn't climb but doesn't press either
- Sink (target_z = box_face_z) — press EE into face proactively, but this couples to the contact normal direction
- Disable lift phase entirely under admit — fallback to push-axis target_z = box_center_z

Recommendation: Hold (simplest, doesn't introduce a press-force mechanism that would re-open the force-magnitude question).

### Task 2.2: Decide where the guard is inserted

- [ ] **Step 1: At the IK tracker (if Stage 1 confirms IK-owner)**

Insert in `_build_guide_path` or `compute_torque` of `reposition_ik.py`. Guard the LIFT-phase target generation:
```python
if admit_active:
    target_z = current_ee_z  # hold
else:
    target_z = lift_height   # normal LTD lift
```

- [ ] **Step 2: At the wrapper (if override is independently involved)**

If the APPROACH-OVERRIDE A_lift_trav also commands lift during c3-mode contact, guard at `wrapper.py` around the override's target generation (line ~1559+).

---

## Stage 3 — Implement the guard (gated on Stage 2 review, TDD)

### Task 3.1: Write the failing test

**Files:**
- Test: `tests/test_reposition_ik.py` (existing) OR `tests/test_contact_admit_guard.py` (new)

- [ ] **Step 1: Write the failing test**

```python
def test_target_z_holds_during_admitted_contact():
    """When admit_active=True, RepositionIKTracker must not increase target_z
    above the EE's current z. Pre-registered SC1: per-tick Δtarget_z <= 0."""
    tracker = RepositionIKTracker(...)  # standard fixture
    ee_pos_now = np.array([0.077, -0.005, 0.045])  # at box face
    target_far = np.array([0.077, -0.005, 0.107])  # 6cm above
    # No admit: tracker should command climb
    out_free = tracker.compute_torque(..., admit_active=False)
    assert out_free.p_des[2] > ee_pos_now[2], "without admit, tracker should climb"
    # Admit: tracker MUST NOT climb
    out_admit = tracker.compute_torque(..., admit_active=True)
    assert out_admit.p_des[2] <= ee_pos_now[2] + 1e-6, (
        f"with admit, target_z={out_admit.p_des[2]:.5f} > ee_z={ee_pos_now[2]:.5f}"
    )
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /root/push_anything_ADMM
pytest tests/test_reposition_ik.py::test_target_z_holds_during_admitted_contact -v
```

Expected: FAIL with either "AttributeError: admit_active" or "AssertionError: with admit, target_z > ee_z" (because the current code doesn't have the guard).

### Task 3.2: Implement minimal guard

**Files:** (one of, per Stage 1 owner)
- Modify: `control/sampling_c3/reposition_ik.py:_build_guide_path` (or wherever Stage 1 confirms)

- [ ] **Step 1: Add `admit_active` parameter to the relevant method**

(Exact code depends on Stage 1 finding; placeholder — replaced after Stage 1.)

- [ ] **Step 2: Run the unit test to verify it passes**

```bash
pytest tests/test_reposition_ik.py::test_target_z_holds_during_admitted_contact -v
```
Expected: PASS.

- [ ] **Step 3: Run the existing reposition tests to verify no regression**

```bash
pytest tests/test_reposition_ik.py -v
```
Expected: ALL PASS.

- [ ] **Step 4: Commit**

```bash
git add control/sampling_c3/reposition_ik.py tests/test_reposition_ik.py
git commit -m "feat: contact-admission guard suspends IK tracker lift during admit"
```

### Task 3.3: Wire the predicate from the wrapper

**Files:**
- Modify: `control/sampling_c3/wrapper.py` (where the IK tracker is called)

- [ ] **Step 1: Compute admit_active in the wrapper's per-tick code**

```python
admit_active = (
    len(getattr(self.base_mpc.formulator, "_last_ee_box_contacts", [])) > 0
)
```

- [ ] **Step 2: Pass admit_active to the IK tracker call**

(Exact line depends on Stage 1; the IK tracker is the `_tracker.compute_torque(...)` call.)

- [ ] **Step 3: Run end-to-end smoke (1 tick) to verify no crash**

```bash
timeout 120 python main.py pushing --task-id 4 --solver c3plus \
  --c3plus-projection lcp --ee-space \
  --sampling-c3 config/sampling_c3_kik.yaml \
  --admm-iter 10 --max-time 0.5 --no-record --seed 4 --name guard_smoke
```
Expected: exit 0, [RESULT] line emitted.

- [ ] **Step 4: Commit**

```bash
git add control/sampling_c3/wrapper.py
git commit -m "feat: wire admit_active predicate from wrapper into IK tracker"
```

---

## Stage 4 — Verify Pre-Registered SC1–SC5 (gated on Stage 3)

### Task 4.1: SC1 — target_z hold check on seed 2 (the known contact-window seed)

- [ ] **Step 1: Run seed 2 LCP with the guard active**

```bash
mkdir -p /root/push_anything_ADMM/contact_guard_v1
timeout 1200 python main.py pushing --task-id 4 --solver c3plus \
  --c3plus-projection lcp --ee-space \
  --sampling-c3 config/sampling_c3_kik.yaml \
  --admm-iter 25 --max-time 3.5 --no-record --seed 2 \
  --name guard_v1_seed2_lcp \
  > /root/push_anything_ADMM/contact_guard_v1/seed2_lcp.log 2>&1
```

- [ ] **Step 2: Verify SC1 — target_z does NOT climb during admit ticks**

```bash
# Find admit window (LCS EE-BOX or A_is_ee=1) and check target_z delta over that window
python /root/push_anything_ADMM/scripts/verify_sc1_target_z_hold.py \
  /root/push_anything_ADMM/contact_guard_v1/seed2_lcp.log
```

Expected output: `[SC1] PASS — max Δtarget_z during admit = 0.000 mm/tick (was +4 mm/tick pre-fix)`.

If FAIL: STOP, debug, re-run from Stage 3. Do NOT proceed to SC2.

### Task 4.2: SC2 — Drake-contact duration check

- [ ] **Step 1: Compute longest contiguous A_is_ee=1 run**

```bash
python -c "
import re, sys
with open('/root/push_anything_ADMM/contact_guard_v1/seed2_lcp.log') as f:
    steps = sorted({int(m.group(1)) for ln in f
                    for m in [re.search(r'A_is_ee=1.*?step=(\d+)', ln)]
                    if m})
if not steps: print('n_cont=0; SC2 unanswerable'); sys.exit(0)
# Find longest contiguous run
runs, cur = [], [steps[0]]
for s in steps[1:]:
    if s == cur[-1]+1: cur.append(s)
    else: runs.append(cur); cur = [s]
runs.append(cur)
longest = max(len(r) for r in runs)
print(f'n_cont={len(steps)}  longest_contiguous={longest}  threshold=15')
print('SC2', 'PASS' if longest >= 15 else 'FAIL')
"
```

Expected: `SC2 PASS  longest_contiguous >= 15`.

If FAIL but longest > 4 (the pre-fix ceiling): partial-progress; report and decide.

### Task 4.3: SC3 — Goal-directed box motion check

- [ ] **Step 1: Read final_obj_xy from the [RESULT] line and project onto g_hat**

```bash
python -c "
import re
log = '/root/push_anything_ADMM/contact_guard_v1/seed2_lcp.log'
with open(log) as f:
    txt = f.read()
m = re.search(r'final_obj_xy=\(([+\-\d.]+), ([+\-\d.]+)\)', txt)
fx, fy = float(m.group(1)), float(m.group(2))
init = (0.0, 0.0)
g_hat = (-1.0, 0.0)  # pushing-W
motion_along_g = (fx - init[0]) * g_hat[0] + (fy - init[1]) * g_hat[1]
print(f'final_obj_xy=({fx:+.4f}, {fy:+.4f})')
print(f'motion_along_g_hat = {motion_along_g*1000:+.2f} mm  (positive = goal-direction)')
print('SC3', 'PASS' if motion_along_g >= 0.005 else 'FAIL (or partial — not-a-failure if non-zero)')
"
```

Expected: motion_along_g_hat ≥ +5 mm (token win — partial goal-directed motion). Full 300 mm not required (per pre-registration: fidelity tax is separate wall).

### Task 4.4: SC4 — No regression on LCP projection + dispatcher

- [ ] **Step 1: λ_n_max bounded distributionally**

```bash
python -c "
import re
lams = []
with open('/root/push_anything_ADMM/contact_guard_v1/seed2_lcp.log') as f:
    for ln in f:
        m = re.search(r'λ_n_max=([\d.eE+\-]+)', ln)
        if m: lams.append(float(m.group(1)))
print(f'n={len(lams)}  max={max(lams):.3f}  ≥5 count={sum(1 for x in lams if x>=5)}')
print('SC4-λ', 'PASS' if max(lams) <= 3.0 else 'FAIL')
"
```

Expected: max ≤ 3.0, ≥5 count = 0.

- [ ] **Step 2: dispatcher switches in 3–10 range**

```bash
grep 'GS-perf' /root/push_anything_ADMM/contact_guard_v1/seed2_lcp.log
```

Expected: `switches=` value in range [3, 10].

### Task 4.5: SC5 — Multi-seed distributional verification (≥ 5 seeds)

- [ ] **Step 1: Run the same 5-seed sweep as `c3plus_projection_sweep` with the guard active**

```bash
PROBE_OUT=/root/push_anything_ADMM/contact_guard_v1_sweep \
  bash /root/push_anything_ADMM/scripts/run_c3plus_projection_sweep.sh
```

(Reuse the sweep script, only --c3plus-projection lcp variant.)

- [ ] **Step 2: Per-seed SC1+SC2+SC4 check**

Run the SC1/SC2/SC4 verifications on each of seeds 0, 1, 2, 3, 4. Report a 5×3 table.

Expected: SC1 PASS on all 5; SC2 PASS on at least 1 (the one with most contact); SC4 PASS on all 5.

- [ ] **Step 3: Commit verification artifacts**

```bash
git add contact_guard_v1_sweep/
git commit -m "verify: SC1-SC5 multi-seed verification artifacts for contact-admission guard"
```

---

## Self-Review

**1. Spec coverage:** SC1–SC5 are mapped to Stage 4 tasks 4.1–4.5 respectively. Stage 1 maps to the user's "/understand which path owns the climb" requirement. Stages 2 and 3 are paper-then-code per the user's "do not pre-write the fix; depends on Stage 1 finding." Spec covered.

**2. Placeholder scan:** Stage 2 has placeholder code for the guard (intentional — exact form depends on Stage 1). Stage 3 Task 3.2 step 1 says "Exact code depends on Stage 1 finding; placeholder." This is acceptable because the plan explicitly gates Stage 2/3 on Stage 1 completion. Other steps have concrete commands.

**3. Type consistency:** `admit_active` (boolean parameter) is the consistent name across Stages 2 and 3. `_last_ee_box_contacts` (list) is the consistent admit predicate source. `target_z` and `p_des[2]` refer to the same value (the EE-position desired z) — documented inline.

---

## Held — execution handoff

Per the user's instruction ("Report: the write-plan with SC1-SC5 pre-registered, then the /understand finding ... then STOP before applying the fix"), the EXECUTION starts at Stage 1 only. Stage 2+ requires the operator's go on the confirmed-owner fix scope.

**Plan saved.** Stage 1 will execute next (read-only investigation). Then STOP.
