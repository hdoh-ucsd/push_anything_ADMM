# Python Port of *Push Anything* — Diagnostic Investigation and Root-Cause Analysis

**Hyungjun Doh** · Department of Electrical and Computer Engineering · UC San Diego
Based on Bui et al., "Push Anything: Single- and Multi-Object Pushing From First Sight with Contact-Implicit MPC," ICRA 2026 (arXiv:2510.19974), and the dairlib `sampling_based_c3` reference (Venkatesh et al., arXiv:2505.13350).

---

## Summary

This report documents a Python/PyDrake reimplementation of the *Push Anything* sampling-based contact-implicit MPC system and, more substantially, the diagnostic investigation that explains why the port plans correct contact forces yet fails to translate the manipuland to its goal. The headline contribution is not a final motion number. It is (1) an attribution methodology that separates *earned* architectural contribution from *kinematic artifact* in the controller's output, and (2) a multi-stage, falsification-driven diagnosis that localizes the performance gap to a single code-confirmed root cause: the executor commands a goal-aligned heuristic force rather than the planner's optimal contact force, a limitation that traces directly to the choice to plan in the robot's full joint space rather than the reference's reduced Cartesian proxy.

The investigation repeatedly used cheap read-only diagnosis to rule out expensive fixes before they were attempted — including a force-magnitude bottleneck, a coordinate-frame bug, a friction-cone modeling gap, and two separate "sticky contact" schemes — each of which a less disciplined process would have built. Every ruled-out hypothesis is itself a result, because each narrowed the search to the true cause.

---

## 1. System overview

The plant is a Franka Panda arm with a spherical pusher (radius 25 mm) acting on a 100 mm cubic box (200 g, friction coefficient μ = 0.4) on a table, simulated in Drake. The task studied here is a westward push: drive the box −0.3 m along the x-axis.

The controller is a bilevel sampling-C3 dispatcher following Venkatesh 2025. At each control tick it gates between a *contact-free* mode, which repositions the end-effector kinematically toward a sampled approach pose, and a *contact-rich* mode, which runs a C3+ contact-implicit MPC solve (Bui 2026, with the η slack variable and closed-form per-component projection). Confirmed at the source level: the solver in every run is genuinely C3+ — the decision vector carries η, and Bui's closed-form projection fires each iteration. The "mode = c3" label in the logs denotes the contact-rich dispatcher mode, not the legacy C3 solver variant; these were carefully distinguished during the audit.

---

## 2. Architecture: port versus reference

The single most consequential design difference between this port and the dairlib reference is the space in which contact is planned. The difference is structural, not a parameter, and it propagates into every downstream behavior the investigation examined.

| Aspect | dairlib reference | This port |
|---|---|---|
| Planning model | 3-DoF floating end-effector proxy | Full 7-DoF Franka |
| Control input | u ∈ ℝ³ (Cartesian EE force) | u ∈ ℝ⁷ (joint torque) |
| Planner output consumed | EE position + force trajectories, published over LCM | (x_seq, λ) retained; joint-torque u_seq discarded |
| Execution | Separate OSC process (1 kHz), Franka-only plant | In-process OSC executor, full box-aware plant |
| Contact force in OSC | External-force tracking (λ_e), no friction-cone contact channel | External-force tracking (λ_ext), no friction-cone channel |

Two clarifications resolved earlier ambiguities in the project's slide deck. First, the port *does* use an Operational Space Controller in the canonical control path; the earlier "no OSC layer" framing is stale. The planner's joint-torque output is discarded, and the executor instead tracks a Cartesian target derived from the planner's predicted next state, with the planner's contact forces entering as a feedforward. Second, the J⁻ᵀ "three failures" critique from the early deck is now half-superseded: the actuator-side failures (Jacobian conditioning, torque saturation) are dispatched by the OSC QP, which respects torque limits and posture; the planner-side conditioning concern remains live, because a 7-DoF C3+ over Franka kinematics inherits pathologies the reference's 3-DoF Cartesian proxy avoids by construction.

A direct source comparison confirmed that the port's force-tracking *mechanism* faithfully mirrors the reference. The reference's push force is produced entirely through an external-force tracking objective (λ_e via `ExternalForceTrackingData`); its `franka_osc_controller` registers position, orientation, and force tracking but **no** friction-cone contact constraint. The port's `λ_ext` channel is the same construct. Where the reference omits a friction-cone channel for this task, so does the port — correctly, because for nonprehensile pushing with a spherical EE the goal is to command a force, not to predict a cone-constrained reaction. The implementations agree on what to include and what to omit.

---

## 3. The attribution methodology

A custom offline visualizer parses each run log into a per-step record and classifies every contact-rich step into one of four categories: *planned-productive* contact (contact-rich mode, real EE–box contact pair, normal force above threshold, contact normal anti-aligned with the goal direction), *planned-unproductive*, *accidental* contact during reposition, and *no-contact* free motion. The classifier credits goal progress to a step only when the box actually moves while a genuine, correctly-directed contact is active. This separation is what distinguishes motion the controller *earned* through correct contact reasoning from motion that is a kinematic side-effect of the arm sweeping past the box.

The methodology proved its value by catching its own measurement error. An interim claim that five of six contact-rich sessions "made contact" rested on the planner's horizon-wide maximum normal force as a contact proxy — a quantity that is nonzero whenever the planner *predicts* contact at any future knot, including box-ground contacts and future-horizon predictions. Re-auditing against the correct per-step proxy (the magnitude of the actual contact normal versus a no-contact sentinel) showed only one of six sessions made real current-step contact. The corrected reading also confirmed that the run's reported earned motion was honest, not an under-count: the sessions that scored zero productive steps genuinely did not move the box. An attribution tool that surfaces and corrects its own over-counting is, for a research instrument, working as intended.

---

## 4. The diagnostic chain

The core of the work is a sequence of hypotheses, each tested by cheap read-only analysis before any implementation, and each either falsified or confirmed with quantitative evidence.

### 4.1 Is the planner force-limited? — Falsified

The first hypothesis was that the 7-DoF joint-space planner could not request large enough contact forces. It was decisively falsified by a one-line physical check: a 200 g box at μ = 0.4 requires only about 0.785 N to slide, while the planner requests a median 8.5 N during productive contact — roughly eleven times the threshold — at zero torque saturation (median commanded torque about 30 N·m against an 87 N·m limit). The force was never the bottleneck, and a Cartesian-proxy rewrite undertaken to address force magnitude would have addressed a problem that does not exist.

### 4.2 Why is productive contact so rare? — Entry geometry

Productive contact occupies only about 1.1% of a run (roughly 170 ms of 15 s), with a median contact window of 20 ms; the arm spends over 90% of every run repositioning. A per-session anatomy revealed the mechanism. The contact-rich mode is entered when the inverse-kinematics tracker reports "arrived" — defined as being within 20 mm of its target. But the target is the *setback* point: a sample projected 30 mm outward along the box face normal (a correct addition that prevents the pusher from colliding with the box during approach). So at the moment the dispatcher switches to contact-rich mode, the end-effector is roughly 35 mm shy of the box surface, no EE–box contact pair exists in the LCS, the normal force is zero, and a contact-loss gate correctly exits the dead session after five steps. Twelve of thirteen contact-rich entries in the canonical run made no contact at all; the single productive session was triggered by a cost-based entry path, not the normal arrival gate. The entry gate, written when the target effectively sat on the box, was never updated when the setback moved the target 30 mm off it — two individually-correct mechanisms left mutually inconsistent.

### 4.3 When contact does form, why does it not sustain? — Force command versus force reaction

In the rare sessions that achieve clean contact, the push still decays. Tracing the executor's internal torques showed the contact feedforward holding below 1 N·m while the tracking term collapsed roughly forty-fold, the end-effector separating from the box, and the planner's own cost rising every step. A source-level comparison against the reference explained why. The reference publishes the planner's *commanded* Cartesian force trajectory and has its OSC track it as a soft cost on a force decision variable — a command that persists across momentary contact loss by construction. The port instead fed the OSC the planner's first-knot complementarity *reaction*, which by the structure of the LCS collapses to zero whenever the predicted contact gap is positive. The reference commands a force; the port observed a reaction. A command persists; a reaction evaporates the instant contact is interrupted.

### 4.4 The force-tracking fix — validated

Implementing the reference's mechanism — adding the external force as a QP decision variable with a soft tracking cost, fed a derived force command with a magnitude floor so it does not collapse on momentary contact loss — produced a clean, validated improvement. Against the baseline, the executor's contact feedforward held at 4.89 N·m instead of collapsing to 0.8 N·m, contact persisted for nineteen consecutive steps (190 ms) instead of one-to-three-step flickers, and westward motion per second improved roughly 3.7×, all with zero QP failures and no instability. A static-configuration unit test confirmed the new decision variable tracks its commanded force to within 0.005 N. The fix worked through exactly the mechanism the diagnosis predicted.

### 4.5 Combining the fixes — a remaining gap, honestly attributed

With the entry gate also re-tuned to admit on contact proximity, the contact-rich entry rate improved markedly. But the combined run did not compound into proportionally more earned motion, and the corrected attribution explained why: the entry gate admits at roughly 85 mm (EE center to box center), while Drake admits a real contact pair only at about 77 mm, and the force-tracking push could not close the remaining 8–13 mm within the disengage window. The gate gets the arm *near* the box; it does not by itself complete the approach into a real, sustained contact.

### 4.6 Hypotheses ruled out

Three further candidate causes were investigated and ruled out before any code was written:

A **coordinate-frame bug** was the leading suspect once parameter tuning kept revealing the next mis-aligned knob — the classic signature of compensating for an error underneath. A seam-by-seam comparison of every frame conversion against the reference found the conventions correct: the quaternion order matches, the end-effector Jacobian is the right dimension and frame, and the goal-alignment computation is internally consistent. The difficulty is not a sign flip or a transposed rotation.

A **friction-cone modeling gap** appeared, at first, to be a clear structural difference: the reference's QP machinery contains a friction-cone-constrained contact channel that the port lacks. Reading the reference's actual push-task controller dissolved this: the reference registers no contact points for the push task, so its friction-cone channel is empty. The cone machinery exists in the general OSC infrastructure (used by the reference's legged-robot controllers) but is deliberately unused for nonprehensile pushing. Adding it to the port would implement something the reference itself rejected for this task — a research extension closing no parity gap. This was the clearest case of the read-first discipline preventing an unnecessary and stability-risky build.

Two **sticky-contact schemes** — a minimum contact-mode residency, and a two-phase approach that descends a final waypoint into the box surface — were each shown to be either harmful or premature. Minimum residency would extend the dead, no-contact sessions the disengage gate exists to clean up. The two-phase descend was implemented and rolled back after it regressed end-effector height tracking; its *intent* (sustain the approach into contact) was correct and is reflected in the eventual force-tracking fix, but its implementation broke a working behavior.

### 4.7 Root cause

With force magnitude, entry geometry, frame conventions, the friction cone, and sticky residency all eliminated, the remaining gap is precise and code-confirmed. The executor commands a force whose *direction* is purely the negated goal direction (−ĝ) and whose *magnitude* is derived from the planner's predicted normal force. This is a heuristic, not the planner's solution. The reference, by contrast, tracks the planner's actual commanded Cartesian force, which captures pushes that deviate from the straight-at-goal direction (for instance, when the optimal action rotates the box or exploits contact geometry). The port cannot supply such a command because its 7-DoF joint-torque planner does not *produce* a Cartesian force; the reference obtains one for free because it plans in a 3-DoF Cartesian proxy. Every thread of the investigation — contact-hold, the force-command source, the residual approach gap — converges on this one architectural choice.

---

## 5. Implementation layers landed

Independently of the final root-cause finding, the investigation produced a sequence of validated, separately-committed fixes that brought the port into closer alignment with the reference and with the paper's specification:

The face-normal sampler replaced a center-radius ring that placed off-axis samples inside the box collision volume (worst case, a 20.7 mm penetration at corner-aimed angles that matched an observed +17 mm adversarial box bump), implementing the paper's outward-normal projection. A contact-loss disengagement gate eliminated a hang in which the dispatcher remained in contact-rich mode without any contact. A rich-to-free buffer refresh implemented the reference's lowest-cost sample selection on mode transition. A contact-run attribution fix restored honest EE–box contact accounting. The force-tracking executor and the contact-proximity entry gate are the two most recent. Each was validated against the attribution methodology and committed atomically.

---

## 6. Results, honestly framed

The motion numbers are evidence, not the headline, and they carry real run-to-run variance owing to first-strike sensitivity in the approach. A representative chain:

| Configuration | Earned westward motion | Productive steps | Sessions making real contact |
|---|---|---|---|
| Canonical (face-normal sampler) | ~3.8 mm | 17 | 1 of 13 |
| Disengagement-gate variant | ~10.3 mm | 83 | 5 of 7 |
| Force-tracking + entry-gate | ~2.8 mm | 22 | 1 of 6 (real-contact proxy) |

The force-tracking validation is the cleanest single result: 3.7× motion per second, contact sustained roughly six times longer, contact feedforward six times higher, through the predicted mechanism. The combined-fix run's lower aggregate number is well-understood — it is approach-gap-limited, not a regression of the force-tracking mechanism, which the per-session attribution confirms. No configuration reaches the goal; the honest statement is that the system makes correct, sustained contact in the sessions it successfully initiates, and the bottleneck on overall progress is now precisely characterized.

---

## 7. Future work

The single change that addresses the root cause is to track the planner's *actual* Cartesian force rather than a goal-aligned heuristic. Two routes exist. The first derives a task-space wrench from the existing 7-DoF C3+ solution and feeds it as the force-tracking command — a self-contained controller change that does not alter the planning architecture. The second adopts the reference's 3-DoF floating-end-effector Cartesian proxy as the planning model, which yields a true Cartesian force command directly and matches the reference's architecture wholesale; this is the larger, more faithful change and would also clarify the planner-side conditioning concern noted in §2. Both are multi-week efforts that warrant a fresh start rather than continuation of the present diagnostic session.

The approach gap (§4.5) is a secondary, separable lever: closing the 8–13 mm between gate admission and Drake contact admission, either by tightening the setback with a velocity cap, raising the approach push force, or a re-implemented descend phase that preserves height tracking.

---

## 8. Slide-deck corrections

For the existing progress deck, four claims require revision before presentation: "no OSC layer" is stale (the OSC is the canonical executor); "F_EE = J⁻ᵀ u_arm" is stale as worded (the planner's joint-torque output is discarded, and contact force enters as a feedforward into the OSC QP); "u ∈ ℝ⁷ joint torque" remains correct for the planner but should note that u_seq is unused downstream; and the "J⁻ᵀ three failures" slide should be reframed from actuator-side (now dispatched by the OSC) to planner-side (still live, and the locus of the root cause).

---

## 9. Closing note on method

The defining discipline of this investigation was to diagnose before fixing, and to treat a ruled-out hypothesis as a result. That discipline repeatedly prevented expensive dead ends: a Cartesian rewrite aimed at a non-existent force limit, a friction cone the reference itself does not use, two sticky-contact schemes that would have extended dead windows or broken height tracking, and a frame-bug hunt through conventions that turned out correct. What remains is a clean, code-grounded account of why a faithful C3+ port plans correct contact yet does not move the box to its goal — and a precisely specified fix. For a research investigation, knowing exactly why something does not work, and what single change would address it, is the result.
