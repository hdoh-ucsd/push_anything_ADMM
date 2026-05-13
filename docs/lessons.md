# Institutional knowledge (cross-round)

Patterns, disciplines, and sizing rules that survive any specific
finding. Each entry stands alone; do not weave a narrative across
entries. Move content here from findings docs when the content is
about *how to investigate*, not about *what we found*.

------------------------------------------------------------

## Static-hold integrator sizing rule (factor of 2)

Validated 2026-05-12, 9.4.5-B sub-attempt 1.5. At steady-state with q_err → 0 and v → 0, joint j's integrator must wind to **2·tau_g(q_target_j) / Ki**, NOT 1·tau_g/Ki. The factor of 2 arises because the PD law applies tau_g(q_target) as a feedforward; the integrator must therefore both cancel that feedforward and provide the negative actuation torque needed for force balance at q_now = q_target. For joint 1's home-hold case, tau_g(q_target) = 24.93 Nm, requiring integrator = 2·24.93/8 = 6.23 rad·s. Measured value at I_max=7.0 (reverted) was −6.12 rad·s — within 2% of the closed-form prediction.

**Implication:** Ki·I_max budget for any static hold must be sized to 2× the maximum gravity load across the loaded joints, not 1×. Step 8's Fix 6 sized to 1× (pushing-task load ~7.39 Nm → I_max=2.0), which was correct for the pushing task (moving target, integrator never asked to reach 2× steady-state) but undersized for home-hold (24.93 Nm). This rule is prospective: apply when re-tuning Ki/I_max for any new task or verification probe.

## Prior-art rediscovery: search project docs early

The 2026-05-11 9.4 → 9.4.3 chain re-derived a mechanism the project had already characterized empirically (TS4↔TS3 = 11.1 mm = one full per-stride distance, documented in `docs/reposition_ik.md:148-156` as step 8 Hypothesis F). The chain was rigorous but inefficient — searching the existing docs for "guide" / "stride" / "Hypothesis F" earlier would have surfaced the prior characterization. Same pattern recurred in 9.4.5-C with the rediscovery of step 8 Candidates 2 and 3 (`docs/step8_sampling_c3_candidates.md`).

**Discipline:** when investigating a mechanism, search project docs early for prior-art entries, especially docs that catalog hypotheses or defects (in this codebase: `docs/reposition_ik.md` Bug catalog, Operational notes, Step 8 closure sections; `docs/step8_sampling_c3_candidates.md` Candidate list; `docs/step9_findings.md` Findings A/B/C and Backlog). A brief grep against the suspected mechanism's keywords before the first probe consistently prevents multi-hour re-derivation chains.

## Configuration-change promotion: re-audit deferred items when upstream ships

Step 8 correctly deferred Hypothesis F: the wrapper at that time wasn't commanding contact-seeking targets (mechanism α was suppressing them), so even a perfect Hypothesis F fix wouldn't have moved the box. With α (commit `3e58cc6`) and Finding-C C-fix (commit `8f0b738`) shipped on 2026-05-11, that rationale no longer held, and Hypothesis F became binding. The mechanism didn't change; the upstream configuration did. Same pattern occurred with step-8 Candidates 2 and 3 (deferred behind executor-bottleneck framing; promoted to binding once Effects A and B were addressed).

**Discipline:** each time a fix lands that touches the upstream conditions of an earlier deferral, walk through the deferred mechanisms and check whether the deferral's rationale still holds. A candidate deferred when condition X was binding may become binding once X is resolved. Low-cost institutional practice that prevents downstream re-derivation chains. The same logic applies to *falsifications* under shifting upstream conditions — see the 9.4.7-A retest of 1d, where the prior falsification reason ("forced c3-mode produces n_λ=0") was directly invalidated by the F2 upstream fix.
