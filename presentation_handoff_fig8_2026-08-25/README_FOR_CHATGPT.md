# Figure 8 experiment — presentation handoff

Use this folder as the complete source packet for making a presentation about
the Figure 8 single-object C3+ experiment.

## Requested presentation sections

1. **Failures and reasons** — distinguish toppling/planar-unreachable failures
   from tight-goal convergence misses.
2. **Future extensions** — present these as proposals, not completed results.
3. **Push T versus Anything letter geometry** — explain that Push T is a
   purpose-built two-link box model, while the Anything objects are imported
   scanned meshes; the `I_shape_block` files are a simplified, single-link
   box decomposition made for controlled geometry comparisons.

## Headline results

- Campaign: 25 objects, target 28 successful randomized goals per object.
- 21/25 objects reached the 28-success target.
- 615 successful, 254 failed, and 12 incomplete attempts were recorded.
- Attempt success rate excluding incomplete attempts: 70.8%.
- Four objects were stopped after repeated failures: Letter E, tape, eraser,
  and gallon milk. They retained 27 partial successes.
- Among their 159 classified failures, 127 (79.9%) involved toppling and loss
  of planar reachability; 32 (20.1%) were tight-gate misses.

Read `PRESENTATION_BRIEF.md` first. It contains a slide-ready narrative,
speaker cautions, and the mesh comparison. `SOURCE_INDEX.md` maps every claim
to supporting files.

## Suggested prompt

> Create a research presentation from the attached folder. Center it on the
> Figure 8 C3+ randomized-goal experiment. Include: (A) failures and their
> evidence-backed reasons, (B) future extensions clearly labeled as proposed,
> and (C) the difference between the canonical Push T model, imported Anything
> letter meshes, and the simplified letter-block surrogate. Use the supplied
> figures and videos. Do not claim that incomplete attempts failed, do not
> generalize the four terminal objects to all 25 objects, and do not present
> proposed explanations or extensions as measured results.

