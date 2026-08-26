# Presentation brief: Figure 8 C3+ object campaign

## Experiment framing

The experiment tests a simulation port of the Push Anything C3+ pipeline on 25
single-object geometries. Each trial draws a randomized tight SE(2) goal and is
successful only when position error is below 2 cm and orientation error is
below 0.1 rad. Each object was targeted for 28 successful trials. Failures and
interrupted runs were retained but were not plotted as successes.

The measured campaign produced 615 successes, 254 failures, and 12 incomplete
attempts. Twenty-one objects reached 28 successes. Four objects were stopped
after repeated failures but still produced partial successes: Letter E (1),
tape (2), gallon milk (6), and eraser (18).

## A. Failures and their reasons

### Failure category 1: toppling and planar unreachability

This is the dominant classified mode for the four terminal objects: 127 of 159
failures (79.9%). All classified gallon-milk failures (32/32) and eraser
failures (90/90), plus 5/19 Letter E failures, contain a
`[PLANAR-UNREACHABLE]` event. Final orientation errors often approach or exceed
1.6 rad. Once the object leaves the assumed planar manipulation envelope, the
planar controller cannot reliably recover it.

Evidence supports describing the observed mechanism as **toppling followed by
loss of planar reachability**. A lower center of mass, more accurate
mass/inertia/contact modeling, or a controller that reasons about 3D attitude
are plausible future remedies, but the campaign itself does not prove which
underlying model parameter caused each topple.

Use `media/gallon_milk_topple_sequence.png` as the main visual. It shows the
initial state, first topple, temporary recovery, re-topple, and final miss.

### Failure category 2: tight-goal convergence miss

Thirty-two of 159 terminal-object failures (20.1%) stayed planar and passed a
looser pose criterion but did not pass the tight 2 cm / 0.1 rad gate. All 18
tape failures were in this category. Typical tape residuals were approximately
0.015–0.019 m translation and 0.19–0.21 rad rotation. Fourteen Letter E failures
showed a similar pattern, typically 0.012–0.015 m translation and 0.16–0.17 rad
rotation.

The important distinction is that these objects reached the target
neighborhood; they were not unreachable. The remaining error was mainly
orientation and exceeded the strict 0.1 rad threshold.

### What did not dominate

No terminal-object failure was classified as a 75-minute campaign watchdog
timeout. Incomplete/interrupted attempts are reported separately and excluded
from failure percentages.

## B. Future extensions (proposals, not completed results)

### Near-term, directly motivated by observed failures

1. **Stability-aware planning:** include tipping margin, support polygon, object
   height, and estimated center of mass in sample and trajectory scoring.
2. **Better physical identification:** estimate object mass, center of mass,
   inertia, friction, and contact compliance instead of relying on shared or
   approximate values.
3. **3D recovery modes:** detect incipient tilt and either avoid the action or
   switch to a recovery controller that reasons about roll and pitch.
4. **Tight-goal finishing controller:** add a near-goal mode specialized for
   reducing the last 0.1–0.2 rad of orientation error without sacrificing
   already-good translation.
5. **Adaptive sampling:** bias contact samples toward actions that improve the
   currently limiting pose component; retain failed samples to avoid repeating
   bad contact locations.
6. **Complete the partial campaigns:** rerun E, tape, eraser, and gallon milk
   after each isolated intervention, with the same seeds and success gate.

### Broader research extensions

7. Multi-object pushing with collision/de-cluttering reasoning.
8. Hardware validation with perception noise and online object-pose estimation.
9. General SE(3) non-prehensile manipulation rather than planar-only goals.
10. Continuous contact-location optimization and comparison with alternative
    contact-implicit solvers such as CRISP.
11. Parallel/GPU candidate evaluation after numerical parity is established.

For a credible study, change one logical mechanism at a time, preserve seeds,
report both success rate and time-to-goal, and retain failure-mode counts.

## C. Push T versus Anything letter geometry

Three terms must be kept separate:

| Property | Canonical Push T | Imported Anything Letter I | Letter I block surrogate |
|---|---|---|---|
| Purpose | Purpose-built Push T benchmark | Scanned/imported object used in Figure 8 | Controlled approximation of the imported mesh |
| Geometry representation | Two analytic boxes | Triangle mesh plus convex collision decomposition | Three analytic boxes plus triangle-soup OBJ for sampling |
| Model topology | Two links welded by one fixed joint | One link, ten convex mesh collision pieces | One link, three box collision pieces |
| Geometry complexity | 2 collision geometries | 10 collision geometries; source OBJ has 5,000 vertices / 10,000 faces | 3 collision geometries; OBJ has 24 vertices / 36 faces |
| Envelope | 0.20 × 0.16 × 0.04 m | 0.16165 × 0.06149 × 0.05197 m | 0.15451 × 0.05600 × 0.05197 m |
| Simulated mass | 1.0 kg total (0.5 kg per link) | 0.05 kg | 0.05 kg |
| Ground witnesses | 3 task-specific points | Mesh/controller-derived witness set | Copied verbatim from matching mesh task |
| Sampling surface | Purpose-built T perimeter/box geometry | Mesh-face normals from imported OBJ | Mesh-face normals from generated box OBJ |

Push T is therefore **not an Anything mesh**. It uses two overlapping 0.16 ×
0.04 × 0.04 m box links welded into a T. Its link topology, mass, dimensions,
contact decomposition, and task tuning differ from the Anything letter family.

The imported Letter I is a single rigid link represented by a dense visual OBJ
and ten convex OBJ pieces for collision. `I_shape_block` is not the original
Anything object and is not equivalent to Push T: it is a hand-authored
three-box approximation that preserves the letter task's mass, friction,
vertical extent, witnesses, and single-link state layout so geometry can be
isolated in a mesh-versus-block comparison.

### Presentation takeaway

Do not attribute a Push T versus letter result solely to “mesh complexity.” The
comparison also changes shape, size, mass, inertia, link/collision topology,
sampling surfaces, and task/controller lineage. The letter mesh-versus-letter
block pair is the cleaner geometry-ablation comparison because its non-geometry
settings were intentionally matched.

## Suggested slide sequence

1. Goal and experimental protocol.
2. Figure 8 distribution: 615 successful trials.
3. Campaign accounting: completed versus terminal objects.
4. Failure taxonomy and counts.
5. Toppling/unreachability sequence.
6. Tight-gate misses: tape and Letter E.
7. Why Push T and Anything letters are different experimental families.
8. Geometry/model comparison table.
9. Near-term extensions tied to failure mechanisms.
10. Broader research roadmap and evaluation plan.

