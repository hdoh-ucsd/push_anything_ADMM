# Path B — continuous contact-location optimization inside Push Anything / C3+

**What Path B is.** CRISP-inspired continuous contact-location optimization
*inside* Push Anything/C3+. The single idea retained from CRISP is **contact
location as an optimization variable**.

**What Path B is not.** CRISP implemented inside C3+. Nothing is transferred
from CRISP B-B: not the dynamics, not the quasi-static limit surface, not the
normal-force-only model, not the pairwise face complementarity, not the solver.
The B-B → C3+ adapter was investigated and declined
(`docs/crisp_c3plus_linearization_study.md`).

Everything below is read out of this repository, not from the paper.

---

## 1. The actual baseline architecture

### 1.0 Which config is canonical (read this first)

The box's canonical controller config must be passed **explicitly**:

```
python main.py pushing --task-id 4 --sampling-c3 config/sampling_c3_kik.yaml --max-time 180
```

A bare `--sampling-c3` loads `config/sampling_c3_params.yaml`, which is a
**different controller**: `kRandomOnCircle`, 3+1 samples, no cost-LCS ranking.
The canonical `sampling_c3_kik.yaml` is `kFaceNormal`, 5+4 samples,
`use_cost_lcs_ranking: true`. Two earlier diagnostic rollouts in this branch
used the bare flag and are therefore **not** baseline measurements.

### 1.1 The pipeline

```
box pose (obj_xy, obj_quat)
    │
    ├─► face selection        uniform random PER SAMPLE over the 4 side faces
    │                          sampling.py:917  rng.integers(0, n_faces)
    │
    ├─► point on face          scalar jitter along the face tangent
    │                          sampling.py:995  jitter ~ U(-range, +range)
    │                          point = face_center + jitter * tang
    │
    ├─► EE placement           project outward along the face normal
    │                          proj_xy = point + setback * n_world
    │                          z       = sampling_height
    │
    ├─► rejection              resample if within sample_reject_clearance
    │                          of the box surface
    │
    ├─► candidate list         [current_EE] + 5 strategy samples
    │                          controller.py:1041 insert(0, ee_pos_now)
    │
    ├─► STAGE 1  planning C3+  ONE LCS built at the current Drake state,
    │                          shared by all candidates.
    │                          Per candidate: x0[7:10] = sample_pos
    │                          inner_solve.py:629
    │
    ├─► STAGE 2  cost ranking  PER candidate: IK to the sample EE pose,
    │                          plant temporarily moved there, a SEPARATE
    │                          cost-LCS built at that arm config, forward
    │                          simulate with PD + PGS LCP, score the
    │                          SIMULATED trajectory
    │                          inner_solve.py:738-790
    │
    └─► argmin                 best candidate drives the OSC target
```

### 1.2 The nine questions, answered from the code

1. **Face selection** — uniform random *per sample* (`rng.integers(0, n_faces)`),
   area-weighted only for `tshape`. **There is no face enumeration.** Each of the
   5 strategy samples independently draws its own face *and* its own point. The
   baseline is therefore **not** "F faces × P points"; it is 5 joint
   (face, point) draws.
2. **Point sampling** — one scalar `jitter ~ U(−range, +range)` along the face
   tangent `t = (−n_y, n_x)`. `range = half_len = box_half = 0.05 m`, except on a
   *goal-aligned* face (`−n·ĝ > GOAL_ALIGN_THRESHOLD = 0.7`) where it shrinks to
   `CENTERED_JITTER_FRACTION = 0.2` × half_len = **0.01 m**.
3. **EE placement** — `proj_xy = point_on_face + setback · n`, `z = sampling_height`.
4. **Normal offsets** — `sampling_setback = 0.027 m` (`sampling_c3_kik.yaml:295`).
   The code floors this at `PUSHER_RADIUS + 0.005 = 0.0245 m`; with
   `PUSHER_RADIUS = 0.0195` the floor **does not fire**, so 0.027 stands. At a
   face-centre target the EE centre sits 0.077 m from the box centre and the
   pusher surface sits **7.5 mm proud** of the face. (The floor's docstring still
   describes a 0.030 setback and a 0.025 pusher radius; both are stale.)
5. **Candidate count** — `num_additional_samples_c3: 5` plus the current EE
   inserted at the front = **6 evaluated candidates** per c3 tick;
   `num_additional_samples_repos: 4` in reposition mode.
6. **What differs between candidates** — the **EE position only**. Same box
   state, same goal, same stage-1 LCS.
7. **Candidate cost** — two stages, see 1.1. Stage 2 scores a *forward
   simulation* of the plan, with object-only cost matrices
   (`_object_only_cost_matrices_ee_space`: `Q[7:10] = 0`, so EE position is not
   itself costed — only the object's resulting motion is).
8. **Does the contact point stay fixed over the horizon?** **Yes.** The sample is
   a single EE position used as `x0`; there is no per-knot contact variable. The
   planner's `x_seq` then evolves `p_ee` freely, but the *sampled contact
   location* is one point per candidate per tick.
9. **How the sample reaches the LCS** — split, and this is the crux:
   - **stage-1 planning LCS**: only through `x0[7:10]`. The matrices are **not**
     rebuilt per candidate. `inner_solve.py:487` states this outright.
   - **stage-2 cost LCS**: fully rebuilt at the sample's IK'd arm configuration.

> The `inner_solve.py:487` docstring ("contact admission reflects the CURRENT EE
> position rather than the hypothetical sample") is **stale for the box**: it
> describes stage 1 only. `use_cost_lcs_ranking: true` added stage 2, which does
> place the plant at each sample's arm config before building the cost-LCS
> (the "Delta-1 gap fix" comment at `inner_solve.py:753-766`).

---

## 2. Contact-location dependency graph (Task 2)

With `s` the continuous coordinate along one selected face:

```
s
├─ affine ─► c(s)        contact point on the face
│            └─ affine ─► p_ee(s) = c(s) + setback·n,  z = sampling_height
│                         ├─ affine ─► stage-1 x0[7:10]
│                         │            └─► C3+ solve  ──► x_seq(s), u_seq(s)
│                         │                              └─► stage-1 cost
│                         └─ NONLINEAR ─► IK: q_arm(s)
│                                        └─► cost-LCS rebuilt at q_arm(s)
│                                             └─► PD + PGS forward sim
│                                                  └─► STAGE-2 CANDIDATE COST
└─ (winner only) ─► OSC target
```

| quantity | depends on `s` | class |
|---|---|---|
| contact point `c(s)` | yes | **A — affine** |
| EE placement `p_ee(s)` | yes | **A — affine** |
| stage-1 `x0[7:10]` | yes | **A — affine** |
| stage-1 `A,B,D,d` / `E,F,H,c` | **no** — built once per tick | invariant |
| planner `x_seq(s)`, `u_seq(s)` | through `x0` only | **B — locally linearizable** |
| stage-1 cost | through `x_seq` | **B** |
| per-sample IK `q_arm(s)` | yes | **B** — iterative, no closed form |
| cost-LCS geometry at `q_arm(s)` | yes | **C — requires LCS reconstruction** |
| forward-sim rollout | yes | **C** |
| **stage-2 candidate cost (the argmin)** | yes | **C** |
| OSC target | winner only | **A** |
| constraints | no | invariant |

**Nothing is class D.** Every dependency is representable; the question is cost,
not feasibility.

**The consequence that shapes everything:** the *cheap* half of the dependency
(`s → p_ee → x0`) is exactly affine, but the *decisive* half — the stage-2 cost
that actually selects the winner — is class C. Evaluating `cost(s)` for one new
`s` costs one IK solve plus one cost-LCS build plus one forward simulation:
**approximately one full candidate evaluation.** The enumeration-versus-iteration
exchange rate is therefore close to 1:1, which is exactly the hypothesis Task 7
exists to test.

---

## 3. Parameterization (Task 1)

For face `f` with world outward unit normal `n_f`, centre
`c_f = obj_xy + box_half·n_f`, tangent `t_f = (−n_{f,y}, n_{f,x})`, and
half-length `L_f`:

```
p0   = c_f − L_f · t_f                       face corner, one end
p1   = c_f + L_f · t_f                       face corner, other end
c(s) = p0 + s (p1 − p0),        s ∈ [0, 1]
p_ee(s) = c(s) + setback · n_f,  z = sampling_height
```

- **Frame.** World xy; `n_f` and `t_f` are body-frame face data rotated through
  `obj_quat`, so the parameterization follows the object as it turns.
- **Bounds.** `s ∈ [0, 1]`, with `s = 0` and `s = 1` exactly at the face corners.
  The baseline's uniform jitter corresponds to `s ~ U(0,1)`; its centred variant
  corresponds to `s ~ U(0.4, 0.6)`.
- **Equivalence to the baseline.** `jitter = (2s − 1)·L_f`. Path B changes *how
  `s` is chosen*, and nothing else about the mapping.
- **Derivatives.** `dp_ee/ds = 2 L_f · t_f`, **constant**; `d²p_ee/ds² = 0`. The
  geometric part of the parameterization is exactly affine and perfectly
  conditioned. All curvature in `cost(s)` therefore comes from IK, the cost-LCS
  rebuild, and the forward simulation — never from the parameterization.
- **Corner behaviour.** `s → 0` or `1` puts the pusher at a face corner, where
  the outward-normal projection can violate `sample_reject_clearance` for a
  rotated box. Path B inherits the baseline's rejection test and additionally
  clamps to `s ∈ [ε, 1−ε]`.
- **Face-boundary behaviour.** No wraparound. Each face is an independent
  interval; moving to an adjacent face is a *discrete* face change, which Path B
  deliberately keeps discrete.

### Why face selection stays discrete

If a candidate carries only the contact representation of face `f`, there is
never more than one face force in play, so CRISP's pairwise exclusivity
`λ_i ⊥ λ_j` is **structurally vacuous** rather than approximated. That isolates
the research question — sampled versus optimized contact *location* — without
touching C3+'s complementarity formulation.

### SE(3) generalization

The mapping is written as **face origin + tangent basis × coordinates**, which
extends from one coordinate to two without structural change:

```
planar:  c(s)    = p0 + s·(L·t)                     1 coordinate,  3×1 basis
cube:    c(u,v)  = p00 + u·e1 + v·e2                2 coordinates, 3×2 basis
```

`dc/d(u,v)` stays a constant basis matrix, so the conditioning argument carries
over. **Design choices that would block the extension, and are therefore
avoided:** hard-coding a single scalar tangent; assuming `z = sampling_height`
(planar-only — for a cube face the third coordinate is a face coordinate, not a
constant); and assuming `L_f = box_half` (must stay per-face). Two of these are
present in the *baseline* sampler and are the reason Path B keeps its own
mapping rather than editing `_face_normal_projection` in place.

---

## 4. Optimization strategy (Task 4)

| | compatible with C3+ | correct | extra iterations | complexity | latency | init sensitivity | → (u,v) |
|---|---|---|---|---|---|---|---|
| **A** `s` inside C3+ | **no** — needs the LCS to depend on `s`, i.e. class C inside the solver: exactly the adapter already declined | — | — | high | — | — | — |
| **B** outer 1-D optimization | yes — `cost(s)` as a black box | yes | 1 candidate eval per iteration | low | linear in iterations | moderate | yes |
| **C** alternating fix-`s` / solve / update-`s` | yes | needs `dcost/ds`, which class C does not provide in closed form; degenerates to B | ≥ B | medium | ≥ B | high | yes |
| **D** coarse sample + local refine | yes | yes | tunable | **lowest** | tunable | **low** — seeded by the coarse best | yes |

**Chosen: D**, implemented on B's machinery with a small, explicit budget.

Reasons, in order of weight:

1. **It reuses the existing per-candidate evaluation unchanged**, so the A/B is
   apples-to-apples: identical cost function, identical LCS construction,
   identical solver settings. Any difference is attributable to *how the contact
   location is chosen*, which is the whole question.
2. **It makes the exchange rate directly measurable.** Baseline spends its budget
   on `P` independent draws; Path B spends it on `1` draw plus `k` refinement
   iterations. Reporting both counts is Task 7.
3. **Lowest initialization sensitivity**, because the coarse best seeds the
   refinement — and initialization sensitivity is the failure mode that killed
   the naive strategies in the B-B study.
4. A is ruled out on architecture, not preference: it *is* the declined adapter.
   C is B with extra machinery unless a cheap `dcost/ds` exists, and class C says
   it does not.

**Prerequisite before implementing.** D is only well-posed if `cost(s)` is
smooth enough that local refinement beats another random draw. Because the
geometric mapping is exactly affine (§3), any roughness comes from IK, the
cost-LCS rebuild, or the forward sim. **That is Task 3, and it must be measured
first** — see `path_b_results.md`.
