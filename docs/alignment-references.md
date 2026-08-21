# Alignment References

Two pointers that govern every alignment-phase decision: where the live plan
lives, and which upstream branch is authoritative.

---

## 1. Current conformance records

The 2026-06-23 alignment plan was superseded by the T-push and
reproduce-dairlib arcs and has been removed from the live documentation. Do not
use its stage table as current project state.

Use these maintained records instead:

- `docs/conformance-map.md` for subsystem-level implementation status;
- `docs/port-todo.md` for reference mechanisms not yet landed;
- `docs/superpowers/investigations/` for dated causal evidence;
- `docs/superpowers/plans/README.md` for work that remains proposed.

### Anti-stale discipline

Treat code and landed commits as authoritative when a dated investigation
conflicts with a current-state document. Update `conformance-map.md` and
`port-todo.md` in the same change that alters the corresponding mechanism.

---

## 2. Authoritative DAIR reference branch

Conform to `push_anything_dev`, NOT `sampling_based_c3_public`. The local
`sampling_based_c3_public` branch that earlier comparisons read is 1490
commits older and lacks the approach-layer code our additions reconstruct.

DAIR repo `dairlib_sampling_c3` (origin: github.com/DAIRLab/dairlib) has two
sampling-c3 branches that diverged:

| Branch | HEAD | Date | Role |
|---|---|---|---|
| `sampling_based_c3_public` | `b52c68d` | 2025-10-17 | older "public" version |
| `push_anything_dev` | `257e3ed` | 2026-06-10 | **1490 commits ahead, authoritative** |

### What push_anything_dev adds

- `examples/sampling_c3/anything/` per-task config dir (NEW; doesn't exist on
  public).
- `sampling_c3_utils.cc` (NEW, 210 lines).
- `systems/controllers/sampling_based_c3_controller.cc`: 2171 → 3319 lines
  (+53%).
- `examples/sampling_c3/generate_samples.cc`: 475 → 871 lines (+83%).
- `examples/sampling_c3/reposition.cc`: unchanged (both branches have
  `RepositionPiecewiseLinear`).

### Approach-layer mechanisms that overlap with our additions

1. **EE_z altitude gate** at `sampling_based_c3_controller.cc:1290-1293` —
   `ee_z_close=true`, `c3_min_clearance=0.01` block c3 entry while EE is above
   sampling_z + clearance. Reference counterpart to our admit-guard.
2. **3-leg high-waypoint reposition** — `RepositionPiecewiseLinear` at
   `reposition.cc:394` with `pwl_waypoint_height=0.0774 m`. Lift up, traverse
   over object, descend onto target.
3. **kMeshNormal face-area-weighted sampling** at `generate_samples.cc:454` —
   uniform-on-face-area, not goal-biased (`barycentric_bias=1`).

### What push_anything_dev does NOT contain

1. A wrong-face commit gate (face_align rejection at c3 entry). Our
   `commit_face_gate` is a genuinely unique addition.
2. A goal-biased sampler.

The single commit `b52c68d` (quaternion Hessian fix) on public didn't make it
to dev — that's the only direction public is ahead of dev.

---

## How to apply

- When asked about alignment status, read `conformance-map.md` and
  `port-todo.md` first.
- When a mechanism lands, update those current-state records in the same
  change.
- When authoring a new plan, link it from `superpowers/plans/README.md` and
  state whether it changes or only measures current behavior.
- When evaluating or conforming to "the DAIR reference": check
  `push_anything_dev` @ `257e3ed`, not `sampling_based_c3_public`. Conforming
  to public would replicate an obsolete reference.
