# Adaptive EE contact height — design

**Date:** 2026-08-08  **Status:** approved (approach (a))  **Scope:** port only

## Problem

The c3-mode EE tracking height is a hardcoded per-task constant in port-frame
coordinates (`sampling_height: 0.034`, with the reference's separate
`z_height` currently `null` so it falls back to 0.034). Those numbers are
hand-translations of the reference's own constants, which live in a *different
world frame* (reference ground top at z=0 but object resting at z=-0.009;
port ground top at z=0 with object resting at +0.020).

Every frame change has silently invalidated the translation — the box-spawn
migration, the ground-plane move, the pusher-radius change. Each time the
symptom was subtle (grazing contact, plateaued IK arrivals, weak normal
loading) rather than an outright failure, so it cost days to localise. The
2026-08-08 `[OSC-DEBUG]` probe measured the current consequence: at
z = 0.034 the pusher sphere rides the T's **top corner** (centre 14 mm above
the T mid-plane, poking 13.5 mm above the top face), producing a tipping
moment and face-skating instead of a square push.

## Key insight

The reference's `z_height = -0.004` is not a physically meaningful world
constant; it is frame-bound. Expressed **relative to the object** it is
frame-invariant:

> pusher-sphere centre **5 mm above the object's mid-plane**
> (equivalently 15 mm below the T's top face)

Deriving the height from object geometry therefore *reproduces* the reference
rather than departing from it, and removes the fragile manual translation.

## Design

A single rule, evaluated from live object geometry:

```
z_track = obj_z_center + contact_height_offset_above_mid
clamped to [ ground_z + r_pusher + eps ,  obj_top - r_pusher - eps ]
```

- `obj_z_center` — the manipuland body origin's world z, queried from the
  plant each planner tick (so it follows settling/tipping).
- `contact_height_offset_above_mid` — one named parameter,
  **default `+0.005`** = the reference's own geometry.
- Clamp keeps the sphere off the table and off the top face if the object is
  short or the offset is misconfigured.

Consumers: `_c3_track_z()` (the c3-mode z-freeze and the c3-entry altitude
ceiling). Sampling height is **unchanged** — approach stays where it is; only
the press plane becomes object-derived.

### Values produced

| task | obj_z_center | h | z_track | sphere bottom | top margin |
|---|---|---|---|---|---|
| push_t (T) | 0.020 | 0.040 | **0.025** | +0.0055 | 0.0150 |
| pushing (box) | 0.050 | 0.100 | **0.055** | +0.0355 | 0.0450 |

T lands **exactly** on the reference's `z_height` in port frame. Pusher
radius 0.0195.

## Conformance status

- **The mechanism is conformance-neutral-to-positive.** It computes the
  reference's own value instead of transcribing it, and stays correct when
  the frame moves — it removes a recurring source of silent conformance
  violation.
- **Only the offset carries conformance weight.** `+0.005` = reference
  geometry (default). `0.0` = true mid-face; better physics (no tipping
  moment, maximum face overlap) but a deliberate 5 mm departure. Isolated to
  one value so the experiment can be run and labelled cleanly.
- **Box extrapolation is an inference, not a transcription.** The reference
  has no box task, so there is nothing to be unfaithful to, but applying a
  T-derived 5 mm offset to a 100 mm cube is a judgement call. A
  height-proportional offset (mid + 12.5% h → 0.0625 for the box) is the
  alternative; rejected for now as unjustified extra machinery (YAGNI).

## Error handling

- Object body/geometry unavailable → fall back to the existing configured
  height and log once (never crash a run).
- Clamp violation (object shorter than 2·r_pusher) → clamp wins, log once.
- `[C3-Z]` one-shot banner reporting source, offset, and resulting z_track,
  so every run self-documents which plane it pressed on.

## Testing

- Unit: rule returns 0.025 for T geometry and 0.055 for box geometry;
  clamps engage for a degenerate 10 mm-tall object; fallback path on a
  missing body.
- Integration: 20 s smoke per task confirming the `[C3-Z]` banner and no
  new EE-ground contacts in the `[F1K]` trace (the 0.025 plane puts the
  sphere bottom 5.5 mm above the table — the main risk to watch).
- Eval: T 180 s vs p155 (contacts 1341, trans 0.1573); box 180 s vs r3
  (loose PASS 0.0386/0.1375).

## Out of scope

Sampling/approach height, the entry gate thresholds, and any change to the
force path. One mechanism, one parameter.
