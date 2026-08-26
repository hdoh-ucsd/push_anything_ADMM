# Source index and provenance

## Campaign-wide evidence

- `data/EXPERIMENT_2026-08-25.json` — canonical campaign counts and IDs.
- `data/FIG8_MESH_28_SUCCESS_RUNS.csv` — all 615 successful records with log,
  seed, first-goal step, and time-to-goal.
- `reports/failure_classification.md` — detailed measured failure taxonomy.
- `media/fig8_randomized_result.png` — distribution used in the README.

## Failure visuals

- `media/gallon_milk_topple_sequence.png` — five-stage primary failure figure.
- `media/*_top2.png` — paired examples for E, tape, gallon milk, and eraser.
- `media/fig8_E_failure.mp4` and `media/fig8_gallon_milk_failure.mp4` — full
  rendered failure examples with controller side panel.

## Successful comparison videos

- `media/push_t_success.mp4` — representative validated Push T success.
- `media/letter_i_success.mp4` — representative validated imported Letter I
  success.

## Exact geometry/model sources

- `geometry/push_t.sdf` — canonical two-link, two-box Push T.
- `geometry/I_shape_texture.sdf` — imported Letter I collision model.
- `geometry/I_shape_texture.obj` — imported Letter I source mesh.
- `geometry/I_shape_block.sdf` and `.obj` — controlled three-box surrogate.
- `geometry/gen_block_assets.py` — documents how block variants preserve
  non-geometry properties and why their OBJ files exist for mesh-normal sampling.

## Supporting technical notes

- `reports/anything_n1_config_delta_audit.md` — differences between the native
  Anything N=1 lineage and the legacy Push T lineage. Treat dated configuration
  rows as historical audit evidence, not necessarily current defaults.
- `reports/research_roadmap.svg` — project-level current/future scope.

## Claim discipline

- “Toppling / planar unreachable” and “tight-gate miss” are measured classifier
  outcomes.
- Specific root causes such as inaccurate inertia or center of mass are
  hypotheses unless explicitly identified as proposals.
- “Stopped after repeated failures” is an operator terminal decision, not a
  mathematical proof that the objects can never succeed.
- The attempt success rate excludes incomplete attempts.
- This is a simulation-port campaign, not the original paper's hardware result.

