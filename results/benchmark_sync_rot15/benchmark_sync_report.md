# benchmark_v2_rot15 synchronization report

## Outcome

- Old benchmark (`benchmark_v1_25pair`): 5 starts x 5 goals = 25 cases / scene.
- New benchmark (`benchmark_v2_rot15`): 5 starts x 1 translational goal x 3 relative yaw demands = 15 cases / scene.
- Total: 90 nominal cases over 6 scenes.
- Geometry gate: **PASS** (90/90 raw-feasible; zero penetrating targets).
- Cross-method numeric parity: **PASS** (90/90).
- Legacy-source numeric audit: **PASS** (210 checks; 0 skipped because an optional checkout was absent).
- Offline launcher-resolution smoke: **PASS** (the `open_task/s01` rotation triplet resolves for both methods without starting a controller).
- Native controller smoke: **BLOCKED** (see Verification and campaign blockers).
- Full 90-case campaign: **NOT RUN**.

## Canonical sources and conventions

- Starts `s01`...`s05` and `g02` XY/yaw source: upstream OIM `examples/poses/{oim_scene}.yaml` at `d6d80a65ba6a45079e1384ba879e54d50d25bc39`. Only the `g02` translation is retained; its historical absolute yaw is deliberately ignored.
- C3+ source audit: `examples/sampling_c3/{demo}/parameters/{sim_params.yaml,goal_params.yaml}` at `bd5ce7b52d7133d54065578e42874d92d3cce95c`.
- Geometry: `benchmarks/rot15/scene_sources.yaml`, transcribed from `tools/scene_smoke/scene_configs/{scene}.yaml` and checked against `systems/controllers/sampling_based_c3_controller.cc`. `icra_sign` uses `push_c_glyph` and the actual three-box C footprint plus all seven fixed glyph hulls.
- Scene IDs are the runner's canonical names: `open_task`, `single_obstacle`, `shelf_gap`, `ycb_clutter`, `icra_sign`, `slalom`. OIM maps only `open_task` to its native `open_table` scene key.
- Yaw composition: `theta_goal = wrap_to_pi(theta_start + delta_theta)` with deltas `0`, `+pi/2` (CCW), `-pi/2` (CW).
- Wrap convention: `[-pi, pi)`.
- C3+ quaternion interface: normalized `[w,x,y,z]`; `Eigen::Quaterniond(v[0],v[1],v[2],v[3])`.
- OIM tabletop interface: `[x,y,yaw]` planar slide/hinge coordinates, so no quaternion is passed. The parity artifact independently constructs SciPy-style `[x,y,z,w]` oracle quaternions and converts them back to yaw.
- Seed policy: `17000 + 100*scene_index + start_index` (zero-based scene index, one-based start index). Rotation does not affect the seed; matched C3+/OIM cases receive the same integer.

## Geometry preflight

`raw_min_signed_distance` reproduces the synchronized controller representation: the orientation-aware `TFootprint`/`CFootprint` boundary samples evaluated against exact AABBs, convex hulls, and the robot-base disc. Negative is penetration, zero is touching, positive is clearance.

| scene | cases | minimum raw clearance | penetrating |
|---|---:|---:|---:|
| `open_task` | 15 | +inf (no obstacles) | 0 |
| `single_obstacle` | 15 | 0.332296495 m | 0 |
| `shelf_gap` | 15 | 0.276095717 m | 0 |
| `ycb_clutter` | 15 | 0.228062576 m | 0 |
| `icra_sign` | 15 | 0.042458338 m | 0 |
| `slalom` | 15 | 0.104296495 m | 0 |

C3+ `lcs_contact` additionally subtracts a 0.01 m hard object-obstacle margin (`phi = d_raw - margin`). OIM at `d6d80a65ba6a45079e1384ba879e54d50d25bc39` has physical MuJoCo nonpenetration and a soft exponential object-obstacle cost, but no extra shared hard object margin. `margin_adjusted_clearance` in the CSV is therefore C3+-specific, not a shared benchmark feasibility definition.

## Cross-method parity

Every canonical row is independently translated to C3+ YAML semantics and OIM planar-pose semantics, serialized, read back, and converted to SE(2). Start XYZ/yaw, goal XYZ/yaw, relative yaw, task seed, manipulated-object identity, footprint type, and case identity all match to tolerance `1e-10`. Obstacle geometry is shared by scene from the authoritative specification rather than regenerated in either adapter.

The intended task seed is explicit and identical in both adapters. The audited C3+ perimeter generator currently constructs `std::mt19937` from `std::random_device`, however, and exposes no task-level seed in these demo YAMLs. OIM consumes `--seed`. Therefore task-pose parity is complete, but common random numbers are a **campaign blocker** until a separately approved C3+ seed-plumbing change is made; this benchmark task does not change sampling behavior.

## Legacy preservation and controller freeze

The legacy `tools/scene_smoke/gen_grid_dirs.py` and `tools/scene_smoke/run_grid_campaign.py` 5x5 path is not modified. No controller, solver, cost, ADMM, MPC, sampling, horizon, threshold, dynamics, contact, or scene-physics parameter is changed. New launch-time C3+ configs are copies of the legacy `sXXg02` task config with only the canonical start pose and g02 goal pose changed. A task receipt stores the intended seed and case identity, but it does not falsely claim that the current C3+ sampler consumes that seed.

## Verification and campaign blockers

- `python -m pytest tests/test_rot15_benchmark.py -q`: **11 passed**.
- `python -m benchmarks.rot15.benchmark validate`: **PASS** (90 cases, geometry, and parity).
- `python -m benchmarks.rot15.launcher --method both --scene open_task --smoke-triplet --dry-run --cap 5 --port-base 19000`: **PASS**, six native command lines resolved and zero simulations started.
- A native OIM parser/launcher smoke stops during import with `ModuleNotFoundError: No module named 'jax'`; the available environment also lacks MuJoCo. No dependency installation was attempted.
- The checked-out OIM submodule is `a954f00047b6f2f302e83176124f482b2ec1a2e7`; the audited canonical OIM source is `d6d80a65ba6a45079e1384ba879e54d50d25bc39`, and the current checkout lacks `examples/pusht/slalom.py`. The source is present in the audited upstream Git object but is not silently copied into this checkout.
- C3+ task-pose consumption can be materialized additively, but deterministic seed consumption is not supported by the audited sampler. The real `--all` launcher therefore has an intentional hard stop.
- The repository-wide `python -m pytest tests` result in this existing dirty environment was **35 failed, 455 passed, 56 skipped, 1 xfailed, 15 errors**. The new rot15 tests all pass; the other failures include unavailable Meshcat/websocket services and pre-existing configuration/progress/mode expectations and were not altered.

The benchmark definitions, geometry gate, and adapter parity are ready. A full campaign and cost fine-tuning remain blocked until C3 seed plumbing is separately approved and verified, and until an OIM environment/checkout containing the audited scene scripts is available. No controller code is changed in this task.

## Files added or modified

- `.gitignore` (track only the small deterministic synchronization artifacts under the otherwise ignored `results/` tree)
- `benchmarks/__init__.py`
- `benchmarks/rot15/__init__.py`
- `benchmarks/rot15/scene_sources.yaml`
- `benchmarks/rot15/benchmark.py`
- `benchmarks/rot15/launcher.py`
- `benchmarks/rot15/oim_entry.py`
- `tests/test_rot15_benchmark.py`
- `results/benchmark_sync_rot15/*` (generated CSV/YAML/report evidence; no simulation logs)

## Reproducibility

- Outer repository commit: `4316b1ea7325b14f1de3dac6fa197e3b6aa4a0d3`.
- Outer repository dirty state at generation: `["M .gitignore", " D results/fig8_consecutive_failures/previews/book.gif", " D results/fig8_consecutive_failures/previews/clamp.gif", " D results/fig8_consecutive_failures/previews/eraser.gif", " D results/fig8_consecutive_failures/previews/gallon-milk.gif", " D results/fig8_consecutive_failures/previews/letter-3.gif", " D results/fig8_consecutive_failures/previews/letter-a.gif", " D results/fig8_consecutive_failures/previews/letter-b.gif", " D results/fig8_consecutive_failures/previews/letter-c.gif", " D results/fig8_consecutive_failures/previews/letter-e.gif", " D results/fig8_consecutive_failures/previews/letter-g.gif", " D results/fig8_consecutive_failures/previews/letter-h.gif", " D results/fig8_consecutive_failures/previews/letter-i.gif", " D results/fig8_consecutive_failures/previews/letter-r.gif", " D results/fig8_consecutive_failures/previews/letter-s.gif", " D results/fig8_consecutive_failures/previews/letter-y.gif", " D results/fig8_consecutive_failures/previews/lotion.gif", " D results/fig8_consecutive_failures/previews/push-t.gif", " D results/fig8_consecutive_failures/previews/tape.gif", " D results/fig8_consecutive_failures/previews/wood-block.gif", " D results/fig8_consecutive_failures/videos/book.mp4", " D results/fig8_consecutive_failures/videos/clamp.mp4", " D results/fig8_consecutive_failures/videos/eraser.mp4", " D results/fig8_consecutive_failures/videos/gallon-milk.mp4", " D results/fig8_consecutive_failures/videos/letter-3.mp4", " D results/fig8_consecutive_failures/videos/letter-a.mp4", " D results/fig8_consecutive_failures/videos/letter-b.mp4", " D results/fig8_consecutive_failures/videos/letter-c.mp4", " D results/fig8_consecutive_failures/videos/letter-e.mp4", " D results/fig8_consecutive_failures/videos/letter-g.mp4", " D results/fig8_consecutive_failures/videos/letter-h.mp4", " D results/fig8_consecutive_failures/videos/letter-i.mp4", " D results/fig8_consecutive_failures/videos/letter-r.mp4", " D results/fig8_consecutive_failures/videos/letter-s.mp4", " D results/fig8_consecutive_failures/videos/letter-y.mp4", " D results/fig8_consecutive_failures/videos/lotion.mp4", " D results/fig8_consecutive_failures/videos/push-t.mp4", " D results/fig8_consecutive_failures/videos/tape.mp4", " D results/fig8_consecutive_failures/videos/wood-block.mp4", " D results/fig8_consecutive_gallery/previews/baby-toy.gif", " D results/fig8_consecutive_gallery/previews/chicken-broth.gif", " D results/fig8_consecutive_gallery/previews/egg-carton.gif", " D results/fig8_consecutive_gallery/previews/expo-box.gif", " D results/fig8_consecutive_gallery/previews/milk-bottle.gif", " D results/fig8_consecutive_gallery/previews/xbox.gif", " D results/fig8_consecutive_gallery/videos/baby-toy.mp4", " D results/fig8_consecutive_gallery/videos/chicken-broth.mp4", " D results/fig8_consecutive_gallery/videos/egg-carton.mp4", " D results/fig8_consecutive_gallery/videos/expo-box.mp4", " D results/fig8_consecutive_gallery/videos/milk-bottle.mp4", " D results/fig8_consecutive_gallery/videos/xbox.mp4", " D results/fig8_success_gallery/index.html", " D results/fig8_success_gallery/previews/baby-toy.gif", " D results/fig8_success_gallery/previews/book.gif", " D results/fig8_success_gallery/previews/chicken-broth.gif", " D results/fig8_success_gallery/previews/clamp.gif", " D results/fig8_success_gallery/previews/egg-carton.gif", " D results/fig8_success_gallery/previews/eraser.gif", " D results/fig8_success_gallery/previews/expo-box.gif", " D results/fig8_success_gallery/previews/gallon-milk.gif", " D results/fig8_success_gallery/previews/letter-3.gif", " D results/fig8_success_gallery/previews/letter-a.gif", " D results/fig8_success_gallery/previews/letter-b.gif", " D results/fig8_success_gallery/previews/letter-c.gif", " D results/fig8_success_gallery/previews/letter-e.gif", " D results/fig8_success_gallery/previews/letter-g.gif", " D results/fig8_success_gallery/previews/letter-h.gif", " D results/fig8_success_gallery/previews/letter-i.gif", " D results/fig8_success_gallery/previews/letter-r.gif", " D results/fig8_success_gallery/previews/letter-s.gif", " D results/fig8_success_gallery/previews/letter-y.gif", " D results/fig8_success_gallery/previews/lotion.gif", " D results/fig8_success_gallery/previews/milk-bottle.gif", " D results/fig8_success_gallery/previews/push-t.gif", " D results/fig8_success_gallery/previews/tape.gif", " D results/fig8_success_gallery/previews/wood-block.gif", " D results/fig8_success_gallery/previews/xbox.gif", " D results/fig8_success_gallery/thumbnails/baby-toy.jpg", " D results/fig8_success_gallery/thumbnails/book.jpg", " D results/fig8_success_gallery/thumbnails/chicken-broth.jpg", " D results/fig8_success_gallery/thumbnails/clamp.jpg", " D results/fig8_success_gallery/thumbnails/egg-carton.jpg", " D results/fig8_success_gallery/thumbnails/eraser.jpg", " D results/fig8_success_gallery/thumbnails/expo-box.jpg", " D results/fig8_success_gallery/thumbnails/gallon-milk.jpg", " D results/fig8_success_gallery/thumbnails/letter-3.jpg", " D results/fig8_success_gallery/thumbnails/letter-a.jpg", " D results/fig8_success_gallery/thumbnails/letter-b.jpg", " D results/fig8_success_gallery/thumbnails/letter-c.jpg", " D results/fig8_success_gallery/thumbnails/letter-e.jpg", " D results/fig8_success_gallery/thumbnails/letter-g.jpg", " D results/fig8_success_gallery/thumbnails/letter-h.jpg", " D results/fig8_success_gallery/thumbnails/letter-i.jpg", " D results/fig8_success_gallery/thumbnails/letter-r.jpg", " D results/fig8_success_gallery/thumbnails/letter-s.jpg", " D results/fig8_success_gallery/thumbnails/letter-y.jpg", " D results/fig8_success_gallery/thumbnails/lotion.jpg", " D results/fig8_success_gallery/thumbnails/milk-bottle.jpg", " D results/fig8_success_gallery/thumbnails/push-t.jpg", " D results/fig8_success_gallery/thumbnails/tape.jpg", " D results/fig8_success_gallery/thumbnails/wood-block.jpg", " D results/fig8_success_gallery/thumbnails/xbox.jpg", " D results/fig8_success_gallery/videos/baby-toy.mp4", " D results/fig8_success_gallery/videos/book.mp4", " D results/fig8_success_gallery/videos/chicken-broth.mp4", " D results/fig8_success_gallery/videos/clamp.mp4", " D results/fig8_success_gallery/videos/egg-carton.mp4", " D results/fig8_success_gallery/videos/eraser.mp4", " D results/fig8_success_gallery/videos/expo-box.mp4", " D results/fig8_success_gallery/videos/gallon-milk.mp4", " D results/fig8_success_gallery/videos/letter-3.mp4", " D results/fig8_success_gallery/videos/letter-a.mp4", " D results/fig8_success_gallery/videos/letter-b.mp4", " D results/fig8_success_gallery/videos/letter-c.mp4", " D results/fig8_success_gallery/videos/letter-e.mp4", " D results/fig8_success_gallery/videos/letter-g.mp4", " D results/fig8_success_gallery/videos/letter-h.mp4", " D results/fig8_success_gallery/videos/letter-i.mp4", " D results/fig8_success_gallery/videos/letter-r.mp4", " D results/fig8_success_gallery/videos/letter-s.mp4", " D results/fig8_success_gallery/videos/letter-y.mp4", " D results/fig8_success_gallery/videos/lotion.mp4", " D results/fig8_success_gallery/videos/milk-bottle.mp4", " D results/fig8_success_gallery/videos/push-t.mp4", " D results/fig8_success_gallery/videos/tape.mp4", " D results/fig8_success_gallery/videos/wood-block.mp4", " D results/fig8_success_gallery/videos/xbox.mp4", "?? benchmarks/", "?? config/snapshots/", "?? docs/C3PLUS_PORT_AND_OBSTACLE_NOTES.md", "?? results/benchmark_sync_rot15/", "?? scripts/c3ab/", "?? tests/test_rot15_benchmark.py"]`.
- Canonical specification SHA-256: `5032a1175ac994b1da59e8bb3db7451d94e413b794e7f08b5b38c2ac737b34d7`.
- Output directory: `/root/push_anything_ADMM/results/benchmark_sync_rot15`.
- Generator: `python -m benchmarks.rot15.benchmark generate`.
- Validator: `python -m benchmarks.rot15.benchmark validate`.

## Artifacts

- `canonical_cases.csv`: 90 canonical task rows.
- `scene_case_summary.csv`: six scene-level count checks.
- `goal_geometry_preflight.csv`: all 90 goal-pose clearances.
- `cross_method_parity.csv`: all 90 adapter parity checks.
- `source_audit.csv`: numeric comparison to the legacy C3+ configs and upstream OIM pose files.
- `c3plus_cases.yaml`, `oim_cases.yaml`: method translations.
- `oim_pose_overrides/*.yaml`: OIM pose-key files generated from the same rows.
- `benchmark_sync_report.md`: this report.
