# Push Anything ADMM

A **PyDrake** port of Contact-Implicit MPC (CI-MPC) for non-prehensile pushing, reimplementing Bui et al., *"Push Anything"* (ICRA 2026) and the DAIR Lab `sampling_c3` stack in Python. A 7-DOF Franka Emika Panda pushes an object toward an SE(2)/SE(3) goal by linearising the Drake plant into a discrete Linear Complementarity System (LCS) each control step and solving an N-step trajectory as one stacked QP with a custom ADMM solver.

Three layers stack up:

- **C3+** (Bui 2026, *ICRA*) — slack-variable ADMM with closed-form per-component projection. **The default solver.**
- **C3** (Aydinoglu et al. 2024, *T-RO*) — consensus ADMM with LCP / Lorentz-cone projection. Retained as a falsification lever (`--solver c3`), not the production path.
- **Sampling-C3** (Venkatesh et al. 2025, *RA-L*) — outer sample-based dispatcher that reasons globally about contact mode, switching between contact-rich pushing and contact-free repositioning.

The project's organising principle is **reference conformance**: where the port and the DAIR Lab reference disagree, the reference wins, and divergences are treated as defects rather than design choices. See [Reference Conformance](#reference-conformance).

---

## Status

Active research port, measured against the paper's **tight goal** tolerance (position < 0.02 m **and** rotation < 0.1 rad):

| Task | Window | Result | Tight goal | Log |
| --- | --- | --- | --- | --- |
| `push_t_mesh` | 180 s | 0.0108 m / 0.0766 rad | **PASS(final)** | `meshT_meshnormal_canon180.log` |
| `pushing` (box) | 180 s | 0.0196 m / 0.0053 rad | **PASS(final)** | `boxgate_sixleg.log` |
| `push_t_block` | 300 s | 0.0195 m / 0.0461 rad | **PASS(final)** | `blockT_muintent_ext300.log` |
| `push_t_block` | 180 s | 0.0197 m / 0.0125 rad (best draw) | **PASS(final)** — 2 of 7 draws pass | 7 same-seed runs, 2026-08-20 |

**Block-T at 180 s varies widely between runs of identical code and seed.** Seven consecutive same-seed 180 s draws finished at, in millimetres: **19.7, 20.0, 20.7, 21.7, 23.8, 33.3, 50.7** (rotation was inside tolerance in every one — the spread is purely translational). Two pass tight: one on the final frame (0.0197 m / 0.0125 rad) and one on the latch (reached 0.0197 m / 0.0830 rad at t = 122.4 s, then settled 0.3 mm out). Five of seven land within 24 mm; the two outliers take a bad shove around t ≈ 35–45 s and never recover.

The driver is the documented Ipopt floating-point floor: a contact event resolves one way or the other and the run follows a different branch from there. Two runs of *identical* code diverge measurably by t ≈ 44 s, so a single 180 s number is a draw from this spread, not a repeatable measurement — quote the distribution, and never conclude anything from one run. The 300 s window is materially safer: the endgame latch typically lands at t ≈ 186 s, just past the canonical cutoff.

> Earlier revisions of this section called the spread "tight (1.7 mm)" from three draws and then "bimodal" from five. Both were over-readings of a small sample; the seven-draw list above is given as raw data rather than a shape.

> A previously-quoted block-T result of 0.0190 m / 0.0887 rad at 180 s (`blockT_retention_canon180.log`, commit `df337c9`) was produced under the **literal-μ** planner friction (`EE-BOX: 1.0`). HEAD ships **intent-μ** (`EE-BOX: 0.4615`, the effective-friction reading of the reference), so that number is not reproducible at HEAD and should not be quoted as a current result.

> **Sticky achievement record (`16f03ac`).** The reference's achieved-goal flag is sticky forever (cc:887-897). The port's authorized release deviation (`achieved_goal_release_loops`) un-pins dispatch after post-latch drift so the arm can re-engage — but it previously also erased the *verdict*: a run that latched joint-tight at t = 122.4 s reported `FAIL(-)` after a 0.4 mm settle triggered the release. The achievement record (`_tight_ever_latched`) is now a separate flag: set the instant both criteria hold, cleared only on a goal change, and read by `[RESULT]` — while the release keeps its dispatch role (it is what rescued the box endgame from stranding). Pinned by `tests/test_tight_latch_record.py`.

For calibration: on the same T-pushing task, the **reference C++ stack never gets closer than 0.143 m in 1800 s** — it stalls in repositioning, completing only 6 repositions in 30 minutes, with 93% of planner loops never reaching the pushing phase. The port's advantage is not a better solver; it is conforming to the authors' `anything` pipeline lineage, whereas the shipped `push_t` demo configs are bit-rotted. Compare `results/reference_pusht_sidepanel.mp4` against `results/blockT_tight_sidepanel.mp4`.

**`push_jack`** (SE(3) 3-armed jack, requiring topple plus reorientation) reaches goals and induces flips, but is not yet a reliable tight-goal task.

---

## Quick Start

```bash
# Canonical T-push (mesh T), 180 s
python main.py push_t_mesh --sampling-c3 config/sampling_c3_kik_t.yaml --max-time 180

# Rectangular block-T
python main.py push_t_block --sampling-c3 config/sampling_c3_kik_t.yaml --max-time 180

# Box push
python main.py pushing --sampling-c3 config/sampling_c3_kik.yaml --max-time 180

# SE(3) jack
python main.py push_jack --sampling-c3 config/sampling_c3_kik_jack.yaml --max-time 180
```

Add `--seed 0` for a reproducible sampler draw (all canonical results above are seeded).

Meshcat opens at `http://127.0.0.1:7000`. Every run mirrors stdout to `results/<stem>.txt` and ends with a machine-readable `[RESULT]` line carrying translational/rotational error and the tight/loose verdicts.

Reference-conformant behavior is the **default** — there are no opt-in conformance flags to remember. Videos are rendered *after* a run from its log, not during it (see [Tooling](#tooling)).

---

## Architecture

The stack is bilevel. An outer sampler decides *where* the end-effector should be; an inner ADMM MPC decides *what force* to apply once it is there. Both feed a 1 kHz Operational-Space Controller that converts Cartesian commands into joint torques.

```
                   ┌────────────────────────────────┐
                   │   Sampling-C3 dispatcher       │  (Venkatesh 2025)
                   │   sample → rank → mode switch  │
                   └───────────────┬────────────────┘
                                   │  per planner tick (~13 Hz)
                   ┌───────────────┴────────────────┐
                   ▼                                ▼
        ┌────────────────────┐          ┌───────────────────────┐
        │  c3 (rich) mode    │          │  free (repos) mode    │
        │  C3+ ADMM MPC      │          │  PWL lift→traverse→   │
        │  u ∈ ℝ³ EE force   │          │  descend trajectory   │
        └─────────┬──────────┘          └──────────┬────────────┘
                  └───────────────┬────────────────┘
                                  ▼
                     ┌─────────────────────────┐
                     │  OSC  (1 kHz)           │
                     │  QP: track ẍ_des,       │
                     │  posture, λ_ext         │
                     └────────────┬────────────┘
                                  ▼
                     ┌─────────────────────────┐
                     │  τ ∈ ℝ⁷  →  Drake plant │
                     └─────────────────────────┘
```

**The MPC decision variable is Cartesian end-effector force** (`u ∈ ℝ³`, Newtons), matching the reference; the planner state is `n_x = 19` (object quaternion and position, EE position, and their velocities). The legacy port-only ℝ⁷ joint-torque planner is still selectable via `--r7` but is not the production path.

> **Historical note.** Earlier revisions of this README described a joint-torque formulation with "OSC not on the roadmap." That is obsolete: EE-space force and the OSC layer are both implemented and are now the default.

---

## Algorithm

### The LCS abstraction

Both solver variants consume the same linearised system:

```
x[k+1] = A x[k] + B u[k] + D λ[k] + d
0 ≤ λ  ⊥  E x[k] + F λ[k] + H u[k] + c  ≥ 0
```

`lcs_formulator.py` builds this live from Drake each tick: signed-distance queries give contact points and normals, `CalcJacobianTranslationalVelocity` gives `J_n`/`J_t`, and first-order autodiff gives the dynamics blocks.

The default contact model is **Anitescu** (folded friction, `n_λ = 4·n_c`), matching the reference's `contact_model: 'anitescu'`. The Stewart-Trinkle formulation (`n_λ = 6·n_c`, with explicit friction-cone slack γ) is retained as a fallback path.

### C3+ (Bui 2026) — default

Augments the decision variable with a slack η, `z = (x, λ, u, η)`, and bakes `η = Ex + Fλ + Hu + c` into the QP as a hard equality. The δ-update then becomes a closed-form case analysis on each scalar pair `(λ°, η°)` (Bui eq. 12, with `r = √(u_λ/u_η)`):

| Case | Condition | δ_λ | δ_η |
| --- | --- | --- | --- |
| 1 | `η° ≥ 0` and `η° ≥ r·λ°` | 0 | η° |
| 2 | `λ° ≥ 0` and `η° < r·λ°` | λ° | 0 |
| 3 | otherwise | 0 | 0 |

This replaces a cone projection with arithmetic — roughly 4–5 orders of magnitude faster on the projection step (Bui Table III).

### C3 (Aydinoglu 2024) — falsification lever

Decision variable `z = (x, λ, u)`; the δ-update solves an LCP per timestep, `δ_λ = solve_lcp(F, E·δ_x + H·δ_u + c)`, producing complementarity-feasible λ by construction. Kept for A/B experiments; the C3+ path has superseded it in production.

### Sampling-C3 outer wrapper

Each planner tick:

1. **Sample** candidate EE placements around the object. The strategy is per-task — `kMeshNormal` (area-weighted draws on mesh side-walls, offset along the face normal) for the mesh T, perimeter/projection strategies elsewhere.
2. **Rank** each sample by a short C3 rollout cost, using an object-outcome-only cost (`cost_type` 5) that zeroes the EE position/velocity terms.
3. **Switch modes** with hysteresis: `c3` (delegate to the inner MPC) or `free` (execute a piecewise-linear lift → traverse → descend reposition).
4. **Retain targets.** Reposition targets change only through the reference's `kToBetterRepos` hysteresis — *not* by per-tick argmin. Pure re-selection was a port divergence that starved the solver with a ~117 mm/tick teleport storm.

---

## Reference Conformance

The reference implementation lives at `/root/reference_repos/dairlib_sampling_c3`, branch `push_anything_dev` @ `257e3ede`. Conformance work follows a few rules learned the expensive way:

- **The reference is the specification.** A metric regression on the way to matching reference behavior is a diagnostic signal, not a reason to revert.
- **No off-reference knob probes.** Tuning with no counterpart in the reference is not a fix.
- **Divergences are catalogued, not rediscovered.** `docs/conformance-map.md` is a ~75-entry subsystem-by-subsystem audit (port line ↔ reference line ↔ load-bearing verdict). Read it before making any conformance change.
- **Lineage matters more than mechanism.** Several long-lived bugs traced to porting the right code from the *wrong* reference demo. The T-pushing sampler is the canonical example: the authors' `anything` pipeline uses `kMeshNormal`, while the `push_t` demo uses perimeter sampling — porting the latter produced an entire class of spin failures.

The reference stack is runnable for head-to-head comparison:

```bash
scripts/reference/run_reference.sh push_t 180 mytag     # → results/reference/push_t_mytag/
scripts/reference/compare_port_vs_reference.py          # aligned metric comparison
```

Related docs: `REFERENCE.md` (quote-and-cite extraction of the reference source) and `REPORT.md` (diagnostic write-up).

---

## Evaluation Protocol

Two tiers, because short runs systematically misrepresent endgame behavior:

| Tier | Duration | Purpose | Quotable? |
| --- | --- | --- | --- |
| **Gate** | 60 s | Fast iteration signal while developing | **No** |
| **Canonical** | 180 s | The number that lands in a claim or this README | Yes |

Further rules that keep results honest:

- **Baseline at the pre-change commit, same protocol** — no comparing against a differently-configured historical run.
- **Never parallelise evaluation runs.** They contend for CPU, and the controller is latency-sensitive (see [Performance](#performance)).
- **No multi-seed sweeps as evidence.** A sweep that averages over a mechanism is not an explanation; root-cause instead.
- **Count log statistics with Python, not grep.** The logs have pathologically long lines, and some grep builds print nothing rather than zero on no-match.

---

## Performance

Runtime is dominated by the outer loop's per-sample ADMM solves — ~85% of wall time, versus ~4% for the 1 kHz OSC path.

A 2026-08-20 profile found that Drake re-derives the QP Hessian's PSD verdict on every `AddQuadraticCost` / `UpdateCoefficients` call when `is_convex` is left unset — an O(n³) check on a 559×559 matrix (push_t_block: N=10, TOT=54), roughly 15,000 times per run (~42% of solver time) — plus a redundant coefficient push that was always overwritten before any solve (~18%). Caching Drake's own verdict and dropping the dead push gave:

| | Before | After |
| --- | --- | --- |
| Wall clock, 60 s sim | 491 s | **340 s** (1.44×) |
| Committed C3+ solve | 78–85 ms | **50–55 ms** |

Verified bit-exact at the solver level: 517/517 `_solve_c3plus` calls return identical `(u_seq, x_seq)` against the pre-change implementation.

> **Perf changes cannot be validated by comparing run logs.** The controller feeds its own measured solve latency into the predicted-x0 clamp (`filtered_solve_time`, mirroring reference `cc:1394,1460`), so a faster solver legitimately produces a different trajectory — exactly as faster hardware would. Two runs of *identical* code already diverge. Validate instead by replaying the old implementation on identical inputs, snapshotting and restoring solver state around each call (`_u_prev_solve` is a warm start).

**Open lever:** sample evaluation is serial (`num_threads_to_use: 1`) while the reference runs `#pragma omp parallel for` with `num_outer_threads: 4`. The port already ships the thread pool and per-thread solver kits; the reference literal is unmeasured here.

Profiling entry point:

```bash
python profiling/profile_run.py <task> <seconds>   # cProfile + SectionTimer
```

Section labels: `lcs.extract_dynamics`, `lcs.geometry_query`, `lcs.calc_jacobians`, `admm.qp_build`, `admm.osqp_solve`, `admm.z_update`, `admm.final_qp`.

> cProfile cannot see pybind11 *instance* methods — Drake `Solve`/`UpdateCoefficients` time is silently folded into the calling Python function's "self" time. Use `line_profiler` to split it out.

---

## Tasks

Task definitions live in `config/tasks.yaml`, and the CLI's task list is read from that file — adding a task requires no Python changes.

| Task | Object | Notes |
| --- | --- | --- |
| `pushing` | box | Box push; `--task-id 1..4` selects N/E/S/W goals |
| `push_t_mesh` | non-convex mesh T | Canonical T task; `kMeshNormal` sampler |
| `push_t_block` | rectangular Push-T | Built from byte-identical reference SDFs |
| `push_h` | H-shape | |
| `push_jack` | 3-armed jack | SE(3); needs topple plus reorientation |
| *imported objects* | milk, clamp, tape, book, … | Fig-8 campaign objects (`scripts/import_anything_objects.py`) |

`hard_pushing`, `shepherding`, and `cube_turning` remain in `tasks.yaml` but are **stale early-phase entries** — selectable, unmaintained.

---

## Repository Structure

```
push_anything_ADMM/
├── main.py                        # Entry point: argparse, env build, sim loop, [RESULT]
├── config/
│   ├── tasks.yaml                 # All task params (geometry, mass, μ, goals, weights)
│   ├── sampling_c3_kik*.yaml      # Per-task sampling-C3 configs (t / h / jack / …)
│   ├── osc_franka.yaml            # OSC gains
│   └── directional_tasks.json     # N/E/S/W goal variants for --task-id
├── sim/env_builder.py             # Drake diagram: table + Panda + pusher + object
├── control/
│   ├── lcs_formulator.py          # Drake plant → A,B,D,d, J_n,J_t, φ, μ, E,F,H,c
│   ├── admm_solver.py             # C3 / C3+ ADMM core (z/δ/ω, adaptive ρ, final QP)
│   ├── ci_mpc_c3plus.py           # C3+ inner controller (EE-space force)
│   ├── ci_mpc_c3.py               # C3 inner controller (legacy lever)
│   ├── lcp_solver.py              # LCP solve for the C3 δ-update
│   ├── task_costs.py              # Quadratic manipulation cost (Q/R/QN)
│   ├── osc/                       # 1 kHz Operational-Space Controller (QP + dynamics)
│   └── sampling_c3/               # Outer wrapper: sampling, ranking, mode switch,
│                                  #   reposition, progress, goal generation, IK
├── tools/visualizer/              # Log → Drake-scene frames → composite video
├── scripts/
│   ├── make_run_video.sh          # Composite run video from a results/ log
│   ├── reference/                 # Reference-stack harness + comparison tooling
│   └── …                          # Diagnostic / sweep / parse helpers
├── profiling/                     # SectionTimer + cProfile wrapper
├── tests/                         # pytest suite (344 cases)
├── docs/conformance-map.md        # Port ↔ reference divergence audit
├── REFERENCE.md                   # Curated reference-source extraction
└── REPORT.md                      # Diagnostic investigation write-up
```

---

## Tooling

**Run videos.** Rendered from a completed run's log, so visualisation never perturbs a timed run:

```bash
scripts/make_run_video.sh <RUN_NAME> --task push_t_block --stride 4 --fps 27
```

This renders Drake-scene frames (`tools/visualizer/render_log_drake_scene.py`; camera via `PORT_CAM_EYE` / `PORT_CAM_TARGET`) and composites a 2080×720 side-panel video with sticky milestones and a rolling log (`paint_log_sidepanel.py`). `--goal-xy` / `--goal-yaw` override the goal ghost for multi-goal segment renders.

**Reference video.** `tools/visualizer/convert_reference_jack_log.py` (and its `push_t` equivalent) converts a reference `c3controller.log` state stream into port log format, so both stacks can be rendered in the same scene and camera for side-by-side comparison.

---

## Testing

```bash
python -m pytest tests/ -q
```

**308 of 344 pass.** The 36 failures are known and concentrated in `test_mode_switch.py` (9), `test_progress.py` (8), `test_lcs_efhc.py` (6), `test_sampling_strategies.py` (3), and the OSC unit tests (5) — mostly stale assertions pinning pre-conformance behavior that later reference-conformance work deliberately changed. They are tracked, not silently tolerated: a new failure outside that set is a real regression.

---

## Dependencies

| Package | Purpose |
| --- | --- |
| `pydrake` | Physics, geometry queries, MathematicalProgram, OSQP, Meshcat |
| `numpy` | Numerics |
| `pyyaml` | Task + controller config loading |
| `pytest` | Test suite |
| `ffmpeg` | Video compositing (`scripts/make_run_video.sh`) |
| `line_profiler` | Optional; line-level solver profiling |

Gurobi is optional and relevant only to the reference C++ stack (see `scripts/reference/`).

---

## Citations

H. Bui et al., **"Push Anything: Single- and Multi-Object Pushing From First Sight with Contact-Implicit MPC,"** arXiv:2510.19974v2, 2025.

A. Aydinoglu, A. Wei, W.-C. Huang, and M. Posa, **"Consensus Complementarity Control for Multi-Contact MPC,"** *IEEE Transactions on Robotics*, vol. 40, pp. 3879–3896, 2024.

S. Venkatesh, B. Bianchini, A. Aydinoglu, W. Yang, and M. Posa, **"Approximating Global Contact-Implicit MPC via Sampling and Local Complementarity,"** *RA-L* 2025, arXiv:2505.13350.

W. Heemels, J. M. Schumacher, and S. Weiland, **"Linear Complementarity Systems,"** *SIAM Journal on Applied Mathematics*, vol. 60, no. 4, pp. 1234–1269, 2000.

---

## Appendix: CLI Reference

```
python main.py TASK [options]
```

| Flag | Default | Description |
| --- | --- | --- |
| `TASK` | `pushing` | Task name; choices read from `config/tasks.yaml` |
| `--sampling-c3 [PATH.yaml]` | off | Enable the sampling-C3 outer controller. Bare flag → `config/sampling_c3_params.yaml` |
| `--solver {c3,c3plus}` | `c3plus` | Inner ADMM solver. `c3` is a deprecated falsification lever |
| `--task-id {1,2,3,4}` | — | Directional goal override (1=N, 2=E, 3=S, 4=W) from `config/directional_tasks.json` |
| `--max-time T` | 8.0 | Simulation duration in seconds |
| `--admm-iter N` | 3 | ADMM iterations per control step |
| `--seed INT` | — | Seed the sampler RNG for reproducible draws |
| `--name BASENAME` | `<task>_<timestamp>` | Log stem under `results/` |
| `--math-diag` | off | Verbose `[MATH.*]` solver diagnostics (zero overhead when off) |
| `--r7` | off | **Legacy**: port-only ℝ⁷ joint-torque full-plant planner instead of EE-space force |

Behavioral variation beyond these flags is driven by YAML config, not CLI switches — deliberately, so a run is reproducible from its config plus its logged `[RUN-META]` line.

### Key MPC parameters

Values are task-conditional (set per task and YAML); the figures below are `push_t_block`'s, as printed in every run's banner.

| Parameter | Value |
| --- | --- |
| Horizon `N` | 10 |
| Planning `dt` | 0.075 s (position and pose) |
| ADMM iterations | 3 |
| Initial ρ | 100.0 |
| EE force limit | 30.0 N |
| Planner state / input | `n_x = 19`, `n_u = 3` |
