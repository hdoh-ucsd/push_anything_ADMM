# Test Baseline and Failure Classification

## Scope

This is a diagnostic baseline only. No numerical algorithm, solver option,
configuration, research output, tolerance, or expected numerical value was
changed during classification.

Baseline commit: `018dffb95c6457ab850ef62d49d80e2b7240f2d2`, with the pre-existing
dirty working tree recorded in `audit_output/test_baseline_20260821/environment.txt`.

Artifacts:

- Human-readable transcript: `audit_output/test_baseline_20260821/pytest.txt`
- JUnit XML: `audit_output/test_baseline_20260821/pytest.junit.xml`
- Environment and package inventory: `audit_output/test_baseline_20260821/environment.txt`
- Exact command: `audit_output/test_baseline_20260821/command.txt`
- Machine-readable classification: `tests/baseline_failures.yaml`

Result: **344 collected; 292 passed; 37 failed; 15 errors** in 2.60 seconds.
All 52 non-passing nodes reproduced in two consecutive runs in this session.

## Classification summary

| Classification | Count | Interpretation |
| --- | ---: | --- |
| Environment-only | 25 | Environment construction was blocked before the scientific assertion by unavailable socket networking required by Meshcat. |
| Stale test | 24 | Assertions target superseded interfaces or behavior that current code comments/history identify as intentional. |
| Configuration drift | 2 | Current canonical YAML values differ from assertions; authority must be confirmed before updating either side. |
| Numerical tolerance | 1 | A hard state bound was exceeded by 25.5067 micrometres; solver residual/options must be measured before any tolerance decision. |
| Dependency drift | 0 | No failure was attributed to a package API/version change. |
| Genuine implementation defect | 0 confirmed | One force-tracking test looked suspicious initially, but current code explicitly defines lambda as a cost-only shadow variable, making the test contract stale. |
| Flaky/non-deterministic | 0 observed | Both suite runs produced the same node-level outcome. |
| Missing asset | 0 | All required assets reached by these tests were present. |
| Unknown | 0 provisional | Authority questions remain, but each mechanical cause is identified. |

“Stale test” does not authorize changing expected values. It means the test and
implementation disagree and repository evidence favors a newer documented
contract. The authoritative scientific intent must still be approved.

## Meshcat analysis

Twenty-five nodes share the root exception `RuntimeError: Meshcat failed to
open a websocket port.` Fifteen occur in module fixtures and are reported as
errors; ten LCS tests construct the environment inside their test bodies and
are therefore reported as failures.

Direct diagnostics established:

- A minimal Python `socket.socket()` call fails immediately with
  `PermissionError: [Errno 1] Operation not permitted`.
- A standalone `pydrake.geometry.Meshcat()` fails identically.
- No relevant Meshcat/Python listeners were visible.
- Tests ran serially; no parallel test plugin was used.
- Failure occurs on the first Meshcat construction.
- The file-descriptor limit is 1,048,576, so descriptor exhaustion is not
  indicated.
- `sim/env_builder.py:822` starts Meshcat unconditionally, including for tests
  that only need plant/LCS/OSC data.

Conclusion for this runner:

| Hypothesis | Finding |
| --- | --- |
| Fixed/default port reuse | Not supported; socket creation itself is denied. |
| Parallel tests | Ruled out for this run. |
| Leaked Meshcat processes | Not supported and cannot explain first-call failure. |
| Unavailable networking | Confirmed root cause. |
| Missing teardown | Not causal here, but a risk on network-enabled runners. |
| Test-environment assumptions | Confirmed: model tests assume visualization networking. |

### Proposed Meshcat test strategy (not implemented)

1. Add a session-scoped capability probe that attempts a localhost socket and
   records whether Meshcat networking is available.
2. Refactor environment construction to accept an explicit visualization mode,
   such as `visualization="meshcat" | "none"`; production defaults remain
   unchanged, while model tests request `none`.
3. Provide shared module/session fixtures for Drake plants and contexts instead
   of constructing a new visualizer per test.
4. If a real Meshcat fixture is required, yield it and explicitly delete/close
   it during fixture teardown; do not rely on garbage collection.
5. Introduce markers:
   - `unit`: no Drake/network;
   - `drake`: Drake plant/model, headless-capable;
   - `meshcat`: explicitly requires localhost networking;
   - `integration`: multi-component controller behavior;
   - `numerical_regression`: fixed numerical contract;
   - `slow`: long-running simulation/benchmark.
6. Make `meshcat` tests skip with an explicit capability reason when networking
   is unavailable. LCS and OSC correctness tests must not be skipped merely
   because visualization is unavailable.
7. Add a teardown regression that repeatedly creates and disposes the Meshcat
   fixture on a network-enabled CI runner to detect port/process leakage.

## Non-environment failure groups

The per-node inventory is in `tests/baseline_failures.yaml`. The evidence-led
groups are:

- Mode switching (10 nodes including the commit-face test): implementation
  documents a reference-derived reversal of near/far `_position` hysteresis and
  reordered repos-to-repos inflation. Older tests encode the prior contract.
- Progress tracking (8 nodes): tests dynamically attach removed tick fields,
  while the implementation reads seconds or explicit `_loops` fields; tests
  also encode the old near-goal selection and absolute cost-drop semantics.
- Sampling strategies (3 nodes): `g_hat` is now intentionally ignored by the
  random-circle sampler, and sphere/perimeter strategies are implemented.
- OSC force tracking (1 node): the test expects lambda to enter dynamics;
  current reference-alignment comments explicitly define lambda as a cost-only
  shadow variable and keep it out of dynamics.
- Canonical configuration (2 nodes): jack initial-goal and progress metric
  assertions differ from current YAML. These require intent confirmation.
- OSC joint-2 default (1 node): test says `W_joint2=0`; YAML documents its
  intentional restoration to `1.0` on 2026-08-14.
- Planner bound (1 node): actual maximum `0.15002550673505122` versus hard bound
  `0.15`, exceeding the test allowance by `24.506735` micrometres and the bound
  itself by `25.506735` micrometres. This is the only tolerance-classified node.

## Smallest numerical core before GPU work

The existing minimum should be run as explicit node groups, not inferred from
the whole suite:

### C3/C3+ and ADMM

- `tests/test_c3plus_vs_c3_smoke.py` (5 currently passing)
- `tests/test_solver_scale_plumbing.py` (11 currently passing)
- `tests/test_planner_workspace_bounds.py` (4 passing, 1 tolerance-classified)

### Complementarity/contact projection

- `tests/test_c3plus_projection_eq12.py` (13 passing)
- `tests/test_lcp_projection.py` (6 passing)
- `tests/test_projection.py` (6 passing)

### LCS construction

- `tests/test_lcs_efhc.py` (6 blocked by Meshcat)
- `tests/test_lcs_jacobian.py` (4 blocked by Meshcat)
- `tests/test_lcs_ee_space_ulin_consistency.py` (2 blocked by Meshcat)
- `tests/test_t_mesh_witness_conformance.py` (2 passing)

### MPC and candidate scoring

- `tests/test_inner_solve.py` (6 passing; trajectory-cost scoring)
- `tests/test_task_costs_lateral.py` (6 passing)
- `tests/test_buffer_travel_normalization.py` (3 passing)
- `tests/test_sampling_strategies.py` after contract reconciliation
- focused mode-switch and progress tests after authority reconciliation

### Deterministic rollout/regression gap

No test currently pins an end-to-end deterministic Push Anything rollout with
a commit/config/seed fixture and numerical trajectory digest. Before GPU work,
add at least one small CPU golden case that records:

- fixed LCS matrices and initial state;
- C3 and C3+ first action/state sequence;
- ADMM primal/dual residual sequence;
- complementarity residual;
- candidate scores and selected mode;
- deterministic short rollout digest.

Golden artifacts must be generated from an approved clean CPU baseline and
stored with explicit absolute/relative comparison rules. GPU comparisons should
be added against that baseline, not replace it.

## Prioritized stabilization plan

1. **P0, low risk:** add headless Drake environment support and test markers;
   rerun the 25 blocked tests without changing plant/dynamics construction.
2. **P0, low risk:** split the baseline command into fast numerical-core,
   headless-Drake, Meshcat, and integration commands.
3. **P0, medium risk:** adjudicate authority for mode-switch/progress semantics
   using reference code, dated plans, and canonical experiment manifests before
   changing any tests.
4. **P0, medium risk:** investigate the planner-bound residual using solver
   status, per-knot constraint residuals, OSQP feasibility settings, and repeated
   runs. Do not change the `1e-6` allowance without this evidence.
5. **P1, low risk:** reconcile clearly implemented sampling strategies and
   removed parameter field names in tests, once authority is approved.
6. **P1, medium risk:** adjudicate the two canonical YAML disagreements and the
   joint-2 OSC default against reported runs before updating assertions.
7. **P1, medium risk:** replace the force-tracking dynamics assertion with a
   test of the approved shadow-variable contract, only after confirming that
   contract is authoritative.
8. **P1, medium/high risk:** create an approved CPU deterministic rollout and
   numerical golden fixtures before GPU solver implementation.
9. **P2, low risk:** add CI jobs/markers and require numerical-core success for
   future GPU changes; keep Meshcat-only tests optional on restricted runners.
