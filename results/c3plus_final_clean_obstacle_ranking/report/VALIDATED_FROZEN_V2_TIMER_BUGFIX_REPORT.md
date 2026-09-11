# VALIDATED_FROZEN_V2 timer bugfix

## Scope

This report records the confirmed monotonic-clock implementation bugfix. No
33-run recovery row, full campaign, cost, or scientific parameter was run or
changed.

## Exact source change

In `systems/controllers/sampling_based_c3_controller.cc`, only the two
ComputePlan solve-time captures changed:

```diff
- auto start = std::chrono::high_resolution_clock::now();
+ auto start = std::chrono::steady_clock::now();
...
- auto finish = std::chrono::high_resolution_clock::now();
+ auto finish = std::chrono::steady_clock::now();
```

The matching duration computation and indexing semantics are unchanged. The
clean bugfix commit is `5dcf82c499ba2ab272f6a9fcb857c7310376549c`; the
execution worktree was already dirty before this build. Other
`high_resolution_clock` uses (C3/cost instrumentation) were not changed.

## Freeze/equivalence checks

- Timer regression and ranking-boundary tests: **6 passed** (`pytest -q
  tests/test_c3plus_timer_bugfix.py tests/test_c3plus_ranking_only_boundary.py`).
- Bash freeze gate, including ranking-equivalence test: **PASS**.
- Inner configuration SHA unchanged: `d75d68292004e74a154a28e03d0a35f46fbb0441e1b3ea6e0ecda437667c0d7d`.
- Static source-diff audit shows no Q/R/G/U, C3, ADMM, contact, dynamics,
  sampler, ranking, or success-criterion changes. Existing finite fixture
  checks therefore pass (`PASS_STATIC_SOURCE_AND_EXISTING_OFFLINE_FIXTURES`).
- The V1/V2 distinction is provenance-only: V1 is the historical defective
  implementation; V2 is the same scientific algorithm with the monotonic
  timer correction.

## V2 artifact

| Item | Value |
|---|---|
| version | `VALIDATED_FROZEN_V2` |
| controller | `external/oim_c++_anything/.claude/worktrees/oim-scene-sync-metrics/bazel-bin/examples/sampling_c3/franka_sampling_c3_controller` |
| controller SHA-256 | `9459c12910649deb2d503106a85dbfdeaba349f0ff5f2354221d5dc7a155b5a6` |
| inner-config SHA-256 | `d75d68292004e74a154a28e03d0a35f46fbb0441e1b3ea6e0ecda437667c0d7d` |
| compiler | GCC/C++ Ubuntu 13.3.0 |
| Bazel | 8.4.0 |

## Authorized diagnostic validations

The selected open, B1, and B2 identities were run sequentially with the V2
binary, 60-second diagnostic wall limits, exact original task identities, and
run-owned cleanup. A diagnostic timeout is not a task failure; it is an
infrastructure validation stop.

| Identity | Attempt | V2 binary confirmed | negative solve time/index | ALLFINITE log | cleanup |
|---|---:|---|---|---|---|
| open/A1 `FCO_082...OPEN-A1` | 05 | yes | none observed | absent | PASS |
| B1 `FCO_037...B1` | 01 | yes | none observed | absent | PASS |
| B2 `FCO_102...B2` | 04 | yes | none observed | absent | PASS |

Each ended as `DIAGNOSTIC_WALL_TIMEOUT` after the short validation window;
none produced `first_nonfinite.log`, negative filtered solve time,
`last_passed_index=-1`, or `PLANNER_ABORT_ALLFINITE_QV`. The historical
pre-fix reproduction did produce the negative-clock/ALLFINITE signature.

## Recovery manifest

`targeted_b1_b2_recovery_v2_manifest.csv` contains exactly **33** rows,
selected only from the prior B1/B2 ALLFINITE invalidations. Receipt paths
were checked to exist. SHA-256:

`07f01afba2fdb6c0a087d78dde12039cfa75e2e6ad54cefb5769d7ba19c556f1`

The manifest is prepared, not dispatched. Workspace failures, valid failures,
B3 rows, and successful rows are excluded.

## Status

`VALIDATED_FROZEN_V2_READY = YES`

`TARGETED_33_RECOVERY_RUNS_HAVE_NOT_BEEN_LAUNCHED.`

