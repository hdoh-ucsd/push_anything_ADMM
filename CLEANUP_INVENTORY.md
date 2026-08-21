# Cleanup Inventory

This inventory records proposed dispositions without moving or deleting files.
Research outputs and historical diagnostics remain user-owned evidence.

## Keep in place

- `control/`, `sim/`, and model assets: active reusable implementation.
- `config/`: active configuration paths used by scripts and results.
- `tests/`: active suite, including currently stale or environment-blocked tests.
- `tools/visualizer/`: reusable result visualization pipeline.
- `scripts/reference/`: coherent reference-comparison workflow.
- `README.md`, `REFERENCE.md`, `REPORT.md`: current project narrative and reports.
- committed audit CSVs: preserve as numerical evidence until provenance is
  transferred to a regression-fixture manifest.

## Safe generated-file candidates

These can be regenerated or are process leftovers, but are not approved for
deletion:

| Path/pattern | Current status | Proposed reversible action |
| --- | --- | --- |
| `__pycache__/`, `*.pyc` | ignored cache | Leave ignored; remove only with explicit approval. |
| `.pytest_cache/` | ignored cache with stale history | Leave ignored; remove only with explicit approval. |
| `gurobi.log` | ignored solver output | Move to a dated local archive or remove after approval. |
| `logs/phase2_sweep.pid` | committed process identifier | Archive with its sweep receipt; later remove from Git history only if explicitly requested. |
| `main.py.tmp.1725.535f0661ccc7` | ignored temporary copy | Compare with `main.py`, then move to `_review/` or remove after approval. |
| `audit_output/exec_qp_sig_*` | untracked QP dumps | Keep until their diagnostic purpose and backup are recorded. |

## Review/archive candidates

- `scripts/probe_*`, `scripts/check_*`, and executable `scripts/test_*`:
  classify by the investigation/result they support before moving.
- `logs/*.log`, `logs/*.out`, and `logs/*.pid`: separate run evidence from
  repository source after checksums and provenance are recorded.
- `docs/superpowers/investigations/` and dated plans: retain as historical
  research records; consider a documented archive, not deletion.
- partial/aborted/timeout files under `results/`: retain until negative-result
  value and canonical-run relationships are reviewed.
- `sim/models/xbox/xbox_backup.obj` and
  `sim/models/expo_box/expo_box_backup.obj`: large backup-named assets; verify
  geometry references and hashes before any move.
- empty `models/drake_models/`: verify no tooling expects this path before
  removal.

## Duplicate candidates

- `audit_output/subtick_pa0p005.csv` and `subtick_pa0p02.csv` were byte-identical
  during the initial audit. Preserve both until experiment intent explains why
  differently named parameter runs produced identical data.
- Large results and model assets are repeated in `.claude/worktrees/`. Clean up
  worktrees only after their branches are merged, backed up, or explicitly
  abandoned.

## Proposed future moves requiring separate review

| Current material | Proposed destination |
| --- | --- |
| experiment/sweep launchers | `experiments/<study>/` or `benchmarks/runners/` |
| parsers and aggregators | `scripts/analysis/` |
| plots | `scripts/plotting/` |
| one-shot probes | `scripts/diagnostics/` or `scripts/archive/` |
| curated numerical baselines | `tests/fixtures/` or `reference_results/` |
| disposable run output | `results/runs/` |
| canonical external results | external versioned storage plus repository manifest |

No move should occur until references in shell scripts, Markdown, configs,
Obsidian notes, and recorded commands have been checked.
