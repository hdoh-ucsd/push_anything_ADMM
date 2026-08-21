# Repository Guide

This guide documents the current layout without moving files or changing import
paths.

| Area | Current role | Intended long-term boundary |
| --- | --- | --- |
| `control/` | C3/C3+, ADMM, LCP/LCS, costs, OSC, sampling-C3 | Reusable solver and controller code |
| `sim/` | Drake environment and object models | Reusable simulation infrastructure |
| `config/` | Tasks, controllers, and experimental variants | Split by task/controller/experiment after manifests exist |
| `main.py` | CLI, simulation loop, logging, and orchestration | Thin CLI over reusable runners |
| `scripts/` | Sweeps, probes, analysis, plotting, verification | Categorize without discarding historical scripts |
| `tests/` | Unit, numerical, model, integration, and regression tests | Categorize after a stable baseline exists |
| `profiling/` | Profiling and pretests | Performance tooling |
| `tools/visualizer/` | Log conversion, rendering, and video overlays | Visualization tooling |
| `audit_output/` | Curated CSV baselines mixed with generated QP dumps | Separate small fixtures from run output |
| `results/` | Ignored logs, arrays, videos, and reference runs | Manifested runs, figures, videos, data, and archive |
| `docs/` | Conformance records, investigations, plans, archive | Decide collaborator-facing vs local documentation |
| `papers/` | Local untracked PDFs | Citation metadata plus external/LFS storage policy |

## Script families

- `run_*`, `chain_*`, and sweep shell scripts launch experiments.
- `aggregate_*`, `analyze_*`, `parse_*`, `summarize_*`, and `sweep_report.py`
  analyze outputs.
- `plot_*` and `tools/visualizer/` create figures and videos.
- `probe_*`, `check_*`, `attribute_*`, `dump_*`, and `emit_*` are diagnostics.
- `verify_*` scripts record mathematical or implementation conformance checks.
- `scripts/test_*.py` are executable diagnostics, not pytest tests.
- `scripts/reference/` compares this port with reference behavior.

No script should be archived or renamed until its inputs, expected outputs,
associated commit/config, and references from research notes have been recorded.

## Result classes

- **Canonical/reference:** reported results and trusted comparison baselines;
  immutable and externally backed up.
- **Regression fixture:** small, reviewed numerical artifacts intentionally kept
  with tests.
- **Working run:** reproducible from a recorded command/config but not published.
- **Diagnostic:** solver traces, QP dumps, render intermediates, and profiler data.
- **Incomplete:** partial, aborted, timed-out, or failed runs; retain until their
  scientific/provenance value is reviewed.
