# Reproducibility Guide

## Audited environment

The first cleanup baseline was collected on 2026-08-21 at commit `018dffb`
with a dirty working tree containing pre-existing untracked audit dumps and
paper PDFs. The active environment contained:

- Python 3.10.20
- Drake 1.51.1
- NumPy 2.2.6
- SciPy 1.15.3
- OSQP 1.1.1
- PyYAML 6.0.3
- Matplotlib 3.10.8
- pytest 9.0.3

`environment.yml` records these observed versions. It is a starting point for
reproduction, not evidence that other versions are incompatible.

## Installation

```bash
conda env create -f environment.yml
conda activate push_anything_admm
```

Drake wheels and their supported platforms can impose additional system
requirements. Video export also requires `ffmpeg`.

## Tests

Run the intended suite from the repository root:

```bash
python -m pytest tests
```

The `pyproject.toml` test path prevents pytest from importing diagnostic scripts
whose filenames begin with `test_`.

### Initial cleanup baseline

The no-cache command used was:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -p no:cacheprovider -q tests
```

Result: **292 passed, 37 failed, 15 errors** in approximately 2.5 seconds.
This is a preservation baseline, not a claim that the suite is healthy.

The 15 errors came from Drake tests attempting to start Meshcat in an
environment where no websocket port was available. The failures included known
test/config drift around progress timing, sampling strategies, mode gates, OSC
defaults, planner workspace bounds, and LCS/OSC tests affected by Meshcat.
Do not update expected values or controller parameters solely to reduce this
count. Diagnose and approve each behavior change separately.

Running pytest without an explicit test path previously collected
`scripts/test_layer1_approach.py`, which starts a simulation at import time and
failed during collection. `pyproject.toml` now restricts collection to `tests/`.

## Canonical run record

For a result intended for comparison, publication, or regression, record:

1. Git commit and whether the tree was dirty.
2. Complete command line.
3. Configuration file and SHA-256 checksum.
4. All command-line overrides.
5. Random seed and any nondeterministic components.
6. Python, Drake, NumPy, OSQP, and platform versions.
7. Start/end time and success criterion.
8. Output directory and a checksum manifest for retained artifacts.

Do not modify a canonical result in place. A rerun should receive a new run ID
and link back to the run it supersedes.

## Results storage

`results/` is intentionally ignored because it contains large generated output.
Ignored does not mean disposable. Before cleanup, copy canonical and reported
results to versioned institutional or object storage and create a small manifest
in the repository. Consider DVC or git-annex for datasets and run artifacts, and
Git LFS only for stable binary assets that genuinely need repository-versioned
distribution.
