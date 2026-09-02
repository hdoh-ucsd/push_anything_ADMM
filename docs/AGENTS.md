# Project Instructions for Coding Agents

This repository is an active robotics research project reproducing and extending
Push Anything, C3/C3+, contact-implicit MPC, and sampling-based manipulation.
Scientific traceability takes priority over cosmetic cleanup.

## Safety and approval

- Never delete files, experiment outputs, or research notes without explicit
  approval.
- Inspect `git status` and `git worktree list` before changing shared paths.
- Preserve unrelated and untracked work. Do not clean the working tree on the
  user's behalf.
- Prefer reversible additions, compatibility wrappers, and documented moves.
- Treat reported, published, reference, and baseline results as immutable.
  Never edit numerical data or logs in place.

## Numerical behavior

- Do not silently change solver tolerances, ADMM parameters, MPC horizons,
  dynamics or contact parameters, random seeds, cost weights, benchmark
  definitions, success criteria, or default controller modes.
- Preserve floating-point ordering and solver selection during structural
  refactors unless numerical equivalence is demonstrated and approved.
- Retain legacy paths until their callers and scientific provenance are known.
- A failing test is not permission to retune a parameter or update an expected
  value. First determine whether the implementation, test, configuration, or
  environment has drifted.

## Architecture

- Keep reusable models, solvers, controllers, and simulation infrastructure
  separate from experiment orchestration and one-off diagnostics.
- Put canonical experiment definitions and commands in dedicated experiment or
  benchmark documentation. Do not hide experiment parameters in shell history.
- Document changed public interfaces, configuration keys, file locations, and
  compatibility behavior.
- Distinguish reproduction work from original research contributions in code,
  documentation, results, and commit messages.

## Testing and reproducibility

- Before a refactor, record the commit, dirty-tree state, environment, and
  relevant baseline test results.
- Run focused tests before and after a change. Run integration and numerical
  regression tests when touching C3/C3+, LCS construction, ADMM,
  complementarity projection, MPC, OSC, or simulation.
- Use `python -m pytest tests` for the repository suite. Root-level pytest
  collection must remain restricted to `tests/`; scripts named `test_*.py` are
  diagnostics, not pytest modules.
- Do not weaken tolerances or remove assertions merely to make a refactor pass.
- Record each canonical experiment's commit SHA, dirty-tree status, command,
  config path and checksum, seed, dependency versions, and output location.
- Keep ignored working results backed up outside Git. Curated regression
  fixtures must be small, documented, and intentionally committed.

## Data and artifacts

- Do not overwrite or rename results until a manifest records their provenance.
- Do not assume that `partial`, `failed`, `old`, `backup`, or empty outputs are
  disposable; they may document an important negative result.
- Keep generated media, logs, caches, and profiler output out of normal commits.
- Store large stable assets through an explicitly documented Git LFS, DVC,
  git-annex, or institutional-storage policy rather than ad hoc commits.
