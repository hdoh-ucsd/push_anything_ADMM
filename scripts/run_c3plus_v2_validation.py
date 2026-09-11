#!/usr/bin/env python3
"""Run one of the three predeclared timer-fix validation identities with V2."""

from __future__ import annotations

import runpy
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ["SAMPLING_C3_VALIDATION_BIN_DIR"] = str(
    ROOT / "external/oim_c++_anything/.claude/worktrees/oim-scene-sync-metrics/bazel-bin/examples/sampling_c3")
impl = ROOT / "results/c3plus_final_clean_obstacle_ranking/allfinite_qv_root_cause/run_targeted_diagnostic.py"
namespace = runpy.run_path(str(impl), run_name="v2_validation_impl")
namespace["DIAG_BIN"] = namespace["PROD_BIN"]
namespace["DIAG"] = namespace["PROD"]
raise SystemExit(namespace["main"]())
