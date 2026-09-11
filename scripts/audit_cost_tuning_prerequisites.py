#!/usr/bin/env python3
"""Read-only prerequisite audit for the small C3+ cost-tuning study.

Exit 0 means every launch gate checked here is ready. Exit 2 means the study
must remain stopped. The script never launches a controller or edits a config.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
C3_ROOT = ROOT / "external/oim_c++_anything/.claude/worktrees/oim-scene-sync-metrics"
SYNC = ROOT / "results/benchmark_sync_rot15"


def read_csv(name: str) -> list[dict[str, str]]:
    with (SYNC / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    canonical = read_csv("canonical_cases.csv")
    parity = read_csv("cross_method_parity.csv")
    geometry = read_csv("goal_geometry_preflight.csv")
    per_scene = Counter(row["scene"] for row in canonical)

    benchmark_ready = (
        len(canonical) == 90
        and len(per_scene) == 6
        and set(per_scene.values()) == {15}
        and len(parity) == 90
        and all(row["parity_status"] == "PASS" for row in parity)
        and len(geometry) == 90
        and all(row["feasibility_status"] == "FEASIBLE" for row in geometry)
    )

    sampler = C3_ROOT / "examples/sampling_c3/generate_samples.cc"
    sampler_text = sampler.read_text()
    rng_hits = [
        {"line": line_no, "text": line.strip()}
        for line_no, line in enumerate(sampler_text.splitlines(), start=1)
        if re.search(r"random_device|mt19937", line)
    ]
    controlled_seed = not any("random_device" in hit["text"] for hit in rng_hits)

    logger_report = C3_ROOT / "results/forensics_logger_v2/FORENSICS_LOGGER_V2_REPORT.md"
    logger_text = logger_report.read_text() if logger_report.exists() else ""
    logger_runtime_gate = (
        bool(logger_text)
        and "one remaining validation item" not in logger_text
        and "initial-C3 transaction bootstrap fix was compiled after" not in logger_text
    )

    result = {
        "benchmark_rot15": {
            "ready": benchmark_ready,
            "canonical_rows": len(canonical),
            "scene_counts": dict(sorted(per_scene.items())),
            "parity_pass_rows": sum(row["parity_status"] == "PASS" for row in parity),
            "feasible_goal_rows": sum(row["feasibility_status"] == "FEASIBLE" for row in geometry),
        },
        "forensics_logger_v2": {
            "source_and_short_smoke_present": bool(logger_text),
            "runtime_acceptance_complete": logger_runtime_gate,
        },
        "c3plus_rng": {
            "controlled_seed_available": controlled_seed,
            "authoritative_sampler": str(sampler.relative_to(C3_ROOT)),
            "rng_source_hits": rng_hits,
        },
        "launch_authorized": benchmark_ready and logger_runtime_gate and controlled_seed,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["launch_authorized"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
