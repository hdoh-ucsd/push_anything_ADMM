#!/usr/bin/env python3
"""Report Table-I-style Jack tight-goal timing from continuous run logs.

Each input log is one continuous experiment. Goal durations are differences
between consecutive ``goal #N REACHED at t=...`` timestamps; the first goal is
measured from simulation time zero. Population standard deviation (ddof=0) is
reported because the collected goals are treated as the complete experiment
set. The paper does not state its ddof convention, so the sample value is also
printed for auditability.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import statistics


REACHED = re.compile(r"\[GOAL-GEN\] goal #(\d+) REACHED at t=([0-9.]+)s")


def durations(path: Path) -> list[float]:
    reached = [(int(n), float(t)) for n, t in REACHED.findall(
        path.read_text(errors="replace"))]
    if not reached:
        return []
    numbers = [n for n, _ in reached]
    if numbers != list(range(1, len(numbers) + 1)):
        raise ValueError(f"{path}: non-consecutive goal numbers {numbers}")
    times = [t for _, t in reached]
    return [times[0], *(b - a for a, b in zip(times, times[1:]))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()

    all_durations: list[float] = []
    for path in args.logs:
        values = durations(path)
        all_durations.extend(values)
        print(f"{path}: n={len(values)} durations_s={values}")

    if not all_durations:
        print("No completed tight goals found.")
        return 1

    mean = statistics.fmean(all_durations)
    sigma_pop = statistics.pstdev(all_durations)
    sigma_sample = (statistics.stdev(all_durations)
                    if len(all_durations) > 1 else float("nan"))
    print()
    print("| Metric | Tight goal time |")
    print("|---|---:|")
    print(f"| Successful goals | {len(all_durations)} |")
    print(f"| Mean ± σ (population, ddof=0) | "
          f"{mean:.2f} ± {sigma_pop:.2f} s |")
    print(f"| Sample standard deviation (ddof=1) | {sigma_sample:.2f} s |")
    print(f"| Range | [{min(all_durations):.2f}, "
          f"{max(all_durations):.2f}] s |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

