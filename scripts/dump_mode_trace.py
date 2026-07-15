#!/usr/bin/env python3
"""Emit a per-step mode trace + RLE summary from a run log.

Output:
  - Total step count, c3 step count + percentage, free step count
  - Run-length encoded mode sequence (e.g. "free×12, c3×3, free×8, ...")
  - First/last switch_reason transitions

Useful for explaining why β changes the mode distribution on a given
task — the RLE makes it obvious whether the dispatcher was bouncing
between modes (phase-oscillation) or genuinely committing to long c3
sessions.
"""
from __future__ import annotations
import re
import sys
from collections import Counter

STEP_RE = re.compile(
    r"^\[STEP\] step=(?P<step>\d+) mode=(?P<mode>\S+) t=(?P<t>[\d.+\-eE]+)s"
    r".* switch=(?P<sw>\S+)"
)


def main():
    if len(sys.argv) < 2:
        print("usage: dump_mode_trace.py <log>", file=sys.stderr)
        sys.exit(2)
    path = sys.argv[1]
    modes: list[str] = []
    switches: list[tuple[int, str, str]] = []  # (step, mode, switch_reason)
    with open(path) as f:
        for line in f:
            m = STEP_RE.match(line)
            if not m:
                continue
            modes.append(m.group("mode"))
            switches.append((int(m.group("step")), m.group("mode"),
                             m.group("sw")))

    n = len(modes)
    if n == 0:
        print(f"{path}: no [STEP] rows")
        return
    counts = Counter(modes)
    print(f"=== {path} ===")
    print(f"total_steps={n}  "
          + "  ".join(f"{k}={v} ({100*v/n:.1f}%)" for k, v in counts.items()))

    # Run-length encoding.
    rle: list[tuple[str, int]] = []
    cur = modes[0]
    run = 0
    for m in modes:
        if m == cur:
            run += 1
        else:
            rle.append((cur, run))
            cur, run = m, 1
    rle.append((cur, run))

    print(f"\nMode runs ({len(rle)} total):")
    parts: list[str] = []
    for m, r in rle:
        parts.append(f"{m}×{r}")
    line = ", ".join(parts)
    if len(line) > 220:
        # truncate but show first 8 + last 8 to keep it readable
        head = ", ".join(parts[:8])
        tail = ", ".join(parts[-8:])
        print(f"  {head}, ... ({len(rle) - 16} more) ..., {tail}")
    else:
        print(f"  {line}")

    # Switch reasons — only show transitions (not steady-state).
    print(f"\nSwitch reasons at mode-change boundaries:")
    prev_mode = None
    for step, mode, sw in switches:
        if mode != prev_mode:
            print(f"  step={step:>4d}  {prev_mode}→{mode:5s}  reason={sw}")
            prev_mode = mode


if __name__ == "__main__":
    main()
