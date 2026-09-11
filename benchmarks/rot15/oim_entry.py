#!/usr/bin/env python3
"""Run one OIM example while redirecting only its pose-definition directory.

This is an adapter, not a controller change: OIM still parses its native
``--start``/``--goal`` keys and receives a normal ``[x, y, yaw]`` pose.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oim-root", type=Path, required=True)
    parser.add_argument("--poses-dir", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = args.oim_root.resolve()
    script = args.script.resolve()
    if not script.is_file() or root not in script.parents:
        raise FileNotFoundError(f"OIM example is not under --oim-root: {script}")
    sys.path.insert(0, str(root))
    from oim.utils import poses  # imported from the selected OIM checkout

    poses.POSES_DIR = str(args.poses_dir.resolve())
    forwarded = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    sys.argv = [str(script), *forwarded]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
