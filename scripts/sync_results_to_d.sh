#!/usr/bin/env bash
# Mirror results/ to D:\projects\push_anything_ADMM\results\ for Windows-side
# access. Safe to chain after sims: only copies deltas (rsync -a).
#
# Usage:
#   bash scripts/sync_results_to_d.sh              # incremental sync
#   bash scripts/sync_results_to_d.sh --dry-run    # preview only
#   bash scripts/sync_results_to_d.sh --delete     # also remove files on D
#                                                    that no longer exist locally
#
# Exits 0 if /d isn't mounted (so chaining `... && sync` doesn't break runs
# on machines without the D drive).

set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)/results"
DST="/d/projects/ERL/push_anything_ADMM/results"

if [[ ! -d /d ]]; then
    echo "[sync] /d not mounted — skipping (no error)."
    exit 0
fi

if [[ ! -d "$SRC" ]]; then
    echo "[sync] source $SRC missing — nothing to mirror." >&2
    exit 0
fi

mkdir -p "$DST"
rsync -a --info=stats2 "$@" "$SRC/" "$DST/"
echo "[sync] mirrored $SRC -> $DST"
