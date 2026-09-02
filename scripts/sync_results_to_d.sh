#!/usr/bin/env bash
# Copy results/ to D:\projects\ERL\push_anything_ADMM\results\ for Windows-side
# access. Existing destination files are versioned before replacement.
#
# Usage:
#   bash scripts/sync_results_to_d.sh              # incremental sync
#   bash scripts/sync_results_to_d.sh --dry-run    # preview only
# No deletion mode is provided: D-drive-only results may carry provenance.
#
# Exits 0 if /d isn't mounted (so chaining `... && sync` doesn't break runs
# on machines without the D drive).

set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)/results"
DST="/d/projects/ERL/push_anything_ADMM/results"
DRY_RUN=()

case "${1:-}" in
    "") ;;
    --dry-run) DRY_RUN=(--dry-run) ;;
    *) echo "usage: $0 [--dry-run]" >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo "usage: $0 [--dry-run]" >&2; exit 2; }

if [[ ! -d /d ]]; then
    echo "[sync] /d not mounted — skipping (no error)."
    exit 0
fi

if ! findmnt -T /d -n -o SOURCE,FSTYPE >/dev/null 2>&1; then
    echo "[sync] /d exists but is not a mounted filesystem — refusing to copy." >&2
    exit 2
fi

if [[ ! -d "$SRC" ]]; then
    echo "[sync] source $SRC missing — nothing to mirror." >&2
    exit 0
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="$DST/.sync_backups/$STAMP"
mkdir -p "$DST"
rsync -a --backup --backup-dir="$BACKUP_DIR" --info=stats2 \
    "${DRY_RUN[@]}" "$SRC/" "$DST/"
if [[ ${#DRY_RUN[@]} -gt 0 ]]; then
    echo "[sync] dry-run complete: $SRC -> $DST"
else
    echo "[sync] copied $SRC -> $DST (replaced files preserved under $BACKUP_DIR)"
fi
