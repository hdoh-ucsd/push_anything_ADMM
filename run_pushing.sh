#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Baseline pushing run — full diagnostics enabled.
#
# Usage:
#   ./run_pushing.sh                       # auto-named: results/pushing_<ts>.{mp4,html,txt}
#   ./run_pushing.sh baseline_v3           # custom-named: results/baseline_v3.{mp4,html,txt}
#
# Outputs (all under results/):
#   <name>.mp4    Top-down rendered video
#   <name>.html   Meshcat HTML replay
#   <name>.txt    Full stdout log (includes [MATH.*] diagnostics)
#
# Flags:
#   --save-video / --video-path  ON by default in main.py; passed explicitly
#                                here to make the baseline self-documenting.
#   --math-diag                  Verbose [MATH.*] solver diagnostics, zero
#                                overhead when off. Auto-captured to .txt log.
# -----------------------------------------------------------------------------
set -euo pipefail
mkdir -p results

NAME="${1:-}"

if [[ -z "$NAME" ]]; then
    # No name given — let main.py auto-name everything as pushing_<timestamp>.*
    python main.py pushing \
        --save-video \
        --video-path \
        --math-diag
else
    # Custom name given. main.py's _Tee logger hardcodes the .txt name as
    # pushing_<timestamp>.txt, so we capture the start time, run, then rename
    # the log to match our custom name afterwards.
    START=$(date +%s)

    python main.py pushing \
        --save-video "results/${NAME}.mp4" \
        --video-path "results/${NAME}.html" \
        --math-diag

    # Find the .txt log created during this run and rename it to <NAME>.txt
    LATEST_TXT=$(find results -maxdepth 1 -name 'pushing_*.txt' \
                  -newermt "@${START}" 2>/dev/null | head -n1 || true)
    if [[ -n "$LATEST_TXT" ]]; then
        mv "$LATEST_TXT" "results/${NAME}.txt"
        echo "→ renamed log: ${LATEST_TXT} → results/${NAME}.txt"
    else
        echo "warn: could not find pushing_*.txt log to rename" >&2
    fi
fi