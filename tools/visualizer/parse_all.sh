#!/usr/bin/env bash
# Parse every run.log under results/ into a JSONL file in audit_output/visualizer_jsonl/
# Skips files already up to date (output newer than input).
#
# Usage:  bash tools/visualizer/parse_all.sh
#         (from repo root)
set -e
cd "$(dirname "$0")/../.."  # cd to repo root

mkdir -p audit_output/visualizer_jsonl

count_done=0
count_skip=0
count_fail=0

for log in results/*/west/run.log; do
    [ -f "$log" ] || continue
    run_name=$(basename "$(dirname "$(dirname "$log")")")
    out="audit_output/visualizer_jsonl/${run_name}.jsonl"

    if [ -f "$out" ] && [ "$out" -nt "$log" ]; then
        count_skip=$((count_skip + 1))
        continue
    fi

    echo "Parsing $log -> $out"
    if python tools/visualizer/parse_log_to_jsonl.py "$log" "$out" > /dev/null 2>&1; then
        count_done=$((count_done + 1))
    else
        echo "  FAILED to parse $log"
        count_fail=$((count_fail + 1))
    fi
done

echo ""
echo "Done. parsed=$count_done skipped=$count_skip failed=$count_fail"
echo "JSONL files in audit_output/visualizer_jsonl/:"
ls -la audit_output/visualizer_jsonl/*.jsonl 2>/dev/null | tail -20
