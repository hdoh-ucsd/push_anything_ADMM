#!/usr/bin/env bash
set -euo pipefail

# Fail-closed V2 gate. This wrapper performs no simulation by itself; it must
# be invoked before any future serial experiment launcher.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKTREE="$ROOT_DIR/external/oim_c++_anything/.claude/worktrees/oim-scene-sync-metrics"
CONTROLLER="$WORKTREE/bazel-bin/examples/sampling_c3/franka_sampling_c3_controller"
INNER_CONFIG="$ROOT_DIR/results/c3plus_final_clean_obstacle_ranking/config/final_clean_obstacle_study.json"
FREEZE="$ROOT_DIR/results/c3plus_final_clean_obstacle_ranking/provenance/validated_frozen_v2.json"

test -f "$FREEZE" || { echo "FREEZE_VALIDATION=FAIL missing V2 receipt" >&2; exit 1; }
test -x "$CONTROLLER" || { echo "FREEZE_VALIDATION=FAIL missing controller" >&2; exit 1; }
test -f "$INNER_CONFIG" || { echo "FREEZE_VALIDATION=FAIL missing inner config" >&2; exit 1; }

expected_binary_sha="$(sed -n 's/.*"controller_binary_sha256": "\([0-9a-f]*\)".*/\1/p' "$FREEZE")"
expected_inner_sha="$(sed -n 's/.*"inner_config_sha256": "\([0-9a-f]*\)".*/\1/p' "$FREEZE")"
actual_binary_sha="$(sha256sum "$CONTROLLER" | awk '{print $1}')"
actual_inner_sha="$(sha256sum "$INNER_CONFIG" | awk '{print $1}')"

test -n "$expected_binary_sha" || { echo "FREEZE_VALIDATION=FAIL malformed V2 receipt" >&2; exit 1; }
test -n "$expected_inner_sha" || { echo "FREEZE_VALIDATION=FAIL malformed inner receipt" >&2; exit 1; }
test "$actual_binary_sha" = "$expected_binary_sha" || {
  echo "FREEZE_VALIDATION=FAIL controller SHA mismatch" >&2; exit 1;
}
test "$actual_inner_sha" = "$expected_inner_sha" || {
  echo "FREEZE_VALIDATION=FAIL inner configuration SHA mismatch" >&2; exit 1;
}

if ! (cd "$ROOT_DIR" && python3 -m pytest -q tests/test_c3plus_ranking_only_boundary.py); then
  echo "FREEZE_VALIDATION=FAIL ranking-equivalence-test" >&2
  exit 1
fi

echo "FREEZE_VALIDATION=PASS"
echo "VALIDATED_FROZEN_VERSION=V2"
echo "CONTROLLER_BINARY_SHA256=$actual_binary_sha"
echo "INNER_CONFIG_SHA256=$actual_inner_sha"

# Optional command after the gate, e.g. a future approved serial dispatcher.
if (($#)); then
  exec "$@"
fi
