#!/usr/bin/env bash
# Conformance pipeline — waits for baseline, then runs 3 sequential ablations.
# Each ablation = (YAML edit + commit) + sweep. Single background watcher.
set -uo pipefail
WAIT_PID="$1"   # baseline sweep PID to wait for
LOG=/root/push_anything_ADMM/conformance_pipeline.log
exec >> "$LOG" 2>&1

echo "[$(date)] waiting on baseline sweep PID=$WAIT_PID"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
echo "[$(date)] baseline sweep finished"
ls conformance_baseline/SUMMARY.txt && cat conformance_baseline/SUMMARY.txt || true

# Pre-flight check before each new sweep.
preflight() {
  local TMP_PCT
  TMP_PCT=$(df --output=pcent /tmp 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
  if [ "${TMP_PCT:-0}" -ge 85 ]; then
    echo "[$(date)] ABORT: /tmp at ${TMP_PCT}%"
    exit 1
  fi
}

# === COMMIT 1 === remove wrong-face commit gate via YAML override
echo "[$(date)] === COMMIT 1: remove wrong-face commit gate ==="
preflight
python3 - <<'PYEDIT1'
from pathlib import Path
p = Path('config/sampling_c3_kik.yaml')
txt = p.read_text()
# Add use_commit_face_gate at top level (right after the w_align/w_travel block).
# Insert after the "w_travel:" line.
import re
new = re.sub(
    r'(^w_travel:[^\n]*\n)',
    r'\1\n# Component 1 conformance — disable wrong-face commit gate (dev push_anything_dev @ 257e3ed has no equivalent).\nuse_commit_face_gate: false\n',
    txt, count=1, flags=re.M)
assert new != txt, "edit failed: w_travel anchor not found"
p.write_text(new)
print("commit1 YAML edit applied")
PYEDIT1
git add config/sampling_c3_kik.yaml
git commit -m "conformance(1/3): disable wrong-face commit gate (use_commit_face_gate=false)

Conform to DAIR push_anything_dev @ 257e3ed. The reference dispatcher
(sampling_based_c3_controller.cc) has NO face_align rejection at c3
entry, so this removes our pre-decide gate (wrapper.py:983-994) and
post-decide override (wrapper.py:1039-1057). Code stays intact;
toggled off via YAML override of params.py:649 default.

Ablation step 1 — measures the commit_face_gate's value to seed-0's
clean-win.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
bash scripts/_conformance_sweep.sh conformance_step1_no_face_gate
echo "[$(date)] commit 1 sweep done"
cat conformance_step1_no_face_gate/SUMMARY.txt || true

# === COMMIT 2 === goal-biased kFaceNormal -> uniform kRandomOnCircle
echo "[$(date)] === COMMIT 2: uniform sampler ==="
preflight
sed -i.bak 's|^  sampling_strategy: kFaceNormal|  sampling_strategy: kRandomOnCircle  # Conformance 2/3 — dev uniform sampler (was kFaceNormal w/ goal-biased jitter)|' config/sampling_c3_kik.yaml
rm -f config/sampling_c3_kik.yaml.bak
git add config/sampling_c3_kik.yaml
git commit -m "conformance(2/3): swap kFaceNormal -> kRandomOnCircle (uniform sampler)

Conform to DAIR push_anything_dev @ 257e3ed. The reference samples
uniformly (barycentric_bias=1 in anything/parameters/sampling_params.yaml;
strategies kRadiallySymmetric/kRandomOnCircle/kMeshNormal — none
goal-biased). Our kFaceNormal sampler (sampling.py:155-291) bakes
goal direction into the per-face tangent jitter (CENTERED_JITTER_FRACTION
on goal-aligned faces, full jitter elsewhere — sampling.py:259-266) —
the 'Push ANYTHING' generality violation.

kRandomOnCircle (sampling.py:96-126) is goal-agnostic — uniform random
angles on the contact ring, equivalent semantics to dev's kRandomOnCircle
(generate_samples.cc:225). Our kRadiallySymmetric was rejected because
it sets sample 0 at proxy_angle = arctan2(-g_hat[1], -g_hat[0]) at
sampling.py:141 — still goal-biased on the proxy.

Ablation step 2 — measures the goal-bias's value.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
bash scripts/_conformance_sweep.sh conformance_step2_uniform_sampler
echo "[$(date)] commit 2 sweep done"
cat conformance_step2_uniform_sampler/SUMMARY.txt || true

# === COMMIT 3 === admit-guard / IK reposition -> dev kPiecewiseLinear + dev pwl_waypoint_height
echo "[$(date)] === COMMIT 3: dev reposition (kPWL + pwl_waypoint_height=0.0774m) ==="
preflight
# YAML: traj_type kIK -> kPiecewiseLinear; pwl_waypoint_height 0.15 -> 0.0774
sed -i.bak \
  -e 's|^  traj_type: kIK|  traj_type: kPiecewiseLinear  # Conformance 3/3 — dev anything/parameters/reposition_params.yaml:11 traj_type:3=kPWL|' \
  -e 's|^  pwl_waypoint_height: 0.15.*|  pwl_waypoint_height: 0.0774  # Conformance 3/3 — dev anything/parameters/reposition_params.yaml:34 (was 0.15m, ours; bypasses our admit-guard since IK tracker is no longer in path)|' \
  config/sampling_c3_kik.yaml
rm -f config/sampling_c3_kik.yaml.bak
# CODE: PiecewiseLinearTracker.compute_torque needs admit_active kwarg (no-op)
# so wrapper.py:1683 call site (which always passes admit_active=) works for
# both trackers. Otherwise switching traj_type to kPiecewiseLinear errors at
# the call site.
python3 - <<'PYEDIT3'
from pathlib import Path
p = Path('control/sampling_c3/reposition.py')
txt = p.read_text()
old = 'def compute_torque(self,\n                       current_q:  np.ndarray,\n                       current_v:  np.ndarray,\n                       plant_ctx,\n                       p_target:   np.ndarray,\n                       dt_osc:     float) -> tuple[np.ndarray, dict]:'
new = 'def compute_torque(self,\n                       current_q:  np.ndarray,\n                       current_v:  np.ndarray,\n                       plant_ctx,\n                       p_target:   np.ndarray,\n                       dt_osc:     float,\n                       admit_active: bool = False) -> tuple[np.ndarray, dict]:'
assert old in txt, "anchor not found in reposition.py compute_torque"
p.write_text(txt.replace(old, new, 1))
print("PWL.compute_torque admit_active kwarg added (no-op)")
PYEDIT3
git add config/sampling_c3_kik.yaml control/sampling_c3/reposition.py
git commit -m "conformance(3/3): kIK -> kPiecewiseLinear + dev pwl_waypoint_height 0.0774m

Conform to DAIR push_anything_dev @ 257e3ed reposition: dev uses
kPiecewiseLinear (3-leg lift/traverse/descend) with pwl_waypoint_height
0.0774m (anything/parameters/reposition_params.yaml:34). We switch
from traj_type=kIK (our IK-based path with the admit-guard EE_z gate at
reposition_ik.py:104 _should_cap_z_safe) to kPiecewiseLinear with the
dev waypoint value (was our 0.15m).

Bypasses our admit-guard implicitly: with kPiecewiseLinear the wrapper
dispatches to PiecewiseLinearTracker (wrapper.py:180), not
RepositionIKTracker — so the admit_active latch + EE_Z_GATE never run.

Code change: add admit_active kwarg to PiecewiseLinearTracker.compute_torque
(no-op) so the wrapper.py:1683 call site keeps working uniformly.

Ablation step 3 — measures the (admit-guard + 0.15m waypoint) combination
vs (no admit-guard + 0.0774m waypoint). Reference's ee_z_close +
c3_min_clearance (sampling_based_c3_controller.cc:1290-1293) is not
ported in this commit — it's not strictly needed once kPWL is the
tracker.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
bash scripts/_conformance_sweep.sh conformance_step3_dev_reposition
echo "[$(date)] commit 3 sweep done"
cat conformance_step3_dev_reposition/SUMMARY.txt || true

# === FINAL ABLATION TABLE ===
echo ""
echo "============================================================"
echo "       CONFORMANCE ABLATION TABLE (seeds 0/2/4)"
echo "============================================================"
for D in conformance_baseline conformance_step1_no_face_gate conformance_step2_uniform_sampler conformance_step3_dev_reposition; do
  echo ""
  echo "=== $D ==="
  if [ -f "$D/SUMMARY.txt" ]; then
    cat "$D/SUMMARY.txt"
  else
    echo "(no SUMMARY.txt)"
  fi
done
echo "[$(date)] PIPELINE FINISHED"
