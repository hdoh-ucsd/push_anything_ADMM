#!/usr/bin/env bash
# Chained conformance commits (1/2/3). Run AFTER baseline seed4 is complete.
# Each: YAML edit + git commit + seeds 0/2/4 sweep + SUMMARY.txt.
# Sequential — never parallel across commits (state contamination risk).
set -uo pipefail

LOG=/root/push_anything_ADMM/conformance_chain.log
exec >> "$LOG" 2>&1

preflight() {
  local TMP_PCT
  TMP_PCT=$(df --output=pcent /tmp 2>/dev/null | tail -1 | tr -d ' %' || echo 0)
  if [ "${TMP_PCT:-0}" -ge 85 ]; then
    echo "[$(date)] ABORT: /tmp at ${TMP_PCT}% — DO NOT rm -rf /tmp/claude-0/*"
    exit 1
  fi
}

# ============================================================
# COMMIT 1 — disable wrong-face commit gate
# ============================================================
echo "[$(date)] === COMMIT 1: disable wrong-face commit gate ==="
preflight
python3 - <<'PYEDIT1'
from pathlib import Path
import re
p = Path('config/sampling_c3_kik.yaml')
txt = p.read_text()
new = re.sub(
    r'(^w_travel:[^\n]*\n)',
    r'\1\n# Component 1 conformance (push_anything_dev @ 257e3ed has no commit-face gate).\nuse_commit_face_gate: false\n',
    txt, count=1, flags=re.M)
assert new != txt, "FAIL: w_travel anchor not found"
p.write_text(new)
print("commit1 YAML edit applied")
PYEDIT1
git add config/sampling_c3_kik.yaml
git commit -m "conformance(1/3): disable wrong-face commit gate (use_commit_face_gate=false)

Conform to DAIR push_anything_dev @ 257e3ed (anything/parameters/).
The reference dispatcher (sampling_based_c3_controller.cc) has NO
face_align rejection at c3 entry. Toggle off via YAML override of
the params.py:649 default; wrapper.py:460 _evaluate_commit_face_gate
early-returns when use_commit_face_gate=False (wrapper.py:473).

Ablation step 1/3 — measures the commit_face_gate's value to seed-0's
clean-win.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
bash scripts/_conformance_sweep.sh conformance_step1_no_face_gate
echo "[$(date)] commit 1 sweep done"
cat conformance_step1_no_face_gate/SUMMARY.txt || true

# ============================================================
# COMMIT 2 — kFaceNormal -> kRandomOnCircle (uniform)
# ============================================================
echo "[$(date)] === COMMIT 2: kFaceNormal -> kRandomOnCircle (uniform sampler) ==="
preflight
sed -i.bak 's|^  sampling_strategy: kFaceNormal|  sampling_strategy: kRandomOnCircle  # Conformance 2/3 — dev uniform sampler (was kFaceNormal w/ goal-biased jitter)|' config/sampling_c3_kik.yaml
rm -f config/sampling_c3_kik.yaml.bak
git add config/sampling_c3_kik.yaml
git commit -m "conformance(2/3): swap kFaceNormal -> kRandomOnCircle (uniform sampler)

Conform to DAIR push_anything_dev @ 257e3ed. dev's
anything/parameters/sampling_params.yaml uses sampling_strategy: 7
(kMeshNormaMultiObject) with barycentric_bias: 1 — uniform face draw
with no goal bias. Our kFaceNormal sampler (sampling.py:155-291) bakes
goal direction into per-face tangent jitter (CENTERED_JITTER_FRACTION
on goal-aligned faces, full jitter elsewhere — sampling.py:259+) and a
face_bias_strength downhill weight on the face draw (sampling.py:226-234).

kRandomOnCircle (sampling.py:96-126) is the task-agnostic equivalent in
our codebase: independent uniform angles on the contact ring with no
g_hat dependency (sampling.py:117 'del g_hat'). kRadiallySymmetric was
rejected because sample 0 sits at proxy_angle = arctan2(-g_hat[1],
-g_hat[0]) at sampling.py:141 — still goal-biased on the proxy.

Ablation step 2/3 — measures the goal-bias's value (sample-cost
discrimination).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
bash scripts/_conformance_sweep.sh conformance_step2_uniform_sampler
echo "[$(date)] commit 2 sweep done"
cat conformance_step2_uniform_sampler/SUMMARY.txt || true

# ============================================================
# COMMIT 3 — kIK + 0.15m waypoint -> kPiecewiseLinear + dev 0.0774m
# ============================================================
echo "[$(date)] === COMMIT 3: dev reposition (kPWL + pwl_waypoint_height=0.0774m) ==="
preflight
sed -i.bak \
  -e 's|^  traj_type: kIK|  traj_type: kPiecewiseLinear  # Conformance 3/3 — dev anything/parameters/reposition_params.yaml:11 traj_type:3=kPWL|' \
  -e 's|^  pwl_waypoint_height: 0.15.*|  pwl_waypoint_height: 0.07738005  # Conformance 3/3 — dev anything/parameters/reposition_params.yaml:34 (was 0.15m ours; bypasses our admit-guard since IK tracker is no longer in path)|' \
  config/sampling_c3_kik.yaml
rm -f config/sampling_c3_kik.yaml.bak
# Ensure PiecewiseLinearTracker.compute_torque accepts admit_active kwarg (no-op),
# so wrapper's call site stays uniform across trackers.
python3 - <<'PYEDIT3'
from pathlib import Path
p = Path('control/sampling_c3/reposition.py')
txt = p.read_text()
old = ('def compute_torque(self,\n'
       '                       current_q:  np.ndarray,\n'
       '                       current_v:  np.ndarray,\n'
       '                       plant_ctx,\n'
       '                       p_target:   np.ndarray,\n'
       '                       dt_osc:     float) -> tuple[np.ndarray, dict]:')
new = ('def compute_torque(self,\n'
       '                       current_q:  np.ndarray,\n'
       '                       current_v:  np.ndarray,\n'
       '                       plant_ctx,\n'
       '                       p_target:   np.ndarray,\n'
       '                       dt_osc:     float,\n'
       '                       admit_active: bool = False) -> tuple[np.ndarray, dict]:')
if old not in txt:
    # Already patched or different signature; skip rather than fail.
    print("PWL.compute_torque anchor not found — skipping kwarg patch")
else:
    p.write_text(txt.replace(old, new, 1))
    print("PWL.compute_torque admit_active kwarg added (no-op)")
PYEDIT3
git add config/sampling_c3_kik.yaml control/sampling_c3/reposition.py
git commit -m "conformance(3/3): kIK -> kPiecewiseLinear + pwl_waypoint_height 0.0774m

Conform to DAIR push_anything_dev @ 257e3ed reposition: dev uses
kPiecewiseLinear (3-leg lift/traverse/descend) with pwl_waypoint_height
0.07738005m (anything/parameters/reposition_params.yaml:11,34). We
switch traj_type from kIK to kPiecewiseLinear with the dev waypoint
value.

Bypasses our admit-guard implicitly: kPiecewiseLinear dispatches to
PiecewiseLinearTracker (wrapper.py instantiates it via traj_type
config), not RepositionIKTracker — so admit_active latch + EE_Z_GATE
in reposition_ik.py never run.

Code change: add admit_active kwarg (no-op default) to
PiecewiseLinearTracker.compute_torque so the wrapper's call site keeps
working uniformly across trackers.

Ablation step 3/3 — measures (admit-guard + 0.15m waypoint) vs
(no admit-guard + 0.07738m waypoint). Reference's ee_z_close +
c3_min_clearance check (sampling_based_c3_controller.cc:1290-1293)
is not ported in this commit — not strictly needed once kPWL is the
tracker.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
bash scripts/_conformance_sweep.sh conformance_step3_dev_reposition
echo "[$(date)] commit 3 sweep done"
cat conformance_step3_dev_reposition/SUMMARY.txt || true

# ============================================================
# FINAL ABLATION TABLE
# ============================================================
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
echo "[$(date)] CHAIN FINISHED"
