# Passive C3+ phantom-contact audit

These investigation tools observe the existing controller; they do not change
controller parameters, contact models, costs, candidate ordering, or solver
returns. They are standalone diagnostics, not pytest modules.

Run from the repository root using the existing Drake environment:

```bash
python tools/diagnostics/audit_phantom_contact.py \
  --output /tmp/UNIQUE_PHANTOM_AUDIT_DIRECTORY --max-time 60 \
  > /tmp/UNIQUE_PHANTOM_AUDIT_TERMINAL.txt 2>&1
python tools/diagnostics/analyze_phantom_contact.py \
  /tmp/UNIQUE_PHANTOM_AUDIT_DIRECTORY
```

Both output directories must be new. The observer invokes the unmodified
`main.py pushing --sampling-c3 config/sampling_c3_kik.yaml --solver c3plus
--seed 42` with the requested observation duration. `main.py` also creates its
normal uniquely named log in `results/`; a copy is saved outside Git in the
chosen output directory. Drake requires permission to open a local Meshcat port.

The output includes source/configuration SHA256 hashes, Git status and commit,
dependency versions, loaded configuration, the command, and a CSV row after
every planner or OSC update. Mode changes are counted only on planner rows.
`row` is zero-based; the corresponding CSV file line is `row + 2`.

Contact is measured using the actual simulation plant's EE/manipuland geometry
IDs, signed-distance queries, and point/hydroelastic contact results. A measured
force norm above `1e-8 N` defines `physical_contact`; signed gap is recorded
separately. Planned contact forces and mode labels are not used to infer contact.
The last integration endpoint can fall after the last recorded control update;
analysis treats an unfinished final C3 episode as censored.

`progress_counter` and `progress_metric` capture the actual decision-time
tracker state, including a no-progress exit immediately before its reset.
`progress_counter_after_control` shows the reset. OSC rows carry the latest
planner snapshot. The configuration-cost window front is valid for the selected
box configuration, whose retained history equals the active window length.

In this observer version, `candidate_object_relative_angle` and
`candidate_object_relative_distance` describe the preceding/pursued **reposition
target**, including during C3. `selected_candidate_*` describes the executed
candidate (current EE in C3). Repeat analysis instead reconstructs target-frame
coordinates at each actual target selection, freezing that anchor through later
object movement. Failed-buffer rejection counts are cumulative predicate
results. New replays also write `candidate_rejections.jsonl`, with a unique
predicate-probe ID, candidate coordinates, and the failed-contact targets
present when rejection occurred. These IDs identify probes before final
candidate indexing, not distinct geometric approaches.

The baseline uses measured solve wall time in trajectory execution. Geometry
queries and CSV I/O occur after the controller's timing scope; scalar diagnostic
copies occur inside it. Seed 42 fixes random initialization, but this baseline
cannot promise bitwise deterministic trajectories across wall-time variations.
The audit must not disable or replace that timing behavior to obtain determinism.

The analysis reports 5/10/20 mm object-frame target-equivalence sensitivity and
distinguishes a repeated failed approach from a repeated approach that fails
again. Existing baseline failures and deleted fixtures are recorded separately;
these tools do not repair or restore them.

For the contact-acquisition robustness extension, the observer additionally
records `contact_acquisition_state`, the controller's force/contact measurement,
entry/acquisition/failure times, attempted world and object-frame targets,
`normal_progress_counter`, and `failed_target_memory_size`. The latter counts
only verified failed-contact targets, separately from legacy failed-buffer
entries. `contact_acquisition_deadline` is the scheduled deadline;
`acquisition_timeout_time` is the actual failure event. The unchanged raw
`physical_contact` threshold remains `1e-8 N`; `controller_physical_contact`
uses the separately logged configured threshold. All additional fields are
copied after the controller returns. Rejection wrappers copy values before
returning the original result; their JSON encoding and I/O happen afterward.

Compare a frozen baseline and patched replay without overwriting either:

```bash
python tools/diagnostics/compare_contact_fix.py \
  /tmp/FROZEN_BASELINE /tmp/PATCHED_REPLAY --output /tmp/NEW_COMPARISON \
  --candidate-order-before "FAIL: baseline sensitivity observed" \
  --candidate-order-after "PASS: see candidate-order test evidence"
```

This writes the comparison table plus separate summaries, episode tables,
transitions, and target attempts. Supply candidate-order labels only from
actual solver test evidence; trajectory analysis does not establish solver
independence. Baseline contact time uses the original independent measurement;
patched acquisition time uses the controller's recorded acquisition event,
so contact after an expired deadline cannot turn a failed episode into success.
The analyzer checks that the normal progress counter remains zero during
acquisition. It reports net object motion before and after acquisition and
preserves unfinished-episode censoring. Time to goal comes from the existing
`FIRST_LATCH` event, whose orientation criterion uses the full rotation;
reported yaw error is a separate planar quantity. Historical failed-target
recurrences do not alone prove a bypass of currently retained memory, because
the existing object-motion retention rules can clear failures.
