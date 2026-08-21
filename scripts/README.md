# Script Index

The scripts directory contains experiment launchers, analysis programs,
diagnostic probes, and historical verification utilities. They remain in their
current paths to preserve commands, notes, and experiment provenance.

Scripts are not part of the pytest suite, even when their filename starts with
`test_`. Run repository tests with `python -m pytest tests`.

## Experiment launchers and sweep control

- `chain_baseline_then_postfix.sh`
- `fig8_replication.sh`
- `fig8_rerun_failed.sh`
- `resume_seed_sign_probe.sh`
- `run_altitude_hold_sweep.sh`
- `run_baseline_seed_extension.sh`
- `run_beta_seed_sweep.sh`
- `run_canonical_baseline_sweep.sh`
- `run_new_defaults.sh`
- `run_postfix_facepicker_sweep.sh`
- `run_seed_sign_probe.sh`
- `run_t_r7_canonical.sh`
- `run_until_complete.sh`

These are candidates for a future `experiments/` or `benchmarks/runners/`
directory. Before moving one, record its expected working directory, config,
inputs, outputs, seed behavior, and associated result set.

## Analysis and aggregation

- `aggregate_beta_sweep.py`
- `analyze_jack_run.py`
- `attribute_ee_outward.py`
- `classify_contact_openings.py`
- `parse_beta_contact_events.py`
- `parse_bilevel_log.py`
- `parse_ltd_sweep.py`
- `parse_q7_wrong_face_reengage.py`
- `parse_seed_sign_probe.py`
- `parse_stage2_L2_sweep.py`
- `parse_sweep_logs.py`
- `reduce_selection.py`
- `summarize_seed_sweep.py`
- `sweep_report.py`

These are candidates for `scripts/analysis/`. Parsers should retain compatibility
with historical log formats.

## Plotting and media

- `make_run_video.sh`
- `plot_fig8.py`
- `plot_wrap_diagnostics.py`

General-purpose rendering utilities also live in `tools/visualizer/`.

## Reference comparison

The `reference/` directory is already a coherent unit:

- `compare_port_vs_reference.py`
- `plot_compare.py`
- `run_reference.sh`
- `summarize_reference_log.py`

Keep reference reproduction distinct from original research extensions.

## Diagnostics and probes

- `check_pose.py`
- `check_vff_z_consistency.py`
- `dump_mode_trace.py`
- `emit_controller_inertia.py`
- `find_drake_releases.py`
- `probe_9_4_5_A1_hold_home_pose.py`
- `probe_9_4_6_lcs_contents.py`
- `probe_9_4_7_B_c3_landscape.py`
- `probe_9_4_7_C_gs_table_analysis.py`
- `probe_f3_p2_metrics.py`
- `probe_hold_pose.py`
- `probe_ik_reachability.py`
- `probe_lambda_regression.py`
- `probe_lambda_regression_v2.py`
- `probe_layer1_ee_to_box.py`
- `probe_lcp_residual.py`
- `probe_stage2_L1.py`
- `probe_stage2_L1_seed1_trace.py`
- `probe_stage2b_rescope.py`
- `probe_stage2b_sampling.py`
- `probe_yaw_xseq.py`
- `probe_yaw_xseq_pushing.py`
- `replay_ref_solve.py`

These are review/archive candidates, not deletion candidates. A probe may encode
the only reproducible record of a negative result or numerical diagnosis.

## Executable diagnostics named as tests

- `test_gravity_sign.py`
- `test_jacobian_conditioning.py`
- `test_layer1_approach.py`
- `test_layer2_rot_bonus.py`
- `test_r7_twist_sign.py`
- `test_yaw_cost_monotonic.py`

These execute experiments or diagnostics directly and may start Drake/Meshcat.
They should eventually be renamed to `check_*` or `probe_*` after references to
their current paths have been inventoried.

## Conformance verification

- `verify_slice_indices.py`
- `verify_stage_A_ee_space_lcs.py`
- `verify_stage_B_admm_dims.py`
- `verify_stage_C_cost_dims.py`
- `verify_stage_D_mpc_endtoend.py`

These protect mathematical structure and should remain distinct from empirical
benchmarks. Suitable assertions may later become numerical-regression tests,
but only after an approved baseline is established.

## Data/model transfer

- `import_anything_objects.py`
- `sync_results_to_d.sh`

Treat these as potentially destructive or external-state-changing utilities.
Inspect their source and resolve exact source/destination paths before running.
