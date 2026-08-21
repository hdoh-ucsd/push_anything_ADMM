# Configuration Guide

Configuration files contain scientific and benchmark behavior. Renaming or
reorganizing them must not change values, defaults, parsing, or command-line
selection.

## Stable entry configurations

- `tasks.yaml`: task definitions, dynamics/contact parameters, costs, goals,
  success criteria, and task-specific controller settings.
- `sampling_c3_params.yaml`: default sampling-C3 configuration.
- `osc_franka.yaml`: operational-space controller gains and limits.
- `directional_tasks.json`: fixed directional pushing goals.

## Named experiment variants

- `sampling_c3_causal_kik.yaml`
- `sampling_c3_kik.yaml`
- `sampling_c3_kik_fast.yaml`
- `sampling_c3_kik_h.yaml`
- `sampling_c3_kik_t.yaml`
- `sampling_c3_kik_jack.yaml`
- `sampling_c3_kik_jack_prefixrank.yaml`
- `sampling_c3_kik_jack_wdfast.yaml`
- `sampling_c3_kik_jack_wdoff.yaml`
- `_pA_v045.yaml`
- `_pA_v090.yaml`

These filenames encode experiment history and must not be normalized until a
manifest maps each name to its purpose, source commit, associated runs, and
superseding configuration if any.

## Provenance rule

For a canonical run, record:

- config path and SHA-256 checksum;
- Git commit and dirty-tree status;
- complete command and overrides;
- random seed;
- environment versions;
- result directory and success criterion.

Never edit a configuration to reinterpret an existing result. Create a new
versioned configuration and link it to the prior one.
