# Reproducibility blocker

The authoritative C3+ sampler used by `SamplingC3Controller::ComputePlan`
cannot consume the deterministic task seed recorded by benchmark v2.

Evidence at C3+ commit `bd5ce7b52d7133d54065578e42874d92d3cce95c`:

- `examples/sampling_c3/generate_samples.cc:517` constructs
  `std::mt19937 gen(std::random_device{}())`.
- The multi-object path repeats this pattern at lines 651 and 656-657.
- `GenerateSampleStates` accepts no RNG or seed argument.
- `SamplingC3Controller::ComputePlan` calls `GenerateSampleStates` without a
  seed.
- `SAMPLING_C3_FORENSICS_SEED_INFO` is metadata only and does not seed the
  controller.

The synchronized manifest's seed policy is valid task provenance, but C3+
does not consume it. Comparing one run per cost setting would therefore mix a
cost change with unrelated candidate draws.

Resolution requires a separately approved sampling/reproducibility change,
followed by verification that identical task seeds reproduce candidate pools
across cost configurations. This study did not make that change because the
task expressly prohibits silently altering sampling logic.
