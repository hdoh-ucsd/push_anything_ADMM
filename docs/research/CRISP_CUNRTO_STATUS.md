# CRISP and cuNRTO Branch Status

**Audit date:** August 26, 2026

**Purpose:** presentation-ready status, blockers, and recommended next steps

## Executive summary

| Direction | Current status | Principal blocker | Decision |
|---|---|---|---|
| CRISP | Standalone planner and studies completed; research branch merged into `main` | Solver fidelity, real-time cost, missing robot/contact execution physics, and negligible benefit from within-face optimization | Retain as research evidence; do not integrate full CRISP into the live controller yet |
| cuNRTO | Design study and GPU benchmarks completed; no dedicated cuNRTO implementation branch | The current workload is too small for GPU efficiency; sparse CPU OSQP dominates and the tested GPU path was slower | Pause until the problem size or architecture changes materially |

## 1. CRISP

### Current status

The `worktree-crisp-bb-pushbox` branch was merged into `main` on August 21,
2026. It produced a standalone CRISP Algorithm 1 SCP solver, the Appendix B-B
quasi-static push-box formulation, comparison against the official C++ source,
translation/rotation/multi-face studies, a CRISP-to-C3+ linearization study,
continuous contact-location Path B, unit tests, and reproducible analysis tools.

The implementation remains deliberately disconnected from the live
sampling-C3 controller. Its primary sources are `control/crisp/README.md`,
`docs/crisp/path_b_results.md`, and
`docs/crisp_c3plus_linearization_study.md`.

### What worked

On the easiest canonical box task—0.15 m straight translation with no
rotation—the standalone planner achieved approximately 4.6 mm position error,
selected the correct pushing face, required no face switching, and produced a
nominal end-effector speed of approximately 0.07 m/s.

At the reference scale it also qualitatively reproduced CRISP's ability to
discover a four-face translation-and-rotation sequence.

### What blocked full integration

1. **Inner QP solver mismatch.** The official implementation uses the
   interior-point PIQP solver; the port uses first-order OSQP. At reference
   scale, the port did not simultaneously reproduce terminal accuracy and
   feasibility. Its best feasible tested point had approximately
   `7.5e-5` violation, versus the reference tolerance of `1e-6` and the paper's
   reported approximately `8.3e-9`.
2. **Pure rotation is invisible from the natural cold start.** Yaw dynamics
   contain a contact-location-times-force product. At an all-zero
   initialization both terms are zero, making the first-order yaw gradient
   exactly zero. The optimizer can report success on a stationary, do-nothing
   trajectory. A nonzero contact warm start restores a gradient but did not
   converge reliably in the tested port.
3. **Multi-face feasibility remains inadequate.** Diagonal and
   translation-plus-rotation goals found multiple faces but smeared forces or
   switched excessively. The reference-scale four-face result ended near 0.69
   constraint violation.
4. **The B-B model omits execution physics.** It does not include Franka
   kinematics, joint and torque limits, workspace reachability, pusher friction,
   contact-point continuity, reposition time, momentum, toppling, or
   out-of-plane motion. Its contact point can effectively teleport between
   faces.
5. **Runtime is outside the live MPC budget.** The live controller period is
   75 ms. The reference-scale standalone solve took roughly 180 seconds and 218
   outer iterations.

### Path B result

Path B retained C3+ and optimized only the contact location along an already
selected face. The measured ceiling was decisive for the canonical box:

- perfect within-face optimization improved ranked cost only 0.7–0.9% over
  five random samples;
- correct face selection was 30–60 times more valuable;
- the existing goal-aligned centering heuristic already sampled near the
  within-face optimum; and
- every new location evaluation required IK, a cost-LCS rebuild, and a forward
  simulation.

Path B was stopped because the measured benefit was negligible, not because it
was technically impossible.

### Recommendation

Do not integrate the complete CRISP B-B formulation into the live controller
yet. Preserve it as a standalone comparison and validation benchmark. The
highest-leverage next experiment is deterministic face enumeration or
goal-conditioned face selection, because the discrete face decision dominates
the continuous within-face location.

## 2. cuNRTO and GPU acceleration

### Current status

There is no dedicated cuNRTO implementation branch. cuNRTO served as the design
basis for the GPU-ADMM investigation on `worktree-gpu-admm-plan`. That branch
has no unique unmerged commits; its profiling and benchmark artifacts were
incorporated into repository history. The frozen evidence is in
`docs/gpu/GPU_BASELINE.md`.

Completed work includes hot-loop profiling, a real-QP corpus, CPU/GPU primitive
benchmarks, candidate-count crossover measurements, numerical parity tests,
optional-backend isolation, and a design for batched on-device C3+ ADMM.

### Measured workload

- 6 candidates per control tick
- horizon approximately 10
- QP dimension approximately 639
- Hessian density 0.161%
- approximately 26 QP solves per tick
- three fixed C3+ ADMM iterations
- approximately 8,000 OSQP inner iterations per tick
- about 81% of wall time in sparse CPU OSQP solves
- only about 0.6% of solve time in the easily parallelized C3+ projection

Even infinite acceleration of the projection would improve end-to-end runtime
by only approximately 0.6%.

### What blocked a cuNRTO-style implementation

1. **Batch size is too small.** The workload has six candidates. The measured
   crossover was approximately 151 candidates with resident data and 389 when
   transfers were included.
2. **GPU primitives were slower.** The GPU lost for every tested batch from 1
   through 64. At batch 32 it ran at approximately `0.10x` CPU speed—about ten
   times slower—because kernel-launch overhead dominated.
3. **Sparse CPU algebra fits the problem better.** CPU OSQP exploits the very
   sparse matrices; a dense batched GPU formulation performs much more
   arithmetic and loses that advantage.
4. **Factorization reuse is limited.** Numerical LCS matrices differ across
   candidate end-effector poses, and the ADMM penalty ramp changes the QP
   diagonal between outer iterations.
5. **Float64 throughput is weak on the available GPU.** The RTX 5070 Ti runs
   float64 at approximately 1/64 of its float32 rate, leaving little raw
   throughput advantage over the CPU for the current numerical path.
6. **cuNRTO does not map directly.** Some of cuNRTO's gains arise from large
   feedback-gain computations and fully on-device workloads; C3+ has no
   equivalent feedback-gain block.

### Recommendation

GPU development should remain paused, not abandoned. Re-profile when candidate
count, horizon, contact count, multi-object scale, or SE(3) state size grows
materially; when a dense solve dominates; or when the complete iterative solve
can remain on-device.

Until then, prioritize CPU-side work: reduce unnecessary OSQP iterations,
improve termination checks, reuse symbolic structure, test CPU candidate
parallelism, reduce full QP solve count, and improve face selection.

## Overall conclusion

CRISP was blocked by model mismatch, solver fidelity, initialization
sensitivity, feasibility, and real-time execution requirements. Its most
directly transferable idea delivered less than 1% measured improvement.

cuNRTO was blocked by workload economics: the present QPs are too small, sparse,
and lightly batched for the GPU, and launch overhead overwhelms the available
parallel work.

The strongest immediate direction is improved discrete contact-mode or face
selection. It has the largest measured leverage and may eventually create the
larger problem structure that justifies revisiting CRISP and cuNRTO.
