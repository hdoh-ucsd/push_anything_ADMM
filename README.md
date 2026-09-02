# Push Anything / C3+ Reproduction

A Python/PyDrake reproduction and research port of *Push Anything*: sampling-based
contact-implicit model predictive control for non-prehensile manipulation with a
Franka Panda. The repository combines local Linear Complementarity System (LCS)
models, C3/C3+ trajectory optimization, candidate contact placement, and
operational-space execution in simulation.

> **Status:** active research code. The planning and simulation stack runs
> end-to-end, but this is not a claim of full paper-level reproduction. Stored
> benchmarks include successful and censored trials, and the current test
> baseline is not fully green.

![System architecture: sampling, local LCS construction, C3/C3+ MPC, OSC, and PyDrake simulation](docs/figures/system_architecture.svg)

## Overview

The implementation follows the central decomposition used by *Push Anything*
and sampling-C3:

- **PyDrake plant:** a Franka Panda with a spherical pusher interacts with
  configurable rigid objects and a table.
- **Candidate sampling:** the outer controller generates possible end-effector
  placements around the current object geometry.
- **Local contact models:** each relevant state/candidate is linearized into an
  LCS with dynamics and complementarity matrices.
- **C3+ / C3 MPC:** the default C3+ path solves a finite-horizon contact-implicit
  problem; C3 remains available as a comparison/falsification path.
- **Mode selection:** candidate objectives and progress logic select either a
  contact-rich MPC trajectory or a contact-free reposition trajectory.
- **Execution:** an operational-space controller tracks the selected trajectory
  at a faster inner cadence and applies torques to the Franka simulation.

The current default planner uses the repository's reduced end-effector-space
formulation (`x ∈ R^19`, `u ∈ R^3` Cartesian force). The older full-plant
joint-torque formulation remains behind `--r7` for historical falsification
runs; it is not the default architecture.

## Experiments

The measured experiment reports are organized into two tracks:

- **3D Jack Manipulation:** full SE(3) translation and support-tripod
  reorientation with the Jack object.
- **Single-Object Pushing:** fixed-goal planar pushing for imported objects and
  the T-shaped benchmark.

See [EXPERIMENTS.md](docs/EXPERIMENTS.md) for commands, measured outcomes,
artifacts, and reporting limitations.

## Current Research

![Research roadmap separating current implementation, active investigations, and planned work](docs/figures/research_roadmap.svg)

The maintained implementation covers the Push Anything reproduction stack,
planar single-object pushing, C3/C3+, imported object geometries, and an
experimental jack task with full orientation goals. Current research directions
include a **CRISP comparison study** and investigation of **continuous contact
location**. There is no CRISP implementation in this repository.

General 3D/SE(3) non-prehensile manipulation, broader cube-object studies, and
GPU acceleration are roadmap items, not completed capabilities or benchmark
claims. Existing experimental task/configuration branches should not be read as
validated general-purpose 3D manipulation.

## Method

![C3+ solver flow: local LCS, stacked QP, ADMM updates, and MPC action](docs/figures/c3plus_solver_flow.svg)

For each local candidate, `LCSFormulator` produces discrete dynamics

```text
x[t+1] = A x[t] + B u[t] + D λ[t] + d
```

and complementarity data based on the current contact geometry. C3+ introduces
the slack

```text
η[t] = E x[t] + F λ[t] + H u[t] + c,
0 ≤ λ[t] ⟂ η[t] ≥ 0.
```

`C3Solver` then alternates:

1. a global constrained-QP update;
2. the C3+ componentwise `(λ, η)` projection;
3. consensus/dual and penalty updates;
4. a final QP/trajectory extraction.

### The C3 algorithm (consensus complementarity control)

This is the formulation of Aydinoglu, Wei & Posa, *Consensus Complementarity
Control for Multi-Contact MPC* ([arXiv:2304.11259](https://arxiv.org/abs/2304.11259),
§IV), restated in this repository's notation. The paper is the authoritative
source; what follows is our summary of its math and how each piece maps to
this port.

**The problem.** Contact-implicit MPC over the LCS is the finite-horizon
program

```text
minimize    Σ_{k=0..N-1} c_k(x_k, λ_k, u_k)  +  c_N(x_N)
subject to  x_{k+1} = A x_k + B u_k + D λ_k + d
            0 ≤ λ_k ⊥ E x_k + F λ_k + H u_k + c ≥ 0
            x_0 = x(0)
```

with quadratic stage costs. The complementarity constraint is what makes
this hard: it is nonconvex and combinatorial (each contact is either open
with zero force or closed with nonnegative force), so the feasible set is a
union of exponentially many pieces — solving it exactly is a mixed-integer
program.

**The consensus split.** Stack each step's variables as
`z_k = (x_k, λ_k, u_k)` and write the feasible set as the intersection of
two sets: `D`, everything that satisfies the *linear* dynamics and input
bounds, and `H`, everything that satisfies the *complementarity* (contact)
conditions. C3 introduces a consensus copy `w_k` of `z_k` (the paper's
δ) and a scaled dual `v_k`, and requires `z ∈ D`, `w ∈ H`, `z = w`. The
nonconvexity is now quarantined inside `H`.

**The ADMM iteration.** With per-step penalty matrices `G_k` (and the
projection metric `U_k`), each iteration alternates three steps:

```text
1. QP step (couples all time steps, convex):
   z^{i+1} = argmin_{z ∈ D}  Σ_k c_k(z_k) + ‖z_k − w_k^i + v_k^i‖²_{G_k}

2. Projection step (decouples per time step, nonconvex but tiny):
   w_k^{i+1} = Π_H( z_k^{i+1} + v_k^i )   for each k independently

3. Dual update:
   v_k^{i+1} = v_k^i + z_k^{i+1} − w_k^{i+1}
```

Step 1 is one convex QP over the whole horizon — this is where the
trajectory is optimized. Step 2 is where contact decisions are made: each
time step's copy is projected onto the complementarity set, and because the
steps decouple, the projections can run in parallel. The paper solves the
projection either exactly (a small per-step MIQP) or with a fast heuristic;
after a fixed number of iterations the first input of the latest QP solution
is applied and the horizon recedes.

**What this buys.** The consensus structure never linearizes the
complementarity away — contact mode sequences are *chosen* by the
projection, not assumed — yet everything expensive is convex. That is what
lets C3 run as real-time MPC through contact.

**Mapping to this port.** `control/admm_solver.py` implements the loop with
the C3+ variant: the slack `η = E x + F λ + H u + c` is added as an explicit
variable, so step 2 becomes a *componentwise* `(λ, η)` projection (each
scalar pair is resolved by a case split) instead of a per-step MIQP —
cheaper, at the cost of a looser projection. `G` and `U` above are exactly
the `consensus_cost_scale`/`projection_cost_scale` matrices in the task
YAMLs, the dual update is the consensus/penalty update in the solver, and
the "final QP" polish (with its contact-weight boost) extracts the executed
trajectory after the last iteration.

### The Anitescu contact formulation

The LCS above needs a discrete-time contact model to define what `λ` *is*.
The port implements both standard time-stepping formulations in
`control/lcs_formulator.py`, defaulting to Anitescu to match the reference
(`c3/multibody/lcs_factory.cc`, `FormulateAnitescuContactDynamics`).

**The baseline: Stewart–Trinkle.** The exact time-stepping model
(Stewart & Trinkle, 1996) keeps three variable groups per contact — the
normal force `λ_n`, the four friction-pyramid edge forces `λ_t`, and a
slack `γ` that converges to the sliding speed — coupled by three
complementarity conditions on the post-step velocity `v⁺`:

```text
0 ≤ λ_n ⊥ φ/dt + J_n v⁺           ≥ 0     # no interpenetration; force only in contact
0 ≤ λ_t ⊥ E_tᵀ γ + J_t v⁺         ≥ 0     # friction opposes each slip direction
0 ≤ γ   ⊥ μ λ_n − E_t λ_t         ≥ 0     # Coulomb cone: |friction| ≤ μ·normal
```

This is physically exact for the pyramid cone, but the third row couples
the force variables to each other, giving `6·n_c` complementarity variables
(with 4 pyramid edges) and a harder projection.

**The Anitescu relaxation.** Anitescu's convex formulation (Anitescu,
*Optimization-based simulation of nonsmooth multibody dynamics*, Math.
Program. 105, 2006) folds the normal direction *into* each friction edge.
One combined Jacobian replaces the three groups:

```text
J_c = E_tᵀ J_n + diag(μ) J_t                # one row per pyramid edge, (4·n_c, n_v)
0 ≤ λ ⊥ φ/dt + J_c v⁺ ≥ 0                   # single complementarity, λ ∈ R^{4·n_c}
```

Each `λ_j` is now a force along a *cone edge* (normal tilted by `μ` into a
tangent direction); the physical normal force is recovered as `E_t λ` and
the friction force as the tangential part of the edge sum. The Coulomb cone
is satisfied by construction — no third complementarity row, no `γ`.

**What is gained and what is given up.** The gain: the per-contact
conditions define a *convex* (cone-complementarity) problem — solutions
always exist, the variable count drops to `4·n_c`, and the LCS blocks
`(D, E, F, H, c)` take the compact folded form shown in the Franka section
(Step 4), with `F = dt·J_c M⁻¹ J_cᵀ` the standard Delassus operator. The
cost: a known relaxation artifact — during sliding, the folded constraint
introduces a small normal "boost" proportional to the slip speed, so a
sliding object can gain `O(μ·dt·|v_t|)` of separation per step (the
boundary-layer effect). At this port's planning `dt` and push speeds the
artifact is well below the goal tolerances, and — decisively for
conformance — the reference stack plans with the same model.

**In the code.** `lcs_formulator.py` builds the Stewart–Trinkle blocks
first (`γ`/`λ_n`/`λ_t` rows) and, when `_contact_model == "anitescu"`
(the default, matching the reference), overwrites `D, E, F, H, c` with the
folded formulation (`lcs_formulator.py:1692-1698`); the Stewart–Trinkle
path is preserved behind `_contact_model == "stewart_trinkle"` for
falsification. The per-pair-type friction map (`mu_per_pair_type`) enters
through `diag(μ)` in `J_c`.

The candidate objective feeds the sampling-C3 dispatcher. As in receding-horizon
MPC, only the first execution interval is applied before the state and local
contact model are refreshed. See `control/admm_solver.py`,
`control/lcs_formulator.py`, `control/ci_mpc_c3plus.py`, and
`control/sampling_c3/` for the implementation.

### Franka arm mathematics

This section builds the Franka stack's math from the ground up: what the
planner's state and input are, where they come from, and every matrix the
port infers from the physics engine each tick.

#### Step 1 — the planner does not control joints

The Franka Panda has seven joints, but the planner never sees them. The
default formulation (Push-Anything §IV-A, `use_ee_space=True` in
`control/ci_mpc_c3plus.py`) reduces the arm to the spherical pusher at its
end effector and plans in a small mixed robot/object state:

```text
x = [ p^EE_W,  q_WO,  p^O_W,  ṗ^EE_W,  ω^O_W,  ṗ^O_W ]  ∈ R^19
      3        4      3       3        3       3
```

- `p^EE_W`, `ṗ^EE_W` — pusher (end-effector) position and velocity in world;
- `q_WO`, `p^O_W` — object orientation (unit quaternion) and position;
- `ω^O_W`, `ṗ^O_W` — object angular and linear velocity.

#### Step 2 — how the input is defined

The input is the **Cartesian force applied at the pusher**, not joint
torques:

```text
u ∈ R^3   [N],    ‖u‖∞ ≤ F_max   (per-task u-force limits; the config's
                                   "torque_limit" is reinterpreted in Newtons)
```

This is the key abstraction of the reduced formulation: the planner asks
"what force should the ball at the fingertip exert," and the downstream OSC
QP is responsible for finding the seven joint torques that realize that
force on the real arm. (The legacy full-plant path behind `--r7` plans
joint torques `u ∈ R^7` directly; it is retained only for falsification
runs.)

#### Step 3 — the physics we start from

Everything is derived from the standard manipulator equation that Drake
evaluates for the coupled arm/object/table plant:

```text
M(q) v̇ + C(q, v) v = τ_g(q) + B u + J_nᵀ λ_n + J_tᵀ λ_t
```

with `M` the mass matrix, `C` Coriolis/centrifugal terms, `τ_g` gravity,
`B` the input map, and `λ_n`/`λ_t` the unknown normal and tangential
contact forces.

#### Step 4 — the matrices we infer each tick

The continuous dynamics above are nonlinear in `(q, v, u)`. Each planning
tick, `LCSFormulator` (`control/lcs_formulator.py`) linearizes them at the
measured state via Drake **autodiff** (Aydinoglu 2024, eq. 8):

```text
f(q, v, u) = M⁻¹ (B u − C v + τ_g)          # unconstrained acceleration
J_f = ∂f/∂(q, v, u) = [J_q  J_v  J_u]        # autodiff Jacobian
d_v = f(q*, v*, u*) − J_f · [q*; v*; u*]     # offset making it exact at the
                                             # linearization point
```

From the same plant context it extracts the **contact geometry**:

| Symbol | Shape | Meaning |
|---|---|---|
| `φ` | `(n_c,)` | signed gap distance per contact pair (negative = penetrating) |
| `J_n` | `(n_c, n_v)` | normal contact Jacobians (maps velocities to gap rates) |
| `J_t` | `(4 n_c, n_v)` | tangential Jacobians, 4 friction-pyramid edges per contact |
| `E_t` | `(n_c, 4 n_c)` | selector summing the four tangent forces of each contact |
| `μ` | scalar / per-pair | friction coefficient(s) from the task config |

The contact pairs are the pusher-vs-object faces plus the object-vs-ground
witness points, so `n_c` changes with the sampled candidate and the object's
pose — these matrices are re-inferred at every tick and for every candidate.

Under the default Anitescu contact model, friction is folded into one
combined contact Jacobian and the discrete-time LCS blocks are assembled as
(`lcs_formulator.py:1692-1698`):

```text
J_c = E_tᵀ J_n + diag(μ) J_t                          # friction-folded Jacobian

x[t+1] = A x[t] + B u[t] + D λ[t] + d                 # dynamics row
0 ≤ λ[t] ⊥ E x[t] + F λ[t] + H u[t] + c ≥ 0           # contact row

D = [ dt² · qdotNv · M⁻¹ J_cᵀ ;  dt · M⁻¹ J_cᵀ ]      # how contact forces move the state
E = [ dt·J_c·J_q + E_tᵀ J_n · vNqdot/dt ;  J_c + dt·J_c·J_v ]
F = dt · J_c · M⁻¹ J_cᵀ                               # contact-force coupling (Delassus)
H = dt · J_c · J_u                                    # how the input opens/closes gaps
c = E_tᵀ φ/dt + dt·J_c·d_v − E_tᵀ J_n · vNqdot · q/dt # gap constants
```

Reading the complementarity row as physics: `E x + F λ + H u + c` predicts
the post-step contact velocity/gap of each friction-pyramid direction;
`0 ≤ λ ⊥ (·) ≥ 0` says a contact force may only push (never pull) and only
while its gap is closed. `A`, `B`, `d` are the corresponding discrete-time
integration of `J_f` and `d_v`.

#### Step 5 — what the solver minimizes over that model

C3+ then solves, exactly as in the xArm section, the quadratic tracking
problem `Σ (x_t − x_d)ᵀ Q (x_t − x_d) + Σ u_tᵀ R u_t` over the LCS, with
ADMM consensus/projection weights `G` and `U` on the `(λ, η)` copies; the
weights come from the per-task YAML (`config/sampling_c3_kik_t.yaml` and
friends), including the final-QP contact boost on the last polish solve.

#### Step 6 — realizing `u` on the seven-joint arm

The OSC (`control/osc/operational_space_controller.py`) closes the gap
between the planner's fiction (a free-flying force ball) and the real arm:
it tracks the planned pusher trajectory and promotes the planner's contact
force to its QP, producing joint torques `τ ∈ R^7` through the same
inverse-dynamics structure as the xArm OSC. Two Franka-specific terms
matter: the reference `joint2` posture pin (`Kp/Kd/W_joint2`,
`joint2_target_rad = 1.1`) that kills the null-space orbit in the endgame,
and `q_init_franka` seeding. The planner's `λ` is a *cost demand*, not a
measured force — the OSC and simulator decide what is physically exerted.

## Single-Object Pushing

The corrected protocol uses one uninterrupted manipulation session per object:
the robot, object, controller state, and random-goal stream carry over across all
28 goals. Each goal has its own 600-second simulated-time limit. A session passes
only if it reaches all 28 goals consecutively; failed sessions are not replaced
with another seed.

The latest-model campaign reran 23 objects on August 28, 2026 and retained the
requested prior outcomes for Eraser and Gallon Milk. The combined result is
**17 passing sessions out of 25 objects**. Full current results and provenance
are in [`FIG8_CONSECUTIVE_28_LATEST_SESSIONS.csv`](FIG8_CONSECUTIVE_28_LATEST_SESSIONS.csv).

| Object | Consecutive goals | Outcome | Failure cause |
|---|---:|:---:|---|
| Letter I | 28/28 | Pass | — |
| Letter C | 28/28 | Pass | — |
| Letter R | 28/28 | Pass | — |
| Letter A | 28/28 | Pass | — |
| Letter Y | 4/28 | Fail | Near-success timeout: 0.0022 m position error and 0.1078 rad orientation error |
| Letter G | 28/28 | Pass | — |
| Letter B | 28/28 | Pass | — |
| Letter 3 | 28/28 | Pass | — |
| Letter H | 28/28 | Pass | — |
| Letter E | 4/28 | Fail | No-contact recovery loop; final errors 0.0263 m and 0.0861 rad |
| Letter S | 28/28 | Pass | — |
| Expo Box | 28/28 | Pass | — |
| Lotion | 28/28 | Pass | — |
| Wood Block | 24/28 | Fail | Planner/reposition stall; final errors 0.0840 m and 1.5641 rad |
| Tape | 2/28 | Fail | Near-success timeout: 0.0154 m and 0.1155 rad, followed by no-contact retries |
| Eraser | 0/28 | Fail | Persistent topple (retained) |
| Milk Bottle | 5/28 | Fail | No-contact recovery loop; final position error 0.1808 m |
| Clamp | 28/28 | Pass | — |
| Chicken Broth | 28/28 | Pass | — |
| Egg Carton | 3/28 | Fail | Inner workspace limit: EE radius 0.2787 m (minimum 0.280 m) |
| Book | 28/28 | Pass | — |
| Baby Toy | 28/28 | Pass | — |
| Gallon Milk | 0/28 | Fail | Outer workspace limit: EE radius 0.7543 m (maximum 0.750 m; retained) |
| Xbox | 28/28 | Pass | — |
| Push T | 28/28 | Pass | — |

The success gate remains reference-conformant: position error must be below
0.02 m and quaternion geodesic orientation error below 0.10 rad simultaneously.
The primary actionable defect is a no-contact recovery loop: after an
unproductive C3 segment, the dispatcher can reposition and select an equivalent
ineffective contact repeatedly until timeout. Candidate failures should be
invalidated, accepted reposition samples should predict contact closure and a
nontrivial force, and repeated retries should force a fresh global sample set.
Workspace targets and their tracked trajectories also need a 5–10 mm safety
margin. Eraser requires separate non-planar topple recovery.

## Object-Informed-Manipulation

The job is to fetch the five tabletop scenarios from the external OIM reference,
preserve their xArm model, scene assets, start/goal poses, and evaluation
protocol, then run each scenario with our C3+ controller and publish comparable
result artifacts. C3+ replans from the measured xArm and object state after each
executed control interval until the OIM goal gate passes or the run budget ends.

![OIM reference scenarios imported into the xArm C3+ evaluation pipeline](docs/figures/oim_c3plus_scenario_pipeline.svg)

### Native C++ integration: `oim_c++_anything`

The native DAIRLab integration is isolated from the modified reference checkout
in a clean Git worktree:

- published branch (README, math, gate ledgers):
  [hdoh-ucsd/dairlib@`oim_c++_anything`](https://github.com/hdoh-ucsd/dairlib/blob/oim_c++_anything/README.md)
- local worktree: `external/oim_c++_anything` (gitignored — clone the
  branch above if it is missing)
- branch: `oim_c++_anything`
- baseline/design commit: `854a8afc`
- canonical task configuration:
  `examples/sampling_c3/oim_t/parameters/oim_t.yaml`
- maintained process diagram and validation gates:
  `examples/sampling_c3/oim_t/ARCHITECTURE.md`

The native stack runs as three LCM processes configured by the single
canonical YAML:

```text
oim_t.yaml
 ├── xarm6_sim
 ├── xarm6_osc_controller
 └── xarm6_sampling_c3_controller
```

#### Pipeline: inputs, outputs, and the equation at each block

Each block below consumes the previous block's output over LCM and publishes
its own. One planning cycle traverses the loop once; the simulator and OSC run
continuously underneath it.

1. **`xarm6_sim` — physics (2 ms step).**
   *Input:* commanded joint torques `τ ∈ R^6`.
   *Output:* measured robot state `(q, q̇) ∈ R^6 × R^6` at 500 Hz and the
   object's spatial pose/velocity `(p^O_W, q_WO, ṗ^O_W, ω^O_W)` at 20 Hz.
   *Equation:* Drake's rigid-body dynamics with hydroelastic/point contact —
   `M(q) q̈ + C(q, q̇) = τ + τ_g + J_cᵀ f_c`.

2. **State reduction (inside `xarm6_sampling_c3_controller`).**
   *Input:* the measured six-joint arm state and object spatial state.
   *Output:* the reduced planning state
   `x = [p^P_W, q_WO, p^O_W, ṗ^P_W, ω^O_W, ṗ^O_W] ∈ R^19`, where the arm is
   collapsed to its stick-tip point `p^P_W` via forward kinematics. The six
   joints never enter the optimization.

3. **Contact sampling.**
   *Input:* the reduced state `x` and the T's exact two-box boundary.
   *Output:* a set of candidate pusher placements — points on the object
   perimeter with outward face normals, lifted to world coordinates at
   sampling height.

4. **LCS linearization (per candidate).**
   *Input:* one candidate pusher position and the current `x`.
   *Output:* a local Linear Complementarity System — matrices
   `(A, B, D, d, E, F, H, c)` with `λ ∈ R^20` contact variables over an
   `N = 5` horizon:
   `x_{t+1} = A x_t + B u_t + D λ_t + d`,
   `0 ≤ λ_t ⊥ E x_t + F λ_t + H u_t + c ≥ 0`.

5. **C3+ solve (per candidate).**
   *Input:* the candidate's LCS, the goal-encoding desired state `x_d`, and
   the cost matrices `(Q, R, G, U)`.
   *Output:* an open-loop plan `{x_t*, u_t*, λ_t*}` minimizing
   `Σ (x_t − x_d)ᵀ Q (x_t − x_d) + Σ u_tᵀ R u_t` by ADMM over consensus
   copies of `(λ, η)`.

6. **Rollout ranking and selection.**
   *Input:* every candidate's plan.
   *Output:* the single executed candidate — each plan is forward-simulated
   through its LCS and scored with the same quadratic error
   `Σ eᵀ Q e + e_Nᵀ Q e_N`, `e = x_t − x_d` (`dynamic_rollout_cost`); the
   argmin wins.

7. **`xarm6_sampling_c3_controller` output — the execution plan.**
   What the controller *gives* is not torques and not the raw C3 solution:
   it publishes one timestamped LCM trajectory
   (`lcmt_timestamped_saved_traj`) holding three time-aligned tracks sampled
   from the winning plan's first execution interval:

   - `end_effector_position_target` — tip position knots `p_des(t) ∈ R^3`,
     from the plan's state trajectory `x_t*` (pushing) or from the
     collision-aware acquisition IK waypoints (repositioning);
   - `end_effector_stick_axis_target` — the commanded stick axis (vertical),
     the orientation reference;
   - `end_effector_force_target` — the feedforward Cartesian force
     `f*(t) ∈ R^3`, which is the C3+ input solution `u_t*` passed through
     one-to-one; this is how the planned contact force reaches execution.

   Only the first interval is executed before the loop replans from the
   measured state (receding horizon).

8. **`xarm6_osc_controller` — operational-space control (500 Hz).**
   *Input:* the three-track plan above and the measured `(q, q̇)` from the
   simulator.
   *What the OSC gives:* the six joint torques `τ ∈ R^6` — it is the only
   block that talks to the motors. Each control tick solves DAIRLab's
   inverse-dynamics QP with decision variables `(v̇, τ, λ)`:

   ```text
   minimize    Σ_i (ÿ_i^cmd − J_i v̇ − J̇_i q̇)ᵀ W_i (ÿ_i^cmd − J_i v̇ − J̇_i q̇)  +  v̇ᵀ W_accel v̇
   subject to  M(q) v̇ + C(q, q̇) = B τ + τ_g + J_extᵀ f*        (inverse dynamics)
               |τ| ≤ τ_max                                       (effort limits)
   ```

   with the commanded task accelerations from PD on each tracking objective:

   ```text
   ÿ_i^cmd = ÿ_i^des + kp (y_i^des − y_i) + kd (ẏ_i^des − ẏ_i)
   ```

   The tracking objectives `y_i` are the tip position (`kp = 200`,
   `kd = 20`), the stick-axis orientation (weighted `0.35² = 0.1225`
   relative to translation, yaw component zeroed for the axisymmetric
   stick), and a 0.01-weight joint-posture regularizer;
   `W_accel = 10⁻⁷ I` regularizes accelerations. The planner's
   `end_effector_force_target` `f*` enters the dynamics constraint as an
   external force, so planned contact forces are actively pressed, not just
   implied by position error. The resulting torque passes through the source
   xArm velocity-servo bridge (gains 300/300/200/200/200/200, MuJoCo
   force-clamp ordering, joint-4 passive spring) before publication. These
   torques close the loop into block 1.

9. **Goal gate (each planning cycle).**
   *Input:* the measured object pose.
   *Equation measured:* `e_p = ||(x, y) − (x_g, y_g)||₂` and
   `e_θ = |wrap(θ − θ_g)|` (see the task definition below). The run ends when
   both pass their tolerances simultaneously, or when the update budget is
   exhausted.

`oim_t.yaml` replaces the legacy task-level composition through
`sim_params.yaml`, `goal_params.yaml`, and
`sampling_c3_controller_params.yaml`. It owns the xArm model contract, OIM T
start and goal in the unwarped `oim_world` frame, simulation timing, success
tolerances, and LCM routing. Algorithm-specific C3+, sampling, repositioning,
progress, and OSC parameter files remain separate until their numerical
provenance has been validated for xArm.

### Task: `open_table`

An xArm6 with a vertical pushing stick must push a planar T block across an
open table from its start pose to a goal pose. The goal variables are the
object's planar SE(2) pose in the `oim_world` frame
(`examples/sampling_c3/oim_t/parameters/oim_t.yaml` in the C++ worktree):

```text
g = (x_g, y_g, θ_g) = (0.381, -0.400, 3.1416)      # object.goal_pose
x^o = (x, y, θ)                                     # measured T pose (yaw from quaternion)
```

Success is a terminal tolerance check on both goal variables simultaneously
(`task.translation_tolerance`, `task.orientation_tolerance`):

```text
e_p = || (x, y) - (x_g, y_g) ||_2         <  0.05 m
e_θ = | wrap(θ - θ_g) |                   <  0.10 rad
```

#### Orientation error and the wrap function

The orientation error is computed in three steps
(`xarm6_full_sampling_c3plus.cc:92-107`):

1. **Yaw extraction.** The measured object quaternion
   `q_WO = (w, x, y, z)` is normalized and reduced to its heading:

   ```text
   θ = atan2( 2(wz + xy),  1 − 2(y² + z²) )
   ```

   This is the standard ZYX yaw formula; roll and pitch are ignored by the
   planar gate (a tilted or toppled T is caught by the separate settle check's
   tilt angle `ψ = acos((R_WO ẑ)·ẑ)`, not by `e_θ`).

2. **Raw difference.** `Δ = θ − θ_g`. This raw value is meaningless as a
   distance, because yaw lives on the circle S¹, not on the real line: the
   values `θ` and `θ + 2π` are the same physical heading, so `Δ` can be off
   by any multiple of `2π` depending on which branch `atan2` returned.

3. **Wrapping.**

   ```text
   wrap(Δ) = atan2(sin Δ, cos Δ)   ∈ (−π, π]
   ```

   Feeding `Δ` through `sin`/`cos` erases every multiple of `2π` (both are
   `2π`-periodic), and `atan2` rebuilds the unique representative in
   `(−π, π]`. The result is the **shortest signed arc** from `θ_g` to `θ` —
   the geodesic distance on the circle. It is exact (no branching or modulo
   edge cases), and its absolute value never exceeds `π`.

Why this matters for `open_table` specifically: the goal heading is
`θ_g = 3.1416 ≈ π`, which sits exactly on the `atan2` branch cut. A T that has
essentially reached the goal can be measured at `θ = +3.10` on one tick and
`θ = −3.10` on the next — the same physical pose, differing only in branch.
Without wrapping, the second measurement scores `|Δ| = |−3.10 − 3.1416| =
6.24 rad`, a catastrophic false failure; wrapped, both score
`e_θ ≈ 0.04 rad` and correctly pass the 0.10 rad gate. The reported terminal
errors in the gate ledgers (e.g. `2.9 rad`) are therefore true remaining
rotations, never branch artifacts.

The same wrap is used everywhere a yaw difference is consumed: the terminal
gate, the per-cycle progress accounting
(`xarm6_full_sampling_c3plus.cc:836-846`), and the settle check's `yaw_delta`.
The task therefore requires roughly 0.8 m of translation plus a genuine ~π
reorientation of the T.

### What we optimize

Each control cycle solves a finite-horizon contact-implicit MPC problem with
C3+ (ADMM over consensus copies) on a locally linearized Linear
Complementarity System. The state is `x ∈ R^19` (pusher position, object
quaternion, object position, then velocities), the input is `u ∈ R^3`
(Cartesian pusher force), contact forces are `λ ∈ R^20`, and the horizon is
`N = 5`:

```text
minimize    Σ_{t=0..N} (x_t - x_d)ᵀ Q (x_t - x_d)  +  Σ_{t=0..N-1} u_tᵀ R u_t

subject to  x_{t+1} = A x_t + B u_t + D λ_t + d          (LCS dynamics)
            0 ≤ λ_t ⊥ E x_t + F λ_t + H u_t + c ≥ 0     (complementarity)
```

The desired state `x_d` encodes the `open_table` goal variables directly: the
object-position slots hold `(x_g, y_g, resting_height)` and the
object-quaternion slots hold `q(θ_g)`, a pure yaw rotation built from
`goal_pose.z()` (`xarm6_full_sampling_c3plus.cc:1597-1605`). The cost
matrices are assembled in `RunSolveAtSampledPusher`
(`xarm6_full_sampling_c3plus.cc:1579-1596`):

```text
Q = state_cost_scale · diag(state_cost_diagonal)
  = 50 · diag(0.01, 0.01, 0.01,          # pusher position
              0.1, 0.1, 0.1, 0.1,        # object quaternion  → orientation goal
              200, 200, 120,             # object position    → translation goal
              5, 5, 5,  0.05 ×6)         # velocities
R = 1.0 · diag(0.01, 0.01, 0.01)
```

so translation error is weighted at an effective 10,000 per m² on object x/y
and orientation enters through the quaternion-error terms. ADMM additionally
carries consensus and projection penalties `G = 0.01·diag(g)` and
`U = 0.26·diag(u)` on the `λ`/`η` copies (weights 2/1 and 20/1); these enforce
the complementarity structure and are not part of the task objective.

Around that inner QP, three more objectives shape the behavior:

- **Sample ranking.** Candidate pusher placements are each solved and then
  scored by forward-simulating the plan through the LCS and accumulating the
  same quadratic error, `Σ eᵀ Q e` plus a terminal term
  (`dynamic_rollout_cost`, `xarm6_full_sampling_c3plus.cc:1740-1834`); the
  minimum-cost candidate is executed. A separate
  `object_yaw_cost_weight: 50.0` biases goal/sample selection toward yaw
  progress.
- **Acquisition IK.** Repositioning to a sampled contact solves an inverse
  kinematics problem with a position constraint on the stick tip and the
  restored source tilt objective `w_tilt · (1 - cos ψ)`, `w_tilt = 80`, which
  keeps the stick vertical inside the feasibility band
  (`xarm6_sampling_c3_controller.cc:120`, `:391-400`).
- **OSC tracking.** The 500 Hz operational-space controller tracks the
  selected trajectory with Cartesian gains `kp = 200`, `kd = 20` and a
  0.01-weight joint-posture regularizer.

In short: the *task* asks for `e_p < 0.05 m` and `e_θ < 0.10 rad` on the T's
planar pose; the *optimizer* minimizes a quadratic penalty on exactly those
two errors (plus small pusher/velocity/effort regularization) subject to
contact-implicit dynamics, and every sampled contact location competes on the
same objective.

### Status

Terminal `open_table` success has not yet been achieved. The chronological
gate records (gates 101–388: contact budgeting, admission gating,
dwell/release cycles, contact receipts, the GATE_380 direction-preserving
limiter + restored `w_tilt = 80` root-cause fix, recovery admission, and
neutral-retreat fallback) live in the C++ worktree with one ledger per gate
range under `examples/sampling_c3/oim_t/` and are summarized in that
worktree's `README.md`. The exact model provenance for the vendored T is in
`examples/sampling_c3/oim_t/OIM_T_PROVENANCE.md`.

## Quick Start

Create the audited environment and run from the repository root:

```bash
conda env create -f environment.yml
conda activate push_anything_admm

# Basic configured task
python main.py pushing

# Sampling-C3 with the default outer-controller configuration
python main.py pushing --sampling-c3 --seed 0 --name pushing_seed0

# Stored T-object workflow configuration
python main.py push_t --max-time 600 \
  --sampling-c3 config/sampling_c3_kik_t.yaml \
  --seed 0 --name push_t_seed0
```

Runs write `results/<name>.txt` and include Git/configuration metadata in the
log. Result media can be rendered separately:

```bash
scripts/make_run_video.sh push_t_seed0 --task push_t
```

See [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) before comparing or reporting
experiments. It records the audited dependency versions, canonical run metadata,
result-storage policy, and current limitations.

## Repository Structure

```text
control/                 C3/C3+, ADMM, LCS/LCP, costs, OSC, sampling-C3
sim/                     PyDrake environment and object models
config/                  tasks, controller settings, experiment variants
scripts/                 launchers, diagnostics, analysis, plotting
tools/visualizer/        log parsing and result/video rendering
tests/                   unit, solver, model, integration, regression tests
docs/                    conformance notes, investigations, generated figures
results/                 ignored working outputs and stored local campaigns
main.py                  CLI, simulation loop, logging, orchestration
```

More detailed maps are in [REPOSITORY_GUIDE.md](docs/REPOSITORY_GUIDE.md) and
the local indexes under `config/`, `scripts/`, and `tests/`.

## Tests

```bash
python -m pytest tests
```

The recorded cleanup baseline collected 344 tests: 292 passed, 37 failed, and
15 errored. Twenty-five non-passing nodes were blocked because the audit runner
forbids the localhost sockets that Meshcat requires; the remaining discrepancies
are classified without changing numerical expectations. See
[TEST_BASELINE.md](docs/TEST_BASELINE.md) and
[`tests/baseline_failures.yaml`](tests/baseline_failures.yaml).

Before changing C3/C3+, ADMM, complementarity projection, LCS construction, or
MPC behavior, establish a clean numerical baseline and preserve the reported
experiment configuration.

## Figure Reproduction

The README figures are regenerated with:

```bash
python docs/figures/generate_readme_figures.py
```

The three SVGs are conceptual diagrams derived from the current source
architecture. The PNG is copied from an existing measured result and is never
recomputed by the README generator.

## References

- H. Bui et al., “Push Anything: Single- and Multi-Object Pushing From First
  Sight with Contact-Implicit MPC,” arXiv:2510.19974, 2025.
- A. Aydinoglu, A. Wei, W.-C. Huang, and M. Posa, “Consensus Complementarity
  Control for Multi-Contact MPC,” *IEEE Transactions on Robotics*, 40,
  3879–3896, 2024.
- S. Venkatesh, B. Bianchini, A. Aydinoglu, W. Yang, and M. Posa,
  “Approximating Global Contact-Implicit MPC via Sampling and Local
  Complementarity,” arXiv:2505.13350, 2025.
- Y. Li, H. Han, S. Kang, J. Ma, and H. Yang, “On the Surprising Robustness of
  Sequential Convex Optimization for Contact-Implicit Motion Planning,”
  arXiv:2502.01055, 2025. Comparison work is a research direction only.
