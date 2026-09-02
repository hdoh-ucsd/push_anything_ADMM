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
formulation ($x \in \mathbb{R}^{19}$,
$u \in \mathbb{R}^3$ Cartesian force). The older full-plant joint-torque
formulation remains behind `--r7` for historical falsification runs; it is not
the default architecture.

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

For each local candidate, `LCSFormulator` produces the discrete dynamics

```math
x_{k+1} = A x_k + B u_k + D\lambda_k + d,
\qquad k=0,\ldots,N-1,
```

and complementarity data based on the current contact geometry. C3+ introduces
the slack

```math
\eta_k := E x_k + F\lambda_k + H u_k + c,
\qquad 0 \leq \lambda_k \perp \eta_k \geq 0.
```

For vectors, $0 \leq a \perp b \geq 0$ means $a \geq 0$, $b \geq 0$,
and $a^\top b=0$ (equivalently, $a_jb_j=0$ for every component $j$).

`C3Solver` then alternates:

1. a global constrained-QP update;
2. the C3+ componentwise $(\lambda,\eta)$ projection;
3. consensus/dual and penalty updates;
4. a final QP/trajectory extraction.

### The C3+ algorithm

C3+ retains the consensus-ADMM scaffold introduced by Aydinoglu, Wei & Posa
in *Consensus Complementarity Control for Multi-Contact MPC*
([arXiv:2304.11259](https://arxiv.org/abs/2304.11259), §IV), then changes the
contact representation and projection as described by Bui et al. The
derivation below starts from C3 and marks the point at which C3+ intervenes.

**Shared contact-implicit problem.** Both algorithms optimize the same
finite-horizon LCS problem:

```math
\begin{aligned}
\underset{\{x_k,u_k,\lambda_k\}}{\mathrm{minimize}}\quad
  & \sum_{k=0}^{N-1} \ell_k(x_k,\lambda_k,u_k) + \ell_N(x_N) \\
\text{subject to}\quad
  & x_{k+1}=Ax_k+Bu_k+D\lambda_k+d,
    && k=0,\ldots,N-1, \\
  & 0\leq\lambda_k\perp Ex_k+F\lambda_k+Hu_k+c\geq0,
    && k=0,\ldots,N-1, \\
  & x_0=x_{\mathrm{meas}}.
\end{aligned}
```

The complementarity constraint makes the feasible set nonconvex and
combinatorial: each contact component is either open with zero force or
closed with zero gap velocity.

**C3 consensus scaffold.** C3 stacks
$z_k:=(x_k,\lambda_k,u_k)$. The set $\mathcal{D}$ contains the linear
dynamics and bounds, while $\mathcal{H}$ contains the complementarity
conditions. A consensus copy $\delta$ and scaled dual $\omega$ impose
$z\in\mathcal{D}$, $\delta\in\mathcal{H}$, and $z=\delta$ through

```math
\begin{aligned}
z^{i+1}
  &=\underset{z\in\mathcal{D}}{\mathrm{arg\,min}}
    \left[J(z)+\frac{1}{2}\sum_{k=0}^{N-1}
    \left\|z_k-\delta_k^i+\omega_k^i\right\|_{G_k}^2\right], \\
\delta_k^{i+1}
  &=\Pi_{\mathcal{H}_k}^{U_k}\!\left(z_k^{i+1}+\omega_k^i\right), \\
\omega_k^{i+1}
  &=\omega_k^i+z_k^{i+1}-\delta_k^{i+1}.
\end{aligned}
```

Here $\|a\|_G^2:=a^\top G a$. The first line is one convex QP coupled across
the horizon. In baseline C3, the second line projects each time step onto
the full complementarity set, using a small MIQP or a heuristic projection.

**C3+ intervention: expose the slack.** C3+ augments each decision block with
the complementarity slack,

```math
\bar z_k:=\left(x_k,\lambda_k,u_k,\eta_k\right),
\qquad
\eta_k=Ex_k+F\lambda_k+Hu_k+c.
```

The hard part of the feasible set is now only the product constraint between
$\lambda_k$ and $\eta_k$. Define

```math
\mathcal{C}:=
\left\{(a,b)\in\mathbb{R}^{n_\lambda}\times\mathbb{R}^{n_\lambda}
:a\geq0,\ b\geq0,\ a\odot b=0\right\},
```

where $\odot$ denotes elementwise multiplication. C3+'s two split sets are

```math
\begin{aligned}
\mathcal{D}_+
  &:=\left\{\bar z:
    \begin{array}{l}
    x_{k+1}=Ax_k+Bu_k+D\lambda_k+d,\\
    \eta_k=Ex_k+F\lambda_k+Hu_k+c,\\
    x_0=x_{\mathrm{meas}},\ \text{and all configured bounds hold}
    \end{array}\right\}, \\
\mathcal{H}_+
  &:=\left\{\bar\delta:
    (\delta_{\lambda,k},\delta_{\eta,k})\in\mathcal{C};\
    \delta_{x,k},\delta_{u,k}\ \text{are free}\right\}.
\end{aligned}
```

C3+ then applies the same ADMM scaffold to the augmented variables:

```math
\begin{aligned}
\bar z^{i+1}
  &=\underset{\bar z\in\mathcal{D}_+}{\mathrm{arg\,min}}
    \left[J(\bar z)+\frac{1}{2}\sum_{k=0}^{N-1}
    \left\|\bar z_k-\bar\delta_k^i+\bar\omega_k^i\right\|_{G_k}^2\right], \\
\bar\delta_k^{i+1}
  &=\Pi_{\mathcal{H}_{+,k}}^{U_k}
    \!\left(\bar z_k^{i+1}+\bar\omega_k^i\right), \\
\bar\omega_k^{i+1}
  &=\bar\omega_k^i+\bar z_k^{i+1}-\bar\delta_k^{i+1}.
\end{aligned}
```

**Closed-form C3+ projection.** Let
$(\lambda_j^\circ,\eta_j^\circ)$ be component $j$ of
$\bar z_k^{i+1}+\bar\omega_k^i$, and define the weight ratio
$r_j:=\sqrt{u_{\lambda,j}/u_{\eta,j}}$. Bui's componentwise projection is

```math
(\delta_{\lambda,j},\delta_{\eta,j})=
\begin{cases}
(0,\eta_j^\circ),
  & \eta_j^\circ\geq0\ \text{and}\
    \eta_j^\circ\geq r_j\lambda_j^\circ,\\
(\lambda_j^\circ,0),
  & \lambda_j^\circ\geq0\ \text{and}\
    \eta_j^\circ<r_j\lambda_j^\circ,\\
(0,0), & \text{otherwise}.
\end{cases}
```

The $x$ and $u$ components pass through the projection unchanged. Thus C3+
preserves C3's horizon-wide convex QP, consensus update, and dual ascent, but
replaces the full per-step contact projection with independent scalar
$(\lambda_j,\eta_j)$ case splits. The projected copy satisfies
$0\leq\delta_\lambda\perp\delta_\eta\geq0$ exactly; the QP copy approaches it
through consensus.

`control/admm_solver.py` implements this augmented loop. Its $G$ and $U$
weights come from the task YAMLs, and its final-QP polish uses the selected
contact branch to extract the trajectory. In receding-horizon operation, only
the first execution interval is applied before the state and local LCS are
rebuilt.

### The Anitescu contact formulation

The LCS above needs a discrete-time contact model to define what $\lambda$ *is*.
The port implements both standard time-stepping formulations in
`control/lcs_formulator.py`, defaulting to Anitescu to match the reference
(`c3/multibody/lcs_factory.cc`, `FormulateAnitescuContactDynamics`).

**The baseline: Stewart–Trinkle.** The exact time-stepping model
(Stewart & Trinkle, 1996) keeps three variable groups per contact — the
normal force $\lambda_n$, the four friction-pyramid edge forces $\lambda_t$,
and a slack $\gamma$ that converges to the sliding speed — coupled by three
complementarity conditions on the post-step velocity $v^+$:

```math
\begin{aligned}
0 &\leq \lambda_n \perp
  \frac{\phi}{\Delta t}+J_n v^+ \geq 0,
  && \text{(nonpenetration)}, \\
0 &\leq \lambda_t \perp
  E_t^\top\gamma+J_t v^+ \geq 0,
  && \text{(maximum dissipation)}, \\
0 &\leq \gamma \perp
  \mathrm{diag}(\boldsymbol\mu)\lambda_n-E_t\lambda_t \geq 0,
  && \text{(friction-pyramid bound)}.
\end{aligned}
```

Here $\lambda_n,\gamma,\phi,\boldsymbol\mu\in\mathbb{R}^{n_c}$,
$\lambda_t\in\mathbb{R}^{4n_c}$, and
$E_t\in\mathbb{R}^{n_c\times4n_c}$ sums the four pyramid-edge components at
each contact. This is physically exact for the pyramid cone, but the third
row couples the force variables to each other, giving $6n_c$ complementarity variables
(with 4 pyramid edges) and a harder projection.

**The Anitescu relaxation.** Anitescu's convex formulation (Anitescu,
*Optimization-based simulation of nonsmooth multibody dynamics*, Math.
Program. 105, 2006) folds the normal direction *into* each friction edge.
One combined Jacobian replaces the three groups. Replicate the per-contact
friction coefficients across their four pyramid edges as
$\bar{\boldsymbol\mu}:=E_t^\top\boldsymbol\mu\in\mathbb{R}^{4n_c}$. Then

```math
\begin{aligned}
J_c
  &:= E_t^\top J_n
      +\mathrm{diag}(\bar{\boldsymbol\mu})J_t
      \in\mathbb{R}^{4n_c\times n_v}, \\
0 &\leq \lambda \perp
  \frac{E_t^\top\phi}{\Delta t}+J_c v^+ \geq 0,
  \qquad \lambda\in\mathbb{R}^{4n_c}.
\end{aligned}
```

Each $\lambda_j$ is now a force along a *cone edge* (normal tilted by $\mu$
into a tangent direction); the normal component is $E_t\lambda$, while the
tangential edge coefficients are
$\mathrm{diag}(\bar{\boldsymbol\mu})\lambda$. The Coulomb cone is
satisfied by construction — no third complementarity row and no $\gamma$.

**What is gained and what is given up.** The gain: the per-contact
conditions admit a convex time-stepping subproblem, the variable count drops
to $4n_c$, and the LCS blocks $(D,E,F,H,c)$ take the compact folded form
shown in the Franka section (Step 4), with
$F=\Delta t\,J_cM^{-1}J_c^\top$ the standard Delassus operator. The
cost: a known relaxation artifact — during sliding, the folded constraint
introduces a small normal "boost" proportional to the slip speed, so a
sliding object can gain $O(\mu\,\Delta t\,\lVert v_t\rVert)$ of separation
per step (the boundary-layer effect). At this port's planning $\Delta t$ and
push speeds the artifact is well below the goal tolerances, and — decisively for
conformance — the reference stack plans with the same model.

**In the code.** `lcs_formulator.py` builds the Stewart–Trinkle blocks
first ($\gamma$/$\lambda_n$/$\lambda_t$ rows) and, when
`_contact_model == "anitescu"`
(the default, matching the reference), overwrites `D, E, F, H, c` with the
folded formulation (`lcs_formulator.py:1692-1698`); the Stewart–Trinkle
path is preserved behind `_contact_model == "stewart_trinkle"` for
falsification. The per-pair-type friction map (`mu_per_pair_type`) enters
through $\mathrm{diag}(\bar{\boldsymbol\mu})$ in $J_c$.

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

Define the object configuration and spatial velocity as

```math
q_O:=\begin{bmatrix}q_{WO}\\{}^W\!p_O\end{bmatrix}\in\mathbb{R}^7,
\qquad
v_O:=\begin{bmatrix}{}^W\!\omega_O\\{}^W\!v_O^{\mathrm{lin}}\end{bmatrix}
\in\mathbb{R}^6.
```

The Python port stores the reduced state in object-first order:

```math
x:=
\begin{bmatrix}
q_O \\ {}^W\!p_{EE} \\ v_O \\ {}^W\!v_{EE}
\end{bmatrix}
\in\mathbb{R}^{7+3+6+3}
=\mathbb{R}^{19}.
```

Here ${}^W\!p_{EE}$ and ${}^W\!v_{EE}$ are the spherical pusher's position
and velocity in the world frame. This storage order matches
`LCSFormulator.BOX_Q_SLOT`, `P_EE_SLOT`, `BOX_V_SLOT`, and `V_EE_SLOT`; it
differs from the actor-first order used by the native C++ reference.

#### Step 2 — how the input is defined

The input is the **Cartesian force applied at the pusher**, not joint
torques:

```math
u_k\in\mathcal{U}:=
\left\{u\in\mathbb{R}^3:\lVert u\rVert_\infty\leq F_{\max}\right\},
\qquad [u_k]=\mathrm{N}.
```

The bound $F_{\max}$ is task-specific; in this formulation the configuration
key `torque_limit` is interpreted in newtons.

This is the key abstraction of the reduced formulation: the planner asks
"what force should the ball at the fingertip exert," and the downstream OSC
QP is responsible for finding the seven joint torques that realize that
force on the real arm. (The legacy full-plant path behind `--r7` plans
joint torques $u\in\mathbb{R}^7$ directly; it is retained only for
falsification runs.)

#### Step 3 — the physics we start from

Drake evaluates the full arm/object/table plant in the standard form

```math
M_{\mathrm{plant}}(q)\dot v+h(q,v)
=\tau_g(q)+B_\tau\tau+J_n^\top\lambda_n+J_t^\top\lambda_t,
```

where $h$ collects Coriolis and centrifugal bias terms and
$\tau\in\mathbb{R}^7$ is joint torque. The reduced planner retains the
object dynamics and contact geometry but replaces the arm dynamics with the
spherical pusher's isotropic point-mass model:

```math
\mathcal{M}(r)\dot\nu+h_r(r,\nu)
=\tau_{g,r}(r)+B_u u+J_n^\top\lambda_n+J_t^\top\lambda_t,
\qquad
\mathcal{M}:=\mathrm{diag}(M_O,m_{EE}I_3).
```

Thus $u\in\mathbb{R}^3$ is Cartesian force in the planner, while the OSC
later maps the planned trajectory and force to the simulator's joint torques.

#### Step 4 — the matrices we infer each tick

The continuous dynamics above are nonlinear in $(q,v,u)$. Each planning
tick, `LCSFormulator` (`control/lcs_formulator.py`) linearizes them at the
measured state via Drake **autodiff** (Aydinoglu 2024, eq. 8):

For the reduced model, let
$r:=[q_O^\top\;({}^W\!p_{EE})^\top]^\top$ and
$\nu:=[v_O^\top\;({}^W\!v_{EE})^\top]^\top$. At the measured linearization
point $(r^\star,\nu^\star)$, the unconstrained generalized acceleration has
the affine model

```math
\begin{aligned}
f(r,\nu,u) &\approx J_r r+J_\nu\nu+J_u u+d_v, \\
J_r &:= \left.\frac{\partial f}{\partial r}\right|_{(r^\star,\nu^\star,0)},
&
J_\nu &:= \left.\frac{\partial f}{\partial \nu}\right|_{(r^\star,\nu^\star,0)}, \\
J_u &:= \left.\frac{\partial f}{\partial u}\right|_{(r^\star,\nu^\star,0)},
&
d_v &:= f(r^\star,\nu^\star,0)-J_r r^\star-J_\nu\nu^\star.
\end{aligned}
```

Because the reduced input channel is linear, this definition of $d_v$ makes
the affine model exact at the measured state for every $u$.

From the same plant context it extracts the **contact geometry**:

| Symbol | Shape | Meaning |
|---|---|---|
| $\phi$ | $\mathbb{R}^{n_c}$ | signed gap distance per contact pair (negative = penetrating) |
| $J_n$ | $\mathbb{R}^{n_c\times n_v}$ | normal contact Jacobian; $J_nv$ gives normal relative velocities |
| $J_t$ | $\mathbb{R}^{4n_c\times n_v}$ | tangential Jacobian, with four friction-pyramid edges per contact |
| $E_t$ | $\mathbb{R}^{n_c\times4n_c}$ | incidence matrix that sums the four edge components at each contact |
| $\boldsymbol\mu$ | $\mathbb{R}^{n_c}$ | per-contact friction coefficients (a scalar configuration value is broadcast) |

The contact pairs are the pusher-vs-object faces plus the object-vs-ground
witness points, so `n_c` changes with the sampled candidate and the object's
pose — these matrices are re-inferred at every tick and for every candidate.

The reduced configuration-rate and velocity maps are

```math
\dot r=\mathcal{N}_{r\nu}\nu,
\qquad
\mathcal{N}_{r\nu}:=\mathrm{diag}(N_O,I_3),
\qquad
\nu=\mathcal{N}_{\nu r}\dot r.
```

The corresponding reduced inertia is

```math
\mathcal{M}:=\mathrm{diag}(M_O,m_{EE}I_3).
```

Here, `M_O` is the object's spatial inertia and `m_EE` is the isotropic point
mass assigned to the pusher in the planning model. All quantities below are
evaluated at the current linearization point; asterisk superscripts are
suppressed for readability.

Under the default Anitescu contact model, friction is folded into one
combined contact Jacobian and the reduced-coordinate LCS blocks are assembled
in `linearize_discrete_ee_space` (`control/lcs_formulator.py:2608-2697`).
Partitioning the folded Jacobian by object and end-effector velocity gives

```math
J_c:=E_t^\top J_n
  +\mathrm{diag}(E_t^\top\boldsymbol\mu)J_t
  =\begin{bmatrix}J_{c,O} & J_{c,EE}\end{bmatrix}.
```

```math
\begin{aligned}
x_{k+1} &= Ax_k+Bu_k+D\lambda_k+d, \\
0 &\leq\lambda_k\perp Ex_k+F\lambda_k+Hu_k+c\geq0,
\end{aligned}
```

with

```math
\begin{aligned}
D &=
\begin{bmatrix}
\Delta t^2\mathcal{N}_{r\nu}\mathcal{M}^{-1}J_c^\top \\
\Delta t\,\mathcal{M}^{-1}J_c^\top
\end{bmatrix}, \\
E &=
\begin{bmatrix}
\Delta t\,J_cJ_r
+\dfrac{1}{\Delta t}E_t^\top J_n\mathcal{N}_{\nu r}
&
J_c+\Delta t\,J_cJ_\nu
\end{bmatrix}, \\
F &= \Delta t\,J_c\mathcal{M}^{-1}J_c^\top, \\
H &= \Delta t\,J_cJ_u, \\
c &= \frac{E_t^\top\phi}{\Delta t}
   +\Delta t\,J_cd_v
   -\frac{E_t^\top J_n\mathcal{N}_{\nu r}r^\star}{\Delta t}.
\end{aligned}
```

Reading the complementarity row as physics: $Ex+F\lambda+Hu+c$ predicts
the post-step contact velocity/gap of each friction-pyramid direction;
$0\leq\lambda\perp(\cdot)\geq0$ says a contact force may only push (never
pull) and only while its gap is closed. $A$, $B$, and $d$ are the corresponding
discrete-time integration of $J_f$ and $d_v$.

#### Step 5 — what the solver minimizes over that model

C3+ then solves, exactly as in the xArm section, the quadratic tracking
problem
$\sum_k\lVert x_k-x_d\rVert_Q^2+\sum_k\lVert u_k\rVert_R^2$ over the LCS,
with ADMM consensus/projection weights $G$ and $U$ on the
$(\lambda,\eta)$ copies; the
weights come from the per-task YAML (`config/sampling_c3_kik_t.yaml` and
friends), including the final-QP contact boost on the last polish solve.

#### Step 6 — realizing `u` on the seven-joint arm

The OSC (`control/osc/operational_space_controller.py`) closes the gap
between the planner's fiction (a free-flying force ball) and the real arm:
it tracks the planned pusher trajectory and promotes the planner's contact
force to its QP, producing joint torques $\tau\in\mathbb{R}^7$ through the same
inverse-dynamics structure as the xArm OSC. Two Franka-specific terms
matter: the reference `joint2` posture pin (`Kp/Kd/W_joint2`,
`joint2_target_rad = 1.1`) that kills the null-space orbit in the endgame,
and `q_init_franka` seeding. The planner's $\lambda$ is a *cost demand*, not a
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
   *Input:* commanded joint torques $\tau\in\mathbb{R}^6$.
   *Output:* measured robot state $(q,\dot q)\in\mathbb{R}^6\times\mathbb{R}^6$
   at 500 Hz and the object's spatial pose/velocity
   $({}^W\!p_O,q_{WO},{}^W\!v_O,{}^W\!\omega_O)$ at 20 Hz.
   *Equation:* Drake's rigid-body dynamics with hydroelastic/point contact —
   $M(q)\ddot q+C(q,\dot q)=\tau+\tau_g+J_c^\top f_c$.

2. **State reduction (inside `xarm6_sampling_c3_controller`).**
   *Input:* the measured six-joint arm state and object spatial state.
   *Output:* the reduced planning state
   $x:=\mathrm{col}({}^W\!p_P,q_{WO},{}^W\!p_O,{}^W\!v_P,
   {}^W\!\omega_O,{}^W\!v_O)\in\mathbb{R}^{19}$, where the arm is
   collapsed to its stick-tip point ${}^W\!p_P$ via forward kinematics. The six
   joints never enter the optimization.

3. **Contact sampling.**
   *Input:* the reduced state `x` and the T's exact two-box boundary.
   *Output:* a set of candidate pusher placements — points on the object
   perimeter with outward face normals, lifted to world coordinates at
   sampling height.

4. **LCS linearization (per candidate).**
   *Input:* one candidate pusher position and the current `x`.
   *Output:* a local Linear Complementarity System — matrices
   $(A,B,D,d,E,F,H,c)$ with $\lambda_k\in\mathbb{R}^{20}$ over a horizon
   $N=5$:

   ```math
   x_{k+1}=Ax_k+Bu_k+D\lambda_k+d,
   \qquad
   0\leq\lambda_k\perp Ex_k+F\lambda_k+Hu_k+c\geq0.
   ```

5. **C3+ solve (per candidate).**
   *Input:* the candidate's LCS, the goal-encoding desired state $x_d$, and
   the cost matrices $(Q,R,G,U)$.
   *Output:* an open-loop plan
   $\{x_k^\star,u_k^\star,\lambda_k^\star\}$ minimizing
   $\sum_k\lVert x_k-x_d\rVert_Q^2+\sum_k\lVert u_k\rVert_R^2$ by ADMM over
   consensus copies of $(\lambda,\eta)$.

6. **Rollout ranking and selection.**
   *Input:* every candidate's plan.
   *Output:* the single executed candidate — each plan is forward-simulated
   through its LCS and scored with the same quadratic error
   $\sum_{k=0}^{N-1}\lVert e_k\rVert_Q^2+\lVert e_N\rVert_Q^2$, where
   $e_k:=x_k-x_d$ (`dynamic_rollout_cost`); the minimum-cost candidate wins.

7. **`xarm6_sampling_c3_controller` output — the execution plan.**
   What the controller *gives* is not torques and not the raw C3 solution:
   it publishes one timestamped LCM trajectory
   (`lcmt_timestamped_saved_traj`) holding three time-aligned tracks sampled
   from the winning plan's first execution interval:

   - `end_effector_position_target` — tip position knots
     $p_{\mathrm{des}}(t)\in\mathbb{R}^3$, from the plan's state trajectory
     $x_k^\star$ (pushing) or from the
     collision-aware acquisition IK waypoints (repositioning);
   - `end_effector_stick_axis_target` — the commanded stick axis (vertical),
     the orientation reference;
   - `end_effector_force_target` — the feedforward Cartesian force
     $f^\star(t)\in\mathbb{R}^3$, which is the C3+ input solution
     $u_k^\star$ passed through
     one-to-one; this is how the planned contact force reaches execution.

   Only the first interval is executed before the loop replans from the
   measured state (receding horizon).

8. **`xarm6_osc_controller` — operational-space control (500 Hz).**
   *Input:* the three-track plan above and the measured $(q,\dot q)$ from the
   simulator.
   *What the OSC gives:* the six joint torques $\tau\in\mathbb{R}^6$ — it is
   the only block that talks to the motors. Each control tick solves DAIRLab's
   inverse-dynamics QP with decision variables $(\dot v,\tau,\lambda)$:

   ```math
   \begin{aligned}
   \underset{\dot v,\tau,\lambda}{\mathrm{minimize}}\quad
     & \sum_i
       \left\|\ddot y_i^{\mathrm{cmd}}-J_i\dot v-\dot J_i\dot q\right\|_{W_i}^2
       +\lVert\dot v\rVert_{W_{\mathrm{accel}}}^2 \\
   \text{subject to}\quad
     & M(q)\dot v+C(q,\dot q)
       =B\tau+\tau_g+J_{\mathrm{ext}}^\top f^\star, \\
     & -\tau_{\max}\leq\tau\leq\tau_{\max}.
   \end{aligned}
   ```

   with the commanded task accelerations from PD on each tracking objective:

   ```math
   \ddot y_i^{\mathrm{cmd}}
   =\ddot y_i^{\mathrm{des}}
    +k_p\bigl(y_i^{\mathrm{des}}-y_i\bigr)
    +k_d\bigl(\dot y_i^{\mathrm{des}}-\dot y_i\bigr).
   ```

   The tracking objectives $y_i$ are the tip position ($k_p=200$,
   $k_d=20$), the stick-axis orientation (weighted $0.35^2=0.1225$
   relative to translation, yaw component zeroed for the axisymmetric
   stick), and a 0.01-weight joint-posture regularizer;
   $W_{\mathrm{accel}}=10^{-7}I$ regularizes accelerations. The planner's
   `end_effector_force_target` $f^\star$ enters the dynamics constraint as an
   external force, so planned contact forces are actively pressed, not just
   implied by position error. The resulting torque passes through the source
   xArm velocity-servo bridge (gains 300/300/200/200/200/200, MuJoCo
   force-clamp ordering, joint-4 passive spring) before publication. These
   torques close the loop into block 1.

9. **Goal gate (each planning cycle).**
   *Input:* the measured object pose.
   *Equation measured:* $e_p=\lVert(x,y)-(x_g,y_g)\rVert_2$ and
   $e_\theta=\lvert\mathrm{wrap}(\theta-\theta_g)\rvert$ (see the task
   definition below). The run ends when
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

```math
g:=(x_g,y_g,\theta_g)=(0.381,-0.400,3.1416),
\qquad
x^o:=(x,y,\theta),
```

where $g$ is `object.goal_pose` and $x^o$ is the measured T pose, with yaw
extracted from its quaternion.

Success is a terminal tolerance check on both goal variables simultaneously
(`task.translation_tolerance`, `task.orientation_tolerance`):

```math
e_p:=\left\lVert
\begin{bmatrix}x\\y\end{bmatrix}
-\begin{bmatrix}x_g\\y_g\end{bmatrix}
\right\rVert_2<0.05\ \mathrm{m},
\qquad
e_\theta:=\left|\mathrm{wrap}(\theta-\theta_g)\right|
<0.10\ \mathrm{rad}.
```

#### Orientation error and the wrap function

The orientation error is computed in three steps
(`xarm6_full_sampling_c3plus.cc:92-107`):

1. **Yaw extraction.** The measured object quaternion
   $q_{WO}=(q_w,q_x,q_y,q_z)$ is normalized and reduced to its heading:

   ```math
   \theta=\mathrm{atan2}
   \!\left(2(q_wq_z+q_xq_y),\ 1-2(q_y^2+q_z^2)\right).
   ```

   This is the standard ZYX yaw formula; roll and pitch are ignored by the
   planar gate (a tilted or toppled T is caught by the separate settle check's
   tilt angle
   $\psi=\arccos\!\left((R_{WO}\hat z)^\top\hat z\right)$, not by $e_\theta$).

2. **Raw difference.** $\Delta\theta:=\theta-\theta_g$. This raw value is
   meaningless as a distance, because yaw lives on the circle $S^1$, not on
   the real line: $\theta$ and $\theta+2\pi$ are the same physical heading,
   so $\Delta\theta$ can be off by any multiple of $2\pi$ depending on which
   branch `atan2` returned.

3. **Wrapping.**

   ```math
   \mathrm{wrap}(\Delta\theta)
   :=\mathrm{atan2}\!\left(\sin\Delta\theta,\cos\Delta\theta\right)
   \in(-\pi,\pi].
   ```

   Feeding $\Delta\theta$ through sine and cosine erases every multiple of
   $2\pi$, and `atan2` rebuilds the unique representative in $(-\pi,\pi]$.
   The result is the **shortest signed arc** from $\theta_g$ to $\theta$ —
   the geodesic distance on the circle. It is exact (no branching or modulo
   edge cases), and its absolute value never exceeds $\pi$.

Why this matters for `open_table` specifically: the goal heading is
$\theta_g=3.1416\approx\pi$, which sits exactly on the `atan2` branch cut. A T
that has essentially reached the goal can be measured at $\theta=+3.10$ on
one tick and $\theta=-3.10$ on the next — the same physical pose, differing
only in branch. Without wrapping, the second measurement scores
$|\Delta\theta|=|-3.10-3.1416|\approx6.24\ \mathrm{rad}$, a catastrophic
false failure; wrapped, both score $e_\theta\approx0.04\ \mathrm{rad}$ and
correctly pass the $0.10\ \mathrm{rad}$ gate. The reported terminal
errors in the gate ledgers (e.g. $2.9\ \mathrm{rad}$) are therefore true
remaining rotations, never branch artifacts.

The same wrap is used everywhere a yaw difference is consumed: the terminal
gate, the per-cycle progress accounting
(`xarm6_full_sampling_c3plus.cc:836-846`), and the settle check's `yaw_delta`.
The task therefore requires roughly $0.8\ \mathrm{m}$ of translation plus a
genuine $\pi$ reorientation of the T.

### What we optimize

Each control cycle solves a finite-horizon contact-implicit MPC problem with
C3+ (ADMM over consensus copies) on a locally linearized Linear
Complementarity System. The state is $x\in\mathbb{R}^{19}$ (pusher position,
object quaternion, object position, then velocities), the input is
$u\in\mathbb{R}^3$ (Cartesian pusher force), contact forces are
$\lambda\in\mathbb{R}^{20}$, and the horizon is $N=5$:

```math
\begin{aligned}
\underset{\{x_k,u_k,\lambda_k\}}{\mathrm{minimize}}\quad
  & \sum_{k=0}^{N}\lVert x_k-x_d\rVert_Q^2
    +\sum_{k=0}^{N-1}\lVert u_k\rVert_R^2 \\
\text{subject to}\quad
  & x_{k+1}=Ax_k+Bu_k+D\lambda_k+d,
    && k=0,\ldots,N-1, \\
  & 0\leq\lambda_k\perp Ex_k+F\lambda_k+Hu_k+c\geq0,
    && k=0,\ldots,N-1, \\
  & x_0=x_{\mathrm{meas}}.
\end{aligned}
```

The desired state $x_d$ encodes the `open_table` goal variables directly: the
object-position slots hold $(x_g,y_g,h_{\mathrm{rest}})$ and the
object-quaternion slots hold $q(\theta_g)$, a pure yaw rotation built from
`goal_pose.z()` (`xarm6_full_sampling_c3plus.cc:1597-1605`). The cost
matrices are assembled in `RunSolveAtSampledPusher`
(`xarm6_full_sampling_c3plus.cc:1579-1596`):

```math
\begin{aligned}
Q
  &= 50\,\mathrm{diag}\!\left(
     \underbrace{0.01\,\mathbf{1}_3}_{\text{pusher position}},
     \underbrace{0.1\,\mathbf{1}_4}_{\text{object quaternion}},
     \underbrace{(200,200,120)}_{\text{object position}},
     \underbrace{5\,\mathbf{1}_3}_{\text{pusher velocity}},
     \underbrace{0.05\,\mathbf{1}_6}_{\text{object velocities}}
     \right), \\
R
  &= 0.01\,I_3.
\end{aligned}
```

so translation error is weighted at an effective 10,000 per m² on object x/y
and orientation enters through the quaternion-error terms. ADMM additionally
carries consensus and projection penalties
$G=0.01\,\mathrm{diag}(g_{\mathrm{ADMM}})$ and
$U=0.26\,\mathrm{diag}(u_{\mathrm{proj}})$ on the
$(\lambda,\eta)$ copies (weights 2/1 and 20/1); these enforce
the complementarity structure and are not part of the task objective.

Around that inner QP, three more objectives shape the behavior:

- **Sample ranking.** Candidate pusher placements are each solved and then
  scored by forward-simulating the plan through the LCS and accumulating the
  same quadratic error,
  $\sum_{k=0}^{N-1}\lVert e_k\rVert_Q^2$ plus a terminal term
  (`dynamic_rollout_cost`, `xarm6_full_sampling_c3plus.cc:1740-1834`); the
  minimum-cost candidate is executed. A separate
  `object_yaw_cost_weight: 50.0` biases goal/sample selection toward yaw
  progress.
- **Acquisition IK.** Repositioning to a sampled contact solves an inverse
  kinematics problem with a position constraint on the stick tip and the
  restored source tilt objective $w_{\mathrm{tilt}}(1-\cos\psi)$,
  $w_{\mathrm{tilt}}=80$, which keeps the stick vertical inside the feasibility band
  (`xarm6_sampling_c3_controller.cc:120`, `:391-400`).
- **OSC tracking.** The 500 Hz operational-space controller tracks the
  selected trajectory with Cartesian gains $k_p=200$, $k_d=20$ and a
  0.01-weight joint-posture regularizer.

In short: the *task* asks for $e_p<0.05\ \mathrm{m}$ and
$e_\theta<0.10\ \mathrm{rad}$ on the T's
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
