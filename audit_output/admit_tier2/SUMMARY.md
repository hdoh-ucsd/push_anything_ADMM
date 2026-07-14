# LCS ADMISSION TIER-2 EVIDENCE (2026-07-14)

Diagnostic commit: `98f5e94` — `PUSHA_ADMIT_T2_DIAG=1`, env-gated, default-OFF byte-identical.

## Runs
- `run_default.log`: `PUSHA_ADMIT_T2_DIAG=1`, pushing --task-id 4, seed 0, admm-iter 25, c3plus, --sampling-c3 config/sampling_c3_kik.yaml, --max-time 2.0 (terminated by 240 s wall timeout at ~1.2 s sim time based on [ADMIT-T2] call=200 hitting step ~200).
- Full raw log: `run_default_full.log` (bulky; not committed).

## Runtime captures

### Init-time disclosure
```
[FILTER INIT] EE body: pusher  geom IDs: [<GeometryId value=223>]
[FILTER INIT] Manipuland geom IDs : 1
[FILTER INIT] Ground geom IDs     : 1  (world_body collision geoms)
[ADMIT-T2] contact_model='stewart_trinkle'   (reference default='anitescu')
[ADMIT-T2] mu=0.4                             (reference: mu_per_pair_type array)
[ADMIT-T2] box_ground_drag=10.0               (port-only viscous A-matrix mod; reference has no analog)
[ADMIT-T2] lcs_explicit_manipuland_ground_contacts=0  object_shape='box'
[ADMIT-T2] ref_pair_admission_planner_lcs=False       (port-only partial-ref flag; tshape-only opt-in)
[ADMIT-T2] always_on_ee_box=False             (env_LCS_ALWAYS_ON_EE_BOX=<unset>)
[ADMIT-T2] normal_compliance_k=0.0  normal_velocity_level=False  normal_phi_clamp=None
[ADMIT-T2] distance_threshold=0.002 m         (reference uses top-N ranking, no threshold)
```

### Per-tick admission (subsampled)
```
[ADMIT-T2] call=0   n_c=1 tags=['BOX-GND'] phi=['+0.0000'] n_ee_box=0 n_box_gnd=1
[ADMIT-T2] call=2   n_c=1 tags=['BOX-GND'] phi=['-0.0000'] n_ee_box=0 n_box_gnd=1
[ADMIT-T2] call=200 n_c=1 tags=['BOX-GND'] phi=['-0.0000'] n_ee_box=0 n_box_gnd=1
```
Port maintains n_c=1 (BOX-GND only) for the entire first 2 s sim window because the EE stays at z ≈ 0.2 m while the box top is at z = 0.10 m (gap ≈ 10 cm, far outside the 2 mm admission threshold). No EE-BOX pair admitted at these sampled ticks. `n_lambda_ST = 6·n_c = 6` per tick (Stewart-Trinkle: γ + λ_n + 4 λ_t = 6 per contact).

Reference-equivalent for this run: Anitescu with num_friction_directions=2 → `n_lambda_Anitescu = 2·2·n_c = 4·n_c` (single λ block, no γ / λ_n / λ_t split). For 1 admitted contact: **6 (port ST) vs 4 (reference Anitescu)** — different LCS dimension, different ADMM projection semantics.

### EE-BOX admission events later in run
```
[CONTACT-ELEM] step=417 element=manipulated_object::collision phi=+0.0018m
[CONTACT-ELEM] step=422 element=manipulated_object::collision phi=+0.0018m
[CONTACT-ELEM] step=423 element=manipulated_object::collision phi=-0.0000m
[CONTACT-ELEM] step=428 element=manipulated_object::collision phi=-0.0000m
[CONTACT-ELEM] step=429 element=manipulated_object::collision phi=+0.0002m
[CONTACT-ELEM] step=434 element=manipulated_object::collision phi=+0.0002m
```
EE-BOX admission begins at step 417 (this is `_diag_step_count`, ~call #417 of `extract_lcs_contacts`) with phi ≈ +1.8 mm — right at the edge of the 2 mm threshold. This is the exact scenario the LCS_ALWAYS_ON_EE_BOX flag was designed to bypass (per the source comment at `lcs_formulator.py:194-208`): "the port's filtered admission at 2 mm only lets the planner REACT once contact is already imminent."

## Reference-side reads (Tier-2 (a))

- **Contact model**: `examples/sampling_c3/anything/parameters/sampling_c3_options.yaml`: `contact_model: 'anitescu'`, `num_friction_directions: 2`. Same for `push_t`. **Reference default = Anitescu everywhere for sampling-c3 tasks.**
- **Pair-selection**: `sampling_based_c3_controller.cc:1580-1615` — `GetResolvedContactPairs` loops each pair-type group and calls `LCSFactory::GetNClosestContactPairs(plant, context, contact_geoms[i], num_to_select)`. Selects N-closest per group by phi (external c3 lib source at `github.com/DAIRLab/c3.git @ 5c08cb2e` — NOT LOCALLY ACCESSIBLE; clone denied). The exact ranking (does it ignore threshold entirely? use secondary tie-break?) is UNKNOWN without the c3 lib source.
- **Pair-list specification**: PRE-SPECIFIED at controller construction. `franka_sampling_c3_controller.cc:124-269` — for each demo, explicitly enumerates the pairs by geometry name (`EE`, `GROUND`, `VERTICAL_LINK`, `HORIZONTAL_LINK`, `TOP_LEFT_SPHERE`, etc.). Object-ground pairs use SPHERE bodies added to the manipuland URDF (`TOP_LEFT_SPHERE`, `TOP_RIGHT_SPHERE`, `BOTTOM_SPHERE` for push_t — three witness points).
- **Contact counts** (from YAML):
  - `anything`: `resolve_contacts_to_lists=[[0, 1, 3, 0, 2]]` → 0 EE-gnd + **1 EE-obj + 3 obj-gnd** + 0 obj-obj + 2 obj-wall = **6 pairs**.
  - `push_t`: `resolve_contacts_to_lists=[[0, 1, 3], [0, 2, 3]]` → planner uses [0]=`0 EE-gnd + 1 EE-T + 3 T-gnd = 4`; cost uses [1]=`0 + 2 + 3 = 5`.
- **Per-pair friction**: `mu_per_pair_type: [0.583, 0.42, 0.375, 0.3, 0.375]` for `anything` (harmonic mean per URDF surface pair). Reference uses **heterogeneous μ**; port uses a **single scalar** from `task_cfg["friction"]`.
- **Ground pair method**: reference builds `SortedPair(SPHERE, GROUND)` explicitly per witness sphere; port uses Drake `PairwiseClosestPoints` which returns a single (manipuland, ground) pair by default (WHICHEVER manipuland vertex the SDF picks as closest to the ground plane).

## Applying the 2.k caution — verifying ALL mechanisms in the "port-only" region

The port has ELEVEN admission-side flags/knobs. Verified each individually for whether it's a live mechanism or truly inert-by-default at runtime:

| Mechanism | Default | Runtime state | Live-by-default? |
|---|---|---|---|
| `distance_threshold=0.002` (hardcoded arg) | 0.002 m | 0.002 m | **YES** — sets the 2 mm admit window |
| `_contact_model` (env LCS_CONTACT_MODEL) | 'stewart_trinkle' | 'stewart_trinkle' | **YES** — sets ST vs Anitescu |
| `mu` (from task_cfg["friction"]) | 0.4 (pushing) | 0.4 | **YES** — friction coefficient |
| `_box_drag_c` (constructor default `box_ground_drag=10.0`) | 10.0 | 10.0 | **YES** — active viscous drag |
| `lcs_explicit_manipuland_ground_contacts` (env LCS_EXPLICIT_MANIPULAND_GND) | 0 | 0 | **inert** |
| `_object_shape` (constructor arg from task_cfg) | "box" | "box" | dispatch-only (sets vertex-set enumeration path if synthesis knob > 0) |
| `_ref_pair_admission_planner_lcs` (yaml `use_reference_pair_admission_planner_lcs`) | False | False | **inert** |
| `_always_on_ee_box` (env LCS_ALWAYS_ON_EE_BOX) | False | False | **inert** |
| `_normal_compliance_k` (env LCS_NORMAL_COMPLIANCE_K) | 0.0 | 0.0 | **inert** |
| `_normal_velocity_level` (env LCS_NORMAL_VELOCITY_LEVEL) | False | False | **inert** |
| `_normal_phi_clamp_v_cap` (env LCS_NORMAL_PHI_CLAMP) | None | None | **inert** |

Live-by-default: **distance_threshold, contact_model=ST, mu=0.4 (scalar), box_ground_drag=10.0**. These four are the load-bearing runtime state of admission. Inert-by-default: **7 mechanisms** — all opt-in flags. The **2.k caution passed**: no hidden live mechanism underneath the inert-by-default tag; each of the 7 was individually verified.
