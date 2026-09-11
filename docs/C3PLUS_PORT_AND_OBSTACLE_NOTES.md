# C3+ (Push-Anything) MJX 포팅 정리 — 원본 대비 구현/추가 + 장애물 회피 심층

**목적**: `Object-Informed-Manipulation-MJX` 레포의 C3+ 구현이 dairlib 원본(drake/C++)에서
무엇을 충실히 가져왔고, 그 위에 무엇을 추가했는지 한 곳에 정리한다. 특히 **장애물 회피**가
어떻게 구성되는지 — 코스트 항인지 제약인지, 파라미터가 무엇인지 — 를 drake/C++로 원본을
보는 사람에게 방향으로 전달하는 것이 초점이다.

핵심 파일: `oim/algs/c3_dynamic.py` (LCS 구성 + C3 outer loop), `oim/worlds/sim3d/build.py`
(컨트롤러 조립), `oim/worlds/sim3d/run.py` (xArm6 실행 루프).

원본 대응: `dairlib/systems/controllers/sampling_based_c3_controller.cc`,
`dairlib/examples/sampling_c3/{generate_samples,goal_generator,quaternion_error_hessian}.cc`,
`.../push_t/parameters/*.yaml`.

---

## 0. 한 줄 요약 (장애물 회피)

> **장애물 회피는 코스트 페널티가 아니라 LCS 비침투 접촉 제약이다.**
> dairlib은 벽(wall)을 plant의 충돌 geometry로 넣어 LCS 접촉쌍에 포함시키고, C3/ADMM이
> 그 상보성(complementarity) 제약을 만족하는 궤적을 자동으로 계획한다. 코스트에는 장애물 항이
> 전혀 없다. 우리 포팅은 이 원칙을 그대로 따르되, 벽 대신 **임의 형상의 SDF 장애물**을 매
> 제어주기마다 물체-장애물 마찰없는 법선 접촉으로 LCS에 주입하도록 일반화했다.

---

## 1. 원본에서 충실히 포팅한 것 (faithful ports)

| 메커니즘 | 원본 위치/파라미터 | 우리 구현 | 비고 |
|---|---|---|---|
| **Contact-implicit LCS** (Anitescu pusher + 지면 Coulomb 마찰) | `LCSFactory` | `build_dynamic_lcs()` | 상태 x=[obj(3), ee(2), vel(5)], pusher 2개 cone-edge + 지면 box friction |
| **C3+ (ADMM + η-slack projection)** | `C3` solver, `projection_type: C3+` | `c3_solve()` | rho, rho_scale, admm_iter 동일 개념 |
| **위치-우선 2단계 코스트 전환** | `cost_switching_threshold_distance=0.05` → `crossed_cost_switching_threshold_` | `crossed` 플래그, `q_theta_eff = where(crossed, q_theta, 0)` | 골 5cm 밖이면 방위 무시(위치만), 안이면 방위 켜짐 |
| **방위 의존 코스트** | `use_quaternion_dependent_cost`, `q_quaternion_dependent_weight=1000`, `hessian_of_squared_quaternion_angle_difference` | 평면 스칼라 yaw로 축약: `q_theta * Δθ²` | 3D 쿼터니온 → 평면 yaw 환원 (아래 §3.2) |
| **Lookahead sub-goal** | `goal_generator.cc: GenerateLineTrajectoryWithLookahead`, `lookahead_step_size=0.15`, `lookahead_angle=2.0`, `angle_hysteresis=0.4` | `step()` 내 sub-goal 계산 + 180° 히스테리시스 | C3는 최종 골이 아니라 현재로부터 ≤0.15m/≤2rad 앞선 sub-goal을 추종 |
| **Unsuccessful sample buffer** | `AddToUnsuccessfulBuffer`, `PruneOutdatedSamplesFromBuffer`, `unsuccessful_radius=0.02`, `N=20`, retention 0.006m/0.05rad | 핸드오프 뱅킹 + 비용배제 + object-move pruning | 죽은 접촉을 물체가 움직일 때까지 후보에서 제외 |
| **C3↔reposition 상대 히스테리시스** | `hyst_*_frac` (0.6/0.7/0.9/0.5/0.7/0.7) | 동일 값 이식 | position/pose 모드별 분리 |
| **무진전 판정** | `kConfigCostDrop`: 35 loop에 config cost 50%↓ | `progress_window=16`, `progress_drop=0.5` (근사) | 정의는 근사, 목적 동일 |
| **랜덤 둘레 접촉 샘플링** | `kRandomOnPerimeter` (push_t) / `sample_projection_clearance` | 경계 무작위 샘플 + shell clearance 0.027 | 물체 둘레에서 접촉 무작위 추출 |
| **장애물 회피 = LCS 접촉 (코스트 아님)** | walls as plant contact geometry (`include_walls`) | `_obs_contacts()` → `build_dynamic_lcs(obs=...)` | **§3에서 심층** |

---

## 2. 우리가 추가로 구현한 것 (additions on top)

원본은 7-DOF Franka + 3D 쿼터니온 물체를 전제한다. 우리는 **평면(2D) 물체 + 6-DOF xArm6**
환경이라 다음을 추가했다.

| 추가분 | 이유 | 위치 |
|---|---|---|
| **평면 환원** (yaw 스칼라 상태, 마찰없는 지면 지지) | sim3d 블록이 평면 구속(T_x,T_y,yaw) | `build_dynamic_lcs`, `_state_cost_hessian` |
| **operational-space(Khatib) 토크 실행 계층** | C3는 평면 EE 속도만 내고, 6-DOF 팔로 매핑 필요 | `run.py: _arm_osc_torque()` |
| **DLS 특이점 처리** (near-singular 방향만 감쇠) | 6-DOF 팔이 folded/stretched 자세에서 lock되는 것 방지 | `run.py`, `osc_dls_eps=0.02` |
| **리치 필터(annulus)** | xArm6 dexterous 한계 밖 접촉 배제 | `base_xy, reach_min, reach_max, reach_penalty` (dairlib `robot_radius_limits`의 xArm6판) |
| **리포지션 z-lift** | 큰 각도 orbit 시 팔이 납작해지는 것 방지 | `osc_repos_lift_z=0.12` 등 (dairlib `RepositionCircular circle_height` 유사) |
| **리포지션 ring march** | 평면 orbit (RepositionCircular의 평면 축약) | `_reposition_move()` |
| **SDF 기반 임의 장애물 접촉** | dairlib은 벽(box)만; 우리는 임의 형상 | `_obs_contacts()` — **§3** |
| **q_pos 12:1 튜닝** | dairlib 위치:방위 계수비(≈12:1) 복원 | `q_pos=1800, q_theta=150` (build.py) |

> 참고: 초기 버전에는 **소프트 장애물 코스트 항**(`w_obstacle`, `obstacle_decay`)이 있었으나,
> 원본에 그런 항이 없다는 걸 확인하고 **제거**했다(“drop obstacle cost” 패치). 로그 json의
> `costs.w_obstacle` 필드는 그 잔재로, 현재 코스트에는 반영되지 않는다.

---

## 3. 장애물 회피 심층 (핵심)

### 3.1 원칙: 코스트가 아니라 LCS 제약

C3+는 contact-implicit MPC다. 접촉(pusher-물체, 물체-지면)이 이미 LCS의 상보성 제약으로
들어가 있다. **장애물도 똑같이 "물체-장애물 마찰없는 법선 접촉"을 LCS에 한 줄 더 추가**하면,
C3/ADMM이 비침투를 만족하는 궤적을 자동으로 낸다. 별도의 회피 코스트가 필요 없다.

- dairlib: 벽을 `MultibodyPlant`의 충돌 geometry로 넣고, LCSFactory가 접촉쌍에서 LCS를
  구성 → 물체-벽 접촉이 상보성 변수로 자동 포함. 코스트(Q,R,quaternion)에는 장애물 항 없음.
- 우리: 벽 대신 임의 형상 SDF. 매 제어주기 `_obs_contacts()`가 물체 footprint에서 각 장애물
  까지의 최근접 접촉(부호거리·법선·레버암)을 계산해 `build_dynamic_lcs(obs=...)`로 주입.

### 3.2 코스트 구성 (장애물은 여기 없음)

평면 상태 `x = [obj_x, obj_y, obj_θ, ee_x, ee_y, (velocities)]`.

```
J(x,u) = Σ_k  (x_k − x_ref_sub)ᵀ Q (x_k − x_ref_sub)   # 상태 추종 (running)
       + (x_N − x_ref_sub)ᵀ Qf (x_N − x_ref_sub)        # 종단
       + Σ_k  uᵀ R u                                     # 입력(평면 EE 속도)
```

- `Q = diag(q_pos, q_pos, q_theta_eff, w_ee, w_ee, w_v...)`
  - `q_pos` : 물체 xy 추종 (=1800)
  - `q_theta_eff = crossed ? q_theta : 0` : 물체 yaw 추종 (=150, 골 5cm 안에서만 켜짐) ← 원본 quaternion cost의 평면판
  - `w_ee`(=10), `w_v`(=0.05) : EE 위치/속도 정규화
- `x_ref_sub` : **lookahead sub-goal** (최종 골 아님). 현재 물체 pose로부터 위치 ≤0.15m,
  yaw ≤2rad 앞선 지점. 큰 회전을 well-conditioned하게 만드는 원본 메커니즘.
- `R = r_r · I` (r_r=0.05) : 평면 EE 속도 입력 페널티.
- **장애물 항 없음.** 장애물은 아래 LCS 제약으로만 들어간다.

### 3.3 LCS에 장애물이 들어가는 방식

각 물체-장애물 접촉당 (마찰없는 법선 1개):

```
phi = sd(object_boundary, obstacle) − obs_margin      # 부호거리 − 안전마진
n   = ∇sd / |∇sd|                                     # world push-away 법선 (2,)
r   = contact_point − object_COM                      # 레버암 (2,)
Jo  = [ n_x, n_y, (r × n) ]      # 물체 평면속도 [ẋ,ẏ,θ̇] → 접근속도.  (5-DOF 상태에선 [n_x,n_y,r×n,0,0])
```

시간이산 비침투 상보성 (Stewart–Trinkle / Anitescu 형식):

```
0  ≤  λ_obs   ⊥   ( Jo · v_next  +  phi/dt )  ≥  0
```

즉 법선 임펄스 `λ_obs ≥ 0` 이고, 스텝 후 gap `Jo·v_next + phi/dt ≥ 0` 이며 둘은 상보. 물체가
장애물로 접근하면 `λ_obs`가 밀어내 침투를 막는다. 이 한 줄이 **지면·pusher 접촉과 완전히
동일한 형태**로 LCS에 추가될 뿐이다 (`build_dynamic_lcs`의 `obs_cols`, `11+o` 행 참조).

### 3.4 `_obs_contacts` 계산 (매 제어주기)

```python
# 물체 footprint 경계점들을 world로 변환 → 각 장애물 SDF의 최근접점 하나를 접촉으로
cw = object_xy + R(θ) · footprint_boundary        # (M,2)
for each obstacle s:
    d, grad = s.sdf_and_grad(cw)                  # (M,), (M,2)
    j = argmin(d)                                 # 가장 가까운 경계점
    phi = d[j] − obs_margin ;  n = grad[j]/|·| ;  r = cw[j] − object_xy
# N_closest 장애물만 선택 (top_k(-phi))  → 고정 크기 LCS
```

포인트: 접촉은 **물체의 footprint 코너 중 장애물에 가장 가까운 점**과 장애물 표면 사이에
잡힌다. `n_obstacles`(=2)개만 유지해 LCS 크기를 고정 → JAX vmap/jit 유지.

### 3.5 drake/C++ 친구에게 주는 방향

1. **회피를 코스트로 넣지 마라.** contact-implicit이라면 장애물은 제약이다. 소프트 페널티는
   튜닝 지옥 + 침투 허용을 낳는다. (우리도 초기에 `w_obstacle` 넣었다가 제거했다.)
2. **가장 쉬운 경로**: 장애물을 `MultibodyPlant`에 충돌 geometry(box/wall)로 추가하고,
   물체-장애물 geometry pair를 LCSFactory가 쓰는 contact set에 포함시키면 끝. C3가 알아서
   비침투 궤적을 낸다. dairlib `include_walls`가 정확히 이 경로다.
3. **임의/이동 장애물이 필요하면**: 매 loop drake의
   `ComputeSignedDistancePairwiseClosestPoints`(또는 `SignedDistanceToPoint`)로 물체-장애물
   최근접쌍의 (phi, n, witness point)을 뽑아, 마찰없는 법선 접촉 한 줄을 LCS에 주입하라.
   레버암 `r × n` 로 회전 커플링까지 정확히 들어간다. 이게 우리 `_obs_contacts` + LCS 주입이다.
4. **안전 여유**: `phi`에서 `obs_margin`(우리 0.01m)을 빼 접촉을 조금 일찍 활성화하면 수치
   침투를 흡수한다.
5. **개수 고정**: 장애물이 많으면 N_closest만 LCS에 넣어라(우리 top_k). LCS/솔버 크기를
   상수로 유지해야 배치/jit·GPU에서 안정적이다.
6. **샘플/리포지션 안전 필터는 별개**: dairlib `filter_samples_for_safety`(workspace_limits)는
   회피가 아니라, 리포지션 목표(EE 재배치 지점)가 벽/작업공간 밖으로 안 나가게 거르는
   별도 장치다. 회피(LCS)와 혼동하지 말 것.

---

## 4. 주요 파라미터 표

### 4.1 C3 / LCS 솔버
| 파라미터 | 값 | 출처 | 의미 |
|---|---|---|---|
| `horizon` (num_knots) | 10 | dairlib N | MPC 예측 지평 |
| `admm_iters` | 3 | dairlib admm_iter | ADMM 반복 |
| `rho`, `rho_scale`, `rho_u` | 0.1, 3.0, 1.0 | dairlib | ADMM 페널티/스케일 |
| gamma (MPC 할인) | 1.0 | dairlib | 할인 없음 |
| plant `mo, Io` | 2.0, 0.005 | 물체 질량/관성 | 평면 물체 |
| `wrench_limit` | task별 (예 [7.85,7.85,0.79]) | 한계면(limit surface) | 병진 대비 회전예산 ~10× 작음 |

### 4.2 코스트 (장애물 항 없음)
| 파라미터 | 값 | 출처 | 의미 |
|---|---|---|---|
| `q_pos` / `qf_pos` | 1800 / 2000 | 우리 튜닝(비율은 dairlib) | 물체 xy 추종 (running/terminal) |
| `q_theta` / `qf_theta` | 150 / 400 | 원본 quaternion cost 평면판 | 물체 yaw 추종 (crossed 시에만) |
| `w_ee`, `w_v` | 10, 0.05 | dairlib q_vector | EE 위치/속도 정규화 |
| `r_r` (R) | 0.05 | dairlib r_vector | 평면 EE 입력 페널티 |
| 위치:방위 계수비 | ≈ 12:1 | dairlib (12000:1000) | 회전 중 위치 유지 우선 |

### 4.3 코스트 전환 / lookahead
| 파라미터 | 값 | 출처 | 의미 |
|---|---|---|---|
| `cost_switching_threshold_distance` | 0.05 m | dairlib push_t | 이 밖이면 방위 무시(위치우선) |
| `look_step` | 0.15 m | dairlib lookahead_step_size | 위치 sub-goal 상한 |
| `look_angle` | 2.0 rad | dairlib lookahead_angle | 방위 sub-goal 상한 |
| `look_hyst` | 0.4 rad | dairlib angle_hysteresis | 180° 근처 회전방향 뒤집힘 방지 |
| `progress_window`, `progress_drop` | 16, 0.5 | dairlib kConfigCostDrop 근사 | 무진전 판정 |

### 4.4 샘플 버퍼 / 히스테리시스
| 파라미터 | 값 | 출처 | 의미 |
|---|---|---|---|
| `num_random` | 8 | dairlib num_additional_samples | loop당 무작위 접촉 수 |
| `shell_clearance` | 0.027 m | dairlib sample_projection_clearance | 표면 밖 standoff |
| `n_unsuccessful` / `unsucc_radius` | 20 / 0.02 m | dairlib | 실패 접촉 버퍼 크기/반경 |
| `unsucc_pos_ret` / `unsucc_ang_ret` | 0.006 m / 0.05 rad | dairlib | 물체 이동 시 버퍼 clear 임계 |
| `n_good` (N_sample_buffer) | 8 | dairlib 200의 축소판 | 양호 접촉 메모리 (원본 대비 얕음) |
| `hyst_*_frac` | 0.6/0.7/0.9/0.5/0.7/0.7 | dairlib | C3↔repos 상대 히스테리시스 |

### 4.5 장애물 (우리 SDF 일반화)
| 파라미터 | 값 | 출처 | 의미 |
|---|---|---|---|
| `n_obstacles` (N_closest) | 2 | 우리 | LCS에 넣을 최근접 장애물 수 (크기 고정) |
| `obs_margin` | 0.01 m | 우리 | 부호거리 안전 여유 (접촉 조기 활성) |
| 접촉 타입 | frictionless normal | dairlib 원칙 | 물체는 장애물 따라 미끄러지되 침투 불가 |
| 코스트 항 | **없음** | dairlib faithful | 회피는 LCS 제약으로만 |

### 4.6 리치 필터 / OSC 실행 (xArm6, 우리 추가)
| 파라미터 | 값 | 의미 |
|---|---|---|
| `reach_min` / `reach_max` | 0.25 / 0.68 m | 팔 dexterous annulus (dairlib robot_radius_limits의 xArm6판) |
| `reach_penalty` | 1e9 | 범위 밖 접촉 후보 배제 |
| `robot_radius` | stick geom에서 읽음 (~0.02) | C3 접촉모델의 pusher 반경 |
| `osc_kv_xy` | 20 | xy op-space 속도게인 (C3 속도 추종) |
| `osc_kp_z / kd_z / z_vmax` | 8 / 60 / 0.3 | tip 하강(접촉고) 제어 |
| `osc_kp_rot / kd_rot` | 100 / 20 | stick 수직 유지(tilt) |
| `osc_dls_eps` | 0.02 | 특이점 근처만 감쇠하는 DLS 비율 |
| `osc_repos_lift_z` | 0.12 m | 리포지션 orbit 중 tip 들어올림 |
| `osc_repos_descend_far/near` | 0.12 / 0.04 m | 접촉고로 되내리는 xy 거리 램프 |

---

## 5. 요약: 친구에게 전할 한 문장

> Contact-implicit MPC(C3/C3+)에서 장애물 회피는 **코스트가 아니라 LCS의 마찰없는 법선
> 비침투 접촉 한 줄**이다. drake라면 장애물을 plant 충돌 geometry로 넣어 LCSFactory 접촉쌍에
> 포함(가장 간단)하거나, 임의/이동 장애물은 매 loop `ComputeSignedDistancePairwiseClosestPoints`로
> (phi, n, witness)를 뽑아 `0 ≤ λ ⊥ (J·v_next + phi/dt)` 행을 LCS에 주입하면 된다. 레버암
> `r×n`으로 회전 커플링까지 자연히 들어가고, 안전여유(margin)와 N_closest 고정만 챙기면 된다.
