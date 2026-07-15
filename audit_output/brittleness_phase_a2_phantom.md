# Phase A2 — phantom-contact cross-tab

**Source:** `nondet_seed0_setarch_hashseed` | **window:** steps 105–250

## Per-run phantom duration vs basin outcome

| run | first_c3 | first_real_admit | phantom_duration (ticks/ms) | goal_dist | basin |
|---|---|---|---|---|---|
| 1 | 110 | — (none in window) | **140** / 1400ms | 0.0985m | GOOD |
| 2 | 111 | — (none in window) | **139** / 1390ms | 0.0779m | GOOD |
| 3 | 110 | — (none in window) | **140** / 1400ms | 0.1510m | BAD |
| 4 | 110 | — (none in window) | **140** / 1400ms | 0.2145m | BAD |

## Phantom-fraction in first 10 c3 ticks per run

| run | c3 ticks in window | phantom ticks | phantom fraction |
|---|---|---|---|
| 1 | 10 | 10 | 100% |
| 2 | 10 | 10 | 100% |
| 3 | 10 | 10 | 100% |
| 4 | 10 | 10 | 100% |

## Discriminator analysis

- GOOD runs phantom_duration: max = 140 ticks
- BAD runs phantom_duration: min = 140 ticks
- **Gap: 0 ticks** (OVERLAP)
