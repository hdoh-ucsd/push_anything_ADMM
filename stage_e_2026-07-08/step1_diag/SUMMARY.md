# Stage E — Step 1: 4-quadrant motion decomposition (2026-07-08)

HEAD = `be45251` + uncommitted §9-leak-revert working tree (bit-identical to `b23fa82` on box path per user memory).

Invocation: `--task-id 4` (West), `--seed 0`, `--max-time 12`, `--admm-iter 25`, `--ee-space`, c3plus/lcp, `--sampling-c3 config/sampling_c3_kik*.yaml`, canonical §7.73 + support bundle env, `PUSHA_REPOSITION_PWL=1`. FT toggled via `PUSHA_DECOUPLE_RECONCILE_FORCE_TRACKING`. `pwl_speed` toggled by yaml overlay (gentle=`sampling_c3_kik.yaml` 0.18, fast=`sampling_c3_kik_fast.yaml` 0.40).

## 4-quadrant results

| Cell                       | goal_motion | free-motion | c3-motion | \|qy\|_max | \|qz\|_max | EE-BOX admit % | wall  | PASS |
|----------------------------|------------:|------------:|----------:|-----------:|-----------:|---------------:|------:|:----:|
| 1a  FT-OFF + gentle (0.18) |   259.6 mm  |   108.7 mm  |  208.7 mm |     0.711  |     0.655  |         57.5 % | 27:40 |  NO  |
| **1b  FT-ON  + gentle (DECISIVE)** | **180.1 mm**  |    **32.1 mm**  |  **246.3 mm** |     **0.709**  |     **0.670**  |         **29.1 %** | 25:28 | **NO** |
| 1c  FT-OFF + fast   (0.40) |   167.6 mm  |   197.0 mm  |  116.7 mm |     0.185  |     0.187  |         34.4 % | 26:14 |  NO  |
| 1d  FT-ON  + fast   (0.40) |    96.0 mm  |   162.1 mm  |  431.7 mm |     0.710  |     0.711  |         36.2 % | 29:51 |  NO  |

Bar per cell (Stage E cumulative bar): `|qy|<0.10 AND |qz|<0.10 AND EE-BOX≥60% AND goal_motion≥20 mm`.

## Read

- **DECISIVE cell (1b) FAILS the Stage E bar.** Motion clears (18 cm), but `|qy|=0.71`, `|qz|=0.67` = ~90°/84° tumble (7× guard); admit only 29 %. The box is being over-driven (§7.31 hammer-blow pattern) — closure is not clean.
- **Motion source is NOT purely brush.** FT-ON gentle (1b) reduces free-mode motion 109 → 32 mm relative to FT-OFF gentle (1a) — consistent with FT-ON adding c3-phase push. But it doesn't zero the free-mode motion, and it doesn't reduce tumble.
- **Gentle vs fast tumble split:** Only FT-OFF + fast (1c) shows low tilt (|qy|=|qz|=0.19). That cell is dominated by free-mode brush (197 mm free, 117 mm c3) — the box slides via brush impact rather than sustained hammer, so it doesn't spin up. All FT-ON cells tumble to |q|=0.71 regardless of speed.
- **EE-BOX admit is uniformly starved.** 29–57 % across all cells — never reaches the 60 % sustained-contact bar. The planner is producing bursts of force that dislodge the pair rather than track it.

## Gate

Per plan (Stage E entry): *"If the decisive cell fails → STOP + report; the cumulative bar is blocked."*

**→ Step 2 (multi-seed cumulative bar) is BLOCKED.**
**→ Step 3 (regression check + mark ALIGNMENT PHASE COMPLETE) is BLOCKED.**

The alignment phase cannot be closed from this substrate. The 75.5 % box-closure that motivated Stage E entry (per `project_S9_leaked_to_box_stage_e_blocked.md`) is confirmed at the seed-0 goal-dist level (goal_motion 180–260 mm here) but is accompanied by severe tumble and starved contact — the closure is the §7.31 over-drive artifact, not clean sustained tracking. Same signature the plan flagged in §7.60.

## Files

- `1{a,b,c,d}_*.log`, `.time`, `_metrics.json` — per-cell raw log, wall-time, extracted metrics
- `extract_metrics.py` (parent dir) — parser
- `smoke_2s.log`, `.time` — health-gate calibration (2 s FT-ON gentle) that first surfaced the tumble

## Wall-time budget

- Diagnostic: 4 × ~27 min = **~1 h 49 min real wall clock**
- Would-have-been Step 2 (skipped): 4 seeds × ≥2 replicates × 12 s = 8-12 runs × 28 min = 3.7–5.6 h
