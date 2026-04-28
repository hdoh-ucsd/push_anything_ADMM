# Push Anything (Drake Port) — Milestone Index

Historical snapshots of the Drake C3 port, each frozen at a natural stopping point.

## 2026-04-21_drake_port_infrastructure
Infrastructure verified (LCS extraction, polyhedral projection, pusher weld, 3-iter ADMM with single contact). Three layered issues remain: approach heuristic unreachable from default pose, IK target produces 2.7mm penetration, vanilla C3 cannot recover from greedy contact-mode selection. The third motivates C3+ (Bui et al. 2025) for the next milestone. See `2026-04-21_drake_port_infrastructure/RESULTS.md`.

## 2026-04-21_drake_port_directional_characterization

Infrastructure verified: LCS extraction, polyhedral projection, pusher weld, 3-iter ADMM with single-contact filter. 4-task directional sweep characterizes vanilla C3's capability envelope:
- Task 2 (east aligned): 34-35% progress, then stalls at stiction-limited fixed point (documented via 30s extended run)
- Tasks 1, 3 (orthogonal): contact-loss failure
- Task 4 (anti-aligned): catastrophic failure
Demonstrates both the greedy-contact-mode limitation AND the LCS-simulator ground-friction mismatch. Includes Meshcat HTML video replays of all 4 scenarios. See `2026-04-21_drake_port_directional_characterization/RESULTS.md`.
