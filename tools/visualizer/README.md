# Push-Anything Attribution Visualizer

## What this is

An offline visualizer for `push_anything_ADMM` simulation logs that distinguishes
**planned productive contact** from **accidental kinematic plowing**. Loads a JSONL
telemetry file and plays it back with:

- Top-down 2D view: EE, box, goal, trail colored by attribution per step.
- Right panel: mode (FREE/RICH), λ_n, contact state, cost gate, box motion.
- Bottom timeline: rich/free mode bands, λ_n trace, cumulative West progress.
- Playback controls: play/pause, scrub, speed 0.25×–8×.
- Optional Drake iframe slot for side-by-side with `verdict_west.html`.

## Attribution categories

Each simulation step is classified into one of four categories based on what's
actually happening:

- **Planned productive** (green): rich mode, λ_n_max > 0.5 N, contact normal aligned
  with goal direction. The architecture is actively pushing the box toward the goal.
- **Planned unproductive** (orange): rich mode active and contact present, but
  λ_n ≈ 0 OR contact normal misaligned. Typically means the LCS has only the
  vertical box-ground contact, not the lateral EE-box contact the planner needs.
- **Accidental contact** (red): free-mode IK tracking happens to bump the box.
  Not the architecture doing its job.
- **No contact / free-mode** (gray): EE tracking a target, no box contact.

The "Cumulative West progress" line at the bottom is the metric to watch:
**how much progress was earned during green segments vs gray ones?**

## Usage

1. Open `index.html` in a browser (no server needed — pure local HTML/JS).
2. Click the file picker, select a `.jsonl` file (e.g. `20260518_174809.jsonl`).
3. Hit Play, or use ← → keys to step, Space to play/pause, click timeline to seek.

## Generating JSONL from a log

```
python parse_log_to_jsonl.py <log_file.txt> [output.jsonl]
```

The parser reads `[GS]`, `[GS-tgt]`, `[C3+]`, `[CONTACT-RUN]`, `[EErel]`, and `[IMP]`
lines and produces one JSON object per simulation step. The first line of the JSONL
file contains `_meta` and `_summary` for the run; the rest are per-step records.

## Drake iframe (optional)

To view alongside Drake's `verdict_west.html`:
1. Check the "Drake iframe" box in the header.
2. Type the path to the Drake HTML file in the URL field (e.g.
   `file:///path/to/results/run/west/verdict_west.html` or a `http://` URL).
3. Drake's 3D scene loads in place of the 2D top-down view; the metrics panel
   stays on the right. Timeline below is shared but not time-synced to Drake's
   internal playback — use Drake's controls for the 3D view, ours for telemetry.

## What the 6 uploaded logs show (cross-check from JSONL parse)

| Run | Sim time | Total West progress | Productive contact | Accidental drift |
|---|---|---|---|---|
| 20260518_174809 ✓ video-verified | 2 s | 3.95 mm | **2.95 mm (75%)** | 1.0 mm |
| 20260518_180051 (1s test) | 1 s | 1.0 mm | 0 | 1.0 mm |
| 20260518_180358 (ground-fric) | 15 s | 1.95 mm | 0 (contact vertical only) | -1.0 mm |
| 20260518_192051 (random ring) | 15 s | 0 mm | 0 | 0 |
| 20260518_200856 (persistence) | 15 s | -10.1 mm East ✗ | 0 | -10.1 mm |
| 20260518_211137 (hyst80) | 15 s | 6.7 mm | 0 (contact vertical only) | 6.7 mm |

**174809 is the only run where the architecture earned its progress.** Every
other run's box motion came from kinematic drift in free mode, not planned
productive C3+ contact. Open 174809.jsonl in the visualizer to verify against
the video.

## Files

- `index.html` — the visualizer (open in browser).
- `parse_log_to_jsonl.py` — parser; converts text logs to JSONL.
- `20260518_174809.jsonl` — parsed video-verified baseline (the canonical anchor).
- `20260516_032821.jsonl` ... `20260518_211137.jsonl` — other parsed runs for comparison.
