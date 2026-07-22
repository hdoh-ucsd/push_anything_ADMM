"""paint_log_scroll.py — overlay the port's actual scrolling log lines on Drake frames.

Replaces the paint_paper_equations.py static equation overlay with the
port's real diagnostic output — a rolling window of the most recent log
lines emitted BEFORE the frame's simulation step. Shows the actual C3+
solver output, mode-switch decisions, contact events, and reference-
conformance parameter loads that produced the tight_goal PASS.

Filter set (in order of visual priority):
    [C3+]                 — per-tick solver output (|u|, λ_n, primal, iters)
    [GS]                  — mode-switch decision (best_k, cost, met_progress)
    [STEP]                — headline state (mode, t, ee, obj, goal_dist)
    [ACHIEVED-FIXED-GOAL] — the tight_goal latch moment (rare, sticky)
    [CROSSED-COST-THRESHOLD] — pose-regime switch (rare, sticky)
    [MATH.setup]          — friction coefficient, torque limit (once)
    [LCS-MU-PER-PAIR]     — per-pair μ setup diagnostic (once)
    [TASK]                — mu_per_pair_type override active (once)
    [RESULT]              — final verdict (very last)

Sticky lines (achieved-goal, crossed-threshold, once-only setup) stay in
the window; per-tick lines roll off after ~10 frames.

Usage:
    python3 tools/visualizer/paint_log_scroll.py \\
        --frames-dir results/<stem>_frames \\
        --log-path   results/<stem>.txt \\
        --output     results/<stem>_logscroll.mp4 \\
        [--fps 30] [--rolling-count 12]
"""
from __future__ import annotations
import argparse
import os
import re
import subprocess
import tempfile
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

_FRAME_RE = re.compile(r"^frame_(\d+)\.png$")
_STEP_RE = re.compile(r"^\[STEP\] step=(\d+) mode=(\w+) t=([\d.]+)s")

# Filter tags — only these get overlaid, in order of screen priority.
_STICKY_TAGS = {
    "[ACHIEVED-FIXED-GOAL]",
    "[CROSSED-COST-THRESHOLD]",
    "[MATH.setup]",
    "[LCS-MU-PER-PAIR]",
    "[TASK]",
    "[C3+] filtered_solve_time",
    "[§7.70]",
    "[ENV]  init arm q",
    "[C3] PUSHA_USE_DRAKE_VIDEO_WRITER",
    "[CONSENSUS-BIND]",
    "[CONSENSUS-DEF]",
    "[CONSENSUS]",
}
_ROLLING_TAGS = {
    "[STEP]",
    "[GS]",
    "[C3+]",
}
_RESULT_TAG = "[RESULT]"

# Colors by tag family.
_TAG_COLORS = {
    "[ACHIEVED-FIXED-GOAL]":   (100, 255, 100),   # bright green
    "[CROSSED-COST-THRESHOLD]": (255, 200, 100),  # amber
    "[RESULT]":                (100, 255, 255),   # cyan
    "[C3+]":                   (200, 200, 255),   # light blue
    "[GS]":                    (255, 220, 180),   # peach
    "[STEP]":                  (220, 220, 220),   # off-white
    "[MATH.setup]":            (255, 220, 100),   # yellow
    "[LCS-MU-PER-PAIR]":       (255, 180, 255),   # magenta
    "[TASK]":                  (255, 180, 255),   # magenta
    "[CONSENSUS-BIND]":        (180, 255, 220),   # aqua
    "[CONSENSUS-DEF]":         (180, 255, 220),   # aqua
    "[CONSENSUS]":             (255, 255, 180),   # pale yellow
}
_DEFAULT_COLOR = (200, 200, 200)


def _load_mono_font(size: int) -> ImageFont.FreeTypeFont:
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono.ttf",
    ):
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def _tag_of(line: str) -> Optional[str]:
    # Match a leading bracketed tag; strip trailing whitespace.
    if not line.startswith("["):
        return None
    end = line.find("]")
    if end < 0:
        return None
    return line[:end + 1]


def _tag_color(tag: str) -> Tuple[int, int, int]:
    return _TAG_COLORS.get(tag, _DEFAULT_COLOR)


def _line_family(line: str) -> str:
    tag = _tag_of(line) or ""
    if tag in _TAG_COLORS:
        return "hi"
    for sticky in _STICKY_TAGS:
        if line.startswith(sticky):
            return "sticky"
    if tag in _ROLLING_TAGS:
        return "rolling"
    return "skip"


def parse_log_by_step(log_path: Path) -> Tuple[Dict[int, List[str]], List[str]]:
    """Bucket log lines by the most recent [STEP] step= number seen.

    Returns:
        by_step: step -> list of log lines emitted BETWEEN this [STEP]
                 and the next.
        header_lines: log lines emitted before the first [STEP]
                      (setup diagnostics that fire once at boot).
    """
    by_step: Dict[int, List[str]] = {}
    header_lines: List[str] = []
    current_step: Optional[int] = None
    with open(log_path, errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = _STEP_RE.match(line)
            if m:
                current_step = int(m.group(1))
                by_step.setdefault(current_step, []).append(line)
                continue
            fam = _line_family(line)
            if fam == "skip":
                continue
            if current_step is None:
                header_lines.append(line)
            else:
                by_step.setdefault(current_step, []).append(line)
    return by_step, header_lines


def _shorten(line: str, max_chars: int) -> str:
    if len(line) <= max_chars:
        return line
    return line[:max_chars - 1] + "…"


def annotate_frame(
    img: Image.Image,
    step: int,
    step_info: Optional[dict],
    rolling_window: List[str],
    sticky_lines: List[str],
    header_lines: List[str],
    font_small: ImageFont.FreeTypeFont,
    font_tiny: ImageFont.FreeTypeFont,
    left_max_chars: int,
    result_line: Optional[str],
) -> Image.Image:
    canvas = img.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    W, H = canvas.size

    # ------------------------------------------------------------------
    # Top-left: rolling recent log lines (per-tick).
    # ------------------------------------------------------------------
    lines = list(rolling_window)
    x0, y0 = 12, 12
    pad = 6
    line_h = 16
    box_h = pad * 2 + line_h * max(len(lines), 1)
    draw.rectangle((x0 - pad, y0 - pad, x0 + left_max_chars * 9 + pad,
                    y0 + box_h - pad), fill=(0, 0, 0))
    y = y0
    for line in lines:
        tag = _tag_of(line) or ""
        color = _tag_color(tag)
        draw.text((x0, y), _shorten(line, left_max_chars),
                  font=font_tiny, fill=color)
        y += line_h

    # ------------------------------------------------------------------
    # Bottom-left: sticky milestone lines (setup + latch + final).
    # ------------------------------------------------------------------
    sticky_disp: List[str] = list(header_lines) + list(sticky_lines)
    if result_line is not None:
        sticky_disp.append(result_line)
    if sticky_disp:
        y_bot = H - (pad * 2 + line_h * len(sticky_disp)) - 12
        draw.rectangle((x0 - pad, y_bot - pad, x0 + left_max_chars * 9 + pad,
                        y_bot + line_h * len(sticky_disp) + pad),
                       fill=(0, 0, 0))
        y = y_bot
        for line in sticky_disp:
            tag = _tag_of(line) or ""
            color = _tag_color(tag)
            draw.text((x0, y), _shorten(line, left_max_chars),
                      font=font_tiny, fill=color)
            y += line_h

    # ------------------------------------------------------------------
    # Right side: current-step readout.
    # ------------------------------------------------------------------
    if step_info is not None:
        readout = [
            f"step={step}",
            f"t={step_info.get('t', 0.0):.2f}s",
            f"mode={step_info.get('mode', '?')}",
        ]
        _draw_readout(draw, (W - 280, 12), readout, font_small)

    return canvas


def _draw_readout(draw, xy, lines, font):
    x0, y0 = xy
    line_h = 20
    pad = 6
    W_est = max((font.getbbox(l)[2] for l in lines), default=0) + pad * 2
    draw.rectangle((x0 - pad, y0 - pad,
                    x0 + W_est - pad, y0 + line_h * len(lines) + pad),
                   fill=(0, 0, 0))
    y = y0
    for line in lines:
        draw.text((x0, y), line, font=font, fill=(200, 255, 200))
        y += line_h


def find_frames(frames_dir: Path):
    entries = []
    for p in sorted(frames_dir.iterdir()):
        m = _FRAME_RE.match(p.name)
        if m:
            entries.append((int(m.group(1)), p))
    return entries


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames-dir", required=True, type=Path)
    ap.add_argument("--log-path", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--rolling-count", type=int, default=10,
                    help="How many rolling log lines to display at top-left.")
    ap.add_argument("--left-max-chars", type=int, default=140,
                    help="Truncate log lines longer than this to fit.")
    args = ap.parse_args()

    print(f"[log-hud] parsing {args.log_path}")
    by_step, header_lines = parse_log_by_step(args.log_path)
    print(f"[log-hud] parsed {len(by_step)} step buckets, "
          f"{len(header_lines)} header lines")

    frames = find_frames(args.frames_dir)
    print(f"[log-hud] found {len(frames)} frames")
    if not frames:
        raise SystemExit("no frames to annotate")

    font_tiny = _load_mono_font(12)
    font_small = _load_mono_font(15)

    # Sticky lines: accumulated across the run, kept on-screen once triggered.
    sticky_lines: List[str] = []
    # Rolling window: last N per-tick lines.
    rolling = deque(maxlen=args.rolling_count)
    result_line: Optional[str] = None

    # First-pass: precompute the [RESULT] line so it can appear at the very
    # end regardless of where it lies in the log.
    with open(args.log_path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(_RESULT_TAG):
                result_line = line
                break

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        prev_step = -1
        step_info_map: Dict[int, dict] = {}
        # We need step_info to render the right-side readout; parse it from
        # the [STEP] lines lazily as we walk step buckets.
        for i, (step, path) in enumerate(frames):
            # Absorb lines from prev_step+1 up through step into buckets.
            for s in range(prev_step + 1, step + 1):
                lines = by_step.get(s, [])
                for line in lines:
                    fam = _line_family(line)
                    if fam == "sticky":
                        # Only add if not already present (prevents dup).
                        if line not in sticky_lines:
                            sticky_lines.append(line)
                    elif fam in ("rolling", "hi") and (line.startswith("[STEP]")
                                                       or line.startswith("[GS]")
                                                       or line.startswith("[C3+]")):
                        rolling.append(line)
                    # Parse [STEP] into step_info
                    m = _STEP_RE.match(line)
                    if m:
                        step_info_map[s] = {
                            "t": float(m.group(3)),
                            "mode": m.group(2),
                        }
            prev_step = step
            info = step_info_map.get(step)
            img = Image.open(path)
            # Merge header_lines into sticky_lines once (they contain
            # early-boot diagnostics like [CONSENSUS-BIND] / [CONSENSUS-DEF]
            # and per-iter [CONSENSUS] blocks that fire BEFORE the first
            # [STEP] line, so they never make it into the by_step buckets).
            if i == 0:
                for hl in header_lines:
                    fam = _line_family(hl)
                    if fam == "sticky" and hl not in sticky_lines:
                        sticky_lines.append(hl)
            out = annotate_frame(img, step, info,
                                 list(rolling),
                                 sticky_lines,
                                 [],  # header already merged above
                                 font_small, font_tiny,
                                 args.left_max_chars,
                                 # Only show RESULT on the last few frames.
                                 result_line if i >= len(frames) - 30 else None)
            out.save(tmp_dir / f"annot_{i:06d}.png", optimize=False)
            if i % 200 == 0:
                print(f"[log-hud] painted {i}/{len(frames)}")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y", "-framerate", str(args.fps),
            "-i", str(tmp_dir / "annot_%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "veryfast", "-crf", "23",
            str(args.output),
        ]
        print(f"[log-hud] encoding → {args.output}")
        subprocess.check_call(cmd)
    print(f"[log-hud] done → {args.output}")


if __name__ == "__main__":
    main()
