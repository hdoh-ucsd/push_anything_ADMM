"""paint_log_sidepanel.py — side-by-side Drake scene + scrolling log panel.

Same log-filter machinery as paint_log_scroll.py, but instead of
overlaying the log lines onto the scene the two are rendered in
separate panels of a wider composite frame: Drake scene on the left,
scrolling log panel on the right. That way the scene is uncovered and
the log panel is dedicated screen real estate.

Layout (defaults):
    [ Drake scene 1280×720 ] [ log panel 800×720 ]  →  composite 2080×720

Log panel contents (right side):
    Top: sticky milestone lines (accumulate as fired) —
         [CONSENSUS-BIND], [CONSENSUS-DEF], [ACHIEVED-FIXED-GOAL],
         [CROSSED-COST-THRESHOLD], [MATH.setup], [LCS-MU-PER-PAIR],
         [TASK], [C3+] filtered_solve_time, [CONSENSUS] i=0/1/2 blocks.
    Middle: rolling window of last N [STEP] / [GS] / [C3+] lines.
    Bottom: current-step readout + final [RESULT] (last 30 frames).

Usage:
    python3 tools/visualizer/paint_log_sidepanel.py \\
        --frames-dir results/<stem>_frames \\
        --log-path   results/<stem>.txt \\
        --output     results/<stem>_sidepanel.mp4 \\
        [--fps 30] [--panel-width 800]
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

_STICKY_TAGS = {
    "[ACHIEVED-FIXED-GOAL]",
    "[CROSSED-COST-THRESHOLD]",
    "[MATH.setup]",
    "[LCS-MU-PER-PAIR]",
    "[TASK]",
    "[C3+] filtered_solve_time",
    "[§7.70]",
    "[ENV]  init arm q",
    "[C3] PORT_USE_DRAKE_VIDEO_WRITER",
    "[CONSENSUS-BIND]",
    "[CONSENSUS-DEF]",
    "[CONSENSUS]",
}
_ROLLING_TAGS = {"[STEP]", "[GS]", "[C3+]"}
_RESULT_TAG = "[RESULT]"

_TAG_COLORS = {
    "[ACHIEVED-FIXED-GOAL]":   (100, 255, 100),
    "[CROSSED-COST-THRESHOLD]": (255, 200, 100),
    "[RESULT]":                (100, 255, 255),
    "[C3+]":                   (200, 200, 255),
    "[GS]":                    (255, 220, 180),
    "[STEP]":                  (220, 220, 220),
    "[MATH.setup]":            (255, 220, 100),
    "[LCS-MU-PER-PAIR]":       (255, 180, 255),
    "[TASK]":                  (255, 180, 255),
    "[CONSENSUS-BIND]":        (180, 255, 220),
    "[CONSENSUS-DEF]":         (180, 255, 220),
    "[CONSENSUS]":             (255, 255, 180),
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
    if not line.startswith("["):
        return None
    end = line.find("]")
    if end < 0:
        return None
    return line[:end + 1]


def _tag_color(tag: str) -> Tuple[int, int, int]:
    return _TAG_COLORS.get(tag, _DEFAULT_COLOR)


def _line_family(line: str) -> str:
    # Sticky prefixes FIRST: most sticky tags ([ACHIEVED-FIXED-GOAL],
    # [TASK], [CONSENSUS*], ...) also have colors, and the old
    # colors-first order classified them "hi" — a family the frame loop
    # only routes to the rolling window for [STEP]/[GS]/[C3+] lines, so
    # every milestone was silently dropped and the sticky section only
    # ever showed colorless boot lines.
    for sticky in _STICKY_TAGS:
        if line.startswith(sticky):
            return "sticky"
    tag = _tag_of(line) or ""
    if tag in _TAG_COLORS:
        return "hi"
    if tag in _ROLLING_TAGS:
        return "rolling"
    return "skip"


def parse_log_by_step(log_path: Path) -> Tuple[Dict[int, List[str]], List[str]]:
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


def _draw_lines(draw, x0, y0, lines, font, max_chars, line_h,
                pad=6):
    """Draw a list of lines starting at (x0, y0). Returns new y."""
    for line in lines:
        tag = _tag_of(line) or ""
        color = _tag_color(tag)
        draw.text((x0, y0), _shorten(line, max_chars),
                  font=font, fill=color)
        y0 += line_h
    return y0


def compose_frame(
    scene_img: Image.Image,
    step: int,
    step_info: Optional[dict],
    rolling_window: List[str],
    sticky_lines: List[str],
    font_tiny: ImageFont.FreeTypeFont,
    font_small: ImageFont.FreeTypeFont,
    panel_width: int,
    panel_max_chars: int,
    result_line: Optional[str],
) -> Image.Image:
    """Left = scene, Right = log panel. Returns composite RGB image."""
    scene = scene_img.convert("RGB")
    W_scene, H = scene.size
    W_out = W_scene + panel_width
    canvas = Image.new("RGB", (W_out, H), (0, 0, 0))
    canvas.paste(scene, (0, 0))
    draw = ImageDraw.Draw(canvas)

    # Panel vertical layout:
    #   [Header — step/t/mode   font_small]
    #   [--- sticky section ---]
    #   [--- rolling section ---]
    #   [--- RESULT section (last 30 frames) ---]
    panel_x0 = W_scene + 12
    line_h = 15
    pad = 8

    # Header
    if step_info is not None:
        hdr = [
            f"step={step}",
            f"t={step_info.get('t', 0.0):.2f}s",
            f"mode={step_info.get('mode', '?')}",
        ]
    else:
        hdr = [f"step={step}"]
    y = 12
    for line in hdr:
        draw.text((panel_x0, y), line, font=font_small,
                  fill=(200, 255, 200))
        y += 20
    y += pad

    # Separator
    draw.line((panel_x0, y, W_out - 12, y), fill=(60, 60, 60), width=1)
    y += pad

    # Sticky milestone lines. Cap the display: with the sticky-vs-hi
    # classification fixed, [CONSENSUS] setup blocks alone can run to ~18
    # lines and starve the rolling window. Milestones accumulate in fire
    # order, so keeping the LAST N keeps late events (ACHIEVED-FIXED-GOAL)
    # visible; an overflow note stands in for the elided early boot lines.
    _STICKY_SHOWN = 12
    draw.text((panel_x0, y), "── milestones (sticky) ──", font=font_tiny,
              fill=(150, 150, 150))
    y += line_h
    shown_sticky = sticky_lines
    if len(sticky_lines) > _STICKY_SHOWN:
        n_hidden = len(sticky_lines) - (_STICKY_SHOWN - 1)
        shown_sticky = ([f"(… +{n_hidden} earlier)"]
                        + sticky_lines[-(_STICKY_SHOWN - 1):])
    y = _draw_lines(draw, panel_x0, y, shown_sticky, font_tiny,
                    panel_max_chars, line_h)
    y += pad

    # Separator
    draw.line((panel_x0, y, W_out - 12, y), fill=(60, 60, 60), width=1)
    y += pad

    # Rolling window
    draw.text((panel_x0, y), "── recent (rolling) ──", font=font_tiny,
              fill=(150, 150, 150))
    y += line_h
    # Budget: compute how many rolling lines fit
    remaining_h = H - y - pad - (line_h * 3 if result_line else 0)
    max_rolling = max(1, remaining_h // line_h)
    _draw_lines(draw, panel_x0, y, rolling_window[-max_rolling:],
                font_tiny, panel_max_chars, line_h)

    # RESULT section (last 30 frames only)
    if result_line is not None:
        # Bottom
        result_h = font_tiny.getbbox(result_line[:panel_max_chars])[3] \
            - font_tiny.getbbox(result_line[:panel_max_chars])[1]
        y_res = H - line_h * 3
        draw.line((panel_x0, y_res, W_out - 12, y_res),
                  fill=(60, 60, 60), width=1)
        draw.text((panel_x0, y_res + pad), "── verdict ──",
                  font=font_tiny, fill=(150, 150, 150))
        draw.text((panel_x0, y_res + pad + line_h),
                  _shorten(result_line, panel_max_chars),
                  font=font_tiny, fill=_tag_color("[RESULT]"))

    return canvas


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
    ap.add_argument("--panel-width", type=int, default=800)
    ap.add_argument("--rolling-count", type=int, default=30)
    ap.add_argument("--panel-max-chars", type=int, default=95)
    args = ap.parse_args()

    print(f"[sidepanel] parsing {args.log_path}")
    by_step, header_lines = parse_log_by_step(args.log_path)
    print(f"[sidepanel] parsed {len(by_step)} step buckets, "
          f"{len(header_lines)} header lines")

    frames = find_frames(args.frames_dir)
    print(f"[sidepanel] found {len(frames)} frames")
    if not frames:
        raise SystemExit("no frames to annotate")

    font_tiny = _load_mono_font(11)
    font_small = _load_mono_font(14)

    # Sticky: seed with header first (fixes the same drop I hit in
    # paint_log_scroll.py — CONSENSUS-BIND fires before the first [STEP]).
    sticky_lines: List[str] = []
    for hl in header_lines:
        if _line_family(hl) == "sticky" and hl not in sticky_lines:
            sticky_lines.append(hl)
    rolling = deque(maxlen=args.rolling_count)
    result_line: Optional[str] = None
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
        for i, (step, path) in enumerate(frames):
            for s in range(prev_step + 1, step + 1):
                lines = by_step.get(s, [])
                for line in lines:
                    fam = _line_family(line)
                    if fam == "sticky":
                        if line not in sticky_lines:
                            sticky_lines.append(line)
                    elif fam in ("rolling", "hi") and (
                            line.startswith("[STEP]")
                            or line.startswith("[GS]")
                            or line.startswith("[C3+]")):
                        rolling.append(line)
                    m = _STEP_RE.match(line)
                    if m:
                        step_info_map[s] = {
                            "t": float(m.group(3)),
                            "mode": m.group(2),
                        }
            prev_step = step
            info = step_info_map.get(step)
            img = Image.open(path)
            out = compose_frame(img, step, info,
                                list(rolling), sticky_lines,
                                font_tiny, font_small,
                                args.panel_width, args.panel_max_chars,
                                result_line if i >= len(frames) - 30
                                else None)
            out.save(tmp_dir / f"annot_{i:06d}.png", optimize=False)
            if i % 200 == 0:
                print(f"[sidepanel] painted {i}/{len(frames)}")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y", "-framerate", str(args.fps),
            "-i", str(tmp_dir / "annot_%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "veryfast", "-crf", "23",
            str(args.output),
        ]
        print(f"[sidepanel] encoding → {args.output}")
        subprocess.check_call(cmd)
    print(f"[sidepanel] done → {args.output}")


if __name__ == "__main__":
    main()
