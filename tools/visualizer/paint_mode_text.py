"""
Paint the per-frame contact-mode indicator onto Drake VTK PNG frames and
encode to MP4.

Usage
-----
    python tools/visualizer/paint_mode_text.py \
        --frames-dir /tmp/claude-0/seed0_drake_frames \
        --output     results/seed0_q7b_drake_annotated.mp4 \
        [--fps 30]

Inputs
------
    <frames-dir>/frame_NNNNNN.png   # one per captured tick (stride applied)
    <frames-dir>/mode_timeline.csv  # step,sim_t,mode,switch  (written by main.py)

Overlay
-------
    Top-left corner: solid color dot + "CONTACT-RICH (C3)" or
    "CONTACT-FREE (REPOSITION)", monospace ~22 px on a 45% black pill,
    12 px inset. Mode is frame-synced by `step` (frame_NNNNNN.png filename
    byte-matches a row in mode_timeline.csv).
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


_FRAME_RE = re.compile(r"^frame_(\d+)\.png$")

_LABEL_C3   = "CONTACT-RICH (C3+)"   # (C3+) default; --solver-label overrides
_LABEL_FREE = "CONTACT-FREE (REPOSITION)"
_LABEL_NA   = "MODE: n/a"

_DOT_C3   = (60, 220, 120, 255)    # green
_DOT_FREE = (240, 180, 60, 255)    # amber
_DOT_NA   = (180, 180, 180, 255)

_TEXT_RGBA  = (255, 255, 255, 255)
_PILL_RGBA  = (0, 0, 0, 115)       # ~45% black

_INSET   = 12
_PAD_X   = 14
_PAD_Y   = 8
_DOT_R   = 7
_GAP     = 10
_FONT_PX = 22


def _load_mono_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def _load_mode_timeline(csv_path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    with open(csv_path) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                step = int(row["step"])
            except (KeyError, ValueError):
                continue
            mode = row.get("mode", "n/a") or "n/a"
            out[step] = mode
    return out


def _mode_to_label_and_dot(mode: str) -> tuple[str, tuple[int, int, int, int]]:
    m = mode.strip().lower()
    if m == "c3" or m == "rich":
        return _LABEL_C3, _DOT_C3
    if m == "free" or m == "repos" or m == "reposition":
        return _LABEL_FREE, _DOT_FREE
    return _LABEL_NA, _DOT_NA


def _paint_one(src: Path, dst: Path, mode: str,
               font: ImageFont.FreeTypeFont) -> None:
    label, dot_rgba = _mode_to_label_and_dot(mode)
    img = Image.open(src).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    pill_w = _PAD_X + (2 * _DOT_R) + _GAP + text_w + _PAD_X
    pill_h = max(_PAD_Y * 2 + text_h, _PAD_Y * 2 + 2 * _DOT_R)
    x0, y0 = _INSET, _INSET
    x1, y1 = x0 + pill_w, y0 + pill_h

    draw.rounded_rectangle([x0, y0, x1, y1], radius=10, fill=_PILL_RGBA)

    dot_cx = x0 + _PAD_X + _DOT_R
    dot_cy = y0 + pill_h // 2
    draw.ellipse(
        [dot_cx - _DOT_R, dot_cy - _DOT_R, dot_cx + _DOT_R, dot_cy + _DOT_R],
        fill=dot_rgba,
    )

    text_x = dot_cx + _DOT_R + _GAP
    text_y = y0 + (pill_h - text_h) // 2 - text_bbox[1]
    draw.text((text_x, text_y), label, font=font, fill=_TEXT_RGBA)

    composed = Image.alpha_composite(img, overlay).convert("RGB")
    composed.save(dst, format="PNG", optimize=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--frames-dir", type=Path, required=True,
                    help="Dir of frame_NNNNNN.png + mode_timeline.csv")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output MP4 path")
    ap.add_argument("--fps", type=float, default=30.0,
                    help="Output frame rate (default 30)")
    ap.add_argument("--keep-painted", action="store_true",
                    help="Keep the painted PNG intermediate dir (debug)")
    ap.add_argument("--solver-label", default="C3+",
                    help="Solver name in the contact-rich banner "
                         "(default C3+, main.py's default; pass C3 for "
                         "--solver c3 runs)")
    args = ap.parse_args()

    global _LABEL_C3
    _LABEL_C3 = f"CONTACT-RICH ({args.solver_label})"

    frames_dir = args.frames_dir.resolve()
    if not frames_dir.is_dir():
        print(f"[paint] frames dir not found: {frames_dir}", file=sys.stderr)
        return 2

    csv_path = frames_dir / "mode_timeline.csv"
    if not csv_path.is_file():
        print(f"[paint] mode_timeline.csv missing in {frames_dir}",
              file=sys.stderr)
        return 2

    mode_by_step = _load_mode_timeline(csv_path)
    print(f"[paint] mode_timeline.csv: {len(mode_by_step)} rows")

    frame_paths = sorted(p for p in frames_dir.iterdir()
                         if _FRAME_RE.match(p.name))
    if not frame_paths:
        print(f"[paint] no frame_*.png in {frames_dir}", file=sys.stderr)
        return 2
    print(f"[paint] frames: {len(frame_paths)} "
          f"(first={frame_paths[0].name}, last={frame_paths[-1].name})")

    font = _load_mono_font(_FONT_PX)

    painted_root = (frames_dir.parent /
                    f"{frames_dir.name}_painted")
    if args.keep_painted and painted_root.exists():
        shutil.rmtree(painted_root)
    painted_root.mkdir(parents=True, exist_ok=True)

    last_mode = "n/a"
    miss_count = 0
    c3_count = 0
    free_count = 0
    for fp in frame_paths:
        step = int(_FRAME_RE.match(fp.name).group(1))
        mode = mode_by_step.get(step)
        if mode is None:
            mode = last_mode
            miss_count += 1
        else:
            last_mode = mode
        if mode == "c3":
            c3_count += 1
        elif mode == "free":
            free_count += 1
        _paint_one(fp, painted_root / fp.name, mode, font)

    print(f"[paint] painted: c3={c3_count} free={free_count} "
          f"missing_mode_lookups={miss_count}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()

    ff_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(args.fps),
        "-pattern_type", "glob",
        "-i", str(painted_root / "frame_*.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-movflags", "+faststart",
        str(args.output),
    ]
    print(f"[paint] ffmpeg: {' '.join(ff_cmd)}")
    rc = subprocess.run(ff_cmd, check=False).returncode
    if rc != 0:
        print(f"[paint] ffmpeg rc={rc}", file=sys.stderr)
        return rc

    if not args.keep_painted:
        shutil.rmtree(painted_root, ignore_errors=True)

    print(f"[paint] OK -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
