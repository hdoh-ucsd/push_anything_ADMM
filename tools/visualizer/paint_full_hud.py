"""paint_full_hud.py — full per-frame HUD overlay for Drake VTK frames.

Reads a run.log for per-step diagnostics ([STEP], [ALL-SAMP],
[CONTACT-CHECK], [DRAKE-CONTACT], [GATE-CONTACT]), overlays a text HUD on
each captured PNG frame, and encodes to MP4 via ffmpeg.

Usage:
    python3 tools/visualizer/paint_full_hud.py \\
        --frames-dir /tmp/frames_run \\
        --log-path   /tmp/run.log \\
        --output     results/hud.mp4 \\
        [--fps 30]
"""
from __future__ import annotations
import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional

from PIL import Image, ImageDraw, ImageFont

_FRAME_RE = re.compile(r"^frame_(\d+)\.png$")

_STEP_COMMON_RE = re.compile(
    r"\[STEP\] step=(\d+) mode=(\w+) t=([\d.]+)s "
    r"ee=\(([+-][\d.]+),([+-][\d.]+),([+-][\d.]+)\) "
    r"obj=\(([+-][\d.]+),([+-][\d.]+),([+-][\d.]+)\) "
    r"goal_dist=([\d.]+)m switch=(\w+)"
)
_STEP_C3_EXTRA_RE = re.compile(
    r"lam_n=([+-]?[\d.]+) lam_t=([+-]?[\d.]+) contact=(\w)"
)
_ALL_SAMP_RE = re.compile(
    r"\[ALL-SAMP\] step=(\d+) mode=(\w+) reason=(\w+) "
    r"k_star=(\d+) best_other=(\S+) target=(\S+) "
    r"labels=\[([^\]]*)\] costs=\[([^\]]*)\]"
)
_CC_RE = re.compile(
    r"\[CONTACT-CHECK\] step=(\d+) .*?"
    r"drake_dist=\s*([+-][\d.]+)\s+delta_mm=\s*([+-][\d.]+)\s+consistent=(\w)"
)
_DC_RE = re.compile(
    r"\[DRAKE-CONTACT\] step=(\d+) n_pairs=(\d+) "
    r"ee_box_normal=([+-]?[\d.]+)"
)


def _load_mono_font(size: int) -> ImageFont.FreeTypeFont:
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    ):
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def parse_log(log_path: Path) -> Dict[int, dict]:
    data: Dict[int, dict] = {}
    with open(log_path, errors="replace") as f:
        for line in f:
            m = _STEP_COMMON_RE.search(line)
            if m:
                step = int(m.group(1))
                d = data.setdefault(step, {})
                d["mode"] = m.group(2)
                d["t"] = float(m.group(3))
                d["ee"] = (float(m.group(4)), float(m.group(5)), float(m.group(6)))
                d["obj"] = (float(m.group(7)), float(m.group(8)), float(m.group(9)))
                d["goal_dist"] = float(m.group(10))
                d["switch"] = m.group(11)
                m2 = _STEP_C3_EXTRA_RE.search(line)
                if m2:
                    d["lam_n"] = float(m2.group(1))
                    d["lam_t"] = float(m2.group(2))
                    d["contact"] = m2.group(3)
                continue
            m = _ALL_SAMP_RE.search(line)
            if m:
                step = int(m.group(1))
                d = data.setdefault(step, {})
                d["k_star"] = int(m.group(4))
                d["best_other"] = m.group(5)
                d["target"] = m.group(6)
                d["labels"] = m.group(7)
                d["costs"] = m.group(8)
                d["reason"] = m.group(3)
                continue
            m = _CC_RE.search(line)
            if m:
                step = int(m.group(1))
                d = data.setdefault(step, {})
                d["drake_dist_m"] = float(m.group(2))
                d["consistent"] = m.group(4)
                continue
            m = _DC_RE.search(line)
            if m:
                step = int(m.group(1))
                d = data.setdefault(step, {})
                d["drake_F_ee_box"] = float(m.group(3))
                d["drake_n_pairs"] = int(m.group(2))
    return data


def _mode_dot_color(mode: str):
    if mode == "c3":
        return (60, 220, 120, 255)   # green
    if mode == "free":
        return (240, 180, 60, 255)   # amber
    return (180, 180, 180, 255)


def paint_frame(img: Image.Image, step: int, info: dict, font, font_lg):
    draw = ImageDraw.Draw(img, "RGBA")

    mode = info.get("mode", "?")
    t = info.get("t", 0.0)
    switch = info.get("switch", "?")

    # ---- HUD lines (left column) ---------------------------------------
    lines: list[str] = [
        f"step={step:4d}   t={t:5.2f}s",
        f"mode={mode:<5s} switch={switch}",
    ]
    ee = info.get("ee")
    obj = info.get("obj")
    if ee is not None:
        lines.append(f"ee   = ({ee[0]:+.3f}, {ee[1]:+.3f}, {ee[2]:+.3f})")
    if obj is not None:
        lines.append(f"box  = ({obj[0]:+.3f}, {obj[1]:+.3f}, {obj[2]:+.3f})")
    gd = info.get("goal_dist")
    if gd is not None:
        lines.append(f"goal_dist = {gd:.3f} m")
    dr = info.get("drake_dist_m")
    if dr is not None:
        cflag = info.get("consistent", "?")
        lines.append(f"gap(drake) = {1000*dr:+7.3f} mm  [LCS ok={cflag}]")
    F = info.get("drake_F_ee_box")
    if F is not None:
        lines.append(f"F(drake ee-box) = {F:6.2f} N")
    ln = info.get("lam_n")
    lt = info.get("lam_t")
    if ln is not None or lt is not None:
        lines.append(
            f"lam_n={ln if ln is not None else 0.0:+.3f}  "
            f"lam_t={lt if lt is not None else 0.0:+.3f}"
        )
    k = info.get("k_star")
    tgt = info.get("target")
    if k is not None:
        lines.append(f"k*={k}   target={tgt}   reason={info.get('reason','-')}")
    labels = info.get("labels")
    if labels:
        lines.append(f"samples: [{labels}]")
    costs = info.get("costs")
    if costs:
        parts = costs.split(",")
        short = ", ".join(f"{float(x):+.0f}" for x in parts[:6])
        if len(parts) > 6:
            short += ", ..."
        lines.append(f"costs:   [{short}]")

    # ---- draw ---------------------------------------------------------
    pad_x, pad_y = 12, 8
    line_h = 22
    widths = [draw.textlength(l, font=font) for l in lines]
    w = int(max(widths)) + 2 * pad_x
    h = line_h * len(lines) + 2 * pad_y
    x0, y0 = 12, 12
    draw.rectangle([x0, y0, x0 + w, y0 + h], fill=(0, 0, 0, 150))

    # mode dot
    dot_r = 7
    dot_cx = x0 + pad_x + dot_r
    dot_cy = y0 + pad_y + line_h - dot_r - 3
    draw.ellipse(
        [dot_cx - dot_r, dot_cy - dot_r, dot_cx + dot_r, dot_cy + dot_r],
        fill=_mode_dot_color(mode),
    )

    tx = x0 + pad_x
    ty = y0 + pad_y
    for i, l in enumerate(lines):
        # first line gets shifted right to sit next to the mode dot
        indent = (2 * dot_r + 8) if i == 1 else 0
        draw.text((tx + indent, ty + i * line_h), l,
                  fill=(255, 255, 255, 255), font=font)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True, type=Path)
    ap.add_argument("--log-path", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    print(f"[HUD] parsing {args.log_path}")
    data = parse_log(args.log_path)
    print(f"[HUD] parsed {len(data)} steps of diagnostic data")

    font = _load_mono_font(18)
    font_lg = _load_mono_font(24)

    pngs = sorted(
        args.frames_dir.glob("frame_*.png"),
        key=lambda p: int(_FRAME_RE.match(p.name).group(1)),
    )
    print(f"[HUD] found {len(pngs)} frames")
    if not pngs:
        raise SystemExit("no frames found")

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        for idx, src in enumerate(pngs):
            step = int(_FRAME_RE.match(src.name).group(1))
            info = data.get(step, {})
            img = Image.open(src).convert("RGBA")
            paint_frame(img, step, info, font, font_lg)
            # sequential renumber for ffmpeg's %06d pattern
            img.convert("RGB").save(td_p / f"annot_{idx:06d}.png")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        print(f"[HUD] encoding {args.output} @ {args.fps} fps")
        cmd = [
            "ffmpeg", "-y", "-framerate", str(args.fps),
            "-i", str(td_p / "annot_%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            str(args.output),
        ]
        subprocess.run(cmd, check=True)
    print(f"[HUD] done → {args.output}")


if __name__ == "__main__":
    main()
