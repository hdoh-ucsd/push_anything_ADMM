"""paint_paper_equations.py — overlay paper equations on Drake VTK frames.

Reads a run.log for per-step state and overlays the Bui et al. 2026 paper's
key equations on each captured PNG frame — the outer C3+ objective (5a-e),
the ADMM iteration steps (7)-(9), the augmented Lagrangian QP (10), and the
componentwise projection case-analysis (12). Encodes to MP4 via ffmpeg.

Paper: Bui et al. "Sampling-Based Contact-Implicit MPC" (arXiv:2510.19974v2).
D-drive mirror at /d/projects/ERL/push_anything_ADMM/paper/2510.19974v2.pdf.

Layout per frame:
  Top-left  block  : (5a)-(5e) outer C3+ objective + LCS constraints (static).
  Bottom-left block: current ADMM step context (7)/(8)/(9)/(10)/(12), varies
                     with mode. Reads mode + λ_n_max from [STEP] log lines.
  Right side       : live state readout (step, t, mode, goal_dist, yaw,
                     lam_n, contact).

Usage:
    python3 tools/visualizer/paint_paper_equations.py \\
        --frames-dir results/tight_goal_p21_refvideo_save_frames \\
        --log-path   results/tight_goal_p21_refvideo_save.txt \\
        --output     results/tight_goal_p21_refvideo_save_paper.mp4 \\
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

_STEP_RE = re.compile(
    r"\[STEP\] step=(\d+) mode=(\w+) t=([\d.]+)s "
    r"ee=\(([+-][\d.]+),([+-][\d.]+),([+-][\d.]+)\) "
    r"obj=\(([+-][\d.]+),([+-][\d.]+),([+-][\d.]+)\) "
    r"goal_dist=([\d.]+)m switch=(\w+)"
)
_C3_EXTRA_RE = re.compile(
    r"lam_n=([+-]?[\d.]+) lam_t=([+-]?[\d.]+) contact=(\w)"
)


# ---------------------------------------------------------------------------
# Static paper-equation blocks — plain-text formatting suitable for TrueType
# monospace rendering. Superscripts and Greek letters rendered as UTF-8.
# ---------------------------------------------------------------------------

_EQ_OUTER = [
    "Bui et al. 2026  ·  Sampling-Based Contact-Implicit MPC  (arXiv:2510.19974v2)",
    "",
    "(5a)   min   Σᴺ⁻¹ₖ₌₀ ( xᵀₖ Qₖ xₖ + uᵀₖ Rₖ uₖ ) + xᵀₙ Qₙ xₙ",
    "       x,u,λ,η",
    "(5b)   s.t.  xₖ₊₁ = A xₖ + B uₖ + D λₖ + d              (LCS dynamics)",
    "(5c)         ηₖ   = E xₖ + F λₖ + H uₖ + c              (complementarity slack)",
    "(5d)         0 ≤ λₖ ⊥ ηₖ ≥ 0                            (complementarity)",
    "(5e)         (xₖ, uₖ) ∈ C                              (workspace + input limits)",
]

_ADMM_HEADER = "ADMM (7)-(9): scaled dual form   ρ={rho}   iter {i}/{N}"

_EQ_ADMM_Z = [
    "(7)  z-update:   z^(i+1) = argmin_z L_ρ( z, δ^(i), w^(i) )",
    "                 QP over (x, λ, u, η) at every knot k=0..N-1",
    "(10) augmented Lagrangian:",
    "     min_z c(z) + Σₖ ρ ‖ zₖ − δ^(i)ₖ + w^(i)ₖ ‖²_G",
]
_EQ_ADMM_DELTA = [
    "(8)  δ-projection:  δₖ^(i+1) = argmin_δₖ L_ρ( z^(i+1)_k, δₖ, w^(i)_k )",
    "(12) componentwise case-analysis:",
    "     (δ_λ, δ_η) = ⎧ (0, η°)  if η° ≥ 0 and η° ≥ √(uᵘ/uᵉ)·λ°   [case 1: η wins]",
    "                  ⎨ (λ°, 0)  if λ° ≥ 0 and η° < √(uᵘ/uᵉ)·λ°    [case 2: λ wins]",
    "                  ⎩ (0, 0)   otherwise                        [case 3: both zero]",
]
_EQ_ADMM_W = [
    "(9)  dual update:  w_k^(i+1) = w_k^(i) + z_k^(i+1) − δ_k^(i+1)",
    "     (scaled dual variables; ρ is fixed penalty per port config)",
]


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
            m = _STEP_RE.search(line)
            if m:
                step = int(m.group(1))
                d = data.setdefault(step, {})
                d["mode"] = m.group(2)
                d["t"] = float(m.group(3))
                d["obj"] = (float(m.group(7)), float(m.group(8)), float(m.group(9)))
                d["goal_dist"] = float(m.group(10))
                d["switch"] = m.group(11)
                c3 = _C3_EXTRA_RE.search(line)
                if c3:
                    d["lam_n"] = float(c3.group(1))
                    d["lam_t"] = float(c3.group(2))
                    d["contact"] = c3.group(3)
    return data


def _draw_text_box(draw: ImageDraw.ImageDraw, xy, lines,
                   font, fill=(255, 255, 255),
                   bg=(0, 0, 0, 180), pad=8, line_spacing=4):
    """Draw a semi-transparent black box behind multi-line text."""
    x0, y0 = xy
    max_w = 0
    total_h = 0
    line_hs = []
    for line in lines:
        bbox = font.getbbox(line)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        max_w = max(max_w, w)
        line_hs.append(h)
        total_h += h + line_spacing
    total_h -= line_spacing
    box = (x0 - pad, y0 - pad,
           x0 + max_w + pad, y0 + total_h + pad)
    # Semi-transparent background via ImageDraw's rectangle on RGB image:
    # draw solid dark rectangle (we're in RGB, not RGBA)
    draw.rectangle(box, fill=(0, 0, 0))
    y = y0
    for line, h in zip(lines, line_hs):
        draw.text((x0, y), line, font=font, fill=fill)
        y += h + line_spacing


def _phase_for_step(step_info: dict) -> str:
    """Choose which ADMM equation block to highlight for this step."""
    if step_info is None:
        return "z"
    mode = step_info.get("mode", "")
    contact = step_info.get("contact", "N")
    lam_n = step_info.get("lam_n", 0.0)
    if mode == "c3":
        # c3 mode: cycle through z/δ/w by contact / λ_n
        if contact == "Y" and lam_n > 0.1:
            return "delta"      # active complementarity projection
        else:
            return "z"          # no contact → QP dominates
    else:
        # free/repos mode: dual step conceptually irrelevant, show w for
        # continuity
        return "w"


def annotate_frame(img: Image.Image, step: int, info: Optional[dict],
                   font_small: ImageFont.FreeTypeFont,
                   font_tiny: ImageFont.FreeTypeFont) -> Image.Image:
    canvas = img.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    W, H = canvas.size

    # --- Top-left: outer objective (static) ---
    _draw_text_box(draw, (12, 12), _EQ_OUTER, font_tiny,
                   fill=(255, 220, 100))

    # --- Bottom-left: ADMM phase block (dynamic) ---
    phase = _phase_for_step(info)
    if phase == "z":
        block = _EQ_ADMM_Z
        phase_name = "PHASE  z-update (QP solve)"
    elif phase == "delta":
        block = _EQ_ADMM_DELTA
        phase_name = "PHASE  δ-projection (case-12)"
    else:
        block = _EQ_ADMM_W
        phase_name = "PHASE  w-update (scaled dual)"
    lines = [phase_name, ""] + list(block)
    # Roughly bottom third of image
    y0 = H - int(H * 0.36)
    _draw_text_box(draw, (12, y0), lines, font_tiny,
                   fill=(180, 220, 255))

    # --- Right side: live state readout ---
    if info is not None:
        readout = [
            f"step={step}",
            f"t={info.get('t', 0.0):.2f}s",
            f"mode={info.get('mode', '?')}",
            f"obj_xy=({info['obj'][0]:+.4f},{info['obj'][1]:+.4f})",
            f"goal_dist={info.get('goal_dist', 0.0)*1000:.1f} mm",
        ]
        if "lam_n" in info:
            readout.append(f"lam_n={info['lam_n']:+.3f} N")
            readout.append(f"contact={info.get('contact', '?')}")
        readout.append(f"switch={info.get('switch', '?')}")
        _draw_text_box(draw, (W - 340, 12), readout, font_small,
                       fill=(200, 255, 200))

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
    args = ap.parse_args()

    print(f"[paper-hud] parsing {args.log_path}")
    steps = parse_log(args.log_path)
    print(f"[paper-hud] parsed {len(steps)} step records")

    frames = find_frames(args.frames_dir)
    print(f"[paper-hud] found {len(frames)} frames")
    if not frames:
        raise SystemExit("no frames to annotate")

    font_tiny = _load_mono_font(14)
    font_small = _load_mono_font(16)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for i, (step, path) in enumerate(frames):
            info = steps.get(step)
            img = Image.open(path)
            out = annotate_frame(img, step, info, font_small, font_tiny)
            out.save(tmp_dir / f"annot_{i:06d}.png", optimize=False)
            if i % 200 == 0:
                print(f"[paper-hud] painted {i}/{len(frames)}")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y", "-framerate", str(args.fps),
            "-i", str(tmp_dir / "annot_%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "veryfast", "-crf", "23",
            str(args.output),
        ]
        print(f"[paper-hud] encoding → {args.output}")
        subprocess.check_call(cmd)
    print(f"[paper-hud] done → {args.output}")


if __name__ == "__main__":
    main()
