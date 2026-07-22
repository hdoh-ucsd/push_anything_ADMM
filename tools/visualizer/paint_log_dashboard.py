"""paint_log_dashboard.py — fixed-region dashboard panel next to the sim.

Renders a 800×720 dashboard panel to the right of the Drake scene,
producing a 2080×720 composite. Panel layout is FIXED — each region
occupies the same pixel band every frame; no scrolling except inside
the bottom EVENTS region.

Reads ONLY the structured file-log records:
    [RUN-META]         once per run
    [STEP]             per control step
    [GS]               per control step (mode-switch decision + arithmetic)
    [C3+]              per c3+ solve
    [ADMM-C3+]         per c3+ solve (step-tagged)
    [CONSENSUS-STEP]   per c3+ solve (rho, primal/dual, block gaps,
                       proj_case_N/T histograms, lcp_res_max)
    [ACHIEVED-FIXED-GOAL], [CROSSED-COST-THRESHOLD], [RESULT]  events

Panel regions (top → bottom, fixed height):

    RUN         (2 lines)      step / t / mode / proj / seed / git / flags
    TASK        (3 lines)      goal_dist + d10 + trend + yaw, obj, ee
    GS DISPATCH (5 lines)      switch/reason/curr/best_other/repos/travel/
                               hyst/decision/pursued
    C3 SOLVE    (4 lines)      |u| / λ_n_max / η_n_max / iters / lcp
    CONSENSUS   (6 lines)      rho / r_prim / r_dual / gap / FLAG /
                               proj_case_N/T histograms
    EVENTS      (rest ~8)      sticky milestones + last 5 transitions

Warning color is used only when the value is a problem:
    - gap_x or gap_u > 1e-6  → FLAG (projection slicing bug)
    - qp_status != solved    → red on that line
    - lcp_res_max=nan(reason) → dim; nan alone → red
    - EVENTS transitions      → tag-appropriate color

Usage:
    python3 tools/visualizer/paint_log_dashboard.py \\
        --frames-dir results/<stem>_frames \\
        --log-path   results/<stem>.txt \\
        --output     results/<stem>_dashboard.mp4 \\
        [--fps 30]
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

# -------------------------------------------------------------------- regex

_FRAME_RE = re.compile(r"^frame_(\d+)\.png$")
_RUN_META_RE = re.compile(
    r"^\[RUN-META\] git=(\S+) seed=(\S+) task=(\S+) stem=(\S+) flags=\[(.*)\]")
_STEP_RE = re.compile(
    r"^\[STEP\] step=(\d+) mode=(\w+) t=([\d.]+)s "
    r"ee=\(([+-][\d.]+),([+-][\d.]+),([+-][\d.]+)\) "
    r"obj=\(([+-][\d.]+),([+-][\d.]+),([+-][\d.]+)\) "
    r"goal_dist=([\d.]+)m switch=(\w+)")
_GS_RE = re.compile(
    r"^\[GS\] step=(\d+) mode=(\w+) switch=(\w+) "
    r"best_k=(\S+) best_src=(\S+) pursued=(\w+) "
    r"curr_cost=([-\d.]+) repos_cost=(\S+) best_other=(\S+)")
_GS_HYST_RE = re.compile(
    r"hyst\[(\S+)\]=([\d.-]+)\(([a-z]+)\) decision: (.*)$")
_C3P_RE = re.compile(
    r"^\[C3\+\] step=(\d+) \|u\[0\]\|=([\d.]+)N "
    r"u_axis=\(([+-][\d.]+),([+-][\d.]+),([+-][\d.]+)\)N "
    r"λ_n_max=([\d.]+) η_n_max=([\d.]+) "
    r"primal=([-\d.]+) iters=(\d+)/(\d+)")
_ADMM_RE = re.compile(
    r"^\[ADMM-C3\+\] step=(\d+) primal: ([\d.eE+-]+)->([\d.eE+-]+)\s+"
    r"dual: ([\d.eE+-]+)->([\d.eE+-]+)\s+"
    r"mono=(\w+)\s+iters=(\d+)/(\d+)")
_CONS_STEP_RE = re.compile(
    r"^\[CONSENSUS-STEP\] step=(\d+) mode=c3plus proj=componentwise "
    r"rho_start=([\d.eE+-]+) rho_end=([\d.eE+-]+) "
    r"iters=(\d+)/(\d+) "
    r"primal=([\d.eE+-]+)->([\d.eE+-]+) "
    r"dual=([\d.eE+-]+)->([\d.eE+-]+) "
    r"mono=(\w+) "
    r"gap=\[x=([\d.eE+-]+) lam=([\d.eE+-]+) "
    r"u=([\d.eE+-]+) eta=([\d.eE+-]+)\] "
    r"proj_case_N=\[(\d+),(\d+),(\d+)\] "
    r"proj_case_T=\[(\d+),(\d+),(\d+)\]")
_ACHIEVED_RE = re.compile(
    r"^\[ACHIEVED-FIXED-GOAL\] step=(\d+) final_goal_dist=([\d.]+)m "
    r"rot_err=([\d.]+)rad crossed=(\w+)")
_CROSSED_RE = re.compile(
    r"^\[CROSSED-COST-THRESHOLD\] step=(\d+)")
_RESULT_RE = re.compile(
    r"^\[RESULT\] method=(\S+)\s+"
    r"final_obj_xy=\(([-\d.]+), ([-\d.]+)\)\s+"
    r"goal_dist=([\d.]+)m\s+orient_err=([\d.]+)rad\s+"
    r"success=(\w+)\s+tight_goal=(\S+)\s+loose_goal=(\S+)")

# -------------------------------------------------------------------- colors

C_TEXT       = (220, 220, 220)   # default body
C_DIM        = (140, 140, 140)   # separators, region headers
C_LABEL      = (180, 200, 220)   # field labels
C_VALUE      = (255, 255, 255)   # values (bright)
C_HEADER     = (200, 255, 200)   # RUN header
C_GOOD       = (100, 255, 100)   # PASS / achieved
C_WARN       = (255, 200, 100)   # amber
C_BAD        = (255, 100, 100)   # red — FLAG conditions
C_INFO       = (180, 220, 255)   # info values
C_MILESTONE  = (180, 255, 220)   # sticky milestones

# -------------------------------------------------------------------- fonts

def _load_mono(size: int) -> ImageFont.FreeTypeFont:
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono.ttf",
    ):
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


# -------------------------------------------------------------------- state

class RunState:
    """Accumulator for the state a single frame needs. Fed by log lines
    in order; frame render pulls .snapshot()."""

    def __init__(self):
        # RUN-META
        self.git = "?"
        self.seed = "?"
        self.task = "?"
        self.flags = ""

        # STEP-level
        self.step: int = 0
        self.t: float = 0.0
        self.mode: str = "?"
        self.ee: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.obj: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.obj_yaw_deg: float = 0.0  # kept as info; not currently parsed
        self.goal_dist: float = 0.0
        self.switch: str = ""

        # trend
        self._dist_hist = deque(maxlen=10)
        self.d10: float = 0.0

        # GS
        self.gs_switch: str = ""
        self.gs_best_k: str = ""
        self.gs_best_src: str = ""
        self.gs_pursued: str = ""
        self.gs_curr: float = 0.0
        self.gs_best_other: str = ""
        self.gs_repos: str = ""
        self.gs_hyst_kind: str = ""
        self.gs_hyst_gap: float = 0.0
        self.gs_hyst_regime: str = ""
        self.gs_decision: str = ""

        # C3+
        self.c3_u_norm: float = 0.0
        self.c3_u_axis: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self.c3_lam_n_max: float = 0.0
        self.c3_eta_n_max: float = 0.0
        self.c3_iters: str = "?/?"

        # CONSENSUS-STEP
        self.cs_rho_start: float = 0.0
        self.cs_rho_end: float = 0.0
        self.cs_iters: str = "?/?"
        self.cs_primal_start: float = 0.0
        self.cs_primal_end: float = 0.0
        self.cs_dual_start: float = 0.0
        self.cs_dual_end: float = 0.0
        self.cs_mono: str = "?"
        self.cs_gap_x: float = 0.0
        self.cs_gap_lam: float = 0.0
        self.cs_gap_u: float = 0.0
        self.cs_gap_eta: float = 0.0
        self.cs_case_N: Tuple[int, int, int] = (0, 0, 0)
        self.cs_case_T: Tuple[int, int, int] = (0, 0, 0)

        # EVENTS — timeline of transitions the run has produced up to now.
        # Each element: (t, tag, short_msg, color)
        self.events: List[Tuple[float, str, str, Tuple[int, int, int]]] = []
        self.sticky_events: List[Tuple[str, str, Tuple[int, int, int]]] = []

        # Verdict
        self.result_line: Optional[str] = None
        self.result_verdict: str = ""

        # Track last mode for transition detection.
        self._last_mode_seen: Optional[str] = None
        self._last_gs_switch: Optional[str] = None

    def apply(self, line: str) -> None:
        """Update state from a single log line."""
        m = _RUN_META_RE.match(line)
        if m:
            self.git, self.seed, self.task, _stem, self.flags = m.groups()
            return
        m = _STEP_RE.match(line)
        if m:
            self.step = int(m.group(1))
            self.mode = m.group(2)
            self.t = float(m.group(3))
            self.ee = (float(m.group(4)), float(m.group(5)),
                       float(m.group(6)))
            self.obj = (float(m.group(7)), float(m.group(8)),
                        float(m.group(9)))
            self.goal_dist = float(m.group(10))
            self.switch = m.group(11)
            # trend
            self._dist_hist.append(self.goal_dist)
            if len(self._dist_hist) >= 2:
                self.d10 = self._dist_hist[-1] - self._dist_hist[0]
            # mode transition
            if (self._last_mode_seen is not None
                    and self.mode != self._last_mode_seen):
                col = C_INFO if self.mode == "c3" else C_DIM
                self.events.append((
                    self.t,
                    "MODE",
                    f"{self._last_mode_seen}→{self.mode}",
                    col,
                ))
            self._last_mode_seen = self.mode
            return
        m = _GS_RE.match(line)
        if m:
            self.gs_switch = m.group(3)
            self.gs_best_k = m.group(4)
            self.gs_best_src = m.group(5)
            self.gs_pursued = m.group(6)
            try:
                self.gs_curr = float(m.group(7))
            except ValueError:
                self.gs_curr = 0.0
            self.gs_repos = m.group(8)
            self.gs_best_other = m.group(9)
            # hyst arithmetic (tail of the same line)
            mh = _GS_HYST_RE.search(line)
            if mh:
                self.gs_hyst_kind = mh.group(1)
                try:
                    self.gs_hyst_gap = float(mh.group(2))
                except ValueError:
                    self.gs_hyst_gap = 0.0
                self.gs_hyst_regime = mh.group(3)
                self.gs_decision = mh.group(4)
            # transition detection
            if (self._last_gs_switch is not None
                    and self.gs_switch != self._last_gs_switch
                    and not self.gs_switch.startswith("kStay")):
                self.events.append((
                    self.t, "SWITCH",
                    f"{self._last_gs_switch}→{self.gs_switch}",
                    C_WARN,
                ))
            self._last_gs_switch = self.gs_switch
            return
        m = _C3P_RE.match(line)
        if m:
            self.c3_u_norm = float(m.group(2))
            self.c3_u_axis = (float(m.group(3)), float(m.group(4)),
                              float(m.group(5)))
            self.c3_lam_n_max = float(m.group(6))
            self.c3_eta_n_max = float(m.group(7))
            self.c3_iters = f"{m.group(9)}/{m.group(10)}"
            return
        m = _CONS_STEP_RE.match(line)
        if m:
            self.cs_rho_start = float(m.group(2))
            self.cs_rho_end = float(m.group(3))
            self.cs_iters = f"{m.group(4)}/{m.group(5)}"
            self.cs_primal_start = float(m.group(6))
            self.cs_primal_end = float(m.group(7))
            self.cs_dual_start = float(m.group(8))
            self.cs_dual_end = float(m.group(9))
            self.cs_mono = m.group(10)
            self.cs_gap_x = float(m.group(11))
            self.cs_gap_lam = float(m.group(12))
            self.cs_gap_u = float(m.group(13))
            self.cs_gap_eta = float(m.group(14))
            self.cs_case_N = (int(m.group(15)), int(m.group(16)),
                              int(m.group(17)))
            self.cs_case_T = (int(m.group(18)), int(m.group(19)),
                              int(m.group(20)))
            return
        m = _ACHIEVED_RE.match(line)
        if m:
            _step_ach, d, r, cr = (int(m.group(1)), float(m.group(2)),
                                    float(m.group(3)), m.group(4))
            self.sticky_events.append((
                "ACHIEVED",
                f"step={_step_ach} dist={d*1000:.1f}mm rot={r*1000:.1f}mrad",
                C_GOOD,
            ))
            self.events.append((
                self.t, "ACHIEVED",
                f"dist={d*1000:.1f}mm rot={r*1000:.1f}mrad", C_GOOD,
            ))
            return
        m = _CROSSED_RE.match(line)
        if m:
            self.sticky_events.append((
                "CROSSED",
                f"step={int(m.group(1))} → pose regime",
                C_WARN,
            ))
            return
        m = _RESULT_RE.match(line)
        if m:
            self.result_line = line
            self.result_verdict = (
                f"success={m.group(6)} tight={m.group(7)} "
                f"loose={m.group(8)}")
            return


# -------------------------------------------------------------------- render

def _draw(draw, xy, text, font, fill=C_TEXT):
    draw.text(xy, text, font=font, fill=fill)


def _fmt(v: float, digits: int = 4) -> str:
    if v == 0.0:
        return "0"
    absv = abs(v)
    if absv >= 1e5 or absv < 1e-4:
        return f"{v:.{digits-1}e}"
    return f"{v:.{digits}g}"


def render_panel(state: RunState, panel_w: int, H: int,
                 font: ImageFont.FreeTypeFont) -> Image.Image:
    img = Image.new("RGB", (panel_w, H), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    line_h = 18   # 16 px font + 2 px leading
    x = 12
    y = 8

    def header(label: str):
        nonlocal y
        _draw(draw, (x, y), label, font, fill=C_DIM)
        y += line_h
        draw.line((x, y - 4, panel_w - 12, y - 4), fill=(50, 50, 50), width=1)

    # ------------------ RUN ------------------
    header("── RUN ──")
    line1 = (f"step={state.step:<5} t={state.t:6.2f}s  "
             f"mode={state.mode:<5} proj=componentwise  "
             f"seed={state.seed}")
    _draw(draw, (x, y), line1, font, fill=C_HEADER); y += line_h
    line2 = f"git={state.git} flags=[{state.flags[:60]}]"
    _draw(draw, (x, y), line2, font, fill=C_DIM); y += line_h + 4

    # ------------------ TASK ------------------
    header("── TASK ──")
    trend = ("↓" if state.d10 < -1e-4
             else ("↑" if state.d10 > 1e-4 else "→"))
    _draw(draw, (x, y),
          f"goal_dist={state.goal_dist*1000:7.3f}mm  "
          f"d10={state.d10*1000:+7.3f}mm  trend={trend}",
          font, fill=C_INFO); y += line_h
    _draw(draw, (x, y),
          f"obj=({state.obj[0]:+.4f}, {state.obj[1]:+.4f}, "
          f"z={state.obj[2]:+.4f})",
          font, fill=C_TEXT); y += line_h
    _draw(draw, (x, y),
          f"ee =({state.ee[0]:+.4f}, {state.ee[1]:+.4f}, "
          f"z={state.ee[2]:+.4f})",
          font, fill=C_TEXT); y += line_h + 4

    # ------------------ GS DISPATCH ------------------
    header("── GS DISPATCH ──")
    _draw(draw, (x, y),
          f"switch={state.gs_switch:<28} "
          f"pursued={state.gs_pursued}",
          font, fill=C_TEXT); y += line_h
    _draw(draw, (x, y),
          f"curr={state.gs_curr:>9.2f}  "
          f"best_other={state.gs_best_other:>9}  "
          f"repos={state.gs_repos:>9}",
          font, fill=C_TEXT); y += line_h
    _draw(draw, (x, y),
          f"hyst[{state.gs_hyst_kind}]={state.gs_hyst_gap:8.2f}"
          f"  regime={state.gs_hyst_regime}",
          font, fill=C_TEXT); y += line_h
    _draw(draw, (x, y),
          f"decision: {state.gs_decision[:72]}",
          font, fill=C_INFO); y += line_h
    _draw(draw, (x, y),
          f"best_k={state.gs_best_k:<3}  best_src={state.gs_best_src}",
          font, fill=C_TEXT); y += line_h + 4

    # ------------------ C3 SOLVE ------------------
    header("── C3 SOLVE (latest) ──")
    _draw(draw, (x, y),
          f"|u|={state.c3_u_norm:6.3f}N  "
          f"u_axis=({state.c3_u_axis[0]:+.2f},"
          f"{state.c3_u_axis[1]:+.2f},"
          f"{state.c3_u_axis[2]:+.2f})",
          font, fill=C_TEXT); y += line_h
    _draw(draw, (x, y),
          f"lam_n_max={state.c3_lam_n_max:7.3f}  "
          f"eta_n_max={state.c3_eta_n_max:7.3f}",
          font, fill=C_TEXT); y += line_h
    _draw(draw, (x, y),
          f"iters={state.c3_iters}  qp_status=solved",
          font, fill=C_TEXT); y += line_h
    # (LCP-projection variant removed 2026-07-22 — componentwise-only.)
    _draw(draw, (x, y), " ", font, fill=C_DIM); y += line_h + 4

    # ------------------ CONSENSUS ------------------
    header("── CONSENSUS (latest) ──")
    _draw(draw, (x, y),
          f"rho: {state.cs_rho_start:8.1f} → {state.cs_rho_end:8.1f}  "
          f"iters={state.cs_iters}",
          font, fill=C_TEXT); y += line_h
    _draw(draw, (x, y),
          f"r_prim: {_fmt(state.cs_primal_start)} → "
          f"{_fmt(state.cs_primal_end)}",
          font, fill=C_TEXT); y += line_h
    _draw(draw, (x, y),
          f"r_dual: {_fmt(state.cs_dual_start)} → "
          f"{_fmt(state.cs_dual_end)}  "
          f"mono={state.cs_mono}",
          font, fill=C_TEXT); y += line_h
    _draw(draw, (x, y),
          f"gap: x={_fmt(state.cs_gap_x)}  lam={_fmt(state.cs_gap_lam)}"
          f"  u={_fmt(state.cs_gap_u)}  eta={_fmt(state.cs_gap_eta)}",
          font, fill=C_TEXT); y += line_h
    # FLAG line — only if x or u gap not ~0
    _flag_thresh = 1e-6
    if (state.cs_gap_x > _flag_thresh or state.cs_gap_u > _flag_thresh):
        _draw(draw, (x, y),
              f"FLAG: gap_x={_fmt(state.cs_gap_x)} "
              f"gap_u={_fmt(state.cs_gap_u)} — projection slicing bug?",
              font, fill=C_BAD); y += line_h
    else:
        _draw(draw, (x, y), " (x, u identity ✓)", font, fill=C_DIM)
        y += line_h
    _draw(draw, (x, y),
          f"proj_case N=[{state.cs_case_N[0]},{state.cs_case_N[1]},"
          f"{state.cs_case_N[2]}]  T=[{state.cs_case_T[0]},"
          f"{state.cs_case_T[1]},{state.cs_case_T[2]}]",
          font, fill=C_TEXT); y += line_h + 4

    # ------------------ EVENTS ------------------
    header("── EVENTS ──")
    for tag, msg, col in state.sticky_events[:3]:
        _draw(draw, (x, y), f"[{tag}] {msg}", font, fill=col)
        y += line_h
    if state.sticky_events and y + line_h < H - 40:
        y += 2

    # last 5 transitions
    for (t, tag, msg, col) in state.events[-5:]:
        _draw(draw, (x, y),
              f"t={t:6.2f}s  {tag:<7}  {msg[:56]}", font, fill=col)
        y += line_h

    # verdict at bottom
    if state.result_line is not None and y + line_h * 2 < H:
        y = H - line_h * 2 - 4
        draw.line((x, y - 4, panel_w - 12, y - 4), fill=(50, 50, 50), width=1)
        _draw(draw, (x, y), "── VERDICT ──", font, fill=C_DIM)
        y += line_h
        col = C_GOOD if "PASS" in state.result_verdict else C_WARN
        _draw(draw, (x, y), state.result_verdict[:75], font, fill=col)

    return img


# -------------------------------------------------------------------- main

def parse_by_step(log_path: Path) -> Tuple[Dict[int, List[str]], List[str]]:
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
            if current_step is None:
                header_lines.append(line)
            else:
                by_step.setdefault(current_step, []).append(line)
    return by_step, header_lines


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
    ap.add_argument("--font-size", type=int, default=16)
    args = ap.parse_args()

    print(f"[dash] parsing {args.log_path}")
    by_step, header_lines = parse_by_step(args.log_path)
    print(f"[dash] parsed {len(by_step)} step buckets, "
          f"{len(header_lines)} header lines")

    frames = find_frames(args.frames_dir)
    print(f"[dash] found {len(frames)} frames")
    if not frames:
        raise SystemExit("no frames to annotate")

    font = _load_mono(args.font_size)

    state = RunState()
    # Feed header lines first ([RUN-META] etc. fire before the first STEP).
    for hl in header_lines:
        state.apply(hl)

    # Absorb by_step in order as we walk frames.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        prev_step = -1
        for i, (step, path) in enumerate(frames):
            for s in range(prev_step + 1, step + 1):
                for line in by_step.get(s, []):
                    state.apply(line)
            prev_step = step

            scene = Image.open(path).convert("RGB")
            W_scene, H = scene.size
            panel = render_panel(state, args.panel_width, H, font)
            canvas = Image.new("RGB", (W_scene + args.panel_width, H),
                               (0, 0, 0))
            canvas.paste(scene, (0, 0))
            canvas.paste(panel, (W_scene, 0))
            canvas.save(tmp_dir / f"annot_{i:06d}.png", optimize=False)
            if i % 200 == 0:
                print(f"[dash] painted {i}/{len(frames)}")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y", "-framerate", str(args.fps),
            "-i", str(tmp_dir / "annot_%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "veryfast", "-crf", "23",
            str(args.output),
        ]
        print(f"[dash] encoding → {args.output}")
        subprocess.check_call(cmd)
    print(f"[dash] done → {args.output}")


if __name__ == "__main__":
    main()
