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
_ROLLING_TAGS = {"[STEP]", "[GS]", "[C3+]", "[CONTACT-RUN]", "[ENTRY-GATE]"}
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
    "[CONTACT-RUN]":           (180, 220, 255),
    "[ENTRY-GATE]":            (255, 140, 140),
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


# ── next-move gauges ────────────────────────────────────────────────────
# Every mode switch in the controller is a countdown against a threshold
# the log already prints: the [GS] decision arithmetic (cost-hysteresis
# flip), the free-mode [STEP] finished_val→finished_thresh (repos arrival
# fires kToC3ReachedReposTarget), the [ENTRY-GATE] block, the
# [CONTACT-RUN] distance to the 2 mm LCS admission, and the consecutive
# no-EE-BOX streak (disengage at 5). The raw lines scroll and truncate;
# the gauge block pins value-vs-threshold pairs at a fixed position so a
# viewer can watch each margin trend toward its flip point.

_DISENGAGE_STREAK = 5      # wrapper contact-loss disengage threshold
_GAUGE_RECENCY = 10        # steps a CONTACT-RUN / ENTRY-GATE stays shown

_G_STEP_RE = re.compile(
    r"^\[STEP\] step=(\d+) mode=(\w+) t=([\d.]+)s")
_G_FLOAT = r"[-+]?[\d.]+"
_G_FIELD_RES = {
    # [STEP] fields (both modes)
    "goal_dist": re.compile(rf"goal_dist=({_G_FLOAT})m"),
    "switch":    re.compile(r"switch=(\w+)"),
    # [STEP] free-mode
    "fin_val":   re.compile(rf"finished_val=({_G_FLOAT})m"),
    "fin_thr":   re.compile(rf"finished_thresh=({_G_FLOAT})m"),
    # [STEP] c3-mode
    "lam_n":     re.compile(rf"lam_n=({_G_FLOAT})"),
    "contact":   re.compile(r"contact=(\w)"),
    "f_cmd":     re.compile(r"f_cmd=\(([^)]+)\)"),
}
_G_GS_RES = {
    "curr_cost":  re.compile(rf"curr_cost=({_G_FLOAT})"),
    "best_other": re.compile(rf"best_other=({_G_FLOAT})"),
    "met":        re.compile(r"met_progress=(\w)"),
    "stall":      re.compile(r"steps_since_improve=(\d+)"),
    "switches":   re.compile(r"switches=(\d+)"),
    "hyst":       re.compile(rf"hyst\[(\w+)\]=({_G_FLOAT})"),
    "decision":   re.compile(r"decision: (.*)$"),
}
_G_CONTACT_RE = re.compile(
    rf"^\[CONTACT-RUN\] step=(\d+) .*distance=({_G_FLOAT}) contact_type=(\S+)")
_G_ENTRY_RE = re.compile(
    r"^\[ENTRY-GATE\] step=(\d+) (ee_to_\w+)=([\d.]+)mm >= thr=([\d.]+)mm")


def _gfloat(s: str) -> Optional[float]:
    try:
        v = float(s)
    except ValueError:
        return None
    return None if v != v else v  # NaN → None


def parse_gauges(log_path: Path) -> Dict[int, dict]:
    """Per-step gauge fields, keyed by each line's OWN step= number.

    ([GS] step=N prints before [STEP] step=N, so the by_step bucketing
    used for the rolling window attributes it to N-1; keying on the
    line's own step field sidesteps that.)
    """
    gauges: Dict[int, dict] = {}

    def g(step: int) -> dict:
        return gauges.setdefault(step, {})

    with open(log_path, errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = _G_STEP_RE.match(line)
            if m:
                d = g(int(m.group(1)))
                d["mode"] = m.group(2)
                for key, rex in _G_FIELD_RES.items():
                    fm = rex.search(line)
                    if fm:
                        d[key] = fm.group(1)
                continue
            if line.startswith("[GS] step="):
                sm = re.match(r"^\[GS\] step=(\d+)", line)
                if not sm:
                    continue
                d = g(int(sm.group(1)))
                for key, rex in _G_GS_RES.items():
                    fm = rex.search(line)
                    if fm:
                        d[key] = fm.group(2) if key == "hyst" else fm.group(1)
                        if key == "hyst":
                            d["hyst_kind"] = fm.group(1)
                continue
            m = _G_CONTACT_RE.match(line)
            if m:
                d = g(int(m.group(1)))
                d["c_dist"], d["c_type"] = m.group(2), m.group(3)
                continue
            m = _G_ENTRY_RE.match(line)
            if m:
                d = g(int(m.group(1)))
                d["eg_label"], d["eg_val"], d["eg_thr"] = \
                    m.group(2), m.group(3), m.group(4)
    return gauges


def _decision_margin(decision: str) -> Optional[float]:
    """Distance-to-flip from the [GS] decision arithmetic.

    Both shapes compare a total against a comparand across ' vs ':
      free: best_other(398.86) vs curr(373.17)+gap(119.66)=492.83
      c3:   best_other(2618.58)+gap(2312.69)=4931.28 vs c3(2720.81)
    A side's value is the number after '=' if present, else the last
    parenthesized number. The mode flips when the sides cross, so
    |lhs - rhs| is the margin regardless of direction.
    """
    parts = decision.split(" vs ")
    if len(parts) != 2:
        return None
    vals = []
    for side in parts:
        m = re.search(rf"=({_G_FLOAT})\s*$", side.strip())
        if not m:
            nums = re.findall(rf"\(({_G_FLOAT})\)", side)
            if not nums:
                return None
            m_val = _gfloat(nums[-1])
        else:
            m_val = _gfloat(m.group(1))
        if m_val is None:
            return None
        vals.append(m_val)
    return abs(vals[0] - vals[1])


_GAUGE_OK = (160, 230, 160)
_GAUGE_WARN = (255, 200, 80)
_GAUGE_TRIP = (255, 100, 100)
_GAUGE_DIM = (150, 150, 150)


def _draw_gauges(draw, x0: int, y: int, g: dict, font, line_h: int) -> int:
    """Fixed-position value-vs-threshold block. Returns new y."""
    draw.text((x0, y), "── next-move gauges ──", font=font, fill=_GAUGE_DIM)
    y += line_h
    if not g:
        return y
    mode = g.get("mode", "?")

    # switch reason + progress stall + switch count
    stall = g.get("stall")
    met = g.get("met")
    color = _GAUGE_OK
    if met == "N":
        color = _GAUGE_WARN  # unproductive-disengage countdown is running
    line = (f"switch  {g.get('switch', '?'):<28s} "
            f"stall {stall or '?'} met={met or '?'} "
            f"switches {g.get('switches', '?')}")
    draw.text((x0, y), line, font=font, fill=color)
    y += line_h

    # cost-hysteresis flip margin (drives kToC3Cost / kToReposCost)
    decision = g.get("decision")
    if decision and decision != "transition":
        margin = _decision_margin(decision)
        hyst = _gfloat(g.get("hyst", ""))
        color = _GAUGE_OK
        if margin is not None and hyst:
            if margin < 0.05 * hyst:
                color = _GAUGE_TRIP
            elif margin < 0.25 * hyst:
                color = _GAUGE_WARN
        mtxt = f"Δ {margin:.1f} → 0" if margin is not None else "Δ ?"
        draw.text((x0, y), f"flip    {mtxt}   {decision}",
                  font=font, fill=color)
        y += line_h

    # repos arrival countdown (fires kToC3ReachedReposTarget at thresh)
    fin_val = _gfloat(g.get("fin_val", ""))
    fin_thr = _gfloat(g.get("fin_thr", ""))
    if mode == "free" and fin_val is not None and fin_thr:
        ratio = fin_val / fin_thr
        color = (_GAUGE_TRIP if ratio <= 1.0
                 else _GAUGE_WARN if ratio <= 2.0 else _GAUGE_OK)
        tag = "  ARRIVED" if ratio <= 1.0 else ""
        draw.text((x0, y),
                  f"repos   ee→tgt {fin_val:.3f}m  finish@{fin_thr:.3f}m{tag}",
                  font=font, fill=color)
        y += line_h

    # entry gate (blocks the free→c3 arrival transition)
    if g.get("_eg_age", 999) <= _GAUGE_RECENCY:
        draw.text((x0, y),
                  f"gate    {g.get('eg_label')} {g.get('eg_val')}mm "
                  f"≥ thr {g.get('eg_thr')}mm  BLOCK",
                  font=font, fill=_GAUGE_TRIP)
        y += line_h

    # c3 executor commitment
    if mode == "c3" and "lam_n" in g:
        draw.text((x0, y),
                  f"c3      λ_n {g['lam_n']}  contact {g.get('contact', '?')}"
                  f"  f_cmd ({g.get('f_cmd', '?')})",
                  font=font, fill=_GAUGE_OK)
        y += line_h

    # contact distance + no-contact disengage streak
    if g.get("_c_age", 999) <= _GAUGE_RECENCY:
        streak = g.get("_streak", 0)
        color = (_GAUGE_TRIP if streak >= _DISENGAGE_STREAK
                 else _GAUGE_WARN if streak >= 3 else _GAUGE_OK)
        draw.text((x0, y),
                  f"contact d {g.get('c_dist')} {g.get('c_type')}   "
                  f"no-contact {streak}/{_DISENGAGE_STREAK}",
                  font=font, fill=color)
        y += line_h

    # goal distance
    gd = _gfloat(g.get("goal_dist", ""))
    if gd is not None:
        draw.text((x0, y), f"goal    dist {gd:.3f}m", font=font,
                  fill=_GAUGE_OK)
        y += line_h
    return y


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


def _shorten_keep_tail(line: str, max_chars: int, head_chars: int = 34) -> str:
    """Middle-ellipsis truncation for rolling log lines.

    The controller's per-step lines put identity first (tag, step, mode,
    ee/obj coordinates — redundant with the scene panel) and the decision
    numbers last (costs, switch reason, λ, f_cmd). Head-only truncation
    therefore amputated exactly the fields a viewer needs; keep the head
    (tag+step+mode) AND the tail instead.
    """
    if len(line) <= max_chars:
        return line
    head = line[:head_chars]
    tail_budget = max_chars - head_chars - 1
    if tail_budget <= 0:
        return line[:max_chars - 1] + "…"
    return head + "…" + line[-tail_budget:]


def _draw_lines(draw, x0, y0, lines, font, max_chars, line_h,
                pad=6, shorten=_shorten):
    """Draw a list of lines starting at (x0, y0). Returns new y."""
    for line in lines:
        tag = _tag_of(line) or ""
        color = _tag_color(tag)
        draw.text((x0, y0), shorten(line, max_chars),
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
    gauge: Optional[dict] = None,
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

    # Next-move gauges: fixed-position value-vs-threshold countdowns
    if gauge is not None:
        y = _draw_gauges(draw, panel_x0, y, gauge, font_tiny, line_h)
        y += pad
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
                font_tiny, panel_max_chars, line_h,
                shorten=_shorten_keep_tail)

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
    ap.add_argument("--panel-max-chars", type=int, default=None,
                    help="Chars per panel line (default: computed from "
                         "--panel-width and the mono font's advance; the "
                         "old hardcoded 95 wasted ~150px of an 800px panel)")
    args = ap.parse_args()

    print(f"[sidepanel] parsing {args.log_path}")
    by_step, header_lines = parse_log_by_step(args.log_path)
    print(f"[sidepanel] parsed {len(by_step)} step buckets, "
          f"{len(header_lines)} header lines")
    gauges_by_step = parse_gauges(args.log_path)
    print(f"[sidepanel] parsed gauges for {len(gauges_by_step)} steps")

    frames = find_frames(args.frames_dir)
    print(f"[sidepanel] found {len(frames)} frames")
    if not frames:
        raise SystemExit("no frames to annotate")

    font_tiny = _load_mono_font(11)
    font_small = _load_mono_font(14)

    if args.panel_max_chars is None:
        # Panel text spans panel_x0 (= scene_W + 12) to composite_W - 12,
        # i.e. panel_width - 24 px; divide by the mono advance.
        char_px = font_tiny.getlength("0") or 7.0
        args.panel_max_chars = max(40, int((args.panel_width - 24) / char_px))
        print(f"[sidepanel] panel_max_chars={args.panel_max_chars} "
              f"(panel_width={args.panel_width}, char_px={char_px:.2f})")

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
        cur_gauge: dict = {}
        for i, (step, path) in enumerate(frames):
            for s in range(prev_step + 1, step + 1):
                lines = by_step.get(s, [])
                for line in lines:
                    fam = _line_family(line)
                    if fam == "sticky":
                        if line not in sticky_lines:
                            sticky_lines.append(line)
                    elif fam in ("rolling", "hi") and \
                            (_tag_of(line) in _ROLLING_TAGS):
                        rolling.append(line)
                    m = _STEP_RE.match(line)
                    if m:
                        step_info_map[s] = {
                            "t": float(m.group(3)),
                            "mode": m.group(2),
                        }
                # Merge this step's gauge fields into the running state.
                gd = gauges_by_step.get(s)
                if gd:
                    new_mode = gd.get("mode")
                    if new_mode and new_mode != cur_gauge.get("mode"):
                        # Mode flipped: drop the other mode's stale fields.
                        for k in ("fin_val", "fin_thr", "lam_n",
                                  "contact", "f_cmd"):
                            cur_gauge.pop(k, None)
                        if new_mode != "c3":
                            cur_gauge["_streak"] = 0
                    if "c_type" in gd:
                        cur_gauge["_streak"] = (
                            cur_gauge.get("_streak", 0) + 1
                            if gd["c_type"] == "NONE" else 0)
                        cur_gauge["_c_step"] = s
                    if "eg_val" in gd:
                        cur_gauge["_eg_step"] = s
                    cur_gauge.update(gd)
            prev_step = step
            info = step_info_map.get(step)
            cur_gauge["_c_age"] = step - cur_gauge.get("_c_step", -999)
            cur_gauge["_eg_age"] = step - cur_gauge.get("_eg_step", -999)
            img = Image.open(path)
            out = compose_frame(img, step, info,
                                list(rolling), sticky_lines,
                                font_tiny, font_small,
                                args.panel_width, args.panel_max_chars,
                                result_line if i >= len(frames) - 30
                                else None,
                                gauge=dict(cur_gauge))
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
