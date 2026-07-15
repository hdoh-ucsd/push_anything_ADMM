#!/usr/bin/env python3
"""Post-process a run.log to compute the β-amendment metrics:

  (1) Direction-fix ratio per contact event: F_goal_on_box / |F_on_box|
      using the realized Drake contact force ([GATE-CONTACT] line), and
      the same projection from the planner-derived command ([STEP] f_cmd).
  (2) Recoil fraction per contact event: peak |Δbox_x| during contact vs
      final |Δbox_x| measured 10 control steps (100 ms) after the EE
      departs (contact=N for ≥ 5 consecutive steps).

A contact event is a contiguous run of [STEP] mode=c3 lines with
contact=Y. We track Δbox_x relative to the box xy at the first step of
the event so the metric is event-relative (sampler-state-independent).

Bin verdicts:
  PASS    = net |Δbox_x| ≥ 5 mm at settle (any single event).
  PARTIAL = peak |Δbox_x| ≥ 5 mm but final < 1 mm (direction fixed,
            recoil dominates).
  FAIL    = peak |Δbox_x| < 1 mm.

Usage:
  python scripts/parse_beta_contact_events.py <run.log>
"""
from __future__ import annotations
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

STEP_RE = re.compile(
    r"^\[STEP\] step=(?P<step>\d+) mode=(?P<mode>\S+) t=(?P<t>[\d.+\-eE]+)s "
    r"ee=\((?P<eex>[+\-\d.eE]+),(?P<eey>[+\-\d.eE]+),(?P<eez>[+\-\d.eE]+)\) "
    r"obj=\((?P<objx>[+\-\d.eE]+),(?P<objy>[+\-\d.eE]+),(?P<objz>[+\-\d.eE]+)\) "
    r"goal_dist=(?P<gd>[\d.+\-eE]+)m"
)
STEP_C3_RE = re.compile(
    r"contact=(?P<contact>[YN]) productive=(?P<prod>[YN]) "
    r"f_cmd=\((?P<fx>[+\-\d.eE]+),(?P<fy>[+\-\d.eE]+),(?P<fz>[+\-\d.eE]+)\)"
)
GATE_RE = re.compile(
    r"^\[GATE-CONTACT\] step=(?P<step>\d+) "
    r"F_W=\((?P<fwx>[+\-\d.eE]+),(?P<fwy>[+\-\d.eE]+),(?P<fwz>[+\-\d.eE]+)\) "
    r"F_on_box=\((?P<fbx>[+\-\d.eE]+),(?P<fby>[+\-\d.eE]+),(?P<fbz>[+\-\d.eE]+)\)"
)
GATE_BOX_RE = re.compile(
    r"^\[GATE-CONTACT\] step=(?P<step>\d+).*?"
    r"box_p=\((?P<bpx>[+\-\d.eE]+),(?P<bpy>[+\-\d.eE]+),(?P<bpz>[+\-\d.eE]+)\)"
)


@dataclass
class StepRow:
    step:    int
    t:       float
    mode:    str
    ee:      tuple[float, float, float]
    obj:     tuple[float, float, float]            # 3-dec from [STEP]
    goal_dist: float
    contact:   Optional[str] = None
    f_cmd:     Optional[tuple[float, float, float]] = None
    F_on_box:  Optional[tuple[float, float, float]] = None
    box_p_hi:  Optional[tuple[float, float, float]] = None  # 5-dec from [GATE-CONTACT]

    @property
    def obj_x(self) -> float:
        return self.box_p_hi[0] if self.box_p_hi is not None else self.obj[0]

    @property
    def obj_y(self) -> float:
        return self.box_p_hi[1] if self.box_p_hi is not None else self.obj[1]


@dataclass
class ContactEvent:
    start_step:   int
    end_step:     int
    rows:         list[StepRow]
    settle_rows:  list[StepRow] = field(default_factory=list)

    @property
    def n_steps(self) -> int:
        return len(self.rows)

    @property
    def duration_ms(self) -> float:
        if len(self.rows) < 2:
            return 0.0
        return 1e3 * (self.rows[-1].t - self.rows[0].t)


def parse_log(path: str) -> tuple[list[StepRow], tuple[float, float]]:
    """Walk the log; for each step pull [STEP] + the matching
    [GATE-CONTACT] for that step (same step number).

    Returns (rows, g_hat_xy). For task-id 4 (west) g_hat = (-1, 0); we
    infer the direction from the *goal_xy* recorded in the task header.
    """
    rows: list[StepRow] = []
    gate_by_step: dict[int, tuple[float, float, float]] = {}
    goal_xy: tuple[float, float] = (0.0, 0.0)
    pending_step: Optional[StepRow] = None
    pending_c3_line: Optional[str] = None

    with open(path) as f:
        for line in f:
            # goal_xy header
            if "[ENV]  Goal coords:" in line:
                m = re.search(r"\[([+\-\d.eE]+),\s*([+\-\d.eE]+)\]", line)
                if m:
                    goal_xy = (float(m.group(1)), float(m.group(2)))
                continue
            m = STEP_RE.match(line)
            if m:
                step_row = StepRow(
                    step=int(m.group("step")),
                    t=float(m.group("t")),
                    mode=m.group("mode"),
                    ee=(float(m.group("eex")),
                        float(m.group("eey")),
                        float(m.group("eez"))),
                    obj=(float(m.group("objx")),
                         float(m.group("objy")),
                         float(m.group("objz"))),
                    goal_dist=float(m.group("gd")),
                )
                m2 = STEP_C3_RE.search(line)
                if m2:
                    step_row.contact = m2.group("contact")
                    step_row.f_cmd = (
                        float(m2.group("fx")),
                        float(m2.group("fy")),
                        float(m2.group("fz")),
                    )
                rows.append(step_row)
                continue
            m = GATE_RE.match(line)
            if m:
                gate_by_step[int(m.group("step"))] = (
                    float(m.group("fbx")),
                    float(m.group("fby")),
                    float(m.group("fbz")),
                )
            mp = GATE_BOX_RE.match(line)
            if mp:
                pass  # box_p stitched below via separate dict
    # Second pass for box_p (regex needs to match the same line as F_on_box;
    # easier to re-walk once at the end).
    box_p_by_step: dict[int, tuple[float, float, float]] = {}
    with open(path) as f:
        for line in f:
            mp = GATE_BOX_RE.match(line)
            if mp:
                box_p_by_step[int(mp.group("step"))] = (
                    float(mp.group("bpx")),
                    float(mp.group("bpy")),
                    float(mp.group("bpz")),
                )

    # Stitch GATE → STEP by step number.
    for r in rows:
        if r.step in gate_by_step:
            r.F_on_box = gate_by_step[r.step]
        if r.step in box_p_by_step:
            r.box_p_hi = box_p_by_step[r.step]

    return rows, goal_xy


def group_contact_events(rows: list[StepRow],
                         settle_steps: int = 10) -> list[ContactEvent]:
    """Group contiguous runs of contact=Y (mode=c3) into events. For each
    event, also retain the next `settle_steps` rows (any mode) after the
    contact ends, so we can measure recoil.
    """
    events: list[ContactEvent] = []
    i = 0
    while i < len(rows):
        if rows[i].mode == "c3" and rows[i].contact == "Y":
            j = i
            while (j < len(rows) and rows[j].mode == "c3"
                   and rows[j].contact == "Y"):
                j += 1
            event = ContactEvent(
                start_step=rows[i].step,
                end_step=rows[j - 1].step,
                rows=rows[i:j],
                settle_rows=rows[j:j + settle_steps],
            )
            events.append(event)
            i = j
        else:
            i += 1
    return events


def event_metrics(ev: ContactEvent,
                  g_hat: tuple[float, float]) -> dict:
    """Compute the direction-fix and recoil numbers for one event."""
    g = (float(g_hat[0]), float(g_hat[1]))
    g_norm = (g[0] ** 2 + g[1] ** 2) ** 0.5
    if g_norm > 1e-9:
        g = (g[0] / g_norm, g[1] / g_norm)
    else:
        g = (-1.0, 0.0)  # degenerate; fall back to west

    obj_x0, obj_y0 = ev.rows[0].obj_x, ev.rows[0].obj_y
    peak_disp = 0.0
    peak_step = ev.start_step
    fcmd_ratios: list[float] = []
    fbox_ratios: list[float] = []
    fbox_mags:   list[float] = []
    fbox_axis_x_frac: list[float] = []   # |Fx|/|F|
    fbox_axis_y_frac: list[float] = []   # |Fy|/|F|
    fbox_axis_z_frac: list[float] = []   # |Fz|/|F|

    # Per-step metrics during the event.
    for r in ev.rows:
        dx = r.obj_x - obj_x0
        dy = r.obj_y - obj_y0
        # Signed displacement along goal direction (positive = toward goal).
        disp_along = dx * g[0] + dy * g[1]
        if abs(disp_along) > abs(peak_disp):
            peak_disp = disp_along
            peak_step = r.step
        # f_cmd is the force ON THE EE (recoil convention). Box reaction
        # = -f_cmd. F_goal_on_box = (-f_cmd) · g_hat = -(f_cmd · g_hat).
        if r.f_cmd is not None:
            fcmd_mag = sum(c * c for c in r.f_cmd) ** 0.5
            fcmd_goal = -(r.f_cmd[0] * g[0] + r.f_cmd[1] * g[1])
            if fcmd_mag > 1e-9:
                fcmd_ratios.append(fcmd_goal / fcmd_mag)
        # F_on_box from Drake — already signed in the world frame as the
        # force on the box. F_goal = F_on_box · g_hat.
        if r.F_on_box is not None:
            fbox_mag = sum(c * c for c in r.F_on_box) ** 0.5
            if fbox_mag > 1e-9:
                fgoal = (r.F_on_box[0] * g[0]
                         + r.F_on_box[1] * g[1])
                fbox_ratios.append(fgoal / fbox_mag)
                fbox_mags.append(fbox_mag)
                fbox_axis_x_frac.append(abs(r.F_on_box[0]) / fbox_mag)
                fbox_axis_y_frac.append(abs(r.F_on_box[1]) / fbox_mag)
                fbox_axis_z_frac.append(abs(r.F_on_box[2]) / fbox_mag)

    # Final displacement: average of the last 5 settle rows (or last available).
    if ev.settle_rows:
        tail = ev.settle_rows[-5:] if len(ev.settle_rows) >= 5 else ev.settle_rows
        final_disp_along = sum(
            (r.obj_x - obj_x0) * g[0] + (r.obj_y - obj_y0) * g[1]
            for r in tail
        ) / len(tail)
    else:
        # No settle window — use last in-contact row.
        last = ev.rows[-1]
        final_disp_along = ((last.obj_x - obj_x0) * g[0]
                            + (last.obj_y - obj_y0) * g[1])

    def _mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")
    def _max(xs):
        return max(xs) if xs else float("nan")

    recoil_frac = (1.0 - final_disp_along / peak_disp
                   if abs(peak_disp) > 1e-9 else float("nan"))

    return dict(
        start_step      = ev.start_step,
        end_step        = ev.end_step,
        n_steps         = ev.n_steps,
        duration_ms     = ev.duration_ms,
        peak_disp_mm    = 1e3 * peak_disp,
        peak_step       = peak_step,
        final_disp_mm   = 1e3 * final_disp_along,
        recoil_frac     = recoil_frac,
        fcmd_ratio_mean = _mean(fcmd_ratios),
        fbox_ratio_mean = _mean(fbox_ratios),
        fbox_ratio_peak = _max(fbox_ratios),
        fbox_mag_mean   = _mean(fbox_mags),
        fbox_mag_peak   = _max(fbox_mags),
        # Per-axis breakdowns of |F_drake| (matches user's "41% lifting,
        # 69% south" report style — fractions of total force magnitude).
        fbox_axis_x_frac_mean = _mean(fbox_axis_x_frac),
        fbox_axis_y_frac_mean = _mean(fbox_axis_y_frac),
        fbox_axis_z_frac_mean = _mean(fbox_axis_z_frac),
    )


def score(metrics_list: list[dict]) -> str:
    """Bin verdict per the user's pre-registration."""
    if not metrics_list:
        return "FAIL (no contact events)"
    best_final = max((m["final_disp_mm"] for m in metrics_list), default=0.0)
    best_peak = max((m["peak_disp_mm"] for m in metrics_list), default=0.0)
    if best_final >= 5.0:
        return f"PASS (best final={best_final:.2f} mm ≥ 5 mm)"
    if best_peak >= 5.0 and best_final < 1.0:
        return (f"PARTIAL (best peak={best_peak:.2f} mm ≥ 5 mm but "
                f"final={best_final:.2f} mm < 1 mm — direction fixed, "
                f"recoil now prime mover)")
    return f"FAIL (best peak={best_peak:.2f} mm < 5 mm; best final={best_final:.2f} mm)"


def main():
    if len(sys.argv) < 2:
        print("usage: parse_beta_contact_events.py <run.log>", file=sys.stderr)
        sys.exit(2)
    path = sys.argv[1]
    rows, goal_xy = parse_log(path)
    if not rows:
        print(f"{path}: no [STEP] rows found")
        sys.exit(0)
    # g_hat = direction from box-start (origin) to goal.
    g_hat = goal_xy  # already vector from origin
    events = group_contact_events(rows)

    print(f"=== {path} ===")
    print(f"goal_xy={goal_xy}  g_hat={g_hat}")
    n_step = len(rows)
    n_c3 = sum(1 for r in rows if r.mode == "c3")
    n_contact = sum(1 for r in rows if r.mode == "c3" and r.contact == "Y")
    last_t = rows[-1].t if rows else 0.0
    print(f"steps={n_step}  c3_steps={n_c3}  contact_steps={n_contact}  "
          f"sim_t={last_t:.3f}s")
    print(f"n_contact_events={len(events)}")
    print()
    if not events:
        print("(no contact events)")
        print()
    for k, ev in enumerate(events):
        m = event_metrics(ev, g_hat)
        print(f"--- event {k}  steps {m['start_step']}..{m['end_step']}  "
              f"({m['n_steps']} steps, {m['duration_ms']:.1f} ms) ---")
        print(f"  peak_disp_along_goal = {m['peak_disp_mm']:+.3f} mm "
              f"(at step {m['peak_step']})")
        print(f"  final_disp_along_goal (5-step settle avg) = "
              f"{m['final_disp_mm']:+.3f} mm")
        print(f"  recoil_frac (1 - final/peak)             = "
              f"{m['recoil_frac']:.3f}")
        print(f"  direction-fix ratio  F_goal/|F| (f_cmd)  = "
              f"{m['fcmd_ratio_mean']:.3f}")
        print(f"  direction-fix ratio  F_goal/|F| (F_drake mean) = "
              f"{m['fbox_ratio_mean']:.3f}  (peak {m['fbox_ratio_peak']:.3f})")
        print(f"  |F_drake| mean = {m['fbox_mag_mean']:.3f} N  "
              f"(peak {m['fbox_mag_peak']:.3f} N)")
        print(f"  |F| per-axis fractions  x={m['fbox_axis_x_frac_mean']:.3f}  "
              f"y={m['fbox_axis_y_frac_mean']:.3f}  "
              f"z={m['fbox_axis_z_frac_mean']:.3f}")
        print()

    print(f"VERDICT: {score([event_metrics(ev, g_hat) for ev in events])}")


if __name__ == "__main__":
    main()
