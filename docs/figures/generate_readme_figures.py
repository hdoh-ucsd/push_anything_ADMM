#!/usr/bin/env python3
"""Generate the conceptual README figures and stage the measured result plot.

The SVGs are deterministic and dependency-free. The result panel is copied
byte-for-byte from the existing Fig. 8 object campaign; this script does not
parse logs or recompute experimental measurements.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures"

INK = "#172033"
MUTED = "#5f6b7a"
BLUE = "#dbeafe"
BLUE_STROKE = "#2563eb"
ORANGE = "#ffedd5"
ORANGE_STROKE = "#ea580c"
GREEN = "#dcfce7"
GREEN_STROKE = "#16803a"
GRAY = "#f3f4f6"
GRAY_STROKE = "#6b7280"
PURPLE = "#f3e8ff"
PURPLE_STROKE = "#7e22ce"


def svg_start(width: int, height: int, title: str, description: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f"<title id=\"title\">{escape(title)}</title>",
        f"<desc id=\"desc\">{escape(description)}</desc>",
        "<defs>",
        '<marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto">',
        f'<path d="M0,0 L9,4.5 L0,9 Z" fill="{MUTED}"/>',
        "</marker>",
        "</defs>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]


def text(lines: list[str], x: float, y: float, value: str, *, size: int = 18,
         weight: int = 500, anchor: str = "middle", color: str = INK) -> None:
    parts = value.split("\n")
    first_y = y - (len(parts) - 1) * size * 0.60
    lines.append(
        f'<text x="{x}" y="{first_y}" text-anchor="{anchor}" '
        f'font-family="Inter,Segoe UI,Arial,sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}">'
    )
    for i, part in enumerate(parts):
        dy = "0" if i == 0 else str(round(size * 1.25, 1))
        lines.append(f'<tspan x="{x}" dy="{dy}">{escape(part)}</tspan>')
    lines.append("</text>")


def box(lines: list[str], x: float, y: float, w: float, h: float, label: str,
        *, fill: str = GRAY, stroke: str = GRAY_STROKE, size: int = 17,
        radius: int = 9) -> None:
    lines.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
    )
    text(lines, x + w / 2, y + h / 2 + size * 0.34, label, size=size)


def arrow(lines: list[str], x1: float, y1: float, x2: float, y2: float,
          *, dashed: bool = False) -> None:
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    lines.append(
        f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{MUTED}" '
        f'stroke-width="2" marker-end="url(#arrow)"{dash}/>'
    )


def write(name: str, lines: list[str]) -> None:
    lines.append("</svg>")
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def architecture() -> None:
    s = svg_start(
        1200, 690, "Push Anything reproduction architecture",
        "Candidate sampling and local LCS models feed C3 or C3+ MPC; the selected "
        "contact-rich or reposition trajectory is executed by OSC on a Franka in PyDrake.",
    )
    text(s, 600, 42, "Push Anything reproduction: planning and execution", size=24, weight=650)

    box(s, 65, 90, 220, 78, "PyDrake scene\nobject + Franka state", fill=BLUE, stroke=BLUE_STROKE)
    box(s, 365, 90, 220, 78, "Candidate contact /\nEE placements", fill=GREEN, stroke=GREEN_STROKE)
    box(s, 665, 90, 220, 78, "Local contact geometry\nand LCS models", fill=PURPLE, stroke=PURPLE_STROKE)
    box(s, 965, 90, 170, 78, "Candidate\nobjectives", fill=ORANGE, stroke=ORANGE_STROKE)
    arrow(s, 285, 129, 365, 129)
    arrow(s, 585, 129, 665, 129)
    arrow(s, 885, 129, 965, 129)

    box(s, 720, 240, 260, 92, "C3+ (default) / C3\ncontact-implicit MPC\nADMM inner solve", fill=ORANGE, stroke=ORANGE_STROKE)
    arrow(s, 800, 168, 800, 240)
    arrow(s, 1050, 168, 940, 240)

    box(s, 330, 240, 260, 92, "Sampling-C3 dispatcher\nselect mode and target", fill=GREEN, stroke=GREEN_STROKE)
    s.append(
        f'<path d="M720,315 L645,350 L540,332" fill="none" stroke="{MUTED}" '
        'stroke-width="2" marker-end="url(#arrow)"/>'
    )

    box(s, 160, 405, 260, 82, "Contact-free branch\nreposition trajectory", fill=GRAY, stroke=GRAY_STROKE)
    box(s, 520, 405, 260, 82, "Contact-rich branch\nplanned MPC trajectory", fill=ORANGE, stroke=ORANGE_STROKE)
    arrow(s, 410, 332, 290, 405)
    arrow(s, 510, 332, 650, 405)

    box(s, 390, 545, 260, 82, "Operational-space control\n1 kHz Franka execution", fill=BLUE, stroke=BLUE_STROKE)
    arrow(s, 290, 487, 455, 545)
    arrow(s, 650, 487, 585, 545)
    box(s, 760, 545, 260, 82, "Apply first interval\nand advance simulation", fill=BLUE, stroke=BLUE_STROKE)
    arrow(s, 650, 586, 760, 586)
    arrow(s, 890, 627, 890, 660)
    s.append(f'<path d="M890,660 L35,660 L35,129 L65,129" fill="none" stroke="{MUTED}" stroke-width="2" marker-end="url(#arrow)"/>')
    text(s, 250, 650, "receding-horizon feedback", size=14, color=MUTED)
    write("system_architecture.svg", s)


def solver_flow() -> None:
    s = svg_start(
        1200, 650, "C3+ solver flow",
        "Current state and local LCS linearization feed a stacked C3+ QP; ADMM alternates "
        "a global QP update, componentwise complementarity projection, and dual and penalty updates.",
    )
    text(s, 600, 42, "C3+ inner solve for one local candidate", size=24, weight=650)
    box(s, 45, 95, 205, 82, "Current state +\nnominal reference", fill=BLUE, stroke=BLUE_STROKE)
    box(s, 315, 95, 205, 82, "Local LCS\nA, B, D, d; E, F, H, c", fill=PURPLE, stroke=PURPLE_STROKE, size=16)
    box(s, 585, 95, 230, 82, "Stacked C3+ QP\nη = Ex + Fλ + Hu + c", fill=ORANGE, stroke=ORANGE_STROKE, size=16)
    arrow(s, 250, 136, 315, 136)
    arrow(s, 520, 136, 585, 136)

    s.append(f'<rect x="180" y="245" width="840" height="210" rx="12" fill="#fafafa" stroke="{GRAY_STROKE}" stroke-width="2"/>')
    text(s, 215, 277, "ADMM iterations", size=18, weight=650, anchor="start")
    box(s, 230, 315, 200, 78, "Global z update\nconstrained QP", fill=BLUE, stroke=BLUE_STROKE)
    box(s, 500, 315, 200, 78, "Local δ update\ncomponentwise (λ, η)", fill=ORANGE, stroke=ORANGE_STROKE, size=16)
    box(s, 760, 315, 220, 78, "Consensus / dual update\nω and penalty scaling", fill=GREEN, stroke=GREEN_STROKE, size=15)
    arrow(s, 430, 354, 500, 354)
    arrow(s, 700, 354, 770, 354)
    s.append(f'<path d="M870,393 L870,430 L330,430 L330,393" fill="none" stroke="{MUTED}" stroke-width="2" marker-end="url(#arrow)"/>')
    arrow(s, 700, 177, 700, 245)

    box(s, 330, 520, 235, 82, "Final QP / trajectory\nand candidate objective", fill=PURPLE, stroke=PURPLE_STROKE, size=16)
    box(s, 655, 520, 215, 82, "First MPC interval\nto the executor", fill=BLUE, stroke=BLUE_STROKE)
    arrow(s, 600, 455, 475, 520)
    arrow(s, 565, 561, 655, 561)
    text(s, 600, 630, "C3 and C3+ share the LCS, cost, and receding-horizon controller interfaces.", size=15, color=MUTED)
    write("c3plus_solver_flow.svg", s)


def roadmap() -> None:
    s = svg_start(
        1200, 410, "Research roadmap",
        "Completed reproduction scope, active comparison and contact-location investigations, and planned future work are separated.",
    )
    text(s, 600, 42, "Research scope and status", size=24, weight=650)
    columns = [
        (45, "CURRENT IMPLEMENTATION", BLUE, BLUE_STROKE,
         ["Push Anything reproduction", "Planar pushing + experimental\njack orientation task", "C3 / C3+ in PyDrake"]),
        (425, "CURRENT RESEARCH", ORANGE, ORANGE_STROKE,
         ["CRISP comparison study", "Continuous contact-location\ninvestigation"]),
        (805, "PLANNED / FUTURE", GRAY, GRAY_STROKE,
         ["General 3D / SE(3)\nnon-prehensile manipulation", "Broader cube-object studies", "GPU acceleration"]),
    ]
    for x, heading, fill, stroke, items in columns:
        s.append(f'<rect x="{x}" y="90" width="350" height="260" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        text(s, x + 175, 125, heading, size=17, weight=700, color=stroke)
        y = 180
        for item in items:
            s.append(f'<circle cx="{x + 34}" cy="{y - 5}" r="5" fill="{stroke}"/>')
            text(s, x + 55, y, item, size=17, anchor="start")
            y += 55
    arrow(s, 395, 220, 425, 220)
    arrow(s, 775, 220, 805, 220, dashed=True)
    text(s, 600, 390, "Roadmap items describe research direction, not completed benchmark claims.", size=15, color=MUTED)
    write("research_roadmap.svg", s)


def stage_measured_result() -> None:
    source = ROOT / "results" / "fig8_objects" / "fig8_time_to_goal.png"
    target = OUT / "fig8_fixed_goal_result.png"
    if not source.exists():
        raise FileNotFoundError(f"measured source figure is missing: {source}")
    shutil.copyfile(source, target)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    architecture()
    solver_flow()
    roadmap()
    stage_measured_result()
    for path in sorted(OUT.glob("*")):
        if path.name != Path(__file__).name:
            print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
