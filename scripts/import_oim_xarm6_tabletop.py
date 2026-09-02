#!/usr/bin/env python3
"""Vendor OIM's five xArm6 tabletop MJCF scenes for Drake.

Drake's MJCF parser supports OBJ but not STL meshes.  The importer preserves
the source include layout and scene files, converts the seven xArm visual /
collision meshes to OBJ, and changes only those mesh filename extensions in
the vendored xarm6.xml.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import trimesh


SCENES = (
    "open_table.xml",
    "single_obstacle.xml",
    "shelf_gap.xml",
    "ycb_clutter.xml",
    "icra_sign.xml",
)


def import_models(source: Path, destination: Path) -> None:
    tabletop = source / "oim/models/xarm6_pusht_tabletop"
    xarm = source / "oim/models/xarm6"
    robot_config = source / "oim/configs/robots/xarm6.yaml"
    if not all((tabletop / name).is_file() for name in SCENES):
        raise FileNotFoundError(f"OIM tabletop scenes not found under {tabletop}")
    if not robot_config.is_file():
        raise FileNotFoundError(f"OIM xArm configuration not found: {robot_config}")

    scene_out = destination / "xarm6_pusht_tabletop"
    xarm_out = destination / "xarm6"
    scene_assets_out = scene_out / "assets"
    scene_assets_out.mkdir(parents=True, exist_ok=True)
    xarm_out.mkdir(parents=True, exist_ok=True)
    xarm_assets_out = xarm_out / "assets"
    xarm_assets_out.mkdir(parents=True, exist_ok=True)
    config_out = destination / "configs/robots"
    config_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(robot_config, config_out / "xarm6.yaml")

    # Includes common.xml, tee.xml, tee_start.xml, and point twins. Keeping
    # the complete small XML set makes upstream diffs and re-imports simple.
    for path in tabletop.glob("*.xml"):
        shutil.copy2(path, scene_out / path.name)
    for path in (tabletop / "assets").iterdir():
        if path.is_file():
            shutil.copy2(path, scene_assets_out / path.name)

    xarm_xml = (xarm / "xarm6.xml").read_text()
    for stl in sorted((xarm / "assets").glob("*.stl")):
        obj_name = stl.with_suffix(".obj").name
        # Drake ignores MJCF's MuJoCo-only maxhullvert attribute and otherwise
        # attempts to construct proximity data from every STL triangle.  Emit
        # the actual convex collision hull up front; it preserves the model's
        # single-convex-hull semantics while avoiding multi-gigabyte startup.
        mesh = trimesh.load_mesh(stl, process=False).convex_hull
        mesh.export(xarm_assets_out / obj_name)
        xarm_xml = xarm_xml.replace(f"assets/{stl.name}", f"assets/{obj_name}")
    (xarm_out / "xarm6.xml").write_text(xarm_xml)

    notice = destination / "UPSTREAM.md"
    notice.write_text(
        "# OIM xArm6 tabletop assets\n\n"
        "Imported from NikolaRaicevic2001/Object-Informed-Manipulation-MJX.\n"
        "The five requested MJCF scenes and their OBJ assets are unchanged.\n"
        "The seven xArm STL meshes are converted to OBJ because Drake's MJCF\n"
        "parser does not accept STL; xarm6.xml changes only those extensions.\n"
        "The source oim/configs/robots/xarm6.yaml is vendored unchanged and\n"
        "is the wrapper's controller, run, and evaluation configuration.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("external/Object-Informed-Manipulation-MJX"),
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("sim/models/oim_xarm6_tabletop"),
    )
    args = parser.parse_args()
    import_models(args.source.resolve(), args.destination.resolve())
    print(f"Imported OIM tabletop models to {args.destination}")


if __name__ == "__main__":
    main()
