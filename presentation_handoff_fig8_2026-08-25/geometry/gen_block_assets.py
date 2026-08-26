#!/usr/bin/env python3
"""Turn hand-authored box decompositions into runnable *_block tasks.

For each object in the merged spec this writes, under sim/models/<name>_block/:
  <link>.sdf   one link, N <box> visual+collision pairs (mirrors the imported
               mesh SDFs' single-link/N-geometry structure, NOT the reference
               push_t.sdf's
               two-link + fixed-joint form, so the floating-base state layout
               is identical to the mesh task)
  <link>.obj   the same boxes as triangles -- REQUIRED: the anything-lineage
               kMeshNormal sampler preprocesses <sdf dir>/<link_name>.obj, so a
               block task without one cannot use the reference sampler.

and emits a tasks.yaml block per object.

Physics values (mass, friction, mu_per_pair_type, cost, controller_inertia) are
copied from the mesh task so the ONLY difference is the geometry.
"""
import json
import os
import sys

import numpy as np
import yaml

REPO = "/root/push_anything_ADMM/.claude/worktrees/fig8-lowcom-single-goal"

HYDRO = """        <drake:proximity_properties>
          <drake:compliant_hydroelastic/>
          <drake:hydroelastic_modulus> 3.0e7 </drake:hydroelastic_modulus>
          <drake:mesh_resolution_hint> 0.02 </drake:mesh_resolution_hint>
          <drake:hunt_crossley_dissipation>10</drake:hunt_crossley_dissipation>
          <drake:mu_dynamic>{mu}</drake:mu_dynamic>
        </drake:proximity_properties>"""


def rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def composite_inertia(boxes, total_mass):
    """Uniform density across all boxes; inertia about the LINK ORIGIN."""
    vols = np.array([b["s"][0] * b["s"][1] * b["s"][2] for b in boxes])
    rho = total_mass / vols.sum()
    I = np.zeros((3, 3))
    com = np.zeros(3)
    for b, v in zip(boxes, vols):
        m = rho * v
        com += m * np.asarray(b["c"], dtype=float)
    com /= total_mass
    for b, v in zip(boxes, vols):
        m = rho * v
        sx, sy, sz = b["s"]
        Ib = np.diag([m * (sy**2 + sz**2) / 12.0,
                      m * (sx**2 + sz**2) / 12.0,
                      m * (sx**2 + sy**2) / 12.0])
        R = rot_z(float(b.get("yaw", 0.0)))
        Ib = R @ Ib @ R.T
        d = np.asarray(b["c"], dtype=float)          # about link origin
        I += Ib + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return I, com, rho


def box_obj(boxes):
    """Boxes -> triangle soup .obj (link frame). Feeds kMeshNormal."""
    V, F = [], []
    corners = np.array([[-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
                        [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1]],
                       dtype=float)
    quads = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
             (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    for b in boxes:
        base = len(V)
        R = rot_z(float(b.get("yaw", 0.0)))
        half = np.asarray(b["s"], dtype=float) / 2.0
        c = np.asarray(b["c"], dtype=float)
        for k in corners:
            V.append(c + R @ (k * half))
        for q in quads:
            # REVERSED winding: load_mesh_faces derives the face normal from
            # vertex order, and kMeshNormal offsets a sample by
            # +buffer_distance ALONG that normal. With the forward winding the
            # normals pointed INWARD, so every sample landed inside the body,
            # failed the clearance test, and the sampler returned only the EE's
            # current position -- n=1 on 100% of ticks, never entering C3+.
            F.append([base + q[0] + 1, base + q[2] + 1, base + q[1] + 1])
            F.append([base + q[0] + 1, base + q[3] + 1, base + q[2] + 1])
    lines = ["# block object, generated from the hand-authored decomposition"]
    lines += [f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" for p in V]
    lines += [f"f {a} {b_} {c_}" for a, b_, c_ in F]
    return "\n".join(lines) + "\n", np.array(V)


def build_sdf(link, boxes, mass, mu, rgba, I, com):
    out = ['<?xml version="1.0"?>',
           "<!-- Block object: hand-authored box decomposition of the imported",
           "     mesh of the same name. Single link, one <box> per piece. -->",
           '<sdf version="1.7">',
           f'  <model name="{link}">',
           f'    <link name="{link}">',
           "      <inertial>",
           f"        <pose>{com[0]:.6f} {com[1]:.6f} {com[2]:.6f} 0 0 0</pose>",
           f"        <mass>{mass}</mass>",
           "        <inertia>",
           f"          <ixx>{I[0,0]:.9g}</ixx>",
           f"          <iyy>{I[1,1]:.9g}</iyy>",
           f"          <izz>{I[2,2]:.9g}</izz>",
           f"          <ixy>{I[0,1]:.9g}</ixy>",
           f"          <ixz>{I[0,2]:.9g}</ixz>",
           f"          <iyz>{I[1,2]:.9g}</iyz>",
           "        </inertia>",
           "      </inertial>"]
    for i, b in enumerate(boxes):
        c = b["c"]
        s = b["s"]
        yaw = float(b.get("yaw", 0.0))
        pose = f"{c[0]:.6f} {c[1]:.6f} {c[2]:.6f} 0 0 {yaw:.6f}"
        size = f"{s[0]:.6f} {s[1]:.6f} {s[2]:.6f}"
        out += [f'      <visual name="box_{i}">',
                f"        <pose>{pose}</pose>",
                "        <geometry>",
                f"          <box><size>{size}</size></box>",
                "        </geometry>",
                "        <material>",
                f"          <diffuse>{rgba[0]} {rgba[1]} {rgba[2]} 1</diffuse>",
                "        </material>",
                "      </visual>",
                f'      <collision name="box_{i}_volume">',
                f"        <pose>{pose}</pose>",
                "        <geometry>",
                f"          <box><size>{size}</size></box>",
                "        </geometry>",
                HYDRO.format(mu=mu),
                "      </collision>"]
    out += ["    </link>", "  </model>", "</sdf>", ""]
    return "\n".join(out)


def footprint_contains(boxes, pt):
    for b in boxes:
        d = np.asarray(pt[:2]) - np.asarray(b["c"][:2])
        R = rot_z(-float(b.get("yaw", 0.0)))[:2, :2]
        l = R @ d
        if abs(l[0]) <= b["s"][0] / 2 + 1e-9 and abs(l[1]) <= b["s"][1] / 2 + 1e-9:
            return True
    return False


def main():
    spec = json.load(open(sys.argv[1]))
    cfg = yaml.safe_load(open(os.path.join(REPO, "config/tasks.yaml")))["tasks"]
    yaml_blocks = []
    report = []

    for name, entry in spec.items():
        boxes = entry["boxes"]
        t = cfg[name]
        link = t["link_name"]
        newname = f"{name}_block"
        d = os.path.join(REPO, "sim/models", newname)
        os.makedirs(d, exist_ok=True)

        mass = float(t["mass"])
        mu = t.get("friction", 0.3)
        # Prefer the BLOCK task's own colour when that task already
        # exists, so a block can be tinted differently from its mesh
        # twin (needed to tell two objects apart in one scene).
        rgba = cfg.get(newname, {}).get(
            "color_rgba", t.get("color_rgba", [0.6, 0.6, 0.6, 1.0]))
        I, com, rho = composite_inertia(boxes, mass)
        obj_txt, V = box_obj(boxes)
        open(os.path.join(d, f"{link}.obj"), "w").write(obj_txt)
        open(os.path.join(d, f"{link}.sdf"), "w").write(
            build_sdf(link, boxes, mass, mu, rgba, I, com))

        zmin = float(V[:, 2].min())
        init_z = -zmin
        # Witnesses are copied VERBATIM from the mesh task.
        #
        # They are reference controller-SDF AABB sphere positions (the 089aaf6
        # witness-triangle fix), so they deliberately sit at bounding-box
        # corners -- in empty space for a letter shape. An earlier version of
        # this script treated "not on material" as an error and pulled them to
        # box centres; that silently shrinks the planner's support polygon and
        # destroys the block-vs-mesh comparison. The block's z range equals the
        # mesh's exactly, so the mesh z is already right.
        wit = [list(map(float, p)) for p in t["ground_witness_points_body"]]
        moved = []

        report.append((name, len(boxes), round(rho, 1), init_z, moved))
        yaml_blocks.append((newname, name, link, newname, init_z, wit, boxes))

    print(f"{'object':20s} {'boxes':>5s} {'rho kg/m3':>10s} {'init_z':>8s}  witnesses_moved")
    for name, nb, rho, iz, moved in report:
        print(f"{name:20s} {nb:5d} {rho:10.1f} {iz:8.4f}  {moved if moved else '-'}")
    json.dump({n: {"init_z": iz, "wit": w}
               for (n, _, _, _, iz, w, _) in yaml_blocks},
              open("/root/.claude/jobs/fd18be42/tmp/blocks/derived.json", "w"),
              indent=1)
    print("\nwrote assets + /root/.claude/jobs/fd18be42/tmp/blocks/derived.json")


if __name__ == "__main__":
    main()
