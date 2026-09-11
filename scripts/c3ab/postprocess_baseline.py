#!/usr/bin/env python3
"""Post-process one frozen-baseline trial dir into metrics.json + trajectory.csv.

Reads state_trace.jsonl (t, q[7], obj[qw,qx,qy,qz,x,y,z], pos_err, ang_err)
and planner.log event strings. Computes per-scene obstacle clearance using the
scenario disc list and the T footprint (union of two boxes) boundary samples.
Pure logging/analysis -- no controller interaction.
"""
import json, sys, math, csv
from pathlib import Path
import numpy as np

SCEN = {
    "open_table": [],
    "single_obstacle": [[0.5, 0.0, 0.0707]],
    "box_clutter": None,   # filled from scenario_params at call time via --discs
    "ycb_clutter": [[0.5, 0.0, 0.0707], [0.37, 0.2, 0.0996]],
    "icra_sign": [],
}
# T footprint boundary samples (body frame, xy), union of crossbar+stem outline
def t_boundary(n=80):
    outline = [(-0.0445, 0.0), (-0.0445, 0.0198), (0.0445, 0.0198), (0.0445, 0.0),
               (0.0099, 0.0), (0.0099, -0.0794), (-0.0099, -0.0794), (-0.0099, 0.0), (-0.0445, 0.0)]
    pts = []
    for (x0, y0), (x1, y1) in zip(outline[:-1], outline[1:]):
        seg = max(2, int(n * math.hypot(x1-x0, y1-y0) / 0.38))
        for k in range(seg):
            a = k / seg
            pts.append((x0 + a*(x1-x0), y0 + a*(y1-y0)))
    return np.array(pts)

BOUND = t_boundary()

def yaw_of(q):
    qw, qx, qy, qz = q
    return math.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))

def min_clearance(x, y, yaw, discs):
    if not discs:
        return None
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    world = BOUND @ R.T + np.array([x, y])
    best = 1e9
    for ox, oy, r in discs:
        d = np.min(np.linalg.norm(world - np.array([ox, oy]), axis=1)) - r
        best = min(best, d)
    return float(best)

def main():
    run_dir = Path(sys.argv[1]); scene = sys.argv[2]
    gx, gy, gyaw = map(float, sys.argv[3:6])
    discs = SCEN.get(scene)
    if len(sys.argv) > 6:
        discs = json.loads(sys.argv[6])
    rows = []
    for line in open(run_dir/"state_trace.jsonl"):
        d = json.loads(line)
        x, y = d["obj"][4], d["obj"][5]
        yw = yaw_of(d["obj"][:4])
        clr = min_clearance(x, y, yw, discs or [])
        rows.append(dict(t=d["t"], x=x, y=y, yaw=yw,
                         pos_err=d["pos_err"], ang_err=d["ang_err"],
                         clearance=clr))
    # planner events
    ev = dict(repositions=0, no_progress=0, buffer_overflow=0,
              switch_to_c3=0, repos_collide=0)
    plog = run_dir/"planner.log"
    if plog.exists():
        txt = plog.read_text(errors="replace")
        ev["no_progress"] = txt.count("Repositioning after not making progress")
        ev["switch_to_c3"] = txt.count("Switching to C3")
        ev["buffer_overflow"] = txt.count("Unsuccessful sample buffer overflow")
        ev["repos_collide"] = txt.count("Previous repositioning target in collision")
        ev["repositions"] = ev["no_progress"] + ev["repos_collide"]
    fin = json.loads([l for l in open(run_dir/"success.log").read().splitlines()
                      if l.startswith("FINAL")][-1][6:])
    pe = np.array([r["pos_err"] for r in rows]); ae = np.array([r["ang_err"] for r in rows])
    clrs = [r["clearance"] for r in rows if r["clearance"] is not None]
    # planar xy error (recorder pos_err is 3D; also compute planar)
    pxy = np.array([math.hypot(r["x"]-gx, r["y"]-gy) for r in rows])
    metrics = {
        "scene": scene, "goal": [gx, gy, gyaw],
        "success_002": fin["first_success_t"] is not None,
        "first_success_t": fin["first_success_t"],
        "end_t": rows[-1]["t"],
        "final_pos_err": float(pe[-1]), "final_xy_err": float(pxy[-1]),
        "final_yaw_err": float(ae[-1]),
        "best_pos_err": float(pe.min()), "best_xy_err": float(pxy.min()),
        "best_yaw_err": float(ae.min()),
        "position_success_002": bool(pxy[-1] <= 0.02),
        "position_success_005": bool(pxy[-1] <= 0.05),
        "orientation_success_01": bool(ae[-1] <= 0.10),
        "full_success_oim_005": bool(np.any((pxy <= 0.05) & (ae <= 0.10))),
        "min_clearance": (min(clrs) if clrs else None),
        "initial_clearance": (clrs[0] if clrs else None),
        "collision": (min(clrs) < 0.0 if clrs else False),
        "events": ev,
        "n_samples": fin.get("samples"),
    }
    with open(run_dir/"metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(run_dir/"trajectory.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(json.dumps(metrics))

if __name__ == "__main__":
    main()
