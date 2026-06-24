"""Stage A rotation-explosion fork resolver (READ-ONLY log analysis).

Resolves Read X (inherent / no-force-regulation) vs Read Y (velocity-impact
trajectory bug) from EXISTING Stage A run.log files. Also reports the c3
mode-flip pattern (rebuild regime split) and EE descend-velocity at contact.

Usage:
    python scripts/_analyze_stage_a_fork.py

No new sims, no port behavior changes. Pure parse of:
    [STAGE-A-TRACE] step, sim_t, mode, phi, box_xy, lam_n_ee_box, qy, qz, finished_repos
    [STEP]          ee=(x,y,z), obj=(x,y,z), target=(x,y,z), ee_stride, mode
    [STAGE-A-PWL]   step=N ... build  (rebuild events; Stage A only)
"""

import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LOG_PATHS = {
    "stage_a/seed0": ROOT / "stage_a" / "seed0" / "run.log",
    "stage_a/seed4": ROOT / "stage_a" / "seed4" / "run.log",
    "stage_a_baseline/seed0": ROOT / "stage_a_baseline" / "seed0" / "run.log",
    "stage_a_baseline/seed4": ROOT / "stage_a_baseline" / "seed4" / "run.log",
}

TRACE_RE = re.compile(
    r"\[STAGE-A-TRACE\] step=(\d+) sim_t=([\d.]+) mode=(\w+) phi=(\S+) "
    r"box_xy=([+-][\d.]+),([+-][\d.]+) lam_n_ee_box=(\S+) "
    r"qy=(\S+) qz=(\S+) finished_repos=(\d)"
)

STEP_RE = re.compile(
    r"\[STEP\] step=(\d+) mode=(\w+) t=([\d.]+)s "
    r"ee=\(([+-][\d.]+),([+-][\d.]+),([+-][\d.]+)\) "
    r"obj=\(([+-][\d.]+),([+-][\d.]+),([+-][\d.]+)\)"
)

PWL_BUILD_RE = re.compile(
    r"\[STAGE-A-PWL\] step=(\d+) sim_t=([\d.]+) build "
    r"p_start=\(([+-][\d.]+),([+-][\d.]+),([+-][\d.]+)\) "
    r"p_target=\(([+-][\d.]+),([+-][\d.]+),([+-][\d.]+)\)"
)


def _f(s):
    if s == "nan":
        return float("nan")
    return float(s)


def parse_log(path):
    trace = []          # list of dict (one per [STAGE-A-TRACE] tick)
    step_rows = {}      # step -> ee=(x,y,z)
    pwl_builds = []     # list of (step, sim_t)
    with open(path) as f:
        for line in f:
            m = TRACE_RE.match(line)
            if m:
                step, t, mode, phi, bx, by, lam, qy, qz, fr = m.groups()
                trace.append({
                    "step": int(step),
                    "sim_t": float(t),
                    "mode": mode,
                    "phi": _f(phi),
                    "box_xy": (_f(bx), _f(by)),
                    "lam_n": _f(lam),
                    "qy": _f(qy),
                    "qz": _f(qz),
                    "finished_repos": int(fr) == 1,
                })
                continue
            m = STEP_RE.match(line)
            if m:
                step, mode, t, ex, ey, ez, ox, oy, oz = m.groups()
                step_rows[int(step)] = {
                    "ee": (float(ex), float(ey), float(ez)),
                    "obj": (float(ox), float(oy), float(oz)),
                    "mode": mode,
                }
                continue
            m = PWL_BUILD_RE.match(line)
            if m:
                step, t, *_ = m.groups()
                pwl_builds.append((int(step), float(t)))
    return trace, step_rows, pwl_builds


def find_c3_episodes(trace):
    """Return list of (start_step, end_step) for each contiguous c3 run."""
    episodes = []
    cur = None
    for t in trace:
        if t["mode"] == "c3":
            if cur is None:
                cur = [t["step"], t["step"]]
            else:
                cur[1] = t["step"]
        else:
            if cur is not None:
                episodes.append(tuple(cur))
                cur = None
    if cur is not None:
        episodes.append(tuple(cur))
    return episodes


def count_mode_flips(trace):
    """Count free->c3 and c3->free transitions; return list of (step, from, to)."""
    flips = []
    for prev, cur in zip(trace[:-1], trace[1:]):
        if prev["mode"] != cur["mode"]:
            flips.append((cur["step"], prev["mode"], cur["mode"]))
    return flips


def first_contact_onset(trace, episode):
    """First step in episode where lam_n is admitted (>0 and not nan)."""
    s_start, s_end = episode
    for t in trace:
        if s_start <= t["step"] <= s_end:
            lam = t["lam_n"]
            if not math.isnan(lam) and lam > 0.0:
                return t["step"]
    return None


def first_global_admit(trace):
    """First step ANYWHERE in the run with admitted EE-BOX pair."""
    for t in trace:
        lam = t["lam_n"]
        if not math.isnan(lam) and lam > 0.0:
            return t["step"]
    return None


def first_admitted_episode(trace, episodes):
    """First c3 episode containing at least one admit tick."""
    for ep in episodes:
        if first_contact_onset(trace, ep) is not None:
            return ep
    return None


def episode_phi_summary(trace, episode):
    s, e = episode
    phis = [t["phi"] for t in trace if s <= t["step"] <= e and not math.isnan(t["phi"])]
    if not phis:
        return None
    return {
        "min_phi_m": min(phis),
        "max_phi_m": max(phis),
        "mean_phi_m": sum(phis) / len(phis),
        "n": len(phis),
    }


def qz_qy_around(trace, center_step, window=5):
    """Return list of (step, qz, qy) in [center-window, center+window]."""
    out = []
    for t in trace:
        if abs(t["step"] - center_step) <= window:
            out.append((t["step"], t["qz"], t["qy"]))
    return out


def slope(rows, key_idx):
    """OLS slope of rows[i][key_idx] vs rows[i][0]."""
    n = len(rows)
    if n < 2:
        return float("nan")
    sx = sum(r[0] for r in rows)
    sy = sum(r[key_idx] for r in rows)
    sxy = sum(r[0] * r[key_idx] for r in rows)
    sxx = sum(r[0] * r[0] for r in rows)
    den = n * sxx - sx * sx
    if den == 0:
        return float("nan")
    return (n * sxy - sx * sy) / den


def fork_diagnose_qz(trace, episode):
    """Compute the spike-vs-ramp signature for the first c3 episode.

    Returns dict with:
      - onset_step
      - qz_pre_5  : avg |qz| in [onset-5, onset-1]
      - qz_post_5 : avg |qz| in [onset+1, onset+5]
      - jump_5_5  : qz_post_5 - qz_pre_5   (sharp jump at contact = SPIKE)
      - slope_episode: OLS slope of |qz| vs step over full episode (rad/tick)
      - episode_qz_total: max |qz| during episode
      - slope_post_first10: slope from onset to onset+10 (early post-contact rate)
      - readY_score, readX_score
    """
    onset = first_contact_onset(trace, episode)
    if onset is None:
        return None
    s_start, s_end = episode
    # |qz| series in episode
    rows = [
        (t["step"], abs(t["qz"]), abs(t["qy"]))
        for t in trace
        if s_start <= t["step"] <= s_end
    ]
    pre = [r for r in rows if onset - 5 <= r[0] <= onset - 1]
    post = [r for r in rows if onset + 1 <= r[0] <= onset + 5]
    qz_pre = sum(r[1] for r in pre) / len(pre) if pre else float("nan")
    qz_post = sum(r[1] for r in post) / len(post) if post else float("nan")
    jump = qz_post - qz_pre if (pre and post) else float("nan")
    slope_ep = slope(rows, 1)
    post10 = [r for r in rows if onset <= r[0] <= onset + 10]
    slope_post10 = slope(post10, 1) if len(post10) >= 2 else float("nan")
    qz_max_ep = max((r[1] for r in rows), default=float("nan"))

    # Heuristic: SPIKE if early post-contact slope ≥ 3× episode-wide slope
    # AND jump_5_5 is ≥ 10% of episode max.
    ratio = (slope_post10 / slope_ep) if (slope_ep and not math.isnan(slope_ep) and slope_ep > 0) else float("nan")
    jump_frac = (jump / qz_max_ep) if (qz_max_ep > 0 and not math.isnan(jump)) else float("nan")

    if not math.isnan(ratio) and not math.isnan(jump_frac):
        spike_like = (ratio >= 3.0) and (jump_frac >= 0.10)
        ramp_like = (ratio <= 1.5) and (jump_frac <= 0.05)
    else:
        spike_like = ramp_like = False

    return {
        "onset_step": onset,
        "episode": (s_start, s_end),
        "qz_pre_5_mean": qz_pre,
        "qz_post_5_mean": qz_post,
        "jump_5_5": jump,
        "slope_full_ep_per_tick": slope_ep,
        "slope_post10_per_tick": slope_post10,
        "ratio_post10_to_ep": ratio,
        "jump_fraction_of_max": jump_frac,
        "qz_max_in_ep": qz_max_ep,
        "verdict": ("SPIKE→READ_Y" if spike_like else
                    "RAMP→READ_X" if ramp_like else
                    "MIXED/ambiguous"),
    }


def ee_descend_velocity(trace, step_rows, onset_step, lookback=5):
    """Finite-difference EE z and ||vEE|| over [onset-lookback, onset-1].

    dt = 0.01 s (100 Hz control loop).
    Returns dict with vz_mean_mps, vxy_mean_mps, vmag_mean_mps over the
    lookback window.
    """
    if onset_step is None:
        return None
    dt = 0.01
    samples = []
    for s in range(onset_step - lookback, onset_step):
        if s in step_rows and (s - 1) in step_rows:
            ex0, ey0, ez0 = step_rows[s - 1]["ee"]
            ex1, ey1, ez1 = step_rows[s]["ee"]
            vx = (ex1 - ex0) / dt
            vy = (ey1 - ey0) / dt
            vz = (ez1 - ez0) / dt
            samples.append((vx, vy, vz))
    if not samples:
        return None
    vz_mean = sum(s[2] for s in samples) / len(samples)
    vxy_mean = sum(math.hypot(s[0], s[1]) for s in samples) / len(samples)
    vmag_mean = sum(math.sqrt(s[0]**2 + s[1]**2 + s[2]**2) for s in samples) / len(samples)
    # Also at-contact-instant (the single step just before onset)
    vz_at = samples[-1][2]
    vmag_at = math.sqrt(samples[-1][0]**2 + samples[-1][1]**2 + samples[-1][2]**2)
    return {
        "lookback_ticks": lookback,
        "dt_s": dt,
        "vz_mean_mps": vz_mean,
        "vxy_mean_mps": vxy_mean,
        "vmag_mean_mps": vmag_mean,
        "vz_at_step_before_onset_mps": vz_at,
        "vmag_at_step_before_onset_mps": vmag_at,
        "n_samples": len(samples),
    }


def rebuild_regime_split(pwl_builds, flips, near_thresh=2):
    """Classify each rebuild as 'at-flip' (within ±near_thresh of any mode
    transition) vs 'healthy' (≥10 ticks from nearest flip). Anything in
    between is 'gray'."""
    flip_steps = [f[0] for f in flips]
    regimes = {"at_flip": 0, "healthy": 0, "gray": 0}
    spacings = []
    prev_step = None
    detail = []
    for (step, _t) in pwl_builds:
        if not flip_steps:
            nearest = None
            regime = "healthy"
        else:
            nearest = min(abs(step - fs) for fs in flip_steps)
            if nearest <= near_thresh:
                regime = "at_flip"
            elif nearest >= 10:
                regime = "healthy"
            else:
                regime = "gray"
        regimes[regime] += 1
        if prev_step is not None:
            spacings.append(step - prev_step)
        prev_step = step
        detail.append((step, regime, nearest))
    return regimes, spacings, detail


def report_log(label, path):
    print(f"\n========== {label} ==========")
    print(f"path: {path}")
    if not path.exists():
        print("  NOT FOUND")
        return None
    trace, step_rows, pwl_builds = parse_log(path)
    print(f"  trace ticks: {len(trace)} | step rows: {len(step_rows)} | PWL rebuilds: {len(pwl_builds)}")

    episodes = find_c3_episodes(trace)
    print(f"  c3 episodes (n={len(episodes)}):")
    for ep in episodes[:10]:
        print(f"    step {ep[0]:4d} .. {ep[1]:4d}  (len={ep[1]-ep[0]+1})")
    if len(episodes) > 10:
        print(f"    ... +{len(episodes)-10} more")

    flips = count_mode_flips(trace)
    free_to_c3 = sum(1 for f in flips if f[1] == "free" and f[2] == "c3")
    c3_to_free = sum(1 for f in flips if f[1] == "c3" and f[2] == "free")
    print(f"  mode flips: free→c3 = {free_to_c3}, c3→free = {c3_to_free}, total = {len(flips)}")

    # (1) qz spike-vs-ramp — FIRST c3 episode + first ADMITTED episode + all
    global_onset = first_global_admit(trace)
    print(f"  first global admit (lam_n>0) at step = {global_onset}")
    if episodes:
        first_ep_phi = episode_phi_summary(trace, episodes[0])
        if first_ep_phi:
            print(f"  first c3 episode {episodes[0]} phi summary: "
                  f"min={first_ep_phi['min_phi_m']*1000:.2f}mm  "
                  f"mean={first_ep_phi['mean_phi_m']*1000:.2f}mm  "
                  f"max={first_ep_phi['max_phi_m']*1000:.2f}mm  (admit_thresh=2mm)")
        diag = fork_diagnose_qz(trace, episodes[0])
        if diag:
            print(f"  [1a] |qz| spike-vs-ramp on FIRST c3 episode {episodes[0]}:")
            for k, v in diag.items():
                if isinstance(v, float):
                    print(f"       {k:30s} = {v:+.6f}")
                else:
                    print(f"       {k:30s} = {v}")
        else:
            print(f"  [1a] First c3 episode {episodes[0]} — no admit in episode, "
                  f"can't anchor a spike test.")

        # Try the first ADMITTED episode
        first_adm_ep = first_admitted_episode(trace, episodes)
        if first_adm_ep and first_adm_ep != episodes[0]:
            diag2 = fork_diagnose_qz(trace, first_adm_ep)
            print(f"  [1b] |qz| spike-vs-ramp on FIRST ADMITTED c3 episode {first_adm_ep}:")
            for k, v in diag2.items():
                if isinstance(v, float):
                    print(f"       {k:30s} = {v:+.6f}")
                else:
                    print(f"       {k:30s} = {v}")

        # Per-episode signature across all admitted episodes
        print("  [1c] |qz| signature across ALL admitted c3 episodes:")
        for i, ep in enumerate(episodes):
            d = fork_diagnose_qz(trace, ep)
            if d is None:
                continue
            print(f"       ep#{i:2d} step {ep[0]:4d}-{ep[1]:4d}  "
                  f"onset@{d['onset_step']:4d}  "
                  f"qz_pre={d['qz_pre_5_mean']:+.5f}  "
                  f"qz_post={d['qz_post_5_mean']:+.5f}  "
                  f"jump5={d['jump_5_5']:+.5f}  "
                  f"qz_max={d['qz_max_in_ep']:+.5f}  "
                  f"verdict={d['verdict']}")

    # (2) c3 mode-flip pattern: rebuild regime split
    if pwl_builds:
        regimes, spacings, detail = rebuild_regime_split(pwl_builds, flips)
        print(f"  [2] Rebuild regime split (n_builds={len(pwl_builds)}):")
        print(f"      at-flip (≤2 ticks of any mode transition): {regimes['at_flip']}")
        print(f"      gray    (3-9 ticks from nearest flip):     {regimes['gray']}")
        print(f"      healthy (≥10 ticks from any flip):         {regimes['healthy']}")
        if spacings:
            spacings_sorted = sorted(spacings)
            n = len(spacings_sorted)
            median = spacings_sorted[n // 2]
            print(f"      rebuild→rebuild spacing (ticks): min={min(spacings)} "
                  f"median={median} max={max(spacings)}")
        # Show all rebuild events:
        print("      rebuild detail (step, regime, ticks-to-nearest-flip):")
        for step, regime, near in detail:
            print(f"        step={step:4d}  {regime:8s}  near_flip={near}")
    else:
        print("  [2] No [STAGE-A-PWL] rebuild events in this log (baseline).")

    # (3) EE descend-velocity at FIRST GLOBAL contact onset
    if global_onset is not None:
        vel = ee_descend_velocity(trace, step_rows, global_onset, lookback=5)
        print(f"  [3] EE velocity in 5 ticks BEFORE first GLOBAL contact onset (step {global_onset}):")
        if vel is None:
            print("      insufficient EE position rows")
        else:
            for k, v in vel.items():
                if isinstance(v, float):
                    print(f"      {k:35s} = {v:+.4f}")
                else:
                    print(f"      {k:35s} = {v}")
        # Also EE z trajectory in lookback window
        print(f"      EE z(m) in [onset-7, onset+2]:")
        for s in range(global_onset - 7, global_onset + 3):
            if s in step_rows:
                ez = step_rows[s]["ee"][2]
                print(f"        step={s:4d}  z={ez:+.4f}m")
    else:
        print("  [3] No global contact onset in this run.")

    return {
        "trace_n": len(trace),
        "episodes": episodes,
        "flips": flips,
        "pwl_builds": pwl_builds,
    }


def main():
    for label, path in LOG_PATHS.items():
        report_log(label, path)


if __name__ == "__main__":
    main()
