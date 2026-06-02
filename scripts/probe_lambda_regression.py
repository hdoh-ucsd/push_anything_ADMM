"""Read-only probe: characterize the lambda-bound regression (Stage 1 vs baseline).

Re-parses existing logs in altitude_hold_sweep/ and contact_guard_v2/ to answer:
  (1) Where do violations (lambda_n_max >= 5) cluster vs contact-window position?
  (2) Is the ADMM residual at violation ticks still ~1e-7 or degraded?
  (3) Is the violation rate per contact tick elevated or roughly constant?
  (4) Benign-vs-serious verdict.
  (5) Do violations track the same windows as the goal-overshoot?

Per-tick records are paired from interleaved blocks:
  [ADMM-C3+] primal: P0->P1  dual: D0->D1  mono=...  iters=...
  [C3+] step=N |u[0]|=...N lambda_n_max=L eta_n_max=...
  [GATE-CONTACT] step=M ... A_is_ee=K ... box_p=(X,Y,Z) ee_p=(...)

Each [GATE-CONTACT] block corresponds to one control loop tick; [C3+]
lines emitted between consecutive GATE-CONTACT events are the per-sample
inner-solve calls for that tick.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

LOG_DIRS = {
    "Stage1":   ("altitude_hold_sweep",  "seed{}_altitude_hold.log"),
    "Baseline": ("contact_guard_v2",     "seed{}_lcp_guard.log"),
}

REPO = Path("/root/push_anything_ADMM")

LAM_VIOL = 5.0  # lambda_n upper bound used by the LCP projection.

# ---- regex ----------------------------------------------------------------
RE_ADMM   = re.compile(r"^\[ADMM-C3\+\]\s+primal:\s*([0-9.eE+-]+)->([0-9.eE+-]+)\s+dual:\s*([0-9.eE+-]+)->([0-9.eE+-]+)")
RE_C3     = re.compile(r"^\[C3\+\]\s+step=(\d+)\s+\|u\[0\]\|=([0-9.eE+-]+)N\s+λ_n_max=([0-9.eE+-]+)\s+η_n_max=([0-9.eE+-]+)\s+primal=([0-9.eE+-]+)")
RE_GATE   = re.compile(r"^\[GATE-CONTACT\]\s+step=(\d+).*?A_is_ee=(\d).*?box_p=\(([+\-0-9.,e]+)\)\s+ee_p=\(([+\-0-9.,e]+)\)")


@dataclass
class InnerSolve:
    c3_step: int
    lam_n_max: float
    eta_n_max: float
    primal_final: float
    dual_final: float

@dataclass
class Tick:
    gate_step: int
    A_is_ee: int
    box_xy: Tuple[float, float]
    box_z: float
    ee_xyz: Tuple[float, float, float]
    inner: List[InnerSolve] = field(default_factory=list)

    @property
    def lam_n_max(self) -> float:
        return max((s.lam_n_max for s in self.inner), default=0.0)

    @property
    def violation(self) -> bool:
        return self.lam_n_max >= LAM_VIOL

    @property
    def primal_at_violation(self) -> float:
        # primal residual on the inner solve that produced the violating lam_n
        if not self.inner:
            return float("nan")
        worst = max(self.inner, key=lambda s: s.lam_n_max)
        return worst.primal_final

    @property
    def dual_at_violation(self) -> float:
        if not self.inner:
            return float("nan")
        worst = max(self.inner, key=lambda s: s.lam_n_max)
        return worst.dual_final


def parse_log(path: Path) -> List[Tick]:
    ticks: List[Tick] = []
    pending_admm: Tuple[float, float] | None = None  # (primal_final, dual_final)
    pending_inner: List[InnerSolve] = []
    with path.open() as f:
        for line in f:
            m = RE_ADMM.match(line)
            if m:
                pending_admm = (float(m.group(2)), float(m.group(4)))
                continue
            m = RE_C3.match(line)
            if m:
                pf = pending_admm[0] if pending_admm else float("nan")
                df = pending_admm[1] if pending_admm else float("nan")
                pending_inner.append(InnerSolve(
                    c3_step=int(m.group(1)),
                    lam_n_max=float(m.group(3)),
                    eta_n_max=float(m.group(4)),
                    primal_final=pf,
                    dual_final=df,
                ))
                pending_admm = None
                continue
            m = RE_GATE.match(line)
            if m:
                box_parts = [float(x) for x in m.group(3).split(",")]
                ee_parts = [float(x) for x in m.group(4).split(",")]
                ticks.append(Tick(
                    gate_step=int(m.group(1)),
                    A_is_ee=int(m.group(2)),
                    box_xy=(box_parts[0], box_parts[1]),
                    box_z=box_parts[2],
                    ee_xyz=(ee_parts[0], ee_parts[1], ee_parts[2]),
                    inner=pending_inner,
                ))
                pending_inner = []
    return ticks


def contact_windows(ticks: List[Tick]) -> List[Tuple[int, int]]:
    """Maximal runs [i_start, i_end] (inclusive) of consecutive A_is_ee==1."""
    runs = []
    i = 0
    while i < len(ticks):
        if ticks[i].A_is_ee == 1:
            j = i
            while j + 1 < len(ticks) and ticks[j + 1].A_is_ee == 1:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def window_for_index(idx: int, runs: List[Tuple[int, int]]):
    for s, e in runs:
        if s <= idx <= e:
            return (s, e)
    return None


def goal_reach_tick(ticks: List[Tick], goal_xy: Tuple[float, float], tol=0.02) -> int:
    """First tick at which |box_xy - goal_xy| <= tol. -1 if never."""
    for i, t in enumerate(ticks):
        dx = t.box_xy[0] - goal_xy[0]
        dy = t.box_xy[1] - goal_xy[1]
        if (dx * dx + dy * dy) ** 0.5 <= tol:
            return i
    return -1


def min_distance_to_goal(ticks: List[Tick], goal_xy):
    best = (float("inf"), -1)
    for i, t in enumerate(ticks):
        d = ((t.box_xy[0] - goal_xy[0]) ** 2 + (t.box_xy[1] - goal_xy[1]) ** 2) ** 0.5
        if d < best[0]:
            best = (d, i)
    return best


def summarize_seed(label: str, seed: int, ticks: List[Tick], goal_xy):
    n_total = len(ticks)
    n_contact = sum(1 for t in ticks if t.A_is_ee == 1)
    violations = [(i, t) for i, t in enumerate(ticks) if t.violation]
    n_viol = len(violations)
    rate_per_contact_tick = (n_viol / n_contact) if n_contact else 0.0
    runs = contact_windows(ticks)
    n_runs = len(runs)
    if runs:
        run_lens = [e - s + 1 for s, e in runs]
        mean_run = sum(run_lens) / len(run_lens)
        max_run = max(run_lens)
    else:
        mean_run = 0
        max_run = 0
    # residual stats: at violation vs contact-non-violation
    prim_v = [t.primal_at_violation for _, t in violations]
    dual_v = [t.dual_at_violation for _, t in violations]
    contact_nonviol = [t for t in ticks if t.A_is_ee == 1 and not t.violation]
    prim_c = [t.primal_at_violation for t in contact_nonviol]  # uses max-lam inner solve
    dual_c = [t.dual_at_violation for t in contact_nonviol]

    def med(xs):
        if not xs:
            return float("nan")
        xs = sorted(xs)
        n = len(xs)
        return xs[n // 2] if n % 2 == 1 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])

    gtick = goal_reach_tick(ticks, goal_xy)
    min_d, min_d_tick = min_distance_to_goal(ticks, goal_xy)
    # Cluster violations by window
    viol_in_window = 0
    viol_window_pos = []  # fractional position within the window where violation sits
    for i, t in violations:
        w = window_for_index(i, runs)
        if w is None:
            continue
        viol_in_window += 1
        s, e = w
        pos = (i - s) / max(1, e - s)
        viol_window_pos.append(pos)
    viol_outside = n_viol - viol_in_window

    print(f"  {label} seed={seed}")
    print(f"    ticks={n_total}  contact_ticks(A_is_ee=1)={n_contact}  violations(>=5)={n_viol}")
    print(f"    violation_rate_per_contact_tick = {rate_per_contact_tick:.4f}")
    print(f"    contact_runs={n_runs}  mean_run_len={mean_run:.1f}  max_run_len={max_run}")
    print(f"    primal_final  median: contact-non-viol={med(prim_c):.3e}  violation={med(prim_v):.3e}")
    print(f"    dual_final    median: contact-non-viol={med(dual_c):.3e}  violation={med(dual_v):.3e}")
    print(f"    violations inside contact-window: {viol_in_window}/{n_viol}")
    if viol_window_pos:
        avg_pos = sum(viol_window_pos) / len(viol_window_pos)
        print(f"    mean fractional position within window (0=start,1=end) = {avg_pos:.2f}")
    print(f"    goal_xy={goal_xy}  goal_reach_tick(<=2cm)={gtick}  min_dist={min_d:.4f}m @ tick={min_d_tick}")
    return {
        "n_total": n_total,
        "n_contact": n_contact,
        "n_viol": n_viol,
        "rate": rate_per_contact_tick,
        "prim_v_med": med(prim_v),
        "prim_c_med": med(prim_c),
        "dual_v_med": med(dual_v),
        "dual_c_med": med(dual_c),
        "viol_indices": [i for i, _ in violations],
        "runs": runs,
        "min_d": min_d,
        "min_d_tick": min_d_tick,
        "goal_reach_tick": gtick,
    }


# Goal XY per task — pushing-W defaults from config/tasks.yaml (task_id=4 = W)
GOAL_XY = (-0.5, 0.0)


def main():
    summary = {}
    for label, (sub, fmt) in LOG_DIRS.items():
        print(f"=== {label} ===")
        summary[label] = {}
        for seed in [2, 4]:  # focus seeds per probe brief
            p = REPO / sub / fmt.format(seed)
            if not p.exists():
                print(f"  missing: {p}")
                continue
            ticks = parse_log(p)
            summary[label][seed] = summarize_seed(label, seed, ticks, GOAL_XY)
            # detail of first few violations
            ticks_with_viol = [(i, t) for i, t in enumerate(ticks) if t.violation]
            for idx, (i, t) in enumerate(ticks_with_viol[:6]):
                print(f"      viol[{idx}] tick={i} gate_step={t.gate_step} lam_n_max={t.lam_n_max:.3f}"
                      f"  primal_final={t.primal_at_violation:.2e}  dual_final={t.dual_at_violation:.2e}"
                      f"  box_xy=({t.box_xy[0]:+.4f},{t.box_xy[1]:+.4f})  ee_xy=({t.ee_xyz[0]:+.4f},{t.ee_xyz[1]:+.4f})")
            if len(ticks_with_viol) > 6:
                last_i, last_t = ticks_with_viol[-1]
                print(f"      ... ({len(ticks_with_viol)-6} more) last tick={last_i} lam_n_max={last_t.lam_n_max:.3f}"
                      f" box_xy=({last_t.box_xy[0]:+.4f},{last_t.box_xy[1]:+.4f})")
        print()

    # (5) shared-cause test: do violations and goal-overshoot share the same time window?
    print("=== Shared-cause test: violation cluster vs min-distance-to-goal tick ===")
    for label in LOG_DIRS:
        for seed in [2, 4]:
            s = summary[label].get(seed)
            if not s:
                continue
            if s["viol_indices"]:
                v_first = s["viol_indices"][0]
                v_last  = s["viol_indices"][-1]
                v_med   = s["viol_indices"][len(s["viol_indices"]) // 2]
                print(f"  {label} seed={seed}: violations span ticks [{v_first}..{v_last}] median={v_med}; "
                      f"min-dist-to-goal at tick={s['min_d_tick']} (dist={s['min_d']:.4f}m)")
            else:
                print(f"  {label} seed={seed}: no violations; min-dist tick={s['min_d_tick']} dist={s['min_d']:.4f}m")


if __name__ == "__main__":
    main()
