"""Compare candidate warm-start semantics across sim logs.

Extracts, per run: candidate selection sequence (argmin_k), mode/switch
behaviour, timing, and the final task outcome -- then reports how far each
semantics diverges from the reference-ordered run.

Usage: python3 scripts/gpu/warmstart_compare.py <label>=<log> [...]
"""
import re
import sys
from collections import Counter


def parse(path):
    t = open(path, errors="replace").read()
    gs = re.findall(r"\[GS\] step=(\d+) mode=(\w+) switch=(\w+) "
                    r"argmin_k=(\S+) exec_src=(\S+)", t)
    perf = re.findall(r"avg_per_step_ms=([\d.]+)\s+full_solves=(\d+)"
                      r"\s+cheap_solves=(\d+)\s+switches=(\d+)", t)
    res = re.findall(r"translational_error=([\d.]+)m\s+"
                     r"rotational_error=([\d.]+)rad\s+success=(\w+)\s+"
                     r"tight_goal=(\S+)\s+loose_goal=(\S+)", t)
    iters = re.findall(r"iters=(\d+)/(\d+)", t)
    gd = [float(x) for x in re.findall(r"goal_dist[=:]\s*([\d.]+)", t)]
    costs = [float(x) for x in
             re.findall(r"curr_cost=([\d.eE+-]+)", t)]
    return dict(
        argmin=[g[3] for g in gs],
        modes=Counter(g[1] for g in gs),
        exec_src=Counter(g[4] for g in gs),
        perf=perf[-1] if perf else None,
        res=res[-1] if res else None,
        admm_iters=[int(a) for a, _ in iters],
        goal=gd, costs=costs, n_gs=len(gs))


def main():
    runs = {}
    for arg in sys.argv[1:]:
        label, path = arg.split("=", 1)
        runs[label] = parse(path)
    if not runs:
        print(__doc__)
        return

    ref_label = next(iter(runs))
    ref = runs[ref_label]

    print(f"{'run':14s} {'ms/step':>9s} {'solves':>8s} {'switch':>7s} "
          f"{'free/c3':>10s} {'trans':>8s} {'rot':>8s} {'tight':>13s}")
    print("-" * 82)
    for label, r in runs.items():
        p, s = r["perf"], r["res"]
        print(f"{label:14s} {p[0]:>9s} {p[1]:>8s} {p[3]:>7s} "
              f"{str(r['modes']['free']) + '/' + str(r['modes'].get('c3', 0)):>10s} "
              f"{s[0]:>8s} {s[1]:>8s} {s[3]:>13s}")

    print(f"\ncandidate selection vs '{ref_label}':")
    for label, r in runs.items():
        n = min(len(ref["argmin"]), len(r["argmin"]))
        if n == 0:
            print(f"  {label:14s} no [GS] lines")
            continue
        same = sum(1 for i in range(n) if ref["argmin"][i] == r["argmin"][i])
        # how long before the first divergence
        first = next((i for i in range(n)
                      if ref["argmin"][i] != r["argmin"][i]), None)
        print(f"  {label:14s} argmin agrees {same}/{n} ({100*same/n:5.1f}%)"
              f"   first divergence at tick "
              f"{first if first is not None else '-'}")

    print("\nargmin_k distribution (which candidate wins, overall):")
    for label, r in runs.items():
        c = Counter(r["argmin"])
        top = ", ".join(f"k={k}:{v}" for k, v in sorted(c.items())[:7])
        print(f"  {label:14s} {top}")

    print("\nADMM iteration counts (should be identical -- fixed at 3):")
    for label, r in runs.items():
        print(f"  {label:14s} {Counter(r['admm_iters'])}")


if __name__ == "__main__":
    main()
