"""Does warm-start divergence grow with position in the candidate chain?

Under "ordered" semantics candidate k is warm-started by a chain of k prior
solves, so a candidate's exposure to the coupling grows with its index. If
divergence from "independent" concentrates on high-k candidates, the effect
scales with candidate count -- which matters directly for a GPU backend,
because batching is only worthwhile at LARGER candidate counts.

Reads two sim logs and reports, per candidate index, how often that index
wins under each semantics and how often the two disagree.
"""
import re
import sys
from collections import Counter, defaultdict


def argmins(path):
    t = open(path, errors="replace").read()
    return re.findall(r"\[GS\] step=\d+ mode=\w+ switch=\w+ argmin_k=(\S+)", t)


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    A, B = argmins(a_path), argmins(b_path)
    n = min(len(A), len(B))
    A, B = A[:n], B[:n]

    disagree_by_a = Counter()
    total_by_a = Counter()
    switched_to = defaultdict(Counter)
    for a, b in zip(A, B):
        total_by_a[a] += 1
        if a != b:
            disagree_by_a[a] += 1
            switched_to[a][b] += 1

    print(f"{n} ticks compared\n")
    print(f"{'A picks k':>10s} {'ticks':>7s} {'disagree':>9s} {'rate':>7s}   "
          f"where B went instead")
    print("-" * 74)
    for k in sorted(total_by_a, key=lambda s: (len(s), s)):
        tot, dis = total_by_a[k], disagree_by_a[k]
        dest = ", ".join(f"{d}:{c}" for d, c in switched_to[k].most_common(4))
        print(f"{k:>10s} {tot:>7d} {dis:>9d} {100*dis/tot:>6.1f}%   {dest}")

    # k=0 is the CURRENT ee placement, not a sampled candidate: it is the
    # one candidate whose problem does not depend on the sampler, so split
    # it out rather than letting it dominate the aggregate.
    non0_tot = sum(v for k, v in total_by_a.items() if k != "0")
    non0_dis = sum(v for k, v in disagree_by_a.items() if k != "0")
    z_tot, z_dis = total_by_a.get("0", 0), disagree_by_a.get("0", 0)
    print(f"\n  k=0 (current EE placement): {z_dis}/{z_tot} disagree "
          f"({100*z_dis/max(z_tot,1):.1f}%)")
    print(f"  k>0 (sampled candidates)  : {non0_dis}/{non0_tot} disagree "
          f"({100*non0_dis/max(non0_tot,1):.1f}%)")


if __name__ == "__main__":
    main()
