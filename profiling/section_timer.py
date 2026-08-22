"""
Lightweight wall-clock accumulator for pipeline section profiling.

Usage
-----
Enable once at startup (e.g. in profile_run.py):

    import profiling.section_timer as ST
    ST.ENABLED = True

Then wrap any code block with the context manager:

    from profiling.section_timer import timed

    with timed("extract_lcs_contacts"):
        phi, J_n, J_t, mu = self.extract_lcs_contacts(ctx)

When ENABLED = False the `timed` context manager is a no-op (one
`if not ENABLED: yield; return` branch — zero allocation per call).

Print a sorted report at any point:

    ST.report()

Design
------
- Module-level globals (no class instantiation overhead).
- Thread-unsafe intentionally: all rollouts run in the same thread.
- Time source: time.perf_counter() — nanosecond resolution on all platforms.
"""
import time
from contextlib import contextmanager
from collections import defaultdict
from typing import Generator

# ---- Global state --------------------------------------------------------

ENABLED: bool = False

# DISTRIBUTIONS: keep every per-call sample, not just the running total.
# Totals alone hide tails, and tails are what matter for real-time MPC --
# the reference_reset experiment moved a mean by 1% while blowing the
# maximum OSQP iteration count up 7-12x. Off by default because retaining
# ~10^5 floats per section costs memory a normal run should not pay.
DISTRIBUTIONS: bool = False

_totals: dict[str, float] = defaultdict(float)   # name -> cumulative seconds
_counts: dict[str, int]   = defaultdict(int)      # name -> call count
_samples: dict[str, list] = defaultdict(list)     # name -> per-call seconds

# ---- Public API ----------------------------------------------------------

@contextmanager
def timed(name: str) -> Generator[None, None, None]:
    """Context manager — records wall time for `name` when ENABLED."""
    if not ENABLED:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        _totals[name] += dt
        _counts[name] += 1
        if DISTRIBUTIONS:
            _samples[name].append(dt)


def samples(name: str = None):
    """Per-call durations in seconds. Empty unless DISTRIBUTIONS was on."""
    return dict(_samples) if name is None else list(_samples.get(name, ()))


def distribution_report(top_n: int = 20) -> str:
    """Per-section latency distribution, in milliseconds.

    Reports mean/p50/p90/p95/p99/max because a mean alone cannot show a
    tail, and a tail is what breaks a real-time controller.
    """
    if not _samples:
        msg = ("[SectionTimer] no distribution data "
               "(set section_timer.DISTRIBUTIONS = True)")
        print(msg)
        return msg

    def q(xs, p):
        xs = sorted(xs)
        if not xs:
            return 0.0
        k = (len(xs) - 1) * p
        lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)

    rows = sorted(_samples.items(), key=lambda kv: -sum(kv[1]))[:top_n]
    w = 78
    lines = ["", "=" * w,
             "  Section Latency Distribution (ms per call)", "=" * w,
             f"  {'Section':<26}{'calls':>8}{'mean':>8}{'p50':>8}"
             f"{'p90':>8}{'p95':>8}{'p99':>8}{'max':>8}",
             "-" * w]
    for name, xs in rows:
        ms = [x * 1000.0 for x in xs]
        lines.append(
            f"  {name:<26}{len(ms):>8,}{sum(ms)/len(ms):>8.2f}"
            f"{q(ms,0.50):>8.2f}{q(ms,0.90):>8.2f}{q(ms,0.95):>8.2f}"
            f"{q(ms,0.99):>8.2f}{max(ms):>8.2f}")
    lines += ["=" * w, ""]
    out = "\n".join(lines)
    print(out)
    return out


def report(top_n: int = 20) -> str:
    """
    Return (and print) a formatted timing report sorted by total time.

    Parameters
    ----------
    top_n : int  Maximum number of sections to show (default 20).

    Returns
    -------
    str  The formatted report text.
    """
    if not _totals:
        msg = "[SectionTimer] No data recorded (ENABLED=False or no calls yet)."
        print(msg)
        return msg

    total_wall = sum(_totals.values())
    rows = sorted(_totals.items(), key=lambda kv: kv[1], reverse=True)[:top_n]

    lines = [
        "",
        "=" * 72,
        f"  Section Timer Report  (total tracked: {total_wall:.3f}s)",
        "=" * 72,
        f"  {'Section':<35} {'Calls':>8} {'Total(s)':>10} {'Mean(ms)':>10} {'%':>6}",
        "-" * 72,
    ]
    for name, sec in rows:
        n    = _counts[name]
        pct  = 100.0 * sec / total_wall if total_wall > 0 else 0.0
        mean = 1000.0 * sec / n if n > 0 else 0.0
        lines.append(
            f"  {name:<35} {n:>8,} {sec:>10.3f} {mean:>10.3f} {pct:>5.1f}%"
        )
    lines += ["=" * 72, ""]

    text = "\n".join(lines)
    print(text)
    return text


def reset() -> None:
    """Clear all accumulated timing data."""
    _totals.clear()
    _counts.clear()
