"""Binomial confidence intervals. Every success rate in this project is reported with one."""

from __future__ import annotations

import math


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for ``successes`` out of ``n`` (default 95%).

    Unlike the normal approximation, it stays inside [0, 1] and is sensible at 0/n and n/n.
    """
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError(f"Need 0 <= successes <= n and n > 0, got {successes}/{n}.")
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar p-value for paired outcomes on the same episodes.

    ``only_a``: episodes only A solved; ``only_b``: only B solved. Episodes both or neither
    solved carry no information about which is better. Under "no difference" the discordant
    episodes split 50/50, so the p-value is a binomial tail, doubled and capped at 1.
    """
    n, k = only_a + only_b, min(only_a, only_b)
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)
