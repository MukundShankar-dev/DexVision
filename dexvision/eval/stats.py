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
