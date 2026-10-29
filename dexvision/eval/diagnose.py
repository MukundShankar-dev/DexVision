"""Why a policy fails: breakdowns of an evaluation report (docs/PLAN.md, V3 diagnostics).

Works on the JSON written by ``dexvision.apps.evaluate`` / ``select_checkpoint``. Nothing here
runs the simulator; every number comes from the episodes' recorded outcome, phase and metrics.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from dexvision.eval.stats import mcnemar_exact, wilson_interval


@dataclass(frozen=True)
class Cell:
    successes: int
    n: int

    @property
    def ci95(self) -> tuple[float, float]:
        return wilson_interval(self.successes, self.n) if self.n else (0.0, 1.0)


def phases(episodes: Sequence[Mapping]) -> dict[str, int]:
    """Phase label -> episode count (``success`` included), most common first."""
    return dict(Counter(e.get("phase") or ("success" if e["success"] else e["failure"])
                        for e in episodes).most_common())


def position_grid(episodes: Sequence[Mapping], center_xy: Sequence[float],
                  half_extent_xy: Sequence[float], bins: int = 3) -> list[list[Cell]]:
    """Success by start position: ``bins`` x ``bins`` cells over the reset area.
    Rows run along x (index 0 = smallest x), columns along y."""
    counts = np.zeros((bins, bins, 2), dtype=int)
    for e in episodes:
        cell = []
        for axis, key in enumerate(("start_x_m", "start_y_m")):
            low = center_xy[axis] - half_extent_xy[axis]
            fraction = (e["metrics"][key] - low) / (2 * half_extent_xy[axis])
            cell.append(int(np.clip(fraction * bins, 0, bins - 1)))
        counts[cell[0], cell[1]] += (int(e["success"]), 1)
    return [[Cell(int(s), int(n)) for s, n in row] for row in counts]


def metric_summary(episodes: Sequence[Mapping], names: Sequence[str]
                   ) -> dict[str, dict[str, float | None]]:
    """Median of each metric over successes and over failures (None when there are none).
    Metrics recorded as -1 ("never happened") are left out of the medians."""
    summary = {}
    for name in names:
        row = {}
        for label, wanted in (("success", True), ("failure", False)):
            values = [e["metrics"][name] for e in episodes
                      if e["success"] is wanted and name in e["metrics"]
                      and e["metrics"][name] != -1]
            row[label] = float(np.median(values)) if values else None
        summary[name] = row
    return summary


@dataclass(frozen=True)
class Paired:
    """Outcomes of two policies on the same seeds."""

    both: int
    only_this: int
    only_reference: int
    neither: int
    p_value: float  # exact McNemar, two-sided
    reference_only_seeds: tuple[int, ...]  # solved by the reference only: look at these first
    reference_only_phases: dict[str, int]  # how this policy failed on those seeds


def paired(episodes: Sequence[Mapping], reference: Sequence[Mapping]) -> Paired:
    """Compare on the seeds both reports share. Raises if they share none."""
    this = {e["seed"]: e for e in episodes}
    other = {e["seed"]: e for e in reference}
    seeds = sorted(this.keys() & other.keys())
    if not seeds:
        raise ValueError("The reports share no seeds")
    table = Counter((this[s]["success"], other[s]["success"]) for s in seeds)
    only_reference = [s for s in seeds if other[s]["success"] and not this[s]["success"]]
    return Paired(
        both=table[True, True], only_this=table[True, False],
        only_reference=table[False, True], neither=table[False, False],
        p_value=mcnemar_exact(table[True, False], table[False, True]),
        reference_only_seeds=tuple(only_reference),
        reference_only_phases=phases([this[s] for s in only_reference]),
    )
