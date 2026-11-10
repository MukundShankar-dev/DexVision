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


def position_grid(episodes: Sequence[Mapping], axes: Sequence[tuple[str, float, float]],
                  bins: int = 3) -> list[list[Cell]]:
    """Success by start position: ``bins`` x ``bins`` cells over the reset range. ``axes`` is
    two (metric, low, high); rows run along the first (index 0 = lowest), columns the second."""
    counts = np.zeros((bins, bins, 2), dtype=int)
    for e in episodes:
        cell = [int(np.clip((e["metrics"][key] - low) / (high - low) * bins, 0, bins - 1))
                for key, low, high in axes]
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


# Fixed bin edges, set before any varied-appearance result existed (docs/PLAN.md, A2).
BRIGHTNESS_EDGES = (0.6, 0.8)  # key light, range 0.4-1.0 (configs/appearance.yaml)
CONTRAST_EDGES = (20.0, 50.0)  # CIE76 colour difference between object and table


TEXTURES = ("plain", "training texture", "held-out texture")


def _bands(edges: Sequence[float]) -> list[str]:
    """Names of the bins the edges cut, low to high."""
    middle = [f"{a:g}-{b:g}" for a, b in zip(edges, edges[1:], strict=False)]
    return [f"< {edges[0]:g}", *middle, f">= {edges[-1]:g}"]


def _band(value: float, edges: Sequence[float]) -> str:
    return _bands(edges)[sum(value >= e for e in edges)]


def _texture(e: Mapping, which: str) -> str:
    if e[f"{which}_texture"] < 0:
        return TEXTURES[0]
    return TEXTURES[2] if e[f"{which}_texture_held_out"] else TEXTURES[1]


def appearance_breakdown(episodes: Sequence[Mapping]) -> dict[str, dict[str, Cell]]:
    """Success by what each episode looked like, from the ``appearance`` an evaluation with
    varied appearance records per episode: key-light brightness, object-table colour contrast,
    and whether the table and the object were plain or carried a training or held-out texture.
    A grouping appears only when the episodes recorded it; its bins run low to high."""
    groupings = {
        "key light brightness": ("key_brightness", _bands(BRIGHTNESS_EDGES), lambda a: _band(
            a["key_brightness"], BRIGHTNESS_EDGES)),
        "object-table contrast (delta E)": ("object_table_delta_e", _bands(CONTRAST_EDGES),
                                            lambda a: _band(a["object_table_delta_e"],
                                                            CONTRAST_EDGES)),
        "table": ("table_texture", TEXTURES, lambda a: _texture(a, "table")),
        "object": ("object_texture", TEXTURES, lambda a: _texture(a, "object")),
    }
    out: dict[str, dict[str, Cell]] = {}
    for title, (key, order, label) in groupings.items():
        counts: dict[str, list[int]] = {}
        for e in episodes:
            look = e.get("appearance") or {}
            if key in look:
                cell = counts.setdefault(label(look), [0, 0])
                cell[0] += int(e["success"])
                cell[1] += 1
        if counts:
            out[title] = {name: Cell(*counts[name]) for name in order if name in counts}
    return out


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
