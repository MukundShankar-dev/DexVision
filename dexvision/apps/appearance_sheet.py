"""Contact sheet of policy-camera frames under varied appearance, for the owner's review (A1).

    python -m dexvision.apps.appearance_sheet --task place_vision --seeds 40 --out sheet.png

Each tile is one seed's first frame after reset, both policy cameras side by side. The episode
is the task's own (object, pose, goal); only its appearance is varied.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from dexvision.eval.evaluator import spec_rig
from dexvision.sim.appearance import FACTORS
from dexvision.sim.workcell import Workcell
from dexvision.tasks import TASK_NAMES, make_task


def sheet(task, seeds: range, columns: int, scale: int) -> np.ndarray:
    """[rows * H, columns * W, 3] uint8: one tile per seed, cameras side by side, 2 px gaps."""
    workcell = Workcell()
    rig = spec_rig(workcell, task.spec)
    tiles = []
    for seed in seeds:
        task.reset(workcell, seed)
        rig.start_episode(seed)
        images = rig.render()
        tiles.append(np.concatenate([images[c] for c in task.spec.cameras], axis=1))
    rig.close()
    h, w, _ = tiles[0].shape
    rows = -(-len(tiles) // columns)
    out = np.full((rows * (h + 2), columns * (w + 2), 3), 255, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, columns)
        out[r * (h + 2):r * (h + 2) + h, c * (w + 2):c * (w + 2) + w] = tile
    return np.repeat(np.repeat(out, scale, axis=0), scale, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=TASK_NAMES, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--appearance", nargs="+", choices=FACTORS, default=list(FACTORS),
                        metavar="FACTOR", help="factors to vary (default: all)")
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--scale", type=int, default=2, help="pixel upscaling for viewing")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    task = make_task(args.task)
    if not task.spec.cameras:
        parser.error(f"{args.task} observes no cameras")
    task.spec = replace(task.spec, appearance=tuple(args.appearance))
    image = sheet(task, task.spec.seeds(args.split, args.seeds), args.columns, args.scale)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.out, image)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
