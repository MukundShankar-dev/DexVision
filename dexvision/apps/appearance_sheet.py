"""Contact sheet of policy-camera frames under varied appearance, for the owner's review (A1).

    python -m dexvision.apps.appearance_sheet --task place_vision --seeds 40 --out sheet.png

Each tile is one seed's first frame after reset: its policy cameras side by side, in the spec's
order, as two separate images. The episode is the task's own (object, pose, goal); only its
appearance is varied.
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

CAMERA_GAP, TILE_GAP = 4, 14  # px before upscaling


def sheet(task, seeds: range, columns: int, scale: int) -> np.ndarray:
    """[rows * H, columns * W, 3] uint8: one tile per seed. A tile is the seed's cameras side by
    side with a narrow white gap between them: the policy sees them as separate images, and
    without the gap the mirrored camera pair reads as one reflected picture."""
    workcell = Workcell()
    rig = spec_rig(workcell, task.spec)
    tiles = []
    for seed in seeds:
        task.reset(workcell, seed)
        rig.start_episode(seed)
        images = rig.render()
        size = rig.config.size_px
        gap = np.full((size, CAMERA_GAP, 3), 255, dtype=np.uint8)
        parts = [part for c in task.spec.cameras for part in (images[c], gap)][:-1]
        tiles.append(np.concatenate(parts, axis=1))
    rig.close()
    h, w, _ = tiles[0].shape
    rows = -(-len(tiles) // columns)
    out = np.full((rows * (h + TILE_GAP), columns * (w + TILE_GAP), 3), 255, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, columns)
        y, x = r * (h + TILE_GAP), c * (w + TILE_GAP)
        out[y:y + h, x:x + w] = tile
    return np.repeat(np.repeat(out, scale, axis=0), scale, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=TASK_NAMES, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--seeds", type=int, default=40)
    parser.add_argument("--appearance", nargs="+", choices=FACTORS, default=list(FACTORS),
                        metavar="FACTOR", help="factors to vary (default: all)")
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--scale", type=int, default=3, help="pixel upscaling for viewing")
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
