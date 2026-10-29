"""Videos of what a vision policy looks at: its camera frames with the encoder's keypoints drawn.

    python -m dexvision.apps.keypoint_video --task pick_vision --checkpoint <run>/step_N.pt \
        --seeds 100003 100017

Each episode is run with the evaluator's rules and saved as runs/videos/keypoints_<task>_<seed>_
<outcome>.mp4: the policy cameras side by side, upscaled, with the 32 keypoints of each camera
as coloured dots (the same colour is the same keypoint throughout). Pick seeds from the
``diagnose`` output, e.g. the ones only the state policy solved.
"""

from __future__ import annotations

import argparse
import colorsys
from collections.abc import Mapping
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

from dexvision import REPO_ROOT
from dexvision.eval.evaluator import camera_rig, run_episode
from dexvision.learning.policy import LearnedPolicy, load_learned_policy
from dexvision.sim.workcell import Workcell
from dexvision.tasks import TASK_NAMES, make_task

SCALE = 4


def draw(frame: np.ndarray, points: np.ndarray, crop_px: int, colours: np.ndarray) -> np.ndarray:
    """frame [H, W, 3] uint8, points [K, 2] (x, y) in [-1, 1] of the centre crop ->
    [H*SCALE, W*SCALE, 3] with a dot per keypoint."""
    big = np.repeat(np.repeat(frame, SCALE, axis=0), SCALE, axis=1)
    offset = (frame.shape[0] - crop_px) / 2
    for (x, y), colour in zip(points, colours, strict=True):
        col = int(round((offset + (x + 1) / 2 * (crop_px - 1) + 0.5) * SCALE))
        row = int(round((offset + (y + 1) / 2 * (crop_px - 1) + 0.5) * SCALE))
        big[max(row - 3, 0):row + 3, max(col - 3, 0):col + 3] = colour
    return big


class KeypointRecorder:
    """Wraps a vision policy; draws its keypoints on every observed frame."""

    def __init__(self, policy: LearnedPolicy) -> None:
        self.policy = policy
        self.name = policy.name
        trained = policy.trained
        self._vision = trained.model.vision
        self._crop = self._vision.crop_px
        k = trained.config["vision"]["keypoints"]
        self._colours = np.array([[int(255 * c) for c in colorsys.hsv_to_rgb(i / k, 1, 1)]
                                  for i in range(k)], dtype=np.uint8)
        self.frames: list[np.ndarray] = []

    def reset(self, context, seed: int) -> None:
        self.policy.reset(context, seed)
        self.frames.clear()

    @torch.no_grad()
    def act(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        cameras = self.policy.trained.cameras
        images = {c: torch.as_tensor(observation[f"rgb/{c}"])[None, None] for c in cameras}
        points = self._vision.eval().keypoints(images)
        self.frames.append(np.concatenate(
            [draw(observation[f"rgb/{c}"], points[c][0, 0].numpy(), self._crop, self._colours)
             for c in cameras], axis=1))
        return self.policy.act(observation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=TASK_NAMES, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "runs" / "videos")
    args = parser.parse_args()

    workcell = Workcell()
    task = make_task(args.task)
    policy, _ = load_learned_policy(args.checkpoint, task.spec)
    if not policy.trained.cameras:
        parser.error("the checkpoint has no cameras")
    recorder = KeypointRecorder(policy)
    rig = camera_rig(workcell, task.spec.cameras)
    for seed in args.seeds:
        result = run_episode(workcell, task, recorder, seed, rig=rig)
        outcome = "success" if result.success else (result.phase or result.failure)
        path = args.out_dir / f"keypoints_{task.spec.name}_{seed}_{outcome}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(path, np.stack(recorder.frames), fps=1.0 / workcell.control_dt,
                    codec="libx264", plugin="pyav")
        print(f"Wrote {path}")
    rig.close()


if __name__ == "__main__":
    main()
