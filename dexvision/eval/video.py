"""Render rollouts from the fixed workcell camera to MP4 (H.264, plays on macOS and Windows)."""

from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

from dexvision.sim.workcell import Workcell


class VideoRecorder:
    def __init__(
        self, workcell: Workcell, camera: str = "fixed", width: int = 640, height: int = 480
    ) -> None:
        self._workcell = workcell
        self._camera = camera
        self._renderer = mujoco.Renderer(workcell.model, height, width)
        self.frames: list[np.ndarray] = []

    def capture(self) -> None:
        self._renderer.update_scene(self._workcell.data, camera=self._camera)
        self.frames.append(self._renderer.render().copy())

    def save(self, path: Path, fps: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(path, np.stack(self.frames), fps=fps, codec="libx264", plugin="pyav")
        self.frames.clear()

    def close(self) -> None:
        self._renderer.close()
