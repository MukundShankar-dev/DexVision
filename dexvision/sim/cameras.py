"""Render the policy cameras (configs/cameras.yaml) from the current simulation state.

Rendering only reads the state: it never changes `MjData`, so a run with cameras is the same run
as one without.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import yaml

from dexvision import REPO_ROOT
from dexvision.sim.workcell import Workcell

DEFAULT_CONFIG = REPO_ROOT / "configs" / "cameras.yaml"


@dataclass(frozen=True)
class CameraConfig:
    cameras: tuple[str, ...]
    size_px: int
    hidden_geom_groups: tuple[int, ...]

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG) -> CameraConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(tuple(raw["cameras"]), int(raw["size_px"]), tuple(raw["hidden_geom_groups"]))


class CameraRig:
    """Offscreen renderer for the policy cameras of one workcell."""

    def __init__(self, workcell: Workcell, config: CameraConfig | None = None) -> None:
        self.config = config or CameraConfig.load()
        self._workcell = workcell
        size = self.config.size_px
        self._renderer = mujoco.Renderer(workcell.model, size, size)
        self._option = mujoco.MjvOption()
        for group in self.config.hidden_geom_groups:
            self._option.geomgroup[group] = 0
        for name in self.config.cameras:
            if mujoco.mj_name2id(workcell.model, mujoco.mjtObj.mjOBJ_CAMERA, name) < 0:
                raise ValueError(f"no camera named {name!r} in the workcell model")

    def render(self) -> dict[str, np.ndarray]:
        """Camera name -> [size_px, size_px, 3] uint8 RGB of the current state."""
        images = {}
        for name in self.config.cameras:
            self._renderer.update_scene(self._workcell.data, camera=name,
                                        scene_option=self._option)
            images[name] = self._renderer.render().copy()
        return images

    def close(self) -> None:
        self._renderer.close()
