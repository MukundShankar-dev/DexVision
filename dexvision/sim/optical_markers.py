"""Explicit render-only optical rigs for the instrumented 5.7 experiment.

Privileged body transforms are used here ONLY to render pixels, never exported
to the estimator. These decorative marker plates have no collision/inertia and
do not modify the physical MuJoCo model or the accepted policies.
"""

from __future__ import annotations

import cv2
import numpy as np


class OpticalMarkerRenderer:
    """Rigid calibrated square labels, visible in the actual RGB image."""

    def __init__(
        self, model, data, atlas: dict, *, camera: str, width=1280, height=960
    ):
        import mujoco

        self.mujoco, self.model, self.data = mujoco, model, data
        self.atlas, self.camera = atlas, camera
        self.renderer = mujoco.Renderer(
            model, height=height, width=width, max_geom=10000
        )
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.tiles = {
            int(key): cv2.aruco.generateImageMarker(dictionary, int(key), 60)[
                ::10, ::10
            ]
            for key in atlas
        }

    def render(self):
        self.renderer.update_scene(self.data, camera=self.camera)
        scene = self.renderer.scene
        for key, marker in self.atlas.items():
            body = self.model.body(marker["body"]).id
            body_rotation = self.data.xmat[body].reshape(3, 3)
            rotation = body_rotation @ np.asarray(marker.get("rotation", np.eye(3)))
            center = self.data.xpos[body] + body_rotation @ np.array(marker["offset_m"])
            side = marker["side_m"]
            # A white quiet zone and real black/white mesh tiles; no label is
            # composited in screen space or supplied as a detection annotation.
            self._box(scene, center, rotation, [side * 0.6, side * 0.6, 0.00005], 1.0)
            tile = side / 6
            for row in range(6):
                for col in range(6):
                    offset = np.array(
                        [
                            (col + 0.5) * tile - side / 2,
                            side / 2 - (row + 0.5) * tile,
                            0.00012,
                        ]
                    )
                    self._box(
                        scene,
                        center + rotation @ offset,
                        rotation,
                        [tile / 2, tile / 2, 0.00005],
                        float(self.tiles[int(key)][row, col]) / 255,
                    )
        return self.renderer.render().copy()

    def _box(self, scene, center, rotation, size, intensity):
        if scene.ngeom >= scene.maxgeom:
            raise ValueError("Optical marker scene geometry capacity exceeded.")
        self.mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            self.mujoco.mjtGeom.mjGEOM_BOX,
            np.asarray(size),
            center,
            rotation.reshape(9),
            np.array([intensity, intensity, intensity, 1.0]),
        )
        scene.ngeom += 1

    def close(self):
        self.renderer.close()
