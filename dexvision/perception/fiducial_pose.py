"""Image-only calibrated pose/identity producer for visible optical rigs.

No simulation object pose, task phase, instance association oracle or pose
fallback is an input. Missing/duplicate/back-facing/bad-fit tags fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class OpticalPose:
    object_id: str
    class_id: str
    position: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    linear_velocity: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]
    timestamp: float
    reprojection_error_px: float
    corners: tuple


class FiducialPoseEstimator:
    def __init__(
        self,
        calibration: dict,
        atlas: dict,
        *,
        maximum_error_px=1.5,
        history_samples=5,
        motion_gap_s=0.51,
    ):
        self.calibration, self.atlas = calibration, atlas
        self.k = np.asarray(calibration["intrinsics"], float)
        self.world_from_camera = np.linalg.inv(
            np.asarray(calibration["camera_from_world"])
        )
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), parameters
        )
        self.maximum_error_px, self.history_samples = maximum_error_px, history_samples
        self.motion_gap_s = motion_gap_s
        self.reset()

    def reset(self):
        self.history = {}

    def infer(self, rgb: np.ndarray, timestamp: float) -> tuple[OpticalPose, ...]:
        if (
            rgb.dtype != np.uint8
            or list(rgb.shape[:2][::-1]) != self.calibration["image_size"]
        ):
            raise ValueError(
                "Fiducial RGB must match the independently calibrated image size."
            )
        corners, ids, _ = self.detector.detectMarkers(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        )
        if ids is None:
            return ()
        flat = ids.ravel().tolist()
        grouped = {}
        for pixels, tag in zip(corners, flat, strict=True):
            if flat.count(tag) != 1 or str(tag) not in self.atlas:
                continue
            marker = self.atlas[str(tag)]
            s = marker["side_m"] / 2
            object_corners = np.array(
                [[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], float
            )
            object_corners = (
                object_corners @ np.asarray(marker.get("rotation", np.eye(3))).T
                + marker["offset_m"]
            )
            grouped.setdefault(marker["object_id"], []).append(
                (object_corners, pixels.reshape(4, 2), marker)
            )
        result = []
        for name, observations in grouped.items():
            if len(observations) < 2:
                continue
            object_corners = np.concatenate([o[0] for o in observations]).astype(float)
            image_corners = np.concatenate([o[1] for o in observations]).astype(float)
            marker = observations[0][2]
            candidates = []
            for pair in combinations(range(len(observations)), 2):
                indices = np.r_[
                    np.arange(pair[0] * 4, pair[0] * 4 + 4),
                    np.arange(pair[1] * 4, pair[1] * 4 + 4),
                ]
                try:
                    ok, rv, tv = cv2.solvePnP(
                        object_corners[indices],
                        image_corners[indices],
                        self.k,
                        None,
                        flags=cv2.SOLVEPNP_SQPNP,
                    )
                except cv2.error:
                    continue
                if not ok or tv[2, 0] <= 0:
                    continue
                projected, _ = cv2.projectPoints(object_corners, rv, tv, self.k, None)
                residual = np.linalg.norm(
                    projected.reshape(-1, 2) - image_corners, axis=1
                ).reshape(-1, 4)
                inliers = np.sqrt(np.mean(residual**2, axis=1)) <= self.maximum_error_px
                if inliers.sum() >= 2:
                    candidates.append(
                        (
                            -int(inliers.sum()),
                            float(np.mean(residual[inliers] ** 2)),
                            pair,
                            inliers,
                            rv,
                            tv,
                        )
                    )
            if not candidates:
                continue
            _, _, _, inliers, rvec, tvec = min(candidates, key=lambda c: c[:3])
            indices = np.flatnonzero(np.repeat(inliers, 4))
            object_corners, image_corners = (
                object_corners[indices],
                image_corners[indices],
            )
            try:
                ok, rvec, tvec = cv2.solvePnP(
                    object_corners,
                    image_corners,
                    self.k,
                    None,
                    flags=cv2.SOLVEPNP_SQPNP,
                )
            except cv2.error:
                continue
            if not ok or tvec[2, 0] <= 0:
                continue
            projected, _ = cv2.projectPoints(object_corners, rvec, tvec, self.k, None)
            error = float(
                np.sqrt(
                    np.mean(
                        np.sum((projected.reshape(-1, 2) - image_corners) ** 2, axis=1)
                    )
                )
            )
            if not np.isfinite(error) or error > self.maximum_error_px:
                continue
            camera_rotation, _ = cv2.Rodrigues(rvec)
            rotation = self.world_from_camera[:3, :3] @ camera_rotation
            position = (self.world_from_camera @ np.r_[tvec.ravel(), 1])[:3]
            quaternion = np.roll(Rotation.from_matrix(rotation).as_quat(), 1)
            history = self.history.get(name, [])
            history = [h for h in history if 0 < timestamp - h[0] <= self.motion_gap_s]
            history.append((timestamp, position, rotation))
            history = history[-self.history_samples :]
            self.history[name] = history
            # A bootstrap observation establishes pose but cannot invent zero
            # physical motion. Two pre-control image samples are mandatory.
            if len(history) < 2:
                continue
            times = np.array([h[0] for h in history])
            times -= times.mean()
            velocity = np.sum(
                times[:, None] * np.array([h[1] for h in history]), axis=0
            ) / (times @ times)
            elapsed = history[-1][0] - history[0][0]
            angular = (
                Rotation.from_matrix(history[-1][2] @ history[0][2].T).as_rotvec()
                / elapsed
            )
            result.append(
                OpticalPose(
                    name,
                    marker["class_id"],
                    tuple(position),
                    tuple(quaternion),
                    tuple(velocity),
                    tuple(angular),
                    timestamp,
                    error,
                    tuple(tuple(p) for p in image_corners),
                )
            )
        return tuple(result)
