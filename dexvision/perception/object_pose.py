"""Metric silhouette hypotheses and an explicit fail-closed schema boundary."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import cv2
import numpy as np
from scipy.optimize import least_squares

from dexvision.perception.object_observations import INFERRED_PERCEPTION, ObjectObservation
from dexvision.perception.object_tracker import Track
from dexvision.sim.world_state import WORLD_STATE_VERSION, WorldState


@dataclass(frozen=True)
class PoseEstimate:
    """World/metre hypothesis with per-field availability, never a truth fill.

    Single RGB and unknown instance dimensions do not certify 6D body pose.
    Orientation is deliberately unavailable; yaw is a symmetry-aware diagnostic
    under an upright primitive assumption. Position error is scored even though
    this assumption is insufficient to qualify policy input.
    """

    track: Track
    position: tuple[float, float, float] | None
    yaw: float | None
    linear_velocity: tuple[float, float, float] | None
    orientation_wxyz: tuple[float, float, float, float] | None
    angular_velocity: tuple[float, float, float] | None
    residual_px: float | None
    rejection_reason: str

    def require_observation(self, *, now: float, minimum_confidence: float,
                            maximum_age_s: float) -> ObjectObservation:
        """Reject unsupported semantic identity/orientation without placeholders."""
        validate_delivery(self.track, now=now, minimum_confidence=minimum_confidence,
                          maximum_age_s=maximum_age_s)
        if self.rejection_reason or self.position is None or self.orientation_wxyz is None:
            raise ValueError(self.rejection_reason or "orientation_unavailable")
        raise ValueError("semantic_identity_unavailable: visual id is not a workcell id")


def validate_delivery(track: Track, *, now: float, minimum_confidence: float,
                      maximum_age_s: float) -> None:
    if (not isfinite(now) or not isfinite(minimum_confidence) or
            not 0 <= minimum_confidence <= 1 or not isfinite(maximum_age_s) or maximum_age_s < 0):
        raise ValueError("Invalid observation delivery limits.")
    if now < track.timestamp or now-track.timestamp > maximum_age_s:
        raise ValueError("stale_or_future_observation")
    if track.ambiguous:
        raise ValueError("ambiguous_identity")
    if track.detection.confidence < minimum_confidence:
        raise ValueError("low_confidence")


def world_state_from_observations(*, timestamp: float, entities: tuple[ObjectObservation, ...],
                                  robot, relations, fixtures, contacts,
                                  field_sources: dict[str, str]) -> WorldState:
    """Build the existing schema only with explicit available sensor inputs.

    Empty contacts/relations mean a *measured* empty set only with provenance.
    A caller cannot turn missing telemetry into a false/empty truth substitute.
    This adapter is unconnected to skill execution in checkpoint 5.6.
    """
    required = ("robot", "relations", "fixtures", "contacts")
    allowed = {"declared_robot_sensor", "declared_fixture_sensor", "inferred_perception"}
    if any(field_sources.get(k) not in allowed for k in required):
        raise ValueError("unsupported_world_fields: explicit measured provenance required")
    if robot is None or any(x is None for x in (relations, fixtures, contacts)):
        raise ValueError("missing_world_fields")
    if any(e.source != INFERRED_PERCEPTION or not e.valid for e in entities):
        raise ValueError("Truth or invalid entity cannot enter inferred world state")
    for entity in entities:
        entity.require_fresh(now=timestamp, maximum_age_s=.1)
        if entity.confidence < .8:
            raise ValueError("low_confidence")
    return WorldState(WORLD_STATE_VERSION, timestamp, "mujoco_world", entities,
                      relations, fixtures, robot, contacts)


class MetricPoseEstimator:
    """Fit projected upright catalog primitive to RGB component extent.

    Size is frozen nominal class geometry, never per-frame truth dimensions.
    Fitting four box edges estimates camera depth and world body centre rather
    than intersecting a visible centroid with the board. Occlusion and scale
    variation remain errors; fit residual/size ambiguity are disclosed.
    """

    def __init__(self, calibration: dict, geometry: dict, *, maximum_gap_s: float,
                 maximum_fit_evaluations: int, maximum_residual_px: float,
                 world_bounds_m: dict | None = None, orientation_anisotropy_min: float = .05,
                 yaw_residual_weight_px: float = 3., silhouette_moments: bool = False,
                 precise_moments: bool = False, optimizer_jacobian: str = "2-point",
                 pixel_edge_correction: float = .5):
        if (calibration.get("version") != "level4/visual-camera-v1" or
                calibration.get("world_frame") != "mujoco_world" or
                calibration.get("length_units") != "m"):
            raise ValueError("Unsupported camera frame/units/calibration.")
        self.k = np.asarray(calibration["intrinsics"], float)
        self.extrinsic = np.asarray(calibration["camera_from_world"], float)
        if (self.k.shape != (3, 3) or self.extrinsic.shape != (4, 4) or
                not np.isfinite(self.k).all() or not np.isfinite(self.extrinsic).all()):
            raise ValueError("Invalid camera matrices.")
        self.geometry = geometry
        self.maximum_gap_s = maximum_gap_s
        self.maximum_fit_evaluations = maximum_fit_evaluations
        self.maximum_residual_px = maximum_residual_px
        self.world_bounds_m = world_bounds_m or {"min": [-.5, -.5, -.2], "max": [.6, .5, .6]}
        self.orientation_anisotropy_min = orientation_anisotropy_min
        self.yaw_residual_weight_px = yaw_residual_weight_px
        self.silhouette_moments = silhouette_moments
        self.precise_moments = precise_moments
        self.optimizer_jacobian = optimizer_jacobian
        self.pixel_edge_correction = pixel_edge_correction
        self.reset()

    def reset(self) -> None:
        self.previous = {}

    def project(self, points: np.ndarray) -> np.ndarray:
        camera = points @ self.extrinsic[:3, :3].T + self.extrinsic[:3, 3]
        if np.any(camera[:, 2] <= 0):
            raise ValueError("Primitive is behind camera.")
        pixels = camera @ self.k.T
        return pixels[:, :2]/pixels[:, 2, None]

    def estimate(self, track: Track) -> PoseEstimate:
        detection = track.detection
        if track.ambiguous:
            return PoseEstimate(track, None, None, None, None, None, None, "ambiguous_identity")
        geometry = self.geometry.get(detection.class_id)
        if geometry is None:
            return PoseEstimate(track, None, None, None, None, None, None, "unsupported_geometry")
        box = np.asarray(detection.box, float)
        # Integer pixel centres: right/bottom exclusive edges become last pixel.
        box[2:] -= 1
        dimensions = np.asarray(geometry["half_size_m"], float)
        points = primitive_points(geometry["shape"], dimensions)
        if geometry.get("axis") == "x":
            points = points[:, [2, 1, 0]]
        center = (box[:2]+box[2:])/2
        ray = np.linalg.solve(self.k, np.r_[center, 1.])
        width = max(1, box[2]-box[0])
        depth = self.k[0, 0] * 2*dimensions[0]/width
        world_from_camera = np.linalg.inv(self.extrinsic)
        initial = (world_from_camera @ np.r_[ray*depth, 1.])[:3]
        # A box with only four edges cannot uniquely determine 3D position and
        # yaw. Add the visible silhouette covariance as an orientation cue.
        y, x = np.nonzero(detection.mask)
        target_moments = None
        if self.silhouette_moments:
            pixel_hull = cv2.convexHull(np.column_stack((x, y)).astype(np.float32)).reshape(-1, 2)
            edge = self.pixel_edge_correction
            footprint = np.concatenate([pixel_hull + [a, b] for a in (-edge, edge) for b in (-edge, edge)])
            if self.precise_moments:
                from dexvision.perception.metric_geometry import convex_polygon, polygon_features
                target_moments = polygon_features(convex_polygon(footprint))
            else:
                target_moments = silhouette_features(cv2.convexHull(footprint.astype(np.float32)))
        covariance = np.cov(np.stack((x, y)))
        observed_angle = .5*np.arctan2(2*covariance[0, 1], covariance[0, 0]-covariance[1, 1])
        eigenvalues = np.linalg.eigvalsh(covariance)
        anisotropy = (eigenvalues[-1]-eigenvalues[0])/max(eigenvalues.sum(), 1e-12)

        def residual(parameters):
            position, yaw = parameters[:3], parameters[3]
            c, s = np.cos(yaw), np.sin(yaw)
            rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
            projected = self.project(points @ rotation.T+position)
            predicted = np.r_[projected.min(0), projected.max(0)]
            if target_moments is not None and self.precise_moments:
                return polygon_features(convex_polygon(projected))-target_moments
            hull = cv2.convexHull(projected.astype(np.float32)).reshape(-1, 2)
            moments = cv2.moments(hull)
            if target_moments is not None:
                return silhouette_features(hull)-target_moments
            angle = .5*np.arctan2(2*moments["mu11"], moments["mu20"]-moments["mu02"])
            yaw_residual = (self.yaw_residual_weight_px*anisotropy*np.sin(angle-observed_angle)
                            if geometry["shape"] == "box" else 0.)
            return np.r_[predicted-box, yaw_residual]

        bounds = ([*self.world_bounds_m["min"], -np.pi], [*self.world_bounds_m["max"], np.pi])
        initial = np.clip(initial, np.asarray(bounds[0])[:3]+1e-6, np.asarray(bounds[1])[:3]-1e-6)
        yaw_starts = (0., np.pi/4) if anisotropy >= self.orientation_anisotropy_min else (0.,)
        results = [least_squares(residual, np.r_[initial, yaw], bounds=bounds,
                                 max_nfev=self.maximum_fit_evaluations, jac=self.optimizer_jacobian)
                   for yaw in yaw_starts if geometry["shape"] == "box"]
        if not results:
            results = [least_squares(residual, np.r_[initial, 0.],
                                     bounds=bounds, max_nfev=self.maximum_fit_evaluations, jac=self.optimizer_jacobian)]
        fit = min(results, key=lambda r: np.linalg.norm(r.fun))
        position = fit.x[:3]
        error = float(np.sqrt(np.mean(fit.fun[:4]**2)))
        velocity = None
        previous = self.previous.get(track.track_id)
        if previous is not None:
            dt = track.timestamp-previous[0]
            if 0 < dt <= self.maximum_gap_s:
                velocity = tuple((position-previous[1])/dt)
        self.previous[track.track_id] = (track.timestamp, position.copy())
        reason = ("silhouette_fit_rejected" if error > self.maximum_residual_px else
                  "uncertified_size_upright_assumption_and_semantic_identity")
        return PoseEstimate(track, tuple(position), float(fit.x[3]), velocity, None, None, error, reason)


def silhouette_features(hull: np.ndarray) -> np.ndarray:
    """Continuous polygon centroid/covariance in pixel units, not a body centre.

    Comparing a projected 3D primitive to these features compensates for its
    visible-surface centroid offset without intersecting that centroid with a
    board plane. Occluded shapes remain explicitly uncertified.
    """
    moments = cv2.moments(hull)
    area = max(moments["m00"], 1e-12)
    std_x = np.sqrt(max(moments["mu20"]/area, 1e-12))
    std_y = np.sqrt(max(moments["mu02"]/area, 1e-12))
    covariance = moments["mu11"]/area
    return np.array([moments["m10"]/area, moments["m01"]/area,
                     std_x, std_y, covariance/max(std_x, std_y)])


def primitive_points(shape: str, dimensions: np.ndarray) -> np.ndarray:
    if shape == "box":
        return np.asarray([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])*dimensions
    if shape != "cylinder":
        raise ValueError("Unsupported primitive shape.")
    angles = np.linspace(0, 2*np.pi, 32, endpoint=False)
    return np.asarray([[dimensions[0]*np.cos(a), dimensions[0]*np.sin(a), z]
                       for a in angles for z in (-dimensions[2], dimensions[2])])
