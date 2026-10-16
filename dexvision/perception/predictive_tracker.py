"""Episode-local association with explicit ambiguity and elapsed-time gaps."""

from __future__ import annotations

from math import isfinite

import numpy as np
from scipy.optimize import linear_sum_assignment

from dexvision.perception.object_detector import Detection


from dexvision.perception.object_tracker import Track


class PredictiveTracker:
    """Image-centre Hungarian association; never consumes ground-truth ids.

    IDs are episode-local visual IDs, not certified workcell semantic IDs.
    Missing tracks expire after max_gap_s. No stale extrapolation is emitted.
    Equally plausible same-class associations reject instead of guessing.
    """

    def __init__(
        self,
        *,
        maximum_distance_px: float,
        maximum_gap_s: float,
        ambiguity_margin_px: float,
        area_ratio_weight_px: float = 0.0,
        maximum_prediction_px: float = 20.0,
        maximum_speed_px_s: float = 400.0,
        velocity_alpha: float = 0.5,
        minimum_motion_confidence: float = 0.8,
        log_area_ratio_cap: float = 2.0,
    ):
        if (
            not all(
                isfinite(v)
                for v in (
                    maximum_distance_px,
                    maximum_gap_s,
                    ambiguity_margin_px,
                    area_ratio_weight_px,
                )
            )
            or min(maximum_distance_px, maximum_gap_s) <= 0
            or ambiguity_margin_px < 0
            or area_ratio_weight_px < 0
        ):
            raise ValueError("Invalid tracker distance/gap/ambiguity configuration.")
        if (
            not all(
                isfinite(v)
                for v in (
                    maximum_prediction_px,
                    maximum_speed_px_s,
                    velocity_alpha,
                    minimum_motion_confidence,
                    log_area_ratio_cap,
                )
            )
            or min(maximum_prediction_px, maximum_speed_px_s, log_area_ratio_cap) <= 0
            or not 0 < velocity_alpha <= 1
            or not 0 <= minimum_motion_confidence <= 1
        ):
            raise ValueError("Invalid predictive tracking bounds.")
        self.maximum_prediction_px = maximum_prediction_px
        self.maximum_speed_px_s = maximum_speed_px_s
        self.velocity_alpha = velocity_alpha
        self.minimum_motion_confidence = minimum_motion_confidence
        self.log_area_ratio_cap = log_area_ratio_cap
        self.maximum_distance_px = maximum_distance_px
        self.maximum_gap_s = maximum_gap_s
        self.ambiguity_margin_px = ambiguity_margin_px
        self.area_ratio_weight_px = area_ratio_weight_px
        self.reset()

    def reset(self) -> None:
        self.previous = {}
        self.velocities = {}
        self.next_id = 0
        self.last_timestamp = None

    def update(
        self, detections: tuple[Detection, ...], timestamp: float
    ) -> tuple[Track, ...]:
        if not isfinite(timestamp) or timestamp < 0:
            raise ValueError("Tracker timestamp must be finite and nonnegative.")
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError(
                "Sequence timestamps must increase; reset between episodes."
            )
        old = [
            t
            for t in self.previous.values()
            if timestamp - t.timestamp <= self.maximum_gap_s
        ]
        distance = np.full((len(old), len(detections)), 1e9)
        for i, track in enumerate(old):
            for j, detection in enumerate(detections):
                if track.detection.class_id == detection.class_id:
                    dt = timestamp - track.timestamp
                    drift = self.velocities.get(track.track_id, np.zeros(2)) * dt
                    drift *= min(
                        1.0,
                        self.maximum_prediction_px
                        / max(float(np.linalg.norm(drift)), 1e-12),
                    )
                    predicted = _center(track.detection) + drift
                    geometric_distance = np.linalg.norm(predicted - _center(detection))
                    if geometric_distance <= self.maximum_distance_px:
                        area_ratio = max(1, detection.mask.sum()) / max(
                            1, track.detection.mask.sum()
                        )
                        distance[i, j] = (
                            geometric_distance
                            + self.area_ratio_weight_px
                            * min(self.log_area_ratio_cap, abs(np.log(area_ratio)))
                        )
        matches = {}
        if distance.size:
            rows, columns = linear_sum_assignment(distance)
            matches = {
                int(j): int(i)
                for i, j in zip(rows, columns)
                if distance[i, j] <= self.maximum_distance_px
            }
        output = []
        for j, detection in enumerate(detections):
            i = matches.get(j)
            # Both old->new and new->old ambiguities matter, including initial
            # overlapping components with no history.
            competitors = [
                _center(d)
                for k, d in enumerate(detections)
                if k != j and d.class_id == detection.class_id
            ]
            ambiguous = any(
                np.linalg.norm(_center(detection) - c) <= self.ambiguity_margin_px
                for c in competitors
            )
            if i is not None:
                candidate = distance[i, j]
                alternatives = np.r_[
                    np.delete(distance[i], j), np.delete(distance[:, j], i)
                ]
                ambiguous |= bool(
                    np.any(alternatives <= candidate + self.ambiguity_margin_px)
                )
                identifier = old[i].track_id
                reacquired = old[i].timestamp != self.last_timestamp
            else:
                identifier = f"visual/{self.next_id:04d}"
                self.next_id += 1
                reacquired = False
            track = Track(identifier, detection, timestamp, ambiguous, reacquired)
            output.append(track)
            # An ambiguous association cannot corrupt later valid history.
            if not ambiguous:
                prior = self.previous.get(identifier)
                if (
                    prior is not None
                    and detection.confidence >= self.minimum_motion_confidence
                    and prior.detection.confidence >= self.minimum_motion_confidence
                    and timestamp > prior.timestamp
                ):
                    velocity = (_center(detection) - _center(prior.detection)) / (
                        timestamp - prior.timestamp
                    )
                    velocity *= min(
                        1.0,
                        self.maximum_speed_px_s
                        / max(float(np.linalg.norm(velocity)), 1e-12),
                    )
                    self.velocities[identifier] = (
                        1 - self.velocity_alpha
                    ) * self.velocities.get(
                        identifier, velocity
                    ) + self.velocity_alpha * velocity
                self.previous[identifier] = track
        self.previous = {
            k: t
            for k, t in self.previous.items()
            if timestamp - t.timestamp <= self.maximum_gap_s
        }
        self.velocities = {
            k: v for k, v in self.velocities.items() if k in self.previous
        }
        self.last_timestamp = timestamp
        return tuple(output)


def _center(detection: Detection) -> np.ndarray:
    box = np.asarray(detection.box, float)
    return (box[:2] + box[2:]) / 2
