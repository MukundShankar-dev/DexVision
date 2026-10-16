"""Associate current learned body centers; publish original visible components."""

from dataclasses import replace

import numpy as np

from dexvision.perception.current_localization import CurrentLocalizationBaseline
from dexvision.perception.object_detector import Detection
from dexvision.perception.object_tracker import Track

ENGINE = "level5/current-rgb-body-tracking-v1"
PARENT_ENGINE = "level5/current-rgb-localization-v1"


def associate_body_centers(
    tracker, estimates, timestamp, project, *, body_classes=("rigid_cuboid",)
):
    """Only inferred current positions and fixed calibration enter association.

    The projected boxes are private association coordinates. Original masks,
    boxes, class scores, and current learned positions are preserved in output.
    Missing/nonfinite localization cannot alter reliable association history.
    """
    association, indices = [], []
    for index, pose in enumerate(estimates):
        if pose.position is None:
            continue
        d = pose.track.detection
        if d.class_id not in body_classes:
            association.append(d)
            indices.append(index)
            continue
        center = np.asarray(project(np.asarray([pose.position])), float)
        if center.shape != (1, 2) or not np.isfinite(center).all():
            continue
        # A fixed two-pixel private box represents the projected body point;
        # its area is not used: the original inferred mask supplies area cost.
        x, y = (int(round(v)) for v in center[0])
        association.append(Detection(d.class_id, (x - 1, y - 1, x + 1, y + 1), d.mask, d.confidence))
        indices.append(index)
    matched = tracker.update(tuple(association), timestamp)
    tracks = dict(zip(indices, matched, strict=True))
    output = []
    for index, pose in enumerate(estimates):
        associated = tracks.get(index)
        if associated is None:
            track = Track(
                f"unlocalized/{timestamp:.17g}/{index}", pose.track.detection,
                timestamp, True, False,
            )
            reason = "ambiguous_identity_missing_position"
        else:
            track = replace(associated, detection=pose.track.detection)
            reason = (
                "ambiguous_identity_position_hypothesis"
                if track.ambiguous else pose.rejection_reason
            )
        output.append(replace(pose, track=track, rejection_reason=reason))
    return tuple(output)


class BodyTrackingBaseline(CurrentLocalizationBaseline):
    """Exact V25 current positions precede causal visual association."""

    def __init__(self, recipe, model):
        if model.get("feature_version") != ENGINE or model.get("inference_variant") != ENGINE:
            raise ValueError("Frozen current-body-center tracking model required.")
        super().__init__(
            recipe,
            {**model, "feature_version": PARENT_ENGINE, "inference_variant": PARENT_ENGINE},
        )
        self.body_classes = tuple(recipe["body_center_tracking_classes"])
        if self.body_classes != ("rigid_cuboid",):
            raise ValueError("V28 freezes body-center association for cuboids only.")

    def infer(self, rgb, timestamp):
        if rgb.shape != self.image_shape:
            raise ValueError("RGB size differs from frozen camera calibration.")
        detections = self.detector.detect(rgb)
        candidates = tuple(
            Track(f"current/{i}", d, timestamp, False, False)
            for i, d in enumerate(detections)
        )
        current = self.pose.estimate_frame(rgb, candidates)
        return associate_body_centers(
            self.tracker, current, timestamp, self.pose.project,
            body_classes=self.body_classes,
        )
