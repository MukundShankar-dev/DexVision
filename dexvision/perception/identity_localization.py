"""Current RGB positions with conservative singleton-class visual identity.

Same-class multiplicity and an ID discontinuity within the frozen history gap
are uncertified. Rejected associations must not train the next association.
This is visual continuity, never a physical or semantic identity certificate.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

from dexvision.perception.current_localization import CurrentLocalizationBaseline
from dexvision.perception.predictive_tracker import PredictiveTracker

ENGINE = "level5/current-rgb-conservative-identity-v1"
PARENT_ENGINE = "level5/current-rgb-localization-v1"


class ConservativeIdentityTracker(PredictiveTracker):
    """Certify only one visually continuous component per class at a time.

    A new ID cannot be certified while that class's last certified ID remains
    fresh. After expiry a new ID is explicitly marked as reacquired. Neither
    thresholds nor predicted positions are changed to force an association.
    """

    def reset(self):
        super().reset()
        self.identity_anchors = {}

    def update(self, detections, timestamp):
        previous = dict(self.previous)
        velocities = dict(self.velocities)
        proposed = super().update(detections, timestamp)
        counts = Counter(t.detection.class_id for t in proposed)
        output = []
        for track in proposed:
            category = track.detection.class_id
            anchor = self.identity_anchors.get(category)
            changed = anchor is not None and anchor[0] != track.track_id
            discontinuity = changed and timestamp - anchor[1] <= self.maximum_gap_s
            ambiguous = track.ambiguous or counts[category] != 1 or discontinuity
            track = replace(
                track,
                ambiguous=ambiguous,
                reacquired=track.reacquired or bool(changed and not ambiguous),
            )
            output.append(track)
            if not ambiguous:
                self.identity_anchors[category] = track.track_id, timestamp

        # super().update has proposed history. Commit only certified outputs;
        # restore earlier history for every veto, then apply ordinary expiry.
        accepted = {t.track_id: t for t in output if not t.ambiguous}
        self.previous = {
            key: track
            for key, track in {**previous, **accepted}.items()
            if timestamp - track.timestamp <= self.maximum_gap_s
        }
        self.velocities = {
            key: self.velocities.get(key, velocities.get(key))
            if key in accepted
            else velocities.get(key)
            for key in self.previous
            if key in velocities or (key in accepted and key in self.velocities)
        }
        return tuple(output)


class IdentityLocalizationBaseline(CurrentLocalizationBaseline):
    """Exact V25 positions; only visual identity certification is amended."""

    def __init__(self, recipe, model):
        if model.get("feature_version") != ENGINE or model.get("inference_variant") != ENGINE:
            raise ValueError("Frozen conservative identity model required.")
        super().__init__(
            recipe,
            {**model, "feature_version": PARENT_ENGINE, "inference_variant": PARENT_ENGINE},
        )
        self.tracker = ConservativeIdentityTracker(
            **recipe["tracker"], **recipe["predictive_tracker"]
        )
