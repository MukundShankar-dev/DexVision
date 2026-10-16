"""Current RGB positions with conservative visual identity continuity.

New same-class births during existing fresh history are uncertified. Existing
multiple tracks keep ordinary association and ambiguity checks. Rejected
associations must not train the next association.
This is visual continuity, never a physical or semantic identity certificate.
"""

from __future__ import annotations

from dataclasses import replace

from dexvision.perception.current_localization import CurrentLocalizationBaseline
from dexvision.perception.predictive_tracker import PredictiveTracker

ENGINE = "level5/current-rgb-conservative-identity-v2"
PARENT_ENGINE = "level5/current-rgb-localization-v1"


class ConservativeIdentityTracker(PredictiveTracker):
    """Keep existing certified visual IDs; reject uncertain replacement births.

    Multiple same-class objects present at entry can retain independent tracks.
    A new same-class birth while any old track is fresh is unsupported: without
    appearance evidence it may be an occlusion fragment or an ID replacement.
    After class-history expiry a fresh ID is explicitly marked as reacquired.
    """

    def reset(self):
        super().reset()
        self.seen_classes = set()

    def update(self, detections, timestamp):
        previous = dict(self.previous)
        velocities = dict(self.velocities)
        proposed = super().update(detections, timestamp)
        output = []
        for track in proposed:
            category = track.detection.class_id
            born = track.track_id not in previous
            recent_same_class = any(
                old.detection.class_id == category
                and timestamp - old.timestamp <= self.maximum_gap_s
                for old in previous.values()
            )
            ambiguous = track.ambiguous or (born and recent_same_class)
            track = replace(
                track,
                ambiguous=ambiguous,
                reacquired=track.reacquired or bool(
                    born and category in self.seen_classes and not ambiguous
                ),
            )
            output.append(track)
            if not ambiguous:
                self.seen_classes.add(category)

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
