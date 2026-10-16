"""An isolated learned-position outlier cannot certify a replacement cube ID."""

import numpy as np

from dexvision.perception.guarded_body_localization import CuboidContinuityTracker
from dexvision.perception.object_detector import Detection


def detection(x, category="rigid_cuboid"):
    return Detection(category, (x, 0, x + 2, 2), np.ones((2, 2), bool), 1.0)


def tracker():
    return CuboidContinuityTracker(maximum_distance_px=60, maximum_gap_s=.51, ambiguity_margin_px=3)


def test_cube_outlier_is_invalid_without_poisoning_reliable_body_history():
    t = tracker()
    first = t.update((detection(10),), 0)[0]
    outlier = t.update((detection(200),), .17)[0]
    assert outlier.ambiguous and outlier.track_id != first.track_id
    assert t.previous[first.track_id].timestamp == 0
    restored = t.update((detection(12),), .34)[0]
    assert not restored.ambiguous and restored.track_id == first.track_id


def test_other_class_later_births_keep_original_tracking_behavior():
    t = tracker()
    first = t.update((detection(10, "rigid_flat_puck"),), 0)[0]
    later = t.update((detection(12, "rigid_flat_puck"), detection(200, "rigid_flat_puck")), .17)
    assert all(not r.ambiguous for r in later)
    assert later[0].track_id == first.track_id
