"""Control inputs must never be fabricated from accepted RGB hypotheses."""

from dataclasses import replace

import numpy as np
import pytest

from dexvision.learning.perception_policy import PerceptionPolicy, unavailable_fields
from dexvision.perception.object_detector import Detection
from dexvision.perception.object_pose import PoseEstimate
from dexvision.perception.object_tracker import Track
from dexvision.perception.world_estimate import inferred_world_estimate


class SpyBaseline:
    def __init__(self, *, ambiguous=False, confidence=1.):
        self.calls = []
        self.resets = 0
        self.ambiguous = ambiguous
        self.confidence = confidence

    def reset(self):
        self.resets += 1

    def infer(self, rgb, timestamp):
        self.calls.append((rgb.copy(), timestamp))
        detection = Detection("rigid_cuboid", (1, 1, 4, 4), np.ones((3, 3), bool), self.confidence)
        track = Track("visual_1", detection, timestamp, self.ambiguous, False)
        return (PoseEstimate(track, (.1, .1, .02), None, None, None, None, None, ""),)


@pytest.mark.parametrize("skill", ["reach", "pick", "place", "push", "press"])
def test_accepted_controls_reject_without_semantic_id_telemetry_or_contact(skill):
    baseline = SpyBaseline()
    policy = PerceptionPolicy(baseline, skill, incompatibility="missing measured fields")
    policy.reset()
    rgb = np.zeros((480, 640, 3), np.uint8)
    decision = policy.observe(rgb, timestamp=2., now=2.)
    assert baseline.resets == 1 and len(baseline.calls) == 1
    assert np.array_equal(baseline.calls[0][0], rgb) and baseline.calls[0][1] == 2.
    assert decision.terminal_result == "rejected_before_execution" and decision.action is None
    assert any("object_id:unavailable" in f for f in decision.unavailable_fields)
    assert any("position:hypothesis" in f for f in decision.unavailable_fields)
    assert "contacts:unavailable" in decision.unavailable_fields
    assert decision.estimate.entities[0].evidence.orientation_wxyz.value is None
    assert decision.estimate.entities[0].evidence.object_id.value is None
    assert decision.estimate.contacts.value is None
    assert decision.delivery_timestamp >= 2.


def test_delivery_latency_and_ambiguous_low_confidence_evidence_remain_visible(monkeypatch):
    times = iter((10., 10.2))
    monkeypatch.setattr("dexvision.learning.perception_policy.time.perf_counter", lambda: next(times))
    policy = PerceptionPolicy(SpyBaseline(ambiguous=True, confidence=.2), "push", incompatibility="contact")
    result = policy.observe(np.zeros((480, 640, 3), np.uint8), timestamp=1., now=1.)
    assert result.delivery_timestamp == pytest.approx(1.2)
    assert "world.timestamp:stale_or_future" in result.unavailable_fields
    assert any("invalid_or_ambiguous" in f for f in result.unavailable_fields)
    assert any("low_confidence" in f for f in result.unavailable_fields)


def test_truth_cannot_be_substituted_even_with_typed_outer_schema():
    world = inferred_world_estimate((), 1.)
    truth = replace(world, source="simulator_ground_truth", **{
        name: replace(getattr(world, name), producer="simulator_ground_truth")
        for name in ("robot", "relations", "fixtures", "contacts")})
    assert "world.source:evaluation_truth_prohibited" in unavailable_fields(truth, now=1.)
    with pytest.raises(ValueError, match="truth"):
        truth.require_control_state(now=1.)


@pytest.mark.parametrize("timestamp,now", [(float("nan"), 0.), (-1., 0.), (1., .9), (0., float("inf"))])
def test_bad_time_rejects_before_inference(timestamp, now):
    baseline = SpyBaseline()
    with pytest.raises(ValueError, match="finite"):
        PerceptionPolicy(baseline, "reach", incompatibility="telemetry").observe(
            np.zeros((480, 640, 3), np.uint8), timestamp=timestamp, now=now)
    assert not baseline.calls


def test_only_calibrated_rgb_can_enter():
    policy = PerceptionPolicy(SpyBaseline(), "reach", incompatibility="telemetry")
    with pytest.raises(ValueError, match="RGB"):
        policy.observe(np.zeros((480, 640, 3)), timestamp=0., now=0.)
    with pytest.raises(TypeError):
        policy.observe(np.zeros((480, 640, 3), np.uint8), timestamp=0., now=0., simulator_truth={})
