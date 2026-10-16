"""5.6 field availability/provenance tests; no simulator, camera, GPU or GUI."""

from dataclasses import replace

import numpy as np
import pytest

from dexvision.evaluation.perception_audit import truth_observation
from dexvision.perception.object_detector import Detection
from dexvision.perception.object_observations import (
    INFERRED_PERCEPTION,
    SIMULATOR_GROUND_TRUTH,
    ObjectObservation,
    ObjectObservationError,
)
from dexvision.perception.object_pose import PoseEstimate
from dexvision.perception.object_tracker import Track
from dexvision.perception.observation_boundary import (
    BODY_ORIGIN,
    WORLD_FRAME,
    BoundaryLimits,
    FieldEvidence,
    ObjectEvidence,
    WorldEvidence,
    evidence_from_pose,
)
from dexvision.sim.world_state import EntityRelation, RobotObservation, WorldState


def evidence(value, *, frame=WORLD_FRAME, units="m", origin=BODY_ORIGIN,
             producer=INFERRED_PERCEPTION, timestamp=1., availability=None):
    return FieldEvidence(value, availability or ("unavailable" if value is None else "available"),
                         producer, timestamp, .9, frame, units, origin)


def object_evidence():
    # Explicit synthetic certified fields, not measurements supplied by V8.
    return ObjectEvidence(
        evidence("item", frame="semantic", units="id", origin="semantic_identity"),
        evidence("rigid_cuboid", frame="semantic", units="id", origin="semantic_identity"),
        evidence((.1, 0., .02)), evidence((1., 0., 0., 0.), units="wxyz_unit"),
        evidence(None, units="m/s"), evidence(None, units="rad/s"),
    )


def world_evidence():
    robot = RobotObservation((0., 0., 0.), (1., 0., 0., 0.),
                             (.1, 0., .1), (1., 0., 0., 0.), True)
    return WorldEvidence(
        (object_evidence(),),
        evidence(robot, units="m,wxyz_unit", origin="named_robot_bodies", producer="declared_robot_sensor"),
        evidence((EntityRelation("item", "board", None, None),), units="relation", origin="named_entities"),
        evidence((), units="m,bool", origin="fixture_rest_axis", producer="declared_fixture_sensor"),
        evidence((), units="id_pairs", origin="named_entities"),
    )


def test_shared_frozen_types_and_optional_motion_with_provenance_retained():
    subject = world_evidence()
    world = subject.require_world_state(now=1.02)
    assert type(world) is WorldState
    item = world.require_entity("item")
    truth = truth_observation({"object_id": "item", "class_id": "rigid_cuboid", "pose_6d": {
        "translation_m": [.1, 0., .02], "rotation_matrix": np.eye(3).tolist()}}, 1.)
    assert type(item) is type(truth) is ObjectObservation
    assert item.source == INFERRED_PERCEPTION and truth.source == SIMULATOR_GROUND_TRUTH
    assert item.position == truth.position and item.orientation_wxyz == truth.orientation_wxyz
    assert item.linear_velocity is None and item.angular_velocity is None
    assert subject.entities[0].linear_velocity.availability == "unavailable"
    assert world.contacts == ()  # Measured empty, not missing contact telemetry.


@pytest.mark.parametrize("field", ["object_id", "class_id", "position", "orientation_wxyz"])
@pytest.mark.parametrize("change", [
    {"availability": "hypothesis"}, {"valid": False}, {"confidence": .79},
    {"timestamp": .89}, {"timestamp": 1.01}, {"producer": SIMULATOR_GROUND_TRUTH},
    {"frame": "camera"}, {"units": "mm"}, {"body_origin": "visible_mask_centroid"},
])
def test_every_required_object_field_fails_closed(field, change):
    subject = object_evidence()
    subject = replace(subject, **{field: replace(getattr(subject, field), **change)})
    with pytest.raises(ObjectObservationError, match=field):
        subject.require_observation(now=1.)


@pytest.mark.parametrize("field", ["robot", "relations", "fixtures", "contacts"])
@pytest.mark.parametrize("change", [
    {"timestamp": .89}, {"timestamp": 1.01}, {"confidence": .79},
    {"valid": False}, {"producer": SIMULATOR_GROUND_TRUTH},
    {"availability": "unavailable", "value": None},
])
def test_world_fields_need_their_own_fresh_valid_producer(field, change):
    subject = world_evidence()
    subject = replace(subject, **{field: replace(getattr(subject, field), **change)})
    with pytest.raises(ObjectObservationError, match=field):
        subject.require_world_state(now=1.)


def test_empty_relations_cannot_hide_missing_entity_state():
    subject = world_evidence()
    with pytest.raises(ObjectObservationError, match="every entity"):
        replace(subject, relations=replace(subject.relations, value=())).require_world_state(now=1.)


@pytest.mark.parametrize("field", ["linear_velocity", "angular_velocity"])
def test_motion_is_optional_only_when_consumer_does_not_require_it(field):
    subject = object_evidence()
    with pytest.raises(ObjectObservationError, match=field):
        subject.require_observation(now=1., required_motion=(field,))
    unknown_motion = replace(getattr(subject, field), value=(9., 8., 7.), availability="hypothesis")
    subject = replace(subject, **{field: unknown_motion})
    assert getattr(subject.require_observation(now=1.), field) is None
    with pytest.raises(ObjectObservationError, match=field):
        subject.require_observation(now=1., required_motion=(field,))
    measured_motion = replace(unknown_motion, availability="available")
    observation = replace(subject, **{field: measured_motion}).require_observation(
        now=1., required_motion=(field,))
    assert getattr(observation, field) == (9., 8., 7.)
    stale_motion = replace(measured_motion, timestamp=.89)
    with pytest.raises(ObjectObservationError, match=field):
        replace(subject, **{field: stale_motion}).require_observation(now=1.)


def test_visual_ids_nonunit_orientation_and_unknown_motion_requirements_reject():
    subject = object_evidence()
    with pytest.raises(ObjectObservationError, match="semantic_identity"):
        replace(subject, object_id=replace(subject.object_id, value="visual/0001")).require_observation(now=1.)
    with pytest.raises(ObjectObservationError, match="unit quaternion"):
        replace(subject, orientation_wxyz=replace(subject.orientation_wxyz, value=(2., 0., 0., 0.))).require_observation(now=1.)
    with pytest.raises(ObjectObservationError, match="Unknown required"):
        subject.require_observation(now=1., required_motion=("speed",))


def test_object_time_and_confidence_are_conservative_across_consumed_fields():
    subject = object_evidence()
    subject = replace(subject, orientation_wxyz=replace(subject.orientation_wxyz, timestamp=.95, confidence=.81))
    item = subject.require_observation(now=1.)
    assert item.timestamp == .95 and item.confidence == .81


def test_rgb_adapter_retains_hypotheses_and_never_imports_withheld_truth():
    mask = np.zeros((4, 4), bool)
    mask[1:3, 1:3] = True
    detection = Detection("rigid_cuboid", (1, 1, 3, 3), mask, .9)
    track = Track("visual/0001", detection, 1., False, False)
    pose = PoseEstimate(track, (.1, 0., .02), 0., None, None, None, .1, "uncertified")
    first = evidence_from_pose(pose)
    for truth_position in ([0., 0., 0.], [9., 8., 7.]):
        truth_observation({"object_id": "withheld", "class_id": "rigid_cuboid", "pose_6d": {
            "translation_m": truth_position, "rotation_matrix": np.eye(3).tolist()}}, 1.)
        assert evidence_from_pose(pose) == first
    assert first.position.value == pose.position and first.position.availability == "hypothesis"
    assert first.object_id.value is None and first.orientation_wxyz.value is None
    assert first.diagnostic_reason == "uncertified"
    with pytest.raises(ObjectObservationError, match="object_id"):
        first.require_observation(now=1.)
    # Adding a quaternion to the raw hypothesis cannot bypass certification.
    assert evidence_from_pose(replace(pose, orientation_wxyz=(1., 0., 0., 0.))).orientation_wxyz.value is None
    assert not evidence_from_pose(replace(pose, track=replace(track, ambiguous=True))).position.valid


@pytest.mark.parametrize("value, availability", [(None, "available"), ((0, 0, 0), "unavailable"), ((0, 0, 0), "unknown")])
def test_availability_and_payload_cannot_contradict(value, availability):
    with pytest.raises(ObjectObservationError):
        evidence(value, availability=availability)


@pytest.mark.parametrize("kwargs", [{"minimum_confidence": float("nan")}, {"minimum_confidence": 2.},
                                  {"maximum_age_s": -1.}, {"maximum_age_s": float("inf")}])
def test_invalid_limits_reject(kwargs):
    with pytest.raises(ObjectObservationError):
        BoundaryLimits(**kwargs)


@pytest.mark.parametrize("now", [float("nan"), float("inf"), -.1])
def test_invalid_delivery_times_reject(now):
    with pytest.raises(ObjectObservationError):
        world_evidence().require_world_state(now=now)
