"""Fresh confirmation integrity, partial-schema parity and truth isolation."""

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from dexvision.evaluation.perception_confirmation import TypedBaseline
from dexvision.logging.perception_confirmation import load_protocol, package_confirmation, verify_dataset
from dexvision.logging.visual_stream import digest_file, write_json
from dexvision.perception.object_detector import Detection
from dexvision.perception.object_observations import (
    SIMULATOR_GROUND_TRUTH, ObjectObservationError, make_object_observation,
)
from dexvision.perception.object_pose import PoseEstimate
from dexvision.perception.object_tracker import Track
from dexvision.perception.perception_baseline import save_artifact_index
from dexvision.perception.world_estimate import (
    WorldEstimate, evaluation_world_estimate, inferred_world_estimate, world_estimate_from_dict,
)
from dexvision.sim.world_state import WORLD_STATE_VERSION, RobotObservation, WorldState


def pose(timestamp=1.):
    detection = Detection("rigid_cuboid", (1, 1, 3, 3), np.ones((2, 2), bool), .9)
    track = Track("visual/0000", detection, timestamp, False, False)
    return PoseEstimate(track, (.1, 0., .02), None, None, None, None, None, "uncertified")


def truth(position=(.1, 0., .02)):
    entity = make_object_observation(object_id="block_small", class_id="rigid_cuboid", position=position,
        orientation_wxyz=(1, 0, 0, 0), linear_velocity=(0, 0, 0), source=SIMULATOR_GROUND_TRUTH,
        confidence=1., timestamp=1., frame="mujoco_world")
    robot = RobotObservation((0, 0, 0), (1, 0, 0, 0), (0, 0, 0), (1, 0, 0, 0), True)
    return WorldState(WORLD_STATE_VERSION, 1., "mujoco_world", (entity,), (), (), robot, ())


def test_actual_inferred_and_truth_adapters_share_partial_type_without_fill():
    inferred = inferred_world_estimate((pose(),), 1.)
    oracle = evaluation_world_estimate(truth())
    assert type(inferred) is type(oracle) is WorldEstimate
    assert inferred.schema_version == oracle.schema_version
    assert inferred.entities[0].evidence.position.value == oracle.entities[0].evidence.position.value
    for name in ("object_id", "orientation_wxyz", "angular_velocity"):
        assert getattr(inferred.entities[0].evidence, name).value is None
    for name in ("robot", "relations", "fixtures", "contacts"):
        assert getattr(inferred, name).availability == "unavailable"
        assert getattr(inferred, name).value is None
    for value in (inferred, oracle):
        with pytest.raises(ObjectObservationError):
            value.require_control_state(now=1.)


@pytest.mark.parametrize("name", ["object_id", "orientation_wxyz", "linear_velocity", "angular_velocity"])
def test_unknown_fields_reject_at_every_read(name):
    entity = inferred_world_estimate((pose(),), 1.).entities[0]
    with pytest.raises(ObjectObservationError):
        entity.require_field(name, now=1., allow_hypothesis=True)


def test_hypothesis_read_requires_explicit_diagnostic_opt_in_and_freshness():
    entity = inferred_world_estimate((pose(),), 1.).entities[0]
    with pytest.raises(ObjectObservationError):
        entity.require_field("position", now=1.)
    assert entity.require_field("position", now=1., allow_hypothesis=True) == (.1, 0., .02)
    for now in (float("nan"), .9, 1.10001):
        with pytest.raises(ObjectObservationError):
            entity.require_field("position", now=now, allow_hypothesis=True)


def test_truth_changes_cannot_change_typed_inference_or_history():
    class Baseline:
        def reset(self):
            pass

        def infer(self, rgb, timestamp):
            return (pose(timestamp),)

    a, b = TypedBaseline(Baseline()), TypedBaseline(Baseline())
    for timestamp in (1., 1.17, 1.51):
        a.infer(np.zeros((4, 4, 3), np.uint8), timestamp)
        evaluation_world_estimate(truth((9., 8., 7.)))
        b.infer(np.zeros((4, 4, 3), np.uint8), timestamp)
    assert a.frames == b.frames
    assert all(f["accepted_control_state_blocked"] for f in a.frames)


def test_envelope_rejects_duplicate_future_and_mixed_truth_fields():
    world = inferred_world_estimate((pose(),), 1.)
    for change in ({"entities": world.entities*2}, {"timestamp": .9},
                   {"robot": replace(world.robot, producer=SIMULATOR_GROUND_TRUTH)},
                   {"frame": "camera"}):
        with pytest.raises(ObjectObservationError):
            replace(world, **change)
    assert asdict(world)["contacts"]["value"] is None


def test_serialized_partial_and_truth_use_same_typed_roundtrip():
    import json
    for world in (inferred_world_estimate((pose(),), 1.), evaluation_world_estimate(truth())):
        restored = world_estimate_from_dict(json.loads(json.dumps(asdict(world))))
        assert type(restored) is WorldEstimate
        assert restored.source == world.source
        assert restored.entities[0].evidence == world.entities[0].evidence
        with pytest.raises(ObjectObservationError):
            restored.require_control_state(now=1.)


def test_scripted_finger_order_survives_sorted_json(tmp_path):
    import json
    import yaml
    from dexvision.logging.perception_confirmation import generate_source
    from dexvision.retargeting.curl_retargeter import CurlRetargeter, load_curl_retargeter_config
    cfg = yaml.safe_load(Path("configs/level5/learning.yaml").read_text())
    names = [f["name"].split("/", 1)[1] for f in cfg["action"]["named_layout"][7:]]
    neutral = CurlRetargeter.from_mapping(load_curl_retargeter_config("configs/level1_teleop.yaml")).map(None)
    case = json.loads(json.dumps({"skill": "press_button", "cell_id": "press_008_centered_nominal",
        "seed": 30057001, "boundary_case": False, "finger_target_names": names,
        "neutral_fingers": {name: neutral[name] for name in names}}, sort_keys=True))
    result = generate_source({"workcell_config": "configs/workcell.yaml", "dataset_config": "configs/level4_dataset.yaml",
                              "maximum_source_actions": 1}, case, tmp_path / "source")
    assert result["frame_count"] == 1
    labels = json.loads((tmp_path / "source/evaluation_only_world_estimates.json").read_text())
    assert world_estimate_from_dict(labels[0]).source == SIMULATOR_GROUND_TRUTH


def fake_protocol(root: Path):
    path = root / "protocol.json"
    write_json(path, {"version": "level5/perception-confirmation-v1", "pins": {}})
    path.with_suffix(".sha256").write_text(f"{digest_file(path)}  protocol.json\n")
    return path


def fake_dataset(root: Path, protocol: Path, *, collision=False):
    root.mkdir()
    write_json(root / "manifest.json", {"protocol_sha256": digest_file(protocol),
        "historical_pixel_collisions": int(collision), "no_filtered_attempts": True})
    (root / "payload.txt").write_text("fresh scripted evidence")
    save_artifact_index(root)


def test_dataset_checksum_independence_and_no_overwrite(tmp_path):
    protocol = fake_protocol(tmp_path)
    root = tmp_path / "data"
    fake_dataset(root, protocol)
    verify_dataset(root, protocol)
    archive = tmp_path / "fresh-v1.tar.gz"
    info = package_confirmation(protocol, root, archive)
    assert info["archive_sha256"] == digest_file(archive)
    assert not any("staging" in name for name in info["payload_files"])
    with pytest.raises(ValueError, match="exists"):
        package_confirmation(protocol, root, archive)
    (root / "payload.txt").write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        verify_dataset(root, protocol)


def test_protocol_pins_and_collision_fail_closed(tmp_path):
    protocol = fake_protocol(tmp_path)
    load_protocol(protocol)
    root = tmp_path / "data"
    fake_dataset(root, protocol, collision=True)
    with pytest.raises(ValueError, match="independence"):
        verify_dataset(root, protocol)
    protocol.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        load_protocol(protocol)
