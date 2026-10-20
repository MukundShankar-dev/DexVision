"""5.7 analytical geometry, sensor firewall and immutable execution contracts."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dexvision.evaluation.instrumented_observer import marker_atlas
from dexvision.evaluation.instrumented_rollouts import clone_runner
from dexvision.perception.fiducial_pose import FiducialPoseEstimator, OpticalPose
from dexvision.perception.instrumented_state import InstrumentedStateEstimator
from dexvision.sim.workcell import Workcell
from dexvision.sim.workcell_sensors import populate_encoder_state, read_workcell_sensors


def analytical_optical():
    k = np.array([[1000.0, 0, 639.5], [0, 1000.0, 479.5], [0, 0, 1.0]])
    atlas = {
        str(i): {
            "object_id": "test_body",
            "class_id": "rigid_cuboid",
            "side_m": 0.05,
            "offset_m": [-0.03 if i == 0 else 0.03, -0.06, 0.1],
            "rotation": Rotation.from_euler("y", -20 if i == 0 else 20, degrees=True)
            .as_matrix()
            .tolist(),
        }
        for i in (0, 1)
    }
    calibration = {
        "intrinsics": k.tolist(),
        "camera_from_world": np.eye(4).tolist(),
        "image_size": [1280, 960],
    }
    position = np.array([0.05, -0.03, 0.7])
    rotation = Rotation.from_euler("xyz", [160.0, 10.0, 15.0], degrees=True).as_matrix()
    corners = []
    for tag in atlas.values():
        s = tag["side_m"] / 2
        points = (
            np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]])
            @ np.array(tag["rotation"]).T
            + tag["offset_m"]
        )
        world = points @ rotation.T + position
        uv = world @ k.T
        corners.append((uv[:, :2] / uv[:, 2, None]).reshape(1, 4, 2))
    estimator = FiducialPoseEstimator(calibration, atlas)
    estimator.detector = type(
        "Detector",
        (),
        {"detectMarkers": lambda _, rgb: (corners, np.array([[0], [1]]), [])},
    )()
    return estimator, position, rotation


def test_nonplanar_marker_pose_recovers_body_origin_without_truth_argument():
    estimator, position, rotation = analytical_optical()
    rgb = np.zeros((960, 1280, 3), np.uint8)
    assert estimator.infer(rgb, 0.0) == ()  # no fabricated zero motion at birth
    (pose,) = estimator.infer(rgb, 0.034)
    assert np.allclose(pose.position, position, atol=1e-8)
    actual = Rotation.from_quat(np.roll(pose.orientation_wxyz, -1)).as_matrix()
    assert np.allclose(actual, rotation, atol=1e-8)
    assert np.linalg.norm(pose.linear_velocity) < 1e-8
    assert np.linalg.norm(pose.angular_velocity) < 1e-8
    with pytest.raises(TypeError):
        estimator.infer(rgb, 0.068, simulator_pose=position)


def test_single_or_duplicate_marker_cannot_supply_certified_pose():
    estimator, _, _ = analytical_optical()
    original = estimator.detector.detectMarkers(None)
    estimator.detector = type(
        "Detector",
        (),
        {"detectMarkers": lambda _, rgb: (original[0][:1], np.array([[0]]), [])},
    )()
    assert estimator.infer(np.zeros((960, 1280, 3), np.uint8), 0.0) == ()
    estimator.detector = type(
        "Detector",
        (),
        {"detectMarkers": lambda _, rgb: (original[0], np.array([[0], [0]]), [])},
    )()
    assert estimator.infer(np.zeros((960, 1280, 3), np.uint8), 0.034) == ()


def test_sensor_ports_exclude_movable_object_state_and_fail_if_missing():
    with Workcell() as workcell:
        workcell.reset(seed=57)
        packet = read_workcell_sensors(workcell.env, workcell.config)
        objects = {o.joint for o in workcell.config.objects}
        assert not objects.intersection(packet.robot_joints)
        import mujoco

        data = mujoco.MjData(workcell.env.model)
        populate_encoder_state(workcell.env.model, data, workcell.config, packet)
        for port in (
            "robot_joints",
            "fixture_joints",
            "contact_switches",
            "hand_target",
            "actuator_commands",
        ):
            with pytest.raises(ValueError, match="unavailable"):
                populate_encoder_state(
                    workcell.env.model,
                    data,
                    workcell.config,
                    replace(packet, **{port: None}),
                )
        with pytest.raises(ValueError, match="Incomplete"):
            populate_encoder_state(
                workcell.env.model,
                data,
                workcell.config,
                replace(packet, robot_joints={}),
            )


def test_same_rgb_and_sensor_packet_is_invariant_to_hidden_object_truth():
    with Workcell() as workcell:
        world = workcell.reset(seed=57)
        entity = world.require_entity("block_small")
        packet = read_workcell_sensors(workcell.env, workcell.config)
        pose = OpticalPose(
            entity.object_id,
            entity.class_id,
            entity.position,
            entity.orientation_wxyz,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            packet.timestamp,
            0.0,
            (),
        )

        class Optical:
            def infer(self, rgb, timestamp):
                return (pose,)

        first = InstrumentedStateEstimator(workcell, Optical(), ("block_small",))
        a = first.observe(np.zeros((1, 1, 3), np.uint8), packet, now=packet.timestamp)
        joint = workcell.env.model.joint(workcell.config.objects[0].joint)
        workcell.env.data.qpos[int(joint.qposadr[0])] += 0.1
        second = InstrumentedStateEstimator(workcell, Optical(), ("block_small",))
        b = second.observe(np.zeros((1, 1, 3), np.uint8), packet, now=packet.timestamp)
        assert a == b
        assert first.workcell.env.data is not workcell.env.data
        preview = first.preview_task(
            skill_name="place_held_object", goal={}, initial_world=a
        )
        assert preview.env.data is first.workcell.env.data
        assert preview.env.data is not workcell.env.data


def test_clone_changes_only_private_observation_globals():
    code = compile("source()", "<private-observation-test>", "eval")
    original = type(lambda: None)(code, {"source": lambda: "truth"})
    inferred = clone_runner(original, {"source": lambda: "inferred"})
    assert original() == "truth" and inferred() == "inferred"
    assert inferred.__code__ is original.__code__


def test_marker_ids_and_rig_geometry_are_static_calibration():
    with Workcell() as workcell:
        recipe = {
            "object_rig_offset_m": [0, -0.06, 0.1],
            "fixture_rig_offset_m": [-0.08, -0.06, 0.05],
            "marker_side_m": 0.05,
            "marker_cube_half_size_m": 0.035,
        }
        atlas = marker_atlas(workcell.config, recipe)
        assert len(atlas) == 6 * (
            len(workcell.config.objects) + len(workcell.config.fixtures)
        )
        assert len({int(k) for k in atlas}) == len(atlas)
        assert all("position_world" not in tag for tag in atlas.values())


def test_estimated_preview_motion_uses_correct_free_joint_angular_frame():
    import mujoco

    with Workcell() as workcell:
        world = workcell.reset(seed=57)
        entity = world.require_entity("block_small")
        quaternion = tuple(np.roll(Rotation.from_euler("z", 0.8).as_quat(), 1))
        pose = OpticalPose(
            entity.object_id,
            entity.class_id,
            entity.position,
            quaternion,
            (0.01, 0.02, 0.03),
            (0.2, -0.3, 0.4),
            world.timestamp,
            0.0,
            (),
        )

        class Optical:
            def infer(self, rgb, timestamp):
                return (pose,)

        estimator = InstrumentedStateEstimator(workcell, Optical(), ("block_small",))
        packet = read_workcell_sensors(workcell.env, workcell.config)
        estimator.observe(np.zeros((1, 1, 3), np.uint8), packet, now=packet.timestamp)
        env = estimator.workcell.env
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(
            env.model,
            env.data,
            mujoco.mjtObj.mjOBJ_BODY,
            env.model.body("block_small").id,
            velocity,
            0,
        )
        assert np.allclose(velocity[:3], pose.angular_velocity)
        assert np.allclose(velocity[3:], pose.linear_velocity)


def test_camera_hold_preserves_acquisition_time_and_expires_without_new_pixels():
    with Workcell() as workcell:
        world = workcell.reset(seed=57)
        entity = world.require_entity("block_small")
        pose = OpticalPose(
            entity.object_id,
            entity.class_id,
            entity.position,
            entity.orientation_wxyz,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            world.timestamp,
            0.0,
            (),
        )

        class Optical:
            def infer(self, rgb, timestamp):
                return (pose,)

        estimator = InstrumentedStateEstimator(workcell, Optical(), ("block_small",))
        packet = read_workcell_sensors(workcell.env, workcell.config)
        estimator.observe(np.empty(0), packet, now=packet.timestamp)
        held = estimator.observe(None, replace(packet, timestamp=0.068), now=0.068)
        assert (
            held.timestamp == 0.068
            and held.require_entity("block_small").timestamp == 0.0
        )
        assert {r.object_id for r in held.relations} == {
            e.object_id for e in held.entities
        }
        with pytest.raises(ValueError, match="Missing fresh"):
            estimator.observe(None, replace(packet, timestamp=0.101), now=0.101)


def test_unobserved_contact_is_not_replaced_by_default_object_pose():
    with Workcell() as workcell:
        world = workcell.reset(seed=57)
        entity = world.require_entity("block_small")
        pose = OpticalPose(
            entity.object_id,
            entity.class_id,
            entity.position,
            entity.orientation_wxyz,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            world.timestamp,
            0.0,
            (),
        )

        class Optical:
            def infer(self, rgb, timestamp):
                return (pose,)

        estimator = InstrumentedStateEstimator(workcell, Optical(), ("block_small",))
        packet = read_workcell_sensors(workcell.env, workcell.config)
        packet = replace(
            packet,
            contact_switches=packet.contact_switches + (("block_large", "rh_palm"),),
        )
        with pytest.raises(ValueError, match="Unobserved movable"):
            estimator.observe(np.empty(0), packet, now=packet.timestamp)


def test_frozen_instrumented_matrices_cover_cells_and_disjoint_resets():
    from types import SimpleNamespace
    from collections import Counter
    from dexvision.evaluation.instrumented_rollouts import case_plan

    runtime = SimpleNamespace(
        inputs=SimpleNamespace(
            skill={
                "evaluation": {
                    split: {"cells": [{"cell_id": str(i)} for i in range(18)]}
                    for split in ("validation", "test")
                }
            }
        )
    )
    protocol = {"seed_starts": {"validation": 57300000, "test": 57900000}}
    validation = case_plan(protocol, runtime, "pick", "validation")
    test = case_plan(protocol, runtime, "pick", "test")
    assert len(test) == 54 and len(validation) == 18
    assert min(Counter(r["cell_id"] for r in test).values()) == 3
    assert not {r["seed"] for r in test} & {r["seed"] for r in validation}


def test_instrumented_protocol_rejects_development_limit_and_changed_source(tmp_path):
    import yaml
    from dexvision.evaluation.instrumented_rollouts import (
        load_instrumented_protocol,
        VERSION,
    )
    from dexvision.learning.training_manifest import digest_file

    source = tmp_path / "source.py"
    source.write_text("immutable = True\n")
    spec = dict(
        version=VERSION,
        checkpoint="5.7",
        training_seeds=[5001, 5002, 5003],
        minimum_confidence=0.8,
        maximum_age_s=0.1,
        camera_stride=3,
        bootstrap_frames=3,
        conditions=[
            "nominal",
            "mild_illumination",
            "partial_occlusion",
            "bounded_distractors",
        ],
        implementation_sha256={str(source): digest_file(source)},
    )
    path = tmp_path / "protocol.yaml"

    def write(value):
        path.write_text(yaml.safe_dump(value))
        path.with_suffix(".sha256").write_text(f"{digest_file(path)}  {path.name}\n")

    write(spec)
    assert load_instrumented_protocol(path)["maximum_age_s"] == 0.1
    write({**spec, "development_case_limit": 1})
    with pytest.raises(ValueError, match="coverage/delivery"):
        load_instrumented_protocol(path)
    write(spec)
    source.write_text("immutable = False\n")
    with pytest.raises(ValueError, match="integration changed"):
        load_instrumented_protocol(path)


def test_instrumented_cli_freeze_evaluate_and_review_are_separate(
    tmp_path, monkeypatch
):
    from dexvision.apps.evaluate_skill import main
    from dexvision.evaluation.instrumented_rollouts import VERSION

    path = tmp_path / "protocol.yaml"
    path.write_text(f"version: {VERSION}\n")
    calls = []
    monkeypatch.setattr(
        "dexvision.evaluation.instrumented_rollouts.freeze_instrumented_matrices",
        lambda p: calls.append("freeze") or {},
    )
    monkeypatch.setattr(
        "dexvision.evaluation.instrumented_rollouts.run_instrumented_evaluation",
        lambda *args, **kwargs: calls.append(kwargs["split"]) or {},
    )
    monkeypatch.setattr(
        "dexvision.evaluation.instrumented_review.review_instrumented",
        lambda p: calls.append("review") or {},
    )
    base = ["--config", str(path), "--all-required-skills", "--track", "perception"]
    assert main([*base, "--freeze-only"]) == 0
    assert (
        main([*base, "--split", "validation", "--output-dir", str(tmp_path / "new")])
        == 0
    )
    assert main([*base, "--viewer", "--review-dir", str(tmp_path / "test")]) == 0
    assert calls == ["freeze", "validation", "review"]
    assert main([*base, "--freeze-only", "--output-dir", str(tmp_path / "new")]) == 2
    assert len(calls) == 3


def test_paired_summary_does_not_hide_failed_physical_audit_or_reuse_counts():
    from types import SimpleNamespace
    from dexvision.evaluation.instrumented_rollouts import summarize_paired

    runtime = SimpleNamespace(
        inputs=SimpleNamespace(
            skill={"skill": "reach_object", "gates": {"mean_action_jerk_max": 0.1}}
        )
    )
    base = dict(
        reset_seed=123,
        checkpoint_sha256="selected",
        object_family="cuboid",
        cell_id="cell",
        target_type="zone",
        terminal_result="succeeded",
        terminal_reason="physical_success",
        safety_counters={},
        executed_actions=10,
        inference_latencies_s=[0.01],
        mean_action_jerk=0.0,
        failure_attribution="none",
    )
    pairs = [
        {
            "state": {**base, "condition": "nominal"},
            "perception": {
                **base,
                "condition": condition,
                "physical_audit": {"passed": condition == "nominal"},
            },
        }
        for condition in ("nominal", "partial_occlusion")
    ]
    result = summarize_paired(pairs, runtime)
    assert result["state"]["attempts"] == 2
    assert result["state"]["unique_reset_count"] == 1
    assert result["perception"]["success_rate"] == 0.5
    assert result["perception"]["original_terminal_success_rate"] == 1.0
    assert result["perception"]["physical_audit_failures"] == 1
    assert not result["gates"]["physical_audits"]
    assert result["conditions"]["nominal"]["numerical_pass"]
    assert not result["conditions"]["partial_occlusion"]["numerical_pass"]
    assert not result["default_registry_admission"]


def test_contact_policy_ignores_truth_latch_and_rest_arguments():
    from types import SimpleNamespace
    from dexvision.evaluation.instrumented_observer import ObservedPolicy

    calls = []

    class Policy:
        def __call__(self, source, prior, **kwargs):
            calls.append(kwargs)
            return prior, np.zeros(1)

    observer = SimpleNamespace(
        policy_calls=0,
        initial=SimpleNamespace(
            require_entity=lambda name: SimpleNamespace(position=(1.0, 2.0, 3.0))
        ),
        row={"entity_id": "block"},
        pressed_once=False,
    )
    policy = ObservedPolicy(Policy(), observer, "push")
    policy(
        np.zeros(1), np.zeros(27), fixture_rest=(99.0, 99.0, 99.0), pressed_once=True
    )
    policy(
        np.zeros(1),
        np.zeros(27),
        fixture_rest=(-99.0, -99.0, -99.0),
        pressed_once=False,
    )
    assert calls == [dict(fixture_rest=(1.0, 2.0, 3.0), pressed_once=False)] * 2


def test_instrumented_review_requires_complete_hash_index_and_keeps_acceptance_pending(
    tmp_path, monkeypatch
):
    import cv2
    import json
    from dexvision.evaluation.perception_rollouts import SKILLS
    from dexvision.evaluation.instrumented_review import (
        index_review,
        review_instrumented,
    )
    from dexvision.perception.perception_baseline import save_artifact_index

    for skill in SKILLS:
        directory = tmp_path / skill / "5001" / "000" / "nominal"
        directory.mkdir(parents=True)
        writer = cv2.VideoWriter(
            str(directory / "review.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32)
        )
        assert writer.isOpened()
        writer.write(np.zeros((32, 32, 3), np.uint8))
        writer.release()
        (directory / "result.json").write_text(
            json.dumps({"terminal_result": "rejected_before_execution"})
        )
    (tmp_path / "report.json").write_text(json.dumps({"split": "test"}))
    assert len(index_review(tmp_path)) == 5
    save_artifact_index(tmp_path)
    monkeypatch.setattr(cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *args: ord("n"))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = review_instrumented(tmp_path)
    assert (
        not result["checkpoint_complete"]
        and result["manual_acceptance"] == "pending_owner_confirmation"
    )
    assert before == {p: p.read_bytes() for p in before}
    index = tmp_path / "SHA256SUMS"
    index.write_text("\n".join(index.read_text().splitlines()[1:]) + "\n")
    with pytest.raises(ValueError, match="Incomplete"):
        review_instrumented(tmp_path)
