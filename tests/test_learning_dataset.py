from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dexvision.learning.datasets import (
    LearningDatasetError,
    load_frozen_reach_datasets,
    load_skill_episodes,
    quaternion_wxyz_to_rotation_6d,
)
from dexvision.learning.splits import (
    EpisodeDescriptor,
    SplitConfig,
    deterministic_episode_split,
)
from dexvision.logging.demo_logger import (
    DemoLogger,
    DemoStepData,
    build_level1_action_schema,
    build_level2_observation_schema,
)


ROOT = Path(__file__).resolve().parents[1]
REACH_GOALS = {
    "reach_target_left": (0.14, -0.10, 0.45),
    "reach_target_center": (0.14, 0.00, 0.49),
    "reach_target_right": (0.14, 0.06, 0.51),
}


def _descriptor(episode_id: str, goal_id: str, session: str | None = None):
    return EpisodeDescriptor(
        episode_id=episode_id,
        goal_id=goal_id,
        data_digest=f"digest-{episode_id}",
        action_schema_version="action-v1",
        observation_schema_version="observation-v2",
        recording_session_id=session,
    )


def _split_config() -> SplitConfig:
    return SplitConfig(
        version="level3/reach-evaluation-v1",
        seed=20260903,
        strategy="stratified_episode_hash",
        train_fraction=0.8,
        validation_fraction=0.2,
    )


def test_frozen_split_uses_exact_seeded_episode_hash_order() -> None:
    episodes = tuple(
        _descriptor(f"{goal}-{index}", goal)
        for goal in ("left", "center")
        for index in range(5)
    )

    manifest = deterministic_episode_split(episodes, _split_config())
    assignments = manifest.assignment_by_episode()

    for goal in ("left", "center"):
        ordered = sorted(
            (episode for episode in episodes if episode.goal_id == goal),
            key=lambda episode: hashlib.sha256(
                f"20260903:{episode.episode_id}".encode()
            ).hexdigest(),
        )
        assert [assignments[episode.episode_id] for episode in ordered] == [
            "train",
            "train",
            "train",
            "train",
            "validation",
        ]
    assert manifest.recording_session_ids_available is False


def test_recording_session_is_never_split() -> None:
    episodes = (
        _descriptor("a", "left", "session-a"),
        _descriptor("b", "left", "session-a"),
        _descriptor("c", "left", "session-b"),
        _descriptor("d", "left", "session-c"),
    )

    manifest = deterministic_episode_split(episodes, _split_config())
    assignments = manifest.assignment_by_episode()

    assert assignments["a"] == assignments["b"]
    assert manifest.recording_session_ids_available is True
    assert all(item.recording_session_id is not None for item in manifest.assignments)


def test_loader_builds_samples_and_fits_normalization_on_training_only(
    tmp_path: Path,
) -> None:
    skill_dir = tmp_path / "data" / "demos" / "raw" / "reach_touch_target"
    episode_specs = [
        (f"episode-{goal}-{index}", goal, position, float(index + goal_index * 10))
        for goal_index, (goal, position) in enumerate(REACH_GOALS.items())
        for index in range(2)
    ]
    for episode_id, goal_id, position, offset in episode_specs:
        _write_reach_episode(skill_dir / episode_id, episode_id, goal_id, position, offset)
    _write_reports(skill_dir, [item[0] for item in episode_specs])

    bundle = load_frozen_reach_datasets(
        tmp_path,
        evaluation_config_path=ROOT / "configs" / "level3_evaluation.yaml",
        normalize=False,
    )

    assert len(bundle.manifest.assignments) == 6
    assert {item.goal_id for item in bundle.manifest.assignments} == set(REACH_GOALS)
    assert len(bundle.train.episodes) == 3
    assert len(bundle.validation.episodes) == 3
    assert len(bundle.test) == 0
    assert set(item.episode_id for item in bundle.train.episodes).isdisjoint(
        item.episode_id for item in bundle.validation.episodes
    )

    sample = bundle.train[0]
    assert set(
        (
            "obs",
            "goal",
            "action",
            "episode_id",
            "demo_id",
            "timestep",
            "tracking_quality",
            "quality_passed",
            "recomputed_success",
        )
    ).issubset(sample)
    assert isinstance(sample["obs"], torch.Tensor)
    assert sample["obs"].dtype == torch.float32
    assert sample["goal"].shape == (3,)
    assert sample["action"].shape == (9,)
    assert sample["quality_passed"] is True
    assert sample["recomputed_success"] is True

    expected_observations = np.concatenate(
        [episode.observations for episode in bundle.train.episodes], axis=0
    )
    assert np.allclose(
        bundle.normalization.observation.mean,
        expected_observations.mean(axis=0),
    )
    assert bundle.normalization.observation.count == expected_observations.shape[0]
    assert bundle.normalization.source_split == "train"
    assert (
        bundle.normalization.dataset_digest == bundle.manifest.dataset_digest
    )

    manifest_path = tmp_path / "outputs" / "split_manifest.json"
    bundle.save_manifest(manifest_path)
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved["version"] == "level3/reach-evaluation-v1"
    assert saved["normalization"]["source_split"] == "train"
    assert len(saved["assignments"]) == 6
    assert all(item["data_digest"] for item in saved["assignments"])


def test_previous_action_is_explicit_and_zero_at_episode_start(tmp_path: Path) -> None:
    skill_dir = tmp_path / "raw" / "reach_touch_target"
    for index, (goal_id, position) in enumerate(REACH_GOALS.items()):
        for repeat in range(2):
            episode_id = f"episode-{index}-{repeat}"
            _write_reach_episode(
                skill_dir / episode_id,
                episode_id,
                goal_id,
                position,
                float(index + repeat),
            )
    episode_ids = [path.name for path in skill_dir.iterdir() if path.is_dir()]
    _write_reports(skill_dir, episode_ids)

    episodes = load_skill_episodes(
        tmp_path,
        skill_name="reach_touch_target",
        include_previous_action=True,
    )

    episode = episodes[0]
    previous_start = len(episode.observation_names) - len(episode.action_names)
    assert np.all(episode.observations[0, previous_start:] == 0.0)
    assert np.array_equal(
        episode.observations[1, previous_start:], episode.actions[0]
    )
    assert all(
        name.startswith("previous_action/")
        for name in episode.observation_names[previous_start:]
    )


def test_missing_layout_field_fails_clearly(tmp_path: Path) -> None:
    skill_dir = tmp_path / "reach_touch_target"
    _write_reach_episode(
        skill_dir / "episode-a",
        "episode-a",
        "reach_target_left",
        REACH_GOALS["reach_target_left"],
        0.0,
    )
    _write_reports(skill_dir, ["episode-a"])

    with pytest.raises(LearningDatasetError, match="missing requested field 'not_saved'"):
        load_skill_episodes(
            skill_dir,
            skill_name="reach_touch_target",
            observation_fields=("robot_qpos", "not_saved"),
        )


def test_mixed_executable_layouts_fail_instead_of_inferring_offsets(
    tmp_path: Path,
) -> None:
    skill_dir = tmp_path / "reach_touch_target"
    _write_reach_episode(
        skill_dir / "episode-a",
        "episode-a",
        "reach_target_left",
        REACH_GOALS["reach_target_left"],
        0.0,
        qpos_dim=4,
    )
    _write_reach_episode(
        skill_dir / "episode-b",
        "episode-b",
        "reach_target_left",
        REACH_GOALS["reach_target_left"],
        1.0,
        qpos_dim=5,
    )
    _write_reports(skill_dir, ["episode-a", "episode-b"])

    with pytest.raises(LearningDatasetError, match="observation schema/layout mismatch"):
        load_skill_episodes(skill_dir, skill_name="reach_touch_target")


def test_quaternion_orientation_uses_sign_invariant_rotation_6d() -> None:
    rotations = quaternion_wxyz_to_rotation_6d(
        np.asarray([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]])
    )

    assert np.allclose(rotations[0], rotations[1])
    assert np.allclose(rotations[0], [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def test_non_clean_episode_is_not_loaded(tmp_path: Path) -> None:
    skill_dir = tmp_path / "reach_touch_target"
    for episode_id in ("clean", "rejected"):
        _write_reach_episode(
            skill_dir / episode_id,
            episode_id,
            "reach_target_left",
            REACH_GOALS["reach_target_left"],
            0.0,
        )
    _write_reports(skill_dir, ["clean", "rejected"], rejected={"rejected"})

    episodes = load_skill_episodes(skill_dir, skill_name="reach_touch_target")

    assert [episode.episode_id for episode in episodes] == ["clean"]


def _write_reach_episode(
    path: Path,
    episode_id: str,
    goal_id: str,
    target_position: tuple[float, float, float],
    offset: float,
    *,
    qpos_dim: int = 4,
) -> None:
    action_schema = build_level1_action_schema(("finger-a", "finger-b"))
    observation_schema = build_level2_observation_schema(
        robot_qpos_dim=qpos_dim,
        robot_qvel_dim=3,
        finger_target_dim=2,
        tracking_quality_dim=4,
        robot_qpos_names=tuple(f"rh_qpos_{index}" for index in range(qpos_dim)),
        robot_qvel_names=("rh_qvel_a", "rh_qvel_b", "rh_qvel_c"),
        actuator_names=("finger-a", "finger-b"),
        finger_joint_qpos_indices=(0, 1),
        finger_joint_qvel_indices=(0, 1),
        finger_joint_names=("rh_joint_a", "rh_joint_b"),
        tracking_quality_names=(
            "detected",
            "handedness_code",
            "hand_tracking_confidence",
            "feature_confidence",
        ),
        object_state_dim=3,
        task_state_dim=3,
        target_state_dim=3,
    )
    logger = DemoLogger(
        path,
        action_schema=action_schema,
        observation_schema=observation_schema,
    )
    logger.start_episode(
        {
            "skill_name": "reach_touch_target",
            "task_name": "Synthetic reach",
            "task_id": "reach_touch_target",
            "episode_id": episode_id,
            "robot_model": "unused.xml",
            "retargeter_config": "unused.yaml",
            "control_rate_hz": 30.0,
            "teleop_config": {},
            "task_config": {
                "required_objects": ["target"],
                "requires_task_state": True,
                "requires_success_metric_inputs": False,
                "required_observation_fields": ["target_state"],
                "resolved_target_source": goal_id,
                "target_position": list(target_position),
            },
        }
    )
    robot_width = qpos_dim + 3 + 2 + 3 + 4
    for timestep in range(3):
        action = np.asarray(
            [
                offset + timestep,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.1,
                0.2,
            ],
            dtype=np.float64,
        )
        robot_state = np.arange(robot_width, dtype=np.float64) + offset + timestep
        robot_state[-4:] = [1.0, 0.0, 0.0, 0.0]
        logger.append(
            DemoStepData(
                features=np.asarray([offset, timestep], dtype=np.float64),
                action=action,
                robot_state=robot_state,
                object_state=np.asarray(target_position, dtype=np.float64),
                task_state=np.asarray(target_position, dtype=np.float64),
                tracking_quality=np.asarray([1.0, 0.0, 0.9, 0.8], dtype=np.float64),
                timestamp=float(timestep) / 30.0,
            )
        )
    logger.close(success=True)


def _write_reports(
    skill_dir: Path,
    episode_ids: list[str],
    *,
    rejected: set[str] | None = None,
) -> None:
    rejected = rejected or set()
    quality_entries = []
    relabel_entries = []
    for episode_id in sorted(episode_ids):
        passed = episode_id not in rejected
        quality_entries.append(
            {
                "episode_directory": episode_id,
                "episode_id": episode_id,
                "passed": passed,
                "failed_filters": [] if passed else ["synthetic_rejection"],
            }
        )
        relabel_entries.append(
            {
                "episode_directory": episode_id,
                "episode_id": episode_id,
                "operator_success": True,
                "recomputed_success": True,
                "labels_agree": True,
            }
        )
    (skill_dir / "quality_report.json").write_text(
        json.dumps({"episodes": quality_entries}), encoding="utf-8"
    )
    (skill_dir / "relabel_report.json").write_text(
        json.dumps({"episodes": relabel_entries}), encoding="utf-8"
    )
