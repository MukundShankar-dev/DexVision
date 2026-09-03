from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from dexvision.learning.datasets import (
    DatasetBundle,
    EpisodeData,
    GoalConditionedSkillDataset,
    fit_training_normalization,
)
from dexvision.learning.models import (
    BASE_ORIENTATION_ACTION_NAMES,
    BASE_POSITION_ACTION_NAMES,
    MLPConfig,
)
from dexvision.learning.splits import SplitAssignment, SplitManifest
from dexvision.learning.train_bc import TrainingConfig, train_behavior_cloning


ACTION_NAMES = (
    BASE_POSITION_ACTION_NAMES
    + BASE_ORIENTATION_ACTION_NAMES
    + ("finger_actuator_targets/finger-a",)
)


def test_best_validation_selection_uses_earliest_exact_tie(
    tmp_path: Path, monkeypatch
) -> None:
    validation_losses = iter((0.4, 0.2, 0.2, 0.3))

    def fake_train_epoch(*_args, **_kwargs) -> float:
        return 0.1

    def fake_evaluate_loss(*_args, **_kwargs) -> float:
        return next(validation_losses)

    monkeypatch.setattr("dexvision.learning.train_bc._train_epoch", fake_train_epoch)
    monkeypatch.setattr(
        "dexvision.learning.train_bc.evaluate_loss", fake_evaluate_loss
    )
    last = tmp_path / "policy_last.pt"
    best = tmp_path / "policy_best.pt"

    result = train_behavior_cloning(
        _bundle(),
        output_path=last,
        best_output_path=best,
        training_config=TrainingConfig(epochs=4, batch_size=2, seed=7),
        model_config=MLPConfig(hidden_dims=(4,)),
    )

    assert result.best_checkpoint_path == best
    assert result.last_checkpoint_path == last
    assert result.selected_epoch == 2
    assert result.selected_validation_loss == 0.2
    assert best.is_file() and last.is_file()
    assert best.with_suffix(".pt.sha256").is_file()
    assert last.with_suffix(".pt.sha256").is_file()
    best_payload = torch.load(best, map_location="cpu", weights_only=True)
    last_payload = torch.load(last, map_location="cpu", weights_only=True)
    assert best_payload["checkpoint_role"] == "best_validation"
    assert best_payload["state_epoch"] == 2
    assert best_payload["selected_epoch"] == 2
    assert best_payload["selection_metric"] == "validation_loss"
    assert best_payload["selection_tie_break"] == "earliest_epoch"
    assert last_payload["checkpoint_role"] == "last"
    assert last_payload["state_epoch"] == 4
    assert len(best_payload["loss_history"]) == 4
    assert best_payload["loss_history"] == last_payload["loss_history"]


def _bundle() -> DatasetBundle:
    episodes = (
        _episode("train", "goal-a"),
        _episode("validation", "goal-a"),
    )
    assignments = tuple(
        SplitAssignment(
            episode_id=episode.episode_id,
            goal_id=episode.goal_id,
            split=split,
            data_digest=episode.data_digest,
            action_schema_version=episode.action_schema_version,
            observation_schema_version=episode.observation_schema_version,
            recording_session_id=None,
        )
        for episode, split in zip(episodes, ("train", "validation"), strict=True)
    )
    manifest = SplitManifest(
        version="test-split-v1",
        seed=3,
        strategy="stratified_episode_hash",
        dataset_digest="selection-test-dataset",
        assignments=assignments,
        action_schema_versions=("level1.13/full-action-v1",),
        observation_schema_versions=("level2/observation-layout-v2",),
        recording_session_ids_available=False,
    )
    normalization = fit_training_normalization(episodes, manifest)
    return DatasetBundle(
        train=GoalConditionedSkillDataset(episodes[:1], normalization=normalization),
        validation=GoalConditionedSkillDataset(
            episodes[1:], normalization=normalization
        ),
        test=GoalConditionedSkillDataset((), normalization=normalization),
        manifest=manifest,
        normalization=normalization,
    )


def _episode(episode_id: str, goal_id: str) -> EpisodeData:
    observations = np.asarray([[0.0], [1.0]], dtype=np.float64)
    actions = np.zeros((2, len(ACTION_NAMES)), dtype=np.float64)
    return EpisodeData(
        episode_id=episode_id,
        goal_id=goal_id,
        recording_session_id=None,
        action_schema_version="level1.13/full-action-v1",
        observation_schema_version="level2/observation-layout-v2",
        data_digest=f"digest-{episode_id}",
        observations=observations,
        goal=np.asarray([0.0]),
        actions=actions,
        observation_names=("state/value",),
        goal_names=("goal/value",),
        action_names=ACTION_NAMES,
        timestamps=np.asarray([0.0, 1.0]),
        tracking_quality=np.ones((2, 1)),
        quality_passed=True,
        recomputed_success=True,
    )
