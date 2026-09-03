from __future__ import annotations

from dataclasses import replace
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
    GoalConditionedMLP,
    MLPConfig,
    PolicySchema,
)
from dexvision.learning.splits import SplitAssignment, SplitManifest
from dexvision.learning.train_bc import (
    TrainingConfig,
    evaluate_loss,
    file_sha256,
    train_behavior_cloning,
)


ACTION_NAMES = (
    BASE_POSITION_ACTION_NAMES
    + BASE_ORIENTATION_ACTION_NAMES
    + ("finger_actuator_targets/finger-a",)
)


def test_tiny_cpu_dataset_overfits_and_saves_reproducibility_metadata(
    tmp_path: Path,
) -> None:
    bundle = _tiny_bundle()
    checkpoint = tmp_path / "policy.pt"

    result = train_behavior_cloning(
        bundle,
        output_path=checkpoint,
        training_config=TrainingConfig(
            epochs=160,
            batch_size=8,
            learning_rate=0.01,
            seed=71,
        ),
        model_config=MLPConfig(hidden_dims=(32, 32), activation="tanh"),
        experiment_config_digest="tiny-config-digest",
    )

    assert result.loss_history[-1]["train_loss"] < 2e-3
    assert result.loss_history[-1]["validation_loss"] < 2e-2
    assert result.checkpoint_digest == file_sha256(checkpoint)
    assert result.digest_path.read_text(encoding="utf-8").split()[0] == result.checkpoint_digest

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["training_checkpoint_version"] == "dexvision/bc-training-v1"
    assert payload["checkpoint_version"] == GoalConditionedMLP.CHECKPOINT_VERSION
    assert payload["completed_epochs"] == 160
    assert len(payload["loss_history"]) == 160
    assert payload["provenance"]["dataset_digest"] == "tiny-dataset-digest"
    assert payload["provenance"]["split_manifest_digest"]
    assert payload["provenance"]["normalization"]["source_split"] == "train"
    assert payload["environment"]["device"] == "cpu"
    assert payload["environment"]["deterministic_algorithms"] is True

    loaded = GoalConditionedMLP.load(checkpoint)
    assert loaded.schema.output_action_names == ACTION_NAMES


def test_validation_is_read_only() -> None:
    bundle = _tiny_bundle()
    schema = PolicySchema.from_episode(bundle.train.episodes[0])
    torch.manual_seed(9)
    model = GoalConditionedMLP(schema, MLPConfig(hidden_dims=(8,))).train()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    loss = evaluate_loss(
        model,
        bundle.validation,
        schema=schema,
        batch_size=4,
    )

    assert np.isfinite(loss)
    assert model.training is True
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert bundle.normalization.observation.count == sum(
        episode.num_steps for episode in bundle.train.episodes
    )


def test_epoch_boundary_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    bundle = _tiny_bundle()
    architecture = MLPConfig(hidden_dims=(12,), activation="tanh")
    full_config = TrainingConfig(
        epochs=8,
        batch_size=7,
        learning_rate=0.005,
        seed=101,
    )
    uninterrupted_path = tmp_path / "uninterrupted.pt"
    resumed_path = tmp_path / "resumed.pt"

    uninterrupted = train_behavior_cloning(
        bundle,
        output_path=uninterrupted_path,
        training_config=full_config,
        model_config=architecture,
        experiment_config_digest="resume-test",
    )
    train_behavior_cloning(
        bundle,
        output_path=resumed_path,
        training_config=replace(full_config, epochs=3),
        model_config=architecture,
        experiment_config_digest="resume-test",
    )
    resumed = train_behavior_cloning(
        bundle,
        output_path=resumed_path,
        training_config=full_config,
        model_config=architecture,
        experiment_config_digest="resume-test",
        resume_from=resumed_path,
    )

    first = torch.load(uninterrupted_path, map_location="cpu", weights_only=True)
    second = torch.load(resumed_path, map_location="cpu", weights_only=True)
    assert uninterrupted.loss_history == resumed.loss_history
    assert first["loss_history"] == second["loss_history"]
    assert all(
        torch.equal(first["state_dict"][name], second["state_dict"][name])
        for name in first["state_dict"]
    )


def test_cli_reports_missing_dataset_cleanly(tmp_path: Path, capsys) -> None:
    from dexvision.apps.train_policy import main

    exit_code = main(
        [
            "--config",
            str(Path(__file__).resolve().parents[1] / "configs" / "level3_bc.yaml"),
            "--dataset-root",
            str(tmp_path / "missing"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "ERROR:" in captured.err
    assert "could not find extracted skill directory" in captured.err


def _tiny_bundle() -> DatasetBundle:
    episodes = (
        _episode("train-a", "goal-a", -0.8, 16),
        _episode("train-b", "goal-b", 0.5, 16),
        _episode("validation-a", "goal-a", -0.6, 12),
    )
    assignments = (
        _assignment(episodes[0], "train"),
        _assignment(episodes[1], "train"),
        _assignment(episodes[2], "validation"),
    )
    manifest = SplitManifest(
        version="tiny-split-v1",
        seed=3,
        strategy="stratified_episode_hash",
        dataset_digest="tiny-dataset-digest",
        assignments=assignments,
        action_schema_versions=("level2/action-v1",),
        observation_schema_versions=("level2/observation-layout-v2",),
        recording_session_ids_available=False,
    )
    normalization = fit_training_normalization(episodes, manifest)
    train = GoalConditionedSkillDataset(episodes[:2], normalization=normalization)
    validation = GoalConditionedSkillDataset(episodes[2:], normalization=normalization)
    return DatasetBundle(
        train=train,
        validation=validation,
        test=GoalConditionedSkillDataset((), normalization=normalization),
        manifest=manifest,
        normalization=normalization,
    )


def _episode(episode_id: str, goal_id: str, goal_value: float, steps: int) -> EpisodeData:
    x = np.linspace(-1.0, 1.0, steps, dtype=np.float64)
    observations = np.column_stack((x, x * x))
    goal = np.asarray([goal_value], dtype=np.float64)
    features = np.column_stack((observations, np.full(steps, goal_value)))
    weights = np.asarray(
        [
            [0.5, -0.2, 0.1, 0.3, -0.4, 0.2, 0.1, -0.3],
            [-0.1, 0.4, 0.2, -0.2, 0.3, -0.3, 0.5, 0.1],
            [0.2, 0.1, -0.3, 0.4, 0.1, 0.2, -0.2, 0.5],
        ],
        dtype=np.float64,
    )
    actions = features @ weights
    return EpisodeData(
        episode_id=episode_id,
        goal_id=goal_id,
        recording_session_id=None,
        action_schema_version="level2/action-v1",
        observation_schema_version="level2/observation-layout-v2",
        data_digest=f"digest-{episode_id}",
        observations=observations,
        goal=goal,
        actions=actions,
        observation_names=("obs/x", "obs/x_squared"),
        goal_names=("goal/value",),
        action_names=ACTION_NAMES,
        timestamps=np.arange(steps, dtype=np.float64) / 30.0,
        tracking_quality=np.ones((steps, 1), dtype=np.float64),
        quality_passed=True,
        recomputed_success=True,
    )


def _assignment(episode: EpisodeData, split: str) -> SplitAssignment:
    return SplitAssignment(
        episode_id=episode.episode_id,
        goal_id=episode.goal_id,
        split=split,
        data_digest=episode.data_digest,
        action_schema_version=episode.action_schema_version,
        observation_schema_version=episode.observation_schema_version,
        recording_session_id=None,
    )
