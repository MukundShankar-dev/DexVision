from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dexvision.learning.models import (
    BASE_ORIENTATION_ACTION_NAMES,
    BASE_POSITION_ACTION_NAMES,
    GoalConditionedMLP,
    LearningModelError,
    MLPConfig,
    PolicySchema,
)


FINGER_ACTION_NAMES = (
    "finger_actuator_targets/rh_A_THJ5",
    "finger_actuator_targets/rh_A_FFJ4",
)
FULL_ACTION_NAMES = (
    BASE_POSITION_ACTION_NAMES + BASE_ORIENTATION_ACTION_NAMES + FINGER_ACTION_NAMES
)


def _schema(
    *, output_action_names: tuple[str, ...] = FULL_ACTION_NAMES
) -> PolicySchema:
    return PolicySchema(
        observation_schema_version="level2/observation-layout-v2",
        action_schema_version="level2/action-v1",
        observation_names=("robot_qpos/rh_WRJ2", "base_position/x", "object_state/x"),
        goal_names=("target_position/x", "target_position/y", "target_position/z"),
        dataset_action_names=FULL_ACTION_NAMES,
        output_action_names=output_action_names,
    )


def test_forward_pass_matches_named_full_action_schema_on_cpu() -> None:
    schema = _schema()
    model = GoalConditionedMLP(
        schema,
        MLPConfig(hidden_dims=(16, 8), activation="relu"),
    ).cpu()

    output = model(torch.zeros(4, 3), torch.ones(4, 3))

    assert output.shape == (4, len(FULL_ACTION_NAMES))
    assert output.device.type == "cpu"
    assert model.output_action_names == FULL_ACTION_NAMES
    assert model.schema.dataset_action_names[:3] == BASE_POSITION_ACTION_NAMES
    assert model.schema.dataset_action_names[3:7] == BASE_ORIENTATION_ACTION_NAMES
    assert model.schema.dataset_action_names[7:] == FINGER_ACTION_NAMES


def test_declared_action_subset_preserves_names_and_selects_targets() -> None:
    subset = BASE_POSITION_ACTION_NAMES + (FINGER_ACTION_NAMES[1],)
    schema = _schema(output_action_names=subset)
    actions = torch.arange(18, dtype=torch.float32).reshape(2, 9)
    model = GoalConditionedMLP(schema, MLPConfig(hidden_dims=(4,), activation="tanh"))

    output = model(torch.zeros(2, 3), torch.zeros(2, 3))
    selected = schema.select_action_targets(actions)

    assert schema.is_action_subset is True
    assert output.shape == (2, 4)
    assert model.output_action_names == subset
    assert torch.equal(selected, actions[:, (0, 1, 2, 8)])


def test_schema_rejects_unnamed_or_reordered_action_layouts() -> None:
    with pytest.raises(LearningModelError, match="begin with the named Level 1.13"):
        PolicySchema(
            observation_schema_version="obs-v2",
            action_schema_version="action-v1",
            observation_names=("obs/x",),
            goal_names=("goal/x",),
            dataset_action_names=("action/0",),
            output_action_names=("action/0",),
        )

    with pytest.raises(LearningModelError, match="preserve the dataset action field order"):
        _schema(output_action_names=(FINGER_ACTION_NAMES[0], BASE_POSITION_ACTION_NAMES[0]))


def test_eval_predictions_are_finite_and_deterministic() -> None:
    torch.manual_seed(17)
    model = GoalConditionedMLP(
        _schema(),
        MLPConfig(hidden_dims=(12, 12), activation="gelu"),
    )
    observations = torch.randn(5, 3)
    goals = torch.randn(5, 3)

    model.eval()
    with torch.no_grad():
        first = model(observations, goals)
        second = model(observations, goals)

    assert torch.isfinite(first).all()
    assert torch.equal(first, second)


def test_save_load_round_trip_preserves_schema_config_and_predictions(
    tmp_path: Path,
) -> None:
    torch.manual_seed(23)
    schema = _schema(output_action_names=BASE_POSITION_ACTION_NAMES)
    config = MLPConfig(hidden_dims=(10, 6), activation="tanh")
    model = GoalConditionedMLP(schema, config).eval()
    observations = torch.randn(3, schema.observation_dim)
    goals = torch.randn(3, schema.goal_dim)
    with torch.no_grad():
        expected = model(observations, goals)

    checkpoint = tmp_path / "nested" / "policy.pt"
    model.save(checkpoint)
    restored = GoalConditionedMLP.load(checkpoint).eval()
    with torch.no_grad():
        actual = restored(observations, goals)

    assert restored.schema == schema
    assert restored.config == config
    assert torch.equal(actual, expected)


def test_forward_rejects_schema_incompatible_batches() -> None:
    model = GoalConditionedMLP(_schema(), MLPConfig(hidden_dims=(4,)))

    with pytest.raises(LearningModelError, match=r"observations must have shape \[batch, 3\]"):
        model(torch.zeros(2, 4), torch.zeros(2, 3))
    with pytest.raises(LearningModelError, match="same batch size"):
        model(torch.zeros(2, 3), torch.zeros(1, 3))
