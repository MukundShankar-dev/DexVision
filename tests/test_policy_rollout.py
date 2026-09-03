from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from dexvision.evaluation.evaluate_policy import (
    ActionBounds,
    BackendState,
    MujocoReachRolloutBackend,
    PolicyEvaluationError,
    ReachEvaluationProtocol,
    ReachScenario,
    evaluate_policy,
    load_reach_evaluation_protocol,
)
from dexvision.learning.models import (
    BASE_ORIENTATION_ACTION_NAMES,
    BASE_POSITION_ACTION_NAMES,
    FINGER_ACTION_PREFIX,
    GoalConditionedMLP,
    MLPConfig,
    PolicySchema,
)
from dexvision.learning.policies import PolicyError, load_checkpoint_policy


ROOT = Path(__file__).resolve().parents[1]
ACTION_NAMES = (
    BASE_POSITION_ACTION_NAMES
    + BASE_ORIENTATION_ACTION_NAMES
    + (f"{FINGER_ACTION_PREFIX}finger",)
)


class DeterministicPolicy:
    observation_names = ("state/value",)
    observation_schema_version = "level2/observation-layout-v2"
    action_schema_version = "level1.13/full-action-v1"
    goal_names = ("target_position/x", "target_position/y", "target_position/z")
    dataset_action_names = ACTION_NAMES
    output_action_names = ACTION_NAMES
    checkpoint_digest = "deterministic-test-policy"
    dataset_digest = "synthetic-dataset"

    def predict(self, observation: np.ndarray, goal: np.ndarray) -> np.ndarray:
        del observation
        return np.asarray([*goal, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


class FakeBackend:
    max_episode_steps = 4
    observation_schema_version = "level2/observation-layout-v2"
    action_schema_version = "level1.13/full-action-v1"

    def __init__(self) -> None:
        self.scenario: ReachScenario | None = None
        self.count = 0

    def reset(self, scenario: ReachScenario) -> BackendState:
        self.scenario = scenario
        self.count = 0
        return BackendState(False, None, 1.0, 0)

    def observation(self, names, *, previous_action):
        del previous_action
        assert tuple(names) == ("state/value",)
        return np.asarray([self.count], dtype=np.float64)

    def initial_action(self, action_names):
        assert tuple(action_names) == ACTION_NAMES
        return np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    def action_bounds(self, action_names):
        assert tuple(action_names) == ACTION_NAMES
        return ActionBounds(
            lower=np.asarray([-1.0] * len(ACTION_NAMES)),
            upper=np.asarray([1.0] * len(ACTION_NAMES)),
            workspace_indices=(0, 1, 2),
            joint_indices=(7,),
        )

    def step(self, action_names, action, *, n_steps):
        del action_names, n_steps
        assert self.scenario is not None
        self.count += 1
        distance = float(
            np.linalg.norm(action[:3] - np.asarray(self.scenario.target_position))
        )
        success = distance < 1e-12 and self.count >= 2
        return BackendState(success, None, distance, self.count)

    def close(self):
        pass


def test_frozen_protocol_builds_exact_35_scenario_matrix() -> None:
    protocol = load_reach_evaluation_protocol(ROOT / "configs/level3_evaluation.yaml")

    assert len(protocol.scenarios) == 35
    assert sum(item.target_group == "training" for item in protocol.scenarios) == 21
    assert sum(item.target_group == "held_out" for item in protocol.scenarios) == 14
    assert len({item.scenario_id for item in protocol.scenarios}) == 35


def test_deterministic_policy_runs_full_matrix_and_saves_every_run(tmp_path: Path) -> None:
    protocol = load_reach_evaluation_protocol(ROOT / "configs/level3_evaluation.yaml")

    report = evaluate_policy(
        DeterministicPolicy(),
        protocol,
        output_dir=tmp_path,
        backend_factory=FakeBackend,
        sim_steps_per_action=1,
    )

    assert report.passed is True
    assert report.metrics["training_target_success_rate"] == 1.0
    assert report.metrics["held_out_target_success_rate"] == 1.0
    assert report.metrics["terminal_reason_distribution"] == {"success": 35}
    assert len(report.results) == 35
    assert (tmp_path / "report.json").is_file()
    assert len(tuple((tmp_path / "trajectories").glob("*.npz"))) == 35
    assert all(Path(result.trajectory_file).is_file() for result in report.results)


def test_action_subset_must_be_named_explicitly(tmp_path: Path) -> None:
    policy = DeterministicPolicy()
    policy.output_action_names = BASE_POSITION_ACTION_NAMES  # type: ignore[misc]
    protocol = _two_scenario_protocol()

    with pytest.raises(PolicyEvaluationError, match="ablation name"):
        evaluate_policy(
            policy,
            protocol,
            output_dir=tmp_path,
            backend_factory=FakeBackend,
        )


def test_checkpoint_loader_verifies_sidecar_and_normalization(tmp_path: Path) -> None:
    schema = PolicySchema(
        observation_schema_version="level2/observation-layout-v2",
        action_schema_version="level1.13/full-action-v1",
        observation_names=("state/value",),
        goal_names=("target_position/x", "target_position/y", "target_position/z"),
        dataset_action_names=ACTION_NAMES,
        output_action_names=ACTION_NAMES,
    )
    model = GoalConditionedMLP(schema, MLPConfig(hidden_dims=(4,)))
    normalization = {
        "source_split": "train",
        "dataset_digest": "dataset-digest",
        "observation": _stats(schema.observation_names),
        "goal": _stats(schema.goal_names),
        "action": _stats(schema.dataset_action_names),
    }
    import json

    normalization_digest = hashlib.sha256(
        json.dumps(normalization, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    checkpoint = tmp_path / "policy.pt"
    torch.save(
        {
            "checkpoint_version": model.CHECKPOINT_VERSION,
            "schema": schema.to_dict(),
            "config": model.config.to_dict(),
            "state_dict": model.state_dict(),
            "training_checkpoint_version": "dexvision/bc-training-v1",
            "provenance": {
                "dataset_digest": "dataset-digest",
                "normalization": normalization,
                "normalization_digest": normalization_digest,
            },
        },
        checkpoint,
    )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    checkpoint.with_suffix(".pt.sha256").write_text(
        f"{digest}  {checkpoint.name}\n", encoding="utf-8"
    )

    loaded = load_checkpoint_policy(
        checkpoint, expected_dataset_digest="dataset-digest"
    )

    assert loaded.checkpoint_digest == digest
    assert loaded.output_action_names == ACTION_NAMES
    prediction = loaded.predict(np.zeros(1), np.zeros(3))
    assert prediction.shape == (len(ACTION_NAMES),)
    assert np.all(np.isfinite(prediction))

    checkpoint.with_suffix(".pt.sha256").write_text(
        f"{'0' * 64}  {checkpoint.name}\n", encoding="utf-8"
    )
    with pytest.raises(PolicyError, match="SHA-256"):
        load_checkpoint_policy(checkpoint)


def test_headless_mujoco_smoke_rollout_with_deterministic_policy(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mujoco")
    backend = MujocoReachRolloutBackend(ROOT / "assets/mujoco/task_board_scene.xml")
    action_names = (
        BASE_POSITION_ACTION_NAMES
        + BASE_ORIENTATION_ACTION_NAMES
        + tuple(
            f"{FINGER_ACTION_PREFIX}{name}"
            for name in (
                backend.task.env._mujoco.mj_id2name(
                    backend.task.env.model,
                    backend.task.env._mujoco.mjtObj.mjOBJ_ACTUATOR,
                    index,
                )
                for index in range(backend.task.env.model.nu)
            )
        )
    )
    backend.close()

    class MujocoSmokePolicy(DeterministicPolicy):
        observation_names = ("base_position/x", "base_position/y", "base_position/z")
        dataset_action_names = action_names
        output_action_names = action_names

        def predict(self, observation, goal):
            del observation
            base = np.asarray(goal) + np.asarray([-0.04, 0.0, -0.337])
            return np.asarray(
                [*base, 1.0, 0.0, 0.0, 0.0, *([0.0] * 20)],
                dtype=np.float64,
            )

    report = evaluate_policy(
        MujocoSmokePolicy(),
        _two_scenario_protocol(),
        output_dir=tmp_path,
        model_path=ROOT / "assets/mujoco/task_board_scene.xml",
        sim_steps_per_action=5,
    )

    assert len(report.results) == 2
    assert all(result.success for result in report.results)
    assert all(result.terminal_reason == "success" for result in report.results)


def _stats(names: tuple[str, ...]) -> dict:
    return {
        "count": 4,
        "names": list(names),
        "mean": [0.0] * len(names),
        "std": [1.0] * len(names),
    }


def _two_scenario_protocol() -> ReachEvaluationProtocol:
    source = load_reach_evaluation_protocol(ROOT / "configs/level3_evaluation.yaml")
    scenarios = (
        source.scenarios[0],
        next(item for item in source.scenarios if item.target_group == "held_out"),
    )
    return replace(source, scenarios=scenarios)
