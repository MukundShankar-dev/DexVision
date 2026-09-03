from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    assert isinstance(document, dict)
    return document


def test_reach_evaluation_protocol_uses_reserved_rollout_targets() -> None:
    protocol = _load_yaml(ROOT / "configs" / "level3_evaluation.yaml")
    dataset = _load_yaml(ROOT / "configs" / "reach_touch_dataset.yaml")

    assert protocol["version"] == "level3/reach-evaluation-v1"
    assert protocol["task_id"] == "reach_touch_target"
    assert protocol["training_targets"] == dataset["training_targets"]
    assert (
        protocol["held_out_rollout_targets"]
        == dataset["held_out_evaluation_targets"]
    )

    training_positions = {
        tuple(value) for value in protocol["training_targets"].values()
    }
    held_out_positions = {
        tuple(value) for value in protocol["held_out_rollout_targets"].values()
    }
    assert training_positions.isdisjoint(held_out_positions)


def test_reach_evaluation_protocol_freezes_safe_distinct_scenarios() -> None:
    protocol = _load_yaml(ROOT / "configs" / "level3_evaluation.yaml")
    split = protocol["offline_split"]
    offsets = protocol["initial_base_position_offsets_m"]
    gates = protocol["acceptance_gates"]

    assert split["train_fraction"] + split["validation_fraction"] == 1.0
    assert split["test_fraction"] == 0.0
    assert split["normalization_source"] == "train_only"
    assert split["group_by_episode"] is True
    assert split["require_recording_session_id"] is False
    assert split["claim_cross_session_generalization"] is False

    offset_vectors = [tuple(value) for value in offsets.values()]
    assert len(offset_vectors) == 7
    assert len(set(offset_vectors)) == 7
    assert (0.0, 0.0, 0.0) in offset_vectors
    assert all(len(offset) == 3 for offset in offset_vectors)
    assert max(abs(value) for offset in offset_vectors for value in offset) <= 0.01

    assert 0.0 < gates["minimum_training_target_success_rate"] <= 1.0
    assert 0.0 < gates["minimum_held_out_target_success_rate"] <= 1.0
    assert gates["maximum_invalid_action_count"] == 0
    assert gates["maximum_workspace_violation_count"] == 0
    assert gates["maximum_joint_limit_violation_count"] == 0
