"""A failed pre-entry setup must never query an uninitialized observer."""

import json

import numpy as np
import pytest

from dexvision.evaluation.instrumented_recovery import (
    failed_place_entry,
    reject_failed_place_setup,
)


def setup_failure(tmp_path):
    directory = tmp_path / "state"
    directory.mkdir()
    result = {
        "setup": {"success": False, "reason": "physical_retention_failed"},
        "terminal_result": "rejected_before_execution",
        "terminal_reason": "place_setup_failed",
        "executed_actions": 0,
        "safety_counters": {"invalid_actions": 1},
        "execution_id": "place/5001/42/state/nominal",
        "entry_state_sha256": None,
    }
    (directory / "result.json").write_text(json.dumps(result))
    np.savez_compressed(
        directory / "trajectory.npz",
        qpos=np.zeros((1, 5)),
        qvel=np.zeros((1, 4)),
        state_timestamps=np.array([1.0]),
    )
    return directory / "result.json"


def test_failed_place_setup_rejects_before_observation_and_preserves_baseline(tmp_path):
    state = setup_failure(tmp_path)
    before = {p: p.read_bytes() for p in state.parent.iterdir()}
    output = tmp_path / "nominal"
    output.mkdir()  # exactly the zero-action interrupted case
    value = reject_failed_place_setup(
        state,
        output,
        condition="nominal",
        allow_empty=True,
        infrastructure_failure=True,
    )
    assert value["executed_actions"] == 0 and value["inference_latencies_s"] == []
    assert (
        value["failure_attribution"] == "physics_setup"
        and value["infrastructure_failure_before_control"]
    )
    assert value["physical_setup_reused_for_preentry_rejection"]
    assert value["safety_counters"] == {"invalid_actions": 1}
    assert value["execution_id"] == "place/5001/42/perception/nominal"
    assert {p: p.read_bytes() for p in before} == before
    with np.load(output / "trajectory.npz", allow_pickle=False) as data:
        assert data["qpos"].shape == (1, 5) and data["applied_actions"].shape == (0, 27)
    with pytest.raises(ValueError, match="Preserve"):
        reject_failed_place_setup(state, output, condition="nominal", allow_empty=True)


def test_valid_or_started_place_attempt_cannot_be_relabelled_setup_failure(tmp_path):
    path = setup_failure(tmp_path)
    result = json.loads(path.read_text())
    result["executed_actions"] = 1
    assert not failed_place_entry(result)
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="Only a measured failed"):
        reject_failed_place_setup(path, tmp_path / "new", condition="nominal")
    result["executed_actions"] = 0
    result["setup"]["success"] = True
    assert not failed_place_entry(result)
