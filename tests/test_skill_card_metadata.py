from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from dexvision.apps.export_skill_metadata import load_task_spec, main
from dexvision.logging.skill_card_metadata import (
    SKILL_METADATA_SCHEMA_VERSION,
    SkillMetadataError,
    build_skill_metadata,
    save_skill_metadata,
)
from dexvision.sim.tasks import (
    BUTTON_PRESS_TASK_ID,
    PUSH_CUBE_TASK_ID,
    REACH_TOUCH_TARGET_TASK_ID,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "assets" / "mujoco" / "task_board_scene.xml"


def _write_summary(path: Path, spec: object) -> Path:
    payload = {
        "version": "level2/dataset-summary-v4",
        "dataset": "data/demos",
        "groups": [
            {
                "skill_name": spec.skill_name,
                "task_id": spec.task_id,
                "action_schema_version": spec.action_schema.version,
                "observation_schema_version": spec.observation_schema.version,
                "num_episodes": 12,
                "clean_success_count": 10,
                "level3_ready": True,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "task_id",
    (REACH_TOUCH_TARGET_TASK_ID, BUTTON_PRESS_TASK_ID, PUSH_CUBE_TASK_ID),
)
def test_metadata_stub_exports_every_implemented_level2_skill(
    tmp_path: Path,
    task_id: str,
) -> None:
    pytest.importorskip("mujoco")
    spec = load_task_spec(task_id, MODEL_PATH)
    summary_path = _write_summary(tmp_path / "dataset_summary.json", spec)

    metadata = build_skill_metadata(spec, dataset_summary_path=summary_path)
    output = save_skill_metadata(metadata, tmp_path / f"{task_id}.json")
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert saved["metadata_schema_version"] == SKILL_METADATA_SCHEMA_VERSION
    assert saved["skill_name"] == spec.skill_name
    assert saved["task_id"] == task_id
    assert saved["policy_checkpoint"] is None
    assert saved["dataset_summary_path"] == str(summary_path)
    assert saved["dataset_summary"]["level3_ready"] is True
    assert saved["success_condition"] == spec.success_condition
    assert saved["failure_conditions"] == list(spec.failure_conditions)
    assert saved["timeout"] == {
        "max_episode_steps": spec.max_episode_steps,
        "units": "control steps",
    }
    assert "success" in saved["terminal_state_fields"]
    assert "failure_reason" in saved["terminal_state_fields"]


def test_action_and_parameter_contracts_are_explicit(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    spec = load_task_spec(BUTTON_PRESS_TASK_ID, MODEL_PATH)
    summary_path = _write_summary(tmp_path / "dataset_summary.json", spec)

    metadata = build_skill_metadata(spec, dataset_summary_path=summary_path)

    assert set(metadata.action_schema) == {
        "base_position_target",
        "base_orientation_target",
        "finger_actuator_targets",
    }
    assert metadata.action_schema["base_position_target"]["shape"] == [3]
    assert metadata.action_schema["base_orientation_target"]["shape"] == [4]
    assert metadata.action_schema["finger_actuator_targets"]["names"]
    for contract in metadata.parameter_schema.values():
        assert contract["type"]
        assert "shape" in contract
        assert contract["units"]
        assert contract["coordinate_frame"]
        assert isinstance(contract["required"], bool)


def test_summary_schema_mismatch_is_rejected(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    spec = load_task_spec(REACH_TOUCH_TARGET_TASK_ID, MODEL_PATH)
    summary_path = _write_summary(tmp_path / "dataset_summary.json", spec)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["groups"][0]["action_schema_version"] = "wrong/action-schema"
    summary_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SkillMetadataError, match="does not match task spec"):
        build_skill_metadata(spec, dataset_summary_path=summary_path)


def test_cli_exports_json_without_policy_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    spec = load_task_spec(PUSH_CUBE_TASK_ID, MODEL_PATH)
    summary_path = _write_summary(tmp_path / "dataset_summary.json", spec)
    output = tmp_path / "push_cube_metadata.json"

    result = main(
        [
            "--task",
            PUSH_CUBE_TASK_ID,
            "--model",
            str(MODEL_PATH),
            "--dataset-summary",
            str(summary_path),
            "--output",
            str(output),
        ]
    )

    assert result == 0
    assert json.loads(output.read_text(encoding="utf-8"))["policy_checkpoint"] is None


def test_cli_help_runs_without_loading_mujoco() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "dexvision.apps.export_skill_metadata", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--dataset-summary" in completed.stdout
    assert "--skill-version" in completed.stdout
