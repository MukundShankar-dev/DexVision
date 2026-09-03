from __future__ import annotations

from pathlib import Path

import torch
import yaml

from dexvision.apps.run_level3_diagnostics import build_parser
from dexvision.evaluation.level3_diagnostics import (
    build_diagnostic_matrix,
    load_level3_diagnostics_config,
)
from dexvision.learning.datasets import (
    ELIGIBILITY_RECOMPUTED_SUCCESS,
    GOAL_INPUT_FIXED_TRAINING_MEAN,
)
from dexvision.learning.models import (
    BASE_ORIENTATION_ACTION_NAMES,
    BASE_POSITION_ACTION_NAMES,
    FINGER_ACTION_PREFIX,
)


ROOT = Path(__file__).resolve().parents[1]


def test_diagnostic_matrix_is_explicit_and_controlled(tmp_path: Path) -> None:
    actions = (
        BASE_POSITION_ACTION_NAMES
        + BASE_ORIENTATION_ACTION_NAMES
        + (f"{FINGER_ACTION_PREFIX}finger-a", f"{FINGER_ACTION_PREFIX}finger-b")
    )
    checkpoint = tmp_path / "baseline.pt"
    torch.save(
        {
            "schema": {"dataset_action_names": list(actions)},
            "provenance": {
                "split_manifest": {
                    "assignments": [
                        {"episode_id": "episode-a", "split": "train"},
                        {"episode_id": "episode-b", "split": "validation"},
                    ]
                }
            },
        },
        checkpoint,
    )
    config_path = tmp_path / "diagnostics.yaml"
    training_configs = {
        "reach_touch_target": ROOT / "configs/level3_reach_bc_v2.yaml",
        "button_press": ROOT / "configs/level3_button_bc.yaml",
        "push_cube_to_target": ROOT / "configs/level3_push_bc.yaml",
    }
    config_path.write_text(
        yaml.safe_dump(
            {
                "version": "level3/data-action-diagnostics-v1",
                "dataset_root": str(tmp_path / "data"),
                "output_directory": str(tmp_path / "outputs"),
                "tasks": {
                    task_id: {
                        "training_config": str(training_config),
                        "baseline_checkpoint": str(checkpoint),
                        "baseline_report": str(tmp_path / f"{task_id}.json"),
                    }
                    for task_id, training_config in training_configs.items()
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_level3_diagnostics_config(config_path)
    matrix = build_diagnostic_matrix(config)

    assert len(matrix) == 13
    assert {item.variant for item in matrix if item.task_id == "reach_touch_target"} == {
        "full",
        "base_only",
        "finger_only",
        "fixed_training_mean",
        "broader_recomputed_success",
    }
    assert all(
        item.training_config.model == matrix[0].training_config.model
        and item.training_config.training == matrix[0].training_config.training
        for item in matrix
    )
    assert next(item for item in matrix if item.variant == "base_only").output_action_names == (
        BASE_POSITION_ACTION_NAMES + BASE_ORIENTATION_ACTION_NAMES
    )
    assert all(
        name.startswith(FINGER_ACTION_PREFIX)
        for name in next(
            item for item in matrix if item.variant == "finger_only"
        ).output_action_names
    )
    assert next(
        item for item in matrix if item.variant == "fixed_training_mean"
    ).goal_input_mode == GOAL_INPUT_FIXED_TRAINING_MEAN
    assert next(
        item for item in matrix if item.variant == "broader_recomputed_success"
    ).eligibility == ELIGIBILITY_RECOMPUTED_SUCCESS
    assert dict(
        next(
            item for item in matrix if item.variant == "broader_recomputed_success"
        ).reference_split_assignments
    ) == {"episode-a": "train", "episode-b": "validation"}


def test_diagnostics_cli_has_headless_defaults() -> None:
    args = build_parser().parse_args([])

    assert args.config == Path("configs/level3_diagnostics.yaml")
    assert args.model == Path("assets/mujoco/task_board_scene.xml")
