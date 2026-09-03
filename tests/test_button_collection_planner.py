from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dexvision.apps import select_button_goal
from dexvision.logging.collection_planner import (
    ButtonPressCollectionPlan,
    format_button_recording_command,
    plan_button_press_collection,
)
from dexvision.logging.dataset_summary import (
    ButtonGoalDefinition,
    ButtonPressDatasetConfig,
)


def _config() -> ButtonPressDatasetConfig:
    return ButtonPressDatasetConfig(
        version="test/button-press-split-v1",
        task_id="button_press",
        minimum_clean_successful_episodes=3,
        minimum_clean_per_training_goal=1,
        position_units="metres",
        press_depth_units="metres",
        coordinate_frame="MuJoCo world",
        training_goals=(
            ButtonGoalDefinition(
                "left_010", "button_left", (0.137, -0.08, 0.40), 0.010
            ),
            ButtonGoalDefinition(
                "center_012", "button_center", (0.137, 0.00, 0.40), 0.012
            ),
            ButtonGoalDefinition(
                "right_014", "button_right", (0.137, 0.08, 0.40), 0.014
            ),
        ),
        held_out_evaluation_goals=(
            ButtonGoalDefinition(
                "center_eval_011", "button_center", (0.137, 0.00, 0.40), 0.011
            ),
        ),
    )


def _write_episode(
    dataset: Path,
    name: str,
    goal: ButtonGoalDefinition,
    *,
    clean: bool,
) -> None:
    episode = dataset / name
    episode.mkdir(parents=True)
    metadata = {
        "task_id": "button_press",
        "task_config": {
            "resolved_button_id": goal.button_id,
            "target_press_depth": goal.target_press_depth,
        },
        "test_clean": clean,
    }
    (episode / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def _test_episode_is_clean(episode_dir: Path) -> bool:
    metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
    return bool(metadata["test_clean"])


def test_plan_balances_clean_button_depth_goals_and_never_overwrites(
    tmp_path: Path,
) -> None:
    config = _config()
    left, center, right = config.training_goals
    _write_episode(tmp_path, "2026-09-02_001", left, clean=False)
    _write_episode(tmp_path, "2026-09-02_002", left, clean=False)
    _write_episode(tmp_path, "2026-09-02_003", center, clean=True)
    _write_episode(tmp_path, "2026-09-02_004", right, clean=True)

    with patch(
        "dexvision.logging.collection_planner._episode_is_clean",
        side_effect=_test_episode_is_clean,
    ):
        plan = plan_button_press_collection(
            tmp_path,
            config=config,
            collection_date=date(2026, 9, 2),
            seed=7,
        )

    assert plan.goal == left
    assert plan.goal_counts == (("left_010", 2), ("center_012", 1), ("right_014", 1))
    assert plan.clean_goal_counts == (
        ("left_010", 0),
        ("center_012", 1),
        ("right_014", 1),
    )
    assert plan.output_directory == tmp_path / "2026-09-02_005"
    assert not plan.output_directory.exists()
    command = format_button_recording_command(plan)
    assert "--button-id button_left" in command
    assert "--target-press-depth 0.01" in command
    assert "--level1-13-full" in command


def test_selector_prints_goal_and_does_not_create_output(
    tmp_path: Path,
    capsys,
) -> None:
    goal = _config().training_goals[0]
    plan = ButtonPressCollectionPlan(
        goal=goal,
        output_directory=tmp_path / "2026-09-02_001",
        goal_counts=((goal.goal_id, 0),),
        clean_goal_counts=((goal.goal_id, 0),),
    )
    args = select_button_goal.build_parser().parse_args(["--dataset", str(tmp_path)])

    with patch.object(
        select_button_goal,
        "plan_button_press_collection",
        return_value=plan,
    ):
        result = select_button_goal.run_selector(args)

    assert result == 0
    output = capsys.readouterr().out
    assert "Selected goal: left_010" in output
    assert "Selected button: button_left" in output
    assert "No dataset files were created or modified." in output
    assert not plan.output_directory.exists()


def test_quality_gate_moves_clean_recording_into_raw(tmp_path: Path) -> None:
    final_output = (
        tmp_path / "data" / "demos" / "raw" / "button_press" / "2026-09-02_001"
    )
    goal = _config().training_goals[0]
    plan = ButtonPressCollectionPlan(goal, final_output, (), ())
    staging = tmp_path / "data" / "demos" / "staging" / "button_press" / "attempt"

    def fake_run(
        arguments: tuple[str, ...], *, check: bool
    ) -> subprocess.CompletedProcess:
        assert str(staging) in arguments
        assert check is False
        staging.mkdir(parents=True)
        return subprocess.CompletedProcess(args=arguments, returncode=0)

    with (
        patch.object(select_button_goal, "_staging_output", return_value=staging),
        patch.object(select_button_goal.subprocess, "run", side_effect=fake_run),
        patch.object(
            select_button_goal,
            "evaluate_episode_quality",
            return_value=SimpleNamespace(passed=True, failed_filters=()),
        ),
    ):
        result = select_button_goal._run_quality_gated_recording(
            plan,
            python_command="mjpython",
        )

    assert result == 0
    assert final_output.is_dir()
    assert not staging.exists()


def test_quality_gate_preserves_failed_recording_outside_raw(tmp_path: Path) -> None:
    final_output = (
        tmp_path / "data" / "demos" / "raw" / "button_press" / "2026-09-02_001"
    )
    goal = _config().training_goals[0]
    plan = ButtonPressCollectionPlan(goal, final_output, (), ())
    staging = tmp_path / "data" / "demos" / "staging" / "button_press" / "attempt"

    def fake_run(
        arguments: tuple[str, ...], *, check: bool
    ) -> subprocess.CompletedProcess:
        assert check is False
        staging.mkdir(parents=True)
        return subprocess.CompletedProcess(args=arguments, returncode=0)

    with (
        patch.object(select_button_goal, "_staging_output", return_value=staging),
        patch.object(select_button_goal.subprocess, "run", side_effect=fake_run),
        patch.object(
            select_button_goal,
            "evaluate_episode_quality",
            return_value=SimpleNamespace(
                passed=False,
                failed_filters=("high_action_jerk",),
            ),
        ),
    ):
        result = select_button_goal._run_quality_gated_recording(
            plan,
            python_command="mjpython",
        )

    rejected = (
        tmp_path / "data" / "demos" / "rejected" / "button_press" / "2026-09-02_001"
    )
    assert result == 1
    assert rejected.is_dir()
    assert not final_output.exists()
