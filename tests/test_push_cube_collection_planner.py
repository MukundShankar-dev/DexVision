from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from dexvision.apps import select_push_cube_goal
from dexvision.logging.collection_planner import (
    PushCubeCollectionPlan,
    format_push_cube_recording_command,
    plan_push_cube_collection,
)
from dexvision.logging.dataset_summary import (
    CubeGoalDefinition,
    PushCubeDatasetConfig,
)


def _config() -> PushCubeDatasetConfig:
    goals = tuple(
        CubeGoalDefinition(
            goal_id=f"{target}_goal",
            object_id="push_cube",
            initial_object_position=(-0.09, lane, -0.015),
            target_source=target,
            target_position=(0.09, lane, -0.015),
            approach_side=approach,
        )
        for target, lane, approach in (
            ("push_target_left", -0.07, "left"),
            ("push_target_center", 0.0, "front"),
            ("push_target_right", 0.07, "right"),
        )
    )
    return PushCubeDatasetConfig(
        version="test/push-cube-split-v1",
        task_id="push_cube_to_target",
        minimum_clean_successful_episodes=3,
        minimum_clean_per_training_goal=1,
        position_units="metres",
        coordinate_frame="MuJoCo world cube centre",
        training_goals=goals,
        held_out_evaluation_goals=(
            CubeGoalDefinition(
                goal_id="held_out",
                object_id="push_cube",
                initial_object_position=(-0.09, -0.035, -0.015),
                target_source="target_pose",
                target_position=(0.09, -0.035, -0.015),
                approach_side="left",
            ),
        ),
    )


def _write_episode(
    dataset: Path,
    name: str,
    goal: CubeGoalDefinition,
    *,
    clean: bool,
) -> None:
    episode = dataset / name
    episode.mkdir(parents=True)
    metadata = {
        "task_id": "push_cube_to_target",
        "task_config": {
            "resolved_object_id": goal.object_id,
            "initial_object_position": goal.initial_object_position,
            "resolved_target_source": goal.target_source,
            "target_position": goal.target_position,
            "resolved_approach_side": goal.approach_side,
        },
        "test_clean": clean,
    }
    (episode / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def _test_episode_is_clean(episode_dir: Path) -> bool:
    metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
    return bool(metadata["test_clean"])


def test_plan_balances_clean_cube_goals_and_never_overwrites(tmp_path: Path) -> None:
    config = _config()
    left, center, right = config.training_goals
    _write_episode(tmp_path, "2026-09-02_001", left, clean=True)
    _write_episode(tmp_path, "2026-09-02_002", left, clean=True)
    _write_episode(tmp_path, "2026-09-02_003", center, clean=False)
    _write_episode(tmp_path, "2026-09-02_004", right, clean=True)

    with patch(
        "dexvision.logging.collection_planner._episode_is_clean",
        side_effect=_test_episode_is_clean,
    ):
        plan = plan_push_cube_collection(
            tmp_path,
            config=config,
            collection_date=date(2026, 9, 2),
            seed=7,
        )

    assert plan.goal == center
    assert plan.goal_counts == (
        ("push_target_left_goal", 2),
        ("push_target_center_goal", 1),
        ("push_target_right_goal", 1),
    )
    assert plan.clean_goal_counts == (
        ("push_target_left_goal", 2),
        ("push_target_center_goal", 0),
        ("push_target_right_goal", 1),
    )
    assert plan.output_directory == tmp_path / "2026-09-02_005"
    assert not plan.output_directory.exists()
    command = format_push_cube_recording_command(plan)
    assert "--object-id push_cube" in command
    assert "--target-zone-id push_target_center" in command
    assert "--approach-side front" in command
    assert "--level1-13-full" in command


def test_selector_prints_goal_and_does_not_create_output(
    tmp_path: Path,
    capsys,
) -> None:
    goal = _config().training_goals[0]
    plan = PushCubeCollectionPlan(
        goal=goal,
        output_directory=tmp_path / "2026-09-02_001",
        goal_counts=((goal.goal_id, 0),),
        clean_goal_counts=((goal.goal_id, 0),),
    )
    args = select_push_cube_goal.build_parser().parse_args(["--dataset", str(tmp_path)])

    with patch.object(
        select_push_cube_goal,
        "plan_push_cube_collection",
        return_value=plan,
    ):
        result = select_push_cube_goal.run_selector(args)

    assert result == 0
    output = capsys.readouterr().out
    assert "Selected goal: push_target_left_goal" in output
    assert "Selected target: push_target_left" in output
    assert "No dataset files were created or modified." in output
    assert not plan.output_directory.exists()


def test_quality_gate_moves_clean_recording_into_raw(tmp_path: Path) -> None:
    final_output = (
        tmp_path / "data" / "demos" / "raw" / "push_cube_to_target" / "2026-09-02_001"
    )
    plan = PushCubeCollectionPlan(_config().training_goals[0], final_output, (), ())
    staging = (
        tmp_path / "data" / "demos" / "staging" / "push_cube_to_target" / "attempt"
    )

    def fake_run(
        arguments: tuple[str, ...], *, check: bool
    ) -> subprocess.CompletedProcess:
        assert str(staging) in arguments
        assert check is False
        staging.mkdir(parents=True)
        return subprocess.CompletedProcess(args=arguments, returncode=0)

    with (
        patch.object(select_push_cube_goal, "_staging_output", return_value=staging),
        patch.object(select_push_cube_goal.subprocess, "run", side_effect=fake_run),
        patch.object(
            select_push_cube_goal,
            "evaluate_episode_quality",
            return_value=SimpleNamespace(passed=True, failed_filters=()),
        ),
    ):
        result = select_push_cube_goal._run_quality_gated_recording(
            plan,
            python_command="mjpython",
        )

    assert result == 0
    assert final_output.is_dir()
    assert not staging.exists()


def test_quality_gate_preserves_failed_recording_outside_raw(tmp_path: Path) -> None:
    final_output = (
        tmp_path / "data" / "demos" / "raw" / "push_cube_to_target" / "2026-09-02_001"
    )
    plan = PushCubeCollectionPlan(_config().training_goals[0], final_output, (), ())
    staging = (
        tmp_path / "data" / "demos" / "staging" / "push_cube_to_target" / "attempt"
    )

    def fake_run(
        arguments: tuple[str, ...], *, check: bool
    ) -> subprocess.CompletedProcess:
        assert check is False
        staging.mkdir(parents=True)
        return subprocess.CompletedProcess(args=arguments, returncode=0)

    with (
        patch.object(select_push_cube_goal, "_staging_output", return_value=staging),
        patch.object(select_push_cube_goal.subprocess, "run", side_effect=fake_run),
        patch.object(
            select_push_cube_goal,
            "evaluate_episode_quality",
            return_value=SimpleNamespace(
                passed=False,
                failed_filters=("high_action_jerk",),
            ),
        ),
    ):
        result = select_push_cube_goal._run_quality_gated_recording(
            plan,
            python_command="mjpython",
        )

    rejected = (
        tmp_path
        / "data"
        / "demos"
        / "rejected"
        / "push_cube_to_target"
        / "2026-09-02_001"
    )
    assert result == 1
    assert rejected.is_dir()
    assert not final_output.exists()
