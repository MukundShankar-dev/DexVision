from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from dexvision.apps import summarize_demos
from dexvision.logging.dataset_summary import (
    ButtonGoalDefinition,
    ButtonPressDatasetConfig,
    DATASET_SUMMARY_VERSION,
    DEFAULT_BUTTON_PRESS_CONFIG,
    ReachTouchDatasetConfig,
    TargetDefinition,
    default_summary_paths,
    load_button_press_dataset_config,
    save_dataset_summary,
    summarize_demo_dataset,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_episode(
    dataset: Path,
    name: str,
    *,
    skill_name: str = "reach_touch_target",
    task_id: str = "reach_touch_target",
    success: bool | None = True,
    frame_count: int = 4,
    tracking_confidence: float = 0.9,
    action_schema_version: str = "level1.13/full-action-v1",
    observation_schema_version: str = "level2/observation-layout-v2",
    target_site: str = "reach_target_center",
) -> Path:
    episode = dataset / name
    episode.mkdir(parents=True)
    metadata = {
        "episode_id": f"{task_id}_{name}",
        "skill_name": skill_name,
        "task_id": task_id,
        "success": success,
        "action_schema_version": action_schema_version,
        "action_schema": {"version": action_schema_version},
        "observation_schema_version": observation_schema_version,
        "observation_schema": {"version": observation_schema_version},
        "tracking_quality_fields": [
            "detected",
            "hand_tracking_confidence",
            "feature_confidence",
        ],
    }
    if task_id == "reach_touch_target":
        target_positions = {
            "reach_target_left": [0.14, -0.10, 0.45],
            "reach_target_center": [0.14, 0.00, 0.49],
            "reach_target_right": [0.14, 0.06, 0.51],
        }
        metadata["task_config"] = {
            "resolved_target_source": target_site,
            "target_position": target_positions[target_site],
        }
    elif task_id == "push_cube_to_target":
        metadata["task_config"] = {
            "resolved_target_source": "push_target_left",
            "target_position": [0.09, -0.07, -0.015],
        }
    elif task_id == "button_press":
        button_positions = {
            "button_left": [0.137, -0.08, 0.40],
            "button_center": [0.137, 0.00, 0.40],
            "button_right": [0.137, 0.08, 0.40],
        }
        metadata["task_config"] = {
            "resolved_button_id": target_site,
            "button_position": button_positions[target_site],
            "target_press_depth": 0.01,
            "initial_button_depth": 0.0,
            "initial_base_position": [0.0, 0.0, 0.14],
            "initial_base_orientation": [1.0, 0.0, 0.0, 0.0],
        }
    (episode / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    np.save(episode / "timestamps.npy", np.arange(frame_count, dtype=np.float64))
    tracking = np.ones((frame_count, 3), dtype=np.float64)
    tracking[:, 1] = tracking_confidence
    np.save(episode / "tracking_quality.npy", tracking)
    return episode


def _write_quality_report(
    dataset: Path,
    entries: tuple[tuple[Path, bool, tuple[str, ...]], ...],
) -> None:
    report = {
        "version": "level2/pilot-quality-report-v1",
        "episodes": [
            {
                "episode_directory": episode.name,
                "episode_id": json.loads(
                    (episode / "metadata.json").read_text(encoding="utf-8")
                )["episode_id"],
                "passed": passed,
                "failed_filters": list(failed_filters),
            }
            for episode, passed, failed_filters in entries
        ],
    }
    (dataset / "quality_report.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )


def _write_relabel_report(
    dataset: Path,
    entries: tuple[tuple[Path, bool | None, bool], ...],
) -> None:
    report = {
        "version": "level2/reach-touch-success-v1",
        "episodes": [
            {
                "episode_directory": episode.name,
                "episode_id": json.loads(
                    (episode / "metadata.json").read_text(encoding="utf-8")
                )["episode_id"],
                "operator_success": operator_success,
                "recomputed_success": recomputed_success,
                "labels_agree": (
                    None
                    if operator_success is None
                    else operator_success == recomputed_success
                ),
            }
            for episode, operator_success, recomputed_success in entries
        ],
    }
    (dataset / "relabel_report.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )


def test_summary_reports_metrics_quality_failures_and_relabel_disagreements(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "raw" / "reach_touch_target"
    first = _write_episode(
        dataset,
        "episode_001",
        success=True,
        frame_count=4,
        tracking_confidence=0.8,
    )
    second = _write_episode(
        dataset,
        "episode_002",
        success=True,
        frame_count=6,
        tracking_confidence=1.0,
    )
    _write_quality_report(
        dataset,
        (
            (first, True, ()),
            (second, False, ("high_action_jerk",)),
        ),
    )
    _write_relabel_report(
        dataset,
        (
            (first, True, True),
            (second, True, False),
        ),
    )

    report = summarize_demo_dataset(tmp_path)

    assert report.version == DATASET_SUMMARY_VERSION
    assert report.num_groups == 1
    assert report.num_episodes == 2
    assert report.raw_episodes_modified is False
    assert report.warnings == ()
    group = report.groups[0]
    assert group.skill_name == "reach_touch_target"
    assert group.task_id == "reach_touch_target"
    assert group.num_episodes == 2
    assert group.num_success == 1
    assert group.num_unlabeled == 0
    assert group.success_rate == 0.5
    assert group.mean_episode_length == 5.0
    assert np.isclose(group.mean_tracking_confidence, 0.9)
    assert group.quality_pass_count == 1
    assert group.quality_fail_count == 1
    assert group.quality_unreported_count == 0
    assert group.relabel_disagreement_count == 1
    assert group.relabel_unreported_count == 0
    assert group.action_schema_version == "level1.13/full-action-v1"
    assert group.observation_schema_version == "level2/observation-layout-v2"
    assert group.clean_success_count == 1
    assert group.target_position_distribution[0].target_id == "reach_target_center"
    assert group.level3_ready is None
    assert group.quality_failures[0].episode_id.endswith("episode_002")
    assert group.quality_failures[0].failed_filters == ("high_action_jerk",)
    assert group.relabel_disagreements[0].operator_success is True
    assert group.relabel_disagreements[0].recomputed_success is False


def test_summary_is_grouped_per_skill_and_reports_missing_coverage(
    tmp_path: Path,
) -> None:
    _write_episode(
        tmp_path / "raw" / "reach_touch_target",
        "reach_001",
    )
    _write_episode(
        tmp_path / "raw" / "free_space_gesture",
        "gesture_001",
        skill_name="free_space_gesture",
        task_id="free_space_gesture",
        success=None,
        action_schema_version="level1.13/full-action-v1",
        observation_schema_version="level2/observation-v1",
    )

    report = summarize_demo_dataset(tmp_path)

    assert [(group.skill_name, group.task_id) for group in report.groups] == [
        ("free_space_gesture", "free_space_gesture"),
        ("reach_touch_target", "reach_touch_target"),
    ]
    gesture = report.groups[0]
    assert gesture.num_unlabeled == 1
    assert gesture.success_rate is None
    assert gesture.quality_unreported_count == 1
    assert gesture.relabel_unreported_count == 1
    assert len(report.warnings) == 4
    assert all("coverage is missing" in warning for warning in report.warnings)


def test_dataset_root_uses_raw_subtree_and_excludes_smoke_recordings(
    tmp_path: Path,
) -> None:
    _write_episode(
        tmp_path / "raw" / "reach_touch_target",
        "raw_episode",
    )
    _write_episode(
        tmp_path / "free_space_smoke_check",
        "smoke_episode",
        skill_name="free_space_gesture",
        task_id="free_space_gesture",
    )

    report = summarize_demo_dataset(tmp_path)

    assert report.num_episodes == 1
    assert report.groups[0].skill_name == "reach_touch_target"


def test_missing_and_empty_datasets_produce_clear_warnings(tmp_path: Path) -> None:
    missing = summarize_demo_dataset(tmp_path / "missing")
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    empty = summarize_demo_dataset(empty_dir)

    assert missing.num_episodes == 0
    assert "does not exist" in missing.warnings[0]
    assert empty.num_episodes == 0
    assert "No episode directories" in empty.warnings[0]


def test_json_and_csv_outputs_are_saved(tmp_path: Path) -> None:
    dataset = tmp_path / "demos"
    _write_episode(dataset, "episode_001")
    report = summarize_demo_dataset(dataset)
    json_path = tmp_path / "reports" / "summary.json"
    csv_path = tmp_path / "reports" / "summary.csv"

    saved_json, saved_csv = save_dataset_summary(
        report,
        json_path=json_path,
        csv_path=csv_path,
    )

    assert saved_json == json_path
    assert saved_csv == csv_path
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["groups"][0]["num_episodes"] == 1
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["skill_name"] == "reach_touch_target"
    assert rows[0]["action_schema_version"] == "level1.13/full-action-v1"
    assert rows[0]["clean_success_count"] == "0"


def test_push_cube_pilot_summary_uses_relabel_and_quality_reports(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "raw" / "push_cube_to_target"
    episode = _write_episode(
        dataset,
        "episode_001",
        skill_name="push_cube_to_target",
        task_id="push_cube_to_target",
        success=False,
    )
    _write_quality_report(dataset, ((episode, True, ()),))
    _write_relabel_report(dataset, ((episode, False, True),))

    report = summarize_demo_dataset(tmp_path)
    group = report.groups[0]

    assert group.skill_name == "push_cube_to_target"
    assert group.task_id == "push_cube_to_target"
    assert group.num_episodes == 1
    assert group.num_success == 1
    assert group.success_rate == 1.0
    assert group.quality_pass_count == 1
    assert group.quality_unreported_count == 0
    assert group.relabel_unreported_count == 0
    assert group.relabel_disagreement_count == 1
    assert group.clean_success_count == 1
    assert group.target_position_distribution[0].target_id == "push_target_left"
    assert group.target_position_distribution[0].quality_pass_count == 1
    assert group.level3_ready is None


def test_reach_touch_summary_marks_balanced_clean_dataset_ready(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "raw" / "reach_touch_target"
    episodes = tuple(
        _write_episode(
            dataset,
            f"episode_{index:03d}",
            target_site=target,
        )
        for index, target in enumerate(
            ("reach_target_left", "reach_target_center", "reach_target_right"),
            start=1,
        )
    )
    _write_quality_report(
        dataset,
        tuple((episode, True, ()) for episode in episodes),
    )
    _write_relabel_report(
        dataset,
        tuple((episode, True, True) for episode in episodes),
    )
    config = ReachTouchDatasetConfig(
        version="test/reach-touch-split-v1",
        task_id="reach_touch_target",
        minimum_clean_successful_episodes=3,
        minimum_clean_per_training_target=1,
        position_units="metres",
        coordinate_frame="MuJoCo world",
        training_targets=(
            TargetDefinition("reach_target_left", (0.14, -0.10, 0.45)),
            TargetDefinition("reach_target_center", (0.14, 0.00, 0.49)),
            TargetDefinition("reach_target_right", (0.14, 0.06, 0.51)),
        ),
        held_out_evaluation_targets=(
            TargetDefinition("reach_eval_left_center", (0.14, -0.05, 0.47)),
        ),
    )

    report = summarize_demo_dataset(tmp_path, reach_touch_config=config)

    group = report.groups[0]
    assert group.clean_success_count == 3
    assert [target.clean_success_count for target in group.target_position_distribution] == [
        1,
        1,
        1,
    ]
    assert group.held_out_evaluation_targets == config.held_out_evaluation_targets
    assert group.readiness_config_version == config.version
    assert group.level3_ready is True
    assert group.readiness_failures == ()


def test_button_summary_reports_initial_and_goal_distributions_and_readiness(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "raw" / "button_press"
    episodes = tuple(
        _write_episode(
            dataset,
            f"episode_{index:03d}",
            skill_name="button_press",
            task_id="button_press",
            target_site=button_id,
        )
        for index, button_id in enumerate(
            ("button_left", "button_center", "button_right"),
            start=1,
        )
    )
    _write_quality_report(
        dataset,
        tuple((episode, True, ()) for episode in episodes),
    )
    _write_relabel_report(
        dataset,
        tuple((episode, True, True) for episode in episodes),
    )
    config = ButtonPressDatasetConfig(
        version="test/button-press-split-v1",
        task_id="button_press",
        minimum_clean_successful_episodes=3,
        minimum_clean_per_training_goal=1,
        position_units="metres",
        press_depth_units="metres",
        coordinate_frame="MuJoCo world",
        training_goals=(
            ButtonGoalDefinition(
                "left_010", "button_left", (0.137, -0.08, 0.40), 0.01
            ),
            ButtonGoalDefinition(
                "center_010", "button_center", (0.137, 0.00, 0.40), 0.01
            ),
            ButtonGoalDefinition(
                "right_010", "button_right", (0.137, 0.08, 0.40), 0.01
            ),
        ),
        held_out_evaluation_goals=(
            ButtonGoalDefinition(
                "center_eval_011", "button_center", (0.137, 0.00, 0.40), 0.011
            ),
        ),
    )

    report = summarize_demo_dataset(tmp_path, button_press_config=config)

    group = report.groups[0]
    assert group.clean_success_count == 3
    assert [goal.clean_success_count for goal in group.button_goal_distribution] == [
        1,
        1,
        1,
    ]
    assert len(group.button_initial_state_distribution) == 3
    assert sum(
        state.num_episodes for state in group.button_initial_state_distribution
    ) == 3
    assert group.held_out_button_goals == config.held_out_evaluation_goals
    assert group.readiness_config_version == config.version
    assert group.minimum_clean_per_training_goal == 1
    assert group.level3_ready is True
    assert group.readiness_failures == ()


def test_button_dataset_config_declares_distinct_held_out_states() -> None:
    config = load_button_press_dataset_config(ROOT / DEFAULT_BUTTON_PRESS_CONFIG)

    assert config.task_id == "button_press"
    assert config.minimum_clean_successful_episodes == 50
    assert config.minimum_clean_per_training_goal == 5
    assert len(config.training_goals) == 9
    assert len(config.held_out_evaluation_goals) == 3
    training_states = {
        (goal.button_id, goal.target_press_depth) for goal in config.training_goals
    }
    held_out_states = {
        (goal.button_id, goal.target_press_depth)
        for goal in config.held_out_evaluation_goals
    }
    assert training_states.isdisjoint(held_out_states)


def test_cli_saves_default_outputs_and_keeps_episode_files_unchanged(
    tmp_path: Path,
    capsys,
) -> None:
    dataset = tmp_path / "demos"
    episode = _write_episode(dataset, "episode_001")
    metadata_before = (episode / "metadata.json").read_bytes()
    timestamps_before = (episode / "timestamps.npy").read_bytes()
    expected_json, expected_csv = default_summary_paths(dataset)

    result = summarize_demos.main(["--dataset", str(dataset)])

    assert result == 0
    assert expected_json.is_file()
    assert expected_csv.is_file()
    assert (episode / "metadata.json").read_bytes() == metadata_before
    assert (episode / "timestamps.npy").read_bytes() == timestamps_before
    output = capsys.readouterr()
    assert "Summary complete: groups=1, episodes=1" in output.out
    assert "WARNING:" in output.err


def test_cli_missing_dataset_warns_and_saves_empty_summary(
    tmp_path: Path,
    capsys,
) -> None:
    dataset = tmp_path / "missing"

    result = summarize_demos.main(["--dataset", str(dataset)])

    assert result == 0
    json_path, csv_path = default_summary_paths(dataset)
    assert json_path.is_file()
    assert csv_path.is_file()
    output = capsys.readouterr()
    assert "WARNING: Dataset directory does not exist" in output.err
