from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from dexvision.apps import select_reach_target
from dexvision.logging.collection_planner import (
    CollectionPlannerError,
    format_recording_command,
    plan_reach_touch_collection,
)


def _write_episode(
    dataset: Path,
    name: str,
    target_site: str,
    *,
    clean: bool = True,
) -> None:
    episode = dataset / name
    episode.mkdir(parents=True)
    metadata = {
        "task_id": "reach_touch_target",
        "task_config": {"resolved_target_source": target_site},
        "test_clean": clean,
    }
    (episode / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_plan_randomizes_only_among_targets_with_fewest_clean_demos(
    tmp_path: Path,
) -> None:
    _write_episode(tmp_path, "2026-07-18_001", "reach_target_left", clean=False)
    _write_episode(tmp_path, "2026-07-18_002", "reach_target_left", clean=False)
    _write_episode(tmp_path, "2026-07-18_003", "reach_target_center")
    _write_episode(tmp_path, "2026-07-18_004", "reach_target_right")

    with patch(
        "dexvision.logging.collection_planner._episode_is_clean",
        side_effect=_test_episode_is_clean,
    ):
        plan = plan_reach_touch_collection(
            tmp_path,
            collection_date=date(2026, 7, 18),
            seed=7,
        )

    assert plan.target_site == "reach_target_left"
    assert plan.target_counts == (
        ("reach_target_left", 2),
        ("reach_target_center", 1),
        ("reach_target_right", 1),
    )
    assert plan.clean_target_counts == (
        ("reach_target_left", 0),
        ("reach_target_center", 1),
        ("reach_target_right", 1),
    )
    assert plan.output_directory == tmp_path / "2026-07-18_005"


def test_tied_selection_is_seeded_and_does_not_create_output(tmp_path: Path) -> None:
    for index, target in enumerate(
        ("reach_target_left", "reach_target_center", "reach_target_right"),
        start=1,
    ):
        _write_episode(tmp_path, f"2026-07-18_{index:03d}", target)

    with patch(
        "dexvision.logging.collection_planner._episode_is_clean",
        side_effect=_test_episode_is_clean,
    ):
        first = plan_reach_touch_collection(
            tmp_path,
            collection_date=date(2026, 7, 18),
            seed=11,
        )
        second = plan_reach_touch_collection(
            tmp_path,
            collection_date=date(2026, 7, 18),
            seed=11,
        )

    assert first.target_site == second.target_site
    assert first.clean_target_counts == first.target_counts
    assert first.output_directory == tmp_path / "2026-07-18_004"
    assert not first.output_directory.exists()
    command = format_recording_command(first)
    assert f"--target-site {first.target_site}" in command
    assert "--output" in command
    assert "--level1-13-full" in command


def test_invalid_target_metadata_is_rejected(tmp_path: Path) -> None:
    _write_episode(tmp_path, "2026-07-18_001", "unknown_target")

    with pytest.raises(CollectionPlannerError, match="unsupported"):
        plan_reach_touch_collection(tmp_path)


def test_selector_prints_ready_to_run_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = select_reach_target.build_parser().parse_args(
        [
            "--dataset",
            str(tmp_path),
            "--date",
            "2026-07-18",
            "--seed",
            "3",
        ]
    )

    assert select_reach_target.run_selector(args) == 0
    output = capsys.readouterr().out
    assert "Selected target:" in output
    assert str(tmp_path / "2026-07-18_001") in output
    assert "dexvision.apps.record_demo" in output
    assert "No dataset files were created or modified." in output


def test_selector_run_launches_planned_recorder(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = select_reach_target.build_parser().parse_args(
        [
            "--dataset",
            str(tmp_path),
            "--date",
            "2026-07-18",
            "--seed",
            "3",
            "--run",
        ]
    )

    with patch.object(
        select_reach_target,
        "_run_quality_gated_recording",
        return_value=0,
    ) as run:
        assert select_reach_target.run_selector(args) == 0

    plan = run.call_args.args[0]
    assert plan.output_directory == tmp_path / "2026-07-18_001"
    assert run.call_args.kwargs == {"python_command": "mjpython"}
    output = capsys.readouterr().out
    assert "Current clean counts:" in output
    assert "No dataset files were created or modified." not in output


def test_quality_gate_moves_clean_recording_into_raw(tmp_path: Path) -> None:
    dataset = tmp_path / "data" / "demos" / "raw" / "reach_touch_target"
    plan = plan_reach_touch_collection(
        dataset,
        collection_date=date(2026, 7, 18),
        seed=1,
    )
    staging = (
        tmp_path
        / "data"
        / "demos"
        / "staging"
        / "reach_touch_target"
        / "attempt"
    )

    def fake_run(arguments: tuple[str, ...], *, check: bool) -> subprocess.CompletedProcess:
        assert str(staging) in arguments
        assert check is False
        staging.mkdir(parents=True)
        return subprocess.CompletedProcess(args=arguments, returncode=0)

    with (
        patch.object(select_reach_target, "_staging_output", return_value=staging),
        patch.object(select_reach_target.subprocess, "run", side_effect=fake_run),
        patch.object(
            select_reach_target,
            "evaluate_episode_quality",
            return_value=SimpleNamespace(passed=True, failed_filters=()),
        ),
    ):
        result = select_reach_target._run_quality_gated_recording(
            plan,
            python_command="mjpython",
        )

    assert result == 0
    assert plan.output_directory.is_dir()
    assert not staging.exists()


def test_quality_gate_preserves_failed_recording_outside_raw(tmp_path: Path) -> None:
    dataset = tmp_path / "data" / "demos" / "raw" / "reach_touch_target"
    plan = plan_reach_touch_collection(
        dataset,
        collection_date=date(2026, 7, 18),
        seed=1,
    )
    staging = (
        tmp_path
        / "data"
        / "demos"
        / "staging"
        / "reach_touch_target"
        / "attempt"
    )

    def fake_run(arguments: tuple[str, ...], *, check: bool) -> subprocess.CompletedProcess:
        assert check is False
        staging.mkdir(parents=True)
        return subprocess.CompletedProcess(args=arguments, returncode=0)

    with (
        patch.object(select_reach_target, "_staging_output", return_value=staging),
        patch.object(select_reach_target.subprocess, "run", side_effect=fake_run),
        patch.object(
            select_reach_target,
            "evaluate_episode_quality",
            return_value=SimpleNamespace(
                passed=False,
                failed_filters=("high_action_jerk",),
            ),
        ),
    ):
        result = select_reach_target._run_quality_gated_recording(
            plan,
            python_command="mjpython",
        )

    rejected = (
        tmp_path
        / "data"
        / "demos"
        / "rejected"
        / "reach_touch_target"
        / "2026-07-18_001"
    )
    assert result == 1
    assert rejected.is_dir()
    assert not plan.output_directory.exists()


def _test_episode_is_clean(episode_dir: Path) -> bool:
    metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
    return bool(metadata["test_clean"])
