from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from dexvision.evaluation.benchmark_retargeters import (
    BENCHMARK_VERSION,
    BenchmarkReport,
    ConfidenceInterval,
    RetargeterMetrics,
    RetargetingBenchmarkError,
    bootstrap_confidence_interval,
    discover_task_episodes,
    joint_limit_violation_rate,
    mean_action_jerk,
    mean_fingertip_error,
    save_benchmark_plot,
    save_benchmark_report,
)


def test_action_jerk_is_zero_for_linear_motion_and_positive_for_jump() -> None:
    linear = np.arange(6, dtype=np.float64)[:, None]
    jump = np.asarray([[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]])

    assert mean_action_jerk(linear) == pytest.approx(0.0)
    assert mean_action_jerk(jump) > 0.0
    assert mean_action_jerk(linear[:3]) == 0.0


def test_joint_limit_violation_rate_counts_scalar_values() -> None:
    actions = np.asarray([[0.0, 1.0], [-0.1, 1.1]])

    rate = joint_limit_violation_rate(
        actions,
        lower_limits=np.asarray([0.0, 0.0]),
        upper_limits=np.asarray([1.0, 1.0]),
    )

    assert rate == pytest.approx(0.5)


def test_mean_fingertip_error_uses_euclidean_distance() -> None:
    target = np.zeros((2, 5, 3), dtype=np.float64)
    predicted = target.copy()
    predicted[..., 0] = 2.0

    assert mean_fingertip_error(predicted, target) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="matching shapes"):
        mean_fingertip_error(predicted[0], target)


def test_episode_bootstrap_interval_is_deterministic_and_contains_mean() -> None:
    values = np.asarray([0.0, 0.0, 1.0, 1.0])

    first = bootstrap_confidence_interval(values, samples=500, seed=7)
    second = bootstrap_confidence_interval(values, samples=500, seed=7)

    assert first == second
    assert first.lower <= float(np.mean(values)) <= first.upper
    assert bootstrap_confidence_interval([0.25]) == ConfidenceInterval(0.25, 0.25)


def test_episode_discovery_is_sorted_and_requires_requested_count(
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "push_cube_to_target"
    for name in ("episode_b", "episode_a"):
        episode_dir = task_dir / name
        episode_dir.mkdir(parents=True)
        (episode_dir / "metadata.json").write_text("{}", encoding="utf-8")
    (task_dir / "quality_report.json").write_text("{}", encoding="utf-8")

    selected = discover_task_episodes(
        tmp_path, task_id="push_cube_to_target", episodes=2
    )

    assert [path.name for path in selected] == ["episode_a", "episode_b"]
    with pytest.raises(RetargetingBenchmarkError, match="requested 3"):
        discover_task_episodes(
            tmp_path, task_id="push_cube_to_target", episodes=3
        )


def test_json_csv_and_svg_outputs_contain_all_required_metrics(tmp_path: Path) -> None:
    metric = RetargeterMetrics(
        retargeter="curl",
        episodes=2,
        frames=12,
        mean_latency_ms=0.25,
        mean_action_jerk=0.1,
        joint_limit_violation_rate=0.0,
        mean_fingertip_error=0.2,
        task_success_rate=1.0,
        mean_fingertip_object_distance_m=0.015,
        fingertip_contact_frame_rate=0.4,
        confidence_intervals={
            "mean_latency_ms": ConfidenceInterval(0.2, 0.3),
            "fingertip_contact_frame_rate": ConfidenceInterval(0.2, 0.6),
        },
    )
    report = BenchmarkReport(
        benchmark_version=BENCHMARK_VERSION,
        generated_at_utc="2026-09-02T00:00:00+00:00",
        task_id="push_cube_to_target",
        episode_ids=("a", "b"),
        config_path="configs/level1_teleop.yaml",
        success_evaluation="synthetic",
        bootstrap_samples=2000,
        bootstrap_seed=0,
        action_jerk_units="normalized actuator units/frame^3",
        fingertip_error_units="palm widths",
        fingertip_object_distance_units="metres",
        metrics=(metric,),
    )
    json_path, csv_path = save_benchmark_report(
        report,
        json_path=tmp_path / "report.json",
        csv_path=tmp_path / "report.csv",
    )
    svg_path = save_benchmark_plot(report, tmp_path / "report.svg")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    svg = svg_path.read_text(encoding="utf-8")

    assert payload["benchmark_version"] == BENCHMARK_VERSION
    assert payload["metrics"][0]["mean_latency_ms"] == 0.25
    assert rows[0]["retargeter"] == "curl"
    assert "mean_action_jerk" in rows[0]
    assert float(rows[0]["mean_latency_ms_ci95_low"]) == 0.2
    assert float(rows[0]["fingertip_contact_frame_rate_ci95_high"]) == 0.6
    assert "Mean latency" in svg
    assert "Task success rate" in svg
    assert "Fingertip-object distance" in svg
