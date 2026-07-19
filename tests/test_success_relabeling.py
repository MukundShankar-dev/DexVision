from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dexvision.apps import relabel_demos
from dexvision.logging.relabel_success import (
    DEFAULT_REPORT_NAME,
    RELABEL_REPORT_VERSION,
    SuccessRelabelError,
    relabel_reach_touch_dataset,
    relabel_reach_touch_episode,
)
from dexvision.sim.tasks import (
    is_button_press_success,
    is_push_cube_success,
    push_cube_distance,
)


def _write_episode(
    dataset: Path,
    name: str,
    *,
    qualifying_frames: tuple[bool, ...],
    operator_success: bool | None,
    task_id: str = "reach_touch_target",
) -> Path:
    episode = dataset / name
    episode.mkdir(parents=True)
    metadata = {
        "episode_id": f"episode-{name}",
        "task_id": task_id,
        "success": operator_success,
        "task_config": {
            "success_metric_inputs": [
                "target_position",
                "touch_position",
                "distance_to_target",
                "palm_contact",
            ]
        },
    }
    (episode / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    target = np.asarray([0.14, 0.0, 0.49], dtype=np.float64)
    rows = []
    for qualifies in qualifying_frames:
        touch = target + (
            np.asarray([0.01, 0.0, 0.0])
            if qualifies
            else np.asarray([0.05, 0.0, 0.0])
        )
        distance = float(np.linalg.norm(touch - target))
        rows.append(
            np.concatenate(
                (
                    target,
                    touch,
                    np.asarray(
                        [distance, float(qualifies)],
                        dtype=np.float64,
                    ),
                )
            )
        )
    np.save(episode / "task_states.npy", np.stack(rows))
    return episode


def test_recomputes_success_from_saved_positions_contact_and_dwell(
    tmp_path: Path,
) -> None:
    episode = _write_episode(
        tmp_path,
        "success",
        qualifying_frames=(False, True, True, True, True, True),
        operator_success=True,
    )

    result = relabel_reach_touch_episode(episode)

    assert result.recomputed_success is True
    assert result.operator_success is True
    assert result.labels_agree is True
    assert result.first_success_frame == 5
    assert result.max_consecutive_contact_frames == 5


def test_dwell_resets_and_preserves_operator_disagreement(tmp_path: Path) -> None:
    episode = _write_episode(
        tmp_path,
        "failure",
        qualifying_frames=(True, True, True, False, True, True, True, True),
        operator_success=True,
    )

    result = relabel_reach_touch_episode(episode)

    assert result.recomputed_success is False
    assert result.operator_success is True
    assert result.labels_agree is False
    assert result.max_consecutive_contact_frames == 4


def test_dataset_report_is_saved_without_rewriting_raw_episode(
    tmp_path: Path,
) -> None:
    episode = _write_episode(
        tmp_path,
        "episode-001",
        qualifying_frames=(True, True, True, True, True),
        operator_success=False,
    )
    metadata_before = (episode / "metadata.json").read_bytes()
    task_states_before = (episode / "task_states.npy").read_bytes()

    exit_code = relabel_demos.main(["--dataset", str(tmp_path)])

    report_path = tmp_path / DEFAULT_REPORT_NAME
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert saved["version"] == RELABEL_REPORT_VERSION
    assert saved["raw_episodes_modified"] is False
    assert saved["episodes"][0]["operator_success"] is False
    assert saved["episodes"][0]["recomputed_success"] is True
    assert saved["label_disagreement_count"] == 1
    assert (episode / "metadata.json").read_bytes() == metadata_before
    assert (episode / "task_states.npy").read_bytes() == task_states_before


def test_dataset_summary_preserves_null_operator_label(tmp_path: Path) -> None:
    _write_episode(
        tmp_path,
        "unlabeled",
        qualifying_frames=(False, False),
        operator_success=None,
    )

    report = relabel_reach_touch_dataset(tmp_path)

    assert report.operator_success_count == 0
    assert report.label_disagreement_count == 0
    assert report.episodes[0].operator_success is None
    assert report.episodes[0].labels_agree is None


def test_missing_task_states_produces_clear_error(tmp_path: Path) -> None:
    episode = _write_episode(
        tmp_path,
        "missing-inputs",
        qualifying_frames=(False,),
        operator_success=False,
    )
    (episode / "task_states.npy").unlink()

    with pytest.raises(SuccessRelabelError, match="Missing success metric inputs"):
        relabel_reach_touch_episode(episode)


def test_missing_metric_columns_produces_clear_error(tmp_path: Path) -> None:
    episode = _write_episode(
        tmp_path,
        "narrow-inputs",
        qualifying_frames=(False,),
        operator_success=False,
    )
    np.save(episode / "task_states.npy", np.zeros((2, 7), dtype=np.float64))

    with pytest.raises(SuccessRelabelError, match="expected at least 8"):
        relabel_reach_touch_episode(episode)


def test_inconsistent_saved_distance_produces_clear_error(tmp_path: Path) -> None:
    episode = _write_episode(
        tmp_path,
        "bad-distance",
        qualifying_frames=(True,),
        operator_success=True,
    )
    task_states = np.load(episode / "task_states.npy")
    task_states[0, 6] = 0.02
    np.save(episode / "task_states.npy", task_states)

    with pytest.raises(SuccessRelabelError, match="inconsistent distance_to_target"):
        relabel_reach_touch_episode(episode)


def test_only_reach_touch_task_is_supported(tmp_path: Path) -> None:
    episode = _write_episode(
        tmp_path,
        "other-task",
        qualifying_frames=(True,),
        operator_success=True,
        task_id="button_press",
    )

    with pytest.raises(SuccessRelabelError, match="only 'reach_touch_target'"):
        relabel_reach_touch_episode(episode)


def test_button_success_recomputes_from_saved_terminal_metrics() -> None:
    saved_success_metrics = {
        "press_depth_m": 0.013,
        "target_press_depth_m": 0.012,
        "button_pressed": True,
        "target_pressed_state": True,
        "dwell_steps": 3,
        "required_dwell_steps": 3,
    }

    assert is_button_press_success(**saved_success_metrics)
    assert not is_button_press_success(
        **{**saved_success_metrics, "press_depth_m": 0.011}
    )
    assert not is_button_press_success(
        **{**saved_success_metrics, "dwell_steps": 2}
    )
    assert not is_button_press_success(
        **{**saved_success_metrics, "button_pressed": False}
    )


def test_button_unpressed_state_target_recomputes_without_live_simulation() -> None:
    assert is_button_press_success(
        press_depth_m=0.001,
        target_press_depth_m=0.012,
        button_pressed=False,
        target_pressed_state=False,
        dwell_steps=3,
        required_dwell_steps=3,
    )


def test_push_cube_success_recomputes_from_saved_terminal_metrics() -> None:
    object_position = np.asarray([0.08, 0.01, -0.015])
    target_position = np.asarray([0.09, 0.0, -0.015])
    saved_distance = push_cube_distance(object_position, target_position)

    assert is_push_cube_success(
        distance_m=saved_distance,
        dwell_steps=5,
        distance_threshold_m=0.035,
        required_dwell_steps=5,
    )
    assert not is_push_cube_success(
        distance_m=saved_distance,
        dwell_steps=4,
        distance_threshold_m=0.035,
        required_dwell_steps=5,
    )
    assert not is_push_cube_success(
        distance_m=0.04,
        dwell_steps=5,
        distance_threshold_m=0.035,
        required_dwell_steps=5,
    )


def test_push_cube_distance_is_planar_and_rejects_non_finite_state() -> None:
    assert push_cube_distance(
        object_position=(0.0, 0.0, -0.015),
        target_position=(0.03, 0.04, 0.50),
    ) == pytest.approx(0.05)
    with pytest.raises(ValueError, match="three finite"):
        push_cube_distance(
            object_position=(0.0, float("nan"), 0.0),
            target_position=(0.0, 0.0, 0.0),
        )
