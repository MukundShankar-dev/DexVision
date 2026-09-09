from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dexvision.evaluation import correction_summary
from dexvision.logging.corrective_demos import build_level4_6_plan
from dexvision.logging.level4_collection import PilotEpisode


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "level4_dataset.yaml"


def test_complete_synthetic_summary_waits_only_for_manual_replay(
    tmp_path: Path, monkeypatch
) -> None:
    plan = build_level4_6_plan(CONFIG_PATH)
    paths_by_slot: dict[tuple[str, int], Path] = {}
    episodes: list[PilotEpisode] = []
    sessions = []
    for item in plan:
        path = tmp_path / item.session_id / "episode_000001"
        path.mkdir(parents=True)
        paths_by_slot[(item.coverage_cell_id, item.repetition)] = path
        source_id = f"nominal_{item.sequence:03d}"
        if item.is_correction:
            assert item.source_failure_cell_id is not None
            source = next(
                episode
                for episode in episodes
                if episode.metadata["level4_6"]["coverage_cell_id"]
                == item.source_failure_cell_id
                and episode.metadata["level4_6"]["repetition"] == item.repetition
            )
            source_id = source.episode_id
            np.save(source.path / "actions.npy", np.zeros((2, 3)))
            np.save(path / "actions.npy", np.zeros((3, 3)))
            intervention_interval = [2, 3]
            outcome = "corrected_success"
            success = True
            stream = "corrective_intervention"
        else:
            intervention_interval = None
            outcome = (
                "aborted_unsafe"
                if item.failure_class
                in {"workspace_violation", "joint_limit_violation"}
                else "failed_unrecovered"
            )
            success = False
            stream = "ordinary_failure"
        metadata = {
            "episode_id": item.episode_id,
            "recording_session_id": item.session_id,
            "source": item.source,
            "data_stream": stream,
            "success": success,
            "initial_state_digest": f"sha256:{item.generation_seed:064x}",
            "goal_condition_id": item.coverage_cell_id,
            "source_episode_id": source_id,
            "trigger_source": "scripted",
            "source_policy_checkpoint": None,
            "source_policy_checkpoint_not_applicable_reason": (
                "trigger_not_policy_rollout"
            ),
            "failure_class": item.failure_class,
            "retryability": item.retryability,
            "intervention_interval": intervention_interval,
            "original_terminal_result": {"success": False},
            "final_outcome": outcome,
            "level4_6": {
                "version": "level4/scripted-corrections-v1",
                "coverage_cell_id": item.coverage_cell_id,
                "repetition": item.repetition,
                "generation_seed": item.generation_seed,
            },
        }
        episodes.append(
            PilotEpisode(
                path=path,
                metadata=metadata,
                review=None,
                size_bytes=1,
                duration_seconds=1.0,
            )
        )
        sessions.append(
            SimpleNamespace(recording_session_id=item.session_id, split=item.split)
        )

    monkeypatch.setattr(
        correction_summary,
        "discover_level4_6_episodes",
        lambda _root: tuple(episodes),
    )
    monkeypatch.setattr(
        correction_summary,
        "load_session_manifest",
        lambda _path: SimpleNamespace(sessions=tuple(sessions)),
    )
    monkeypatch.setattr(
        correction_summary,
        "select_level4_training_streams",
        lambda values, *, include_corrections: tuple(sorted(values, key=str)),
    )
    monkeypatch.setattr(
        correction_summary,
        "recompute_corrected_pick_place_outcome",
        lambda *_args, **_kwargs: {
            "pick_success": True,
            "final_place_success": True,
            "corrected_success": True,
        },
    )

    report = correction_summary.summarize_corrections(
        config_path=CONFIG_PATH,
        dataset_dir=tmp_path,
    )

    assert report["episode_count"] == 120
    assert report["complete_cell_count"] == 12
    assert report["episodes_by_stream"] == {
        "corrective_intervention": 30,
        "ordinary_failure": 90,
    }
    assert report["pre_intervention_prefix"]["passed"] is True
    assert report["independent_outcome_recomputation"]["passed"] is True
    assert report["unsafe_failures_abort_only"] is True
    assert report["deterministic_include_exclude"] is True
    assert report["identical_split_comparison_ready"] is True
    assert report["automated_requirements_passed"] is True
    assert report["status"] == "manual_verification_required"
    assert report["checkpoint_complete"] is False


def test_empty_dataset_reports_all_slots_missing(tmp_path: Path, monkeypatch) -> None:
    manifest = SimpleNamespace(sessions=())
    monkeypatch.setattr(correction_summary, "load_session_manifest", lambda _path: manifest)

    report = correction_summary.summarize_corrections(
        config_path=CONFIG_PATH,
        dataset_dir=tmp_path,
    )

    assert report["episode_count"] == 0
    assert report["complete_cell_count"] == 0
    assert len(report["issues"]) == 120
    assert report["automated_requirements_passed"] is False
