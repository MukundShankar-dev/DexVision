from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from dexvision.evaluation import level4_scaling_probe
from dexvision.logging.level4_collection import PilotEpisode, PilotReview


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "level4_dataset.yaml"


def _accepted_review(episode_id: str) -> PilotReview:
    return PilotReview(
        episode_id=episode_id,
        schema_validation=True,
        timestamp_alignment=True,
        headless_replay=True,
        terminal_metric_recomputation=True,
        recomputed_success=True,
        operator_label_agreement=True,
        quality_thresholds=True,
        coverage_assignment=True,
        split_session_leakage_audit=True,
        expert_accepted=True,
    )


def test_nested_probe_reuses_one_recipe_and_validation_matrix(monkeypatch) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    cells = [
        cell
        for cell in config["coverage_cells"]
        if cell["data_group"] in {"reach", "pick_place", "push", "button"}
        and cell["split_owner"] in {"train", "validation"}
    ]
    object_positions = {
        object_id: [index * 0.01, 0.0, 0.02]
        for index, object_id in enumerate(
            [
                "block_small",
                "block_large",
                "cylinder_short",
                "cylinder_tall",
                "puck_light",
                "puck_heavy",
                "start_button",
                "return_bin_left",
                "return_bin_right",
                "inspection_pad",
                "setup_slot_a",
                "setup_slot_b",
            ]
        )
    }
    episodes = []
    sessions = []
    for cell in cells:
        split = cell["split_owner"]
        count = 16 if split == "train" else 8
        skill = {
            "reach": "reach_object",
            "pick_place": "pick_place_sequence",
            "push": "push_object_to_target",
            "button": "press_button",
        }[cell["data_group"]]
        for repetition in range(1, count + 1):
            episode_id = f"{cell['id']}_{repetition:02d}"
            session_id = f"session_{episode_id}"
            sessions.append(
                SimpleNamespace(recording_session_id=session_id, split=split)
            )
            variation = {
                "controller_position_offset_m": [
                    repetition * 1e-6,
                    -repetition * 1e-6,
                    repetition * 0.5e-6,
                ],
                "object_scale_multiplier": 1.0,
                "object_mass_multiplier": 1.0,
                "object_friction_multiplier": 1.0,
            }
            episodes.append(
                PilotEpisode(
                    path=Path("unused"),
                    metadata={
                        "episode_id": episode_id,
                        "recording_session_id": session_id,
                        "skill_name": skill,
                        "goal_condition_id": cell["id"],
                        "source": "scripted",
                        "random_seed": repetition,
                        "reset_state": {"entity_positions_m": object_positions},
                        "procedural_expansion": {
                            "repetition": repetition,
                            "variation": variation,
                        },
                    },
                    review=_accepted_review(episode_id),
                    size_bytes=1,
                    duration_seconds=1.0,
                )
            )

    monkeypatch.setattr(
        level4_scaling_probe, "discover_pilot_episodes", lambda _path: tuple(episodes)
    )
    monkeypatch.setattr(
        level4_scaling_probe,
        "load_session_manifest",
        lambda _path: SimpleNamespace(sessions=tuple(sessions)),
    )
    validation_matrices = []

    def fake_evaluate(_model, **kwargs):
        validation = kwargs["validation_episodes"]
        validation_matrices.append([item.episode_id for item in validation])
        counts = len(validation)
        return {
            "validation_rollout_count": counts,
            "validation_success_count": counts,
            "aggregate_validation_success": 1.0,
            "worst_cell_validation_success": 1.0,
            "validation_success_by_cell": {},
            "safety_violation_count": 0,
            "invalid_action_count": 0,
        }

    monkeypatch.setattr(
        level4_scaling_probe, "_evaluate_validation_rollouts", fake_evaluate
    )
    report = level4_scaling_probe.run_level4_scaling_probe(
        config_path=CONFIG_PATH,
        dataset_dir=Path("unused"),
    )

    assert [item["episodes_per_training_cell"] for item in report["tranches"]] == [
        4,
        8,
        16,
    ]
    assert len({tuple(matrix) for matrix in validation_matrices}) == 1
    assert report["decision_uses_test_data"] is False
    assert report["test_episode_count_inspected"] == 0
    assert report["improvement_8_to_16"] == 0.0
    assert report["dataset_sufficient"] is True


def test_scaling_report_is_written_atomically(tmp_path: Path) -> None:
    output = level4_scaling_probe.save_scaling_probe_report(
        {"version": "test", "dataset_sufficient": True},
        tmp_path / "scaling_probe.json",
    )

    assert output.exists()
    assert "dataset_sufficient" in output.read_text(encoding="utf-8")
