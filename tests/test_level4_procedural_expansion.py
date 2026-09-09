from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from dexvision.evaluation.dataset_coverage import _procedural_expansion_summary
from dexvision.logging.level4_collection import (
    PilotEpisode,
    PilotReview,
    build_level4_procedural_expansion_plan,
    initial_state_digest,
)


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


def test_v19_plan_adds_exactly_890_unique_assignments_with_fresh_test() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    plan = build_level4_procedural_expansion_plan(CONFIG_PATH)

    assert config["version"] == "level4/workcell-dataset-plan-v19"
    assert len(plan) == 890
    assert len({item.seed for item in plan}) == 890
    assert len({item.session_id for item in plan}) == 890
    assert all(
        item.session_id.startswith("level45b_v19_test_s_")
        and item.episode_id_prefix == "level45b_v19"
        and item.seed >= 1990000
        for item in plan
        if item.split == "test"
    )
    assert all(
        not item.session_id.startswith(
            (
                "level45b_test_d_",
                "level45b_v5_test_e_",
                "level45b_v6_test_f_",
                "level45b_v7_test_g_",
                "level45b_v8_test_h_",
            )
        )
        for item in plan
    )
    assert Counter(item.data_group for item in plan) == {
        "reach": 140,
        "pick_place": 438,
        "push": 172,
        "button": 140,
    }
    by_cell: dict[str, list[int]] = defaultdict(list)
    for item in plan:
        by_cell[item.coverage_cell_id].append(item.repetition)
        assert item.variation["seed"] == item.seed
        assert item.variation["case_class"] == (
            "boundary" if item.repetition == 16 else "nominal"
        )
    assert len(by_cell) == 62
    assert all(max(repetitions) == 16 for repetitions in by_cell.values())


def test_procedural_variation_and_initial_digest_are_deterministic() -> None:
    first = build_level4_procedural_expansion_plan(CONFIG_PATH)
    second = build_level4_procedural_expansion_plan(CONFIG_PATH)

    assert first == second
    digests = {
        initial_state_digest(
            initial_state={"seed": item.seed, "cell": item.coverage_cell_id},
            procedural_variation=item.variation,
        )
        for item in first
    }
    assert len(digests) == len(first)


def test_complete_synthetic_expansion_waits_only_for_manual_replay(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "level4_dataset.yaml"
    config_path.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    scaling_path = tmp_path / "outputs" / "level4" / "scaling_probe_v1"
    scaling_path.mkdir(parents=True)
    (scaling_path / "scaling_probe.json").write_text(
        json.dumps(
            {
                "status": "sufficient",
                "dataset_sufficient": True,
                "decision_uses_test_data": False,
            }
        ),
        encoding="utf-8",
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cells = {
        cell["id"]: cell
        for cell in config["coverage_cells"]
        if cell["data_group"] in {"reach", "pick_place", "push", "button"}
    }
    sessions: dict[str, str] = {}
    episodes: list[PilotEpisode] = []
    sequence = 0
    for cell_id, cell in cells.items():
        split = cell["split_owner"]
        anchor_count = cell["minimum_accepted_by_split"][split]
        skill = {
            "reach": "reach_object",
            "pick_place": "pick_place_sequence",
            "push": "push_object_to_target",
            "button": "press_button",
        }[cell["data_group"]]
        for anchor in range(anchor_count):
            sequence += 1
            session = f"anchor_{cell_id}_{anchor}"
            sessions[session] = split
            episode_id = f"anchor_{sequence:06d}"
            episodes.append(
                PilotEpisode(
                    path=tmp_path,
                    metadata={
                        "episode_id": episode_id,
                        "recording_session_id": session,
                        "skill_name": skill,
                        "goal_condition_id": cell_id,
                        "source": "scripted",
                        "random_seed": sequence,
                    },
                    review=_accepted_review(episode_id),
                    size_bytes=100,
                    duration_seconds=1.0,
                )
            )
    for item in build_level4_procedural_expansion_plan(config_path):
        episode_id = f"level45b_{item.sequence:06d}"
        episode_dir = tmp_path / "episodes" / episode_id
        episode_dir.mkdir(parents=True)
        actions = np.zeros((2, 27), dtype=np.float64)
        actions[:, 0] = item.sequence * 1e-5
        np.save(episode_dir / "applied_actions.npy", actions)
        np.save(episode_dir / "timestamps.npy", np.arange(2, dtype=float))
        sessions[item.session_id] = item.split
        digest = initial_state_digest(
            initial_state={"cell": item.coverage_cell_id, "seed": item.seed},
            procedural_variation=item.variation,
        )
        episodes.append(
            PilotEpisode(
                path=episode_dir,
                metadata={
                    "episode_id": episode_id,
                    "recording_session_id": item.session_id,
                    "skill_name": item.skill_name,
                    "goal_condition_id": item.coverage_cell_id,
                    "source": "scripted",
                    "random_seed": item.seed,
                    "initial_state_digest": digest,
                    "procedural_expansion": {
                        "repetition": item.repetition,
                        "variation": item.variation,
                    },
                },
                review=_accepted_review(episode_id),
                size_bytes=200,
                duration_seconds=1.0,
            )
        )

    report = _procedural_expansion_summary(
        config,
        config_path=config_path,
        dataset_root=tmp_path,
        episodes=episodes,
        sessions=sessions,
        cells=cells,
        manual_reviews=(),
    )

    assert report["accepted_nominal_episode_count"] == 992
    assert report["accepted_procedural_episode_count"] == 890
    assert report["coverage_matrix"]["complete_cell_count"] == 62
    assert report["duplicate_seed_count"] == 0
    assert report["duplicate_initial_state_digest_count"] == 0
    assert report["similarity_audit"]["passed"] is True
    assert report["automated_requirements_passed"] is True
    assert report["status"] == "manual_verification_required"
    assert report["checkpoint_complete"] is False
