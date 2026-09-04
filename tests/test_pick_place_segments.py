from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dexvision.apps.validate_level4_episode import validate_episode_directory
from dexvision.logging.demo_logger import (
    action_schema_from_metadata,
    load_logged_demo,
)
from dexvision.logging.phase_labels import (
    derive_pick_place_segments,
    validate_phase_intervals,
)
from test_level4_place_expert import record_pick_place


def test_complete_recording_yields_compatible_reach_pick_and_place_segments(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mujoco")
    episode_dir = record_pick_place(
        tmp_path,
        cell="pp_block_small_inspection_pad",
        seed=4,
        name="segment_contract",
    )
    episode = load_logged_demo(episode_dir)
    intervals = validate_phase_intervals(
        episode.metadata["phase_intervals"],
        frame_count=episode.actions.shape[0],
        phases=episode.online_phases.tolist(),
    )
    segments = derive_pick_place_segments(
        intervals, frame_count=episode.actions.shape[0]
    )
    report = validate_episode_directory(episode_dir)
    action_schema = action_schema_from_metadata(episode.metadata)

    assert [segment.skill_name for segment in segments] == [
        "reach_object",
        "pick_object",
        "place_held_object",
    ]
    assert report["derived_segments"] == [segment.to_dict() for segment in segments]
    assert segments[0].start_frame == 0
    assert segments[0].end_frame == segments[1].start_frame
    assert segments[1].end_frame == segments[2].start_frame
    assert segments[2].end_frame <= episode.actions.shape[0]
    assert episode.actions.shape[1] == action_schema.action_dim
    for segment in segments:
        action_slice = episode.actions[segment.start_frame : segment.end_frame]
        assert action_slice.ndim == 2
        assert action_slice.shape[0] > 0
        assert action_slice.shape[1] == action_schema.action_dim
        assert np.all(np.isfinite(action_slice))
