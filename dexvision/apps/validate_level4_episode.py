"""Validate one saved Level 4 episode without camera, GUI, or MuJoCo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dexvision.logging.dataset_schema import (
    DemoSchemaError,
    LEVEL4_EPISODE_SCHEMA_VERSION,
    validate_demo,
)
from dexvision.logging.demo_logger import (
    DemoLoggerError,
    action_schema_from_metadata,
    load_logged_demo,
    observation_schema_from_metadata,
)
from dexvision.logging.phase_labels import (
    PhaseLabelError,
    derive_pick_place_segments,
    phase_disagreement_report,
    validate_phase_intervals,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a session-aware DexVision Level 4 episode directory."
    )
    parser.add_argument("--episode", type=Path, required=True, help="Saved episode directory.")
    return parser


def validate_episode_directory(path: str | Path) -> dict[str, object]:
    """Load, validate, and summarize one Level 4 episode directory."""

    episode = load_logged_demo(path)
    if episode.metadata.get("episode_schema_version") != LEVEL4_EPISODE_SCHEMA_VERSION:
        raise DemoSchemaError(
            f"episode does not declare '{LEVEL4_EPISODE_SCHEMA_VERSION}'."
        )
    action_schema = action_schema_from_metadata(episode.metadata)
    observation_schema = observation_schema_from_metadata(episode.metadata)
    validate_demo(
        episode,
        action_schema=action_schema,
        observation_schema=observation_schema,
    )
    intervals = validate_phase_intervals(
        episode.metadata["phase_intervals"],
        frame_count=int(episode.timestamps.shape[0]),
        phases=episode.online_phases.tolist(),
    )
    audited = (
        episode.audited_phases.tolist()
        if episode.audited_phases is not None
        else [None] * int(episode.timestamps.shape[0])
    )
    report: dict[str, object] = {
        "episode_id": episode.metadata["episode_id"],
        "recording_session_id": episode.metadata["recording_session_id"],
        "frames": int(episode.timestamps.shape[0]),
        "phase_intervals": len(intervals),
        "phase_disagreement": phase_disagreement_report(
            episode.online_phases.tolist(),
            audited,
        ),
        "rgb_enabled": episode.rgb_frames is not None,
    }
    if episode.metadata["skill_name"] == "pick_place_sequence":
        report["derived_segments"] = [
            segment.to_dict()
            for segment in derive_pick_place_segments(
                intervals,
                frame_count=int(episode.timestamps.shape[0]),
            )
        ]
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate_episode_directory(args.episode)
    except (DemoLoggerError, DemoSchemaError, PhaseLabelError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    disagreement = report["phase_disagreement"]
    print("DexVision Level 4 episode validation: PASS")
    print(f"Episode: {report['episode_id']}")
    print(f"Session: {report['recording_session_id']}")
    print(f"Frames: {report['frames']}")
    print(f"Phase intervals: {report['phase_intervals']}")
    print(
        "Audited phase disagreement: "
        f"{disagreement['disagreement_count']}/{disagreement['audited_frame_count']}"
    )
    print(f"RGB enabled: {report['rgb_enabled']}")
    if "derived_segments" in report:
        print("Derived segments: reach_object, pick_object, place_held_object")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
