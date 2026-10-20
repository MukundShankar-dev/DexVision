"""Read-only review HUD derived from saved actions; original clips stay intact.

The frozen recorder's proxy-only policy-call counter was not forwarded to its
HUD. Replace that diagnostic header in memory with recorded applied-action
counts/XYZ and measured phase, never with a fabricated policy-call count.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from dexvision.evaluation.perception_rollouts import SKILLS
from dexvision.learning.train_skill import _verify_run
from dexvision.learning.training_manifest import digest_file


def annotate_review_frame(frame, index, *, skill, result, trace, observations):
    """Return a display copy plus an audit-friendly caption description."""
    actions = trace.get("applied_actions", np.empty((0, 27)))
    total = len(actions)
    applied_frames = (total + 2) // 3
    terminal = index >= applied_frames
    count = total if terminal else min(total, 1 + index * 3)
    action = actions[count - 1, :3] if count else None
    phase = "unavailable"
    orientation = None
    times = observations.get("timestamps", np.empty(0))
    if len(times):
        stamp = (
            trace["state_timestamps"][-1]
            if terminal
            else trace["state_timestamps"][count - 1]
        )
        nearest = int(np.argmin(abs(times - stamp)))
        phase = str(observations["phases"][nearest])
        if "consumed_pose" in observations:
            orientation = observations["consumed_pose"][nearest, 3:7].tolist()
    description = dict(
        applied=count,
        total=total,
        terminal_frame=terminal,
        phase=phase,
        xyz=action.tolist() if action is not None else None,
        orientation_wxyz=orientation,
    )
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, 44), (0, 0, 0), -1)
    xyz = np.array2string(action, precision=3) if action is not None else "none"
    line = f"{skill} {phase} | applied {count}/{total} | command_xyz={xyz}"
    quaternion = (
        np.array2string(np.asarray(orientation), precision=3)
        if orientation is not None
        else "unavailable"
    )
    second = (
        f"terminal: {result['terminal_result']} | observed q_wxyz={quaternion}"
        if terminal
        else f"{result['condition']} | observed q_wxyz={quaternion}"
    )
    for text, y in ((line, 18), (second, 36)):
        cv2.putText(
            panel,
            text[:105],
            (5, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
        )
    return panel, description


def review_instrumented_results(directory):
    root = Path(directory)
    _verify_run(root)
    indexed = {
        line.split("  ", 1)[1]
        for line in (root / "SHA256SUMS").read_text().splitlines()
    }
    present = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.name != "SHA256SUMS"
    }
    if (
        indexed != present
        or json.loads((root / "report.json").read_text())["split"] != "test"
    ):
        raise ValueError("Review requires complete immutable held-out evidence")
    schedule = json.loads((root / "review_schedule.json").read_text())
    if {row["skill"] for row in schedule} != set(SKILLS):
        raise ValueError("Review must cover every required skill")
    try:
        for row in schedule:
            clip = root / row["clip"]
            result_path = root / row["result"]
            if (
                not clip.resolve().is_relative_to(root.resolve())
                or not result_path.resolve().is_relative_to(root.resolve())
                or digest_file(clip) != row["sha256"]
                or digest_file(result_path) != row["result_sha256"]
            ):
                raise ValueError("Saved review clip or result changed")
            result = json.loads(result_path.read_text())
            with np.load(
                result_path.parent / "trajectory.npz", allow_pickle=False
            ) as data:
                trace = {name: data[name].copy() for name in data.files}
            observations = {}
            if (result_path.parent / "observations.npz").exists():
                with np.load(
                    result_path.parent / "observations.npz", allow_pickle=False
                ) as data:
                    observations = {name: data[name].copy() for name in data.files}
            print(
                f"Review {row['skill']}/{result['condition']}: {result['terminal_result']} / {result['terminal_reason']}",
                flush=True,
            )
            capture = cv2.VideoCapture(str(clip))
            index = 0
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    panel, _ = annotate_review_frame(
                        frame,
                        index,
                        skill=row["skill"],
                        result=result,
                        trace=trace,
                        observations=observations,
                    )
                    cv2.imshow("5.7 saved results: n next, Esc cancels", panel)
                    index += 1
                    key = cv2.waitKey(100) & 0xFF
                    if key == 27:
                        raise ValueError(
                            "Owner review cancelled; acceptance remains pending"
                        )
                    if key == ord("n"):
                        break
            finally:
                capture.release()
    except cv2.error as exc:
        raise ValueError(f"OpenCV review display unavailable: {exc}") from exc
    finally:
        cv2.destroyAllWindows()
    return {
        "clips": len(schedule),
        "checkpoint_complete": False,
        "manual_acceptance": "pending_owner_confirmation",
        "display": "Recorded action-count/XYZ HUD; original pixels/poses/timestamps and files preserved",
    }
