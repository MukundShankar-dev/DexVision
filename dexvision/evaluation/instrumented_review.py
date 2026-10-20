"""Integrity-checked saved 5.7 held-out rollout playback; never grants acceptance."""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from dexvision.evaluation.perception_rollouts import SKILLS
from dexvision.logging.visual_stream import digest_file, write_json


def index_review(root):
    root = Path(root)
    schedule = []
    for skill in SKILLS:
        clips = sorted((root / skill / "5001").glob("*/nominal/review.avi"))
        if not clips:
            raise ValueError(f"Missing predeclared held-out review clip for {skill}")
        # Fixed first recorded nominal case, plus every other predeclared clip.
        for path in sorted((root / skill / "5001").glob("*/*/review.avi")):
            result = path.with_name("result.json")
            capture = cv2.VideoCapture(str(path))
            count = 0
            while True:
                ok, _ = capture.read()
                if not ok:
                    break
                count += 1
            capture.release()
            if count < 1:
                raise ValueError("Unreadable or empty held-out review clip")
            schedule.append(
                {
                    "skill": skill,
                    "clip": path.relative_to(root).as_posix(),
                    "sha256": digest_file(path),
                    "frames": count,
                    "result": result.relative_to(root).as_posix(),
                    "result_sha256": digest_file(result),
                }
            )
    write_json(root / "review_schedule.json", schedule)
    return schedule


def review_instrumented(directory):
    root = Path(directory)
    from dexvision.learning.train_skill import _verify_run

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
    if indexed != present:
        raise ValueError("Incomplete held-out artifact index")
    report = json.loads((root / "report.json").read_text())
    if report["split"] != "test":
        raise ValueError("Checkpoint acceptance requires the frozen held-out report")
    schedule = json.loads((root / "review_schedule.json").read_text())
    if {r["skill"] for r in schedule} != set(SKILLS):
        raise ValueError("Review must cover all five required skills")
    for row in schedule:
        for key, sha in (("clip", "sha256"), ("result", "result_sha256")):
            path = root / row[key]
            if (
                not path.resolve().is_relative_to(root.resolve())
                or digest_file(path) != row[sha]
            ):
                raise ValueError("Held-out review media or terminal result changed")
    try:
        for row in schedule:
            capture = cv2.VideoCapture(str(root / row["clip"]))
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    cv2.imshow("5.7 held-out review: n next clip, Esc cancels", frame)
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
        "manual_acceptance": "pending_owner_confirmation",
        "checkpoint_complete": False,
    }
