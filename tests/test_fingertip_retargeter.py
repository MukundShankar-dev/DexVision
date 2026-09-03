from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dexvision.retargeting.fingertip_ik_retargeter import (
    FINGERTIP_NAMES,
    FingertipIKRetargeter,
    FingertipRetargeterError,
    compute_normalized_fingertip_targets,
)


ROOT = Path(__file__).resolve().parents[1]
TELEOP_CONFIG_PATH = ROOT / "configs" / "level1_teleop.yaml"

_LONG_FINGER_INDICES = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
_CURL_DIRECTIONS = {
    "index": 1.0,
    "middle": 1.0,
    "ring": -1.0,
    "pinky": -1.0,
}


def _open_hand() -> np.ndarray:
    landmarks = np.zeros((21, 3), dtype=np.float64)
    landmarks[1:5] = [
        [-0.25, 0.20, 0.0],
        [-0.38, 0.34, 0.0],
        [-0.51, 0.48, 0.0],
        [-0.64, 0.62, 0.0],
    ]
    landmarks[5:9] = [
        [-0.18, 0.36, 0.0],
        [-0.18, 0.58, 0.0],
        [-0.18, 0.78, 0.0],
        [-0.18, 0.98, 0.0],
    ]
    landmarks[9:13] = [
        [0.00, 0.40, 0.0],
        [0.00, 0.66, 0.0],
        [0.00, 0.90, 0.0],
        [0.00, 1.12, 0.0],
    ]
    landmarks[13:17] = [
        [0.18, 0.36, 0.0],
        [0.18, 0.58, 0.0],
        [0.18, 0.78, 0.0],
        [0.18, 0.96, 0.0],
    ]
    landmarks[17:21] = [
        [0.34, 0.30, 0.0],
        [0.34, 0.50, 0.0],
        [0.34, 0.66, 0.0],
        [0.34, 0.82, 0.0],
    ]
    return landmarks


def _fist() -> np.ndarray:
    landmarks = _open_hand()
    landmarks[1:5] = [
        [-0.25, 0.20, 0.0],
        [-0.37, 0.32, 0.0],
        [-0.25, 0.32, 0.0],
        [-0.25, 0.20, 0.0],
    ]
    for finger, (base, pip, dip, tip) in _LONG_FINGER_INDICES.items():
        direction = _CURL_DIRECTIONS[finger]
        landmarks[pip] = landmarks[base] + [0.0, 0.20, 0.0]
        landmarks[dip] = landmarks[pip] + [0.16 * direction, 0.0, 0.0]
        landmarks[tip] = landmarks[dip] + [0.0, -0.16, 0.0]
    return landmarks


def test_fingertip_targets_are_palm_normalized_and_scale_invariant() -> None:
    landmarks = _open_hand()

    targets = compute_normalized_fingertip_targets(landmarks)
    transformed = compute_normalized_fingertip_targets(
        landmarks * 2.75 + np.asarray([3.0, -4.0, 1.5])
    )

    assert FINGERTIP_NAMES == ("thumb", "index", "middle", "ring", "pinky")
    assert targets.shape == (5, 3)
    assert np.all(np.isfinite(targets))
    assert transformed == pytest.approx(targets)


def test_open_and_fist_landmarks_produce_distinct_joint_targets() -> None:
    retargeter = FingertipIKRetargeter.from_yaml(TELEOP_CONFIG_PATH)

    open_targets = retargeter.map(_open_hand())
    fist_targets = retargeter.map(_fist())

    assert not retargeter.last_used_fallback
    assert set(open_targets) == set(fist_targets)
    for target_name in ("rh_A_FFJ0", "rh_A_MFJ0", "rh_A_RFJ0", "rh_A_LFJ0"):
        assert open_targets[target_name] < fist_targets[target_name]
    assert open_targets["rh_A_THJ1"] < fist_targets["rh_A_THJ1"]


def test_outputs_obey_every_configured_joint_limit() -> None:
    retargeter = FingertipIKRetargeter.from_yaml(TELEOP_CONFIG_PATH)

    for landmarks in (_open_hand(), _fist()):
        targets = retargeter.map(landmarks)
        limits = {
            target.name: target.limit
            for target in retargeter.config.static_targets
        }
        for finger in retargeter.config.fingers:
            limits.update({target.name: target.limit for target in finger.targets})

        assert all(
            limits[name].minimum <= value <= limits[name].maximum
            for name, value in targets.items()
        )


def test_first_solve_failure_uses_safe_open_fallback() -> None:
    retargeter = FingertipIKRetargeter.from_yaml(TELEOP_CONFIG_PATH)

    fallback = retargeter.map(np.zeros((21, 3), dtype=np.float64))

    assert retargeter.last_used_fallback
    assert fallback == retargeter.fallback_targets()
    assert fallback["rh_A_FFJ0"] == pytest.approx(0.05)
    assert fallback["rh_A_MFJ0"] == pytest.approx(0.05)


def test_solve_failure_after_valid_frame_uses_last_valid_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retargeter = FingertipIKRetargeter.from_yaml(TELEOP_CONFIG_PATH)
    previous = retargeter.map(_open_hand())

    def fail_solve(_landmarks: np.ndarray) -> dict[str, float]:
        raise RuntimeError("synthetic solve failure")

    monkeypatch.setattr(retargeter, "solve", fail_solve)

    assert retargeter.map(_fist()) == previous
    assert retargeter.last_used_fallback


def test_tracking_result_input_and_low_confidence_fallback() -> None:
    retargeter = FingertipIKRetargeter.from_mapping(
        {
            "retargeting": {
                "type": "fingertip_ik",
                "min_confidence": 0.5,
                "fingers": {
                    "index": {
                        "targets": {
                            "index_motor": {
                                "open": -2.0,
                                "closed": 2.0,
                                "min": 0.0,
                                "max": 1.0,
                            }
                        }
                    }
                },
            }
        }
    )
    tracked = SimpleNamespace(
        detected=True,
        image_landmarks=_fist(),
        world_landmarks=None,
        confidence=1.0,
    )
    low_confidence = SimpleNamespace(
        detected=True,
        image_landmarks=_open_hand(),
        world_landmarks=None,
        confidence=0.1,
    )

    assert retargeter.map(tracked)["index_motor"] == pytest.approx(1.0)
    assert retargeter.map(low_confidence)["index_motor"] == pytest.approx(0.0)
    assert retargeter.last_used_fallback


def test_public_target_computation_rejects_invalid_landmarks() -> None:
    with pytest.raises(FingertipRetargeterError, match="shape"):
        compute_normalized_fingertip_targets(np.zeros((20, 3)))
    with pytest.raises(FingertipRetargeterError, match="finite"):
        compute_normalized_fingertip_targets(np.full((21, 3), np.nan))
