from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from dexvision.retargeting.optimization_retargeter import (
    OptimizationRetargeter,
    OptimizationWeights,
    _OptimizerResult,
)


ROOT = Path(__file__).resolve().parents[1]
TELEOP_CONFIG_PATH = ROOT / "configs" / "level1_teleop.yaml"
_LONG_FINGERS = {
    "index": (5, 6, 7, 8, 1.0),
    "middle": (9, 10, 11, 12, 1.0),
    "ring": (13, 14, 15, 16, -1.0),
    "pinky": (17, 18, 19, 20, -1.0),
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
    for base, pip, dip, tip, direction in _LONG_FINGERS.values():
        landmarks[pip] = landmarks[base] + [0.0, 0.20, 0.0]
        landmarks[dip] = landmarks[pip] + [0.16 * direction, 0.0, 0.0]
        landmarks[tip] = landmarks[dip] + [0.0, -0.16, 0.0]
    return landmarks


def test_optimizer_returns_distinct_valid_open_and_fist_targets() -> None:
    retargeter = OptimizationRetargeter.from_yaml(TELEOP_CONFIG_PATH)

    open_targets = retargeter.map(_open_hand())
    fist_targets = retargeter.map(_fist())

    assert set(open_targets) == set(fist_targets)
    assert not retargeter.last_used_fallback
    assert retargeter.last_stats.success
    assert retargeter.last_stats.backend in {"scipy", "projected-gradient"}
    for name in ("rh_A_FFJ0", "rh_A_MFJ0", "rh_A_RFJ0", "rh_A_LFJ0"):
        assert fist_targets[name] > open_targets[name]


def test_every_result_respects_configured_limits() -> None:
    retargeter = OptimizationRetargeter.from_yaml(TELEOP_CONFIG_PATH)
    limits = {target.name: target.limit for target in retargeter.config.static_targets}
    for finger in retargeter.config.fingers:
        limits.update({target.name: target.limit for target in finger.targets})

    for landmarks in (_open_hand(), _fist()):
        targets = retargeter.map(landmarks)
        assert all(
            limits[name].minimum <= value <= limits[name].maximum
            for name, value in targets.items()
        )


def test_smoothness_penalty_reduces_frame_to_frame_jump() -> None:
    unsmoothed = OptimizationRetargeter.from_yaml(TELEOP_CONFIG_PATH)
    unsmoothed.weights = OptimizationWeights(smoothness=0.0)
    smoothed = OptimizationRetargeter.from_yaml(TELEOP_CONFIG_PATH)
    smoothed.weights = OptimizationWeights(smoothness=10.0)

    open_unsmoothed = unsmoothed.map(_open_hand())
    open_smoothed = smoothed.map(_open_hand())
    fist_unsmoothed = unsmoothed.map(_fist())
    fist_smoothed = smoothed.map(_fist())

    name = "rh_A_FFJ0"
    unsmoothed_jump = abs(fist_unsmoothed[name] - open_unsmoothed[name])
    smoothed_jump = abs(fist_smoothed[name] - open_smoothed[name])
    assert smoothed_jump < unsmoothed_jump


def test_failed_optimization_uses_last_valid_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retargeter = OptimizationRetargeter.from_yaml(TELEOP_CONFIG_PATH)
    previous = retargeter.map(_open_hand())

    def fail_optimizer(*_args: object, **_kwargs: object) -> _OptimizerResult:
        return _OptimizerResult(
            controls=np.full(5, np.nan),
            objective_value=np.nan,
            success=False,
            backend="synthetic-failure",
        )

    monkeypatch.setattr(retargeter, "_run_optimizer", fail_optimizer)

    assert retargeter.map(_fist()) == previous
    assert retargeter.last_used_fallback
    assert not retargeter.last_stats.success


def test_first_invalid_frame_uses_safe_open_pose() -> None:
    retargeter = OptimizationRetargeter.from_yaml(TELEOP_CONFIG_PATH)

    targets = retargeter.map(np.zeros((21, 3), dtype=np.float64))

    assert targets == retargeter.fallback_targets()
    assert targets["rh_A_FFJ0"] == pytest.approx(0.05)
    assert retargeter.last_used_fallback


def test_solve_time_is_recorded_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    retargeter = OptimizationRetargeter.from_yaml(TELEOP_CONFIG_PATH)

    with caplog.at_level(logging.INFO):
        retargeter.map(_open_hand())

    assert retargeter.last_solve_time_seconds >= 0.0
    assert "Optimization retargeting solve time" in caplog.text


def test_mapping_accepts_optimization_settings() -> None:
    retargeter = OptimizationRetargeter.from_mapping(
        {
            "retargeting": {
                "type": "optimization",
                "optimization": {
                    "fingertip_weight": 2.0,
                    "joint_limit_weight": 50.0,
                    "smoothness_weight": 0.25,
                    "max_iterations": 25,
                },
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

    assert retargeter.weights == OptimizationWeights(2.0, 50.0, 0.25)
    assert retargeter.max_iterations == 25
    assert 0.0 <= retargeter.map(_fist())["index_motor"] <= 1.0
