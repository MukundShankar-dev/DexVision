from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from dexvision.features.hand_features import HandFeatures
from dexvision.retargeting.curl_retargeter import (
    CurlRetargeter,
    CurlRetargeterError,
    load_curl_retargeter_config,
)


ROOT = Path(__file__).resolve().parents[1]
TELEOP_CONFIG_PATH = ROOT / "configs" / "level1_teleop.yaml"

EXPECTED_SHADOW_TARGETS = {
    "rh_A_WRJ2",
    "rh_A_WRJ1",
    "rh_A_THJ5",
    "rh_A_THJ4",
    "rh_A_THJ3",
    "rh_A_THJ2",
    "rh_A_THJ1",
    "rh_A_FFJ4",
    "rh_A_FFJ3",
    "rh_A_FFJ0",
    "rh_A_MFJ4",
    "rh_A_MFJ3",
    "rh_A_MFJ0",
    "rh_A_RFJ4",
    "rh_A_RFJ3",
    "rh_A_RFJ0",
    "rh_A_LFJ5",
    "rh_A_LFJ4",
    "rh_A_LFJ3",
    "rh_A_LFJ0",
}


def _features(
    *,
    thumb: float = 0.0,
    index: float = 0.0,
    middle: float = 0.0,
    ring: float = 0.0,
    pinky: float = 0.0,
    confidence: float = 1.0,
) -> HandFeatures:
    return HandFeatures(
        thumb_curl=thumb,
        index_curl=index,
        middle_curl=middle,
        ring_curl=ring,
        pinky_curl=pinky,
        pinch_thumb_index=0.0,
        palm_roll_proxy=0.0,
        palm_pitch_proxy=0.0,
        confidence=confidence,
    )


def test_default_level1_config_loads_shadow_hand_targets() -> None:
    retargeter = CurlRetargeter.from_yaml(TELEOP_CONFIG_PATH)
    targets = retargeter.map(_features())

    assert set(targets) == EXPECTED_SHADOW_TARGETS
    assert targets["rh_A_WRJ2"] == pytest.approx(0.0)
    assert targets["rh_A_WRJ1"] == pytest.approx(0.0)


def test_open_hand_maps_to_open_robot_hand() -> None:
    retargeter = CurlRetargeter.from_yaml(TELEOP_CONFIG_PATH)

    targets = retargeter.map(_features())

    assert targets["rh_A_THJ5"] == pytest.approx(0.2)
    assert targets["rh_A_THJ4"] == pytest.approx(0.1)
    assert targets["rh_A_FFJ3"] == pytest.approx(-0.15)
    assert targets["rh_A_FFJ0"] == pytest.approx(0.05)
    assert targets["rh_A_MFJ3"] == pytest.approx(-0.15)
    assert targets["rh_A_RFJ0"] == pytest.approx(0.05)
    assert targets["rh_A_LFJ5"] == pytest.approx(0.1)
    assert _all_targets_obey_config_limits(retargeter, targets)


def test_fist_maps_to_closed_robot_hand() -> None:
    retargeter = CurlRetargeter.from_yaml(TELEOP_CONFIG_PATH)

    targets = retargeter.map(
        _features(thumb=1.0, index=1.0, middle=1.0, ring=1.0, pinky=1.0),
        robot_state={"ignored": True},
    )

    assert targets["rh_A_THJ4"] == pytest.approx(0.9)
    assert targets["rh_A_THJ1"] == pytest.approx(1.05)
    assert targets["rh_A_FFJ3"] == pytest.approx(1.2)
    assert targets["rh_A_FFJ0"] == pytest.approx(2.25)
    assert targets["rh_A_MFJ0"] == pytest.approx(2.25)
    assert targets["rh_A_RFJ3"] == pytest.approx(1.2)
    assert targets["rh_A_LFJ0"] == pytest.approx(2.25)
    assert _all_targets_obey_config_limits(retargeter, targets)


def test_pointing_hand_keeps_index_open_and_curls_other_fingers() -> None:
    retargeter = CurlRetargeter.from_yaml(TELEOP_CONFIG_PATH)

    targets = retargeter.map(_features(index=0.0, middle=1.0, ring=1.0, pinky=1.0))

    assert targets["rh_A_FFJ3"] == pytest.approx(-0.15)
    assert targets["rh_A_FFJ0"] == pytest.approx(0.05)
    assert targets["rh_A_MFJ3"] == pytest.approx(1.2)
    assert targets["rh_A_MFJ0"] == pytest.approx(2.25)
    assert targets["rh_A_RFJ3"] == pytest.approx(1.2)
    assert targets["rh_A_LFJ3"] == pytest.approx(1.2)


def test_missing_and_low_confidence_features_map_to_open_pose() -> None:
    raw_config = copy.deepcopy(load_curl_retargeter_config(TELEOP_CONFIG_PATH))
    raw_config["retargeting"]["min_confidence"] = 0.5
    retargeter = CurlRetargeter.from_mapping(raw_config)
    open_targets = retargeter.map(_features(confidence=1.0))

    assert retargeter.map(None) == open_targets
    assert retargeter.map(
        _features(thumb=1.0, index=1.0, middle=1.0, ring=1.0, pinky=1.0, confidence=0.1)
    ) == open_targets


def test_nonfinite_features_are_sanitized_to_finite_targets() -> None:
    retargeter = CurlRetargeter.from_yaml(TELEOP_CONFIG_PATH)
    dirty_features = HandFeatures(
        thumb_curl=float("nan"),
        index_curl=float("inf"),
        middle_curl=-1.0,
        ring_curl=0.5,
        pinky_curl=2.0,
        pinch_thumb_index=float("nan"),
        palm_roll_proxy=float("inf"),
        palm_pitch_proxy=float("-inf"),
        confidence=1.0,
    )

    targets = retargeter.map(dirty_features)

    assert all(math.isfinite(target) for target in targets.values())
    assert targets["rh_A_FFJ0"] == pytest.approx(0.05)
    assert targets["rh_A_LFJ0"] == pytest.approx(2.25)
    assert _all_targets_obey_config_limits(retargeter, targets)


def test_joint_limit_clipping_is_applied_after_interpolation() -> None:
    retargeter = CurlRetargeter.from_mapping(
        {
            "retargeting": {
                "type": "curl",
                "fingers": {
                    "index": {
                        "feature": "index_curl",
                        "targets": {
                            "index_motor": {
                                "open": -2.0,
                                "closed": 2.0,
                                "min": 0.0,
                                "max": 1.0,
                            }
                        },
                    }
                },
            }
        }
    )

    assert retargeter.map(_features(index=0.0))["index_motor"] == pytest.approx(0.0)
    assert retargeter.map(_features(index=1.0))["index_motor"] == pytest.approx(1.0)


def test_per_finger_scale_config_changes_effective_curl() -> None:
    retargeter = CurlRetargeter.from_mapping(
        {
            "retargeting": {
                "type": "curl",
                "fingers": {
                    "index": {
                        "feature": "index_curl",
                        "scale": 0.5,
                        "targets": {
                            "index_motor": {
                                "open": 0.0,
                                "closed": 2.0,
                                "min": 0.0,
                                "max": 2.0,
                            }
                        },
                    }
                },
            }
        }
    )

    targets = retargeter.map(_features(index=1.0))

    assert targets["index_motor"] == pytest.approx(1.0)


def test_invalid_feature_name_is_rejected() -> None:
    with pytest.raises(CurlRetargeterError, match="feature"):
        CurlRetargeter.from_mapping(
            {
                "retargeting": {
                    "type": "curl",
                    "fingers": {
                        "index": {
                            "feature": "pinch_thumb_index",
                            "targets": {
                                "index_motor": {
                                    "open": 0.0,
                                    "closed": 1.0,
                                    "min": 0.0,
                                    "max": 1.0,
                                }
                            },
                        }
                    },
                }
            }
        )


def _all_targets_obey_config_limits(retargeter: CurlRetargeter, targets: dict[str, float]) -> bool:
    limits = {}
    for static_target in retargeter.config.static_targets:
        limits[static_target.name] = static_target.limit
    for finger in retargeter.config.fingers:
        for target in finger.targets:
            limits[target.name] = target.limit

    return all(
        limits[name].minimum <= value <= limits[name].maximum
        for name, value in targets.items()
    )
