"""Coverage cache merge uses TRAIN-only arrays with frozen allocation weights."""

import numpy as np
import pytest
from dexvision.evaluation.perception_metrics import CORE_CLASSES
from dexvision.learning.perception_coverage import merge_training_examples


def test_coverage_merge_retains_original_and_allocation_mass():
    previous = {
        c: (np.zeros((2, 706)), np.ones((2, 3)), np.array([1.0, 3.0]))
        for c in CORE_CLASSES
    }
    extra = {
        c: (np.ones((3, 706)), np.zeros((3, 3)), np.array([1.0, 2.0, 3.0]))
        for c in CORE_CLASSES
    }
    result = merge_training_examples(previous, extra, amendment_fraction=0.8)
    x, y, w = result["rigid_cuboid"]
    assert x.shape == (5, 706) and y.shape == (5, 3)
    assert w[:2].sum() == pytest.approx(0.2) and w[2:].sum() == pytest.approx(0.8)
    assert w[0] / w[1] == pytest.approx(1 / 3) and w[3] / w[4] == pytest.approx(2 / 3)
    for c in CORE_CLASSES:
        if c != "rigid_cuboid":
            assert result[c] is previous[c]
    assert not previous["rigid_cuboid"][0].any() and previous["rigid_cuboid"][
        2
    ].tolist() == [1.0, 3.0]


@pytest.mark.parametrize("mass", [0.0, 1.0, -1.0, float("nan")])
def test_invalid_coverage_mass_fails(mass):
    with pytest.raises(ValueError, match="mass"):
        merge_training_examples({}, {}, amendment_fraction=mass)


def test_wrong_cached_feature_contract_rejects():
    previous = {
        c: (np.zeros((2, 370)), np.ones((2, 3)), np.ones(2)) for c in CORE_CLASSES
    }
    with pytest.raises(ValueError, match="shapes"):
        merge_training_examples(previous, previous, amendment_fraction=0.8)
