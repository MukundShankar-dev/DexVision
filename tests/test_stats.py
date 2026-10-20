import pytest

from dexvision.eval.stats import wilson_interval


@pytest.mark.parametrize(
    ("k", "n", "expected"),
    [
        (0, 10, (0.0, 0.2775)),
        (10, 10, (0.7225, 1.0)),
        (5, 10, (0.2366, 0.7634)),
        (98, 100, (0.9300, 0.9945)),
    ],
)
def test_wilson_matches_reference_values(k, n, expected):
    lo, hi = wilson_interval(k, n)
    assert lo == pytest.approx(expected[0], abs=1e-4)
    assert hi == pytest.approx(expected[1], abs=1e-4)


def test_wilson_rejects_impossible_counts():
    with pytest.raises(ValueError):
        wilson_interval(3, 2)
    with pytest.raises(ValueError):
        wilson_interval(0, 0)
