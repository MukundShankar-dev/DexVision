"""Rare primary phases retain loss mass without exposing phase to inference."""

from collections import Counter, defaultdict
import pytest
from dexvision.learning.perception_context_phase import phase_sample_weight


def test_role_phase_and_episode_weight_totals_are_balanced():
    rows = []
    for episode in range(30):
        for index in range(10 + episode):
            rows.append((str(episode), "approach", True))
    for index in range(3):
        rows.append(("rare", "place", True))
    for episode in range(60):
        for index in range(8):
            rows.append((str(episode), "place", False))
    counts = Counter(rows)
    phase_episodes = defaultdict(set)
    for episode, phase, primary in rows:
        phase_episodes[primary, phase].add(episode)
    totals = defaultdict(float)
    for episode, phase, primary in rows:
        totals[primary, phase] += phase_sample_weight(
            episode, phase, primary, counts, phase_episodes
        )
    assert totals[True, "approach"] == pytest.approx(4.0)
    assert totals[True, "place"] == pytest.approx(4.0)
    assert totals[False, "place"] == pytest.approx(1.0)
