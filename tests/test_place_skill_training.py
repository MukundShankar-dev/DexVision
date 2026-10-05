"""Focused checks for the versioned place recipe and causal learner interface."""
from __future__ import annotations

import numpy as np
import pytest

from dexvision.learning.place_policy import (
    PlaceDataset, decode_place, load_place_spec, place_action_plan, place_features,
    place_targets,
)
from dexvision.learning.skill_datasets import SkillEpisode, feature_schema

CONFIG = 'configs/level5/place_policy_v1.yaml'


def test_place_recipe_binds_frozen_split_and_runtime_workspace():
    spec, inputs = load_place_spec(CONFIG)
    assert inputs.skill['skill'] == 'place_held_object'
    assert len(spec['seeds']) == 3
    assert place_action_plan(spec, inputs)['action']['named_layout'][2]['lower'] == 0.01
    assert sum(np.asarray(spec['output_scales']) > 0) == 19


def test_place_features_are_causal_and_zero_output_holds():
    spec, inputs = load_place_spec(CONFIG)
    names = list(feature_schema(inputs.plan, inputs.skill)[0])
    source = np.zeros(len(names))
    for axis, value in zip('xyz', (0.01, -0.02, 0.14), strict=True):
        source[names.index(f'prior_applied_actions.base_position_target/{axis}')] = value
        source[names.index(f'entity.position_m.{"xyz".index(axis)}')] = value + 0.01
        source[names.index(f'goal.target_position.{"xyz".index(axis)}')] = value + 0.03
    source[names.index('robot.rotation6d.0')] = 1
    source[names.index('robot.rotation6d.4')] = 1
    features, output_names = place_features(source, names, spec)
    assert np.allclose(features[:3], 0.2)
    assert 'phase.transport' in output_names
    prior = np.zeros(27)
    prior[:3] = [0.01, -0.02, 0.14]
    prior[3] = 1
    assert np.array_equal(decode_place(np.zeros(26), prior, spec, place_action_plan(spec, inputs)), prior)
    with pytest.raises(ValueError, match='Invalid bounded'):
        decode_place(np.full(26, np.nan), prior, spec, place_action_plan(spec, inputs))


def test_place_source_invariants_reject_unlicensed_finger_motion():
    spec, inputs = load_place_spec(CONFIG)
    prior = np.zeros((2, 27))
    prior[:, 3] = 1
    applied = prior.copy()
    applied[0, 9] = -0.01  # active release channel
    episode = type('Episode', (), {'records': {'applied_actions': applied,
        'prior_applied_actions': prior, 'safety_masks': np.zeros((2, 27))}, 'frames': np.arange(2)})()
    plan = place_action_plan(spec, inputs)
    targets, masks = place_targets(episode, plan, spec)
    assert targets[0, 8] < 0 and masks[0, 8] == 1
    applied[0, 7] = 0.01  # frozen wrist channel
    with pytest.raises(ValueError, match='invariants'):
        place_targets(episode, plan, spec)


def test_place_sampler_visits_short_release_phase():
    norm = {'mean': [0.], 'std': [1.]}
    episode = SkillEpisode('synthetic', 'train', 'session', 'cell', 'digest',
        np.zeros((101, 1)), np.zeros((101, 26)), np.ones((101, 26)),
        np.arange(101), np.arange(101, dtype=float),
        tuple(['transport'] * 100 + ['release']), None, {})
    dataset = PlaceDataset([episode], norm)
    indices = dataset.sample_indices(np.random.default_rng(3), 1000)
    assert (indices == 100).sum() > 350
