"""Checkpoint 5.5 signed push interface, physical outcome and headless smoke checks."""
from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

from dexvision.evaluation.contact_policy import (
    ContactEvidence, contact_scenarios, read_contact_entries, run_contact_rollout,
)
from dexvision.evaluation.evaluate_skill import safe_action
from dexvision.learning.contact_policy import (
    contact_action_plan, contact_features, contact_targets, decode_contact, load_contact_spec,
)
from dexvision.learning.skill_datasets import feature_schema

CONFIG = 'configs/level5/push_policy_v1.yaml'


def test_push_recipe_independent_signed_channels_and_zero_hold():
    spec, inputs = load_contact_spec(CONFIG)
    plan = contact_action_plan(spec, inputs)
    assert spec['max_epochs'] == 200
    assert spec['candidate_epochs'] == [20, 80, 200]
    assert np.flatnonzero(spec['output_scales']).tolist() == [0, 1]
    assert plan['action']['named_layout'][2]['lower'] == 0.01
    prior = np.zeros(27)
    prior[3] = 1
    prior[2] = 0.1
    assert np.array_equal(decode_contact(np.zeros(26), prior, spec, plan), prior)
    prediction = np.zeros(26)
    prediction[:2] = [-1, 1]
    action = decode_contact(prediction, prior, spec, plan)
    assert np.allclose(action[:2], [-0.005, 0.005])
    assert np.array_equal(action[2:], prior[2:])
    action[0] = 1
    assert safe_action(action, prior, plan)[3] == 'workspace_violation'
    with pytest.raises(ValueError, match='Invalid bounded'):
        decode_contact(np.full(26, np.nan), prior, spec, plan)


def test_push_features_require_object_motion_and_target_geometry():
    spec, inputs = load_contact_spec(CONFIG)
    names = list(feature_schema(inputs.plan, inputs.skill)[0])
    source = np.zeros(len(names))
    source[names.index('robot.rotation6d.0')] = 1
    source[names.index('robot.rotation6d.4')] = 1
    source[names.index('entity.position_m.0')] = 0.02
    source[names.index('goal.target_position.0')] = 0.05
    source[names.index('entity.linear_velocity_mps.0')] = 0.01
    features, labels = contact_features(source, names, spec)
    assert features[labels.index('entity_to_zone/x')] == pytest.approx(0.3)
    assert features[labels.index('entity_velocity/x')] == pytest.approx(0.1)
    assert 'entity.contact_with_hand.0' in labels


def test_push_targets_reject_unproven_height_invariance():
    spec, inputs = load_contact_spec(CONFIG)
    prior = np.zeros((2, 27))
    prior[:, 3] = 1
    applied = prior.copy()
    applied[:, :2] = [[0.003, -0.002], [-0.001, 0.003]]
    episode = type('Episode', (), {'records': {'applied_actions': applied,
        'prior_applied_actions': prior, 'safety_masks': np.zeros((2, 27))}, 'frames': np.arange(2)})()
    target, mask = contact_targets(episode, contact_action_plan(spec, inputs), spec)
    assert target[0, 1] < 0 and target[1, 0] < 0
    assert mask.sum() == 4
    applied[0, 2] = 0.001
    with pytest.raises(ValueError, match='invariants'):
        contact_targets(episode, contact_action_plan(spec, inputs), spec)


def test_push_success_requires_physical_settling_not_hand_arrival():
    spec, _ = load_contact_spec(CONFIG)
    evidence = ContactEvidence(spec, {'object_id': 'block_small'})
    world = type('World', (), {'contacts': (('rh_palm', 'block_small'),)})()
    values = {'planar_object_to_target_distance_m': 0.2}
    for _ in range(10):
        assert not evidence.update(world, values, False)['success']
    for _ in range(4):
        assert not evidence.update(world, values, True)['success']
    assert evidence.update(world, values, True)['success']
    assert not evidence.update(world, values, False)['success']


def test_push_matrix_and_headless_zero_policy_retains_failed_attempt(tmp_path):
    spec, inputs = load_contact_spec(CONFIG)
    scenarios = contact_scenarios(inputs, 'validation')
    assert len(scenarios) == inputs.skill['evaluation']['validation']['rollouts_per_track_per_training_seed']
    row, seed, frame, condition = scenarios[0]
    entries = read_contact_entries(inputs, [scenarios[0]], release_root=inputs.root)
    skill = copy.deepcopy(inputs.skill)
    skill['contract']['timeout_steps'] = 2
    limited = replace(inputs, skill=skill)
    result = run_contact_rollout(limited, spec, row, *entries[row['episode_id']],
        lambda source, prior, **kwargs: (prior.copy(), np.zeros(26)), condition=condition, output_dir=tmp_path / 'attempt')
    assert result['terminal_result'] in {'timed_out', 'rejected_before_execution', 'failed'}
    assert all(result['preconditions'].values())
    assert result['reset_seed'] == seed and result['start_frame'] == frame
    assert result['physical_audit']['passed']
    assert (tmp_path / 'attempt' / 'trajectory.npz').exists()
    with pytest.raises(ValueError):
        contact_scenarios(inputs, 'train')
