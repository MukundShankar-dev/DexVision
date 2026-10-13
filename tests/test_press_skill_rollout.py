"""Checkpoint 5.5 fixture-frame depth, bidirectional press/release and bundle checks."""
from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

from dexvision.evaluation.contact_policy import (
    ContactEvidence, contact_scenarios, create_contact_task, physical_metrics, read_contact_entries,
    run_contact_rollout, wrong_contact,
)
from dexvision.evaluation.evaluate_skill import reach_features
from dexvision.evaluation.pick_policy import physical_evidence_world
from dexvision.learning.contact_policy import (
    ContactPolicy, contact_action_plan, contact_features, contact_schema, contact_settings,
    decode_contact, load_contact_spec,
)
from dexvision.learning.place_policy import PlaceDataset
from dexvision.learning.skill_datasets import SkillEpisode, feature_schema
from dexvision.learning.train_skill import _checksum_index, _json, train_skill
from dexvision.learning.training_manifest import build_training_manifest, content_digest

CONFIG = 'configs/level5/press_policy_v1.yaml'


def test_press_axis_and_bidirectional_decoder():
    spec, inputs = load_contact_spec(CONFIG)
    assert spec['fixture_axis_world'] == [1., 0., 0.]
    assert np.flatnonzero(spec['output_scales']).tolist() == [0]
    prior = np.zeros(27)
    prior[3] = 1
    for sign in (-1, 0, 1):
        prediction = np.zeros(26)
        prediction[0] = sign
        action = decode_contact(prediction, prior, spec, contact_action_plan(spec, inputs))
        assert action[0] == pytest.approx(sign * 0.025)
        assert np.array_equal(action[1:], prior[1:])


def test_press_depth_alignment_and_causal_latch_features():
    spec, inputs = load_contact_spec(CONFIG)
    names = list(feature_schema(inputs.plan, inputs.skill)[0])
    source = np.zeros(len(names))
    rest = np.array([0.1, -0.1, 0.2])
    for j in range(3):
        source[names.index(f'entity.position_m.{j}')] = rest[j] + (0.012 if j == 0 else 0)
    source[names.index('goal.target_depth.0')] = 0.014
    features, labels = contact_features(source, names, spec, fixture_rest=rest, pressed_once=True)
    assert features[labels.index('fixture_depth')] == pytest.approx(1.2)
    assert features[labels.index('fixture_depth_shortfall')] == pytest.approx(0.2)
    assert features[labels.index('pressed_once')] == 1
    with pytest.raises(ValueError, match='rest pose'):
        contact_features(source, names, spec)


def test_press_dwell_then_release_is_required_and_overshoot_visible():
    spec, _ = load_contact_spec(CONFIG)
    evidence = ContactEvidence(spec, {'button_id': 'start_button'})
    contact = type('World', (), {'contacts': (('rh_palm', 'start_button'),)})()
    released = type('World', (), {'contacts': ()})()
    for _ in range(3):
        assert not evidence.update(contact, {'press_depth_m': 0.014}, True)['success']
    assert evidence.pressed_once
    for _ in range(2):
        assert not evidence.update(released, {'press_depth_m': 0.001}, False)['success']
    assert evidence.update(released, {'press_depth_m': 0.001}, False)['success']
    unsafe = ContactEvidence(spec, {'button_id': 'start_button'})
    assert unsafe.update(contact, {'press_depth_m': 0.03}, True)['overshoot']


def test_press_live_depth_matches_fixture_and_zero_policy_audits(tmp_path):
    spec, inputs = load_contact_spec(CONFIG)
    scenario = contact_scenarios(inputs, 'validation')[0]
    row, _, _, condition = scenario
    metadata, prefix = read_contact_entries(inputs, [scenario], inputs.root)[row['episode_id']]
    with create_contact_task(inputs, row, metadata, prefix) as task:
        world = physical_evidence_world(task)
        prior = prefix[-1]
        source = reach_features(task, world, row, inputs, prior, prior, np.zeros(27), task._phase)
        rest = metadata['task_config']['initial_state']['entity_positions_m'][row['entity_id']]
        features, names = contact_features(source, list(feature_schema(inputs.plan, inputs.skill)[0]),
            spec, fixture_rest=rest)
        assert features[names.index('fixture_depth')] * spec['depth_scale_m'] == pytest.approx(
            world.require_fixture(row['entity_id']).press_depth_m, abs=1e-10)
        metrics, _ = physical_metrics(task, world)
        assert metrics['press_depth_m'] == world.require_fixture(row['entity_id']).press_depth_m
        wrong_world = replace(world, contacts=(('rh_palm', 'return_bin_right'),))
        assert wrong_contact(wrong_world, world, row, spec)[0]
    skill = copy.deepcopy(inputs.skill)
    skill['contract']['timeout_steps'] = 2
    result = run_contact_rollout(replace(inputs, skill=skill), spec, row, metadata, prefix,
        lambda source, prior, **kwargs: (prior.copy(), np.zeros(26)), condition=condition, output_dir=tmp_path / 'attempt')
    assert result['terminal_result'] != 'succeeded'
    assert all(result['preconditions'].values())
    assert result['physical_audit']['passed']


def test_contact_tiny_cpu_bundle_and_recipe_rejection(tmp_path):
    spec, inputs = load_contact_spec(CONFIG)
    schema = contact_schema(spec, inputs)
    width = len(schema['input_names'])
    owners = build_training_manifest(inputs, seed=5001, device='cpu', model='baseline')
    norm = {'mean': [0.] * width, 'std': [1.] * width, 'ordered_feature_names': schema['input_names'],
        'training_episode_ids': [r['episode_id'] for r in owners['assignments']['train']]}
    norm['normalization_digest'] = content_digest(norm)
    datasets = {}
    for split in ('train', 'validation'):
        episode = SkillEpisode(split, split, split, split, 'synthetic', np.zeros((2, width)),
            np.zeros((2, 26)), np.tile(np.r_[1., np.zeros(25)], (2, 1)), np.arange(2),
            np.arange(2, dtype=float), ('fixture_contact', 'retract'), None, {})
        datasets[split] = PlaceDataset([episode], norm)
    owners.update(contact_recipe=spec, contact_recipe_sha256=content_digest(spec))
    owners.pop('manifest_digest')
    owners['manifest_digest'] = content_digest(owners)
    run = tmp_path / 'run'
    train_skill(datasets, schema=schema, settings=contact_settings(spec, inputs), manifest=owners,
        output_dir=run, config_snapshot={}, stop_after_steps=1)
    policy = ContactPolicy(run / 'last.pt', spec, inputs)
    assert policy.model.schema == schema
    changed = copy.deepcopy(spec)
    changed['depth_scale_m'] *= 2
    with pytest.raises(ValueError, match='bundle mismatch'):
        ContactPolicy(run / 'last.pt', changed, inputs)
    _json(run / 'test_state_attempt.json', {'test': 'reserved'})
    _checksum_index(run)
