"""Signed push frame, immutable source contract and independent failure evidence."""
from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

from dexvision.evaluation.contact_policy import contact_scenarios, read_contact_entries, run_contact_rollout
from dexvision.learning.contact_policy import contact_action_plan
from dexvision.learning.push_repair import decode_push, load_push_spec, push_basis, push_features, runtime_spec
from dexvision.learning.skill_datasets import feature_schema

CONFIG = 'configs/level5/push_policy_v2.yaml'


def test_signed_task_basis_preserves_retraction_and_lateral_correction():
    spec, _ = load_push_spec(CONFIG)
    basis = push_basis([-.03, .06, 0], [0, 0, 0])
    prior = np.zeros(27)
    prior[3] = 1
    prior[2] = .1
    assert np.array_equal(decode_push(np.zeros(26), prior, basis, spec), prior)
    prediction = np.zeros(26)
    prediction[:2] = [-.5, .3]
    action = decode_push(prediction, prior, basis, spec)
    assert np.allclose(basis.T @ (action[:2] - prior[:2]), [-.0025, .0015])
    assert np.array_equal(action[2:], prior[2:])
    prediction[:2] = [1, 1]
    assert np.max(np.abs(decode_push(prediction, prior, basis, spec)[:2])) <= .005 + 1e-12
    with pytest.raises(ValueError, match='Degenerate'):
        push_basis([0, 0, 0], [0, 0, 0])
    with pytest.raises(ValueError, match='Invalid bounded'):
        decode_push(np.full(26, np.nan), prior, basis, spec)


def test_task_features_are_causal_and_use_fixed_entry_direction():
    spec, inputs = load_push_spec(CONFIG)
    names = list(feature_schema(inputs.plan, inputs.skill)[0])
    source = np.zeros(len(names))
    source[names.index('goal.target_position.1')] = .1
    source[names.index('entity.position_m.0')] = .02
    source[names.index('entity.linear_velocity_mps.1')] = .01
    features, labels, basis = push_features(source, names, spec, fixture_rest=[0, 0, 0])
    assert np.allclose(basis[:, 0], [0, 1])
    assert features[labels.index('object_to_goal/along')] == pytest.approx(1)
    assert features[labels.index('object_to_goal/lateral')] == pytest.approx(.2)
    assert features[labels.index('object_velocity/along')] == pytest.approx(.1)
    assert 'entity.contact_with_hand.0' in labels
    assert all('target_type' not in n and 'entity_id' not in n for n in labels)
    with pytest.raises(ValueError, match='goal moved'):
        push_features(np.stack([source, source + .01]), names, spec, fixture_rest=[0, 0, 0])


def test_repair_preserves_physical_gates_budget_and_workspace():
    spec, inputs = load_push_spec(CONFIG)
    assert inputs.skill['gates']['state_success_min'] == .8
    assert inputs.skill['gates']['state_worst_family_success_min'] == .65
    assert spec['max_epochs'] == 200
    assert spec['seeds'] == [5001, 5002, 5003]
    assert spec['confirmation']['fresh_reset_seed_start'] == 20555000
    assert contact_action_plan(spec, inputs)['action'] == contact_action_plan(runtime_spec(spec), inputs)['action']


def test_headless_zero_push_still_counts_failure_and_audits_state(tmp_path):
    spec, inputs = load_push_spec(CONFIG)
    scenario = contact_scenarios(inputs, 'validation')[0]
    row, _, _, condition = scenario
    metadata, prefix = read_contact_entries(inputs, [scenario], inputs.root)[row['episode_id']]
    skill = copy.deepcopy(inputs.skill)
    skill['contract']['timeout_steps'] = 2
    result = run_contact_rollout(replace(inputs, skill=skill), runtime_spec(spec), row, metadata, prefix,
        lambda source, prior, **kw: (prior.copy(), np.zeros(26)), condition=condition, output_dir=tmp_path / 'failed')
    assert result['terminal_result'] != 'succeeded'
    assert result['physical_audit']['passed']
    assert (tmp_path / 'failed' / 'trajectory.npz').exists()


def test_settled_setup_is_exact_hold_without_modifying_source_or_actions():
    from dexvision.evaluation.push_settled import load_settled_spec, settled_prefix
    spec, parent, setup, inputs = load_settled_spec('configs/level5/push_policy_v3.yaml')
    assert spec['setup_hold_samples'] == 120
    assert parent['version'] == 'level5/push-policy-v2'
    assert setup['confirmation']['fresh_reset_seed_start'] == 20755000
    prefix = np.arange(54, dtype=float).reshape(2, 27)
    original = prefix.copy()
    held = settled_prefix(prefix, spec['setup_hold_samples'])
    assert len(held) == 122
    assert np.array_equal(prefix, original)
    assert np.array_equal(held[:2], original)
    assert np.array_equal(held[2:], np.repeat(original[-1:], 120, axis=0))
    with pytest.raises(ValueError, match='finite nonempty'):
        settled_prefix(np.empty((0, 27)), 120)
    assert inputs.skill['gates']['state_success_min'] == .8


def test_distance_cap_preserves_near_goal_negative_and_lateral_error():
    from dexvision.learning.push_distance import load_push_spec, push_features
    spec, inputs = load_push_spec('configs/level5/push_policy_v4.yaml')
    names = list(feature_schema(inputs.plan, inputs.skill)[0])
    source = np.zeros(len(names))
    source[names.index('goal.target_position.0')] = .1
    source[names.index('entity.position_m.1')] = .04
    values, labels, _ = push_features(source, names, spec, fixture_rest=[0, 0, 0])
    along = labels.index('capped_object_to_goal/along')
    lateral = labels.index('object_to_goal/lateral')
    assert values[along] == pytest.approx(.5)
    assert values[lateral] == pytest.approx(-.4)
    source[names.index('entity.position_m.0')] = .07
    assert push_features(source, names, spec, fixture_rest=[0, 0, 0])[0][along] == pytest.approx(.3)
    source[names.index('entity.position_m.0')] = .12
    assert push_features(source, names, spec, fixture_rest=[0, 0, 0])[0][along] == pytest.approx(-.2)


def test_distance_matrix_intervals_are_evaluation_only_and_do_not_mutate_source():
    from dexvision.evaluation.push_distance import matrix_inputs
    spec, inputs = load_push_spec(CONFIG)
    scenario = contact_scenarios(inputs, 'validation')[0]
    row, seed, _, _ = scenario
    before = copy.deepcopy(row)
    frozen = {'split': 'validation', 'records': [{'row': row, 'prefix_actions': np.zeros((3, 27)).tolist()}],
        'matrix': {'cells': [{'cell_id': row['cell_id'], 'episode_ids': [row['episode_id']],
            'reset_seeds': [seed], 'start_frames': [3]}], 'unique_reset_count': 1,
            'rollouts_per_track_per_training_seed': 4}}
    result = matrix_inputs(inputs, frozen)
    assert row == before
    assert len(contact_scenarios(result, 'validation')) == 4
    assert inputs.skill['evaluation']['validation']['unique_reset_count'] == 30


def test_entry_geometry_removes_constant_family_height_and_contact_offset_cues():
    from dexvision.learning.push_canonical import load_push_spec, push_features
    spec, inputs = load_push_spec('configs/level5/push_policy_v5.yaml')
    names = list(feature_schema(inputs.plan, inputs.skill)[0])
    cube = np.zeros(len(names))
    cube[names.index('goal.target_position.0')] = .15
    cube[names.index('entity.position_m.2')] = .03
    cube[names.index('prior_applied_actions.base_position_target/z')] = .09
    cube[names.index('entity.class.cuboid')] = 1
    puck = cube.copy()
    puck[names.index('entity.position_m.2')] = .005
    puck[names.index('prior_applied_actions.base_position_target/x')] = -.07
    puck[names.index('prior_applied_actions.base_position_target/z')] = .102
    puck[names.index('entity.class.cuboid')] = 0
    puck[names.index('entity.class.flat_puck')] = 1
    a, labels, _ = push_features(cube, names, spec, fixture_rest=[0, 0, .03])
    b, other, _ = push_features(puck, names, spec, fixture_rest=[0, 0, .005])
    assert labels == other and len(labels) == 19
    assert np.array_equal(a, b)
    assert all('class.' not in n and 'command_height' not in n for n in labels)
    before = cube.copy()
    current = cube.copy()
    current[names.index('entity.position_m.0')] = .01
    x, _, _ = push_features(current, names, spec, fixture_rest=[0, 0, .03], entry_source=cube)
    assert x[labels.index('hand_object_geometry_change/along')] == pytest.approx(.1)
    assert np.array_equal(cube, before)


def test_canonical_decoder_learns_only_signed_rate_and_holds_other_channels():
    from dexvision.learning.push_canonical import decode_push, load_push_spec
    from dexvision.learning.push_repair import push_basis
    spec, _ = load_push_spec('configs/level5/push_policy_v5.yaml')
    prior = np.zeros(27)
    prior[3] = 1
    prior[2] = .1
    prediction = np.ones(26)
    prediction[0] = 0
    basis = push_basis([.1, .1, 0], [0, 0, 0])
    assert np.array_equal(decode_push(prediction, prior, basis, spec), prior)
    prediction[0] = -.5
    result = decode_push(prediction, prior, basis, spec)
    assert np.allclose(basis.T @ (result[:2] - prior[:2]), [-.0025, 0.])
    assert np.array_equal(result[2:], prior[2:])


def test_push_success_latch_separates_post_success_actions_without_future_labels():
    from dexvision.learning.push_latched import load_push_spec, push_features
    spec, inputs = load_push_spec('configs/level5/push_policy_v6.yaml')
    names = list(feature_schema(inputs.plan, inputs.skill)[0])
    source = np.zeros(len(names))
    source[names.index('goal.target_position.0')] = .1
    a, labels, _ = push_features(source, names, spec, fixture_rest=[0, 0, 0], qualified_once=False)
    b, _, _ = push_features(source, names, spec, fixture_rest=[0, 0, 0], qualified_once=True)
    assert len(labels) == 20
    assert labels[-1] == 'physical_goal_succeeded_once'
    assert np.array_equal(a[:-1], b[:-1])
    assert a[-1] == 0 and b[-1] == 1
    with pytest.raises(ValueError, match='latch must be boolean'):
        push_features(source, names, spec, fixture_rest=[0, 0, 0], qualified_once=2)


def test_predicate_push_preserves_goal_resolution_without_continuous_rate_proxy():
    from dexvision.learning.push_predicates import load_push_spec, push_features
    spec, inputs = load_push_spec('configs/level5/push_policy_v7.yaml')
    names = list(feature_schema(inputs.plan, inputs.skill)[0])
    reference = np.zeros(len(names))
    reference[names.index('goal.target_position.0')] = .15
    current = reference.copy()
    current[names.index('entity.linear_velocity_mps.0')] = .01
    a, labels, _ = push_features(current, names, spec, fixture_rest=[0, 0, 0], entry_source=reference)
    current[names.index('entity.linear_velocity_mps.0')] = .015
    b, _, _ = push_features(current, names, spec, fixture_rest=[0, 0, 0], entry_source=reference)
    assert len(labels) == 12
    assert np.array_equal(a, b)
    current[names.index('entity.linear_velocity_mps.0')] = .021
    c, _, _ = push_features(current, names, spec, fixture_rest=[0, 0, 0], entry_source=reference)
    assert a[labels.index('object_speed_within_original_limit')] == 1
    assert c[labels.index('object_speed_within_original_limit')] == 0
    current[names.index('entity.position_m.0')] = .12
    assert push_features(current, names, spec, fixture_rest=[0, 0, 0], entry_source=reference)[0][0] == pytest.approx(.3)


def test_guarded_workspace_preserves_signed_direction_and_zero_hold():
    from dexvision.learning.push_guarded import load_push_spec, project_workspace
    spec, _ = load_push_spec('configs/level5/push_policy_v8.yaml')
    prior = np.zeros(27)
    prior[1:3] = [-.178, .1]
    proposed = prior.copy()
    proposed[:2] += [.003, -.005]
    actual, fraction = project_workspace(prior, proposed, spec['workspace_bounds'])
    assert fraction == pytest.approx(.4)
    np.testing.assert_allclose(actual[:2] - prior[:2], fraction * (proposed[:2] - prior[:2]))
    assert actual[1] == -.18 and np.array_equal(actual[2:], prior[2:])
    hold, fraction = project_workspace(prior, prior, spec['workspace_bounds'])
    assert fraction == 1 and np.array_equal(hold, prior)


def test_guarded_phase_alias_and_settle_hold_preserve_learned_signed_output():
    from dexvision.learning.push_guarded import PushPolicy, load_push_spec
    spec, _ = load_push_spec('configs/level5/push_policy_v8.yaml')

    class Parent:
        qualified_once = False

        def __call__(self, source, prior, **kwargs):
            self.source = source.copy()
            proposed = prior.copy()
            proposed[0] -= .002
            return proposed, np.full(26, -.4)

        def reset(self):
            self.qualified_once = False

    policy = object.__new__(PushPolicy)
    policy.parent, policy.spec, policy.events = Parent(), spec, []
    policy.names = ['phase.approach', 'phase.push_contact', 'phase.settle']
    source, prior = np.array([1., 0, 0]), np.zeros(27)
    prior[2] = .1
    raw, prediction = policy(source, prior)
    assert np.array_equal(source, [1, 0, 0])
    assert np.array_equal(policy.parent.source, [0, 1, 0])
    assert raw[0] == -.002 and prediction[0] == -.4
    raw, _ = policy(np.array([0., 0, 1]), prior)
    assert np.array_equal(raw, prior) and policy.events[-1]['settle_hold']
    policy.parent.qualified_once = True
    raw, _ = policy(np.array([0., 0, 1]), prior)
    assert raw[0] == -.002 and not policy.events[-1]['settle_hold']
    policy.reset()
    assert not policy.parent.qualified_once and policy.events == []


def test_guarded_decoder_audit_rejects_hidden_bounds_or_holds(tmp_path):
    import json
    from dexvision.learning.push_guarded import audit_decoder_events, load_push_spec
    spec, _ = load_push_spec('configs/level5/push_policy_v8.yaml')
    prior = np.zeros(27)
    prior[2] = .1
    proposed = prior.copy()
    proposed[0] = .001
    event = {'entry_phase_alias': False, 'settle_hold': True, 'workspace_fraction': np.float64(1.),
        'unbounded_learned_proposal': proposed.tolist()}
    path = tmp_path / 'trajectory.npz'
    np.savez(path, raw_actions=prior[None], prior_applied_actions=prior[None])
    summary = audit_decoder_events([event], path, spec)
    assert summary['settle_hold_samples'] == 1
    json.dumps(summary)
    event['settle_hold'] = False
    with pytest.raises(ValueError, match='evidence mismatch'):
        audit_decoder_events([event], path, spec)
