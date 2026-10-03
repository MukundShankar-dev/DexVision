"""Pick representation, action invariants, causality and tiny CPU training."""
from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from dexvision.learning.phase_losses import phase_errors, pick_phases
from dexvision.learning.pick_policy import (
    boundary_context, load_pick_spec, pick_features, pick_schema, pick_settings, pick_targets,
)
from dexvision.learning.skill_datasets import SkillDataset, feature_schema, vectorize_episode
from dexvision.learning.skill_models import SkillMLP, decode_delta
from dexvision.learning.train_skill import load_checkpoint, train_skill
from dexvision.learning.training_manifest import ReleaseSource, build_training_manifest, select_streams


@pytest.fixture(scope='module')
def pick_inputs():
    return load_pick_spec('configs/level5/pick_policy_v2.yaml')


@pytest.fixture(scope='module')
def pick_examples(pick_inputs):
    spec, inputs = pick_inputs
    rows = [select_streams(inputs)['expert_success'][s][0] for s in ('train', 'validation')]
    root = inputs.root if all((inputs.root / 'data/demos/level4' / r['source_path']).is_dir() for r in rows) else None
    records = ReleaseSource(inputs, root).read_episode_records(rows)
    return [(row, records[row['episode_id']], vectorize_episode(inputs, row, records[row['episode_id']])) for row in rows]


def test_recipe_lock_and_parent_preservation(pick_inputs, tmp_path):
    spec, inputs = pick_inputs
    assert spec['max_epochs'] == 200
    assert inputs.plan['training']['max_epochs'] == 100
    assert pick_settings(spec, inputs)['early_stop']['patience_epochs'] > 200
    path = tmp_path / 'recipe.yaml'
    path.write_text('changed')
    path.with_suffix('.sha256').write_text('bad')
    with pytest.raises(ValueError, match='checksum'):
        load_pick_spec(path)


def test_pick_phase_is_causal_and_loss_retains_failed_phase():
    phase = pick_phases(np.array([0, 1, 1, 0]), np.array([0, .02, .05, .05]), .04)
    assert list(phase) == ['acquire', 'lift', 'stabilize', 'acquire']
    errors = phase_errors(np.ones((4, 2)), np.zeros((4, 2)), np.ones((4, 2)), phase)
    assert errors['acquire'] == {'frames': 2, 'active_values': 4, 'masked_mse': 1.0}


def test_pick_target_roundtrip_zero_action_and_required_closure(pick_inputs, pick_examples):
    spec, inputs = pick_inputs
    for _, _, episode in pick_examples:
        targets, masks = pick_targets(episode, inputs.plan, spec)
        for index in (0, len(episode.frames) // 2, len(episode.frames) - 1):
            prior = episode.records['prior_applied_actions'][index]
            decoded = decode_delta(targets[index] * spec['output_scales'], prior, 'acquire', inputs.plan)
            np.testing.assert_allclose(decoded, episode.records['applied_actions'][index], atol=1e-12)
            np.testing.assert_allclose(decode_delta(np.zeros(26), prior, 'acquire', inputs.plan), prior, atol=1e-12)
        assert np.any(targets[0, 8:] != 0)  # closure starts before contact-derived phase catches up
        assert masks[:, 3:6].all()  # keep orientation active throughout the pick
        changed = copy.deepcopy(episode)
        changed.records['applied_actions'][0, 0] += .0001
        with pytest.raises(ValueError, match='invariants'):
            pick_targets(changed, inputs.plan, spec)


def test_transform_ignores_sequence_annotations_and_global_xy(pick_inputs, pick_examples):
    spec, inputs = pick_inputs
    names = list(feature_schema(inputs.plan, inputs.skill)[0])
    episode = pick_examples[0][2]
    context = boundary_context(episode.inputs[0], names)
    original, output_names, phases = pick_features(episode.inputs, names, context, spec)
    changed = episode.inputs.copy()
    for j, name in enumerate(names):
        if name.startswith('phase.'):
            changed[:, j] = 0
    for j, name in enumerate(names):
        if name in {'robot.position.x', 'robot.position.y', 'entity.position_m.0', 'entity.position_m.1',
                    'prior_applied_actions.base_position_target/x', 'prior_applied_actions.base_position_target/y'}:
            changed[:, j] += .2
    altered, _, new_phases = pick_features(changed, names, context, spec)
    np.testing.assert_allclose(original, altered, atol=1e-14)
    assert np.array_equal(phases, new_phases)
    assert not any('transport' in name for name in output_names)
    assert original.shape[1] == len(pick_schema(spec, inputs)['input_names'])
    model = SkillMLP(pick_schema(spec, inputs))
    assert model(torch.tensor(original[:2], dtype=torch.float32)).shape == (2, 26)


def test_tiny_pick_training_reloads_and_finishes_budget(pick_inputs, pick_examples, tmp_path):
    spec, inputs = pick_inputs
    names = list(feature_schema(inputs.plan, inputs.skill)[0])
    datasets = {}
    for row, _, e in pick_examples:
        x, feature_names, phases = pick_features(e.inputs, names, boundary_context(e.inputs[0], names), spec)
        y, masks = pick_targets(e, inputs.plan, spec)
        small = replace(e, inputs=x[:8], targets=y[:8], masks=masks[:8], frames=e.frames[:8],
                        timestamps=e.timestamps[:8], phases=tuple(phases[:8]))
        norm = {'mean': [0.] * len(feature_names), 'std': [1.] * len(feature_names)}
        datasets[row['split']] = SkillDataset([small], norm)
    settings = pick_settings(spec, inputs)
    settings['max_epochs'] = 2
    settings['selection']['candidate_epochs'] = [1, 2]
    manifest = build_training_manifest(inputs, seed=5001, device='cpu', model='baseline')
    result = train_skill(datasets, schema=pick_schema(spec, inputs), settings=settings, manifest=manifest,
                         output_dir=tmp_path / 'run', config_snapshot={'test': True})
    assert result['finished'] and result['completed_epochs'] == 2
    checkpoint = load_checkpoint(tmp_path / 'run' / 'last.pt')
    model = SkillMLP(checkpoint['schema'])
    model.load_state_dict(checkpoint['model'], strict=True)
    assert checkpoint['training_state']['global_step'] == 2
    assert not (tmp_path / 'run' / 'selected.pt').exists()


def test_preflight_detects_generic_workspace_conflict_and_preserves_runtime(pick_inputs, pick_examples):
    from dexvision.learning.pick_policy import audit_action_bounds, pick_action_plan
    spec, inputs = pick_inputs
    episodes = [example[2] for example in pick_examples]
    assert audit_action_bounds(episodes, inputs.plan)
    runtime_plan = pick_action_plan(spec, inputs)
    assert not audit_action_bounds(episodes, runtime_plan)
    assert runtime_plan['action']['named_layout'][2]['lower'] == .01
    assert inputs.plan['action']['named_layout'][2]['lower'] == .08
    assert runtime_plan['action']['named_layout'][3:] == inputs.plan['action']['named_layout'][3:]


def test_repair_synergy_exact_roundtrip_bounds_and_causality(pick_examples):
    from dexvision.learning.pick_policy import closure_fraction, decode_pick
    spec, inputs = load_pick_spec('configs/level5/pick_policy_v3.yaml')
    for _, _, episode in pick_examples:
        target, mask = pick_targets(episode, inputs.plan, spec)
        assert mask[:, 8].all() and not mask[:, 9:].any()
        for index in range(len(episode.frames)):
            prior = episode.records['prior_applied_actions'][index]
            np.testing.assert_allclose(decode_pick(target[index], prior, spec, inputs.plan),
                                       episode.records['applied_actions'][index], atol=1e-12)
            np.testing.assert_allclose(decode_pick(np.zeros(26), prior, spec, inputs.plan), prior, atol=1e-12)
    prior = pick_examples[0][2].records['prior_applied_actions'][0].copy()
    prediction = np.zeros(26)
    prediction[8] = 1
    for _ in range(100):
        prior = decode_pick(prediction, prior, spec, inputs.plan)
    assert closure_fraction(prior[7:], spec) == pytest.approx(1)
    np.testing.assert_allclose(prior[7:], spec['closure']['closed_targets'])
    prediction[8] = -1
    for _ in range(100):
        prior = decode_pick(prediction, prior, spec, inputs.plan)
    assert closure_fraction(prior[7:], spec) == pytest.approx(0)
    prior[9] += .001
    with pytest.raises(ValueError, match='synergy'):
        decode_pick(np.zeros(26), prior, spec, inputs.plan)
    for bad in (float('nan'), 2.):
        prediction[8] = bad
        with pytest.raises(ValueError, match='bounded pick prediction'):
            decode_pick(prediction, prior, spec, inputs.plan)
    altered = copy.deepcopy(pick_examples[0][2])
    altered.records['applied_actions'][0, 9] += .001
    with pytest.raises(ValueError, match='synergy'):
        pick_targets(altered, inputs.plan, spec)


def test_repair_refuses_exposed_test_before_loading_weights(tmp_path):
    from dexvision.evaluation.pick_policy import evaluate_pick
    with pytest.raises(ValueError, match='original test is exposed'):
        evaluate_pick('configs/level5/pick_policy_v3.yaml', tmp_path / 'missing.pt', tmp_path / 'test')
    assert not (tmp_path / 'test').exists()


def test_v4_geometry_roundtrip_and_full_closure_interlock(pick_examples):
    from dexvision.learning.pick_policy import decode_pick
    spec, inputs = load_pick_spec('configs/level5/pick_policy_v4.yaml')
    names = list(feature_schema(inputs.plan, inputs.skill)[0])
    for _, _, episode in pick_examples:
        context = boundary_context(episode.inputs[0], names, spec)
        target, mask = pick_targets(episode, inputs.plan, spec, names=names)
        features, feature_names, _ = pick_features(episode.inputs, names, context, spec)
        assert features.shape[1] == len(feature_names) == 28
        for index in range(len(episode.frames)):
            prior = episode.records['prior_applied_actions'][index]
            options = {'features': episode.inputs[index], 'names': names, 'context': context}
            np.testing.assert_allclose(decode_pick(target[index], prior, spec, inputs.plan, **options),
                                       episode.records['applied_actions'][index], atol=1e-10)
            np.testing.assert_allclose(decode_pick(np.zeros(26), prior, spec, inputs.plan, **options), prior, atol=1e-12)
        prior = episode.records['prior_applied_actions'][0]
        gain = np.zeros(26)
        gain[2] = 1
        action = decode_pick(gain, prior, spec, inputs.plan, features=episode.inputs[0], names=names, context=context)
        assert action[2] == prior[2]  # an early learned lift request cannot bypass closure
        bad = copy.deepcopy(episode)
        bad.records['applied_actions'][0, 2] += .0001
        with pytest.raises(ValueError, match='before full closure'):
            pick_targets(bad, inputs.plan, spec, names=names)


def test_v4_features_are_equivariant_to_common_world_yaw(pick_examples):
    from dexvision.features.hand_base import rotation_matrix_to_quaternion, quaternion_to_rotation_matrix
    from dexvision.learning.skill_datasets import rotation6d
    spec, inputs = load_pick_spec('configs/level5/pick_policy_v4.yaml')
    names = list(feature_schema(inputs.plan, inputs.skill)[0])
    source = pick_examples[0][2].inputs[21].copy()
    angle = .7
    rotate = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
    moved = source.copy()
    for keys in ([f'robot.position.{a}' for a in 'xyz'],
                 [f'entity.position_m.{j}' for j in range(3)],
                 [f'prior_applied_actions.base_position_target/{a}' for a in 'xyz'],
                 [f'entity.linear_velocity_mps.{j}' for j in range(3)],
                 [f'entity.angular_velocity_radps.{j}' for j in range(3)]):
        ix = [names.index(k) for k in keys]
        moved[ix] = rotate @ source[ix]
    for prefix in ('robot.rotation6d.', 'entity.rotation_6d.'):
        ix = [names.index(f'{prefix}{j}') for j in range(6)]
        v = source[ix]
        matrix = rotate @ np.column_stack((v[:3], v[3:], np.cross(v[:3], v[3:])))
        moved[ix] = matrix[:, :2].T.reshape(6)
    ix = [names.index(f'prior_applied_actions.base_orientation_target/q{a}') for a in 'wxyz']
    moved[ix] = rotation_matrix_to_quaternion(rotate @ quaternion_to_rotation_matrix(source[ix]))
    context = boundary_context(pick_examples[0][2].inputs[0], names, spec)
    moved_context = context.copy()
    ref = np.column_stack((context[1:4], context[4:7], np.cross(context[1:4], context[4:7])))
    moved_context[1:7] = rotation6d(rotation_matrix_to_quaternion(rotate @ ref))
    a = pick_features(source, names, context, spec)[0]
    b = pick_features(moved, names, moved_context, spec)[0]
    np.testing.assert_allclose(a, b, atol=1e-12)
