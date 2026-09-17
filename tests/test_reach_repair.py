"""Regression checks for the measured reach training failure and versioned repair."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from dexvision.learning.reach_repair import load_repair_spec, physical_scales, transform_features
from dexvision.learning.skill_datasets import feature_schema
from dexvision.learning.skill_models import SkillMLP, decode_delta, model_schema


@pytest.fixture(scope='module')
def frozen():
    return load_repair_spec('configs/level5/reach_repair_v1.yaml')


def test_relative_features_use_command_error_and_measured_tracking_lag(frozen):
    _, inputs = frozen
    names = list(feature_schema(inputs.plan, inputs.skill)[0])
    values = np.zeros((2, len(names)))
    for row in values:
        row[:3] = [.04, .03, .15]
        for i, axis in enumerate('xyz'):
            row[names.index(f'prior_applied_actions.base_position_target/{axis}')] = [.05, .02, .16][i]
            row[names.index(f'goal.approach_position.{i}')] = [.1, -.04, .2][i]
    result, output_names = transform_features(values, names, 'relative_goal')
    np.testing.assert_allclose(result, [[.05, -.06, .04, .01, -.01, .01]] * 2)
    assert len(output_names) == 6
    # Unrelated entities/class columns cannot introduce untrained weights at rollout.
    values[:, names.index('entity.position_m.2')] = 100
    np.testing.assert_array_equal(transform_features(values, names, 'relative_goal')[0], result)


def test_physical_scaling_does_not_amplify_near_constant_rotations(frozen):
    spec, inputs = frozen
    names = list(feature_schema(inputs.plan, inputs.skill)[0])
    scales = physical_scales(names, inputs.plan, spec)
    idx = names.index('robot.rotation6d.4')
    assert .01 / 8.205477788271441e-6 > 1000  # Measured original training divisor.
    assert .01 / scales[idx] == pytest.approx(.01)
    assert scales[names.index('robot.position.x')] == .1
    assert scales[names.index('prior_applied_actions.base_position_target/x')] == pytest.approx(.4)
    bad = copy.deepcopy(spec)
    bad['physical_scales']['position_m'] = 0
    with pytest.raises(ValueError, match='positive'):
        physical_scales(names, inputs.plan, bad)


def test_invariant_heads_prevent_accumulating_zero_target_prediction_bias(frozen):
    _, inputs = frozen
    prior = np.zeros(27)
    prior[2:4] = [.2, 1]
    prediction = np.full(26, .1)
    prediction[:3] = 0
    uncorrected = prior.copy()
    for _ in range(20):
        uncorrected = decode_delta(prediction, uncorrected, 'approach', inputs.plan)
    assert uncorrected[7] > inputs.plan['action']['named_layout'][7]['upper']
    prediction[3:] = 0  # Only permitted after training-data invariant proof.
    corrected = prior.copy()
    for _ in range(20):
        corrected = decode_delta(prediction, corrected, 'approach', inputs.plan)
    np.testing.assert_array_equal(corrected, prior)


def test_relative_model_still_learns_actions_and_has_no_goal_servo_fallback(frozen):
    _, inputs = frozen
    schema = model_schema(inputs.plan, inputs.skill)
    schema['input_names'] = ['goal_error/x', 'goal_error/y', 'goal_error/z', 'tracking_lag/x', 'tracking_lag/y', 'tracking_lag/z']
    model = SkillMLP(schema)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    predictions = model(torch.ones(1, 6)).detach().numpy()[0]
    np.testing.assert_array_equal(predictions, np.zeros(26))
    prior = np.zeros(27)
    prior[2:4] = [.2, 1]
    np.testing.assert_array_equal(decode_delta(predictions, prior, 'approach', inputs.plan), prior)


def test_repair_spec_tampering_and_future_skill_are_rejected(frozen, tmp_path):
    from dexvision.learning.training_manifest import digest_file

    path = tmp_path / 'repair.yaml'
    path.write_text(Path('configs/level5/reach_repair_v1.yaml').read_text())
    path.with_suffix('.sha256').write_text(f'{digest_file(path)}  {path.name}\n')
    path.write_text(path.read_text() + '\nextra: tampering\n')
    with pytest.raises(ValueError, match='checksum'):
        load_repair_spec(path)


def test_cli_rejects_missing_training_inputs(capsys, tmp_path):
    from dexvision.apps.repair_reach import main

    assert main(['train', '--output-dir', str(tmp_path/'absent')]) == 2
    assert '--cache-dir' in capsys.readouterr().err
    assert not (tmp_path/'absent').exists()


@pytest.fixture
def synthetic_cache(frozen, tmp_path):
    """Frozen ownership with synthetic tensors; no ignored output/release dependency."""
    from dexvision.learning.skill_datasets import SkillEpisode, fit_normalization
    from dexvision.learning.training_manifest import skill_interval
    from dexvision.learning.train_skill import _checksum_index, _json

    _, inputs = frozen
    names, continuous = feature_schema(inputs.plan, inputs.skill)
    root = tmp_path / 'cache'
    root.mkdir()
    train_episodes = []
    for split in ('train', 'validation'):
        xs, ys, ids, cells = [], [], [], []
        for row in inputs.splits[split]['episodes']:
            interval = skill_interval(inputs.skill, row)
            if interval is None:
                continue
            start, end = interval
            count = end - start
            x = np.zeros((count, len(names)))
            for a in (0, 1):
                x[:, names.index(f'goal.approach_position.{a}')] = .1
            x[:, names.index('goal.approach_position.2')] = -.05
            y = np.zeros((count, 26))
            y[:, :3] = [.1, .1, -.1]
            xs.append(x)
            ys.append(y)
            ids += [row['episode_id']] * count
            cells += [row['cell_id']] * count
            if split == 'train':
                train_episodes.append(SkillEpisode(row['episode_id'], split, row['recording_session_id'],
                    row['cell_id'], row['episode_digest'], x, y, np.ones_like(y), np.arange(start, end),
                    np.arange(count) * .034, ('approach',) * count, (start, end), {}))
        x, y = np.concatenate(xs), np.concatenate(ys)
        np.savez_compressed(root / f'{split}.npz', inputs=x, targets=y, masks=np.ones_like(y),
                            episode_ids=np.array(ids), cell_ids=np.array(cells))
    _json(root / 'normalization.json', fit_normalization(train_episodes, names, continuous, inputs.digests))
    _json(root / 'input_digests.json', inputs.digests)
    _checksum_index(root)
    return root


def test_cache_invariants_ownership_and_tampering(frozen, synthetic_cache):
    from dexvision.learning.reach_repair import repair_datasets
    from dexvision.learning.train_skill import _checksum_index

    spec, inputs = frozen
    datasets, zero, _ = repair_datasets(spec, inputs, synthetic_cache, 'relative_goal_hold')
    assert zero == [False] * 3 + [True] * 23
    assert datasets['train'].inputs.shape[1] == 6
    assert not datasets['train'].masks[:, 3:].any()
    path = synthetic_cache / 'train.npz'
    with np.load(path) as data:
        values = {key:data[key] for key in data.files}
    values['episode_ids'][0] = 'foreign_episode'
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match='artifact changed'):
        repair_datasets(spec, inputs, synthetic_cache, 'relative_goal_hold')
    _checksum_index(synthetic_cache)
    with pytest.raises(ValueError, match='membership'):
        repair_datasets(spec, inputs, synthetic_cache, 'relative_goal_hold')


def test_goal_plane_training_reload_and_rotational_equivariance(synthetic_cache, tmp_path, monkeypatch):
    from dexvision.learning import reach_repair as repair
    from dexvision.learning.train_skill import load_checkpoint

    spec, inputs = load_repair_spec('configs/level5/reach_repair_v2.yaml')
    spec = copy.deepcopy(spec)
    spec['max_epochs'], spec['candidate_epochs'] = 2, [1, 2]
    monkeypatch.setattr(repair, 'load_repair_spec', lambda path: (spec, inputs))
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        run = tmp_path / 'run'
        result = repair.train_repair('synthetic', synthetic_cache, run, arm='goal_plane_hold', seed=5001)
        assert result['completed_epochs'] == 2 and result['finished']
        checkpoint = load_checkpoint(run / 'last.pt')
        assert checkpoint['settings']['early_stop']['patience_epochs'] == 3
        policy = repair.RepairPolicy(run / 'last.pt', spec, inputs)
        names = list(feature_schema(inputs.plan, inputs.skill)[0])
        features = np.zeros(len(names))
        for axis in (0, 1):
            features[names.index(f'goal.approach_position.{axis}')] = .1
        prior = np.zeros(27)
        prior[3] = 1
        action, prediction = policy(features, prior, 'approach')
        assert np.isfinite(action).all()
        assert action[0] == pytest.approx(action[1])
        assert 0 <= action[0] <= .025
        np.testing.assert_array_equal(action[3:], prior[3:])
        rotated = features.copy()
        rotated[names.index('goal.approach_position.0')] = -.1
        rotated_action, _ = policy(rotated, prior, 'approach')
        np.testing.assert_allclose(rotated_action[:3], [-action[1], action[0], action[2]], atol=1e-8)
        for parameter in policy.model.parameters():
            parameter.data.zero_()
        zero_action, zero_prediction = policy(features, prior, 'approach')
        np.testing.assert_array_equal(zero_action, prior)
        np.testing.assert_array_equal(zero_prediction, np.zeros(26))
    finally:
        torch.set_num_threads(old_threads)


def test_weld_calibration_compares_the_same_body_origin():
    from dexvision.learning.reach_repair import calibrated_features
    from dexvision.learning.skill_datasets import rotation6d

    spec, inputs = load_repair_spec('configs/level5/reach_repair_v3.yaml')
    recipe = spec['models']['goal_plane_calibrated_hold']
    names = list(feature_schema(inputs.plan, inputs.skill)[0])
    values = np.zeros((2, len(names)))
    # Known command at [0.1, 0.2, 0.2], identity rotation, satisfied frozen weld.
    values[:, :3] = np.array([.1, .2, .2]) + spec['weld_relpose'][:3]
    values[:, 3:9] = rotation6d(spec['weld_relpose'][3:])
    corrected = calibrated_features(values, names, spec, recipe)
    np.testing.assert_allclose(corrected[:, :3], [[.1, .2, .2]] * 2, atol=1e-14)
    # A real 2 mm displacement is preserved instead of erased as an offset.
    values[1, 0] += .002
    corrected = calibrated_features(values, names, spec, recipe)
    assert corrected[1, 0] - corrected[0, 0] == pytest.approx(.002)
    np.testing.assert_array_equal(corrected[:, 3:], values[:, 3:])


def test_physical_pose_audit_does_not_confuse_commanded_and_measured_success():
    from dexvision.evaluation.reach_repair_review import measured_pose_evidence

    spec, inputs = load_repair_spec('configs/level5/reach_repair_v3.yaml')
    goal = [.1, .05, .16, 1., 0., 0., 0.]
    qpos = np.zeros((7, 7))
    qpos[:, :3] = np.asarray(goal[:3]) + spec['weld_relpose'][:3]
    qpos[:, 3:7] = spec['weld_relpose'][3:]
    names = list(feature_schema(inputs.plan, inputs.skill)[0])
    recipe = spec['models']['goal_plane_calibrated_hold']
    metric = inputs.skill['contract']['success_metric']
    passed = measured_pose_evidence(qpos, 0, goal, names, spec, recipe, metric)
    assert passed['measured_pose_and_dwell_passed']
    assert passed['final_measured_distance_m'] < 1e-12
    qpos[-1, 0] += .04
    failed = measured_pose_evidence(qpos, 0, goal, names, spec, recipe, metric)
    assert not failed['measured_pose_and_dwell_passed']
    assert failed['final_measured_pose_dwell'] == 0


def test_method_selection_binds_frozen_confirmation_and_rejects_test_evidence(tmp_path):
    import json

    from dexvision.evaluation.reach_repair import select_repair_method
    from dexvision.learning.train_skill import _checksum_index, _json
    from dexvision.learning.training_manifest import content_digest, digest_file

    config = 'configs/level5/reach_repair_v3.yaml'
    spec, _ = load_repair_spec(config)
    reset, screen = tmp_path / 'reset', tmp_path / 'screen'
    reset.mkdir()
    screen.mkdir()
    _json(reset / 'resets.json', {'repair_spec_sha256': content_digest(spec), 'purpose': 'synthetic binding guard'})
    _checksum_index(reset)
    report = {'repair_spec_sha256': content_digest(spec), 'split': 'test', 'screen_only': True,
              'seed': 5001, 'repair_arm': 'goal_plane_calibrated_hold', 'epoch': 100,
              'safety_violation_count': 0, 'success_rate': 1., 'worst_cell_success_rate': 1.,
              'normalized_terminal_error': .1, 'mean_action_jerk': .001, 'state_numerical_gates_passed': True}
    _json(screen / 'screen.json', {'reports': [report], 'best_epoch': 100})
    _checksum_index(screen)
    with pytest.raises(ValueError, match='selection evidence'):
        select_repair_method(config, [screen], tmp_path / 'rejected', reset_dir=reset)
    assert not (tmp_path / 'rejected').exists()
    report['split'] = 'validation'
    _json(screen / 'screen.json', {'reports': [report], 'best_epoch': 100})
    _checksum_index(screen)
    decision = select_repair_method(config, [screen], tmp_path / 'decision', reset_dir=reset)
    assert decision['reset_manifest_sha256'] == digest_file(reset / 'resets.json')
    assert not decision['confirmation_used']
    saved = json.loads((tmp_path / 'decision/method_selection.json').read_text())
    assert saved['selected_arm'] == 'goal_plane_calibrated_hold'


def test_goal_plane_height_retains_clearance_state_when_relative_goal_is_same(frozen):
    _, inputs = frozen
    names = list(feature_schema(inputs.plan, inputs.skill)[0])
    values = np.zeros((2, len(names)))
    goal_z = names.index('goal.approach_position.2')
    prior_z = names.index('prior_applied_actions.base_position_target/z')
    values[:, goal_z] = [.14, .15]
    values[:, prior_z] = [.21, .22]
    values[:, 2] = [.20, .21]
    values[:, names.index('goal.approach_position.0')] = .12
    relative, _ = transform_features(values, names, 'goal_plane')
    np.testing.assert_allclose(relative[0], relative[1])
    with_height, output_names = transform_features(values, names, 'goal_plane_height')
    np.testing.assert_allclose(with_height[:, :4], relative)
    np.testing.assert_allclose(with_height[:, 4], [.21, .22])
    assert output_names[-1] == 'goal_plane/height'
