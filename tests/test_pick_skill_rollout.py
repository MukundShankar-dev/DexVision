"""Pick entry isolation, physical evidence, unsafe failures and complete matrices."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from dexvision.evaluation.evaluate_skill import reach_features
from dexvision.evaluation.pick_policy import (
    PickEvidence, create_pick_task, pick_preconditions, pick_scenarios, read_pick_entries, run_pick_rollout,
)
from dexvision.learning.pick_policy import boundary_context, load_pick_spec, pick_features
from dexvision.learning.skill_datasets import feature_schema, vectorize_episode
from dexvision.learning.training_manifest import ReleaseSource


@pytest.fixture(scope='module')
def fixture():
    spec, inputs = load_pick_spec('configs/level5/pick_policy_v2.yaml')
    scenario = pick_scenarios(inputs, 'validation')[0]
    row = scenario[0]
    root = inputs.root if (inputs.root / 'data/demos/level4' / row['source_path']).is_dir() else None
    entry = read_pick_entries(inputs, [scenario], root)[row['episode_id']]
    return spec, inputs, scenario, entry, root


def test_exact_prefix_and_offline_live_feature_parity(fixture):
    spec, inputs, (row, _, start, _), (metadata, prefix), root = fixture
    assert len(prefix) == start
    record = ReleaseSource(inputs, root).read_episode_records([row])[row['episode_id']]
    offline = vectorize_episode(inputs, row, record)
    with create_pick_task(inputs, row, metadata, prefix) as task:
        world = task.workcell.get_world_state()
        live = reach_features(task, world, row, inputs, prefix[-1], prefix[-1], np.zeros(27), task._phase)
        names = list(feature_schema(inputs.plan, inputs.skill)[0])
        a = pick_features(live, names, boundary_context(live, names), spec)[0]
        b = pick_features(offline.inputs[0], names, boundary_context(offline.inputs[0], names), spec)[0]
        np.testing.assert_allclose(a, b, atol=1e-7, rtol=0)
        assert task._pick_task._dwell_steps == 0
        assert all(pick_preconditions(world, row, spec).values())


def changed_world(world, object_id, *, height=.06, held=True, speed=0., supported=False):
    entity = world.require_entity(object_id)
    entity = replace(entity, position=(*entity.position[:2], height), linear_velocity=(0., 0., speed))
    relation = replace(world.relation_for(object_id), held_by='rh_palm' if held else None,
                       supported_by='table' if supported else None)
    return replace(world, entities=tuple(entity if e.object_id == object_id else e for e in world.entities),
                   relations=tuple(relation if r.object_id == object_id else r for r in world.relations),
                   contacts=((object_id, 'rh_ffdistal'), (object_id, 'rh_mfdistal')) if held else ())


def test_physical_success_needs_contacts_height_velocity_and_consecutive_dwell(fixture):
    spec, inputs, (row, _, _, _), (metadata, prefix), _ = fixture
    with create_pick_task(inputs, row, metadata, prefix) as task:
        world = task.workcell.get_world_state()
        evidence = PickEvidence(row['entity_id'], 0., spec['terminal'])
        assert not evidence.update(changed_world(world, row['entity_id'], held=False))['success']
        assert not evidence.update(changed_world(world, row['entity_id'], speed=.1))['success']
        assert not evidence.update(changed_world(world, row['entity_id'], supported=True))['success']
        good = changed_world(world, row['entity_id'])
        for _ in range(9):
            assert not evidence.update(good)['success']
        assert evidence.update(good)['success']
        evidence.update(changed_world(world, row['entity_id'], held=False))
        assert evidence.stable_dwell == 0
        for _ in range(5):
            result = evidence.update(changed_world(world, row['entity_id'], held=False))
        assert result['slip_drop']


def test_unsafe_policy_is_retained_and_cannot_be_rescued_by_clipping(fixture, tmp_path):
    spec, inputs, (row, _, _, condition), entry, _ = fixture
    class Unsafe:
        def reset(self, features):
            pass
        def __call__(self, features, prior, phase):
            raw = prior.copy()
            raw[0] = 100.
            return raw, np.zeros(26)
    result = run_pick_rollout(inputs, spec, row, *entry, Unsafe(), condition=condition, output_dir=tmp_path / 'attempt')
    assert result['terminal_reason'] == 'workspace_violation'
    assert result['safety_counters']['workspace_violations'] > 0
    assert result['terminal_result'] == 'failed'
    assert (tmp_path / 'attempt' / 'trajectory.npz').is_file()


def test_frozen_matrices_and_held_entry_rejection(fixture):
    spec, inputs, (row, _, _, _), entry, _ = fixture
    assert len(pick_scenarios(inputs, 'validation')) == 120
    assert len(pick_scenarios(inputs, 'test')) == 216
    with create_pick_task(inputs, row, *entry) as task:
        world = changed_world(task.workcell.get_world_state(), row['entity_id'], held=True, supported=True)
        assert not pick_preconditions(world, row, spec)['unheld']
    with pytest.raises(ValueError):
        pick_scenarios(inputs, 'train')


def test_rejected_attempt_keeps_denominator_without_policy_actions(fixture, tmp_path, monkeypatch):
    import dexvision.evaluation.pick_policy as evaluation
    spec, inputs, (row, _, _, condition), entry, _ = fixture
    monkeypatch.setattr(evaluation, 'pick_preconditions', lambda *args: {'unheld': False})
    class NeverAct:
        def reset(self, features):
            pass
        def __call__(self, *args):
            raise AssertionError('Rejected entry must never execute policy')
    result = run_pick_rollout(inputs, spec, row, *entry, NeverAct(), condition=condition, output_dir=tmp_path / 'rejected')
    assert result['terminal_result'] == 'rejected_before_execution'
    assert result['normalized_terminal_error'] is None
    with np.load(tmp_path / 'rejected' / 'trajectory.npz', allow_pickle=False) as data:
        assert len(data['qpos']) == 1


def test_single_test_reservation_checked_before_reading_release(fixture, tmp_path, monkeypatch):
    import json
    import torch
    import dexvision.evaluation.pick_policy as evaluation
    spec, inputs, *_ = fixture
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'validation_candidates').mkdir()
    source = run / 'validation_candidates/epoch_0020.pt'
    source.write_bytes(b'candidate')
    (run / 'selection.json').write_text(json.dumps({'status': 'selected_from_validation', 'epoch': 20,
        'source_checkpoint_sha256': evaluation.digest_file(source)}))
    (run / 'test_state_attempt.json').write_text('{}')
    checkpoint = {'manifest': {}, 'model': {'weight': torch.zeros(1)}}
    monkeypatch.setattr(evaluation, 'load_pick_spec', lambda _: (spec, inputs))
    monkeypatch.setattr(evaluation, 'load_checkpoint', lambda _: checkpoint)
    monkeypatch.setattr(evaluation, 'PickPolicy', lambda *args: object())
    monkeypatch.setattr(evaluation, '_verify_run', lambda _: None)
    def forbidden(*args):
        raise AssertionError('Test arrays cannot be read after reservation')
    monkeypatch.setattr(evaluation, 'read_pick_entries', forbidden)
    with pytest.raises(ValueError, match='already reserved'):
        evaluation.evaluate_pick('unused', run / 'selected.pt', tmp_path / 'repeat')
    assert not (tmp_path / 'repeat').exists()


def test_pick_execution_never_applies_source_orientation_hold(fixture, tmp_path, monkeypatch):
    from dexvision.sim.workcell import Workcell
    spec, inputs, (row, _, _, condition), entry, _ = fixture
    def forbidden(*args, **kwargs):
        raise AssertionError('Learned policy physics must not overwrite object rotation')
    monkeypatch.setattr(Workcell, 'preserve_object_orientation', forbidden)
    class Stop:
        def reset(self, _):
            pass
        def __call__(self, _, prior, phase):
            raw = prior.copy()
            raw[0] = 100
            return raw, np.zeros(26)
    result = run_pick_rollout(inputs, spec, row, *entry, Stop(), condition=condition, output_dir=tmp_path / 'ordinary')
    assert result['terminal_reason'] == 'workspace_violation'


def test_repair_confirmation_requires_all_seeds_and_never_reserves_failed_gate(tmp_path, monkeypatch):
    import dexvision.evaluation.pick_repair as repair
    spec, inputs = load_pick_spec('configs/level5/pick_policy_v3.yaml')
    monkeypatch.setattr(repair, 'read_confirmation', lambda *args: {'input_digests': inputs.digests})
    def failed(*args):
        raise ValueError('Every pick seed must pass full validation before confirmation')
    monkeypatch.setattr(repair, 'verify_pick_selection', failed)
    run = tmp_path / 'seed_5001'
    run.mkdir()
    with pytest.raises(ValueError, match='Every pick seed'):
        repair.confirm_pick('configs/level5/pick_policy_v3.yaml', run, tmp_path / 'resets', tmp_path / 'confirm')
    assert not (run / 'fresh_confirmation_attempt.json').exists()
    assert not (tmp_path / 'confirm').exists()


def test_saved_physics_audit_rejects_success_with_stationary_supported_object(fixture):
    from dexvision.evaluation.pick_repair import physical_pick_audit
    spec, inputs, (row, _, _, _), (metadata, prefix), _ = fixture
    with create_pick_task(inputs, row, metadata, prefix) as task:
        values = {'qpos': np.repeat(task.env.data.qpos[None], 12, axis=0),
                  'qvel': np.repeat(task.env.data.qvel[None], 12, axis=0),
                  'applied_actions': np.repeat(prefix[-1][None], 11, axis=0),
                  'state_timestamps': np.arange(12) * inputs.plan['action']['control_interval_s']}
        result = {'object_instance': row['entity_id'], 'terminal_result': 'succeeded'}
        audit = physical_pick_audit(task, values, result, spec)
        assert not audit['passed'] and not audit['physical_success']


def test_conditioned_setup_replans_pose_and_never_closes_or_holds_object(fixture, monkeypatch):
    from dexvision.evaluation.pick_repair import conditioned_pick_prefix
    from dexvision.sim.level4_expert import GraspFamilyTemplate, _conditioned_grasp_orientation
    from dexvision.sim.workcell import Workcell
    spec, inputs, (row, _, _, _), (metadata, source_prefix), _ = fixture
    def forbidden(*args, **kwargs):
        raise AssertionError('Setup must not overwrite object orientation')
    monkeypatch.setattr(Workcell, 'preserve_object_orientation', forbidden)
    with create_pick_task(inputs, row, metadata, np.empty((0, 27))) as task:
        prefix, audit = conditioned_pick_prefix(task, metadata, source_prefix, inputs)
        template = GraspFamilyTemplate.from_mapping(metadata['teleop_config']['scripted_expert']['grasp']['family_templates']['cuboid'])
        entity = task.initial_world_state.require_entity(row['entity_id'])
        np.testing.assert_allclose(prefix[-1, :3], np.asarray(entity.position) + template.object_relative_position_m, atol=1e-10)
        np.testing.assert_allclose(prefix[-1, 3:7], _conditioned_grasp_orientation(template, np.asarray(entity.orientation_wxyz)), atol=1e-10)
        np.testing.assert_allclose(prefix[:, 7:], np.broadcast_to(metadata['initial_applied_action'][7:], prefix[:, 7:].shape))
        assert audit['post_boundary_actions_read'] == audit['object_pose_writes'] == 0


def test_conditioned_setup_explicitly_recognizes_historical_yaw_rule(fixture):
    import copy
    from dexvision.evaluation.pick_repair import conditioned_pick_prefix
    spec, inputs, (row, _, _, _), (metadata, source_prefix), _ = fixture
    metadata = copy.deepcopy(metadata)
    obj = metadata['task_config']['initial_state']['objects'][row['entity_id']]
    obj['orientation_wxyz'][3] = abs(obj['orientation_wxyz'][3])
    source_prefix = source_prefix.copy()
    source_prefix[-1, 3:7] = metadata['teleop_config']['scripted_expert']['grasp']['family_templates']['cuboid']['wrist_orientation_wxyz']
    source_prefix[-1, 3:7] /= np.linalg.norm(source_prefix[-1, 3:7])
    with create_pick_task(inputs, row, metadata, np.empty((0, 27))) as task:
        with pytest.raises(ValueError, match='declared grasp template'):
            conditioned_pick_prefix(task, metadata, source_prefix, inputs)
        _, audit = conditioned_pick_prefix(task, metadata, source_prefix, inputs, allow_legacy_yaw=True)
        assert audit['source_orientation_rule'] == 'historical_negative_yaw_only'


def test_setup_protocol_locks_all_selected_weights(tmp_path):
    import yaml
    from dexvision.evaluation.pick_setup import load_setup_spec
    from dexvision.learning.training_manifest import digest_file
    from pathlib import Path
    protocol = yaml.safe_load(Path('configs/level5/pick_setup_v2.yaml').read_text())
    for seed in protocol['seeds']:
        p = tmp_path / f'{seed}.pt'
        p.write_bytes(str(seed).encode())
        protocol['selected_weights'][str(seed)] = {'path': str(p), 'sha256': digest_file(p)}
    p = tmp_path / 'protocol.yaml'
    p.write_text(yaml.safe_dump(protocol))
    p.with_suffix('.sha256').write_text(f'{digest_file(p)}  {p.name}\n')
    assert load_setup_spec(p)[0] == protocol
    (tmp_path / '5002.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='cannot change or reselect'):
        load_setup_spec(p)


def test_collection_expert_setup_stops_before_acquire_and_matches_source(fixture, monkeypatch):
    from dexvision.evaluation.pick_repair import expert_pick_prefix
    spec, inputs = load_pick_spec('configs/level5/pick_policy_v4.yaml')
    _, _, (row, _, _, _), (metadata, source_prefix), _ = fixture
    with create_pick_task(inputs, row, metadata, np.empty((0, 27))) as task:
        def forbidden(*args, **kwargs):
            raise AssertionError('Live setup cannot overwrite object rotation')
        monkeypatch.setattr(task.workcell, 'preserve_object_orientation', forbidden)
        prefix, audit = expert_pick_prefix(task, spec, inputs)
        assert not audit['setup_failed']
        assert audit['live_acquire_actions_queried'] == audit['live_acquire_actions_applied'] == 0
        assert audit['live_object_pose_writes'] == 0
        np.testing.assert_allclose(prefix, source_prefix, atol=1e-10)
        np.testing.assert_allclose(prefix[:, 7:], np.broadcast_to(spec['closure']['open_targets'], prefix[:, 7:].shape), atol=1e-12)


@pytest.mark.parametrize('failure', [None, 'confirmation', 'audit'])
def test_setup_review_requires_all_seed_confirmations_and_audits(tmp_path, monkeypatch, failure):
    import json
    import dexvision.evaluation.pick_setup_review as review
    from dexvision.learning.training_manifest import digest_file
    protocol = {'seeds': [5001, 5002, 5003], 'output_dir': str(tmp_path)}
    monkeypatch.setattr(review, 'load_setup_spec', lambda _: (protocol, {}, object()))
    monkeypatch.setattr(review, '_load_resets', lambda *args: {})
    monkeypatch.setattr(review, '_verify_run', lambda *args: None)
    monkeypatch.setattr(review, '_verify_report', lambda report, *args: {
        'state_numerical_gates_passed': report['passed'], 'rollout_count': 1})
    for seed in protocol['seeds']:
        directory = tmp_path / f'confirmation_{seed}'
        directory.mkdir()
        (directory / 'report.json').write_text(json.dumps({'seed': seed, 'checkpoint_sha256': str(seed),
            'passed': not (failure == 'confirmation' and seed == 5003)}))
        audit_dir = tmp_path / f'audit_{seed}'
        audit_dir.mkdir()
        (audit_dir / 'audit.json').write_text(json.dumps({'passed': True, 'audits': [{'passed': True}],
            'attempts': 1, 'checkpoint_sha256': str(seed),
            'report_sha256': 'wrong' if failure == 'audit' and seed == 5003 else digest_file(directory / 'report.json')}))
    if failure:
        with pytest.raises(ValueError, match='Every pick seed|matching passing physical audits'):
            review.review_inputs('unused', tmp_path / 'resets', 5001)
    else:
        assert review.review_inputs('unused', tmp_path / 'resets', 5001)[3]['seed'] == 5001


@pytest.mark.parametrize('failure', [None, 'confirmation', 'audit', 'coverage'])
def test_repair_review_requires_all_seed_physical_evidence(tmp_path, monkeypatch, failure):
    import json
    import dexvision.evaluation.pick_repair as repair
    from dexvision.learning.training_manifest import digest_file
    spec = {'seeds': [5001, 5002, 5003], 'output_dir': str(tmp_path)}
    reports = {}
    for seed in spec['seeds']:
        directory = tmp_path / f'confirmation_{seed}'
        directory.mkdir()
        result = {'episode_id': 'one', 'render_condition': 'nominal'}
        report = {'seed': seed, 'checkpoint_sha256': str(seed), 'rollout_count': 1, 'results': [result]}
        reports[directory] = report
        (directory / 'report.json').write_text(json.dumps(report))
        audit_dir = tmp_path / f'audit_{seed}'
        audit_dir.mkdir()
        audited = {**result, 'passed': True}
        if failure == 'coverage' and seed == 5003:
            audited['episode_id'] = 'another'
        (audit_dir / 'audit.json').write_text(json.dumps({'passed': True, 'audits': [audited],
            'attempt_count': 1, 'checkpoint_sha256': str(seed),
            'confirmation_sha256': 'wrong' if failure == 'audit' and seed == 5003 else digest_file(directory / 'report.json')}))
    def load(config, run, reset, directory):
        if failure == 'confirmation' and reports[directory]['seed'] == 5003:
            raise ValueError('Saved pick review requires passing confirmation')
        return spec, {}, {}, reports[directory]
    monkeypatch.setattr(repair, '_load_saved_confirmation', load)
    monkeypatch.setattr(repair, '_verify_run', lambda *args: None)
    args = ('unused', tmp_path / 'seed_5001', tmp_path / 'resets', tmp_path / 'confirmation_5001')
    if failure:
        with pytest.raises(ValueError, match='passing confirmation|matching passing physical audits'):
            repair.verify_pick_review(*args)
    else:
        assert repair.verify_pick_review(*args)[3]['seed'] == 5001
