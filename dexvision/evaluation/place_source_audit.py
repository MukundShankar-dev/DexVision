"""Independent saved-state audit of additive 5.4A place supervision."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dexvision.evaluation.evaluate_skill import apply_action
from dexvision.evaluation.pick_policy import physical_evidence_world
from dexvision.evaluation.place_policy import PlaceEvidence
from dexvision.learning.skill_datasets import feature_schema
from dexvision.learning.train_skill import _checksum_index, _json, _verify_run
from dexvision.learning.training_manifest import content_digest, digest_file
from dexvision.logging.level4_collection import WorkcellPilotTask


def audit_place_payload(place, payload_dir):
    """Recompute every source place state from saved qpos/qvel, without stepping."""
    spec, inputs = place
    payload_dir = Path(payload_dir)
    _verify_run(payload_dir)
    metadata = json.loads((payload_dir / 'metadata.json').read_text())
    result = json.loads((payload_dir / 'result.json').read_text())
    object_id = metadata['typed_goal']['object_id']
    target_id = metadata['typed_goal']['target_id']
    names, _ = feature_schema(inputs.plan, inputs.skill)
    with np.load(payload_dir / 'trajectory.npz', allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    count = len(arrays['actions'])
    if (count <= 0 or arrays['actions'].shape != (count, 27)
            or arrays['observations'].shape != (count, len(names))
            or arrays['prior_applied_actions'].shape != (count, 27)
            or arrays['safety_masks'].shape != (count, 27)
            or len(arrays['qpos']) != count + 1 or len(arrays['qvel']) != count + 1
            or len(arrays['timestamps']) != count + 1
            or np.any(arrays['safety_masks'])
            or not np.isfinite(np.r_[arrays['observations'].ravel(), arrays['actions'].ravel()]).all()
            or not np.all(np.diff(arrays['timestamps']) > 0)):
        raise ValueError('Invalid saved place payload shape, timing or safety mask')
    if not np.array_equal(arrays['actions'][:-1], arrays['prior_applied_actions'][1:]):
        raise ValueError('Place payload prior/action alignment mismatch')
    opening = np.flatnonzero(np.linalg.norm(
        arrays['actions'][:, 7:] - arrays['prior_applied_actions'][:, 7:], axis=1) > 1e-8)
    if not len(opening):
        raise ValueError('Successful place payload never commands finger opening')
    release_command_index = int(opening[0])
    if result['place_actions'] != count or result['terminal_result'] != 'succeeded':
        raise ValueError('Place payload receipt differs from saved trajectory')
    board = inputs.requirements['workcell']['board_workspace']
    off_board, pre_release_hand_table, post_release_hand_table = 0, 0, 0
    before_release_command_hand_table, release_window_hand_table = 0, 0
    with WorkcellPilotTask(
            workcell_config=inputs.root / inputs.plan['inputs']['workcell_config']['path'],
            dataset_config=inputs.root / inputs.plan['inputs']['dataset_config']['path'],
            skill_name='pick_place_sequence', goal_condition_id=metadata['cell_id'],
            seed=metadata['random_seed'], procedural_variation=metadata['procedural_variation']) as task:

        def physical(index):
            command = (arrays['prior_applied_actions'][0] if index == 0
                       else arrays['actions'][index - 1])
            apply_action(task, command, inputs.plan['action']['named_layout'])
            task.env.data.qpos[:] = arrays['qpos'][index]
            task.env.data.qvel[:] = arrays['qvel'][index]
            task.env.data.time = arrays['timestamps'][index]
            task.env._mujoco.mj_forward(task.env.model, task.env.data)
            return physical_evidence_world(task)

        initial = physical(0)
        relation = initial.relation_for(object_id)
        if relation.held_by != 'rh_palm' or relation.supported_by is not None:
            raise ValueError('Saved place entry is not physically held')
        evidence = PlaceEvidence(object_id, target_id, spec['terminal'],
            initial.require_entity(object_id).position)
        for index in range(1, count + 1):
            world = physical(index)
            metrics = evidence.update(world, SimpleNamespace(values={}, success=False))
            xy = np.asarray(world.require_entity(object_id).position[:2])
            off_board += int(np.any(xy < np.asarray(board['min_xy_m']) + board['safe_edge_margin_m'])
                or np.any(xy > np.asarray(board['max_xy_m']) - board['safe_edge_margin_m']))
            hand_table = int(any('workcell_table' in pair and
                any(name.startswith('rh_') for name in pair) for pair in world.contacts))
            if evidence.released:
                post_release_hand_table += hand_table
            else:
                pre_release_hand_table += hand_table
            if index < release_command_index:
                before_release_command_hand_table += hand_table
            else:
                release_window_hand_table += hand_table
        final = world.require_entity(object_id)
        final_target = world.require_entity(target_id)
        same_pose = (np.allclose(final.position, result['physical_final_context']['object_position_m'],
            atol=1e-12, rtol=0) and np.allclose(final_target.position,
            result['physical_final_context']['target_position_m'], atol=1e-12, rtol=0))
    passed = bool(metrics['success'] and not evidence.premature_release
        and not evidence.bounced and off_board == 0
        and before_release_command_hand_table == 0 and same_pose)
    return {'episode_id': metadata['episode_id'], 'passed': passed, 'frames': count,
        'transported': evidence.transported, 'settle_steps': evidence.settle_dwell,
        'premature_release': evidence.premature_release, 'bounced': evidence.bounced,
        'object_workspace_samples': off_board,
        'pre_release_hand_table_samples': pre_release_hand_table,
        'post_release_hand_table_samples': post_release_hand_table,
        'before_release_command_hand_table_samples': before_release_command_hand_table,
        'release_window_hand_table_samples': release_window_hand_table,
        'final_pose_matches_source': same_pose,
        'payload_sha256': digest_file(payload_dir / 'trajectory.npz')}


def audit_place_data(config, output_dir):
    """Audit every successful captured episode before immutable publication."""
    from dexvision.logging.place_amendment import _frozen

    plan, _, _, place, _, frozen = _frozen(config, output_dir)
    root = Path(output_dir)
    directory = root / 'audit'
    records, excluded = [], []
    for split in ('train', 'validation'):
        stage = root / split
        _verify_run(stage)
        report = json.loads((stage / 'report.json').read_text())
        expected = {item['episode_id'] for item in frozen['assignments'][split]}
        if (report['plan_digest'] != content_digest(plan)
                or len(report['results']) != len(expected)
                or {item['episode_id'] for item in report['results']} != expected):
            raise ValueError('Place source audit requires complete frozen split reports')
        for item in report['results']:
            payload = stage / item['episode_id'] / 'payload'
            if item['payload']:
                audit = audit_place_payload(place, payload)
                if audit['episode_id'] != item['episode_id']:
                    raise ValueError('Place source audit episode ownership mismatch')
                records.append({'split': split, **audit})
            else:
                if payload.exists():
                    raise ValueError('Failed place attempt has a training payload')
                excluded.append({'split': split, 'episode_id': item['episode_id'],
                                 'reason': item['terminal_reason']})
    report = {'version': 'level5/place-source-physical-audit-v1',
        'plan_digest': content_digest(plan), 'passed': all(item['passed'] for item in records),
        'audited_successes': len(records), 'excluded_failures': excluded,
        'audits': records, 'method': 'Saved qpos/qvel MuJoCo forward and physical place terminal reconstruction'}
    directory.mkdir(parents=True, exist_ok=False)
    _json(directory / 'report.json', report)
    _checksum_index(directory)
    return {key: value for key, value in report.items() if key not in {'audits', 'excluded_failures'}}
