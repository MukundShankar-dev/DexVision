"""Explicit additive-release adapter; original Level 4 ownership remains historical."""
from __future__ import annotations

import copy
import io
import json
import tarfile
from dataclasses import replace

import numpy as np

from dexvision.learning.skill_datasets import SkillDataset, SkillEpisode, feature_schema
from dexvision.learning.training_manifest import content_digest, digest_file, safe_path


def amendment_manifest(inputs, spec):
    pin = spec['unassisted_dataset']
    path = safe_path(inputs.root, pin['manifest'])
    if digest_file(path) != pin['manifest_sha256']:
        raise ValueError('Unassisted pick manifest checksum mismatch')
    manifest = json.loads(path.read_text())
    if (manifest['version'] != pin.get('version', 'level5/pick-unassisted-data-v1')
            or manifest['archive_sha256'] != pin['archive_sha256']):
        raise ValueError('Unassisted pick release binding mismatch')
    return manifest


def amended_inputs(inputs, spec):
    """Replace only V5 training/validation membership with separately declared ownership."""
    manifest = amendment_manifest(inputs, spec)
    splits, skill, snapshot = copy.deepcopy(inputs.splits), copy.deepcopy(inputs.skill), copy.deepcopy(inputs.config_snapshot)
    cells = {c['id']: c for c in inputs.requirements['coverage_cells']}
    episode_owners, session_owners, seed_owners = {}, {}, {}
    for split in ('train', 'validation'):
        rows = []
        for result in manifest['reports'][split]['results']:
            eid, session, seed = result['episode_id'], result['session_id'], result['seed']
            if eid in episode_owners or seed in seed_owners or session_owners.setdefault(session, split) != split:
                raise ValueError('Additive release has duplicate or cross-split owners')
            episode_owners[eid], seed_owners[seed] = split, split
            cell = cells[result['cell']]
            prefix = f'datasets/{manifest["plan"]["release_name"]}/episodes/{eid}/'
            file_hashes = {name: manifest['files'][prefix + name]['sha256']
                           for name in ('metadata.json', 'result.json', 'trajectory.npz')}
            start, end = result['entry_frame'] or 0, result['action_count']
            rows.append({'episode_id': eid, 'split': split, 'recording_session_id': session,
                'random_seed': seed, 'cell_id': result['cell'], 'entity_id': result['object_id'],
                'typed_goal': {'object_id': cell['object_id'], 'target_id': cell['target_id']},
                'target_id': cell['target_id'], 'episode_digest': content_digest(file_hashes),
                'skill': skill['source_skill'], 'completed_skill': 'pick_object',
                'source_scope': 'sequence scene, standalone pick only; no place completion claim',
                'stream': 'expert_success' if result['success'] else 'failed_teacher',
                'frames': end, 'entry_frame': start, 'training_target_interval': [start, end],
                'segments': [{'skill_name': 'pick_object', 'start_frame': start, 'end_frame': end}],
                'phase_intervals': [{'phase': 'acquire', 'start_frame': start, 'end_frame': end}],
                'data_amendment': manifest['version'], 'payload_prefix': prefix, 'file_sha256': file_hashes})
        splits[split] = {'episodes': rows, 'images': [], 'normalization_inputs': [
            {'episode_id': r['episode_id'], 'episode_digest': r['episode_digest'],
             'frame_interval': r['training_target_interval']} for r in rows
            if split == 'train' and r['stream'] == 'expert_success']}
    conditions = inputs.config_snapshot['evaluation']['rollouts']['perception_conditions']
    matrix_cells = []
    for cell_id in manifest['plan']['splits']['validation']['cells']:
        rows = [r for r in splits['validation']['episodes'] if r['cell_id'] == cell_id]
        matrix_cells.append({'cell_id': cell_id, 'episode_ids': [r['episode_id'] for r in rows],
            'reset_seeds': [r['random_seed'] for r in rows], 'start_frames': [r['entry_frame'] for r in rows]})
    size = len(splits['validation']['episodes'])
    skill['evaluation']['validation'] = {'cells': matrix_cells, 'unique_reset_count': size,
        'rollouts_per_track_per_training_seed': size * len(conditions)}
    snapshot['unassisted_pick_amendment'] = spec['unassisted_dataset']
    digests = {**inputs.digests, 'pick_additive_manifest_sha256': spec['unassisted_dataset']['manifest_sha256'],
        'pick_additive_archive_sha256': manifest['archive_sha256']}
    return replace(inputs, splits=splits, skill=skill, digests=digests, config_snapshot=snapshot)


def read_amendment_payload(inputs, spec, rows):
    manifest = amendment_manifest(inputs, spec)
    archive = safe_path(inputs.root, manifest['archive'])
    if digest_file(archive) != manifest['archive_sha256']:
        raise ValueError('Unassisted archive missing/changed; retrieve the immutable LFS release')
    wanted = {r['payload_prefix'] + name for r in rows for name in r['file_sha256']}
    result, seen = {}, set()
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            safe_path(inputs.root, member.name)
            if not member.isfile() or member.name in seen or member.name not in manifest['files']:
                raise ValueError('Unsafe, duplicate or unlisted pick archive member')
            seen.add(member.name)
            if member.name in wanted:
                with tar.extractfile(member) as stream:
                    data = stream.read()
                import hashlib
                if hashlib.sha256(data).hexdigest() != manifest['files'][member.name]['sha256']:
                    raise ValueError('Pick payload checksum mismatch')
                result[member.name] = data
    if set(result) != wanted or seen != set(manifest['files']):
        raise ValueError('Incomplete pick archive')
    return result


def raw_amendment_datasets(inputs, spec):
    rows = [r for s in ('train', 'validation') for r in inputs.splits[s]['episodes'] if r['stream'] == 'expert_success']
    payload = read_amendment_payload(inputs, spec, rows)
    names, continuous = feature_schema(inputs.plan, inputs.skill)
    normalization = {'version': 'level5/pick-unassisted-fixed-scales-v1', 'ordered_feature_names': list(names),
        'mean': [0.] * len(names), 'std': [1.] * len(names), 'continuous_mask': continuous.tolist(),
        'training_episode_ids': [r['episode_id'] for r in rows if r['split'] == 'train'],
        'source': 'new immutable unassisted release; fixed physical scales, no validation fitting'}
    normalization['normalization_digest'] = content_digest(normalization)
    episodes = {'train': [], 'validation': []}
    for row in rows:
        start, end = row['training_target_interval']
        with np.load(io.BytesIO(payload[row['payload_prefix'] + 'trajectory.npz']), allow_pickle=False) as a:
            arrays = {k: a[k].copy() for k in a.files}
        if (arrays['observations'].shape != (row['frames'], len(names))
                or arrays['actions'].shape != (row['frames'], 27)
                or len(arrays['qpos']) != row['frames'] + 1
                or np.any(arrays['safety_masks'])):
            raise ValueError('Unassisted episode shapes/safety differ from frozen ownership')
        records = {'applied_actions': arrays['actions'][start:end],
            'prior_applied_actions': arrays['prior_applied_actions'][start:end], 'safety_masks': arrays['safety_masks'][start:end]}
        episodes[row['split']].append(SkillEpisode(row['episode_id'], row['split'], row['recording_session_id'],
            row['cell_id'], row['episode_digest'], arrays['observations'][start:end],
            np.zeros((end-start, 26)), np.ones((end-start, 26)), np.arange(start, end),
            arrays['timestamps'][start:end], ('acquire',) * (end-start),
            (start, end) if row['split'] == 'train' else None, records))
    return {split: SkillDataset(values, normalization) for split, values in episodes.items()}


def amendment_entries(inputs, spec, rows):
    payload = read_amendment_payload(inputs, spec, rows)
    entries = {}
    for row in rows:
        config = json.loads(payload[row['payload_prefix'] + 'metadata.json'])
        with np.load(io.BytesIO(payload[row['payload_prefix'] + 'trajectory.npz']), allow_pickle=False) as a:
            prefix = a['actions'][:row['entry_frame']].copy()
            initial_action = a['prior_applied_actions'][0].copy() if len(a.get('prior_applied_actions', [])) else None
        if initial_action is None:
            from dexvision.logging.level4_collection import WorkcellPilotTask
            with WorkcellPilotTask(workcell_config=inputs.root / inputs.plan['inputs']['workcell_config']['path'],
                    dataset_config=inputs.root / inputs.plan['inputs']['dataset_config']['path'], skill_name='pick_place_sequence',
                    goal_condition_id=row['cell_id'], seed=row['random_seed'],
                    procedural_variation=config['procedural_variation']) as task:
                world = task.initial_world_state
                initial_action = np.r_[world.robot.base_position, world.robot.base_orientation_wxyz, spec['closure']['open_targets']]
        metadata = {'episode_id': row['episode_id'], 'random_seed': row['random_seed'],
            'recording_session_id': row['recording_session_id'], 'typed_goal': config['parameters'],
            'task_config': config, 'initial_online_phase': 'approach', 'initial_applied_action': initial_action.tolist(),
            'source': 'unassisted_scripted_pick_amendment'}
        if metadata['typed_goal'] != row['typed_goal']:
            raise ValueError('Unassisted goal/source mismatch')
        entries[row['episode_id']] = metadata, prefix
    return entries


def audit_amendment(config, output_dir):
    """Reconstruct released physical outcomes from saved states, without integration."""
    from pathlib import Path
    from dexvision.evaluation.pick_policy import create_pick_task
    from dexvision.evaluation.pick_repair import physical_pick_audit
    from dexvision.learning.pick_policy import load_pick_spec
    from dexvision.learning.train_skill import _checksum_index, _json
    spec, inputs = load_pick_spec(config)
    rows = [r for s in ('train', 'validation') for r in inputs.splits[s]['episodes']]
    payload = read_amendment_payload(inputs, spec, rows)
    entries = amendment_entries(inputs, spec, rows)
    audits = []
    for row in rows:
        metadata, prefix = entries[row['episode_id']]
        with np.load(io.BytesIO(payload[row['payload_prefix'] + 'trajectory.npz']), allow_pickle=False) as a:
            start = row['entry_frame']
            values = {'qpos': a['qpos'][start:], 'qvel': a['qvel'][start:],
                'applied_actions': a['actions'][start:], 'state_timestamps': a['timestamps'][start:]}
            with create_pick_task(inputs, row, metadata, prefix) as task:
                audit = physical_pick_audit(task, values, {'object_instance': row['entity_id'],
                    'terminal_result': 'succeeded' if row['stream'] == 'expert_success' else 'failed'}, spec)
        audits.append({'episode_id': row['episode_id'], 'split': row['split'], **audit})
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    report = {'passed': all(a['passed'] for a in audits), 'episodes': len(audits), 'audits': audits,
        'release': spec['unassisted_dataset'], 'method': 'saved qpos/qvel contact, lift, retention and dwell reconstruction'}
    _json(directory / 'audit.json', report)
    _checksum_index(directory)
    return {k: v for k, v in report.items() if k != 'audits'}


def verify_amendment_audit(inputs, spec):
    """Block training when any source claim disagrees with physical reconstruction."""
    from dexvision.learning.train_skill import _verify_run
    directory = safe_path(inputs.root, spec['source_physical_audit'])
    _verify_run(directory)
    report = json.loads((directory / 'audit.json').read_text())
    expected = {(r['episode_id'], split) for split in ('train', 'validation')
                for r in inputs.splits[split]['episodes']}
    actual = {(a['episode_id'], a['split']) for a in report['audits']}
    if (not report['passed'] or not all(a['passed'] for a in report['audits'])
            or report['release'] != spec['unassisted_dataset']
            or report['episodes'] != len(expected) or len(report['audits']) != len(expected)
            or actual != expected):
        raise ValueError('Training requires a complete passing physical source audit for this release')
