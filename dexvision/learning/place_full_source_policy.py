"""V19 place learning on the audited, all-cell V3 source."""
from __future__ import annotations

import hashlib
import io
import json
import tarfile
import copy
from pathlib import Path

import numpy as np
import yaml

from dexvision.learning.place_geometric_release_policy import (
    GeometricReleasePlacePolicy, load_geometric_release_spec,
)
from dexvision.learning.place_policy import (
    PlaceDataset, place_action_plan, place_features,
    place_schema, place_settings,
)
from dexvision.learning.skill_datasets import SkillEpisode, action_targets, feature_schema
from dexvision.learning.skill_models import SkillMLP
from dexvision.learning.train_skill import load_checkpoint, train_skill
from dexvision.learning.training_manifest import content_digest, digest_file

VERSION = 'level5/place-policy-full-source-v19'
SOURCE_VERSION = 'level5/place-unassisted-data-v3'
AUDITED_SUCCESSES = 354
DECODER_FIELDS = (
    'geometry', 'terminal', 'workspace_bounds', 'open_targets', 'closed_targets',
    'release_step_by_family', 'weld_relpose', 'position_scale_m',
    'velocity_scale_mps', 'motion_output_scales', 'release_gate_threshold',
    'alignment_stop_distance_m', 'alignment_step_m', 'setup_teacher_recipe',
    'setup_teacher_recipe_sha256',
)


def load_full_source_spec(config):
    path = Path(config)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError('Place V19 recipe checksum mismatch')
    spec = yaml.safe_load(path.read_text())
    if spec.get('version') != VERSION or spec.get('checkpoint') != '5.4A':
        raise ValueError('Unsupported place V19 recipe')
    parent = Path(spec['parent_policy_recipe'])
    if digest_file(parent) != spec['parent_policy_recipe_sha256']:
        raise ValueError('Place V19 decoder parent changed')
    _, old, inputs = load_geometric_release_spec(parent)
    for name in DECODER_FIELDS:
        if spec[name] != old[name]:
            raise ValueError(f'Place V19 decoder field changed: {name}')
    if (spec['seeds'] != old['seeds'] or spec['max_epochs'] != old['max_epochs']
            or spec['candidate_epochs'] != old['candidate_epochs']
            or spec['validation_seed_start'] != 1330260900
            or spec['validation_resets_per_cell'] != 2
            or spec['selection_seed_start'] != 1340260900
            or spec['confirmation_seed_start'] != 1360260900):
        raise ValueError('Place V19 budget or fresh reset boundary changed')
    for name, checksum in spec['implementation_sha256'].items():
        if digest_file(inputs.root / name) != checksum:
            raise ValueError(f'Place V19 implementation changed: {name}')
    for name in ('manifest', 'archive', 'source_physical_audit'):
        if digest_file(Path(spec['additive_data'][name])) != spec['additive_data'][name + '_sha256']:
            raise ValueError(f'Place V19 source changed: {name}')
    if digest_file(Path(spec['setup_teacher_recipe'])) != spec['setup_teacher_recipe_sha256']:
        raise ValueError('Place V19 setup teacher changed')
    for seed in spec['seeds']:
        entry = spec['decoder_bootstrap_checkpoints'][str(seed)]
        if digest_file(Path(entry['path'])) != entry['sha256']:
            raise ValueError(f'Place V19 decoder bootstrap changed for seed {seed}')
    return spec, inputs


def _released_payloads(spec):
    """Verify every V3 archive member and its independent physical audit."""
    pin = spec['additive_data']
    manifest = json.loads(Path(pin['manifest']).read_text())
    if (manifest['archive_sha256'] != pin['archive_sha256']
            or manifest['version'] != SOURCE_VERSION
            or not manifest['physical_audit']['passed']
            or manifest['physical_audit_sha256'] != pin['source_physical_audit_sha256']):
        raise ValueError('Place V19 source release binding mismatch')
    audit = json.loads(Path(pin['source_physical_audit']).read_text())
    if (audit != manifest['physical_audit']
            or audit['audited_successes'] != AUDITED_SUCCESSES
            or any(not row['passed'] for row in audit['audits'])):
        raise ValueError('Place V19 source physical audit is incomplete')
    wanted = set(manifest['files'])
    payload = {}
    with tarfile.open(pin['archive'], 'r:gz') as archive:
        for member in archive:
            path = Path(member.name)
            if (not member.isfile() or member.name not in wanted
                    or member.name in payload or '..' in path.parts or path.is_absolute()):
                raise ValueError('Unsafe, duplicate or unlisted place V19 member')
            with archive.extractfile(member) as stream:
                blob = stream.read()
            if hashlib.sha256(blob).hexdigest() != manifest['files'][member.name]['sha256']:
                raise ValueError('Place V19 release member checksum mismatch')
            payload[member.name] = blob
    if set(payload) != wanted:
        raise ValueError('Place V19 release is incomplete')
    return manifest, payload


def full_source_datasets(spec, inputs):
    """Read successful V3 place intervals with causal, phase-balanced labels."""
    manifest, payload = _released_payloads(spec)
    source_names = list(feature_schema(inputs.plan, inputs.skill)[0])
    plan = place_action_plan(spec, inputs)
    episodes = {'train': [], 'validation': []}
    owners = set()
    feature_names = None
    for split in episodes:
        for row in manifest['reports'][split]['results']:
            if not row['payload']:
                continue
            eid, session = row['episode_id'], row['session_id']
            if eid in owners:
                raise ValueError('Duplicate place V19 episode owner')
            owners.add(eid)
            prefix = f"datasets/{manifest['plan']['release_name']}/episodes/{eid}/"
            meta = json.loads(payload[prefix + 'metadata.json'])
            result = json.loads(payload[prefix + 'result.json'])
            with np.load(io.BytesIO(payload[prefix + 'trajectory.npz']), allow_pickle=False) as data:
                arrays = {key: data[key].copy() for key in data.files}
            length = len(arrays['actions'])
            if (meta['episode_id'] != eid or meta['session_id'] != session
                    or result['terminal_result'] != 'succeeded'
                    or arrays['observations'].shape != (length, len(source_names))
                    or arrays['actions'].shape != (length, 27)
                    or np.any(arrays['safety_masks'])):
                raise ValueError('Invalid place V19 source alignment')
            x, names = place_features(arrays['observations'], source_names, spec)
            if feature_names is None:
                feature_names = names
            elif feature_names != names:
                raise ValueError('Place V19 feature schema changed between episodes')
            y, masks = action_targets(arrays['actions'], arrays['prior_applied_actions'],
                arrays['safety_masks'], ('transport',) * length, plan, 'baseline')
            release = arrays['expert_phases'] == 'latched_release'
            y[:, 8] = np.where(release, 1., -1.)
            masks[:, 6:] = 0.
            masks[:, 8] = 1.
            if np.any(np.abs(y[:, :6]) > 1 + 1e-5):
                raise ValueError('Place V19 source motion exceeds bounded action head')
            # Explicit move and release scores; source phase is only a training label.
            moving = np.linalg.norm(y[:, :3], axis=1) > 1e-8
            y[:, 0] = np.where(moving, 1., -1.)
            masks[:, :] = 0.
            masks[:, 0] = 1.
            masks[:, 3:6] = 1.
            masks[:, 8] = 1.
            digest = content_digest({name: manifest['files'][prefix + name]['sha256']
                for name in ('metadata.json', 'result.json', 'trajectory.npz')})
            episodes[split].append(SkillEpisode(eid, split, session, row['cell'], digest,
                x, y, masks, np.arange(length), arrays['timestamps'][:-1],
                tuple(arrays['expert_phases'].tolist()),
                (0, length) if split == 'train' else None, {}))
    if {e.session_id for e in episodes['train']} & {e.session_id for e in episodes['validation']}:
        raise ValueError('Place V19 training/validation session leakage')
    norm = {'version': VERSION, 'ordered_feature_names': feature_names,
        'mean': [0.] * len(feature_names), 'std': [1.] * len(feature_names),
        'continuous_mask': [True] * len(feature_names),
        'training_episode_ids': [e.episode_id for e in episodes['train']],
        'rule': 'causal_phase_balanced_fixed_physical_scales',
        'archive_sha256': manifest['archive_sha256']}
    norm['normalization_digest'] = content_digest(norm)
    return {split: PlaceDataset(values, norm) for split, values in episodes.items()}


def full_source_validation_matrix(spec, inputs):
    """New all-cell validation resets, disjoint from both V3 source splits."""
    from dexvision.evaluation.place_repair_policy import repair_matrix

    amended = copy.deepcopy(spec)
    amended['confirmation']['seed_start'] = spec['validation_seed_start']
    amended['confirmation']['resets_per_cell'] = spec['validation_resets_per_cell']
    return repair_matrix(amended, inputs, 'confirmation')


def train_full_source(config, *, seed, output_dir=None, dry_run=False, resume=False):
    spec, inputs = load_full_source_spec(config)
    if seed not in spec['seeds']:
        raise ValueError('Seed outside frozen V19 recipe')
    datasets = full_source_datasets(spec, inputs)
    if dry_run:
        return {split: {'episodes': len(data.episodes), 'frames': len(data)}
                for split, data in datasets.items()}
    validation = full_source_validation_matrix(spec, inputs)
    conditions = inputs.config_snapshot['evaluation']['rollouts']['perception_conditions']
    manifest = {'version': VERSION, 'seed': seed, 'device': 'cpu', 'model': 'baseline',
        'recipe_sha256': digest_file(Path(config)), 'recipe': spec,
        'source_adapter': SOURCE_VERSION, 'additive_data': spec['additive_data'],
        'validation_matrix_sha256': content_digest(validation),
        'validation_rollout_count': len(validation) * len(conditions),
        'assignments': {split: [{'episode_id': e.episode_id, 'session_id': e.session_id,
            'episode_digest': e.episode_digest} for e in data.episodes]
            for split, data in datasets.items()},
        'implementation_sha256': {name: digest_file(inputs.root / name) for name in (
            'dexvision/learning/place_full_source_policy.py',
            'dexvision/learning/place_policy.py',
            'dexvision/learning/skill_models.py',
            'dexvision/learning/train_skill.py')},
    }
    manifest['manifest_digest'] = content_digest(manifest)
    schema = place_schema(spec, inputs)
    schema.update(observation_contract=VERSION, action_contract=VERSION)
    settings = place_settings(spec, inputs)
    settings['loss']['group_weights'] = spec['loss_group_weights']
    directory = Path(output_dir or f"{spec['output_dir']}/seed_{seed}")
    report = train_skill(datasets, schema=schema, settings=settings, manifest=manifest,
        output_dir=directory, config_snapshot={'place_full_source_recipe': spec}, resume=resume)
    return {key: value for key, value in report.items() if key != 'history'}


class FullSourcePlacePolicy(GeometricReleasePlacePolicy):
    """Use V3-trained scores in the frozen V18 geometric decoder."""

    def __init__(self, checkpoint_path, config):
        spec, inputs = load_full_source_spec(config)
        checkpoint = load_checkpoint(checkpoint_path)
        manifest = checkpoint['manifest']
        if (manifest['version'] != VERSION or manifest['recipe'] != spec
                or manifest['recipe_sha256'] != digest_file(Path(config))
                or manifest['seed'] not in spec['seeds']):
            raise ValueError('Place V19 checkpoint/recipe mismatch')
        for name, checksum in manifest['implementation_sha256'].items():
            if digest_file(inputs.root / name) != checksum:
                raise ValueError(f'Place V19 execution code changed: {name}')
        bootstrap = spec['decoder_bootstrap_checkpoints'][str(manifest['seed'])]['path']
        super().__init__(bootstrap, spec['parent_policy_recipe'])
        model = SkillMLP(checkpoint['schema'])
        model.load_state_dict(checkpoint['model'], strict=True)
        model.eval()
        self.model = model
        self.seed = manifest['seed']
