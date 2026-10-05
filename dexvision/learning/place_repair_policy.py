"""Independent 5.4A state policy trained on the additive unassisted place source."""
from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import torch
import yaml

from dexvision.learning.place_policy import (
    PlaceDataset, load_place_spec, place_action_plan, place_features, place_schema,
    place_settings,
)
from dexvision.learning.skill_datasets import SkillEpisode, action_targets, feature_schema
from dexvision.learning.skill_models import SkillMLP, decode_delta
from dexvision.learning.train_skill import load_checkpoint, train_skill
from dexvision.learning.training_manifest import content_digest, digest_file

VERSION = 'level5/place-policy-repair-v2'


def load_repair_spec(path):
    path = Path(path)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError('Place repair recipe checksum mismatch')
    spec = yaml.safe_load(path.read_text())
    if spec.get('version') != VERSION or spec.get('checkpoint') != '5.4A':
        raise ValueError('Unsupported place repair recipe')
    parent = Path(spec['parent_place_recipe'])
    if digest_file(parent) != spec['parent_place_recipe_sha256']:
        raise ValueError('Place repair parent changed')
    _, inputs = load_place_spec(parent)
    if digest_file(Path(spec['setup_teacher_recipe'])) != spec['setup_teacher_recipe_sha256']:
        raise ValueError('Place repair setup teacher changed')
    for key in ('manifest', 'archive', 'source_physical_audit'):
        if digest_file(Path(spec['additive_data'][key])) != spec['additive_data'][key + '_sha256']:
            raise ValueError(f'Place repair {key} changed')
    if spec['workspace_bounds'] != {'min': inputs.requirements['workcell']['safe_workspace']['min'],
                                    'max': inputs.requirements['workcell']['safe_workspace']['max']}:
        raise ValueError('Place repair workspace changed')
    if spec['terminal'] != yaml.safe_load(parent.read_text())['terminal']:
        raise ValueError('Place repair terminal changed')
    if spec['seeds'] != [5001, 5002, 5003] or spec['max_epochs'] < 200:
        raise ValueError('Place repair requires three full-budget seeds')
    return spec, inputs


def _released_payloads(spec):
    """Verify every member and return only success payloads, never failed attempts."""
    pin = spec['additive_data']
    manifest = json.loads(Path(pin['manifest']).read_text())
    if (manifest['archive_sha256'] != pin['archive_sha256']
            or manifest['version'] != 'level5/place-unassisted-data-v2'
            or not manifest['physical_audit']['passed']
            or manifest['physical_audit_sha256'] != pin['source_physical_audit_sha256']):
        raise ValueError('Place additive release binding mismatch')
    audit = json.loads(Path(pin['source_physical_audit']).read_text())
    if (audit != manifest['physical_audit'] or audit['audited_successes'] != 187
            or any(not row['passed'] for row in audit['audits'])):
        raise ValueError('Place source physical audit is incomplete')
    wanted = set(manifest['files'])
    payload = {}
    with tarfile.open(pin['archive'], 'r:gz') as archive:
        for member in archive:
            if (not member.isfile() or member.name not in wanted or member.name in payload
                    or '..' in Path(member.name).parts or Path(member.name).is_absolute()):
                raise ValueError('Unsafe, duplicate or unlisted place release member')
            with archive.extractfile(member) as stream:
                blob = stream.read()
            if hashlib.sha256(blob).hexdigest() != manifest['files'][member.name]['sha256']:
                raise ValueError('Place release member checksum mismatch')
            payload[member.name] = blob
    if set(payload) != wanted:
        raise ValueError('Place release is incomplete')
    return manifest, payload


def repair_datasets(spec, inputs):
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
                raise ValueError('Duplicate place episode owner')
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
                raise ValueError('Invalid place release source alignment')
            x, names = place_features(arrays['observations'], source_names, spec)
            if feature_names is None:
                feature_names = names
            elif feature_names != names:
                raise ValueError('Place feature schema changed between episodes')
            y, masks = action_targets(arrays['actions'], arrays['prior_applied_actions'],
                arrays['safety_masks'], ('transport',) * length, plan, 'baseline')
            # The learned output is translation, rotation and a release gate.
            # One source-proven coupled opening vector is decoded below.
            release = arrays['expert_phases'] == 'latched_release'
            y[:, 8] = np.where(release, 1., -1.)
            masks[:, 6:] = 0.
            masks[:, 8] = 1.
            if np.any(np.abs(y[:, :6]) > 1 + 1e-5):
                raise ValueError('Place source motion exceeds bounded action head')
            digest = content_digest({name: manifest['files'][prefix + name]['sha256']
                for name in ('metadata.json', 'result.json', 'trajectory.npz')})
            episodes[split].append(SkillEpisode(eid, split, session, row['cell'], digest,
                x, y, masks, np.arange(length), arrays['timestamps'][:-1],
                tuple(arrays['expert_phases'].tolist()),
                (0, length) if split == 'train' else None, {}))
    if {e.session_id for e in episodes['train']} & {e.session_id for e in episodes['validation']}:
        raise ValueError('Place training/validation session leakage')
    norm = {'version': VERSION, 'ordered_feature_names': feature_names,
        'mean': [0.] * len(feature_names), 'std': [1.] * len(feature_names),
        'continuous_mask': [True] * len(feature_names),
        'training_episode_ids': [e.episode_id for e in episodes['train']],
        'rule': 'fixed_recipe_physical_scales_no_validation_fit',
        'archive_sha256': manifest['archive_sha256']}
    norm['normalization_digest'] = content_digest(norm)
    return {split: PlaceDataset(values, norm) for split, values in episodes.items()}


def train_repair(config, *, seed, output_dir=None, dry_run=False, resume=False):
    from dexvision.evaluation.place_repair_policy import repair_matrix

    spec, inputs = load_repair_spec(config)
    if seed not in spec['seeds']:
        raise ValueError('Seed outside frozen place repair recipe')
    datasets = repair_datasets(spec, inputs)
    if dry_run:
        return {split: {'episodes': len(data.episodes), 'frames': len(data)}
                for split, data in datasets.items()}
    validation = repair_matrix(spec, inputs, 'validation')
    conditions = inputs.config_snapshot['evaluation']['rollouts']['perception_conditions']
    manifest = {'version': VERSION, 'seed': seed, 'device': 'cpu', 'model': 'baseline',
        'recipe_sha256': digest_file(Path(config)), 'recipe': spec,
        'additive_data': spec['additive_data'],
        'validation_matrix_sha256': content_digest(validation),
        'validation_rollout_count': len(validation) * len(conditions),
        'assignments': {split: [{'episode_id': e.episode_id, 'session_id': e.session_id,
            'episode_digest': e.episode_digest} for e in data.episodes]
            for split, data in datasets.items()},
        'implementation_sha256': {name: digest_file(inputs.root / name) for name in (
            'dexvision/learning/place_repair_policy.py', 'dexvision/learning/place_policy.py',
            'dexvision/learning/skill_models.py', 'dexvision/learning/train_skill.py',
            'dexvision/evaluation/place_repair_policy.py')},
    }
    manifest['manifest_digest'] = content_digest(manifest)
    schema = place_schema(spec, inputs)
    schema.update(observation_contract=VERSION, action_contract=VERSION)
    settings = place_settings(spec, inputs)
    settings['loss']['group_weights'] = spec['loss_group_weights']
    directory = Path(output_dir or f"{spec['output_dir']}/seed_{seed}")
    report = train_skill(datasets, schema=schema, settings=settings, manifest=manifest,
        output_dir=directory, config_snapshot={'place_repair_recipe': spec}, resume=resume)
    return {key: value for key, value in report.items() if key != 'history'}


class RepairPolicy:
    """Learn motion and release timing; decode a coupled source-proven opening."""

    def __init__(self, checkpoint_path, config):
        spec, inputs = load_repair_spec(config)
        checkpoint = load_checkpoint(checkpoint_path)
        manifest = checkpoint['manifest']
        if (manifest['recipe'] != spec or manifest['recipe_sha256'] != digest_file(Path(config))
                or manifest['seed'] not in spec['seeds']):
            raise ValueError('Place repair policy/recipe mismatch')
        for name, digest in manifest['implementation_sha256'].items():
            if digest_file(inputs.root / name) != digest:
                raise ValueError('Place repair execution code changed after training')
        self.model = SkillMLP(checkpoint['schema'])
        self.model.load_state_dict(checkpoint['model'], strict=True)
        self.model.eval()
        self.spec, self.inputs = spec, inputs
        self.names = list(feature_schema(inputs.plan, inputs.skill)[0])
        self.plan = place_action_plan(spec, inputs)
        self.latch = None

    def reset(self):
        self.latch = None

    def __call__(self, source, prior, phase=None):
        from dexvision.evaluation.place_repair import release_fingers

        features, _ = place_features(source, self.names, self.spec)
        with torch.inference_mode():
            prediction = self.model(torch.from_numpy(features.astype(np.float32)[None]))[0].numpy()
        prior = np.asarray(prior, float)
        out = prior.copy()
        index = {name: i for i, name in enumerate(self.names)}
        object_position = source[[index[f'entity.position_m.{j}'] for j in range(3)]]
        target_position = source[[index[f'goal.target_position.{j}'] for j in range(3)]]
        distance = float(np.linalg.norm(target_position - object_position))
        held = source[index['entity.held.0']] > .5
        family = next((kind for kind in ('cuboid', 'cylinder', 'flat_puck')
            if source[index[f'entity.class.{kind}']] > .5), None)
        if family is None:
            return np.full(27, np.nan), prediction
        if (self.latch is None and held and distance <= self.spec['terminal']['release_distance_m']
                and prediction[8] > self.spec['release_gate_threshold']):
            self.latch = prior[:7].copy()
        if self.latch is not None:
            out[:7] = self.latch
            out[7:] = release_fingers(prior[7:], self.spec['open_targets'],
                self.spec['closed_targets'], self.spec['release_step_by_family'][family])
        else:
            scaled = np.zeros(26)
            scaled[:6] = prediction[:6] * np.asarray(self.spec['motion_output_scales'])
            out = decode_delta(scaled, prior, 'transport', self.plan)
            out[7:] = prior[7:]
            if held and distance <= self.spec['terminal']['release_distance_m']:
                # Preserve the source's bounded physical alignment inside the
                # release gate while the learned release score is still low.
                error = target_position - object_position
                step = min(max(0., distance - self.spec['alignment_stop_distance_m']),
                    self.spec['alignment_step_m'])
                out[:3] = prior[:3] + error / max(distance, 1e-12) * step
        out[:3] = np.clip(out[:3], self.spec['workspace_bounds']['min'],
                          self.spec['workspace_bounds']['max'])
        return out, prediction
