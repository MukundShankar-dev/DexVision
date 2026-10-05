"""Versioned, state-grounded place policy and immutable training cache."""
from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

from dexvision.learning.pick_policy import audit_action_bounds
from dexvision.learning.reach_repair import calibrated_features
from dexvision.learning.skill_datasets import SkillDataset, SkillEpisode, action_targets, feature_schema, load_skill_datasets
from dexvision.learning.skill_models import SkillMLP, decode_delta, model_schema
from dexvision.learning.train_skill import _checksum_index, _json, _verify_run, load_checkpoint, train_skill
from dexvision.learning.training_manifest import build_training_manifest, content_digest, digest_file, load_training_inputs, skill_interval

VERSION = 'level5/place-policy-v1'


def load_place_spec(path):
    path = Path(path)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError('Place recipe checksum mismatch')
    spec = yaml.safe_load(path.read_text())
    if spec.get('version') != VERSION or spec.get('checkpoint') != '5.4':
        raise ValueError('Unsupported place recipe')
    inputs = load_training_inputs(spec['parent_config'])
    if inputs.skill['skill'] != 'place_held_object':
        raise ValueError('Place recipe must use the frozen place split')
    workspace = inputs.requirements['workcell']['safe_workspace']
    if spec['workspace_bounds'] != {'min': workspace['min'], 'max': workspace['max']}:
        raise ValueError('Place workspace differs from collection runtime')
    import xml.etree.ElementTree as ET
    config_path = inputs.root / inputs.plan['inputs']['workcell_config']['path']
    config = yaml.safe_load(config_path.read_text())
    weld = ET.parse((config_path.parent / config['model_path']).resolve()).find(
        f".//weld[@name='{config['scene']['hand_base_weld']}']")
    if weld is None or list(map(float, weld.attrib['relpose'].split())) != spec['weld_relpose']:
        raise ValueError('Place measured-frame calibration differs from model')
    scales = np.asarray(spec['output_scales'], float)
    if scales.shape != (26,) or not np.isfinite(scales).all() or np.any((scales < 0) | (scales > 1)):
        raise ValueError('Invalid place output scales')
    if not np.all(scales[:6] > 0) or not np.any(scales[8:] > 0):
        raise ValueError('Place must learn transport, lowering, rotation and release')
    if spec['position_scale_m'] <= 0 or spec['velocity_scale_mps'] <= 0:
        raise ValueError('Place physical scales must be positive')
    return spec, inputs


def place_action_plan(spec, inputs):
    plan = copy.deepcopy(inputs.plan)
    for j, field in enumerate(plan['action']['named_layout'][:3]):
        field['lower'] = spec['workspace_bounds']['min'][j]
        field['upper'] = spec['workspace_bounds']['max'][j]
    # Place needs translation during transport and release opening near target.
    # The frozen generic phase mask is historical and cannot mask the release.
    plan['action']['phase_relevance']['transport'] = {
        group: True for group in plan['action']['phase_relevance']['transport']}
    return plan


def place_features(values, names, spec):
    """Transform [..., 195] pre-action state into fixed-scale causal geometry."""
    values = np.asarray(values, float)
    if values.shape[-1] != len(names) or not np.isfinite(values).all():
        raise ValueError('Invalid place source features')
    index = {name: column for column, name in enumerate(names)}

    def get(keys):
        return values[..., [index[k] for k in keys]]

    axes = 'xyz'
    prior = get([f'prior_applied_actions.base_position_target/{a}' for a in axes])
    obj = get([f'entity.position_m.{i}' for i in range(3)])
    goal = get([f'goal.target_position.{i}' for i in range(3)])
    measured = calibrated_features(values, names, spec, {'calibrate_measured_frame': True})[..., :3]
    position_scale = spec['position_scale_m']
    parts = [(goal - obj) / position_scale, (goal - prior) / position_scale,
             (obj - prior) / position_scale, (prior - measured) / position_scale,
             prior[..., 2, None] / position_scale,
             get([f'entity.linear_velocity_mps.{i}' for i in range(3)]) / spec['velocity_scale_mps'],
             get([f'entity.angular_velocity_radps.{i}' for i in range(3)])]
    output_names = [*[f'object_to_target/{a}' for a in axes], *[f'command_to_target/{a}' for a in axes],
                    *[f'command_to_object/{a}' for a in axes], *[f'tracking_lag/{a}' for a in axes],
                    'command_height', *[f'object_velocity/{a}' for a in axes],
                    *[f'object_angular_velocity/{a}' for a in axes]]
    prefixes = ('entity.rotation_6d.', 'goal.target_rotation.', 'robot.rotation6d.',
                'finger_joint_positions.', 'prior_applied_actions.base_orientation_target/',
                'prior_applied_actions.finger_actuator_targets/', 'entity.supported.',
                'entity.held.', 'entity.contact_with_hand.', 'entity.class.', 'goal.target_type.',
                'phase.transport', 'phase.place', 'phase.release', 'phase.settle', 'phase.retract')
    selected = [name for name in names if name.startswith(prefixes)]
    parts.append(get(selected))
    output_names.extend(selected)
    return np.concatenate(parts, axis=-1), output_names


def place_targets(episode, plan, spec):
    # All channels stay eligible across phases. The per-channel audit below is
    # the only authority for holding a channel fixed.
    targets, masks = action_targets(episode.records['applied_actions'],
        episode.records['prior_applied_actions'], episode.records['safety_masks'],
        ('transport',) * len(episode.frames), plan, 'baseline')
    scales = np.asarray(spec['output_scales'])
    inactive = scales == 0
    if np.any(np.abs(targets[:, inactive]) > spec['invariant_tolerance']):
        raise ValueError('Place source violates frozen action invariants')
    targets = targets / np.where(inactive, 1, scales)
    if np.any(np.abs(targets[masks.astype(bool)]) > 1 + 1e-5):
        raise ValueError('Place targets exceed recipe scales')
    masks[:, inactive] = 0
    return targets, masks


class PlaceDataset(SkillDataset):
    """Uniform cell, episode and present causal phase to expose short release intervals."""

    def __init__(self, episodes, normalization):
        super().__init__(episodes, normalization)
        self.phase_buckets = {}
        offset = 0
        for episode in self.episodes:
            local = {}
            for phase in set(episode.phases):
                local[phase] = offset + np.flatnonzero(np.asarray(episode.phases) == phase)
            self.phase_buckets.setdefault(episode.cell_id, []).append(local)
            offset += len(episode.frames)

    def sample_indices(self, rng, count):
        cells = sorted(self.phase_buckets)
        result = []
        for _ in range(count):
            episodes = self.phase_buckets[cells[int(rng.integers(len(cells)))]]
            phases = episodes[int(rng.integers(len(episodes)))]
            frames = phases[sorted(phases)[int(rng.integers(len(phases)))]]
            result.append(frames[int(rng.integers(len(frames)))])
        return np.asarray(result, np.int64)


def prepare_place_cache(config, directory, release_root=None, *, datasets=None):
    spec, inputs = load_place_spec(config)
    if datasets is None:
        datasets, _ = load_skill_datasets(inputs, release_root=release_root)
    if set(datasets) != {'train', 'validation'}:
        raise ValueError('Place cache accepts train/validation only')
    plan = place_action_plan(spec, inputs)
    issues = audit_action_bounds([e for d in datasets.values() for e in d.episodes], plan)
    if issues:
        raise ValueError(f'Place action safety preflight failed: {issues[0]}')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    names = list(feature_schema(inputs.plan, inputs.skill)[0])
    inventory, audit = {}, {}
    for split, dataset in datasets.items():
        inventory[split] = []
        xs, ys, phases = [], [], []
        for episode in dataset.episodes:
            x, feature_names = place_features(episode.inputs, names, spec)
            y, mask = place_targets(episode, plan, spec)
            file = f'{split}/{episode.episode_id}.npz'
            (directory / split).mkdir(exist_ok=True)
            np.savez_compressed(directory / file, inputs=x, targets=y, masks=mask,
                frames=episode.frames, timestamps=episode.timestamps, phases=episode.phases)
            inventory[split].append({'episode_id': episode.episode_id, 'session_id': episode.session_id,
                'cell_id': episode.cell_id, 'episode_digest': episode.episode_digest,
                'normalization_interval': episode.normalization_interval, 'file': file})
            xs.append(x)
            ys.append(y)
            phases.extend(episode.phases)
        x, y = np.concatenate(xs), np.concatenate(ys)
        audit[split] = {'episodes': len(dataset.episodes), 'frames': len(x),
            'phases': dict(Counter(phases)), 'feature_min': x.min(0).tolist(),
            'feature_max': x.max(0).tolist(), 'feature_std': x.std(0).tolist(),
            'target_min': y.min(0).tolist(), 'target_max': y.max(0).tolist(),
            'target_std': y.std(0).tolist()}
    manifest = {'version': VERSION, 'recipe_sha256': content_digest(spec), 'input_digests': inputs.digests,
        'feature_names': feature_names, 'episodes': inventory, 'audit': audit,
        'safety_preflight': {'passed': True, 'action_plan_sha256': content_digest(plan['action'])},
        'normalization_source': datasets['train'].normalization}
    _json(directory / 'cache.json', manifest)
    _checksum_index(directory)
    return {'cache': str(directory), 'audit': {s: {k: v for k, v in a.items()
        if k in {'episodes', 'frames', 'phases'}} for s, a in audit.items()}}


def place_datasets(spec, inputs, directory):
    directory = Path(directory)
    _verify_run(directory)
    cache = json.loads((directory / 'cache.json').read_text())
    if cache['recipe_sha256'] != content_digest(spec) or cache['input_digests'] != inputs.digests:
        raise ValueError('Place cache recipe/source mismatch')
    expected = {'passed': True, 'action_plan_sha256': content_digest(place_action_plan(spec, inputs)['action'])}
    if cache.get('safety_preflight') != expected:
        raise ValueError('Place cache safety preflight mismatch')
    norm = {**cache['normalization_source'], 'version': VERSION,
        'ordered_feature_names': cache['feature_names'], 'mean': [0.] * len(cache['feature_names']),
        'std': [1.] * len(cache['feature_names']), 'continuous_mask': [True] * len(cache['feature_names']),
        'rule': 'fixed_recipe_physical_scales', 'recipe_sha256': content_digest(spec)}
    norm.pop('normalization_digest')
    norm['normalization_digest'] = content_digest(norm)
    result = {}
    for split in ('train', 'validation'):
        rows = {r['episode_id']: r for r in inputs.splits[split]['episodes'] if skill_interval(inputs.skill, r)}
        items = cache['episodes'][split]
        if {e['episode_id'] for e in items} != set(rows) or len(items) != len(rows):
            raise ValueError('Place cache membership differs from frozen split')
        episodes = []
        for item in items:
            row = rows[item['episode_id']]
            if (item['session_id'], item['cell_id'], item['episode_digest']) != (
                    row['recording_session_id'], row['cell_id'], row['episode_digest']):
                raise ValueError('Place cache ownership mismatch')
            with np.load(directory / item['file'], allow_pickle=False) as data:
                start, end = skill_interval(inputs.skill, row)
                if not np.array_equal(data['frames'], np.arange(start, end)):
                    raise ValueError('Place cache interval mismatch')
                episodes.append(SkillEpisode(item['episode_id'], split, item['session_id'], item['cell_id'],
                    item['episode_digest'], data['inputs'], data['targets'], data['masks'], data['frames'],
                    data['timestamps'], tuple(data['phases']),
                    tuple(item['normalization_interval']) if item['normalization_interval'] else None, {}))
        result[split] = PlaceDataset(episodes, norm)
    return result


def place_schema(spec, inputs):
    names = list(feature_schema(inputs.plan, inputs.skill)[0])
    _, feature_names = place_features(np.zeros(len(names)), names, spec)
    schema = model_schema(inputs.plan, inputs.skill)
    schema.update(input_names=feature_names, observation_width=len(feature_names), goal_width=0,
        observation_contract=VERSION, action_contract=VERSION)
    return schema


def place_settings(spec, inputs):
    settings = copy.deepcopy(inputs.plan['training'])
    settings['max_epochs'] = spec['max_epochs']
    settings['early_stop']['patience_epochs'] = spec['max_epochs'] + 1
    settings['selection']['candidate_epochs'] = spec['candidate_epochs']
    settings['sampling'] = 'uniform_cell_episode_causal_phase_frame_with_replacement'
    return settings


def train_place(config, *, release_root=None, output_dir=None, seed=None, device='cpu',
                model='baseline', dry_run=False, resume=False, stop_after_steps=None, cache_dir=None):
    spec, inputs = load_place_spec(config)
    if device != 'cpu' or model != 'baseline' or seed not in spec['seeds']:
        raise ValueError('Place v1 requires CPU baseline and a frozen seed')
    cache_dir = Path(cache_dir or spec['cache_dir'])
    if not cache_dir.exists():
        prepare_place_cache(config, cache_dir, release_root)
    datasets = place_datasets(spec, inputs, cache_dir)
    if dry_run:
        return {'recipe': VERSION, 'frames': {s: len(d) for s, d in datasets.items()},
            'phases': {s: dict(Counter(p for e in d.episodes for p in e.phases)) for s, d in datasets.items()}}
    manifest = build_training_manifest(inputs, seed=seed, device=device, model=model)
    manifest.update(place_recipe=spec, place_recipe_sha256=content_digest(spec),
        cache_sha256=digest_file(cache_dir / 'SHA256SUMS'), sampling=place_settings(spec, inputs)['sampling'])
    manifest.pop('manifest_digest')
    manifest['manifest_digest'] = content_digest(manifest)
    directory = Path(output_dir or f"{spec['output_dir']}/seed_{seed}")
    report = train_skill(datasets, schema=place_schema(spec, inputs), settings=place_settings(spec, inputs),
        manifest=manifest, output_dir=directory, config_snapshot={**inputs.config_snapshot, 'place_recipe': spec},
        resume=resume, stop_after_steps=stop_after_steps)
    return {k: v for k, v in report.items() if k != 'history'}


class PlacePolicy:
    """Bind selected weights to the exact transform, scales, decoder and cadence."""

    def __init__(self, path, spec, inputs):
        checkpoint = load_checkpoint(path)
        manifest, norm = checkpoint['manifest'], checkpoint['normalization']
        if (manifest.get('place_recipe_sha256') != content_digest(spec)
                or manifest.get('place_recipe') != spec or manifest['digests'] != inputs.digests
                or manifest['seed'] not in spec['seeds'] or checkpoint['schema'] != place_schema(spec, inputs)
                or checkpoint['settings'] != place_settings(spec, inputs)):
            raise ValueError('Place policy bundle mismatch')
        execution_modules = (
            'dexvision/learning/place_policy.py', 'dexvision/evaluation/place_policy.py',
            'dexvision/learning/train_skill.py', 'dexvision/learning/skill_models.py',
            'dexvision/learning/skill_datasets.py', 'dexvision/evaluation/evaluate_skill.py',
            'dexvision/evaluation/pick_policy.py', 'dexvision/logging/level4_collection.py',
        )
        if any(manifest['implementation_sha256'].get(name) != digest_file(inputs.root / name)
               for name in execution_modules):
            raise ValueError('Place execution code changed since training; preserve the run')
        for value, key in ((manifest, 'manifest_digest'), (norm, 'normalization_digest')):
            if content_digest({k: v for k, v in value.items() if k != key}) != value[key]:
                raise ValueError('Place policy digest mismatch')
        if (norm['ordered_feature_names'] != checkpoint['schema']['input_names']
                or norm['mean'] != [0.] * len(norm['mean']) or norm['std'] != [1.] * len(norm['std'])
                or set(norm['training_episode_ids']) != {r['episode_id'] for r in manifest['assignments']['train']}):
            raise ValueError('Place policy normalization mismatch')
        self.model = SkillMLP(checkpoint['schema'])
        self.model.load_state_dict(checkpoint['model'], strict=True)
        self.model.eval()
        self.spec, self.inputs = spec, inputs
        self.names = list(feature_schema(inputs.plan, inputs.skill)[0])
        self.action_plan = place_action_plan(spec, inputs)

    def __call__(self, source, prior, phase=None):
        features, _ = place_features(source, self.names, self.spec)
        with torch.inference_mode():
            prediction = self.model(torch.from_numpy(features.astype(np.float32)[None]))[0].numpy()
        try:
            action = decode_place(prediction, prior, self.spec, self.action_plan)
        except ValueError:
            action = np.full(27, np.nan)
        return action, prediction


def decode_place(prediction, prior, spec, plan):
    prediction = np.asarray(prediction, float)
    if prediction.shape != (26,) or not np.isfinite(prediction).all() or np.any(np.abs(prediction) > 1 + 1e-6):
        raise ValueError('Invalid bounded place prediction')
    scaled = prediction * np.asarray(spec['output_scales'])
    return decode_delta(scaled, prior, 'transport', plan)
