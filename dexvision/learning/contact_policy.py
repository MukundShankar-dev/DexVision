"""Independently frozen contact-phase push and press behavior-cloning bundles."""
from __future__ import annotations

import copy
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

from dexvision.learning.pick_policy import audit_action_bounds
from dexvision.learning.place_policy import PlaceDataset
from dexvision.learning.reach_repair import calibrated_features
from dexvision.learning.skill_datasets import SkillEpisode, action_targets, feature_schema, load_skill_datasets
from dexvision.learning.skill_models import SkillMLP, decode_delta, model_schema
from dexvision.learning.train_skill import _checksum_index, _json, _verify_run, load_checkpoint, train_skill
from dexvision.learning.training_manifest import (
    ReleaseSource, build_training_manifest, content_digest, digest_file, load_training_inputs, skill_interval,
)

VERSIONS = {'level5/push-policy-v1': 'push_object_to_target', 'level5/press-policy-v1': 'press_button'}
EXECUTION_MODULES = (
    'dexvision/learning/contact_policy.py', 'dexvision/evaluation/contact_policy.py',
    'dexvision/learning/train_skill.py', 'dexvision/learning/skill_models.py',
    'dexvision/learning/skill_datasets.py', 'dexvision/learning/place_policy.py',
    'dexvision/learning/reach_repair.py', 'dexvision/evaluation/evaluate_skill.py',
    'dexvision/evaluation/pick_policy.py', 'dexvision/logging/level4_collection.py',
    'dexvision/sim/workcell.py',
)


def load_contact_spec(path):
    path = Path(path)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError('Contact recipe checksum mismatch')
    spec = yaml.safe_load(path.read_text())
    if spec.get('version') not in VERSIONS or spec.get('checkpoint') != '5.5':
        raise ValueError('Unsupported contact recipe')
    inputs = load_training_inputs(spec['parent_config'])
    if inputs.skill['skill'] != VERSIONS[spec['version']]:
        raise ValueError('Contact recipe uses wrong frozen skill/split')
    workspace = inputs.requirements['workcell']['safe_workspace']
    if spec['workspace_bounds'] != {'min': workspace['min'], 'max': workspace['max']}:
        raise ValueError('Contact workspace differs from collection runtime')
    config_path = inputs.root / inputs.plan['inputs']['workcell_config']['path']
    config = yaml.safe_load(config_path.read_text())
    model = ET.parse((config_path.parent / config['model_path']).resolve())
    weld = model.find(f".//weld[@name='{config['scene']['hand_base_weld']}']")
    if weld is None or list(map(float, weld.attrib['relpose'].split())) != spec['weld_relpose']:
        raise ValueError('Contact calibration differs from model weld')
    if inputs.skill['skill'] == 'press_button':
        joint = model.find(".//joint[@name='start_button_joint']")
        if joint is None or list(map(float, joint.attrib['axis'].split())) != spec['fixture_axis_world']:
            raise ValueError('Press axis differs from frozen fixture')
    scales = np.asarray(spec['output_scales'], float)
    active = [0, 1] if inputs.skill['skill'] == 'push_object_to_target' else [0]
    if (scales.shape != (26,) or not np.isfinite(scales).all() or np.any(scales < 0)
            or np.any(scales > 1) or list(np.flatnonzero(scales)) != active):
        raise ValueError('Contact recipe active channels invalid')
    if (spec['seeds'] != [5001, 5002, 5003] or spec['max_epochs'] < 1
            or not spec['candidate_epochs'] or max(spec['candidate_epochs']) != spec['max_epochs']
            or any(s <= 0 for s in (spec['position_scale_m'], spec['velocity_scale_mps'], spec['depth_scale_m']))):
        raise ValueError('Invalid frozen contact budget or physical scales')
    return spec, inputs


def contact_action_plan(spec, inputs):
    plan = copy.deepcopy(inputs.plan)
    for j, field in enumerate(plan['action']['named_layout'][:3]):
        field.update(lower=spec['workspace_bounds']['min'][j], upper=spec['workspace_bounds']['max'][j])
    # Named channel audit, rather than historical phase masks, licenses holding.
    for relevance in plan['action']['phase_relevance'].values():
        relevance.update({group: True for group in relevance})
    return plan


def contact_features(values, names, spec, *, fixture_rest=None, pressed_once=False):
    """[..., source width] pre-action state -> fixed physical relative features.

    Press depth is the measured fixture displacement along its pinned +X slide
    axis. The known fixture rest pose is entry context, not a future label.
    pressed_once is a causal latch of the physical depth/state dwell.
    """
    values = np.asarray(values, float)
    if values.shape[-1] != len(names) or not np.isfinite(values).all():
        raise ValueError('Invalid contact source features')
    index = {name: column for column, name in enumerate(names)}

    def get(keys):
        return values[..., [index[k] for k in keys]]

    prior = get([f'prior_applied_actions.base_position_target/{a}' for a in 'xyz'])
    obj = get([f'entity.position_m.{i}' for i in range(3)])
    measured = calibrated_features(values, names, spec, {'calibrate_measured_frame': True})[..., :3]
    parts = [(obj - prior) / spec['position_scale_m'], (prior - measured) / spec['position_scale_m'],
             prior[..., 2, None] / spec['position_scale_m'],
             get([f'entity.linear_velocity_mps.{i}' for i in range(3)]) / spec['velocity_scale_mps'],
             get([f'entity.angular_velocity_radps.{i}' for i in range(3)])]
    output_names = [*[f'command_to_entity/{a}' for a in 'xyz'], *[f'tracking_lag/{a}' for a in 'xyz'],
                    'command_height', *[f'entity_velocity/{a}' for a in 'xyz'],
                    *[f'entity_angular_velocity/{a}' for a in 'xyz']]
    if spec['version'] == 'level5/push-policy-v1':
        goal = get([f'goal.target_position.{i}' for i in range(3)])
        parts.extend([(goal - obj) / spec['position_scale_m'], (goal - prior) / spec['position_scale_m']])
        output_names.extend([*[f'entity_to_zone/{a}' for a in 'xyz'], *[f'command_to_zone/{a}' for a in 'xyz']])
        prefixes = ('phase.push_contact', 'phase.settle', 'phase.retract', 'goal.target_type.')
    else:
        rest = np.asarray(fixture_rest, float)
        if rest.shape != (3,) or not np.isfinite(rest).all():
            raise ValueError('Press needs a finite known fixture rest pose')
        depth = (obj - rest) @ np.asarray(spec['fixture_axis_world'])
        target = get(['goal.target_depth.0'])[..., 0]
        latch = np.broadcast_to(np.asarray(pressed_once, float), depth.shape)
        if np.any((latch != 0) & (latch != 1)):
            raise ValueError('Press completion latch must be boolean')
        parts.extend([np.stack((depth / spec['depth_scale_m'], (target - depth) / spec['depth_scale_m'], latch), axis=-1),
                      (prior - rest) / spec['position_scale_m']])
        output_names.extend(['fixture_depth', 'fixture_depth_shortfall', 'pressed_once',
                             *[f'fixture_to_command/{a}' for a in 'xyz']])
        prefixes = ()
    selected = [n for n in names if n.startswith(('entity.rotation_6d.', 'robot.rotation6d.',
        'entity.supported.', 'entity.held.', 'entity.contact_with_hand.', 'entity.class.', *prefixes))]
    parts.append(get(selected))
    output_names.extend(selected)
    return np.concatenate(parts, axis=-1), output_names


def contact_targets(episode, plan, spec):
    y, mask = action_targets(episode.records['applied_actions'], episode.records['prior_applied_actions'],
        episode.records['safety_masks'], ('approach',) * len(episode.frames), plan, 'baseline')
    scales = np.asarray(spec['output_scales'])
    inactive = scales == 0
    if np.any(np.abs(y[:, inactive]) > spec['invariant_tolerance']):
        raise ValueError('Contact source violates whole-interval action invariants')
    y /= np.where(inactive, 1, scales)
    if np.any(np.abs(y[mask.astype(bool)]) > 1 + 1e-5):
        raise ValueError('Contact action targets exceed frozen scales')
    mask[:, inactive] = 0
    return y, mask


def prepare_contact_cache(config, directory, release_root=None, *, datasets=None):
    spec, inputs = load_contact_spec(config)
    if datasets is None:
        datasets, _ = load_skill_datasets(inputs, release_root=release_root)
    if set(datasets) != {'train', 'validation'}:
        raise ValueError('Contact cache accepts train/validation only')
    plan = contact_action_plan(spec, inputs)
    issues = audit_action_bounds([e for d in datasets.values() for e in d.episodes], plan)
    if issues:
        raise ValueError(f'Contact safety preflight failed: {issues[0]}')
    rows = {r['episode_id']: r for s in ('train', 'validation') for r in inputs.splits[s]['episodes']
            if skill_interval(inputs.skill, r)}
    paths = {eid: f"data/demos/level4/{r['source_path']}/metadata.json" for eid, r in rows.items()}
    blobs = ReleaseSource(inputs, release_root).read_many(list(paths.values()))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    names = list(feature_schema(inputs.plan, inputs.skill)[0])
    inventory, audit = {}, {}
    for split, dataset in datasets.items():
        inventory[split] = []
        xs, ys, phases = [], [], []
        for episode in dataset.episodes:
            metadata = json.loads(blobs[paths[episode.episode_id]])
            rest = metadata['task_config']['initial_state']['entity_positions_m'][rows[episode.episode_id]['entity_id']]
            latch = np.maximum.accumulate(episode.inputs[:, names.index('phase.retract')] > 0)
            x, feature_names = contact_features(episode.inputs, names, spec, fixture_rest=rest, pressed_once=latch)
            y, mask = contact_targets(episode, plan, spec)
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
        audit[split] = {'episodes': len(dataset.episodes), 'frames': len(x), 'phases': dict(Counter(phases)),
            'feature_min': x.min(0).tolist(), 'feature_max': x.max(0).tolist(), 'feature_std': x.std(0).tolist(),
            'target_min': y.min(0).tolist(), 'target_max': y.max(0).tolist(), 'target_std': y.std(0).tolist()}
    _json(directory / 'cache.json', {'version': spec['version'], 'recipe_sha256': content_digest(spec),
        'input_digests': inputs.digests, 'feature_names': feature_names, 'episodes': inventory, 'audit': audit,
        'safety_preflight': {'passed': True, 'action_plan_sha256': content_digest(plan['action'])},
        'normalization_source': datasets['train'].normalization})
    _checksum_index(directory)
    return {'cache': str(directory), 'audit': audit}


def contact_datasets(spec, inputs, directory):
    directory = Path(directory)
    _verify_run(directory)
    cache = json.loads((directory / 'cache.json').read_text())
    if (cache['recipe_sha256'] != content_digest(spec) or cache['input_digests'] != inputs.digests
            or cache['safety_preflight'] != {'passed': True,
                'action_plan_sha256': content_digest(contact_action_plan(spec, inputs)['action'])}):
        raise ValueError('Contact cache recipe/source/preflight mismatch')
    norm = {**cache['normalization_source'], 'version': spec['version'],
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
            raise ValueError('Contact cache membership differs from frozen split')
        episodes = []
        for item in items:
            row = rows[item['episode_id']]
            if (item['session_id'], item['cell_id'], item['episode_digest']) != (
                    row['recording_session_id'], row['cell_id'], row['episode_digest']):
                raise ValueError('Contact cache ownership mismatch')
            with np.load(directory / item['file'], allow_pickle=False) as data:
                start, end = skill_interval(inputs.skill, row)
                if not np.array_equal(data['frames'], np.arange(start, end)):
                    raise ValueError('Contact cache interval mismatch')
                episodes.append(SkillEpisode(item['episode_id'], split, item['session_id'], item['cell_id'],
                    item['episode_digest'], data['inputs'], data['targets'], data['masks'], data['frames'],
                    data['timestamps'], tuple(data['phases']),
                    tuple(item['normalization_interval']) if item['normalization_interval'] else None, {}))
        result[split] = PlaceDataset(episodes, norm)
    return result


def contact_schema(spec, inputs):
    names = list(feature_schema(inputs.plan, inputs.skill)[0])
    _, feature_names = contact_features(np.zeros(len(names)), names, spec, fixture_rest=np.zeros(3))
    schema = model_schema(inputs.plan, inputs.skill)
    schema.update(input_names=feature_names, observation_width=len(feature_names), goal_width=0,
        observation_contract=spec['version'], action_contract=spec['version'])
    return schema


def contact_settings(spec, inputs):
    settings = copy.deepcopy(inputs.plan['training'])
    settings['max_epochs'] = spec['max_epochs']
    settings['early_stop']['patience_epochs'] = spec['max_epochs'] + 1
    settings['selection']['candidate_epochs'] = spec['candidate_epochs']
    settings['sampling'] = 'uniform_cell_episode_causal_phase_frame_with_replacement'
    return settings


def train_contact(config, *, release_root=None, output_dir=None, seed=None, device='cpu', model='baseline',
                  dry_run=False, resume=False, stop_after_steps=None, cache_dir=None):
    spec, inputs = load_contact_spec(config)
    seed = spec['seeds'][0] if seed is None else seed
    if device != 'cpu' or model != 'baseline' or seed not in spec['seeds']:
        raise ValueError('Contact v1 requires CPU baseline and a frozen seed')
    cache_dir = Path(cache_dir or spec['cache_dir'])
    if not cache_dir.exists():
        prepare_contact_cache(config, cache_dir, release_root)
    datasets = contact_datasets(spec, inputs, cache_dir)
    if dry_run:
        return {'recipe': spec['version'], 'frames': {s: len(d) for s, d in datasets.items()},
            'phases': {s: dict(Counter(p for e in d.episodes for p in e.phases)) for s, d in datasets.items()}}
    manifest = build_training_manifest(inputs, seed=seed, device=device, model=model)
    manifest.update(contact_recipe=spec, contact_recipe_sha256=content_digest(spec),
        cache_sha256=digest_file(cache_dir / 'SHA256SUMS'), sampling=contact_settings(spec, inputs)['sampling'])
    manifest.pop('manifest_digest')
    manifest['manifest_digest'] = content_digest(manifest)
    directory = Path(output_dir or f"{spec['output_dir']}/seed_{seed}")
    report = train_skill(datasets, schema=contact_schema(spec, inputs), settings=contact_settings(spec, inputs),
        manifest=manifest, output_dir=directory, config_snapshot={**inputs.config_snapshot, 'contact_recipe': spec},
        resume=resume, stop_after_steps=stop_after_steps)
    return {k: v for k, v in report.items() if k != 'history'}


def decode_contact(prediction, prior, spec, plan):
    prediction = np.asarray(prediction, float)
    if prediction.shape != (26,) or not np.isfinite(prediction).all() or np.any(np.abs(prediction) > 1 + 1e-6):
        raise ValueError('Invalid bounded contact prediction')
    return decode_delta(prediction * np.asarray(spec['output_scales']), prior, 'approach', plan)


class ContactPolicy:
    """Selected weights plus exact transform, scales, invariants and control cadence."""

    def __init__(self, path, spec, inputs):
        checkpoint = load_checkpoint(path)
        manifest, norm = checkpoint['manifest'], checkpoint['normalization']
        if (manifest.get('contact_recipe_sha256') != content_digest(spec)
                or manifest.get('contact_recipe') != spec or manifest['digests'] != inputs.digests
                or manifest['seed'] not in spec['seeds'] or checkpoint['schema'] != contact_schema(spec, inputs)
                or checkpoint['settings'] != contact_settings(spec, inputs)):
            raise ValueError('Contact policy bundle mismatch')
        if any(manifest['implementation_sha256'].get(name) != digest_file(inputs.root / name)
               for name in EXECUTION_MODULES):
            raise ValueError('Contact execution code changed since training; preserve the run')
        for value, key in ((manifest, 'manifest_digest'), (norm, 'normalization_digest')):
            if content_digest({k: v for k, v in value.items() if k != key}) != value[key]:
                raise ValueError('Contact policy digest mismatch')
        if (norm['ordered_feature_names'] != checkpoint['schema']['input_names']
                or norm['mean'] != [0.] * len(norm['mean']) or norm['std'] != [1.] * len(norm['std'])
                or set(norm['training_episode_ids']) != {r['episode_id'] for r in manifest['assignments']['train']}):
            raise ValueError('Contact policy normalization mismatch')
        self.model = SkillMLP(checkpoint['schema'])
        self.model.load_state_dict(checkpoint['model'], strict=True)
        self.model.eval()
        self.spec, self.inputs = spec, inputs
        self.names = list(feature_schema(inputs.plan, inputs.skill)[0])
        self.action_plan = contact_action_plan(spec, inputs)

    def __call__(self, source, prior, *, fixture_rest=None, pressed_once=False):
        features, _ = contact_features(source, self.names, self.spec,
            fixture_rest=fixture_rest, pressed_once=pressed_once)
        with torch.inference_mode():
            prediction = self.model(torch.from_numpy(features.astype(np.float32)[None]))[0].numpy()
        try:
            action = decode_contact(prediction, prior, self.spec, self.action_plan)
        except ValueError:
            action = np.full(27, np.nan)
        return action, prediction
