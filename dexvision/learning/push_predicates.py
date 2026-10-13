"""Checkpoint 5.5 bounded long-range observation around signed task-frame push."""
from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

from dexvision.learning.contact_policy import (
    EXECUTION_MODULES, contact_action_plan, contact_datasets, contact_settings,
    contact_targets, load_contact_spec,
)
from dexvision.learning.pick_policy import audit_action_bounds
from dexvision.learning.reach_repair import calibrated_features
from dexvision.learning.skill_datasets import feature_schema, load_skill_datasets
from dexvision.learning.skill_models import SkillMLP, model_schema
from dexvision.learning.train_skill import _checksum_index, _json, load_checkpoint, train_skill
from dexvision.learning.training_manifest import (
    ReleaseSource, build_training_manifest, content_digest, digest_file, skill_interval,
)

VERSION = 'level5/push-policy-v7'
MODULES = (*EXECUTION_MODULES, 'dexvision/learning/push_repair.py', 'dexvision/evaluation/push_repair.py',
    'dexvision/sim/level4_expert.py', 'dexvision/sim/scripted_expert_settings.py',
    'dexvision/learning/push_distance.py', 'dexvision/evaluation/push_distance.py',
    'dexvision/evaluation/push_settled.py',
    'dexvision/learning/push_canonical.py', 'dexvision/evaluation/push_canonical.py',
    'dexvision/learning/push_latched.py', 'dexvision/evaluation/push_latched.py',
    'dexvision/learning/push_predicates.py', 'dexvision/evaluation/push_predicates.py')


def load_push_spec(path):
    path = Path(path)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError('Push repair recipe checksum mismatch')
    spec = yaml.safe_load(path.read_text())
    parent, inputs = load_contact_spec(spec['parent_recipe'])
    if (spec['version'] != VERSION or spec['checkpoint'] != '5.5'
            or digest_file(Path(spec['parent_recipe'])) != spec['parent_recipe_sha256']):
        raise ValueError('Unsupported push repair parent/version')
    for key in ('parent_config', 'workspace_bounds', 'weld_relpose', 'terminal', 'seeds', 'preconditions'):
        if spec[key] != parent[key]:
            raise ValueError(f'Push repair changed frozen {key}')
    if spec['delta_scale_m'] != 0.005 or spec['max_epochs'] != 200 or spec['candidate_epochs'] != [20, 80, 200]:
        raise ValueError('Push repair scale/budget mismatch')
    if spec['goal_distance_cap_m'] != 0.05 or spec['setup_hold_samples'] != 120:
        raise ValueError('Unsupported frozen push distance/setup transform')
    if spec['angular_velocity_scale_radps'] != 5. or spec['height_change_scale_m'] != .01:
        raise ValueError('Canonical physical scales changed')
    if (spec['lateral_contact_envelope_m'] != .035 or spec['tracking_envelope_m'] != .01
            or spec['object_speed_limit_mps'] != .02):
        raise ValueError('Predicate observation thresholds changed')
    return spec, inputs


def runtime_spec(spec):
    """Use the unchanged V1 physical task contract; learned transform is V2."""
    result = copy.deepcopy(spec)
    result['version'] = 'level5/push-policy-v1'
    return result


def push_basis(goal, rest):
    """Fixed entry-context basis [world XY, task along/lateral], never future state."""
    goal, rest = np.asarray(goal, float), np.asarray(rest, float)
    if goal.shape != (3,) or rest.shape != (3,) or not np.isfinite(np.r_[goal, rest]).all():
        raise ValueError('Push needs finite initial object and goal poses')
    vector = (goal - rest)[:2]
    length = np.linalg.norm(vector)
    if length < 1e-8:
        raise ValueError('Degenerate push direction')
    along = vector / length
    return np.column_stack((along, [-along[1], along[0]]))


def push_features(values, names, spec, *, fixture_rest, entry_source=None, qualified_once=False):
    """Planar goal errors and causal physical predicates, no continuous own-rate proxy."""
    values = np.asarray(values, float)
    reference = values.reshape(-1, len(names))[0] if entry_source is None else np.asarray(entry_source, float)
    if (values.shape[-1] != len(names) or reference.shape != (len(names),)
            or not np.isfinite(values).all() or not np.isfinite(reference).all()):
        raise ValueError('Invalid predicate push observation/reference')
    index = {n: j for j, n in enumerate(names)}
    def get(data, prefix, count=3):
        return data[..., [index[f'{prefix}.{j}'] for j in range(count)]]
    goal, obj = get(values, 'goal.target_position'), get(values, 'entity.position_m')
    if not np.allclose(goal, get(reference, 'goal.target_position'), atol=1e-12, rtol=0):
        raise ValueError('Predicate push goal changed without reset')
    basis = push_basis(goal.reshape(-1, 3)[0], fixture_rest)
    error = (goal - obj)[..., :2] @ basis
    along = np.minimum(error[..., 0], spec['goal_distance_cap_m']) / spec['position_scale_m']
    radial = np.minimum(np.linalg.norm(error, axis=-1), spec['goal_distance_cap_m']) / spec['position_scale_m']
    measured = calibrated_features(values, names, spec, {'calibrate_measured_frame': True})[..., :3]
    first_measured = calibrated_features(reference, names, spec, {'calibrate_measured_frame': True})[..., :3]
    geometry_change = ((obj - measured) - (get(reference, 'entity.position_m') - first_measured))[..., :2] @ basis
    aligned = np.abs(geometry_change[..., 1]) <= spec['lateral_contact_envelope_m']
    speed = np.linalg.norm(get(values, 'entity.linear_velocity_mps'), axis=-1)
    slow = speed <= spec['object_speed_limit_mps']
    cols = [index[f'prior_applied_actions.base_position_target/{a}'] for a in 'xyz']
    tracking = (values[..., cols] - measured) - (reference[cols] - first_measured)
    tracked = np.linalg.norm(tracking[..., :2], axis=-1) <= spec['tracking_envelope_m']
    parts = [np.stack((along, radial, aligned, slow, tracked), axis=-1)]
    labels = ['goal_error/capped_signed_along', 'goal_error/capped_radial',
        'contact_geometry_within_lateral_envelope', 'object_speed_within_original_limit', 'tracking_within_envelope']
    selected = [n for n in names if n.startswith(('entity.contact_with_hand.', 'entity.supported.',
        'phase.approach', 'phase.push_contact', 'phase.settle', 'phase.retract'))]
    parts.append(values[..., [index[n] for n in selected]])
    labels.extend(selected)
    latch = np.broadcast_to(np.asarray(qualified_once, float), values.shape[:-1])
    if np.any((latch != 0) & (latch != 1)):
        raise ValueError('Push completion latch must be boolean')
    parts.append(latch[..., None])
    labels.append('physical_goal_succeeded_once')
    return np.concatenate(parts, axis=-1), labels, basis


def push_schema(spec, inputs):
    names = list(feature_schema(inputs.plan, inputs.skill)[0])
    values = np.zeros(len(names))
    values[names.index('goal.target_position.0')] = 0.1
    _, labels, _ = push_features(values, names, spec, fixture_rest=np.zeros(3))
    schema = model_schema(inputs.plan, inputs.skill)
    schema.update(input_names=labels, observation_width=len(labels), goal_width=0,
        observation_contract=spec['version'], action_contract=spec['version'])
    schema['output_names'][:2] = ['task_signed_along_delta', 'task_lateral_held_zero']
    return schema


def prepare_push_cache(config, directory, release_root=None, *, datasets=None):
    spec, inputs = load_push_spec(config)
    if datasets is None:
        datasets, _ = load_skill_datasets(inputs, release_root=release_root)
    if set(datasets) != {'train', 'validation'}:
        raise ValueError('Push repair supervision accepts train/validation only')
    plan = contact_action_plan(spec, inputs)
    issues = audit_action_bounds([e for d in datasets.values() for e in d.episodes], plan)
    if issues:
        raise ValueError(f'Push safety preflight: {issues[0]}')
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
            x, labels, basis = push_features(episode.inputs, names, spec, fixture_rest=rest, qualified_once=latch)
            _, mask = contact_targets(episode, plan, spec)
            delta = (episode.records['applied_actions'] - episode.records['prior_applied_actions'])[:, :2]
            y = np.zeros((len(x), 26))
            y[:, :2] = delta @ basis / spec['delta_scale_m']
            if np.any(np.abs(y[:, 1] * spec['delta_scale_m']) > spec['invariant_tolerance']):
                raise ValueError('Canonical push source violates audited task-lateral invariant')
            y[:, 1] = 0
            mask[:, 1] = 0
            if np.any(np.abs(y) > 1 + 1e-5):
                raise ValueError('Signed task-frame supervision exceeds frozen scale')
            file = f'{split}/{episode.episode_id}.npz'
            (directory / split).mkdir(exist_ok=True)
            np.savez_compressed(directory / file, inputs=x, targets=y, masks=mask, frames=episode.frames,
                timestamps=episode.timestamps, phases=episode.phases)
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
        'input_digests': inputs.digests, 'feature_names': labels, 'episodes': inventory, 'audit': audit,
        'safety_preflight': {'passed': True, 'action_plan_sha256': content_digest(plan['action'])},
        'normalization_source': datasets['train'].normalization})
    _checksum_index(directory)
    return {'cache': str(directory), 'audit': audit}


def train_push(config, *, release_root=None, output_dir=None, seed=None, device='cpu', model='baseline',
               dry_run=False, resume=False, stop_after_steps=None, cache_dir=None):
    spec, inputs = load_push_spec(config)
    seed = spec['seeds'][0] if seed is None else seed
    if device != 'cpu' or model != 'baseline' or seed not in spec['seeds']:
        raise ValueError('Push repair requires frozen CPU baseline seed')
    cache_dir = Path(cache_dir or spec['cache_dir'])
    if not cache_dir.exists():
        prepare_push_cache(config, cache_dir, release_root)
    datasets = contact_datasets(spec, inputs, cache_dir)
    if dry_run:
        return {'recipe': spec['version'], 'frames': {s: len(d) for s, d in datasets.items()}}
    from dexvision.evaluation.push_predicates import read_entry_matrix
    validation = read_entry_matrix(spec, inputs, 'validation')
    read_entry_matrix(spec, inputs, 'test')
    manifest = build_training_manifest(inputs, seed=seed, device=device, model=model)
    manifest.update(original_validation_matrix_sha256=manifest['validation_matrix_sha256'],
        validation_matrix_sha256=content_digest(validation['matrix']),
        validation_rollout_count=validation['matrix']['rollouts_per_track_per_training_seed'],
        validation_resets_sha256=digest_file(Path(spec['output_dir']) / 'validation_resets/resets.json'))
    manifest.update(contact_recipe=spec, contact_recipe_sha256=content_digest(spec),
        cache_sha256=digest_file(cache_dir / 'SHA256SUMS'), sampling=contact_settings(spec, inputs)['sampling'],
        confirmation_resets_sha256=digest_file(Path(spec['output_dir']) / 'confirmation_resets' / 'resets.json'))
    manifest.pop('manifest_digest')
    manifest['manifest_digest'] = content_digest(manifest)
    report = train_skill(datasets, schema=push_schema(spec, inputs), settings=contact_settings(spec, inputs),
        manifest=manifest, output_dir=Path(output_dir or f"{spec['output_dir']}/seed_{seed}"),
        config_snapshot={**inputs.config_snapshot, 'contact_recipe': spec}, resume=resume,
        stop_after_steps=stop_after_steps)
    return {k: v for k, v in report.items() if k != 'history'}


def decode_push(prediction, prior, basis, spec):
    prediction, prior = np.asarray(prediction, float), np.asarray(prior, float)
    if prediction.shape != (26,) or prior.shape != (27,) or not np.isfinite(np.r_[prediction, prior]).all() \
            or np.any(np.abs(prediction) > 1 + 1e-6):
        raise ValueError('Invalid bounded push prediction')
    local = np.array([prediction[0] * spec['delta_scale_m'], 0.])
    delta = basis @ local
    # Preserve direction while obeying the original per-world-component cap.
    delta /= max(1., np.max(np.abs(delta)) / spec['delta_scale_m'])
    result = prior.copy()
    result[:2] += delta
    return result


class PushPolicy:
    """Signed learned task deltas; fixed measured entry basis and held invariant channels."""
    def __init__(self, path, spec, inputs):
        checkpoint = load_checkpoint(path)
        manifest, norm = checkpoint['manifest'], checkpoint['normalization']
        if (manifest.get('contact_recipe') != spec or manifest.get('contact_recipe_sha256') != content_digest(spec)
                or manifest['digests'] != inputs.digests or checkpoint['schema'] != push_schema(spec, inputs)
                or checkpoint['settings'] != contact_settings(spec, inputs)):
            raise ValueError('Push repair bundle mismatch')
        if any(manifest['implementation_sha256'].get(n) != digest_file(inputs.root / n) for n in MODULES):
            raise ValueError('Push execution code changed since training')
        for value, key in ((manifest, 'manifest_digest'), (norm, 'normalization_digest')):
            if content_digest({k: v for k, v in value.items() if k != key}) != value[key]:
                raise ValueError('Push bundle digest mismatch')
        if norm['mean'] != [0.] * len(norm['mean']) or norm['std'] != [1.] * len(norm['std']):
            raise ValueError('Push physical normalization mismatch')
        self.model = SkillMLP(checkpoint['schema'])
        self.model.load_state_dict(checkpoint['model'], strict=True)
        self.model.eval()
        self.spec, self.inputs = spec, inputs
        self.names = list(feature_schema(inputs.plan, inputs.skill)[0])
        self.entry_source = None
        self.qualified_once = False

    def reset(self):
        self.entry_source = None
        self.qualified_once = False

    def __call__(self, source, prior, *, fixture_rest=None, **kwargs):
        if self.entry_source is None:
            self.entry_source = np.asarray(source, float).copy()
        self.qualified_once |= bool(source[self.names.index('phase.retract')] > 0)
        features, _, basis = push_features(source, self.names, self.spec, fixture_rest=fixture_rest,
            entry_source=self.entry_source, qualified_once=self.qualified_once)
        with torch.inference_mode():
            prediction = self.model(torch.from_numpy(features.astype(np.float32)[None]))[0].numpy()
        return decode_push(prediction, prior, basis, self.spec), prediction
