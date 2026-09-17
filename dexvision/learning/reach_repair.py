"""Versioned reach-only repair: physical scales and demonstrable action invariants.

The original frozen Level 5 loader, models, trainer and checkpoints are unchanged.
This adapter transforms named causal features and trains through that same trainer.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from dexvision.learning.skill_datasets import SkillDataset, SkillEpisode, feature_schema, load_skill_datasets
from dexvision.learning.skill_models import SkillMLP, decode_delta, model_schema
from dexvision.learning.train_skill import _checksum_index, _json, _verify_run, load_checkpoint, train_skill
from dexvision.learning.training_manifest import build_training_manifest, content_digest, digest_file, load_training_inputs, skill_interval


def load_repair_spec(path: str | Path):
    path = Path(path)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError('Repair specification checksum mismatch')
    spec = yaml.safe_load(path.read_text())
    if spec['version'] not in {'level5/reach-repair-v1', 'level5/reach-repair-v2', 'level5/reach-repair-v3', 'level5/reach-repair-v4', 'level5/reach-repair-v5'}:
        raise ValueError('Unsupported repair specification')
    inputs = load_training_inputs(spec['parent_config'])
    if inputs.skill['skill'] != 'reach_object':
        raise ValueError('Repair is reach-only')
    if 'weld_relpose' in spec:
        import xml.etree.ElementTree as ET

        config_path = inputs.root / inputs.plan['inputs']['workcell_config']['path']
        config = yaml.safe_load(config_path.read_text())
        model_path = (config_path.parent / config['model_path']).resolve()
        weld = ET.parse(model_path).find(f".//weld[@name='{config['scene']['hand_base_weld']}']")
        if weld is None or list(map(float, weld.attrib['relpose'].split())) != spec['weld_relpose']:
            raise ValueError('Repair measured-frame transform differs from frozen model weld')
    return spec, inputs


def prepare_repair_cache(spec_path, output_dir, release_root=None):
    _, inputs = load_repair_spec(spec_path)
    datasets, _ = load_skill_datasets(inputs, release_root=release_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    for split, data in datasets.items():
        x = np.concatenate([e.inputs for e in data.episodes])
        y = np.concatenate([e.targets for e in data.episodes])
        np.savez_compressed(output_dir / f'{split}.npz', inputs=x, targets=y,
            masks=np.concatenate([e.masks for e in data.episodes]),
            prior=np.concatenate([e.records['prior_applied_actions'] for e in data.episodes]),
            episode_ids=np.array([e.episode_id for e in data.episodes for _ in e.frames]),
            cell_ids=np.array([e.cell_id for e in data.episodes for _ in e.frames]))
        _json(output_dir / f'{split}_statistics.json', {'frames': len(x), 'target_min': y.min(axis=0).tolist(),
              'target_max': y.max(axis=0).tolist(), 'target_std': y.std(axis=0).tolist(),
              'max_absolute_target': np.abs(y).max(axis=0).tolist()})
    _json(output_dir / 'normalization.json', datasets['train'].normalization)
    _json(output_dir / 'input_digests.json', inputs.digests)
    _checksum_index(output_dir)
    return {'cache': str(output_dir), 'train_frames': len(datasets['train']), 'validation_frames': len(datasets['validation'])}


def calibrated_features(values, names, spec, recipe):
    """Express measured forearm origin in the mocap command frame using the weld."""
    values = np.asarray(values, float).copy()
    if not recipe.get('calibrate_measured_frame', False):
        return values
    from dexvision.learning.skill_datasets import rotation6d

    if list(names[3:9]) != [f'robot.rotation6d.{i}' for i in range(6)]:
        raise ValueError('Measured rotation layout differs from calibration contract')
    relative = np.asarray(spec['weld_relpose'])
    rotation = rotation6d(relative[3:])
    relative_rotation = np.column_stack((rotation[:3], rotation[3:], np.cross(rotation[:3], rotation[3:])))
    first, second = values[..., 3:6], values[..., 6:9]
    measured_rotation = np.stack((first, second, np.cross(first, second)), axis=-1)
    target_rotation = measured_rotation @ relative_rotation.T
    values[..., :3] -= np.einsum('...ij,j->...i', target_rotation, relative[:3])
    return values


def transform_features(values, names, kind):
    """[... ,192] causal inputs -> full vector or [goal error xyz, tracking lag xyz]."""
    values = np.asarray(values, float)
    if values.shape[-1] != len(names) or not np.isfinite(values).all():
        raise ValueError('Invalid named repair features')
    if kind == 'full_physical':
        return values.copy(), list(names)
    if kind not in {'relative_goal', 'goal_plane', 'goal_plane_height'}:
        raise ValueError('Unknown repair feature representation')
    goal = [names.index(f'goal.approach_position.{i}') for i in range(3)]
    prior = [names.index(f'prior_applied_actions.base_position_target/{axis}') for axis in 'xyz']
    measured = list(range(3))
    if list(names[:3]) != ['robot.position.x', 'robot.position.y', 'robot.position.z']:
        raise ValueError('Measured position layout differs from repair contract')
    result = np.concatenate((values[..., goal] - values[..., prior], values[..., prior] - values[..., measured]), axis=-1)
    if kind.startswith('goal_plane'):
        distance = np.linalg.norm(result[..., :2], axis=-1)
        unit = result[..., :2] / np.maximum(distance[..., None], 1e-12)
        radial_lag = np.sum(result[..., 3:5] * unit, axis=-1)
        output = np.stack((distance, result[..., 2], radial_lag, result[..., 5]), axis=-1)
        output_names = ['goal_plane/distance', 'goal_plane/vertical_error', 'goal_plane/radial_lag', 'goal_plane/vertical_lag']
        if kind == 'goal_plane_height':
            output = np.concatenate((output, values[..., prior[2], None]), axis=-1)
            output_names.append('goal_plane/height')
        return output, output_names
    return result, [f'goal_error/{a}' for a in 'xyz'] + [f'tracking_lag/{a}' for a in 'xyz']


def physical_scales(names, plan, spec):
    """Fixed SI scales; no near-zero empirical divisor and no validation fitting."""
    units = spec['physical_scales']
    action = {f['name']: f['upper'] - f['lower'] for f in plan['action']['named_layout']}
    scales = []
    for name in names:
        if name.startswith(('goal_plane/', 'goal_error/', 'tracking_lag/', 'robot.position.', 'entity.position_m.', 'goal.approach_position.')):
            value = units['position_m']
        elif name.startswith(('prior_applied_actions.', 'prior_commanded_actions.')):
            value = action[name.split('.', 1)[1]]
        elif name.startswith(('robot.rotation', 'entity.rotation', 'goal.approach_rotation.')):
            value = units['rotation_6d']
        elif name.startswith('finger_joint_positions.'):
            value = units['joint_position_rad']
        elif name.startswith('finger_joint_velocities.'):
            value = units['joint_velocity_radps']
        elif 'velocity' in name:
            value = units['angular_velocity_radps'] if ('angular' in name or '/w' in name) else units['linear_velocity_mps']
        else:
            value = units['scalar_or_binary']
        scales.append(value)
    result = np.array(scales, float)
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError('Physical feature scales must be finite and positive')
    return result


def repair_datasets(spec, inputs, cache_dir, arm):
    cache_dir = Path(cache_dir)
    _verify_run(cache_dir)
    if json.loads((cache_dir / 'input_digests.json').read_text()) != inputs.digests:
        raise ValueError('Cache belongs to different frozen inputs')
    recipe = spec['models'][arm]
    original_names, _ = feature_schema(inputs.plan, inputs.skill)
    arrays = {s: np.load(cache_dir / f'{s}.npz', allow_pickle=False) for s in ('train', 'validation')}
    training_targets = arrays['train']['targets']
    zero = np.all(np.abs(training_targets) <= spec['zero_target_tolerance'], axis=0)
    if not np.any(~zero):
        raise ValueError('Training data contains no learned action signal')
    if recipe['hold_zero_targets'] and (np.any(zero[:3]) or not np.all(zero[3:])):
        raise ValueError('This reach repair requires varying translation and demonstrated zero rotation/joint deltas')
    old_norm = json.loads((cache_dir / 'normalization.json').read_text())
    _, names = transform_features(arrays['train']['inputs'], list(original_names), recipe['features'])
    norm = {**old_norm, 'version': 'level5/physical-normalization-v1', 'ordered_feature_names': names,
            'mean': np.zeros(len(names)).tolist(), 'std': physical_scales(names, inputs.plan, spec).tolist(),
            'continuous_mask': [True] * len(names), 'statistics_rule': 'fixed_physical_scales_no_empirical_fit',
            'repair_spec_sha256': content_digest(spec)}
    norm.pop('normalization_digest')
    norm['normalization_digest'] = content_digest(norm)
    datasets = {}
    owners = {r['episode_id']: r for r in inputs.splits['train']['normalization_inputs']}
    for split, data in arrays.items():
        rows = {r['episode_id']: r for r in inputs.splits[split]['episodes'] if skill_interval(inputs.skill, r) is not None}
        if set(data['episode_ids']) != set(rows):
            raise ValueError('Cache membership differs from frozen reach split')
        episodes = []
        for eid, row in sorted(rows.items()):
            pick = data['episode_ids'] == eid
            start, end = skill_interval(inputs.skill, row)
            if int(pick.sum()) != end - start or set(data['cell_ids'][pick]) != {row['cell_id']}:
                raise ValueError('Cache episode interval or cell differs from frozen ownership')
            x, _ = transform_features(calibrated_features(data['inputs'][pick], original_names, spec, recipe), list(original_names), recipe['features'])
            masks = data['masks'][pick].copy()
            if recipe['hold_zero_targets']:
                masks[:, zero] = 0
            targets = data['targets'][pick].copy()
            if recipe['features'].startswith('goal_plane'):
                relative, _ = transform_features(data['inputs'][pick], list(original_names), 'relative_goal')
                distance = np.linalg.norm(relative[:, :2], axis=1)
                unit = relative[:, :2] / np.maximum(distance[:, None], 1e-12)
                caps = np.array([f['max_change_per_sample'] for f in inputs.plan['action']['named_layout'][:3]])
                delta = targets[:, :2] * caps[:2]
                radial = np.sum(delta * unit, axis=1)
                tolerance = spec['goal_plane']['tangential_tolerance_m']
                if (np.max(np.abs(delta - radial[:, None] * unit)) > tolerance
                        or np.min(radial) < -tolerance or np.max(radial - distance) > tolerance):
                    raise ValueError('Reach data violates preregistered planar action invariants')
                fraction = radial / np.maximum(np.minimum(distance, spec['goal_plane']['maximum_planar_step_m']), 1e-12)
                vertical = targets[:, 2].copy()
                targets[:] = 0
                targets[:, :2] = np.stack((fraction, vertical), axis=1)
                masks[:] = 0
                masks[:, :2] = 1
            owner = owners.get(eid)
            episodes.append(SkillEpisode(eid, split, row['recording_session_id'], row['cell_id'], row['episode_digest'],
                x, targets, masks, np.arange(start, end),
                (np.arange(start, end) + 1) * inputs.plan['action']['control_interval_s'],
                ('approach',) * (end - start), tuple(owner['frame_interval']) if owner else None, {}))
        datasets[split] = SkillDataset(episodes, norm)
    for data in arrays.values():
        data.close()
    return datasets, zero.tolist(), names


def train_repair(spec_path, cache_dir, output_dir, *, arm, seed, resume=False):
    spec, inputs = load_repair_spec(spec_path)
    if seed not in spec['seeds'] or arm not in spec['models']:
        raise ValueError('Choose a preregistered repair arm and seed')
    datasets, zero, names = repair_datasets(spec, inputs, cache_dir, arm)
    schema = model_schema(inputs.plan, inputs.skill)
    schema.update(input_names=names, observation_width=len(names), goal_width=0)
    if spec['models'][arm]['features'].startswith('goal_plane'):
        schema['output_names'] = ['planar_fraction', 'vertical_delta', *[f'unused/{i}' for i in range(24)]]
    settings = copy.deepcopy(inputs.plan['training'])
    settings['max_epochs'] = spec['max_epochs']
    settings['early_stop']['patience_epochs'] = spec['max_epochs'] + 1
    settings['selection']['candidate_epochs'] = spec['candidate_epochs']
    manifest = build_training_manifest(inputs, seed=seed, device='cpu', model='baseline')
    manifest.update(repair_arm=arm, repair_spec=spec, repair_spec_sha256=content_digest(spec), zero_target_channels=zero,
                    cache_sha256=digest_file(Path(cache_dir) / 'SHA256SUMS'))
    manifest.pop('manifest_digest')
    manifest['manifest_digest'] = content_digest(manifest)
    return train_skill(datasets, schema=schema, settings=settings, manifest=manifest, output_dir=output_dir,
                       config_snapshot={**inputs.config_snapshot, 'repair': spec}, resume=resume)


class RepairPolicy:
    """Learned standalone reach controller with explicit train-supported invariants."""

    def __init__(self, checkpoint_path, spec, inputs):
        checkpoint = load_checkpoint(checkpoint_path)
        manifest = checkpoint['manifest']
        if (manifest['digests'] != inputs.digests or manifest['repair_spec_sha256'] != content_digest(spec)
                or content_digest({k:v for k,v in manifest.items() if k != 'manifest_digest'}) != manifest['manifest_digest']):
            raise ValueError('Repair checkpoint provenance mismatch')
        self.recipe = spec['models'][manifest['repair_arm']]
        self.names = list(feature_schema(inputs.plan, inputs.skill)[0])
        self.zero = np.array(manifest['zero_target_channels'], bool)
        self.norm = checkpoint['normalization']
        if content_digest({k:v for k,v in self.norm.items() if k != 'normalization_digest'}) != self.norm['normalization_digest']:
            raise ValueError('Repair normalization digest mismatch')
        expected_names = transform_features(np.zeros(len(self.names)), self.names, self.recipe['features'])[1]
        if (self.norm['ordered_feature_names'] != expected_names
                or not np.array_equal(self.norm['mean'], np.zeros(len(expected_names)))
                or not np.array_equal(self.norm['std'], physical_scales(expected_names, inputs.plan, spec))
                or set(self.norm['training_episode_ids']) != {r['episode_id'] for r in manifest['assignments']['train']}
                or manifest['seed'] not in spec['seeds']
                or self.zero.shape != (26,)
                or self.recipe['hold_zero_targets'] and (np.any(self.zero[:3]) or not np.all(self.zero[3:]))):
            raise ValueError('Repair feature scales, ownership or action invariant contract mismatch')
        self.model = SkillMLP(checkpoint['schema'])
        self.model.load_state_dict(checkpoint['model'], strict=True)
        self.model.eval()
        self.plan = inputs.plan
        self.spec = spec

    def __call__(self, features, prior, phase):
        x, names = transform_features(calibrated_features(features, self.names, self.spec, self.recipe), self.names, self.recipe['features'])
        if names != self.norm['ordered_feature_names']:
            raise ValueError('Repair live feature schema mismatch')
        x = ((x - self.norm['mean']) / self.norm['std']).astype(np.float32)
        with torch.inference_mode():
            prediction = self.model(torch.from_numpy(x[None]))[0].numpy()
        if self.recipe['features'].startswith('goal_plane'):
            relative, _ = transform_features(features, self.names, 'relative_goal')
            distance = np.linalg.norm(relative[:2])
            unit = relative[:2] / max(distance, 1e-12)
            caps = np.array([f['max_change_per_sample'] for f in self.plan['action']['named_layout'][:3]])
            expanded = np.zeros(26)
            # An explicit learned action parameterization, not post-supervisor clipping.
            # Zero network => zero displacement; direction alone cannot execute a reach.
            expanded[:2] = max(0.0, float(prediction[0])) * min(distance, self.spec['goal_plane']['maximum_planar_step_m']) * unit / caps[:2]
            expanded[2] = prediction[1]
            prediction = expanded
        elif self.recipe['hold_zero_targets']:
            prediction[self.zero] = 0
        try:
            action = decode_delta(prediction, prior, phase, self.plan)
        except ValueError:
            action = np.full(27, np.nan)
        return action, prediction
