"""Versioned state pick recipe, verified cache and complete learned policy bundle."""
from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

from dexvision.learning.phase_losses import phase_errors, pick_phases
from dexvision.learning.reach_repair import calibrated_features
from dexvision.learning.skill_datasets import (
    SkillDataset, SkillEpisode, action_targets, feature_schema, load_skill_datasets,
)
from dexvision.learning.skill_models import SkillMLP, decode_delta, model_schema
from dexvision.learning.train_skill import _checksum_index, _json, _verify_run, load_checkpoint, train_skill
from dexvision.learning.training_manifest import (
    build_training_manifest, content_digest, digest_file, load_training_inputs, skill_interval,
)

VERSION = "level5/pick-policy-v1"
VERSIONS = {VERSION, "level5/pick-policy-v2", "level5/pick-policy-v3", "level5/pick-policy-v4", "level5/pick-policy-v5", "level5/pick-policy-v6", "level5/pick-policy-v7"}


def load_pick_spec(path):
    path = Path(path)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError("Pick recipe checksum mismatch")
    spec = yaml.safe_load(path.read_text())
    if spec['version'] not in VERSIONS or spec['checkpoint'] != ('5.3A' if 'closure' in spec else '5.3'):
        raise ValueError("Unsupported pick recipe")
    inputs = load_training_inputs(spec['parent_config'])
    if inputs.skill['skill'] != 'pick_object':
        raise ValueError("This recipe supports pick_object only")
    import xml.etree.ElementTree as ET
    config_path = inputs.root / inputs.plan['inputs']['workcell_config']['path']
    config = yaml.safe_load(config_path.read_text())
    weld = ET.parse((config_path.parent / config['model_path']).resolve()).find(
        f".//weld[@name='{config['scene']['hand_base_weld']}']")
    if weld is None or list(map(float, weld.attrib['relpose'].split())) != spec['weld_relpose']:
        raise ValueError("Pick measured-frame calibration differs from model")
    scales = np.asarray(spec['output_scales'], float)
    if scales.shape != (26,) or not np.isfinite(scales).all() or np.any((scales < 0) | (scales > 1)):
        raise ValueError("Invalid pick output scales")
    if not np.all(scales[3:6] > 0) or not np.any(scales[8:] > 0):
        raise ValueError("Pick must retain rotation and closure outputs")
    if spec['version'] != VERSION:
        workspace = inputs.requirements['workcell']['safe_workspace']
        if spec['workspace_authority'] != 'frozen_level4_recording_runtime' or spec['workspace_bounds'] != {
                'min': workspace['min'], 'max': workspace['max']}:
            raise ValueError("Pick workspace must exactly preserve the frozen collection runtime")
    if 'closure' in spec:
        closure = spec['closure']
        start, end = (np.asarray(closure[k], float) for k in ('open_targets', 'closed_targets'))
        if (start.shape != (20,) or end.shape != (20,) or not np.isfinite(np.r_[start, end]).all()
                or not 0 < closure['maximum_step'] <= 0.05 or np.linalg.norm(end - start) <= 0):
            raise ValueError('Invalid closure synergy recipe')
        fields = inputs.plan['action']['named_layout'][7:]
        lower, upper, caps = (np.asarray([f[k] for f in fields]) for k in ('lower', 'upper', 'max_change_per_sample'))
        if (np.any(start < lower) or np.any(start > upper) or np.any(end < lower) or np.any(end > upper)
                or np.any(np.abs(end - start) * closure['maximum_step'] > caps + 1e-12)):
            raise ValueError('Closure synergy violates frozen actuator or rate bounds')
    if 'unassisted_dataset' in spec:
        from dexvision.learning.pick_data import amended_inputs
        inputs = amended_inputs(inputs, spec)
    return spec, inputs


def pick_action_plan(spec, inputs):
    """V2 binds the actual released recorder workspace, preserving v1 metadata."""
    plan = copy.deepcopy(inputs.plan)
    if spec['version'] != VERSION:
        for j, field in enumerate(plan['action']['named_layout'][:3]):
            field['lower'] = spec['workspace_bounds']['min'][j]
            field['upper'] = spec['workspace_bounds']['max'][j]
    return plan


def audit_action_bounds(episodes, plan):
    """Fail before training if expert targets or their priors contradict safety."""
    fields = plan['action']['named_layout']
    lower, upper = (np.asarray([f[k] for f in fields]) for k in ('lower', 'upper'))
    issues = []
    for e in episodes:
        for key in ('applied_actions', 'prior_applied_actions'):
            values = e.records[key]
            bad = (values < lower - 1e-12) | (values > upper + 1e-12)
            if bad.any():
                issues.append({'episode_id': e.episode_id, 'record': key, 'violating_values': int(bad.sum()),
                    'fields': [f['name'] for j, f in enumerate(fields) if bad[:, j].any()]})
    return issues


def boundary_context(values, names, spec=None):
    """[7] entry object height/rotation; V4 adds entry command height as value 8."""
    indices = [names.index('entity.position_m.2'), *[names.index(f'entity.rotation_6d.{j}') for j in range(6)]]
    if spec and 'geometry' in spec:
        indices.append(names.index('prior_applied_actions.base_position_target/z'))
    return np.asarray(values)[indices].copy()


def pick_features(values, names, context, spec, *, history=None):
    """[...,194] source state -> calibrated relative features and causal phases.

    Context is frozen at skill entry and contains no future state or expert stage.
    Output values are already scaled with recipe constants, never fitted on validation.
    """
    values = np.asarray(values, float)
    if values.shape[-1] != len(names) or not np.isfinite(values).all():
        raise ValueError("Invalid pick source features")
    context = np.asarray(context, float)
    if context.shape != (8 if 'geometry' in spec else 7,) or not np.isfinite(context).all():
        raise ValueError("Invalid pick boundary context")
    if 'geometry' in spec:
        from dexvision.learning.pick_geometry import compact_pick_features
        output, output_names, phases = compact_pick_features(values, names, context, spec)
        if 'orientation_feedback' in spec:
            from dexvision.learning.pick_feedback import closed_contact_counts
            counts = (np.asarray(history['closed_contact_dwell']) if history is not None
                      else closed_contact_counts(values, names, spec))
            if values.ndim == 1 and counts.ndim:
                counts = counts[0]
            counts = np.broadcast_to(counts, values.shape[:-1])
            output = np.concatenate((output, np.minimum(counts / spec['closed_contact_scale'], 1.)[..., None]), axis=-1)
            output_names = [*output_names, 'causal_closed_contact_dwell']
        return output, output_names, phases
    calibrated = calibrated_features(values, names, spec, {'calibrate_measured_frame': True})
    def get(keys):
        return values[..., [names.index(k) for k in keys]]
    prior = get([f'prior_applied_actions.base_position_target/{a}' for a in 'xyz'])
    entity = get([f'entity.position_m.{j}' for j in range(3)])
    height = entity[..., 2] - context[0]
    phases = pick_phases(get(['entity.held.0'])[..., 0], height, spec['lift_height_m'])
    parts = [(entity - prior) / spec['position_scale_m'],
             (prior - calibrated[..., :3]) / spec['position_scale_m'],
             prior[..., 2, None] / spec['position_scale_m'], height[..., None] / spec['position_scale_m']]
    output_names = [*[f'object_minus_command/{a}' for a in 'xyz'],
                    *[f'tracking_lag/{a}' for a in 'xyz'], 'command_height', 'object_lift_height']
    prefixes = ('robot.rotation6d.', 'robot.velocity.', 'finger_joint_positions.',
                'finger_joint_velocities.', 'prior_applied_actions.base_orientation_target/',
                'prior_applied_actions.finger_actuator_targets/', 'entity.rotation_6d.',
                'entity.linear_velocity_mps.', 'entity.angular_velocity_radps.',
                'entity.supported.', 'entity.held.', 'entity.contact_with_hand.', 'entity.class.')
    selected = [n for n in names if n.startswith(prefixes)]
    parts.append(get(selected))  # SI radians, m/s, rad/s and unitless all have fixed scale 1.
    output_names.extend(selected)
    parts.append(np.broadcast_to(context[1:], (*values.shape[:-1], 6)))
    output_names.extend(f'entry_object_rotation/{j}' for j in range(6))
    parts.append(np.stack([phases == p for p in ('acquire', 'lift', 'stabilize')], axis=-1))
    output_names.extend(f'pick_phase/{p}' for p in ('acquire', 'lift', 'stabilize'))
    return np.concatenate(parts, axis=-1), output_names, phases


def pick_targets(episode, plan, spec, *, names=None):
    """Keep closure supervision before first contact; no reach/sequence phase masks."""
    targets, masks = action_targets(episode.records['applied_actions'], episode.records['prior_applied_actions'],
        episode.records['safety_masks'], ('acquire',) * len(episode.frames), plan, 'baseline')
    scales = np.asarray(spec['output_scales'])
    inactive = scales == 0
    if np.any(np.abs(targets[:, inactive]) > spec['invariant_tolerance']):
        raise ValueError("Pick data violates frozen action invariants")
    result = targets / np.where(inactive, 1, scales)
    if np.any(np.abs(result[masks.astype(bool)]) > 1 + 1e-5):
        raise ValueError("Pick targets exceed recipe output scales")
    masks[:, inactive] = 0
    if 'closure' in spec:
        prior = episode.records['prior_applied_actions'][:, 7:]
        current = episode.records['applied_actions'][:, 7:]
        before = closure_fraction(prior, spec)
        after = closure_fraction(current, spec)
        change = (after - before) / spec['closure']['maximum_step']
        if np.any(np.abs(change) > 1 + 1e-10):
            raise ValueError('Closure targets exceed the frozen synergy step')
        result[:, 8:] = 0
        masks[:, 8:] = 0
        result[:, 8] = change
        masks[:, 8] = 1
    if 'geometry' in spec:
        from dexvision.learning.pick_geometry import geometry_targets
        if names is None:
            names = list(feature_schema(plan, load_training_inputs(spec['parent_config']).skill)[0])
        result, masks = geometry_targets(episode, result, masks, spec, names)
    return result, masks


def closure_fraction(targets, spec):
    """Project [...,20] commands onto the training-proven finger synergy; reject residuals."""
    start = np.asarray(spec['closure']['open_targets'], float)
    direction = np.asarray(spec['closure']['closed_targets'], float) - start
    targets = np.asarray(targets, float)
    fraction = (targets - start) @ direction / (direction @ direction)
    if (not np.isfinite(targets).all() or np.any(fraction < -1e-8) or np.any(fraction > 1 + 1e-8)
            or not np.allclose(targets, start + fraction[..., None] * direction, atol=1e-8, rtol=0)):
        raise ValueError('Command is outside the audited closure synergy')
    return np.clip(fraction, 0, 1)


def decode_pick(prediction, prior, spec, plan, *, features=None, names=None, context=None):
    """Bounded learned lift/rotation and, in v3, one learned closure increment."""
    prediction = np.asarray(prediction, float)
    if prediction.shape != (26,) or not np.isfinite(prediction).all() or np.any(np.abs(prediction) > 1 + 1e-6):
        raise ValueError('Invalid bounded pick prediction')
    if 'geometry' in spec:
        from dexvision.learning.pick_geometry import geometry_action
        if features is None or names is None or context is None:
            raise ValueError('V4 decoding requires causal geometry and boundary context')
        return geometry_action(prediction, prior, spec, plan, features, names, context)
    scaled = prediction * np.asarray(spec['output_scales'])
    if 'closure' not in spec:
        return decode_delta(scaled, prior, 'acquire', plan)
    scaled[8:] = 0
    action = decode_delta(scaled, prior, 'acquire', plan)
    fraction = closure_fraction(np.asarray(prior)[7:], spec)
    fraction = np.clip(fraction + float(prediction[8]) * spec['closure']['maximum_step'], 0, 1)
    start = np.asarray(spec['closure']['open_targets'])
    direction = np.asarray(spec['closure']['closed_targets']) - start
    action[7:] = start + fraction * direction
    return action


def prepare_pick_cache(spec_path, output_dir, release_root=None, *, datasets=None):
    spec, inputs = load_pick_spec(spec_path)
    if datasets is None:
        if 'unassisted_dataset' in spec:
            from dexvision.learning.pick_data import raw_amendment_datasets
            datasets = raw_amendment_datasets(inputs, spec)
        else:
            datasets, _ = load_skill_datasets(inputs, release_root=release_root)
    issues = audit_action_bounds([e for d in datasets.values() for e in d.episodes], pick_action_plan(spec, inputs))
    if issues:
        raise ValueError(f"Pick safety preflight failed: {issues[0]}; preserve data and reconcile versioned runtime bounds")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    names = list(feature_schema(inputs.plan, inputs.skill)[0])
    inventory, audit = {}, {}
    for split, data in datasets.items():
        if split not in {'train', 'validation'}:
            raise ValueError("Test data cannot enter pick cache")
        inventory[split] = []
        all_x, all_y, phases = [], [], []
        for e in data.episodes:
            context = boundary_context(e.inputs[0], names, spec)
            x, feature_names, phase = pick_features(e.inputs, names, context, spec)
            y, mask = pick_targets(e, inputs.plan, spec, names=names)
            file = f'{split}/{e.episode_id}.npz'
            (directory / split).mkdir(exist_ok=True)
            np.savez_compressed(directory / file, inputs=x, targets=y, masks=mask,
                frames=e.frames, timestamps=e.timestamps, phases=phase, context=context)
            inventory[split].append({'episode_id': e.episode_id, 'session_id': e.session_id, 'cell_id': e.cell_id,
                'episode_digest': e.episode_digest, 'normalization_interval': e.normalization_interval, 'file': file})
            all_x.append(x)
            all_y.append(y)
            phases.extend(phase.tolist())
        x, y = np.concatenate(all_x), np.concatenate(all_y)
        audit[split] = {'episodes': len(data.episodes), 'frames': len(x), 'phases': dict(Counter(phases)),
            'feature_min': x.min(axis=0).tolist(), 'feature_max': x.max(axis=0).tolist(),
            'feature_std': x.std(axis=0).tolist(), 'target_min': y.min(axis=0).tolist(),
            'target_max': y.max(axis=0).tolist(), 'target_std': y.std(axis=0).tolist()}
    manifest = {'version': VERSION, 'recipe_sha256': content_digest(spec), 'input_digests': inputs.digests,
                'feature_names': feature_names, 'episodes': inventory, 'audit': audit,
                'safety_preflight': {'passed': True, 'action_plan_sha256': content_digest(pick_action_plan(spec, inputs)['action'])},
                'normalization_source': datasets['train'].normalization}
    _json(directory / 'cache.json', manifest)
    _checksum_index(directory)
    return {'cache': str(directory), 'audit': {s: {k: v for k, v in a.items() if k in {'episodes', 'frames', 'phases'}} for s, a in audit.items()}}


def pick_datasets(spec, inputs, directory):
    directory = Path(directory)
    _verify_run(directory)
    cache = json.loads((directory / 'cache.json').read_text())
    if cache['recipe_sha256'] != content_digest(spec) or cache['input_digests'] != inputs.digests:
        raise ValueError("Pick cache recipe/source mismatch")
    if cache.get('safety_preflight') != {'passed': True, 'action_plan_sha256': content_digest(pick_action_plan(spec, inputs)['action'])}:
        raise ValueError("Pick cache lacks matching safety preflight; preserve historical attempts")
    norm = {**cache['normalization_source'], 'version': VERSION, 'ordered_feature_names': cache['feature_names'],
            'mean': [0.] * len(cache['feature_names']), 'std': [1.] * len(cache['feature_names']),
            'continuous_mask': [True] * len(cache['feature_names']), 'rule': 'fixed_recipe_physical_scales',
            'recipe_sha256': content_digest(spec)}
    norm.pop('normalization_digest')
    norm['normalization_digest'] = content_digest(norm)
    result = {}
    for split in ('train', 'validation'):
        rows = {r['episode_id']: r for r in inputs.splits[split]['episodes'] if skill_interval(inputs.skill, r)}
        items = cache['episodes'][split]
        if {e['episode_id'] for e in items} != set(rows) or len(items) != len(rows):
            raise ValueError("Pick cache membership differs from frozen split")
        episodes = []
        for e in items:
            row = rows[e['episode_id']]
            if (e['session_id'], e['cell_id'], e['episode_digest']) != (row['recording_session_id'], row['cell_id'], row['episode_digest']):
                raise ValueError("Pick cache ownership mismatch")
            with np.load(directory / e['file'], allow_pickle=False) as a:
                start, end = skill_interval(inputs.skill, row)
                if not np.array_equal(a['frames'], np.arange(start, end)):
                    raise ValueError("Pick cache frame interval mismatch")
                episodes.append(SkillEpisode(e['episode_id'], split, e['session_id'], e['cell_id'], e['episode_digest'],
                    a['inputs'], a['targets'], a['masks'], a['frames'], a['timestamps'], tuple(a['phases']),
                    tuple(e['normalization_interval']) if e['normalization_interval'] else None, {}))
        result[split] = SkillDataset(episodes, norm)
    return result


def pick_schema(spec, inputs):
    names = list(feature_schema(inputs.plan, inputs.skill)[0])
    _, feature_names, _ = pick_features(np.zeros(len(names)), names, np.zeros(8 if 'geometry' in spec else 7), spec,
        history={'closed_contact_dwell': 0})
    schema = model_schema(inputs.plan, inputs.skill)
    schema.update(input_names=feature_names, observation_width=len(feature_names), goal_width=0,
                  observation_contract=spec['version'], action_contract=spec['version'])
    if 'closure' in spec:
        schema['output_names'][8] = 'closure_fraction_delta_over_maximum_step'
    if 'geometry' in spec:
        schema['output_names'][3:6] = [f'hand_frame_rotation_delta/{a}' for a in 'xyz']
    if 'orientation_feedback' in spec:
        schema['output_names'][3:6] = ['orientation_correction_magnitude', 'inactive_orientation_y', 'inactive_orientation_z']
    if 'binary_action_gates' in spec:
        schema['output_names'][2] = 'binary_lift_gate_score'
        schema['output_names'][8] = 'binary_close_gate_score'
    return schema


def pick_settings(spec, inputs):
    settings = copy.deepcopy(inputs.plan['training'])
    settings['max_epochs'] = spec['max_epochs']
    settings['early_stop']['patience_epochs'] = spec['max_epochs'] + 1
    settings['selection']['candidate_epochs'] = spec['candidate_epochs']
    return settings


def train_pick(config, *, release_root=None, output_dir=None, seed=None, device='cpu',
               model='baseline', dry_run=False, resume=False, stop_after_steps=None, cache_dir=None):
    spec, inputs = load_pick_spec(config)
    if 'source_physical_audit' in spec:
        from dexvision.learning.pick_data import verify_amendment_audit
        verify_amendment_audit(inputs, spec)
    if device != 'cpu' or model != 'baseline' or seed not in spec['seeds']:
        raise ValueError("Pick v1 requires CPU baseline and a preregistered seed")
    cache_dir = Path(cache_dir or spec['cache_dir'])
    if not cache_dir.exists():
        prepare_pick_cache(config, cache_dir, release_root)
    datasets = pick_datasets(spec, inputs, cache_dir)
    if dry_run:
        return {'recipe': spec['version'], 'frames': {s: len(d) for s, d in datasets.items()}}
    manifest = build_training_manifest(inputs, seed=seed, device=device, model=model)
    manifest.update(pick_recipe=spec, pick_recipe_sha256=content_digest(spec),
                    cache_sha256=digest_file(cache_dir / 'SHA256SUMS'))
    manifest.pop('manifest_digest')
    manifest['manifest_digest'] = content_digest(manifest)
    directory = Path(output_dir or f"{spec['output_dir']}/seed_{seed}")
    report = train_skill(datasets, schema=pick_schema(spec, inputs), settings=pick_settings(spec, inputs),
        manifest=manifest, output_dir=directory, config_snapshot={**inputs.config_snapshot, 'pick_recipe': spec},
        resume=resume, stop_after_steps=stop_after_steps)
    if report['finished']:
        policy = PickPolicy(directory / 'last.pt', spec, inputs)
        errors = {}
        for split, data in datasets.items():
            with torch.inference_mode():
                predictions = policy.model(torch.from_numpy(data.inputs)).numpy()
            errors[split] = phase_errors(predictions, data.targets, data.masks,
                                        [p for e in data.episodes for p in e.phases])
        _json(directory / 'last_epoch_phase_errors.json', errors)
        _checksum_index(directory)
    return {k: v for k, v in report.items() if k != 'history'}


class PickPolicy:
    """Weights, transform, fixed scales and active action decoder travel together."""

    def __init__(self, path, spec, inputs):
        checkpoint = load_checkpoint(path)
        manifest, norm = checkpoint['manifest'], checkpoint['normalization']
        if (manifest.get('pick_recipe_sha256') != content_digest(spec) or manifest.get('pick_recipe') != spec
                or manifest['digests'] != inputs.digests or manifest['seed'] not in spec['seeds']
                or checkpoint['schema'] != pick_schema(spec, inputs) or checkpoint['settings'] != pick_settings(spec, inputs)):
            raise ValueError("Pick checkpoint bundle mismatch")
        for value, key in ((manifest, 'manifest_digest'), (norm, 'normalization_digest')):
            if content_digest({k: v for k, v in value.items() if k != key}) != value[key]:
                raise ValueError("Pick bundle digest mismatch")
        if (norm['ordered_feature_names'] != checkpoint['schema']['input_names']
                or norm['mean'] != [0.] * len(norm['mean']) or norm['std'] != [1.] * len(norm['std'])
                or set(norm['training_episode_ids']) != {r['episode_id'] for r in manifest['assignments']['train']}):
            raise ValueError("Pick normalization mismatch")
        self.model = SkillMLP(checkpoint['schema'])
        self.model.load_state_dict(checkpoint['model'], strict=True)
        self.model.eval()
        self.spec, self.inputs = spec, inputs
        self.action_plan = pick_action_plan(spec, inputs)
        self.names = list(feature_schema(inputs.plan, inputs.skill)[0])
        self.context = None

    def reset(self, source_features):
        self.context = boundary_context(source_features, self.names, self.spec)
        self.closed_contact_dwell = 0

    def __call__(self, features, prior, phase=None):
        if self.context is None:
            raise ValueError("Pick policy needs boundary context before inference")
        history = None
        if 'orientation_feedback' in self.spec:
            active = closure_fraction(np.asarray(prior)[7:], self.spec) >= 1 - 1e-8 and features[self.names.index('entity.held.0')] > 0.5
            self.closed_contact_dwell = self.closed_contact_dwell + 1 if active else 0
            history = {'closed_contact_dwell': self.closed_contact_dwell}
        values, _, _ = pick_features(features, self.names, self.context, self.spec, history=history)
        with torch.inference_mode():
            prediction = self.model(torch.from_numpy(values.astype(np.float32)[None]))[0].numpy()
        try:
            action = decode_pick(prediction, prior, self.spec, self.inputs.plan,
                features=features, names=self.names, context=self.context)
        except ValueError:
            action = np.full(27, np.nan)
        return action, prediction
