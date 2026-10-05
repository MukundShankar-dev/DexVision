"""V6 place repair: bounded physical motion with learned release and orientation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from dexvision.learning.place_geometry_policy import geometry_datasets
from dexvision.learning.place_policy import load_place_spec, place_action_plan, place_features, place_schema, place_settings
from dexvision.learning.skill_datasets import feature_schema
from dexvision.learning.skill_models import SkillMLP, decode_delta
from dexvision.learning.train_skill import load_checkpoint, train_skill
from dexvision.learning.training_manifest import content_digest, digest_file

VERSION = 'level5/place-policy-guarded-v6'


def load_guarded_spec(config):
    path = Path(config)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError('Place guarded recipe checksum mismatch')
    spec = yaml.safe_load(path.read_text())
    if spec['version'] != VERSION or spec['checkpoint'] != '5.4A':
        raise ValueError('Unsupported place guarded recipe')
    parent = Path(spec['parent_place_recipe'])
    if digest_file(parent) != spec['parent_place_recipe_sha256']:
        raise ValueError('Place guarded parent changed')
    _, inputs = load_place_spec(parent)
    for key in ('manifest', 'archive', 'source_physical_audit'):
        if digest_file(Path(spec['additive_data'][key])) != spec['additive_data'][key + '_sha256']:
            raise ValueError(f'Place guarded {key} changed')
    if digest_file(Path(spec['setup_teacher_recipe'])) != spec['setup_teacher_recipe_sha256']:
        raise ValueError('Place guarded setup teacher changed')
    parent_spec = yaml.safe_load(parent.read_text())
    if (spec['terminal'] != parent_spec['terminal']
            or spec['workspace_bounds'] != parent_spec['workspace_bounds']
            or spec['seeds'] != [5001, 5002, 5003] or spec['max_epochs'] < 200):
        raise ValueError('Place guarded weakens frozen parent contract')
    if set(spec['geometry']['release_goal_error_by_family_m']) != {'cuboid', 'cylinder', 'flat_puck'}:
        raise ValueError('Place guarded needs audited family release geometry')
    if spec['geometry']['transport_step_m'] != 0.005 or spec['geometry']['descent_step_m'] != 0.001:
        raise ValueError('Place guarded differs from the resolved source control step')
    if (spec['geometry']['release_distance_by_family_m']['cuboid'] != 0.0245
            or spec['release_step_by_family']['cylinder'] != 0.0125
            or any(value > spec['terminal']['release_distance_m']
                   for value in spec['geometry']['release_distance_by_family_m'].values())):
        raise ValueError('Place guarded release controls differ from frozen V6 recipe')
    return spec, inputs


def train_guarded(config, *, seed, output_dir=None, dry_run=False, resume=False):
    from dexvision.evaluation.place_repair_policy import repair_matrix

    spec, inputs = load_guarded_spec(config)
    if seed not in spec['seeds']:
        raise ValueError('Seed outside frozen place guarded recipe')
    datasets = geometry_datasets(spec, inputs)
    for dataset in datasets.values():
        norm = {**dataset.normalization, 'version': VERSION,
                'rule': 'causal_object_feedback_fixed_physical_scales'}
        norm.pop('normalization_digest')
        norm['normalization_digest'] = content_digest(norm)
        dataset.normalization = norm
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
            'dexvision/learning/place_guarded_policy.py',
            'dexvision/evaluation/place_guarded_policy.py',
            'dexvision/learning/place_geometry_policy.py',
            'dexvision/learning/place_repair_policy.py',
            'dexvision/evaluation/place_repair_policy.py',
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
        output_dir=directory, config_snapshot={'place_guarded_recipe': spec}, resume=resume)
    return {key: value for key, value in report.items() if key != 'history'}


class GuardedPolicy:
    """Use physical waypoints while held; learned scores control rotation and release."""

    def __init__(self, checkpoint_path, config):
        spec, inputs = load_guarded_spec(config)
        checkpoint = load_checkpoint(checkpoint_path)
        manifest = checkpoint['manifest']
        if (manifest['recipe'] != spec or manifest['recipe_sha256'] != digest_file(Path(config))
                or manifest['seed'] not in spec['seeds']):
            raise ValueError('Place guarded bundle mismatch')
        for name, digest in manifest['implementation_sha256'].items():
            if digest_file(inputs.root / name) != digest:
                raise ValueError('Place guarded execution code changed after training')
        self.model = SkillMLP(checkpoint['schema'])
        self.model.load_state_dict(checkpoint['model'], strict=True)
        self.model.eval()
        self.spec, self.inputs = spec, inputs
        self.names = list(feature_schema(inputs.plan, inputs.skill)[0])
        self.plan = place_action_plan(spec, inputs)
        self.index = {name: i for i, name in enumerate(self.names)}
        self.object_id = None
        self.reset()

    def bind(self, row):
        self.object_id = row['entity_id']

    def reset(self):
        self.latch = None
        self.stage = 'raise'

    def _goal_error(self, family):
        return np.asarray(self.spec['geometry']['release_goal_error_by_family_m'][family], float)

    def _waypoint(self, object_position, target_position, prior, family):
        geometry = self.spec['geometry']
        transport_z = geometry['transport_object_height_by_object_m'].get(self.object_id,
            geometry['transport_object_height_by_family_m'].get(family,
                geometry['transport_object_height_default_m']))
        desired = target_position - self._goal_error(family)
        while True:
            if self.stage == 'raise':
                if object_position[2] >= transport_z - geometry['waypoint_tolerance_m']:
                    self.stage = 'planar'
                    continue
                target = prior.copy()
                target[2] += max(0., transport_z - object_position[2])
                return target, geometry['transport_step_m']
            if self.stage == 'planar':
                error_xy = desired[:2] - object_position[:2]
                if np.linalg.norm(error_xy) <= geometry['planar_tolerance_m']:
                    self.stage = 'lower'
                    continue
                target = prior.copy()
                target[:2] += error_xy
                return target, geometry['transport_step_m']
            return prior + (desired - object_position), geometry['descent_step_m']

    def __call__(self, source, prior, phase=None):
        from dexvision.evaluation.place_repair import release_fingers

        if self.object_id is None:
            raise ValueError('Place guarded policy needs a typed object request')
        features, _ = place_features(source, self.names, self.spec)
        with torch.inference_mode():
            prediction = self.model(torch.from_numpy(features.astype(np.float32)[None]))[0].numpy()
        if not np.isfinite(prediction).all():
            return np.full(27, np.nan), prediction
        prior = np.asarray(prior, float)
        index = self.index
        object_position = source[[index[f'entity.position_m.{j}'] for j in range(3)]]
        target_position = source[[index[f'goal.target_position.{j}'] for j in range(3)]]
        family = next((kind for kind in ('cuboid', 'cylinder', 'flat_puck')
            if source[index[f'entity.class.{kind}']] > .5), None)
        if family is None:
            return np.full(27, np.nan), prediction
        distance = float(np.linalg.norm(target_position - object_position))
        held = source[index['entity.held.0']] > .5
        trigger = self.spec['geometry']['release_distance_by_family_m'][family]
        if (self.latch is None and held and distance <= trigger
                and prediction[8] > self.spec['release_gate_threshold']):
            self.latch = prior[:7].copy()
        if self.latch is not None:
            out = prior.copy()
            out[:7] = self.latch
            out[7:] = release_fingers(prior[7:], self.spec['open_targets'],
                self.spec['closed_targets'], self.spec['release_step_by_family'][family])
            return out, prediction
        rotation = np.zeros(26)
        rotation[3:6] = prediction[3:6]
        out = decode_delta(rotation, prior, 'transport', self.plan)
        out[7:] = prior[7:]
        if ((held and distance > trigger)
                or prediction[0] > self.spec['geometry']['move_gate_threshold']):
            waypoint, step = self._waypoint(object_position, target_position, prior[:3], family)
            delta = waypoint - prior[:3]
            length = float(np.linalg.norm(delta))
            out[:3] = prior[:3] + delta / max(length, 1e-12) * min(length, step)
        out[:3] = np.clip(out[:3], self.spec['workspace_bounds']['min'],
                          self.spec['workspace_bounds']['max'])
        return out, prediction
