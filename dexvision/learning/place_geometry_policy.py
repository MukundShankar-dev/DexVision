"""V3 place repair: learned motion/release gates with explicit goal geometry."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from dexvision.learning.place_policy import (
    PlaceDataset, load_place_spec, place_action_plan, place_features, place_schema, place_settings,
)
from dexvision.learning.place_repair_policy import (
    VERSION as SOURCE_VERSION, repair_datasets,
)
from dexvision.learning.skill_datasets import SkillEpisode, feature_schema
from dexvision.learning.skill_models import SkillMLP, decode_delta
from dexvision.learning.train_skill import load_checkpoint, train_skill
from dexvision.learning.training_manifest import content_digest, digest_file

VERSION = 'level5/place-policy-geometry-v3'


def load_geometry_spec(config):
    path = Path(config)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError('Place geometry recipe checksum mismatch')
    spec = yaml.safe_load(path.read_text())
    if spec['version'] != VERSION or spec['checkpoint'] != '5.4A':
        raise ValueError('Unsupported place geometry recipe')
    parent = Path(spec['parent_place_recipe'])
    if digest_file(parent) != spec['parent_place_recipe_sha256']:
        raise ValueError('Place geometry parent changed')
    _, inputs = load_place_spec(parent)
    for key in ('manifest', 'archive', 'source_physical_audit'):
        if digest_file(Path(spec['additive_data'][key])) != spec['additive_data'][key + '_sha256']:
            raise ValueError(f'Place geometry {key} changed')
    if digest_file(Path(spec['setup_teacher_recipe'])) != spec['setup_teacher_recipe_sha256']:
        raise ValueError('Place geometry setup teacher changed')
    parent_spec = yaml.safe_load(parent.read_text())
    if (spec['terminal'] != parent_spec['terminal']
            or spec['workspace_bounds'] != parent_spec['workspace_bounds']
            or spec['seeds'] != [5001, 5002, 5003] or spec['max_epochs'] < 200):
        raise ValueError('Place geometry weakens frozen parent contract')
    return spec, inputs


def geometry_datasets(spec, inputs):
    """Supervise a causal move gate and release gate; no expert stage is an input."""
    base = repair_datasets(spec, inputs)
    result = {}
    for split, dataset in base.items():
        episodes = []
        for episode in dataset.episodes:
            y = episode.targets.copy()
            mask = np.zeros_like(episode.masks)
            moving = np.linalg.norm(episode.targets[:, :3], axis=1) > 1e-8
            y[:, 0] = np.where(moving, 1., -1.)
            mask[:, 0] = 1.
            mask[:, 3:6] = 1.
            mask[:, 8] = 1.
            episodes.append(SkillEpisode(episode.episode_id, split, episode.session_id,
                episode.cell_id, episode.episode_digest, episode.inputs, y, mask,
                episode.frames, episode.timestamps, episode.phases,
                episode.normalization_interval, {}))
        norm = {**dataset.normalization, 'version': VERSION,
                'rule': 'causal_geometry_fixed_physical_scales'}
        norm.pop('normalization_digest')
        norm['normalization_digest'] = content_digest(norm)
        result[split] = PlaceDataset(episodes, norm)
    return result


def train_geometry(config, *, seed, output_dir=None, dry_run=False, resume=False):
    from dexvision.evaluation.place_repair_policy import repair_matrix

    spec, inputs = load_geometry_spec(config)
    if seed not in spec['seeds']:
        raise ValueError('Seed outside frozen place geometry recipe')
    datasets = geometry_datasets(spec, inputs)
    if dry_run:
        return {split: {'episodes': len(data.episodes), 'frames': len(data)}
                for split, data in datasets.items()}
    validation = repair_matrix(spec, inputs, 'validation')
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
            'dexvision/learning/place_geometry_policy.py',
            'dexvision/evaluation/place_geometry_policy.py',
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
        output_dir=directory, config_snapshot={'place_geometry_recipe': spec}, resume=resume)
    return {key: value for key, value in report.items() if key != 'history'}


class GeometryPolicy:
    """Learn when to move and release; use typed goal geometry for direction."""

    def __init__(self, checkpoint_path, config):
        spec, inputs = load_geometry_spec(config)
        checkpoint = load_checkpoint(checkpoint_path)
        manifest = checkpoint['manifest']
        if (manifest['recipe'] != spec or manifest['recipe_sha256'] != digest_file(Path(config))
                or manifest['seed'] not in spec['seeds']):
            raise ValueError('Place geometry bundle mismatch')
        for name, digest in manifest['implementation_sha256'].items():
            if digest_file(inputs.root / name) != digest:
                raise ValueError('Place geometry execution code changed after training')
        self.model = SkillMLP(checkpoint['schema'])
        self.model.load_state_dict(checkpoint['model'], strict=True)
        self.model.eval()
        self.spec, self.inputs = spec, inputs
        self.names = list(feature_schema(inputs.plan, inputs.skill)[0])
        self.plan = place_action_plan(spec, inputs)
        self.object_id = None
        self.reset()

    def bind(self, row):
        self.object_id = row['entity_id']

    def reset(self):
        self.latch = None
        self.offset = None
        self.stage = 'raise'

    def _waypoint(self, object_position, target_position, prior):
        geometry = self.spec['geometry']
        family = self.family
        transport_z = geometry['transport_object_height_by_object_m'].get(self.object_id,
            geometry['transport_object_height_by_family_m'].get(family,
                geometry['transport_object_height_default_m']))
        over_target = prior.copy()
        over_target[:2] += target_position[:2] - object_position[:2]
        desired = target_position.copy()
        desired[:2] += geometry['target_offset_xy_by_family_m'][family]
        desired[0] = min(desired[0], geometry['maximum_placement_center_x_m'])
        desired[2] = max(target_position[2],
                         geometry['resting_height_by_object_m'][self.object_id]) + \
            geometry['placement_clearance_by_family_m'].get(family,
                geometry['placement_clearance_default_m'])
        placement = desired - self.offset
        while True:
            if self.stage == 'raise':
                if object_position[2] >= transport_z - geometry['waypoint_tolerance_m']:
                    self.stage = 'planar'
                    continue
                waypoint = prior.copy()
                waypoint[2] += max(0., transport_z - object_position[2])
                return waypoint, geometry['transport_step_m']
            if self.stage == 'planar':
                if np.linalg.norm(over_target[:2] - prior[:2]) <= geometry['planar_tolerance_m']:
                    self.stage = 'lower'
                    continue
                return over_target, geometry['transport_step_m']
            return placement, geometry['descent_step_m']

    def __call__(self, source, prior, phase=None):
        from dexvision.evaluation.place_repair import release_fingers

        if self.object_id is None:
            raise ValueError('Place geometry policy needs a typed object request')
        features, _ = place_features(source, self.names, self.spec)
        with torch.inference_mode():
            prediction = self.model(torch.from_numpy(features.astype(np.float32)[None]))[0].numpy()
        if not np.isfinite(prediction).all():
            return np.full(27, np.nan), prediction
        prior = np.asarray(prior, float)
        index = {name: i for i, name in enumerate(self.names)}
        object_position = source[[index[f'entity.position_m.{j}'] for j in range(3)]]
        target_position = source[[index[f'goal.target_position.{j}'] for j in range(3)]]
        family = next((kind for kind in ('cuboid', 'cylinder', 'flat_puck')
            if source[index[f'entity.class.{kind}']] > .5), None)
        if family is None:
            return np.full(27, np.nan), prediction
        self.family = family
        if self.offset is None:
            self.offset = object_position - prior[:3]
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
        if held and distance <= self.spec['terminal']['release_distance_m']:
            error = target_position - object_position
            step = min(max(0., distance - self.spec['alignment_stop_distance_m']),
                       self.spec['alignment_step_m'])
            out[:3] = prior[:3] + error / max(distance, 1e-12) * step
        elif prediction[0] > self.spec['geometry']['move_gate_threshold']:
            waypoint, step = self._waypoint(object_position, target_position, prior[:3])
            delta = waypoint - prior[:3]
            length = float(np.linalg.norm(delta))
            out[:3] = prior[:3] + delta / max(length, 1e-12) * min(length, step)
        out[:3] = np.clip(out[:3], self.spec['workspace_bounds']['min'],
                          self.spec['workspace_bounds']['max'])
        return out, prediction
