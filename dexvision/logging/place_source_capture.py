"""Causal place-only payload from an exact replay of scripted source actions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dexvision.evaluation.evaluate_skill import apply_action, reach_features, safe_action
from dexvision.evaluation.pick_policy import physical_evidence_world
from dexvision.learning.place_policy import place_action_plan
from dexvision.learning.skill_datasets import feature_schema
from dexvision.learning.train_skill import _checksum_index, _json
from dexvision.learning.training_manifest import content_digest, digest_file
from dexvision.logging.level4_collection import WorkcellPilotTask


def capture_causal_place_trace(place, row, metadata, source_dir, output_dir, *, session_id):
    """Replay selected actions, requiring every saved physical state to match.

    The output contains only the place interval. Pick and post-pick setup are
    replayed to establish its held-object entry and excluded from supervision.
    Observations are recorded before action t; labels are the applied action t.
    """
    _, inputs = place
    source_dir = Path(source_dir)
    outcome = json.loads((source_dir / 'result.json').read_text())
    if (outcome['terminal_result'] != 'succeeded' or any(outcome['safety_counters'].values())
            or outcome['episode_id'] != row['episode_id']):
        raise ValueError('Only a successful zero-safety source can enter place supervision')
    directory = Path(output_dir)
    fields = inputs.plan['action']['named_layout']
    steps = inputs.plan['action']['simulation_steps_per_action']
    plan = place_action_plan(place[0], inputs)
    names, _ = feature_schema(inputs.plan, inputs.skill)
    arrays = {key: [] for key in ('observations', 'actions', 'prior_applied_actions',
        'safety_masks', 'phases', 'expert_phases', 'qpos', 'qvel', 'timestamps')}
    prior = None
    with WorkcellPilotTask(
            workcell_config=inputs.root / inputs.plan['inputs']['workcell_config']['path'],
            dataset_config=inputs.root / inputs.plan['inputs']['dataset_config']['path'],
            skill_name='pick_place_sequence', goal_condition_id=row['cell_id'], seed=metadata['random_seed'],
            procedural_variation=metadata['task_config'].get('procedural_variation')) as task:
        if task.goal != row['typed_goal']:
            raise ValueError('Causal place capture goal differs from source reset')
        for filename, stage in (('pick_setup_trajectory.npz', 'pick'),
                                ('place_setup_trajectory.npz', 'setup'),
                                ('trajectory.npz', 'place')):
            path = source_dir / filename
            if not path.is_file():
                raise ValueError(f'Missing source stage: {filename}')
            with np.load(path, allow_pickle=False) as saved:
                actions = saved['actions']
                if actions.ndim != 2 or actions.shape[1] != 27 \
                        or len(saved['qpos']) != len(actions) + 1:
                    raise ValueError(f'Invalid source stage arrays: {filename}')

                def verify_state(index):
                    for key, actual in (('qpos', task.env.data.qpos), ('qvel', task.env.data.qvel),
                                        ('timestamps', task.env.data.time)):
                        if not np.array_equal(np.asarray(actual), np.asarray(saved[key][index])):
                            raise ValueError(f'Place source replay mismatch: {filename} {index} {key}')

                verify_state(0)
                if stage == 'place':
                    if prior is None:
                        raise ValueError('Place capture needs a verified held-object setup')
                    world = physical_evidence_world(task)
                    relation = world.relation_for(row['entity_id'])
                    if relation.held_by != 'rh_palm' or relation.supported_by is not None:
                        raise ValueError('Place capture entry is not physically held')
                    for key, value in (('qpos', task.env.data.qpos), ('qvel', task.env.data.qvel),
                                       ('timestamps', task.env.data.time)):
                        arrays[key].append(np.asarray(value).copy())
                for index, action in enumerate(actions):
                    if stage == 'place':
                        world = physical_evidence_world(task)
                        phase = task._phase
                        feature = reach_features(task, world, row, inputs, prior, prior,
                            np.zeros(27), phase)
                        if feature.shape != (len(names),):
                            raise ValueError('Causal place feature shape differs from frozen schema')
                        applied, mask, count, unsafe = safe_action(action, prior, plan)
                        if unsafe or any(count.values()) or not np.allclose(
                                applied, action, atol=1e-12, rtol=0):
                            raise ValueError('Place source action conflicts with frozen safety bounds')
                        arrays['observations'].append(feature)
                        arrays['actions'].append(action.copy())
                        arrays['prior_applied_actions'].append(prior.copy())
                        arrays['safety_masks'].append(mask.copy())
                        arrays['phases'].append(phase)
                        arrays['expert_phases'].append(saved['phases'][index])
                    apply_action(task, action, fields)
                    if stage == 'pick':
                        task.workcell.step(n_steps=steps)
                    else:
                        task.step(n_steps=steps)
                    verify_state(index + 1)
                    prior = action.copy()
                    if stage == 'place':
                        for key, value in (('qpos', task.env.data.qpos), ('qvel', task.env.data.qvel),
                                           ('timestamps', task.env.data.time)):
                            arrays[key].append(np.asarray(value).copy())
        final_world = physical_evidence_world(task)
        final_object = final_world.require_entity(row['entity_id'])
        if not np.allclose(final_object.position, outcome['final_physical_context']['object_position_m'],
                           atol=1e-12, rtol=0):
            raise ValueError('Replayed final place pose differs from source result')
    payload = {key: np.asarray(value) for key, value in arrays.items()}
    if payload['observations'].shape != (outcome['place_actions'], len(names)) \
            or payload['actions'].shape != (outcome['place_actions'], 27):
        raise ValueError('Incomplete causal place payload')
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory / 'trajectory.npz', **payload)
    _json(directory / 'metadata.json', {'episode_id': row['episode_id'], 'session_id': session_id,
        'cell_id': row['cell_id'], 'random_seed': metadata['random_seed'], 'typed_goal': row['typed_goal'],
        'procedural_variation': metadata['task_config'].get('procedural_variation'),
        'source_result_sha256': digest_file(source_dir / 'result.json'),
        'feature_schema_sha256': content_digest(list(names)),
        'place_start': 'physically_held_after_unassisted_pick_and_bounded_setup',
        'source_prefix_excluded_from_supervision': True})
    _json(directory / 'result.json', {'episode_id': row['episode_id'], 'session_id': session_id,
        'terminal_result': outcome['terminal_result'], 'terminal_reason': outcome['terminal_reason'],
        'place_actions': outcome['place_actions'], 'safety_counters': outcome['safety_counters'],
        'source_result_sha256': digest_file(source_dir / 'result.json'),
        'source_replay_exact': True, 'causal_observations': True,
        'physical_final_context': outcome['final_physical_context']})
    _checksum_index(directory)
    return {'episode_id': row['episode_id'], 'frames': outcome['place_actions'],
            'source_replay_exact': True, 'payload_sha256': digest_file(directory / 'trajectory.npz')}
