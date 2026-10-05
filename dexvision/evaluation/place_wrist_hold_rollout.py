"""V24 copied-state place rollout; keeps V20 setup, metrics and saved-state format."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from dexvision.evaluation.benchmark_retargeters import mean_action_jerk
from dexvision.evaluation.evaluate_skill import SAFETY_KEYS, apply_action, reach_features, safe_action
from dexvision.evaluation.pick_policy import physical_evidence_world
from dexvision.evaluation.place_policy import PlaceEvidence, place_preconditions
from dexvision.evaluation.place_repair_policy import _setup
from dexvision.evaluation.place_wrist_hold_preview import preview_place_action
from dexvision.learning.place_policy import place_action_plan
from dexvision.learning.train_skill import _checksum_index, _json
from dexvision.logging.level4_collection import WorkcellPilotTask

def run_wrist_hold_rollout(spec, inputs, teacher, pick, row, policy, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    fields = inputs.plan['action']['named_layout']
    plan = place_action_plan(spec, inputs)
    counters = dict.fromkeys(SAFETY_KEYS, 0)
    counters.update(object_workspace_violations=0, unwanted_contact=0)
    arrays = defaultdict(list)
    terminal, reason, metrics = 'timed_out', 'unsettled', {}
    with WorkcellPilotTask(
            workcell_config=inputs.root / inputs.plan['inputs']['workcell_config']['path'],
            dataset_config=inputs.root / inputs.plan['inputs']['dataset_config']['path'],
            skill_name='pick_place_sequence', goal_condition_id=row['cell_id'], seed=row['seed'],
            procedural_variation=row['variation']) as task:
        if task.goal != row['typed_goal']:
            raise ValueError('Place repair reset goal mismatch')
        setup, prior = _setup(task, row, teacher, pick, (spec, inputs))
        _json(directory / 'setup.json', setup)
        for key, value in setup.get('safety_counters', {}).items():
            counters[key] = counters.get(key, 0) + value
        world = physical_evidence_world(task)
        object_id, target_id = row['entity_id'], row['typed_goal']['target_id']
        family = next(o.family for o in task.workcell.config.objects if o.object_id == object_id)
        evidence = PlaceEvidence(object_id, target_id, spec['terminal'], world.require_entity(object_id).position)
        checks = place_preconditions(world, row)
        for key, value in (('qpos', task.env.data.qpos), ('qvel', task.env.data.qvel),
                           ('state_timestamps', task.env.data.time)):
            arrays[key].append(np.asarray(value).copy())
        if not setup['success'] or not all(checks.values()):
            terminal, reason = 'rejected_before_execution', setup['reason'] if not setup['success'] \
                else 'rejected_precondition'
        else:
            policy.reset()
            commanded, safety = prior.copy(), np.zeros(27)
            for _ in range(teacher[0]['maximum_place_actions']):
                world = physical_evidence_world(task)
                source = reach_features(task, world, row, inputs, commanded, prior, safety, task._phase)
                raw, prediction = policy(source, prior, task._phase)
                applied, mask, count, unsafe = safe_action(raw, prior, plan)
                applied = preview_place_action(task, applied, prior, fields, inputs, row, policy)
                for key in SAFETY_KEYS:
                    counters[key] += count[key]
                for key, value in (('observations', source), ('raw_actions', raw),
                    ('applied_actions', applied), ('policy_predictions', prediction),
                    ('prior_applied_actions', prior), ('safety_masks', mask)):
                    arrays[key].append(np.asarray(value).copy())
                apply_action(task, applied, fields)
                task.step(n_steps=inputs.plan['action']['simulation_steps_per_action'])
                world = physical_evidence_world(task)
                frozen = SimpleNamespace(values=task.current_state.task_values,
                                         success=task.current_state.success)
                metrics = evidence.update(world, frozen)
                for key, value in (('qpos', task.env.data.qpos), ('qvel', task.env.data.qvel),
                                   ('state_timestamps', task.env.data.time)):
                    arrays[key].append(np.asarray(value).copy())
                board = inputs.requirements['workcell']['board_workspace']
                xy = np.asarray(world.require_entity(object_id).position[:2])
                off_board = bool(np.any(xy < np.asarray(board['min_xy_m']) + board['safe_edge_margin_m'])
                    or np.any(xy > np.asarray(board['max_xy_m']) - board['safe_edge_margin_m']))
                counters['object_workspace_violations'] += int(off_board)
                entities = set(task.workcell.config.object_ids) | set(task.workcell.config.fixture_ids)
                unwanted = sum(any(name.startswith('rh_') for name in pair) and
                    any(name in entities and name not in {object_id, target_id} for name in pair)
                    for pair in world.contacts)
                counters['unwanted_contact'] += unwanted
                metrics['unwanted_contact_count'] = unwanted
                failure = unsafe or ('object_workspace_violation' if off_board else
                    'unwanted_contact' if unwanted else
                    'premature_release' if evidence.premature_release else
                    'bounce' if evidence.bounced else None)
                prior, commanded, safety = applied, raw, mask
                if failure:
                    terminal, reason = 'failed', failure
                    break
                if metrics['success']:
                    terminal, reason = 'succeeded', 'physical_settled_placement_passed'
                    break
        end = float(task.env.data.time)
    np.savez_compressed(directory / 'trajectory.npz',
        **{key: np.asarray(value) for key, value in arrays.items()})
    bounds = np.asarray([f['upper'] - f['lower'] for f in fields])
    actions = np.asarray(arrays['applied_actions']) if arrays['applied_actions'] else np.empty((0, 27))
    result = {'episode_id': row['episode_id'], 'reset_seed': row['seed'], 'cell_id': row['cell_id'],
        'session': row['session'], 'object_family': family, 'object_instance': object_id,
        'target_id': target_id, 'target_type': inputs.skill['target_type_mapping'][target_id],
        'boundary_class': 'boundary' if row['boundary'] else 'nominal',
        'render_condition': row['render_condition'],
        'terminal_result': terminal, 'terminal_reason': reason, 'final_metrics': metrics,
        'preconditions': checks, 'setup': setup, 'safety_counters': counters,
        'premature_release': evidence.premature_release, 'nominally_placed': evidence.nominally_placed,
        'bounced': evidence.bounced, 'transported': evidence.transported,
        'released': evidence.released,
        'settled': evidence.settle_dwell >= spec['terminal']['consecutive_samples'] and not evidence.bounced,
        'normalized_terminal_error': metrics['object_to_target_distance_m'] /
            spec['terminal']['target_distance_m'] if metrics else None,
        'mean_action_jerk': mean_action_jerk(actions / bounds) if len(actions) else 0.,
        'end_timestamp': end, 'trajectory_path': 'trajectory.npz'}
    _json(directory / 'result.json', result)
    _checksum_index(directory)
    return result

