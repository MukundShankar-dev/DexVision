"""Versioned unassisted place-teacher feasibility for the 5.4A repair."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from dexvision.evaluation.evaluate_skill import SAFETY_KEYS, apply_action, safe_action
from dexvision.evaluation.pick_policy import physical_evidence_world
from dexvision.evaluation.pick_teacher import build_teacher, execute_teacher, load_teacher_spec
from dexvision.evaluation.place_policy import PlaceEvidence, place_action_plan, place_scenarios, read_place_entries
from dexvision.learning.place_policy import load_place_spec
from dexvision.learning.train_skill import _checksum_index, _json
from dexvision.learning.training_manifest import content_digest, digest_file
from dexvision.logging.level4_collection import WorkcellPilotTask, sample_level4_procedural_variation
from dexvision.sim.level4_expert import (DeterministicPlaceConfig, DeterministicPlaceExpert,
    _bounded_quaternion_step, _quaternion_multiply)
from dexvision.sim.scripted_expert_settings import resolve_scripted_expert_settings


def load_place_teacher_spec(path):
    path = Path(path)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError('Place teacher recipe checksum mismatch')
    recipe = yaml.safe_load(path.read_text())
    if recipe.get('version') not in {'level5/place-teacher-v1', 'level5/place-teacher-v2',
                                     'level5/place-teacher-v3', 'level5/place-teacher-v4'} \
            or recipe.get('checkpoint') != '5.4A':
        raise ValueError('Unsupported place teacher recipe')
    for key, digest_key in (('place_recipe', 'place_recipe_sha256'), ('pick_teacher_recipe', 'pick_teacher_recipe_sha256')):
        if digest_file(Path(recipe[key])) != recipe[digest_key]:
            raise ValueError(f'Place teacher {key} changed')
    place_spec, place_inputs = load_place_spec(recipe['place_recipe'])
    pick_recipe, pick_spec, pick_inputs = load_teacher_spec(recipe['pick_teacher_recipe'])
    limits = recipe['release_distance_by_family_m']
    if set(limits) != {'cuboid', 'cylinder', 'flat_puck'} or any(
            not 0 < value < place_spec['terminal']['release_distance_m'] for value in limits.values()):
        raise ValueError('Place teacher release distances must remain inside frozen release gate')
    if not 0 < recipe['placement_clearance_m'] < 0.07 or not 0 < recipe['alignment_step_m'] <= 0.002:
        raise ValueError('Invalid place teacher geometry')
    if recipe['version'].endswith(('v2', 'v3', 'v4')) and not 0 < recipe.get('release_step_scale', 0) <= 1:
        raise ValueError('Place teacher requires a bounded release-step scale')
    if recipe['version'].endswith('v3'):
        _validate_place_teacher_v3(recipe)
    if recipe['version'].endswith('v4'):
        _validate_place_teacher_v4(recipe)
    return recipe, (place_spec, place_inputs), (pick_recipe, pick_spec, pick_inputs)


def _validate_place_teacher_v3(recipe):
    setup = recipe['post_pick_setup']
    if setup['pick_lift_distance_by_object_m'] != {'block_large': 0.10}:
        raise ValueError('Place V3 pick-lift override must be declared exactly')
    if setup['tighten_excess_fraction_by_object'] != {'block_large': 0.10}:
        raise ValueError('Place V3 grip tightening must be declared exactly')
    if not 0 < setup['finger_step_rad'] <= 0.025 or not 0 < setup['yaw_step_rad'] <= 0.015:
        raise ValueError('Place V3 post-pick steps exceed audited bounds')
    if not 0 < setup['maximum_actions_per_stage'] <= 100:
        raise ValueError('Invalid Place V3 setup action limit')
    for angles in setup['yaw_degrees_by_object_target'].values():
        if any(not -25 <= value <= 0 for value in angles.values()):
            raise ValueError('Place V3 post-pick yaw exceeds audited range')
    inboard = recipe['inboard_raise']
    if inboard['family'] != 'flat_puck' or not -0.15 < inboard['below_object_x_m'] < -0.12 \
            or not 0 < inboard['shift_x_m'] <= 0.005:
        raise ValueError('Invalid Place V3 inboard raise')
    overrides = recipe['place_config_overrides']
    for values in (overrides['transport_height_by_family_m'],
                   overrides['transport_height_by_object_m']):
        if any(not 0.07 <= value <= 0.12 for value in values.values()):
            raise ValueError('Invalid Place V3 transport height')
    if any(not 0.01 <= value < 0.07 for value in
           overrides['placement_clearance_by_family_m'].values()):
        raise ValueError('Invalid Place V3 placement clearance')


def _validate_place_teacher_v4(recipe):
    setup = recipe['post_pick_setup']
    if setup['pick_lift_distance_by_object_m'] != {'block_large': 0.10}:
        raise ValueError('Invalid Place V4 pick-lift override')
    fractions = setup['tighten_excess_fraction_by_object']
    if set(fractions) != {'block_large', 'cylinder_tall'} or any(
            not 0 < value <= 0.2 for value in fractions.values()):
        raise ValueError('Invalid Place V4 grip tightening')
    shifts = setup['tighten_translate_x_m_by_object']
    if set(shifts) != {'block_large'} or any(not 0 < value <= 0.003 for value in shifts.values()):
        raise ValueError('Invalid Place V4 coupled inward motion')
    if not 0 < setup['finger_step_rad'] <= 0.025 or not 0 < setup['yaw_step_rad'] <= 0.015 \
            or not 0 < setup['maximum_actions_per_stage'] <= 100:
        raise ValueError('Invalid Place V4 setup step')
    for angles in setup['yaw_degrees_by_object_target'].values():
        if any(not -25 <= value <= 0 for value in angles.values()):
            raise ValueError('Invalid Place V4 wrist yaw')
    scales = recipe['release_step_scale_by_family']
    if set(scales) != {'cuboid', 'cylinder', 'flat_puck'} or any(
            not 0 < value <= 1 for value in scales.values()):
        raise ValueError('Invalid Place V4 release-step scale')
    inboard = recipe['inboard_raise']
    if inboard['family'] != 'flat_puck' or not -0.15 < inboard['below_object_x_m'] < -0.12 \
            or not 0 < inboard['shift_x_m'] <= 0.005:
        raise ValueError('Invalid Place V4 inboard raise')
    overrides = recipe['place_config_overrides']
    for values in (overrides['transport_height_by_family_m'],
                   overrides['transport_height_by_object_m']):
        if any(not 0.07 <= value <= 0.12 for value in values.values()):
            raise ValueError('Invalid Place V4 transport height')
    if any(not 0.01 <= value < 0.07 for value in
           overrides['placement_clearance_by_family_m'].values()):
        raise ValueError('Invalid Place V4 placement clearance')


def release_fingers(prior, open_targets, closed_targets, release_step):
    """One source-proven rank-one opening increment, capped at the open pose."""
    prior, opened, closed = (np.asarray(value, float) for value in (prior, open_targets, closed_targets))
    if prior.shape != (20,) or opened.shape != (20,) or closed.shape != (20,):
        raise ValueError('Place release requires twenty ordered actuator targets')
    if not 0 < release_step <= 1 or not np.isfinite(np.r_[prior, opened, closed]).all():
        raise ValueError('Invalid place release step or actuator value')
    direction = opened - closed
    if np.any(direction > 1e-12):
        raise ValueError('Place release expects the audited opening direction')
    return np.maximum(prior + release_step * direction, opened)


def place_teacher_action(requested, prior, *, held, error, family, latch, recipe, place_spec,
                         open_targets, closed_targets, release_step):
    """Couple release fingers and hold the command pose until physical settling."""
    raw = np.asarray(requested, float).copy()
    prior = np.asarray(prior, float)
    error = np.asarray(error, float)
    if raw.shape != (27,) or prior.shape != (27,) or error.shape != (3,):
        raise ValueError('Invalid place teacher action or goal error')
    distance = float(np.linalg.norm(error))
    if held and distance <= recipe['release_distance_by_family_m'][family] and latch is None:
        latch = prior[:7].copy()
    if latch is not None:
        raw[:7] = latch
        raw[7:] = release_fingers(prior[7:], open_targets, closed_targets, release_step)
    else:
        raw[7:] = prior[7:]
        if held and distance <= place_spec['terminal']['release_distance_m']:
            step = min(max(0., distance - recipe['alignment_stop_distance_m']), recipe['alignment_step_m'])
            raw[:3] = prior[:3] + (error / max(distance, 1e-12)) * step
    bounded = np.clip(raw[:3], place_spec['workspace_bounds']['min'], place_spec['workspace_bounds']['max'])
    saturation = bool(np.any(bounded != raw[:3]))
    raw[:3] = bounded
    return raw, latch, saturation


def _place_pick_recipe(task, pick_recipe, recipe):
    """Bind an explicit 5.4A lift extension without editing the 5.3A teacher."""
    object_id = task.goal['object_id']
    lift = recipe['post_pick_setup']['pick_lift_distance_by_object_m'].get(object_id)
    if lift is None:
        return pick_recipe
    settings = resolve_scripted_expert_settings(task, 'scripted_grasp')
    family = next(item.family for item in task.workcell.config.objects if item.object_id == object_id)
    templates = deepcopy(settings['family_templates'])
    fallbacks = deepcopy(settings.get('family_fallback_templates', {}))
    templates[family]['lift_distance_m'] = lift
    for template in fallbacks.get(family, ()):
        template['lift_distance_m'] = lift
    runtime = deepcopy(pick_recipe)
    runtime['grasp_overrides'] = {**runtime['grasp_overrides'],
        'family_templates': templates, 'family_fallback_templates': fallbacks}
    return runtime


def _place_config(task, family, object_id, recipe):
    settings = resolve_scripted_expert_settings(task, 'scripted_place')
    changes = {'placement_clearance_m': recipe['placement_clearance_m'],
        'release_position_tolerance_m': recipe['release_position_tolerance_m']}
    if recipe['version'].endswith(('v3', 'v4')):
        overrides = recipe['place_config_overrides']
        changes['transport_object_height_m'] = overrides['transport_height_by_object_m'].get(
            object_id, overrides['transport_height_by_family_m'].get(
                family, settings['transport_object_height_m']))
        changes['placement_clearance_m'] = overrides['placement_clearance_by_family_m'].get(
            family, changes['placement_clearance_m'])
    return replace(DeterministicPlaceConfig.from_mapping(settings), **changes)


def _configure_inboard_raise(expert, world, row, family, recipe):
    """Shift only the first held-object lift waypoint, before the expert moves."""
    if not recipe['version'].endswith(('v3', 'v4')):
        return 0.0
    config = recipe['inboard_raise']
    if family != config['family'] or world.require_entity(row['entity_id']).position[0] \
            >= config['below_object_x_m']:
        return 0.0
    first = expert.waypoints[0]
    first[0] += config['shift_x_m']
    expert._waypoints = (first,)
    return config['shift_x_m']


def _post_pick_setup(task, pick_arrays, row, recipe, pick_spec, place_spec, place_inputs, counters):
    """Run bounded grip/yaw actions, checking physical retention every sample."""
    setup = recipe['post_pick_setup']
    arrays = defaultdict(list)
    arrays['actions'] = []
    arrays['phases'] = []
    prior = np.asarray(pick_arrays['actions'][-1], float).copy()
    board = place_inputs.requirements['workcell']['board_workspace']
    plan = place_action_plan(place_spec, place_inputs)
    for key, value in (('qpos', task.env.data.qpos), ('qvel', task.env.data.qvel),
                       ('timestamps', task.env.data.time)):
        arrays[key].append(np.asarray(value).copy())
    result = {'success': True, 'reason': 'verified_held_place_entry', 'action_count': 0,
        'tighten_actions': 0, 'yaw_actions': 0, 'yaw_degrees': 0}

    def step(raw, phase):
        nonlocal prior
        applied, _, count, unsafe = safe_action(raw, prior, plan)
        for key in SAFETY_KEYS:
            counters[key] += count[key]
        if unsafe:
            return unsafe
        apply_action(task, applied, plan['action']['named_layout'])
        task.step(n_steps=place_inputs.plan['action']['simulation_steps_per_action'])
        world = physical_evidence_world(task)
        arrays['actions'].append(applied.copy())
        arrays['phases'].append(phase)
        for key, value in (('qpos', task.env.data.qpos), ('qvel', task.env.data.qvel),
                           ('timestamps', task.env.data.time)):
            arrays[key].append(np.asarray(value).copy())
        result['action_count'] += 1
        result[phase + '_actions'] += 1
        prior = applied
        relation = world.relation_for(row['entity_id'])
        if relation.held_by != 'rh_palm' or relation.supported_by is not None:
            return 'physical_retention_failed'
        xy = np.asarray(world.require_entity(row['entity_id']).position[:2])
        offboard = bool(np.any(xy < np.asarray(board['min_xy_m']) + board['safe_edge_margin_m'])
            or np.any(xy > np.asarray(board['max_xy_m']) - board['safe_edge_margin_m']))
        counters['object_workspace_violations'] += int(offboard)
        if offboard:
            return 'object_workspace_violation'
        if any('workcell_table' in pair and any(name.startswith('rh_') for name in pair)
               for pair in world.contacts):
            return 'hand_table_contact'
        entities = set(task.workcell.config.object_ids) | set(task.workcell.config.fixture_ids)
        unwanted = any(any(name.startswith('rh_') for name in pair)
            and any(name in entities and name not in {row['entity_id'], row['typed_goal']['target_id']}
                    for name in pair) for pair in world.contacts)
        counters['unwanted_contact'] += int(unwanted)
        if unwanted:
            return 'unwanted_contact'
        return None

    object_id = row['entity_id']
    fraction = setup['tighten_excess_fraction_by_object'].get(object_id, 0.0)
    if fraction:
        opened = np.asarray(pick_spec['closure']['open_targets'], float)
        closed = np.asarray(pick_spec['closure']['closed_targets'], float)
        limits = task.env.model.actuator_ctrlrange
        target = np.clip(closed + fraction * (closed - opened), limits[:, 0], limits[:, 1])
        for _ in range(setup['maximum_actions_per_stage']):
            raw = prior.copy()
            raw[7:] += np.clip(target - prior[7:], -setup['finger_step_rad'], setup['finger_step_rad'])
            if recipe['version'].endswith('v4'):
                raw[0] += setup['tighten_translate_x_m_by_object'].get(object_id, 0.0)
            failure = step(raw, 'tighten')
            if failure:
                result.update(success=False, reason=failure)
                return result, prior, arrays
            if np.max(np.abs(prior[7:] - target)) < 1e-8:
                break
        else:
            result.update(success=False, reason='tighten_timeout')
            return result, prior, arrays

    degrees = setup['yaw_degrees_by_object_target'].get(object_id, {}).get(
        row['typed_goal']['target_id'], 0)
    result['yaw_degrees'] = degrees
    if degrees:
        half = math.radians(degrees) / 2
        target = _quaternion_multiply(prior[3:7],
            np.asarray([math.cos(half), 0., 0., math.sin(half)]))
        for _ in range(setup['maximum_actions_per_stage']):
            raw = prior.copy()
            raw[3:7] = _bounded_quaternion_step(prior[3:7], target, setup['yaw_step_rad'])
            failure = step(raw, 'yaw')
            if failure:
                result.update(success=False, reason=failure)
                return result, prior, arrays
            if np.linalg.norm(prior[3:7] - target) < 1e-6:
                break
        else:
            result.update(success=False, reason='yaw_timeout')
    return result, prior, arrays


def run_place_teacher_attempt(recipe, place, pick, row, metadata, output_dir):
    """Run ordinary-physics pick setup, then a place teacher without object writes."""
    place_spec, place_inputs = place
    pick_recipe, pick_spec, pick_inputs = pick
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    fields = place_inputs.plan['action']['named_layout']
    place_plan = place_action_plan(place_spec, place_inputs)
    counters = dict.fromkeys(SAFETY_KEYS, 0)
    counters['unwanted_contact'] = 0
    results = {'episode_id': row['episode_id'], 'cell_id': row['cell_id'],
               'reset_seed': metadata['random_seed'], 'source_digest': row['episode_digest']}
    with WorkcellPilotTask(
            workcell_config=place_inputs.root / place_inputs.plan['inputs']['workcell_config']['path'],
            dataset_config=place_inputs.root / place_inputs.plan['inputs']['dataset_config']['path'],
            skill_name='pick_place_sequence', goal_condition_id=row['cell_id'], seed=metadata['random_seed'],
            procedural_variation=metadata['task_config'].get('procedural_variation')) as task:
        if task.goal != row['typed_goal']:
            raise ValueError('Fresh teacher goal differs from the development reset')
        family = next(item.family for item in task.workcell.config.objects
                      if item.object_id == row['entity_id'])
        results['object_family'] = family
        results['target_id'] = row['typed_goal']['target_id']
        runtime_pick = _place_pick_recipe(task, pick_recipe, recipe) if recipe['version'].endswith(('v3', 'v4')) \
            else pick_recipe
        pick_expert, preflight, grasp_config = build_teacher(task, runtime_pick, pick_spec, pick_inputs)
        pick_result, pick_arrays = execute_teacher(task, pick_expert, preflight, runtime_pick, pick_spec, pick_inputs)
        results['pick_setup'] = pick_result
        setup_result, setup_prior = None, None
        if recipe['version'].endswith(('v3', 'v4')):
            np.savez_compressed(directory / 'pick_setup_trajectory.npz',
                **{key: np.asarray(value) for key, value in pick_arrays.items()})
            if pick_result['success']:
                initial = physical_evidence_world(task)
                relation = initial.relation_for(row['entity_id'])
                if relation.held_by == 'rh_palm' and relation.supported_by is None:
                    setup_result, setup_prior, setup_arrays = _post_pick_setup(task, pick_arrays, row,
                        recipe, pick_spec, place_spec, place_inputs, counters)
                    results['place_setup'] = setup_result
                    saved_setup = {key: np.asarray(value) for key, value in setup_arrays.items()}
                    if not len(saved_setup['actions']):
                        saved_setup['actions'] = np.empty((0, 27))
                    np.savez_compressed(directory / 'place_setup_trajectory.npz', **saved_setup)
        if not pick_result['success']:
            results.update(terminal_result='rejected_before_execution', terminal_reason='pick_setup_failed',
                           safety_counters=counters, place_actions=0)
        elif setup_result is not None and not setup_result['success']:
            results.update(terminal_result='rejected_before_execution', terminal_reason='place_setup_failed',
                           safety_counters=counters, place_actions=0)
        else:
            world = physical_evidence_world(task)
            relation = world.relation_for(row['entity_id'])
            if relation.held_by != 'rh_palm' or relation.supported_by is not None:
                results.update(terminal_result='rejected_before_execution', terminal_reason='held_precondition_failed',
                               safety_counters=counters, place_actions=0)
            else:
                names = [field['name'].split('/')[-1] for field in fields[7:]]
                opened = np.asarray(pick_spec['closure']['open_targets'], float)
                closed = np.asarray(pick_spec['closure']['closed_targets'], float)
                config = _place_config(task, family, row['entity_id'], recipe)
                expert = DeterministicPlaceExpert(open_finger_targets=dict(zip(names, opened, strict=True)),
                    closed_finger_targets=dict(zip(names, closed, strict=True)), grasp_config=grasp_config,
                    config=config, validate_on_reset=False)
                expert.reset(task, world)
                shift = _configure_inboard_raise(expert, world, row, family, recipe)
                if recipe['version'].endswith(('v3', 'v4')):
                    results['inboard_raise_shift_m'] = shift
                prior = setup_prior if setup_prior is not None else np.asarray(pick_arrays['actions'][-1], float).copy()
                evidence = PlaceEvidence(row['entity_id'], row['typed_goal']['target_id'],
                    place_spec['terminal'], world.require_entity(row['entity_id']).position)
                latch, metrics, terminal, reason = None, {}, 'timed_out', 'unsettled'
                last_context = {}
                arrays = defaultdict(list)
                for key, value in (('qpos', task.env.data.qpos), ('qvel', task.env.data.qvel),
                                   ('timestamps', task.env.data.time)):
                    arrays[key].append(np.asarray(value).copy())
                for _ in range(recipe['maximum_place_actions']):
                    world = physical_evidence_world(task)
                    obj = np.asarray(world.require_entity(row['entity_id']).position)
                    target = np.asarray(world.require_entity(row['typed_goal']['target_id']).position)
                    held = world.relation_for(row['entity_id']).held_by == 'rh_palm'
                    last_context = {'object_position_m': obj.tolist(), 'target_position_m': target.tolist(),
                        'distance_m': float(np.linalg.norm(target - obj)), 'held': held,
                        'command_position_m': prior[:3].tolist(), 'latch_active': latch is not None}
                    if held and np.linalg.norm(target - obj) <= recipe['release_distance_by_family_m'][family] \
                            and latch is None:
                        latch = prior[:7].copy()
                    if latch is None:
                        requested, phase, _, expert_reason = expert.step(world)
                        if expert_reason:
                            terminal, reason = 'failed', 'teacher_' + expert_reason
                            break
                        raw = requested.as_array()
                    else:
                        phase, raw = 'latched_release', prior.copy()
                    raw, latch, saturated = place_teacher_action(raw, prior, held=held, error=target-obj,
                        family=family, latch=latch, recipe=recipe, place_spec=place_spec,
                        open_targets=opened, closed_targets=closed,
                        release_step=config.release_synergy_step *
                            recipe.get('release_step_scale_by_family', {}).get(
                                family, recipe.get('release_step_scale', 1.0)))
                    counters.setdefault('decoder_workspace_saturations', 0)
                    counters['decoder_workspace_saturations'] += int(saturated)
                    applied, mask, count, unsafe = safe_action(raw, prior, place_plan)
                    for key in SAFETY_KEYS:
                        counters[key] += count[key]
                    if unsafe:
                        terminal, reason = 'failed', unsafe
                        break
                    apply_action(task, applied, fields)
                    task.step(n_steps=place_inputs.plan['action']['simulation_steps_per_action'])
                    world = physical_evidence_world(task)
                    frozen = SimpleNamespace(values=task.current_state.task_values, success=task.current_state.success)
                    metrics = evidence.update(world, frozen)
                    arrays['actions'].append(applied.copy())
                    arrays['phases'].append(phase)
                    for key, value in (('qpos', task.env.data.qpos), ('qvel', task.env.data.qvel),
                                       ('timestamps', task.env.data.time)):
                        arrays[key].append(np.asarray(value).copy())
                    board = place_inputs.requirements['workcell']['board_workspace']
                    xy = np.asarray(world.require_entity(row['entity_id']).position[:2])
                    off_board = bool(np.any(xy < np.asarray(board['min_xy_m']) + board['safe_edge_margin_m'])
                        or np.any(xy > np.asarray(board['max_xy_m']) - board['safe_edge_margin_m']))
                    counters['object_workspace_violations'] += int(off_board)
                    entities = set(task.workcell.config.object_ids) | set(task.workcell.config.fixture_ids)
                    unwanted = any(any(name.startswith('rh_') for name in pair)
                        and any(name in entities and name not in {row['entity_id'], row['typed_goal']['target_id']}
                                for name in pair) for pair in world.contacts)
                    counters['unwanted_contact'] += int(unwanted)
                    failure = ('object_workspace_violation' if off_board else 'unwanted_contact' if unwanted else
                        'premature_release' if evidence.premature_release else 'bounce' if evidence.bounced else None)
                    prior = applied
                    if failure:
                        terminal, reason = 'failed', failure
                        break
                    if metrics['success']:
                        terminal, reason = 'succeeded', 'physical_settled_placement_passed'
                        break
                np.savez_compressed(directory / 'trajectory.npz', **{key: np.asarray(value)
                    for key, value in arrays.items()})
                final_object = world.require_entity(row['entity_id'])
                final_target = world.require_entity(row['typed_goal']['target_id'])
                results.update(terminal_result=terminal, terminal_reason=reason, object_family=family,
                    place_actions=len(arrays['actions']), final_metrics=metrics, safety_counters=counters,
                    last_pre_action_context=last_context,
                    final_physical_context={'object_position_m': final_object.position,
                        'target_position_m': final_target.position,
                        'object_held_by': world.relation_for(row['entity_id']).held_by,
                        'object_supported_by': world.relation_for(row['entity_id']).supported_by},
                    disclosure=('Unassisted scripted pick, bounded grip/yaw place setup, and place teacher; '
                        'no post-entry object pose writes' if recipe['version'].endswith(('v3', 'v4')) else
                        'Unassisted scripted pick setup and place teacher; no post-entry object pose writes'))
    _json(directory / 'result.json', results)
    _checksum_index(directory)
    return results


def qualify_place_teacher(config, output_dir):
    """Run every frozen validation start once as an exposed teacher diagnostic."""
    recipe, place, pick = load_place_teacher_spec(config)
    _, inputs = place
    scenarios = place_scenarios(inputs, 'validation')
    entries = read_place_entries(inputs, scenarios)
    unique = {}
    for row, _, _, _ in scenarios:
        unique.setdefault(row['episode_id'], row)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    results = []
    for index, row in enumerate(unique.values()):
        metadata, _, _ = entries[row['episode_id']]
        result = run_place_teacher_attempt(recipe, place, pick, row, metadata,
            directory / f"{index:02d}_{row['episode_id']}")
        results.append(result)
        print(f"{row['episode_id']}: {result['terminal_reason']}", flush=True)
    summary = summarize_teacher_results(recipe, results, config,
        'Validation resets are exposed teacher development cases, not policy qualification')
    _json(directory / 'summary.json', summary)
    _checksum_index(directory)
    return summary


def summarize_teacher_results(recipe, results, config, disclosure):
    if not results:
        raise ValueError('Place teacher summary needs attempts')
    family = {}
    for name in ('cuboid', 'cylinder', 'flat_puck'):
        group = [result for result in results if result.get('object_family') == name]
        family[name] = {'count': len(group), 'successes': sum(r['terminal_result'] == 'succeeded' for r in group)}
    if any(not item['count'] for item in family.values()):
        raise ValueError('Place teacher summary requires every object family')
    summary = {'version': recipe['version'], 'development_only': True, 'count': len(results),
        'successes': sum(r['terminal_result'] == 'succeeded' for r in results),
        'reasons': dict(Counter(r['terminal_reason'] for r in results)), 'by_family': family,
        'safety_counters': {key: sum(r['safety_counters'].get(key, 0) for r in results)
            for key in (*SAFETY_KEYS, 'unwanted_contact', 'decoder_workspace_saturations')},
        'source_recipe_sha256': digest_file(Path(config)),
        'disclosure': disclosure}
    summary['teacher_feasibility_passed'] = (summary['successes'] / len(results) >= recipe['gates']['success_min']
        and all(item['successes'] / item['count'] >= recipe['gates']['worst_family_success_min']
            for item in family.values())
        and all(value == 0 for value in summary['safety_counters'].values()))
    return summary


def qualify_place_teacher_coverage(config, output_dir):
    """Probe all six instances and five target cells on new development seeds."""
    path = Path(config)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError('Place coverage recipe checksum mismatch')
    coverage = yaml.safe_load(path.read_text())
    if coverage.get('version') not in {'level5/place-teacher-coverage-v1',
                                       'level5/place-teacher-coverage-v2',
                                       'level5/place-teacher-coverage-v3'} \
            or coverage.get('checkpoint') != '5.4A':
        raise ValueError('Unsupported place teacher coverage recipe')
    teacher_path = Path(coverage['teacher_recipe'])
    if digest_file(teacher_path) != coverage['teacher_recipe_sha256']:
        raise ValueError('Place coverage teacher recipe changed')
    recipe, place, pick = load_place_teacher_spec(teacher_path)
    _, inputs = place
    config_path = inputs.root / inputs.plan['inputs']['dataset_config']['path']
    dataset = yaml.safe_load(config_path.read_text())
    cells = {cell['id']: cell for cell in dataset['coverage_cells']
        if cell.get('data_group') == 'pick_place'}
    if set(coverage['cells']) != set(cells) or len(coverage['cells']) != 30:
        raise ValueError('Place coverage must freeze every pick/place cell exactly once')
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    results, reset_manifest = [], []
    for index, cell_id in enumerate(coverage['cells']):
        cell = cells[cell_id]
        seed = coverage['seed_start'] + index
        variation = sample_level4_procedural_variation(
            inputs.requirements['level4_5b_procedural_expansion'],
            skill_name='pick_place_sequence', seed=seed,
            boundary_case=coverage['boundary_case'])
        goal = {'object_id': cell['object_id'], 'target_id': cell['target_id']}
        metadata = {'random_seed': seed, 'task_config': {'procedural_variation': variation}}
        row = {'episode_id': f'place_teacher_coverage_{seed}', 'cell_id': cell_id,
            'entity_id': cell['object_id'], 'typed_goal': goal,
            'episode_digest': content_digest({'cell': cell_id, 'seed': seed, 'variation': variation})}
        reset_manifest.append({'episode_id': row['episode_id'], 'cell_id': cell_id,
            'seed': seed, 'goal': goal, 'variation': variation})
        result = run_place_teacher_attempt(recipe, place, pick, row, metadata,
            directory / f"{index:02d}_{row['episode_id']}")
        results.append(result)
        print(f"{row['episode_id']} {cell_id}: {result['terminal_reason']}", flush=True)
    summary = summarize_teacher_results(recipe, results, teacher_path,
        'Fresh development seeds across all cells; prior families and goal types exposed, no policy qualification')
    summary['coverage_recipe_sha256'] = digest_file(path)
    summary['target_counts'] = dict(Counter(r['target_id'] for r in results))
    summary['target_successes'] = dict(Counter(r['target_id'] for r in results
        if r['terminal_result'] == 'succeeded'))
    _json(directory / 'resets.json', {'version': coverage['version'], 'resets': reset_manifest})
    _json(directory / 'summary.json', summary)
    _checksum_index(directory)
    return summary
