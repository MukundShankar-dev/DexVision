"""Versioned unassisted pick-teacher feasibility before additive data collection."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import yaml

from dexvision.evaluation.evaluate_skill import SAFETY_KEYS, apply_action, reach_features, safe_action
from dexvision.evaluation.pick_policy import PickEvidence, physical_evidence_world, pick_preconditions
from dexvision.learning.pick_policy import load_pick_spec, pick_action_plan
from dexvision.learning.train_skill import _checksum_index, _json
from dexvision.learning.training_manifest import content_digest, digest_file
from dexvision.logging.level4_collection import WorkcellPilotTask, sample_level4_procedural_variation
from dexvision.sim.level4_expert import (
    DeterministicGraspLiftConfig, DeterministicGraspLiftExpert, DeterministicPickPlaceExpert,
    DeterministicPlaceConfig, WaypointValidationResult, _ExpertTaskView,
)
from dexvision.sim.scripted_expert_settings import resolve_scripted_expert_settings


def load_teacher_spec(path):
    path = Path(path)
    if path.with_suffix('.sha256').read_text().strip() != f'{digest_file(path)}  {path.name}':
        raise ValueError('Teacher recipe checksum mismatch')
    recipe = yaml.safe_load(path.read_text())
    if recipe['checkpoint'] != '5.3A' or digest_file(Path(recipe['parent_recipe'])) != recipe['parent_sha256']:
        raise ValueError('Teacher recipe checkpoint/parent mismatch')
    if recipe['grasp_overrides']['orientation_preservation_policy'] != 'ordinary_physics_no_object_pose_writes':
        raise ValueError('New teacher must use ordinary unassisted physics')
    spec, inputs = load_pick_spec(recipe['parent_recipe'])
    return recipe, spec, inputs


def build_teacher(task, recipe, spec, inputs):
    """Use the frozen fallback templates but qualify every candidate without holds."""
    if recipe.get('exact_controller_preflight'):
        return build_exact_teacher(task, recipe, spec, inputs)
    names = [f['name'].split('/')[-1] for f in inputs.plan['action']['named_layout'][7:]]
    settings = resolve_scripted_expert_settings(task, 'scripted_grasp')
    settings.update(recipe['grasp_overrides'])
    expert = DeterministicPickPlaceExpert(
        open_finger_targets=dict(zip(names, spec['closure']['open_targets'], strict=True)),
        closed_finger_targets=dict(zip(names, spec['closure']['closed_targets'], strict=True)),
        grasp_config=DeterministicGraspLiftConfig.from_mapping(settings),
        place_config=DeterministicPlaceConfig.from_mapping(resolve_scripted_expert_settings(task, 'scripted_place')),
        validate_on_reset=False)
    expert.reset(task, task.initial_world_state)
    # The standalone grasp controller never transitions into place.
    return expert._grasp, expert.validation, expert.grasp_config


def build_exact_teacher(task, recipe, spec, inputs):
    """Select the first candidate passing the exact live loop on a fresh state copy."""
    names = [f['name'].split('/')[-1] for f in inputs.plan['action']['named_layout'][7:]]
    settings = resolve_scripted_expert_settings(task, 'scripted_grasp')
    settings.update(recipe['grasp_overrides'])
    base = DeterministicGraspLiftConfig.from_mapping(settings)
    object_id = task.goal['object_id']
    family = next(o.family for o in task.workcell.config.objects if o.object_id == object_id)
    candidates = [base, *[replace(base, family_templates={**base.family_templates, family: t})
        for t in (base.family_fallback_templates or {}).get(family, ())]]

    def make(live, config):
        view = _ExpertTaskView(skill_name='pick_object', goal={'object_id': object_id},
            workcell=live.workcell, collection_config=live.collection_config)
        expert = DeterministicGraspLiftExpert(
            open_finger_targets=dict(zip(names, spec['closure']['open_targets'], strict=True)),
            closed_finger_targets=dict(zip(names, spec['closure']['closed_targets'], strict=True)),
            config=config, validate_on_reset=False)
        expert.reset(view, live.initial_world_state)
        return expert

    audits = []
    selected, validation = base, WaypointValidationResult(False, 'no_unassisted_candidate', 0, 0.)
    for candidate in candidates:
        with WorkcellPilotTask(workcell_config=inputs.root / inputs.plan['inputs']['workcell_config']['path'],
                dataset_config=inputs.root / inputs.plan['inputs']['dataset_config']['path'],
                skill_name='pick_place_sequence', goal_condition_id=task.goal_condition_id, seed=int(task.workcell._seed),
                procedural_variation=task.procedural_variation) as scratch:
            if not np.array_equal(scratch.env.data.qpos, task.env.data.qpos):
                raise ValueError('Teacher preflight must start from the exact same initial state')
            expert = make(scratch, candidate)
            result, _ = execute_teacher(scratch, expert, expert.validation, recipe, spec, inputs)
        audits.append({'template': asdict(candidate.family_templates[family]), **result})
        if result['success'] and not any(result['safety_counters'].values()):
            selected = candidate
            validation = WaypointValidationResult(True, None, result['action_count'], 0.)
            break
    expert = make(task, selected)
    expert.preflight_candidates = audits
    return expert, validation, selected


def execute_teacher(task, expert, preflight, recipe, spec, inputs):
    """One identical ordinary-physics loop for scratch preflight and live data."""
    initial = task.initial_world_state
    object_id = task.goal['object_id']
    row = {'entity_id': object_id, 'typed_goal': task.goal}
    plan = pick_action_plan(spec, inputs)
    arrays = defaultdict(list)
    evidence = PickEvidence(object_id, initial.require_entity(object_id).position[2], spec['terminal'])
    prior = np.r_[initial.robot.base_position, initial.robot.base_orientation_wxyz, spec['closure']['open_targets']]
    counters = dict.fromkeys(SAFETY_KEYS, 0)
    reason, success, entry, metrics = 'preflight_' + str(preflight.reason), False, None, {}
    for key, value in [('qpos', task.env.data.qpos), ('qvel', task.env.data.qvel), ('timestamps', task.env.data.time)]:
        arrays[key].append(np.asarray(value).copy())
    if preflight.valid:
        reason = 'timeout'
        for index in range(recipe['maximum_actions']):
            world = task.workcell.get_world_state()
            if (recipe.get('orientation_reference') == 'skill_entry' and entry is None
                    and expert._waypoint_index == len(expert.waypoints) - 1
                    and np.linalg.norm(expert._last_requested_position - expert.waypoints[-1]) <= expert.config.waypoint_tolerance_m):
                expert._initial_object_orientation = np.asarray(world.require_entity(object_id).orientation_wxyz)
            raw, phase, done, controller_reason = expert.step(world)
            if phase != 'approach' and entry is None:
                entry = index
                checks = pick_preconditions(world, row, spec)
                task._pick_task.reset_dwell()
                if not all(checks.values()):
                    reason = 'rejected_precondition'
                    break
            if controller_reason:
                reason = 'teacher_' + controller_reason
                break
            action, mask, counts, unsafe = safe_action(raw.as_array(), prior, plan)
            for key in counters:
                counters[key] += counts[key]
            if recipe.get('record_causal_features'):
                held = world.relation_for(object_id).held_by is not None
                height = world.require_entity(object_id).position[2] - initial.require_entity(object_id).position[2]
                causal_phase = 'stabilize' if held and height >= spec['lift_height_m'] else 'lift' if held else 'acquire'
                previous_mask = arrays['safety_masks'][-1] if arrays['safety_masks'] else np.zeros(27)
                arrays['observations'].append(reach_features(task, world, row, inputs, prior, prior, previous_mask, causal_phase))
                arrays['prior_applied_actions'].append(prior.copy())
            apply_action(task, action, plan['action']['named_layout'])
            world = task.workcell.step(n_steps=plan['action']['simulation_steps_per_action'])
            if recipe.get('physical_evidence_state') == 'recomputed_post_integration_copy':
                world = physical_evidence_world(task)
            for key, value in [('qpos', task.env.data.qpos), ('qvel', task.env.data.qvel),
                    ('timestamps', task.env.data.time), ('actions', action), ('safety_masks', mask)]:
                arrays[key].append(np.asarray(value).copy())
            arrays['phases'].append(phase)
            board = inputs.requirements['workcell']['board_workspace']
            xy = np.asarray(world.require_entity(object_id).position[:2])
            off_board = (np.any(xy < np.asarray(board['min_xy_m']) + board['safe_edge_margin_m'])
                         or np.any(xy > np.asarray(board['max_xy_m']) - board['safe_edge_margin_m']))
            counters['object_workspace_violations'] += int(off_board)
            if unsafe or off_board:
                reason = unsafe or 'object_workspace_violation'
                break
            if entry is not None:
                metrics = evidence.update(world)
                historical = task._pick_task.evaluate(world)
                if metrics['slip_drop'] or metrics['wrong_object_held']:
                    reason = 'slip_drop' if metrics['slip_drop'] else 'wrong_object_held'
                    break
                if metrics['success'] and historical.success:
                    success, reason = True, 'physical_lift_and_stable_dwell_passed'
                    break
            prior = action
    return {'success': success, 'reason': reason, 'entry_frame': entry,
            'action_count': len(arrays['actions']), 'safety_counters': counters, 'metrics': metrics}, arrays


def unassisted_pick_prefix(task, recipe, spec, inputs):
    """Apply only the chosen open-hand approach; no acquire action is queried."""
    if not recipe.get('exact_controller_preflight'):
        raise ValueError('Unassisted setup requires the identical controller-loop preflight')
    expert, validation, selected = build_exact_teacher(task, recipe, spec, inputs)
    audit = {'preflight': asdict(validation), 'selected_grasp_config': asdict(selected),
        'preflight_candidates': expert.preflight_candidates, 'object_pose_writes_after_reset': 0,
        'live_acquire_actions_queried': 0, 'live_acquire_actions_applied': 0,
        'physics': 'ordinary in scratch preflight and live approach; no object orientation hold'}
    if not validation.valid:
        return np.empty((0, 27)), {**audit, 'setup_failed': True}
    initial = task.initial_world_state
    prior = np.r_[initial.robot.base_position, initial.robot.base_orientation_wxyz, spec['closure']['open_targets']]
    plan, actions = pick_action_plan(spec, inputs), []
    for _ in range(recipe['maximum_actions']):
        if (expert._waypoint_index == len(expert.waypoints) - 1 and
                np.linalg.norm(expert._last_requested_position - expert.waypoints[-1]) <= expert.config.waypoint_tolerance_m):
            return np.asarray(actions), {**audit, 'setup_failed': False, 'prefix_frames': len(actions)}
        raw, phase, _, reason = expert.step(task.workcell.get_world_state())
        if reason or phase != 'approach':
            raise ValueError(f'Unassisted setup crossed entry or failed: {phase}/{reason}')
        action, _, _, unsafe = safe_action(raw.as_array(), prior, plan)
        if unsafe or not np.allclose(action[7:], spec['closure']['open_targets'], atol=1e-12):
            raise ValueError('Unassisted setup must contain safe open-hand approach commands only')
        apply_action(task, action, plan['action']['named_layout'])
        task.workcell.step(n_steps=plan['action']['simulation_steps_per_action'])
        prior = action
        actions.append(action.copy())
    raise ValueError('Unassisted setup approach timed out')


def teacher_attempt(recipe, spec, inputs, cell, seed, boundary, output_dir):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    variation = sample_level4_procedural_variation(inputs.requirements['level4_5b_procedural_expansion'],
        skill_name='pick_place_sequence', seed=seed, boundary_case=boundary)
    with WorkcellPilotTask(workcell_config=inputs.root / inputs.plan['inputs']['workcell_config']['path'],
            dataset_config=inputs.root / inputs.plan['inputs']['dataset_config']['path'],
            skill_name='pick_place_sequence', goal_condition_id=cell, seed=seed,
            procedural_variation=variation) as task:
        object_id = task.goal['object_id']
        family = next(o.family for o in task.workcell.config.objects if o.object_id == object_id)
        metadata = task.metadata_task_config()
        expert, preflight, selected_config = build_teacher(task, recipe, spec, inputs)
        outcome, arrays = execute_teacher(task, expert, preflight, recipe, spec, inputs)
        result = {'seed': seed, 'cell': cell, 'family': family, 'object_id': object_id,
            'variation': variation, **outcome, 'preflight': asdict(preflight),
            'selected_config': asdict(selected_config),
            'preflight_candidates': getattr(expert, 'preflight_candidates', []),
            'object_pose_writes_after_reset': 0, 'qualification_scope': 'teacher feasibility, not learned-policy qualification'}
        _json(directory / 'metadata.json', metadata)
    np.savez_compressed(directory / 'trajectory.npz', **{k: np.asarray(v) for k, v in arrays.items()})
    _json(directory / 'result.json', result)
    _checksum_index(directory)
    return result


def qualify_teacher(config, output_dir):
    recipe, spec, inputs = load_teacher_spec(config)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    records = [{'cell': cell, 'seed': recipe['seed_start'] + i * 100 + j, 'boundary': bool(j % 2)}
        for i, cell in enumerate(recipe['cells']) for j in range(recipe['resets_per_cell'])]
    _json(directory / 'manifest.json', {'recipe': recipe, 'recipe_digest': content_digest(recipe),
        'input_digests': inputs.digests, 'records': records, 'purpose': 'development feasibility; not held-out confirmation'})
    results = []
    for record in records:
        result = teacher_attempt(recipe, spec, inputs, **record, output_dir=directory / str(record['seed']))
        results.append(result)
        print(f"Teacher {record['seed']} {result['object_id']}: {result['reason']}", flush=True)
    families = {f: [r for r in results if r['family'] == f] for f in {r['family'] for r in results}}
    success = sum(r['success'] for r in results) / len(results)
    family_rates = {f: sum(r['success'] for r in rows) / len(rows) for f, rows in families.items()}
    safety = sum(sum(r['safety_counters'].values()) for r in results)
    gates = {'success': success >= recipe['gates']['success_min'],
        'worst_family': min(family_rates.values()) >= recipe['gates']['worst_family_success_min'],
        'safety': safety <= recipe['gates']['safety_violations_max']}
    report = {'success_rate': success, 'family_rates': family_rates, 'safety_violations': safety,
        'reasons': dict(Counter(r['reason'] for r in results)), 'gates': gates, 'passed': all(gates.values()),
        'results': results, 'learned_policy_qualified': False, 'training_data_collected': False}
    _json(directory / 'report.json', report)
    _checksum_index(directory)
    return {k: v for k, v in report.items() if k != 'results'}
