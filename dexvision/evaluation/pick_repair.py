"""Fresh pick repair confirmation; no continuation from the exposed original test."""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from dexvision.evaluation.pick_policy import (
    create_pick_task, pick_preconditions, pick_scenarios, read_pick_entries,
    run_pick_rollout, summarize_pick,
)
from dexvision.learning.pick_policy import PickPolicy, load_pick_spec
from dexvision.learning.train_skill import _checksum_index, _json, _verify_run, load_checkpoint
from dexvision.learning.training_manifest import content_digest, digest_file
from dexvision.logging.level4_collection import WorkcellPilotTask, sample_level4_procedural_variation


def read_confirmation(spec, directory):
    directory = Path(directory)
    _verify_run(directory)
    frozen = json.loads((directory / 'resets.json').read_text())
    if frozen['recipe_sha256'] != content_digest(spec):
        raise ValueError('Pick fresh resets belong to another recipe')
    return frozen


def freeze_pick_confirmation(config, output_dir, release_root=None):
    """Freeze new physics resets and source pre-acquisition prefixes, without policy actions.

    Source controls stop strictly before the original acquire boundary. Their
    reuse is declared: fresh procedural physics, not a newly learned approach.
    No reset is filtered by prefix quality or a later policy outcome.
    """
    spec, inputs = load_pick_spec(config)
    if spec['checkpoint'] != '5.3A':
        raise ValueError('Fresh confirmation is for pick repair only')
    root = Path(spec['output_dir'])
    if list(root.glob('seed_*/selected.pt')):
        raise ValueError('Freeze confirmation before any repair selection')
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    scenarios = pick_scenarios(inputs, 'test')
    entries = read_pick_entries(inputs, scenarios, release_root, spec=spec)
    by_cell = {}
    for row, _, _, _ in scenarios:
        by_cell.setdefault(row['cell_id'], row)
    records, cells = [], []
    settings = spec['confirmation']
    for cell_index, (cell_id, source_row) in enumerate(by_cell.items()):
        source_metadata, prefix = entries[source_row['episode_id']]
        cell = {'cell_id': cell_id, 'episode_ids': [], 'reset_seeds': [], 'start_frames': []}
        for repetition in range(settings['resets_per_source_cell']):
            seed = settings['fresh_reset_seed_start'] + cell_index * 100 + repetition
            variation = sample_level4_procedural_variation(inputs.requirements['level4_5b_procedural_expansion'],
                skill_name='pick_place_sequence', seed=seed, boundary_case=bool(repetition % 2))
            with WorkcellPilotTask(workcell_config=inputs.root / inputs.plan['inputs']['workcell_config']['path'],
                    dataset_config=inputs.root / inputs.plan['inputs']['dataset_config']['path'],
                    skill_name='pick_place_sequence', goal_condition_id=cell_id, seed=seed,
                    procedural_variation=variation) as task:
                metadata = copy.deepcopy(source_metadata)
                eid = f'pick_confirmation_{seed}'
                metadata.update(episode_id=eid, random_seed=seed, typed_goal=task.goal,
                    recording_session_id='evaluation_only_no_recording_session',
                    task_config=task.metadata_task_config(), initial_online_phase='approach',
                    source='fresh_evaluation_reset_source_prefix_only')
                row = {**source_row, 'episode_id': eid, 'random_seed': seed, 'typed_goal': task.goal,
                       'recording_session_id': metadata['recording_session_id'],
                       'episode_digest': content_digest(metadata)}
                setup_audit = None
                if settings.get('setup_mode') == 'unassisted_exact_teacher':
                    from dexvision.evaluation.pick_teacher import load_teacher_spec, unassisted_pick_prefix
                    if digest_file(Path(spec['teacher_recipe'])) != spec['teacher_recipe_sha256']:
                        raise ValueError('Pick confirmation teacher recipe changed')
                    teacher, _, _ = load_teacher_spec(spec['teacher_recipe'])
                    prefix, setup_audit = unassisted_pick_prefix(task, teacher, spec, inputs)
            with create_pick_task(inputs, row, metadata, prefix) as task:
                checks = pick_preconditions(task.workcell.get_world_state(), row, spec)
            records.append({'row': row, 'metadata': metadata, 'prefix_actions': prefix.tolist(),
                            'prefix_source_episode': source_row['episode_id'],
                            'prefix_source_digest': source_row['episode_digest'], 'entry_checks': checks,
                            **({'setup_audit': setup_audit} if setup_audit is not None else {})})
            cell['episode_ids'].append(eid)
            cell['reset_seeds'].append(seed)
            cell['start_frames'].append(len(prefix))
            print(f'Frozen pick reset {eid}: entry={all(checks.values())}', flush=True)
        cells.append(cell)
    conditions = inputs.config_snapshot['evaluation']['rollouts']['perception_conditions']
    frozen = {'version': 'level5/pick-fresh-confirmation-v1', 'recipe_sha256': content_digest(spec),
              'input_digests': inputs.digests, 'records': records, 'disclosure': settings['disclosure'],
              'setup': ('Exact unassisted teacher preflight and open-hand approach; no live acquire command or policy query'
                        if settings.get('setup_mode') == 'unassisted_exact_teacher' else
                        'Frozen source prefix only; fresh procedural physics; no filtering or policy execution'),
              'matrix': {'cells': cells, 'unique_reset_count': len(records),
                         'rollouts_per_track_per_training_seed': len(records) * len(conditions)}}
    _json(directory / 'resets.json', frozen)
    _checksum_index(directory)
    return {'unique_resets': len(records), 'valid_entries': sum(all(r['entry_checks'].values()) for r in records),
            'sha256': digest_file(directory / 'resets.json')}


def verify_pick_selection(run, spec, inputs):
    """Bind selected weights to full passing validation and the exact policy bundle."""
    run = Path(run)
    _verify_run(run)
    selection = json.loads((run / 'selection.json').read_text())
    checkpoint = load_checkpoint(run / 'selected.pt')
    candidate_path = run / 'validation_candidates' / f"epoch_{selection['epoch']:04d}.pt"
    candidate = load_checkpoint(candidate_path)
    PickPolicy(run / 'selected.pt', spec, inputs)
    if (selection['status'] != 'selected_from_validation'
            or digest_file(candidate_path) != selection['source_checkpoint_sha256']
            or any(not torch.equal(v, candidate['model'][k]) for k, v in checkpoint['model'].items())):
        raise ValueError('Pick selected weights differ from validation candidate')
    report = next(r for r in selection['reports'] if r['epoch'] == selection['epoch'])
    if (not report['state_numerical_gates_passed'] or report['split'] != 'validation'
            or report['recipe_sha256'] != content_digest(spec)
            or report['checkpoint_sha256'] != digest_file(candidate_path)
            or report['matrix_sha256'] != content_digest(inputs.skill['evaluation']['validation'])):
        raise ValueError('Every pick seed must pass full validation before confirmation')
    expected = summarize_pick(report['results'], inputs, 'validation')
    if any(report.get(k) != expected[k] for k in ('gates', 'state_numerical_gates_passed', 'rollout_count')):
        raise ValueError('Pick validation summary does not match complete attempts')
    return selection, checkpoint


def confirm_pick(config, run_dir, reset_dir, output_dir):
    spec, inputs = load_pick_spec(config)
    run, directory, reset_dir = Path(run_dir), Path(output_dir), Path(reset_dir)
    frozen = read_confirmation(spec, reset_dir)
    if frozen['input_digests'] != inputs.digests:
        raise ValueError('Fresh confirmation source bindings differ')
    all_selections = {}
    for seed in spec['seeds']:
        sibling = run.parent / f'seed_{seed}'
        verify_pick_selection(sibling, spec, inputs)
        all_selections[str(seed)] = digest_file(sibling / 'selected.pt')
    selection, checkpoint = verify_pick_selection(run, spec, inputs)
    policy = PickPolicy(run / 'selected.pt', spec, inputs)
    provenance = {'version': 'level5/pick-confirmation-v1', 'split': 'fresh_confirmation', 'track': 'state',
        'skill': 'pick_object', 'seed': checkpoint['manifest']['seed'], 'epoch': selection['epoch'],
        'recipe_sha256': content_digest(spec), 'input_digests': inputs.digests,
        'checkpoint_sha256': digest_file(run / 'selected.pt'), 'all_selected_sha256': all_selections,
        'reset_manifest_sha256': digest_file(reset_dir / 'resets.json'), 'output_dir': str(directory.resolve()),
        'disclosure': frozen['disclosure'], 'selection_uses_confirmation': False,
        'execution_implementation_sha256': {p.relative_to(inputs.root).as_posix(): digest_file(p)
            for p in sorted((inputs.root / 'dexvision').rglob('*.py'))}}
    reservation = run / 'fresh_confirmation_attempt.json'
    if reservation.exists():
        raise ValueError('One-time pick confirmation already reserved; do not repeat')
    directory.mkdir(parents=True, exist_ok=False)
    with reservation.open('x') as stream:
        json.dump(provenance, stream, indent=2)
    _checksum_index(run)
    _json(directory / 'provenance.json', provenance)
    results = []
    for record in frozen['records']:
        row = record['row']
        for condition in inputs.config_snapshot['evaluation']['rollouts']['perception_conditions']:
            child = directory / f"{len(results):03d}_{row['episode_id']}_{condition}"
            result = run_pick_rollout(inputs, spec, row, record['metadata'], np.asarray(record['prefix_actions']),
                policy, condition=condition, output_dir=child)
            result['result_path'] = f'{child.name}/result.json'
            results.append(result)
            if len(results) % 20 == 0:
                print(f"Pick confirmation seed {provenance['seed']}: {len(results)}/{frozen['matrix']['rollouts_per_track_per_training_seed']}", flush=True)
    skill, splits = copy.deepcopy(inputs.skill), copy.deepcopy(inputs.splits)
    skill['evaluation']['test'] = frozen['matrix']
    splits['test']['episodes'] = [r['row'] for r in frozen['records']]
    expected = ({(r['row']['episode_id'], c) for r in frozen['records']
                 for c in inputs.config_snapshot['evaluation']['rollouts']['perception_conditions']}
                if spec['confirmation'].get('setup_mode') == 'unassisted_exact_teacher' else None)
    summary = summarize_pick(results, replace(inputs, skill=skill, splits=splits), 'test', expected_attempts=expected)
    report = {**provenance, **summary, 'results': results}
    _json(directory / 'report.json', report)
    _checksum_index(directory)
    return {k: v for k, v in report.items() if k not in {'results', 'input_digests', 'execution_implementation_sha256', 'breakdowns'}}


def _load_saved_confirmation(config, run_dir, reset_dir, report_dir):
    spec, inputs = load_pick_spec(config)
    run, reset_dir, directory = map(Path, (run_dir, reset_dir, report_dir))
    for path in (run, directory):
        _verify_run(path)
    frozen = read_confirmation(spec, reset_dir)
    report = json.loads((directory / 'report.json').read_text())
    receipt = json.loads((run / 'fresh_confirmation_attempt.json').read_text())
    if (report['checkpoint_sha256'] != digest_file(run / 'selected.pt')
            or report['recipe_sha256'] != content_digest(spec)
            or report['reset_manifest_sha256'] != digest_file(reset_dir / 'resets.json')
            or receipt['checkpoint_sha256'] != report['checkpoint_sha256']
            or Path(receipt['output_dir']).resolve() != directory.resolve()
            or not report['state_numerical_gates_passed']):
        raise ValueError('Saved pick review requires passing confirmation for these exact weights and resets')
    return spec, inputs, frozen, report


def physical_pick_audit(task, trajectory, result, spec):
    """Recompute success from saved MuJoCo states without integrating physics.

    Independent of reported success/contact/height values: rebuild contacts from
    qpos/qvel and check physical retention, lift, speed and unsupported dwell.
    """
    from dexvision.evaluation.evaluate_skill import apply_action

    env = task.env
    object_id = result['object_instance']
    initial_z = task.initial_world_state.require_entity(object_id).position[2]
    commands = trajectory.get('applied_actions', np.empty((0, 27)))
    if len(trajectory['qpos']) != len(commands) + 1 or len(trajectory['qvel']) != len(commands) + 1:
        raise ValueError('Saved pick trajectory has mismatched action/state lengths')
    terminal = spec['terminal']
    consecutive, longest, acquired, gap, dropped = 0, 0, False, 0, False
    for index in range(1, len(trajectory['qpos'])):
        apply_action(task, commands[index - 1], task.collection_config['action_contract']['named_layout'])
        env.data.qpos[:] = trajectory['qpos'][index]
        env.data.qvel[:] = trajectory['qvel'][index]
        env.data.time = trajectory['state_timestamps'][index]
        env._mujoco.mj_forward(env.model, env.data)
        world = task.workcell.get_world_state()
        entity = world.require_entity(object_id)
        contact_bodies = {name for pair in world.contacts if object_id in pair for name in pair if name.startswith('rh_')}
        relation = world.relation_for(object_id)
        retained = len(contact_bodies) >= 2 and relation.held_by == 'rh_palm'
        acquired |= retained
        gap = 0 if retained else gap + int(acquired)
        dropped |= gap > terminal['maximum_retention_gap_samples']
        stable = (retained and relation.supported_by is None
                  and entity.position[2] - initial_z >= terminal['minimum_height_m']
                  and entity.linear_velocity is not None
                  and np.linalg.norm(entity.linear_velocity) <= terminal['maximum_speed_mps'])
        consecutive = consecutive + 1 if stable else 0
        longest = max(longest, consecutive)
    claimed = result['terminal_result'] == 'succeeded'
    physical_success = longest >= terminal['consecutive_samples'] and not dropped
    return {'passed': claimed == physical_success, 'claimed_success': claimed,
            'physical_success': physical_success, 'longest_stable_dwell': longest, 'drop_detected': bool(dropped)}


def audit_pick_confirmation(config, run_dir, reset_dir, report_dir, output_dir):
    spec, inputs, frozen, report = _load_saved_confirmation(config, run_dir, reset_dir, report_dir)
    records = {r['row']['episode_id']: r for r in frozen['records']}
    audits = []
    for result in report['results']:
        record = records[result['episode_id']]
        child = Path(report_dir) / Path(result['result_path']).parent
        with create_pick_task(inputs, record['row'], record['metadata'], np.asarray(record['prefix_actions'])) as task:
            with np.load(child / 'trajectory.npz', allow_pickle=False) as values:
                audit = physical_pick_audit(task, values, result, spec)
        audits.append({'episode_id': result['episode_id'], 'render_condition': result['render_condition'], **audit})
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    audit = {'passed': all(a['passed'] for a in audits), 'attempt_count': len(audits), 'audits': audits,
             'confirmation_sha256': digest_file(Path(report_dir) / 'report.json'),
             'checkpoint_sha256': report['checkpoint_sha256'], 'owner_acceptance': 'pending'}
    _json(directory / 'audit.json', audit)
    _checksum_index(directory)
    return {k: v for k, v in audit.items() if k != 'audits'}


def verify_pick_review(config, run_dir, reset_dir, report_dir):
    """Require every seed's numerical result and matching physical audit."""
    spec, inputs, frozen, report = _load_saved_confirmation(config, run_dir, reset_dir, report_dir)
    root = Path(spec['output_dir'])
    for seed in spec['seeds']:
        directory = root / f'confirmation_{seed}'
        _, _, _, sibling = _load_saved_confirmation(config, root / f'seed_{seed}', reset_dir, directory)
        audit_dir = root / f'audit_{seed}'
        _verify_run(audit_dir)
        audit = json.loads((audit_dir / 'audit.json').read_text())
        expected = {(r['episode_id'], r['render_condition']) for r in sibling['results']}
        actual = {(r['episode_id'], r['render_condition']) for r in audit['audits']}
        if (not audit['passed'] or not all(a['passed'] for a in audit['audits'])
                or audit['attempt_count'] != sibling['rollout_count']
                or len(audit['audits']) != sibling['rollout_count'] or actual != expected
                or audit['confirmation_sha256'] != digest_file(directory / 'report.json')
                or audit['checkpoint_sha256'] != sibling['checkpoint_sha256']):
            raise ValueError('Manual review requires matching passing physical audits for every seed')
    return spec, inputs, frozen, report


def review_pick_confirmation(config, run_dir, reset_dir, report_dir):
    """Replay saved confirmation states; user alone supplies visual acceptance."""
    import time
    import mujoco.viewer
    from dexvision.evaluation.evaluate_skill import apply_action

    spec, inputs, frozen, report = verify_pick_review(config, run_dir, reset_dir, report_dir)
    records = {r['row']['episode_id']: r for r in frozen['records']}
    chosen = {}
    for result in report['results']:
        key = ((result['object_instance'], result['boundary_class'], 'succeeded')
               if result['terminal_result'] == 'succeeded' else ('failure', result['terminal_reason']))
        chosen.setdefault(key, result)
    for result in chosen.values():
        record = records[result['episode_id']]
        child = Path(report_dir) / Path(result['result_path']).parent
        with create_pick_task(inputs, record['row'], record['metadata'], np.asarray(record['prefix_actions'])) as task:
            print(f"Pick seed {report['seed']} {result['object_instance']} {result['boundary_class']} "
                  f"{result['terminal_reason']}: saved physical rollout", flush=True)
            with np.load(child / 'trajectory.npz', allow_pickle=False) as values, mujoco.viewer.launch_passive(task.env.model, task.env.data) as viewer:
                viewer.cam.lookat[:] = [0.04, 0., 0.08]
                viewer.cam.distance = 0.65
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -25
                for index, (qpos, qvel) in enumerate(zip(values['qpos'], values['qvel'], strict=True)):
                    if not viewer.is_running():
                        return {'manual_acceptance': 'pending_owner_confirmation', 'cancelled': True}
                    if index:
                        apply_action(task, values['applied_actions'][index - 1], inputs.plan['action']['named_layout'])
                    task.env.data.qpos[:] = qpos
                    task.env.data.qvel[:] = qvel
                    task.env.data.time = values['state_timestamps'][index]
                    task.env._mujoco.mj_forward(task.env.model, task.env.data)
                    viewer.sync()
                    time.sleep(inputs.plan['action']['control_interval_s'])
    return {'manual_acceptance': 'pending_owner_confirmation', 'reviewed': len(chosen)}


def conditioned_pick_prefix(task, metadata, source_prefix, inputs, *, allow_legacy_yaw=False):
    """Plan an open-hand approach from the frozen family template and fresh object pose.

    Only the source prefix's final pre-acquisition command is consulted. This
    scripted setup stops before closure and never supplies a policy continuation.
    """
    from dexvision.evaluation.evaluate_skill import apply_action, safe_action
    from dexvision.learning.skill_datasets import relative_rotation
    from dexvision.sim.level4_expert import GraspFamilyTemplate, _bounded_quaternion_step, _conditioned_grasp_orientation

    if len(source_prefix) == 0:
        raise ValueError('Conditioned pick setup requires a source approach prefix')
    object_id = task.goal['object_id']
    original = metadata['task_config']['initial_state']['objects'][object_id]
    fresh = task.initial_world_state.require_entity(object_id)
    settings = metadata['teleop_config']['scripted_expert']['grasp']
    family = next(o.family for o in task.workcell.config.objects if o.object_id == object_id)
    candidates = [settings['family_templates'][family], *settings.get('family_fallback_templates', {}).get(family, [])]
    selected = None
    source_rule = 'declared_signed_yaw'
    for item in candidates:
        candidate = GraspFamilyTemplate.from_mapping(item)
        source_orientation = _conditioned_grasp_orientation(candidate, np.asarray(original['orientation_wxyz']))
        tolerance = max(1e-8, float(settings['waypoint_tolerance_m']))
        matches_position = np.allclose(np.asarray(source_prefix[-1][:3]) - original['position_m'],
            candidate.object_relative_position_m, atol=tolerance, rtol=0)
        matches_orientation = np.linalg.norm(relative_rotation(source_prefix[-1][3:7], source_orientation)) <= tolerance
        if matches_position and not matches_orientation and allow_legacy_yaw:
            w, x, y, z = original['orientation_wxyz']
            yaw = min(0., float(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))))
            historical = _conditioned_grasp_orientation(candidate, np.array([np.cos(yaw/2), 0., 0., np.sin(yaw/2)]))
            matches_orientation = np.linalg.norm(relative_rotation(source_prefix[-1][3:7], historical)) <= tolerance
            if matches_orientation:
                source_rule = 'historical_negative_yaw_only'
        if matches_position and matches_orientation:
            selected = candidate
            break
    if selected is None:
        raise ValueError('Source approach does not match a declared grasp template')
    goal = np.asarray(source_prefix[-1]).copy()
    goal[:3] = np.asarray(fresh.position) + np.asarray(selected.object_relative_position_m)
    goal[3:7] = _conditioned_grasp_orientation(selected, np.asarray(fresh.orientation_wxyz))
    prior = np.asarray(metadata['initial_applied_action']).copy()
    world = task.initial_world_state
    prior[:7] = [*world.robot.base_position, *world.robot.base_orientation_wxyz]
    if not np.allclose(goal[7:], prior[7:], atol=1e-12):
        raise ValueError('Source prefix already closes fingers; setup must stop before acquisition')
    plan = copy.deepcopy(inputs.plan)
    workspace = inputs.requirements['workcell']['safe_workspace']
    for j, field in enumerate(plan['action']['named_layout'][:3]):
        field['lower'], field['upper'] = workspace['min'][j], workspace['max'][j]
    # Same declared Level 4 approach speeds; no post-entry expert control.
    transit = max(float(settings['transit_height_m']), float(prior[2]), float(goal[2]))
    route = [(np.array([prior[0], prior[1], transit]), prior[3:7].copy(), float(settings['transit_step_m'])),
             (np.array([prior[0], prior[1], transit]), goal[3:7], float(settings['transit_step_m'])),
             (np.array([goal[0], goal[1], transit]), goal[3:7], float(settings['transit_step_m'])),
             (goal[:3], goal[3:7], float(settings['descent_step_m']))]
    actions = []
    for destination, orientation, step in route:
        for _ in range(1000):
            delta = destination - prior[:3]
            distance = float(np.linalg.norm(delta))
            angle = float(np.linalg.norm(relative_rotation(orientation, prior[3:7])))
            if distance <= 1e-12 and angle <= 1e-12:
                break
            raw = prior.copy()
            raw[:3] += delta * min(1., step / max(distance, 1e-12))
            raw[3:7] = _bounded_quaternion_step(prior[3:7], orientation,
                min(float(settings['orientation_step_rad']), plan['action']['maximum_angular_change_rad']))
            applied, _, _, unsafe = safe_action(raw, prior, plan)
            if unsafe:
                raise ValueError(f'Conditioned scripted approach violates frozen limits: {unsafe}')
            apply_action(task, applied, plan['action']['named_layout'])
            task.step(n_steps=plan['action']['simulation_steps_per_action'])
            actions.append(applied.copy())
            prior = applied
        else:
            raise ValueError('Conditioned scripted approach did not converge')
    if not actions or not np.allclose(actions[-1][:3], goal[:3], atol=1e-10):
        raise ValueError('Conditioned approach did not reach the declared source-relative pose')
    return np.asarray(actions), {'source_relative_command_m': (np.asarray(source_prefix[-1][:3]) - original['position_m']).tolist(),
        'fresh_relative_command_m': (actions[-1][:3] - fresh.position).tolist(),
        'yaw_gain': selected.negative_object_yaw_to_wrist_yaw_gain,
        'source_orientation_rule': source_rule,
        'required_wrist_reorientation_rad': float(np.linalg.norm(relative_rotation(goal[3:7], source_prefix[-1][3:7]))),
        'source_reset_position_m': original['position_m'], 'fresh_reset_position_m': list(fresh.position),
        'source_prefix_frames': len(source_prefix), 'conditioned_prefix_frames': len(actions),
        'post_boundary_actions_read': 0, 'object_pose_writes': 0}


def expert_pick_prefix(task, spec, inputs):
    """Use the frozen collection expert solely to establish a fresh approach boundary.

    Its deterministic copied-state template preflight is disclosed; the live task
    receives only open-hand approach commands. No acquire action is queried or
    applied, and no live object pose/orientation is overwritten.
    """
    from dataclasses import asdict
    from dexvision.evaluation.evaluate_skill import apply_action, safe_action
    from dexvision.learning.pick_policy import pick_action_plan
    from dexvision.sim.level4_expert import DeterministicGraspLiftConfig, DeterministicPickPlaceExpert, DeterministicPlaceConfig
    from dexvision.sim.scripted_expert_settings import resolve_scripted_expert_settings

    fields = inputs.plan['action']['named_layout']
    names = [f['name'].split('/')[-1] for f in fields[7:]]
    expert = DeterministicPickPlaceExpert(
        open_finger_targets=dict(zip(names, spec['closure']['open_targets'], strict=True)),
        closed_finger_targets=dict(zip(names, spec['closure']['closed_targets'], strict=True)),
        grasp_config=DeterministicGraspLiftConfig.from_mapping(resolve_scripted_expert_settings(task, 'scripted_grasp')),
        place_config=DeterministicPlaceConfig.from_mapping(resolve_scripted_expert_settings(task, 'scripted_place')),
        validate_on_reset=False)
    expert.reset(task, task.initial_world_state)
    audit = {'preflight': asdict(expert.validation),
             'preflight_assistance': 'Frozen scripted expert validates grasp on a copy with declared rotation hold; no policy queried',
             'live_object_pose_writes': 0, 'live_acquire_actions_queried': 0, 'live_acquire_actions_applied': 0,
             'selected_grasp_config': asdict(expert.grasp_config)}
    if not expert.validation.valid:
        return np.empty((0, 27)), {**audit, 'setup_failed': True, 'reason': expert.validation.reason}
    world = task.initial_world_state
    prior = np.r_[world.robot.base_position, world.robot.base_orientation_wxyz, spec['closure']['open_targets']]
    plan = pick_action_plan(spec, inputs)
    actions = []
    grasp = expert._grasp
    for _ in range(1000):
        if (grasp._waypoint_index == len(grasp.waypoints) - 1
                and np.linalg.norm(grasp._last_requested_position - grasp.waypoints[-1]) <= grasp.config.waypoint_tolerance_m):
            break
        requested, phase, _, reason = expert.step(task.workcell.get_world_state())
        if reason or phase != 'approach':
            raise ValueError(f'Expert prefix crossed the declared boundary or failed: {phase}/{reason}')
        action, _, _, unsafe = safe_action(requested.as_array(), prior, plan)
        if unsafe or not np.allclose(action[7:], spec['closure']['open_targets'], atol=1e-12):
            raise ValueError(f'Expert approach violates safety or starts closure: {unsafe}')
        apply_action(task, action, fields)
        task.step(n_steps=plan['action']['simulation_steps_per_action'])
        actions.append(action.copy())
        prior = action
    else:
        raise ValueError('Scripted approach failed to reach its boundary')
    return np.asarray(actions), {**audit, 'setup_failed': False, 'prefix_frames': len(actions)}
