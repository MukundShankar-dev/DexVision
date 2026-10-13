"""Standalone contact policies, frozen matrices and post-integration evidence."""
from __future__ import annotations

import io
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from dexvision.evaluation.benchmark_retargeters import mean_action_jerk
from dexvision.evaluation.evaluate_skill import SAFETY_KEYS, apply_action, reach_features, safe_action
from dexvision.evaluation.pick_policy import physical_evidence_world
from dexvision.learning.contact_policy import ContactPolicy, contact_action_plan, load_contact_spec
from dexvision.learning.skill_datasets import _restore_initial
from dexvision.learning.train_skill import _checksum_index, _json, _verify_run, load_checkpoint, select_checkpoint
from dexvision.learning.training_manifest import ReleaseSource, content_digest, digest_file, skill_interval
from dexvision.logging.level4_collection import WorkcellPilotTask
from dexvision.sim.workcell import _compute_task_metrics


def contact_scenarios(inputs, split):
    if split not in {'validation', 'test'}:
        raise ValueError('Contact evaluation requires validation or test')
    matrix = inputs.skill['evaluation'][split]
    rows = {r['episode_id']: r for r in inputs.splits[split]['episodes']}
    scenarios = []
    for cell in matrix['cells']:
        for eid, seed, frame in zip(cell['episode_ids'], cell['reset_seeds'], cell['start_frames'], strict=True):
            row = rows[eid]
            if row['cell_id'] != cell['cell_id'] or frame != skill_interval(inputs.skill, row)[0]:
                raise ValueError('Contact matrix boundary/ownership mismatch')
            for condition in inputs.config_snapshot['evaluation']['rollouts']['perception_conditions']:
                scenarios.append((row, seed, frame, condition))
    if (len(scenarios) != matrix['rollouts_per_track_per_training_seed']
            or len({r[0]['episode_id'] for r in scenarios}) != matrix['unique_reset_count']):
        raise ValueError('Incomplete contact matrix')
    return scenarios


def read_contact_entries(inputs, scenarios, release_root=None):
    """Only checksum-verified metadata and strictly pre-entry actions are exposed."""
    unique = {row['episode_id']: (row, seed, frame) for row, seed, frame, _ in scenarios}
    paths = [f"data/demos/level4/{r['source_path']}/{name}"
        for r, _, _ in unique.values() for name in ('metadata.json', 'applied_actions.npy')]
    blobs = ReleaseSource(inputs, release_root).read_many(paths)
    entries = {}
    for eid, (row, seed, frame) in unique.items():
        prefix = f"data/demos/level4/{row['source_path']}/"
        metadata = json.loads(blobs[prefix + 'metadata.json'])
        if (metadata['random_seed'] != seed or metadata['typed_goal'] != row['typed_goal']
                or metadata['episode_id'] != eid or metadata['recording_session_id'] != row['recording_session_id']):
            raise ValueError('Contact entry metadata mismatch')
        actions = np.load(io.BytesIO(blobs[prefix + 'applied_actions.npy']), allow_pickle=False)[:frame].copy()
        if actions.shape != (frame, 27) or not np.isfinite(actions).all():
            raise ValueError('Invalid contact setup prefix')
        entries[eid] = (metadata, actions)
    return entries


def create_contact_task(inputs, row, metadata, prefix):
    task = WorkcellPilotTask(workcell_config=inputs.root / inputs.plan['inputs']['workcell_config']['path'],
        dataset_config=inputs.root / inputs.plan['inputs']['dataset_config']['path'], skill_name=inputs.skill['skill'],
        goal_condition_id=row['cell_id'], seed=metadata['random_seed'],
        procedural_variation=metadata['task_config'].get('procedural_variation'))
    try:
        _restore_initial(task, metadata)
        for action in prefix:
            apply_action(task, action, inputs.plan['action']['named_layout'])
            task.step(n_steps=inputs.plan['action']['simulation_steps_per_action'])
        task._task.reset_dwell()
        return task
    except Exception:
        task.close()
        raise


def contact_preconditions(world, row, spec):
    entity = world.require_entity(row['entity_id'])
    distance = float(np.linalg.norm(np.asarray(world.robot.end_effector_position) - entity.position))
    checks = {'correct_entity': row['entity_id'] == row['typed_goal'].get('object_id', row['typed_goal'].get('button_id')),
        'fresh_observation': entity.timestamp == world.timestamp and entity.confidence == 1.,
        'approach_metric_passed': distance <= spec['preconditions']['maximum_hand_distance_m']}
    if spec['version'] == 'level5/push-policy-v1':
        relation = world.relation_for(row['entity_id'])
        checks.update(supported=relation.supported_by is not None,
            family=entity.class_id.removeprefix('rigid_') in {'cuboid', 'flat_puck'})
    else:
        checks['fixture_present'] = world.require_fixture(row['entity_id']).fixture_id == row['entity_id']
    return checks


@dataclass
class ContactEvidence:
    """Physical dwell, contact loss, overshoot, release and wrong-contact evidence."""
    spec: dict
    goal: dict
    dwell: int = 0
    pressed_once: bool = False
    release_dwell: int = 0
    had_contact: bool = False
    contact_loss_samples: int = 0
    overshot: bool = False
    push_direction: tuple[float, float] | None = None

    def update(self, world, values, qualifies):
        entity_id = self.goal.get('object_id', self.goal.get('button_id'))
        contact = any(entity_id in pair and any(n.startswith('rh_') for n in pair) for pair in world.contacts)
        self.had_contact |= contact
        self.contact_loss_samples += int(self.had_contact and not contact)
        self.dwell = self.dwell + 1 if qualifies else 0
        metrics = {**values, 'physical_contact': contact, 'dwell_steps': self.dwell,
            'contact_loss_samples': self.contact_loss_samples}
        if self.spec['version'] == 'level5/press-policy-v1':
            self.pressed_once |= self.dwell >= self.spec['terminal']['press_dwell_samples']
            released = (self.pressed_once and values['press_depth_m'] <= self.spec['terminal']['release_depth_m']
                        and not contact)
            self.release_dwell = self.release_dwell + 1 if released else 0
            self.overshot |= values['press_depth_m'] > self.spec['terminal']['maximum_depth_m']
            metrics.update(pressed_once=self.pressed_once, release_dwell_steps=self.release_dwell,
                overshoot=self.overshot, success=self.release_dwell >= self.spec['terminal']['release_dwell_samples']
                and not self.overshot)
        else:
            if self.push_direction is not None:
                position = np.asarray(world.require_entity(entity_id).position[:2])
                target = np.asarray(world.require_entity(self.goal['target_zone']).position[:2])
                self.overshot |= float((position - target) @ np.asarray(self.push_direction)) > 0.035
            metrics.update(overshoot=self.overshot,
                success=self.dwell >= self.spec['terminal']['settle_dwell_samples'])
        return metrics


def physical_metrics(task, world):
    return _compute_task_metrics(task.skill_name, task.goal, world, task.initial_world_state, task.workcell.config)


def contact_evidence(spec, goal, world):
    evidence = ContactEvidence(spec, goal)
    if spec['version'] == 'level5/push-policy-v1':
        direction = np.asarray(world.require_entity(goal['target_zone']).position[:2]) - np.asarray(
            world.require_entity(goal['object_id']).position[:2])
        length = float(np.linalg.norm(direction))
        if length > 1e-12:
            evidence.push_direction = tuple(direction / length)
    return evidence


def wrong_contact(world, initial, row, spec):
    allowed = {row['entity_id'], row['typed_goal'].get('target_zone')}
    other_entities = {e.object_id for e in world.entities} - allowed
    wrong = any(any(n.startswith('rh_') for n in pair) and any(n in other_entities for n in pair)
                for pair in world.contacts)
    displaced = any(np.linalg.norm(np.asarray(e.position) - initial.require_entity(e.object_id).position)
        > spec['terminal']['maximum_non_target_displacement_m']
        for e in world.entities if e.object_id in other_entities and e.class_id.startswith('rigid_'))
    activation = any(f.fixture_id != row['entity_id'] and f.press_depth_m > 0.002 for f in world.fixtures)
    return bool(wrong or displaced or activation), int(activation)


def run_contact_rollout(inputs, spec, row, metadata, prefix, policy, *, condition, output_dir):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    plan = contact_action_plan(spec, inputs)
    arrays, worlds, history = defaultdict(list), [], []
    counts = dict.fromkeys(SAFETY_KEYS, 0)
    terminal, reason, metrics = 'timed_out', 'timeout', {}
    wrong_count, activations = 0, 0
    with create_contact_task(inputs, row, metadata, prefix) as task:
        initial = physical_evidence_world(task)
        prior = prefix[-1].copy() if len(prefix) else np.asarray(metadata['initial_applied_action'], float)
        commanded, safety = prior.copy(), np.zeros(27)
        checks = contact_preconditions(initial, row, spec)
        evidence = contact_evidence(spec, row['typed_goal'], initial)
        rest = metadata['task_config']['initial_state']['entity_positions_m'][row['entity_id']]
        start = float(task.env.data.time)
        worlds.append(asdict(initial))
        for key, value in (('qpos', task.env.data.qpos), ('qvel', task.env.data.qvel), ('state_timestamps', start)):
            arrays[key].append(np.asarray(value).copy())
        if not all(checks.values()):
            terminal, reason = 'rejected_before_execution', 'rejected_precondition'
        else:
            for _ in range(inputs.skill['contract']['timeout_steps']):
                world = physical_evidence_world(task)
                source = reach_features(task, world, row, inputs, commanded, prior, safety, task._phase)
                raw, prediction = policy(source, prior, fixture_rest=rest, pressed_once=evidence.pressed_once)
                applied, mask, count, unsafe = safe_action(raw, prior, plan)
                for key in SAFETY_KEYS:
                    counts[key] += count[key]
                for key, value in (('observations', source), ('raw_actions', raw), ('commanded_actions', raw),
                    ('applied_actions', applied), ('policy_predictions', prediction), ('prior_applied_actions', prior),
                    ('prior_commanded_actions', commanded), ('safety_masks', mask), ('phases', task._phase),
                    ('action_timestamps', float(task.env.data.time))):
                    arrays[key].append(np.asarray(value).copy())
                apply_action(task, applied, plan['action']['named_layout'])
                task.step(n_steps=inputs.plan['action']['simulation_steps_per_action'])
                world = physical_evidence_world(task)
                values, qualifies = physical_metrics(task, world)
                metrics = evidence.update(world, values, qualifies)
                # Feed the next sample a phase from this post-integration state,
                # rather than mj_step's stale derived-field task snapshot.
                if spec['version'] == 'level5/push-policy-v1':
                    task._phase = ('retract' if metrics['success'] else 'settle' if qualifies else
                        'push_contact' if evidence.had_contact else 'approach')
                wrong, activated = wrong_contact(world, initial, row, spec)
                wrong_count += int(wrong)
                activations += activated
                metrics.update(wrong_contact=wrong, wrong_button_activation=activated)
                if spec['version'] == 'level5/push-policy-v1':
                    off_board = not values['object_on_board']
                    counts['object_workspace_violations'] += int(off_board)
                else:
                    off_board = False
                failure = unsafe or ('object_workspace_violation' if off_board else 'wrong_contact' if wrong
                    else 'fixture_overshoot' if metrics['overshoot'] else None)
                history.append(metrics)
                worlds.append(asdict(world))
                for key, value in (('qpos', task.env.data.qpos), ('qvel', task.env.data.qvel),
                    ('state_timestamps', float(task.env.data.time)), ('safety_reasons', failure or 'none')):
                    arrays[key].append(np.asarray(value).copy())
                prior, commanded, safety = applied, raw, mask
                if failure:
                    terminal, reason = 'failed', failure
                    break
                if metrics['success']:
                    terminal, reason = 'succeeded', 'physical_contact_task_passed'
                    break
        end = float(task.env.data.time)
        entity = initial.require_entity(row['entity_id'])
        variation = metadata['task_config'].get('procedural_variation') or {}
        object_spec = next((o for o in task.workcell.config.objects if o.object_id == row['entity_id']), None)
        geom = task.env.model.geom(object_spec.geom if object_spec else 'start_button_geom')
        size, friction = geom.size.copy().tolist(), geom.friction.copy().tolist()
        mass = float(task.env.model.body(object_spec.body if object_spec else 'start_button').mass[0])
    np.savez_compressed(directory / 'trajectory.npz', **{k: np.asarray(v) for k, v in arrays.items()})
    _json(directory / 'world_states.json', {'states': worlds, 'metrics': history})
    _json(directory / 'setup.json', {'metadata': metadata, 'prefix_actions': prefix.tolist(),
        'disclosure': 'Scripted approach prefix excluded; no post-entry expert queries or object pose writes'})
    metric = 'planar_object_to_target_distance_m' if spec['version'] == 'level5/push-policy-v1' else 'press_depth_m'
    error = (metrics[metric] / 0.035 if metric.startswith('planar') else
        max(0., row['typed_goal']['target_press_depth_m'] - metrics[metric]) / spec['depth_scale_m']) if metrics else None
    result = {'episode_id': row['episode_id'], 'episode_digest': row['episode_digest'],
        'reset_seed': metadata['random_seed'], 'start_frame': len(prefix), 'cell_id': row['cell_id'],
        'session': row['recording_session_id'], 'source': row['source'], 'object_family': entity.class_id,
        'object_instance': row['entity_id'], 'target_id': row['typed_goal'].get('target_zone', row['entity_id']),
        'procedural_variation': variation, 'render_condition': condition,
        'object_size_m': size, 'object_mass_kg': mass, 'object_friction': friction,
        'start_position_m': list(entity.position), 'push_direction_xy': evidence.push_direction,
        'boundary_class': variation.get('case_class', 'unspecified_in_source'),
        'target_depth_m': row['typed_goal'].get('target_press_depth_m'),
        'render_condition_effect': 'none_on_state_or_physics; no_RGB_claim', 'preconditions': checks,
        'terminal_result': terminal, 'terminal_reason': reason, 'final_metrics': metrics,
        'start_timestamp': start, 'end_timestamp': end, 'retry_count': 0, 'safety_counters': counts,
        'normalized_terminal_error': error, 'wrong_contact_count': wrong_count,
        'wrong_button_activation_count': activations, 'contact_loss_samples': evidence.contact_loss_samples,
        'mean_action_jerk': mean_action_jerk(np.asarray(arrays['applied_actions']) / np.asarray([
            f['upper'] - f['lower'] for f in inputs.plan['action']['named_layout']])) if arrays['applied_actions'] else 0.,
        'trajectory_path': 'trajectory.npz', 'world_states_path': 'world_states.json'}
    _json(directory / 'result.json', result)
    result['physical_audit'] = audit_contact_attempt(inputs, spec, row, directory)
    _json(directory / 'result.json', result)
    _checksum_index(directory)
    return result


def audit_contact_attempt(inputs, spec, row, directory):
    """Reconstruct every saved qpos/qvel on a separate simulator; never integrate."""
    directory = Path(directory)
    setup = json.loads((directory / 'setup.json').read_text())
    saved = json.loads((directory / 'world_states.json').read_text())
    result = json.loads((directory / 'result.json').read_text())
    issues = []
    with create_contact_task(inputs, row, setup['metadata'], np.asarray(setup['prefix_actions'])) as task, \
            np.load(directory / 'trajectory.npz', allow_pickle=False) as data:
        if len(data['qpos']) != len(saved['metrics']) + 1:
            raise ValueError('Contact audit state/action alignment mismatch')
        initial = physical_evidence_world(task)
        evidence = contact_evidence(spec, row['typed_goal'], initial)
        for index in range(1, len(data['qpos'])):
            apply_action(task, data['applied_actions'][index - 1], inputs.plan['action']['named_layout'])
            task.env.data.qpos[:] = data['qpos'][index]
            task.env.data.qvel[:] = data['qvel'][index]
            task.env.data.time = float(data['state_timestamps'][index])
            world = physical_evidence_world(task)
            values, qualifies = physical_metrics(task, world)
            metrics = evidence.update(world, values, qualifies)
            wrong, activated = wrong_contact(world, initial, row, spec)
            metrics.update(wrong_contact=wrong, wrong_button_activation=activated)
            if content_digest(metrics) != content_digest(saved['metrics'][index - 1]):
                issues.append(f'physical metric mismatch at {index}')
        if result['terminal_result'] == 'succeeded' and (not saved['metrics'] or not metrics['success']):
            issues.append('saved physical terminal dwell does not pass')
    return {'passed': not issues, 'states_checked': len(saved['metrics']), 'issues': issues}


def summarize_contact(results, inputs, split):
    expected = {(r['episode_id'], c) for r, _, _, c in contact_scenarios(inputs, split)}
    if len(results) != len(expected) or {(r['episode_id'], r['render_condition']) for r in results} != expected:
        raise ValueError('Missing, duplicate or extra contact attempts')
    breakdowns = {}
    for field in ('source', 'session', 'object_family', 'object_instance', 'target_id', 'cell_id', 'render_condition',
                  'boundary_class', 'object_size_m', 'object_mass_kg', 'object_friction', 'push_direction_xy',
                  'target_depth_m'):
        groups = defaultdict(list)
        for r in results:
            key = r[field]
            groups[json.dumps(key, sort_keys=True) if not isinstance(key, str) else key].append(r)
        breakdowns[field] = {k: {'count': len(g),
            'success_rate': sum(r['terminal_result'] == 'succeeded' for r in g) / len(g),
            'terminal_reasons': dict(Counter(r['terminal_reason'] for r in g))} for k, g in sorted(groups.items())}
    success = sum(r['terminal_result'] == 'succeeded' for r in results) / len(results)
    worst = min(x['success_rate'] for x in breakdowns['object_family'].values())
    counts = {k: sum(r['safety_counters'][k] for r in results) for k in SAFETY_KEYS}
    wrong = sum(r['wrong_contact_count'] for r in results)
    activation = sum(r['wrong_button_activation_count'] for r in results)
    physical = sum(not r['physical_audit']['passed'] for r in results)
    jerk = float(np.mean([r['mean_action_jerk'] for r in results]))
    gates = {'aggregate_success': success >= inputs.skill['gates']['state_success_min'],
        'worst_family_success': worst >= inputs.skill['gates']['state_worst_family_success_min'],
        'mean_action_jerk': jerk <= inputs.skill['gates']['mean_action_jerk_max'],
        'wrong_contact': wrong == 0, 'wrong_button_activation': activation == 0, 'physical_audit': physical == 0,
        **{k: v == 0 for k, v in counts.items()}, 'explicit_terminal_results': all(
            r['terminal_result'] in inputs.config_snapshot['evaluation']['terminal_results'] for r in results)}
    errors = [r['normalized_terminal_error'] for r in results if r['normalized_terminal_error'] is not None]
    return {'rollout_count': len(results), 'unique_reset_count': len({r['episode_id'] for r in results}),
        'success_rate': success, 'worst_family_success_rate': worst,
        'worst_cell_success_rate': min(x['success_rate'] for x in breakdowns['cell_id'].values()),
        'mean_action_jerk': jerk, 'safety_violation_count': sum(counts.values()) + wrong + activation + physical,
        'safety_counters': counts, 'wrong_contact_count': wrong, 'wrong_button_activation_count': activation,
        'physical_audit_failure_count': physical, 'normalized_terminal_error': float(np.mean(errors)) if errors else None,
        'terminal_error_unavailable_reason': None if errors else 'rejected_precondition',
        'breakdowns': breakdowns, 'gates': gates, 'state_numerical_gates_passed': all(gates.values()),
        'status': 'experimental' if all(gates.values()) else 'failed', 'default_registry_admission': False}


def evaluate_contact(config, checkpoint_path, output_dir, *, split='test', release_root=None):
    spec, inputs = load_contact_spec(config)
    checkpoint_path, directory = Path(checkpoint_path), Path(output_dir)
    policy = ContactPolicy(checkpoint_path, spec, inputs)
    checkpoint = load_checkpoint(checkpoint_path)
    manifest = checkpoint['manifest']
    run = checkpoint_path.parent.parent if checkpoint_path.parent.name == 'validation_candidates' else checkpoint_path.parent
    if split == 'test':
        _verify_run(run)
        selection = json.loads((run / 'selection.json').read_text())
        if checkpoint_path.name != 'selected.pt' or selection['status'] != 'selected_from_validation':
            raise ValueError('Contact test requires validation-selected weights')
        candidate_path = run / 'validation_candidates' / f"epoch_{selection['epoch']:04d}.pt"
        candidate = load_checkpoint(candidate_path)
        if digest_file(candidate_path) != selection['source_checkpoint_sha256'] or any(
                not torch.equal(v, candidate['model'][k]) for k, v in checkpoint['model'].items()):
            raise ValueError('Selected contact weights differ from validation candidate')
        if (run / 'test_state_attempt.json').exists():
            raise ValueError('Single contact test attempt already reserved')
    scenarios = contact_scenarios(inputs, split)
    entries = read_contact_entries(inputs, scenarios, release_root)
    plan = contact_action_plan(spec, inputs)
    for metadata, prefix in entries.values():
        prior = prefix[-1] if len(prefix) else np.asarray(metadata['initial_applied_action'])
        if safe_action(prior, prior, plan)[3]:
            raise ValueError('Contact entry conflicts with frozen safety bounds')
    directory.mkdir(parents=True, exist_ok=False)
    provenance = {'version': 'level5/contact-evaluation-v1', 'skill': inputs.skill['skill'],
        'track': 'state', 'split': split, 'seed': manifest['seed'],
        'epoch': checkpoint['training_state']['completed_epochs'], 'checkpoint_sha256': digest_file(checkpoint_path),
        'checkpoint_path': str(checkpoint_path.resolve()), 'output_dir': str(directory.resolve()),
        'manifest_digest': manifest['manifest_digest'], 'validation_matrix_sha256': manifest['validation_matrix_sha256'],
        'matrix_sha256': content_digest(inputs.skill['evaluation'][split]), 'input_digests': inputs.digests,
        'recipe_sha256': content_digest(spec), 'selection_uses_test': False}
    _json(directory / 'provenance.json', provenance)
    if split == 'test':
        with (run / 'test_state_attempt.json').open('x') as stream:
            json.dump(provenance, stream, indent=2)
    results = []
    for index, (row, _, _, condition) in enumerate(scenarios):
        if index % 20 == 0:
            print(f"{inputs.skill['skill']} {split}: {index}/{len(scenarios)} seed {manifest['seed']}", flush=True)
        child = directory / f"{index:03d}_{row['episode_id']}_{condition}"
        result = run_contact_rollout(inputs, spec, row, *entries[row['episode_id']], policy,
            condition=condition, output_dir=child)
        result['result_path'] = f'{child.name}/result.json'
        results.append(result)
    report = {**provenance, **summarize_contact(results, inputs, split), 'results': results,
        'training_history': checkpoint['training_state']['history']}
    _json(directory / 'report.json', report)
    _checksum_index(directory)
    if split == 'test':
        _checksum_index(run)
    return report


def select_contact(config, run_dir, output_dir, release_root=None):
    run, directory = Path(run_dir), Path(output_dir)
    _verify_run(run)
    checkpoint = load_checkpoint(run / 'last.pt')
    if not checkpoint['training_state']['finished'] or (run / 'selected.pt').exists():
        raise ValueError('Contact selection requires finished unselected training')
    directory.mkdir(parents=True, exist_ok=False)
    reports = []
    for epoch, item in sorted(checkpoint['training_state']['candidates'].items(), key=lambda x: int(x[0])):
        print(f'Contact validation candidate {epoch}: {run}', flush=True)
        reports.append(evaluate_contact(config, run / item['path'], directory / f'epoch_{int(epoch):04d}',
            split='validation', release_root=release_root))
    _checksum_index(run)
    result = select_checkpoint(run, reports)
    _checksum_index(directory)
    return result


def review_contact(config, checkpoint_path):
    """Replay one success from each held-out cell; leave acceptance to the owner."""
    import mujoco.viewer

    spec, inputs = load_contact_spec(config)
    checkpoint_path = Path(checkpoint_path)
    ContactPolicy(checkpoint_path, spec, inputs)
    _verify_run(checkpoint_path.parent)
    attempt = json.loads((checkpoint_path.parent / 'test_state_attempt.json').read_text())
    directory = Path(attempt['output_dir'])
    _verify_run(directory)
    report = json.loads((directory / 'report.json').read_text())
    if report['checkpoint_sha256'] != digest_file(checkpoint_path) or not report['state_numerical_gates_passed']:
        raise ValueError('Contact viewer requires passing numerical and physical report for these weights')
    chosen = {}
    for result in report['results']:
        if result['terminal_result'] == 'succeeded':
            chosen.setdefault(result['cell_id'], result)
    if set(chosen) != set(report['breakdowns']['cell_id']):
        raise ValueError('Contact viewer needs a saved success in every held-out cell')
    rows = {r['episode_id']: r for r in inputs.splits['test']['episodes']}
    for result in chosen.values():
        child = directory / Path(result['result_path']).parent
        _verify_run(child)
        setup = json.loads((child / 'setup.json').read_text())
        with create_contact_task(inputs, rows[result['episode_id']], setup['metadata'],
                np.asarray(setup['prefix_actions'])) as task:
            print(f"Contact review seed {report['seed']} {result['cell_id']}: {result['terminal_reason']}", flush=True)
            with np.load(child / 'trajectory.npz', allow_pickle=False) as data, \
                    mujoco.viewer.launch_passive(task.env.model, task.env.data) as viewer:
                for index, (qpos, qvel) in enumerate(zip(data['qpos'], data['qvel'], strict=True)):
                    if not viewer.is_running():
                        return {'manual_acceptance': 'pending_owner_confirmation', 'cancelled': True}
                    if index:
                        apply_action(task, data['applied_actions'][index - 1], inputs.plan['action']['named_layout'])
                    task.env.data.qpos[:] = qpos
                    task.env.data.qvel[:] = qvel
                    task.env.data.time = data['state_timestamps'][index]
                    task.env._mujoco.mj_forward(task.env.model, task.env.data)
                    viewer.sync()
                    time.sleep(inputs.plan['action']['control_interval_s'])
    return {'manual_acceptance': 'pending_owner_confirmation', 'reviewed': len(chosen)}


def run_contact_evaluation(config, *, checkpoint, output_dir, release_root=None, track='state', split='test',
                           select_candidates=None, viewer=False, scenario_set=None):
    if track != 'state':
        raise ValueError('Checkpoint 5.5 implements state contact policies only')
    if viewer:
        if scenario_set != 'manual_stratified' or checkpoint is None or select_candidates:
            raise ValueError('Contact viewer needs selected checkpoint and manual_stratified')
        return review_contact(config, checkpoint)
    if scenario_set is not None or output_dir is None:
        raise ValueError('Contact evaluation needs a new output directory')
    if select_candidates is not None:
        if checkpoint is not None:
            raise ValueError('Choose checkpoint or candidate selection')
        return select_contact(config, select_candidates, output_dir, release_root)
    if checkpoint is None:
        raise ValueError('Contact evaluation requires checkpoint')
    return evaluate_contact(config, checkpoint, output_dir, split=split, release_root=release_root)
