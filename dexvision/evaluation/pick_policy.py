"""Frozen standalone pick evaluation with scripted entry and physical outcome evidence."""
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
from dexvision.learning.phase_losses import phase_errors
from dexvision.learning.pick_policy import PickPolicy, load_pick_spec, pick_action_plan, pick_datasets
from dexvision.learning.skill_datasets import _restore_initial
from dexvision.learning.train_skill import _checksum_index, _json, _verify_run, load_checkpoint, select_checkpoint
from dexvision.learning.training_manifest import ReleaseSource, content_digest, digest_file, skill_interval
from dexvision.logging.level4_collection import WorkcellPilotTask


def pick_scenarios(inputs, split):
    if split not in {'validation', 'test'}:
        raise ValueError("Pick evaluation requires validation or test")
    matrix = inputs.skill['evaluation'][split]
    rows = {r['episode_id']: r for r in inputs.splits[split]['episodes']}
    result = []
    for cell in matrix['cells']:
        for eid, seed, frame in zip(cell['episode_ids'], cell['reset_seeds'], cell['start_frames'], strict=True):
            row = rows[eid]
            expected_frame = row['entry_frame'] if row.get('data_amendment') else skill_interval(inputs.skill, row)[0]
            if row['cell_id'] != cell['cell_id'] or frame != expected_frame:
                raise ValueError("Pick matrix start/ownership mismatch")
            for condition in inputs.config_snapshot['evaluation']['rollouts']['perception_conditions']:
                result.append((row, seed, frame, condition))
    if (len(result) != matrix['rollouts_per_track_per_training_seed'] or
            len({r[0]['episode_id'] for r in result}) != matrix['unique_reset_count']):
        raise ValueError("Incomplete pick matrix")
    return result


def read_pick_entries(inputs, scenarios, release_root=None, *, spec=None):
    """Verify release bytes and expose only the strictly pre-boundary action prefix."""
    unique = {r['episode_id']: (r, seed, frame) for r, seed, frame, _ in scenarios}
    if any(r.get('data_amendment') for r, _, _ in unique.values()):
        if spec is None or not all(r.get('data_amendment') for r, _, _ in unique.values()):
            raise ValueError('Additive pick entries require their explicit recipe and unmixed ownership')
        from dexvision.learning.pick_data import amendment_entries
        return amendment_entries(inputs, spec, [r for r, _, _ in unique.values()])
    names = [f"data/demos/level4/{r['source_path']}/{name}"
             for r, _, _ in unique.values() for name in ('metadata.json', 'applied_actions.npy')]
    blobs = ReleaseSource(inputs, release_root).read_many(names)
    entries = {}
    for eid, (row, seed, frame) in unique.items():
        prefix = f"data/demos/level4/{row['source_path']}/"
        metadata = json.loads(blobs[prefix + 'metadata.json'])
        if (metadata['random_seed'] != seed or metadata['typed_goal'] != row['typed_goal']
                or metadata['episode_id'] != eid or metadata['recording_session_id'] != row['recording_session_id']):
            raise ValueError("Pick entry metadata mismatch")
        actions = np.load(io.BytesIO(blobs[prefix + 'applied_actions.npy']), allow_pickle=False)[:frame].copy()
        if actions.shape != (frame, 27) or not np.isfinite(actions).all():
            raise ValueError("Invalid pick setup prefix")
        entries[eid] = (metadata, actions)
    return entries


def create_pick_task(inputs, row, metadata, prefix):
    """Replay frozen setup only. Return a live task with terminal dwell reset."""
    task = WorkcellPilotTask(workcell_config=inputs.root / inputs.plan['inputs']['workcell_config']['path'],
        dataset_config=inputs.root / inputs.plan['inputs']['dataset_config']['path'],
        skill_name='pick_place_sequence', goal_condition_id=row['cell_id'], seed=metadata['random_seed'],
        procedural_variation=metadata['task_config'].get('procedural_variation'))
    try:
        _restore_initial(task, metadata)
        for action in prefix:
            apply_action(task, action, inputs.plan['action']['named_layout'])
            task.step(n_steps=inputs.plan['action']['simulation_steps_per_action'])
        task._pick_task.reset_dwell()
        return task
    except Exception:
        task.close()
        raise


def pick_preconditions(world, row, spec):
    relation = world.relation_for(row['entity_id'])
    entity = world.require_entity(row['entity_id'])
    distance = float(np.linalg.norm(np.asarray(world.robot.end_effector_position) - entity.position))
    return {'correct_object': entity.object_id == row['typed_goal']['object_id'],
            'supported': relation.supported_by is not None,
            'unheld': relation.held_by is None,
            'approach_metric_passed': distance <= spec['preconditions']['maximum_grasp_site_distance_m'],
            'fresh_observation': entity.timestamp == world.timestamp and entity.confidence == 1.0}


def physical_evidence_world(task):
    """Recompute post-integration geometry/velocity on a copy, preserving live dynamics.

    mj_step leaves derived fields at its final pre-integration state. Terminal
    evidence must instead agree with the saved qpos/qvel audit. Neither forward
    dynamics nor its warm-start updates are allowed to mutate the live data here.
    """
    env = task.env
    live = env.data
    scratch = env._mujoco.MjData(env.model)
    env._mujoco.mj_copyData(scratch, env.model, live)
    try:
        env.data = scratch
        env._mujoco.mj_forward(env.model, scratch)
        return task.workcell.get_world_state()
    finally:
        env.data = live


@dataclass
class PickEvidence:
    """Stateful physical acquisition, lift, retention and consecutive stable dwell."""
    object_id: str
    initial_height: float
    spec: dict
    acquired: bool = False
    lifted: bool = False
    had_contact: bool = False
    retention_gap: int = 0
    stable_dwell: int = 0
    maximum_height: float = 0.0

    def update(self, world):
        entity = world.require_entity(self.object_id)
        relation = world.relation_for(self.object_id)
        contacts = {other for pair in world.contacts if self.object_id in pair
                    for other in pair if other.startswith('rh_')}
        held = len(contacts) >= 2 and relation.held_by == 'rh_palm'
        height = float(entity.position[2] - self.initial_height)
        velocity = entity.linear_velocity
        speed = float(np.linalg.norm(velocity)) if velocity is not None else None
        self.had_contact |= bool(contacts)
        self.acquired |= held
        self.lifted |= held and height >= self.spec['minimum_height_m']
        self.maximum_height = max(self.maximum_height, height)
        self.retention_gap = 0 if held else self.retention_gap + int(self.acquired)
        stable = (held and height >= self.spec['minimum_height_m'] and relation.supported_by is None
                  and speed is not None and speed <= self.spec['maximum_speed_mps'])
        self.stable_dwell = self.stable_dwell + 1 if stable else 0
        dropped = self.acquired and self.retention_gap > self.spec['maximum_retention_gap_samples']
        wrong = any(r.held_by is not None and r.object_id != self.object_id for r in world.relations)
        return {'held_object_id': self.object_id if held else None, 'contact_body_count': len(contacts),
                'object_height_above_support_m': height, 'object_speed_mps': speed,
                'object_supported': relation.supported_by is not None, 'acquired': self.acquired,
                'lift_height_passed': self.lifted, 'stable_held_dwell': self.stable_dwell,
                'maximum_height_m': self.maximum_height, 'retention_gap': self.retention_gap,
                'slip_drop': dropped, 'wrong_object_held': wrong,
                'success': self.stable_dwell >= self.spec['consecutive_samples']}


def run_pick_rollout(inputs, spec, row, metadata, prefix, policy, *, condition, output_dir):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    action_plan = pick_action_plan(spec, inputs)
    fields = action_plan['action']['named_layout']
    arrays, worlds, metrics_history = defaultdict(list), [], []
    counts = dict.fromkeys(SAFETY_KEYS, 0)
    terminal, reason = 'timed_out', 'failed_acquisition'
    metrics = {}
    with create_pick_task(inputs, row, metadata, prefix) as task:
        env = task.env
        initial = task.workcell.get_world_state()
        prior = prefix[-1].copy() if len(prefix) else np.asarray(metadata['initial_applied_action'], float)
        commanded, safety = prior.copy(), np.zeros(27)
        checks = pick_preconditions(initial, row, spec)
        object_id = row['entity_id']
        evidence = PickEvidence(object_id, task.initial_world_state.require_entity(object_id).position[2], spec['terminal'])
        source = reach_features(task, initial, row, inputs, commanded, prior, safety, task._phase)
        policy.reset(source)
        start = float(env.data.time)
        worlds.append(asdict(initial))
        for key, value in (('qpos', env.data.qpos), ('qvel', env.data.qvel), ('state_timestamps', start)):
            arrays[key].append(np.asarray(value).copy())
        if not all(checks.values()):
            terminal, reason = 'rejected_before_execution', 'rejected_precondition'
        else:
            for _ in range(inputs.skill['contract']['timeout_steps']):
                world = task.workcell.get_world_state()
                source = reach_features(task, world, row, inputs, commanded, prior, safety, task._phase)
                raw, prediction = policy(source, prior, task._phase)
                applied, mask, count, unsafe = safe_action(raw, prior, action_plan)
                for key in SAFETY_KEYS:
                    counts[key] += count[key]
                for key, value in (('observations', source), ('raw_actions', raw), ('commanded_actions', raw),
                    ('applied_actions', applied), ('policy_predictions', prediction), ('prior_applied_actions', prior),
                    ('prior_commanded_actions', commanded), ('safety_masks', mask), ('action_timestamps', float(env.data.time))):
                    arrays[key].append(np.asarray(value).copy())
                apply_action(task, applied, fields)
                world = task.workcell.step(n_steps=inputs.plan['action']['simulation_steps_per_action'])
                if spec.get('physical_evidence_state') == 'recomputed_post_integration_copy':
                    world = physical_evidence_world(task)
                metrics = evidence.update(world)
                historical = task._pick_task.evaluate(world)
                metrics['frozen_pick_metrics'] = dict(historical.values)
                metrics['frozen_pick_success'] = historical.success
                metrics['success'] = metrics['success'] and historical.success
                metrics['maximum_contact_penetration_m'] = max(
                    (max(0., -float(env.data.contact[j].dist)) for j in range(env.data.ncon)), default=0.)
                metrics['hand_table_contact_count'] = sum(
                    task.workcell.config.scene['table_body'] in pair and any(n.startswith('rh_') for n in pair)
                    for pair in world.contacts)
                metrics_history.append(metrics)
                worlds.append(asdict(world))
                arrays['qpos'].append(env.data.qpos.copy())
                arrays['qvel'].append(env.data.qvel.copy())
                arrays['state_timestamps'].append(float(env.data.time))
                phase = 'stabilize' if metrics['held_object_id'] and metrics['object_height_above_support_m'] >= spec['lift_height_m'] else (
                    'lift' if metrics['held_object_id'] else 'acquire')
                arrays['physical_phases'].append(phase)
                # Non-target objects are intentionally parked outside the board at reset.
                board = inputs.requirements['workcell']['board_workspace']
                xy = np.asarray(world.require_entity(object_id).position[:2])
                off_board = bool(np.any(xy < np.asarray(board['min_xy_m']) + board['safe_edge_margin_m']) or
                                 np.any(xy > np.asarray(board['max_xy_m']) - board['safe_edge_margin_m']))
                counts['object_workspace_violations'] += int(off_board)
                entities = set(task.workcell.config.object_ids) | set(task.workcell.config.fixture_ids)
                wrong = metrics['wrong_object_held'] or any(any(n.startswith('rh_') for n in pair) and
                    any(n in entities and n != object_id for n in pair) for pair in world.contacts)
                failure = unsafe or ('object_workspace_violation' if off_board else
                    'wrong_contact' if wrong else 'slip_drop' if metrics['slip_drop'] else None)
                arrays['safety_reasons'].append(failure or 'none')
                prior, commanded, safety = applied, raw, mask
                if failure:
                    terminal, reason = 'failed', failure
                    break
                if metrics['success']:
                    terminal, reason = 'succeeded', 'physical_lift_and_stable_dwell_passed'
                    break
            else:
                reason = 'unstable_hold' if evidence.lifted else 'insufficient_lift' if evidence.acquired else 'failed_acquisition'
        end = float(env.data.time)
        object_spec = next(o for o in task.workcell.config.objects if o.object_id == object_id)
        body = env.model.body(object_id)
        geom = env.model.geom(object_spec.geom)
        physical_parameters = {'mass_kg': float(body.mass[0]),
                               'geom_size_m': geom.size.tolist(), 'geom_friction': geom.friction.tolist(),
                               'approach_distance_m': float(np.linalg.norm(np.asarray(initial.robot.end_effector_position) - initial.require_entity(object_id).position))}
    saved = {key: np.asarray(value) for key, value in arrays.items()}
    np.savez_compressed(directory / 'trajectory.npz', **saved)
    _json(directory / 'world_states.json', {'states': worlds, 'metrics': metrics_history})
    _json(directory / 'setup.json', {'metadata': metadata, 'prefix_actions': prefix.tolist(),
          'disclosure': 'Scripted setup before exact frozen boundary; no expert action after boundary'})
    error = max(0., spec['lift_height_m'] - metrics['object_height_above_support_m']) / spec['lift_height_m'] if metrics else None
    variation = metadata['task_config'].get('procedural_variation') or {}
    result = {'episode_id': row['episode_id'], 'episode_digest': row['episode_digest'], 'reset_seed': metadata['random_seed'],
        'start_frame': len(prefix), 'cell_id': row['cell_id'], 'session': row['recording_session_id'],
        'object_family': object_spec.family, 'object_instance': object_id, 'target_id': object_id,
        'target_type': object_spec.family, 'boundary_class': variation.get('case_class', 'unspecified_in_source'),
        'render_condition': condition, 'render_condition_effect': 'none_on_state_or_physics; no_RGB_claim',
        'physical_parameters': physical_parameters, 'procedural_variation': variation,
        'preconditions': checks, 'terminal_result': terminal, 'terminal_reason': reason, 'final_metrics': metrics,
        'start_timestamp': start, 'end_timestamp': end, 'retry_count': 0, 'safety_counters': counts,
        'normalized_terminal_error': error, 'acquired': evidence.acquired, 'lifted': evidence.lifted,
        'slip_drop': bool(metrics.get('slip_drop', False)),
        'mean_action_jerk': mean_action_jerk(np.asarray(arrays['applied_actions']) / np.asarray([
            f['upper'] - f['lower'] for f in inputs.plan['action']['named_layout']])) if arrays['applied_actions'] else 0.,
        'trajectory_path': 'trajectory.npz', 'world_states_path': 'world_states.json'}
    for field, parameter in (('size', 'geom_size_m'), ('mass', 'mass_kg'), ('friction', 'geom_friction'), ('approach', 'approach_distance_m')):
        result[field] = json.dumps(physical_parameters[parameter], sort_keys=True)
    _json(directory / 'result.json', result)
    _checksum_index(directory)
    return result


def summarize_pick(results, inputs, split, *, expected_attempts=None):
    expected = (expected_attempts if expected_attempts is not None else
                {(r['episode_id'], c) for r, _, _, c in pick_scenarios(inputs, split)})
    if len(results) != len(expected) or {(r['episode_id'], r['render_condition']) for r in results} != expected:
        raise ValueError("Missing, duplicate or extra pick attempts")
    breakdowns = {}
    fields = ('session', 'object_family', 'object_instance', 'target_id', 'target_type', 'cell_id', 'boundary_class', 'render_condition',
              'size', 'mass', 'friction', 'approach')
    for field in fields:
        groups = defaultdict(list)
        for r in results:
            groups[r[field]].append(r)
        breakdowns[field] = {k: {'count': len(v), 'success_rate': sum(r['terminal_result'] == 'succeeded' for r in v) / len(v),
            'acquisition_rate': sum(r['acquired'] for r in v) / len(v), 'drop_rate': sum(r['slip_drop'] for r in v) / len(v),
            'terminal_reasons': dict(Counter(r['terminal_reason'] for r in v))} for k, v in sorted(groups.items())}
    success = sum(r['terminal_result'] == 'succeeded' for r in results) / len(results)
    worst = min(v['success_rate'] for v in breakdowns['object_family'].values())
    counts = {k: sum(r['safety_counters'][k] for r in results) for k in SAFETY_KEYS}
    jerk = float(np.mean([r['mean_action_jerk'] for r in results]))
    drop = sum(r['slip_drop'] for r in results) / len(results)
    gates = {'aggregate_success': success >= inputs.skill['gates']['state_success_min'],
             'worst_family_success': worst >= inputs.skill['gates']['state_worst_family_success_min'],
             'drop_rate': drop <= inputs.skill['gates']['drop_rate_max'],
             'mean_action_jerk': jerk <= inputs.skill['gates']['mean_action_jerk_max'],
             **{k: v == 0 for k, v in counts.items()},
             'explicit_terminal_results': all(r['terminal_result'] in inputs.config_snapshot['evaluation']['terminal_results'] for r in results)}
    errors = [r['normalized_terminal_error'] for r in results]
    return {'rollout_count': len(results), 'unique_reset_count': len({r['episode_id'] for r in results}),
        'success_rate': success, 'worst_family_success_rate': worst,
        'worst_cell_success_rate': min(v['success_rate'] for v in breakdowns['cell_id'].values()),
        'acquisition_rate': sum(r['acquired'] for r in results) / len(results),
        'lift_height_rate': sum(r['lifted'] for r in results) / len(results), 'drop_rate': drop,
        'normalized_terminal_error': float(np.mean(errors)) if all(v is not None for v in errors) else None,
        'terminal_error_unavailable_reason': 'rejected_precondition' if any(v is None for v in errors) else None,
        'safety_counters': counts, 'safety_violation_count': sum(counts.values()), 'mean_action_jerk': jerk,
        'gates': gates, 'state_numerical_gates_passed': all(gates.values()), 'state_qualified': False,
        'manual_acceptance': 'pending_owner_confirmation' if all(gates.values()) else 'not_required_numerical_gates_failed',
        'status': 'failed' if sum(counts.values()) else 'experimental', 'breakdowns': breakdowns,
        'terminal_reasons': dict(Counter(r['terminal_reason'] for r in results))}


def evaluate_pick(config, checkpoint_path, output_dir, *, split='test', release_root=None):
    spec, inputs = load_pick_spec(config)
    if spec['checkpoint'] == '5.3A' and split == 'test':
        raise ValueError('Pick repair requires fresh confirmation; original test is exposed')
    checkpoint_path, directory = Path(checkpoint_path), Path(output_dir)
    checkpoint = load_checkpoint(checkpoint_path)
    policy = PickPolicy(checkpoint_path, spec, inputs)
    manifest = checkpoint['manifest']
    run = checkpoint_path.parent.parent if checkpoint_path.parent.name == 'validation_candidates' else checkpoint_path.parent
    if split == 'test':
        _verify_run(run)
        selection = json.loads((run / 'selection.json').read_text())
        if checkpoint_path.name != 'selected.pt' or selection['status'] != 'selected_from_validation':
            raise ValueError("Test requires validation-selected pick weights")
        source = run / 'validation_candidates' / f"epoch_{selection['epoch']:04d}.pt"
        candidate = load_checkpoint(source)
        if digest_file(source) != selection['source_checkpoint_sha256'] or any(
                not torch.equal(v, candidate['model'][k]) for k, v in checkpoint['model'].items()):
            raise ValueError("Selected pick weights differ from validation candidate")
        if (run / 'test_state_attempt.json').exists():
            raise ValueError("Single pick test attempt already reserved; do not repeat")
    scenarios = pick_scenarios(inputs, split)
    entries = read_pick_entries(inputs, scenarios, release_root, spec=spec)
    for metadata, prefix in entries.values():
        prior = prefix[-1] if len(prefix) else np.asarray(metadata['initial_applied_action'])
        if safe_action(prior, prior, pick_action_plan(spec, inputs))[3]:
            raise ValueError("Pick entry conflicts with recipe safety bounds; stop before evaluating policy")
    directory.mkdir(parents=True, exist_ok=False)
    provenance = {'version': 'level5/pick-evaluation-v1', 'skill': 'pick_object', 'track': 'state', 'split': split,
        'seed': manifest['seed'], 'epoch': checkpoint['training_state']['completed_epochs'],
        'checkpoint_sha256': digest_file(checkpoint_path), 'checkpoint_path': str(checkpoint_path.resolve()),
        'output_dir': str(directory.resolve()), 'manifest_digest': manifest['manifest_digest'],
        'validation_matrix_sha256': manifest['validation_matrix_sha256'],
        'matrix_sha256': content_digest(inputs.skill['evaluation'][split]), 'input_digests': inputs.digests,
        'recipe_sha256': content_digest(spec), 'selection_uses_test': False,
        'execution_implementation_sha256': {p.relative_to(inputs.root).as_posix(): digest_file(p)
            for p in sorted((inputs.root / 'dexvision').rglob('*.py'))}}
    _json(directory / 'provenance.json', provenance)
    if split == 'test':
        with (run / 'test_state_attempt.json').open('x') as stream:
            json.dump(provenance, stream, indent=2)
    results = []
    for index, (row, _, _, condition) in enumerate(scenarios):
        if index % 20 == 0:
            print(f"Pick {split}: {index}/{len(scenarios)} seed {manifest['seed']}", flush=True)
        child = directory / f"{index:03d}_{row['episode_id']}_{condition}"
        result = run_pick_rollout(inputs, spec, row, *entries[row['episode_id']], policy,
                                  condition=condition, output_dir=child)
        result['result_path'] = f'{child.name}/result.json'
        results.append(result)
    offline_errors = None
    if Path(spec['cache_dir']).exists():
        data = pick_datasets(spec, inputs, spec['cache_dir'])['validation']
        with torch.inference_mode():
            predictions = policy.model(torch.from_numpy(data.inputs)).numpy()
        offline_errors = phase_errors(predictions, data.targets, data.masks, [p for e in data.episodes for p in e.phases])
    report = {**provenance, **summarize_pick(results, inputs, split), 'results': results,
              'phase_conditioned_offline_validation_error': offline_errors,
              'training_history': checkpoint['training_state']['history']}
    _json(directory / 'report.json', report)
    _checksum_index(directory)
    if split == 'test':
        _checksum_index(run)
    return report


def select_pick(config, run_dir, output_dir, release_root=None):
    spec, _ = load_pick_spec(config)
    if spec['checkpoint'] == '5.3A':
        from dexvision.evaluation.pick_repair import read_confirmation
        read_confirmation(spec, Path(spec['output_dir']) / 'confirmation_resets')
    run, directory = Path(run_dir), Path(output_dir)
    _verify_run(run)
    checkpoint = load_checkpoint(run / 'last.pt')
    if not checkpoint['training_state']['finished'] or (run / 'selected.pt').exists():
        raise ValueError("Pick selection requires finished unselected training")
    directory.mkdir(parents=True, exist_ok=False)
    reports = []
    for epoch, item in sorted(checkpoint['training_state']['candidates'].items(), key=lambda x: int(x[0])):
        print(f"Pick validation candidate {epoch}: {run}", flush=True)
        reports.append(evaluate_pick(config, run / item['path'], directory / f'epoch_{int(epoch):04d}',
                                     split='validation', release_root=release_root))
    _checksum_index(run)
    result = select_checkpoint(run, reports)
    _checksum_index(directory)
    return result


def review_pick(config, checkpoint_path):
    """Replay recorded physical states. Never rerun held-out physics or claim acceptance."""
    import mujoco.viewer
    spec, inputs = load_pick_spec(config)
    checkpoint_path = Path(checkpoint_path)
    _verify_run(checkpoint_path.parent)
    attempt = json.loads((checkpoint_path.parent / 'test_state_attempt.json').read_text())
    directory = Path(attempt['output_dir'])
    _verify_run(directory)
    report = json.loads((directory / 'report.json').read_text())
    if report['checkpoint_sha256'] != digest_file(checkpoint_path) or not report['state_numerical_gates_passed']:
        raise ValueError("Pick viewer requires passing numerical report for these weights")
    chosen = {}
    for result in report['results']:
        chosen.setdefault((result['object_family'], result['boundary_class']), result)
    rows = {r['episode_id']: r for r in inputs.splits['test']['episodes']}
    for result in chosen.values():
        child = directory / Path(result['result_path']).parent
        setup = json.loads((child / 'setup.json').read_text())
        with create_pick_task(inputs, rows[result['episode_id']], setup['metadata'], np.asarray(setup['prefix_actions'])) as task:
            print(f"Pick review seed {report['seed']} {result['object_family']}: {result['terminal_reason']}", flush=True)
            with np.load(child / 'trajectory.npz', allow_pickle=False) as trajectory, mujoco.viewer.launch_passive(task.env.model, task.env.data) as viewer:
                for index, (qpos, qvel) in enumerate(zip(trajectory['qpos'], trajectory['qvel'], strict=True)):
                    if not viewer.is_running():
                        return {'manual_acceptance': 'pending_owner_confirmation', 'cancelled': True}
                    if index:
                        apply_action(task, trajectory['applied_actions'][index - 1], inputs.plan['action']['named_layout'])
                    task.env.data.qpos[:] = qpos
                    task.env.data.qvel[:] = qvel
                    task.env.data.time = trajectory['state_timestamps'][index]
                    task.env._mujoco.mj_forward(task.env.model, task.env.data)
                    viewer.sync()
                    time.sleep(inputs.plan['action']['control_interval_s'])
    return {'manual_acceptance': 'pending_owner_confirmation', 'reviewed': len(chosen)}


def run_pick_evaluation(config, *, checkpoint, output_dir, release_root=None, track='state', split='test',
                        select_candidates=None, viewer=False, scenario_set=None):
    if track != 'state':
        raise ValueError("Checkpoint 5.3 implements state pick only")
    if viewer:
        if scenario_set != 'manual_stratified' or checkpoint is None or select_candidates:
            raise ValueError("Pick viewer needs selected checkpoint and manual_stratified")
        return review_pick(config, checkpoint)
    if scenario_set is not None or output_dir is None:
        raise ValueError("Pick evaluation needs a new output directory")
    if select_candidates is not None:
        if checkpoint is not None:
            raise ValueError("Choose checkpoint or candidate selection")
        return select_pick(config, select_candidates, output_dir, release_root)
    if checkpoint is None:
        raise ValueError("Pick evaluation requires checkpoint")
    return evaluate_pick(config, checkpoint, output_dir, split=split, release_root=release_root)
