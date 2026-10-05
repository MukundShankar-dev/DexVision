"""Standalone state place rollouts, physical outcome gates and saved-state review."""
from __future__ import annotations

import io
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from dexvision.evaluation.benchmark_retargeters import mean_action_jerk
from dexvision.evaluation.evaluate_skill import SAFETY_KEYS, apply_action, reach_features, safe_action
from dexvision.evaluation.pick_policy import physical_evidence_world
from dexvision.learning.place_policy import PlacePolicy, load_place_spec, place_action_plan
from dexvision.learning.skill_datasets import _restore_initial
from dexvision.learning.train_skill import _checksum_index, _json, _verify_run, load_checkpoint, select_checkpoint
from dexvision.learning.training_manifest import ReleaseSource, content_digest, digest_file, skill_interval
from dexvision.logging.level4_collection import WorkcellPilotTask


def place_scenarios(inputs, split):
    if split not in {'validation', 'test'}:
        raise ValueError('Place evaluation requires validation or test')
    matrix = inputs.skill['evaluation'][split]
    rows = {r['episode_id']: r for r in inputs.splits[split]['episodes']}
    scenarios = []
    for cell in matrix['cells']:
        for eid, seed, frame in zip(cell['episode_ids'], cell['reset_seeds'], cell['start_frames'], strict=True):
            row = rows[eid]
            if row['cell_id'] != cell['cell_id'] or frame != skill_interval(inputs.skill, row)[0]:
                raise ValueError('Place matrix boundary/ownership mismatch')
            for condition in inputs.config_snapshot['evaluation']['rollouts']['perception_conditions']:
                scenarios.append((row, seed, frame, condition))
    if (len(scenarios) != matrix['rollouts_per_track_per_training_seed']
            or len({r[0]['episode_id'] for r in scenarios}) != matrix['unique_reset_count']):
        raise ValueError('Incomplete place matrix')
    return scenarios


def read_place_entries(inputs, scenarios, release_root=None):
    """Read only checksum-verified metadata and the strictly pre-entry prefix."""
    unique = {row['episode_id']: (row, seed, frame) for row, seed, frame, _ in scenarios}
    files = ('metadata.json', 'applied_actions.npy', 'online_phases.npy')
    names = [f"data/demos/level4/{row['source_path']}/{name}"
             for row, _, _ in unique.values() for name in files]
    blobs = ReleaseSource(inputs, release_root).read_many(names)
    entries = {}
    for eid, (row, seed, frame) in unique.items():
        prefix = f"data/demos/level4/{row['source_path']}/"
        metadata = json.loads(blobs[prefix + 'metadata.json'])
        if (metadata['random_seed'] != seed or metadata['typed_goal'] != row['typed_goal']
                or metadata['episode_id'] != eid or metadata['recording_session_id'] != row['recording_session_id']):
            raise ValueError('Place entry metadata mismatch')
        actions = np.load(io.BytesIO(blobs[prefix + 'applied_actions.npy']), allow_pickle=False)[:frame].copy()
        phases = np.load(io.BytesIO(blobs[prefix + 'online_phases.npy']), allow_pickle=False)[:frame].copy()
        if actions.shape != (frame, 27) or phases.shape != (frame,) or not np.isfinite(actions).all():
            raise ValueError('Invalid place setup prefix')
        entries[eid] = (metadata, actions, phases)
    return entries


def create_place_task(inputs, row, metadata, prefix, phases):
    """Replay the exact scripted pre-entry setup; after entry no object pose writes."""
    task = WorkcellPilotTask(workcell_config=inputs.root / inputs.plan['inputs']['workcell_config']['path'],
        dataset_config=inputs.root / inputs.plan['inputs']['dataset_config']['path'],
        skill_name='pick_place_sequence', goal_condition_id=row['cell_id'], seed=metadata['random_seed'],
        procedural_variation=metadata['task_config'].get('procedural_variation'))
    try:
        _restore_initial(task, metadata)
        grasp = metadata['teleop_config']['scripted_expert']['grasp']
        hold = grasp.get('orientation_preservation_policy') == 'shape_aware_hammer_grip_with_world_orientation_hold'
        chunk = int(grasp.get('orientation_hold_chunk_steps') or inputs.plan['action']['simulation_steps_per_action'])
        if chunk <= 0:
            raise ValueError('Invalid source orientation-hold cadence')
        steps = inputs.plan['action']['simulation_steps_per_action']
        object_id = row['entity_id']
        orientation = task.initial_world_state.require_entity(object_id).orientation_wxyz
        for action, phase in zip(prefix, phases, strict=True):
            apply_action(task, action, inputs.plan['action']['named_layout'])
            if hold and phase in {'lift', 'stabilize', 'transport', 'place'}:
                remaining = steps
                while remaining > chunk:
                    task.workcell.preserve_object_orientation(object_id, orientation)
                    task.workcell.step(n_steps=chunk)
                    remaining -= chunk
                task.workcell.preserve_object_orientation(object_id, orientation)
                task.step(n_steps=remaining)
            else:
                task.step(n_steps=steps)
        task._place_task.reset_dwell()
        return task
    except Exception:
        task.close()
        raise


def place_preconditions(world, row):
    object_id = row['typed_goal']['object_id']
    relation = world.relation_for(object_id)
    entity = world.require_entity(object_id)
    target = world.require_entity(row['typed_goal']['target_id'])
    held = [r.object_id for r in world.relations if r.held_by is not None]
    return {'exactly_one_held': held == [object_id], 'correct_object': entity.object_id == row['entity_id'],
        'unsupported': relation.supported_by is None, 'fresh_observation':
        entity.timestamp == world.timestamp and entity.confidence == 1.0
        and target.timestamp == world.timestamp and target.confidence == 1.0}


@dataclass
class PlaceEvidence:
    """Measured retention, release, target placement and consecutive settling."""
    object_id: str
    target_id: str
    spec: dict
    initial_position: tuple[float, float, float]
    transported: bool = False
    lowered: bool = False
    released: bool = False
    premature_release: bool = False
    nominally_placed: bool = False
    settle_dwell: int = 0
    max_post_release_speed: float = 0.0
    had_support_after_release: bool = False
    bounced: bool = False
    release_step: int | None = None
    steps: int = 0

    def update(self, world, frozen):
        self.steps += 1
        entity = world.require_entity(self.object_id)
        target = world.require_entity(self.target_id)
        relation = world.relation_for(self.object_id)
        held = relation.held_by == 'rh_palm'
        supported = relation.supported_by is not None
        distance = float(np.linalg.norm(np.asarray(entity.position) - target.position))
        speed = float(np.linalg.norm(entity.linear_velocity)) if entity.linear_velocity is not None else float('inf')
        angular = float(np.linalg.norm(entity.angular_velocity)) if entity.angular_velocity is not None else float('inf')
        self.transported |= held and float(np.linalg.norm(np.asarray(entity.position[:2])
            - np.asarray(self.initial_position[:2]))) >= self.spec['minimum_transport_displacement_m']
        self.lowered |= held and distance <= self.spec['release_distance_m']
        if not held and not self.released:
            self.released = True
            self.release_step = self.steps
            self.premature_release = distance > self.spec['release_distance_m']
            self.nominally_placed = not self.premature_release
        if self.released:
            self.max_post_release_speed = max(self.max_post_release_speed, speed)
            self.bounced |= self.had_support_after_release and (
                not supported or distance > self.spec['target_distance_m'])
            self.had_support_after_release |= supported
        stable = (self.released and not held and supported and distance <= self.spec['target_distance_m']
            and speed <= self.spec['maximum_linear_speed_mps']
            and angular <= self.spec['maximum_angular_speed_radps'])
        self.settle_dwell = self.settle_dwell + 1 if stable else 0
        return {'held_object_id': self.object_id if held else None, 'object_to_target_distance_m': distance,
            'object_linear_speed_mps': speed, 'object_angular_speed_radps': angular,
            'object_supported': supported, 'transported': self.transported, 'lowered': self.lowered,
            'released': self.released, 'premature_release': self.premature_release,
            'nominally_placed': self.nominally_placed, 'settle_steps': self.settle_dwell,
            'post_release_max_speed_mps': self.max_post_release_speed,
            'release_step': self.release_step, 'bounced': self.bounced,
            'frozen_sequence_metrics': dict(frozen.values), 'frozen_sequence_success': frozen.success,
            'success': self.settle_dwell >= self.spec['consecutive_samples'] and not self.bounced}


def run_place_rollout(inputs, spec, row, metadata, prefix, phases, policy, *, condition, output_dir):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    plan = place_action_plan(spec, inputs)
    fields = plan['action']['named_layout']
    arrays, worlds, metrics_history = defaultdict(list), [], []
    counts = dict.fromkeys(SAFETY_KEYS, 0)
    terminal, reason, metrics = 'timed_out', 'unsettled', {}
    with create_place_task(inputs, row, metadata, prefix, phases) as task:
        env = task.env
        initial = physical_evidence_world(task)
        prior = prefix[-1].copy() if len(prefix) else np.asarray(metadata['initial_applied_action'], float)
        commanded, safety = prior.copy(), np.zeros(27)
        checks = place_preconditions(initial, row)
        object_id, target_id = row['entity_id'], row['typed_goal']['target_id']
        evidence = PlaceEvidence(object_id, target_id, spec['terminal'], initial.require_entity(object_id).position)
        start = float(env.data.time)
        worlds.append(asdict(initial))
        for key, value in (('qpos', env.data.qpos), ('qvel', env.data.qvel), ('state_timestamps', start)):
            arrays[key].append(np.asarray(value).copy())
        if not all(checks.values()):
            terminal, reason = 'rejected_before_execution', 'rejected_precondition'
        else:
            for _ in range(inputs.skill['contract']['timeout_steps']):
                world = physical_evidence_world(task)
                source = reach_features(task, world, row, inputs, commanded, prior, safety, task._phase)
                raw, prediction = policy(source, prior, task._phase)
                applied, mask, count, unsafe = safe_action(raw, prior, plan)
                for key in SAFETY_KEYS:
                    counts[key] += count[key]
                for key, value in (('observations', source), ('raw_actions', raw), ('commanded_actions', raw),
                    ('applied_actions', applied), ('policy_predictions', prediction), ('prior_applied_actions', prior),
                    ('prior_commanded_actions', commanded), ('safety_masks', mask), ('action_timestamps', float(env.data.time))):
                    arrays[key].append(np.asarray(value).copy())
                apply_action(task, applied, fields)
                task.step(n_steps=inputs.plan['action']['simulation_steps_per_action'])
                world = physical_evidence_world(task)
                frozen = SimpleNamespace(values=task.current_state.task_values,
                    success=task.current_state.success)
                metrics = evidence.update(world, frozen)
                metrics_history.append(metrics)
                worlds.append(asdict(world))
                arrays['qpos'].append(env.data.qpos.copy())
                arrays['qvel'].append(env.data.qvel.copy())
                arrays['state_timestamps'].append(float(env.data.time))
                arrays['physical_phases'].append('settle' if evidence.released else
                    'lower' if evidence.lowered else 'transport')
                board = inputs.requirements['workcell']['board_workspace']
                xy = np.asarray(world.require_entity(object_id).position[:2])
                off_board = bool(np.any(xy < np.asarray(board['min_xy_m']) + board['safe_edge_margin_m']) or
                    np.any(xy > np.asarray(board['max_xy_m']) - board['safe_edge_margin_m']))
                counts['object_workspace_violations'] += int(off_board)
                entities = set(task.workcell.config.object_ids) | set(task.workcell.config.fixture_ids)
                wrong = any(r.held_by is not None and r.object_id != object_id for r in world.relations)
                unwanted_contacts = sum(any(name.startswith('rh_') for name in pair) and
                    any(name in entities and name not in {object_id, target_id} for name in pair)
                    for pair in world.contacts)
                metrics['unwanted_contact_count'] = unwanted_contacts
                failure = unsafe or ('object_workspace_violation' if off_board else
                    'wrong_object_held' if wrong else 'unwanted_contact' if unwanted_contacts else
                    'premature_release' if evidence.premature_release else 'bounce' if evidence.bounced else None)
                arrays['safety_reasons'].append(failure or 'none')
                prior, commanded, safety = applied, raw, mask
                if failure:
                    terminal, reason = 'failed', failure
                    break
                if metrics['success']:
                    terminal, reason = 'succeeded', 'physical_settled_placement_passed'
                    break
        end = float(env.data.time)
        object_spec = next(o for o in task.workcell.config.objects if o.object_id == object_id)
        variation = metadata['task_config'].get('procedural_variation') or {}
    np.savez_compressed(directory / 'trajectory.npz', **{k: np.asarray(v) for k, v in arrays.items()})
    _json(directory / 'world_states.json', {'states': worlds, 'metrics': metrics_history})
    _json(directory / 'setup.json', {'metadata': metadata, 'prefix_actions': prefix.tolist(),
        'prefix_phases': phases.tolist(), 'disclosure': 'Scripted setup only; no post-entry expert or orientation hold'})
    error = metrics['object_to_target_distance_m'] / spec['terminal']['target_distance_m'] if metrics else None
    result = {'episode_id': row['episode_id'], 'episode_digest': row['episode_digest'],
        'reset_seed': metadata['random_seed'], 'start_frame': len(prefix), 'cell_id': row['cell_id'],
        'session': row['recording_session_id'], 'source': row['source'],
        'object_family': object_spec.family, 'object_instance': object_id, 'target_id': target_id,
        'target_type': inputs.skill['target_type_mapping'][target_id],
        'boundary_class': variation.get('case_class', 'unspecified_in_source'),
        'render_condition': condition, 'render_condition_effect': 'none_on_state_or_physics; no_RGB_claim',
        'preconditions': checks, 'terminal_result': terminal, 'terminal_reason': reason, 'final_metrics': metrics,
        'start_timestamp': start, 'end_timestamp': end, 'retry_count': 0, 'safety_counters': counts,
        'normalized_terminal_error': error, 'transported': evidence.transported,
        'lowered': evidence.lowered, 'released': evidence.released,
        'premature_release': evidence.premature_release, 'nominally_placed': evidence.nominally_placed,
        'bounced': evidence.bounced,
        'settled': evidence.settle_dwell >= spec['terminal']['consecutive_samples'] and not evidence.bounced,
        'mean_action_jerk': mean_action_jerk(np.asarray(arrays['applied_actions']) / np.asarray([
            f['upper'] - f['lower'] for f in inputs.plan['action']['named_layout']])) if arrays['applied_actions'] else 0.,
        'trajectory_path': 'trajectory.npz', 'world_states_path': 'world_states.json'}
    _json(directory / 'result.json', result)
    _checksum_index(directory)
    return result


def summarize_place(results, inputs, split):
    expected = {(r['episode_id'], condition) for r, _, _, condition in place_scenarios(inputs, split)}
    if len(results) != len(expected) or {(r['episode_id'], r['render_condition']) for r in results} != expected:
        raise ValueError('Missing, duplicate or extra place attempts')
    breakdowns = {}
    for field in ('source', 'session', 'object_family', 'object_instance', 'target_id', 'target_type',
                  'cell_id', 'boundary_class', 'render_condition'):
        groups = defaultdict(list)
        for result in results:
            groups[result[field]].append(result)
        breakdowns[field] = {key: {'count': len(group),
            'success_rate': sum(r['terminal_result'] == 'succeeded' for r in group) / len(group),
            'premature_release_rate': sum(r['premature_release'] for r in group) / len(group),
            'terminal_reasons': dict(Counter(r['terminal_reason'] for r in group))}
            for key, group in sorted(groups.items())}
    success = sum(r['terminal_result'] == 'succeeded' for r in results) / len(results)
    worst = min(x['success_rate'] for x in breakdowns['object_family'].values())
    counts = {key: sum(r['safety_counters'][key] for r in results) for key in SAFETY_KEYS}
    premature = sum(r['premature_release'] for r in results) / len(results)
    bounced = sum(r['bounced'] for r in results) / len(results)
    unwanted = sum(r['final_metrics'].get('unwanted_contact_count', 0) for r in results)
    nominal = [r for r in results if r['nominally_placed']]
    settled = sum(r['settled'] for r in nominal) / len(nominal) if nominal else 0.
    jerk = float(np.mean([r['mean_action_jerk'] for r in results]))
    gates = {'aggregate_success': success >= inputs.skill['gates']['state_success_min'],
        'worst_family_success': worst >= inputs.skill['gates']['state_worst_family_success_min'],
        'premature_release_rate': premature <= inputs.skill['gates']['premature_release_rate_max'],
        'settled_among_nominally_placed': settled >= inputs.skill['gates']['settled_among_nominally_placed_min'],
        'mean_action_jerk': jerk <= inputs.skill['gates']['mean_action_jerk_max'],
        'unwanted_contact': unwanted == 0,
        **{key: value == 0 for key, value in counts.items()},
        'explicit_terminal_results': all(r['terminal_result'] in inputs.config_snapshot['evaluation']['terminal_results']
            for r in results)}
    errors = [r['normalized_terminal_error'] for r in results]
    return {'rollout_count': len(results), 'unique_reset_count': len({r['episode_id'] for r in results}),
        'success_rate': success, 'worst_family_success_rate': worst,
        'worst_cell_success_rate': min(x['success_rate'] for x in breakdowns['cell_id'].values()),
        'transport_retention_rate': sum(r['transported'] for r in results) / len(results),
        'target_placement_rate': sum(r['nominally_placed'] for r in results) / len(results),
        'release_rate': sum(r['released'] for r in results) / len(results),
        'premature_release_rate': premature, 'settled_among_nominally_placed': settled,
        'bounce_rate': bounced, 'unwanted_contact_count': unwanted,
        'mean_action_jerk': jerk, 'safety_violation_count': sum(counts.values()), 'safety_counters': counts,
        'normalized_terminal_error': float(np.mean([x for x in errors if x is not None])) if any(
            x is not None for x in errors) else None,
        'terminal_error_unavailable_reason': None if any(x is not None for x in errors) else 'rejected_precondition',
        'breakdowns': breakdowns, 'gates': gates, 'state_numerical_gates_passed': all(gates.values())}


def evaluate_place(config, checkpoint_path, output_dir, *, split='test', release_root=None):
    spec, inputs = load_place_spec(config)
    checkpoint_path, directory = Path(checkpoint_path), Path(output_dir)
    policy = PlacePolicy(checkpoint_path, spec, inputs)
    checkpoint = load_checkpoint(checkpoint_path)
    manifest = checkpoint['manifest']
    run = checkpoint_path.parent.parent if checkpoint_path.parent.name == 'validation_candidates' else checkpoint_path.parent
    if split == 'test':
        _verify_run(run)
        selection = json.loads((run / 'selection.json').read_text())
        if checkpoint_path.name != 'selected.pt' or selection['status'] != 'selected_from_validation':
            raise ValueError('Place test requires validation-selected weights')
        candidate_path = run / 'validation_candidates' / f"epoch_{selection['epoch']:04d}.pt"
        candidate = load_checkpoint(candidate_path)
        if digest_file(candidate_path) != selection['source_checkpoint_sha256'] or any(
                not torch.equal(value, candidate['model'][key]) for key, value in checkpoint['model'].items()):
            raise ValueError('Selected place weights differ from validation candidate')
        if (run / 'test_state_attempt.json').exists():
            raise ValueError('Single place test attempt already reserved')
    scenarios = place_scenarios(inputs, split)
    entries = read_place_entries(inputs, scenarios, release_root)
    plan = place_action_plan(spec, inputs)
    for metadata, prefix, _ in entries.values():
        prior = prefix[-1] if len(prefix) else np.asarray(metadata['initial_applied_action'])
        if safe_action(prior, prior, plan)[3]:
            raise ValueError('Place entry conflicts with frozen safety bounds')
    directory.mkdir(parents=True, exist_ok=False)
    provenance = {'version': 'level5/place-evaluation-v1', 'skill': 'place_held_object',
        'track': 'state', 'split': split, 'seed': manifest['seed'],
        'epoch': checkpoint['training_state']['completed_epochs'],
        'checkpoint_sha256': digest_file(checkpoint_path), 'checkpoint_path': str(checkpoint_path.resolve()),
        'output_dir': str(directory.resolve()), 'manifest_digest': manifest['manifest_digest'],
        'validation_matrix_sha256': manifest['validation_matrix_sha256'],
        'matrix_sha256': content_digest(inputs.skill['evaluation'][split]),
        'input_digests': inputs.digests, 'recipe_sha256': content_digest(spec),
        'selection_uses_test': False, 'execution_implementation_sha256': {
            p.relative_to(inputs.root).as_posix(): digest_file(p)
            for p in sorted((inputs.root / 'dexvision').rglob('*.py'))}}
    _json(directory / 'provenance.json', provenance)
    if split == 'test':
        with (run / 'test_state_attempt.json').open('x') as stream:
            json.dump(provenance, stream, indent=2)
    results = []
    for index, (row, _, _, condition) in enumerate(scenarios):
        if index % 20 == 0:
            print(f"Place {split}: {index}/{len(scenarios)} seed {manifest['seed']}", flush=True)
        child = directory / f"{index:03d}_{row['episode_id']}_{condition}"
        result = run_place_rollout(inputs, spec, row, *entries[row['episode_id']], policy,
            condition=condition, output_dir=child)
        result['result_path'] = f'{child.name}/result.json'
        results.append(result)
    report = {**provenance, **summarize_place(results, inputs, split), 'results': results,
        'training_history': checkpoint['training_state']['history']}
    _json(directory / 'report.json', report)
    _checksum_index(directory)
    if split == 'test':
        _checksum_index(run)
    return report


def select_place(config, run_dir, output_dir, release_root=None):
    run, directory = Path(run_dir), Path(output_dir)
    _verify_run(run)
    checkpoint = load_checkpoint(run / 'last.pt')
    if not checkpoint['training_state']['finished'] or (run / 'selected.pt').exists():
        raise ValueError('Place selection requires finished unselected training')
    directory.mkdir(parents=True, exist_ok=False)
    reports = []
    for epoch, item in sorted(checkpoint['training_state']['candidates'].items(), key=lambda x: int(x[0])):
        print(f'Place validation candidate {epoch}: {run}', flush=True)
        reports.append(evaluate_place(config, run / item['path'], directory / f'epoch_{int(epoch):04d}',
            split='validation', release_root=release_root))
    _checksum_index(run)
    result = select_checkpoint(run, reports)
    _checksum_index(directory)
    return result


def review_place(config, checkpoint_path):
    """Replay recorded physical states, leaving acceptance to the owner."""
    import mujoco.viewer

    _, inputs = load_place_spec(config)
    checkpoint_path = Path(checkpoint_path)
    _verify_run(checkpoint_path.parent)
    attempt = json.loads((checkpoint_path.parent / 'test_state_attempt.json').read_text())
    directory = Path(attempt['output_dir'])
    _verify_run(directory)
    report = json.loads((directory / 'report.json').read_text())
    if report['checkpoint_sha256'] != digest_file(checkpoint_path) or not report['state_numerical_gates_passed']:
        raise ValueError('Place viewer requires passing numerical report for these weights')
    chosen = {}
    for result in report['results']:
        chosen.setdefault(result['target_type'], result)
    rows = {r['episode_id']: r for r in inputs.splits['test']['episodes']}
    for result in chosen.values():
        child = directory / Path(result['result_path']).parent
        setup = json.loads((child / 'setup.json').read_text())
        with create_place_task(inputs, rows[result['episode_id']], setup['metadata'],
                np.asarray(setup['prefix_actions']), np.asarray(setup['prefix_phases'])) as task:
            print(f"Place review seed {report['seed']} {result['target_type']}: {result['terminal_reason']}", flush=True)
            with np.load(child / 'trajectory.npz', allow_pickle=False) as trajectory, \
                    mujoco.viewer.launch_passive(task.env.model, task.env.data) as viewer:
                for index, (qpos, qvel) in enumerate(zip(trajectory['qpos'], trajectory['qvel'], strict=True)):
                    if not viewer.is_running():
                        return {'manual_acceptance': 'pending_owner_confirmation', 'cancelled': True}
                    if index:
                        apply_action(task, trajectory['applied_actions'][index - 1],
                            inputs.plan['action']['named_layout'])
                    task.env.data.qpos[:] = qpos
                    task.env.data.qvel[:] = qvel
                    task.env.data.time = trajectory['state_timestamps'][index]
                    task.env._mujoco.mj_forward(task.env.model, task.env.data)
                    viewer.sync()
                    time.sleep(inputs.plan['action']['control_interval_s'])
    return {'manual_acceptance': 'pending_owner_confirmation', 'reviewed': len(chosen)}


def run_place_evaluation(config, *, checkpoint, output_dir, release_root=None, track='state', split='test',
                         select_candidates=None, viewer=False, scenario_set=None):
    if track != 'state':
        raise ValueError('Checkpoint 5.4 implements state place only')
    if viewer:
        if scenario_set != 'manual_stratified' or checkpoint is None or select_candidates:
            raise ValueError('Place viewer needs selected checkpoint and manual_stratified')
        return review_place(config, checkpoint)
    if scenario_set is not None or output_dir is None:
        raise ValueError('Place evaluation needs a new output directory')
    if select_candidates is not None:
        if checkpoint is not None:
            raise ValueError('Choose checkpoint or candidate selection')
        return select_place(config, select_candidates, output_dir, release_root)
    if checkpoint is None:
        raise ValueError('Place evaluation requires checkpoint')
    return evaluate_place(config, checkpoint, output_dir, split=split, release_root=release_root)
