"""Independent MuJoCo saved-state reconstruction for a selected place repair."""
from __future__ import annotations

import json
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from dexvision.evaluation.evaluate_skill import apply_action
from dexvision.evaluation.pick_policy import physical_evidence_world
from dexvision.evaluation.place_policy import PlaceEvidence
from dexvision.evaluation.place_repair_policy import repair_matrix
from dexvision.learning.place_cadence_policy import load_cadence_spec
from dexvision.learning.place_guarded_policy import load_guarded_spec
from dexvision.learning.place_safety_policy import load_safety_spec
from dexvision.learning.place_slow_policy import load_slow_spec
from dexvision.learning.place_clearance_policy import load_clearance_spec
from dexvision.learning.place_slot_clearance_policy import load_slot_clearance_spec
from dexvision.learning.place_adaptive_policy import load_adaptive_spec
from dexvision.learning.place_recovery_policy import load_recovery_spec
from dexvision.learning.place_wrist_policy import load_wrist_spec
from dexvision.learning.place_physical_policy import load_physical_spec
from dexvision.learning.place_contact_policy import load_contact_spec
from dexvision.learning.train_skill import _checksum_index, _json, _verify_run, load_checkpoint
from dexvision.learning.training_manifest import content_digest, digest_file
from dexvision.logging.level4_collection import WorkcellPilotTask


def audit_repair_rollout(spec, inputs, row, result, child):
    """Restore every saved state and recompute the terminal from physical pose."""
    child = Path(child)
    _verify_run(child)
    saved_result = json.loads((child / 'result.json').read_text())
    if saved_result != {key: value for key, value in result.items() if key != 'result_path'}:
        raise ValueError('Place attempt result differs from report')
    with np.load(child / 'trajectory.npz', allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    if 'applied_actions' not in arrays:
        if (result['terminal_result'] != 'rejected_before_execution'
                or result['terminal_reason'] not in {
                    'pick_setup_failed', 'held_precondition_failed',
                    'place_setup_failed', 'rejected_precondition'}
                or set(arrays) != {'qpos', 'qvel', 'state_timestamps'}
                or len(arrays['qpos']) != 1 or len(arrays['qvel']) != 1
                or arrays['state_timestamps'].shape != (1,)
                or not np.isfinite(np.concatenate((arrays['qpos'].ravel(),
                    arrays['qvel'].ravel(), arrays['state_timestamps']))).all()):
            raise ValueError('Invalid rejected-before-place saved trajectory')
        return {'episode_id': row['episode_id'], 'render_condition': row['render_condition'],
            'passed': not any(result['safety_counters'].values()), 'frames': 0,
            'reported_success': False, 'physical_success': False,
            'excluded_before_place': True, 'first_opening_step': None,
            'trajectory_sha256': digest_file(child / 'trajectory.npz'),
            'result_sha256': digest_file(child / 'result.json')}
    count = len(arrays['applied_actions'])
    if (arrays['qpos'].shape[0] != count + 1 or arrays['qvel'].shape[0] != count + 1
            or arrays['state_timestamps'].shape != (count + 1,)
            or (count and (arrays['applied_actions'].shape != (count, 27)
                or arrays['prior_applied_actions'].shape != (count, 27)))
            or not np.isfinite(np.concatenate((arrays['qpos'].ravel(), arrays['qvel'].ravel(),
                arrays['state_timestamps'].ravel()))).all()
            or (count and not np.all(np.diff(arrays['state_timestamps']) > 0))):
        raise ValueError('Invalid saved place policy trajectory')
    if count and not np.array_equal(arrays['applied_actions'][:-1], arrays['prior_applied_actions'][1:]):
        raise ValueError('Place policy action/history alignment changed')
    opening = (np.flatnonzero(np.linalg.norm(arrays['applied_actions'][:, 7:]
        - arrays['prior_applied_actions'][:, 7:], axis=1) > 1e-8) if count else np.array([], int))
    first_opening = int(opening[0]) if len(opening) else None
    object_id, target_id = row['entity_id'], row['typed_goal']['target_id']
    board = inputs.requirements['workcell']['board_workspace']
    off_board, before_release_hand_table, release_window_hand_table = 0, 0, 0
    with WorkcellPilotTask(
            workcell_config=inputs.root / inputs.plan['inputs']['workcell_config']['path'],
            dataset_config=inputs.root / inputs.plan['inputs']['dataset_config']['path'],
            skill_name='pick_place_sequence', goal_condition_id=row['cell_id'], seed=row['seed'],
            procedural_variation=row['variation']) as task:

        def physical(index):
            if count:
                command = arrays['prior_applied_actions'][0] if index == 0 \
                    else arrays['applied_actions'][index - 1]
                apply_action(task, command, inputs.plan['action']['named_layout'])
            task.env.data.qpos[:] = arrays['qpos'][index]
            task.env.data.qvel[:] = arrays['qvel'][index]
            task.env.data.time = arrays['state_timestamps'][index]
            task.env._mujoco.mj_forward(task.env.model, task.env.data)
            return physical_evidence_world(task)

        initial = physical(0)
        evidence = PlaceEvidence(object_id, target_id, spec['terminal'],
                                 initial.require_entity(object_id).position)
        metrics = None
        for index in range(1, count + 1):
            world = physical(index)
            metrics = evidence.update(world, SimpleNamespace(values={}, success=False))
            xy = np.asarray(world.require_entity(object_id).position[:2])
            off_board += int(np.any(xy < np.asarray(board['min_xy_m']) + board['safe_edge_margin_m'])
                or np.any(xy > np.asarray(board['max_xy_m']) - board['safe_edge_margin_m']))
            hand_table = int(any('workcell_table' in pair and
                any(name.startswith('rh_') for name in pair) for pair in world.contacts))
            if first_opening is None or index <= first_opening:
                before_release_hand_table += hand_table
            else:
                release_window_hand_table += hand_table
        final_world = physical(count)
        final_object = final_world.require_entity(object_id)
        final_target = final_world.require_entity(target_id)
    reported_success = result['terminal_result'] == 'succeeded'
    physical_success = bool(metrics and metrics['success'])
    distance_match = bool(metrics is None or np.isclose(
        metrics['object_to_target_distance_m'],
        result['final_metrics']['object_to_target_distance_m'], atol=1e-8, rtol=0))
    passed = bool((not reported_success or (physical_success and first_opening is not None
        and not evidence.premature_release and not evidence.bounced and off_board == 0
        and before_release_hand_table == 0 and not any(result['safety_counters'].values())))
        and (reported_success or not physical_success) and distance_match)
    return {'episode_id': row['episode_id'], 'render_condition': row['render_condition'],
        'passed': passed, 'frames': count, 'reported_success': reported_success,
        'physical_success': physical_success, 'first_opening_step': first_opening,
        'premature_release': evidence.premature_release, 'bounced': evidence.bounced,
        'settle_steps': evidence.settle_dwell, 'object_workspace_samples': off_board,
        'before_release_command_hand_table_samples': before_release_hand_table,
        'release_window_hand_table_samples': release_window_hand_table,
        'final_distance_matches_report': distance_match,
        'final_object_position_m': list(final_object.position),
        'final_target_position_m': list(final_target.position),
        'trajectory_sha256': digest_file(child / 'trajectory.npz'),
        'result_sha256': digest_file(child / 'result.json')}


def audit_cadence_confirmation(config, checkpoint_path, report_dir, output_dir,
                               *, selection_config=None):
    version = yaml.safe_load(Path(config).read_text())['version']
    if version == 'level5/place-policy-pad-release-v25':
        from dexvision.learning.place_pad_release_policy import load_pad_release_spec
        overlay, spec, inputs = load_pad_release_spec(config)
        spec = copy.deepcopy(spec)
        spec['confirmation']['seed_start'] = overlay['confirmation_seed_start']
    elif version == 'level5/place-policy-wrist-hold-v24':
        from dexvision.learning.place_wrist_hold_policy import load_wrist_hold_spec
        overlay, spec, inputs = load_wrist_hold_spec(config)
        spec = copy.deepcopy(spec)
        spec['confirmation']['seed_start'] = overlay['confirmation_seed_start']
    elif version == 'level5/place-policy-bin-opening-v23':
        from dexvision.learning.place_bin_opening_policy import load_bin_opening_spec
        overlay, spec, inputs = load_bin_opening_spec(config)
        spec = copy.deepcopy(spec)
        spec['confirmation']['seed_start'] = overlay['confirmation_seed_start']
    elif version == 'level5/place-policy-supported-descent-v22':
        from dexvision.learning.place_supported_descent_policy import load_supported_descent_spec
        overlay, spec, inputs = load_supported_descent_spec(config)
        spec = copy.deepcopy(spec)
        spec['confirmation']['seed_start'] = overlay['confirmation_seed_start']
    elif version == 'level5/place-policy-shadow-v21':
        from dexvision.learning.place_shadow_policy import load_shadow_spec
        overlay, spec, inputs = load_shadow_spec(config)
        spec = copy.deepcopy(spec)
        spec['confirmation']['seed_start'] = overlay['confirmation_seed_start']
    elif version == 'level5/place-policy-object-calibration-v20':
        from dexvision.learning.place_object_calibration_policy import load_object_calibration_spec
        overlay, spec, inputs = load_object_calibration_spec(config)
        spec = copy.deepcopy(spec)
        spec['confirmation']['seed_start'] = overlay['confirmation_seed_start']
    elif version == 'level5/place-policy-full-source-v19':
        from dexvision.learning.place_full_source_policy import load_full_source_spec
        spec, inputs = load_full_source_spec(config)
        spec = copy.deepcopy(spec)
        spec['confirmation']['seed_start'] = spec['confirmation_seed_start']
    elif version == 'level5/place-policy-geometric-release-v18':
        from dexvision.learning.place_geometric_release_policy import load_geometric_release_spec
        overlay, spec, inputs = load_geometric_release_spec(config)
        spec = copy.deepcopy(spec)
        spec['confirmation']['seed_start'] = overlay['confirmation_seed_start']
    elif version == 'level5/place-policy-source-wrist-v17':
        from dexvision.learning.place_source_wrist_policy import load_source_wrist_spec
        overlay, spec, inputs = load_source_wrist_spec(config)
        spec = copy.deepcopy(spec)
        spec['confirmation']['seed_start'] = overlay['confirmation_seed_start']
    elif version == 'level5/place-policy-contact-v16':
        overlay, spec, inputs = load_contact_spec(config)
        spec = copy.deepcopy(spec)
        spec['confirmation']['seed_start'] = overlay['confirmation_seed_start']
    elif version == 'level5/place-policy-physical-v15':
        _, spec, inputs = load_physical_spec(config)
    elif version == 'level5/place-policy-wrist-v14':
        _, spec, inputs = load_wrist_spec(config)
    elif version == 'level5/place-policy-recovery-v13':
        _, spec, inputs = load_recovery_spec(config)
    elif version == 'level5/place-policy-adaptive-v12':
        _, spec, inputs = load_adaptive_spec(config)
    elif version == 'level5/place-policy-slot-clearance-v11':
        _, spec, inputs = load_slot_clearance_spec(config)
    elif version == 'level5/place-policy-clearance-v10':
        _, spec, inputs = load_clearance_spec(config)
    elif version == 'level5/place-policy-slow-v9':
        _, spec, inputs = load_slow_spec(config)
    elif version == 'level5/place-policy-safety-v8':
        _, spec, inputs = load_safety_spec(config)
    elif version == 'level5/place-policy-guarded-v6':
        spec, inputs = load_guarded_spec(config)
    elif version == 'level5/place-policy-cadence-v5':
        spec, inputs = load_cadence_spec(config)
    else:
        raise ValueError('Unsupported place repair audit recipe')
    checkpoint_path, report_dir, output_dir = map(Path, (checkpoint_path, report_dir, output_dir))
    if checkpoint_path.name != 'selected.pt':
        raise ValueError('Place physical audit requires selected weights')
    run = checkpoint_path.parent
    _verify_run(run)
    selected = json.loads((run / 'selection.json').read_text())
    candidate = run / 'validation_candidates' / f"epoch_{selected['epoch']:04d}.pt"
    if (selected['status'] != 'selected_from_validation'
            or digest_file(candidate) != selected['source_checkpoint_sha256']):
        raise ValueError('Place selection evidence differs from candidate')
    weights = load_checkpoint(checkpoint_path)['model']
    source = load_checkpoint(candidate)['model']
    if any(not torch.equal(weights[key], value) for key, value in source.items()):
        raise ValueError('Selected place weights differ from candidate')
    if selected.get('version') in {'level5/place-selection-v7', 'level5/place-safety-selection-v8',
                                   'level5/place-slow-selection-v9',
                                   'level5/place-clearance-selection-v10',
                                   'level5/place-slot-clearance-selection-v11',
                                   'level5/place-adaptive-selection-v12',
                                   'level5/place-recovery-selection-v13',
                                   'level5/place-wrist-selection-v14',
                                   'level5/place-physical-selection-v15',
                                   'level5/place-contact-selection-v16',
                                   'level5/place-source-wrist-selection-v17',
                                   'level5/place-geometric-release-selection-v18',
                                   'level5/place-full-source-selection-v19',
                                   'level5/place-object-calibration-selection-v20',
                                   'level5/place-shadow-selection-v21',
                                   'level5/place-supported-descent-selection-v22',
                                   'level5/place-bin-opening-selection-v23',
                                   'level5/place-wrist-hold-selection-v24',
                                   'level5/place-pad-release-selection-v25'}:
        if selected['version'] == 'level5/place-pad-release-selection-v25':
            from dexvision.evaluation.place_pad_release_selection import load_selection_spec
        elif selected['version'] == 'level5/place-wrist-hold-selection-v24':
            from dexvision.evaluation.place_wrist_hold_selection import load_selection_spec
        elif selected['version'] == 'level5/place-bin-opening-selection-v23':
            from dexvision.evaluation.place_bin_opening_selection import load_selection_spec
        elif selected['version'] == 'level5/place-supported-descent-selection-v22':
            from dexvision.evaluation.place_supported_descent_selection import load_selection_spec
        elif selected['version'] == 'level5/place-shadow-selection-v21':
            from dexvision.evaluation.place_shadow_selection import load_selection_spec
        elif selected['version'] == 'level5/place-object-calibration-selection-v20':
            from dexvision.evaluation.place_object_calibration_selection import load_selection_spec
        elif selected['version'] == 'level5/place-full-source-selection-v19':
            from dexvision.evaluation.place_full_source_selection import load_selection_spec
        elif selected['version'] == 'level5/place-geometric-release-selection-v18':
            from dexvision.evaluation.place_geometric_release_selection import load_selection_spec
        elif selected['version'] == 'level5/place-source-wrist-selection-v17':
            from dexvision.evaluation.place_source_wrist_selection import load_selection_spec
        elif selected['version'] == 'level5/place-contact-selection-v16':
            from dexvision.evaluation.place_contact_selection import load_selection_spec
        elif selected['version'] == 'level5/place-physical-selection-v15':
            from dexvision.evaluation.place_physical_selection import load_selection_spec
        elif selected['version'] == 'level5/place-wrist-selection-v14':
            from dexvision.evaluation.place_wrist_selection import load_selection_spec
        elif selected['version'] == 'level5/place-recovery-selection-v13':
            from dexvision.evaluation.place_recovery_selection import load_selection_spec
        elif selected['version'] == 'level5/place-adaptive-selection-v12':
            from dexvision.evaluation.place_adaptive_selection import load_selection_spec
        elif selected['version'] == 'level5/place-slot-clearance-selection-v11':
            from dexvision.evaluation.place_slot_clearance_selection import load_selection_spec
        elif selected['version'] == 'level5/place-clearance-selection-v10':
            from dexvision.evaluation.place_clearance_selection import load_selection_spec
        elif selected['version'] == 'level5/place-slow-selection-v9':
            from dexvision.evaluation.place_slow_selection import load_selection_spec
        elif selected['version'] == 'level5/place-safety-selection-v8':
            from dexvision.evaluation.place_safety_selection import load_selection_spec
        else:
            from dexvision.evaluation.place_selection import load_selection_spec

        if selection_config is None:
            raise ValueError('Place audit requires its selection recipe')
        protocol = load_selection_spec(selection_config)[0]
        seed = load_checkpoint(checkpoint_path)['manifest']['seed']
        screen_dir = Path(protocol['output_dir']) / 'screens' / f'seed_{seed}' / \
            f"epoch_{selected['epoch']:04d}"
        validation_root = protocol.get('validation_output_dir', protocol.get('policy_output_dir'))
        validation_dir = Path(validation_root) / f'validation_seed{seed}' / \
            f"epoch_{selected['epoch']:04d}"
        _verify_run(screen_dir)
        _verify_run(validation_dir)
        screen = json.loads((screen_dir / 'report.json').read_text())
        validation = json.loads((validation_dir / 'report.json').read_text())
        if (selected['selection_recipe_sha256'] != digest_file(Path(selection_config))
                or selected['all_target_report_sha256'] != digest_file(screen_dir / 'report.json')
                or selected['validation_report_sha256'] != digest_file(validation_dir / 'report.json')
                or not screen['state_numerical_gates_passed']
                or not validation['state_numerical_gates_passed']
                or not selected['all_target_gates_passed']
                or not selected['validation_gates_passed']):
            raise ValueError('Place selection evidence failed the independent audit')
        if selected['version'] in {'level5/place-contact-selection-v16',
                                   'level5/place-source-wrist-selection-v17',
                                   'level5/place-geometric-release-selection-v18',
                                   'level5/place-full-source-selection-v19',
                                   'level5/place-object-calibration-selection-v20',
                                   'level5/place-shadow-selection-v21',
                                   'level5/place-supported-descent-selection-v22',
                                   'level5/place-bin-opening-selection-v23',
                                   'level5/place-wrist-hold-selection-v24',
                                   'level5/place-pad-release-selection-v25'} and (
                not screen['physical_audit_passed']
                or not selected['all_target_physical_audit_passed']):
            raise ValueError('Place selection physical audit failed')
    elif selection_config is not None:
        raise ValueError('Place selection recipe supplied for an unselected bundle')
    _verify_run(report_dir)
    report = json.loads((report_dir / 'report.json').read_text())
    matrix = repair_matrix(spec, inputs, 'confirmation')
    conditions = inputs.config_snapshot['evaluation']['rollouts']['perception_conditions']
    expected = {(row['episode_id'], condition) for row in matrix for condition in conditions}
    received = {(item['episode_id'], item['render_condition']) for item in report['results']}
    if (report['split'] != 'confirmation' or report['checkpoint_sha256'] != digest_file(checkpoint_path)
            or report['recipe_sha256'] != digest_file(Path(config))
            or received != expected or len(report['results']) != len(expected)
            or report['matrix_sha256'] != content_digest([
                {**row, 'render_condition': condition} for row in matrix for condition in conditions])):
        raise ValueError('Place confirmation matrix or selected bundle mismatch')
    rows = {row['episode_id']: row for row in matrix}
    audits = []
    for index, item in enumerate(report['results']):
        if index % 20 == 0:
            print(f'Place physical audit {index}/{len(report["results"])}', flush=True)
        child = report_dir / Path(item['result_path']).parent
        row = {**rows[item['episode_id']], 'render_condition': item['render_condition']}
        audits.append(audit_repair_rollout(spec, inputs, row, item, child))
    output_dir.mkdir(parents=True, exist_ok=False)
    result = {'version': 'level5/place-policy-physical-audit-v1',
        'passed': all(item['passed'] for item in audits),
        'audited_attempts': len(audits), 'physical_successes': sum(item['physical_success'] for item in audits),
        'report_sha256': digest_file(report_dir / 'report.json'),
        'selected_checkpoint_sha256': digest_file(checkpoint_path),
        'selection_evidence_sha256': digest_file(run / 'selection.json'),
        'selection_recipe_sha256': digest_file(Path(selection_config)) if selection_config else None,
        'recipe_sha256': digest_file(Path(config)),
        'method': 'Saved qpos/qvel and prior command restored before independent MuJoCo forward',
        'audits': audits}
    _json(output_dir / 'report.json', result)
    _checksum_index(output_dir)
    return {key: value for key, value in result.items() if key != 'audits'}
