"""An amendment cannot hide a missing, altered, or cross-split replacement."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from dexvision.evaluation.dataset_audit import apply_puck_replacement
from dexvision.evaluation.split_audit import content_digest, file_inventory
from dexvision.logging.visual_stream import digest_file


@pytest.fixture
def amendment(tmp_path):
    config = tmp_path / 'dataset.yaml'
    config.write_text('frozen: true\n')
    episodes, sessions = [], {}
    for eid, seed in [('old', 1), ('new', 2)]:
        path = tmp_path / eid
        path.mkdir()
        (path / 'actions.npy').write_bytes(eid.encode())
        episodes.append(SimpleNamespace(episode_id=eid, session_id=eid, path=path,
            goal_condition_id='pp_puck_light_return_bin_left', skill_name='pick_place_sequence',
            source='scripted', metadata={'random_seed': seed, 'operator_id': 'expert'}))
        sessions[eid] = SimpleNamespace(split='train')
    receipt_path = tmp_path / 'receipt.json'
    plan = {'version': 'level4/puck-replacement-plan-v1',
            'original_episode_id': 'old', 'replacement_episode_id': 'new',
            'original_episode_digest': content_digest(file_inventory(episodes[0].path)),
            'cell_id': episodes[0].goal_condition_id, 'skill': 'pick_place_sequence',
            'split': 'train', 'recording_session_id': 'new', 'seed': 2,
            'operator_id': 'expert', 'dataset_config': str(config),
            'dataset_config_sha256': digest_file(config), 'receipt_path': str(receipt_path)}
    plan_path = tmp_path / 'plan.yaml'
    plan_path.write_text(yaml.safe_dump(plan))
    receipt_path.write_text(json.dumps({'version': 'level4/puck-replacement-receipt-v1',
        'plan_sha256': digest_file(plan_path),
        'replacement_episode_digest': content_digest(file_inventory(episodes[1].path))}))
    return episodes, sessions, plan_path


def test_valid_amendment_preserves_original_and_exposes_lineage(amendment):
    episodes, sessions, plan = amendment
    before = [file_inventory(e.path) for e in episodes]
    active, excluded, hashes = apply_puck_replacement(episodes, plan_path=plan, sessions=sessions)
    assert active == [episodes[1]]
    assert excluded['episode_id'] == 'old'
    assert excluded['replacement_episode_id'] == 'new'
    assert excluded['episode_digest'] == content_digest(before[0])
    assert [file_inventory(e.path) for e in episodes] == before
    assert hashes[str(plan)] == digest_file(plan)


@pytest.mark.parametrize('failure', ['missing', 'duplicate', 'original_bytes', 'replacement_bytes',
                                     'split', 'seed', 'plan', 'receipt'])
def test_invalid_amendment_cannot_remove_original(amendment, failure):
    episodes, sessions, plan = amendment
    if failure == 'missing':
        episodes = episodes[:1]
    elif failure == 'duplicate':
        episodes = [*episodes, episodes[1]]
    elif failure.endswith('_bytes'):
        index = 0 if failure == 'original_bytes' else 1
        (episodes[index].path / 'actions.npy').write_bytes(b'changed')
    elif failure == 'split':
        sessions['new'].split = 'test'
    elif failure == 'seed':
        episodes[1].metadata['random_seed'] = 3
    elif failure == 'plan':
        plan.write_text(plan.read_text() + '# changed\n')
    else:
        receipt = Path(yaml.safe_load(plan.read_text())['receipt_path'])
        payload = json.loads(receipt.read_text())
        payload['replacement_episode_digest'] = 'untrusted'
        receipt.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        apply_puck_replacement(episodes, plan_path=plan, sessions=sessions)
