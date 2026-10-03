"""Frozen session ownership and immutable publication for the pick data amendment."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import dexvision.logging.pick_amendment as amendment


def test_frozen_collection_has_disjoint_episode_seed_and_session_owners(tmp_path, monkeypatch):
    plan = {'splits': {s: {'seed_start': start, 'episodes_per_cell': 2, 'cells': ['cell_a', 'cell_b']}
                      for s, start in [('train', 1000), ('validation', 2000)]}}
    monkeypatch.setattr(amendment, 'load_data_plan', lambda _: (plan, {}, {}, SimpleNamespace(digests={})))
    assert amendment.freeze_data_plan('unused', tmp_path / 'data') == {'train': 4, 'validation': 4}
    frozen = json.loads((tmp_path / 'data/plan.json').read_text())
    for key in ('seed', 'episode_id', 'session_id'):
        assert not {r[key] for r in frozen['assignments']['train']} & {r[key] for r in frozen['assignments']['validation']}
    with pytest.raises(FileExistsError):
        amendment.freeze_data_plan('unused', tmp_path / 'data')


def test_publication_never_overwrites_an_existing_release(tmp_path, monkeypatch):
    plan = {'release_name': 'existing'}
    monkeypatch.setattr(amendment, '_frozen', lambda *args: (plan, {}, {}, None, {}))
    archive = tmp_path / 'existing.tar.gz'
    archive.write_bytes(b'original release')
    with pytest.raises(FileExistsError, match='Immutable'):
        amendment.publish_data('unused', tmp_path / 'staging', tmp_path)
    assert archive.read_bytes() == b'original release'


@pytest.mark.parametrize('failure', [None, 'failed', 'missing', 'release'])
def test_training_requires_complete_matching_physical_source_audit(tmp_path, failure):
    from dexvision.learning.pick_data import verify_amendment_audit
    from dexvision.learning.train_skill import _checksum_index
    spec = {'source_physical_audit': 'audit', 'unassisted_dataset': {'archive_sha256': 'frozen'}}
    inputs = SimpleNamespace(root=tmp_path, splits={'train': {'episodes': [{'episode_id': 'train'}]},
                                                  'validation': {'episodes': [{'episode_id': 'val'}]}})
    audits = [{'episode_id': eid, 'split': split, 'passed': not (failure == 'failed' and split == 'train')}
              for eid, split in [('train', 'train'), ('val', 'validation')]]
    report = {'passed': True, 'release': spec['unassisted_dataset'], 'episodes': 2, 'audits': audits}
    if failure == 'missing':
        report['audits'].pop()
    if failure == 'release':
        report['release'] = {'archive_sha256': 'another'}
    directory = tmp_path / 'audit'
    directory.mkdir()
    (directory / 'audit.json').write_text(json.dumps(report))
    _checksum_index(directory)
    if failure:
        with pytest.raises(ValueError, match='complete passing physical source audit'):
            verify_amendment_audit(inputs, spec)
    else:
        verify_amendment_audit(inputs, spec)
