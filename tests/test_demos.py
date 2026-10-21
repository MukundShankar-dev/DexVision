import json

import numpy as np
import pytest

from dexvision.data.demos import (
    DemoRecorder,
    generate,
    load_data_config,
    load_episode,
    noise_std_vector,
    replay_matches,
)
from dexvision.eval.evaluator import run_episode
from dexvision.policies import make_policy
from dexvision.sim.workcell import ACTION_DIM, Workcell
from dexvision.tasks import make_task


@pytest.fixture(scope="module")
def workcell():
    return Workcell()


@pytest.fixture(scope="module")
def dataset(workcell, tmp_path_factory):
    config = dict(load_data_config("pick"), episodes=3)
    out = tmp_path_factory.mktemp("demos") / "pick"
    task = make_task("pick")
    expert = make_policy("expert", workcell.action_space, "pick")
    manifest = generate(workcell, task, expert, config, out)
    return out, manifest, task


def test_action_space_clip_matches_step_limits(workcell):
    space = workcell.action_space
    action = np.full(ACTION_DIM, 10.0)
    clipped = space.clip(action)
    assert np.linalg.norm(clipped[:3]) == pytest.approx(space.max_translation_m)
    assert np.linalg.norm(clipped[3:6]) == pytest.approx(space.max_rotation_rad)
    np.testing.assert_array_equal(clipped[6:], space.target_high)
    inside = space.hold()
    np.testing.assert_array_equal(space.clip(inside), inside)


def test_recorder_labels_are_clean_and_execution_is_noisy(workcell):
    task = make_task("pick")
    expert = make_policy("expert", workcell.action_space, "pick")
    std = noise_std_vector(load_data_config("pick")["noise_std"])
    recorder = DemoRecorder(expert, workcell.action_space, std)
    result = run_episode(workcell, task, recorder, 7)
    arrays = recorder.arrays(result)
    T = result.steps
    assert arrays["action"].shape == arrays["executed_action"].shape == (T, ACTION_DIM)
    assert all(arrays[f"obs/{k}"].shape[0] == T for k in task.spec.observation)
    noise = arrays["executed_action"] - arrays["action"]
    assert np.all(np.abs(noise[:, :3].std(axis=0) - std[:3]) < 0.5 * std[:3])
    space, labels = workcell.action_space, arrays["action"]
    assert np.all(np.linalg.norm(labels[:, :3], axis=1) <= space.max_translation_m + 1e-12)
    assert np.all(np.linalg.norm(labels[:, 3:6], axis=1) <= space.max_rotation_rad + 1e-12)
    assert np.all((labels[:, 6:] >= space.target_low) & (labels[:, 6:] <= space.target_high))


def test_zero_noise_recording_equals_the_plain_expert(workcell):
    task = make_task("pick")
    expert = make_policy("expert", workcell.action_space, "pick")
    recorder = DemoRecorder(expert, workcell.action_space, np.zeros(ACTION_DIM))
    recorded = run_episode(workcell, task, recorder, 4)
    plain = run_episode(workcell, task, make_policy("expert", workcell.action_space, "pick"), 4)
    assert recorded == plain


def test_generated_episodes_replay_exactly(workcell, dataset):
    out, manifest, task = dataset
    paths = sorted(out.glob("episode_*.npz"))
    assert len(paths) == manifest["kept"]["successes"] > 0
    assert manifest["attempted"] == 3 and manifest["seeds"] == [0, 3]
    for path in paths:
        episode = load_episode(path)
        assert bool(episode["success"])
        assert replay_matches(workcell, task, episode)


def test_tampered_actions_do_not_replay(workcell, dataset):
    out, _, task = dataset
    episode = load_episode(sorted(out.glob("episode_*.npz"))[0])
    episode["executed_action"] = episode["executed_action"].copy()
    episode["executed_action"][5, 0] += 0.005
    assert not replay_matches(workcell, task, episode)


def test_manifest_records_provenance_and_hashes(dataset):
    out, manifest, _ = dataset
    on_disk = json.loads((out / "manifest.json").read_text())
    assert on_disk["files"] == manifest["files"]
    assert set(manifest["files"]) == {p.name for p in out.glob("episode_*.npz")}
    assert {"commit", "dirty"} <= set(manifest["code"])
    assert manifest["config"]["noise_std"]["translation_m"] > 0
    with pytest.raises(FileExistsError):
        generate(None, None, None, load_data_config("pick"), out)
