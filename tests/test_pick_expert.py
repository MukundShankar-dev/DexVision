import math

import numpy as np
import pytest

from dexvision.data.demos import DemoRecorder, load_data_config, noise_std_vector
from dexvision.eval.evaluator import run_episode
from dexvision.policies.pick_expert import PickExpert, grasp_yaw, wrap
from dexvision.sim.workcell import OBJECT_IDS, Workcell
from dexvision.tasks import make_task
from dexvision.tasks.features import TaskContext, rotation6d, rotation_from_6d


def yaw_rotation(yaw):
    return rotation_from_6d(rotation6d([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]))


def test_rotation_from_6d_inverts_rotation6d():
    q = np.array([0.8, 0.2, -0.3, 0.4])
    q /= np.linalg.norm(q)
    matrix = rotation_from_6d(rotation6d(q))
    np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(rotation6d(q), matrix[:, :2].T.reshape(6))


def test_wrap():
    assert wrap(3 * math.pi / 2) == pytest.approx(-math.pi / 2)
    assert wrap(0.7, math.pi / 2) == pytest.approx(0.7)  # already inside (-pi/4, pi/4]
    assert wrap(1.0, math.pi / 2) == pytest.approx(1.0 - math.pi / 2)


@pytest.mark.parametrize("object_yaw", [-2.5, -0.4, 0.0, 0.3, 1.9, 3.0])
def test_grasp_yaw_aligns_fingers_with_the_short_side(object_yaw):
    rotation = yaw_rotation(object_yaw)
    assert grasp_yaw(rotation, "cylinder", np.array([0.02, 0.03, 0])) == 0.0
    cube = grasp_yaw(rotation, "box", np.array([0.0175, 0.0175, 0.0175]))
    assert abs(cube) <= math.pi / 4 + 1e-9
    assert math.cos(4 * (cube - object_yaw)) == pytest.approx(1.0)
    # block_large: the short side is the object's y axis.
    hand = grasp_yaw(rotation, "box", np.array([0.025, 0.015, 0.02]))
    assert abs(hand) <= math.pi / 2 + 1e-9
    fingers = np.array([math.cos(hand), math.sin(hand), 0.0])
    assert abs(fingers @ rotation[:, 1]) == pytest.approx(1.0)


@pytest.fixture(scope="module")
def workcell():
    return Workcell()


@pytest.mark.parametrize("seed", [0, 3, 5])  # cylinder_tall, puck_heavy, block_small
def test_expert_picks_on_train_seeds(workcell, seed):
    task = make_task("pick")
    result = run_episode(workcell, task, PickExpert(workcell.action_space), seed)
    assert result.success, result


def test_grasp_is_chosen_by_object_height_only(workcell):
    expert = PickExpert(workcell.action_space)
    for name in OBJECT_IDS:
        shape, size = workcell.object_geometry(name)
        expert.reset(TaskContext("pick", name, shape, size), 0)
        expected = "flat" if name.startswith("puck") else "default"
        assert expert._plan is expert._grasps[expected], name


def test_flat_grasp_survives_dart_noise_on_a_light_puck(workcell):
    task = make_task("pick")
    seed = next(s for s in range(100)
                if task.objects[int(np.random.default_rng([s, 1]).integers(6))] == "puck_light")
    noise = noise_std_vector(load_data_config("pick")["noise_std"])
    recorder = DemoRecorder(PickExpert(workcell.action_space), workcell.action_space, noise)
    assert run_episode(workcell, task, recorder, seed).success


def test_expert_uses_only_declared_observation_fields(workcell):
    task = make_task("pick")
    expert = PickExpert(workcell.action_space)
    context, observation = task.reset(workcell, 1)
    expert.reset(context, 1)

    class Strict(dict):
        def __getitem__(self, key):
            assert key in task.spec.observation, key
            return super().__getitem__(key)

    from dexvision.tasks.features import observation_dict

    expert.act(Strict(observation_dict(task.spec.observation, observation, context)))
