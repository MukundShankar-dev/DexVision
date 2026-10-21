import numpy as np
import pytest

from dexvision.sim.workcell import Workcell
from dexvision.tasks import TASKS, make_task
from dexvision.tasks.features import FEATURES, observation_dict, rotation6d

EXPECTED_SIZES = {
    "palm_position": 3, "palm_rotation6d": 6, "hand_qpos": 24, "hand_qvel": 24,
    "fingertip_positions": 15, "object_position": 3, "object_rotation6d": 6,
    "object_linear_velocity": 3, "object_angular_velocity": 3, "object_shape": 2,
    "object_size": 3, "grasp_position": 3, "goal_position": 3,
}


def test_rotation6d_of_identity_and_quarter_turn():
    np.testing.assert_allclose(rotation6d([1, 0, 0, 0]), [1, 0, 0, 0, 1, 0])
    yaw90 = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]
    np.testing.assert_allclose(rotation6d(yaw90), [0, 1, 0, -1, 0, 0], atol=1e-12)


def test_every_feature_has_documented_size():
    assert set(FEATURES) == set(EXPECTED_SIZES)
    workcell = Workcell()
    covered = set()
    for skill in TASKS:
        task = make_task(skill)
        context, observation = task.reset(workcell, 100000)
        values = observation_dict(task.spec.observation, observation, context)
        for name, value in values.items():
            assert value.shape == (EXPECTED_SIZES[name],), (skill, name)
            assert np.all(np.isfinite(value)), (skill, name)
        covered |= set(values)
    assert covered == set(FEATURES)  # every feature is declared by some task and checked


def test_unknown_feature_is_an_error():
    workcell = Workcell()
    task = make_task("pick")
    context, observation = task.reset(workcell, 100000)
    with pytest.raises(KeyError):
        observation_dict(("object_position", "contact_forces"), observation, context)
