from dataclasses import replace

import numpy as np
import pytest

from dexvision.eval.evaluator import run_episode
from dexvision.policies import make_policy
from dexvision.sim.workcell import Contact, Workcell
from dexvision.tasks import make_task
from dexvision.tasks.features import observation_dict
from dexvision.tasks.reach import hand_collisions, reach_qualifies


def test_predicate_uses_a_distance_tolerance():
    goal = np.array([0.0, 0.0, 0.1])
    assert reach_qualifies(goal + [0.015, 0, 0], goal, 0.015)
    assert not reach_qualifies(goal + [0.0151, 0, 0], goal, 0.015)


def test_hand_collisions_ignore_hand_self_contact():
    contacts = (Contact("rh_ffdistal", "rh_mfdistal", 1.0), Contact("table", "rh_palm", 3.0),
                Contact("rh_thdistal", "tray_left", 1.0), Contact("table", "block_small", 1.0))
    assert hand_collisions(contacts) == {"table", "tray_left"}


@pytest.fixture(scope="module")
def workcell():
    return Workcell()


def test_goals_are_inside_the_box_and_away_from_the_start(workcell):
    task = make_task("reach")
    lo = task._center - task._half_extent
    hi = task._center + task._half_extent
    for seed in range(20):
        context, observation = task.reset(workcell, seed)
        goal = context.goal_position
        assert np.all(goal >= lo) and np.all(goal <= hi)
        assert np.linalg.norm(goal - observation.grasp_position) >= 0.05
        assert context.target_object is None
    first, _ = task.reset(workcell, 7)
    again, _ = task.reset(workcell, 7)
    np.testing.assert_array_equal(first.goal_position, again.goal_position)


def test_success_needs_the_measured_point_to_dwell(workcell):
    task = make_task("reach")
    context, observation = task.reset(workcell, 3)
    at_goal = replace(observation, grasp_position=context.goal_position.copy(),
                      command_position=context.goal_position + 0.1)
    checks = [task.check(at_goal, ()) for _ in range(task.dwell_steps)]
    assert all(c.qualifies for c in checks)
    assert [c.success for c in checks] == [False] * (task.dwell_steps - 1) + [True]
    # the command at the goal while the measured point is elsewhere does not count
    commanded_only = replace(observation, command_position=context.goal_position.copy())
    assert not task.check(commanded_only, ()).qualifies


def test_touching_the_table_fails_the_episode(workcell):
    task = make_task("reach")
    context, observation = task.reset(workcell, 3)
    at_goal = replace(observation, grasp_position=context.goal_position.copy())
    check = task.check(at_goal, (Contact("rh_palm", "table", 5.0),))
    assert check.failure == "hand_collision" and not check.success


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_expert_reaches_on_train_seeds(workcell, seed):
    task = make_task("reach")
    assert run_episode(workcell, task, make_policy("expert", workcell.action_space, "reach"),
                       seed).success


def test_reach_expert_uses_only_declared_fields(workcell):
    task = make_task("reach")
    expert = make_policy("expert", workcell.action_space, "reach")
    context, observation = task.reset(workcell, 1)
    expert.reset(context, 1)

    class Strict(dict):
        def __getitem__(self, key):
            assert key in task.spec.observation, key
            return super().__getitem__(key)

    expert.act(Strict(observation_dict(task.spec.observation, observation, context)))


def test_every_skill_has_its_own_expert(workcell):
    assert type(make_policy("expert", workcell.action_space, "pick")).__name__ == "PickExpert"
    assert type(make_policy("expert", workcell.action_space, "reach")).__name__ == "ReachExpert"
    with pytest.raises(KeyError):
        make_policy("expert", workcell.action_space, "juggle")
