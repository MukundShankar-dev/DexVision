import math
from dataclasses import replace

import numpy as np
import pytest

from dexvision.eval.evaluator import run_episode
from dexvision.policies import make_policy
from dexvision.policies.push_expert import support_distance
from dexvision.sim.workcell import Contact, Workcell
from dexvision.tasks import make_task
from dexvision.tasks.features import observation_dict
from dexvision.tasks.push import push_qualifies


def test_predicate_needs_the_object_near_the_goal_and_still_resting():
    goal = np.array([0.0, 0.0, 0.02])
    assert push_qualifies(np.array([0.019, 0.0, 0.02]), goal, 0.02, 0.02, 0.01)
    assert not push_qualifies(np.array([0.021, 0.0, 0.02]), goal, 0.02, 0.02, 0.01)
    assert not push_qualifies(np.array([0.0, 0.0, 0.035]), goal, 0.02, 0.02, 0.01)  # lifted
    assert not push_qualifies(np.array([0.0, 0.0, 0.005]), goal, 0.02, 0.02, 0.01)  # toppled


def test_support_distance_of_boxes_and_cylinders():
    eye = np.eye(3)
    assert support_distance("cylinder", np.array([0.02, 0.03, 0]), eye,
                            np.array([0.6, 0.8, 0])) == 0.02
    size = np.array([0.025, 0.015, 0.02])
    assert support_distance("box", size, eye, np.array([1.0, 0, 0])) == pytest.approx(0.025)
    diagonal = np.array([math.sqrt(0.5), math.sqrt(0.5), 0])
    assert support_distance("box", size, eye, diagonal) == pytest.approx(0.04 * math.sqrt(0.5))


@pytest.fixture(scope="module")
def workcell():
    return Workcell()


def test_goals_are_a_short_push_not_toward_the_arm(workcell):
    task = make_task("push")
    for seed in range(20):
        context, observation = task.reset(workcell, seed)
        start = observation.objects[context.target_object].position
        offset = context.goal_position - start
        assert 0.05 <= np.linalg.norm(offset[:2]) <= 0.08
        assert abs(math.atan2(offset[1], offset[0])) <= math.pi / 2 + 1e-9
        assert offset[2] == 0 and context.target_object in task.objects
        angle = task.episode_metrics()["push_angle_deg"]
        assert angle == pytest.approx(math.degrees(math.atan2(offset[1], offset[0])), abs=0.1)


def test_success_dwell_and_collision_failures(workcell):
    task = make_task("push")
    context, observation = task.reset(workcell, 2)
    target = context.target_object
    objects = dict(observation.objects)
    goal_xy = context.goal_position.copy()
    goal_xy[2] = objects[target].position[2]
    objects[target] = replace(objects[target], position=goal_xy)
    at_goal = replace(observation, objects=objects)
    checks = [task.check(at_goal, ()) for _ in range(task.dwell_steps)]
    assert [c.success for c in checks][-2:] == [False, True]
    touching_table = task.check(at_goal, (Contact("rh_ffdistal", "table", 2.0),))
    assert touching_table.failure is None  # pushing happens just above the table
    hit = task.check(at_goal, (Contact("fixture_wall", "rh_forearm", 2.0),))
    assert hit.failure == "hand_collision" and not hit.success


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_push_expert_succeeds_on_train_seeds(workcell, seed):
    task = make_task("push")
    assert run_episode(workcell, task, make_policy("expert", workcell.action_space, "push"),
                       seed).success


def test_push_expert_uses_only_declared_fields(workcell):
    task = make_task("push")
    expert = make_policy("expert", workcell.action_space, "push")
    context, observation = task.reset(workcell, 1)
    expert.reset(context, 1)

    class Strict(dict):
        def __getitem__(self, key):
            assert key in task.spec.observation, key
            return super().__getitem__(key)

    expert.act(Strict(observation_dict(task.spec.observation, observation, context)))


@pytest.mark.parametrize(("push_yaw_deg", "pusher"), [(0, "wall"), (40, "wall"), (50, "palm"),
                                                      (90, "palm"), (-50, "back"), (-90, "back")])
def test_push_expert_faces_the_push_with_the_least_hand_turn(workcell, push_yaw_deg, pusher):
    task = make_task("push")
    expert = make_policy("expert", workcell.action_space, "push")
    context, observation = task.reset(workcell, 0)
    obs = observation_dict(task.spec.observation, observation, context)
    goal = obs["object_position"] + 0.06 * np.array(
        [math.cos(math.radians(push_yaw_deg)), math.sin(math.radians(push_yaw_deg)), 0.0])
    expert.reset(replace(context, goal_position=goal), 0)
    expert.act({**obs, "goal_position": goal})
    assert expert.pusher == pusher
