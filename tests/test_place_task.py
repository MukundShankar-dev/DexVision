import math
from dataclasses import replace

import numpy as np
import pytest

from dexvision.eval.evaluator import run_episode
from dexvision.policies import make_policy
from dexvision.sim.workcell import Contact, Workcell
from dexvision.tasks import make_task
from dexvision.tasks.features import observation_dict
from dexvision.tasks.place import tilt


@pytest.fixture(scope="module")
def workcell():
    return Workcell()


def moved(observation, target, position, orientation=None, velocity=None):
    """The observation with the target object's measured state replaced."""
    state = observation.objects[target]
    state = replace(state, position=np.asarray(position, dtype=float),
                    orientation=state.orientation if orientation is None else orientation,
                    linear_velocity=state.linear_velocity if velocity is None else velocity)
    return replace(observation, objects={**observation.objects, target: state})


def test_tilt_of_upright_and_tipped_bodies():
    assert tilt(np.array([1.0, 0, 0, 0])) == pytest.approx(0.0)
    quarter = np.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0])  # 90 deg about x
    assert tilt(quarter) == pytest.approx(math.pi / 2)


def test_goals_are_in_the_region_and_a_real_carry_away(workcell):
    task = make_task("place")
    reset = task.spec.raw["reset"]
    for seed in range(30):
        context, observation = task.reset(workcell, seed)
        start = observation.objects[context.target_object].position
        goal = context.goal_position
        assert reset["goal_x_m"][0] <= goal[0] <= reset["goal_x_m"][1]
        assert reset["goal_y_m"][0] <= goal[1] <= reset["goal_y_m"][1]
        assert np.linalg.norm(goal[:2] - start[:2]) >= reset["min_goal_distance_m"]
        assert goal[2] == start[2]  # the object's centre when resting


def run(task, observations, contacts=()):
    return [task.check(o, contacts) for o in observations]


def test_success_needs_a_carry_then_a_released_upright_rest_on_the_goal(workcell):
    task = make_task("place")
    context, observation = task.reset(workcell, 3)
    target, goal = context.target_object, context.goal_position
    at_goal = moved(observation, target, goal)
    # slid to the goal without being lifted: never succeeds
    assert not any(c.success for c in run(task, [at_goal] * 30))

    task.reset(workcell, 3)
    lifted = moved(observation, target, goal + [0, 0, task.min_lift_m + 0.01])
    checks = run(task, [lifted] + [at_goal] * task.dwell_steps)
    assert [c.success for c in checks][-2:] == [False, True]


def test_holding_tipped_off_goal_or_dropped_through_do_not_count(workcell):
    task = make_task("place")
    context, observation = task.reset(workcell, 3)
    target, goal = context.target_object, context.goal_position
    lift = moved(observation, target, goal + [0, 0, 0.05])
    held = (Contact(target, "rh_thdistal", 1.0),)
    tipped = np.array([math.cos(0.15), math.sin(0.15), 0, 0])  # 0.3 rad
    for state, contacts in [
        (moved(observation, target, goal), held),  # still in the hand
        (moved(observation, target, goal, orientation=tipped), ()),  # on its side-ish
        (moved(observation, target, goal + [0.025, 0, 0]), ()),  # 2.5 cm off
        (moved(observation, target, goal + [0, 0, 0.01]), ()),  # not resting on the table
    ]:
        task.reset(workcell, 3)
        assert not any(c.success for c in run(task, [lift] + [state] * 20, contacts))


def test_landing_speed_and_failures(workcell):
    task = make_task("place")
    context, observation = task.reset(workcell, 3)
    target, goal = context.target_object, context.goal_position
    run(task, [moved(observation, target, goal + [0, 0, 0.05])])
    landing = moved(observation, target, goal, velocity=np.array([0.0, 0.0, -0.3]))
    task.check(landing, (Contact(target, "table", 1.0),))
    assert task.episode_metrics()["landing_speed_mps"] == pytest.approx(0.3)

    hit = task.check(observation, (Contact("tray_left", "rh_ffdistal", 1.0),))
    assert hit.failure == "hand_collision"
    assert task.check(observation, (Contact("table", "rh_thdistal", 1.0),)).failure is None
    fell = moved(observation, target, [0.3, 0.0, -0.05])
    assert task.check(fell, ()).failure == "object_fell_off_table"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_place_expert_succeeds_on_train_seeds(workcell, seed):
    task = make_task("place")
    expert = make_policy("expert", workcell.action_space, "place")
    assert run_episode(workcell, task, expert, seed).success


def test_place_expert_uses_only_declared_fields(workcell):
    task = make_task("place")
    expert = make_policy("expert", workcell.action_space, "place")
    context, observation = task.reset(workcell, 1)
    expert.reset(context, 1)

    class Strict(dict):
        def __getitem__(self, key):
            assert key in task.spec.observation, key
            return super().__getitem__(key)

    fields = observation_dict(task.spec.observation, observation, context)
    for _ in range(5):
        expert.act(Strict(fields))
