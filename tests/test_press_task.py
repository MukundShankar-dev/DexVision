from dataclasses import replace

import pytest

from dexvision.eval.evaluator import run_episode
from dexvision.policies import make_policy
from dexvision.sim.workcell import Contact, Workcell
from dexvision.tasks import make_task
from dexvision.tasks.features import observation_dict


@pytest.fixture(scope="module")
def workcell():
    return Workcell()


def run_depths(task, observation, depths, contacts=()):
    return [task.check(replace(observation, button_depth=d), contacts) for d in depths]


def test_success_needs_a_full_press_then_a_release(workcell):
    task = make_task("press")
    _, observation = task.reset(workcell, 0)
    # released without ever pressing: never succeeds
    assert not any(c.success for c in run_depths(task, observation, [0.0] * 20))
    # a press one step too short does not count
    short = [0.016] * (task.press_steps - 1) + [0.0] * 20
    assert not any(c.success for c in run_depths(task, observation, short))
    # a full press, then released for release_steps
    checks = run_depths(task, observation, [0.016] * task.press_steps + [0.0] * task.release_steps)
    assert [c.success for c in checks][-2:] == [False, True]


def test_release_does_not_count_while_the_hand_touches_the_button(workcell):
    task = make_task("press")
    _, observation = task.reset(workcell, 0)
    run_depths(task, observation, [0.016] * task.press_steps)
    touching = (Contact("rh_ffdistal", "button", 0.1),)
    checks = run_depths(task, observation, [0.0] * 20, touching)
    assert not any(c.success for c in checks)


def test_overpressing_and_collisions_fail(workcell):
    task = make_task("press")
    _, observation = task.reset(workcell, 0)
    assert run_depths(task, observation, [0.025])[0].failure == "button_overpressed"
    hit = task.check(observation, (Contact("fixture_wall", "rh_ffdistal", 1.0),))
    assert hit.failure == "hand_collision"
    assert task.check(observation, (Contact("button", "rh_ffdistal", 1.0),)).failure is None


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_press_expert_succeeds_on_train_seeds(workcell, seed):
    task = make_task("press")
    assert run_episode(workcell, task, make_policy("expert", workcell.action_space, "press"),
                       seed).success


def test_press_expert_uses_only_declared_fields(workcell):
    task = make_task("press")
    expert = make_policy("expert", workcell.action_space, "press")
    context, observation = task.reset(workcell, 1)
    expert.reset(context, 1)

    class Strict(dict):
        def __getitem__(self, key):
            assert key in task.spec.observation, key
            return super().__getitem__(key)

    expert.act(Strict(observation_dict(task.spec.observation, observation, context)))
