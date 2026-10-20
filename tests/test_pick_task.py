from dataclasses import replace

import numpy as np
import pytest

from dexvision.sim.workcell import Contact, Workcell
from dexvision.tasks import make_task
from dexvision.tasks.contacts import held_by_opposition
from dexvision.tasks.pick import pick_qualifies

THUMB, INDEX, MIDDLE = "rh_thdistal", "rh_ffdistal", "rh_mfdistal"


@pytest.mark.parametrize(
    ("touching", "held"),
    [
        ({THUMB, INDEX}, True),
        ({THUMB, INDEX, MIDDLE, "rh_palm"}, True),
        ({INDEX, MIDDLE}, False),  # no thumb: pushed, not grasped
        ({THUMB}, False),  # one digit cannot oppose itself
        ({THUMB, "rh_palm"}, False),  # palm is not a finger
        ({THUMB, INDEX, "table"}, False),  # still resting on the table
        (set(), False),
    ],
)
def test_held_requires_thumb_opposition_and_nothing_else(touching, held):
    assert held_by_opposition(touching) is held


def test_pick_predicate_requires_lift():
    assert pick_qualifies(0.05, {THUMB, INDEX}, 0.05)
    assert not pick_qualifies(0.049, {THUMB, INDEX}, 0.05)


@pytest.fixture(scope="module")
def workcell():
    return Workcell()


def lifted(observation, target, dz):
    """A constructed observation with the target raised by dz. The simulator is not touched."""
    objects = dict(observation.objects)
    objects[target] = replace(objects[target], position=objects[target].position + [0, 0, dz])
    return replace(observation, objects=objects)


def grasp_contacts(target):
    return (Contact(target, THUMB, 2.0), Contact(INDEX, target, 2.0))


def test_success_needs_full_dwell_and_resets_when_broken(workcell):
    task = make_task("pick")
    context, observation = task.reset(workcell, 100000)
    target = context.target_object
    up = lifted(observation, target, 0.06)
    for _ in range(task.dwell_steps - 1):
        check = task.check(up, grasp_contacts(target))
        assert check.qualifies and not check.success
    assert not task.check(up, ()).qualifies  # contact lost: dwell restarts
    for _ in range(task.dwell_steps - 1):
        assert not task.check(up, grasp_contacts(target)).success
    assert task.check(up, grasp_contacts(target)).success


def test_object_on_floor_is_a_failure(workcell):
    task = make_task("pick")
    context, observation = task.reset(workcell, 100001)
    check = task.check(lifted(observation, context.target_object, -0.1), ())
    assert check.failure == "object_fell_off_table"


def test_reset_samples_every_object_and_respects_hand_jitter(workcell):
    task = make_task("pick")
    targets = set()
    for seed in range(100000, 100030):
        context, observation = task.reset(workcell, seed)
        targets.add(context.target_object)
        assert context.target_object in task.objects
        assert observation.objects[context.target_object].position[2] > 0
        offset = observation.command_position - np.array([-0.16, 0.0, 0.20])
        assert np.all(np.abs(offset) <= 0.02 + 1e-9)
    assert targets == set(task.objects)


def test_untouched_object_never_succeeds(workcell):
    task = make_task("pick")
    context, observation = task.reset(workcell, 100002)
    for _ in range(20):
        observation = workcell.step(workcell.action_space.hold())
        assert not task.check(observation, workcell.contacts()).qualifies
