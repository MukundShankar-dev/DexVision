"""Task specs, success predicates and observation features, one module per skill."""

from dexvision.tasks.pick import PickTask
from dexvision.tasks.push import PushTask
from dexvision.tasks.reach import ReachTask
from dexvision.tasks.spec import TaskSpec, load_task_spec

TASKS = {"pick": PickTask, "push": PushTask, "reach": ReachTask}


def make_task(skill: str):
    spec: TaskSpec = load_task_spec(skill)
    return TASKS[spec.skill](spec)
