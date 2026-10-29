"""Task specs, success predicates and observation features, one module per skill."""

from dexvision.tasks.pick import PickTask
from dexvision.tasks.place import PlaceTask
from dexvision.tasks.press import PressTask
from dexvision.tasks.push import PushTask
from dexvision.tasks.reach import ReachTask
from dexvision.tasks.spec import TASK_CONFIG_DIR, TaskSpec, load_task_spec

TASKS = {"pick": PickTask, "place": PlaceTask, "press": PressTask, "push": PushTask,
         "reach": ReachTask}
# every spec in configs/tasks/: the skills and their variants (e.g. pick_vision)
TASK_NAMES = tuple(sorted(p.stem for p in TASK_CONFIG_DIR.glob("*.yaml")))


def make_task(name: str):
    spec: TaskSpec = load_task_spec(name)
    return TASKS[spec.skill](spec)
