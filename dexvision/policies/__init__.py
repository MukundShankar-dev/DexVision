"""Controllers behind the common ``Policy`` interface."""

from dexvision.policies.baselines import RandomPolicy, ZeroPolicy
from dexvision.policies.pick_expert import PickExpert
from dexvision.policies.push_expert import PushExpert
from dexvision.policies.reach_expert import ReachExpert
from dexvision.sim.workcell import ActionSpace

POLICIES = ("zero", "random", "expert")
# one scripted expert per skill
EXPERTS = {"pick": PickExpert, "push": PushExpert, "reach": ReachExpert}


def make_policy(name: str, action_space: ActionSpace, skill: str):
    """``name`` is "zero", "random" or "expert"; "expert" means ``skill``'s scripted expert."""
    if name == "zero":
        return ZeroPolicy(action_space)
    if name == "random":
        return RandomPolicy(action_space)
    if name == "expert":
        if skill not in EXPERTS:
            raise KeyError(f"No scripted expert for '{skill}'. Available: {sorted(EXPERTS)}")
        return EXPERTS[skill](action_space)
    raise KeyError(f"Unknown policy '{name}'. Available: {list(POLICIES)}")
