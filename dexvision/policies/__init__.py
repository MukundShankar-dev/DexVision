"""Controllers behind the common ``Policy`` interface."""

from dexvision.policies.baselines import RandomPolicy, ZeroPolicy
from dexvision.sim.workcell import ActionSpace

POLICIES = {"zero": ZeroPolicy, "random": RandomPolicy}


def make_policy(name: str, action_space: ActionSpace):
    if name not in POLICIES:
        raise KeyError(f"Unknown policy '{name}'. Available: {sorted(POLICIES)}")
    return POLICIES[name](action_space)
