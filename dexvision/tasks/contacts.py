"""Classify physical contacts by which part of the hand they involve."""

from __future__ import annotations

from collections.abc import Iterable

from dexvision.sim.workcell import Contact

# Digit links that can hold an object. Knuckles, the thumb base and the little-finger
# metacarpal sit inside the palm, so contacts on them count as palm contacts.
THUMB_LINKS = frozenset({"rh_thproximal", "rh_thhub", "rh_thmiddle", "rh_thdistal"})
FINGER_LINKS = frozenset(
    f"rh_{finger}{link}"
    for finger in ("ff", "mf", "rf", "lf")
    for link in ("proximal", "middle", "distal")
)


def is_hand(body: str) -> bool:
    return body.startswith("rh_")


def bodies_touching(contacts: Iterable[Contact], body: str) -> set[str]:
    """Names of all bodies currently in contact with ``body``."""
    touching = set()
    for contact in contacts:
        if contact.body1 == body:
            touching.add(contact.body2)
        elif contact.body2 == body:
            touching.add(contact.body1)
    return touching


def held_by_opposition(touching: set[str]) -> bool:
    """True if the thumb and at least one other finger touch the object, and nothing else does
    apart from the hand. This is the physical signature of a grasp, not of a nudge."""
    if any(not is_hand(body) for body in touching):
        return False
    return bool(touching & THUMB_LINKS) and bool(touching & FINGER_LINKS)
