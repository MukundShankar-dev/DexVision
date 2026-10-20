"""Open the workcell in the interactive MuJoCo viewer (macOS: run with ``mjpython``).

    mjpython -m dexvision.apps.view_workcell --seed 0
"""

from __future__ import annotations

import argparse

import mujoco.viewer

from dexvision.sim.workcell import OBJECT_IDS, Placement, Workcell

# A spread-out layout for looking at the scene. Task specs define the real reset distributions.
VIEW_LAYOUT = {
    name: Placement((x, y), (0.015, 0.015))
    for name, (x, y) in zip(
        OBJECT_IDS,
        [(-0.12, -0.10), (-0.12, 0.0), (-0.12, 0.10), (-0.04, -0.10), (-0.04, 0.0), (-0.04, 0.10)],
        strict=True,
    )
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    workcell = Workcell()
    workcell.reset(args.seed, VIEW_LAYOUT)
    print(f"Workcell reset with seed {args.seed}. Close the viewer window to exit.")
    mujoco.viewer.launch(workcell.model, workcell.data)


if __name__ == "__main__":
    main()
