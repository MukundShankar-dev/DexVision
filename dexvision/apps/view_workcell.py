"""Open the workcell in the MuJoCo viewer, stepping it through the normal action path.

    mjpython -m dexvision.apps.view_workcell --seed 0    # macOS
    python -m dexvision.apps.view_workcell --seed 0      # Windows / Linux
"""

from __future__ import annotations

import argparse
import time

import mujoco.viewer
import numpy as np

from dexvision.sim.workcell import (
    ACTION_DIM,
    ACTUATOR_TARGETS,
    OBJECT_IDS,
    Placement,
    Workcell,
)

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
    hold = np.zeros(ACTION_DIM)
    hold[ACTUATOR_TARGETS] = workcell.neutral_actuator_targets
    print(f"Workcell reset with seed {args.seed}. The hand holds still; close the window to exit.")
    # The passive viewer is the only one mjpython supports on macOS. We step the scene ourselves,
    # through Workcell.step, so what you see is exactly what policies experience.
    with mujoco.viewer.launch_passive(workcell.model, workcell.data) as viewer:
        while viewer.is_running():
            started = time.perf_counter()
            workcell.step(hold)
            viewer.sync()
            time.sleep(max(0.0, workcell.control_dt - (time.perf_counter() - started)))


if __name__ == "__main__":
    main()
