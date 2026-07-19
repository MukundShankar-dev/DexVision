# DexVision / Hand2Bot

DexVision is a staged robotics and computer-vision project for controlling a
simulated dexterous robot hand from live hand-pose tracking. The current Level 2
work turns the completed Level 1 OpenCV, MediaPipe, and MuJoCo teleoperation
pipeline into reproducible demonstration datasets.

## Clean setup

Install Git and a Conda distribution such as Miniconda or Miniforge, clone this
repository, and run the following commands from the repository root:

```bash
conda env create -f environment.yml
conda activate dexvision
python -m dexvision.apps.health_check
```

The environment specification installs Python 3.11, the runtime dependencies,
pytest, and Ruff. To update an existing environment after the specification
changes:

```bash
conda env update --name dexvision --file environment.yml --prune
conda activate dexvision
```

PyTorch is reserved for Level 3 learning work and is an optional dependency:

```bash
python -m pip install -e ".[learning]"
```

The live demo uses these repository assets:

```text
assets/models/hand_landmarker.task
assets/mujoco/hand_scene.xml
configs/level1_teleop.yaml
```

## Level 1 Demo

### macOS

Run viewer-based applications from Terminal or iTerm with `mjpython`, which is
installed with MuJoCo:

```bash
mjpython -m dexvision.apps.run_level1_teleop --camera-id 0 --show-camera-window --print-interval 10
```

macOS may ask for camera permission the first time the application runs. The
automated checks do not open a camera or GUI.

### Windows

Run the same module from Anaconda Prompt or PowerShell after activating the
environment:

```powershell
python -m dexvision.apps.run_level1_teleop --camera-id 0 --show-camera-window --print-interval 10
```

Allow camera access in Windows privacy settings if the live application cannot
open the selected camera.

The demo opens the MuJoCo hand viewer and a camera overlay with landmarks,
finger-control bars, tracking confidence, FPS, and tracking-loss status. Press
`q` in the camera overlay, close the viewer, or press `Ctrl-C` to stop.

For a short demo video or GIF, record the camera overlay and MuJoCo viewer with
the operating system screen recorder while the demo is running.

## Development checks

Always run checks in the `dexvision` environment:

```bash
conda activate dexvision
ruff check dexvision tests
pytest
python -m dexvision.apps.health_check
```

Automated tests use synthetic camera data and do not require a webcam, GPU, or
visible MuJoCo GUI.

## Demonstration data

Operator-recorded demos under `data/demos/` are local data and are intentionally
ignored by Git. Creating or updating the environment does not remove them. See
the [Level 2 dataset runbook](docs/level2_dataset_runbook.md) for the collection
layout and commands.

Project staging and checkpoint status are documented in
[CURRENT_STATUS](docs/CURRENT_STATUS.md) and the
[Level 2 progress file](docs/progress_level_2.md).

### Current Level 2 status

Level 2.7F is complete. The `reach_touch_target` dataset contains 55 clean
successful demonstrations with balanced left/center/right coverage
(`18/18/19`) and is marked Level 3-ready. The five-demo `button_press` pilot
covers all three configured buttons; every retained episode validates,
replays, recomputes as successful, and passes quality filtering.

The deterministic `push_cube_to_target` task schema, MuJoCo fixture, reset,
object/target state extraction, and saved-state success metric are implemented.
The next checkpoint is Level 2.7G — Push-Cube Pilot. Cube recording/replay
restoration, dataset relabeling/filtering/summary integration, pilot
demonstrations, dataset scale-up, retargeter baselines, and the final Level 2
benchmark/results work are not yet complete.

## Known Limitations

This is a simulated teleoperation and dataset pipeline, not a real-robot
controller. The thumb mapping is intentionally conservative, and pinch and
peace-sign poses remain approximate. Tracking quality depends on lighting,
camera placement, and whether the input is mirrored. Use
`--assume-mirrored-input` only for selfie-mirrored camera feeds.
