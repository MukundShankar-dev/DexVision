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

Level 2.10B is complete. The three manipulation datasets currently marked Level
3-ready are:

- `reach_touch_target`: 55 clean successes with balanced target coverage
  (`18/18/19`)
- `button_press`: 55 clean successes across nine configured button/depth goals
- `push_cube_to_target`: 101 clean successes across three lane-aligned goals
  (`33/34/34`)

Every retained push-cube episode validates, completes semantic headless replay,
recomputes as successful, and passes quality filtering. Its versioned split
also reserves three interpolated cube start/target-pose states for Level 3
evaluation.

Policy-free Level 2 skill metadata can be exported from each implemented task
spec and the matching dataset-summary group:

```bash
python -m dexvision.apps.export_skill_metadata --task reach_touch_target
python -m dexvision.apps.export_skill_metadata --task button_press
python -m dexvision.apps.export_skill_metadata --task push_cube_to_target
```

These stubs intentionally leave `policy_checkpoint` unset for Level 3.

Level 2.8 adds a second, approximate retargeting baseline alongside the existing
curl retargeter. It converts MediaPipe-compatible hand landmarks into five
palm-local fingertip targets normalized by palm width, maps fingertip extension
to bounded Shadow Hand controls, and safely falls back when target extraction
or solving fails. Run its synthetic checks with:

```bash
pytest tests/test_fingertip_retargeter.py
```

Level 2.9 adds an optimization retargeter that minimizes palm-local fingertip
error, configured actuator-limit violations, and temporal changes in finger
controls. It uses bounded SciPy optimization when available, retains a
deterministic projected-gradient path when SciPy is absent, clips outputs, and
falls back to last-valid or safe-open targets. Run its synthetic checks with:

```bash
pytest tests/test_optimization_retargeter.py
```

Level 2.10B validates all three retargeters on all 101 saved
`push_cube_to_target` episodes (7,176 frames). The shared human feature and
landmark streams feed each retargeter, and counterfactual headless MuJoCo replay
combines each method's finger targets with the recorded base actions. Values in
brackets are deterministic 95% episode-bootstrap intervals from 2,000 resamples
with seed 0:

| Retargeter | Latency ms [95% CI] | Jerk [95% CI] | Surrogate tip error [95% CI] | Success [95% CI] | Tip-cube distance m [95% CI] | Tip-contact rate [95% CI] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Curl | 0.0766 [0.0763, 0.0768] | 0.01547 [0.01465, 0.01634] | 0.4854 [0.4767, 0.4938] | 0.871 [0.802, 0.931] | 0.10573 [0.10541, 0.10604] | 0.00867 [0.00592, 0.01168] |
| Fingertip | 0.1014 [0.0998, 0.1041] | 0.01872 [0.01736, 0.02022] | 0.4546 [0.4454, 0.4637] | 0.832 [0.762, 0.901] | 0.10632 [0.10604, 0.10658] | 0.00574 [0.00368, 0.00815] |
| Optimization | 1.3040 [1.2661, 1.3696] | 0.01798 [0.01668, 0.01941] | 0.4546 [0.4454, 0.4637] | 0.871 [0.802, 0.931] | 0.10585 [0.10552, 0.10616] | 0.00988 [0.00715, 0.01286] |

All three methods had zero joint-limit violations. Latency varies by machine;
action jerk is normalized actuator units per frame cubed, surrogate fingertip
error is normalized by palm width, and tip-cube measurements use the five
distal collision geoms in MuJoCo. Generate the versioned JSON, flattened CSV,
and six-panel SVG artifacts with:

```bash
python -m dexvision.apps.benchmark_retargeters --task push_cube_to_target --episodes 101 --bootstrap-samples 2000
```

The original demonstrations and base trajectories were collected with curl
retargeting. Reusing those base trajectories makes this a controlled
counterfactual finger-retargeting comparison, but it does not remove that curl
bias. No independent live fingertip/optimization trajectories were collected
or claimed; doing so requires a new operator collection campaign.

No policy training or Level 5 orchestration has been implemented.

## Known Limitations

This is a simulated teleoperation and dataset pipeline, not a real-robot
controller. The thumb mapping is intentionally conservative, and pinch and
peace-sign poses remain approximate. Tracking quality depends on lighting,
camera placement, and whether the input is mirrored. Use
`--assume-mirrored-input` only for selfie-mirrored camera feeds.
The Level 2.8 fingertip baseline is a geometric approximation rather than a
numerical robot-model IK solve. The Level 2.10 fingertip-error metric uses the
same palm-local surrogate for a consistent comparison and is not a measured
physical robot fingertip distance.
