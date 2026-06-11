# DexVision / Hand2Bot

DexVision is a staged robotics and computer-vision project for controlling a
simulated dexterous robot hand from live hand-pose tracking.

Level 1 focuses on real-time camera-to-MuJoCo teleoperation using OpenCV,
MediaPipe hand landmarks, local hand features, smoothing, a curl/bend
retargeter, and the vendored Shadow Hand MuJoCo model.

## Setup

Use the project Conda environment before running the apps or tests:

```bash
conda activate dexvision
```

The Level 1 demo expects these local assets to exist:

```text
assets/models/hand_landmarker.task
assets/mujoco/hand_scene.xml
configs/level1_teleop.yaml
```

## Level 1 Demo

On macOS, run the polished demo from a regular Terminal or iTerm session with
`mjpython` so the MuJoCo viewer can open:

```bash
mjpython -m dexvision.apps.run_level1_teleop --camera-id 0 --show-camera-window --print-interval 10
```

On Windows, run the same module with Python:

```bash
python -m dexvision.apps.run_level1_teleop --camera-id 0 --show-camera-window --print-interval 10
```

The demo opens the MuJoCo hand viewer and, when `--show-camera-window` is set, a
camera overlay window with landmarks, finger-control bars, tracking confidence,
FPS, and tracking-loss status. Press `q` in the camera overlay, close the MuJoCo
viewer, or press `Ctrl-C` in the terminal to stop.

Useful options:

```bash
python -m dexvision.apps.run_level1_teleop --help
python -m dexvision.apps.run_level1_teleop --camera-id 1 --width 640 --height 480 --show-camera-window
python -m dexvision.apps.run_level1_teleop --camera-id 0 --assume-mirrored-input --show-camera-window
```

For a short demo video or GIF, record both the camera overlay and MuJoCo viewer
with the operating system screen recorder while running the demo command above.
On macOS, `Shift-Command-5` can record a selected portion of the screen.

## Automated Checks

Run the focused Level 1 demo checks:

```bash
python -m dexvision.apps.run_level1_teleop --help
pytest tests/test_run_level1_teleop.py tests/test_curl_retargeter.py tests/test_one_finger_teleop.py
```

Run the full automated suite:

```bash
pytest
```

Automated tests use synthetic camera data and do not require a webcam, GPU, or
visible MuJoCo GUI.

## Known Limitations

This is a Level 1 teleoperation demo, not a real-robot controller.

The thumb mapping is intentionally conservative and less expressive than the
long-finger bend controls.

Pinch and peace-sign poses are approximate because the current retargeter maps
simple hand features to Shadow Hand actuator targets.

Tracking quality depends on lighting, camera placement, and whether the input is
mirrored. Use `--assume-mirrored-input` only for selfie-mirrored camera feeds.

When tracking is lost or confidence is low, controls decay or hold according to
the smoothing configuration instead of trying to infer unseen fingers.
