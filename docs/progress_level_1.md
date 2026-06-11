# Progress Level 1 — Real-Time Hand Tracking and MuJoCo Teleoperation

Level 1 goal:

> Use a normal camera to track a human hand and control a simulated robot hand in MuJoCo in real time.

This level is not one task. It is a sequence of small subsystem checkpoints. Do not combine steps until each one works independently.

---

## Level 1.0 — Repo and Health Check

### Goal

Prove the repo can run Python modules cleanly.

### Files

```text
dexvision/apps/health_check.py
tests/test_imports.py
```

### Build

`health_check.py` should check imports:

```text
Python
NumPy
OpenCV
MediaPipe
MuJoCo
PyTorch, optional for later
```

### Run

```bash
python -m dexvision.apps.health_check
pytest tests/test_imports.py
```

### Pass Criteria

```text
[x] Script runs with python -m
[x] Missing dependencies print clear messages
[x] Tests pass
[x] No camera or MuJoCo window is opened yet
```

### Codex Prompt

```text
Implement only dexvision/apps/health_check.py and tests/test_imports.py.
The health check should import numpy, cv2, mediapipe, mujoco, and torch if available.
It should print OK/MISSING for each package and should not crash if an optional package is missing.
Do not add camera, MuJoCo scene loading, or hand tracking.
```

---

## Level 1.1 — Camera Smoke Test

### Goal

Open a webcam or phone-as-webcam feed with OpenCV.

### Files

```text
dexvision/camera/opencv_camera.py
dexvision/apps/check_camera.py
tests/test_camera.py
```

### Build

`OpenCVCamera` should:

```text
accept camera_id
accept width and height
open cv2.VideoCapture
read frames
return success flag and frame
release cleanly
```

`check_camera.py` should:

```text
open camera
show live feed
print/display FPS
quit on q
```

### Run

```bash
python -m dexvision.apps.check_camera --camera-id 0 --width 1280 --height 720
pytest tests/test_camera.py
```

### Pass Criteria

```text
[x] Live camera feed appears
[x] FPS is visible
[x] q quits cleanly
[x] Code handles missing camera with clear error
[x] Automated tests do not require a real webcam
```

### Manual Debug

```text
If camera feed is black, try --camera-id 1 or 2.
If phone webcam is used, confirm OS sees it as a webcam.
If FPS is poor, try --width 640 --height 480.
```

### Codex Prompt

```text
Create dexvision/camera/opencv_camera.py and dexvision/apps/check_camera.py only.
The app should open an OpenCV camera feed, display FPS, accept --camera-id, --width, and --height, and exit cleanly on q.
Add tests using a mock or synthetic frame source if needed.
Do not add MediaPipe or MuJoCo.
```

---

## Level 1.2 — Hand Landmark Tracking

### Goal

Draw hand landmarks on the camera feed.

### Files

```text
dexvision/perception/hand_tracker.py
dexvision/perception/visualization.py
dexvision/apps/check_hand_tracking.py
tests/test_hand_tracker_schema.py
```

### Build

Create a `HandTrackingResult` dataclass:

```python
@dataclass
class HandTrackingResult:
    detected: bool
    handedness: str | None
    confidence: float
    image_landmarks: np.ndarray | None  # [21, 3]
    world_landmarks: np.ndarray | None  # [21, 3]
    timestamp: float
```

`check_hand_tracking.py` should:

```text
use OpenCVCamera
run hand detector
draw 21 landmarks
draw skeleton lines
show anatomical handedness/confidence
handle no-hand frames
```

### Run

```bash
python -m dexvision.apps.check_hand_tracking --camera-id 0
pytest tests/test_hand_tracker_schema.py
```

### Pass Criteria

```text
[x] Landmarks appear on hand
[x] Anatomical handedness is displayed
[x] Confidence is displayed
[x] No crash when hand leaves frame
[x] Output schema is stable
```

### Manual Tests

```text
Open palm
Closed fist
Index point
Fast movement
Hand leaves frame
Low light
Left hand displays Left, right hand displays Right
```

### Handedness Note

MediaPipe handedness assumes selfie-mirrored input. DexVision corrects labels
for normal unmirrored OpenCV frames by default. If a camera source already
provides mirrored images, pass `--assume-mirrored-input`.

### Codex Prompt

```text
Implement dexvision/perception/hand_tracker.py, dexvision/perception/visualization.py, and dexvision/apps/check_hand_tracking.py.
Use the existing OpenCVCamera wrapper.
Return a HandTrackingResult dataclass with detected, handedness, confidence, image_landmarks, world_landmarks, and timestamp.
Handedness should be anatomical Left/Right for normal unmirrored OpenCV input.
Draw landmarks and skeleton on the frame.
Do not add MuJoCo, feature extraction, or retargeting.
```

---

## Level 1.3 — Finger Feature Extraction

### Goal

Convert raw landmarks into usable control values.

### Files

```text
dexvision/features/hand_features.py
dexvision/apps/check_hand_features.py
tests/test_hand_features.py
```

### Features

Compute:

```text
thumb_curl
index_curl
middle_curl
ring_curl
pinky_curl
pinch_thumb_index
palm_roll_proxy
palm_pitch_proxy
confidence
```

Normalize finger curl:

```text
0.0 = open/extended
1.0 = curled/closed
```

### Run

```bash
python -m dexvision.apps.check_hand_features --camera-id 0
pytest tests/test_hand_features.py
```

### Pass Criteria

```text
[x] Open hand gives low curl values
[x] Fist gives high curl values
[x] Pointing gives low index curl and high other curls
[x] Thumb-index pinch value changes visibly
[x] No NaNs when tracking is missing
```

### Visualization

`check_hand_features.py` should show:

```text
camera feed with landmarks
live bars for each finger curl
pinch value
confidence
```

### Codex Prompt

```text
Implement dexvision/features/hand_features.py and dexvision/apps/check_hand_features.py.
Use existing hand tracking output and compute normalized finger curl values and thumb-index pinch distance.
Add a simple visual overlay with live feature bars.
Add tests using synthetic landmark arrays.
Do not add MuJoCo or robot retargeting yet.
```

---

## Level 1.3B — Local Per-Finger Feature Extractor Replacement

### Goal

Replace the Level 1.3 feature extraction internals with a local per-finger
model. Each non-thumb finger should compute curl from its own MCP/PIP/DIP/TIP
joint chain instead of global fingertip-to-palm distances, so moving one finger
does not strongly change another finger's curl, extension, or up/down state.

This is a blocking revisit discovered during Level 1.10 manual teleop testing.
Do not continue to full-hand teleop until this checkpoint passes and Level 1.10
is manually re-verified.

### Files

```text
dexvision/features/hand_features.py
dexvision/apps/check_hand_features.py
tests/test_hand_features.py
```

Optional diagnostic-only files may be added if they stay camera/feature focused
and do not introduce MuJoCo or retargeting.

### Problem Observed

During Level 1.10 testing, `check_one_finger_teleop.py` showed that the robot
index finger followed the reported `index_effective` value and non-index robot
fingers stayed neutral/open. However, the raw feature itself was unstable:

```text
index extended / pointing: I ~= 0.17
index curled / fist:       I ~= 0.70-0.78
similar-looking index pose with other fingers changed: I could jump into curled range
```

That behavior indicates feature/landmark coupling in Level 1.3, not leakage in
the Level 1.10 retargeter.

### Build

Implement structured feature output:

```text
FingerState:
  curl: float 0.0-1.0
  extension: float 0.0-1.0
  abduction: float | None
  is_up: bool
  valid: bool

PalmState:
  origin
  local axes
  valid

HandFeatures:
  thumb/index/middle/ring/pinky: FingerState
  palm: PalmState
  compatibility scalar fields such as index_curl, middle_curl, etc.
```

For index/middle/ring/pinky:

```text
Use only that finger's MCP/PIP/DIP/TIP landmarks for curl.
Compute curl primarily from local joint bend angles.
Compute extension from the fingertip relative to that finger's MCP in a palm-local frame.
Determine is_up from both extension and curl.
Avoid wrist-to-tip or global fingertip-distance as the main curl score.
```

For thumb:

```text
Use separate CMC/MCP/IP/TIP logic.
Report thumb curl, extension, and an opposition/pinch-relevant signal when available.
Do not reuse the exact four-finger formula blindly.
```

Palm frame:

```text
Use wrist, index MCP, middle MCP, and pinky MCP.
Use the frame to normalize orientation/extension, not as the only curl signal.
```

Calibration:

```text
Add an optional calibration API for open-hand baseline, fist baseline, and
per-finger min/max normalization. It may be a dataclass/API placeholder with a
clear TODO if there is no live calibration flow yet.
```

Keep the public `HandFeatures` contract:

```text
0.0 = open/extended
1.0 = curled/closed
Legacy scalar fields still work for Level 1.4+ callers.
curl and extension are independent; do not force curl + extension = 1.
```

Control-signal decision:

```text
For index/middle/ring/pinky:
  bend = clamp(1.0 - extension, 0.0, 1.0)

extension high -> bend low -> robot finger open
extension low  -> bend high -> robot finger closed
```

Raw `curl` remains a diagnostic/local geometric signal. Do not use raw
long-finger `*_curl` as the Level 1.10 robot-control signal while this decision
is active. Thumb control remains conservative and still needs a separate live
decision.

Do not change robot joint limits or MuJoCo model files to mask feature issues.

Update `check_hand_features.py` so the live overlay shows per-finger curl,
per-finger extension, derived bend where available, and `is_up`, making
coupling easy to spot.

### Run

```bash
pytest tests/test_hand_features.py
pytest tests/test_smoothing.py tests/test_curl_retargeter.py tests/test_one_finger_teleop.py
python -m dexvision.apps.check_hand_features --camera-id 0
```

Then re-run the Level 1.10 manual diagnostic:

```bash
mjpython -m dexvision.apps.check_one_finger_teleop --camera-id 0 --show-camera-window --print-interval 10
```

### Pass Criteria

```text
[x] Synthetic open hand gives low curl/high extension for all fingers.
[x] Synthetic fist gives high curl for all fingers.
[x] Synthetic index-only-up, middle-only-up, ring-only-up, and pinky-only-up each affect only the intended finger.
[x] Peace sign keeps index/middle up and ring/pinky curled.
[x] Synthetic moving index does not significantly change middle/ring/pinky computed curl or extension.
[x] Pinch remains visible through thumb/index distance and thumb opposition signal.
[x] No NaNs when tracking is missing or landmarks are invalid.
[x] Existing feature-schema callers still pass automated tests.
[x] Manual review finds long-finger extension values track live poses well.
[x] Manual review accepts derived long-finger bend values as meaningful enough for robot control.
[x] Thumb live behavior is acceptable for the current conservative/non-blocking Level 1.10 path.
[x] Manual live verification passes overall.
```

### Status

Automated checks passed on June 10, 2026 with `pytest tests/test_hand_features.py`,
affected schema-callers, and the full `pytest` suite.

Manual screenshot review on June 10, 2026 found the replacement much better for
the four long fingers. The `extension` rows look close to the intended live
poses across open hand, fist, single-finger-up poses, peace sign, and partial
fist. However, raw `curl` values do not look intuitive or consistently
correlated with the live pose. That finding is resolved for Level 1.10 by
using derived long-finger bend from extension while keeping raw `*_curl` as a
diagnostic signal. Fuller thumb behavior can be revisited during full-hand
teleop tuning.

Control-signal decision used for Level 1.10:

```text
index_bend/middle_bend/ring_bend/pinky_bend = 1.0 - extension
```

The downstream pipeline now uses long-finger `*_bend` fields for robot control,
while raw `*_curl` fields remain visible for diagnostics. Thumb control stays on
the existing conservative thumb behavior until a separate thumb decision is
made.

Manual verification passed on June 11, 2026 after the user reported the live
feature viewer and one-finger teleop behavior were clean. This closes the
reopened Level 1.3B blocking checkpoint.

Affected downstream pieces updated for this decision:

```text
configs/level1_teleop.yaml
  Maps thumb_curl plus index_bend/middle_bend/ring_bend/pinky_bend.

dexvision/retargeting/curl_retargeter.py
  Interpolates robot targets from configured control fields and keeps legacy
  *_curl fields available.

dexvision/apps/check_one_finger_teleop.py
  Drives the robot index finger from smoothed index_bend and prints raw curl,
  raw extension, smoothed extension, and bend side by side.

dexvision/apps/calibrate_index_curl.py
  Kept under the legacy module name, but now reports index_curl, index_extension,
  and index_bend as an index signal inspection tool.

dexvision/features/smoothing.py
  Smooths structured extension values; bend is derived from smoothed extension.
```

### Manual Debug

Use the feature viewer and the Level 1.10 camera overlay to compare the yellow
finger chains with the per-finger curl/extension/bend/up rows.

```text
Open palm: all long fingers should have high extension and UP.
Fist: all long fingers should have low extension and not UP.
Index/middle/ring/pinky-only-up: only the raised finger should be UP.
Peace sign: index and middle should be UP; ring and pinky should stay curled.
Pinch: pinch value should drop when thumb tip approaches index tip.
Palm rotation: slight rotation should not swing unrelated finger rows heavily.
Raw curl is currently diagnostic only; do not treat a live curl mismatch as
a robot-control failure unless the derived bend signal is also wrong.
```

### Codex Prompt

```text
Redo Level 1.3 feature extraction as Level 1.3B.
Read docs/CURRENT_STATUS.md, docs/progress_level_1.md, and docs/module_contracts.md.
Replace global/scalar feature extraction with structured local per-finger
FingerState/PalmState/HandFeatures output. Preserve scalar compatibility fields.
Do not modify MuJoCo, retargeting, Level 2, or learning.
Add synthetic tests for open hand, fist, each single-finger-up pose, peace sign,
index motion locality, pinch, and invalid/no-hand frames.
Run automated tests, then stop for manual live verification.
```

---

## Level 1.4 — Feature Smoothing

### Goal

Reduce jitter before connecting features to robot joints.

### Files

```text
dexvision/features/smoothing.py
dexvision/apps/check_smoothing.py
tests/test_smoothing.py
```

### Build

Start with:

```text
Exponential moving average
```

Later optional:

```text
One Euro Filter
velocity clipping
confidence-based hold
```

### Run

```bash
python -m dexvision.apps.check_smoothing --camera-id 0
pytest tests/test_smoothing.py
```

### Pass Criteria

```text
[x] Smoothed values jitter less than raw values
[x] Movement still feels responsive
[x] Lost tracking does not produce NaNs
[x] Confidence drop can freeze or decay values safely
```

### Codex Prompt

```text
Implement dexvision/features/smoothing.py and dexvision/apps/check_smoothing.py.
Start with an exponential moving average smoother for scalar hand features.
Show raw and smoothed feature values in the app.
Add tests for step response, noisy input, and missing values.
Do not add MuJoCo.
```

---

## Level 1.5 — MuJoCo Import and Simple Scene

### Goal

Confirm MuJoCo works independently.

### Files

```text
assets/mujoco/simple_scene.xml
dexvision/sim/mujoco_env.py
dexvision/apps/check_mujoco.py
tests/test_mujoco_load.py
```

### Build

Start with a simple scene:

```text
ground plane
light
cube
camera
```

No hand model yet.

### Run

```bash
python -m dexvision.apps.check_mujoco --model assets/mujoco/simple_scene.xml
pytest tests/test_mujoco_load.py
```

### Pass Criteria

```text
[x] MuJoCo model loads
[x] Simulation steps
[x] Viewer opens if requested
[x] Headless load test passes
[x] No camera/MediaPipe involved
```

### Codex Prompt

```text
Create a minimal MuJoCo XML scene at assets/mujoco/simple_scene.xml.
Implement dexvision/sim/mujoco_env.py with a simple load/reset/step API.
Implement dexvision/apps/check_mujoco.py that loads the scene and steps it.
Add a headless test that only verifies the model loads and can step.
Do not add camera or hand tracking.
```

---

## Level 1.6 — Robot Hand Model Load

### Goal

Load a robot hand model in MuJoCo.

Note: this checkpoint originally introduced the simple hand at
`assets/mujoco/hand_scene.xml`. Level 1.8B later separates that model into
`assets/mujoco/debug_hand_scene.xml` and reserves `assets/mujoco/hand_scene.xml`
for the final hand decision.

### Files

```text
assets/mujoco/hand_scene.xml
dexvision/sim/hand_model.py
dexvision/apps/check_hand_model.py
tests/test_hand_model.py
```

### Build

The hand can be simple at first.

Minimum acceptable model:

```text
palm body
thumb with 2-3 joints
index/middle/ring/pinky with 2-3 joints each
position actuators
joint limits
```

If using an existing model, document the source in `assets/mujoco/README.md`.

### Run

```bash
python -m dexvision.apps.check_hand_model --model assets/mujoco/hand_scene.xml
pytest tests/test_hand_model.py
```

### Pass Criteria

```text
[x] Hand scene loads
[x] Joint names are discoverable
[x] Actuator names are discoverable
[x] Joint limits are available
[x] Simulation is stable at rest
```

### Codex Prompt

```text
Create or integrate a simple MuJoCo hand scene at assets/mujoco/hand_scene.xml.
Add a utility that prints joint and actuator names.
Add tests that verify the model loads and contains expected controllable joints.
Do not connect camera or MediaPipe.
```

---

## Level 1.7 — Move One Robot Joint Manually

### Goal

Control one robot joint from code.

### Files

```text
dexvision/apps/check_one_joint.py
```

### Build

Script behavior:

```text
load hand scene
find one index finger joint/actuator
drive it with a sine wave or open/close loop
render viewer
```

### Run

```bash
python -m dexvision.apps.check_one_joint --joint index_mcp
```

### Pass Criteria

```text
[x] One robot joint visibly moves
[x] Joint target stays within limits
[x] No unstable simulation
[x] Joint name and target are printed
```

### Codex Prompt

```text
Implement dexvision/apps/check_one_joint.py.
It should load the hand MuJoCo model and drive exactly one selected joint or actuator using a simple periodic command.
Print joint name, target, and current value.
Do not add camera or hand tracking.
```

---

## Level 1.8 — Move All Robot Fingers Manually

### Goal

Script robot gestures without camera input.

### Files

```text
dexvision/apps/check_hand_actuation.py
configs/hand_gestures.yaml
```

### Build

Script gestures:

```text
open hand
fist
point
pinch
peace sign
relax
```

### Run

```bash
python -m dexvision.apps.check_hand_actuation
```

### Pass Criteria

```text
[x] Robot can open/close
[x] Robot can point
[x] Robot can pinch approximately
[x] Joint limits are respected
[x] Gesture config is readable
```

### Codex Prompt

```text
Implement dexvision/apps/check_hand_actuation.py and configs/hand_gestures.yaml.
The app should play scripted robot hand gestures using joint target dictionaries.
Do not add camera, MediaPipe, or human feature extraction.
```

---

## Level 1.8B — Final Hand Model Decision and Separation

### Goal

Separate the current smoke-test hand from the hand model that will define the
real action space for recording demonstrations and training policies.

The current simple MuJoCo hand may be kept as a debug hand for loading,
actuation, and viewer smoke tests. Serious Level 2 demo recording and Level 3
learning should not start until the final hand model is selected and documented.
Changing the hand model after collecting demonstrations may invalidate recorded
datasets, action arrays, retargeting configs, benchmarks, and trained policies.

### Files

```text
docs/robot_hand_model.md
assets/mujoco/debug_hand_scene.xml
assets/mujoco/hand_scene.xml
configs/level1_teleop.yaml, when retargeting is implemented
```

### Build

Document:

```text
debug hand path
final hand path
joint names
actuator names
joint limits
neutral pose
known visual/control limitations
decision: debug-only or final
```

### Run

```bash
python -m dexvision.apps.check_hand_model --model assets/mujoco/debug_hand_scene.xml
python -m dexvision.apps.check_hand_actuation --model assets/mujoco/debug_hand_scene.xml --headless
```

When a final hand is chosen:

```bash
python -m dexvision.apps.check_hand_model --model assets/mujoco/hand_scene.xml
python -m dexvision.apps.check_hand_actuation --model assets/mujoco/hand_scene.xml
```

### Pass Criteria

```text
[x] debug hand and final hand are clearly separated
[x] final hand model choice is documented
[x] joint names, actuator names, joint limits, and neutral pose are documented
[x] scripted gestures work on whichever hand is marked final
[x] retargeting config points to the final hand
```

### Codex Prompt

```text
Document the hand-model decision before retargeting, demo recording, or learning.
The existing simple hand may become assets/mujoco/debug_hand_scene.xml.
Do not collect demos, train policies, or pretend the final hand is chosen until
docs/robot_hand_model.md documents that decision and the final hand passes the
scripted gesture check.
```

---

## Level 1.9 — Curl Retargeter

### Goal

Map configured human hand control features to robot joint targets.

### Files

```text
dexvision/retargeting/curl_retargeter.py
configs/level1_teleop.yaml
tests/test_curl_retargeter.py
```

### Build

Input:

```text
HandFeatures
```

Output:

```text
dict[joint_name, target_value]
```

Requirements:

```text
obey joint limits
handle missing features
support per-finger scaling
configurable joint mapping
```

### Run

```bash
pytest tests/test_curl_retargeter.py
```

### Pass Criteria

```text
[x] Open hand maps to open robot hand
[x] Fist maps to closed robot hand
[x] Outputs obey limits
[x] Per-finger scale config works
```

### Codex Prompt

```text
Implement dexvision/retargeting/curl_retargeter.py.
It should map configured HandFeatures control fields to robot joint target dictionaries using a YAML config.
Add tests for open hand, fist, pointing, and joint-limit clipping.
Do not add live camera or MuJoCo runtime code in this module.
```

---

## Level 1.10 — One Real Finger Controls One Robot Finger

### Goal

First full integration, but only one finger.

### Files

```text
dexvision/apps/check_one_finger_teleop.py
```

### Pipeline

```text
camera
→ hand tracking
→ feature extraction
→ smoothing
→ index_bend
→ index robot joint target
→ MuJoCo
```

### Run

```bash
mjpython -m dexvision.apps.check_one_finger_teleop --camera-id 0 --show-camera-window --print-interval 10
```

### Pass Criteria

```text
[x] Real index finger open -> robot index finger opens
[x] Real index finger curls -> robot index finger curls
[x] Other fingers stay fixed
[x] Tracking loss handled safely
[x] Jitter is acceptable
```

### Status

Manual one-finger teleop verification passed on June 11, 2026 after switching
the control signal from raw `index_curl` to derived `index_bend = 1.0 -
smoothed index extension`.

The MuJoCo index finger follows `index_bend`, non-index robot fingers remain
neutral/open, and printed diagnostics show raw index curl, raw index extension,
smoothed index extension, and index bend side by side.

### Codex Prompt

```text
Implement dexvision/apps/check_one_finger_teleop.py.
Use existing camera, hand tracker, feature extractor, smoother, MuJoCo env, and retargeter.
Only connect the human index finger to the robot index finger.
Other robot fingers should stay neutral.
Do not implement all-finger teleop yet.
```

---

## Level 1.11 — Full Hand Teleoperation

### Goal

Real human hand controls robot hand.

### Files

```text
dexvision/apps/run_level1_teleop.py
configs/level1_teleop.yaml
```

### Run

```bash
python -m dexvision.apps.run_level1_teleop --camera-id 0
```

### Pass Criteria

```text
[x] Open palm works
[x] Fist works
[x] Point works
[x] Pinch roughly works
[x] Peace sign roughly works
[x] Tracking loss safe behavior works
[x] Latency feels acceptable
```

### Status

Automated checks passed on June 11, 2026 using
`conda run -n dexvision python -m dexvision.apps.run_level1_teleop --help`,
`conda run -n dexvision pytest tests/test_run_level1_teleop.py tests/test_curl_retargeter.py tests/test_one_finger_teleop.py`,
and `conda run -n dexvision pytest` with 138 passed.

Manual full-hand teleop verification passed on June 11, 2026 after the user
reported the live Level 1.11 behavior looked good.

### Codex Prompt

```text
Implement dexvision/apps/run_level1_teleop.py.
Connect the existing camera, hand tracker, feature extractor, smoother, retargeter, and MuJoCo hand environment.
Support full-hand control-field teleoperation using the Level 1.3B long-finger bend decision.
Add CLI options for camera id, config path, and model path.
Do not add demo recording or imitation learning.
```

---

## Level 1.12 — Polished Level 1 Demo

### Goal

Make the demo presentable.

### Required Visuals

```text
webcam with landmarks
finger feature bars
tracking confidence
FPS
MuJoCo hand viewer
status messages for tracking loss
```

### Optional Visuals

```text
raw vs smoothed values
robot target vs actual joint values
gesture label
short recorded demo video
```

### Run

```bash
mjpython -m dexvision.apps.run_level1_teleop --camera-id 0 --show-camera-window --print-interval 10
```

### Automated Checks

```bash
conda run -n dexvision python -m dexvision.apps.run_level1_teleop --help
conda run -n dexvision pytest tests/test_run_level1_teleop.py tests/test_level1_demo_docs.py tests/test_curl_retargeter.py tests/test_one_finger_teleop.py
conda run -n dexvision pytest
```

### Pass Criteria

```text
[x] Demo can be shown to someone else
[x] README has Level 1 instructions
[x] Known limitations documented
[x] Short video/GIF captured
```

### Status

Implementation and automated checks passed on June 11, 2026. Manual live demo
verification passed after the user confirmed the polished demo looked good.

### Codex Prompt

```text
Polish the Level 1 demo by improving overlays, CLI flags, logging, and README instructions.
Do not change core algorithms unless needed.
Do not add Level 2 recording yet.
```

---

## Level 1.13 — Hand Base / Wrist Pose Control

### Goal

Use the tracked human palm/wrist pose to control the simulated Shadow Hand base
pose in MuJoCo while preserving the existing finger teleoperation path.

Fixed-base finger teleop is not sufficient for meaningful object manipulation:
the hand can articulate its fingers, but it cannot reach, approach, translate
around, or rotate around objects. Serious Level 2 manipulation demonstrations
should not start until Level 1.13 is complete or explicitly deferred.

### Files

```text
dexvision/features/hand_base.py
dexvision/sim/hand_base_control.py
dexvision/sim/mujoco_env.py
dexvision/apps/run_level1_teleop.py
dexvision/apps/check_hand_base_control.py
configs/level1_teleop.yaml
assets/mujoco/hand_scene.xml
assets/mujoco/menagerie/shadow_hand/right_hand_dexvision.xml
tests/test_hand_base_control.py
```

### Sub-Checkpoints

Level 1.13 is staged so base pose control can become real hand mirroring over
time without turning into one oversized checkpoint.

#### Level 1.13A — Base x/y Translation

```text
[x] Use normalized image-space palm-center coordinates from wrist, index MCP, middle MCP, and pinky MCP.
[x] Press c in the camera overlay to capture neutral_palm_center and neutral robot base pose.
[x] Map image x to robot lateral motion and image y to robot height by default.
[x] Keep the base workspace clamped so the hand stays above the floor.
[x] Leave finger teleoperation unchanged.
[x] Manually verified by the user on June 11, 2026.
```

#### Level 1.13B — Depth / In-Out Base Control

```text
[x] Use monocular image-space hand scale as a depth proxy.
[x] Press c to capture neutral_palm_center, neutral_hand_scale, and neutral robot base pose.
[x] Keep existing x/y translation behavior unchanged.
[x] Map larger hand scale to closer-to-camera depth motion along the configured depth axis.
[x] Map smaller hand scale to the opposite depth direction.
[x] Clamp depth to configured safe limits and the base workspace.
[x] Smooth depth and apply a deadband near neutral to avoid jitter.
[x] On tracking loss, hold the last valid depth target by default or decay depth to neutral when configured.
[x] Show overlay/debug values for calibration, hand scale, neutral scale, depth delta, depth target, clamp/limit state, x/y/base target, and fingers.
[x] Start the simulated Shadow Hand upright with palm facing forward toward the camera/approach axis.
[x] Existing finger teleoperation still works.
[x] No palm/wrist rotation is applied for this checkpoint.
[x] Manually verified by the user on June 11, 2026.
```

#### Level 1.13C — Relative Palm Rotation

```text
[x] Estimate palm orientation from wrist/index MCP/middle MCP/pinky MCP.
[x] Store calibrated human palm rotation and calibrated robot base rotation when c is pressed.
[x] Compute target_robot_rotation from calibrated relative human palm rotation.
[x] Apply orientation only when explicitly enabled.
[x] Add roll/pitch/yaw DOF staging so roll-only can be tested first.
[x] Keep camera-to-MuJoCo axis signs/remap configurable for tuning.
[x] Smooth, clamp, deadband, and rate-limit orientation so palm rotations do not spin wildly.
[x] Manually verify roll-only first, then pitch/yaw if roll behavior is stable.
```

#### Level 1.13D — Final Pose-Control Polish/Safety

```text
[x] Increase the default roll/pitch/yaw envelope beyond the initial conservative 45 degree cap.
[x] Add CLI overrides for roll/pitch/yaw clamps, orientation smoothing, deadband, and rate limit.
[x] Re-check translation, depth, rotation, fingers, tracking loss, reacquire, and workspace limits together.
[x] Confirm the Shadow Hand can approach/retract and rotate without floor collisions or sudden jumps.
[x] Document final manual verification command, tuning notes, and known caveats.
[x] Decide whether any remaining 1.13 behavior is good enough for Level 2 or explicitly deferred.
```

Serious Level 2 manipulation demos, recording, replay, and learning should wait
until Level 1.13A/B/C are complete, or until the user explicitly defers the
remaining 1.13 sub-checkpoints.

### Config Options

```text
base_control.enable_base_control
base_control.base_control_mode: image_2d
base_control.base_fixed_z
base_control.base_position_scale_x
base_control.base_position_scale_y
base_control.base_image_y_axis: height or approach
base_control.enable_depth_control
base_control.depth_source: palm_width or robust_palm_scale
base_control.depth_axis: x/y/z, or named axis approach/lateral/height
base_control.depth_scale
base_control.depth_sign
base_control.depth_min
base_control.depth_max
base_control.depth_smoothing_alpha
base_control.depth_deadband
base_control.depth_hold_on_tracking_loss
base_control.base_workspace_min
base_control.base_workspace_max
base_control.base_smoothing_alpha
base_control.enable_base_orientation
base_control.orientation_mode: relative_palm
base_control.orientation_dofs: roll, pitch, yaw subset
base_control.orientation_axis_signs
base_control.orientation_remap_matrix
base_control.max_roll_deg
base_control.max_pitch_deg
base_control.max_yaw_deg
base_control.orientation_smoothing_alpha
base_control.orientation_deadband_deg
CLI overrides: --max-roll-deg, --max-pitch-deg, --max-yaw-deg
CLI overrides: --orientation-smoothing-alpha, --orientation-deadband-deg, --max-rotation-step-deg
base_control.base_orientation_axis_signs
base_control.base_orientation_remap_matrix
base_control.position_source
base_control.position_mode
base_control.position_scale
base_control.position_offset
base_control.rotation_mode
base_control.rotation_offset_quat
base_control.workspace_limits
base_control.max_position_step
base_control.max_rotation_step_degrees
base_control.reset_base_pose: tracking_reacquire
```

### Run

Automated:

```bash
conda run -n dexvision python -m dexvision.apps.run_level1_teleop --help
conda run -n dexvision python -m dexvision.apps.check_hand_base_control --help
conda run -n dexvision pytest tests/test_hand_base_control.py tests/test_run_level1_teleop.py
```

Manual:

```bash
mjpython -m dexvision.apps.check_hand_base_control --camera-id 0 --print-interval 10
```

Final full teleop with base/depth/orientation/finger control enabled:

```bash
mjpython -m dexvision.apps.run_level1_teleop --camera-id 0 --show-camera-window --enable-base-control --enable-depth-control --enable-base-orientation --orientation-dofs roll,pitch,yaw --print-interval 10
```

Finger-only fixed-base teleop must still work with base control disabled:

```bash
mjpython -m dexvision.apps.run_level1_teleop --camera-id 0 --show-camera-window --print-interval 10
```

### Pass Criteria

```text
[x] Synthetic palm pose estimator returns valid normalized axes/quaternion.
[x] Synthetic image palm-center estimator returns the normalized palm mean.
[x] Synthetic hand-scale estimator returns palm_width and robust_palm_scale.
[x] Invalid landmarks fail safely with finite neutral output.
[x] Palm-center delta calculation is correct after calibration.
[x] Pressing c stores neutral hand scale for depth calibration.
[x] Larger hand scale produces depth movement in the configured direction.
[x] Smaller hand scale produces depth movement in the opposite direction.
[x] Depth deadband suppresses small hand-scale noise near neutral.
[x] Depth clamp keeps targets inside configured depth and workspace limits.
[x] Workspace clamp keeps targets inside configured bounds.
[x] Default orientation control is disabled.
[x] Orientation is not required for Level 1.13B and is not enabled by default.
[x] Config defaults preserve old fixed-base behavior.
[x] Pressing c with open palm facing camera makes that pose neutral.
[x] Moving the real hand left/right moves the simulated hand base left/right clearly.
[x] Moving the real hand up/down moves the simulated hand base up/down clearly.
[x] Moving the real hand toward the camera moves the simulated hand in/out along the configured depth axis.
[x] Moving the real hand away from the camera moves the simulated hand in the opposite depth direction.
[x] At startup/reset, the simulated hand is upright with the palm facing forward like the user's calibration pose.
[x] The hand stays above the floor.
[x] Base and depth motion are smoothed and bounded.
[x] Tracking loss holds the last valid base target or safely returns toward neutral without a jump.
[x] Existing finger teleop still works with base control disabled.
[x] Existing finger teleop still works with base/depth control enabled.
[x] No rotation is applied for Level 1.13B.
[x] Palm frame axes are normalized and orthogonal for synthetic landmarks.
[x] Palm rotation quaternion is valid.
[x] Pressing c stores neutral human palm rotation and neutral robot base rotation.
[x] Relative palm rotation is identity immediately after calibration.
[x] Synthetic palm roll produces the expected roll delta.
[x] Orientation DOFs can stage roll-only before pitch/yaw.
[x] Max roll/pitch/yaw clamps bound relative rotation.
[x] Orientation deadband suppresses tiny palm-rotation jitter.
[x] Tracking loss holds the last valid orientation target.
[x] Orientation disabled preserves translation/depth-only behavior.
[x] Roll-only manual verification passes with no wild spinning.
[x] Pitch/yaw manual verification passes with consistent bounded motion.
[x] Wider default orientation limits allow table-pickup style wrist poses beyond the initial 45 degree caps.
[x] Orientation tuning CLI flags are visible in Level 1.13 help.
[x] Wider 1.13D orientation limits are manually verified with translation/depth/fingers together.
```

### Status

Initial implementation and automated checks passed on June 11, 2026 using
`conda run -n dexvision python -m dexvision.apps.run_level1_teleop --help`,
`conda run -n dexvision python -m dexvision.apps.check_hand_base_control --help`,
`conda run -n dexvision pytest tests/test_hand_base_control.py tests/test_run_level1_teleop.py`,
MuJoCo-adjacent hand model/actuation checks, and `conda run -n dexvision pytest`
with 149 passed.

Manual testing on June 11, 2026 found two issues: after a while the live app
stopped on a high-qvel instability guard during hand-base control, and the
visual palm/robot orientation felt under-rotated for upright hand poses. A
stabilization patch now rate-limits mocap translation/rotation, lowers base
smoothing alpha, and switches the enabled base-control default rotation mode to
`palm_absolute` with a configurable `rotation_offset_quat`.

Follow-up manual testing on June 11, 2026 found that the behavior looked better
with the user's right hand than left hand, and that wrist rotation/fist poses
could still push the base down/sideways into the workspace clamp. A follow-up
patch now uses the wrist as the default base translation anchor, keeps
`base_control.position_source: palm_center` available for explicit experiments,
mirrors anatomical left-hand orientation vectors into the right Shadow Hand
target convention, and uses a higher qvel guard only while base control is
enabled so bounded mocap-weld free-joint transients do not stop the app. Updated
automated checks passed on June 11, 2026 using
`conda run -n dexvision pytest tests/test_hand_base_control.py tests/test_run_level1_teleop.py`
with 31 passed, a headless 720-frame MuJoCo base-control stress check
(`max_abs_qpos=1.071107`, `max_abs_qvel=60.115224`), and
`conda run -n dexvision pytest` with 160 passed.

Additional manual testing on June 11, 2026 still showed a bounded qvel guard
stop (`max_abs_qpos=0.897984`, `max_abs_qvel=152.293542`) when the left hand
entered the frame, and the user suspected stale relative mapping after tracking
loss/re-entry. A follow-up patch changes the default to
`base_control.position_mode: absolute`, resets base smoothing/source neutral on
tracking loss and large reacquire jumps, keeps relative first-valid-frame
mapping available only through `base_control.position_mode: relative`, and
raises the base-control-only qvel guard further while preserving the stricter
finger-only teleop guard.
Updated automated checks passed on June 11, 2026 using
`conda run -n dexvision pytest tests/test_hand_base_control.py tests/test_run_level1_teleop.py`
with 34 passed, a headless 900-frame tracking-loss/reacquire MuJoCo stress
check (`max_abs_qpos=1.083558`, `max_abs_qvel=58.271888`), both Level 1.13 CLI
help commands, and `conda run -n dexvision pytest` with 163 passed.

Additional manual testing on June 11, 2026 showed the mapping was still not
usable enough: base x was nearly always zero, base y only changed slightly,
base z clamped near the floor, movement did not intuitively follow the real
hand, and orientation remained premature/noisy. The checkpoint is therefore
being simplified into staged verification. The Level 1.13 path now uses
calibrated `image_2d` relative pose mapping: translation is driven from
palm-center deltas and optional orientation is driven from palm-orientation
deltas relative to the pose captured by `c`.
Updated automated checks for the simplified path passed on June 11, 2026 using
`conda run -n dexvision pytest tests/test_hand_base_control.py tests/test_run_level1_teleop.py`
with 46 passed, both Level 1.13 CLI help commands, and
`conda run -n dexvision pytest` with 175 passed.

The latest orientation update keeps orientation disabled by default but makes
`--enable-base-orientation` use calibrated relative palm orientation. Pressing
`c` now stores both the current human palm orientation and the neutral robot
base orientation, so the user does not need to hold their hand in the same
orientation as the robot model during calibration. Default image up/down now
maps to robot height. Depth/in-out control is handled separately by
Level 1.13B using monocular hand scale.
Updated automated checks passed on June 11, 2026 using
`conda run -n dexvision pytest tests/test_hand_base_control.py tests/test_run_level1_teleop.py`
with 53 passed, both Level 1.13 CLI help commands, and
`conda run -n dexvision pytest` with 182 passed.

Level 1.13A base x/y translation was manually verified by the user on
June 11, 2026: left/right and up/down hand motion moved the simulated Shadow
Hand base as intended, and finger teleop still worked.

Level 1.13B depth/in-out control manually passed on June 11, 2026 after the
user confirmed the depth behavior worked and the upright palm-forward neutral
start pose looked good. The implementation uses monocular image-space hand
scale as the depth proxy and keeps palm/wrist rotation out of scope. Updated
automated checks passed using `conda run -n dexvision pytest` with 193 passed.

Level 1.13C relative palm/wrist rotation implementation is automated-test
ready and has passed manual camera plus MuJoCo verification. It keeps
orientation disabled unless `--enable-base-orientation` is passed, calibrates
the current human palm orientation to the robot neutral/base orientation when
`c` is pressed, supports `--orientation-dofs roll` for the first roll-only
check, and keeps roll/pitch/yaw signs, remap, clamps, smoothing, and deadband
configurable. Automated checks passed on June 11, 2026 using
`conda run -n dexvision python -m dexvision.apps.run_level1_teleop --help`,
`conda run -n dexvision python -m dexvision.apps.check_hand_base_control --help`,
`conda run -n dexvision pytest tests/test_hand_base_control.py tests/test_run_level1_teleop.py`
with 73 passed, and `conda run -n dexvision pytest` with 202 passed. Ruff was
not available in the `dexvision` environment.
Manual verification passed after the user confirmed roll, pitch, yaw, and all
three axes together worked in the live loop, then pushed commit
`5fe920d initial roll/pitch/yaw`.

Level 1.13D final pose-control polish/safety manually passed after the user
confirmed the widened roll/pitch/yaw range is enough for the current scope. The
1.13D improvement widens the default orientation envelope from the initial
conservative 45 degree caps to roll/yaw 120 degrees and pitch 110 degrees,
increases the base orientation rate limit to 8 degrees per frame, lowers the
orientation deadband to 0.5 degrees, and exposes live CLI overrides for
table-pickup style tuning. Final Level 1 full teleop with base/depth/
orientation/finger control was manually smoke-tested by the user for the
current scope with no reported crashes. Level 1 is complete.

Final Level 1 capabilities:

```text
Camera-based hand landmark tracking with anatomical handedness labels.
Per-finger bend controls for Shadow Hand finger teleoperation.
Calibrated image-space base x/y translation.
Monocular hand-scale depth/in-out control.
Calibrated relative palm/wrist roll, pitch, and yaw base orientation control.
Live overlay/debug values for tracking, calibration, base targets, depth, orientation, and finger bends.
Configurable smoothing, deadbands, clamps, workspace limits, orientation DOFs, signs, remap, and CLI tuning overrides.
```

Known Level 1 limitations:

```text
Depth is monocular hand-scale based, not metric 3D reconstruction.
Orientation mapping is calibrated and tunable, but still camera/viewpoint dependent.
The thumb mapping remains conservative compared with the long-finger bend controls.
No demonstration recording, replay, learning, two-hand control, object task logic, or Level 2 data schema has been implemented yet.
```

### Manual Debug

```text
Press c after placing the real hand in a comfortable centered pose.
Press r to reset the simulated base to neutral and clear the calibration.
Default base translation uses image-space palm-center deltas, not raw MediaPipe world coordinates.
Default depth uses robust_palm_scale and maps approach/retract to robot base x.
Move your hand closer to the camera to increase hand scale; move it away to decrease hand scale.
Default image up/down maps to robot height while depth maps to the configured depth axis.
If left/right feels backwards, change the sign of base_control.base_position_scale_x.
If up/down feels backwards, change the sign of base_control.base_position_scale_y.
If in/out feels backwards, change base_control.depth_sign.
If motion feels too large or too small, tune base_control.base_position_scale_x/y.
If depth motion feels too large or too small, tune base_control.depth_scale.
If depth jitters near neutral, increase base_control.depth_deadband or lower base_control.depth_smoothing_alpha.
If the hand reaches outside the useful scene area, tune base_control.base_workspace_min/max.
If motion feels jerky, lower base_control.base_smoothing_alpha or base_control.max_position_step.
Use --enable-base-orientation to test relative palm orientation matching.
Use --orientation-dofs roll to test roll-only first.
Use --orientation-dofs roll,pitch,yaw to test all orientation axes after roll is stable.
If orientation axes feel backwards, tune base_control.orientation_axis_signs.
If orientation axes are cross-wired, tune base_control.orientation_remap_matrix.
If orientation moves too far, lower base_control.max_roll_deg/max_pitch_deg/max_yaw_deg.
If orientation jitters, increase base_control.orientation_deadband_deg or lower base_control.orientation_smoothing_alpha.
For wider table-pickup style testing, tune live with --max-roll-deg, --max-pitch-deg, --max-yaw-deg, --orientation-smoothing-alpha, --orientation-deadband-deg, and --max-rotation-step-deg.
```

### Codex Prompt

```text
Implement Level 1.13D final pose-control polish/safety only.
Keep the completed Level 1.13A x/y translation and Level 1.13B depth/in-out
control and completed Level 1.13C relative palm rotation working. Improve the
combined base pose-control behavior in small, testable steps: orientation range,
workspace safety, translation/depth/orientation interaction, tracking loss,
reacquire behavior, and manual verification notes.
Do not add recording, replay, learning, two-hand control, or Level 2 features.
Add automated tests for each small safety/tuning change and stop for manual
verification before marking Level 1.13D complete.
Do not implement Level 2 demo recording, replay, or learning.
Stop after automated checks and wait for manual verification.
```

---

# Level 1 Completion Checklist

```text
[x] Health check works
[x] Camera feed works
[x] Hand landmarks work
[x] Finger features work
[x] Smoothing works
[x] Simple MuJoCo scene works
[x] Hand model loads
[x] One robot joint moves manually
[x] All robot fingers move manually
[x] Curl retargeter passes tests
[x] One real finger controls one robot finger
[x] Full hand teleop works
[x] Demo overlay is presentable
[x] README has Level 1 usage
[x] Hand base x/y translation works
[x] Hand base depth/in-out control works
[x] Hand base relative palm rotation works
[x] Hand base/wrist pose control final polish works
```
