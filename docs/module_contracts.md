# Module Contracts

This file defines expected inputs/outputs between modules. Keep it updated as the repo evolves.

---

## Camera

Module:

```text
dexvision/camera/opencv_camera.py
```

Contract:

```python
success: bool
frame: np.ndarray  # BGR, shape [H, W, 3], dtype uint8
timestamp: float
```

Rules:

```text
No MediaPipe in camera module.
No MuJoCo in camera module.
No global camera initialization on import.
```

---

## Hand Tracking

Module:

```text
dexvision/perception/hand_tracker.py
```

Contract:

```python
@dataclass
class HandTrackingResult:
    detected: bool
    handedness: str | None  # Anatomical "Left"/"Right" by default.
    confidence: float
    image_landmarks: np.ndarray | None  # [21, 3]
    world_landmarks: np.ndarray | None  # [21, 3]
    timestamp: float
```

Rules:

```text
No robot joint logic here.
No MuJoCo here.
Must handle no-hand frames.
Default handedness is corrected for normal unmirrored OpenCV input.
Use assume_mirrored_input=True only for selfie-mirrored camera images.
```

---

## Hand Features

Module:

```text
dexvision/features/hand_features.py
```

Contract:

```python
@dataclass(frozen=True)
class FingerState:
    curl: float
    extension: float
    abduction: float | None
    is_up: bool
    valid: bool

@dataclass(frozen=True)
class PalmState:
    origin: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    z_axis: np.ndarray
    valid: bool

@dataclass(frozen=True)
class HandFeatures:
    thumb: FingerState
    index: FingerState
    middle: FingerState
    ring: FingerState
    pinky: FingerState
    palm: PalmState
    pinch_thumb_index: float
    confidence: float

    # Compatibility properties for Level 1.4+ callers:
    thumb_curl: float
    index_curl: float
    middle_curl: float
    ring_curl: float
    pinky_curl: float
    index_bend: float
    middle_bend: float
    ring_bend: float
    pinky_bend: float
    palm_roll: float
    palm_pitch: float
    palm_yaw: float
```

Rules:

```text
0.0 curl means open.
1.0 curl means closed.
0.0 extension means folded toward the palm.
1.0 extension means extended away from that finger's MCP.
curl and extension are independent measurements; they are not expected to sum to 1.0.
curl is a diagnostic/local geometric bend signal.
extension is currently the preferred long-finger robot-control signal.
For index/middle/ring/pinky, bend = clamp(1.0 - extension, 0.0, 1.0).
0.0 bend means robot finger open/extended; 1.0 bend means robot finger closed/folded.
Index/middle/ring/pinky curl is computed from only that finger's MCP/PIP/DIP/TIP local joint angles.
Index/middle/ring/pinky extension is measured from fingertip motion relative to that finger's MCP in a palm-local frame.
Thumb curl uses CMC/MCP/IP/TIP with separate thumb logic; thumb abduction may carry an opposition/pinch-relevant signal.
Thumb bend is not part of the current contract; thumb robot control still needs a separate decision.
PalmState is built from wrist, index MCP, middle MCP, and pinky MCP.
Legacy scalar fields such as index_curl must keep working and must mirror the corresponding FingerState.curl.
Long-finger bend fields are derived compatibility/control properties and must mirror 1.0 - the corresponding FingerState.extension.
Live Level 1.3B review found long-finger extension much more intuitive/reliable than raw curl.
No MuJoCo-specific joint names here.
```

---

## Hand Base Target

Module:

```text
dexvision/features/hand_base.py
```

Contract:

```python
@dataclass(frozen=True)
class HandBaseTarget:
    position: np.ndarray  # shape [3]
    orientation_quat: np.ndarray  # MuJoCo [w, x, y, z], shape [4]
    confidence: float
    valid: bool

@dataclass(frozen=True)
class ImagePalmCenterTarget:
    palm_center: np.ndarray  # normalized image coordinates, shape [2]
    hand_scale: float  # normalized image-space scale, e.g. palm width
    confidence: float
    valid: bool
```

Rules:

```text
HandBaseTarget is built from wrist, index MCP, middle MCP, and pinky MCP.
ImagePalmCenterTarget is built from normalized image-space wrist/index MCP/middle MCP/pinky MCP palm-center coordinates.
Level 1.13 default base translation uses calibrated ImagePalmCenterTarget deltas, not raw absolute MediaPipe world coordinates.
Pressing c in the live overlay captures neutral_palm_center and the current human palm orientation as the human neutral pose.
For Level 1.13B, pressing c also captures neutral_hand_scale and the current robot base pose.
The robot base pose after calibration is neutral_robot_base plus configured translation and orientation deltas from that human neutral pose.
Default image_2d translation maps image x to robot lateral motion and image y to robot height.
When depth control is enabled, monocular hand-scale changes map in/out motion along the configured depth axis.
HandBaseTarget orientation estimation is used for optional relative orientation matching when --enable-base-orientation is passed.
Level 1.13C orientation control is wrist/base orientation control, not finger articulation.
orientation_mode: relative_palm maps human palm rotation deltas from the calibrated pose to the calibrated robot base pose.
orientation_dofs can stage roll-only testing before enabling roll,pitch,yaw together.
Orientation deltas can be adjusted with orientation_axis_signs/base_orientation_axis_signs and orientation_remap_matrix/base_orientation_remap_matrix.
Relative orientation is bounded by max_roll_deg, max_pitch_deg, max_yaw_deg, orientation_smoothing_alpha, orientation_deadband_deg, and max_rotation_step_degrees.
Live apps may override those Level 1.13D tuning values with CLI flags for manual table-pickup style testing.
On tracking loss or invalid palm frames, orientation should hold the last valid target instead of updating from invalid landmarks.
The pose_3d target position is a wrist-origin control signal by default; palm-center remains an explicit option.
The target position is not a robot joint command.
Anatomical left-hand orientation vectors may be mirrored into the right-hand robot target convention.
orientation_quat must be normalized and compatible with MuJoCo's wxyz convention.
Invalid or degenerate palm landmarks must return valid=False with finite neutral values.
No MuJoCo model, mocap-body, or actuator names here.
```

---

## Smoothing

Module:

```text
dexvision/features/smoothing.py
```

Contract:

```python
smoothed_features = smoother.update(features)
```

Rules:

```text
No camera access here.
No MuJoCo access here.
Must handle missing or low-confidence input.
```

---

## Retargeting

Module:

```text
dexvision/retargeting/*.py
```

Contract:

```python
joint_targets: dict[str, float] = retargeter.map(features_or_landmarks, robot_state=None)
```

Rules:

```text
Output must obey joint limits.
Retargeter can know robot joint names.
Retargeter should not step MuJoCo directly.
For free_space_gesture pinch collection, the curl retargeter may apply a bounded thumb-index pinch overlay from pinch_thumb_index while keeping the saved full action schema unchanged.
The pinch overlay may use simple hand-shape gates, such as requiring index bend and open middle/ring/pinky fingers, so loose thumb-index distance thresholds do not corrupt fist demos.
```

---

## MuJoCo Environment

Module:

```text
dexvision/sim/mujoco_env.py
```

Contract:

```python
state = env.reset()
state = env.step(action)
env.set_joint_targets(joint_targets)
state = env.get_state()
```

Rules:

```text
No camera logic here.
No MediaPipe logic here.
Should support headless load tests.
```

---

## Task Specs

Module:

```text
dexvision/sim/tasks.py
```

Contract:

```python
@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    skill_name: str
    required_objects: tuple[str, ...]
    observation_schema: "ObservationSchema"
    action_schema: "ActionSchema"
    success_condition: str
    failure_conditions: tuple[str, ...]
    max_episode_steps: int
    reset_config: dict
```

Rules:

```text
TaskSpec defines resettable MuJoCo task environments for Level 2 demos.
Task ids should be stable, for example reach_touch_target, button_press, and push_cube_to_target.
Each task must expose state fields needed to compute success/failure after recording.
Task specs should not train policies or call learning modules.
```

---

## Action Schema

Module:

```text
dexvision/logging/dataset_schema.py
```

Contract:

```python
@dataclass(frozen=True)
class ActionSchema:
    version: str
    base_position_target: slice | tuple[int, int]  # shape [3]
    base_orientation_target: slice | tuple[int, int]  # replay may use MuJoCo wxyz quaternion, shape [4]
    finger_actuator_targets: slice | tuple[int, int]  # shape [N]
    representation_notes: dict
```

Rules:

```text
Every Level 2 demo must preserve the full Level 1.13 action at each timestep.
The policy action includes base position target, base orientation target, and finger actuator targets.
Replay storage may keep MuJoCo wxyz quaternions.
Learning datasets may convert orientation to 6D rotation or another stable representation.
Action subsets may be exposed for ablations, but saved demos must retain the full action.
```

---

## Observation Schema

Module:

```text
dexvision/logging/dataset_schema.py
```

Contract:

```python
@dataclass(frozen=True)
class ObservationFieldLayout:
    source_array: str  # robot_states, tracking_quality, object_states, or task_states
    shape: tuple[int, ...]
    dtype: str
    units: str
    coordinate_frame: str
    normalization: str
    column_range: tuple[int, int] | None
    column_indices: tuple[int, ...]
    names: tuple[str, ...]
    optional: bool
    absence_rule: str | None
    mask_field: str | None

@dataclass(frozen=True)
class ObservationSchema:
    version: str
    fields: tuple[str, ...]
    shapes: dict[str, tuple[int, ...]]
    optional_fields: tuple[str, ...]
    layouts: dict[str, ObservationFieldLayout]
    compatibility_notes: tuple[str, ...]
```

Expected Level 2/3 fields:

```text
robot qpos/qvel
actuator controls
hand/base pose
hand/base velocity, if available
finger joint positions
finger joint velocities
object pose/velocity, when present
target pose
task state
tracking quality
success metric inputs
metadata/config snapshot
```

Rules:

```text
Observation schemas must be versioned and saved with demos.
level2/observation-layout-v2 is the default executable observation schema.
Every v2 field maps to one saved dense source array through an explicit column
range or ordered column indices.
robot_states.npy is packed as robot qpos, robot qvel, actuator controls,
commanded base position, and commanded base orientation.
Robot qpos/qvel and actuator-control layouts preserve the recorded MuJoCo
degree-of-freedom and actuator names in exact array order.
Finger joint fields select named MuJoCo joint columns from qpos/qvel; they are
not inferred from actuator-target count.
Each layout declares shape, float dtype, units, coordinate frame, optional
behavior, and future-learning normalization guidance.
Dense source-array widths must exactly match the maximum column declared by
their executable layouts.
object_states.npy and task_states.npy may be absent only when every field using
that source is optional and declares an absence rule or mask field.
Fields that are absent for a task should be masked or explicitly marked optional.
Legacy level2/observation-v1 free-space demos remain full-action replayable
through an explicit shape-only compatibility adapter. The adapter does not
invent unavailable finger-joint column mappings; observation extraction
requires migration to v2.
Future skill policies must declare the observation schema they were trained on.
```

---

## Demo Episode Schema

Module:

```text
dexvision/logging/dataset_schema.py
```

Contract:

```python
@dataclass
class DemoEpisode:
    metadata: dict
    landmarks: np.ndarray | None
    features: np.ndarray
    actions: np.ndarray
    robot_states: np.ndarray
    object_states: np.ndarray | None
    task_states: np.ndarray | None
    tracking_quality: np.ndarray
    timestamps: np.ndarray
    success: bool | None
```

Rules:

```text
metadata must include skill_name, task_id, episode_id, action_schema version, observation_schema version, robot model/config, task config, and teleop config snapshot.
free_space_gesture demos may include an optional gesture_label metadata field: open_palm, fist, point, pinch, peace_sign, or wave.
actions must preserve base_position_target, base_orientation_target, and finger_actuator_targets.
robot/task/object state must preserve inputs needed for replay, quality filtering, and task-specific success relabeling.
DemoEpisode validation should use synthetic arrays and should not require camera, GUI, or learning code.
```

---

## Demo Logging

Module:

```text
dexvision/logging/demo_logger.py
```

Contract:

```python
logger.start_episode(metadata)
logger.append(step_data)
logger.close(success=...)
```

Rules:

```text
Must save metadata and arrays.
Must validate array lengths.
Should not require video recording.
Must record skill_name/task_id when logging task demos.
Must preserve the full Level 1.13 action schema: base position target, base orientation target, and finger actuator targets.
Manual free_space_gesture recording should not append live frames until c successfully calibrates/centers the hand.
Should save task/object state and success metric inputs when present.
```

---

## Skill Cards

Module:

```text
dexvision/learning/skill_cards.py
```

Contract:

```python
@dataclass(frozen=True)
class SkillCard:
    skill_name: str
    task_id: str
    policy_checkpoint: str
    observation_schema: str
    action_schema: str
    inputs: dict
    preconditions: tuple[str, ...]
    success_condition: str
    failure_conditions: tuple[str, ...]
    metrics: dict
    known_limitations: tuple[str, ...]
```

Rules:

```text
Skill cards describe trained Level 3 policies for future Level 5 orchestration.
They must declare observation/action schemas and success/failure metrics.
The action schema must include base position target, base orientation target, and finger actuator targets.
Skill cards are metadata only; they must not implement an LLM planner or long-horizon orchestration.
```

---

## Learning Dataset

Module:

```text
dexvision/learning/datasets.py
```

Contract:

```python
sample = {
  "obs": Tensor,  # hand/base pose and velocity, finger qpos/qvel, object pose, target pose
  "action": Tensor,  # base position target, base orientation target, finger actuator targets
  "demo_id": str,
  "timestep": int,
}
```

Rules:

```text
No live camera.
No live teleop.
Should work from saved demos only.
Saved demos should preserve the full Level 1.13 base-plus-finger action at each timestep.
Early learning experiments may expose action subsets, but should not require recollecting demos.
Replay storage may keep MuJoCo wxyz quaternions; learning datasets may convert orientation to 6D or another stable representation.
```
