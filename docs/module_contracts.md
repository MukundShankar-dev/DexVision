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

fingertip_targets: np.ndarray = compute_normalized_fingertip_targets(landmarks)
# shape [5, 3], ordered thumb/index/middle/ring/pinky
```

Rules:

```text
Output must obey joint limits.
Retargeter can know robot joint names.
Retargeter should not step MuJoCo directly.
The Level 2.8 fingertip baseline accepts a MediaPipe-compatible landmark array
or a hand-tracking result and computes five palm-local fingertip positions.
Fingertip positions are normalized by palm width and use axes that cross the
palm from pinky to index, point from wrist toward the fingers, and follow the
palm normal.
The approximate fingertip solve maps long-finger palm-axis extension and thumb
CMC-to-tip extension to the existing configured Shadow Hand open/closed targets.
All returned targets must be clipped to the configured inclusive limits.
Missing or low-confidence tracking returns the safe open target set. If a solve
fails after a valid frame, the fingertip baseline returns the last valid target
set; before any valid solve it returns the safe open target set.
The Level 2.9 optimization retargeter minimizes palm-local 3D fingertip error,
configured actuator-limit violations, and change from the previous normalized
finger-control solution. The non-negative objective weights and positive
iteration limit are configurable.
The optimization retargeter accepts a MediaPipe-compatible landmark array or a
hand-tracking result. It uses bounded SciPy optimization when SciPy is available
and a deterministic projected-gradient solver otherwise.
Every optimization result is clipped to configured inclusive target limits.
Failed solves return the last valid target set, or the safe open target set
before the first valid solve. Each attempted solve records elapsed time,
objective value when available, success, and optimizer backend, and successful
solve time is logged.
For free_space_gesture pinch collection, the curl retargeter may apply a bounded thumb-index pinch overlay from pinch_thumb_index while keeping the saved full action schema unchanged.
The pinch overlay may use simple hand-shape gates, such as requiring index bend and open middle/ring/pinky fingers, so loose thumb-index distance thresholds do not corrupt fist demos.
```

---

## Retargeting Benchmark

Module:

```text
dexvision/evaluation/benchmark_retargeters.py
```

Contract:

```python
report = run_benchmark(
    episode_dirs,
    task_id="push_cube_to_target",
    config_path="configs/level1_teleop.yaml",
    retargeter_names=("curl", "fingertip", "optimization"),
    success_evaluator=replay_push_cube_metrics,
    bootstrap_samples=2000,
    bootstrap_seed=0,
)
save_benchmark_report(report, json_path=..., csv_path=...)
save_benchmark_plot(report, output_path=...)
```

Rules:

```text
Every retargeter must receive the same ordered saved episodes and frame streams.
The curl benchmark consumes the saved named feature layout; fingertip and
optimization methods consume the saved MediaPipe-compatible landmarks.
Latency measures only retargeter mapping time, not metric calculation or replay.
Action jerk is the mean L2 third difference after actuator-limit normalization.
Joint-limit rate counts scalar outputs outside the configured inclusive limits.
Fingertip error is palm-width-normalized Euclidean error under the shared
benchmark fingertip surrogate.
For push_cube_to_target, task success is recomputed by counterfactual headless
MuJoCo replay with recorded base actions and benchmarked finger targets.
Aggregate means carry deterministic 95% episode-bootstrap confidence intervals.
Push-cube replay measures signed distance from the cube geom to the five
collidable distal-fingertip geoms and the fraction of frames with fingertip
contact. These are actual MuJoCo geometry measurements, distinct from the
palm-local kinematic-surrogate fingertip error.
The recorded base trajectories were collected with curl retargeting, so the
counterfactual comparison must disclose curl-derived trajectory bias and must
not claim independent per-retargeter operator demonstrations.
Recorded raw demonstrations are read-only. JSON, CSV, and SVG reports are
written only to the requested report directory.
No policy training or live camera access belongs in this module.
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
    parameter_type: type
    parameter_schema: Mapping[str, Mapping[str, Any]]
    state_fields: tuple[str, ...]
    success_metric_inputs: tuple[str, ...]
    terminal_state_schema: Mapping[str, Mapping[str, Any]]
```

Rules:

```text
TaskSpec defines resettable MuJoCo task environments for Level 2 demos.
Task ids should be stable, for example reach_touch_target, button_press, and push_cube_to_target.
Each task must expose state fields needed to compute success/failure after recording.
Task specs should not train policies or call learning modules.
reach_touch_target success requires physical contact between the rh_palm body
and the collidable active target marker for the configured dwell duration.
The saved task state must include the palm_contact flag, closest contact
position, target position, contact-to-target distance, and dwell count so
success can be recomputed later.
button_press parameters declare a configured button_id plus either a press-depth
threshold or pressed-state target, with an optional world-frame approach pose.
Button reset selection must be deterministic for a fixed seed.
The saved button task state must include button identity and position, current
and target press depth, current and target pressed state, dwell count, optional
approach pose, terminal state, and deterministic initial robot/button state.
button_press success must be recomputable from the saved press depth, target
depth, current/target pressed states, and consecutive dwell count.
Button task code must not record demonstrations or add cube-push/learning work
in Level 2.7D.
When button_press runs in the shared task-board model, reach-touch target sites
and the active reach marker must be hidden; the active marker must also be
non-colliding. Reach-touch task instances must retain their original fixtures.
After reset, the selected button must be bright green and every non-target
button dark gray. Button replay must restore the same saved target cue.
push_cube_to_target parameters declare a configured object_id plus either a
world-frame target_pose or named target_zone_id, with an optional approach_side.
Cube start and target-zone selection must be deterministic for a fixed seed.
The saved cube object state must include world-frame position and orientation
plus free-joint linear and angular velocity.
The saved cube task state must include object identity, target identity and
position, target radius, optional approach side, planar object-to-target
distance, dwell count, terminal state, and deterministic initial object/robot
state.
push_cube_to_target success must be recomputable from the saved object position,
target position, planar distance, target radius, and consecutive dwell count.
Level 2.7G records push_cube_to_target pilot episodes with the full Level 1.13
action, 13-column cube pose/velocity object state, and the complete PushCubeState
task array. Recording metadata preserves the resolved object/start, target cue,
target radius, dwell requirement, and optional approach side.
Cube mode hides the vertical reach/button wall plus all reach and button
fixtures so the horizontal orange-cube/green-target task is visually isolated.
Reset aligns the hand free joint and mocap target behind the cube with the palm
upright and facing the cube, with fingers up, before stepping. The operator
calibrates with the real palm facing the camera and fingers up; increasing
camera-relative hand scale must then move the simulated palm along its normal
toward the cube. The free joint and mocap target must remain aligned so reset
does not sweep the robot through the object.
The task uses the model's palm-facing reset quaternion, not the visually
opposite back-of-hand quaternion. Live base orientation must remain disabled
for cube recording even when the Level 1.13 full-control preset is requested;
translation, camera-relative depth, and finger controls remain active. The
saved effective teleop snapshot must report these resolved enable flags.
Cube mode hides and disables forearm/wrist geometry where those support bodies
extend below the tabletop, while palm and finger visuals and collision geometry
remain active. Hand height is fixed for this planar task; image up/down motion
must not drive the palm into the table. Named target zones use the matching
lateral cube-start lane, and lateral position remains fixed on that lane. Thus
the live pilot requires only palm-normal depth motion and cannot saturate or
drift sideways. The live viewer starts from a three-quarter angle showing the
palm, cube, and target together.
Live cube recording does not enforce the task spec's nominal step timeout;
recording continues until success, a workspace safety failure, or operator q.
Push-cube recording must request and save an explicit operator success/failure
label just like reach-touch and button-press pilots. Automated relabeling must
retain a null operator label for older raw recordings rather than rewriting
their metadata.
The effective task-specific base neutral/workspace override must be saved in
the teleop config snapshot for quality filtering.
The cube recording profile uses task-specific monocular-depth gain, fixed
lateral position/height/orientation, moderate depth smoothing, a smaller depth
deadband, and bounded forward workspace/step limits.
These limits prevent table penetration, abrupt scoop/lift motion, and
overtravel while preserving responsive control. Effective values must be
recorded, not silently inherited from the base YAML.
Live cube recording must also advance MuJoCo by at least one nominal camera
control period per frame. With the current 0.002-second model timestep and
30 Hz control rate this is 17 simulation steps per frame; using the generic
2-step default causes visible mocap-weld lag. Save the effective
sim_steps_per_frame value in recording metadata.
Replay must use that saved `sim_steps_per_frame` cadence by default so recorded
contact dynamics are reproduced. An explicit `--sim-steps-per-action` may
override it; legacy demos without saved cadence fall back to one step.
Quality filtering must accept workspace axes whose minimum equals maximum;
such axes are intentionally fixed and are excluded from limit-hit accounting.
Dataset summaries must include push-cube target-position distributions and
count a clean success when the episode both recomputes as successful and
passes quality filtering. This target distribution is the audit source for
coverage of all three configured cube lanes.
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

## Success Relabeling

Module:

```text
dexvision/logging/relabel_success.py
```

Contract:

```python
report = relabel_demo_dataset(dataset_dir)
save_relabel_report(report, output_path)
```

Rules:

```text
Task-specific dispatch supports reach_touch_target, button_press, and
push_cube_to_target.
For reach_touch_target, recompute distance from saved target and touch positions
and validate the saved distance.
For reach_touch_target, recompute consecutive qualifying contact frames from
saved palm-contact flags and use the fixed task distance/dwell thresholds.
For button_press, recompute success from saved press depth, target press depth,
button/target pressed states, and dwell count.
For push_cube_to_target, recompute planar object-to-target distance from saved
object and target positions, validate the saved distance and consecutive dwell
count, and require the 13-column cube pose/velocity object state.
Cube target position, target radius, object id, target source, and dwell
requirement must remain reconstructable from immutable episode metadata/state.
Button target depth/state must stay constant within an episode, and the saved
button id plus dwell requirement must be present in task metadata.
Preserve the operator label and recomputed label together in the audit report.
Never rewrite raw episode metadata or arrays.
Missing or inconsistent metric inputs must produce clear errors.
```

---

## Demo Quality Filtering

Module:

```text
dexvision/logging/quality_filters.py
```

Contract:

```python
report = filter_demo_dataset(dataset_dir, thresholds=QualityThresholds())
save_quality_report(report, output_path)
```

Rules:

```text
Level 2.7 evaluates saved reach_touch_target, button_press, and
push_cube_to_target pilot episodes; scaled reach-touch evaluation remains
supported.
Quality thresholds must be versioned, configurable, and embedded in each report.
Filters cover mean tracking confidence, missing frames, feature jitter, action
jerk, actuator-limit hits, recomputed task failure, and workspace-limit hits.
Actuator and workspace bounds come from the saved episode metadata/config snapshot.
Task success must use the task-specific recomputed label.
Reports group results by skill_name and task_id.
Reports may be added beside a dataset, but raw episode metadata and arrays must
never be deleted or rewritten.
Missing, inconsistent, or non-finite quality inputs must produce clear errors.
```

---

## Scaled Reach-Touch Collection Gate

Modules:

```text
dexvision/logging/collection_planner.py
dexvision/apps/select_reach_target.py
```

Contract:

```python
plan = plan_reach_touch_collection(dataset_dir)
python -m dexvision.apps.select_reach_target --run
```

Rules:

```text
Target selection balances quality-passed successful episodes, not raw attempts.
Ties between least-represented clean targets are selected randomly.
The next raw episode path must never overwrite an existing directory.
Quality-gated attempts record into a staging directory first.
Only episodes passing the current Level 2.7 quality filters move into raw/.
Rejected or invalid attempts remain outside raw/ for audit and tuning.
Existing raw episode directories must never be deleted or rewritten.
```

---

## Scaled Button-Press Collection Gate

Modules:

```text
dexvision/logging/collection_planner.py
dexvision/apps/select_button_goal.py
```

Contract:

```python
plan = plan_button_press_collection(dataset_dir)
python -m dexvision.apps.select_button_goal --run
```

Rules:

```text
The versioned button dataset config defines training button/depth goals and
held-out evaluation states before scaled collection starts.
Goal selection balances quality-passed successful episodes by exact configured
button id and target press depth, not raw attempts or button id alone.
Quality-gated attempts record into a staging directory first.
Only episodes passing the current Level 2.7 quality filters move into raw/.
Rejected or invalid attempts remain outside raw/ for audit and tuning.
Existing raw episode directories must never be deleted or rewritten.
Held-out button/depth states must never be selected for Level 2 training data.
```

---

## Scaled Push-Cube Collection Gate

Modules:

```text
dexvision/logging/collection_planner.py
dexvision/apps/select_push_cube_goal.py
```

Contract:

```python
plan = plan_push_cube_collection(dataset_dir)
python -m dexvision.apps.select_push_cube_goal --run
```

Rules:

```text
The versioned push-cube dataset config defines exact training start/target/
approach goals and held-out evaluation starts/target poses before scale-up.
Goal selection balances quality-passed successful episodes, not raw attempts.
Quality-gated attempts record into a staging directory first.
Only episodes passing the current Level 2.7 quality filters move into raw/.
Rejected or invalid attempts remain outside raw/ for audit and tuning.
Existing raw episode directories must never be deleted or rewritten.
Held-out cube starts and target poses must never be selected for Level 2
training data.
```

---

## Dataset Summary and Task Readiness

Module:

```text
dexvision/logging/dataset_summary.py
```

Contract:

```python
config = load_reach_touch_dataset_config("configs/reach_touch_dataset.yaml")
button_config = load_button_press_dataset_config("configs/button_press_dataset.yaml")
cube_config = load_push_cube_dataset_config("configs/push_cube_dataset.yaml")
report = summarize_demo_dataset(
    dataset_dir,
    reach_touch_config=config,
    button_press_config=button_config,
    push_cube_config=cube_config,
)
```

Rules:

```text
The reach-touch train/held-out split is versioned in YAML.
Training and held-out target ids and positions must be distinct.
Readiness uses recomputed success plus quality-pass results, not operator labels alone.
The summary reports raw, recomputed-success, quality-pass, and clean-success counts per target.
Level 3 readiness requires complete relabel/quality coverage, no disagreements,
one action/observation schema version, the configured clean total and per-target
minimums, and declared uncontaminated held-out evaluation targets.
Summary generation remains read-only with respect to raw episode directories.
Button-press summaries report exact button/depth goal counts plus observed
button and robot initial-state distributions.
Button readiness requires complete relabel/quality coverage, no label
disagreements, one action/observation schema version, the configured clean total
and per-goal minimums, and declared uncontaminated held-out button/depth states.
Push-cube summaries report exact cube start/target/approach goal counts plus
observed cube and robot initial-state distributions.
Push-cube readiness requires complete relabel/quality coverage, no label
disagreements, one action/observation schema version, the configured clean total
and per-goal minimums, and declared uncontaminated held-out cube starts and
target poses.
```

---

## Skill Card Metadata Stubs

Module:

```text
dexvision/logging/skill_card_metadata.py
```

Contract:

```python
metadata = build_skill_metadata(
    task_spec,
    dataset_summary_path="data/demos/reports/summaries/dataset_summary.json",
)
save_skill_metadata(metadata, output_path)
```

Rules:

```text
Level 2 metadata stubs are exported from executable task specs and the matching
dataset-summary group.
Stubs declare the full Level 1.13 base-position, base-orientation, and finger-
actuator action layout.
Every typed goal parameter declares type, shape, units, coordinate frame, and
whether it is required.
Success/failure conditions, timeout, terminal-state fields, preconditions,
dataset-summary provenance/readiness, and known limitations are preserved.
The task-spec and dataset-summary action/observation schema versions must agree.
policy_checkpoint remains null until Level 3 trains and exports a policy.
Metadata export does not train or evaluate a policy and does not implement
skill orchestration.
```

---

## Level 5 Full-Scale Skill Qualification

Level 5 consumes the immutable Level 4 dataset release and frozen split
manifests. It must not mutate accepted episodes or silently recollect data to
repair a failed policy.

Rules:

```text
Every training run records dataset, split, config, schema, seed, and checkpoint
digests.
Ground-truth-state and perception-grounded evaluations remain separate.
Held-out sessions, object instances, goal regions, and scenes are not used for
checkpoint or hyperparameter selection.
Every core skill receives one status: qualified, experimental, or failed.
Only qualified and schema-compatible skills are enabled in the default
supervised registry.
Experimental and failed skills remain visible in reports.
When data are insufficient, create a new Level 4 release rather than modifying
the old one in place.
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
Skill cards describe qualified Level 5 policies for future Level 7 orchestration.
Level 3 feasibility checkpoints may emit provisional metadata, but provisional
or failed policies must not appear as qualified skills by default.
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
  "obs": Tensor,  # named robot/base/finger/object state plus tracking quality
  "goal": Tensor,  # task-typed goal parameters resolved from saved metadata
  "action": Tensor,  # base position target, base orientation target, finger actuator targets
  "episode_id": str,
  "demo_id": str,
  "goal_id": str,
  "timestep": int,
  "timestamp": float,
  "tracking_quality": Tensor,
  "quality_passed": bool,
  "recomputed_success": bool,
}
```

Rules:

```text
No live camera.
No live teleop.
Should work from saved demos only.
Clean datasets require complete saved quality and recomputed-success reports.
Observation columns are resolved through executable saved layouts and named
MuJoCo fields; packed offsets and dimensions must never be inferred.
Task-irrelevant fixture joints are excluded by their saved names and the
task's required-object declaration.
Saved wxyz base orientation observations are converted to a sign-invariant 6D
rotation representation. Stored action quaternions remain unchanged.
Reach, button, and push goals are encoded from explicit typed task_config
fields rather than dynamic target_state prefixes.
Saved demos should preserve the full Level 1.13 base-plus-finger action at each timestep.
Early learning experiments may expose action subsets, but should not require recollecting demos.
Replay storage may keep MuJoCo wxyz quaternions; learning datasets may convert orientation to 6D or another stable representation.
The first reach split is defined by level3/reach-evaluation-v1.
Offline demonstration validation uses only configured training targets and is
separate from closed-loop held-out-goal rollout evaluation.
Existing episodes without recording-session ids may be split by whole episode,
but results must not claim cross-session generalization.
When a real session id is present, all episodes from that session must remain in
one split.
Normalization statistics must come from the training partition only.
Split manifests save every episode assignment, goal id, recording-session-id
availability, per-episode data digest, dataset digest, schema versions, and
training-only normalization statistics.
Reserved held-out targets must not be added to demonstration training or
validation partitions merely to satisfy a test-split requirement.
```

---

## Level 4 Comprehensive Dataset Contract

The immutable Level 2 release remains the input to Level 3 feasibility work.
Level 4 creates a new release; it must not rewrite or relabel the Level 2
archive in place.

Every new Level 4 episode must add these provenance fields to the existing
episode contract:

```text
recording_session_id: genuine stable id for one collection session
operator_id: stable local/pseudonymous label
source: teleoperation, scripted, policy_rollout, or corrective_intervention
object_instance_ids: concrete objects used in the episode
goal_condition_id: frozen coverage-matrix cell
camera_or_render_config: null only when no image stream is claimed
intervention_interval and failure_reason: required for corrective data
code/config/schema versions and random seed
```

Rules:

```text
Do not invent session ids for Level 2 legacy episodes.
Keep complete sessions together whenever a split claims session independence.
Keep held-out object instances and goal regions out of training when the
evaluation protocol claims generalization to them.
Expert successes, ordinary failures, policy rollouts, and corrective
interventions must remain distinguishable.
Synchronized image/state/action streams must have validated alignment.
Every immutable release needs a manifest, checksum, schema versions, split
manifests, coverage report, and retrieval instructions.
```

---

## World State and Object Observation

Planned Level 4 modules:

```text
dexvision/sim/world_state.py
dexvision/perception/object_observations.py
```

Minimum conceptual contract:

```python
@dataclass(frozen=True)
class ObjectObservation:
    object_id: str
    class_id: str
    position: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    linear_velocity: tuple[float, float, float] | None
    source: str
    confidence: float
    timestamp: float
    frame: str
```

Rules:

```text
Object ids are stable within an episode and deterministic after reset.
Units and coordinate frames are explicit.
Simulator ground truth and inferred perception share one typed outer schema.
Source and confidence distinguish measured/inferred state from ground truth.
Stale, missing, or ambiguous state must fail clearly at the skill boundary.
A detector, tracker, or compact VLM may populate semantic/object hypotheses;
none may bypass metric pose validation or safety checks.
```
