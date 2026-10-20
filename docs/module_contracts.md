# Module Contracts

This file defines expected inputs/outputs between modules. Keep it updated as the repo evolves.

Current staging: Level 5.6 is complete with V29/fresh V8 automated gates and
owner manual acceptance on October 16, 2026, bound in
[level5_perception_acceptance.json](level5_perception_acceptance.json).
All five accepted inferred-control tracks and default registry admission remain
unsupported. Level 5.7 is unstarted; this completion request does not authorize
starting it. [CURRENT_STATUS.md](CURRENT_STATUS.md) governs checkpoint selection.
Versioned contracts below preserve earlier methods and their historical gates.

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

    # Present only when episode_schema_version == level4/episode-v1:
    requested_actions: np.ndarray | None
    commanded_actions: np.ndarray | None
    applied_actions: np.ndarray | None
    prior_commanded_actions: np.ndarray | None
    prior_applied_actions: np.ndarray | None
    safety_masks: np.ndarray | None
    safety_reasons: np.ndarray | None
    request_sources: np.ndarray | None
    online_phases: np.ndarray | None
    audited_phases: np.ndarray | None
    phase_relevance_masks: np.ndarray | None
    intervention_flags: np.ndarray | None
    failure_reasons: np.ndarray | None
    action_timestamps: np.ndarray | None
    task_timestamps: np.ndarray | None
    state_timestamps: np.ndarray | None
    rgb_frames: np.ndarray | None
    rgb_timestamps: np.ndarray | None
```

Rules:

```text
metadata must include skill_name, task_id, episode_id, action_schema version, observation_schema version, robot model/config, task config, and teleop config snapshot.
free_space_gesture demos may include an optional gesture_label metadata field: open_palm, fist, point, pinch, peace_sign, or wave.
actions must preserve base_position_target, base_orientation_target, and finger_actuator_targets.
robot/task/object state must preserve inputs needed for replay, quality filtering, and task-specific success relabeling.
DemoEpisode validation should use synthetic arrays and should not require camera, GUI, or learning code.
Level 4 fields remain absent for legacy Level 2 episodes; loaders must not invent session or provenance values.
For Level 4, actions.npy exactly mirrors applied_actions.npy while requested, commanded, prior, and safety arrays preserve the reconstructable control path.
Level 4 phase intervals use inclusive start and exclusive end frame indices and must reconstruct online_phases.npy without gaps or overlaps.
Level 4 state, task, action, and optional RGB timestamps must satisfy the frozen alignment tolerances.
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
When Level 4 metadata is declared, missing per-sample requested/commanded/applied fields may be losslessly adapted from an unchanged legacy action, but explicit safety changes must carry matching masks and stable reasons.
Level 4 writes are atomic and append-only. The overwrite flag cannot replace an existing Level 4 episode.
```

---

## Level 4 Session Manifest

Module:

```text
dexvision/logging/session_manifest.py
```

Contract:

```python
manifest = append_session_manifest(path, recording_session)
episode_dir = next_episode_directory(dataset_dir, recording_session_id=session_id)
```

Rules:

```text
Each recording_session_id identifies one genuine process/calibration block and is unique in the manifest.
Each session records operator_id, whole-session split, process-start timestamp, reset seed, and calibration-record digest.
Appending a duplicate session id fails; resuming episode allocation skips every existing directory and never overwrites it.
Legacy Level 2 episodes are not added to a Level 4 session manifest.
```

---

## Level 4 Phase Labels

Module:

```text
dexvision/logging/phase_labels.py
```

Contract:

```python
phase = tracker.update(current_state)
intervals = phases_to_intervals(online_phases)
segments = derive_pick_place_segments(intervals, frame_count=frame_count)
report = phase_disagreement_report(online_phases, audited_phases)
```

Rules:

```text
The tracker evaluates ordered transitions from only current and prior state and takes at most one edge per control sample.
Audited annotations are stored separately and never influence an online transition.
Saved intervals are monotonic, non-overlapping, gap-free, and exactly reconstruct the per-frame online labels.
A complete pick/place sequence derives reach_object, pick_object, and place_held_object segments without counting them as new episodes.
Per-frame action-relevance masks must exactly match the versioned mask for the saved online phase.
```

---

## Level 4 Collection Pilot and Coverage

Modules:

```text
dexvision/logging/level4_collection.py
dexvision/evaluation/dataset_coverage.py
dexvision/apps/record_demo.py
```

Contract:

```python
episodes = discover_pilot_episodes(dataset_dir)
report = summarize_level4_coverage(
    config_path="configs/level4_dataset.yaml",
    dataset_dir="data/demos/level4_pilot",
)
save_coverage_report(report, output_path)
```

Rules:

```text
Pilot acceptance evidence is an append-only sidecar and never rewrites episode metadata or arrays.
Later user-confirmed visible replays append to a separate dataset-level manual replay manifest.
The level4_workcell recorder resolves a frozen coverage cell into one reach, complete pick/place, push, or press task and records all six object states plus task metrics.
Its logical base control point is the palm/grasp site; the weld offset keeps the long forearm outside the task board, and reset aligns the dynamic free joint to that weld before simulation advances.
Historical live-control pilot code remains readable only to reproduce the recorded rejection evidence. It is not an allowed source for new Level 4+ collection, corrections, training, or qualification.
The reach marker is a pre-grasp cue high enough to keep the Shadow Hand clear of staged objects; dense reach task state preserves distance, orientation error, and scene disturbance separately from dwell.
The selected reach entity is enclosed by a bright emissive magenta non-colliding wireframe cage, while a separate floating three-axis cyan cross marks the desired palm position. The hand-attached grasp site is hidden.
Scripted expert execution uses the safe transit plane and protected target corridor and records its frozen motion parameters in episode metadata.
`--workcell-dry-run` exercises a resolved scripted workcell cell through temporary logger storage without adding a session-manifest entry or retaining an episode.
Each recorder invocation creates one genuine process/calibration session manifest entry and one collision-free append-only episode path.
An expert-accepted episode passes every frozen acceptance gate, has matching operator and recomputed success labels, and carries no rejection reason.
Ordinary failures and rejected attempts remain visible and never count as expert successes.
Session split ownership must agree with both the session manifest and coverage cell.
Complete pick/place episodes count once as episodes and yield separate reach, pick, and place segment counts.
Coverage reports include pilot rates, phase agreement, replay/recomputation evidence, object families, target types, rejection reasons, collection time, and storage projection.
Headless replay evidence does not satisfy the manual replay gate.
Coverage counts remain provisional until the pilot passes and the user confirms one visible replay of every required skill.
```

Level 4.3 scripted-expert contract:

```python
class ExpertController:
    def reset(self, task: object, world_state: WorldState) -> None: ...

    def step(
        self, world_state: WorldState
    ) -> tuple[RequestedAction, str, bool, str | None]: ...
```

Rules:

```text
Expert controllers use privileged simulator WorldState and causal current/prior state only.
The common expert boundary is deterministic for a frozen task, reset state, config, and seed.
Experts produce the existing complete named requested-action layout before the recorder sees the sample; there is no parallel expert episode format.
For source=scripted, requested_action is the nominal scripted action. A bounded learning residual is derived read-only as applied_action - requested_action.
Commanded/applied actions, prior actions, per-field safety masks/reasons, phases, timestamps, and provenance retain the Level 4.2 contract.
The logical motion point is the palm/grasp site. Candidate waypoint segments are checked in a copied MuJoCo state before execution.
The first planner is only rise, horizontal transit, protected pre-contact approach, and corridor descent. A general planner is evidence-triggered, not part of the initial contract.
Button press fixes posture/orientation and moves along the fixture normal; push fixes posture/orientation/fingers and moves along a task-local forward axis during contact.
Grasp templates are object-family-specific and expose one scalar open-to-closed synergy. Place begins only from a verified held object.
Causal phases may be policy inputs but audited future knowledge may not be used online.
The initial learned adapter consumes simulator state and emits a bounded task-local Cartesian delta or residual. Deterministic code expands it to the full requested-action layout before logging.
The first learned model is a small MLP. RGB, VLMs, LLMs, larger models, and action chunking are outside the initial Level 4.3 learning probes.
```

Levels 4.3A--E implement this boundary in
`dexvision/sim/level4_expert.py`. `SafeWaypointReachExpert` emits the existing
27-field named action and validates rise/horizontal/corridor/descent candidates
in copied MuJoCo state. `DeterministicButtonPressExpert` retains one fixed
neutral finger posture and base orientation, reaches a task-relative
pre-contact pose, moves only along the button's positive joint normal during
contact, satisfies the existing depth/dwell metric, and retracts through the
same corridor until contact is released. Its copied-state qualification rejects
workspace or robot-joint-limit violations, non-target object disturbance,
table contact, and any hand contact other than hand self-contact or contact with
the requested button. `DeterministicPushExpert` constructs a task-local planar
axis from the saved object start and target, uses a family-specific partial
index curl with the other fingers flexed, rotates only at the safe transit
plane, and then freezes height, orientation, and finger targets throughout
contact. Standalone push trials park non-target objects on the lower floor.
Contact translation is positive along that axis until the distance, speed,
table-support, upright-tilt, and dwell metric passes, followed by an axial
retract. The final metric must remain qualified for five released samples;
transient success is not latched. Its copied-state validator rejects board or
workspace exits, tilt above 10 degrees, non-target contacts, limit violations,
and planar neighbor disturbance. Qualification also runs the actual feedback
controller in one copied state and replays the exact resulting action sequence
in a second independent copy. A bounded family-specific lateral/contact-height
portfolio may select the first candidate that passes both executions; it does
not relax task or safety metrics. Contact loss may trigger only the configured
bounded re-approach count. When contact is first established at approach
completion, the controller must emit `push_contact` before `settle`, even if the
same physical contact already moved the object inside the terminal radius.
`record_demo --source scripted` routes `reach_object`,
`press_button`, `push_object_to_target`, and standalone `pick_object` through
these experts and the existing Level 4 logger. `DeterministicGraspLiftExpert`
uses configuration-owned cuboid, cylinder, and flat-puck object-relative
templates, deterministic retargeter open/full-flexion endpoints, and one scalar
closure synergy per family. Standalone grasp qualification retains the selected
object's seeded pose and parks non-target objects on the lower floor so the
known clutter limitation does not masquerade as a grasp failure. Its copied
trajectory and live controller require initial table support, two or more hand
contacts for the held relation, at least 0.040 m lift, loss of table support,
retention, and ten stable samples at no more than 0.020 m/s. Table contact is
allowed during acquisition and early lift only; after the lift threshold,
wrong-object, fixture, or table contact fails qualification. Complete
pick/place composes the shared grasp stage through the same expert and
action/logger boundary. Its copied-state qualification records the exact
candidate actions and requires that same open-loop sequence to acquire, retain,
transport, release, and settle successfully in a fresh independent copy; a
closed-loop-only success cannot qualify a saved demonstration. Fatal expert
reasons terminate recording immediately and remain rejected evidence. For the
4.3E orientation requirement, each family uses
a configuration-owned 80--90 degree side-on wrist pose analogous to a
hammer-curl grip. Cuboid wrist yaw is conditioned on the object's seeded yaw;
axially symmetric objects ignore irrelevant spin. A declared rotation-only
world-orientation hold is applied during lift, stabilize, transport, and place;
it does not set object position or linear velocity, and it ends before release.
`DeterministicPlaceExpert` accepts only a genuinely held selected object,
preserves the seeded object orientation, transports the grasp site above the
resolved target, descends to a configuration-owned release height, opens the
same scalar synergy, clears the released object vertically, waits for supported
linear and angular settling, and retracts. Family-specific planar release
offsets compensate deterministic post-release drift without weakening the
frozen 0.025 m three-dimensional place metric. Standalone pick and composed
pick/place apply task-local six-dimensional table contact and explicit
round-object rolling friction; push retains the baseline model contact
parameters. These exact overrides, the initial object quaternion, and the
source/destination cues are recorded in immutable metadata and restored by
replay. Replay reapplies the rotation-only hold at a bounded internal cadence
so visible and headless trajectories retain the same orientation. Copied-state
validation rejects premature release, unsafe hand contact,
workspace or joint-limit violations, and non-target disturbance. Complete
episodes retain the existing phase vocabulary and derive compatible reach,
pick, and place segments from the stored causal intervals.

The button body pose is stored in every Level 4 initial-state entity map and is
restored before replay, so episodes remain deterministic across the reachable
fixture placement introduced for 4.3B. A scripted button episode is labeled a
success if the causal metric succeeds at any press sample, while recording
continues through the required retract/release phase. Its final released state
therefore need not itself satisfy the transient press metric.

All scripted episodes must both attain their metric and finish the expert's
declared terminal phase before they can be labeled successful; reaching a frame
limit after a transient success is a failed/incomplete recording. Reach ends at
its qualified goal, button finishes after a latched press and complete release,
push finishes only while its final released state remains qualified, and
standalone grasp ends only after the stable held-object dwell.

Level 4.3F expert/replay audit modules:

```text
dexvision/evaluation/level4_expert_audit.py
dexvision/apps/audit_level4_experts.py
```

Contract:

```python
report = audit_expert_architecture(
    episode_dirs,
    config_path=dataset_config,
    workcell_config=workcell_config,
)
save_expert_audit_report(report, output_path)
```

Rules:

```text
The frozen audit requires at least two accepted resets for every scripted source skill: reach, button, push, standalone grasp-and-lift, and complete pick/place.
Each episode is loaded through the existing schema validator; the audit never introduces a parallel episode format.
The saved seed, coverage cell, typed goal, task config, and complete reset state must regenerate the same initial task before replay. Expert audits replay directly from that prepared task state; they must not reset it a second time and discard the single-object trial preparation.
Requested, commanded, applied, and prior action arrays, script request-source labels, causal phase intervals, aligned state/action/task timestamps, schema versions, and identity provenance must be complete.
Headless replay uses the saved simulation cadence and recomputes the task metric from replayed world state; complete pick/place must demonstrate both a held-object pick and a successful terminal place.
Accepted episodes must agree with their operator success label, contain zero safety/intervention evidence, and keep planar non-target disturbance within the skill's configured limit.
Ordinary failures remain listed with rejection reasons and never increment accepted source or derived-skill counts.
Generated JSON reports are written atomically outside immutable episode directories and do not create or mutate pilot-review decisions.
Visible replay remains a separate manual gate and cannot be satisfied by this headless audit.
```

Level 4.3G state-only button learning modules:

```text
configs/level4_button_learning_pilot.yaml
dexvision/learning/level4_lowdim.py
dexvision/evaluation/level4_button_learning.py
```

Contract:

```python
config, digest = load_button_learning_config(path)
trajectories = collect_button_expert_trajectories(config)
training = train_button_delta_policy(trajectories, config)
rollouts = evaluate_button_policy(training.policy, config)
report = run_button_learning_pilot(config_path=path)
```

Rules:

```text
The pilot contains exactly 20 scripted successes owned by one train, one validation, and one untouched test session in a 14/3/3 episode split.
Approach-class offsets and deterministic seed jitter alter the physical hand reset while remaining inside the frozen safe workspace.
Workcell.set_hand_base_reset_pose aligns both the mocap target and welded free hand base directly, avoiding a reset-only dynamic transient.
The observation is simulator-state-only and causal: end-effector-to-button relative pose, button state, six-dimensional base velocity, current phase one-hot, previous applied XYZ delta, and target press depth.
The learned output is only fixture-frame dx/dy/dz. Deterministic code supplies the complete fixed orientation and finger posture and applies frozen phase-local delta limits before producing the existing full named action.
Exactly one 64-by-64 tanh MLP with phase-balanced MSE, training-only normalization, and validation-loss epoch selection is allowed before any diagnosis.
Training consumes only the train session, selects by validation, and reports the untouched test loss separately.
Closed-loop qualification uses 20 separately seeded test-owned resets, requires at least 0.80 success, and counts workspace, joint-limit, wrong-button, unintended-contact, and invalid-action violations explicitly.
Rollout failures must receive a causal diagnostic category before increasing the 20-success dataset or changing the model class. RGB, action chunking, larger models, and expert fallback are prohibited.
The qualified formulation retains the same 20 episodes and MLP after diagnosing contact-induced open-finger joint-limit failures and correcting only the deterministic fixed posture and contact-axis adapter.
```

Level 4.3H state-only push learning modules:

```text
configs/level4_push_learning_pilot.yaml
dexvision/learning/level4_push_lowdim.py
dexvision/evaluation/level4_push_learning.py
```

Contract:

```python
config, digest = load_push_learning_config(path)
trajectories = collect_push_expert_trajectories(config)
training = train_push_delta_policy(trajectories, config)
rollouts = evaluate_push_policy(training.policy, config)
report = run_push_learning_pilot(config_path=path)
```

Rules:

```text
The push probe reuses the qualified causal observation convention, training-split-only population normalization, phase input, previous applied XYZ delta, 17-simulation-step cadence, shared task-local XYZ adapter, and 64-by-64 tanh MLP class.
The observation adds only push state: end-effector/object/goal geometry in the initial object-to-target frame, relative wrist orientation, object and base velocities, object family, and the target stop distance.
The scripted expert remains the nominal safe-waypoint, orientation, finger-posture, contact-axis, settle, and retract controller. The learned three-value output is only a bounded task-frame tracking residual around that nominal request.
Residual magnitude is at most 1 mm during free-space approach and 0.5 mm along the task axis during contact/retract; settle emits zero residual. Deterministic code rejects lateral/vertical contact residuals and expands the result to the existing full named action.
Exactly 20 scripted successes use whole-session 14/3/3 ownership over mutually disjoint probe conditions. Model selection uses validation loss and test loss remains untouched by training.
Closed-loop qualification uses 20 separately seeded held-out resets across cuboid and flat-puck conditions, requires at least 0.70 success, and explicitly counts board exits, neighbor disturbance, workspace/joint/contact/tilt safety failures, and invalid actions.
The provisional right-bin coverage cells are excluded because copied-state qualification measured fixture-path and safe-workspace infeasibility under the current controller; Level 4.3I must revise the provisional collection matrix rather than relabel these cells as qualified.
The initial learned-only delta formulation failed 0/20 from compounding approach error. That measured diagnosis justified the already-specified nominal-plus-residual formulation; it did not justify action chunking. No image input, chunking, larger model, extra recipe, or data increase is permitted by this checkpoint.
```

Level 4.3I final source/count freeze:

```text
configs/level4_dataset.yaml was level4/workcell-dataset-plan-v2 at the Level 4.3I freeze while the episode schema remained level4/episode-v1.
Every required coverage row has one required_source, one split_owner, and explicit train/validation/test accepted minima.
The matrix minimum is 114 and the planning maximum is 140: reach 20, pick/place 42, push 20, button 20, and failure/correction 12.
The accepted v2 snapshot had source minima of scripted 98, human-controlled 13, policy_rollout 0, and corrective_intervention 3; provenance classes may never be merged or relabeled.
The v3 decision supersedes those human-controlled cells before active collection. Scripted experts supply every required nominal skill and deterministic scripted controllers supply the required corrections.
The 114-episode v3 matrix is an immutable integration/coverage anchor, not the final learning-data sufficiency claim. Level 4.5B must publish a versioned v4 expansion with at least 16 unique accepted episodes per nominal coverage cell: reach 160, pick/place 480, push 192, and button 160. Level 4.6 must provide at least 10 unique episodes per failure/correction cell, or 120 total, producing a 1,112-episode release-candidate floor.
Every expansion episode has a unique reset seed and initial-state digest and samples frozen continuous in-cell variation. A validation-only nested-subset scaling probe may require a larger versioned tranche; test-owned episodes cannot determine the dataset size or randomization bounds.
Push coverage is restricted to the 12 Level 4.3H-qualified condition cells. The eight removed provisional cells remain named with evidence-backed exclusion reasons.
Whole-session split ownership, held-out object/goal rules, train-only normalization, and untouched test restrictions remain mandatory.
The old storage projection applies only to the 140-episode anchor ceiling. Level 4.5B must reproject state/action and sampled-RGB payloads at the 1,112-episode release floor and the possible 32-per-nominal-cell expansion ceiling. Use Git LFS only below the documented quota; otherwise keep manifests, splits, checksums, and retrieval instructions in Git and use immutable external object storage for payloads.
Level 4.4 cannot start until the user accepts this matrix and Level 4.3I is marked complete.
```

Level 4.4 core collection planning and audit:

```python
assignments = build_level4_core_collection_plan("configs/level4_dataset.yaml")
report = summarize_level4_coverage(
    config_path="configs/level4_dataset.yaml",
    dataset_dir="data/demos/level4",
)
core = report["level4_4_core_collection"]
```

Rules:

```text
The core plan expands only reach, push, and button minima: 60 accepted episodes
across 32 cells. The user-directed v3 revision requires all 60 assignments to
be scripted. Historical human-controlled episodes keep their provenance locally
but are excluded from the active dataset, release, training, and evaluation
plan.
Every assignment names one frozen source, split, session slot, repetition, and
deterministic reset seed; pick/place and failure/correction cells are excluded.
The four held-out push cells override the generic test-seed sequence with seeds
200, 1004, 201, and 202 already qualified by Level 4.3H. Overrides may name only
a core cell with exactly one required episode and may not weaken safety gates.
Recorder-side frozen-owner enforcement rejects source or split mismatches before
creating an episode or session manifest entry.
The Level 4.4 verdict requires every core cell, the frozen source minima, at
least two train sessions plus one validation and one test session, session and
target balance, held-out isolation, and append-only review evidence.
Rejected or failed attempts never satisfy expert minima and remain discoverable
through their non-accepted review records.
Test-owned data may be collected and audited only after train and validation are
frozen; it may not influence tuning, normalization, thresholds, or checkpoint
selection.
```

Level 4.5A complete pick/place anchor planning and audit:

```python
assignments = build_level4_pick_place_collection_plan(
    "configs/level4_dataset.yaml"
)
report = summarize_level4_coverage(
    config_path="configs/level4_dataset.yaml",
    dataset_dir="data/demos/level4",
)
anchor = report["level4_5a_pick_place_collection"]
```

Rules:

```text
The anchor plan expands only the frozen 30 complete pick/place cells into 42
accepted scripted assignments. Every accepted complete episode must derive one
reach_object, one pick_object, and one place_held_object segment from executable
saved phase boundaries; a failed phase cannot supply a later success segment.
Pick/place tasks isolate the unrelated vertical button fixture. Return-bin wall
geometries remain visible but are collision-disabled because MuJoCo cylinders
are solid rather than hollow receptacle rims; the target floors, placement
metrics, object contacts, disturbance checks, and joint-limit checks remain.
The versioned v2 anchor may merge object-specific grasp points and placement-
center corrections into the frozen family templates without changing success
or safety thresholds. Validation, recording, and replay reapply the rotation-
only orientation hold at the same eight-simulation-step cadence.
The planner assigns whole train, validation, and test sessions and fixed reset
seeds. Test generation starts only after train/validation choices are frozen.
Held-out outcomes may be audited and reported but may not tune seeds, expert
parameters, normalization, safety thresholds, or checkpoint selection.
The automated verdict requires 42 accepted episodes across all 30 cells,
complete object-family/instance/target/session/source counts, recomputable phase
segments, split isolation, and append-only review evidence. Manual completion
then requires at least six visible replays spanning every object family and
target type; the checkpoint remains incomplete until the user confirms them.
```

Level 4.5B procedural expansion and scaling probe:

```python
assignments = build_level4_procedural_expansion_plan(
    "configs/level4_dataset.yaml"
)
report = run_level4_scaling_probe(
    config_path="configs/level4_dataset.yaml",
    dataset_dir="data/demos/level4",
)
coverage = summarize_level4_coverage(
    config_path="configs/level4_dataset.yaml",
    dataset_dir="data/demos/level4",
)["level4_5b_procedural_expansion"]
```

Rules:

```text
The active v19 plan contributes exactly 890 assignments to the 102 immutable nominal
anchors, producing 16 accepted episodes in each of 62 split-owned nominal
cells and 992 nominal episodes overall when complete.
Every assignment has a deterministic unique reset seed, initial-state digest,
continuous in-cell variation, and one append-only split-owned session. A failed
attempt remains rejected evidence and never satisfies an accepted slot.
The coverage audit verifies counts, seed and digest uniqueness, initial-state
and action-trajectory separation, storage projection, and manual-review state.
The scaling probe uses only accepted train/validation episodes. Its 4/8/16
training subsets are nested, use one model recipe and seed, normalize from the
active training subset, and replay the same fixed validation matrix.
Test generation is rejected unless the saved scaling report is sufficient and
explicitly records that it inspected no test episodes. If the user authorizes
diagnosis of an exposed test failure, that entire namespace must first be
quarantined from active coverage and model selection, and a successor plan must
freeze a new version, session prefix, episode prefix, and seed base before the
failure is inspected. Only an untouched successor namespace may provide active
test coverage; no exposed result may change thresholds, model selection, or the
planned data volume.
The automated verdict requires all 992 accepted nominal episodes and all
independence and scaling gates. Manual completion additionally requires visible
nominal and boundary replays for each of reach, complete pick/place, push, and
button press, confirmed by the user.
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

## Goal-Conditioned MLP Policy

Module:

```text
dexvision/learning/models.py
```

Contract:

```python
schema = PolicySchema(
    observation_schema_version=...,
    action_schema_version=...,
    observation_names=...,
    goal_names=...,
    dataset_action_names=...,
    output_action_names=...,
)
normalized_action = model(normalized_observations, normalized_goals)
# inputs: [batch, observation_dim], [batch, goal_dim]
# output: [batch, len(schema.output_action_names)]
```

Rules:

```text
The model contract records exact observation, goal, dataset-action, and output-action field names plus schema versions.
The full dataset action layout begins with named Level 1.13 base-position and quaternion-orientation fields and ends with one or more named finger-actuator fields.
The default output layout matches the complete dataset action layout.
An action ablation must declare a non-empty, order-preserving named subset; the model must not infer a subset from dimensions.
The model consumes normalized values and predicts normalized action values.
Saved checkpoints contain the model format version, schema contract, architecture config, and weights.
Loading reconstructs the model from saved schema/config metadata and requires an exact weight match.
Level 3.2 contains no optimizer, training loop, rollout, camera, or MuJoCo runtime.
```

---

## Behavior-Cloning Training

Module:

```text
dexvision/learning/train_bc.py
```

Contract:

```python
result = train_behavior_cloning(
    bundle,
    output_path=...,
    training_config=...,
    model_config=...,
    output_action_names=None,
    resume_from=None,
)
```

Rules:

```text
Training and validation consume the same training-only normalization metadata.
Validation runs without gradients and must not update model parameters or normalization.
The declared PolicySchema selects full-action or named-subset targets; training must not infer action offsets.
Level 3.3 uses deterministic CPU execution and an epoch-derived shuffle seed.
Epoch-boundary checkpoints save model and optimizer state, completed epoch count, and complete train/validation loss history.
Level 3.5B writes distinct last and best-validation checkpoints with SHA-256 sidecars.
Best-checkpoint selection uses offline validation loss only and breaks exact ties by earliest epoch.
Both checkpoint roles record the selected epoch/metric, full loss history, and the epoch represented by their saved state.
Checkpoints preserve the Level 3.2 model fields so GoalConditionedMLP.load can load trained weights directly.
Saved provenance includes config/model/schema versions, dataset and split-manifest digests, training-only normalization and its digest, seed, and environment metadata.
Every written checkpoint has a sibling SHA-256 digest file; resume verifies it when present.
Resume requires exact schema, architecture, dataset, split, normalization, and non-epoch training settings.
The preserved Level 3.3 configuration trains only reach_touch_target under the frozen reach split.
Level 3.5B uses the same dataset, schema, model, and training interfaces for reach_touch_target, button_press, and push_cube_to_target.
Each task's offline split comes only from its frozen evaluation protocol, and reserved rollout-only goals cannot enter training.
No camera, live teleoperation, or MuJoCo rollout belongs in this training module.
```

---

## Closed-Loop Policy Inference and Task Evaluation

Modules:

```text
dexvision/learning/policies.py
dexvision/evaluation/evaluate_policy.py
```

Contract:

```python
policy = load_checkpoint_policy(checkpoint_path, expected_dataset_digest=...)
protocol = load_reach_evaluation_protocol("configs/level3_evaluation.yaml")
report = evaluate_policy(policy, protocol, output_dir=...)
```

Rules:

```text
Policy loading requires and verifies the checkpoint SHA-256 sidecar.
The checkpoint model, exact named schemas, training-only normalization, and
dataset/normalization digests must agree before inference.
Inference consumes raw named observations and typed goals, normalizes them
with saved training statistics, and returns denormalized named actions.
An action-subset checkpoint requires an explicit ablation name; otherwise the
complete Level 1.13 base-position, quaternion-orientation, and finger layout is
required.
The reach scenario matrix comes only from level3/reach-evaluation-v1: three
training targets, two held-out targets, and seven fixed initial-base offsets.
Held-out scenarios are evaluation-only and must not tune the policy or protocol.
Base actions use the Level 1.13 base workspace; finger actions use the MuJoCo
actuator control ranges. Violating predictions are clipped before application
and terminate with an explicit recorded safety reason.
Every rollout, including every failure, saves observations, raw and applied
actions, distances, target/offset metadata, and its terminal reason.
Reports preserve checkpoint, dataset, protocol, and effective rollout-config
digests and separately report training-target and held-out-target success.
Normalized action jerk uses the complete applied action layout normalized by
its declared workspace, quaternion-component, and actuator ranges.
No camera, live teleoperation, button/push training, or held-out retuning
belongs in Level 3.4.
Level 3.5B adds button and push training/evaluation under their unchanged
Level 3.5A protocols. Their reports preserve offline selected epoch/loss and
dataset, split, training-config, schema, checkpoint, rollout-config, and
protocol digests. Every rollout failure remains saved and held-out outcomes
never select an epoch or model setting.
```

---

## Level 3 Data and Action-Space Diagnostics

Module:

```text
dexvision/evaluation/level3_diagnostics.py
```

Contract:

```python
config = load_level3_diagnostics_config("configs/level3_diagnostics.yaml")
report = run_level3_diagnostics(config)
```

Rules:

```text
Full, base-only, finger-only, and fixed-training-mean controls use the same
task-specific seed, hidden layers, frozen protocol, and scenario order.
Action subsets retain the complete named Level 1.13 dataset layout; omitted
fields hold their previous applied values during rollout.
Fixed-goal control retains the declared goal layout and model width but feeds
the training-mean goal, which is zero after training-only normalization.
A broader-data comparison may include recomputed-success episodes that failed
quality filtering, but shared clean episodes must retain their baseline split.
Do not create a broader-data run when no additional eligible episodes exist.
Summary tables include training/held-out success, final task error, normalized
jerk, and invalid/workspace/object-workspace/joint-limit violation counts.
Offline validation error is grouped by named action field and goal condition.
Reports preserve dataset, split, checkpoint, schema, training-config, and
protocol provenance and separate measured conclusions from hypotheses.
Raw demonstrations and prior Level 3 checkpoints/reports remain read-only.
```

---

## Level 4 Scripted Failures and Corrective Demonstrations

Modules:

```text
dexvision/logging/corrective_demos.py
dexvision/evaluation/correction_summary.py
dexvision/apps/generate_scripted_correction.py
```

Contract:

```python
plan = build_level4_6_plan("configs/level4_dataset.yaml")
episode_dir = generate_scripted_correction(
    source_rollout=failure_episode_dir,
    output_dir=output_dir,
    assignment=assignment,
    dataset_dir="data/demos/level4",
)
report = summarize_corrections(
    config_path="configs/level4_dataset.yaml",
    dataset_dir="data/demos/level4",
)
selected = select_level4_training_streams(
    episode_dirs,
    include_corrections=True,
)
```

Rules:

```text
The frozen plan contains exactly 12 cells and 10 active episodes per cell:
nine ordinary scripted failure cells and three corrective-intervention cells.
Every correction links to one ordinary failure, retains that episode as an
unchanged pre-intervention prefix, and stores failure class, retryability,
intervention interval, trigger source, original terminal result, and final
correction outcome as distinct fields.
Policy-triggered corrections require source_policy_checkpoint. Teleoperation
and scripted triggers require a null checkpoint plus the stable reason
trigger_not_policy_rollout.
Required corrections are simulator-state-grounded deterministic expert
continuations. They never require camera, hand-tracking, or human-control input.
Workspace and joint-limit violations preserve the unsafe request, record the
clipped/rejected applied action and safety reason, remain abort-only, and never
provide correction targets.
Each active cell has ten unique source episodes and stored initial-state
digests. Corrections may share a reset only with their explicitly linked
failure prefix.
The summary independently replays all corrections and requires both a pick and
a successful final settled placement. Saved success labels alone are not
qualification evidence.
Baseline and correction-trained selections use the same stable split ownership;
ordinary failures and policy rollouts are never silently promoted to expert
targets.
Superseded attempts remain immutable diagnostic evidence in the append-only
correction quarantine manifest and do not count toward active coverage.
Visible replay approvals are stored append-only in
`data/demos/level4/correction_manual_replay_manifest.json`; the correction
summary may report checkpoint completion only after one active correction has
an explicit passed review. The user accepted `level46_2091` on September 9,
2026, completing the Level 4.6 manual gate.
```

---

## Level 4 Rendered Visual Grounding Dataset

Modules:

```text
dexvision/perception/render_annotations.py
dexvision/logging/visual_stream.py
dexvision/apps/export_visual_dataset.py
```

Contract:

```python
report = export_visual_dataset(
    config_path="configs/level4_visual_dataset.yaml",
    dataset_dir="data/demos/level4",
    output_dir="data/visual/level4",
)
```

Rules:

```text
Level 4.7 derives RGB from exact named saved qpos/qvel and state timestamps;
rendering never integrates actions or rewrites source episodes.
The fixed level4/visual-camera-v1 calibration is explicit in the export config.
Every RGB frame links to its source episode/session/frame, action timestamp,
calibration, visible instance mask, boxes, stable ids, classes, and metric poses.
Invisible entities have zero visible pixels, null boxes, retained simulator
poses, and the shared occluded_or_out_of_frame status; no amodal mask is claimed.
The source task's static positions, object scale, and pick/place fixture
isolation are restored. Privileged cue geoms and label sites are hidden.
Whole-session and frozen cell ownership determine split assignment. Quarantined
or unaccepted episodes cannot enter the compact nominal scripted source set.
Held-out instances and the held-out right-bin region are hidden in non-test
renders, including background geometry. Source simulator truth stays intact.
Nominal, mild illumination, partial occlusion, and bounded distractor conditions
share one camera. Render-only geometry never changes the source trajectory.
Partial occlusion is bounded by measured goal-mask loss, with deterministic
card narrowing or explicit absence when the visible goal cannot support it.
All training conditions precede validation and test during export. Exact
cross-split pixel duplicates are explicitly excluded and reported; held-out
pixels cannot select training frames. Coverage counts require exported frames.
Coverage and storage reports expose shortages; no duplicated image repairs a
missing cell. Target coverage is split-permitted coverage, with all five targets
covered by the union of splits.
Source/image/config/implementation/asset digests and license attribution are
saved with the output. An existing output directory is never overwritten.
The custom workcell license remains unspecified and must be resolved before a
later release. This checkpoint neither packages data nor trains perception.
Manual review uses the 48 labeled panels across four condition sheets. Automated
checks never mark Level 4.7 complete without explicit user confirmation.
```

---

## Level 4 Dataset Audit and Frozen Splits

Modules:

```text
dexvision/evaluation/dataset_audit.py
dexvision/evaluation/split_audit.py
dexvision/apps/audit_level4_dataset.py
```

Contract:

```python
report = audit_level4_dataset(
    config_path="configs/level4_dataset.yaml",
    splits_path="configs/level4_splits.yaml",
    dataset_dir="data/demos/level4",
    output_dir="outputs/level4/audit",
)
```

Rules:

```text
The audit is read-only for nominal, failure, correction, visual and legacy data.
Existing output directories cannot be overwritten; repeat audits use new paths.
Active membership follows accepted scripted reviews and frozen quarantine rules.
An explicitly configured puck replacement amendment may supersede one preserved
training episode. Its versioned plan and receipt bind the original/replacement
file digests, frozen dataset config, new recording seed/session/operator, and
matching skill/cell/whole-session split. Missing, duplicate, changed or mismatched
evidence stops the audit. The original remains unchanged and its exclusion
retains its digest and replacement id. The replacement's manifest records
replaces_episode_id and still requires fresh replay, quality and leakage checks;
saved qualification receipts cannot bypass those gates. Historical collection
inventories retain originals; active coverage uses the amended membership.
The optional button_replacement_plan similarly binds exactly the eight
non-training button duplicates in the parent audit to new same-cell/same-split
episodes. The existing frozen procedural sampler supplies physical goal and
controller variation; the original dataset config and all task thresholds stay
unchanged. Each assignment records in a fresh process and session. A collision,
failed replay or insufficient trajectory-descriptor distance stops collection;
no test-outcome tuning or automatic retry is permitted. Recorder metadata uses
button_replacement provenance, not a fabricated Level 4.5B repetition. Replay
restores that amendment's saved task_config.procedural_variation. Originals,
reviews, visual sources and prior audit artifacts remain unchanged. Explicit
manifest supersession never bypasses fresh replay, coverage or split checks.
Fresh schema and headless replay results override stale nominal acceptance flags.
Correction prefixes, outcomes, source lineage and abort-only unsafe failures are
independently checked with the Level 4.6 summary.
Whole sessions, task-relevant held-out ids/regions and source lineage own one split.
Shared training-pool identities/goals across sessions are permitted explicitly.
Raw background object state remains archival truth and is excluded from learner
and normalization inputs; unrestricted raw-state learning is not isolation-safe.
Every rendered frame is decoded and checked for source/image hashes, alignment,
mask/box/visibility agreement, exact simulator poses/ids/classes and calibration.
Per-split manifests preserve relative-file SHA-256, named schemas, phase/skill
intervals, source lineage and baseline/correction target intervals.
Normalization inputs are explicit training expert episodes/frames only; no
normalization statistics, training, packaging or release is performed here.
A failing audit writes a versioned collection amendment and leaves Level 4.8
incomplete. It never edits accepted episodes or silently repairs shortages.
```

Level 4.8 completed on September 14, 2026. The final audit at
`outputs/level4/audit_v4_final` passes all 1,112 active episodes and 74 coverage
cells with zero issues. Its train/validation/test manifests are frozen and
contain explicit lineage for nine replacements; all originals remain preserved.
This qualified the dataset audit only. Release packaging had not started at
that point; its subsequent Level 4.9 completion is recorded below. Level 5 has
not started.


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
source_episode_id: required for corrective_intervention
trigger_source: required for corrective_intervention; identifies whether the
  triggering episode was teleoperation, scripted, or policy_rollout
source_policy_checkpoint: required only when trigger_source is policy_rollout;
  otherwise null with a stable not-applicable reason
object_instance_ids: concrete objects used in the episode
goal_condition_id: frozen coverage-matrix cell
camera_or_render_config: null only when no image stream is claimed
intervention_interval and failure_reason: required for corrective data
code/config/schema versions and random seed
```

Every control sample must preserve:

```text
requested_action and request source (operator, script, or policy)
commanded_action before safety handling
applied_action after clipping/rate limiting/rejection/safe fallback
prior commanded and applied actions
per-field safety mask and stable safety reason code
action frame, units, bounds, rate limits, and control interval
online phase and versioned phase-specific action relevance mask
resulting state timestamp
```

The complete named commanded/applied layouts remain present even when a phase
uses only a subset. Quaternions are normalized and sign-continuous with the
previous applied quaternion, with a frozen first-sample canonical-sign rule.
The action record must support deterministic derivation of bounded residual
targets. Unsafe commands may remain failure evidence but are never expert
applied-action targets.

Rules:

```text
Do not invent session ids for Level 2 legacy episodes.
Require at least four genuine Level 4 sessions: two training, one validation,
and one untouched test session. Keep every complete session in exactly one split.
Keep held-out object instances and goal regions out of training when the
evaluation protocol claims generalization to them.
Assign every coverage cell to a split and freeze a per-split accepted minimum;
global episode totals cannot compensate for a missing required cell.
Expert successes, ordinary failures, policy rollouts, and corrective
interventions must remain distinguishable.
Derive the online phase causally from current/prior world state and task
metrics; retain audited annotations separately and report disagreements.
Synchronized image/state/action streams must have validated alignment.
The fixed-camera visual matrix covers nominal, mild-illumination,
partial-occlusion, and bounded-distractor conditions without implying
cross-camera or real-world transfer.
Every immutable release needs a manifest, checksum, schema versions, split
manifests, coverage report, and retrieval instructions.
```

---

## World State and Object Observation

Modules:

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

@dataclass(frozen=True)
class WorldState:
    schema_version: str
    timestamp: float
    frame: str
    entities: tuple[ObjectObservation, ...]
    relations: tuple[EntityRelation, ...]
    fixtures: tuple[FixtureObservation, ...]
    robot: RobotObservation
    contacts: tuple[tuple[str, str], ...]
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
Level 4.1 simulator truth uses source=simulator_ground_truth and confidence=1.
Future inferred observations use source=inferred_perception but the same frozen
ObjectObservation type and validation rules.
Workcell task factories validate goals against configs/level4_dataset.yaml and
compute the frozen reach, pick, place-held, push, and press terminal metrics.
The operator-facing return-bin mapping is return_bin_left=[0.10, 0.11, 0.02] m
and return_bin_right=[0.10, -0.11, 0.02] m; scene labels and task ids must not
swap these sides.
Consecutive qualifying dwell is task-local and causal; a nonqualifying sample
resets it to zero.
The optional rotate_dial task remains disabled.
```

---

## Level 4 Immutable Release Packaging and Verification

Modules: `dexvision/logging/dataset_release.py`,
`dexvision/apps/prepare_dataset_release.py`, and
`dexvision/apps/verify_dataset_release.py`.

The packager uses the passing frozen audit as the payload allowlist. It includes
all active episode streams, the complete visual export, frozen splits, exact
named schemas, audit/source/config digests, replacement recording snapshots,
licenses and the handoff. It never scans excluded working episodes into targets,
rewrites raw data, overwrites an archive, or modifies the Level 2 release.

Archive metadata and gzip timestamps are deterministic. The verifier checks
metadata checksums, audit/split binding, required episode/visual files, the
archive digest, and the exact member set with streaming per-file size/hash
verification. Absolute/traversal/Windows-drive paths, symlinks, hard links,
special files, sparse entries and duplicate members are rejected. Restoring
requires a nonexistent directory; failed partial output is retained and cannot
be silently reused. No GUI, GPU or camera is involved.

Optional Git LFS retrieval creates an empty repository and isolated LFS cache,
requests the exact manifest object id and verifies its size/hash before restore.
A returned LFS pointer or failed download is an error. The caller must obtain
trusted metadata from the project; checksums alone do not authenticate an author.

On September 14, 2026, the owner selected Git LFS and approved Apache-2.0 for
included project workcell/source/configuration and CC BY 4.0 for generated data.
The notices in `docs/licenses/level4-v1/` apply prospectively without rewriting
historical export/audit reports. Third-party Shadow Hand notices remain intact.
The release config documents host limits but does not invent an account-wide
remaining quota. A build-time candidate manifest never constitutes user manual
acceptance. Later verification receipts and status docs record subsequent events.

The Level 5 handoff is data-interface documentation only. Frozen split rows and
`schemas.json` define exact layouts; whole-session ownership, task-relevant
object selection, and training-only normalization inputs remain mandatory.
No learning or policy qualification is implemented by checkpoint 4.9.


Level 4.9 completion: source and metadata are published, both dataset releases
passed clean-clone retrieval and readback, and the owner accepted delegated
verification. The verifier resolves historical candidate status using a new
checksum-bound `completion_receipt.json`; it validates the archive/manifest
binding, verified clone commit, legacy retrieval, automated checks and owner
acceptance fields. The archive and original payload manifest remain unchanged.
`--require-ready` passes only with that evidence. Level 5 has not started.

---

## Level 5.2 State-Grounded Reach Evaluation

Modules: `dexvision/evaluation/evaluate_skill.py` and
`dexvision/apps/evaluate_skill.py`.

The evaluator accepts only the frozen `reach_object` state track. Each candidate
uses the complete validation matrix before `select_checkpoint` applies the
existing ranking. Test requires `selected.pt`; an exclusive run-local reservation
prevents repeating the single test evaluation under another output path.

Reach starts at frame zero. Reset metadata comes from checksum-verified release
bytes; no test action arrays or expert continuation are read. Live 192-value
features match the causal training layout, including measured free-joint pose,
named hand state, prior actions/safety, physical phase, selected entity, and
numeric goal. Baseline actions decode from the previous applied action. The
frozen absolute-action reference uses identical data, inputs, seeds, sampling,
selection and reset matrices, with its declared reference head/loss.

Every physical reset executes independently under each of the four condition
labels. Render-only conditions have no effect on state inputs or physics; this
track makes no RGB/perception claim. Every attempted rollout retains the initial
state, requested/commanded/applied action history, safety masks/reasons, named
observations, state/action timestamps, full qpos/qvel and typed world states,
terminal metrics and reason. Rejections remain in the denominator with an
unavailable terminal error. Raw invalid/workspace/actuator-limit predictions
remain failures even after a finite fallback or clipping. Wrong-entity contact
and objects leaving the board also fail explicitly.

The existing frozen WorldState reach pose/dwell metric uses the mocap base
target; the learner's proprioception uses measured free-joint pose. Physical
object displacement supplies the scene-disturbance metric. Actuator target-limit
counts follow Level 3; they are not a claim of zero soft-joint qpos overshoot.
These semantics remain visible in the comparison report.

Reports retain per-seed condition/cell/family/instance/target/session/boundary
breakdowns, full input/checkpoint and execution-code digests, training curves,
and historical Level 3 safety categories. The pinned Level 3 checkpoint's
contact-only xyz goal and legacy observation contract are incompatible with
Level 4 approach-pose goals; no synthetic adapter or paired transfer claim is
made. State-only numerical success remains experimental pending the required
manual and later perception gates. Manual review replays saved states and never
reruns held-out physics or records owner acceptance automatically. No future
skill, perception pipeline, runtime registry, correction training or executor
is implemented here.

---

## Level 5.2A Reach Training Repair

Modules: `dexvision/learning/reach_repair.py`,
`dexvision/evaluation/reach_repair.py`,
`dexvision/evaluation/reach_repair_review.py`, and
`dexvision/apps/repair_reach.py`. Method evidence and limits are documented in
`docs/level5_reach_repair.md`.

Each repair recipe has its own version and checksum. Historical v1 Level 5
configs, datasets, labels, models and evaluation artifacts remain unchanged.
The cache contains only checksum-verified train/validation slices with frozen
whole-episode/cell ownership. Physical feature scales are fixed by the recipe;
validation and confirmation cannot fit them. Training runs through the entire
preregistered budget and retains early, late and best-offline candidates.
Selection requires full closed-loop validation evidence bound to each candidate.

The V3 reach representation is four physically scaled values: planar goal
distance, vertical goal error, radial tracking lag and vertical tracking lag.
Measured forearm pose is converted through the model's frozen weld transform
into the same body origin as the action command before computing tracking lag.
This is a rigid coordinate conversion, not a fitted bias correction. The model
XML and recipe transform must agree. The historical 192-field source schema
remains readable and unchanged.

The MLP learns planar progress fraction and vertical displacement. The decoder
constrains planar movement to nonnegative progress toward the goal without
overshoot. Training proves this constraint and the zero orientation/joint
deltas before enabling them. The complete 27-field commanded/applied record
and original safety checks remain mandatory. Zero model output produces zero
movement. No expert or saved continuation is consulted during inference.
This reach-specific structure does not authorize fixed joints, fixed
orientation, or straight planar paths for contact or obstacle-avoidance skills.

The selected method binds a previously frozen fresh-reset manifest. Confirmation
requires all three full-validation selections to pass and verifies their model
weights against the evaluated candidates. An exclusive per-run reservation
prevents repeating confirmation under a new output path. Fresh seeds do not
make previously evaluated task families unseen; that limitation is explicit.
All failed attempts remain in metrics and artifacts.

Saved-state audit independently computes physical pose and dwell by converting
the measured forearm pose into the command frame. It does not integrate physics
or treat commanded pose as measured pose. Manual review replays saved qpos/qvel
with the corresponding commands, shows each family/boundary stratum, and leaves
acceptance pending owner confirmation. Neither the audit nor opening the viewer
qualifies a policy. Later perception qualification remains a separate checkpoint.


V5 is the final automatically verified repair recipe. It adds previous applied
command height as a fifth input, keeping the calibrated goal-plane decoder.
Its `SkillMLP` is 5 -> 128/ReLU -> 128/ReLU -> bounded output heads.
The existing heads retain widths 3/3/2/18 (26 outputs); only coordinates 0 and 1
are supervised and decoded as planar fraction and normalized vertical delta.
The other 24 coordinates are masked and ignored, not learned joint control.
Masked MSE fits scripted expert actions; the decoder supplies planar direction,
nonnegative progress and no planar overshoot. Inference reads current simulator
state on every control sample; it uses no RGB, online weight updates, expert
action query or saved trajectory continuation. The full training recipe and
input/output units are in **Final V5 learning method** in the method report.
All three seeds run 1,000 epochs; closed-loop validation selects epochs 300,
207 and 300. Each passes 120/120 validation and 240/240 fresh confirmation
attempts with zero safety violations; all saved confirmation trajectories also
pass physical pose/dwell audit. The owner accepted the review and completion on
September 17, 2026; `docs/level5_reach_acceptance.json` binds the acceptance to
the selected model digests. The bounded state reach gate is complete; this
does not establish perception qualification or default registry admission. Later skills
must follow the per-skill audit and selection decisions in
`docs/level5_reach_repair.md`; the reach-specific posture and planar constraints
must not be copied into contact skills without independent evidence.

---

## Prospective Level 5.3+ Policy Bundle Contract

Planning only; no future loader, skill or executor is implemented by this
section. `docs/level5_training_guidance.md` is the method amendment for unstarted
skills. Historical Level 3 and frozen Level 5.0 schemas remain readable and
unchanged; they do not mandate the same input/output representation for a new
versioned experiment.

A future skill recipe must explicitly bind the source schema, causal feature
transform, body-origin calibration, physical scales or training-only statistics
with justified floors, named active output targets, invariant channels, action
decoder, phase/history rules and control interval. The selected weights alone
are not a complete deployable policy. All components and their code/config
digests travel together through validation, perception, registry and release.

Train/live feature parity, target decoding, zero-output behaviour and raw-action
safety checks need tests. Do not infer a channel is safely invariant from its
being absent in one phase; verify training variation and whole-skill semantics.
Perception may replace the declared entity observation source, but cannot
secretly use truth in derived goals or phases. Terminal success needs measured
physical evidence, not only commanded pose. New fields/contracts require a
versioned recipe and executable support in the appropriate future checkpoint.

Keep the original safety/qualification gates, split isolation and per-seed
reporting. Freeze a training budget and candidate schedule, then select by full
closed-loop validation before one-time test/confirmation. Offline-loss patience
from the historical v1 trainer is not the default for future skill recipes.

---

## Level 5.3 State-Grounded Pick Policy

Modules: `dexvision/learning/pick_policy.py`,
`dexvision/learning/phase_losses.py`, and
`dexvision/evaluation/pick_policy.py`. The existing training/evaluation apps
dispatch only explicitly versioned pick recipes to these adapters. Method,
negative attempts and exact commands are in `docs/level5_pick_results.md`.

The pick bundle binds its selected weights to 121 causal features, the verified
forearm-to-command body-origin transform, fixed physical scales, entry object
height/orientation, seventeen active outputs, the bounded delta decoder and the
original control cadence. Current contact-derived retention and measured height
define acquire/lift/stabilize; saved sequence phases, expert stage and future
success do not enter policy inputs. Varying closure and orientation channels
remain active. Nine zero-delta channels are held only after a whole-interval
training-data invariant check. Zero output holds every command.

A checksum-bound cache preserves whole episode/session/cell ownership and the
exact frozen pick slices. Source/prior actions must pass a preflight against
the recipe's safety bounds. Full-budget training retains early/late/best-offline
candidates; all receive the full validation matrix for each of three seeds.
One-time test evaluation requires weights identical to the selected candidate
and an exclusive run-local reservation. Failed and interrupted attempts remain
available; no test action can become a learned target or a policy continuation.

V1 preserves the diagnosed metadata conflict. V2 explicitly uses the immutable
Level 4 recorder and task runtime's workspace (`workcell.safe_workspace`, z min
0.01 m), verified against the frozen config, instead of the generic action
layout's stale z min 0.08 m. Historical layouts/config locks are unchanged.
Actuator limits, rate caps, zero-safety and task/jerk thresholds are unchanged;
jerk keeps its original frozen normalization ranges. This is a versioned binding
to the original collection-time workspace, not a new wider runtime envelope.

The standalone evaluator replays only the prefix before the exact frozen entry,
then resets dwell and rejects unsupported, held, distant or stale entries
without another start search. Scripted setup is excluded from policy metrics.
Physical success requires correct object contact/retention, measured lift,
velocity and unsupported stable dwell, plus the original frozen pick metric.
Every attempted rollout retains raw/applied actions, safety reasons, physical
states and acquisition/lift/drop/hold results. Background objects intentionally
parked off the board are distinguished from the selected object. This establishes
neither learned reach-to-pick composition nor perception qualification. Any
triggered viewer acceptance remains pending until owner confirmation.

---

## Level 5.3A Pick Repair Amendment

`configs/level5/pick_policy_v3.yaml` freezes a training-proven rank-one closure
representation. The 26-value MLP interface remains, but output 8 means a signed
closure fraction increment divided by 0.05; other finger outputs are masked and
ignored. The decoder supplies the common direction between audited open/closed
postures and bounds aperture to [0,1]. Targets must reconstruct this direction
exactly; incompatible prior commands are rejected. This prevents independent
finger drift while retaining learned closure, lift and three orientation outputs.
It does not claim independently learned full-hand control or vision.

Source demonstrations used object-orientation overwrites during lifting.
The repair preserves that source provenance but prohibits those overwrites in
policy evaluation. A saved-action validation ablation is diagnostic, never a
policy or confirmation controller. The original 5.3 report remains historical.

The repair app freezes fresh procedural resets before validation selection,
using only pre-acquisition source prefixes. All entries remain in the evaluation
denominator, including rejected preconditions. It verifies passing full-matrix
validation for all three seeds and exact selected weights before reserving a
single fresh confirmation per run. Exposed original tests are forbidden for
repair recipes. Fresh seeds do not make exposed families or instances unseen.
Saved-state audit reconstructs contacts, measured lift and stable dwell without
physics integration. Review replays saved physical states, and leaves visual
acceptance to the owner. No later skill is authorized by this amendment.

V4 adds the independently audited compact-geometry variant in
`dexvision/learning/pick_geometry.py`. Its 28 inputs use prior-command hand-frame
relative geometry and causal entry context; the latter adds entry command
height to object height/orientation. Rotation outputs are learned hand-frame
increments converted back to world. Puck rotation stays fixed only after a
whole-training invariant check. The decoder prevents lift before full closure,
keeps closure/lift monotone, and bounds lift by demonstrated family travel.
At saturated closure/lift bounds the corresponding gains are unidentifiable and
masked; all recorded actions must still reconstruct exactly. V3 artifacts retain
their original semantics and failed confirmation. V4 receives its own fresh
reset manifest and cannot rerun either exposed confirmation as untouched evidence.


The separate `pick_setup_v3` protocol corrects fresh-reset approach setup while
locking all three V4 selected weight hashes. The shared
`dexvision/sim/scripted_expert_settings.py` resolver preserves the collection
app's exact settings. The frozen collection expert supplies only open-hand
approach commands, stopping before acquire. Its copied-state grasp-template
preflight retains historical orientation assistance on the scratch copy only;
live setup and policy rollout never overwrite object poses. This limits any
qualification to scripted family-aware approach starts. The earlier literal
source-prefix failures remain preserved and are not relabeled as successes.

Fresh validation and confirmation matrices are frozen together before evaluation.
No reset filtering, replacement, retraining or reselection is allowed. Every seed
must pass full validation before one reserved confirmation. The guarded setup
viewer additionally requires all three passing confirmations and matching saved
physical audits; it replays saved states and includes recorded failure reasons.
Only the owner can supply manual acceptance. Original families are exposed, and
no perception, arbitrary-start, skill-composition or deployment claim follows.


The owner-authorized teacher amendment adds an explicit ordinary-physics grasp
mode while preserving historical defaults. `pick_teacher_v2` tests each original
fallback using the identical controller loop and initial state used live; no
object-orientation writes occur in either. Its 24/24 development result and the
unchanged V4 policy's 24/24 diagnostic result motivate `pick_setup_v4`: fixed
weights, a checksum-bound teacher recipe, and separately frozen fresh validation
and confirmation. This version replaces scratch orientation assistance with
ordinary-physics planning but still uses full-state scripted grasp selection
before entry. No learned-policy command is queried during setup; no expert
continuation is used after entry. The setup viewer covers nominal/boundary cases
for every object instance and requires all three numerical and physical audits.
The approved additive data route is versioned separately below.


The additive pick release adapter (`learning/pick_data.py`) verifies compressed
archive and per-member hashes, preserves original Level 4 ownership documents,
and declares its separate train/validation episode, session and seed ownership.
All six instances are now exposed; fresh confirmation establishes only new-reset
performance. A source scene named pick/place does not imply any completed place
trajectory. Raw working episodes and old immutable releases remain unchanged.

V5 introduces 29 inputs: V4 geometry plus a causal consecutive full-closure/held
counter scaled by ten. Three learned scalar outputs control closure increment,
lift speed and orientation-correction magnitude. The explicit direction decoder
uses current object versus entry orientation (cuboid full rotation, cylinder axis,
puck fixed), bounds remaining error and forbids reverse correction. Source-action
reconstruction and offline/online history parity are required. This is constrained
state behavior cloning, with engineered directions and bounds, not learned vision.

V5 was stopped before training because the independent V1 source audit found
191/192 physically passing. MuJoCo's cached derived state after integration gave
one extra apparent stable sample. Teacher V4 and the prospective V6 policy instead
use `physical_evidence_world` for terminal evidence: copy MuJoCo data, recompute
contacts/pose/velocity, and read the resulting world without changing live data,
warm starts, controller inputs or commands. Original recipes retain their cached
state semantics. No speed, contact, dwell or safety threshold changes. Data V2
re-records the same cases in a new immutable archive, preserving V1 and its failure.
The repair viewer requires every seed's passing numerical result and matching
complete saved-state audit, covers each instance/boundary and failure reason, and
leaves acceptance pending owner confirmation.


V7 separately freezes binary close/lift gates after the training audit proves
all active targets are 0 or 1 (closure is always 1). The MLP still emits bounded
scores, but values above the configured 0.5 midpoint enable the demonstrated
maximum closing/lift rate. Rate, direction, aperture/travel bounds and the
full-closure interlock are engineered; zero output holds. Orientation magnitude
remains continuously learned. Cache construction rejects nonbinary active source
targets instead of silently quantizing data. The schema names gate scores
explicitly. Fixed-weight validation diagnostics compare timing with source data;
they do not qualify models or consume confirmation. V6 failures remain immutable,
and V7 must complete its own budget, full candidate selection and fresh reset
confirmation before any owner review.


Level 5.3A V7 completion: all three selected policies pass the original bounded
state pick gates (212/216 fresh confirmation successes each, zero safety
violations), with 648/648 physical-audit agreements. Owner manual acceptance was
confirmed October 3, 2026 and is bound to model/recipe/report hashes in
`docs/level5_pick_acceptance.json`. The six instances are training-exposed and
scripted family-aware approach/preflight remains required. This is not vision,
arbitrary-start, composition, default registry or deployment qualification.
Level 5.4 has not started. Generated weights and run evidence remain saved
locally in Git-ignored outputs; the completion commit contains source, recipes,
documentation and additive LFS dataset releases, not a published model release.

---

## Level 5.4 State-Grounded Place-Held-Object Policy

Modules: `dexvision/learning/place_policy.py` and
`dexvision/evaluation/place_policy.py`. The CLI training/evaluation apps dispatch
only the independently locked `level5/place-policy-v1` recipe to these modules.
The historical Level 5.0 place config, Level 4 dataset and split ownership are
unchanged. Exact method and negative results are in `docs/level5_place_results.md`.

The bundle binds selected weights to a causal held-object/target transform,
verified forearm-to-command weld calibration, fixed physical scales, nineteen
active output deltas, nine whole-training invariant channels, the original
control cadence and the Level 4 collection runtime workspace. Finger-release
channels stay active. A zero network output holds the full prior command; raw
actions still pass the frozen rate, workspace and actuator safety checks.
The short place/release intervals use a declared cell/episode/causal-phase/frame
sampler without admitting validation/test targets to training. All training
seeds run the complete budget; every retained candidate receives the full
closed-loop validation matrix before one selected held-out evaluation.

The evaluator replays the checksum-verified scripted prefix through the exact
place boundary, including the source's declared rotation-only orientation hold
before entry. It rejects missing, wrong, multiple, supported and stale held
objects. After entry, policy control uses ordinary physics with no object pose
writes or source-action continuation. Physical outcome evidence is recomputed
post-integration on a copied MuJoCo state: retention, release distance,
support, linear/angular speed, bounce and ten-sample settling must agree with
the terminal result. All attempts, including precondition and safety failures,
remain in the denominator and saved trajectories.

V1 state qualification is negative: selected epochs 20/56/20 yield 4/216,
0/216 and 12/216 held-out successes, with premature release and safety failures.
The conditional viewer review is not triggered. The saved-state audit verifies
the physical basis of the sixteen reported successes, not overall
qualification. There is no perception, default registry, composition or
deployment admission, and no Level 5.5 implementation follows from this result.

---

## Level 5.4A Place Repair Source Feasibility and Policy Qualification (Complete)

V25 passed the frozen state gates, independent physical audits and owner
manual acceptance on October 5, 2026. The acceptance receipt is
`docs/level5_place_acceptance.json`; the accepted runtime is described at
the end of this section. Pending and failed statuses in the history below
refer to earlier versions and do not describe the accepted V25 bundle.

### Preserved source and policy repair history

`dexvision/evaluation/place_repair.py` and
`dexvision/apps/qualify_place_teacher.py` run checksum-locked scripted-teacher
development screens. They do not load, train, select or qualify a learned place
policy. The completed V1 policy modules, frozen Level 4 releases, original
terminal/safety gates and negative test artifacts remain unchanged.

An exact ordinary-physics pick teacher establishes the verified held-object
precondition on a fresh deterministic task reset. The place teacher then
uses ordinary physics with measured object-to-target correction inside the
frozen 0.025 m release gate, source-proven coupled finger opening, and a
causal hand-pose latch during release and settling. Setup and place failures
remain counted, with per-attempt actions and physical qpos/qvel saved. Every
recipe and development reset matrix is versioned and independently locked.

The V2 teacher passes its original exposed slot-B development screen (29/30,
zero safety), but the separately frozen all-cell fresh-seed screen fails
(15/30, including table contact, physical drops and object workspace exits).
The next frozen V3 source passed 30/30 on the exposed all-cell development
matrix, then failed a newly seeded all-cell screen at 22/30, with two setup
board-margin violations. Its bounded grip/yaw setup and every physical state
are retained as negative source evidence. The V3 result is development-only;
the source remains unqualified for a dataset release.
Frozen V4 passed 29/30 exposed `830260900` resets, then failed 26/30 fresh
`840260900` resets. An exact-reset copied-task candidate search, binding the
unchanged V4 controller and frozen candidate geometry, is under a separate
fresh development screen. The frozen search passed **29/30**, with 9/10
worst-family success and zero safety violations; one premature release remains
failed. Every preview and selected live continuation is retained. This clears
scripted-source feasibility only. Repair evidence and the exact commands are in
`docs/level5_place_repair.md`. A future policy still requires its own frozen
training, full validation selection, fresh one-time confirmation, saved-state
audit and owner viewer acceptance before checkpoint completion.

The first frozen additive data plan exposed a no-success fallback flaw at
`860260906`: the V1 planner ran a board-violating candidate despite safer
failed candidates. Collection stopped; no release was made. Search V2 changes
only that fallback. Its separately frozen 60-reset nominal/boundary screen
passed 59/60 with 19/20 worst-family success and zero selected live safety.
Additive data V2 has new frozen assignments and passed 140/144 training and
47/48 validation resets with zero live safety violations. Its independent
saved-state audit passed all 187 successful trajectories; five failed attempts
remain counted and excluded from supervision. The checksum-locked, LFS-tracked
`datasets/level5-place-unassisted-v2.tar.gz` archive contains only canonical
causal place payloads. The unassisted scripted pick/setup prefix is replayed
exactly to establish the held-object entry, then excluded from place labels.
The 195-field pre-action source includes causal task phase but never the
expert's future phase. The additive release does not qualify a learned policy;
frozen recipe, three-seed training, full validation selection, fresh one-time
confirmation, policy physical audit and conditional owner viewer acceptance
remain required.

The V6 learned repair is separate from the scripted teacher. Its causal
195-field state observation feeds an MLP trained on the 140 successful V2
training episodes, with 47 successful episodes reserved for offline loss.
The learned outputs are move, orientation and release scores. The decoder
supplies measured-object waypoint direction, the verified 0.005 m transport
and 0.001 m descent bounds, training-only family release offsets, a forced
waypoint while held outside the release region, a 0.0245 m cuboid trigger
inside the unchanged 0.025 m physical limit, coupled finger opening and
the fixed safety checks. It is neither learned vision nor unconstrained
full-hand control. V5's slot-B validation selections passed numerically but
its separate all-target development screen failed zero safety; its reserved
confirmation was never run. V6 passed separate 27/30 nominal and 27/30
boundary all-target development screens with zero safety. Full V6 validation
selected candidates, but a separate all-target development screen exposed a
seed-5002 safety failure. The frozen V7 all-target selector likewise rejected
both validation-eligible seed-5002 candidates, so no three-seed V7 bundle or
confirmation was produced.

V8 retains the exact fully trained V6 weights and learned outputs. Its
checksum-locked decoder overlay pauses transport when the commanded hand and
measured held object separate beyond the declared bound, uses a smaller
nonplanar cuboid opening, and permits a geometric release for an
outward-moving cuboid near a receptacle edge within the original 0.025 m
release tolerance. These are engineered state-feedback and safety rules, not
learned vision or new learned action dimensions. Each frozen V6 candidate
receives full V8 validation. The V8 selection protocol screens only candidates
passing all original state gates on a distinct fresh all-target reset matrix;
selection does not consume the separate one-time confirmation. V8 selected
seed 5001 and 5002 bundles but rejected both seed-5003 candidates on settling
or object-workspace safety, so it never ran confirmation. V9 locks smaller
coupled finger-opening steps for nonplanar cuboids, cylinders and pucks on the
same checksum-locked V8 control path and exact V6 learned weights. It freezes
candidate epochs before full V9 validation and a distinct fresh all-target
selection screen. The exposed V8 starts are development evidence only.
V9 passed full validation, but its fresh all-target screen rejected seeds
5002 and 5003 on the unchanged settling gate. No V9 three-seed bundle or
confirmation exists. Independent physical reconstruction and owner viewer
acceptance remain required before 5.4A can be marked complete.

V10 added only an engineered 0–3 mm upward release-hand clearance keyed to
object family and target type at first release latch. It preserved exact V6
weights and every frozen gate. The complete exposed 1000/1020 screens passed
all three seeds, but full validation rejected seed 5002 on 36/64 cuboid
successes. Its fresh selection and confirmation were not used. V11 leaves
the hand at its original release height for cuboids over either placement
slot, while retaining V10 clearance elsewhere. This zero-slot-rise rule was
frozen from the V10 bounce diagnosis. All three candidates passed complete
V11 validation, but the fresh 1060 selection rejected seeds 5002 and 5003
on the unchanged settling gate. The original 930 confirmation, saved-state
confirmation audit and owner viewer acceptance remain unstarted. V11 does
not claim learned vision, learned release geometry or qualification.

V12 retains those exact V6 learned weights and V11 safety/clearance controls.
Its checksum-locked decoder holds the first computed descent hand waypoint
only for a heavy puck sent to the inspection pad when the pre-action entry
command/object vertical offset is below 0.053 m. It also chooses between two
bounded rank-one finger-opening steps for cuboids sent to the left receptacle
from the current command/object vertical gap and measured downward velocity.
Both controls are engineered state-feedback rules; the learned move,
orientation and release scores remain active, and no new learned action
dimension or vision claim is introduced. The original 930-series
confirmation and every frozen 5.4 terminal/safety gate remain unchanged.
The independent saved-state audit rejected a V12 nominal success after 15
hand-table-contact frames before first opening. Partial V12 validation was
stopped; 1080 selection and 930 confirmation were unused. V13 preserves the
same weights, V12 decoder and all frozen gates. Only for the measured
short-grasp heavy-puck inspection-pad negative lateral drift state, while
held and descending, it permits coupled opening at 0.024 m target distance,
inside the original 0.025 m release limit. This is engineered physical
feedback; it does not add learned outputs or a vision claim. Complete V13
validation, separate fresh 1100 all-target selection, one-time 930
confirmation, independent physical audit and owner viewer acceptance are
required before 5.4A can be marked complete.

V13 completed all three 192-rollout validation matrices and passed each
frozen state gate. Its new 1100 all-target selection rejected seeds 5001
and 5002 for one object-workspace violation each and seed 5003 for 51/59
settled nominal placements. No V13 selected bundle or 930 confirmation
exists; 1100 is now exposed development evidence. Subsequent repair must
freeze another distinct selection reset set and preserve the V13 failure.

V14 adds only an engineered conditional wrist hold after the verified setup.
For a large block over the left bin, it holds the wrist when the entry
quaternion scalar exceeds 0.82. For a short cylinder over slot B, it holds
when the first learned quaternion command decreases that scalar. For a heavy
puck over slot A, it holds on that first-command direction or an entry
command/object vertical offset below 0.053 m. Other learned orientation,
release and motion outputs remain active. V6 weights, V13 physical release
guard, all original terminal/safety limits and the one-time 930 confirmation
are unchanged. Exposed 1100 targeted results were development evidence only.
Complete V14 validation passed all three seeds, but the fresh 1120 all-target
selection rejected seed 5002 for one object-workspace violation and seed 5003
for 51/59 settled nominal placements. No V14 three-seed bundle or 930
confirmation exists. Subsequent repair must use a distinct unused selection
set and pass every remaining 5.4A qualification gate.

V15 retains the V6 model and V14 decoder. It selects the existing 0.00625
or 0.0125 coupled opening step for short cylinders and heavy pucks from
pre-action target, wrist scalar, command/object grasp offset and object
vertical speed. For a held large block rising near the left-bin margin, it
commands at most 2 mm inboard and 3 mm upward rather than continuing descent.
These are engineered state-feedback controls with no new learned outputs.
Changed attempts on exposed 1100 and 1120 matrices passed saved-state audits;
merging with unchanged attempts passes the original numerical gates. V15
1140 selection and the original 930 confirmation were unused when V15 froze.
Complete validation, the fresh 1140 all-target selection and the one-time 930
confirmation passed all original numerical state gates for all three seeds,
but independent saved-state audit rejected all three selected bundles. The
audit reconstructed pre-opening hand-table contact on 8, 2 and 7 distinct
reported successes, respectively. The 930 matrix is consumed and exposed;
V15 has no physical qualification or owner viewer acceptance. Subsequent
repair must preserve those reports, freeze distinct unused selection and
confirmation matrices, and pass the independent physical audit before viewer
review. Exploratory source-grounded wrist/height controls on the exposed 930
resets remain development evidence and are not a selected recipe.

V16 retains the exact V6 learned weights and overlays measured-object wrist
feedback and bounded command-height guards. All three 192-rollout validation
matrices passed numerical gates, but the fresh 1160 all-cell selection rejected
each seed: seed 5001 missed settling and one saved-state physical check; seeds
5002 and 5003 had one and three saved-state contact failures. No V16 bundle or
1180 confirmation exists. V3 additive source covers all 30 cells in both
splits and passed 354/354 independent successful-trajectory audits; it does
not itself qualify a learned policy. V17 holds the setup wrist for round
objects, adjusts large-block pad clearance by 0.5 mm, and enables existing
object-orientation feedback for two seed-5003 cells. It reuses the V2-trained
V6 weights, so V3 is engineering evidence rather than new supervision for
those weights. Exposed 1160 targeted replays are diagnostic only. V17 freezes
distinct unused 1260 selection and 1280 confirmation sets, with unchanged
numerical gates, full saved-state audit and conditional owner viewer review.
Full V17 validation and fresh 1260 selection passed all state and saved-state
selection gates, but the one-time 1280 confirmation failed seed 5003's 90%
settling gate (192/220 nominal placements settled). The 720/720 independent
saved-state audit was physically consistent but did not override that numerical
failure. V18 retains the same learned weights and freezes a seed-5003
large-block opening fallback at the original 0.0245 m cuboid release bound
for pad/slot targets. It responds to the exposed 1280 learned-score failure;
those diagnostic resets cannot qualify V18. Distinct unused 1300 selection
and 1320 confirmation resets are frozen, with the same automated and manual
gates. V17 and its 1280 confirmation remain negative evidence.
V18 complete validation passed all three seeds, but its fresh 1300 selection
rejected all three on the 90% settling gate; seed 5001 had one additional
saved-state physical failure. The 1320 confirmation is unused. This prevents
qualification and viewer review, and suggests the limited V2 training
coverage needs separate investigation before further all-cell qualification.

V19 trains new causal move, orientation and release scores on the audited V3
release: 237 successful training and 117 source-validation episodes, with all
30 object/target cells represented in each split. The V6 checkpoint is loaded
only to establish the pinned V18 decoder controls; V19 weights replace the
model before action inference. The explicit geometric waypoint, fixed joints,
family release bounds, physical scales, safety checks and 5.4 gates remain
engineered. Expert phase is a training label, never a runtime input. Three
200-epoch runs and all nine 1330 full closed-loop validation matrices finished;
epochs 20, 20 and 200 passed validation. Fresh 1340 selection then rejected
every seed on settling (50/59, 51/58, 50/58). Saved-state audit also found
one reported-success pre-opening hand-table contact each for seeds 5001 and
5003. The 1340 set is exposed, 1360 one-time confirmation unused, and no V19
bundle or owner viewer acceptance exists. Exposed supported-opening probes
regressed full-matrix settling and do not change this frozen result.

V20 binds the exact V19 learned move, orientation and release scores to an
object-calibrated decoder. The six target-minus-object release waypoint
vectors are medians from audited V3 **training** episodes only, recomputed
and checked against the frozen recipe at load time. The decoder also descends
at most 1 mm per action while an opening cylinder remains held and more than
8 mm above target, and keeps the small-block slot-B hand command at or above
0.073 m. All other V18/V19 controls and original 5.4 terminal and safety
limits remain. Exposed 1340 development passed numerical and saved-state
physical gates for three fixed V19 epochs, but this is not qualification.
The exposed 1330 validation matrix must be fully rerun for V20. Distinct
fresh 1380 selection and one-time 1400 confirmation, independent physical
audit and owner viewer acceptance still gate 5.4A completion.
Complete V20 validation passed all three state matrices, but fresh 1380
selection rejected seed 5002 for two reported-success pre-opening
hand-table contacts and seed 5003 for one object-workspace violation.
Seed 5001 passed its selection gates; no three-seed bundle exists. The
1380 matrix is exposed and 1400 confirmation remains unused. V20 has no
physical qualification or owner viewer acceptance.

### Accepted V25 runtime

The final V25 runtime binds those exact V19 weights (epochs 20, 20 and 200)
to the frozen V24 wrist hold and copied-state collision preview, plus an
engineered opening trigger for the large block on the inspection pad at
the unchanged 0.0245 m held-object distance bound. Recipe loading verifies
the complete parent chain and code pins; qualification and saved-state
review use this same bundle. Learned move, orientation and release scores
remain distinct from engineered geometry, bounds, fixed joints, wrist hold,
preview and opening controls. No runtime expert phase or vision claim is added.
Fresh 1580 selection passes all numerical gates and 180/180 physical audits.
One-time 1600 confirmation passes every original state gate for all seeds
(196/240, 204/240 and 192/240 successes), with 720/720 independent audits
and verified artifact integrity. Counted failures and previously exposed
families remain disclosed. The owner confirmed the prescribed V25 viewer
review on October 5, 2026 with "Passed". The separate
`docs/level5_place_acceptance.json` binds that acceptance to the three exact
bundles and their qualification evidence; historical automated receipts
retain their then-pending manual status. Checkpoint 5.4A is complete with
bounded state qualification, without perception or default registry
admission. See `docs/level5_place_repair_verification_v25.json` for hashes
and `docs/level5_place_repair.md` for exact commands and manual criteria.

---

## Level 5.5 State-Grounded Push and Press

`dexvision/learning/contact_policy.py` and
`dexvision/evaluation/contact_policy.py` implement only standalone state-based
contact-phase push/press. Each has an independent recipe/sidecar checksum and
uses the unchanged Level 4 split, initial setup and exact first-contact boundary.
Scripted prefixes establish entry and are excluded from policy supervision,
jerk and task metrics. Inference receives no expert or saved continuation.
Dial remains absent under the unchanged manifest.

Push learns signed planar deltas at fixed 5 mm/component scale from object-
to-zone geometry, hand/object tracking, motion, support/contact and causal
physical phase. Press learns signed world-X deltas at the original 25 mm cap,
with fixture-relative geometry, measured slide displacement, target shortfall
and a causal pressed-dwell latch. The +X fixture axis and measured/commanded
body-origin calibration must agree with the checksum-pinned model. Source
whole-interval audits license fixed height/orientation/joints for push and
fixed Y/Z/orientation/joints for press; these are contact-entry invariants, not
learned full-hand control, free-space approach or vision. Zero output holds.

Both recipes use fixed physical feature scales, three full 200-epoch budgets
and cell/episode/present-phase/frame sampling. Early/late/final/best-offline
weights receive full validation. Selected weights bind transforms, scales,
decoder, cadence, recipe/source/schema/code digests and candidate selection.
Exclusive test reservations prevent retesting under another output path.
All failures remain counted; conditions repeat identical state physics and
do not qualify RGB robustness or globally unseen families.

Every terminal metric uses copied post-integration physical state. Separate
saved-state reconstruction checks every attempt without integrating physics.
Push requires object-zone settling, support, tilt, speed and board containment;
hand arrival is insufficient. Press requires physical depth/state dwell and
subsequent no-contact release. Contact loss, overshoot, wrong contacts,
non-target disturbance/activation, raw safety violations and timeouts are
retained. Original jerk/qualification gates remain unchanged. Conditional
owner saved-state viewer acceptance is still required for numerical passes;
no perception, registry, executor or deployment admission is implemented.
Method and exact commands are in `docs/level5_contact_results.md`.

The independent 5.5 V2 push repair lives in
`dexvision/learning/push_repair.py` and `dexvision/evaluation/push_repair.py`.
Its frozen initial-object-to-static-goal XY basis rotates task-relative geometry,
calibrated tracking lag, linear velocity and learned signed along/lateral deltas.
The unchanged 5 mm world-component cap scales the rotated vector uniformly;
zero holds and negative/lateral outputs remain valid. No positive-only progress
or online expert action is imposed. The unchanged V1 physical evaluator owns
safety, terminal dwell and all-attempt independent saved-state audits. Accepted
press execution code is immutable. V2 uses the original train/validation splits
and complete candidate selection, then a separately frozen, disjoint procedural
confirmation of all original test cells. Failed setup and policy attempts cannot
be filtered; each seed has one exclusive confirmation reservation. Fresh resets
of V1-exposed families do not establish unseen-family or perception qualification.
See `docs/level5_push_repair.md` for the source audit, exact recipe, commands and
the historical owner acceptance gate, since satisfied by the final V8 receipt
below. This amendment stays within checkpoint 5.5.

The separate V3 overlay (`dexvision/evaluation/push_settled.py`) binds the exact
V2 full-budget candidates to a fixed 120-sample ordinary-physics hold of the
last source-prefix command before every learned entry. This addresses a parked
non-target puck disturbance reproduced with zero policy output; it does not
relax disturbance thresholds or alter post-entry learned features/actions.
The engineered setup interval is disclosed and excluded from learned control
and jerk/task metrics. V3 repeats complete candidate validation and reserves a
new disjoint confirmation matrix. Its bundle includes `entry_overlay.json`,
parent weights/transforms and both original-reset and actual-prefix hashes.
The viewer requires all three fresh confirmations to pass before owner review.
A passing automated report alone cannot complete checkpoint 5.5.

V4 is a separately trained 5.5 distance representation repair in
`dexvision/learning/push_distance.py` and `dexvision/evaluation/push_distance.py`.
It caps only positive task-along goal distance at 0.05 m in the observation,
licensed by the source-only invariant far-distance action audit. Negative,
lateral and near-goal errors remain exact; the signed action head/decoder is
unchanged. Physical success/error metrics use uncapped measured distance.
Every candidate evaluates all 30 original validation resets plus eight fresh
all-cell selection resets, frozen before training and bound to the expanded
manifest's matrix/digest/count. These fresh selection families are already
exposed by V3's failed consumed confirmation; they do not supervise the model.
The separate 32-reset confirmation is disjoint and reserved once. The original
supervised split and archive remain immutable. The inherited source first-
contact rule defines the source/supervised boundary; actual execution entry
also includes the explicitly frozen 120-sample hold, with prefix lengths/hashes
in the matrix and saved setup. No failed V3 reset is retried or filtered.

V5 (`dexvision/learning/push_canonical.py`,
`dexvision/evaluation/push_canonical.py`) uses first-observation-relative measured
hand/object geometry and tracking changes, planar goal errors, object motion,
height change, contact/support and causal phase. No constant family/height/
contact-offset cue enters the model. Reset the reference for each independent
request/rollout; it is causal entry calibration, not future state or recurrent
history. Fixed physical scales are 0.1 m, 0.1 m/s, 5 rad/s and 0.01 m. Source
whole-interval task-lateral invariance plus independent long-cuboid source
feasibility license the one learned signed along-rate mask. Decoder sets lateral
zero, rotates the signed rate into world XY and holds other channels; zero holds
and negative retraction remain valid. This is independently grounded push scope,
not generic reuse of reach. V5 freezes new all-cell selection and confirmation
before three full training budgets and complete candidate validation. Uncapped
physical gates and owner acceptance still govern 5.5 completion.

V6 (`dexvision/learning/push_latched.py`, `dexvision/evaluation/push_latched.py`)
adds one causal completion-state bit to V5: maximum of current/past physical
push `phase.retract` indicators. That existing producer sets retract only on
current `TaskMetricResult.success`; it is not the saved expert stage or the
retrospective episode annotation. Reset entry reference and completion latch
for every independent rollout. Offline construction uses only observations up
to each action; no target action backdates the latch. Source post-success
retraction labels remain intact and separate from nominal progress before the
standalone physical endpoint. V6 retains the signed one-rate decoder and all
physical/safety gates, then binds new disjoint selection/confirmation to full
training budgets. Owner acceptance remains mandatory after numerical passes.


V7 (`dexvision/learning/push_predicates.py`,
`dexvision/evaluation/push_predicates.py`) has 12 causal inputs: signed along
and radial goal distance capped at 0.05 m, entry-relative lateral alignment
within 0.035 m, current object speed within the original 0.02 m/s physical
limit, entry-relative tracking change within 0.01 m, physical contact/support,
four current physical phases and the completion latch. Position scale is
0.1 m. Signed/near-goal distances remain exact; the physical goal metric is
uncapped. Predicates remove continuous lag/motion proxies that correlate with
the source's command magnitude without replacing the learned action with a
minimum progress rule. The MLP still predicts signed along displacement;
geometry determines only the audited direction/held channels and safety bounds.
Freeze 46 unique selection resets (30 original plus 16 fresh, four per cell)
and 32 disjoint confirmation resets before training. Failed setup is counted
as rejection, never filtered or replaced. Full budgets, complete candidate
selection, one-time confirmation and owner acceptance remain mandatory.


V8 (`dexvision/learning/push_guarded.py`,
`dexvision/evaluation/push_guarded.py`) preserves exact fully trained V7
candidate weights, schema, normalization and training manifest. Its separate
`entry_overlay.json` binds execution source, recipes, parent artifacts and
new validation/confirmation matrices. The training manifest retains the
original V7 selection metadata; the overlay owns its separately frozen
62-reset/248-attempt execution selection. No new fit or modified source labels
is implied. Every original candidate is reselected by the complete new matrix.

V8 has three explicit engineered operations. Before completion, physical
approach is encoded as nominal contact progress: source audit contains
48 training and nine validation advancing samples before true contact, while
source `phase.approach` never varies. Hold only currently physically qualifying
settle states until the original five-sample dwell completes. This prevents
a learned transient negative action from invalidating qualification; a
separate saved-state physical check verifies every hold. Uniformly scale signed
XY steps to the unchanged original command workspace; negative steps and zero
hold remain possible. All 13,508 perfect source commands are unchanged by
that bound. Log unbounded learned proposals, alias/hold flags and workspace
fractions, and independently reconstruct every executed decoder action.
These engineered operations are disclosed separately from learned signed
progress; boundary-limited attempts still count as failures when goals fail.
No unsafe motion, object pose write, online expert or success-gate relaxation
is permitted. Three passing one-time confirmations and owner acceptance are
still required; no subsequent checkpoint is authorized.


V8 automated qualification completes: all 12 full248-attempt candidate
matrices pass; selected epochs 200/61/20. One-time fresh confirmation passes
116/128, 116/128, 120/128 with worst-family 81.25%, 81.25%, 87.5%, zero executed
safety/wrong contact, and 384/384 independent physical/decoder/phase audits.
Confirmation retains12/12/8 timeouts and discloses348/532/876 workspace-bound
samples plus 468/476/480 settling holds. Exact artifacts/source/reset/code
bindings and successful actual contact pass verification. This is the complete
engineered/learned state bundle, not learned safety or vision. Accepted press
is unchanged. Owner push viewer acceptance ("Passed", October 13, 2026) is
bound separately in `docs/level5_push_acceptance.json`; checkpoint 5.5 is
complete. Historical automated receipts remain unchanged. Checkpoint 5.6 is
unstarted; perception and default registry admission remain unqualified.

---

## Level 5.6 Perception Baseline and Accepted-Bundle Compatibility

Owner-authorized publication includes code, compact V25/V29 models and
metadata/acceptance/status docs. New bulk dataset archives and complete
multipart payloads remain local; they are not available from a clone alone.
`datasets/README.md` documents exact-byte local restoration. Scientific and
acceptance bindings are preserved; no model/schema qualification changes.

**Current outcome: 5.6 complete; V29/fresh V8 automated gates and owner manual acceptance pass.**
The owner confirmed the held-out review passed on October 16, 2026.
Last Completed is **5.6**; Next Target is **5.7 (unstarted)**, on branch `main`.
`docs/level5_perception_acceptance.json` binds this owner-reported acceptance to
the exact frozen artifacts; no agent visual observation is claimed. Historical
automated receipts retain their then-pending manual status. No 5.7 work is
authorized by this completion request. The earlier V8 validation diagnostic
review remains separately accepted.

V29 retains exact V25 selected seed 11/source epoch 3,000 current-RGB MLP weights,
TRAIN transforms, fixed 8 px / 8 px / 10 mm output scales and calibrated ray decoder,
with zero new optimizer updates. All three original current candidates and all
six source fits remain preserved. Cuboid association uses the calibrated
projection of the current learned body center, with a cuboid-only uncertain
replacement-ID guard and history rollback. Other classes keep original birth/
visible-box association. Published masks, boxes and positions are unchanged.
Original tracker limits and all original visual gates remain unchanged.
Added identity coverage requires 99% unambiguous primary matches overall and 95%
in every supported primary class/phase; V29 passes both.

Validation: 5,923/5,923 primary matches and positions, zero identity switches, median
0.904 mm/p95 3.414 mm;
all 545 six-phase queries pass (worst phase p95 14.082 mm).
Fresh V8: 512 ordinary-physics source resets/26,687 RGB frames,
primary recall 100.0000%, median 0.954 mm,
p95 3.675 mm, 14,244 positions and
0 primary ID switches. All original/global/core/primary, supported phase,
position coverage, added identity coverage, and mandatory release/settle gates
pass independently. Release: 46 visible from
13 physical sources;
settle: 24 visible from
13 physical sources.

Full software: 1,177 passed/1 skipped; focused 64, Ruff and
whitespace pass. Actual 64 validation truth-isolation queries and 48 cached/uncached runtime
paths, exact accepted-transform oracles, portable RGB relocation, archive/member
readbacks, source/body-origin/schema audit, all 512 video decodes, typed/raw
parity and actual consumed-protocol retry rejection pass. The review has 16
panels and 36 fixed-plus-worst temporal clips.
Evidence: `docs/level5_perception_verification_v29.json`.

Model semantic SHA: `3a25ea7b49cc05ce7d67e89226a7e8c231a7feaea945d4863e7ed56a5cf5cadd`.
Active current-weight SHA: `0ba58825982fbfd714f028c3b8a9c4195e9f4ec5befe678e5273025863bbdb85`.
The baseline V2 and fresh V2–V7 consumed tests stay permanently closed, with all
archives, data, failed/positive candidates, reports and reservations preserved.
V7 failed one primary cuboid/release ID switch; V26/V27 failed independent identity
coverage. V28 positive validation and the independently frozen V29 guard plan
are preserved. Fresh V8 is now consumed too and cannot be retried or used for
training/selection. All source outcomes were retained; no test tuning or reuse.

The availability-aware schema is shared by truth and inference. Physical
linear/angular motion, full orientation, semantic identity, robot/fixture
telemetry and contacts remain explicitly unavailable and fail closed. All five
accepted inferred-control tracks remain unsupported. Qualification scope is
calibrated simulator pixel-center images on exposed task families/cells;
no real-camera, unseen-family, hardware, full-feature or closed-loop control
qualification is claimed. Python traced peak: 32,454,158 bytes for
model construction and 64 queries after imports; native/full-media peak remains
unmeasured and RTX 5070 Ti is untested.

Manual command (dexvision Conda environment, repository root):
`python -m dexvision.apps.repair_perception review --evaluation-dir outputs/level5/perception_confirmation_v8/evaluation`.
The owner confirmed all prescribed review criteria passed on October 16, 2026;
`docs/level5_perception_acceptance.json` records acceptance of the 16 panels and
36 scheduled clips. The command above is retained for saved-media reproduction;
no manual check remains for 5.6. Stop before 5.7.

Earlier paragraphs preserve the status of their historical attempts.

Current V19 selects seed3/epoch3,000 after two full-budget metric-loss fits on
frozen developmentV3 plus targetedV4 coverage. All original/core/primary and
stronger class/phase validation gates pass. The input is causal RGB-motion706;
learned outputs are body-center UV offsets and world height, with per-class
TRAIN centers and fixed8px/8px/10mm output scales. A calibrated ray decoder
produces world/metre body-position hypotheses. No full orientation, semantic
identity or sensor/contact state is invented. The shared partial schema guards
all five incompatible accepted controls. Its one-time fresh V4 test fails cuboid
stabilization median13.801mm>10mm, despite passing original/core/primary gates.
All four consumed sets remain closed. V20 retains identical inference while
fitting a separately frozen TRAIN/validation stabilization-window amendment.
V21's three full fits pass stabilization but fail lift; V22 changes loss
weights only, using complete ordered TRAIN provenance and equal role/phase/
physical-source mass. All features/targets, inference and metric gates remain
unchanged; no new test is opened. Both full-sequence strict validation and query validation (>=8visible queries,
>=4physical sources,recall>=95%,pose coverage>=99%,median<=10mm,p95<=25mm)
precede another independently frozen fresh confirmation. No phase/source/label
enters inference. Qualification/manual acceptance remain pending; no5.7 work.
`docs/level5_perception_verification_v19.json` binds the exact readiness artifacts.
Earlier paragraphs below retain their historical negative results.

Latest continuation: V11 passes pixel-center validation but fresh confirmation
V3 fails primary position/identity gates. All three consumed sets remain closed;
new development V3 and targeted cuboid development V4 with versioned
RGB-context / causal-tracker coverage repair remain in 5.6. No accepted runtime/schema is changed, full control fields remain
unsupported, no new owner acceptance is triggered, and 5.7 remains unstarted.
`docs/level5_perception_development.md` and the immutable V3 failure receipt
record the current work. Earlier method/result paragraphs below are historical.

October 14 fresh-confirmation amendment: the new V2 protocol/data were frozen
before generation/evaluation. The exact V8 model fails unchanged core/primary
gates on 32 new sessions / 1,641 RGB frames. Both historical baseline V2 test
and fresh confirmation V2 are consumed; neither can fit/select or retest a
repair. Failed scripted sources and every planned case remain counted.
The shared `level5/world-estimate-v1` type now permits explicitly unavailable
object/world fields for both true and inferred producers. Stable visual ids
remain distinct from unsupported semantic ids. `ObjectObservation`/`WorldState`
and every accepted runtime remain immutable behind the full field boundary.
All five inferred-control dispositions stay unsupported, including place's
privileged collision preview. Typed shape and calibration checks do not qualify
control or a failing visual method. See `docs/level5_perception_confirmation.md`
and `docs/level5_perception_confirmation_v2.json` for evidence and commands.

Contract amended October 14, 2026. V8 RGB component detection/grouping,
episode-local tracking and TRAIN-only local body-centre regression pass the
original aggregate and supplemental primary numerical gates on complete
validation. Failed V1–V7 evidence and the consumed negative V2 held-out result
remain intact; repair never reruns or tunes against that set. Frozen repair
recipes reject test evaluation before model/image loading. The selected model
contains its TRAIN feature statistics/exemplars, UV/height scales, temporal
feature precision and exact calibrated ray decoder. No full orientation,
semantic identity, physical contact/phase or fresh confirmation is claimed.

Owner validation diagnostic review is confirmed complete in
`docs/level5_perception_validation_acceptance.json`. Checkpoint 5.6 remains open
pending a passing independently frozen confirmation and its conditional owner
review; the new shared partial schema is implemented, with unsupported full
control inputs explicit. Current commands/limits are in
`docs/level5_perception_repair.md`, with bindings in
`docs/level5_perception_repair_verification_v8.json`. Historical negative results
and the original receipt remain unchanged. Accepted runtimes/object schemas
and datasets are unchanged; no 5.7 rollout is implemented.
The authoritative build/checklist/manual criteria are in the Level 5.6 section
of `docs/progress_level_5.md`. Accepted recipes and selected artifacts come from
the reach V5, pick V7, place V25 (V19 weights), push V8 (V7 weights) and press V1
owner acceptance receipts. Historical 5.0 configs and all accepted runtime
bundles remain immutable inputs to this work.

The perception boundary uses `ObjectObservation` and `WorldState`, preserving
stable semantic ids, metre/world-frame positions, wxyz orientation, timestamps,
confidence and validity. An inferred observation has `source=inferred_perception`.
Optional linear/angular velocity must have explicit availability. Per-field
provenance/availability is required in the proposed adapter/report; this prose
does not introduce a new implemented schema or grant permission to invent
missing measurements. Relations, contacts, fixture depth, robot telemetry,
entry references, derived goals and causal phase/history also need declared
producers. Schema shape alone is insufficient proof of equivalent inputs.

Maintain a field-source inventory naming consumer, units/body origin/frame,
time alignment, confidence/validity and reset/history rules. Producers are
inferred vision, declared robot/fixture sensors, frozen calibration/geometry,
evaluation-only truth or unsupported. Simulator contact/state availability
does not establish a deployment sensor. Never silently fill unknown state
with identity orientation, zero motion, false held/contact, or an empty list
and then present it as a valid estimate. Reject or explicitly block consumers
that require unavailable fields. Truth labels and parity/evaluation oracles
must be separate from policy-visible inferred features and action decisions.

In particular, the V25 evaluation path uses `place_wrist_hold_preview.py` and
`place_collision_preview.py`. The latter copies live MuJoCo data and scores
actions using predicted contacts/object positions. This is a privileged
dependency even when the neural input pose is replaced. Record an unsupported
perception compatibility disposition until an explicitly versioned later
integration amendment provides and evaluates an estimated-state alternative;
do not modify or disable the accepted preview in 5.6.

Bind the immutable visual manifest and `level4/visual-camera-v1` calibration.
Audit 3D body-center/height assumptions, pose symmetries, occlusion and object
coverage against later skill source releases/setups. Invisible truth pose
labels are not detections. Preserve split/session/episode ownership and
geometric consistency across render variants/augmentations. Track within each
episode, respecting subsampling/excluded-frame timestamp gaps; velocity uses
elapsed time. Freeze gap/reset/reacquisition rules and score temporal sequences.

The separate `configs/level5/perception.yaml` recipe and sidecar checksum must
freeze the method, supervision, scales, budget/seeds, candidate selection,
validation-selected confidence/age handling, metrics/denominators and artifact
digests before held-out evaluation. Explicitly record inheritance or versioned
changes to the parent 0.8 confidence and 0.1-second age limits without editing
the historical locks. Retain the original recall, position-error, identity-switch
and rejection gates. Add orientation, motion, fixture-depth, availability,
latency and task-predicate diagnostics with limits declared before test; generic
pose accuracy does not guarantee compatibility with millimetre contact rules.
Missing estimates and unsupported strata remain visible in the report.

5.6 can qualify its perception baseline only after its own automated gates and
owner panel/video acceptance pass, with an explicit compatibility disposition
for each skill. It does not qualify any closed-loop skill, authorize a truth
fallback, admit a skill to the default registry or start 5.7. Unsupported inputs
and privileged preview dependencies block the affected later perception track.

The additive offline `dexvision/perception/observation_boundary.py` adapter
retains `FieldEvidence` for each object/world field. Availability distinguishes
available measurements, uncertified hypotheses and absent values. Conversion
checks declared producer, frame/units/body origin, validity, confidence and
each field's acquisition time against the inherited 0.8/0.1 s limits. Optional
unavailable motion remains `None`; consumers can require motion explicitly.
Every entity requires a relation record; measured empty sets require available
evidence. Object time/confidence conservatively use the oldest/lowest consumed
field. This does not perform temporal interpolation or establish synchronized
sensor accuracy. Complete synthetic evidence converts to the original frozen
types without editing them. V8 RGB hypotheses keep semantic identity and full
orientation unavailable and reject; neither provenance declarations nor this
adapter certify a deployed producer or qualify any accepted skill. There is no
policy/action integration or change to the frozen V8 implementation pins.

---

## Historical Level 5.7 v1 Inferred-Control Rejection Boundary

Modules: `dexvision/learning/perception_policy.py` and
`dexvision/evaluation/perception_rollouts.py`, dispatched through
`dexvision.apps.evaluate_skill` with the separately checksum-locked
`configs/level5/evaluation_perception.yaml` protocol.

The accepted 5.6 visual baseline still cannot supply a full accepted control
state for any required skill. `PerceptionPolicy` is rejection-only; it accepts
RGB/acquisition/delivery times, resets episode history and returns the shared
typed estimate, explicit unavailable/uncertified/stale/low-confidence fields,
and `rejected_before_execution` with no action. Inference elapsed time contributes
to delivery age. A schema-shaped estimate does not certify a measured producer.
No simulator, task-derived goal, privileged preview, semantic-id oracle,
default-state filling or truth callback enters this interface. Place V25 remains
blocked by its unchanged live-state collision preview as well as missing fields.

The evaluator verifies accepted selected-weight/receipt/recipe/runtime-sidecar
bindings for all three seeds of all five skills. It saves/reserves its new
validation preflight matrix before inference. Fresh neutral-scene reset RGB/time
are the only inferred inputs. These are not task entry states or qualifying
closed-loop rollouts. All rejections stay counted, with no action or simulator
mutation. Paired state execution, physical success/safety, worst-family rates
and degradation remain absent/null rather than substituted from historical
confirmation. Actual inference latency and cadence misses are reported.

Saved five-panel rejection review checks hashes, visual-hypothesis labelling,
explicit missing fields and report/terminal consistency. It never records owner
acceptance and cannot replace the required held-out action/terminal rollout
review. Completion remains blocked on observable accepted inputs, a supported
versioned integration and all original numerical/manual gates. No 5.8 work or
default registry admission is authorized by this implementation.

## Level 5.7 instrumented observation amendment

The owner delegated a low-storage source amendment after the v1 rejection
preflight. The v2 `evaluation_perception.yaml` protocol is separate from V29.
Its sources and limits are specified in `docs/level5_perception_integration.md`.
V29's natural RGB control dispositions remain unsupported.

`OpticalMarkerRenderer` uses privileged body transforms only to render visible
rigid marker plates. `FiducialPoseEstimator.infer(rgb, timestamp)` accepts pixels
and acquisition time only; two tag faces, calibrated geometry and distinct-time
pose history produce body-center identity, full wxyz pose and world motion.
Missing/duplicate/poorly reprojected tags reject. No motion is fabricated at
birth, and a held camera sample retains its original age. The original 0.8
confidence and 0.1-second age limits remain; the deterministic 0.99 geometry
confidence is not a calibrated probability.

`WorkcellSensorPacket` declares ideal simulated robot/base encoders/pose,
mocap command feedback, actuator commands, fixture slide encoders and named
contact switches. No movable object pose/motion is exported. This is privileged
instrumented simulation, not unmarked RGB or real hardware qualification.
`InstrumentedStateEstimator` owns fresh MjData and reconstructs robot/fixture
measurements and observed optical object state. Static targets use calibrated
geometry. An unexpected interaction with an unobserved movable body rejects.
Place's unchanged preview logic consumes only this partial estimated copy;
unknown parked objects are excluded rather than filled from truth/default poses.
The independent state reference retains its original live-data preview.

Private rollout function globals substitute only declared observation,
precondition and preview inputs. Original accepted policy weights/transforms,
decoder classes and evaluator bytecode are preserved; the checksum-pinned
`dexvision/evaluation/evaluate_skill.py` is unchanged. Policy-visible phases,
press latches and reach derived approach poses use measured inputs only.
Evaluation truth scores physical terminals/safety and post-proposal errors; it
cannot supply features, an action preview or a missing field. This evaluator
is not qualification of an observation-only deployed supervisor.

Freeze both entry matrices and all source/input hashes before evaluation. Use
three exact receipt-bound models, all required cells, >=30 unique held-out resets
per skill and >=3 per required cell, each crossed with four original conditions.
One independent state execution is reused for four visual comparisons without
inflating unique-state counts. Identical three-tick prior-command holds and
entry hashes establish paired starts. Validation-only bias/dropout/latency/port
probes precede one-time disjoint test evaluation. Failed setups/rollouts stay in
the denominator. No new training, large image dataset or registry admission.

Saved compact float64 traces, per-condition/family/cell summaries, physical
reconstruction audits and fixed first-initialized-attempt review clips remain under a 640 MiB
experiment cap. Hashed saved-media playback never records owner acceptance.
5.7 completion remains pending owner manual confirmation. No 5.8 work follows.

The completed 5.7 automated experiment and manual-review handoff are bound in
`docs/level5_perception_integration_verification_v2.json`. The original one-time
test is consumed. A separately locked recovery runner preserves all completed
requests and rejects a measured failed paired place setup before observation;
no consumed control is retried. Read-only event and canonical-group audits fill
missing event evidence and preserve request-defined groups without changing
thresholds or raw reports. The corrected saved-media player derives its header
from recorded actions/pose, preserving all original clips and hashes. No track
passes every numerical gate; owner review is still pending. Last Completed is
5.6, Next Target is 5.7, and no 5.8/default registry work is authorized here.
