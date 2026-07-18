# Task Board Environment

This document defines the initial Level 2 MuJoCo task board for structured
skill-demonstration recording. It is a design checkpoint, not an implementation
of recording, replay, learning, or orchestration.

The task board is a skill-learning environment. Level 2 defines resettable
tasks and records demonstrations; Level 3 trains one or more reusable low-level
skill policies from those demonstrations; a future Level 5 planner may compose
trained skills into longer tasks. Level 5 composition is not implemented here.

Level 2 should use the completed Level 1.13 teleoperation action space:

```text
base position target
base orientation target
finger actuator targets
```

Every recorded episode should also include robot qpos/qvel, hand/base pose,
object or task state, target pose, tracking quality, success metric inputs, and
a metadata/config snapshot so Level 3 can train from reproducible skill demos.

---

## Environment Overview

### Tabletop Workspace

The task board is a fixed tabletop MuJoCo scene with a bounded workspace in
front of the Shadow Hand. The board should be simple, resettable, and readable
from state arrays without relying on visual inspection.

The workspace should include:

```text
flat table or board plane
visible origin or workspace markers
fixed target markers
simple movable objects
task-specific fixtures such as a button or dial
simple graspable objects and placement targets
```

The initial implementation should favor deterministic starts. Randomized
starts can be added later, but each episode must still save the sampled initial
state in metadata or task state.

### Robot

The robot is the selected right Shadow Hand from
`assets/mujoco/hand_scene.xml`, controlled through the Level 1.13 interface.

The task board assumes:

```text
base x/y control from calibrated image-space palm motion
depth/in-out control from monocular hand scale
relative wrist/palm orientation when enabled
finger bend actuator targets from the Level 1 retargeter
```

The task environment should not define a new teleoperation interface. It should
consume the same commands that Level 1 emits so recorded demos match live
operator behavior.

### Objects

Use simple objects with explicit named bodies/sites/joints so task state can be
read without fragile geometry assumptions:

```text
target marker or touch site
pressable button with scalar press depth or state
small cube
target zone for cube pushing
graspable block or small object
placement zone or receptacle
dial with a hinge angle
small pinch object for the later stretch task
```

Objects should be lightweight enough for stable local simulation on macOS and
Windows, and all important state should be available through MuJoCo qpos/qvel,
body pose, joint value, site pose, or task-specific helper functions.

### Reset And Metrics

Every task must provide a reset path that restores:

```text
robot neutral pose
robot base neutral pose
object poses and velocities
task target state
success metric accumulators
```

Each task must define fixed success and failure criteria. Demo recording should
save enough state to recompute the final success label after the episode.

---

## Common Action Space

All initial tasks use the same recorded action fields, even when a task only
needs part of the command:

```text
base_position_target: [3]
base_orientation_target: [4], MuJoCo wxyz quaternion
finger_actuator_targets: [N]
```

The dense action array used by Level 2.1 may pack these fields, but metadata
must include an action schema that reconstructs the components exactly.

---

## Common Observation And State Fields

All episodes should record:

```text
timestamp
hand landmarks, optional but preferred
hand features
tracking quality
base position target
base orientation target
finger actuator targets
robot qpos
robot qvel
task name
task phase or label, optional
success metric inputs
metadata/config snapshot
```

Object tasks should additionally record object and target state, such as:

```text
object pose
object velocity
target pose
button press depth/state
dial angle and angular velocity
distance-to-goal metric
contact/touch flag when available
```

---

## Skill Parameters And Composition Contracts

Tasks that are intended to become reusable Level 3 skills must be
goal-parameterized. A fixed target or object may be used for a smoke test, but
the task schema and saved episodes must still declare the parameter values that
produced each trajectory.

Initial parameter contracts:

| Skill | Required parameters |
|---|---|
| `reach_touch_target` | `target_pose` or named `target_site` |
| `reach_object` | `object_id`, optional object-relative `approach_pose` |
| `button_press` | `button_id`, target press depth/state |
| `push_cube_to_target` | `object_id`, `target_pose` or `target_zone_id` |
| `grasp_object` | `object_id`, optional grasp family or approach pose |
| `pinch_lift_object` | `object_id`, target lift height |
| `place_object` | `object_id`, target pose or receptacle id |
| `release_object` | `object_id`, target release pose or zone |
| `rotate_dial` | `dial_id`, target angle or range |

Every parameter must declare:

```text
type and shape
units
coordinate frame
allowed range or named-id source
whether it is required
how it is represented in the recorded observation/task state
```

Every task intended for future composition must also declare a transition
contract:

```text
preconditions
valid initial-state envelope
success terminal-state envelope
failure terminal states
timeout
which world-state fields change
whether a retry is safe
```

The Level 3 policy should run closed loop from observation plus typed goal
parameters. The future Level 5 planner should invoke those parameters through a
supervised executor and receive a structured terminal result; it should not
control per-timestep actions.

---

## Initial Task Set

The initial task set is staged from simple operator/action-distribution demos
to contact and object manipulation. These are primitive skills, not a solved
long-horizon application. They could support a future constrained sandwich
environment, but object perception, symbolic state, sequencing, and long-horizon
planning remain future Level 5 concerns.

Curriculum notes:

```text
reach_touch_target is first because it validates base/wrist control without object dynamics.
button_press is the first contact skill.
push_cube_to_target introduces object dynamics.
grasp_object and pinch_lift_object are later because grasp stability is harder.
place_object requires a successful grasp/lift prerequisite.
release_object is separate because placing requires controlled release.
rotate_dial requires wrist/base orientation and contact control, so it is not an early first task.
```

### free_space_gesture

Stage:
Early/core calibration and action-distribution skill.

Composition note:
Treat this as a calibration and data-pipeline dataset by default, not as a
future orchestration skill. It only becomes a reusable skill if a later
checkpoint defines a concrete goal parameter and success condition for an
autonomous gesture request.

Purpose:
Record full-hand and wrist/base teleoperation without object contact.

Required objects:
No task objects are required. Optional visual markers may be present on the
tabletop for operator orientation.

Initial state:
The Shadow Hand starts at the calibrated neutral base pose with fingers open or
relaxed. No object state is reset.

Action space:
Full Level 1.13 action space: base position target, base orientation target,
and finger actuator targets.

Observation/state fields:

```text
hand landmarks
hand features
tracking quality
base target pose
finger actuator targets
robot qpos/qvel
optional gesture_label
```

Demonstration requirements:
Record clean open/close, pointing, pinching, waving, wrist rotation, base
translation, and tracking-loss recovery examples without contacting objects.

Success condition:
The episode is valid if tracking remains usable for the configured fraction of
frames and the robot stays within configured safety bounds.

Failure condition:
Tracking is missing for too much of the episode, robot state becomes unstable,
or the operator aborts.

Max episode length:
10 seconds, or 300 frames at 30 Hz.

Why useful for Level 3 learning:
It gives Level 3 clean examples of the action distribution before contact-rich
tasks: opening, closing, pointing, pinching, wrist rotation, base motion, and
tracking-loss behavior.

Why this skill matters for future orchestration:
It is mostly a calibration and representation dataset, not a high-level
orchestration skill. It helps later policies understand safe neutral motions
and hand poses.

### reach_touch_target

Stage:
Early/core first learned skill.

Purpose:
Teach the hand base and wrist to approach a fixed target on the board without
requiring object manipulation.

Required objects:
A fixed target site or small pad on the tabletop.

Initial state:
The target is fixed or sampled from a small set of known board locations. The
robot starts from the calibrated neutral pose.

Action space:
Full Level 1.13 action space. Finger targets may remain mostly open, but they
must still be recorded.

Observation/state fields:

```text
target position
base target pose
finger actuator targets
robot qpos/qvel
fingertip or palm site position
distance to target
tracking quality
```

Demonstration requirements:
Record smooth reaches from neutral to several fixed target sites. Include
approaches with different wrist orientations when safe, but avoid object
contact dynamics.

Success condition:
The configured fingertip, palm, or touch site reaches within a fixed distance
threshold of the target and holds it for a short dwell window.

Failure condition:
The episode times out, tracking quality is too low, or the hand leaves the
workspace bounds.

Max episode length:
8 seconds, or 240 frames at 30 Hz.

Why useful for Level 3 learning:
It isolates spatial reaching, depth control, and wrist alignment before adding
object dynamics.

Why this skill matters for future orchestration:
Most longer tasks begin with moving the hand to an object, fixture, or target
region. Future planners can parameterize this skill with a target pose or named
MuJoCo site.

### reach_object

Stage:
Early/core object-approach skill after `reach_touch_target`.

Purpose:
Move the hand to a pre-grasp or touch pose near a named object without yet
requiring grasp closure.

Required objects:
A simple named object, such as a cube or small block, with a saved object pose
and an optional pre-grasp target site.

Initial state:
The object starts at a known reachable pose. The robot starts from the
calibrated neutral pose.

Action space:
Full Level 1.13 action space. The expected behavior mainly uses base pose and
wrist orientation; finger targets should remain safe/open or task-appropriate.

Observation/state fields:

```text
object pose
pre-grasp or touch target pose
base target pose
finger actuator targets
robot qpos/qvel
distance to object/pre-grasp site
tracking quality
```

Demonstration requirements:
Record approach trajectories to objects at several reachable board positions.
Avoid requiring object displacement or stable grasp during this skill.

Success condition:
The configured hand site reaches the object-relative pre-grasp/touch threshold
and holds for a short dwell window without disturbing the object beyond a small
tolerance.

Failure condition:
The episode times out, the object is knocked out of tolerance, tracking quality
is too low, or the hand leaves the workspace bounds.

Max episode length:
8 seconds, or 240 frames at 30 Hz.

Why useful for Level 3 learning:
It bridges target reaching and manipulation by adding object-relative target
poses while still avoiding difficult object dynamics.

Why this skill matters for future orchestration:
Future planners can call `reach_object(object_id)` before grasping, pushing, or
tool interaction.

### button_press

Stage:
Early/core first contact skill.

Purpose:
Record approach, alignment, and deliberate contact against a simple one-degree
fixture.

Naming note:
The concrete task id is `button_press`. A future skill API may expose the
general alias `press_button`.

Required objects:
A pressable button with a named joint or scalar press state, plus an optional
visual target marker.

Initial state:
The button starts unpressed at a fixed pose or a known sampled pose. The robot
starts at the calibrated neutral pose.

Action space:
Full Level 1.13 action space. The expected behavior uses base pose, wrist
orientation, and one or more finger actuator targets.

Observation/state fields:

```text
button pose
button press depth or joint value
button pressed flag
base target pose
finger actuator targets
robot qpos/qvel
tracking quality
```

Demonstration requirements:
Record stable approaches, clear press events, and short dwell holds. Demos
should include enough state to relabel success from button press depth/state.

Success condition:
The button press depth crosses the configured pressed threshold and remains
pressed for the required dwell window.

Failure condition:
The episode times out, the hand knocks the button fixture out of bounds, robot
state becomes unstable, or tracking quality is too low.

Max episode length:
8 seconds, or 240 frames at 30 Hz.

Why useful for Level 3 learning:
It introduces contact timing and force-direction consistency while keeping the
object state simple and easy to score.

Why this skill matters for future orchestration:
Button or switch activation is a common low-level primitive for future
procedural environments, and it tests contact without free object motion.

### push_cube_to_target

Stage:
Core manipulation skill after reaching and simple contact.

Purpose:
Record the first planar object manipulation task with a movable object and a
fixed target zone.

Naming note:
The concrete first task is `push_cube_to_target`. A future generalized skill
card may expose `push_object_to_target(object_id, target_pose)`.

Required objects:
A small cube and a target zone on the tabletop.

Initial state:
The cube starts at a known pose or one of a small set of sampled poses. The
target zone is fixed or sampled independently within the reachable board area.
The robot starts at the calibrated neutral pose.

Action space:
Full Level 1.13 action space. Successful demos should use base movement, wrist
orientation, and finger posture to contact and push the cube.

Observation/state fields:

```text
cube pose
cube linear/angular velocity
target zone pose
cube-to-target distance
base target pose
finger actuator targets
robot qpos/qvel
tracking quality
```

Demonstration requirements:
Record full approach, contact, push, and settle phases. Demos must save cube
pose/velocity, target zone pose, and distance-to-target metric inputs so success
can be relabeled after recording.

Success condition:
The cube center ends within a fixed distance threshold of the target zone and,
optionally, remains there for a short dwell window.

Failure condition:
The cube leaves the board, the episode times out, tracking quality is too low,
or the robot state becomes unstable.

Max episode length:
12 seconds, or 360 frames at 30 Hz.

Why useful for Level 3 learning:
It is the first object task with visible cause and effect, useful for learning
contact-rich hand/base coordination without requiring grasp stability.

Why this skill matters for future orchestration:
Future planners can use this as `push_object_to_target(object_id, target_pose)`
when a task requires moving an object across a surface without grasping.

### grasp_object

Stage:
Later/stretch grasp-formation skill.

Purpose:
Record closing the fingers around a reachable object to establish a stable
grasp without requiring a lift.

Required objects:
A graspable block or small object with stable collision geometry and named
object pose.

Initial state:
The object starts at a fixed reachable pose. The robot starts from neutral or
from the successful terminal region of `reach_object`.

Action space:
Full Level 1.13 action space. Successful demos require base pose, wrist
orientation, and coordinated finger actuator targets.

Observation/state fields:

```text
object pose
object velocity
base target pose
finger actuator targets
robot qpos/qvel
contact or proximity flags, if available
grasp stability proxy
tracking quality
```

Demonstration requirements:
Record approach-to-close transitions with visible stable contact. Demos should
avoid lifting so grasp formation can be scored separately from lift success.

Success condition:
The object remains within the grasp region while configured finger/hand sites
maintain contact or proximity for a dwell window.

Failure condition:
The object is knocked away, the hand closes without enclosing the object,
tracking quality is too low, or the episode times out.

Max episode length:
10 seconds, or 300 frames at 30 Hz.

Why useful for Level 3 learning:
It isolates grasp formation before adding vertical lift and placement.

Why this skill matters for future orchestration:
Future planners can use `grasp_object(object_id)` before lift, place, or tool
skills. It is intentionally later because grasp success is more fragile than
reaching or pushing.

### pinch_lift_object

Stage:
Later/stretch lift skill.

Purpose:
Record a pinch or small-object grasp followed by a short vertical lift.

Required objects:
A small pinchable object with stable collision geometry and a lift-height
target.

Initial state:
The object starts at a fixed reachable pose on the tabletop. The robot starts
from neutral, from `reach_object`, or from a successful `grasp_object` terminal
state when staged replay is available.

Action space:
Full Level 1.13 action space. Successful demos require coordinated base pose,
wrist orientation, thumb/index or multi-finger closure, and lift motion.

Observation/state fields:

```text
object pose
object velocity
lift height
target lift height
base target pose
finger actuator targets
robot qpos/qvel
tracking quality
optional contact flags
```

Demonstration requirements:
Record stable pinch/grasp closure, lift, and short hold. Save object height and
drop/failure state so success can be relabeled.

Success condition:
The object is lifted above the target height and held for the configured dwell
window without leaving the workspace.

Failure condition:
The object is dropped, leaves the workspace, never clears the table, tracking
quality is too low, or the robot state becomes unstable.

Max episode length:
12 seconds, or 360 frames at 30 Hz.

Why useful for Level 3 learning:
It combines grasp stability, vertical motion, and object state feedback.

Why this skill matters for future orchestration:
Future planners can use `pinch_lift_object(object_id)` for constrained pick
steps after a reachable object and feasible grasp are established.

### place_object

Stage:
Later/stretch placement skill requiring a grasp/lift prerequisite.

Purpose:
Move a held object to a target zone or receptacle without dropping it early.

Required objects:
A grasped/lifted object and a target placement zone, plate, tray, or named site.

Initial state:
The object starts already grasped or lifted, either from a scripted reset or a
successful `grasp_object`/`pinch_lift_object` terminal state.

Action space:
Full Level 1.13 action space. Successful demos use base pose, wrist orientation,
and finger targets that maintain grip until the release phase.

Observation/state fields:

```text
object pose
object velocity
target placement pose
base target pose
finger actuator targets
robot qpos/qvel
held-object flag or contact proxy
distance to placement target
tracking quality
```

Demonstration requirements:
Record transport and stabilization over the placement target. Release may be
recorded as an explicit terminal phase, but `release_object` should also be
available as a separate skill for controlled study.

Success condition:
The object reaches the placement target within tolerance while still controlled
by the hand or after a valid terminal release.

Failure condition:
The object is dropped early, misses the target, leaves the workspace, or the
episode times out.

Max episode length:
12 seconds, or 360 frames at 30 Hz.

Why useful for Level 3 learning:
It teaches object transport and target-relative positioning after grasp/lift
skills are reliable.

Why this skill matters for future orchestration:
Future planners can use `place_object(object_id, target_pose)` for structured
assembly or constrained food-preparation environments.

### release_object

Stage:
Later/stretch terminal control skill.

Purpose:
Open or relax the hand in a controlled way so a held object is released at the
desired location.

Required objects:
A held object and a target surface, zone, or receptacle.

Initial state:
The object starts held at or near the target placement pose. The robot starts
from a stable grasp/lift or placement pre-release state.

Action space:
Full Level 1.13 action space. The expected behavior focuses on finger actuator
targets while keeping base/wrist pose stable.

Observation/state fields:

```text
object pose
object velocity
target release pose
base target pose
finger actuator targets
robot qpos/qvel
contact or held-object proxy
tracking quality
```

Demonstration requirements:
Record gradual controlled opening, not just abrupt dropping. Save final object
pose and velocity to score stable release.

Success condition:
The object separates from the hand and settles within the target tolerance
without excessive bounce or drift.

Failure condition:
The object remains stuck, is dropped outside the target, is launched by the
release, or the episode times out.

Max episode length:
6 seconds, or 180 frames at 30 Hz.

Why useful for Level 3 learning:
It separates terminal release control from reaching and placement, making
placing easier to evaluate and compose.

Why this skill matters for future orchestration:
Future planners can explicitly call `release_object()` after `place_object`
instead of hoping release is hidden inside another policy.

### rotate_dial

Stage:
Later/core contact-orientation skill after reaching and simple contact.

Purpose:
Record dexterous wrist/finger contact for changing a constrained object's
angle.

Required objects:
A dial with a named hinge joint, angle limits, and a visible target angle or
target range.

Initial state:
The dial starts at a known initial angle. The target angle is fixed for early
demos and can later be sampled from a small set.

Action space:
Full Level 1.13 action space. Successful demos should combine base pose,
orientation, and finger actuator targets.

Observation/state fields:

```text
dial angle
dial angular velocity
target angle or target range
angle error
base target pose
finger actuator targets
robot qpos/qvel
tracking quality
```

Demonstration requirements:
Record approach, contact, rotation toward target, and dwell in the target range.
Demos must save dial angle, target angle/range, and angle error for relabeling.

Success condition:
The dial angle reaches the target range and remains within tolerance for the
configured dwell window.

Failure condition:
The episode times out, the dial exceeds safe limits, tracking quality is too
low, or the robot state becomes unstable.

Max episode length:
10 seconds, or 300 frames at 30 Hz.

Why useful for Level 3 learning:
It adds constrained-object manipulation and teaches policies to coordinate
contact point, wrist orientation, and incremental finger motion.

Why this skill matters for future orchestration:
Future planners can use `rotate_dial(dial_id, target_angle)` when a constrained
environment requires angular adjustment. It is not an early first task because
it needs wrist/base orientation plus contact control.

### optional later: spread_over_surface

Stage:
Optional later extension.

Purpose:
Move a deformable or granular proxy across a surface in a controlled region.

Required objects:
A simple spreadable proxy, tool, or surface marker. This should wait until core
rigid-object skills are reliable.

Initial state:
The proxy starts in a bounded region on a target surface.

Action space:
Full Level 1.13 action space, possibly with a simple tool-use fixture later.

Observation/state fields:

```text
surface region state
proxy coverage metric
base target pose
finger actuator targets
robot qpos/qvel
tracking quality
```

Demonstration requirements:
Record only after core rigid-object tasks work. Define coverage metrics before
collecting demos.

Success condition:
Coverage or surface metric reaches the configured threshold.

Failure condition:
The proxy leaves the target region, metrics cannot be measured, or the episode
times out.

Max episode length:
15 seconds, or 450 frames at 30 Hz.

Why useful for Level 3 learning:
It tests more continuous contact behavior after rigid-object primitives.

Why this skill matters for future orchestration:
It could support constrained future food-preparation tasks, but it is optional
and not required for Levels 2 or 3.

### optional later: tool_use_simple

Stage:
Optional later extension.

Purpose:
Use a simple tool or tool-like fixture for a constrained one-step task.

Required objects:
A lightweight tool, holder, and target fixture with measurable state.

Initial state:
The tool starts in a known pose, or already grasped for the first version.

Action space:
Full Level 1.13 action space.

Observation/state fields:

```text
tool pose
target fixture state
base target pose
finger actuator targets
robot qpos/qvel
tracking quality
```

Demonstration requirements:
Keep the task narrow and measurable; do not introduce broad tool-use planning.

Success condition:
The tool changes the target fixture state within a fixed tolerance.

Failure condition:
The tool is dropped, target state is not changed, or the episode times out.

Max episode length:
15 seconds, or 450 frames at 30 Hz.

Why useful for Level 3 learning:
It would test transfer from grasp/contact skills to a constrained tool fixture.

Why this skill matters for future orchestration:
It may become a useful primitive for long-horizon tasks, but it should remain
optional until core reach/contact/push/grasp/place skills are reliable.

---

## Out Of Scope

Level 2 should not train policies, add Level 3 learning code, or implement
future Level 5 skill orchestration. Level 2's job is to produce reliable,
validated demonstrations and benchmarks that Level 3 can consume later.
