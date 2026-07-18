# Robot Hand Model Decision

This file prevents DexVision from accidentally recording demonstrations or
training policies against a placeholder hand model.

---

## Current Model Status

The original simple MuJoCo hand is now the debug/smoke-test hand only. It is
acceptable for Level 1 checks that prove MuJoCo loading, named actuators,
joint-limit validation, and scripted finger movement work.

The selected final Level 1.8B hand is the right Shadow Hand E3M5 model from
Google DeepMind MuJoCo Menagerie. This model defines the intended action space
for Level 2 demonstration recording and Level 3 learning. Manual visual
verification passed for Level 1.8B after the user inspected the MuJoCo viewer
and confirmed that scripted open/fist/point/pinch/peace/relax gestures moved
the Shadow Hand clearly enough for the checkpoint.

Changing the hand after collecting demos can invalidate:

```text
recorded action arrays
retargeting configs
replay behavior
benchmark results
trained policies
```

---

## Debug Hand Path

```text
assets/mujoco/debug_hand_scene.xml
```

This is the current simple hand used by debug and smoke-test apps. It is not the
final training hand.

Debug gesture config:

```text
configs/debug_hand_gestures.yaml
```

---

## Final Hand Path

```text
assets/mujoco/hand_scene.xml
```

Status:

```text
selected: Shadow Hand E3M5 right hand from Google DeepMind MuJoCo Menagerie
manual visual verification: passed for Level 1.8B on June 10, 2026
final action space: Shadow Hand actuator and joint set, plus Level 1.13 base pose targets
```

Vendored source directory:

```text
assets/mujoco/menagerie/shadow_hand/
```

Final gesture config:

```text
configs/hand_gestures.yaml
```

Future Level 1 retargeting config seed:

```text
configs/level1_teleop.yaml
```

---

## Source And License

Source:

```text
https://github.com/google-deepmind/mujoco_menagerie/tree/main/shadow_hand
```

Upstream model:

```text
Shadow Hand E3M5 Description (MJCF)
```

License:

```text
Apache-2.0
Copyright 2022 Shadow Robot Company Ltd
```

The upstream README states that the original URDF and assets were provided by
Shadow Robot Company under the Apache-2.0 license. The vendored directory keeps
the upstream `README.md`, `LICENSE`, and `CHANGELOG.md` files.

DexVision local adapter:

```text
assets/mujoco/menagerie/shadow_hand/right_hand_dexvision.xml
```

This file is copied from upstream `right_hand.xml` with only the `meshdir`
changed so `assets/mujoco/hand_scene.xml` can include the model from the
DexVision asset root.

---

## Final Hand Joint Names

```text
rh_WRJ2
rh_WRJ1
rh_FFJ4
rh_FFJ3
rh_FFJ2
rh_FFJ1
rh_MFJ4
rh_MFJ3
rh_MFJ2
rh_MFJ1
rh_RFJ4
rh_RFJ3
rh_RFJ2
rh_RFJ1
rh_LFJ5
rh_LFJ4
rh_LFJ3
rh_LFJ2
rh_LFJ1
rh_THJ5
rh_THJ4
rh_THJ3
rh_THJ2
rh_THJ1
```

---

## Final Hand Actuator Names

```text
rh_A_WRJ2 -> joint rh_WRJ2
rh_A_WRJ1 -> joint rh_WRJ1
rh_A_THJ5 -> joint rh_THJ5
rh_A_THJ4 -> joint rh_THJ4
rh_A_THJ3 -> joint rh_THJ3
rh_A_THJ2 -> joint rh_THJ2
rh_A_THJ1 -> joint rh_THJ1
rh_A_FFJ4 -> joint rh_FFJ4
rh_A_FFJ3 -> joint rh_FFJ3
rh_A_FFJ0 -> tendon rh_FFJ0
rh_A_MFJ4 -> joint rh_MFJ4
rh_A_MFJ3 -> joint rh_MFJ3
rh_A_MFJ0 -> tendon rh_MFJ0
rh_A_RFJ4 -> joint rh_RFJ4
rh_A_RFJ3 -> joint rh_RFJ3
rh_A_RFJ0 -> tendon rh_RFJ0
rh_A_LFJ5 -> joint rh_LFJ5
rh_A_LFJ4 -> joint rh_LFJ4
rh_A_LFJ3 -> joint rh_LFJ3
rh_A_LFJ0 -> tendon rh_LFJ0
```

Tendon actuators ending in `J0` couple each finger's middle and distal joints.

---

## Final Hand Joint Limits

```text
rh_WRJ2: [-0.524, 0.175]
rh_WRJ1: [-0.698, 0.489]
rh_FFJ4: [-0.349, 0.349]
rh_FFJ3: [-0.262, 1.571]
rh_FFJ2: [0.000, 1.571]
rh_FFJ1: [0.000, 1.571]
rh_MFJ4: [-0.349, 0.349]
rh_MFJ3: [-0.262, 1.571]
rh_MFJ2: [0.000, 1.571]
rh_MFJ1: [0.000, 1.571]
rh_RFJ4: [-0.349, 0.349]
rh_RFJ3: [-0.262, 1.571]
rh_RFJ2: [0.000, 1.571]
rh_RFJ1: [0.000, 1.571]
rh_LFJ5: [0.000, 0.785]
rh_LFJ4: [-0.349, 0.349]
rh_LFJ3: [-0.262, 1.571]
rh_LFJ2: [0.000, 1.571]
rh_LFJ1: [0.000, 1.571]
rh_THJ5: [-1.047, 1.047]
rh_THJ4: [0.000, 1.222]
rh_THJ3: [-0.209, 0.209]
rh_THJ2: [-0.698, 0.698]
rh_THJ1: [-0.262, 1.571]
```

---

## Final Hand Actuator Control Ranges

```text
rh_A_WRJ2: [-0.524, 0.175]
rh_A_WRJ1: [-0.698, 0.489]
rh_A_THJ5: [-1.047, 1.047]
rh_A_THJ4: [0.000, 1.222]
rh_A_THJ3: [-0.209, 0.209]
rh_A_THJ2: [-0.698, 0.698]
rh_A_THJ1: [-0.262, 1.571]
rh_A_FFJ4: [-0.349, 0.349]
rh_A_FFJ3: [-0.262, 1.571]
rh_A_FFJ0: [0.000, 3.142]
rh_A_MFJ4: [-0.349, 0.349]
rh_A_MFJ3: [-0.262, 1.571]
rh_A_MFJ0: [0.000, 3.142]
rh_A_RFJ4: [-0.349, 0.349]
rh_A_RFJ3: [-0.262, 1.571]
rh_A_RFJ0: [0.000, 3.142]
rh_A_LFJ5: [0.000, 0.785]
rh_A_LFJ4: [-0.349, 0.349]
rh_A_LFJ3: [-0.262, 1.571]
rh_A_LFJ0: [0.000, 3.142]
```

---

## Useful Body Names

```text
palm: rh_palm
thumb fingertip: rh_thdistal
index fingertip: rh_ffdistal
middle fingertip: rh_mfdistal
ring fingertip: rh_rfdistal
pinky fingertip: rh_lfdistal
forearm: rh_forearm
wrist: rh_wrist
```

---

## Neutral Pose

Final hand neutral pose for Level 1.8B:

```text
all actuator controls = 0.0
all joint positions reset by MuJoCo = 0.0
```

The `relax` gesture in `configs/hand_gestures.yaml` sends all actuator controls
to `0.0`.

---

## Known Visual/Control Limitations

Final Shadow Hand limitations for the current checkpoint:

```text
manual visual verification passed for Level 1.8B after user visual inspection
scripted gestures are approximate and not meant to be final task policies
pinch is approximate and may need visual tuning
peace sign is approximate and may need visual tuning
thumb control remains conservative in the current Level 1 teleop path
finger middle/distal joints use tendon coupling through J0 actuators
no Level 2 demo recording has been validated against this hand yet
no Level 3 learning has been validated against this hand yet
```

Current debug hand limitations:

```text
coarse box palm and capsule fingers
only two joints per non-thumb finger
thumb geometry and motion are approximate
no tendon coupling
no realistic contact pads or fingernails
gesture shapes are approximate
not validated for object manipulation demos
not validated as a stable final action space for learning
```

---

## Decision

Decision:

```text
final model selected: Shadow Hand E3M5 right hand
debug hand retained: assets/mujoco/debug_hand_scene.xml
manual viewer verification passed; Level 1.8B is complete
final action space defined by the Shadow Hand actuator/joint set and Level 1.13 base pose command
```

Completed before serious Level 2 demos or Level 3 training:

```text
1. manually verified assets/mujoco/hand_scene.xml in the MuJoCo viewer
2. confirmed scripted gestures visibly moved the Shadow Hand clearly enough for Level 1.8B
3. confirmed no checkpoint-blocking instability errors during Level 1.8B verification
4. implemented Level 1.9 retargeting against configs/level1_teleop.yaml
5. completed Level 1.13D full teleoperation with base position, base orientation, and finger targets
```

Remaining limitations before high-quality manipulation datasets:

```text
1. tune approximate pinch/peace behavior only if future task demos need it
2. keep thumb behavior conservative until a dedicated thumb-control revisit
3. validate Level 2 task recording against the final full action schema
```
