# MuJoCo assets

`workcell.xml` is the only scene. Its physics choices are documented in its header comment.
Changing them invalidates every recorded result (AGENTS.md rule 5).

## Shadow Hand

Vendored from
[MuJoCo Menagerie `shadow_hand`](https://github.com/google-deepmind/mujoco_menagerie/tree/main/shadow_hand)
under Apache-2.0 (Copyright 2022 Shadow Robot Company Ltd). Keep `LICENSE` with the files.

`right_hand.xml` and `scene_right.xml` are unmodified upstream files.
`right_hand_dexvision.xml` is a copy of `right_hand.xml` with exactly these changes:

- `meshdir` points at `menagerie/shadow_hand/assets` so `workcell.xml` can include it.
- `rh_forearm` has a free joint, so the forearm can float on the mocap weld that stands in for an arm.
- `rh_forearm` is gravity-compensated (`gravcomp="1"`), as an arm controller would do. The wrist,
  palm and fingers are not compensated: their own actuators carry them.

Actuator gains and force limits are the upstream values.
