# MuJoCo Assets

DexVision keeps two hand models separated:

```text
assets/mujoco/debug_hand_scene.xml
assets/mujoco/hand_scene.xml
```

`debug_hand_scene.xml` is the original simple two-joint-per-finger smoke-test
hand. It remains useful for fast loading, actuator, and viewer checks, but it is
not the final Level 2/3 data-collection hand.

`hand_scene.xml` is the final hand scene for Level 1.8B and loads the right
Shadow Hand E3M5 model from:

```text
assets/mujoco/menagerie/shadow_hand/
```

## Shadow Hand Source

Source:

```text
https://github.com/google-deepmind/mujoco_menagerie/tree/main/shadow_hand
```

The vendored upstream files include:

```text
CHANGELOG.md
LICENSE
README.md
keyframes.xml
left_hand.xml
right_hand.xml
scene_left.xml
scene_right.xml
shadow_hand.png
assets/*.obj
```

`right_hand_dexvision.xml` is a local adapter copied from upstream
`right_hand.xml` with only the `meshdir` adjusted so that
`assets/mujoco/hand_scene.xml` can include it from the DexVision asset root.

## License

The upstream Shadow Hand directory is licensed under Apache-2.0. Its `LICENSE`
file includes:

```text
Copyright 2022 Shadow Robot Company Ltd
```

Keep `assets/mujoco/menagerie/shadow_hand/LICENSE` with the vendored assets.
