# DexVision

Manipulation skills (reach, pick, place, push, press) for a simulated Shadow
Hand in MuJoCo:

- **Experts.** Closed-loop scripted experts generate demonstrations.
- **Learning.** One small Diffusion Policy learns every skill.
- **Physics.** No orientation holds, no friction boosts, and success is
  measured on the physical hand and object.
- **Evaluation.** Eval-first: task spec, baselines and gates come before
  training, and every test-split run is logged.

The project is being rebuilt. See [`STATUS.md`](STATUS.md) for where it
stands and [`docs/PLAN.md`](docs/PLAN.md) for the checkpoints. The original
pipeline is preserved at git tag `legacy-v1`; see
[`docs/LEGACY.md`](docs/LEGACY.md) for why it was retired.

## Setup

```bash
conda env create -f environment.yml
conda activate dexvision
ruff check dexvision tests
pytest
```

On macOS, run viewer apps with `mjpython`. Automated tests need no GUI, GPU or
camera.

## Credits

The Shadow Hand model is from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(Apache-2.0; see `assets/mujoco/menagerie/shadow_hand/LICENSE`).
