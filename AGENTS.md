# AGENTS.md

This repository is a staged robotics/computer-vision project. AI coding agents must work incrementally and must not implement broad project areas in one pass.

Project name: **DexVision / Hand2Bot**

Goal:

> Build a simulator-first, vision-grounded imitation-learning and reusable
> manipulation-skill pipeline using MuJoCo and PyTorch, with deterministic
> scripted experts for data generation and language-guided orchestration only
> after the skills qualify.

---

## 1. Core Agent Rule

Do not build large chunks at once.

For every task:

1. Read the relevant `docs/progress_level_*.md` file.
2. Select exactly one checkpoint.
3. Implement only the files needed for that checkpoint.
4. Add or update tests for that checkpoint.
5. Provide the exact commands to run.
6. Do not move to the next checkpoint until this one passes.

---


## 1A. Current Status File

Before selecting any work, always read:

```text
docs/CURRENT_STATUS.md
```

That file defines:

```text
active level
active progress file
last completed checkpoint
next target checkpoint
current branch
manual verification status
```

Do not choose checkpoints from other progress files unless `docs/CURRENT_STATUS.md` is updated or the user explicitly overrides it.

Do not advance to the next checkpoint automatically after finishing the current checkpoint.

If a checkpoint requires manual verification, stop after automated checks pass and give the user the exact command and pass/fail criteria. Do not mark the checkpoint complete until the user confirms the manual check passed.

Current handoff (October 16, 2026): Level 5.6 is complete after V29/fresh V8
automated checks and owner manual acceptance, recorded in
`docs/level5_perception_acceptance.json`. The active progress file remains
`docs/progress_level_5.md`. Last Completed is 5.6; Next Target is 5.7, unstarted.
The owner explicitly requested stopping before 5.7; do not start it without a
new user request. All five accepted inferred-control tracks remain unsupported;
5.6 completion does not qualify closed-loop control or default registry admission.
Preserve frozen releases, consumed tests and historical receipts unchanged.
The owner authorized publishing code, compact V25/V29 model bundles and
metadata/acceptance/status documentation while keeping large 5.6 data payloads
local to respect the reported 10 GiB LFS allowance. This is an explicit 5.6
exception to the general release publication rule below. Do not force-add bulk
5.6 archives or parts. A clone alone does not contain these new datasets.
Eight large archives are retained as complete verified local parts. Recreate
whole archives when needed with
`python -m dexvision.apps.restore_multipart_releases`; it does not download data.
Duplicate extracted image/source copies for consumed confirmation V2–V7 were
evicted after every payload hash was verified. V8 working data, active training
inputs, models, results, protocols and failed-attempt evidence remain preserved.
See `datasets/README.md`, `docs/level5_perception_publication.json` and local
`outputs/maintenance/disk_cleanup_20261016/` receipts for storage/restoration.
The approved code/model/metadata publication was pushed to `origin/main` as
`8dd467a`. Large payloads remain local. Local LFS object and temporary caches
were cleared after publication; preserve the ordinary Git commit history.

---

## 1B. Dataset Release and Git LFS Rules

Operator-owned working data remains under `data/demos/` and is intentionally
ignored by Git. Do not force-add individual working episodes or rewrite raw
episode files.

Immutable dataset releases live under `datasets/` as compressed archives. They
are tracked with Git LFS through `.gitattributes` and accompanied by:

```text
datasets/<release>.tar.gz
datasets/<release>.tar.gz.sha256
datasets/<release>_manifest.json
```

Before using a release after cloning:

```bash
git lfs install
git lfs pull
shasum -a 256 -c datasets/<release>.tar.gz.sha256
tar -xzf datasets/<release>.tar.gz
```

Rules for future agents:

```text
Never overwrite an existing release archive.
Create a new version when intentionally publishing changed data.
Keep generated staging data out of release archives.
Update the manifest, archive checksum, README, and status/progress docs together.
Verify the LFS attribute before committing an archive.
Do not claim cross-session provenance when the metadata does not contain it.
```

The current release process and contents are documented in
`datasets/README.md`.

---

## 2. Project Levels

The project is organized into seven levels:

```text
docs/progress_level_1.md
  Historical real-time hand-tracking control prototype; complete and retained.

docs/progress_level_2.md
  Demonstration recording, replay, data quality, and retargeting benchmarks.

docs/progress_level_3.md
  Learning feasibility on the existing narrow Level 2 datasets.

docs/progress_level_4.md
  Comprehensive multi-session dataset collection and versioned release.

docs/progress_level_5.md
  Full-scale skill learning, qualification, and supervised runtime.

docs/progress_level_6.md
  Portfolio polish, robustness, reproducibility, and optional advanced extensions.

docs/level7_future.md and docs/skill_orchestration_future.md
  Future language-guided skill orchestration; not currently active.
```

Agents should respect the current level and not prematurely implement future-level features.

Level 1 and the immutable Level 2 release preserve the original camera/hand-
control experiment as historical work. It is not an active Level 4+ data or
control path. Future agents must not require live hand-pose control, human-
operated demonstrations, or human corrective interventions for any remaining
checkpoint. Level 4+ nominal and corrective trajectories come from deterministic
scripted experts or explicitly qualified policies. Manual MuJoCo viewer review
is still allowed because visual verification is not a control source.

---

## 3. Do Not Overreach

Do not add these until explicitly requested:

```text
ROS2
Isaac Sim
Docker
web dashboard
cloud training
large foundation models
real robot hardware
complex RL
multi-camera calibration
```

This project should first work locally on Mac and Windows.

---

## 4. Cross-Platform Requirement

The code should run on:

```text
macOS
Windows
```

Avoid Linux-only assumptions.

Use:

```text
pathlib.Path instead of string path hacks
argparse for scripts
YAML configs for tunable parameters
clear error messages for missing camera/model files
```

Do not hardcode absolute paths.

---

## 5. Preferred Tech Stack

Use:

```text
Python 3.11+
OpenCV for camera I/O and overlays
MediaPipe for hand landmarks
MuJoCo for simulation
NumPy for numeric processing
PyTorch for learning in Level 3+
pytest for tests
ruff for linting
YAML for configs
```

Use the dedicated Conda environment for local development:

```bash
conda activate dexvision
```

Do not assume the base Conda environment has the required packages installed.

Do not replace MuJoCo with PyBullet unless explicitly asked.

Do not replace OpenCV with another camera stack unless explicitly asked.

---

## 6. Package Design Rules

Follow this module separation:

```text
dexvision/camera/
  Camera input only.

dexvision/perception/
  Hand landmark detection and visualization.

dexvision/features/
  Landmark-to-feature conversion and smoothing.

dexvision/retargeting/
  Human hand feature to robot joint target mapping.

dexvision/sim/
  MuJoCo model loading, stepping, tasks, and state extraction.

dexvision/logging/
  Demonstration recording, replay, validation, and quality filtering.

dexvision/learning/
  PyTorch datasets, models, training, and policies.

dexvision/evaluation/
  Metrics, plots, and benchmark scripts.

dexvision/apps/
  CLI entry points only. Apps connect modules; they should not contain core logic.
```

---

## 7. App Script Rules

Every script in `dexvision/apps/` should:

1. Be runnable with `python -m dexvision.apps.<script_name>`.
2. Use `argparse`.
3. Print clear startup information.
4. Fail gracefully when dependencies or devices are missing.
5. Avoid storing business logic inside the app file.

Good example:

```bash
python -m dexvision.apps.check_camera --camera-id 0 --width 1280 --height 720
```

---

## 8. Testing Rules

For every checkpoint, add at least one of:

```text
unit test
smoke test script
manual acceptance test checklist
```

Use synthetic data when testing camera-independent logic.

Do not require a webcam for automated tests.

Do not require a GPU for automated tests.

Do not require a visible MuJoCo GUI for automated tests unless the checkpoint explicitly concerns visualization.

---

## 9. Definition of Done for a Checkpoint

A checkpoint is done only when:

```text
the requested command runs
tests pass
the implementation is isolated to the relevant module
the progress markdown checkbox is updated
the result is documented with any important caveats
```

---

## 10. Coding Style

Prefer:

```text
small classes
dataclasses for structured outputs
explicit shapes in docstrings
type hints
clear error messages
configurable thresholds
```

Avoid:

```text
global mutable state
giant notebooks as the main implementation
magic numbers hidden in functions
side effects at import time
hardcoded camera IDs
hardcoded local paths
```

---

## 11. Standard Response from Coding Agent

When finishing a task, respond with:

```text
Implemented:
- file/module summary

How to run:
- command 1
- command 2

Tests:
- command and result

Notes:
- assumptions
- known limitations
- next checkpoint recommendation
```

---

## 12. Current Priority

Default priority order:

```text
1. Make the subsystem run.
2. Add tests and error handling.
3. Add visualization.
4. Add config.
5. Refactor only when needed.
```

Do not prematurely optimize.

---

## 13. Learning Method Handoff

Before Level 5 training or evaluation work, read
`docs/level5_reach_repair.md`, `docs/level5_training_guidance.md`, and the Level 5.2A contract in
`docs/module_contracts.md`. Keep checkpoint selection and completion governed
by `docs/CURRENT_STATUS.md` and the active progress file.

The reach V5 method uses state-based behavior cloning with a small MLP and an
explicit geometric action decoder. Document learned outputs separately from
engineered direction, bounds, fixed joints and safety checks; do not describe
this as learned vision or unconstrained full-hand control.

For later skills, independently audit coordinate frames, feature scales,
action variation and phase information. Reach-specific fixed posture and
straight planar movement require new evidence before reuse. Select models by
closed-loop validation, preserve failed attempts, and distinguish the selected
checkpoint from the last training epoch. Keep confirmation isolated from
selection and disclose prior exposure to evaluated task families. These rules
do not authorize starting a future checkpoint or bypassing manual acceptance.

For unstarted skills, freeze a separate recipe with explicit feature/scaling,
active-action/decoder and training-budget choices; do not inherit the old
full-action and offline-loss early-stop settings silently. The original 5.0
learning plan and config locks are historical evidence and must stay intact.
Package selected weights with their exact transform, scales, decoder and
control cadence. The same bundle must be used in qualification and deployment.
