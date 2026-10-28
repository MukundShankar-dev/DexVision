# Rebuild plan

Goal: learn reach, pick, place, push and press with a simulated Shadow Hand.
All five use one Diffusion Policy, trained on honest physics and evaluated
eval-first. Pick goes first, with close owner review at every checkpoint.

Each checkpoint lists its pass criteria. Do one checkpoint, then stop for
review.

---

## R0 — Harness reset

- Tag `legacy-v1` holds the full original pipeline, including the
  uncommitted 5.7 work and the old agent guide.
- `main` contains only the lean harness (this file, `AGENTS.md`,
  `STATUS.md`, `docs/LEGACY.md`), packaging, and the scene assets that are
  kept.
- `.gitattributes` forces LF line endings. Data directories are untracked
  but kept on disk.

**Pass:** `pip install -e ".[dev]"` works; `ruff check` and `pytest` (empty
or smoke suite) pass; the owner approves the harness.

## R1 — Honest simulator core

`dexvision/sim/`: a small `Workcell` built on the existing scene and hand.

- **Hand base.** Gravity compensation on the hand bodies. The weld is tuned
  so the measured palm tracks the mocap target to within a few mm at rest.
  The tracking error is measured and reported.
- **Observations.** Read the measured palm and fingertip poses, hand joint
  positions and velocities, and object pose and velocity. Never the mocap
  target.
- **Object physics.**
  - Nominal friction and mass, set once in the XML with a justification.
  - No orientation holds, no friction multipliers.
  - Bin walls collide.
- **Reset.** Rebuilds model parameters from a pristine copy every time, so
  nothing accumulates. Object positions are randomized over centimetres;
  yaw is randomized.
- **Tests:**
  - no state writes after reset (wrap `MjData` writes in tests);
  - reset is deterministic per seed;
  - repeated resets do not drift model parameters;
  - the measured palm differs from the command when blocked.

**Pass:** tests pass. A short report lists measured weld tracking error, finger
force limits, and object masses and frictions.

## R2 — Pick task spec, evaluator and baselines (eval-first)

- **Task spec** in `configs/tasks/pick.yaml`:
  - object set and reset distribution;
  - observation list;
  - action space (shared by all skills: 6-D wrist pose delta plus 20 finger
    targets);
  - horizon, success predicate and safety limits;
  - seed ranges for train, val and test.
- **Success** (measured): the object is at least 5 cm above its rest height
  for at least 0.5 s, is not touching the table, and is in contact with at
  least two distinct finger links, one of them the thumb. Read from contacts
  on physical geoms.
- **One evaluator** for all skills. It reports unique episodes with Wilson
  CIs and per-object breakdowns, appends test runs to the ledger, and saves
  a few rollout videos.
- **Baselines on val:** zero action and random action.

**Pass:** the predicate has unit tests on constructed states, and baselines
are reported with CIs.

## R3 — Pick scripted expert

- A closed-loop state-feedback expert:
  - approach above the object;
  - descend;
  - thumb-opposition grasp with independent finger targets;
  - lift.
- It has privileged state (declared) and no lookahead. Parameters may depend
  only on object shape class and dimensions.
- The expert exposes `act(obs) -> action` from any state, so DAgger and DART
  are possible later.
- Evaluate on val with CIs. If it is weak on some objects, report that; do
  not tune physics.

**Pass:** expert val success is reported honestly. The owner reviews videos
and agrees the grasp looks physically plausible. The gate for learned
policies is fixed now, in `STATUS.md`.

## R4 — Data generation

- Expert rollouts on train seeds, with optional DART-style noise: execute
  `action + noise`, record the clean expert action.
- Only successful episodes are kept, and the filter rate is reported.
- Store data as compressed `.npz` per episode plus a JSON manifest (git
  commit, config, counts). No per-file source hashes.

**Pass:** dataset stats are reported, and a replay of recorded actions
reproduces the recorded outcomes.

## R5 — Diffusion Policy (method only)

- `dexvision/learning/diffusion.py`:
  - 1-D temporal conv UNet with FiLM conditioning;
  - DDPM training and DDIM sampling;
  - observation history 2, prediction horizon 16, execute 8;
  - normalization fitted on train only.
- An MLP behaviour-cloning baseline on the same interface.
- **Tests:** shapes; normalization round-trip; overfits a tiny synthetic
  dataset; deterministic with a fixed seed on CPU.

**Pass:** tests pass. The hyperparameters are written once and shared by
every skill.

## R6 — Train and evaluate pick

- Train Diffusion Policy and MLP-BC on the pick data, and select by val
  success.
- Report val results with CIs next to the zero, random and expert
  baselines.
- One test run, appended to the ledger.

**Pass:** the result is reported against the pre-written gate, pass or fail.
If it fails, diagnose before changing anything. Candidate fixes are DAgger
using the expert, or more data. Method changes follow `AGENTS.md` §3.

## R7+ — Remaining skills with the same method

Order: reach, push, press, place. Each repeats R2–R6 with no method
changes. Only the task spec and expert are new.

## V — Perception (draft, for owner review)

Goal: replace the privileged simulator state with camera images. The policy
keeps what a real robot measures itself (hand joint positions and velocities,
palm and fingertip poses) and the instruction (`goal_position`). It loses
`object_*` and `button_position`, and must see those instead.

Rules carried over:
- The vision method is a **new method** (`AGENTS.md` §3): the same Diffusion
  Policy (horizons, UNet, training settings) plus one image encoder, shared
  by every skill. State-based results stay as they are, as the ceiling.
- Each skill gets a vision variant of its spec (only the observation list
  changes) with its own new test seed range, written before looking.
- The camera setup is fixed before training and never tuned to make a policy
  pass (as physics is, rule 5).

### V1 — Cameras and image observations

- Two fixed cameras chosen to see the whole work area, the hand and the
  wall: the existing front-left view and one from the opposite side. A wrist
  camera is left out at first (simpler; revisit if occlusion by the hand
  proves to be the main failure).
- RGB, 96 × 96, rendered from the simulation after each step (measured, not
  privileged: what a camera would see).
- Demos store images alongside the state fields, so the state and vision
  tracks train on the same episodes. Images are uint8, compressed.
- **Measure:** render time per frame on an idle GPU (about 30 ms while the
  GPU was training), and dataset size per skill.
- **Tests:** images are deterministic per seed; rendering does not change the
  simulation (state is bit-identical with and without cameras); the camera
  poses never change between episodes.

**Pass:** the owner looks at sample frames from every skill and agrees the
cameras show what the task needs (object, goal marker, button).

### V2 — Vision method (method only)

- `dexvision/learning/vision.py`: a small ResNet-style encoder per camera
  (GroupNorm, spatial softmax keypoints), trained end to end from scratch,
  following Chi et al. 2023. Random-crop augmentation during training only.
  Its features join the proprioception vector that conditions the UNet.
- MLP-BC gets the same encoder, so the comparison stays like for like.
- **Tests:** shapes; overfits a tiny image dataset; deterministic on CPU.
- **Measure:** training time for 60k steps on the desktop. If it is far over
  about 3 hours per skill, propose a smaller encoder before going on.

**Pass:** tests pass; hyperparameters are written once for every skill.

### V3 — Pick from images (closely supervised, as R2–R6)

- Spec `pick` vision variant, new test range; baselines on val: the
  state-based Diffusion policy (ceiling), and the vision policy with images
  blanked to their training mean (must collapse, as the hidden-input checks
  did).
- Gate written before training, against the state-based result.
- Train, select on val, one ledgered test per method.

### V4+ — Press, push, place from images

Same procedure, one skill per checkpoint. Reach needs no perception (its
goal is an instruction), so it is not redone.

## Later (not planned in detail yet)

- Visual randomization (lighting, textures, camera jitter) once the plain
  vision track works.
- Skill composition.
