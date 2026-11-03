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
- Cameras framed on the work area (about 30 cm across), not the whole scene:
  the current view spans about 65 cm, about 7 mm per pixel at 96 px, which
  leaves a 2 cm goal tolerance about 3 pixels wide.
- RGB, rendered from the simulation after each step (measured, not
  privileged: what a camera would see). Resolution is chosen in V1 from
  sample frames and measured cost: 96 × 96 to start (the size Diffusion
  Policy used on its 2D pushing task), 128 × 128 if 96 is too coarse.
- `button_depth` is a sensor on the button, not on the robot. Whether press
  keeps it is decided with the owner before press is redone from images.
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

- Spec `pick_vision`: the `pick` spec with `object_*` removed from the
  observations and the two cameras added; same train and val seeds (so the
  state and vision policies are compared seed by seed on val), new test
  range. Demos use the state track's expert, seeds and noise, with images.
- Gate written before training, against the state-based result.
- Train, select on val, one ledgered test per method.

**Diagnostics, built and checked before training.** The evals must say *why*
a policy fails, not only how often. Each question gets its own measurement:

| Question | Measurement |
|---|---|
| Do 96 px images hold the object's position and shape at all? | Perception probe: the same encoder trained alone to predict object position (cm error) and shape (accuracy) on val frames, split by episode phase (before contact, during grasp). Runs before policy training. |
| Did the policy's encoder learn it? | The same probe on the trained policy's frozen encoder features, compared with the above. |
| Does the policy use the images, and which camera? | Val with both cameras, then each camera alone, replaced by the training-mean image. |
| Where in the task does it fail? | Every episode gets a measured phase: no contact, contact without grasp, grasped but not lifted, lifted but not held 0.5 s, object knocked off. Plus continuous metrics: closest grasp-point-to-object distance, how far the object was pushed before the grasp, first contact and grasp steps. |
| Is it vision-specific? | Paired comparison with the state policy on the same val seeds: both / only state / only vision / neither, exact McNemar test. |
| Is it localized? | Success by object and by start position (3 × 3 grid over the reset area), with CIs. |
| Learning or closed loop? | Offline action error on held-out val-seed demos, image vs state policy. Low offline error with low success points to compounding errors (DAgger territory). |
| What is it looking at? | Videos of failures with the 32 keypoints of each camera drawn on the policy's frames. |

All of these run on val. The phase labels use only the success predicate's
own pieces (contact, opposition grasp, lift, dwell), so they add no new
thresholds.

### V4+ — Press, push, place from images

Same procedure, one skill per checkpoint. Reach needs no perception (its
goal is an instruction), so it is not redone.

## A — Appearance variation (draft, for owner review)

**Why.** The end goal is a library of reusable skills that a specialized
LLM strings together through tool calls. A skill used as a tool runs in
scenes nobody recorded demos in. Today every training and test frame has the
same lighting, the same grey table and the same colour per object. So V3–V6
show the skills work in one scene, not that they are reusable. This track
varies lighting, the table and its surroundings, and object and button
appearance, measures what that does to the existing policies, and then
trains on it.

**Scope.** Lighting, table and surroundings, and object and button
appearance. Camera pose jitter and distractor objects are left out (see
"After A").

### What varies, and how

Everything is drawn once per episode at reset, from its own random stream
`default_rng([seed, 2])`. Stream 1 is the task's (target and goal), and the
workcell's stream places objects. So a seed's physical episode (object, pose,
goal, hand jitter) is identical with and without variation. That makes every
comparison paired, seed by seed, against the V3–V6 results.

| Factor | Parameters (exact ranges in `configs/appearance.yaml`; each includes today's scene) |
|---|---|
| Key light (directional) | direction: azimuth anywhere, elevation 40–85°; brightness 0.4–1.0; colour tint ±15% per channel (warm to cool) |
| Fill, camera light and ambient | fill brightness 0.2–0.8; camera light 0.1–0.5; ambient 0.05–0.35; specular 0–0.5; shadows always on (they show height) |
| Table top and wall | any hue, saturation 0–0.35, value 0.15–0.75; 70% of episodes add a texture from the bank (below), tinted by that colour |
| Trays, floor | same scheme, drawn independently (the sky is not in view of the policy cameras) |
| Object | any hue, saturation 0.2–1, value 0.25–0.95; 50% add a texture from the bank; shininess and specularity in small ranges |
| Button | as the object |

- **Texture bank:** 16 procedural textures (noise, stripes as wood grain,
  checker, speckle at several scales). They are generated from a fixed seed
  and added to the model once at load with `MjSpec`, so no image files are
  added. 12 are used for training. 4 are held out and appear only in val and
  test, so results can show whether a policy handles textures it never saw.
- **Implementation:** the appearance is drawn once per episode, then
  applied to the rendered scene (`mjvScene`) just before each frame is
  drawn. The model itself is never changed, and physics never reads the
  scene. Each factor has its own random sub-stream, so one-factor runs share
  their draws with the all-factor run.
- **Unchanged:** the camera poses, the hidden goal marker, the geometry and
  everything physical.

**Rules carried over.**
- Rule 1 (no state writes mid-episode) holds: nothing changes after reset.
- These ranges are treated like physics (rule 5): set once with a reason,
  never tuned to make a policy pass. The reason is "plausible indoor scenes",
  not "what the policy can handle".
- Measured cases where an object ends up nearly the same colour as the table
  are kept, not filtered out. Success is reported by object–table contrast
  instead.

### Decisions for the owner

1. **Object colour: fully random (recommended) or jittered around today's
   colours.**
   - Today colour is how a policy can tell objects apart, including the
     light puck from the heavy one, which differ only slightly in size (3.0 cm
     radius × 1.2 cm vs 2.6 × 1.6 cm).
   - Fully random colour forces identity to come from shape, as on a real
     robot, where mass is invisible. That may expose MLP-BC's heavy-puck
     weakness in Diffusion too.
   - For the LLM interface, colour should not be the identity handle the
     skill depends on. The caller names the target by position (see "After
     A"), so this costs nothing there.
2. **Held-out textures in val and test (recommended), or the full bank
   everywhere.**
3. **Variation level per skill spec:** `appearance: nominal | varied` in the
   task spec. Each varied spec gets a new test seed range, written down
   before looking: 340000–380000, 10000 per skill.

### A1 — Appearance variation in the simulator (no training)

- `configs/appearance.yaml` holds the ranges above.
- `dexvision/sim/appearance.py` draws and applies one appearance per
  episode. No factors (the default) is exactly today's scene.
- Task specs and data configs take `appearance: [lighting, surfaces,
  objects]`. `evaluate --appearance FACTOR...` overrides it for one val run,
  and `apps/appearance_sheet` makes the contact sheets.
- **Tests:**
  - Physics is bit-identical: same seed and actions, different appearances,
    identical state trajectory.
  - Nominal mode renders exactly today's frames.
  - Appearance is deterministic per seed and differs across seeds.
  - Nothing changes after reset.
  - Held-out textures never appear in train seeds.
- **Measure:** render time per step with variation vs nominal (it should
  be the same).

**Pass:** the owner reviews a contact sheet of about 40 varied frames per
skill from both policy cameras and agrees the scenes are plausible and the
task stays visible to a person.

### A2 — How robust are the V3–V6 policies? (eval-first, no training)

This runs the existing policies on val with varied appearance and paired
seeds, before training anything new. The questions the evals must answer:

| Question | Measurement |
|---|---|
| How much does varied appearance hurt? | Diffusion and MLP-BC from V3–V6, all factors on: paired with nominal on the same 200 val seeds (McNemar), failure phases. |
| Which factor hurts? | One factor at a time: lighting only, table only, objects only. |
| Where does it hurt? | Success binned by key-light brightness, by object–table contrast, by training vs held-out texture, and by object. |
| Is the information still in the pixels? | The scratch probe (object position, identity) trained and tested on varied frames. In particular: can the two pucks still be told apart at 96 px without colour? |
| What does the encoder track? | Keypoint videos of varied failures. |
| Sanity | The state expert is still 200/200, and appearance cannot reach it. |

**Pass:** the measurements are reported. The A3 gate is written from them
and approved before any training.

### A3+ — Train on varied scenes, one skill per checkpoint

- Same demos, same seeds, same expert. Only the rendering is varied, so the
  dataset changes and the method does not. `method.yaml` is untouched, and
  random crop is still the only augmentation.
  - Adding colour-jitter augmentation would be a method change, re-run on
    every skill. It is held back unless A3 shows varied data is not enough.
- Pick first, closely supervised, then press, push and place.
- **Reported per skill:**
  - test on varied scenes (one ledgered run per method);
  - val on varied and on nominal scenes (did robustness cost nominal
    performance?);
  - the A2 diagnostics again.
- **Storage:** about 2.4 GB of image demos for the four skills on the desktop
  (it has 831 GB free). The nominal V3–V6 datasets can be deleted, since
  they are reproducible from the seeds.

### After A (toward skills as LLM tools; not planned in detail)

What a skill needs before an LLM can call it as a tool, roughly in order:

1. **Start anywhere a previous skill could leave things:** a hand pose not at
   neutral, or an object already in hand. This needs reset distributions
   widened to the end states of the other skills.
2. **Knows when it is done:** today the evaluator's privileged success
   predicate ends an episode. A tool must report done or failed itself,
   from what it observes (a learned or measured termination signal), so
   the caller can react to failures.
3. **Target as an argument:** with distractor objects on the table, the
   caller says which one, as a position hint (`target_position`, like
   `goal_position`) that a perception or VLM layer on the LLM side supplies.
4. **Composition harness:** a scripted sequencer chaining tools, such as
   pick then place, or push then press, evaluated as whole tasks. This is
   eval-first, before any LLM.
5. **LLM orchestrator:** tool schemas (name, arguments, preconditions, done
   and failure reports) over the same harness. `AGENTS.md` lists language
   orchestration as out of scope until requested, so it gets updated when we
   get there.
- Camera pose jitter belongs with step 1 or with a real-camera setup.
