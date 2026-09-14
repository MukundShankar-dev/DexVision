# Level 5.0 — Learning, Evaluation, and Artifact Freeze

Protocol: `level5/learning-plan-v1`, September 14, 2026.

This checkpoint freezes configuration and acceptance tests only. It does not
implement loaders, train policies, evaluate held-out policies, fit normalization,
or qualify skills. Level 5.1 remains unstarted. The owner explicitly activated
Level 5 after Level 4.9 completion; the initial branch was `main` and the working
tree was clean.

## Inputs and readiness

`configs/level5/learning.yaml` binds the release, archive, completion receipt,
schemas, three frozen splits, Level 3 report, audit, and effective Level 4
configs by SHA-256. `configs/level5/SHA256SUMS` locks this plan and every Level 5
YAML. Paths resolve from the repository root; the future loader takes a separate
release-root argument. Changing data requires a versioned Level 4 amendment and
release. Changing this protocol requires a new documented version before the
affected experiment; test outcomes cannot authorize tuning.

The read-only release verifier passed with `--require-ready`: 33,350 files,
1,112 active episodes, 2,633 visual frames, and publication readiness. Dataset
digest is `5a9a96575914e1b7604c71e10edde8d50025b293d1dd22e0820b3f2c2920989b`.
The authoritative active counts are 416 training, 144 validation, and 432 test
nominal experts, plus separately identified failures/corrections. Historical
collection inventories include nine superseded originals; they are not the
learner membership authority.

The original Level 4.5B scaling report is embedded verbatim in the learning
config and its source hash is pinned. The 4/8/16-per-cell probe obtained
72/72, 72/72, and 71/72 validation successes. The largest tranche's worst cell
was 7/8, with zero invalid actions and zero safety violations. The 8-to-16
change was -0.0138889, below the frozen 0.03 material-improvement threshold;
aggregate 0.986111 and worst-cell 0.875 exceed 0.95 and 0.80. No test episodes
were inspected for that decision. The final audit reports zero procedural
exact-action duplicates and a minimum descriptor distance of
0.00028136836130646044, above its unchanged 1e-8 threshold. All final audit
issues are empty. These gates permit this freeze.

The probe used the nominal expert plus a bounded 1 mm residual. It supports
readiness under that narrow recipe; it does not establish standalone policy
sample sufficiency or a learned improvement over the expert. It also predates
nine audited replacement episodes. Their immutable lineage and fresh audit
qualify final membership; the old curve is not relabeled as a rerun on it.
The predeclared failure response to insufficient scaling, failed readiness, or
failed final similarity is to stop and request a new Level 4 version, without
using held-out policy outcomes to decide whether to collect more data.

## Evidence and model decision

The final diagnosis in `docs/level3_results.md` is pipeline-go, policy-no-go.
Level 3.4's 7 workspace plus 14 joint-limit terminations (21 total) remain
negative evidence despite finite actions and low jerk. Correct validation
selection made reach worse: zero success, 20 workspace and 15 joint-limit
violations. Button and push often failed at the first action. Base-only
ablations changed safety and button success but did not qualify a controller.
There is no measured temporal-aliasing result justifying a GRU yet.

Retain the 128/128 ReLU goal-conditioned full-action MLP as the reference
family, using the same new data/inputs/sampling where comparisons permit.
The starting standalone candidate is a 128/128 phase-conditioned MLP with
separate bounded translation, rotation-vector, wrist, and finger heads. Its
reference is the **previous applied action**, not the deterministic expert.
The previous commanded/applied actions and causal phase are explicit inputs.
Uniform cell/episode/frame sampling prevents long episodes or larger cells
from dominating the baseline. Group-normalized phase-masked loss gives base,
rotation, wrist, and finger groups equal weight. Safety checks remain outside
the model and every raw violation is reported even if application is clipped.

Short-history GRU escalation requires measured validation aliasing resolved by
history or controlled error accumulation after action/safety are fixed.
ACT-style chunking requires a written temporal/compounding-error hypothesis
and a separately frozen comparison. This checkpoint does not implement either.
No learned regrasp, drop recovery, diffusion, RL, foundation policy, or human
control is introduced. A legacy Level 3 checkpoint can only be evaluated in
5.2 after an explicit schema/goal compatibility check; an incompatible case
must be reported, not silently adapted and called the same policy.

## Executable data contracts

Each skill YAML includes the complete frozen Level 4 goal, precondition,
terminal-condition, timeout, and failure contract. Planner-facing place takes
a target, while its executable goal binds the unique held object as context;
press retains its frozen depth/state parameters. String ids validate requests
and select entities, but never become ordinal or trainable instance-id inputs.
Optional dial is absent because Level 4 did not promote it.

Resolve robot columns by the saved layout names. The policy has exactly 183
common input values in the ordered `feature_groups`, plus each skill's numeric
goal vector (reach 9, pick 11, place 12, push 6, press 2). Inputs include actual
base free-joint pose/velocity, 24 named hand positions and velocities, previous
actions/safety masks, causal phase, and one selected entity. Rotations use the
first two matrix columns as six values. Zero unavailable velocities and mark
their validity false. Entity classes use the frozen five-way one-hot vocabulary.
Task success and safety metrics belong to the supervisor, not the policy input.
Relations are derived causally from the typed world-state interface.

The saved `base_position`/`base_orientation` layout contains commanded targets,
not measured base pose. Use `rh_base_freejoint` qpos for measured pose. The
saved finger layout contains `start_button_joint`; filter to names beginning
`rh_` for the 24 hand joints. Button depth is a task fixture measurement in
metres, not a hand joint in radians. Do not feed all 168 robot-state columns,
all background objects, tracking quality, or retrospective annotations.

Saved state row t is the result of action t. The feature for action t uses
state row t-1 (or the recreated reset state at t=0), prior-action arrays at t,
and safety mask t-1. Reconstruct phase before action t through the causal state
machine; never use a transition caused by action t as its own input. The loader
must preserve frame timestamps and reject skew beyond the frozen contract.
Only training-manifest normalization inputs intersected with each skill's
eligible interval may fit continuous observation/goal means and population
standard deviations. No normalization is fitted in 5.0.

Targets preserve the complete 27-field applied layout. The standalone model
predicts 26 values: translation 3, rotation vector 3, wrist 2, fingers 18.
Translation/joint deltas use each recorded maximum change per sample; rotation
uses a capped 0.139626 rad vector, left-composed with the prior quaternion.
Normalize and maintain quaternion sign continuity. Irrelevant groups hold their
previous applied values. Record requested, commanded, applied, prior actions,
safety masks/reasons, phase, and timestamps. Every invalid/workspace/joint
violation fails qualification; clipping cannot hide it.

The executable release runs 17 simulator steps of 0.002 s per action (0.034 s),
although the nominal recorder setting says 30 Hz. The protocol uses the measured
0.034 s interval for replay/control. Jerk retains the project's normalized
third-difference per-control-sample convention, not SI jerk per second cubed.

## Training and evaluation

All hyperparameters, seeds (5001/5002/5003), 100-epoch cap, Adam settings,
batch size, stopping rule, and candidate epochs are in the learning config.
Every seed retains its own selected checkpoint; no best-seed reporting.
Offline validation loss supplies early stopping. Candidate selection then uses
only the explicit validation rollout matrix, ordered by safety violations,
success, worst-cell success, normalized terminal error, offline loss, and epoch.
The candidate set includes each reached tenth epoch, best offline epoch, and
last executed epoch. Deduplicate candidates. Test is run once per selected
checkpoint per track; no test reranking or hyperparameter search.

Matrices are explicit cell records with aligned episode-id, reset-seed, and
start-frame arrays. Selection is mechanical: lexicographic episode ids within
each split-owned cell, with max(3, ceil(30/cell count)) examples per cell.
Only frozen nominal release members are eligible. The exact metadata restores
goals, variation, dynamics, and initial state, including replacement provenance.
Recreate the source task and replay only actions before start_frame to establish
the standalone skill precondition. This is disclosed scripted setup, excluded
from policy success and policy jerk. No prefix or demonstration action is
available to the standalone policy after its start. Reset terminal dwell at
that boundary; reject and save an invalid precondition without trying another
frame. This is standalone skill testing, not a composed pilot claim.

Pick starts at acquire; place starts at transport. Push and press start before
the first push/fixture-contact action inside their saved segments, satisfying
the approach-envelope contract. Training and expert statistics use that same
interval intersection. No new labels or data are written.

| Skill | Training expert intervals | Validation resets | Test resets | Expert p95 jerk | Frozen maximum mean jerk |
|---|---:|---:|---:|---:|---:|
| Reach | 80 | 30 | 30 | 0.0190957903481 | 0.0238697379352 |
| Pick | 144 | 30 | 54 | 0.00250200504018 | 0.00312750630023 |
| Place | 144 | 30 | 54 | 0.00841966173101 | 0.0105245771638 |
| Push | 96 | 30 | 32 | 0.00164420371487 | 0.00205525464359 |
| Press | 96 | 30 | 30 | 0.00333333333333 | 0.00416666666667 |

Pick/place intervals share 144 parent training episodes; they are not counted
as additional recordings. Jerk is mean L2 third difference of all 27 applied
columns normalized by declared field ranges. Compute one value per training
expert interval, then the linear-interpolated 95th percentile; gates equal
1.25 times that percentile. Per-episode metrics are included for independent
archive readback. The unrounded YAML values are authoritative.

Every physical reset crosses all four frozen render conditions in **both**
tracks, so state and perception pair exactly. These repetitions do not count
as additional independent resets. Test rollouts per seed/track are therefore
120 reach, 216 pick, 216 place, 128 push, 120 press. Report each seed, condition,
cell, object family/instance, target family/id, session, and boundary class.
Every rejected precondition remains in the denominator. Each seed must pass;
pooling cannot rescue a failing seed or required family.

All roadmap success/safety gates are retained. Mean jerk is over all scheduled
rollout metrics per seed/track; short or rejected trajectories cannot bypass
success gates. Empty required groups fail. A state-only pass remains
experimental until perception, compatibility, and later manual gates pass.
Unsafe/incompatible/unrunnable artifacts are failed and disabled. Every result
has one of the five frozen terminal statuses, provenance, final metrics,
trajectory paths and explicit failure reasons.

Perception uses the fixed Level 4 calibration and conditions, robot
proprioception and frozen target geometry, but inferred goal-object identity
and pose. No simulator-truth fallback is permitted. Missing/stale/ambiguous
perception rejects or aborts explicitly. The unchanged detector/pose gates
are in evaluation_common.yaml; no perception model is built here.

## Claims, artifacts, and limitations

Freeze four comparison arms: unmodified expert, standalone policy, expert plus
zero residual, expert plus learned residual. An expert residual is opt-in at
5.8; nominal expert requested/applied actions often coincide, so zero targets
must not be relabeled as useful learned corrections. Expert-assisted learning
requires at least a 0.05 absolute success gain OR a 10% reduction in paired
normalized terminal error, without reducing success and while passing every
safety/worst-family gate. This must hold on validation and the single declared
test comparison. Zero-error expert baselines cannot demonstrate relative error
improvement. No improvement means `scripted_expert` with an experimental learned
component; residual assistance never proves standalone learning.

Outputs use immutable run directories under `outputs/level5/`, recording all
seeds, candidates, inputs, environment, optimizer/RNG/selection state, failures,
trajectories and SHA-256. Resume requires identical input digests and sampler
state. Never silently commit working outputs. Future publication uses versioned
`releases/level5-v1` metadata with checked LFS limits or explicitly selected
immutable storage; no binary or release publication occurs in 5.0.

Claims remain bounded to the released rigid-object workcell, isolated
pick/place fixtures and assisted orientation, supported push cells, and one
camera. There is no real-world, cross-camera, arbitrary-object, or human-control
claim. Frozen unsupported cells remain unsupported. Data deficiencies trigger
new versions, never edits to v1 or test-guided recollection.

## Verification

Run from the repository root:

```bash
conda run -n dexvision python -m dexvision.apps.verify_dataset_release --release-dir datasets/level4-v1 --require-ready
conda run -n dexvision pytest -q tests/test_level5_learning_plan.py tests/test_level5_evaluation_configs.py
conda run -n dexvision ruff check tests/test_level5_learning_plan.py tests/test_level5_evaluation_configs.py
conda run -n dexvision ruff check dexvision tests
conda run -n dexvision pytest -q
git diff --check
```

Tests validate readiness rejection cases, immutable hashes, named schemas,
whole-session/split isolation, numerical gates, exact reset membership, causal
selection rules, and expert jerk. The archive-readback test recomputes jerk and
checks reset metadata directly from the immutable release; it explicitly skips
only when the LFS payload is unavailable. No GUI, GPU, webcam, or model download
is required. No manual verification is required for checkpoint 5.0.
