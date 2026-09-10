# Level 4.8 Dataset Audit and Frozen Split Manifests

The audit covers the active Level 4 nominal and failure/correction dataset and
its completed Level 4.7 visual export. Source episodes, historical attempts,
quarantines, visual artifacts, and the immutable Level 2 release remain
read-only. Level 4.9 packaging has not started.

## Reproduce

From the repository root, use the dedicated environment:

```bash
conda activate dexvision
python -m dexvision.apps.audit_level4_dataset --config configs/level4_dataset.yaml --splits configs/level4_splits.yaml --dataset-dir data/demos/level4 --output-dir outputs/level4/audit
conda run -n dexvision pytest -q tests/test_level4_dataset_audit.py tests/test_level4_split_audit.py
conda run -n dexvision ruff check dexvision tests
conda run -n dexvision pytest -q
```

An existing output directory is never overwritten. For a repeat audit, use a
new directory such as `outputs/level4/audit_v3` (both `audit` and `audit_v2`
already exist). Four spawned worker processes
perform independent headless replays; no GUI, webcam, GPU, training, or human
control is required. Exit status 0 means all audit gates passed, 1 means a
completed audit found issues, and 2 means an input/dependency could not be read
or validated. A failed completed audit writes `collection_amendment_v1.json`;
it does not modify accepted data or authorize additional collection.

## Files changed for this checkpoint

| File | Purpose |
|---|---|
| `dexvision/evaluation/dataset_audit.py` | Schema, fresh replay, coverage, visual truth and amendment orchestration |
| `dexvision/evaluation/split_audit.py` | Split isolation, content hashes and diagnostic/frozen manifests |
| `dexvision/apps/audit_level4_dataset.py` | Headless audit CLI with explicit exit statuses |
| `configs/level4_splits.yaml` | Digest-bound split and normalization policy |
| `tests/test_level4_dataset_audit.py` | Read-only boundaries, schema/replay rejection and annotation truth |
| `tests/test_level4_split_audit.py` | Leakage, checksums, normalization and failed-manifest regressions |
| `tests/test_level3_results.py` | Replace obsolete 4.6/4.7 status assertions with active-heading agreement |
| `tests/test_roadmap_docs.py` | Check current checkpoint ordering without pinning obsolete completion ids |
| `docs/module_contracts.md` | Level 4.8 module contract |
| `docs/progress_level_4.md` | Implementation evidence and incomplete checkpoint status |
| `docs/CURRENT_STATUS.md` | Keep 4.7/4.8 current and record audit blockers |
| `docs/level4_dataset_report.md` | Reproduction, artifacts, measured results and limitations |

## Generated evidence

All generated evidence lives under the selected audit directory:

- `report.json`: episode and segment counts by skill, phase, session, object,
  target, and outcome; all 74 coverage cells with separate split minima;
  fresh quality/replay results; nominal independence/scaling evidence;
  failure/correction coverage; visual coverage/integrity; rejection and
  quarantine reasons; source/config/code digests and limitations.
- `train.json`, `validation.json`, `test.json`: deterministic complete-episode
  ownership, source lineage, exact source-file SHA-256 values, episode digests,
  named observation/action schemas, phase and skill intervals, training-target
  intervals, and image assignments/checksums. Each manifest carries a canonical
  JSON digest excluding its own `manifest_digest` field.
- `file_sha256.json`: SHA-256 for the other generated audit files. This index
  excludes itself; it is not a dataset-release checksum or release archive.

The dataset digest hashes the canonical mapping of episode id to episode digest.
Each episode digest hashes its relative-file-to-SHA-256 mapping. Visual payload,
configuration, and implementation digests are reported separately. Paths inside
episode inventories are relative, so copying the dataset does not alter its
content digest.

## Acceptance and learning boundaries

Nominal expert acceptance is re-evaluated using the established scripted-expert
contract: full schema/provenance, causal phases, timestamp alignment, zero safety
interventions, bounded non-target disturbance, matching reset, complete headless
replay, and independently recomputed terminal metrics. Audited phase disagreement
must also remain within the frozen threshold. Saved acceptance booleans alone
cannot pass the audit. Legacy camera-confidence filters are not evidence of
scripted-expert quality.

Ordinary failures remain evidence with no training-target interval. Correction
records retain their source failure, original outcome, and intervention interval.
The correction summary verifies unchanged prefixes, conditional provenance,
abort-only unsafe failures, and fresh successful pick/place replays. Correction
target intervals must contain no safety-mask violations. Baseline and optional
correction training inherit the same frozen split ownership.

Only explicitly listed **training expert-success episodes and frame intervals**
are normalization inputs. Validation, test, corrections, ordinary failures,
quarantined attempts, and retrospective annotations are excluded. No statistics
are fitted here. Future learners must resolve the saved named schemas and use
only task-relevant entity state and typed goals; full archival background object
state is excluded. Raw episodes intentionally retain simulator truth for all
objects, including held-out background identities. The audit makes no claim
that feeding unrestricted raw state to a learner would preserve isolation.

Shared training-pool objects, the common button, and non-held-out goals recur
across splits by design. Held-out object instances (`block_large`,
`cylinder_tall`, `puck_heavy`) and `return_bin_right` remain test-only task
conditions. Complete sessions and all linked nominal/failure/correction records
must retain one split. Exact nominal action or initial-state duplicates cannot
cross splits. Rendered held-out background annotations/pixels are excluded from
non-test exports; decoded RGB hashes cannot cross splits.

Every RGB/mask pair is decoded and checked against saved hashes, mask extents,
visibility, stable ids/classes, the restored simulator poses, source state/action
timestamps, and the fixed camera calibration. The original Level 4.7 contact-sheet
approval remains immutable and its artifact hashes are verified.

## Scope and known biases

The dataset uses deterministic simulator experts, task-specific approach
geometry, isolated pick/place fixtures and an assisted held orientation. It does
not establish arbitrary-object grasping, natural clutter robustness, human or
cross-operator imitation, real-world transfer, or cross-camera generalization.
The visual dataset is a compact nominal subset; failures and corrections do not
claim an RGB stream. Phase-membership counts overlap; segment counts never add
to the recorded-episode total. All unsupported frozen cells remain explicit in
the report's `coverage_exclusions` rather than being filled with duplicate data.

The custom workcell asset license remains unspecified. The owner must resolve
it before Level 4.9 release packaging. This checkpoint does not publish an
archive, change collection parameters, train a model, or qualify a skill.

## Results

The first audit at `outputs/level4/audit` completed with 1,112 active episodes:
992 saved nominal expert successes, 90 ordinary failures, and 30 corrections.
It preserved/excluded 2,734 historical attempts (2,709 quarantined diagnostics
and 25 other unaccepted/non-scripted attempts). All 30 correction outcomes and
the 2,633-frame visual export passed independent verification.

| Split | Saved nominal successes | Ordinary failures | Corrections | All active |
|---|---:|---:|---:|---:|
| Train | 416 | 50 | 20 | 486 |
| Validation | 144 | 20 | 10 | 174 |
| Test | 432 | 20 | 0 | 452 |

The large test partition reflects the frozen held-out object/goal matrix;
pick/place supplies 288 of its 432 nominal episodes. The dataset is not balanced
by skill or split, and there are no test-owned corrective interventions. Segment
counts describe qualified nominal skill segments; corrections are identified
separately by bounded intervention intervals, not silently counted as additional
nominal episodes or automatically segmented policies. Outcome counters preserve
the saved labels; `audit_passed`, each record's fresh quality/replay result, and
`fresh_episode_audit_fail_ids` separately identify recomputed failures.

Fresh replay rejected `level45b_000293` in training cell
`pp_puck_light_return_bin_left`: terminal task success failed and the saved
success label disagreed. A focused independent replay acquired the puck but
never qualified placement: it settled 0.0704992619 m from the target, outside
the 0.025 m tolerance, with no safety intervention and matching reset metadata.
This leaves 15 freshly qualified nominal episodes against that cell's frozen
minimum of 16; the original accepted files and thresholds remain unchanged.

### Isolated puck replay investigation (September 10, 2026)

The saved terminal task state reports successful placement, whereas a fresh
replay does not. Comparing all 329 saved action rows against the recorded
generalized positions shows divergence during approach/contact, before the
orientation-held lift. At the first `acquire` row (index 80), the maximum
absolute generalized-position difference is 0.0056439809. Replaying those same
actions through the recorder's `_step_scripted_workcell` path produces the
same differences as `replay_loaded_demo`; removing the extra replay quaternion
normalization also leaves the result unchanged. These checks did not establish
a stepping or normalization defect that can safely be patched.

Physics reconstruction remains an unresolved hypothesis. The episode stores
`code_version: working-tree`, a generic config version, baseline contact
friction and sampled procedural multipliers, but no exact recording-time
configuration snapshot or resolved physics snapshot. Current reconstruction
selects a 1.8 puck friction base multiplier. Diagnostic replays using the other
declared schedule values (1.5 and 1.7), a unit base, or the saved baseline
contact friction did not reproduce the recorded generalized-position
trajectory. None of these diagnostic settings was adopted. A successful
terminal outcome under altered physics alone would not prove compatibility.

No replay compatibility fix has been verified. The episode remains rejected;
recovering the recording-time configuration or collecting a separately
versioned, independently qualified replacement is still necessary to resolve
it. This isolated investigation changed no simulation code, collection
configuration, episode, review, visual export, or existing audit artifact, and
did not address button duplicates. The 26 checkpoint regression tests pass.

Reproduce the single-episode qualification without collecting or writing data:

```bash
conda run -n dexvision python -c 'import json; from dexvision.evaluation.level4_expert_audit import audit_scripted_episode; result = audit_scripted_episode("data/demos/level4/level45b_train_a_000293_pp_puck_light_return_bin_left/episode_000001", config_path="configs/level4_dataset.yaml", workcell_config="configs/workcell.yaml"); print(json.dumps(result.to_dict(), indent=2)); raise SystemExit(0 if result.accepted else 1)'
```

Pass requires `accepted: true`, recomputed success and label agreement, with no
rejection reasons. The current expected result is exit 1 with
`recomputed_task_failure` and `operator_label_disagreement`. This is an automated
gate; no manual verification can substitute for it.

### Remaining audit gates

Twenty immutable button anchor episodes form two identical action-trajectory
groups spanning train, validation, and test. Eight non-training episodes match
training trajectories. This is action-stream duplication, not a claim that every
raw observation byte is identical. The report identifies every involved episode
and digest, so the overlap cannot be hidden by random background reset metadata.

The final audit at `outputs/level4/audit_v2` completed with exit status 1.
It reports 11 diagnostic messages describing the failed replay, its coverage
shortage and the eight cross-split matches in two action groups. Its manifests
explicitly carry `dataset_audit_passed` and a diagnostic-only status when any gate fails.
Failed episodes contribute no training target, normalization input, or qualified
cell count. The versioned amendment records affected cells, rejected episode ids,
and cross-split duplicate groups. It requires an append-only plan and distinct
replacement evidence, preserving all accepted/raw data and frozen thresholds.
That audit performed no replacement collection or split reassignment. The later
user-authorized puck-only amendment is documented below.

Level 4.8 remains incomplete. No manual verification is required; Level 4.9
must not start until a versioned amendment resolves these automated data gates.


Automated implementation checks: 26 focused checkpoint tests, 17 status/docs
regressions, repository-wide Ruff and whitespace checks pass. The final full
suite completed with **622 passed, 1 skipped in 487.05 seconds** after correcting
two obsolete status assertions. The skipped test is
`tests/test_render_annotations.py:62`: offscreen OpenGL was unavailable because
of an invalid CoreGraphics connection. The Level 4.8 saved-frame decoding and
simulator-truth audit needs no offscreen renderer and passed in full. Independent
read-back verification passes for every audit-file checksum, canonical manifest
digest, all 1,112 source episode inventories and the complete visual export.
Both audit runs have the same source dataset digest:
`4c9c5d0aa7a63a18aeb4b0417eee1ddc047f70d374d94e99743d561d45216361`.
The final manifest lists 415 train-only normalization episodes and none from
validation or test; every manifest remains diagnostic-only until the complete
dataset audit passes.


## Commands executed

The two read-only dataset runs used the following command with output directories
`outputs/level4/audit` and `outputs/level4/audit_v2`, respectively. Both returned
status 1 for the recorded data blockers; neither directory was overwritten.

```bash
conda run --no-capture-output -n dexvision python -m dexvision.apps.audit_level4_dataset --config configs/level4_dataset.yaml --splits configs/level4_splits.yaml --dataset-dir data/demos/level4 --output-dir outputs/level4/audit_v2
conda run -n dexvision pytest -q tests/test_level4_dataset_audit.py tests/test_level4_split_audit.py
conda run -n dexvision pytest -q tests/test_level3_results.py tests/test_roadmap_docs.py
conda run -n dexvision ruff check dexvision tests
conda run --no-capture-output -n dexvision pytest -q -rs
git diff --check
```

The first regression attempts exposed two status-only assertions still pinned
to Level 4.6/4.7. They now verify that the current headings exist and are ordered
in the active progress file while retaining the historical completion checks.
The independent integrity check additionally rehashed every active episode and
visual artifact, verified generated checksums and canonical manifest digests,
and checked that the failed episode has no training or normalization interval.

## Puck-only replacement amendment (September 10, 2026)

The user authorized collecting one replacement after the isolated replay
investigation could not establish a safe compatibility patch. The frozen plan
is `configs/level4_puck_replacement_v1.yaml`. It selected unused seed 480001,
training cell `pp_puck_light_return_bin_left`, a new session, and the existing
nominal scripted expert before collection. This is a new seeded nominal trial,
not a reconstruction of the old procedural assignment; no controller, physics,
success threshold, held-out configuration, or original collection plan changed.

The first attempt, `level48_puck_v1_000001`, recorded 332 frames and passed an
independent replay of its saved actions: complete pick and terminal place,
matching reset and labels, zero safety violations, and maximum neighbor
movement 1.4569e-12 m. Its original recorder-created session and metadata are
retained. The new review and qualification receipt were written once.
`outputs/level4/puck_replacement_v1/receipt.json` binds the plan and episode
inventory, qualification result, code/config snapshot, simulator asset hashes,
and Python/MuJoCo/NumPy versions. Recording code and YAML files are preserved
under that directory's `recording_snapshot/`.

The optional `replacement_plan` in `configs/level4_splits.yaml` activates the
amendment. The audit checks both episodes' hashes, the plan/receipt/config
binding, matching cell and training ownership, and the new seed/session/operator.
Missing, duplicated, changed, or cross-split evidence stops the audit. The
replacement still receives the normal fresh replay and quality checks on every
audit; the receipt cannot override a failed replay. Its manifest explicitly
records `replaces_episode_id`. The old `level45b_000293` remains unchanged on
disk with its original review and an explicit, digest-bound exclusion from the
active release candidate. It contributes no training or normalization frames.
The historical collection summary continues to inventory preserved originals;
the active manifests and fresh coverage counts use the amended membership.

Reproduce qualification (read-only, exit 0 required):

```bash
conda run -n dexvision python -c 'import json; from dexvision.evaluation.level4_expert_audit import audit_scripted_episode; a = audit_scripted_episode("data/demos/level4/level48_puck_v1_train_000001/episode_000001", config_path="configs/level4_dataset.yaml", workcell_config="configs/workcell.yaml"); print(json.dumps(a.to_dict(), indent=2)); raise SystemExit(0 if a.accepted else 1)'
conda run -n dexvision pytest -q tests/test_level4_puck_replacement.py tests/test_level4_dataset_audit.py tests/test_level4_split_audit.py
```

Exact collection command used (do not rerun into the existing session):

```bash
conda run -n dexvision python -m dexvision.apps.record_demo --task level4_workcell --skill pick_place_sequence --source scripted --goal-condition-id pp_puck_light_return_bin_left --task-seed 480001 --session-id level48_puck_v1_train_000001 --operator-id scripted_puck_replacement_v1 --session-split train --level4-dataset-config configs/level4_dataset.yaml --level4-dataset-dir data/demos/level4 --workcell-config configs/workcell.yaml --episode-id level48_puck_v1_000001 --enforce-frozen-cell-owner --print-interval 1000
```

The complete audit rerun uses `--output-dir outputs/level4/audit_v3`; existing
audits remain immutable. For another run, choose a new output directory.
The focused suite passes 35 tests and repository-wide Ruff passes. No manual
verification is required for 4.8. Button duplicates are outside this amendment;
4.8 must remain incomplete until those separate leakage failures are resolved.

Final amendment verification: `outputs/level4/audit_v3` completed with **1,112 /
1,112 fresh episode checks passing**, **74 / 74 coverage cells passing**, and
**16 / 16 qualified episodes in the puck training cell**. All 30 corrections
and the full visual audit pass. The eight remaining diagnostics are exactly
the original cross-split button matches in two unchanged action groups; exit 1
is expected until that separate issue is resolved. There are no remaining
failed-episode ids or coverage shortages. All split manifests remain
diagnostic-only. The new dataset digest is
`37436d7d4225086ac59d7d148c7cc7fa7359996f913bfb3b4271eb784d1aecfe`.
The training normalization list contains 416 episodes, including all 332 frames
of the replacement and no frames of the preserved original.

The full suite passed with **631 passed, 1 skipped in 523.19 seconds**. The
skip is the platform-dependent offscreen OpenGL test. The 17 status/documentation
regressions also pass. Independent read-back verifies all manifest digests,
audit artifact checksums, replacement lineage and source hashes. Every one of
the previous 1,112 active episode inventories, all pre-existing session entries,
all other root provenance files, audit v2 artifacts, and the complete visual
export remain unchanged. The only root session-manifest change is the new
recorder-created session. Replacement plan, recording snapshot and simulator
asset hashes match the saved receipt. No manual verification is required.

Additional files for this isolated amendment: `configs/level4_puck_replacement_v1.yaml`,
`tests/test_level4_puck_replacement.py`; amendment handling in
`dexvision/evaluation/dataset_audit.py`; activation in `configs/level4_splits.yaml`;
and status, progress, module-contract and report updates. Data and qualification
artifacts remain ignored working data; no dataset release or commit was created.
