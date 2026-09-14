# Level 4 v1 handoff to Level 5

This handoff freezes data interfaces only. Level 4.9 remains pending publication
and the owner's clean-directory restore confirmation. No Level 5 training,
policy qualification, normalization fitting or runtime implementation is included.

## Membership and split ownership

The authoritative inputs are `datasets/level4-v1/splits/train.json`,
`validation.json`, and `test.json`, copied byte-for-byte from the passing
`outputs/level4/audit_v4_final` audit. The dataset digest is
`5a9a96575914e1b7604c71e10edde8d50025b293d1dd22e0820b3f2c2920989b`.

There are 1,112 active episodes: 486 training, 174 validation, and 452 test.
Each episode has a genuine session; all 1,112 sessions belong wholly to one
split. There are 992 nominal expert successes, 90 ordinary failures, and 30
corrective interventions. All 74 required coverage cells pass. Complete
pick/place sequences contain multiple skill segments but count as one episode.
The nine superseded originals are excluded explicitly by the audit; the local
working data, excluded diagnostic attempts and prior reports remain unchanged.
Select episodes from these manifests, never by scanning all local directories.

## Exact layouts and goals

`schemas.json` enumerates complete named observation/action layouts and their
per-episode binding. Each frozen split row also contains the full
`observation_schema`, `action_schema`, `schema_versions`, `typed_goal`,
`phase_intervals`, derived `segments`, and `training_target_interval`.
`source_path` is relative to restored `data/demos/level4`.

The saved observation schema is `level2/observation-layout-v2`. The older
requirements file's conceptual `level4/observation-v1` label does not replace
this executable per-episode layout. Resolve columns using `layouts` and the
saved names, source arrays, column ranges/indices, shapes, dtypes, units and
frames; do not infer offsets from vector width. The 168-column `robot_states`
array includes scene joints. Using it indiscriminately would expose held-out
background objects: learner inputs must select named robot fields and only the
task-relevant entity and typed goal. The same exclusion applies to object-state
arrays. Background simulator state is retained as archival replay truth.

Every saved action has 27 fields under `level1.13/full-action-v1`:

| Columns (end exclusive) | Meaning |
|---|---|
| 0:3 | World-frame base target position x, y, z in metres |
| 3:7 | Normalized, sign-continuous base target quaternion w, x, y, z |
| 7:27 | Named actuator targets, including wrist and finger fields |

The exact actuator order is `rh_A_WRJ2`, `rh_A_WRJ1`, `rh_A_THJ5`, `rh_A_THJ4`,
`rh_A_THJ3`, `rh_A_THJ2`, `rh_A_THJ1`, `rh_A_FFJ4`, `rh_A_FFJ3`, `rh_A_FFJ0`,
`rh_A_MFJ4`, `rh_A_MFJ3`, `rh_A_MFJ0`, `rh_A_RFJ4`, `rh_A_RFJ3`, `rh_A_RFJ0`,
`rh_A_LFJ5`, `rh_A_LFJ4`, `rh_A_LFJ3`, `rh_A_LFJ0`.

Requested, commanded, applied and prior actions are separate full-width arrays.
`actions.npy` mirrors `applied_actions.npy`. Per-field safety masks/reasons,
control interval, timestamps and causal phase relevance are preserved under
`level4/request-command-apply-v1`, `level4/action-safety-v1` and
`level4/causal-phase-v1`. Bounds, units and rate limits are frozen in each
episode's `action_contract` and the archived dataset configuration. Never turn
unsafe requested commands into expert targets.

Typed goal contracts are frozen in `configs/level4_dataset.yaml` under the
five required `skills`; resolved values are stored in each split row:

| Skill or recorded sequence | Resolved goal |
|---|---|
| reach_object | entity_id and world-frame approach_pose |
| pick_object segment | object_id from the parent pick/place sequence |
| place_held_object segment | target_id from the parent pick/place sequence |
| pick_place_sequence | object_id and target_id |
| push_object_to_target | object_id and target_zone |
| press_button | button_id, target_press_depth_m, target_pressed_state |

Segment intervals are start-inclusive/end-exclusive. Preserve the saved
selection intervals when exposing nominal and corrective targets. An ordinary
failure has no expert target interval; a correction's pre-intervention failure
prefix remains distinct from its recovery targets and original outcome.

## Normalization and visual supervision

Only the 416 explicitly listed training expert episodes/intervals in
`train.json:normalization_inputs` may fit normalization. Validation/test lists
are empty. Exclude failures, corrections, background entities and retrospective
annotations from those inputs. No statistics are fitted by this release.

The complete single-camera export contains 2,633 RGB/mask frame pairs from 64
active nominal source episodes. `data/visual/level4/frames.jsonl` binds labels,
visibility, boxes, metric poses, ids, source indices/timestamps and calibration.
`calibration.json`, source checksums, condition/split assignments, exclusion
reports, contact sheets and the user's review receipt accompany the images.
Invisible entities retain poses with null boxes and empty visible masks.
The source annotation format uses world-frame translation and rotation matrices;
no amodal segmentation or real-camera provenance is claimed. Failures and
corrections have no rendered stream in this release.

## Versions, limitations and recovery

The audit source commit is `41a5d9998058178eebc8a4f40dd633a621de04fe`.
The archive carries source/config hashes, the frozen audit, replacement receipts
and their historical recording snapshots. Some original episodes say
`code_version: working-tree`; the release source pin must not be presented as
proof of their exact original recording commit. Effective recording configs
and available snapshot hashes remain their original provenance evidence.
The release tooling, handoff and license additions are independently pinned by
archived file hashes. The archive, metadata and individual payload files have
separate SHA-256 checks. The Level 2 release remains independently retrievable.

Deterministic simulator experts and one fixed camera support bounded rigid-object
skills only. There is no human, cross-operator, cross-camera or real-world
transfer claim. Pick/place uses isolated fixtures and assisted orientation.
Eight unsupported push cells remain explicitly excluded; shared training-pool
objects and goals intentionally recur across sessions. Exact duplicate rendered
frames were excluded, never copied to repair coverage. See the unchanged audit
report for counts, source lineage, rejected attempts and claim boundaries.

Use a clean, nonexistent restoration directory and require every archive/file
checksum to pass. If a transfer fails, retrieve the same immutable LFS object
again and restore into another new directory. Never overwrite the v1 archive,
accepted source episodes, frozen splits, or the Level 2 archive. Intentional
changes require a new release version and a new audit. LFS storage counts all
retained object versions; downloading consumes the owner's bandwidth. Budget
at least 4 GB of free disk for the compressed archive, restored payload and
verification overhead. Keep a verified offline copy; GitHub's account-wide
remaining quota is not exposed by the current credentials.
