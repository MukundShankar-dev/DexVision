# Versioned Dataset Releases

Level 4 v1 is complete and independently retrievable through Git LFS as
`level4-v1.tar.gz`. Source and release metadata are published. Frozen metadata,
schemas, splits, licenses, completion evidence and retrieval instructions are in
[`level4-v1/README.md`](level4-v1/README.md). Level 5 is active; its additive unassisted pick amendment is described below.
The Level 2 release described below remains unchanged.

The editable working tree under `data/demos/` remains ignored by Git. Published
dataset snapshots are immutable archives in this directory and are stored with
Git LFS.

The Level 2 snapshot contains:

```text
data/demos/raw/
data/demos/rejected/
data/demos/reports/
```

It excludes temporary staging data and the one-off top-level recorder smoke
directory.

This snapshot is the immutable input to the Level 3 learning-feasibility work.
It is not the project's final comprehensive skill dataset and must not be used
to claim cross-session, cross-object, cross-camera, or open-world
generalization. Level 4 has published a separate versioned release with genuine
session ids, broader objects/goals, grasp-lift-place and recovery coverage, and
visual grounding data. Level 5 will train and qualify the full skill set from
that frozen release. The Level 2 archive must not be overwritten when the
Level 4 release is created.

`dexvision_level2_v1_manifest.json` records the archive digest, task counts,
frame counts, report digests, and the known absence of recording-session ids.

After cloning, install Git LFS, download the archive, verify it, and extract it
from the repository root:

```bash
git lfs install
git lfs pull
shasum -a 256 -c datasets/dexvision_level2_v1.tar.gz.sha256
tar -xzf datasets/dexvision_level2_v1.tar.gz
```

Do not modify an existing release archive. Create a new version when the
dataset intentionally changes.

For a Level 4 release, keep manifests, checksums, and split metadata in Git. A
bounded archive may use Git LFS; if synchronized image streams exceed hosting
quotas, store the immutable payload in documented versioned artifact storage
and keep its retrieval instructions and cryptographic digests here.


## Level 5 unassisted pick amendment v1

`level5-pick-unassisted-v1.tar.gz` is an additive, immutable **state-only** release
for owner-authorized checkpoint 5.3A. It contains 144 training and 48 validation
standalone pick trajectories from the exact-loop deterministic teacher V3, with
no object-orientation hold or object-pose writes after reset in scratch or live
execution. Every frozen collection attempt was reported successful; none was
replaced. The later independent saved-state audit verifies only **191/192**:
one final speed exceeds 0.0200 m/s after recomputation. V1 is preserved as
superseded evidence and must not be described as fully physically verified.
The corrected V2 collection repeats the same cases without replacement.

The six named instances are already exposed in development and are intentionally
included in the new training amendment. This does not alter historical Level 4
split ownership or claim unseen-instance/family generalization. The new whole
simulator sessions are named per split/cell in the manifest (18 training sessions,
six validation sessions); they are deterministic simulation runs, not human,
camera, cross-operator or cross-camera sessions. The source scene is the existing
pick/place task, but recordings terminate at standalone pick success and contain
no completed-place labels or RGB/perception supervision.

The archive contains only canonical `datasets/level5-pick-unassisted-v1/episodes/`
payloads: metadata, result receipts, and compressed causal observations, actions,
safety masks and physical states. It contains no staging directory, training
cache, temporary diagnostic script, model checkpoint or source asset copy.
The original `data/demos/` and all prior release archives remain unchanged.
`level5-pick-unassisted-v1_manifest.json` binds each payload digest, plan, teacher
recipe/feasibility hashes, episode/session assignments, all attempts and physical
parameter/settings provenance. Source simulator assets retain their existing
licenses; they are not redistributed in this archive.

```bash
git lfs install
git lfs pull
shasum -a 256 -c datasets/level5-pick-unassisted-v1.tar.gz.sha256
# Optional inspection extraction; training reads the verified archive directly.
tar -xzf datasets/level5-pick-unassisted-v1.tar.gz
```

The original Level 4 release metadata/assets remain required. Exact collection,
verification, training and qualification commands are in
[`../docs/level5_pick_repair.md`](../docs/level5_pick_repair.md). Never overwrite
this archive. A changed teacher, data selection or content requires a new version.
No remote upload or dataset commit is implied by local packaging.

## Level 5 unassisted pick amendment v2

`level5-pick-unassisted-v2.tar.gz` preserves the same frozen 144 training and
48 validation reset cases and split ownership as V1, using new episode/session
identifiers. Teacher V4 recomputes physical success from post-integration state
on a separate MuJoCo data copy. This fixes the cached-state discrepancy above
without changing live physics, thresholds, fallback order or replacing resets.
All 192 collection attempts report success and zero safety violations; independent
saved-state verification is recorded in the 5.3A method report. V1 stays immutable.

The archive and `level5-pick-unassisted-v2_manifest.json` use the same canonical
three-file episode format and provenance disclosures as V1. V6 and V7 policy recipes
pin both hashes and require the complete passing source audit before training.
This remains exposed-instance, state-only standalone pick data, with no place,
vision, human-demonstration or arbitrary-start qualification claim.

```bash
git lfs install
git lfs pull
shasum -a 256 -c datasets/level5-pick-unassisted-v2.tar.gz.sha256
tar -xzf datasets/level5-pick-unassisted-v2.tar.gz
```

Training can read the verified archive directly; extraction is optional. No
existing release or raw working episode is overwritten, and local packaging does
not imply a remote upload or commit.
