# Versioned Dataset Releases

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
generalization. Level 4 will publish a separate versioned release with genuine
session ids, broader objects/goals, grasp-lift-place and recovery coverage, and
visual grounding data. The Level 2 archive must not be overwritten when that
release is created.

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
