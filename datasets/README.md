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
