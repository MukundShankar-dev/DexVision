"""Generate expert demonstrations for a skill and check that they replay.

    python -m dexvision.apps.generate_demos --config pick --verify 50

Episodes go to datasets/<task>/<config name>/ (untracked). The manifest's summary, without the
per-file hashes, is copied to results/<task>/dataset_<config name>.json (tracked).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dexvision import REPO_ROOT
from dexvision.data.demos import generate, load_data_config, load_episode, replay_matches
from dexvision.policies import make_policy
from dexvision.sim.workcell import Workcell
from dexvision.tasks import make_task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="pick", help="name in configs/data/ or a path")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--verify", type=int, default=50, help="replay the first N kept episodes")
    args = parser.parse_args()

    config = load_data_config(args.config)
    name = Path(args.config).stem
    out = args.out or REPO_ROOT / "datasets" / config["task"] / name
    workcell = Workcell()
    task = make_task(config["task"])
    expert = make_policy(config["policy"], workcell.action_space, config["task"])

    def progress(done: int, total: int, result) -> None:
        if done % 100 == 0 or done == total:
            print(f"  {done}/{total} episodes", flush=True)

    print(f"Generating {config['episodes']} {config['task']} episodes into {out}")
    started = time.perf_counter()
    manifest = generate(workcell, task, expert, config, out, progress)
    kept = manifest["kept"]
    print(f"Kept {kept['successes']}/{kept['n']}, 95% CI [{kept['ci95'][0]:.1%}, "
          f"{kept['ci95'][1]:.1%}]; dropped {manifest['dropped']}; "
          f"{manifest['transitions']} transitions ({time.perf_counter() - started:.0f} s)")

    paths = sorted(out.glob("episode_*.npz"))[: args.verify]
    matched = sum(replay_matches(workcell, task, load_episode(p)) for p in paths)
    print(f"Replay: {matched}/{len(paths)} episodes reproduce outcome and observations exactly")
    summary = {k: v for k, v in manifest.items() if k != "files"}
    summary["replay_check"] = {"episodes": len(paths), "matched": matched}
    result_path = REPO_ROOT / "results" / config["task"] / f"dataset_{name}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {out / 'manifest.json'} and {result_path}")


if __name__ == "__main__":
    main()
