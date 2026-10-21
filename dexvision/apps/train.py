"""Train the method (or the MLP baseline) on a skill's demonstrations.

    python -m dexvision.apps.train --task pick --method diffusion --device cuda

Checkpoints (EMA weights) go to runs/<task>/<method>_seed<seed>/step_<N>.pt every
``checkpoint_every`` steps, with the loss log in train_log.csv. The directory must not exist.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dexvision import REPO_ROOT
from dexvision.learning.dataset import DemoWindows
from dexvision.learning.train import MODELS, load_method_config, train
from dexvision.tasks import TASKS, load_task_spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--method", choices=sorted(MODELS), required=True)
    parser.add_argument("--data", type=Path, default=None,
                        help="episode directory (default datasets/<task>/<task>)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=None,
                        help="override the method's step count (smoke runs only)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    spec = load_task_spec(args.task)
    config = load_method_config()
    data = args.data or REPO_ROOT / "datasets" / args.task / args.task
    out = args.out or REPO_ROOT / "runs" / args.task / f"{args.method}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    horizons = config["horizons"]
    windows = DemoWindows.from_directory(data, spec.observation, horizons["observation"],
                                         horizons["prediction"])
    if windows.n_episodes != len(manifest["files"]):
        raise RuntimeError(f"{data} holds {windows.n_episodes} episodes; the manifest lists "
                           f"{len(manifest['files'])}")
    print(f"{args.method} on {args.task}: {windows.n_episodes} episodes, {len(windows)} windows, "
          f"device {args.device}", flush=True)

    record = {"skill": spec.skill, "dataset": str(data.resolve().relative_to(REPO_ROOT))
              if data.resolve().is_relative_to(REPO_ROOT) else str(data),
              "dataset_files_sha256": manifest["files_sha256"],
              "dataset_code": manifest["code"], "steps_override": args.steps}
    log = (out / "train_log.csv").open("w", encoding="utf-8")
    log.write("step,loss,seconds\n")
    started = time.perf_counter()
    running = []

    def on_step(step: int, loss: float) -> None:
        running.append(loss)
        if step % 100 == 0:
            mean = sum(running) / len(running)
            running.clear()
            elapsed = time.perf_counter() - started
            log.write(f"{step},{mean:.6f},{elapsed:.1f}\n")
            log.flush()
            if step % 1000 == 0:
                print(f"  step {step}: loss {mean:.4f} ({elapsed / 60:.1f} min)", flush=True)

    def on_checkpoint(trained) -> None:
        path = out / f"step_{trained.step:06d}.pt"
        trained.save(path)
        print(f"  saved {path.name}", flush=True)

    train(args.method, windows, config, steps=args.steps, seed=args.seed, device=args.device,
          record=record, checkpoint=on_checkpoint, log=on_step)
    log.close()
    print(f"Done in {(time.perf_counter() - started) / 60:.1f} min: {out}")


if __name__ == "__main__":
    main()
