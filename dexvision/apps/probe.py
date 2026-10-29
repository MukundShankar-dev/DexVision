"""Perception probe and offline action error (docs/PLAN.md, V3; dexvision/eval/probe.py).

    # before training: can the encoder architecture read object position and identity at all?
    python -m dexvision.apps.probe --task pick_vision --device cuda \
        --train-data datasets/pick/pick_vision --val-data datasets/pick/pick_vision_val
    # after training: what did a policy's encoder learn, and how well does it fit held-out demos?
    python -m dexvision.apps.probe --task pick_vision --device cuda --checkpoint <run>/step_N.pt \
        --train-data datasets/pick/pick_vision --val-data datasets/pick/pick_vision_val

Without --checkpoint it runs the scratch probe. With one, it runs the frozen probe on the
checkpoint's encoder (if it has cameras) and the offline action error on the val demos (a state
checkpoint works too, with --task pick). Reports go to results/<task>/probe_<name>.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from dexvision import REPO_ROOT
from dexvision.data.demos import load_episode
from dexvision.eval.probe import (
    FeatureProbe,
    ScratchProbe,
    fit_probe,
    frozen_features,
    load_probe_frames,
    offline_action_error,
    probe_frames,
    probe_report,
)
from dexvision.learning.train import Trained, load_method_config
from dexvision.provenance import code_version
from dexvision.tasks import TASK_NAMES, load_task_spec


def _paths(directory: Path) -> list[Path]:
    return sorted(Path(directory).glob("episode_*.npz"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=TASK_NAMES, required=True)
    parser.add_argument("--train-data", type=Path, default=None, help="needed for a probe")
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=10000, help="probe training steps")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--offline-windows", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    spec = load_task_spec(args.task)
    objects = list(spec.raw["objects"])
    trained = Trained.load(args.checkpoint, args.device) if args.checkpoint else None
    cameras = trained.cameras if trained else spec.cameras
    started = time.perf_counter()
    val_episodes = [load_episode(p) for p in _paths(args.val_data)]
    result: dict = {"task": spec.name, "code": code_version(), "train_data": str(args.train_data),
                    "val_data": str(args.val_data), "val_episodes": len(val_episodes),
                    "settings": {k: getattr(args, k) for k in
                                 ("steps", "batch", "lr", "offline_windows", "seed")}}
    if trained:
        result["checkpoint"] = str(args.checkpoint)
        result["offline_action_error"] = offline_action_error(
            trained, val_episodes, args.offline_windows, args.seed, args.device)
        print(f"Offline action error (normalized units), {trained.method}:")
        for bucket, row in result["offline_action_error"].items():
            print(f"  {bucket:7s} {row}")

    if cameras:
        if args.train_data is None:
            parser.error("a probe needs --train-data")
        train = load_probe_frames(_paths(args.train_data), cameras, objects)
        val = probe_frames(val_episodes, cameras, objects)
        print(f"Probe frames: {len(train.position)} train, {len(val.position)} val", flush=True)
        if trained:
            name = "frozen"
            features = {split: frozen_features(trained.model.vision, frames.images, args.device)
                        for split, frames in (("train", train), ("val", val))}
            model = FeatureProbe(features["train"].shape[1], len(objects))

            def inputs_for(split):
                return lambda rows: features[split][rows.to(args.device)]
        else:
            name = "scratch"
            model = ScratchProbe(cameras, load_method_config()["vision"], len(objects))

            def inputs_for(split):
                frames = train if split == "train" else val
                return lambda rows: {c: torch.as_tensor(x[rows.numpy()], device=args.device)
                                     for c, x in frames.images.items()}
        scale = fit_probe(model, inputs_for("train"), train, steps=args.steps, batch=args.batch,
                          lr=args.lr, device=args.device, seed=args.seed)
        result[f"probe_{name}"] = probe_report(model, inputs_for("val"), val, scale)
        print(f"Probe ({name}) on val frames:")
        for bucket, row in result[f"probe_{name}"].items():
            print(f"  {bucket:7s} {row}")

    label = f"{trained.method}_step{trained.step}" if trained else "scratch"
    out = args.out or REPO_ROOT / "results" / spec.name / f"probe_{label}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {out} ({(time.perf_counter() - started) / 60:.1f} min)")


if __name__ == "__main__":
    main()
