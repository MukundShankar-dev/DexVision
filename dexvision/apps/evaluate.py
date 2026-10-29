"""Evaluate a policy on a task split and write the report.

    python -m dexvision.apps.evaluate --task pick --policy zero --episodes 100 --videos 2
    python -m dexvision.apps.evaluate --task pick --checkpoint runs/pick/<run>/step_060000.pt

Reports go to results/<task>/<policy>_<split>.json (a checkpoint is named by its method) and
videos to runs/videos/. Test-split runs
count against the task's test budget and are logged in results/test_ledger.jsonl.

--dart-noise executes every action with the DART noise from configs/data/<task>.yaml, to measure
how robust a policy (usually the expert) is to small execution errors. Val split only.

--blank CAMERA [CAMERA ...] replaces those cameras' images with the training-mean image, to
measure how much a vision policy relies on each camera. Val split only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from dexvision import REPO_ROOT
from dexvision.data.demos import DemoRecorder, load_data_config, noise_std_vector
from dexvision.eval.evaluator import evaluate
from dexvision.learning.policy import load_learned_policy
from dexvision.policies import POLICIES, make_policy
from dexvision.sim.workcell import Workcell
from dexvision.tasks import TASK_NAMES, make_task


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=TASK_NAMES, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--policy", choices=sorted(POLICIES))
    source.add_argument("--checkpoint", type=Path, help="a trained model from apps.train")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--videos", type=int, default=0, help="record the first N episodes")
    parser.add_argument("--video-dir", type=Path, default=REPO_ROOT / "runs" / "videos")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dart-noise", action="store_true",
                        help="execute actions with the task's DART noise (val only)")
    parser.add_argument("--blank", nargs="+", default=(), metavar="CAMERA",
                        help="replace these cameras' images with the training mean (val only)")
    parser.add_argument("--progress-every", type=int, default=20, metavar="N",
                        help="print a progress line every N episodes (0: never)")
    args = parser.parse_args()
    if args.dart_noise and args.split != "val":
        parser.error("--dart-noise is a robustness measurement on val, not a test result")
    if args.blank and (args.split != "val" or not args.checkpoint):
        parser.error("--blank is a diagnostic for a trained vision policy on val")

    workcell = Workcell()
    task = make_task(args.task)
    info = {}
    if args.checkpoint:
        policy, info = load_learned_policy(args.checkpoint, task.spec, blank=args.blank)
    else:
        policy = make_policy(args.policy, workcell.action_space, task.spec.skill)
    if args.dart_noise:
        noise = noise_std_vector(load_data_config(args.task)["noise_std"])
        name = policy.name
        policy = DemoRecorder(policy, workcell.action_space, noise)
        policy.name = f"{name}_dart"
    print(f"Evaluating {policy.name} on {args.task}/{args.split}: {args.episodes} episodes",
          flush=True)
    started = time.perf_counter()

    def progress(episodes, total):
        if args.progress_every and len(episodes) % args.progress_every == 0:
            done = len(episodes)
            elapsed = time.perf_counter() - started
            print(f"  {done}/{total} episodes, {sum(e.success for e in episodes)} successes, "
                  f"{elapsed / done:.1f} s/episode, ~{elapsed / done * (total - done) / 60:.0f} "
                  f"min left", flush=True)

    report = evaluate(workcell, task, policy, args.split, args.episodes, video_dir=args.video_dir,
                      n_videos=args.videos, policy_info=info, on_episode=progress)
    elapsed = time.perf_counter() - started

    out = args.out or REPO_ROOT / "results" / args.task / f"{policy.name}_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=1) + "\n", encoding="utf-8")
    o = report.overall
    print(f"Success {o.successes}/{o.n} = {o.successes / o.n:.1%}, "
          f"95% CI [{o.ci95[0]:.1%}, {o.ci95[1]:.1%}]  ({elapsed:.0f} s)")
    for name, rate in report.by_object.items():
        lo, hi = rate.ci95
        print(f"  {name:15s} {rate.successes:3d}/{rate.n:<3d} CI [{lo:.0%}, {hi:.0%}]")
    print(f"  failures: {report.failures}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
