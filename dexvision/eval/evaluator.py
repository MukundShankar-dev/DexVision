"""One evaluator for every skill and every policy.

Counting rules (AGENTS.md rule 7): one seed is one unique episode, every rate carries a Wilson
95% interval, and test-split runs are budgeted and appended to the ledger.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dexvision.eval.ledger import DEFAULT_LEDGER, append_test_run, check_test_budget
from dexvision.eval.stats import wilson_interval
from dexvision.eval.video import VideoRecorder
from dexvision.policies.base import Policy
from dexvision.provenance import code_version
from dexvision.sim.workcell import Workcell
from dexvision.tasks.features import observation_dict


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    target_object: str
    success: bool
    steps: int
    failure: str | None
    metrics: dict[str, float]


@dataclass(frozen=True)
class Rate:
    successes: int
    n: int
    ci95: tuple[float, float]

    @classmethod
    def of(cls, results: list[EpisodeResult]) -> Rate:
        k = sum(r.success for r in results)
        lo, hi = wilson_interval(k, len(results))
        return cls(k, len(results), (round(lo, 4), round(hi, 4)))


@dataclass(frozen=True)
class EvalReport:
    skill: str
    policy: str
    split: str
    overall: Rate
    by_object: dict[str, Rate]
    failures: dict[str, int]
    episodes: list[EpisodeResult] = field(repr=False)
    code: dict[str, object]

    def to_dict(self) -> dict:
        return asdict(self)


def run_episode(
    workcell: Workcell, task, policy: Policy, seed: int, recorder: VideoRecorder | None = None
) -> EpisodeResult:
    context, observation = task.reset(workcell, seed)
    policy.reset(context, seed)
    if recorder:
        recorder.capture()
    fields = task.spec.observation
    for step in range(1, task.spec.horizon_steps + 1):
        action = policy.act(observation_dict(fields, observation, context))
        observation = workcell.step(action)
        check = task.check(observation, workcell.contacts())
        if recorder:
            recorder.capture()
        if check.success or check.failure:
            return EpisodeResult(
                seed, context.target_object, check.success, step, check.failure,
                task.episode_metrics(),
            )
    return EpisodeResult(
        seed, context.target_object, False, task.spec.horizon_steps, "timeout",
        task.episode_metrics(),
    )


def evaluate(
    workcell: Workcell,
    task,
    policy: Policy,
    split: str,
    n_episodes: int,
    *,
    video_dir: Path | None = None,
    n_videos: int = 0,
    ledger_path: Path = DEFAULT_LEDGER,
) -> EvalReport:
    """Run ``n_episodes`` on the first seeds of ``split``. Test runs are budgeted and logged."""
    skill = task.spec.skill
    seeds = task.spec.seeds(split, n_episodes)
    if split == "test":
        check_test_budget(skill, policy.name, task.spec.test_budget, ledger_path)
    recorder = VideoRecorder(workcell) if video_dir and n_videos else None
    episodes = []
    for index, seed in enumerate(seeds):
        record = recorder if recorder and index < n_videos else None
        result = run_episode(workcell, task, policy, seed, record)
        if record:
            outcome = "success" if result.success else result.failure
            record.save(video_dir / f"{skill}_{policy.name}_{split}_{seed}_{outcome}.mp4",
                        fps=1.0 / workcell.control_dt)
        episodes.append(result)
    if recorder:
        recorder.close()

    grouped = defaultdict(list)
    failures = defaultdict(int)
    for result in episodes:
        grouped[result.target_object].append(result)
        if result.failure:
            failures[result.failure] += 1
    report = EvalReport(
        skill=skill,
        policy=policy.name,
        split=split,
        overall=Rate.of(episodes),
        by_object={name: Rate.of(group) for name, group in sorted(grouped.items())},
        failures=dict(failures),
        episodes=episodes,
        code=code_version(),
    )
    if split == "test":
        append_test_run(
            {"skill": skill, "policy": policy.name, "n": report.overall.n,
             "successes": report.overall.successes, "ci95": report.overall.ci95,
             "seeds": [seeds.start, seeds.stop], "code": report.code},
            ledger_path,
        )
    return report
