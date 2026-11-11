"""One evaluator for every skill and every policy.

Counting rules (AGENTS.md rule 7): one seed is one unique episode, every rate carries a Wilson
95% interval, and test-split runs are budgeted and appended to the ledger.
"""

from __future__ import annotations

import multiprocessing
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from dexvision.eval.ledger import DEFAULT_LEDGER, append_test_run, check_test_budget
from dexvision.eval.stats import wilson_interval
from dexvision.eval.video import VideoRecorder
from dexvision.policies.base import Policy
from dexvision.provenance import code_version
from dexvision.sim.appearance import AppearanceSampler
from dexvision.sim.cameras import CameraConfig, CameraRig
from dexvision.sim.workcell import Workcell
from dexvision.tasks.features import observation_dict

# Builds a policy inside a worker process: a module-level function or functools.partial of one.
PolicyFactory = Callable[[Workcell], Policy]
CHUNK = 10  # seeds per worker job; progress is reported per finished chunk


def worker_pool(workers: int) -> ProcessPoolExecutor:
    """The process pool every parallel path uses. Workers are spawned, not forked: a forked child
    of a process that already holds an OpenGL context hangs in its own renderer on Linux, and
    macOS and Windows spawn anyway, so every platform now runs the same way."""
    return ProcessPoolExecutor(max_workers=workers,
                               mp_context=multiprocessing.get_context("spawn"))

@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    target_object: str
    success: bool
    steps: int
    failure: str | None
    metrics: dict[str, float]
    phase: str | None = None  # how far the episode got, for tasks that define phases
    appearance: dict | None = None  # varied appearance: light, textures, contrast (diagnostics)


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
    task: str  # the spec's name, e.g. pick or pick_vision
    skill: str
    policy: str
    split: str
    overall: Rate
    by_object: dict[str, Rate]
    failures: dict[str, int]
    episodes: list[EpisodeResult] = field(repr=False)
    code: dict[str, object]
    policy_info: dict[str, object] = field(default_factory=dict)  # e.g. checkpoint path and hash
    appearance: tuple[str, ...] = ()  # varied appearance factors; () is the nominal scene

    def to_dict(self) -> dict:
        return asdict(self)


def camera_rig(workcell: Workcell, cameras, appearance=(), train_seeds=(0, 0)
               ) -> CameraRig | None:
    """A rig rendering ``cameras`` (names from configs/cameras.yaml), or None for no cameras.
    ``appearance`` lists the varied factors; held-out textures never show in ``train_seeds``."""
    if not cameras:
        return None
    sampler = AppearanceSampler(workcell.model, appearance, train_seeds) if appearance else None
    return CameraRig(workcell, replace(CameraConfig.load(), cameras=tuple(cameras)), sampler)


def spec_rig(workcell: Workcell, spec) -> CameraRig | None:
    """The rig a task spec declares: its cameras and appearance."""
    return camera_rig(workcell, spec.cameras, spec.appearance, spec.splits["train"])


def run_episode(
    workcell: Workcell, task, policy: Policy, seed: int, recorder: VideoRecorder | None = None,
    rig: CameraRig | None = None,
) -> EpisodeResult:
    """One episode. With a ``rig``, the policy also observes ``rgb/<camera>`` images rendered
    from the state it acts on."""
    context, observation = task.reset(workcell, seed)
    policy.reset(context, seed)
    if rig:
        rig.start_episode(seed)
    if recorder:
        recorder.capture()
    fields = task.spec.observation

    def observe(observation) -> dict:
        observed = observation_dict(fields, observation, context)
        if rig:
            observed.update({f"rgb/{name}": image for name, image in rig.render().items()})
        return observed

    for step in range(1, task.spec.horizon_steps + 1):
        action = policy.act(observe(observation))
        observation = workcell.step(action)
        check = task.check(observation, workcell.contacts())
        if recorder:
            recorder.capture()
        if check.success or check.failure:
            return _result(task, context, seed, check.success, step, check.failure, rig)
    return _result(task, context, seed, False, task.spec.horizon_steps, "timeout", rig)


def _result(task, context, seed: int, success: bool, steps: int, failure: str | None,
            rig: CameraRig | None) -> EpisodeResult:
    metrics = task.episode_metrics()
    phase = task.phase(success, failure, metrics) if hasattr(task, "phase") else None
    appearance = rig.sampler.describe(rig.appearance, context.target_object) \
        if rig and rig.appearance else None
    return EpisodeResult(seed, context.target_object or "none", success, steps, failure, metrics,
                         phase, appearance)


def _run_seeds(job: tuple[PolicyFactory, object, list[int]]) -> list[EpisodeResult]:
    """Worker: its own simulator, policy and cameras. One torch thread per worker, so N workers
    use N cores instead of fighting over them."""
    import torch

    factory, task, seeds = job
    torch.set_num_threads(1)
    workcell = Workcell()
    policy = factory(workcell)
    rig = spec_rig(workcell, task.spec)
    try:
        return [run_episode(workcell, task, policy, seed, rig=rig) for seed in seeds]
    finally:
        if rig:
            rig.close()


def _run_serial(workcell, task, policy, seeds, split, video_dir, n_videos, on_episode
                ) -> list[EpisodeResult]:
    recorder = VideoRecorder(workcell) if video_dir and n_videos else None
    rig = spec_rig(workcell, task.spec)
    episodes = []
    for index, seed in enumerate(seeds):
        record = recorder if recorder and index < n_videos else None
        result = run_episode(workcell, task, policy, seed, record, rig)
        if record:
            outcome = "success" if result.success else result.failure
            record.save(video_dir / f"{task.spec.skill}_{policy.name}_{split}_{seed}_{outcome}.mp4",
                        fps=1.0 / workcell.control_dt)
        episodes.append(result)
        if on_episode:
            on_episode(episodes, len(seeds))
    if recorder:
        recorder.close()
    if rig:
        rig.close()
    return episodes


def _run_parallel(task, seeds, workers, policy_factory, on_episode) -> list[EpisodeResult]:
    jobs = [(policy_factory, task, list(seeds[i:i + CHUNK])) for i in range(0, len(seeds), CHUNK)]
    episodes = []
    with worker_pool(workers) as pool:
        for chunk in pool.map(_run_seeds, jobs):  # in seed order
            for result in chunk:
                episodes.append(result)
                if on_episode:
                    on_episode(episodes, len(seeds))
    return episodes


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
    policy_info: dict[str, object] | None = None,
    on_episode: Callable[[list[EpisodeResult], int], None] | None = None,
    workers: int = 1,
    policy_factory: PolicyFactory | None = None,
) -> EvalReport:
    """Run ``n_episodes`` on the first seeds of ``split``. Test runs are budgeted and logged.

    ``on_episode(episodes_so_far, n_episodes)`` is called after every episode (progress output).
    With ``workers`` > 1, episodes run in that many processes, each building its policy with
    ``policy_factory``; every episode is seeded, so the results equal a serial run's.
    """
    skill = task.spec.skill
    seeds = task.spec.seeds(split, n_episodes)
    if split == "test":
        check_test_budget(skill, policy.name, task.spec.test_budget, task.spec.splits["test"],
                          ledger_path)
    if workers > 1:
        if policy_factory is None or n_videos:
            raise ValueError("Parallel evaluation needs a policy_factory and records no videos")
        episodes = _run_parallel(task, seeds, workers, policy_factory, on_episode)
    else:
        episodes = _run_serial(workcell, task, policy, seeds, split, video_dir, n_videos,
                               on_episode)

    grouped = defaultdict(list)
    failures = defaultdict(int)
    for result in episodes:
        grouped[result.target_object].append(result)
        if result.failure:
            failures[result.failure] += 1
    report = EvalReport(
        task=task.spec.name,
        skill=skill,
        policy=policy.name,
        split=split,
        overall=Rate.of(episodes),
        by_object={name: Rate.of(group) for name, group in sorted(grouped.items())},
        failures=dict(failures),
        episodes=episodes,
        code=code_version(),
        policy_info=dict(policy_info or {}),
        appearance=task.spec.appearance,
    )
    if split == "test":
        append_test_run(
            {"task": task.spec.name, "skill": skill, "policy": policy.name, "n": report.overall.n,
             "successes": report.overall.successes, "ci95": report.overall.ci95,
             "seeds": [seeds.start, seeds.stop], "code": report.code,
             "policy_info": report.policy_info},
            ledger_path,
        )
    return report
