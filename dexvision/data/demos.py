"""Record expert demonstrations and replay them.

Each kept episode is one compressed ``episode_<seed>.npz`` holding, for T control steps:

    obs/<field>       [T, d]   the declared observation field before step t's action
    obs/rgb/<camera>  [T, H, W, 3] uint8, when the data config lists ``cameras``
    action            [T, 26]  the clean expert action, clipped by ``ActionSpace.clip`` (the label)
    executed_action   [T, 26]  action + noise, exactly what was sent to ``Workcell.step``
    seed, success, steps, target_object, failure   scalars

The rollout loop is the evaluator's ``run_episode``, so recorded and evaluated episodes obey the
same reset, termination and success rules. ``manifest.json`` records provenance, the resolved
configs, counts and a SHA-256 per file.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from dexvision import REPO_ROOT
from dexvision.eval.evaluator import EpisodeResult, Rate, camera_rig, run_episode
from dexvision.policies.base import Policy
from dexvision.provenance import code_version
from dexvision.sim.workcell import (
    ACTION_DIM,
    ACTUATOR_TARGETS,
    ROTATION,
    TRANSLATION,
    ActionSpace,
    Workcell,
)
from dexvision.tasks.features import TaskContext

DATA_CONFIG_DIR = REPO_ROOT / "configs" / "data"


def noise_std_vector(noise_std: Mapping[str, float]) -> np.ndarray:
    """Config ``noise_std`` -> [26] per-component standard deviations."""
    std = np.zeros(ACTION_DIM)
    std[TRANSLATION] = noise_std["translation_m"]
    std[ROTATION] = noise_std["rotation_rad"]
    std[ACTUATOR_TARGETS] = noise_std["targets_rad"]
    return std


class DemoRecorder:
    """Wraps an expert. Records observations and clean actions; executes them with noise."""

    def __init__(self, expert: Policy, action_space: ActionSpace, noise_std: np.ndarray) -> None:
        self.expert = expert
        self.name = expert.name
        self._space = action_space
        self._noise_std = np.asarray(noise_std, dtype=float)

    def reset(self, context: TaskContext, seed: int) -> None:
        self.expert.reset(context, seed)
        self._rng = np.random.default_rng([seed, 3])
        self.observations: list[dict[str, np.ndarray]] = []
        self.actions: list[np.ndarray] = []
        self.executed: list[np.ndarray] = []

    def act(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        clean = self._space.clip(self.expert.act(observation))
        executed = clean + self._rng.normal(0.0, self._noise_std)
        self.observations.append({k: np.array(v) for k, v in observation.items()})
        self.actions.append(clean)
        self.executed.append(executed)
        return executed

    def arrays(self, result: EpisodeResult) -> dict[str, np.ndarray]:
        fields = self.observations[0].keys()
        arrays = {f"obs/{k}": np.stack([o[k] for o in self.observations]) for k in fields}
        arrays["action"] = np.stack(self.actions)
        arrays["executed_action"] = np.stack(self.executed)
        arrays["seed"] = np.array(result.seed)
        arrays["success"] = np.array(result.success)
        arrays["steps"] = np.array(result.steps)
        arrays["target_object"] = np.array(result.target_object)
        arrays["failure"] = np.array(result.failure or "")
        return arrays


class ReplayPolicy:
    """Sends a recorded action sequence open loop and records what it observes. If the episode
    outlasts the recording (it diverged), it holds still: no motion, last finger targets."""

    name = "replay"

    def __init__(self, executed_actions: np.ndarray) -> None:
        self._actions = executed_actions

    def reset(self, context: TaskContext, seed: int) -> None:
        self._t = 0
        self.observations: list[dict[str, np.ndarray]] = []

    def act(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        self.observations.append({k: np.array(v) for k, v in observation.items()})
        if self._t < len(self._actions):
            action = self._actions[self._t]
        else:
            action = self._actions[-1].copy()
            action[:6] = 0.0
        self._t += 1
        return action


def load_episode(path: Path, images: bool = True) -> dict[str, np.ndarray]:
    """All arrays of one episode file; without the ``obs/rgb/*`` frames if not ``images``."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files
                if images or not key.startswith("obs/rgb/")}


def replay_matches(workcell: Workcell, task, episode: Mapping[str, np.ndarray]) -> bool:
    """Replay ``executed_action`` from the episode's seed. True when the outcome, step count and
    every recorded observation match bit for bit."""
    replay = ReplayPolicy(episode["executed_action"])
    result = run_episode(workcell, task, replay, int(episode["seed"]))
    if (result.success, result.steps) != (bool(episode["success"]), int(episode["steps"])):
        return False
    return all(
        np.array_equal(np.stack([o[k] for o in replay.observations]), episode[f"obs/{k}"])
        for k in task.spec.observation
    )


def load_data_config(name_or_path: str | Path) -> dict:
    path = Path(name_or_path)
    if not path.suffix:
        path = DATA_CONFIG_DIR / f"{name_or_path}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(workcell: Workcell, task, expert: Policy, config: Mapping, out_dir: Path,
            seeds: Sequence[int], progress=None) -> tuple[list[EpisodeResult], dict[str, str]]:
    """Roll out ``seeds`` in order; save the successes. -> results, {file name: SHA-256}."""
    recorder = DemoRecorder(expert, workcell.action_space, noise_std_vector(config["noise_std"]))
    rig = camera_rig(workcell, config.get("cameras", ()))
    results, files = [], {}
    for seed in seeds:
        result = run_episode(workcell, task, recorder, seed, rig=rig)
        results.append(result)
        if result.success:
            path = out_dir / f"episode_{seed:06d}.npz"
            np.savez_compressed(path, **recorder.arrays(result))
            files[path.name] = _sha256(path)
        if progress:
            progress(len(results), len(seeds), result)
    if rig:
        rig.close()
    return results, files


def _record_in_worker(job: tuple[Mapping, str, list[int]]
                      ) -> tuple[list[EpisodeResult], dict[str, str]]:
    from dexvision.policies import make_policy
    from dexvision.tasks import make_task

    config, out_dir, seeds = job
    workcell = Workcell()
    expert = make_policy(config["policy"], workcell.action_space, config["task"])
    return _record(workcell, make_task(config["task"]), expert, config, Path(out_dir), seeds)


def generate(
    workcell: Workcell, task, expert: Policy, config: Mapping, out_dir: Path,
    progress=None, workers: int = 1,
) -> dict:
    """Roll out ``expert`` on the first ``config['episodes']`` seeds of ``config['split']``,
    keep the successes in ``out_dir`` and return the manifest (also written to the directory).

    With ``workers`` > 1, chunks of seeds run in separate processes, each building its own
    workcell, task and expert from the config. Every episode depends only on its seed, so the
    files are the same either way."""
    out_dir.mkdir(parents=True, exist_ok=False)
    seeds = list(task.spec.seeds(config["split"], int(config["episodes"])))
    if workers <= 1:
        results, files = _record(workcell, task, expert, config, out_dir, seeds, progress)
    else:
        chunks = [seeds[i:i + 20] for i in range(0, len(seeds), 20)]
        results, files = [], {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            jobs = [pool.submit(_record_in_worker, (dict(config), str(out_dir), chunk))
                    for chunk in chunks]
            for job in as_completed(jobs):
                chunk_results, chunk_files = job.result()
                results.extend(chunk_results)
                files.update(chunk_files)
                if progress:
                    progress(len(results), len(seeds), chunk_results[-1])
        results.sort(key=lambda r: r.seed)
        files = dict(sorted(files.items()))
    manifest = build_manifest(task, expert, config, results, files)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", "utf-8")
    return manifest


def build_manifest(task, expert, config, results: list[EpisodeResult], files: dict) -> dict:
    kept = [r for r in results if r.success]
    by_object: dict[str, list[EpisodeResult]] = {}
    for r in results:
        by_object.setdefault(r.target_object, []).append(r)
    failures: dict[str, int] = {}
    for r in results:
        if r.failure:
            failures[r.failure] = failures.get(r.failure, 0) + 1
    steps = np.array([r.steps for r in kept]) if kept else np.zeros(1)
    return {
        "code": code_version(),
        "config": dict(config),
        "task_spec": task.spec.raw,
        "expert_config": getattr(expert, "config", None),
        "seeds": [results[0].seed, results[-1].seed + 1],
        "attempted": len(results),
        "kept": asdict(Rate.of(results)),
        "kept_by_object": {k: asdict(Rate.of(v)) for k, v in sorted(by_object.items())},
        "dropped": failures,
        "transitions": int(sum(r.steps for r in kept)),
        "steps_per_episode": {"min": int(steps.min()), "median": float(np.median(steps)),
                              "max": int(steps.max())},
        "files": files,
        "files_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
    }
