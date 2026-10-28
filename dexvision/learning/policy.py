"""A trained model behind the common ``Policy`` interface (receding-horizon action chunks).

Each inference predicts Tp actions from the last To observations; the first ``execution``
actions are executed before predicting again. Sampling noise comes from a generator seeded with
the episode seed, so evaluation is reproducible. The policy has no skill-specific code. A policy
trained with cameras also reads ``rgb/<camera>`` images ([H, W, 3] uint8) from the observation.
"""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from dexvision import REPO_ROOT
from dexvision.learning.dataset import observation_vector
from dexvision.learning.train import Trained
from dexvision.tasks.features import TaskContext
from dexvision.tasks.spec import TaskSpec


class LearnedPolicy:
    def __init__(self, trained: Trained, device: str = "cpu", name: str | None = None) -> None:
        self.trained = trained
        self.name = name or trained.method
        self._device = device
        horizons = trained.config["horizons"]
        self._obs_horizon, self._execution = horizons["observation"], horizons["execution"]
        self.inferences = 0

    def reset(self, context: TaskContext, seed: int) -> None:
        self._history: deque[np.ndarray] = deque(maxlen=self._obs_horizon)
        self._image_history: deque[dict[str, np.ndarray]] = deque(maxlen=self._obs_horizon)
        self._queue: deque[np.ndarray] = deque()
        self._generator = torch.Generator().manual_seed(int(seed))

    def act(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        obs = observation_vector(self.trained.fields, observation)
        images = {c: np.asarray(observation[f"rgb/{c}"]) for c in self.trained.cameras}
        self._history.append(obs)
        self._image_history.append(images)
        while len(self._history) < self._obs_horizon:  # episode start: repeat the first
            self._history.appendleft(obs)
            self._image_history.appendleft(images)
        if not self._queue:
            self._queue.extend(self._predict())
        return self._queue.popleft()

    def _predict(self) -> np.ndarray:
        """-> the next ``execution`` actions [execution, A] in action units."""
        t = self.trained
        obs = torch.as_tensor(t.obs_norm.encode(np.stack(self._history)), dtype=torch.float32,
                              device=self._device)[None]
        images = {c: torch.as_tensor(np.stack([h[c] for h in self._image_history]),
                                     device=self._device)[None]
                  for c in t.cameras} or None
        chunk = t.model.predict(obs, self._generator, images)[0].cpu().double().numpy()
        self.inferences += 1
        return t.action_norm.decode(chunk)[: self._execution]


def load_learned_policy(path: Path, spec: TaskSpec, device: str = "cpu"
                        ) -> tuple[LearnedPolicy, dict[str, object]]:
    """Load a checkpoint for ``spec``'s skill. Refuses checkpoints trained for another skill or
    on other observation fields. Returns the policy and its provenance for the report."""
    path = Path(path).resolve()
    trained = Trained.load(path, device)
    if trained.record.get("skill") != spec.skill:
        raise ValueError(f"{path} was trained for {trained.record.get('skill')!r}, "
                         f"not {spec.skill!r}")
    if trained.fields != spec.observation:
        raise ValueError(f"{path} observes {trained.fields}, the task declares {spec.observation}")
    try:
        shown = str(path.relative_to(REPO_ROOT))
    except ValueError:
        shown = str(path)
    info = {
        "checkpoint": shown,
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "method": trained.method,
        "step": trained.step,
        "training": trained.record,
    }
    return LearnedPolicy(trained, device), info
