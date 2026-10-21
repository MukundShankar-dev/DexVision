"""A trained model behind the common ``Policy`` interface (receding-horizon action chunks).

Each inference predicts Tp actions from the last To observations; the first ``execution``
actions are executed before predicting again. Sampling noise comes from a generator seeded with
the episode seed, so evaluation is reproducible. The policy has no skill-specific code.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping

import numpy as np
import torch

from dexvision.learning.dataset import observation_vector
from dexvision.learning.train import Trained
from dexvision.tasks.features import TaskContext


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
        self._queue: deque[np.ndarray] = deque()
        self._generator = torch.Generator().manual_seed(int(seed))

    def act(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        obs = observation_vector(self.trained.fields, observation)
        self._history.append(obs)
        while len(self._history) < self._obs_horizon:  # episode start: repeat the first
            self._history.appendleft(obs)
        if not self._queue:
            self._queue.extend(self._predict())
        return self._queue.popleft()

    def _predict(self) -> np.ndarray:
        """-> the next ``execution`` actions [execution, A] in action units."""
        t = self.trained
        obs = torch.as_tensor(t.obs_norm.encode(np.stack(self._history)), dtype=torch.float32,
                              device=self._device)[None]
        chunk = t.model.predict(obs, self._generator)[0].cpu().double().numpy()
        self.inferences += 1
        return t.action_norm.decode(chunk)[: self._execution]
