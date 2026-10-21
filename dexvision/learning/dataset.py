"""Training windows from demonstration episodes (see ``dexvision/data/demos.py``).

Sample t of an episode pairs

    observations  obs[t - To + 1] .. obs[t]     [To, D]   (before the episode: obs[0] repeated)
    actions       action[t] .. action[t + Tp - 1]  [Tp, A]  (after the episode: last repeated)

where To and Tp are the observation and prediction horizons. Observation fields are
concatenated in the task spec's declared order.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from dexvision.data.demos import load_episode


def observation_vector(fields: Sequence[str], arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate declared fields: each [..., d_i] -> [..., sum d_i]."""
    return np.concatenate([np.asarray(arrays[f], dtype=np.float64) for f in fields], axis=-1)


class DemoWindows:
    """All windows of a set of episodes, held in memory.

    ``observations`` [N, D] and ``actions`` [N, A] are the stacked steps of every episode;
    ``obs_index`` [M, To] and ``action_index`` [M, Tp] select the rows of each window.
    """

    def __init__(self, episodes: Sequence[dict[str, np.ndarray]], fields: Sequence[str],
                 obs_horizon: int, pred_horizon: int) -> None:
        if not episodes:
            raise ValueError("No episodes")
        self.fields = tuple(fields)
        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon
        obs, actions, obs_index, action_index = [], [], [], []
        start = 0
        for episode in episodes:
            o = observation_vector(self.fields, {f: episode[f"obs/{f}"] for f in self.fields})
            a = np.asarray(episode["action"], dtype=np.float64)
            n = len(a)
            if len(o) != n:
                raise ValueError("Observation and action lengths differ")
            t = np.arange(n)
            obs_index.append(start + np.clip(t[:, None] + np.arange(1 - obs_horizon, 1), 0, n - 1))
            action_index.append(start + np.clip(t[:, None] + np.arange(pred_horizon), 0, n - 1))
            obs.append(o)
            actions.append(a)
            start += n
        self.observations = np.concatenate(obs)
        self.actions = np.concatenate(actions)
        self.obs_index = np.concatenate(obs_index)
        self.action_index = np.concatenate(action_index)
        self.n_episodes = len(episodes)

    def __len__(self) -> int:
        return len(self.obs_index)

    def window(self, i) -> tuple[np.ndarray, np.ndarray]:
        """Index or index array -> observations [..., To, D], actions [..., Tp, A]."""
        return self.observations[self.obs_index[i]], self.actions[self.action_index[i]]

    @classmethod
    def from_directory(cls, directory: Path, fields: Sequence[str], obs_horizon: int,
                       pred_horizon: int) -> DemoWindows:
        paths = sorted(Path(directory).glob("episode_*.npz"))
        return cls([load_episode(p) for p in paths], fields, obs_horizon, pred_horizon)
