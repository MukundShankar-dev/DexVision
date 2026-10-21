import copy

import numpy as np
import pytest
import torch

from dexvision.learning.dataset import DemoWindows
from dexvision.learning.diffusion import DiffusionModel, squared_cosine_alphas_cumprod
from dexvision.learning.normalize import Normalizer
from dexvision.learning.policy import LearnedPolicy
from dexvision.learning.train import Trained, build_model, load_method_config, train

FIELDS = ("code", "phase")


def tiny_config(steps=300):
    config = copy.deepcopy(load_method_config())
    config["diffusion"].update(down_dims=[32, 64], step_embedding_dim=32, group_norm_groups=4)
    config["mlp"]["hidden_dims"] = [128, 128]
    config["training"].update(steps=steps, batch_size=64, warmup_steps=20,
                              learning_rate=1e-3, checkpoint_every=10**9)
    return config


def synthetic_episodes(n_episodes=4, length=24, action_dim=3):
    """Each episode has its own code; actions are smooth functions of code and time, and the
    last action dimension is constant (like an unused actuator)."""
    episodes = []
    for e in range(n_episodes):
        t = np.arange(length)
        code = np.tile([np.cos(2 * np.pi * e / n_episodes), np.sin(2 * np.pi * e / n_episodes)],
                       (length, 1))
        actions = np.stack([np.sin(0.3 * t + e), np.cos(0.2 * t - e) * (e + 1) / n_episodes,
                            np.full(length, 0.5)], axis=1)[:, :action_dim]
        episodes.append({"obs/code": code, "obs/phase": (t / length)[:, None],
                         "action": actions})
    return episodes


def test_normalizer_round_trip_and_constant_dimensions():
    data = np.array([[0.0, 5.0, 2.0], [1.0, 5.0, -2.0], [0.5, 5.0, 0.0]])
    norm = Normalizer.fit(data)
    encoded = norm.encode(data)
    assert encoded.min() >= -1 and encoded.max() <= 1
    np.testing.assert_allclose(encoded[:, 1], 0.0)
    np.testing.assert_allclose(norm.decode(encoded), data)
    tensor = torch.as_tensor(data)
    torch.testing.assert_close(norm.decode(norm.encode(tensor)), tensor)
    restored = Normalizer.from_state(norm.state())
    np.testing.assert_array_equal(restored.encode(data), encoded)


def test_windows_pad_history_at_start_and_actions_at_end():
    windows = DemoWindows(synthetic_episodes(2, 5), FIELDS, obs_horizon=2, pred_horizon=4)
    assert len(windows) == 10
    obs, actions = windows.window(0)
    assert obs.shape == (2, 3) and actions.shape == (4, 3)
    np.testing.assert_array_equal(obs[0], obs[1])  # history before t = 0 repeats obs[0]
    obs, actions = windows.window(4)  # last step of episode 0
    np.testing.assert_array_equal(actions, np.repeat(actions[:1], 4, axis=0))
    obs, actions = windows.window(5)  # first step of episode 1: no leakage from episode 0
    np.testing.assert_array_equal(obs[0], obs[1])
    episode1 = synthetic_episodes(2, 5)[1]
    np.testing.assert_array_equal(actions, episode1["action"][:4])
    np.testing.assert_array_equal(windows.window(np.array([0, 5]))[1].shape, (2, 4, 3))


def test_model_shapes_with_the_method_config():
    config = load_method_config()
    horizons = config["horizons"]
    obs = torch.zeros(3, horizons["observation"], 92)
    for method in ("diffusion", "mlp"):
        model = build_model(method, 92, 26, config).eval()
        assert model.loss(obs, torch.zeros(3, horizons["prediction"], 26)).shape == ()
        assert model.predict(obs, torch.Generator().manual_seed(0)).shape == (
            3, horizons["prediction"], 26)


def test_noise_schedule_runs_from_signal_to_noise():
    a = squared_cosine_alphas_cumprod(100)
    assert a.shape == (100,) and torch.all(a[1:] < a[:-1])
    assert a[0] > 0.99 and a[-1] < 1e-3


def fit_error(method, config, n_episodes=4, length=24, seed=0):
    """Mean absolute action error of the trained model on its own training windows."""
    windows = DemoWindows(synthetic_episodes(n_episodes, length), FIELDS, obs_horizon=2,
                          pred_horizon=16)
    trained = train(method, windows, config, seed=seed)
    obs, actions = windows.window(np.arange(len(windows)))
    obs = torch.as_tensor(trained.obs_norm.encode(obs), dtype=torch.float32)
    chunk = trained.model.predict(obs, torch.Generator().manual_seed(1)).double().numpy()
    return np.abs(trained.action_norm.decode(chunk) - actions).mean()


def tiny_diffusion_config(steps):
    config = tiny_config(steps)
    config["diffusion"].update(down_dims=[16, 32], step_embedding_dim=16)
    config["training"].update(batch_size=32, learning_rate=3e-3)
    return config


def test_mlp_overfits_a_tiny_dataset():
    assert fit_error("mlp", tiny_config(steps=1000)) < 0.05


def test_diffusion_learns_a_tiny_dataset():
    """Fast check: the error falls well below an untrained model's. Full convergence is slow on
    CPU (about 19 ms per step for this tiny UNet); see the slow test below."""
    untrained = fit_error("diffusion", tiny_diffusion_config(steps=1), n_episodes=2, length=12)
    trained = fit_error("diffusion", tiny_diffusion_config(steps=1000), n_episodes=2, length=12)
    assert trained < 0.2 and trained < 0.4 * untrained, (trained, untrained)


@pytest.mark.slow
def test_diffusion_overfits_a_tiny_dataset():
    config = tiny_config(steps=8000)
    assert fit_error("diffusion", config) < 0.02


def test_training_and_sampling_are_deterministic_on_cpu():
    windows = DemoWindows(synthetic_episodes(), FIELDS, obs_horizon=2, pred_horizon=16)
    first = train("diffusion", windows, tiny_config(), steps=20, seed=3)
    second = train("diffusion", windows, tiny_config(), steps=20, seed=3)
    for a, b in zip(first.model.state_dict().values(), second.model.state_dict().values(),
                    strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    obs = torch.zeros(2, 2, 3)
    torch.testing.assert_close(first.model.predict(obs, torch.Generator().manual_seed(7)),
                               second.model.predict(obs, torch.Generator().manual_seed(7)),
                               rtol=0, atol=0)
    other = train("diffusion", windows, tiny_config(), steps=20, seed=4)
    assert not torch.equal(next(other.model.parameters()), next(first.model.parameters()))


def test_checkpoint_round_trip(tmp_path):
    windows = DemoWindows(synthetic_episodes(), FIELDS, obs_horizon=2, pred_horizon=16)
    trained = train("diffusion", windows, tiny_config(), steps=5, seed=0,
                    record={"dataset_sha256": "abc"})
    trained.save(tmp_path / "model.pt")
    loaded = Trained.load(tmp_path / "model.pt")
    assert loaded.fields == FIELDS and loaded.step == 5 and loaded.method == "diffusion"
    assert loaded.record["dataset_sha256"] == "abc" and "commit" in loaded.record["code"]
    obs = torch.ones(1, 2, 3)
    torch.testing.assert_close(trained.model.predict(obs, torch.Generator().manual_seed(0)),
                               loaded.model.predict(obs, torch.Generator().manual_seed(0)))


def test_policy_executes_chunks_and_reads_only_its_fields():
    windows = DemoWindows(synthetic_episodes(), FIELDS, obs_horizon=2, pred_horizon=16)
    trained = train("diffusion", windows, tiny_config(), steps=5, seed=0)
    execution = trained.config["horizons"]["execution"]

    class Strict(dict):
        def __getitem__(self, key):
            assert key in FIELDS, key
            return super().__getitem__(key)

    observation = Strict(code=np.zeros(2), phase=np.zeros(1), extra=np.zeros(4))
    policy = LearnedPolicy(trained)
    policy.reset(None, seed=11)
    first = [policy.act(observation) for _ in range(execution + 1)]
    assert policy.inferences == 2 and first[0].shape == (3,)
    policy.reset(None, seed=11)
    again = [policy.act(observation) for _ in range(execution + 1)]
    np.testing.assert_array_equal(np.stack(first), np.stack(again))


def test_mismatched_window_horizons_are_rejected():
    windows = DemoWindows(synthetic_episodes(), FIELDS, obs_horizon=2, pred_horizon=8)
    with pytest.raises(ValueError):
        train("mlp", windows, tiny_config(), steps=1)


def test_unknown_method_is_rejected():
    with pytest.raises(KeyError):
        build_model("transformer", 3, 3, load_method_config())
    assert isinstance(build_model("diffusion", 3, 3, load_method_config()), DiffusionModel)
