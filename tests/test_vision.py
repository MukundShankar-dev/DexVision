import copy

import numpy as np
import pytest
import torch

from dexvision.learning.dataset import DemoWindows
from dexvision.learning.policy import LearnedPolicy
from dexvision.learning.train import Trained, build_model, load_method_config, train
from dexvision.learning.vision import CameraEncoders, crop

FIELDS = ("phase",)
CAMERAS = ("cam",)
SIZE = 32


def tiny_config(steps=300):
    config = copy.deepcopy(load_method_config())
    config["diffusion"].update(down_dims=[16, 32], step_embedding_dim=16, group_norm_groups=4)
    config["mlp"]["hidden_dims"] = [128, 128]
    # a 30 of 32 px crop: a crop shifts the square but not its target position, which leaves an
    # error floor that grows with the crop margin
    config["vision"].update(crop_px=30, channels=[8, 16], group_norm_groups=4, keypoints=8,
                            feature_dim=16)
    config["training"].update(steps=steps, batch_size=32, warmup_steps=20, learning_rate=1e-3,
                              checkpoint_every=10**9)
    return config


def square_episodes(n_episodes=12, length=6, seed=0):
    """A bright square sits somewhere in a dark image for a whole episode; the action is its
    position. The low-dimensional observation carries nothing about it, so a policy can only
    succeed by reading the image."""
    rng = np.random.default_rng(seed)
    episodes = []
    for _ in range(n_episodes):
        y, x = rng.integers(6, SIZE - 12, size=2)
        frame = np.zeros((SIZE, SIZE, 3), np.uint8)
        frame[y:y + 6, x:x + 6] = 255
        t = np.arange(length)
        episodes.append({"obs/phase": (t / length)[:, None],
                         "obs/rgb/cam": np.repeat(frame[None], length, axis=0),
                         "action": np.tile([x / SIZE, y / SIZE], (length, 1))})
    return episodes


def windows(episodes):
    return DemoWindows(episodes, FIELDS, obs_horizon=2, pred_horizon=16, cameras=CAMERAS)


def test_windows_carry_images_on_the_observation_rows():
    w = windows(square_episodes(2, 4))
    images = w.window_images(np.array([0, 5]))
    assert images["cam"].shape == (2, 2, SIZE, SIZE, 3) and images["cam"].dtype == np.uint8
    np.testing.assert_array_equal(images["cam"][1, 0], square_episodes(2, 4)[1]["obs/rgb/cam"][0])


def test_crop_is_centred_when_acting_and_shared_over_history_when_training():
    images = torch.arange(2 * 2 * 8 * 8 * 3, dtype=torch.int64).view(2, 2, 8, 8, 3)
    centre = crop(images, 6, random=False)
    assert torch.equal(centre, images[:, :, 1:7, 1:7])
    torch.manual_seed(0)
    random = crop(images, 6, random=True)
    assert random.shape == (2, 2, 6, 6, 3)
    for b in range(2):  # each sample's crop is some 6x6 window, the same at both steps
        offsets = [(top, left) for top in range(3) for left in range(3)
                   if torch.equal(random[b], images[b, :, top:top + 6, left:left + 6])]
        assert len(offsets) == 1


def test_encoder_and_model_shapes():
    config = tiny_config()
    encoder = CameraEncoders(("a", "b"), config["vision"]).eval()
    images = {c: torch.zeros(3, 2, SIZE, SIZE, 3, dtype=torch.uint8) for c in ("a", "b")}
    assert encoder(images).shape == (3, 2, 2 * config["vision"]["feature_dim"])
    obs, actions = torch.zeros(3, 2, 5), torch.zeros(3, 16, 4)
    for method in ("diffusion", "mlp"):
        model = build_model(method, 5, 4, config, ("a", "b")).eval()
        assert model.loss(obs, actions, images).shape == ()
        assert model.predict(obs, torch.Generator().manual_seed(0), images).shape == (3, 16, 4)


def test_models_without_cameras_have_no_image_encoder():
    config = load_method_config()
    for method in ("diffusion", "mlp"):
        model = build_model(method, 5, 4, config)
        assert not any(name.startswith("vision.") for name in model.state_dict())


def image_fit_error(trained, w):
    obs, actions = w.window(np.arange(len(w)))
    obs = torch.as_tensor(trained.obs_norm.encode(obs), dtype=torch.float32)
    images = {c: torch.as_tensor(v) for c, v in w.window_images(np.arange(len(w))).items()}
    chunk = trained.model.predict(obs, torch.Generator().manual_seed(1), images)
    return np.abs(trained.action_norm.decode(chunk.double().numpy()) - actions).mean()


def test_mlp_learns_actions_that_only_the_image_shows():
    w = windows(square_episodes())
    trained = train("mlp", w, tiny_config(steps=600), seed=0)
    spread = np.abs(w.actions - w.actions.mean(axis=0)).mean()  # a constant guess
    error = image_fit_error(trained, w)
    assert error < 0.25 * spread, (error, spread)


@pytest.mark.slow
def test_diffusion_learns_actions_that_only_the_image_shows():
    w = windows(square_episodes())
    trained = train("diffusion", w, tiny_config(steps=3000), seed=0)
    spread = np.abs(w.actions - w.actions.mean(axis=0)).mean()
    assert image_fit_error(trained, w) < 0.25 * spread


def test_training_with_images_is_deterministic_on_cpu():
    w = windows(square_episodes(4, 4))
    first = train("diffusion", w, tiny_config(), steps=10, seed=2)
    second = train("diffusion", w, tiny_config(), steps=10, seed=2)
    for a, b in zip(first.model.state_dict().values(), second.model.state_dict().values(),
                    strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_checkpoint_round_trip_with_cameras(tmp_path):
    w = windows(square_episodes(4, 4))
    trained = train("mlp", w, tiny_config(), steps=5, seed=0)
    trained.save(tmp_path / "model.pt")
    loaded = Trained.load(tmp_path / "model.pt")
    assert loaded.cameras == CAMERAS
    obs = torch.zeros(1, 2, 1)
    images = {"cam": torch.as_tensor(w.window_images(0)["cam"])[None]}
    torch.testing.assert_close(trained.model.predict(obs, None, images),
                               loaded.model.predict(obs, None, images))


def test_policy_reads_its_camera_images():
    w = windows(square_episodes(4, 4))
    trained = train("mlp", w, tiny_config(), steps=5, seed=0)
    policy = LearnedPolicy(trained)
    policy.reset(None, 0)
    frame = square_episodes(4, 4)[0]["obs/rgb/cam"][0]
    assert policy.act({"phase": np.zeros(1), "rgb/cam": frame}).shape == (2,)
    policy.reset(None, 0)
    with pytest.raises(KeyError):
        policy.act({"phase": np.zeros(1)})


def test_windows_from_files_match_windows_in_memory(tmp_path):
    episodes = square_episodes(3, 5)
    for i, e in enumerate(episodes):
        np.savez_compressed(tmp_path / f"episode_{i:06d}.npz", **e)
    loaded = DemoWindows.from_directory(tmp_path, FIELDS, 2, 16, CAMERAS)
    expected = windows(episodes)
    assert loaded.cameras == CAMERAS
    np.testing.assert_array_equal(loaded.images["cam"], expected.images["cam"])
    np.testing.assert_array_equal(loaded.observations, expected.observations)
    np.testing.assert_array_equal(loaded.obs_index, expected.obs_index)
