from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dexvision.apps import replay_demo as replay_app
from dexvision.logging import replay_demo as replay_module
from dexvision.logging.demo_logger import (
    DemoLogger,
    DemoStepData,
    build_level1_action_schema,
    build_level2_observation_schema,
)
from dexvision.logging.replay_demo import (
    DemoReplayError,
    iter_replay_steps,
    load_replay_demo,
    replay_loaded_demo,
)
from dexvision.sim.mujoco_env import MujocoEnv
from dexvision.sim.tasks import DEFAULT_TASK_BOARD_MODEL, PushCubeTask


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "assets" / "mujoco" / "hand_scene.xml"
TARGET_NAMES = ("rh_A_WRJ2", "rh_A_WRJ1")


class FakeReplayEnv:
    def __init__(self) -> None:
        self.reset_count = 0
        self.mocap_calls: list[tuple[str, np.ndarray, np.ndarray]] = []
        self.joint_target_calls: list[dict[str, float]] = []
        self.step_calls: list[int] = []
        self.time = 0.0

    def reset(self) -> object:
        self.reset_count += 1
        self.time = 0.0
        return SimpleNamespace(time=self.time)

    def set_mocap_pose(
        self,
        body_name: str,
        *,
        position: np.ndarray,
        orientation_quat: np.ndarray,
    ) -> None:
        self.mocap_calls.append(
            (
                body_name,
                np.asarray(position, dtype=np.float64).copy(),
                np.asarray(orientation_quat, dtype=np.float64).copy(),
            )
        )

    def set_joint_targets(self, joint_targets: dict[str, float]) -> None:
        self.joint_target_calls.append(dict(joint_targets))

    def step(self, *, n_steps: int = 1) -> object:
        self.step_calls.append(n_steps)
        self.time += 0.002 * n_steps
        return SimpleNamespace(time=self.time)


def _metadata() -> dict:
    return {
        "skill_name": "free_space_gesture",
        "task_name": "Free Space Gesture",
        "task_id": "free_space_gesture",
        "episode_id": "replay_0001",
        "robot_model": str(MODEL_PATH),
        "retargeter_config": "configs/level1_teleop.yaml",
        "control_rate_hz": 30.0,
        "teleop_config": {
            "base_control": {
                "enable_base_control": True,
                "mocap_body": "dexvision_hand_base_target",
            }
        },
        "task_config": {
            "required_objects": (),
            "requires_task_state": False,
            "requires_success_metric_inputs": False,
            "required_observation_fields": (),
        },
    }


def _step(index: int, action_dim: int) -> DemoStepData:
    action = np.zeros(action_dim, dtype=np.float64)
    action[:3] = np.asarray([0.01 * index, -0.02 * index, 0.14], dtype=np.float64)
    action[3:7] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    action[7:9] = np.asarray([0.05 * index, -0.03 * index], dtype=np.float64)
    return DemoStepData(
        features=np.full(14, index, dtype=np.float64),
        action=action,
        robot_state=np.full(4 + 4 + len(TARGET_NAMES) + 3 + 4, index, dtype=np.float64),
        tracking_quality=np.asarray([1.0, 0.0, 0.9, 0.8, 0.0, 0.0], dtype=np.float64),
        timestamp=float(index) * 0.1,
        landmarks=np.full((21, 3), index, dtype=np.float64),
    )


def _write_demo(tmp_path: Path, *, steps: int = 3) -> Path:
    demo_dir = tmp_path / "free_space_gesture" / "2026-06-14_001"
    action_schema = build_level1_action_schema(TARGET_NAMES)
    observation_schema = build_level2_observation_schema(
        robot_qpos_dim=4,
        robot_qvel_dim=4,
        finger_target_dim=len(TARGET_NAMES),
        tracking_quality_dim=6,
    )
    logger = DemoLogger(
        demo_dir,
        action_schema=action_schema,
        observation_schema=observation_schema,
    )
    logger.start_episode(_metadata())
    for index in range(steps):
        logger.append(_step(index, action_schema.action_dim))
    logger.close(success=True)
    return demo_dir


def test_load_replay_demo_reconstructs_saved_schema_and_steps(tmp_path: Path) -> None:
    demo_dir = _write_demo(tmp_path)

    loaded = load_replay_demo(demo_dir)
    replay_steps = iter_replay_steps(loaded)

    assert loaded.action_schema.action_dim == 9
    assert loaded.finger_target_names == TARGET_NAMES
    assert loaded.model_path == MODEL_PATH
    assert loaded.mocap_body_name == "dexvision_hand_base_target"
    assert len(replay_steps) == 3
    assert replay_steps[2].base_position_target == pytest.approx([0.02, -0.04, 0.14])
    assert replay_steps[2].base_orientation_target == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert replay_steps[2].finger_actuator_targets == pytest.approx(
        {"rh_A_WRJ2": 0.1, "rh_A_WRJ1": -0.06}
    )


def test_replay_loaded_demo_applies_full_base_and_finger_actions(tmp_path: Path) -> None:
    loaded = load_replay_demo(_write_demo(tmp_path))
    env = FakeReplayEnv()
    sleeps: list[float] = []

    result = replay_loaded_demo(
        loaded,
        env,
        speed=2.0,
        sim_steps_per_action=3,
        sleep_fn=sleeps.append,
    )

    assert result.steps_replayed == 3
    assert result.final_sim_time == pytest.approx(0.018)
    assert env.reset_count == 1
    assert env.step_calls == [3, 3, 3]
    assert [call[0] for call in env.mocap_calls] == ["dexvision_hand_base_target"] * 3
    assert env.mocap_calls[1][1] == pytest.approx([0.01, -0.02, 0.14])
    assert env.joint_target_calls[2] == pytest.approx({"rh_A_WRJ2": 0.1, "rh_A_WRJ1": -0.06})
    assert sleeps == pytest.approx([0.05, 0.05])


def test_replay_loaded_demo_can_preserve_a_prepared_environment(tmp_path: Path) -> None:
    loaded = load_replay_demo(_write_demo(tmp_path, steps=1))
    env = FakeReplayEnv()
    env.time = 4.0

    result = replay_loaded_demo(
        loaded,
        env,
        speed=1000.0,
        sim_steps_per_action=2,
        reset_env=False,
        sleep_fn=lambda _delay: None,
    )

    assert env.reset_count == 0
    assert result.final_sim_time == pytest.approx(4.004)


def test_replay_loader_reports_missing_required_arrays(tmp_path: Path) -> None:
    demo_dir = _write_demo(tmp_path)
    (demo_dir / "actions.npy").unlink()

    with pytest.raises(DemoReplayError, match="actions.npy"):
        load_replay_demo(demo_dir)


def test_replay_loader_requires_finger_target_names(tmp_path: Path) -> None:
    demo_dir = _write_demo(tmp_path)
    metadata_path = demo_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("finger_target_names", None)
    metadata["action_schema"]["representation_notes"].pop("finger_target_names", None)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(DemoReplayError, match="finger_target_names"):
        load_replay_demo(demo_dir)


def test_replay_loader_explicitly_adapts_legacy_level2_observation_v1(
    tmp_path: Path,
) -> None:
    demo_dir = _write_demo(tmp_path)
    metadata_path = demo_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    observation = metadata["observation_schema"]
    observation["version"] = "level2/observation-v1"
    observation["fields"].remove("actuator_controls")
    observation["shapes"].pop("actuator_controls")
    observation.pop("layouts")
    metadata["observation_schema_version"] = "level2/observation-v1"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    loaded = load_replay_demo(demo_dir)

    assert loaded.observation_schema.executable is False
    assert "shape-only" in loaded.observation_schema.compatibility_notes[0]
    assert len(iter_replay_steps(loaded)) == 3


def test_replay_demo_parser_accepts_progress_command() -> None:
    parser = replay_app.build_parser()

    args = parser.parse_args(
        [
            "--demo",
            "data/demos/raw/free_space_gesture/2026-06-14_001",
            "--headless",
            "--speed",
            "0.5",
        ]
    )

    assert args.demo == Path("data/demos/raw/free_space_gesture/2026-06-14_001")
    assert args.headless is True
    assert args.speed == pytest.approx(0.5)
    assert args.sim_steps_per_action is None


def test_replay_demo_uses_saved_recording_cadence_by_default() -> None:
    loaded = SimpleNamespace(
        episode=SimpleNamespace(
            metadata={"recording": {"sim_steps_per_frame": 17}}
        )
    )

    assert replay_app._resolve_sim_steps_per_action(loaded, None) == 17
    assert replay_app._resolve_sim_steps_per_action(loaded, 3) == 3


def test_replay_demo_uses_one_step_for_legacy_recordings() -> None:
    loaded = SimpleNamespace(episode=SimpleNamespace(metadata={}))

    assert replay_app._resolve_sim_steps_per_action(loaded, None) == 1


def test_replay_demo_rejects_invalid_saved_recording_cadence() -> None:
    loaded = SimpleNamespace(
        episode=SimpleNamespace(
            metadata={"recording": {"sim_steps_per_frame": 0}}
        )
    )

    with pytest.raises(DemoReplayError, match="sim_steps_per_frame"):
        replay_app._resolve_sim_steps_per_action(loaded, None)


def test_replay_demo_headless_smoke_runs_with_mujoco(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    demo_dir = _write_demo(tmp_path, steps=2)

    result = replay_app.main(
        [
            "--demo",
            str(demo_dir),
            "--headless",
            "--speed",
            "1000",
            "--max-steps",
            "2",
        ]
    )

    assert result == 0


def test_push_cube_semantic_replay_restores_saved_start_and_target() -> None:
    pytest.importorskip("mujoco")
    with PushCubeTask(DEFAULT_TASK_BOARD_MODEL) as task:
        initial = task.reset(seed=4)
        metadata = {
            "task_id": "push_cube_to_target",
            "task_config": {
                "resolved_object_id": initial.object_id,
                "initial_object_position": [0.071, -0.019, -0.015],
                "initial_object_orientation": [1.0, 0.0, 0.0, 0.0],
                "initial_object_linear_velocity": [0.0, 0.0, 0.0],
                "initial_object_angular_velocity": [0.0, 0.0, 0.0],
                "initial_base_position": [-0.52, 0.07, 0.02],
                "initial_base_orientation": [
                    0.7071067811865476,
                    0.0,
                    0.7071067811865476,
                    0.0,
                ],
                "base_free_joint": "rh_base_freejoint",
                "target_position": [-0.082, 0.041, -0.015],
                "target_marker_body": task.config.target_marker_body,
            },
        }

    loaded = SimpleNamespace(
        episode=SimpleNamespace(metadata=metadata),
        mocap_body_name="dexvision_hand_base_target",
    )
    with MujocoEnv(DEFAULT_TASK_BOARD_MODEL) as env:
        env.reset()
        replay_module._restore_task_replay_state(loaded, env)
        joint_id = env._mujoco.mj_name2id(
            env.model,
            env._mujoco.mjtObj.mjOBJ_JOINT,
            "push_cube_joint",
        )
        qpos_address = int(env.model.jnt_qposadr[joint_id])
        base_joint_id = env._mujoco.mj_name2id(
            env.model,
            env._mujoco.mjtObj.mjOBJ_JOINT,
            "rh_base_freejoint",
        )
        base_qpos_address = int(env.model.jnt_qposadr[base_joint_id])
        marker_position, _ = env.get_mocap_pose("push_cube_target_marker")

        assert env.data.qpos[qpos_address : qpos_address + 3] == pytest.approx(
            [0.071, -0.019, -0.015]
        )
        assert env.data.qpos[
            base_qpos_address : base_qpos_address + 3
        ] == pytest.approx([-0.52, 0.07, 0.02])
        assert marker_position == pytest.approx([-0.082, 0.041, -0.015])


def test_push_cube_semantic_replay_rejects_malformed_start_state() -> None:
    loaded = SimpleNamespace(
        episode=SimpleNamespace(
            metadata={
                "task_id": "push_cube_to_target",
                "task_config": {
                    "resolved_object_id": "push_cube",
                    "initial_object_position": [0.0, 0.0],
                    "initial_object_orientation": [1.0, 0.0, 0.0, 0.0],
                    "initial_object_linear_velocity": [0.0, 0.0, 0.0],
                    "initial_object_angular_velocity": [0.0, 0.0, 0.0],
                    "initial_base_position": [-0.52, 0.0, 0.02],
                    "initial_base_orientation": [
                        0.7071067811865476,
                        0.0,
                        0.7071067811865476,
                        0.0,
                    ],
                    "target_position": [0.0, 0.0, -0.015],
                },
            }
        ),
        mocap_body_name="dexvision_hand_base_target",
    )

    with pytest.raises(DemoReplayError, match="initial_object_position"):
        replay_module._restore_task_replay_state(loaded, FakeReplayEnv())
