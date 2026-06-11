"""Small MuJoCo environment wrapper for DexVision simulation checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import mujoco
except ImportError as exc:  # pragma: no cover - exercised when dependency is absent.
    mujoco = None  # type: ignore[assignment]
    _MUJOCO_IMPORT_ERROR: ImportError | None = exc
else:
    _MUJOCO_IMPORT_ERROR = None


class MujocoError(RuntimeError):
    """Raised when MuJoCo cannot load or step a model."""


@dataclass(frozen=True)
class MujocoState:
    """Snapshot of a MuJoCo simulation state.

    Attributes:
        time: Simulation time in seconds.
        qpos: Generalized positions with shape ``[model.nq]``.
        qvel: Generalized velocities with shape ``[model.nv]``.
        ctrl: Actuator controls with shape ``[model.nu]``.
    """

    time: float
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray


class MujocoEnv:
    """Load, reset, and step a MuJoCo model in headless mode."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise MujocoError(f"MuJoCo model file does not exist: {self.model_path}")
        if not self.model_path.is_file():
            raise MujocoError(f"MuJoCo model path is not a file: {self.model_path}")

        mujoco_module = _load_mujoco()
        try:
            self.model = mujoco_module.MjModel.from_xml_path(str(self.model_path))
        except Exception as exc:  # pragma: no cover - exact MuJoCo exception varies.
            raise MujocoError(f"Failed to load MuJoCo model '{self.model_path}': {exc}") from exc

        self.data = mujoco_module.MjData(self.model)
        self._mujoco = mujoco_module

    def reset(self) -> MujocoState:
        """Reset the simulation data and return the initial state."""

        self._mujoco.mj_resetData(self.model, self.data)
        self._mujoco.mj_forward(self.model, self.data)
        return self.get_state()

    def step(
        self,
        action: Sequence[float] | np.ndarray | Mapping[str, float] | None = None,
        *,
        n_steps: int = 1,
    ) -> MujocoState:
        """Apply an optional action and advance the simulation.

        Args:
            action: Optional actuator controls. A sequence must have length
                ``model.nu``. A mapping is interpreted as actuator-name targets.
            n_steps: Number of MuJoCo integration steps to run.
        """

        if n_steps <= 0:
            raise ValueError("n_steps must be a positive integer.")

        if action is not None:
            if isinstance(action, Mapping):
                self.set_joint_targets(action)
            else:
                self._set_ctrl_array(action)

        for _ in range(n_steps):
            self._mujoco.mj_step(self.model, self.data)

        return self.get_state()

    def set_joint_targets(self, joint_targets: Mapping[str, float]) -> None:
        """Set actuator controls by actuator name.

        The Level 1.5 scene has no actuators, so non-empty target mappings will
        raise a clear error until later hand-model checkpoints add actuators.
        """

        for actuator_name, target in joint_targets.items():
            actuator_id = self._mujoco.mj_name2id(
                self.model,
                self._mujoco.mjtObj.mjOBJ_ACTUATOR,
                actuator_name,
            )
            if actuator_id < 0:
                raise MujocoError(f"Unknown MuJoCo actuator target: {actuator_name}")
            self.data.ctrl[actuator_id] = float(target)

    def set_mocap_pose(
        self,
        body_name: str,
        *,
        position: Sequence[float] | np.ndarray,
        orientation_quat: Sequence[float] | np.ndarray,
    ) -> None:
        """Set a named MuJoCo mocap body pose.

        ``orientation_quat`` uses MuJoCo's ``[w, x, y, z]`` convention.
        """

        mocap_id = self._mocap_id_for_body(body_name)
        position_array = np.asarray(position, dtype=np.float64)
        quat_array = np.asarray(orientation_quat, dtype=np.float64)
        if position_array.shape != (3,):
            raise MujocoError(
                f"Mocap position for '{body_name}' must have shape [3], "
                f"got {position_array.shape}."
            )
        if quat_array.shape != (4,):
            raise MujocoError(
                f"Mocap orientation for '{body_name}' must have shape [4], "
                f"got {quat_array.shape}."
            )
        if not np.all(np.isfinite(position_array)) or not np.all(np.isfinite(quat_array)):
            raise MujocoError(f"Mocap pose for '{body_name}' must be finite.")
        quat_norm = float(np.linalg.norm(quat_array))
        if quat_norm <= 0.0:
            raise MujocoError(f"Mocap orientation for '{body_name}' must be non-zero.")

        self.data.mocap_pos[mocap_id] = position_array
        self.data.mocap_quat[mocap_id] = quat_array / quat_norm

    def get_mocap_pose(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return copies of a named mocap body's position and quaternion."""

        mocap_id = self._mocap_id_for_body(body_name)
        return (
            np.asarray(self.data.mocap_pos[mocap_id], dtype=np.float64).copy(),
            np.asarray(self.data.mocap_quat[mocap_id], dtype=np.float64).copy(),
        )

    def get_state(self) -> MujocoState:
        """Return a copy of the current simulation state."""

        return MujocoState(
            time=float(self.data.time),
            qpos=np.asarray(self.data.qpos, dtype=np.float64).copy(),
            qvel=np.asarray(self.data.qvel, dtype=np.float64).copy(),
            ctrl=np.asarray(self.data.ctrl, dtype=np.float64).copy(),
        )

    def close(self) -> None:
        """Release environment resources.

        MuJoCo owns no external handles for this headless wrapper, but a close
        method keeps the app and future callers explicit about lifecycle.
        """

    def __enter__(self) -> "MujocoEnv":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _set_ctrl_array(self, action: Sequence[float] | np.ndarray) -> None:
        action_array = np.asarray(action, dtype=np.float64)
        expected_shape = (self.model.nu,)
        if action_array.shape != expected_shape:
            raise MujocoError(
                "MuJoCo action shape mismatch: "
                f"expected {expected_shape}, got {action_array.shape}."
            )
        self.data.ctrl[:] = action_array

    def _mocap_id_for_body(self, body_name: str) -> int:
        if not body_name:
            raise MujocoError("Mocap body name cannot be empty.")
        body_id = self._mujoco.mj_name2id(
            self.model,
            self._mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )
        if body_id < 0:
            raise MujocoError(f"Unknown MuJoCo mocap body: {body_name}")
        mocap_id = int(self.model.body_mocapid[body_id])
        if mocap_id < 0:
            raise MujocoError(f"MuJoCo body '{body_name}' is not a mocap body.")
        return mocap_id


def _load_mujoco() -> Any:
    if mujoco is None:
        raise MujocoError(
            "MuJoCo is required for simulation. Activate the dexvision Conda "
            "environment or install the package providing 'mujoco' "
            f"({_MUJOCO_IMPORT_ERROR})."
        )
    return mujoco
