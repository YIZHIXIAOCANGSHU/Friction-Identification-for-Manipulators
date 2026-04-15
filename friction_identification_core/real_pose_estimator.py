from __future__ import annotations

import numpy as np

from .am_d02_scene import build_am_d02_model


class RealPoseEstimator:
    """Compute end-effector pose from measured joint states for live visualization."""

    def __init__(
        self,
        *,
        model_path: str,
        joint_names: list[str],
        end_effector_body: str,
        tcp_offset: np.ndarray,
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        self.model = build_am_d02_model(model_path, np.asarray(tcp_offset, dtype=np.float64))
        self.data = mujoco.MjData(self.model)
        self.qpos_addrs = []

        for name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"找不到关节: {name}")
            self.qpos_addrs.append(self.model.jnt_qposadr[joint_id])

        self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, end_effector_body)
        if self.ee_body_id < 0:
            raise ValueError(f"找不到末端 body: {end_effector_body}")

    def compute(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.size != len(self.qpos_addrs):
            raise ValueError("q size does not match the configured robot joints.")

        self.data.qpos[:] = 0.0
        for addr, value in zip(self.qpos_addrs, q):
            self.data.qpos[addr] = value
        self._mujoco.mj_forward(self.model, self.data)
        return self.data.xpos[self.ee_body_id].copy(), self.data.xquat[self.ee_body_id].copy()
