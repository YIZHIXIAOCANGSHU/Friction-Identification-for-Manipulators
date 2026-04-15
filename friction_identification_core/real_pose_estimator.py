from __future__ import annotations

"""Forward-kinematics helper for visualizing real robot joint states in MuJoCo."""

import time

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
        render: bool = True,
        viewer_fps: float = 30.0,
    ) -> None:
        import mujoco
        import mujoco.viewer

        self._mujoco = mujoco
        self._viewer_module = mujoco.viewer
        self.model = build_am_d02_model(model_path, np.asarray(tcp_offset, dtype=np.float64))
        self.data = mujoco.MjData(self.model)
        self.qpos_addrs = []
        self.viewer = None
        self._viewer_period_s = 1.0 / max(float(viewer_fps), 1.0)
        self._last_viewer_sync = 0.0

        # Cache the qpos slots corresponding to the configured actuated joints.
        for name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"找不到关节: {name}")
            self.qpos_addrs.append(self.model.jnt_qposadr[joint_id])

        self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, end_effector_body)
        if self.ee_body_id < 0:
            raise ValueError(f"找不到末端 body: {end_effector_body}")

        if render:
            self.viewer = self._viewer_module.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
            )
            self.viewer.cam.azimuth = 135
            self.viewer.cam.elevation = -20
            self.viewer.cam.distance = 1.8
            self.viewer.cam.lookat[:] = [0.0, 0.0, 1.1]

    def update(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward the latest measured joint angles and return TCP pose."""

        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.size != len(self.qpos_addrs):
            raise ValueError("q size does not match the configured robot joints.")

        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        for addr, value in zip(self.qpos_addrs, q):
            self.data.qpos[addr] = value
        self._mujoco.mj_forward(self.model, self.data)

        if self.viewer is not None:
            if not self.viewer.is_running():
                self.viewer.close()
                self.viewer = None
            else:
                now = time.perf_counter()
                # Throttle the passive viewer to avoid tying UI refresh to UART rate.
                if now - self._last_viewer_sync >= self._viewer_period_s:
                    self.viewer.sync()
                    self._last_viewer_sync = now

        return self.data.xpos[self.ee_body_id].copy(), self.data.xquat[self.ee_body_id].copy()

    def close(self) -> None:
        """Close the optional passive viewer."""

        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
