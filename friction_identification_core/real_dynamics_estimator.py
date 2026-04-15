from __future__ import annotations

"""Inverse-dynamics helper used to isolate real-world friction residual torque."""

import numpy as np

from .am_d02_scene import build_am_d02_model


class RealDynamicsEstimator:
    """Rigid-body inverse dynamics without friction for real-data residual fitting."""

    def __init__(
        self,
        *,
        model_path: str,
        joint_names: list[str],
        tcp_offset: np.ndarray,
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        self.model = build_am_d02_model(model_path, np.asarray(tcp_offset, dtype=np.float64))
        self.data = mujoco.MjData(self.model)
        self.qpos_addrs = []
        self.dof_addrs = []

        for name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"找不到关节: {name}")
            self.qpos_addrs.append(self.model.jnt_qposadr[joint_id])
            self.dof_addrs.append(self.model.jnt_dofadr[joint_id])

        self.qpos_addrs = np.asarray(self.qpos_addrs, dtype=np.int32)
        self.dof_addrs = np.asarray(self.dof_addrs, dtype=np.int32)
        # Zero these terms so inverse dynamics returns only rigid-body contributions.
        self.model.dof_frictionloss[self.dof_addrs] = 0.0
        self.model.dof_damping[self.dof_addrs] = 0.0

    def _assign_state(self, q: np.ndarray, qd: np.ndarray | None = None, qdd: np.ndarray | None = None) -> None:
        """Write one joint state sample into MuJoCo state buffers."""

        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.size != self.qpos_addrs.size:
            raise ValueError("q size does not match configured robot joints.")

        if qd is None:
            qd = np.zeros(self.dof_addrs.size, dtype=np.float64)
        else:
            qd = np.asarray(qd, dtype=np.float64).reshape(-1)
        if qdd is None:
            qdd = np.zeros(self.dof_addrs.size, dtype=np.float64)
        else:
            qdd = np.asarray(qdd, dtype=np.float64).reshape(-1)

        if qd.size != self.dof_addrs.size or qdd.size != self.dof_addrs.size:
            raise ValueError("qd/qdd size does not match configured robot joints.")

        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        for idx in range(self.qpos_addrs.size):
            self.data.qpos[self.qpos_addrs[idx]] = q[idx]
            self.data.qvel[self.dof_addrs[idx]] = qd[idx]
            self.data.qacc[self.dof_addrs[idx]] = qdd[idx]

    def inverse_dynamics(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
        """Return rigid-body inverse dynamics torque for one state sample."""

        q = np.asarray(q, dtype=np.float64).reshape(-1)
        qd = np.asarray(qd, dtype=np.float64).reshape(-1)
        qdd = np.asarray(qdd, dtype=np.float64).reshape(-1)
        if q.size != self.qpos_addrs.size or qd.size != self.dof_addrs.size or qdd.size != self.dof_addrs.size:
            raise ValueError("q/qd/qdd size does not match configured robot joints.")

        self._assign_state(q, qd, qdd)
        self._mujoco.mj_inverse(self.model, self.data)
        return self.data.qfrc_inverse[self.dof_addrs].copy()

    def gravity_torque(self, q: np.ndarray) -> np.ndarray:
        """Return gravity/bias torque for a static configuration."""

        self._assign_state(q)
        self._mujoco.mj_forward(self.model, self.data)
        return self.data.qfrc_bias[self.dof_addrs].copy()

    def bias_torque(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """Return Coriolis plus gravity bias torque for a state sample."""

        self._assign_state(q, qd)
        self._mujoco.mj_forward(self.model, self.data)
        return self.data.qfrc_bias[self.dof_addrs].copy()

    def batch_inverse_dynamics(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
        """Evaluate inverse dynamics over a batch of [N, J] trajectories."""

        q = np.asarray(q, dtype=np.float64)
        qd = np.asarray(qd, dtype=np.float64)
        qdd = np.asarray(qdd, dtype=np.float64)
        if q.shape != qd.shape or q.shape != qdd.shape or q.ndim != 2:
            raise ValueError("q/qd/qdd must be same-shape 2D arrays.")

        tau = np.zeros_like(q, dtype=np.float64)
        for sample_idx in range(q.shape[0]):
            tau[sample_idx] = self.inverse_dynamics(q[sample_idx], qd[sample_idx], qdd[sample_idx])
        return tau
