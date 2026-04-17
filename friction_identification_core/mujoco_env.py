from __future__ import annotations

import mujoco
import numpy as np

from friction_identification_core.config import Config
from friction_identification_core.trajectory import (
    ReferenceTrajectory,
    build_excitation_trajectory,
    build_quintic_point_to_point_trajectory,
)
from friction_identification_core.mujoco_support import build_am_d02_model


def _load_model(model_path: str, tcp_offset: np.ndarray) -> mujoco.MjModel:
    if model_path.lower().endswith(".urdf"):
        return build_am_d02_model(model_path, np.asarray(tcp_offset, dtype=np.float64))
    return mujoco.MjModel.from_xml_path(model_path)


class MujocoEnvironment:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.model = _load_model(str(config.robot.urdf_path), config.robot.tcp_offset)
        self.inverse_model = _load_model(str(config.robot.urdf_path), config.robot.tcp_offset)
        self.inverse_data = mujoco.MjData(self.inverse_model)

        self.model.opt.timestep = float(config.sampling.timestep)
        self.inverse_model.opt.timestep = float(config.sampling.timestep)
        self.joint_names = list(config.robot.joint_names)
        self.joint_ids: list[int] = []
        self.qpos_addrs: list[int] = []
        self.dof_addrs: list[int] = []

        for name in self.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"找不到关节: {name}")
            self.joint_ids.append(joint_id)
            self.qpos_addrs.append(int(self.model.jnt_qposadr[joint_id]))
            self.dof_addrs.append(int(self.model.jnt_dofadr[joint_id]))

        self.joint_ids = np.asarray(self.joint_ids, dtype=np.int32)
        self.qpos_addrs = np.asarray(self.qpos_addrs, dtype=np.int32)
        self.dof_addrs = np.asarray(self.dof_addrs, dtype=np.int32)

        self.ee_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            config.robot.end_effector_body,
        )
        if self.ee_body_id < 0:
            raise ValueError(f"找不到末端 body: {config.robot.end_effector_body}")

        self._apply_joint_limit_overrides(config.robot.joint_limits)
        self._apply_friction_overrides(
            self.model,
            np.zeros(config.joint_count, dtype=np.float64),
            np.zeros(config.joint_count, dtype=np.float64),
        )
        self._apply_friction_overrides(
            self.inverse_model,
            np.zeros(config.joint_count, dtype=np.float64),
            np.zeros(config.joint_count, dtype=np.float64),
        )

        self.model.geom_contype[:] = 0
        self.model.geom_conaffinity[:] = 0
        self.inverse_model.geom_contype[:] = 0
        self.inverse_model.geom_conaffinity[:] = 0

    def _apply_joint_limit_overrides(self, joint_limit_overrides: np.ndarray) -> None:
        joint_limit_overrides = np.asarray(joint_limit_overrides, dtype=np.float64)
        if joint_limit_overrides.shape != (self.config.joint_count, 2):
            raise ValueError("joint_limit_overrides must have shape [num_joints, 2].")
        for joint_id, limits in zip(self.joint_ids, joint_limit_overrides):
            self.model.jnt_limited[joint_id] = 1
            self.model.jnt_range[joint_id] = limits
            self.inverse_model.jnt_limited[joint_id] = 1
            self.inverse_model.jnt_range[joint_id] = limits

    def _apply_friction_overrides(
        self,
        model: mujoco.MjModel,
        coulomb: np.ndarray,
        viscous: np.ndarray,
    ) -> None:
        coulomb = np.asarray(coulomb, dtype=np.float64).reshape(-1)
        viscous = np.asarray(viscous, dtype=np.float64).reshape(-1)
        model.dof_frictionloss[self.dof_addrs] = coulomb
        model.dof_damping[self.dof_addrs] = viscous

    def _assign_inverse_state(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> None:
        self.inverse_data.qpos[:] = 0.0
        self.inverse_data.qvel[:] = 0.0
        self.inverse_data.qacc[:] = 0.0
        for qpos_addr, dof_addr, q_i, qd_i, qdd_i in zip(self.qpos_addrs, self.dof_addrs, q, qd, qdd):
            self.inverse_data.qpos[qpos_addr] = q_i
            self.inverse_data.qvel[dof_addr] = qd_i
            self.inverse_data.qacc[dof_addr] = qdd_i

    def inverse_dynamics(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
        self._assign_inverse_state(
            np.asarray(q, dtype=np.float64).reshape(-1),
            np.asarray(qd, dtype=np.float64).reshape(-1),
            np.asarray(qdd, dtype=np.float64).reshape(-1),
        )
        mujoco.mj_inverse(self.inverse_model, self.inverse_data)
        return self.inverse_data.qfrc_inverse[self.dof_addrs].copy()

    def build_excitation_reference(self) -> ReferenceTrajectory:
        return build_excitation_trajectory(self.config)

    def build_startup_reference(self, start_q: np.ndarray, goal_q: np.ndarray) -> ReferenceTrajectory | None:
        start_q = np.asarray(start_q, dtype=np.float64).reshape(-1)
        goal_q = np.asarray(goal_q, dtype=np.float64).reshape(-1)
        if np.allclose(start_q, goal_q, atol=1e-9):
            return None

        sample_rate = float(self.config.sampling.rate)
        unit_reference = build_quintic_point_to_point_trajectory(
            start_q=start_q,
            goal_q=goal_q,
            duration=1.0,
            sample_rate=sample_rate,
            settle_duration=0.0,
        )
        ee_pos_unit, _ = self.evaluate_end_effector_trajectory(unit_reference.q_cmd)
        if ee_pos_unit.shape[0] >= 2:
            ee_speed_unit = np.linalg.norm(np.diff(ee_pos_unit, axis=0), axis=1) * sample_rate
            peak_ee_speed_unit = float(np.max(ee_speed_unit))
        else:
            peak_ee_speed_unit = 0.0

        transition_cfg = self.config.identification.transition
        max_ee_speed = max(float(transition_cfg.max_ee_speed), 1e-6)
        duration = max(float(transition_cfg.min_duration), peak_ee_speed_unit / max_ee_speed)
        return build_quintic_point_to_point_trajectory(
            start_q=start_q,
            goal_q=goal_q,
            duration=duration,
            sample_rate=sample_rate,
            settle_duration=float(transition_cfg.settle_duration),
        )

    def evaluate_end_effector_trajectory(self, q_cmd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q_cmd = np.asarray(q_cmd, dtype=np.float64)
        if q_cmd.ndim != 2 or q_cmd.shape[1] != self.config.joint_count:
            raise ValueError("q_cmd must have shape [N, joint_count].")

        data = mujoco.MjData(self.model)
        ee_pos = np.zeros((q_cmd.shape[0], 3), dtype=np.float64)
        ee_quat = np.zeros((q_cmd.shape[0], 4), dtype=np.float64)
        for sample_idx, q_sample in enumerate(q_cmd):
            data.qpos[:] = 0.0
            data.qvel[:] = 0.0
            data.qacc[:] = 0.0
            for qpos_addr, value in zip(self.qpos_addrs, q_sample):
                data.qpos[qpos_addr] = value
            mujoco.mj_forward(self.model, data)
            ee_pos[sample_idx] = data.xpos[self.ee_body_id].copy()
            ee_quat[sample_idx] = data.xquat[self.ee_body_id].copy()
        return ee_pos, ee_quat

    def close(self) -> None:
        pass
