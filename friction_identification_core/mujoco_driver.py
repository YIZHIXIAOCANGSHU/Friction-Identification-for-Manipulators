from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np

from .models import FrictionSampleBatch


@dataclass
class MujocoFrictionCollectionConfig:
    duration: float = 18.0
    sample_rate: float = 400.0
    timestep: float = 0.0005
    base_frequency: float = 0.12
    amplitude_scale: float = 0.18
    render: bool = False
    realtime: bool = False
    feedback_scale: float = 0.12


class MujocoFrictionCollector:
    def __init__(
        self,
        model_path: str,
        joint_names: list[str],
        actuator_names: list[str],
        *,
        timestep: float = 0.0005,
        render: bool = False,
    ):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = timestep
        self.render = render
        self.viewer = None
        self.joint_names = joint_names
        self.actuator_names = actuator_names
        self.joint_ids = []
        self.dof_addrs = []
        self.qpos_addrs = []
        self.actuator_ids = []

        for name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"找不到关节: {name}")
            self.joint_ids.append(joint_id)
            self.dof_addrs.append(self.model.jnt_dofadr[joint_id])
            self.qpos_addrs.append(self.model.jnt_qposadr[joint_id])

        for name in actuator_names:
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator_id < 0:
                raise ValueError(f"找不到执行器: {name}")
            self.actuator_ids.append(actuator_id)

        self.kp = np.array([100.0, 100.0, 80.0, 80.0, 50.0, 50.0, 50.0], dtype=np.float64)
        self.kd = np.array([10.0, 10.0, 8.0, 8.0, 5.0, 5.0, 5.0], dtype=np.float64)

    def _get_home_joint_positions(self) -> np.ndarray:
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            return np.array([self.model.key_qpos[key_id, addr] for addr in self.qpos_addrs], dtype=np.float64)

        ranges = self.model.jnt_range[self.joint_ids]
        centers = np.mean(ranges, axis=1)
        return centers.astype(np.float64)

    def get_true_friction_parameters(self) -> tuple[np.ndarray, np.ndarray]:
        coulomb = np.array([self.model.dof_frictionloss[addr] for addr in self.dof_addrs], dtype=np.float64)
        viscous = np.array([self.model.dof_damping[addr] for addr in self.dof_addrs], dtype=np.float64)
        return coulomb, viscous

    def generate_excitation_trajectory(
        self,
        *,
        duration: float,
        sample_rate: float,
        base_frequency: float,
        amplitude_scale: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        num_samples = int(duration * sample_rate)
        t = np.linspace(0.0, duration, num_samples, endpoint=False)
        num_joints = len(self.joint_ids)
        q_cmd = np.zeros((num_samples, num_joints), dtype=np.float64)
        qd_cmd = np.zeros_like(q_cmd)
        qdd_cmd = np.zeros_like(q_cmd)

        ranges = self.model.jnt_range[self.joint_ids]
        home = self._get_home_joint_positions()

        for joint_idx in range(num_joints):
            lower, upper = ranges[joint_idx]
            if lower < upper:
                center = 0.5 * (lower + upper)
                half_range = 0.5 * (upper - lower)
                amplitude = min(amplitude_scale * half_range, 0.35 * half_range, 0.18)
            else:
                center = home[joint_idx]
                amplitude = 0.18

            base = base_frequency * (1.0 + 0.14 * joint_idx)
            omega_1 = 2.0 * np.pi * base
            omega_2 = 2.0 * omega_1
            phase_1 = 0.45 * joint_idx
            phase_2 = 0.9 + 0.3 * joint_idx
            amp_2 = 0.35 * amplitude

            q_cmd[:, joint_idx] = (
                center
                + amplitude * np.sin(omega_1 * t + phase_1)
                + amp_2 * np.sin(omega_2 * t + phase_2)
            )
            qd_cmd[:, joint_idx] = (
                amplitude * omega_1 * np.cos(omega_1 * t + phase_1)
                + amp_2 * omega_2 * np.cos(omega_2 * t + phase_2)
            )
            qdd_cmd[:, joint_idx] = (
                -amplitude * omega_1 ** 2 * np.sin(omega_1 * t + phase_1)
                - amp_2 * omega_2 ** 2 * np.sin(omega_2 * t + phase_2)
            )

        return t, q_cmd, qd_cmd, qdd_cmd

    def reset(self, q_init: np.ndarray) -> None:
        mujoco.mj_resetData(self.model, self.data)
        for qpos_addr, value in zip(self.qpos_addrs, q_init):
            self.data.qpos[qpos_addr] = value
        mujoco.mj_forward(self.model, self.data)

    def _inverse_dynamics(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
        qpos_backup = self.data.qpos.copy()
        qvel_backup = self.data.qvel.copy()
        qacc_backup = self.data.qacc.copy()

        for qpos_addr, dof_addr, q_i, qd_i, qdd_i in zip(self.qpos_addrs, self.dof_addrs, q, qd, qdd):
            self.data.qpos[qpos_addr] = q_i
            self.data.qvel[dof_addr] = qd_i
            self.data.qacc[dof_addr] = qdd_i

        mujoco.mj_inverse(self.model, self.data)
        tau = np.array([self.data.qfrc_inverse[dof_addr] for dof_addr in self.dof_addrs], dtype=np.float64)

        self.data.qpos[:] = qpos_backup
        self.data.qvel[:] = qvel_backup
        self.data.qacc[:] = qacc_backup
        return tau

    def _get_joint_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = np.array([self.data.qpos[addr] for addr in self.qpos_addrs], dtype=np.float64)
        qd = np.array([self.data.qvel[addr] for addr in self.dof_addrs], dtype=np.float64)
        qdd = np.array([self.data.qacc[addr] for addr in self.dof_addrs], dtype=np.float64)
        return q, qd, qdd

    def _get_friction_components(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tau_passive = -np.array([self.data.qfrc_passive[addr] for addr in self.dof_addrs], dtype=np.float64)
        tau_constraint = -np.array([self.data.qfrc_constraint[addr] for addr in self.dof_addrs], dtype=np.float64)
        return tau_passive, tau_constraint, tau_passive + tau_constraint

    def _set_torque(self, tau: np.ndarray) -> None:
        for actuator_id, torque in zip(self.actuator_ids, tau):
            self.data.ctrl[actuator_id] = float(torque)

    def collect(
        self,
        q_cmd: np.ndarray,
        qd_cmd: np.ndarray,
        qdd_cmd: np.ndarray,
        *,
        sample_rate: float,
        feedback_scale: float,
        realtime: bool,
    ) -> FrictionSampleBatch:
        num_samples = q_cmd.shape[0]
        self.reset(q_cmd[0])

        if self.render:
            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
            )
            self.viewer.cam.azimuth = 135
            self.viewer.cam.elevation = -20
            self.viewer.cam.distance = 2.0
            self.viewer.cam.lookat[:] = [0.0, 0.0, 0.55]

        time_log = np.zeros(num_samples, dtype=np.float64)
        q_log = np.zeros_like(q_cmd)
        qd_log = np.zeros_like(q_cmd)
        qdd_log = np.zeros_like(q_cmd)
        tau_ctrl_log = np.zeros_like(q_cmd)
        tau_passive_log = np.zeros_like(q_cmd)
        tau_constraint_log = np.zeros_like(q_cmd)
        tau_friction_log = np.zeros_like(q_cmd)

        sim_time = 0.0
        sample_dt = 1.0 / sample_rate
        wall_start = time.time()

        for sample_idx in range(num_samples):
            q_curr, qd_curr, _ = self._get_joint_state()
            tau_ff = self._inverse_dynamics(q_cmd[sample_idx], qd_cmd[sample_idx], qdd_cmd[sample_idx])
            tau_fb = self.kp * (q_cmd[sample_idx] - q_curr) + self.kd * (qd_cmd[sample_idx] - qd_curr)
            tau_ctrl = tau_ff + feedback_scale * tau_fb
            self._set_torque(tau_ctrl)

            next_sample_time = (sample_idx + 1) * sample_dt
            while sim_time + 1e-12 < next_sample_time:
                mujoco.mj_step(self.model, self.data)
                sim_time += self.model.opt.timestep
                if self.viewer is not None and self.viewer.is_running():
                    self.viewer.sync()
                if realtime:
                    elapsed = time.time() - wall_start
                    if sim_time > elapsed:
                        time.sleep(sim_time - elapsed)

            q_meas, qd_meas, qdd_meas = self._get_joint_state()
            tau_passive, tau_constraint, tau_friction = self._get_friction_components()

            time_log[sample_idx] = sim_time
            q_log[sample_idx] = q_meas
            qd_log[sample_idx] = qd_meas
            qdd_log[sample_idx] = qdd_meas
            tau_ctrl_log[sample_idx] = tau_ctrl
            tau_passive_log[sample_idx] = tau_passive
            tau_constraint_log[sample_idx] = tau_constraint
            tau_friction_log[sample_idx] = tau_friction

        return FrictionSampleBatch(
            time=time_log,
            q=q_log,
            qd=qd_log,
            qdd=qdd_log,
            q_cmd=q_cmd,
            qd_cmd=qd_cmd,
            qdd_cmd=qdd_cmd,
            tau_ctrl=tau_ctrl_log,
            tau_passive=tau_passive_log,
            tau_constraint=tau_constraint_log,
            tau_friction=tau_friction_log,
        )

    def build_clean_sample_mask(
        self,
        batch: FrictionSampleBatch,
        *,
        limit_margin: float = 0.05,
        constraint_tolerance: float = 0.35,
    ) -> np.ndarray:
        lower = self.model.jnt_range[self.joint_ids, 0][None, :]
        upper = self.model.jnt_range[self.joint_ids, 1][None, :]
        margin_to_limits = np.minimum(batch.q - lower, upper - batch.q)
        away_from_limits = np.all(margin_to_limits > limit_margin, axis=1)
        constraint_is_clean = np.all(np.abs(batch.tau_constraint) < constraint_tolerance, axis=1)
        finite = np.all(np.isfinite(batch.qd), axis=1) & np.all(np.isfinite(batch.tau_friction), axis=1)
        moving = np.any(np.abs(batch.qd) > 0.02, axis=1)
        return away_from_limits & constraint_is_clean & finite & moving

    def run_openarm_collection(self, config: MujocoFrictionCollectionConfig) -> FrictionSampleBatch:
        t, q_cmd, qd_cmd, qdd_cmd = self.generate_excitation_trajectory(
            duration=config.duration,
            sample_rate=config.sample_rate,
            base_frequency=config.base_frequency,
            amplitude_scale=config.amplitude_scale,
        )
        batch = self.collect(
            q_cmd=q_cmd,
            qd_cmd=qd_cmd,
            qdd_cmd=qdd_cmd,
            sample_rate=config.sample_rate,
            feedback_scale=config.feedback_scale,
            realtime=config.realtime,
        )
        batch.time[:] = t[: batch.time.shape[0]]
        return batch

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
