from __future__ import annotations

import time
from dataclasses import dataclass

import mujoco
import mujoco.viewer
import numpy as np

from friction_identification_core.config import Config
from friction_identification_core.core.controller import FrictionIdentificationController
from friction_identification_core.core.models import FrictionSampleBatch
from friction_identification_core.core.safety import SafetyGuard
from friction_identification_core.core.trajectory import (
    ReferenceTrajectory,
    build_excitation_trajectory,
    build_quintic_point_to_point_trajectory,
    resolve_joint_limit_arrays,
)
from friction_identification_core.utils.mujoco import build_am_d02_model


@dataclass(frozen=True)
class SimulationArtifacts:
    raw_batch: FrictionSampleBatch
    clean_mask: np.ndarray


def _load_model(model_path: str, tcp_offset: np.ndarray) -> mujoco.MjModel:
    if model_path.lower().endswith(".urdf"):
        return build_am_d02_model(model_path, np.asarray(tcp_offset, dtype=np.float64))
    return mujoco.MjModel.from_xml_path(model_path)


def _build_simulation_clean_sample_mask(
    *,
    q: np.ndarray,
    qd: np.ndarray,
    tau_constraint: np.ndarray,
    tau_friction: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    limited: np.ndarray,
    limit_margin: float,
    constraint_tolerance: float,
    active_joints: np.ndarray,
    min_motion_speed: float = 0.02,
) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    qd = np.asarray(qd, dtype=np.float64)
    tau_constraint = np.asarray(tau_constraint, dtype=np.float64)
    tau_friction = np.asarray(tau_friction, dtype=np.float64)
    active_joint_mask = np.asarray(active_joints, dtype=bool).reshape(-1)

    active_limited = np.asarray(limited, dtype=bool).reshape(-1) & active_joint_mask
    if np.any(active_limited):
        margin_to_limits = np.minimum(q - lower[None, :], upper[None, :] - q)
        margin_to_limits[:, ~active_limited] = np.inf
        away_from_limits = np.all(margin_to_limits > float(limit_margin), axis=1)
    else:
        away_from_limits = np.ones(q.shape[0], dtype=bool)

    constraint_is_clean = np.all(
        np.abs(tau_constraint[:, active_joint_mask]) < float(constraint_tolerance),
        axis=1,
    )
    finite = (
        np.all(np.isfinite(q[:, active_joint_mask]), axis=1)
        & np.all(np.isfinite(qd[:, active_joint_mask]), axis=1)
        & np.all(np.isfinite(tau_friction[:, active_joint_mask]), axis=1)
    )
    moving = np.any(np.abs(qd[:, active_joint_mask]) > float(min_motion_speed), axis=1)
    return away_from_limits & constraint_is_clean & finite & moving


class MujocoEnvironment:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.model = _load_model(str(config.robot.urdf_path), config.robot.tcp_offset)
        self.inverse_model = _load_model(str(config.robot.urdf_path), config.robot.tcp_offset)
        self.data = mujoco.MjData(self.model)
        self.inverse_data = mujoco.MjData(self.inverse_model)
        self.viewer = None

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
            config.simulation_friction.coulomb,
            config.simulation_friction.viscous,
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

    def _open_viewer(self) -> None:
        if not self.config.visualization.render:
            return
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self.viewer = mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=False,
            show_right_ui=False,
        )
        self.viewer.cam.azimuth = 135
        self.viewer.cam.elevation = -20
        self.viewer.cam.distance = 1.8
        self.viewer.cam.lookat[:] = [0.0, 0.0, 1.1]

    def _step_until(self, target_time: float, *, sim_time: float, wall_start: float, realtime: bool) -> float:
        while sim_time + 1e-12 < target_time:
            mujoco.mj_step(self.model, self.data)
            sim_time += self.model.opt.timestep
            if self.viewer is not None and self.viewer.is_running():
                self.viewer.sync()
            if realtime:
                elapsed = time.time() - wall_start
                if sim_time > elapsed:
                    time.sleep(sim_time - elapsed)
        return sim_time

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

    def bias_torque(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        self._assign_inverse_state(
            np.asarray(q, dtype=np.float64).reshape(-1),
            np.asarray(qd, dtype=np.float64).reshape(-1),
            np.zeros(self.config.joint_count, dtype=np.float64),
        )
        mujoco.mj_forward(self.inverse_model, self.inverse_data)
        return self.inverse_data.qfrc_bias[self.dof_addrs].copy()

    def _get_joint_limits(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        joint_limits = self.config.robot.joint_limits
        return resolve_joint_limit_arrays(joint_limits)

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

    def get_true_friction_parameters(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.model.dof_frictionloss[self.dof_addrs].copy(),
            self.model.dof_damping[self.dof_addrs].copy(),
        )

    def build_clean_sample_mask(self, batch: FrictionSampleBatch) -> np.ndarray:
        lower, upper, limited = self._get_joint_limits()
        return _build_simulation_clean_sample_mask(
            q=batch.q,
            qd=batch.qd,
            tau_constraint=batch.tau_constraint,
            tau_friction=batch.tau_friction,
            lower=lower,
            upper=upper,
            limited=limited,
            limit_margin=float(self.config.safety.joint_limit_margin),
            constraint_tolerance=0.35,
            active_joints=self.config.target_joint_mask,
        )

    def reset(self, q_init: np.ndarray) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        for qpos_addr, value in zip(self.qpos_addrs, np.asarray(q_init, dtype=np.float64).reshape(-1)):
            self.data.qpos[qpos_addr] = value
        mujoco.mj_forward(self.model, self.data)

    def _get_joint_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = self.data.qpos[self.qpos_addrs].copy()
        qd = self.data.qvel[self.dof_addrs].copy()
        qdd = self.data.qacc[self.dof_addrs].copy()
        return q, qd, qdd

    def _get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.xpos[self.ee_body_id].copy(), self.data.xquat[self.ee_body_id].copy()

    def _get_friction_components(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tau_passive = -self.data.qfrc_passive[self.dof_addrs].copy()
        tau_constraint = -self.data.qfrc_constraint[self.dof_addrs].copy()
        return tau_passive, tau_constraint, tau_passive + tau_constraint

    def _set_torque(self, tau: np.ndarray) -> None:
        tau = np.asarray(tau, dtype=np.float64).reshape(-1)
        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        self.data.qfrc_applied[self.dof_addrs] = tau

    def _run_reference_unlogged(
        self,
        reference: ReferenceTrajectory,
        controller: FrictionIdentificationController,
        safety: SafetyGuard,
        *,
        realtime: bool,
    ) -> None:
        sim_time = 0.0
        sample_dt = 1.0 / self.config.sampling.rate
        wall_start = time.time()
        for sample_idx in range(reference.q_cmd.shape[0]):
            q_curr, qd_curr, _ = self._get_joint_state()
            safety.assert_joint_limits(q_curr)
            _, _, tau = controller.compute_torque(
                q_cmd=reference.q_cmd[sample_idx],
                qd_cmd=reference.qd_cmd[sample_idx],
                qdd_cmd=reference.qdd_cmd[sample_idx],
                q_curr=q_curr,
                qd_curr=qd_curr,
            )
            self._set_torque(tau)
            sim_time = self._step_until(
                (sample_idx + 1) * sample_dt,
                sim_time=sim_time,
                wall_start=wall_start,
                realtime=realtime,
            )
            q_meas, _, _ = self._get_joint_state()
            safety.assert_joint_limits(q_meas)

    def run_reference_trajectory(
        self,
        reference: ReferenceTrajectory,
        controller: FrictionIdentificationController,
        safety: SafetyGuard,
        *,
        startup_reference: ReferenceTrajectory | None = None,
        realtime: bool = False,
    ) -> FrictionSampleBatch:
        num_samples = reference.q_cmd.shape[0]
        ee_pos_cmd, ee_quat_cmd = self.evaluate_end_effector_trajectory(reference.q_cmd)

        initial_q = startup_reference.q_cmd[0] if startup_reference is not None else reference.q_cmd[0]
        self.reset(initial_q)
        self._open_viewer()

        if startup_reference is not None:
            self._run_reference_unlogged(startup_reference, controller, safety, realtime=realtime)

        time_log = np.asarray(reference.time, dtype=np.float64).copy()
        q_log = np.zeros_like(reference.q_cmd)
        qd_log = np.zeros_like(reference.q_cmd)
        qdd_log = np.zeros_like(reference.q_cmd)
        tau_ctrl_log = np.zeros_like(reference.q_cmd)
        tau_passive_log = np.zeros_like(reference.q_cmd)
        tau_constraint_log = np.zeros_like(reference.q_cmd)
        tau_friction_log = np.zeros_like(reference.q_cmd)
        ee_pos_log = np.zeros_like(ee_pos_cmd)
        ee_quat_log = np.zeros_like(ee_quat_cmd)

        sim_time = 0.0
        wall_start = time.time()
        sample_dt = 1.0 / self.config.sampling.rate

        for sample_idx in range(num_samples):
            q_curr, qd_curr, _ = self._get_joint_state()
            safety.assert_joint_limits(q_curr)
            _, _, tau = controller.compute_torque(
                q_cmd=reference.q_cmd[sample_idx],
                qd_cmd=reference.qd_cmd[sample_idx],
                qdd_cmd=reference.qdd_cmd[sample_idx],
                q_curr=q_curr,
                qd_curr=qd_curr,
            )
            self._set_torque(tau)
            sim_time = self._step_until(
                (sample_idx + 1) * sample_dt,
                sim_time=sim_time,
                wall_start=wall_start,
                realtime=realtime,
            )

            q_meas, qd_meas, qdd_meas = self._get_joint_state()
            safety.assert_joint_limits(q_meas)
            ee_pos_meas, ee_quat_meas = self._get_ee_pose()
            tau_passive, tau_constraint, tau_friction = self._get_friction_components()

            q_log[sample_idx] = q_meas
            qd_log[sample_idx] = qd_meas
            qdd_log[sample_idx] = qdd_meas
            tau_ctrl_log[sample_idx] = tau
            tau_passive_log[sample_idx] = tau_passive
            tau_constraint_log[sample_idx] = tau_constraint
            tau_friction_log[sample_idx] = tau_friction
            ee_pos_log[sample_idx] = ee_pos_meas
            ee_quat_log[sample_idx] = ee_quat_meas

        self._set_torque(np.zeros(self.config.joint_count, dtype=np.float64))
        return FrictionSampleBatch(
            time=time_log,
            q=q_log,
            qd=qd_log,
            qdd=qdd_log,
            ee_pos=ee_pos_log,
            ee_quat=ee_quat_log,
            q_cmd=reference.q_cmd.copy(),
            qd_cmd=reference.qd_cmd.copy(),
            qdd_cmd=reference.qdd_cmd.copy(),
            ee_pos_cmd=ee_pos_cmd,
            ee_quat_cmd=ee_quat_cmd,
            tau_ctrl=tau_ctrl_log,
            tau_passive=tau_passive_log,
            tau_constraint=tau_constraint_log,
            tau_friction=tau_friction_log,
        )

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
