from __future__ import annotations

"""MuJoCo 采集层，复用共享核心逻辑完成激励生成、跟踪和日志记录。"""

import time
from dataclasses import dataclass
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np

from .am_d02_scene import build_am_d02_model
from .models import FrictionSampleBatch
from .shared_logic import (
    ReferenceTrajectory,
    build_quintic_point_to_point_trajectory,
    build_joint_excitation_plan,
    build_simulation_clean_sample_mask,
    find_joint_limit_violation,
    generate_segmented_excitation_trajectory,
    shape_limit_aware_torque_command,
    shrink_joint_limit_window,
)


class MujocoSafetyLimitExceeded(RuntimeError):
    """Raised when the MuJoCo state leaves the configured joint-safety envelope."""

    pass


SIM_EXCITATION_LIMIT_MARGIN_RAD = np.deg2rad(8.0)
SIM_EXCITATION_SOFT_MARGIN_RAD = np.deg2rad(18.0)
SIM_EXCITATION_RECENTER_TORQUE_RATIO = 0.16


@dataclass
class MujocoFrictionCollectionConfig:
    """High-level knobs for one MuJoCo excitation and logging run."""

    duration: float = 30.0
    sample_rate: float = 400.0
    timestep: float = 0.0005
    base_frequency: float = 0.12
    amplitude_scale: float = 0.18
    render: bool = False
    realtime: bool = False
    feedback_scale: float = 0.12


@dataclass(frozen=True)
class TrackingControlCommand:
    """统一描述一次轨迹跟踪输出的力矩及其限位整形结果。"""

    tau_command: np.ndarray
    tau_feedforward: np.ndarray
    tau_feedback: np.ndarray
    raw_tau_command: np.ndarray
    blocked_mask: np.ndarray
    scale_factors: np.ndarray


class MujocoFrictionCollector:
    """负责 MuJoCo 执行层，核心轨迹与筛样规则由共享模块提供。"""

    def __init__(
        self,
        model_path: str,
        joint_names: list[str],
        actuator_names: Optional[list[str]] = None,
        *,
        timestep: float = 0.0005,
        render: bool = False,
        home_qpos: Optional[np.ndarray] = None,
        end_effector_body: Optional[str] = None,
        tcp_offset: Optional[np.ndarray] = None,
        torque_limits: Optional[np.ndarray] = None,
        joint_limit_overrides: Optional[np.ndarray] = None,
        friction_loss: Optional[np.ndarray] = None,
        damping: Optional[np.ndarray] = None,
        inverse_friction_loss: Optional[np.ndarray] = None,
        inverse_damping: Optional[np.ndarray] = None,
    ):
        self.model = self._load_model(model_path=model_path, tcp_offset=tcp_offset)
        self.inverse_model = self._load_model(model_path=model_path, tcp_offset=tcp_offset)
        self.data = mujoco.MjData(self.model)
        self.inverse_data = mujoco.MjData(self.inverse_model)
        self.model.opt.timestep = timestep
        self.inverse_model.opt.timestep = timestep
        self.render = render
        self.viewer = None
        self.joint_names = list(joint_names)
        self.actuator_names = list(actuator_names or [])
        self.home_qpos = None if home_qpos is None else np.asarray(home_qpos, dtype=np.float64).copy()
        self.torque_limits = None if torque_limits is None else np.asarray(torque_limits, dtype=np.float64).reshape(-1)
        self.joint_ids = []
        self.dof_addrs = []
        self.qpos_addrs = []
        self.actuator_ids = []

        for name in self.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"找不到关节: {name}")
            self.joint_ids.append(joint_id)
            self.dof_addrs.append(self.model.jnt_dofadr[joint_id])
            self.qpos_addrs.append(self.model.jnt_qposadr[joint_id])

        for name in self.actuator_names:
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator_id < 0:
                raise ValueError(f"找不到执行器: {name}")
            self.actuator_ids.append(actuator_id)

        self.joint_ids = np.asarray(self.joint_ids, dtype=np.int32)
        self.dof_addrs = np.asarray(self.dof_addrs, dtype=np.int32)
        self.qpos_addrs = np.asarray(self.qpos_addrs, dtype=np.int32)
        self.actuator_ids = np.asarray(self.actuator_ids, dtype=np.int32)
        if self.torque_limits is None:
            self.torque_limits = np.full(len(self.joint_ids), np.inf, dtype=np.float64)
        elif self.torque_limits.size != len(self.joint_ids):
            raise ValueError("torque_limits size must match the number of joints.")
        self.ee_body_id = -1
        if end_effector_body is not None:
            self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, end_effector_body)
            if self.ee_body_id < 0:
                raise ValueError(f"找不到末端 body: {end_effector_body}")

        if joint_limit_overrides is not None:
            self._apply_joint_limit_overrides(joint_limit_overrides)
        self._apply_friction_overrides(friction_loss=friction_loss, damping=damping)
        self._apply_inverse_friction_overrides(
            friction_loss=friction_loss if inverse_friction_loss is None else inverse_friction_loss,
            damping=damping if inverse_damping is None else inverse_damping,
        )

        # Friction identification only needs dynamics from the articulated model.
        # Disable contact collisions to avoid large artificial constraint torques.
        self.model.geom_contype[:] = 0
        self.model.geom_conaffinity[:] = 0
        self.inverse_model.geom_contype[:] = 0
        self.inverse_model.geom_conaffinity[:] = 0

        self.kp = np.array([115.0, 100.0, 35.0, 40.0, 22.0, 20.0, 20.0], dtype=np.float64)
        self.kd = np.array([5.0, 5.0, 2.2, 2.0, 1.0, 1.0, 1.0], dtype=np.float64)

    @staticmethod
    def _load_model(model_path: str, tcp_offset: Optional[np.ndarray]) -> mujoco.MjModel:
        """Load either a URDF-backed robot or a ready-made MuJoCo XML scene."""

        if model_path.lower().endswith(".urdf"):
            if tcp_offset is None:
                raise ValueError("tcp_offset is required when loading a URDF model.")
            return build_am_d02_model(model_path, np.asarray(tcp_offset, dtype=np.float64))
        return mujoco.MjModel.from_xml_path(model_path)

    @staticmethod
    def _print_status(message: str) -> None:
        print(f"[INFO] {message}", flush=True)

    @staticmethod
    def _maybe_report_progress(
        *,
        label: str,
        completed: int,
        total: int,
        last_reported_percent: int,
        percent_step: int = 10,
        extra: str = "",
    ) -> int:
        """Emit coarse progress logs without flooding the console."""

        if total <= 0:
            return last_reported_percent

        if completed >= total:
            percent = 100
            if last_reported_percent >= 100:
                return last_reported_percent
        else:
            percent = int(100.0 * completed / total)
            next_threshold = max(percent_step, last_reported_percent + percent_step)
            if percent < next_threshold:
                return last_reported_percent

        suffix = f", {extra}" if extra else ""
        MujocoFrictionCollector._print_status(
            f"{label}: {percent:3d}% ({completed}/{total}{suffix})"
        )
        return percent

    def _apply_joint_limit_overrides(self, joint_limit_overrides: np.ndarray) -> None:
        """Overwrite joint limits in both forward and inverse MuJoCo models."""

        joint_limit_overrides = np.asarray(joint_limit_overrides, dtype=np.float64)
        if joint_limit_overrides.shape != (len(self.joint_ids), 2):
            raise ValueError("joint_limit_overrides must have shape [num_joints, 2].")
        for joint_id, limits in zip(self.joint_ids, joint_limit_overrides):
            self.model.jnt_limited[joint_id] = 1
            self.model.jnt_range[joint_id] = limits
            self.inverse_model.jnt_limited[joint_id] = 1
            self.inverse_model.jnt_range[joint_id] = limits

    def _apply_friction_overrides(
        self,
        *,
        friction_loss: Optional[np.ndarray],
        damping: Optional[np.ndarray],
    ) -> None:
        self._apply_friction_overrides_to_model(
            self.model,
            friction_loss=friction_loss,
            damping=damping,
        )

    def _apply_inverse_friction_overrides(
        self,
        *,
        friction_loss: Optional[np.ndarray],
        damping: Optional[np.ndarray],
    ) -> None:
        self._apply_friction_overrides_to_model(
            self.inverse_model,
            friction_loss=friction_loss,
            damping=damping,
        )

    def _apply_friction_overrides_to_model(
        self,
        model: mujoco.MjModel,
        *,
        friction_loss: Optional[np.ndarray],
        damping: Optional[np.ndarray],
    ) -> None:
        """Write friction-related parameters into the selected MuJoCo model."""

        if friction_loss is not None:
            friction_loss = np.asarray(friction_loss, dtype=np.float64).reshape(-1)
            if friction_loss.size != len(self.dof_addrs):
                raise ValueError("friction_loss size must match the number of joints.")
            model.dof_frictionloss[self.dof_addrs] = friction_loss
        if damping is not None:
            damping = np.asarray(damping, dtype=np.float64).reshape(-1)
            if damping.size != len(self.dof_addrs):
                raise ValueError("damping size must match the number of joints.")
            model.dof_damping[self.dof_addrs] = damping

    def _get_home_joint_positions(self) -> np.ndarray:
        """Resolve the preferred home pose from config, keyframe, or joint centers."""

        if self.home_qpos is not None:
            return self.home_qpos.copy()

        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            return np.array([self.model.key_qpos[key_id, addr] for addr in self.qpos_addrs], dtype=np.float64)

        ranges = self.model.jnt_range[self.joint_ids]
        centers = np.mean(ranges, axis=1)
        return centers.astype(np.float64)

    def get_true_friction_parameters(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the simulation model's ground-truth friction parameters."""

        coulomb = np.array([self.model.dof_frictionloss[addr] for addr in self.dof_addrs], dtype=np.float64)
        viscous = np.array([self.model.dof_damping[addr] for addr in self.dof_addrs], dtype=np.float64)
        return coulomb, viscous

    def get_controller_friction_parameters(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the inverse model parameters currently used for feedforward control."""

        coulomb = np.array(
            [self.inverse_model.dof_frictionloss[addr] for addr in self.dof_addrs],
            dtype=np.float64,
        )
        viscous = np.array(
            [self.inverse_model.dof_damping[addr] for addr in self.dof_addrs],
            dtype=np.float64,
        )
        return coulomb, viscous

    def set_inverse_friction_parameters(
        self,
        *,
        friction_loss: Optional[np.ndarray],
        damping: Optional[np.ndarray],
    ) -> None:
        """Update the inverse-dynamics controller's friction parameters."""

        self._apply_inverse_friction_overrides(
            friction_loss=friction_loss,
            damping=damping,
        )

    def _get_joint_limits(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return per-joint lower/upper bounds and a valid-limit mask."""

        lower = np.full(len(self.joint_ids), -np.inf, dtype=np.float64)
        upper = np.full(len(self.joint_ids), np.inf, dtype=np.float64)
        limited = np.zeros(len(self.joint_ids), dtype=bool)
        for idx, joint_id in enumerate(self.joint_ids):
            if not self.model.jnt_limited[joint_id]:
                continue
            lo, hi = self.model.jnt_range[joint_id]
            lower[idx] = min(lo, hi)
            upper[idx] = max(lo, hi)
            limited[idx] = True
        return lower, upper, limited

    def _get_excitation_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Return a conservative working window inside the hard joint limits."""

        lower, upper, limited = self._get_joint_limits()
        margin = SIM_EXCITATION_LIMIT_MARGIN_RAD
        if np.any(limited):
            finite_span = upper[limited] - lower[limited]
            if finite_span.size > 0:
                margin = min(float(margin), 0.49 * float(np.min(finite_span)))
        return shrink_joint_limit_window(
            lower,
            upper,
            limited,
            margin=margin,
            keep_midpoint_inside=True,
        )

    def _raise_if_joint_limits_exceeded(self, q: np.ndarray) -> None:
        """Abort the current run once the measured state crosses a hard limit."""

        violation_message = find_joint_limit_violation(
            q=q,
            joint_names=self.joint_names,
            joint_limits=np.column_stack(self._get_joint_limits()[:2]),
            margin=0.0,
        )
        if violation_message is not None:
            raise MujocoSafetyLimitExceeded(violation_message)

    def _build_joint_excitation_plan(
        self,
        *,
        amplitude_scale: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Choose a safe excitation center and amplitude for every joint."""

        lower, upper, limited = self._get_joint_limits()
        plan = build_joint_excitation_plan(
            home_qpos=self._get_home_joint_positions(),
            joint_limits=np.column_stack((lower, upper)),
            limited=limited,
            amplitude_scale=amplitude_scale,
        )
        return (
            plan.centers.copy(),
            plan.amplitudes.copy(),
            plan.safe_lower.copy(),
            plan.safe_upper.copy(),
            plan.limited.copy(),
        )

    def build_excitation_reference(
        self,
        *,
        duration: float,
        sample_rate: float,
        base_frequency: float,
        amplitude_scale: float,
    ) -> ReferenceTrajectory:
        """Build the shared excitation reference used by both simulation and real runs."""

        lower, upper, limited = self._get_joint_limits()
        return generate_segmented_excitation_trajectory(
            home_qpos=self._get_home_joint_positions(),
            joint_limits=np.column_stack((lower, upper)),
            limited=limited,
            duration=duration,
            sample_rate=sample_rate,
            base_frequency=base_frequency,
            amplitude_scale=amplitude_scale,
        )

    def build_transition_reference(
        self,
        *,
        start_q: np.ndarray,
        goal_q: np.ndarray,
        sample_rate: float,
        max_ee_speed: float,
        min_duration: float,
        settle_duration: float,
    ) -> tuple[ReferenceTrajectory, float]:
        """根据当前姿态在线规划过渡段，并按末端速度目标自动拉长时间。"""

        start_q = np.asarray(start_q, dtype=np.float64).reshape(-1)
        goal_q = np.asarray(goal_q, dtype=np.float64).reshape(-1)
        if start_q.shape != goal_q.shape:
            raise ValueError("start_q and goal_q must share the same shape.")

        max_ee_speed = max(float(max_ee_speed), 1e-6)
        min_duration = max(float(min_duration), 1e-3)
        settle_duration = max(float(settle_duration), 0.0)
        displacement = float(np.linalg.norm(goal_q - start_q))

        if displacement < 1e-6:
            duration = min_duration
        else:
            unit_reference = build_quintic_point_to_point_trajectory(
                start_q=start_q,
                goal_q=goal_q,
                duration=1.0,
                sample_rate=sample_rate,
                settle_duration=0.0,
            )
            ee_pos_unit, _ = self.evaluate_end_effector_trajectory(unit_reference.q_cmd)
            if ee_pos_unit.shape[0] >= 2:
                ee_speed_unit = np.linalg.norm(np.diff(ee_pos_unit, axis=0), axis=1) * float(sample_rate)
                peak_ee_speed_unit = float(np.max(ee_speed_unit))
            else:
                peak_ee_speed_unit = 0.0
            duration = max(min_duration, peak_ee_speed_unit / max_ee_speed)

        reference = build_quintic_point_to_point_trajectory(
            start_q=start_q,
            goal_q=goal_q,
            duration=duration,
            sample_rate=sample_rate,
            settle_duration=settle_duration,
        )
        return reference, float(duration)

    def generate_excitation_trajectory(
        self,
        *,
        duration: float,
        sample_rate: float,
        base_frequency: float,
        amplitude_scale: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate a segmented reference motion that excites joints one by one."""

        reference = self.build_excitation_reference(
            duration=duration,
            sample_rate=sample_rate,
            base_frequency=base_frequency,
            amplitude_scale=amplitude_scale,
        )
        return (
            reference.time.copy(),
            reference.q_cmd.copy(),
            reference.qd_cmd.copy(),
            reference.qdd_cmd.copy(),
        )

    def reset(self, q_init: np.ndarray) -> None:
        """Reset the forward simulation to a specific initial joint configuration."""

        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        for qpos_addr, value in zip(self.qpos_addrs, q_init):
            self.data.qpos[qpos_addr] = value
        mujoco.mj_forward(self.model, self.data)

    def _inverse_dynamics(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
        """Evaluate inverse dynamics without disturbing the current sim state."""

        qpos_backup = self.inverse_data.qpos.copy()
        qvel_backup = self.inverse_data.qvel.copy()
        qacc_backup = self.inverse_data.qacc.copy()

        for qpos_addr, dof_addr, q_i, qd_i, qdd_i in zip(self.qpos_addrs, self.dof_addrs, q, qd, qdd):
            self.inverse_data.qpos[qpos_addr] = q_i
            self.inverse_data.qvel[dof_addr] = qd_i
            self.inverse_data.qacc[dof_addr] = qdd_i

        mujoco.mj_inverse(self.inverse_model, self.inverse_data)
        tau = np.array(
            [self.inverse_data.qfrc_inverse[dof_addr] for dof_addr in self.dof_addrs],
            dtype=np.float64,
        )

        self.inverse_data.qpos[:] = qpos_backup
        self.inverse_data.qvel[:] = qvel_backup
        self.inverse_data.qacc[:] = qacc_backup
        mujoco.mj_forward(self.inverse_model, self.inverse_data)
        return tau

    def compute_bias_torque(
        self,
        *,
        q_curr: np.ndarray,
        qd_curr: np.ndarray,
    ) -> np.ndarray:
        """Return gravity plus velocity-dependent bias torque from the friction-free inverse model."""

        qpos_backup = self.inverse_data.qpos.copy()
        qvel_backup = self.inverse_data.qvel.copy()
        qacc_backup = self.inverse_data.qacc.copy()

        for qpos_addr, dof_addr, q_i, qd_i in zip(self.qpos_addrs, self.dof_addrs, q_curr, qd_curr):
            self.inverse_data.qpos[qpos_addr] = q_i
            self.inverse_data.qvel[dof_addr] = qd_i
            self.inverse_data.qacc[dof_addr] = 0.0

        mujoco.mj_forward(self.inverse_model, self.inverse_data)
        tau = np.array(
            [self.inverse_data.qfrc_bias[dof_addr] for dof_addr in self.dof_addrs],
            dtype=np.float64,
        )

        self.inverse_data.qpos[:] = qpos_backup
        self.inverse_data.qvel[:] = qvel_backup
        self.inverse_data.qacc[:] = qacc_backup
        mujoco.mj_forward(self.inverse_model, self.inverse_data)
        return tau

    def _get_joint_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read measured joint position, velocity, and acceleration from MuJoCo."""

        q = np.array([self.data.qpos[addr] for addr in self.qpos_addrs], dtype=np.float64)
        qd = np.array([self.data.qvel[addr] for addr in self.dof_addrs], dtype=np.float64)
        qdd = np.array([self.data.qacc[addr] for addr in self.dof_addrs], dtype=np.float64)
        return q, qd, qdd

    def _get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the current end-effector pose if configured."""

        if self.ee_body_id < 0:
            return np.zeros(3, dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return (
            self.data.xpos[self.ee_body_id].copy(),
            self.data.xquat[self.ee_body_id].copy(),
        )

    def _get_friction_components(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split passive and constraint torques for later sample filtering."""

        tau_passive = -np.array([self.data.qfrc_passive[addr] for addr in self.dof_addrs], dtype=np.float64)
        tau_constraint = -np.array([self.data.qfrc_constraint[addr] for addr in self.dof_addrs], dtype=np.float64)
        return tau_passive, tau_constraint, tau_passive + tau_constraint

    def _set_torque(self, tau: np.ndarray) -> None:
        """Apply joint torque through actuators or directly through qfrc_applied."""

        if self.actuator_ids.size > 0:
            self.data.ctrl[:] = 0.0
            for actuator_id, torque in zip(self.actuator_ids, tau):
                self.data.ctrl[actuator_id] = float(torque)
            return

        self.data.qfrc_applied[:] = 0.0
        self.data.qfrc_applied[self.dof_addrs] = tau

    def compute_tracking_torque(
        self,
        *,
        q_cmd: np.ndarray,
        qd_cmd: np.ndarray,
        qdd_cmd: np.ndarray,
        q_curr: np.ndarray,
        qd_curr: np.ndarray,
        feedback_scale: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Combine inverse-dynamics feedforward and PD feedback torque."""

        tau_ff = self._inverse_dynamics(q_cmd, qd_cmd, qdd_cmd)
        tau_fb = self.kp * (q_cmd - q_curr) + self.kd * (qd_cmd - qd_curr)
        tau_ctrl = tau_ff + feedback_scale * tau_fb
        return tau_ff, tau_fb, tau_ctrl

    def compute_safe_tracking_torque(
        self,
        *,
        q_cmd: np.ndarray,
        qd_cmd: np.ndarray,
        qdd_cmd: np.ndarray,
        q_curr: np.ndarray,
        qd_curr: np.ndarray,
        feedback_scale: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply torque clipping and joint-limit-aware shaping to the tracking controller."""

        command = self.compute_safe_tracking_command(
            q_cmd=q_cmd,
            qd_cmd=qd_cmd,
            qdd_cmd=qdd_cmd,
            q_curr=q_curr,
            qd_curr=qd_curr,
            feedback_scale=feedback_scale,
        )
        return (
            command.tau_feedforward,
            command.tau_feedback,
            command.tau_command,
        )

    def compute_safe_tracking_command(
        self,
        *,
        q_cmd: np.ndarray,
        qd_cmd: np.ndarray,
        qdd_cmd: np.ndarray,
        q_curr: np.ndarray,
        qd_curr: np.ndarray,
        feedback_scale: float,
        lower: Optional[np.ndarray] = None,
        upper: Optional[np.ndarray] = None,
        soft_margin: float = SIM_EXCITATION_SOFT_MARGIN_RAD,
        recenter_torque_ratio: float = SIM_EXCITATION_RECENTER_TORQUE_RATIO,
    ) -> TrackingControlCommand:
        """共享仿真/真机的轨迹跟踪控制核心。"""

        tau_ff, tau_fb, tau_ctrl = self.compute_tracking_torque(
            q_cmd=q_cmd,
            qd_cmd=qd_cmd,
            qdd_cmd=qdd_cmd,
            q_curr=q_curr,
            qd_curr=qd_curr,
            feedback_scale=feedback_scale,
        )
        raw_tau_command = np.clip(tau_ctrl, -self.torque_limits, self.torque_limits)
        if lower is None or upper is None:
            lower, upper = self._get_excitation_limits()
        else:
            lower = np.asarray(lower, dtype=np.float64).reshape(-1)
            upper = np.asarray(upper, dtype=np.float64).reshape(-1)

        tau_command, blocked_mask, scale_factors = shape_limit_aware_torque_command(
            q=q_curr,
            torque_command=raw_tau_command,
            torque_limits=self.torque_limits,
            lower=lower,
            upper=upper,
            soft_margin=soft_margin,
            recenter_torque_ratio=recenter_torque_ratio,
        )
        return TrackingControlCommand(
            tau_command=tau_command,
            tau_feedforward=tau_ff,
            tau_feedback=tau_fb,
            raw_tau_command=raw_tau_command,
            blocked_mask=blocked_mask,
            scale_factors=scale_factors,
        )

    def evaluate_end_effector_trajectory(self, q_traj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Replay a joint trajectory to obtain its corresponding TCP path."""

        num_samples = q_traj.shape[0]
        ee_pos = np.zeros((num_samples, 3), dtype=np.float64)
        ee_quat = np.zeros((num_samples, 4), dtype=np.float64)

        qpos_backup = self.data.qpos.copy()
        qvel_backup = self.data.qvel.copy()
        qacc_backup = self.data.qacc.copy()

        try:
            self._print_status(f"Evaluating end-effector reference trajectory for {num_samples} samples")
            last_reported_percent = 0
            self.data.qvel[:] = 0.0
            self.data.qacc[:] = 0.0
            for sample_idx, q_sample in enumerate(q_traj):
                for qpos_addr, q_i in zip(self.qpos_addrs, q_sample):
                    self.data.qpos[qpos_addr] = q_i
                mujoco.mj_forward(self.model, self.data)
                ee_pos[sample_idx], ee_quat[sample_idx] = self._get_ee_pose()
                last_reported_percent = self._maybe_report_progress(
                    label="EE reference evaluation",
                    completed=sample_idx + 1,
                    total=num_samples,
                    last_reported_percent=last_reported_percent,
                    percent_step=20,
                )
        finally:
            self.data.qpos[:] = qpos_backup
            self.data.qvel[:] = qvel_backup
            self.data.qacc[:] = qacc_backup
            mujoco.mj_forward(self.model, self.data)

        return ee_pos, ee_quat

    def collect(
        self,
        q_cmd: np.ndarray,
        qd_cmd: np.ndarray,
        qdd_cmd: np.ndarray,
        ee_pos_cmd: np.ndarray,
        ee_quat_cmd: np.ndarray,
        *,
        sample_rate: float,
        feedback_scale: float,
        realtime: bool,
    ) -> FrictionSampleBatch:
        """Track a reference trajectory and log the resulting simulation signals."""

        num_samples = q_cmd.shape[0]
        self.reset(q_cmd[0])

        if self.render:
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

        time_log = np.zeros(num_samples, dtype=np.float64)
        q_log = np.zeros_like(q_cmd)
        qd_log = np.zeros_like(q_cmd)
        qdd_log = np.zeros_like(q_cmd)
        ee_pos_log = np.zeros_like(ee_pos_cmd)
        ee_quat_log = np.zeros_like(ee_quat_cmd)
        tau_ctrl_log = np.zeros_like(q_cmd)
        tau_passive_log = np.zeros_like(q_cmd)
        tau_constraint_log = np.zeros_like(q_cmd)
        tau_friction_log = np.zeros_like(q_cmd)

        sim_time = 0.0
        sample_dt = 1.0 / sample_rate
        wall_start = time.time()
        self._print_status(
            "Collecting MuJoCo samples "
            f"(duration={num_samples / sample_rate:.1f}s, sample_rate={sample_rate:.1f}Hz, "
            f"samples={num_samples})"
        )
        last_reported_percent = 0

        for sample_idx in range(num_samples):
            q_curr, qd_curr, _ = self._get_joint_state()
            self._raise_if_joint_limits_exceeded(q_curr)
            command = self.compute_safe_tracking_command(
                q_cmd=q_cmd[sample_idx],
                qd_cmd=qd_cmd[sample_idx],
                qdd_cmd=qdd_cmd[sample_idx],
                q_curr=q_curr,
                qd_curr=qd_curr,
                feedback_scale=feedback_scale,
            )
            self._set_torque(command.tau_command)

            next_sample_time = (sample_idx + 1) * sample_dt
            # Step the simulator until the next logging instant is reached.
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
            try:
                self._raise_if_joint_limits_exceeded(q_meas)
            except MujocoSafetyLimitExceeded:
                self._set_torque(np.zeros(len(self.dof_addrs), dtype=np.float64))
                raise
            ee_pos_meas, ee_quat_meas = self._get_ee_pose()
            tau_passive, tau_constraint, tau_friction = self._get_friction_components()

            time_log[sample_idx] = sim_time
            q_log[sample_idx] = q_meas
            qd_log[sample_idx] = qd_meas
            qdd_log[sample_idx] = qdd_meas
            ee_pos_log[sample_idx] = ee_pos_meas
            ee_quat_log[sample_idx] = ee_quat_meas
            tau_ctrl_log[sample_idx] = command.tau_command
            tau_passive_log[sample_idx] = tau_passive
            tau_constraint_log[sample_idx] = tau_constraint
            tau_friction_log[sample_idx] = tau_friction
            last_reported_percent = self._maybe_report_progress(
                label="MuJoCo collection",
                completed=sample_idx + 1,
                total=num_samples,
                last_reported_percent=last_reported_percent,
                percent_step=5,
                extra=f"sim_time={sim_time:.1f}s/{num_samples / sample_rate:.1f}s",
            )

        wall_elapsed = time.time() - wall_start
        self._print_status(
            f"MuJoCo collection complete in {wall_elapsed:.1f}s wall time"
        )

        return FrictionSampleBatch(
            time=time_log,
            q=q_log,
            qd=qd_log,
            qdd=qdd_log,
            ee_pos=ee_pos_log,
            ee_quat=ee_quat_log,
            q_cmd=q_cmd,
            qd_cmd=qd_cmd,
            qdd_cmd=qdd_cmd,
            ee_pos_cmd=ee_pos_cmd,
            ee_quat_cmd=ee_quat_cmd,
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
        """Filter out samples near joint limits, constraints, or near-zero motion."""

        lower, upper, limited = self._get_joint_limits()
        return build_simulation_clean_sample_mask(
            q=batch.q,
            qd=batch.qd,
            tau_constraint=batch.tau_constraint,
            tau_friction=batch.tau_friction,
            lower=lower,
            upper=upper,
            limited=limited,
            limit_margin=limit_margin,
            constraint_tolerance=constraint_tolerance,
        )

    def run_openarm_collection(self, config: MujocoFrictionCollectionConfig) -> FrictionSampleBatch:
        """Generate a default excitation trajectory and immediately collect it."""

        self._print_status(
            "Generating segmented joint excitation trajectory "
            f"(duration={config.duration:.1f}s, base_frequency={config.base_frequency:.3f}Hz, "
            f"amplitude_scale={config.amplitude_scale:.3f})"
        )
        t, q_cmd, qd_cmd, qdd_cmd = self.generate_excitation_trajectory(
            duration=config.duration,
            sample_rate=config.sample_rate,
            base_frequency=config.base_frequency,
            amplitude_scale=config.amplitude_scale,
        )
        self._print_status(f"Excitation trajectory ready with {t.shape[0]} samples")
        return self.run_reference_trajectory(
            q_cmd=q_cmd,
            qd_cmd=qd_cmd,
            qdd_cmd=qdd_cmd,
            sample_rate=config.sample_rate,
            feedback_scale=config.feedback_scale,
            realtime=config.realtime,
            time_reference=t,
        )

    def run_reference_trajectory(
        self,
        *,
        q_cmd: np.ndarray,
        qd_cmd: np.ndarray,
        qdd_cmd: np.ndarray,
        sample_rate: float,
        feedback_scale: float,
        realtime: bool,
        time_reference: Optional[np.ndarray] = None,
        ee_pos_cmd: Optional[np.ndarray] = None,
        ee_quat_cmd: Optional[np.ndarray] = None,
    ) -> FrictionSampleBatch:
        """Track a provided reference trajectory and align its logged timestamps."""

        if ee_pos_cmd is None or ee_quat_cmd is None:
            ee_pos_cmd, ee_quat_cmd = self.evaluate_end_effector_trajectory(q_cmd)

        batch = self.collect(
            q_cmd=q_cmd,
            qd_cmd=qd_cmd,
            qdd_cmd=qdd_cmd,
            ee_pos_cmd=ee_pos_cmd,
            ee_quat_cmd=ee_quat_cmd,
            sample_rate=sample_rate,
            feedback_scale=feedback_scale,
            realtime=realtime,
        )
        if time_reference is None:
            batch.time[:] = np.arange(batch.time.shape[0], dtype=np.float64) / float(sample_rate)
        else:
            time_reference = np.asarray(time_reference, dtype=np.float64).reshape(-1)
            if time_reference.shape[0] != batch.time.shape[0]:
                raise ValueError("time_reference length must match the number of trajectory samples.")
            batch.time[:] = time_reference
        return batch

    def close(self) -> None:
        """Close the optional passive MuJoCo viewer."""

        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
