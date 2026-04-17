from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from friction_identification_core.config import Config
from friction_identification_core.core.controller import FrictionIdentificationController
from friction_identification_core.core.estimator import fit_multijoint_friction
from friction_identification_core.core.models import JointFrictionParameters
from friction_identification_core.core.safety import SafetyGuard
from friction_identification_core.core.trajectory import (
    ReferenceTrajectory,
    build_startup_pose,
    sample_reference_trajectory,
)
from friction_identification_core.hardware.serial_protocol import (
    RECV_FRAME_SIZE,
    SEND_FRAME_SIZE,
    SerialFrameReader,
    TorqueCommandFramePacker,
)
from friction_identification_core.simulation.mujoco_env import MujocoEnvironment
from friction_identification_core.utils.logging import ensure_directory, log_info, write_json
from friction_identification_core.utils.mujoco import build_am_d02_model
from friction_identification_core.utils.visualization import build_hardware_reporter, build_pose_estimator


@dataclass
class LiveReferenceState:
    excitation_reference: ReferenceTrajectory
    sample_rate: float
    startup_reference: ReferenceTrajectory | None = None
    startup_duration: float = 0.0
    reference_start_time: float | None = None
    trajectory_elapsed_s: float = 0.0
    last_elapsed_s: float | None = None

    def initialize(self, env: MujocoEnvironment, q_start: np.ndarray, startup_target: np.ndarray, elapsed_s: float) -> None:
        if self.reference_start_time is not None:
            return
        self.reference_start_time = float(elapsed_s)
        self.last_elapsed_s = float(elapsed_s)
        self.trajectory_elapsed_s = 0.0
        self.startup_reference = env.build_startup_reference(q_start, startup_target)
        if self.startup_reference is not None and self.startup_reference.time.size > 0:
            self.startup_duration = float(self.startup_reference.time[-1] + 1.0 / self.sample_rate)

    def sample(self, elapsed_s: float, *, max_step_s: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.reference_start_time is None:
            raise RuntimeError("Live reference has not been initialized.")
        if self.last_elapsed_s is None:
            self.last_elapsed_s = float(elapsed_s)

        delta_s = max(float(elapsed_s) - self.last_elapsed_s, 0.0)
        self.last_elapsed_s = float(elapsed_s)
        if max_step_s is not None and max_step_s > 0.0:
            delta_s = min(delta_s, float(max_step_s))
        self.trajectory_elapsed_s += delta_s

        local_elapsed = self.trajectory_elapsed_s
        if self.startup_reference is not None and local_elapsed < self.startup_duration:
            return sample_reference_trajectory(self.startup_reference, local_elapsed, wrap=False)
        excitation_elapsed = max(local_elapsed - self.startup_duration, 0.0)
        return sample_reference_trajectory(self.excitation_reference, excitation_elapsed, wrap=False)

    def is_complete(self) -> bool:
        if self.reference_start_time is None:
            return False
        total_duration = self.startup_duration + float(self.excitation_reference.time[-1] + 1.0 / self.sample_rate)
        return self.trajectory_elapsed_s >= total_duration


class RigidBodyDynamics:
    """MuJoCo rigid-body inverse dynamics without joint friction terms."""

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
        self.model.dof_frictionloss[self.dof_addrs] = 0.0
        self.model.dof_damping[self.dof_addrs] = 0.0

    def _assign_state(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> None:
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        for idx in range(self.qpos_addrs.size):
            self.data.qpos[self.qpos_addrs[idx]] = q[idx]
            self.data.qvel[self.dof_addrs[idx]] = qd[idx]
            self.data.qacc[self.dof_addrs[idx]] = qdd[idx]

    def inverse_dynamics(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        qd = np.asarray(qd, dtype=np.float64).reshape(-1)
        qdd = np.asarray(qdd, dtype=np.float64).reshape(-1)
        self._assign_state(q, qd, qdd)
        self._mujoco.mj_inverse(self.model, self.data)
        return self.data.qfrc_inverse[self.dof_addrs].copy()

    def batch_inverse_dynamics(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        qd = np.asarray(qd, dtype=np.float64)
        qdd = np.asarray(qdd, dtype=np.float64)
        if q.shape != qd.shape or q.shape != qdd.shape or q.ndim != 2:
            raise ValueError("q/qd/qdd must be same-shape 2D arrays.")

        tau = np.zeros_like(q, dtype=np.float64)
        for sample_idx in range(q.shape[0]):
            tau[sample_idx] = self.inverse_dynamics(q[sample_idx], qd[sample_idx], qdd[sample_idx])
        return tau


def _load_existing_summary(summary_path: Path, joint_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not summary_path.exists():
        return (
            np.zeros(joint_count, dtype=np.float64),
            np.zeros(joint_count, dtype=np.float64),
            np.zeros(joint_count, dtype=np.float64),
        )

    with open(summary_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    def _read_vector(key: str) -> np.ndarray:
        values = np.asarray(payload.get(key, [0.0] * joint_count), dtype=np.float64).reshape(-1)
        if values.size != joint_count:
            return np.zeros(joint_count, dtype=np.float64)
        return values

    return _read_vector("estimated_coulomb"), _read_vector("estimated_viscous"), _read_vector("estimated_offset")


def _load_compensation_parameters(summary_path: Path, joint_count: int) -> list[JointFrictionParameters]:
    coulomb, viscous, offset = _load_existing_summary(summary_path, joint_count)
    return [
        JointFrictionParameters(
            coulomb=float(coulomb[idx]),
            viscous=float(viscous[idx]),
            offset=float(offset[idx]),
            velocity_scale=0.03,
        )
        for idx in range(joint_count)
    ]


def _predict_compensation_torque(
    velocity: np.ndarray,
    parameters: list[JointFrictionParameters],
    torque_limits: np.ndarray,
) -> np.ndarray:
    velocity = np.asarray(velocity, dtype=np.float64).reshape(-1)
    torque_limits = np.asarray(torque_limits, dtype=np.float64).reshape(-1)
    torque = np.zeros_like(velocity)
    for idx, param in enumerate(parameters):
        scale = max(float(param.velocity_scale), 1e-6)
        torque[idx] = param.coulomb * np.tanh(velocity[idx] / scale) + param.viscous * velocity[idx] + param.offset
    return np.clip(torque, -torque_limits, torque_limits)


def _build_residual_clean_sample_mask(
    *,
    q: np.ndarray,
    qd: np.ndarray,
    tau_residual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    torque_limits: np.ndarray,
    active_joints: np.ndarray,
    min_motion_speed: float,
    torque_limit_scale: float = 1.5,
) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    qd = np.asarray(qd, dtype=np.float64)
    tau_residual = np.asarray(tau_residual, dtype=np.float64)
    active_joint_mask = np.asarray(active_joints, dtype=bool).reshape(-1)

    within_window = np.all(
        ((q >= lower[None, :]) & (q <= upper[None, :])) | (~active_joint_mask)[None, :],
        axis=1,
    )
    finite = (
        np.all(np.isfinite(q[:, active_joint_mask]), axis=1)
        & np.all(np.isfinite(qd[:, active_joint_mask]), axis=1)
        & np.all(np.isfinite(tau_residual[:, active_joint_mask]), axis=1)
    )
    moving = np.any(np.abs(qd[:, active_joint_mask]) >= float(min_motion_speed), axis=1)
    residual_reasonable = np.all(
        np.abs(tau_residual[:, active_joint_mask])
        <= (np.asarray(torque_limits, dtype=np.float64)[active_joint_mask] * float(torque_limit_scale)),
        axis=1,
    )
    return finite & within_window & moving & residual_reasonable


def _save_capture(
    config: Config,
    *,
    mode: str,
    termination_reason: str,
    time_log: list[float],
    q_log: list[np.ndarray],
    qd_log: list[np.ndarray],
    q_cmd_log: list[np.ndarray],
    qd_cmd_log: list[np.ndarray],
    tau_measured_log: list[np.ndarray],
    tau_command_log: list[np.ndarray],
    tau_feedforward_log: list[np.ndarray],
    tau_feedback_log: list[np.ndarray],
    mos_temp_log: list[np.ndarray],
    coil_temp_log: list[np.ndarray],
    ee_pos_log: list[np.ndarray],
    ee_quat_log: list[np.ndarray],
    uart_cycle_hz_log: list[float],
    uart_latency_ms_log: list[float],
    uart_transfer_kbps_log: list[float],
) -> tuple[Path, Path]:
    results_dir = ensure_directory(config.results_dir)
    prefix = f"{config.output.hardware_capture_prefix}_{mode}_joint_{config.target_joint + 1}"
    npz_path = results_dir / f"{prefix}.npz"
    json_path = results_dir / f"{prefix}.json"

    q = np.asarray(q_log, dtype=np.float64).reshape(-1, config.joint_count)
    qd = np.asarray(qd_log, dtype=np.float64).reshape(-1, config.joint_count)
    q_cmd = np.asarray(q_cmd_log, dtype=np.float64).reshape(-1, config.joint_count)
    qd_cmd = np.asarray(qd_cmd_log, dtype=np.float64).reshape(-1, config.joint_count)
    tau_measured = np.asarray(tau_measured_log, dtype=np.float64).reshape(-1, config.joint_count)
    tau_command = np.asarray(tau_command_log, dtype=np.float64).reshape(-1, config.joint_count)
    tau_feedforward = np.asarray(tau_feedforward_log, dtype=np.float64).reshape(-1, config.joint_count)
    tau_feedback = np.asarray(tau_feedback_log, dtype=np.float64).reshape(-1, config.joint_count)
    mos_temp = np.asarray(mos_temp_log, dtype=np.float64).reshape(-1, config.joint_count)
    coil_temp = np.asarray(coil_temp_log, dtype=np.float64).reshape(-1, config.joint_count)

    if ee_pos_log:
        ee_pos = np.asarray(ee_pos_log, dtype=np.float64).reshape(-1, 3)
        ee_quat = np.asarray(ee_quat_log, dtype=np.float64).reshape(-1, 4)
    else:
        ee_pos = np.zeros((q.shape[0], 3), dtype=np.float64)
        ee_quat = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64), (q.shape[0], 1))

    np.savez(
        npz_path,
        time=np.asarray(time_log, dtype=np.float64),
        q=q,
        qd=qd,
        q_cmd=q_cmd,
        qd_cmd=qd_cmd,
        tau_measured=tau_measured,
        tau_command=tau_command,
        tau_feedforward=tau_feedforward,
        tau_feedback=tau_feedback,
        mos_temperature=mos_temp,
        coil_temperature=coil_temp,
        ee_pos=ee_pos,
        ee_quat=ee_quat,
        uart_cycle_hz=np.asarray(uart_cycle_hz_log, dtype=np.float64),
        uart_latency_ms=np.asarray(uart_latency_ms_log, dtype=np.float64),
        uart_transfer_kbps=np.asarray(uart_transfer_kbps_log, dtype=np.float64),
    )

    summary = {
        "mode": mode,
        "config_path": str(config.config_path),
        "target_joint": int(config.target_joint),
        "target_joint_name": config.target_joint_name,
        "sample_count": int(len(time_log)),
        "duration_s": float(time_log[-1]) if time_log else 0.0,
        "termination_reason": termination_reason,
        "mean_uart_cycle_hz": float(np.mean(uart_cycle_hz_log)) if uart_cycle_hz_log else 0.0,
        "mean_uart_latency_ms": float(np.mean(uart_latency_ms_log)) if uart_latency_ms_log else 0.0,
        "mean_uart_transfer_kbps": float(np.mean(uart_transfer_kbps_log)) if uart_transfer_kbps_log else 0.0,
    }
    write_json(json_path, summary)
    return npz_path, json_path


def _identify_from_capture(
    config: Config,
    *,
    time_s: np.ndarray,
    q: np.ndarray,
    qd: np.ndarray,
    tau_measured: np.ndarray,
) -> tuple[Path, Path] | None:
    if time_s.size < 32:
        log_info("真机样本过少，跳过实际数据辨识。")
        return None

    results_dir = ensure_directory(config.results_dir)
    prefix = f"{config.output.hardware_ident_prefix}_joint_{config.target_joint + 1}"
    npz_path = results_dir / f"{prefix}.npz"
    json_path = results_dir / f"{prefix}.json"

    gradient_order = 2 if time_s.size >= 3 else 1
    qdd = np.gradient(qd, time_s, axis=0, edge_order=gradient_order)
    dynamics = RigidBodyDynamics(
        model_path=str(config.robot.urdf_path),
        joint_names=list(config.robot.joint_names),
        tcp_offset=config.robot.tcp_offset,
    )
    tau_rigid = dynamics.batch_inverse_dynamics(q, qd, qdd)
    tau_friction = tau_measured - tau_rigid

    safety = SafetyGuard(config, active_joint_mask=config.target_joint_mask)
    lower, upper = safety.safe_joint_window()
    clean_mask = _build_residual_clean_sample_mask(
        q=q,
        qd=qd,
        tau_residual=tau_friction,
        lower=lower,
        upper=upper,
        torque_limits=config.robot.torque_limits,
        active_joints=config.target_joint_mask,
        min_motion_speed=max(float(config.fitting.min_velocity_threshold), 0.01),
    )

    retained = int(np.count_nonzero(clean_mask))
    if retained < 16:
        log_info("筛样后真机样本不足，跳过实际数据辨识。")
        return None

    qd_clean = qd[clean_mask]
    tau_clean = tau_friction[clean_mask]
    validation_mask = np.zeros(qd_clean.shape[0], dtype=bool)
    validation_mask[::5] = True
    validation_mask[: min(20, validation_mask.size)] = False
    if validation_mask.size > 0 and np.all(validation_mask):
        validation_mask[-1] = False

    result = fit_multijoint_friction(
        velocity=qd_clean[:, config.target_joint_mask],
        torque=tau_clean[:, config.target_joint_mask],
        joint_names=[config.target_joint_name],
        validation_mask=validation_mask,
        velocity_scale=config.fitting.velocity_scale,
        regularization=config.fitting.regularization,
        max_iterations=config.fitting.max_iterations,
        huber_delta=config.fitting.huber_delta,
        min_velocity_threshold=config.fitting.min_velocity_threshold,
    )

    estimated_coulomb, estimated_viscous, estimated_offset = _load_existing_summary(
        config.summary_path,
        config.joint_count,
    )
    estimated_coulomb[config.target_joint] = result.parameters[0].coulomb
    estimated_viscous[config.target_joint] = result.parameters[0].viscous
    estimated_offset[config.target_joint] = result.parameters[0].offset

    np.savez(
        npz_path,
        time=time_s,
        q=q,
        qd=qd,
        qdd=qdd,
        tau_measured=tau_measured,
        tau_rigid=tau_rigid,
        tau_friction=tau_friction,
        clean_mask=clean_mask,
    )

    summary = {
        "mode": "hardware_identification",
        "config_path": str(config.config_path),
        "target_joint": int(config.target_joint),
        "target_joint_name": config.target_joint_name,
        "retained_samples": retained,
        "estimated_coulomb": estimated_coulomb.tolist(),
        "estimated_viscous": estimated_viscous.tolist(),
        "estimated_offset": estimated_offset.tolist(),
        "updated_joint": {
            "index": int(config.target_joint),
            "name": config.target_joint_name,
            "coulomb": float(result.parameters[0].coulomb),
            "viscous": float(result.parameters[0].viscous),
            "offset": float(result.parameters[0].offset),
            "validation_rmse": float(result.validation_rmse[0]),
            "validation_r2": float(result.validation_r2[0]),
        },
        "velocity_scale": float(config.fitting.velocity_scale),
    }
    write_json(json_path, summary)
    write_json(config.summary_path, summary)
    log_info(f"真机辨识结果已保存: {json_path}")
    return npz_path, json_path


def run_hardware(config: Config, *, mode: str) -> tuple[Path, Path]:
    if mode not in {"collect", "compensate", "full_feedforward"}:
        raise ValueError("mode must be 'collect', 'compensate', or 'full_feedforward'.")

    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("缺少 pyserial，请先安装 requirements.txt 中的依赖。") from exc

    env = MujocoEnvironment(config)
    safety = SafetyGuard(config, active_joint_mask=config.target_joint_mask)
    controller = FrictionIdentificationController(config, env, safety=safety)
    reporter = build_hardware_reporter(config)
    pose_estimator = build_pose_estimator(config)
    parameters = _load_compensation_parameters(config.summary_path, config.joint_count) if mode in ("compensate", "full_feedforward") else None

    reference_state = None
    reference_max_step_s = None
    if mode in ("collect", "full_feedforward"):
        excitation_reference = env.build_excitation_reference()
        startup_target = build_startup_pose(config, excitation_reference)
        reference_state = LiveReferenceState(
            excitation_reference=excitation_reference,
            sample_rate=config.sampling.rate,
        )
        reference_max_step_s = (
            max(float(config.sampling.hardware_reference_step_factor), 1.0) / float(config.sampling.rate)
        )

    frame_reader = SerialFrameReader()
    frame_packer = TorqueCommandFramePacker()
    zero_frame = frame_packer.pack(np.zeros(config.joint_count, dtype=np.float32))

    q = np.zeros(config.joint_count, dtype=np.float64)
    qd = np.zeros(config.joint_count, dtype=np.float64)
    tau_measured = np.zeros(config.joint_count, dtype=np.float64)
    mos_temp = np.zeros(config.joint_count, dtype=np.float64)
    coil_temp = np.zeros(config.joint_count, dtype=np.float64)
    target_joint_idx = int(config.target_joint)
    last_feedback_time = np.full(config.joint_count, np.nan, dtype=np.float64)

    time_log: list[float] = []
    q_log: list[np.ndarray] = []
    qd_log: list[np.ndarray] = []
    q_cmd_log: list[np.ndarray] = []
    qd_cmd_log: list[np.ndarray] = []
    tau_measured_log: list[np.ndarray] = []
    tau_command_log: list[np.ndarray] = []
    tau_feedforward_log: list[np.ndarray] = []
    tau_feedback_log: list[np.ndarray] = []
    mos_temp_log: list[np.ndarray] = []
    coil_temp_log: list[np.ndarray] = []
    ee_pos_log: list[np.ndarray] = []
    ee_quat_log: list[np.ndarray] = []
    uart_cycle_hz_log: list[float] = []
    uart_latency_ms_log: list[float] = []
    uart_transfer_kbps_log: list[float] = []

    start_time = None
    last_cycle_end = None
    step_index = 0
    termination_reason = "completed"
    bytes_per_cycle = RECV_FRAME_SIZE * config.joint_count + SEND_FRAME_SIZE
    command_refresh_period_s = 1.0 / max(float(config.sampling.rate), 1.0)
    nominal_feedback_cycle_s = max((bytes_per_cycle * 10.0) / max(float(config.serial.baudrate), 1.0), command_refresh_period_s)
    feedback_stale_timeout_s = max(nominal_feedback_cycle_s * 3.0, 0.1)
    last_command_frame = zero_frame
    last_command_send_time = None
    last_feedback_wait_log_time = None

    log_info(
        "开始真机运行: "
        f"mode={mode}, target=J{config.target_joint + 1}({config.target_joint_name}), "
        f"port={config.serial.port}, baudrate={config.serial.baudrate}"
    )

    try:
        ser = serial.Serial(config.serial.port, config.serial.baudrate, timeout=0)
        ser.reset_input_buffer()
        ser.write(last_command_frame)
        last_command_send_time = time.perf_counter()
    except Exception as exc:
        if reporter is not None:
            reporter.close()
        if pose_estimator is not None:
            pose_estimator.close()
        env.close()
        raise RuntimeError(f"无法打开串口 {config.serial.port}: {exc}") from exc

    try:
        while True:
            bytes_waiting = frame_reader.read_available(ser)
            emitted_sample = False

            while True:
                frame = frame_reader.pop_frame()
                if frame is None:
                    break

                if not 1 <= frame.motor_id <= config.joint_count:
                    continue

                idx = frame.motor_id - 1
                frame_time = time.perf_counter()
                q[idx] = frame.position
                qd[idx] = frame.velocity
                tau_measured[idx] = frame.torque
                mos_temp[idx] = frame.mos_temperature
                coil_temp[idx] = frame.coil_temperature
                last_feedback_time[idx] = frame_time
            now = time.perf_counter()
            target_feedback_available = np.isfinite(last_feedback_time[target_joint_idx])
            target_feedback_fresh = target_feedback_available and (
                now - float(last_feedback_time[target_joint_idx])
            ) < feedback_stale_timeout_s

            if target_feedback_fresh and (
                last_cycle_end is None
                or (now - last_cycle_end) >= command_refresh_period_s
            ):
                cycle_end = now
                if start_time is None:
                    start_time = cycle_end
                elapsed_s = cycle_end - start_time
                cycle_period = (cycle_end - last_cycle_end) if last_cycle_end is not None else 0.0
                last_cycle_end = cycle_end

                safety.assert_joint_limits(q)

                if mode == "collect":
                    assert reference_state is not None
                    reference_state.initialize(env, q, startup_target, elapsed_s)
                    q_cmd_ref, qd_cmd_ref, qdd_cmd_ref = reference_state.sample(
                        elapsed_s,
                        max_step_s=reference_max_step_s,
                    )
                    tau_ff, tau_fb, tau_command = controller.compute_torque(
                        q_cmd=q_cmd_ref,
                        qd_cmd=qd_cmd_ref,
                        qdd_cmd=qdd_cmd_ref,
                        q_curr=q,
                        qd_curr=qd,
                    )
                elif mode == "full_feedforward":
                    # 完整前馈模式: tau_ff(刚体动力学前馈) + 摩擦力补偿 + 反馈
                    assert reference_state is not None
                    assert parameters is not None
                    reference_state.initialize(env, q, startup_target, elapsed_s)
                    q_cmd_ref, qd_cmd_ref, qdd_cmd_ref = reference_state.sample(
                        elapsed_s,
                        max_step_s=reference_max_step_s,
                    )
                    # 计算刚体动力学前馈和反馈项
                    tau_ff, tau_fb, _ = controller.compute_torque(
                        q_cmd=q_cmd_ref,
                        qd_cmd=qd_cmd_ref,
                        qdd_cmd=qdd_cmd_ref,
                        q_curr=q,
                        qd_curr=qd,
                    )
                    # 计算摩擦力补偿 (使用期望速度 qd_cmd_ref)
                    tau_friction = _predict_compensation_torque(
                        qd_cmd_ref,
                        parameters,
                        torque_limits=config.robot.torque_limits,
                    )
                    # 完整前馈 = 刚体动力学前馈 + 摩擦力补偿
                    tau_ff_total = tau_ff + tau_friction
                    # 完整控制力矩 = 完整前馈 + 反馈
                    tau_command = safety.soften_torque_near_joint_limits(
                        q,
                        safety.clamp_torque(tau_ff_total + controller.feedback_scale * tau_fb),
                    )
                    tau_ff = tau_ff_total  # 记录完整前馈
                else:
                    assert parameters is not None
                    q_cmd_ref = q.copy()
                    qd_cmd_ref = np.zeros_like(q)
                    qdd_cmd_ref = np.zeros_like(q)
                    tau_ff = np.zeros_like(q)
                    tau_fb = np.zeros_like(q)
                    tau_command = safety.soften_torque_near_joint_limits(
                        q,
                        safety.clamp_torque(
                            _predict_compensation_torque(
                                qd,
                                parameters,
                                torque_limits=config.robot.torque_limits,
                            )
                        ),
                    )

                last_command_frame = frame_packer.pack(tau_command.astype(np.float32))
                ser.write(last_command_frame)
                last_command_send_time = time.perf_counter()
                emitted_sample = True
                step_index += 1
                last_feedback_wait_log_time = None

                ee_pos = None
                ee_quat = None
                if pose_estimator is not None:
                    ee_pos, ee_quat = pose_estimator.update(q)
                    ee_pos_log.append(np.asarray(ee_pos, dtype=np.float64))
                    ee_quat_log.append(np.asarray(ee_quat, dtype=np.float64))

                time_log.append(float(elapsed_s))
                q_log.append(q.copy())
                qd_log.append(qd.copy())
                q_cmd_log.append(q_cmd_ref.copy())
                qd_cmd_log.append(qd_cmd_ref.copy())
                tau_measured_log.append(tau_measured.copy())
                tau_command_log.append(tau_command.copy())
                tau_feedforward_log.append(tau_ff.copy())
                tau_feedback_log.append(tau_fb.copy())
                mos_temp_log.append(mos_temp.copy())
                coil_temp_log.append(coil_temp.copy())

                uart_latency_ms = cycle_period * 1000.0 if cycle_period > 0.0 else 0.0
                uart_cycle_hz = 1.0 / cycle_period if cycle_period > 1e-9 else 0.0
                uart_transfer_kbps = (bytes_per_cycle * 8.0 / 1000.0) * uart_cycle_hz if uart_cycle_hz > 0.0 else 0.0
                uart_cycle_hz_log.append(float(uart_cycle_hz))
                uart_latency_ms_log.append(float(uart_latency_ms))
                uart_transfer_kbps_log.append(float(uart_transfer_kbps))

                if reporter is not None and step_index % max(config.visualization.rerun_log_stride, 1) == 0:
                    reporter.log_step(
                        elapsed_s=elapsed_s,
                        step_index=step_index,
                        q=q,
                        qd=qd,
                        tau_measured=tau_measured,
                        tau_command=tau_command,
                        mos_temperature=mos_temp,
                        coil_temperature=coil_temp,
                        uart_cycle_hz=uart_cycle_hz,
                        uart_latency_ms=uart_latency_ms,
                        uart_transfer_kbps=uart_transfer_kbps,
                        ee_pos=ee_pos,
                        ee_quat=ee_quat,
                        rx_text=None,
                        tx_text=None,
                    )

                if mode in ("collect", "full_feedforward") and reference_state.is_complete():
                    termination_reason = "collection_complete"
                    raise KeyboardInterrupt

            if not emitted_sample:
                now = time.perf_counter()
                if (
                    last_command_send_time is None
                    or (now - last_command_send_time) >= command_refresh_period_s
                ):
                    if not target_feedback_fresh:
                        last_command_frame = zero_frame
                    ser.write(last_command_frame)
                    last_command_send_time = now

                if not target_feedback_fresh and (
                    last_feedback_wait_log_time is None
                    or (now - last_feedback_wait_log_time) >= 1.0
                ):
                    if target_feedback_available:
                        last_seen_ms = (now - float(last_feedback_time[target_joint_idx])) * 1000.0
                        detail = f"最近一次收到 J{target_joint_idx + 1} 反馈已过去 {last_seen_ms:.0f} ms"
                    else:
                        detail = f"尚未收到 J{target_joint_idx + 1} 反馈"
                    log_info(
                        "等待目标关节反馈，"
                        f"{detail}，继续发送零力矩保持安全。"
                    )
                    last_feedback_wait_log_time = now

            if not emitted_sample or bytes_waiting <= 0:
                time.sleep(0.0005)

    except KeyboardInterrupt:
        pass
    except Exception:
        termination_reason = "error"
        try:
            ser.write(zero_frame)
        finally:
            raise
    finally:
        try:
            ser.write(zero_frame)
        except Exception:
            pass
        ser.close()
        if reporter is not None:
            reporter.close()
        if pose_estimator is not None:
            pose_estimator.close()
        env.close()

    capture_paths = _save_capture(
        config,
        mode=mode,
        termination_reason=termination_reason,
        time_log=time_log,
        q_log=q_log,
        qd_log=qd_log,
        q_cmd_log=q_cmd_log,
        qd_cmd_log=qd_cmd_log,
        tau_measured_log=tau_measured_log,
        tau_command_log=tau_command_log,
        tau_feedforward_log=tau_feedforward_log,
        tau_feedback_log=tau_feedback_log,
        mos_temp_log=mos_temp_log,
        coil_temp_log=coil_temp_log,
        ee_pos_log=ee_pos_log,
        ee_quat_log=ee_quat_log,
        uart_cycle_hz_log=uart_cycle_hz_log,
        uart_latency_ms_log=uart_latency_ms_log,
        uart_transfer_kbps_log=uart_transfer_kbps_log,
    )
    log_info(f"真机采集结果已保存: {capture_paths[0]}")

    if mode == "collect" and time_log:
        _identify_from_capture(
            config,
            time_s=np.asarray(time_log, dtype=np.float64),
            q=np.asarray(q_log, dtype=np.float64),
            qd=np.asarray(qd_log, dtype=np.float64),
            tau_measured=np.asarray(tau_measured_log, dtype=np.float64),
        )

    return capture_paths
