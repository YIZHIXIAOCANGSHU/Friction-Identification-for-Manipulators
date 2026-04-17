from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from friction_identification_core.config import Config
from friction_identification_core.controller import (
    FrictionIdentificationController,
    SafetyGuard,
    load_compensation_parameters,
    predict_compensation_torque,
)
from friction_identification_core.models import CollectedData, FrictionIdentificationResult, IdentificationInputs
from friction_identification_core.serial_protocol import (
    RECV_FRAME_SIZE,
    SEND_FRAME_SIZE,
    SerialFrameReader,
    TorqueCommandFramePacker,
)
from friction_identification_core.mujoco_env import MujocoEnvironment
from friction_identification_core.mujoco_support import build_am_d02_model
from friction_identification_core.runtime import log_info
from friction_identification_core.trajectory import (
    ReferenceTrajectory,
    build_startup_pose,
    sample_reference_trajectory,
)
from friction_identification_core.visualization import build_hardware_reporter, build_pose_estimator


@dataclass
class LiveReferenceState:
    excitation_reference: ReferenceTrajectory
    sample_rate: float
    startup_reference: ReferenceTrajectory | None = None
    startup_duration: float = 0.0
    reference_start_time: float | None = None
    trajectory_elapsed_s: float = 0.0
    last_elapsed_s: float | None = None

    def initialize(
        self,
        env: MujocoEnvironment,
        q_start: np.ndarray,
        startup_target: np.ndarray,
        elapsed_s: float,
    ) -> None:
        if self.reference_start_time is not None:
            return
        self.reference_start_time = float(elapsed_s)
        self.last_elapsed_s = float(elapsed_s)
        self.trajectory_elapsed_s = 0.0
        self.startup_reference = env.build_startup_reference(q_start, startup_target)
        if self.startup_reference is not None and self.startup_reference.time.size > 0:
            self.startup_duration = float(
                self.startup_reference.time[-1] + 1.0 / self.sample_rate
            )

    def sample(
        self,
        elapsed_s: float,
        *,
        max_step_s: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        total_duration = self.startup_duration + float(
            self.excitation_reference.time[-1] + 1.0 / self.sample_rate
        )
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
        <= (
            np.asarray(torque_limits, dtype=np.float64)[active_joint_mask]
            * float(torque_limit_scale)
        ),
        axis=1,
    )
    return finite & within_window & moving & residual_reasonable


class HardwareSource:
    source_name = "hardware"

    def __init__(self, config: Config) -> None:
        self.config = config
        self.env = MujocoEnvironment(config)
        self.reporter = build_hardware_reporter(config)
        self.pose_estimator = build_pose_estimator(config)
        self.inverse_dynamics_backend = self.env

    def build_reference(self) -> ReferenceTrajectory:
        return self.env.build_excitation_reference()

    def supports_identification(self, mode: str) -> bool:
        return mode == "collect"

    def collect(
        self,
        *,
        mode: str,
        reference: ReferenceTrajectory | None,
        controller: FrictionIdentificationController,
        safety: SafetyGuard,
    ) -> CollectedData:
        if mode not in {"collect", "compensate", "full_feedforward"}:
            raise ValueError("mode must be 'collect', 'compensate', or 'full_feedforward'.")
        if mode in {"collect", "full_feedforward"} and reference is None:
            reference = self.build_reference()

        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("缺少 pyserial，请先安装 requirements.txt 中的依赖。") from exc

        parameters = (
            load_compensation_parameters(self.config.summary_path, self.config.joint_count)
            if mode in {"compensate", "full_feedforward"}
            else None
        )

        reference_state = None
        reference_max_step_s = None
        startup_target = None
        if mode in {"collect", "full_feedforward"}:
            assert reference is not None
            startup_target = build_startup_pose(self.config, reference)
            reference_state = LiveReferenceState(
                excitation_reference=reference,
                sample_rate=self.config.sampling.rate,
            )
            reference_max_step_s = (
                max(float(self.config.sampling.hardware_reference_step_factor), 1.0)
                / float(self.config.sampling.rate)
            )

        frame_reader = SerialFrameReader()
        frame_packer = TorqueCommandFramePacker()
        zero_frame = frame_packer.pack(np.zeros(self.config.joint_count, dtype=np.float32))

        q = np.zeros(self.config.joint_count, dtype=np.float64)
        qd = np.zeros(self.config.joint_count, dtype=np.float64)
        tau_measured = np.zeros(self.config.joint_count, dtype=np.float64)
        mos_temp = np.zeros(self.config.joint_count, dtype=np.float64)
        coil_temp = np.zeros(self.config.joint_count, dtype=np.float64)
        target_joint_idx = int(self.config.target_joint)
        last_feedback_time = np.full(self.config.joint_count, np.nan, dtype=np.float64)

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
        bytes_per_cycle = RECV_FRAME_SIZE * self.config.joint_count + SEND_FRAME_SIZE
        command_refresh_period_s = 1.0 / max(float(self.config.sampling.rate), 1.0)
        nominal_feedback_cycle_s = max(
            (bytes_per_cycle * 10.0) / max(float(self.config.serial.baudrate), 1.0),
            command_refresh_period_s,
        )
        feedback_stale_timeout_s = max(nominal_feedback_cycle_s * 3.0, 0.1)
        last_command_frame = zero_frame
        last_command_send_time = None
        last_feedback_wait_log_time = None

        log_info(
            "开始真机运行: "
            f"mode={mode}, target=J{self.config.target_joint + 1}({self.config.target_joint_name}), "
            f"port={self.config.serial.port}, baudrate={self.config.serial.baudrate}"
        )

        ser = None
        try:
            ser = serial.Serial(self.config.serial.port, self.config.serial.baudrate, timeout=0)
            ser.reset_input_buffer()
            ser.write(last_command_frame)
            last_command_send_time = time.perf_counter()

            while True:
                bytes_waiting = frame_reader.read_available(ser)
                emitted_sample = False

                while True:
                    frame = frame_reader.pop_frame()
                    if frame is None:
                        break

                    if not 1 <= frame.motor_id <= self.config.joint_count:
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
                    last_cycle_end is None or (now - last_cycle_end) >= command_refresh_period_s
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
                        assert startup_target is not None
                        reference_state.initialize(self.env, q, startup_target, elapsed_s)
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
                        assert reference_state is not None
                        assert startup_target is not None
                        assert parameters is not None
                        reference_state.initialize(self.env, q, startup_target, elapsed_s)
                        q_cmd_ref, qd_cmd_ref, qdd_cmd_ref = reference_state.sample(
                            elapsed_s,
                            max_step_s=reference_max_step_s,
                        )
                        tau_ff, tau_fb, _ = controller.compute_torque(
                            q_cmd=q_cmd_ref,
                            qd_cmd=qd_cmd_ref,
                            qdd_cmd=qdd_cmd_ref,
                            q_curr=q,
                            qd_curr=qd,
                        )
                        tau_friction = predict_compensation_torque(
                            qd_cmd_ref,
                            parameters,
                            torque_limits=self.config.robot.torque_limits,
                        )
                        tau_ff_total = tau_ff + tau_friction
                        tau_command = safety.soften_torque_near_joint_limits(
                            q,
                            safety.clamp_torque(
                                tau_ff_total + controller.feedback_scale * tau_fb
                            ),
                        )
                        tau_ff = tau_ff_total
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
                                predict_compensation_torque(
                                    qd,
                                    parameters,
                                    torque_limits=self.config.robot.torque_limits,
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
                    if self.pose_estimator is not None:
                        ee_pos, ee_quat = self.pose_estimator.update(q)
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
                    uart_transfer_kbps = (
                        (bytes_per_cycle * 8.0 / 1000.0) * uart_cycle_hz
                        if uart_cycle_hz > 0.0
                        else 0.0
                    )
                    uart_cycle_hz_log.append(float(uart_cycle_hz))
                    uart_latency_ms_log.append(float(uart_latency_ms))
                    uart_transfer_kbps_log.append(float(uart_transfer_kbps))

                    if self.reporter is not None and (
                        step_index % max(self.config.visualization.rerun_log_stride, 1) == 0
                    ):
                        self.reporter.log_step(
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

                    if mode in {"collect", "full_feedforward"} and reference_state.is_complete():
                        termination_reason = "collection_complete"
                        break

                if mode in {"collect", "full_feedforward"} and termination_reason == "collection_complete":
                    break

                if not emitted_sample:
                    now = time.perf_counter()
                    if last_command_send_time is None or (
                        now - last_command_send_time
                    ) >= command_refresh_period_s:
                        if not target_feedback_fresh:
                            last_command_frame = zero_frame
                        ser.write(last_command_frame)
                        last_command_send_time = now

                    if not target_feedback_fresh and (
                        last_feedback_wait_log_time is None
                        or (now - last_feedback_wait_log_time) >= 1.0
                    ):
                        if target_feedback_available:
                            last_seen_ms = (
                                now - float(last_feedback_time[target_joint_idx])
                            ) * 1000.0
                            detail = (
                                f"最近一次收到 J{target_joint_idx + 1} 反馈已过去 {last_seen_ms:.0f} ms"
                            )
                        else:
                            detail = f"尚未收到 J{target_joint_idx + 1} 反馈"
                        log_info(f"等待目标关节反馈，{detail}，继续发送零力矩保持安全。")
                        last_feedback_wait_log_time = now

                    if not emitted_sample or bytes_waiting <= 0:
                        time.sleep(0.0005)

        except KeyboardInterrupt:
            termination_reason = "interrupted"
        except Exception:
            termination_reason = "error"
            raise
        finally:
            if ser is not None:
                try:
                    ser.write(zero_frame)
                except Exception:
                    pass
                try:
                    ser.close()
                except Exception:
                    pass

        q_array = np.asarray(q_log, dtype=np.float64).reshape(-1, self.config.joint_count)
        if ee_pos_log:
            ee_pos = np.asarray(ee_pos_log, dtype=np.float64).reshape(-1, 3)
            ee_quat = np.asarray(ee_quat_log, dtype=np.float64).reshape(-1, 4)
        else:
            ee_pos = np.zeros((q_array.shape[0], 3), dtype=np.float64)
            ee_quat = np.tile(
                np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
                (q_array.shape[0], 1),
            )

        return CollectedData(
            source=self.source_name,
            mode=mode,
            time=np.asarray(time_log, dtype=np.float64),
            q=q_array,
            qd=np.asarray(qd_log, dtype=np.float64).reshape(-1, self.config.joint_count),
            q_cmd=np.asarray(q_cmd_log, dtype=np.float64).reshape(-1, self.config.joint_count),
            qd_cmd=np.asarray(qd_cmd_log, dtype=np.float64).reshape(-1, self.config.joint_count),
            tau_command=np.asarray(tau_command_log, dtype=np.float64).reshape(-1, self.config.joint_count),
            tau_measured=np.asarray(tau_measured_log, dtype=np.float64).reshape(-1, self.config.joint_count),
            tau_feedforward=np.asarray(
                tau_feedforward_log,
                dtype=np.float64,
            ).reshape(-1, self.config.joint_count),
            tau_feedback=np.asarray(
                tau_feedback_log,
                dtype=np.float64,
            ).reshape(-1, self.config.joint_count),
            ee_pos=ee_pos,
            ee_quat=ee_quat,
            mos_temperature=np.asarray(
                mos_temp_log,
                dtype=np.float64,
            ).reshape(-1, self.config.joint_count),
            coil_temperature=np.asarray(
                coil_temp_log,
                dtype=np.float64,
            ).reshape(-1, self.config.joint_count),
            uart_cycle_hz=np.asarray(uart_cycle_hz_log, dtype=np.float64),
            uart_latency_ms=np.asarray(uart_latency_ms_log, dtype=np.float64),
            uart_transfer_kbps=np.asarray(uart_transfer_kbps_log, dtype=np.float64),
            metadata={"termination_reason": termination_reason},
        )

    def prepare_identification(self, data: CollectedData) -> IdentificationInputs | None:
        if data.sample_count < 32:
            log_info("真机样本过少，跳过实际数据辨识。")
            return None

        gradient_order = 2 if data.sample_count >= 3 else 1
        qdd = np.gradient(data.qd, data.time, axis=0, edge_order=gradient_order)
        dynamics = RigidBodyDynamics(
            model_path=str(self.config.robot.urdf_path),
            joint_names=list(self.config.robot.joint_names),
            tcp_offset=self.config.robot.tcp_offset,
        )
        tau_rigid = dynamics.batch_inverse_dynamics(data.q, data.qd, qdd)
        tau_friction = data.tau_measured - tau_rigid

        safety = SafetyGuard(self.config, active_joint_mask=self.config.target_joint_mask)
        lower, upper = safety.safe_joint_window()
        clean_mask = _build_residual_clean_sample_mask(
            q=data.q,
            qd=data.qd,
            tau_residual=tau_friction,
            lower=lower,
            upper=upper,
            torque_limits=self.config.robot.torque_limits,
            active_joints=self.config.target_joint_mask,
            min_motion_speed=max(float(self.config.fitting.min_velocity_threshold), 0.01),
        )
        retained = int(np.count_nonzero(clean_mask))
        if retained < 16:
            log_info("筛样后真机样本不足，跳过实际数据辨识。")
            return None

        data.qdd = qdd
        data.tau_rigid = tau_rigid
        data.tau_friction = tau_friction
        data.clean_mask = clean_mask

        qd_clean = data.qd[clean_mask]
        tau_clean = tau_friction[clean_mask]
        return IdentificationInputs(
            velocity=qd_clean[:, self.config.target_joint_mask],
            torque=tau_clean[:, self.config.target_joint_mask],
            joint_names=[self.config.target_joint_name],
            clean_mask=clean_mask,
            metadata={"retained_samples": retained},
        )

    def finalize(
        self,
        data: CollectedData | None,
        result: FrictionIdentificationResult | None,
    ) -> None:
        if self.reporter is not None:
            self.reporter.close()
            self.reporter = None
        if self.pose_estimator is not None:
            self.pose_estimator.close()
            self.pose_estimator = None
        self.env.close()
