from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from friction_identification_core.config import Config
from friction_identification_core.controller import (
    FrictionIdentificationController,
    SafetyGuard,
    has_compensation_results,
    load_compensation_parameters,
    predict_compensation_torque,
)
from friction_identification_core.models import CollectedData, FrictionIdentificationResult, IdentificationInputs
from friction_identification_core.mujoco_env import MujocoEnvironment
from friction_identification_core.mujoco_support import build_am_d02_model
from friction_identification_core.results import IdentificationResults
from friction_identification_core.runtime import log_info
from friction_identification_core.serial_protocol import (
    RECV_FRAME_SIZE,
    SEND_FRAME_SIZE,
    SerialFrameReader,
    TorqueCommandFramePacker,
)
from friction_identification_core.status import (
    compute_limit_margin_remaining,
    compute_range_ratio,
    compute_rotation_state,
)
from friction_identification_core.trajectory import (
    ReferenceSample,
    ReferenceTrajectory,
    build_startup_pose,
    resolve_joint_window,
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
            self.startup_duration = float(self.startup_reference.time[-1] + 1.0 / self.sample_rate)

    def sample(
        self,
        elapsed_s: float,
        *,
        max_step_s: float | None = None,
    ) -> ReferenceSample:
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
        <= np.asarray(torque_limits, dtype=np.float64)[active_joint_mask] * float(torque_limit_scale),
        axis=1,
    )
    return finite & within_window & moving & residual_reasonable


def _smooth_velocity_and_estimate_acceleration(
    time_samples: np.ndarray,
    velocity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | str]]:
    time_samples = np.asarray(time_samples, dtype=np.float64).reshape(-1)
    velocity = np.asarray(velocity, dtype=np.float64)
    if velocity.ndim != 2 or velocity.shape[0] != time_samples.size:
        raise ValueError("velocity must have shape [N, J] aligned with time_samples.")

    gradient_order = 2 if time_samples.size >= 3 else 1
    sample_dt = np.diff(time_samples)
    positive_dt = sample_dt[sample_dt > 1e-9]
    if positive_dt.size == 0:
        qdd = np.gradient(velocity, time_samples, axis=0, edge_order=gradient_order)
        return velocity.copy(), qdd, {"velocity_filter": "gradient"}

    polyorder = 3 if time_samples.size >= 7 else max(1, time_samples.size - 3)
    max_window = min(15, time_samples.size if time_samples.size % 2 == 1 else time_samples.size - 1)
    min_window = max(5, polyorder + 2)
    if min_window % 2 == 0:
        min_window += 1
    window_length = max_window

    if window_length < min_window:
        qdd = np.gradient(velocity, time_samples, axis=0, edge_order=gradient_order)
        return velocity.copy(), qdd, {"velocity_filter": "gradient"}

    delta = float(np.mean(positive_dt))
    qd_filtered = savgol_filter(
        velocity,
        window_length=window_length,
        polyorder=polyorder,
        deriv=0,
        delta=delta,
        axis=0,
        mode="interp",
    )
    qdd = savgol_filter(
        qd_filtered,
        window_length=window_length,
        polyorder=polyorder,
        deriv=1,
        delta=delta,
        axis=0,
        mode="interp",
    )
    return qd_filtered, qdd, {
        "velocity_filter": "savgol",
        "velocity_filter_window_length": int(window_length),
        "velocity_filter_polyorder": int(polyorder),
    }


class HardwareSource:
    source_name = "hardware"

    def __init__(self, config: Config) -> None:
        self.config = config
        self.env = MujocoEnvironment(config)
        self.reporter = build_hardware_reporter(config)
        self.pose_estimator = build_pose_estimator(config)
        self.inverse_dynamics_backend = self.env
        self._summary_published = False

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
        batch_index: int = 1,
        total_batches: int = 1,
    ) -> CollectedData:
        if mode not in {"collect", "compensate"}:
            raise ValueError("mode must be 'collect' or 'compensate'.")
        if mode == "collect" and reference is None:
            reference = self.build_reference()

        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("缺少 pyserial，请先安装 requirements.txt 中的依赖。") from exc

        parameters = None
        if mode == "compensate":
            if not has_compensation_results(self.config.summary_path):
                log_info("未找到历史辨识汇总，补偿模式将使用零摩擦补偿力矩。")
            parameters = load_compensation_parameters(self.config.summary_path, self.config.joint_count)

        reference_state = None
        reference_max_step_s = None
        startup_target = None
        if mode == "collect":
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

        frame_reader = SerialFrameReader(max_motor_id=self.config.joint_count)
        frame_packer = TorqueCommandFramePacker()
        zero_frame = frame_packer.pack(np.zeros(self.config.joint_count, dtype=np.float32))

        q = np.zeros(self.config.joint_count, dtype=np.float64)
        qd = np.zeros(self.config.joint_count, dtype=np.float64)
        tau_measured = np.zeros(self.config.joint_count, dtype=np.float64)
        mos_temp = np.zeros(self.config.joint_count, dtype=np.float64)
        coil_temp = np.zeros(self.config.joint_count, dtype=np.float64)
        last_feedback_time = np.full(self.config.joint_count, np.nan, dtype=np.float64)
        feedback_group_mask = np.zeros(self.config.joint_count, dtype=bool)
        previous_qd = np.zeros(self.config.joint_count, dtype=np.float64)

        time_log: list[float] = []
        q_log: list[np.ndarray] = []
        qd_log: list[np.ndarray] = []
        q_cmd_log: list[np.ndarray] = []
        qd_cmd_log: list[np.ndarray] = []
        qdd_cmd_log: list[np.ndarray] = []
        tau_measured_log: list[np.ndarray] = []
        tau_command_log: list[np.ndarray] = []
        tau_track_ff_log: list[np.ndarray] = []
        tau_track_fb_log: list[np.ndarray] = []
        tau_friction_comp_log: list[np.ndarray] = []
        tau_residual_log: list[np.ndarray] = []
        rotation_state_log: list[np.ndarray] = []
        range_ratio_log: list[np.ndarray] = []
        limit_margin_log: list[np.ndarray] = []
        batch_index_log: list[int] = []
        phase_name_log: list[str] = []
        mos_temp_log: list[np.ndarray] = []
        coil_temp_log: list[np.ndarray] = []
        ee_pos_log: list[np.ndarray] = []
        ee_quat_log: list[np.ndarray] = []
        uart_cycle_hz_log: list[float] = []
        uart_latency_ms_log: list[float] = []
        uart_transfer_kbps_log: list[float] = []

        active_joint_mask = self.config.active_joint_mask
        active_joint_indices = np.flatnonzero(active_joint_mask)
        active_window_mode = self.config.identification.excitation.window_mode
        motion_lower, motion_upper, _ = resolve_joint_window(
            self.config.robot.joint_limits,
            safety_margin=self.config.safety.joint_limit_margin,
            window_mode=active_window_mode,
        )
        valid_sample_count = 0

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
            "开始真机并行运行: "
            f"mode={mode}, batch={batch_index}/{total_batches}, "
            f"active_joints={[int(idx) + 1 for idx in active_joint_indices]}, "
            f"port={self.config.serial.port}, baudrate={self.config.serial.baudrate}"
        )
        log_info("串口回传按 active_joints 凑齐一轮后再发送下一次控制/记录样本。")
        if nominal_feedback_cycle_s > (command_refresh_period_s * 1.25):
            log_info(
                "UART 理论完整收发周期约 "
                f"{nominal_feedback_cycle_s * 1000.0:.1f} ms "
                f"({1.0 / nominal_feedback_cycle_s:.1f} Hz)，"
                f"当前 sampling.rate={self.config.sampling.rate:.1f} Hz 将以串口实际反馈节奏为准。"
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
                feedback_group_ready = False

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
                    feedback_group_mask[idx] = True

                    if np.all(feedback_group_mask[active_joint_mask]):
                        feedback_group_ready = True
                        break

                now = time.perf_counter()
                feedback_group_ready = feedback_group_ready or np.all(feedback_group_mask[active_joint_mask])

                if feedback_group_ready and (
                    last_cycle_end is None or (now - last_cycle_end) >= command_refresh_period_s
                ):
                    cycle_end = now
                    if start_time is None:
                        start_time = cycle_end
                    elapsed_s = cycle_end - start_time
                    cycle_period = (cycle_end - last_cycle_end) if last_cycle_end is not None else 0.0
                    last_cycle_end = cycle_end

                    safety.assert_joint_limits(q, window_mode=active_window_mode)

                    if mode == "collect":
                        assert reference_state is not None
                        assert startup_target is not None
                        reference_state.initialize(self.env, q, startup_target, elapsed_s)
                        reference_sample = reference_state.sample(
                            elapsed_s,
                            max_step_s=reference_max_step_s,
                        )
                        q_cmd_ref = reference_sample.q_cmd
                        qd_cmd_ref = reference_sample.qd_cmd
                        qdd_cmd_ref = reference_sample.qdd_cmd
                        phase_name = reference_sample.phase_name or "collect"
                        tau_track_ff, tau_track_fb, tau_command = controller.compute_torque(
                            q_cmd=q_cmd_ref,
                            qd_cmd=qd_cmd_ref,
                            qdd_cmd=qdd_cmd_ref,
                            q_curr=q,
                            qd_curr=qd,
                        )
                        tau_friction_comp = np.zeros_like(q)
                    else:
                        assert parameters is not None
                        q_cmd_ref = q.copy()
                        qd_cmd_ref = np.zeros_like(q)
                        qdd_cmd_ref = np.zeros_like(q)
                        phase_name = "compensate"
                        tau_track_ff = np.zeros_like(q)
                        tau_track_fb = np.zeros_like(q)
                        tau_friction_comp = safety.soften_torque_near_joint_limits(
                            q,
                            safety.clamp_torque(
                                predict_compensation_torque(
                                    qd,
                                    parameters,
                                    torque_limits=self.config.robot.torque_limits,
                                )
                            ),
                            window_mode=active_window_mode,
                        )
                        tau_command = tau_friction_comp.copy()

                    qdd_live = (
                        (qd - previous_qd) / cycle_period
                        if cycle_period > 1e-9
                        else np.zeros_like(qd)
                    )
                    previous_qd = qd.copy()
                    tau_rigid_live = self.env.inverse_dynamics(q, qd, qdd_live)
                    tau_residual = tau_measured - tau_rigid_live
                    rotation_state = compute_rotation_state(
                        qd,
                        velocity_eps=self.config.status.velocity_eps,
                    )
                    range_ratio = compute_range_ratio(q, motion_lower, motion_upper)
                    limit_margin_remaining = compute_limit_margin_remaining(q, motion_lower, motion_upper)
                    sample_valid = bool(
                        _build_residual_clean_sample_mask(
                            q=q[None, :],
                            qd=qd[None, :],
                            tau_residual=tau_residual[None, :],
                            lower=motion_lower,
                            upper=motion_upper,
                            torque_limits=self.config.robot.torque_limits,
                            active_joints=active_joint_mask,
                            min_motion_speed=max(float(self.config.fitting.min_velocity_threshold), 0.01),
                        )[0]
                    )
                    valid_sample_count += int(sample_valid)
                    valid_sample_ratio = valid_sample_count / max(step_index + 1, 1)

                    last_command_frame = frame_packer.pack(tau_command.astype(np.float32))
                    ser.write(last_command_frame)
                    last_command_send_time = time.perf_counter()
                    emitted_sample = True
                    step_index += 1
                    last_feedback_wait_log_time = None
                    feedback_group_mask[:] = False

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
                    qdd_cmd_log.append(qdd_cmd_ref.copy())
                    tau_measured_log.append(tau_measured.copy())
                    tau_command_log.append(tau_command.copy())
                    tau_track_ff_log.append(tau_track_ff.copy())
                    tau_track_fb_log.append(tau_track_fb.copy())
                    tau_friction_comp_log.append(tau_friction_comp.copy())
                    tau_residual_log.append(tau_residual.copy())
                    rotation_state_log.append(rotation_state.copy())
                    range_ratio_log.append(range_ratio.copy())
                    limit_margin_log.append(limit_margin_remaining.copy())
                    batch_index_log.append(batch_index)
                    phase_name_log.append(phase_name)
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
                            batch_index=batch_index,
                            total_batches=total_batches,
                            q=q,
                            qd=qd,
                            q_cmd=q_cmd_ref,
                            qd_cmd=qd_cmd_ref,
                            tau_measured=tau_measured,
                            tau_command=tau_command,
                            tau_track_ff=tau_track_ff,
                            tau_track_fb=tau_track_fb,
                            tau_friction_comp=tau_friction_comp,
                            tau_residual=tau_residual,
                            rotation_state=rotation_state,
                            range_ratio=range_ratio,
                            limit_margin_remaining=limit_margin_remaining,
                            mos_temperature=mos_temp,
                            coil_temperature=coil_temp,
                            uart_cycle_hz=uart_cycle_hz,
                            uart_latency_ms=uart_latency_ms,
                            uart_transfer_kbps=uart_transfer_kbps,
                            valid_sample_ratio=valid_sample_ratio,
                            phase_name=phase_name,
                            ee_pos=ee_pos,
                            ee_quat=ee_quat,
                        )

                    if mode == "collect" and reference_state.is_complete():
                        termination_reason = "collection_complete"
                        break

                if mode == "collect" and termination_reason == "collection_complete":
                    break

                if not emitted_sample:
                    now = time.perf_counter()
                    if last_command_send_time is None or (now - last_command_send_time) >= command_refresh_period_s:
                        if not feedback_group_ready:
                            last_command_frame = zero_frame
                        ser.write(last_command_frame)
                        last_command_send_time = now

                    if not feedback_group_ready and (
                        last_feedback_wait_log_time is None
                        or (now - last_feedback_wait_log_time) >= 1.0
                    ):
                        pending = [int(idx) + 1 for idx in active_joint_indices if not feedback_group_mask[idx]]
                        missing = [idx for idx in pending if not np.isfinite(last_feedback_time[idx - 1])]
                        stale = [
                            idx
                            for idx in pending
                            if np.isfinite(last_feedback_time[idx - 1])
                            and (now - float(last_feedback_time[idx - 1])) >= feedback_stale_timeout_s
                        ]
                        detail = f"pending={pending}, missing={missing}, stale={stale}"
                        log_info(f"等待本轮回传凑齐 active_joints，继续发送零力矩保持安全。{detail}")
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
            ee_quat = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64), (q_array.shape[0], 1))

        return CollectedData(
            source=self.source_name,
            mode=mode,
            time=np.asarray(time_log, dtype=np.float64),
            q=q_array,
            qd=np.asarray(qd_log, dtype=np.float64).reshape(-1, self.config.joint_count),
            q_cmd=np.asarray(q_cmd_log, dtype=np.float64).reshape(-1, self.config.joint_count),
            qd_cmd=np.asarray(qd_cmd_log, dtype=np.float64).reshape(-1, self.config.joint_count),
            qdd_cmd=np.asarray(qdd_cmd_log, dtype=np.float64).reshape(-1, self.config.joint_count),
            tau_command=np.asarray(tau_command_log, dtype=np.float64).reshape(-1, self.config.joint_count),
            tau_measured=np.asarray(tau_measured_log, dtype=np.float64).reshape(-1, self.config.joint_count),
            tau_track_ff=np.asarray(tau_track_ff_log, dtype=np.float64).reshape(-1, self.config.joint_count),
            tau_track_fb=np.asarray(tau_track_fb_log, dtype=np.float64).reshape(-1, self.config.joint_count),
            tau_friction_comp=np.asarray(
                tau_friction_comp_log,
                dtype=np.float64,
            ).reshape(-1, self.config.joint_count),
            tau_residual=np.asarray(tau_residual_log, dtype=np.float64).reshape(-1, self.config.joint_count),
            rotation_state=np.asarray(rotation_state_log, dtype=np.int8).reshape(-1, self.config.joint_count),
            range_ratio=np.asarray(range_ratio_log, dtype=np.float64).reshape(-1, self.config.joint_count),
            limit_margin_remaining=np.asarray(
                limit_margin_log,
                dtype=np.float64,
            ).reshape(-1, self.config.joint_count),
            batch_index=np.asarray(batch_index_log, dtype=np.int64),
            phase_name=np.asarray(phase_name_log, dtype="<U32"),
            ee_pos=ee_pos,
            ee_quat=ee_quat,
            mos_temperature=np.asarray(mos_temp_log, dtype=np.float64).reshape(-1, self.config.joint_count),
            coil_temperature=np.asarray(coil_temp_log, dtype=np.float64).reshape(-1, self.config.joint_count),
            uart_cycle_hz=np.asarray(uart_cycle_hz_log, dtype=np.float64),
            uart_latency_ms=np.asarray(uart_latency_ms_log, dtype=np.float64),
            uart_transfer_kbps=np.asarray(uart_transfer_kbps_log, dtype=np.float64),
            metadata={
                "termination_reason": termination_reason,
                "batch_index": batch_index,
                "total_batches": total_batches,
                "fit_joint_indices": active_joint_indices.tolist(),
                "active_joint_indices": active_joint_indices.tolist(),
                "valid_sample_ratio_live": (valid_sample_count / max(step_index, 1)) if step_index > 0 else 0.0,
            },
        )

    def prepare_identification(self, data: CollectedData) -> IdentificationInputs | None:
        if data.sample_count < 32:
            log_info("真机样本过少，跳过实际数据辨识。")
            return None

        qd_filtered, qdd, filter_metadata = _smooth_velocity_and_estimate_acceleration(data.time, data.qd)
        dynamics = RigidBodyDynamics(
            model_path=str(self.config.robot.urdf_path),
            joint_names=list(self.config.robot.joint_names),
            tcp_offset=self.config.robot.tcp_offset,
        )
        tau_rigid = dynamics.batch_inverse_dynamics(data.q, qd_filtered, qdd)
        tau_residual = data.tau_measured - tau_rigid

        lower, upper, _ = resolve_joint_window(
            self.config.robot.joint_limits,
            safety_margin=self.config.safety.joint_limit_margin,
            window_mode=self.config.identification.excitation.window_mode,
        )
        clean_mask = _build_residual_clean_sample_mask(
            q=data.q,
            qd=qd_filtered,
            tau_residual=tau_residual,
            lower=lower,
            upper=upper,
            torque_limits=self.config.robot.torque_limits,
            active_joints=self.config.active_joint_mask,
            min_motion_speed=max(float(self.config.fitting.min_velocity_threshold), 0.01),
        )
        retained = int(np.count_nonzero(clean_mask))
        if retained < 16:
            log_info("筛样后真机样本不足，跳过实际数据辨识。")
            return None

        data.qdd = qdd
        data.tau_rigid = tau_rigid
        data.tau_residual = tau_residual
        data.tau_friction = tau_residual
        data.clean_mask = clean_mask
        data.metadata.update(
            filter_metadata,
            retained_samples=retained,
            valid_sample_ratio=(retained / max(data.sample_count, 1)),
            fit_joint_indices=self.config.active_joint_indices.tolist(),
        )

        qd_clean = qd_filtered[clean_mask][:, self.config.active_joint_mask]
        tau_clean = tau_residual[clean_mask][:, self.config.active_joint_mask]
        return IdentificationInputs(
            velocity=qd_clean,
            torque=tau_clean,
            joint_names=self.config.active_joint_names,
            clean_mask=clean_mask,
            metadata={**filter_metadata, "retained_samples": retained},
        )

    def finalize(
        self,
        data: CollectedData | None,
        result: FrictionIdentificationResult | None,
    ) -> None:
        if self.reporter is not None and result is not None and not self._summary_published:
            self.reporter.log_identification_summary(
                result,
                active_joint_names=self.config.active_joint_names,
                active_joint_indices=self.config.active_joint_indices.tolist(),
            )
        if self.reporter is not None:
            self.reporter.close()
            self.reporter = None
        if self.pose_estimator is not None:
            self.pose_estimator.close()
            self.pose_estimator = None
        self.env.close()

    def publish_summary(self, summary: IdentificationResults) -> None:
        if self.reporter is not None:
            self.reporter.log_loaded_summary(summary)
            self._summary_published = True
