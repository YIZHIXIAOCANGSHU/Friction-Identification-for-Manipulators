from __future__ import annotations

from collections import deque
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
    JointFeedbackFrame,
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
        *,
        startup_duration_override: float | None = None,
    ) -> None:
        if self.reference_start_time is not None:
            return
        self.reference_start_time = float(elapsed_s)
        self.last_elapsed_s = float(elapsed_s)
        self.trajectory_elapsed_s = 0.0
        self.startup_reference = env.build_startup_reference(
            q_start,
            startup_target,
            duration_override=startup_duration_override,
        )
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


class FeedbackCycleWindow:
    """Track active-joint refresh order between two emitted samples."""

    def __init__(self, *, active_joint_mask: np.ndarray) -> None:
        self.active_joint_mask = np.asarray(active_joint_mask, dtype=bool).reshape(-1)
        self.joint_count = int(self.active_joint_mask.size)
        self.active_joint_count = int(np.count_nonzero(self.active_joint_mask))
        self._window: deque[int] = deque()
        self._seen = np.zeros(self.joint_count, dtype=bool)

    def push(self, joint_index: int) -> None:
        joint_index = int(joint_index)
        if not 0 <= joint_index < self.joint_count:
            return
        if not self.active_joint_mask[joint_index]:
            return
        if self._seen[joint_index]:
            return
        self._window.append(joint_index)
        self._seen[joint_index] = True

    def is_ready(self) -> bool:
        return bool(self.active_joint_count > 0 and np.all(self._seen[self.active_joint_mask]))

    def current_joint_ids(self) -> list[int]:
        return [int(joint_index) + 1 for joint_index in self._window]

    def pending_joint_ids(self) -> list[int]:
        pending_mask = self.active_joint_mask & (~self._seen)
        return [int(joint_index) + 1 for joint_index in np.flatnonzero(pending_mask)]

    def progress(self) -> tuple[int, int]:
        return int(np.count_nonzero(self._seen[self.active_joint_mask])), self.active_joint_count

    def advance_after_emit(self) -> list[int]:
        ready_joint_ids = self.current_joint_ids()
        self._window.clear()
        self._seen[:] = False
        return ready_joint_ids


def _initialized_feedback_joint_ids(
    *,
    active_joint_mask: np.ndarray,
    last_feedback_time: np.ndarray,
) -> list[int]:
    initialized_mask = np.asarray(active_joint_mask, dtype=bool).reshape(-1) & np.isfinite(
        np.asarray(last_feedback_time, dtype=np.float64).reshape(-1)
    )
    return [int(joint_index) + 1 for joint_index in np.flatnonzero(initialized_mask)]


def _fresh_feedback_joint_ids(
    *,
    active_joint_mask: np.ndarray,
    last_feedback_time: np.ndarray,
    now: float,
    stale_timeout_s: float,
) -> list[int]:
    active_joint_mask = np.asarray(active_joint_mask, dtype=bool).reshape(-1)
    last_feedback_time = np.asarray(last_feedback_time, dtype=np.float64).reshape(-1)
    fresh_mask = np.zeros_like(active_joint_mask, dtype=bool)
    initialized_mask = active_joint_mask & np.isfinite(last_feedback_time)
    if np.any(initialized_mask):
        fresh_mask[initialized_mask] = (
            np.maximum(float(now) - last_feedback_time[initialized_mask], 0.0)
            <= float(stale_timeout_s)
    )
    return [int(joint_index) + 1 for joint_index in np.flatnonzero(fresh_mask)]


def _ordered_unique_joint_ids(joint_indices: list[int]) -> list[int]:
    ordered: list[int] = []
    seen: set[int] = set()
    for joint_index in joint_indices:
        joint_index = int(joint_index)
        if joint_index in seen:
            continue
        ordered.append(joint_index)
        seen.add(joint_index)
    return ordered


def _apply_feedback_frame(
    *,
    frame: JointFeedbackFrame,
    joint_count: int,
    q: np.ndarray,
    qd: np.ndarray,
    tau_measured: np.ndarray,
    mos_temp: np.ndarray,
    coil_temp: np.ndarray,
    last_feedback_time: np.ndarray,
) -> int | None:
    motor_id = int(frame.motor_id)
    if not 1 <= motor_id <= int(joint_count):
        return None

    joint_index = motor_id - 1
    q[joint_index] = float(frame.position)
    qd[joint_index] = float(frame.velocity)
    tau_measured[joint_index] = float(frame.torque)
    mos_temp[joint_index] = float(frame.mos_temperature)
    coil_temp[joint_index] = float(frame.coil_temperature)
    last_feedback_time[joint_index] = time.perf_counter()
    return joint_index


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


def _build_residual_clean_joint_mask(
    *,
    q: np.ndarray,
    qd: np.ndarray,
    tau_residual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    torque_limits: np.ndarray,
    active_joints: np.ndarray,
    min_motion_speed: float,
    refreshed_mask: np.ndarray | None = None,
    torque_limit_scale: float = 1.5,
) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    qd = np.asarray(qd, dtype=np.float64)
    tau_residual = np.asarray(tau_residual, dtype=np.float64)
    active_joint_mask = np.asarray(active_joints, dtype=bool).reshape(-1)
    if q.shape != qd.shape or q.shape != tau_residual.shape or q.ndim != 2:
        raise ValueError("q/qd/tau_residual 必须是同形状二维数组 [N, J]。")

    within_window = (q >= lower[None, :]) & (q <= upper[None, :])
    finite = np.isfinite(q) & np.isfinite(qd) & np.isfinite(tau_residual)
    moving = np.abs(qd) >= float(min_motion_speed)
    residual_reasonable = (
        np.abs(tau_residual)
        <= np.asarray(torque_limits, dtype=np.float64)[None, :] * float(torque_limit_scale)
    )

    joint_mask = finite & within_window & moving & residual_reasonable
    if refreshed_mask is not None:
        refreshed_mask = np.asarray(refreshed_mask, dtype=bool)
        if refreshed_mask.shape != joint_mask.shape:
            raise ValueError("refreshed_mask 必须与 q/qd/tau_residual 形状一致。")
        joint_mask &= refreshed_mask
    joint_mask[:, ~active_joint_mask] = False
    return joint_mask


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
    refreshed_mask: np.ndarray | None = None,
    torque_limit_scale: float = 1.5,
) -> np.ndarray:
    joint_mask = _build_residual_clean_joint_mask(
        q=q,
        qd=qd,
        tau_residual=tau_residual,
        lower=lower,
        upper=upper,
        torque_limits=torque_limits,
        active_joints=active_joints,
        min_motion_speed=min_motion_speed,
        refreshed_mask=refreshed_mask,
        torque_limit_scale=torque_limit_scale,
    )
    active_joint_mask = np.asarray(active_joints, dtype=bool).reshape(-1)
    if not np.any(active_joint_mask):
        return np.zeros(joint_mask.shape[0], dtype=bool)
    return np.any(joint_mask[:, active_joint_mask], axis=1)


def _build_motion_sufficient_joint_mask(
    *,
    q: np.ndarray,
    qd: np.ndarray,
    q_cmd: np.ndarray,
    qd_cmd: np.ndarray,
    active_joints: np.ndarray,
    min_motion_speed: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    q = np.asarray(q, dtype=np.float64)
    qd = np.asarray(qd, dtype=np.float64)
    q_cmd = np.asarray(q_cmd, dtype=np.float64)
    qd_cmd = np.asarray(qd_cmd, dtype=np.float64)
    active_joint_mask = np.asarray(active_joints, dtype=bool).reshape(-1)
    if q.shape != qd.shape or q.shape != q_cmd.shape or q.shape != qd_cmd.shape or q.ndim != 2:
        raise ValueError("q/qd/q_cmd/qd_cmd must be same-shape 2D arrays.")

    actual_q_span = np.nanmax(q, axis=0) - np.nanmin(q, axis=0)
    commanded_q_span = np.nanmax(q_cmd, axis=0) - np.nanmin(q_cmd, axis=0)
    actual_speed_p95 = np.nanpercentile(np.abs(qd), 95.0, axis=0)
    commanded_speed_p95 = np.nanpercentile(np.abs(qd_cmd), 95.0, axis=0)

    # Reject joints that only exhibit encoder noise while the reference commands
    # large position or velocity excursions.
    min_motion_speed = max(float(min_motion_speed), 0.01)
    min_q_span_abs = 1e-3
    min_q_span_ratio = 0.02
    min_speed_ratio = 0.05

    sufficient_q_span = actual_q_span >= np.maximum(min_q_span_abs, commanded_q_span * min_q_span_ratio)
    sufficient_speed = actual_speed_p95 >= np.maximum(min_motion_speed, commanded_speed_p95 * min_speed_ratio)
    commanded_motion = (commanded_q_span >= min_q_span_abs) | (commanded_speed_p95 >= min_motion_speed)

    motion_sufficient = (~active_joint_mask) | (~commanded_motion) | sufficient_q_span | sufficient_speed
    metrics = {
        "actual_q_span": np.asarray(actual_q_span, dtype=np.float64),
        "commanded_q_span": np.asarray(commanded_q_span, dtype=np.float64),
        "actual_speed_p95": np.asarray(actual_speed_p95, dtype=np.float64),
        "commanded_speed_p95": np.asarray(commanded_speed_p95, dtype=np.float64),
    }
    return motion_sufficient, metrics


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

    def build_reference(self, *, joint_index: int | None = None) -> ReferenceTrajectory:
        duration = (
            float(self.config.identification.sequential.joint_duration)
            if joint_index is not None
            else None
        )
        return self.env.build_excitation_reference(joint_index=joint_index, duration=duration)

    def supports_identification(self, mode: str) -> bool:
        return mode in {"collect", "sequential"}

    def collect_single_joint(
        self,
        *,
        joint_index: int,
        reference: ReferenceTrajectory,
        controller: FrictionIdentificationController,
        safety: SafetyGuard,
        batch_index: int,
        total_batches: int,
        group_index: int,
        total_groups: int,
    ) -> CollectedData:
        return self.collect(
            mode="sequential",
            reference=reference,
            controller=controller,
            safety=safety,
            batch_index=batch_index,
            total_batches=total_batches,
            target_joint_index=joint_index,
            group_index=group_index,
            total_groups=total_groups,
        )

    def _collect_sequential_joint(
        self,
        *,
        serial_module,
        reference: ReferenceTrajectory,
        controller: FrictionIdentificationController,
        safety: SafetyGuard,
        batch_index: int,
        total_batches: int,
        target_joint_index: int,
        group_index: int,
        total_groups: int,
    ) -> CollectedData:
        frame_packer = TorqueCommandFramePacker()
        zero_frame = frame_packer.pack(np.zeros(self.config.joint_count, dtype=np.float32))
        target_joint_index = int(target_joint_index)
        target_joint_id = target_joint_index + 1

        q = np.zeros(self.config.joint_count, dtype=np.float64)
        qd = np.zeros(self.config.joint_count, dtype=np.float64)
        tau_measured = np.zeros(self.config.joint_count, dtype=np.float64)
        mos_temp = np.zeros(self.config.joint_count, dtype=np.float64)
        coil_temp = np.full(self.config.joint_count, np.nan, dtype=np.float64)
        last_feedback_time = np.full(self.config.joint_count, np.nan, dtype=np.float64)
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
        joint_refresh_mask_log: list[np.ndarray] = []
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

        active_joint_mask = np.zeros(self.config.joint_count, dtype=bool)
        active_joint_mask[target_joint_index] = True
        active_joint_indices = np.asarray([target_joint_index], dtype=np.int64)
        active_joint_ids = [target_joint_id]
        active_window_mode = self.config.identification.excitation.window_mode
        motion_lower, motion_upper, _ = resolve_joint_window(
            self.config.robot.joint_limits,
            safety_margin=self.config.safety.joint_limit_margin,
            window_mode=active_window_mode,
        )
        valid_sample_count = 0

        reference_state = LiveReferenceState(
            excitation_reference=reference,
            sample_rate=self.config.sampling.rate,
        )
        startup_target = build_startup_pose(self.config, reference)
        startup_duration_override = float(self.config.identification.sequential.zero_position_duration)
        reference_max_step_s = (
            max(float(self.config.sampling.hardware_reference_step_factor), 1.0)
            / float(self.config.sampling.rate)
        )

        start_time = None
        last_cycle_end = None
        step_index = 0
        termination_reason = "completed"
        feedback_sweep_bytes = RECV_FRAME_SIZE * self.config.joint_count + SEND_FRAME_SIZE
        command_refresh_period_s = 1.0 / max(float(self.config.sampling.rate), 1.0)
        nominal_feedback_sweep_s = max(
            (feedback_sweep_bytes * 10.0) / max(float(self.config.serial.baudrate), 1.0),
            command_refresh_period_s,
        )
        last_command_frame = zero_frame
        allow_command_hold = False
        last_command_send_time = None
        last_feedback_wait_log_time = None
        last_sample_emit_time = None
        feedback_wait_log_grace_s = max(nominal_feedback_sweep_s, 1.0)
        frame_reader = SerialFrameReader(max_motor_id=self.config.joint_count)
        feedback_joint_indices_since_emit: list[int] = []
        target_init_announced = False

        log_info(
            "开始真机逐电机运行: "
            f"mode=sequential, batch={batch_index}/{total_batches}, "
            f"target_joint=J{target_joint_id}, group={group_index}/{total_groups}, "
            f"port={self.config.serial.port}, baudrate={self.config.serial.baudrate}"
        )
        log_info(
            "逐电机串口反馈按单电机帧解析: "
            "只要求目标关节 J"
            f"{target_joint_id} 先收到首帧；之后控制与采样按周期推进，不再区分反馈新旧。"
        )
        log_info(
            "当前串口单轮反馈扫帧 + 指令发送周期估计约 "
            f"{nominal_feedback_sweep_s * 1000.0:.1f} ms "
            f"({1.0 / nominal_feedback_sweep_s:.1f} Hz)。"
        )

        ser = None
        try:
            ser = serial_module.Serial(self.config.serial.port, self.config.serial.baudrate, timeout=0)
            ser.reset_input_buffer()
            ser.write(last_command_frame)
            last_command_send_time = time.perf_counter()
            if self.reporter is not None and hasattr(self.reporter, "set_focus_joint"):
                self.reporter.set_focus_joint(target_joint_index)

            while True:
                bytes_waiting = frame_reader.read_available(ser)
                updated_joint_indices: list[int] = []
                while True:
                    frame = frame_reader.pop_frame()
                    if frame is None:
                        break
                    joint_index = _apply_feedback_frame(
                        frame=frame,
                        joint_count=self.config.joint_count,
                        q=q,
                        qd=qd,
                        tau_measured=tau_measured,
                        mos_temp=mos_temp,
                        coil_temp=coil_temp,
                        last_feedback_time=last_feedback_time,
                    )
                    if joint_index is not None:
                        updated_joint_indices.append(joint_index)
                if updated_joint_indices:
                    feedback_joint_indices_since_emit.extend(updated_joint_indices)

                now = time.perf_counter()
                initialized_joint_ids = _initialized_feedback_joint_ids(
                    active_joint_mask=np.ones(self.config.joint_count, dtype=bool),
                    last_feedback_time=last_feedback_time,
                )
                target_joint_initialized = bool(np.isfinite(last_feedback_time[target_joint_index]))
                recent_feedback_joint_ids = [
                    joint_idx + 1 for joint_idx in _ordered_unique_joint_ids(feedback_joint_indices_since_emit)
                ]

                if target_joint_initialized and not target_init_announced:
                    log_info(
                        "目标关节首帧已收到，开始按控制周期推进: "
                        f"target=J{target_joint_id}, "
                        f"initialized={initialized_joint_ids}, "
                        f"recent_feedback={recent_feedback_joint_ids or ['none']}"
                    )
                    target_init_announced = True

                if (
                    target_joint_initialized
                    and (last_cycle_end is None or (now - last_cycle_end) >= command_refresh_period_s)
                ):
                    cycle_end = now
                    if start_time is None:
                        start_time = cycle_end
                    elapsed_s = cycle_end - start_time
                    cycle_period = (cycle_end - last_cycle_end) if last_cycle_end is not None else 0.0
                    last_cycle_end = cycle_end

                    safety.assert_joint_limits(q, window_mode=active_window_mode)
                    reference_state.initialize(
                        self.env,
                        q,
                        startup_target,
                        elapsed_s,
                        startup_duration_override=startup_duration_override,
                    )
                    reference_sample = reference_state.sample(
                        elapsed_s,
                        max_step_s=reference_max_step_s,
                    )
                    q_cmd_ref = reference_sample.q_cmd
                    qd_cmd_ref = reference_sample.qd_cmd
                    qdd_cmd_ref = reference_sample.qdd_cmd
                    phase_name = reference_sample.phase_name or "sequential"
                    tau_track_ff, tau_track_fb, tau_command = controller.compute_torque(
                        q_cmd=q_cmd_ref,
                        qd_cmd=qd_cmd_ref,
                        qdd_cmd=qdd_cmd_ref,
                        q_curr=q,
                        qd_curr=qd,
                    )
                    tau_friction_comp = np.zeros_like(q)

                    qdd_live = (
                        (qd - previous_qd) / cycle_period
                        if cycle_period > 1e-9
                        else np.zeros_like(qd)
                    )
                    previous_qd = qd.copy()
                    tau_rigid_live = self.env.inverse_dynamics(q, qd, qdd_live)
                    tau_residual = tau_measured - tau_rigid_live
                    feedback_cycle_joint_indices = _ordered_unique_joint_ids(feedback_joint_indices_since_emit)
                    feedback_cycle_joint_ids = [joint_idx + 1 for joint_idx in feedback_cycle_joint_indices]
                    joint_refresh_mask = np.zeros(self.config.joint_count, dtype=bool)
                    joint_refresh_mask[target_joint_index] = True
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
                            refreshed_mask=joint_refresh_mask[None, :],
                        )[0]
                    )
                    valid_sample_count += int(sample_valid)
                    valid_sample_ratio = valid_sample_count / max(step_index + 1, 1)
                    feedback_frames_in_sample = max(len(feedback_cycle_joint_indices), 1)
                    sample_uart_bytes = feedback_frames_in_sample * RECV_FRAME_SIZE + SEND_FRAME_SIZE

                    last_command_frame = frame_packer.pack(tau_command.astype(np.float32))
                    allow_command_hold = True
                    ser.write(last_command_frame)
                    last_command_send_time = time.perf_counter()
                    last_sample_emit_time = cycle_end
                    step_index += 1
                    last_feedback_wait_log_time = None
                    feedback_joint_indices_since_emit.clear()

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
                    joint_refresh_mask_log.append(joint_refresh_mask.copy())
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
                        (sample_uart_bytes * 8.0 / 1000.0) * uart_cycle_hz
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
                            active_joint_ids=active_joint_ids,
                            feedback_cycle_joint_ids=feedback_cycle_joint_ids,
                            ee_pos=ee_pos,
                            ee_quat=ee_quat,
                        )

                    if reference_state.is_complete():
                        termination_reason = "collection_complete"
                        break

                if termination_reason == "collection_complete":
                    break

                if last_command_send_time is None or (now - last_command_send_time) >= command_refresh_period_s:
                    command_frame = last_command_frame if allow_command_hold else zero_frame
                    ser.write(command_frame)
                    last_command_send_time = now

                sample_stream_stalled = last_sample_emit_time is None or (
                    now - float(last_sample_emit_time)
                ) >= feedback_wait_log_grace_s
                if sample_stream_stalled and (
                    last_feedback_wait_log_time is None
                    or (now - last_feedback_wait_log_time) >= 1.0
                ):
                    if not target_joint_initialized:
                        log_info(
                            "逐电机启动中，等待目标关节首帧初始化: "
                            f"target=J{target_joint_id}, "
                            f"initialized={initialized_joint_ids}, "
                            f"recent_feedback={recent_feedback_joint_ids or ['none']}"
                        )
                    else:
                        log_info(
                            "目标关节已完成初始化，后续按控制周期推进。"
                            f"target=J{target_joint_id}, "
                            f"recent_feedback={recent_feedback_joint_ids or ['none']}"
                        )
                    last_feedback_wait_log_time = now

                if bytes_waiting <= 0:
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
            mode="sequential",
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
            joint_refresh_mask=np.asarray(joint_refresh_mask_log, dtype=bool).reshape(-1, self.config.joint_count),
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
                "identification_mode": "sequential",
                "batch_index": batch_index,
                "total_batches": total_batches,
                "group_index": group_index,
                "total_groups": total_groups,
                "target_joint_index": target_joint_index,
                "fit_joint_indices": active_joint_indices.tolist(),
                "active_joint_indices": active_joint_indices.tolist(),
                "valid_sample_ratio_live": (valid_sample_count / max(step_index, 1)) if step_index > 0 else 0.0,
            },
        )

    def collect(
        self,
        *,
        mode: str,
        reference: ReferenceTrajectory | None,
        controller: FrictionIdentificationController,
        safety: SafetyGuard,
        batch_index: int = 1,
        total_batches: int = 1,
        target_joint_index: int | None = None,
        group_index: int = 1,
        total_groups: int = 1,
    ) -> CollectedData:
        if mode not in {"collect", "sequential", "compensate"}:
            raise ValueError("mode must be 'collect', 'sequential', or 'compensate'.")
        if mode == "collect" and reference is None:
            reference = self.build_reference()
        if mode == "sequential":
            if target_joint_index is None:
                raise ValueError("sequential mode requires target_joint_index.")
            if reference is None:
                reference = self.build_reference(joint_index=target_joint_index)

        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("缺少 pyserial，请先安装 requirements.txt 中的依赖。") from exc

        if mode == "sequential":
            assert reference is not None
            assert target_joint_index is not None
            return self._collect_sequential_joint(
                serial_module=serial,
                reference=reference,
                controller=controller,
                safety=safety,
                batch_index=batch_index,
                total_batches=total_batches,
                target_joint_index=target_joint_index,
                group_index=group_index,
                total_groups=total_groups,
            )

        parameters = None
        if mode == "compensate":
            if not has_compensation_results(self.config.summary_path):
                log_info("未找到历史辨识汇总，补偿模式将使用零摩擦补偿力矩。")
            parameters = load_compensation_parameters(self.config.summary_path, self.config.joint_count)

        reference_state = None
        reference_max_step_s = None
        startup_target = None
        startup_duration_override = None
        if mode in {"collect", "sequential"}:
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
            if mode == "sequential":
                startup_duration_override = float(self.config.identification.sequential.zero_position_duration)

        frame_reader = SerialFrameReader(max_motor_id=self.config.joint_count)
        frame_packer = TorqueCommandFramePacker()
        zero_frame = frame_packer.pack(np.zeros(self.config.joint_count, dtype=np.float32))

        q = np.zeros(self.config.joint_count, dtype=np.float64)
        qd = np.zeros(self.config.joint_count, dtype=np.float64)
        tau_measured = np.zeros(self.config.joint_count, dtype=np.float64)
        mos_temp = np.zeros(self.config.joint_count, dtype=np.float64)
        coil_temp = np.full(self.config.joint_count, np.nan, dtype=np.float64)
        last_feedback_time = np.full(self.config.joint_count, np.nan, dtype=np.float64)
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
        joint_refresh_mask_log: list[np.ndarray] = []
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

        active_joint_mask = self.config.active_joint_mask.copy()
        if mode == "sequential":
            active_joint_mask[:] = False
            active_joint_mask[int(target_joint_index)] = True
        active_joint_indices = np.flatnonzero(active_joint_mask)
        active_joint_ids = [int(idx) + 1 for idx in active_joint_indices]
        feedback_window = FeedbackCycleWindow(active_joint_mask=active_joint_mask)
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
        feedback_sweep_bytes = RECV_FRAME_SIZE * self.config.joint_count + SEND_FRAME_SIZE
        command_refresh_period_s = 1.0 / max(float(self.config.sampling.rate), 1.0)
        nominal_feedback_sweep_s = max(
            (feedback_sweep_bytes * 10.0) / max(float(self.config.serial.baudrate), 1.0),
            command_refresh_period_s,
        )
        feedback_stale_timeout_s = max(
            nominal_feedback_sweep_s * float(self.config.serial.feedback_stale_timeout_factor),
            0.1,
        )
        last_command_frame = zero_frame
        allow_command_hold = False
        last_command_send_time = None
        last_feedback_wait_log_time = None
        last_sample_emit_time = None
        feedback_frames_since_emit = 0
        feedback_wait_log_grace_s = max(feedback_stale_timeout_s, 1.0)

        run_label = "逐电机" if mode == "sequential" else "并行"
        log_info(
            f"开始真机{run_label}运行: "
            f"mode={mode}, batch={batch_index}/{total_batches}, "
            f"active_joints={active_joint_ids}, "
            f"port={self.config.serial.port}, baudrate={self.config.serial.baudrate}"
        )
        if mode == "sequential" and target_joint_index is not None:
            log_info(
                f"当前辨识关节: J{int(target_joint_index) + 1}, "
                f"group={int(group_index)}/{int(total_groups)}。"
            )
        else:
            log_info(
                "串口回传按单电机反馈帧逐个汇总；控制推进基于 active_joints"
                " 最近一轮有效反馈，不要求把多个电机拼成一个大帧。"
            )
        if nominal_feedback_sweep_s > (command_refresh_period_s * 1.25):
            log_info(
                "UART 理论单轮反馈扫帧 + 指令发送周期约 "
                f"{nominal_feedback_sweep_s * 1000.0:.1f} ms "
                f"({1.0 / nominal_feedback_sweep_s:.1f} Hz)，"
                f"当前 sampling.rate={self.config.sampling.rate:.1f} Hz 将以串口实际反馈节奏为准。"
            )
        log_info(
            "当前反馈新鲜窗口约 "
            f"{feedback_stale_timeout_s * 1000.0:.1f} ms "
            f"(factor={self.config.serial.feedback_stale_timeout_factor:.2f})。"
        )

        ser = None
        try:
            ser = serial.Serial(self.config.serial.port, self.config.serial.baudrate, timeout=0)
            ser.reset_input_buffer()
            ser.write(last_command_frame)
            last_command_send_time = time.perf_counter()
            if mode == "sequential" and target_joint_index is not None:
                if hasattr(self.reporter, "set_focus_joint"):
                    self.reporter.set_focus_joint(int(target_joint_index))

            while True:
                bytes_waiting = frame_reader.read_available(ser)
                emitted_sample = False
                feedback_group_ready = False
                feedback_cycle_joint_ids: list[int] = []

                while True:
                    frame = frame_reader.pop_frame()
                    if frame is None:
                        break
                    idx = _apply_feedback_frame(
                        frame=frame,
                        joint_count=self.config.joint_count,
                        q=q,
                        qd=qd,
                        tau_measured=tau_measured,
                        mos_temp=mos_temp,
                        coil_temp=coil_temp,
                        last_feedback_time=last_feedback_time,
                    )
                    if idx is None:
                        continue
                    feedback_window.push(idx)
                    if active_joint_mask[idx]:
                        feedback_frames_since_emit += 1

                    if feedback_window.is_ready():
                        feedback_group_ready = True
                        feedback_cycle_joint_ids = feedback_window.current_joint_ids()
                        break

                now = time.perf_counter()
                initialized_joint_ids = _initialized_feedback_joint_ids(
                    active_joint_mask=active_joint_mask,
                    last_feedback_time=last_feedback_time,
                )
                fresh_joint_ids = _fresh_feedback_joint_ids(
                    active_joint_mask=active_joint_mask,
                    last_feedback_time=last_feedback_time,
                    now=now,
                    stale_timeout_s=feedback_stale_timeout_s,
                )
                feedback_initialized = len(initialized_joint_ids) == len(active_joint_ids)
                feedback_fresh = len(fresh_joint_ids) == len(active_joint_ids)
                feedback_control_ready = feedback_initialized and feedback_fresh
                feedback_snapshot_ready = (
                    feedback_frames_since_emit > 0
                    and feedback_control_ready
                )
                if not feedback_cycle_joint_ids and feedback_window.is_ready():
                    feedback_cycle_joint_ids = feedback_window.current_joint_ids()
                if not feedback_cycle_joint_ids and feedback_snapshot_ready:
                    feedback_cycle_joint_ids = list(fresh_joint_ids)
                feedback_group_ready = feedback_group_ready or feedback_snapshot_ready or bool(feedback_cycle_joint_ids)
                if not feedback_control_ready:
                    allow_command_hold = False

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

                    if mode in {"collect", "sequential"}:
                        assert reference_state is not None
                        assert startup_target is not None
                        reference_state.initialize(
                            self.env,
                            q,
                            startup_target,
                            elapsed_s,
                            startup_duration_override=startup_duration_override,
                        )
                        reference_sample = reference_state.sample(
                            elapsed_s,
                            max_step_s=reference_max_step_s,
                        )
                        q_cmd_ref = reference_sample.q_cmd
                        qd_cmd_ref = reference_sample.qd_cmd
                        qdd_cmd_ref = reference_sample.qdd_cmd
                        phase_name = reference_sample.phase_name or mode
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
                    joint_refresh_mask = np.zeros(self.config.joint_count, dtype=bool)
                    if feedback_cycle_joint_ids:
                        joint_refresh_mask[np.asarray(feedback_cycle_joint_ids, dtype=np.int64) - 1] = True
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
                            refreshed_mask=joint_refresh_mask[None, :],
                        )[0]
                    )
                    valid_sample_count += int(sample_valid)
                    valid_sample_ratio = valid_sample_count / max(step_index + 1, 1)
                    feedback_frames_in_sample = max(int(feedback_frames_since_emit), 1)
                    sample_uart_bytes = feedback_frames_in_sample * RECV_FRAME_SIZE + SEND_FRAME_SIZE

                    last_command_frame = frame_packer.pack(tau_command.astype(np.float32))
                    allow_command_hold = True
                    ser.write(last_command_frame)
                    last_command_send_time = time.perf_counter()
                    last_sample_emit_time = cycle_end
                    emitted_sample = True
                    step_index += 1
                    last_feedback_wait_log_time = None
                    emitted_feedback_joint_ids = list(feedback_cycle_joint_ids)
                    feedback_window.advance_after_emit()
                    feedback_cycle_joint_ids = emitted_feedback_joint_ids
                    feedback_frames_since_emit = 0

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
                    joint_refresh_mask_log.append(joint_refresh_mask.copy())
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
                        (sample_uart_bytes * 8.0 / 1000.0) * uart_cycle_hz
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
                            active_joint_ids=active_joint_ids,
                            feedback_cycle_joint_ids=feedback_cycle_joint_ids,
                            ee_pos=ee_pos,
                            ee_quat=ee_quat,
                        )

                    if mode in {"collect", "sequential"} and reference_state.is_complete():
                        termination_reason = "collection_complete"
                        break

                if mode in {"collect", "sequential"} and termination_reason == "collection_complete":
                    break

                if not emitted_sample:
                    now = time.perf_counter()
                    if last_command_send_time is None or (now - last_command_send_time) >= command_refresh_period_s:
                        command_frame = last_command_frame if (allow_command_hold and feedback_control_ready) else zero_frame
                        ser.write(command_frame)
                        last_command_send_time = now

                    sample_stream_stalled = last_sample_emit_time is None or (
                        now - float(last_sample_emit_time)
                    ) >= feedback_wait_log_grace_s
                    if not feedback_group_ready and sample_stream_stalled and (
                        last_feedback_wait_log_time is None
                        or (now - last_feedback_wait_log_time) >= 1.0
                    ):
                        pending = feedback_window.pending_joint_ids()
                        missing = [idx for idx in active_joint_ids if idx not in initialized_joint_ids]
                        fresh_pending = [idx for idx in active_joint_ids if idx not in fresh_joint_ids]
                        stale = [
                            idx
                            for idx in fresh_pending
                            if np.isfinite(last_feedback_time[idx - 1])
                            and (now - float(last_feedback_time[idx - 1])) >= feedback_stale_timeout_s
                        ]
                        progress_count, progress_total = feedback_window.progress()
                        covered_joint_ids = feedback_window.current_joint_ids()
                        detail = (
                            f"covered={covered_joint_ids}, progress={progress_count}/{progress_total}, "
                            f"initialized={initialized_joint_ids}, fresh={fresh_joint_ids}, "
                            f"pending={pending}, missing={missing}, stale={stale}"
                        )
                        if last_sample_emit_time is None:
                            stall_detail = "启动阶段尚未形成完整样本。"
                        else:
                            stall_detail = (
                                "距离上一条已发出样本 "
                                f"{(now - float(last_sample_emit_time)) * 1000.0:.1f} ms。"
                            )
                        log_info(
                            "等待 active_joints 形成完整新鲜快照，继续发送零力矩保持安全。"
                            f"{stall_detail}{detail}"
                        )
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
            joint_refresh_mask=np.asarray(joint_refresh_mask_log, dtype=bool).reshape(-1, self.config.joint_count),
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
                "identification_mode": mode,
                "batch_index": batch_index,
                "total_batches": total_batches,
                "group_index": group_index,
                "total_groups": total_groups,
                "target_joint_index": (int(target_joint_index) if target_joint_index is not None else None),
                "fit_joint_indices": active_joint_indices.tolist(),
                "active_joint_indices": active_joint_indices.tolist(),
                "valid_sample_ratio_live": (valid_sample_count / max(step_index, 1)) if step_index > 0 else 0.0,
            },
        )

    def prepare_identification(self, data: CollectedData) -> IdentificationInputs | None:
        if data.sample_count < 16:
            log_info("真机样本过少，跳过实际数据辨识。")
            return None

        fit_joint_indices = np.asarray(
            data.metadata.get("fit_joint_indices", self.config.active_joint_indices.tolist()),
            dtype=np.int64,
        ).reshape(-1)
        if fit_joint_indices.size == 0:
            log_info("未找到可辨识关节索引，跳过实际数据辨识。")
            return None
        if np.any(fit_joint_indices < 0) or np.any(fit_joint_indices >= self.config.joint_count):
            raise ValueError("fit_joint_indices contains out-of-range joint indices.")
        fit_joint_mask = np.zeros(self.config.joint_count, dtype=bool)
        fit_joint_mask[fit_joint_indices] = True

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
        joint_refresh_mask = (
            np.asarray(data.joint_refresh_mask, dtype=bool).reshape(data.sample_count, self.config.joint_count)
            if data.joint_refresh_mask is not None
            else np.ones((data.sample_count, self.config.joint_count), dtype=bool)
        )
        motion_sufficient_mask, motion_metrics = _build_motion_sufficient_joint_mask(
            q=data.q,
            qd=qd_filtered,
            q_cmd=data.q_cmd,
            qd_cmd=data.qd_cmd,
            active_joints=fit_joint_mask,
            min_motion_speed=max(float(self.config.fitting.min_velocity_threshold), 0.01),
        )
        motion_insufficient_joint_indices = [
            int(joint_idx) for joint_idx in fit_joint_indices if not bool(motion_sufficient_mask[joint_idx])
        ]
        fit_joint_mask &= motion_sufficient_mask
        surviving_fit_joint_indices = np.flatnonzero(fit_joint_mask)
        if surviving_fit_joint_indices.size == 0:
            failing_summary = ", ".join(
                "J"
                f"{int(joint_idx) + 1}"
                f"(q_span={motion_metrics['actual_q_span'][joint_idx]:.4f}, "
                f"q_cmd_span={motion_metrics['commanded_q_span'][joint_idx]:.4f}, "
                f"qd95={motion_metrics['actual_speed_p95'][joint_idx]:.4f}, "
                f"qd_cmd95={motion_metrics['commanded_speed_p95'][joint_idx]:.4f})"
                for joint_idx in fit_joint_indices
            )
            log_info(
                "检测到真机反馈未形成足够实际运动，跳过实际数据辨识。"
                f" {failing_summary}"
            )
            data.metadata.update(
                motion_sufficient_joint_indices=[],
                motion_insufficient_joint_indices=motion_insufficient_joint_indices,
                actual_q_span=motion_metrics["actual_q_span"].tolist(),
                commanded_q_span=motion_metrics["commanded_q_span"].tolist(),
                actual_speed_p95=motion_metrics["actual_speed_p95"].tolist(),
                commanded_speed_p95=motion_metrics["commanded_speed_p95"].tolist(),
            )
            return None

        joint_clean_mask = _build_residual_clean_joint_mask(
            q=data.q,
            qd=qd_filtered,
            tau_residual=tau_residual,
            lower=lower,
            upper=upper,
            torque_limits=self.config.robot.torque_limits,
            active_joints=fit_joint_mask,
            min_motion_speed=max(float(self.config.fitting.min_velocity_threshold), 0.01),
            refreshed_mask=joint_refresh_mask,
        )
        clean_mask = np.any(joint_clean_mask[:, fit_joint_mask], axis=1)
        retained_per_joint = np.count_nonzero(joint_clean_mask, axis=0)
        active_retained = retained_per_joint[fit_joint_mask]
        if not np.any(active_retained >= 8):
            log_info("筛样后真机样本不足，跳过实际数据辨识。")
            return None
        active_ratio = active_retained / max(data.sample_count, 1)
        retained_summary = ", ".join(
            f"J{joint_idx + 1}={int(retained_per_joint[joint_idx])}"
            for joint_idx in surviving_fit_joint_indices
        )
        log_info(f"按关节筛样保留样本: {retained_summary}")

        data.qdd = qdd
        data.tau_rigid = tau_rigid
        data.tau_residual = tau_residual
        data.tau_friction = tau_residual
        data.clean_mask = clean_mask
        data.joint_clean_mask = joint_clean_mask
        data.metadata.update(
            filter_metadata,
            retained_samples=int(np.count_nonzero(clean_mask)),
            retained_samples_per_joint=retained_per_joint.tolist(),
            valid_sample_ratio=float(np.count_nonzero(clean_mask) / max(data.sample_count, 1)),
            valid_sample_ratio_per_joint=np.asarray(retained_per_joint / max(data.sample_count, 1), dtype=np.float64).tolist(),
            fit_joint_indices=surviving_fit_joint_indices.tolist(),
            motion_sufficient_joint_indices=surviving_fit_joint_indices.tolist(),
            motion_insufficient_joint_indices=motion_insufficient_joint_indices,
            actual_q_span=motion_metrics["actual_q_span"].tolist(),
            commanded_q_span=motion_metrics["commanded_q_span"].tolist(),
            actual_speed_p95=motion_metrics["actual_speed_p95"].tolist(),
            commanded_speed_p95=motion_metrics["commanded_speed_p95"].tolist(),
        )

        qd_fit = qd_filtered[:, fit_joint_mask]
        tau_fit = tau_residual[:, fit_joint_mask]
        sample_mask = joint_clean_mask[:, fit_joint_mask]
        return IdentificationInputs(
            velocity=qd_fit,
            torque=tau_fit,
            joint_names=[self.config.robot.joint_names[int(idx)] for idx in surviving_fit_joint_indices],
            clean_mask=clean_mask,
            sample_mask=sample_mask,
            metadata={
                **filter_metadata,
                "retained_samples": int(np.count_nonzero(clean_mask)),
                "retained_samples_per_joint": active_retained.tolist(),
                "valid_sample_ratio_per_joint": active_ratio.tolist(),
                "motion_sufficient_joint_indices": surviving_fit_joint_indices.tolist(),
            },
        )

    def finalize(
        self,
        data: CollectedData | None,
        result: FrictionIdentificationResult | None,
    ) -> None:
        if self.reporter is not None and result is not None and not self._summary_published:
            fit_joint_indices = (
                np.asarray(
                    data.metadata.get("fit_joint_indices", self.config.active_joint_indices.tolist()),
                    dtype=np.int64,
                ).reshape(-1)
                if data is not None
                else self.config.active_joint_indices
            )
            self.reporter.log_identification_summary(
                result,
                active_joint_names=[self.config.robot.joint_names[int(idx)] for idx in fit_joint_indices],
                active_joint_indices=fit_joint_indices.tolist(),
            )
        if self.reporter is not None:
            self.reporter.close()
            self.reporter = None
        if self.pose_estimator is not None:
            self.pose_estimator.close()
            self.pose_estimator = None
        self.env.close()

    def publish_identification_result(
        self,
        data: CollectedData,
        result: FrictionIdentificationResult,
    ) -> None:
        if self.reporter is None:
            return
        fit_joint_indices = np.asarray(
            data.metadata.get("fit_joint_indices", self.config.active_joint_indices.tolist()),
            dtype=np.int64,
        ).reshape(-1)
        self.reporter.log_identification_summary(
            result,
            active_joint_names=[self.config.robot.joint_names[int(idx)] for idx in fit_joint_indices],
            active_joint_indices=fit_joint_indices.tolist(),
        )

    def publish_summary(self, summary: IdentificationResults) -> None:
        if self.reporter is not None:
            self.reporter.log_loaded_summary(summary)
            self._summary_published = True
