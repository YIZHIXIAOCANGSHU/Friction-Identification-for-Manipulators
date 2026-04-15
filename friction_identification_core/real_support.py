from __future__ import annotations

"""Reusable real-UART helpers extracted from the CLI entry point."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .config import DEFAULT_FRICTION_CONFIG
from .estimator import fit_multijoint_friction
from .models import JointFrictionParameters
from .mujoco_driver import MujocoFrictionCollector, TrackingControlCommand
from .runtime import log_info, write_json
from .shared_logic import (
    ReferenceTrajectory,
    build_position_window_mask,
    build_residual_clean_sample_mask,
    find_joint_limit_violation,
    predict_friction_compensation_torque,
    sample_reference_trajectory,
    shrink_joint_limit_window,
)


@dataclass
class RealCollectionController:
    """保存真机采集时复用的共享参考轨迹与控制器对象。"""

    collector: MujocoFrictionCollector
    reference: ReferenceTrajectory
    reference_duration: float
    reference_sample_rate: float
    wrap_reference: bool
    time_reference: np.ndarray
    q_cmd_reference: np.ndarray
    qd_cmd_reference: np.ndarray
    qdd_cmd_reference: np.ndarray
    feedback_scale: float
    transition_max_ee_speed: float
    transition_min_duration: float
    transition_settle_duration: float
    runtime_transition_reference: ReferenceTrajectory | None = None
    runtime_transition_duration: float = 0.0
    runtime_reference_start_time: float | None = None


@dataclass
class RealControlCommand:
    """统一描述一次真机控制循环中生成出的参考量与下发力矩。"""

    tau_command: np.ndarray
    tau_feedforward: np.ndarray
    tau_feedback: np.ndarray
    raw_tau_command: np.ndarray
    blocked_mask: np.ndarray
    scale_factors: np.ndarray
    excitation_valid: bool
    q_cmd_ref: np.ndarray | None
    qd_cmd_ref: np.ndarray | None
    qdd_cmd_ref: np.ndarray | None


class SafetyLimitExceededError(RuntimeError):
    """Raised when measured joints exceed the configured safety envelope."""

    pass


SAFETY_LIMIT_MARGIN_RAD = np.deg2rad(5.0)
REAL_EXCITATION_LIMIT_MARGIN_RAD = np.deg2rad(8.0)
EXCITATION_SOFT_MARGIN_RAD = np.deg2rad(18.0)
EXCITATION_RECENTER_TORQUE_RATIO = 0.16


def load_identified_friction_parameters(summary_path: Path) -> list[JointFrictionParameters]:
    """Load previously identified friction parameters from a JSON summary."""

    if not summary_path.exists():
        raise FileNotFoundError(f"未找到辨识结果文件: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as file:
        summary = json.load(file)

    coulomb = np.asarray(summary.get("estimated_coulomb"), dtype=np.float64).reshape(-1)
    viscous = np.asarray(summary.get("estimated_viscous"), dtype=np.float64).reshape(-1)
    offset = np.asarray(summary.get("estimated_offset", [0.0] * coulomb.size), dtype=np.float64).reshape(-1)
    velocity_scale = float(summary.get("velocity_scale", DEFAULT_FRICTION_CONFIG.fit.velocity_scale))

    if coulomb.size != 7 or viscous.size != 7 or offset.size != 7:
        raise ValueError("辨识结果中的 estimated_coulomb / estimated_viscous / estimated_offset 必须都是 7 维。")

    return [
        JointFrictionParameters(
            coulomb=float(coulomb[idx]),
            viscous=float(viscous[idx]),
            offset=float(offset[idx]),
            velocity_scale=velocity_scale,
        )
        for idx in range(7)
    ]


def predict_joint_friction_torque(velocity: np.ndarray, parameters: list[JointFrictionParameters]) -> np.ndarray:
    """按统一摩擦模型计算真机补偿力矩。"""

    return predict_friction_compensation_torque(
        velocity,
        parameters,
        torque_limits=DEFAULT_FRICTION_CONFIG.model.torque_limits,
    )


def build_real_collection_controller(
    *,
    duration_s: float,
    base_frequency: float,
    amplitude_scale: float,
    feedback_scale: float,
    transition_max_ee_speed: float,
    transition_min_duration: float,
    transition_settle_duration: float,
) -> RealCollectionController:
    """复用共享激励生成逻辑，为真机采集准备参考轨迹。"""

    model_config = DEFAULT_FRICTION_CONFIG.model
    collection_config = DEFAULT_FRICTION_CONFIG.collection
    reference_duration = float(duration_s) if duration_s > 0.0 else float(collection_config.duration)
    reference_duration = max(reference_duration, 1e-3)

    collector = MujocoFrictionCollector(
        model_path=str(model_config.urdf_path),
        joint_names=list(model_config.joint_names),
        actuator_names=None,
        timestep=collection_config.timestep,
        render=False,
        home_qpos=model_config.home_qpos,
        end_effector_body=model_config.end_effector_body,
        tcp_offset=model_config.tcp_offset,
        torque_limits=model_config.torque_limits,
        joint_limit_overrides=model_config.joint_limits,
        friction_loss=model_config.friction_loss,
        damping=model_config.damping,
        inverse_friction_loss=np.zeros_like(model_config.friction_loss),
        inverse_damping=np.zeros_like(model_config.damping),
    )
    reference = collector.build_excitation_reference(
        duration=reference_duration,
        sample_rate=collection_config.sample_rate,
        base_frequency=base_frequency,
        amplitude_scale=amplitude_scale,
    )
    return RealCollectionController(
        collector=collector,
        reference=reference,
        reference_duration=reference_duration,
        reference_sample_rate=float(collection_config.sample_rate),
        wrap_reference=duration_s <= 0.0,
        time_reference=reference.time.copy(),
        q_cmd_reference=reference.q_cmd.copy(),
        qd_cmd_reference=reference.qd_cmd.copy(),
        qdd_cmd_reference=reference.qdd_cmd.copy(),
        feedback_scale=float(feedback_scale),
        transition_max_ee_speed=float(transition_max_ee_speed),
        transition_min_duration=float(transition_min_duration),
        transition_settle_duration=float(transition_settle_duration),
    )


def initialize_runtime_transition(
    controller: RealCollectionController,
    q_start: np.ndarray,
    elapsed_s: float,
) -> None:
    """首次拿到完整状态后，在线生成当前姿态到激励起始姿态的过渡段。"""

    if controller.runtime_reference_start_time is not None:
        return

    target_q = controller.q_cmd_reference[0].copy()
    transition_reference, motion_duration = controller.collector.build_transition_reference(
        start_q=q_start,
        goal_q=target_q,
        sample_rate=controller.reference_sample_rate,
        max_ee_speed=controller.transition_max_ee_speed,
        min_duration=controller.transition_min_duration,
        settle_duration=controller.transition_settle_duration,
    )
    controller.runtime_transition_reference = transition_reference
    controller.runtime_transition_duration = float(
        transition_reference.time[-1] + (1.0 / controller.reference_sample_rate)
    )
    controller.runtime_reference_start_time = float(elapsed_s)

    q_distance = float(np.linalg.norm(np.asarray(q_start, dtype=np.float64) - target_q))
    log_info(
        "采集起步过渡已初始化: "
        f"joint_distance={q_distance:.4f} rad, "
        f"motion_duration={motion_duration:.3f}s, "
        f"settle_duration={controller.transition_settle_duration:.3f}s, "
        f"target_ee_speed<={controller.transition_max_ee_speed:.3f} m/s"
    )


def sample_collection_reference(
    controller: RealCollectionController,
    elapsed_s: float,
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按当前实时时间采样共享参考轨迹。"""

    initialize_runtime_transition(controller, q_start=q, elapsed_s=elapsed_s)
    assert controller.runtime_reference_start_time is not None

    local_elapsed_s = max(float(elapsed_s) - controller.runtime_reference_start_time, 0.0)
    if (
        controller.runtime_transition_reference is not None
        and local_elapsed_s < controller.runtime_transition_duration
    ):
        return sample_reference_trajectory(
            controller.runtime_transition_reference,
            local_elapsed_s,
            wrap=False,
        )

    excitation_elapsed_s = max(local_elapsed_s - controller.runtime_transition_duration, 0.0)
    return sample_reference_trajectory(
        controller.reference,
        excitation_elapsed_s,
        wrap=controller.wrap_reference,
    )


def has_completed_startup_transition(
    controller: RealCollectionController,
    *,
    elapsed_s: float,
    q: np.ndarray,
) -> bool:
    """判断是否已经运行到预期的激励起始姿态并完成稳定保持。"""

    initialize_runtime_transition(controller, q_start=q, elapsed_s=elapsed_s)
    if controller.runtime_reference_start_time is None:
        return False

    local_elapsed_s = max(float(elapsed_s) - controller.runtime_reference_start_time, 0.0)
    return local_elapsed_s >= controller.runtime_transition_duration


def get_real_excitation_limits() -> tuple[np.ndarray, np.ndarray]:
    """给真机采集生成一组比硬限位更保守的工作窗口。"""

    model_config = DEFAULT_FRICTION_CONFIG.model
    sample_filter = DEFAULT_FRICTION_CONFIG.sample_filter
    joint_limits = np.asarray(model_config.joint_limits, dtype=np.float64)
    limit_margin = max(float(sample_filter.limit_margin), float(REAL_EXCITATION_LIMIT_MARGIN_RAD))
    lower, upper = shrink_joint_limit_window(
        joint_limits[:, 0],
        joint_limits[:, 1],
        np.ones(joint_limits.shape[0], dtype=bool),
        margin=limit_margin,
        keep_midpoint_inside=True,
    )
    return lower, upper


def build_real_excitation_valid_mask(q: np.ndarray) -> np.ndarray:
    """标记哪些样本仍处于真机保守激励工作窗内。"""

    q = np.asarray(q, dtype=np.float64)
    lower, upper = get_real_excitation_limits()
    return build_position_window_mask(q, lower=lower, upper=upper)


def check_joint_limits_or_raise(q: np.ndarray) -> None:
    """一旦越过硬安全限位就立即停机，避免继续下发力矩。"""

    violation_message = find_joint_limit_violation(
        q=q,
        joint_names=DEFAULT_FRICTION_CONFIG.model.joint_names,
        joint_limits=DEFAULT_FRICTION_CONFIG.model.joint_limits,
        margin=SAFETY_LIMIT_MARGIN_RAD,
    )
    if violation_message is None:
        return
    raise SafetyLimitExceededError(f"{violation_message} (原始限位外扩 5.0 deg)")


def maybe_build_pose_estimator(*, render: bool, viewer_fps: float):
    """Best-effort construction of the MuJoCo live pose visualizer."""

    try:
        from .real_pose_estimator import RealPoseEstimator

        model = DEFAULT_FRICTION_CONFIG.model
        return RealPoseEstimator(
            model_path=str(model.urdf_path),
            joint_names=list(model.joint_names),
            end_effector_body=model.end_effector_body,
            tcp_offset=model.tcp_offset,
            render=render,
            viewer_fps=viewer_fps,
        )
    except Exception as exc:
        if render:
            log_info(f"MuJoCo 可视化初始化失败，将仅记录关节与力矩数据: {exc}")
        else:
            log_info(f"MuJoCo 位姿估计不可用，将仅记录关节与力矩数据: {exc}")
        return None


def build_real_clean_sample_mask(q: np.ndarray, tau_residual: np.ndarray) -> np.ndarray:
    """按共享筛样逻辑保留可用于真机摩擦拟合的样本。"""

    model_config = DEFAULT_FRICTION_CONFIG.model
    lower, upper = get_real_excitation_limits()
    return build_residual_clean_sample_mask(
        q=q,
        tau_residual=tau_residual,
        lower=lower,
        upper=upper,
        torque_limits=model_config.torque_limits,
        torque_limit_scale=1.5,
    )


def compute_collect_bias_compensation_torque(
    *,
    q: np.ndarray,
    qd: np.ndarray,
    collection_controller: RealCollectionController,
) -> np.ndarray:
    """Compute gravity/Coriolis bias compensation for collect mode without the excitation term."""

    torque_limits = DEFAULT_FRICTION_CONFIG.model.torque_limits
    tau_bias = collection_controller.collector.compute_bias_torque(q_curr=q, qd_curr=qd)
    return np.clip(tau_bias, -torque_limits, torque_limits)


def compute_realtime_control_command(
    *,
    control_mode: str,
    q: np.ndarray,
    qd: np.ndarray,
    elapsed_s: float,
    parameters: list[JointFrictionParameters] | None,
    collection_controller: RealCollectionController | None,
) -> RealControlCommand:
    """将真机循环中的控制计算收口为单一函数，入口层只负责 I/O。"""

    if control_mode == "compensate":
        if parameters is None:
            raise ValueError("parameters are required in compensate mode.")
        tau_command = predict_joint_friction_torque(qd, parameters)
        return RealControlCommand(
            tau_command=tau_command,
            tau_feedforward=np.zeros_like(q),
            tau_feedback=np.zeros_like(q),
            raw_tau_command=tau_command.copy(),
            blocked_mask=np.zeros_like(q, dtype=bool),
            scale_factors=np.ones_like(q),
            excitation_valid=True,
            q_cmd_ref=None,
            qd_cmd_ref=None,
            qdd_cmd_ref=None,
        )

    if collection_controller is None:
        raise ValueError("collection_controller is required in collect mode.")

    q_cmd_ref, qd_cmd_ref, qdd_cmd_ref = sample_collection_reference(collection_controller, elapsed_s, q)
    lower, upper = get_real_excitation_limits()
    tracking_command: TrackingControlCommand = collection_controller.collector.compute_safe_tracking_command(
        q_cmd=q_cmd_ref,
        qd_cmd=qd_cmd_ref,
        qdd_cmd=qdd_cmd_ref,
        q_curr=q,
        qd_curr=qd,
        feedback_scale=collection_controller.feedback_scale,
        lower=lower,
        upper=upper,
        soft_margin=EXCITATION_SOFT_MARGIN_RAD,
        recenter_torque_ratio=EXCITATION_RECENTER_TORQUE_RATIO,
    )
    excitation_valid = bool(build_real_excitation_valid_mask(q.reshape(1, -1))[0])
    return RealControlCommand(
        tau_command=tracking_command.tau_command,
        tau_feedforward=tracking_command.tau_feedforward,
        tau_feedback=tracking_command.tau_feedback,
        raw_tau_command=tracking_command.raw_tau_command,
        blocked_mask=tracking_command.blocked_mask,
        scale_factors=tracking_command.scale_factors,
        excitation_valid=excitation_valid,
        q_cmd_ref=q_cmd_ref,
        qd_cmd_ref=qd_cmd_ref,
        qdd_cmd_ref=qdd_cmd_ref,
    )


def identify_real_friction_from_capture(
    *,
    time_s: np.ndarray,
    q: np.ndarray,
    qd: np.ndarray,
    tau_measured: np.ndarray,
    output_dir: Path,
    capture_prefix: str,
    output_prefix: str,
    summary_path: Path,
) -> None:
    """Fit friction parameters from saved real UART capture data."""

    if time_s.size < 32:
        log_info("有效真机样本太少，跳过实际数据辨识。")
        return

    from .real_dynamics_estimator import RealDynamicsEstimator

    model_config = DEFAULT_FRICTION_CONFIG.model
    fit_config = DEFAULT_FRICTION_CONFIG.fit
    sample_filter = DEFAULT_FRICTION_CONFIG.sample_filter

    gradient_order = 2 if time_s.size >= 3 else 1
    qdd = np.gradient(qd, time_s, axis=0, edge_order=gradient_order)
    rigid_estimator = RealDynamicsEstimator(
        model_path=str(model_config.urdf_path),
        joint_names=list(model_config.joint_names),
        tcp_offset=model_config.tcp_offset,
    )
    tau_rigid = rigid_estimator.batch_inverse_dynamics(q, qd, qdd)
    tau_friction = tau_measured - tau_rigid

    clean_mask = build_real_clean_sample_mask(q, tau_friction)
    retained = int(np.count_nonzero(clean_mask))
    if retained < 32:
        log_info("筛选后真机样本不足，跳过实际数据辨识。")
        return

    q_clean = q[clean_mask]
    qd_clean = qd[clean_mask]
    tau_measured_clean = tau_measured[clean_mask]
    tau_rigid_clean = tau_rigid[clean_mask]
    tau_friction_clean = tau_friction[clean_mask]
    time_clean = time_s[clean_mask]

    validation_mask = sample_filter.build_validation_mask(time_clean.shape[0])
    result = fit_multijoint_friction(
        velocity=qd_clean,
        torque=tau_friction_clean,
        joint_names=model_config.joint_names,
        validation_mask=validation_mask,
        velocity_scale=fit_config.velocity_scale,
        regularization=fit_config.regularization,
        max_iterations=fit_config.max_iterations,
        huber_delta=fit_config.huber_delta,
        min_velocity_threshold=fit_config.min_velocity_threshold,
        progress_callback=lambda current, total, joint_name: log_info(
            f"Fitting real joint {current}/{total}: {joint_name}"
        ),
    )

    estimated_coulomb = np.array([params.coulomb for params in result.parameters], dtype=np.float64)
    estimated_viscous = np.array([params.viscous for params in result.parameters], dtype=np.float64)
    estimated_offset = np.array([params.offset for params in result.parameters], dtype=np.float64)

    npz_path = output_dir / f"{output_prefix}.npz"
    json_path = output_dir / f"{output_prefix}.json"
    np.savez(
        npz_path,
        time=time_s,
        q=q,
        qd=qd,
        qdd=qdd,
        tau_measured=tau_measured,
        tau_rigid=tau_rigid,
        tau_friction_est=tau_friction,
        clean_mask=clean_mask,
        time_clean=time_clean,
        q_clean=q_clean,
        qd_clean=qd_clean,
        tau_measured_clean=tau_measured_clean,
        tau_rigid_clean=tau_rigid_clean,
        tau_friction_clean=tau_friction_clean,
        tau_pred=result.predicted_torque,
        train_mask=result.train_mask,
        validation_mask=result.validation_mask,
    )

    summary = {
        "source_capture": str(output_dir / f"{capture_prefix}.npz"),
        "sample_count": int(time_s.shape[0]),
        "retained_samples": retained,
        "joint_names": list(model_config.joint_names),
        "joint_limits_rad": model_config.joint_limits.tolist(),
        "excitation_filter_limits_rad": np.column_stack(get_real_excitation_limits()).tolist(),
        "estimated_coulomb": estimated_coulomb.tolist(),
        "estimated_viscous": estimated_viscous.tolist(),
        "estimated_offset": estimated_offset.tolist(),
        "velocity_scale": float(fit_config.velocity_scale),
        "train_rmse": result.train_rmse.tolist(),
        "validation_rmse": result.validation_rmse.tolist(),
        "train_r2": result.train_r2.tolist(),
        "validation_r2": result.validation_r2.tolist(),
        "mean_validation_rmse": float(np.nanmean(result.validation_rmse)),
        "mean_validation_r2": float(np.nanmean(result.validation_r2)),
        "tau_friction_estimation_note": "tau_friction_est = tau_measured - rigid_body_inverse_dynamics(q, qd, qdd)",
    }
    write_json(json_path, summary)
    write_json(summary_path, summary)

    log_info(f"真机辨识结果已保存: {npz_path}")
    log_info(f"真机辨识摘要已保存: {json_path}")
    log_info(f"真机辨识摘要同步写入: {summary_path}")
    for joint_name, params, rmse in zip(result.joint_names, result.parameters, result.validation_rmse):
        log_info(
            f"{joint_name}: fc={params.coulomb:.6f}, fv={params.viscous:.6f}, "
            f"offset={params.offset:.6f}, val_rmse={rmse:.6f}"
        )


def save_capture(
    *,
    output_dir: Path,
    output_prefix: str,
    control_mode: str,
    control_output_mode: str,
    send_enabled: bool,
    send_bias_compensation_only: bool,
    time_log: list[float],
    q_log: list[np.ndarray],
    qd_log: list[np.ndarray],
    tau_measured_log: list[np.ndarray],
    tau_command_log: list[np.ndarray],
    excitation_valid_log: list[bool],
    excitation_blocked_log: list[np.ndarray],
    mos_temp_log: list[np.ndarray],
    coil_temp_log: list[np.ndarray],
    ee_pos_log: list[np.ndarray],
    ee_quat_log: list[np.ndarray],
    uart_cycle_hz_log: list[float],
    uart_latency_ms_log: list[float],
    uart_transfer_kbps_log: list[float],
    collection_controller: str,
    collection_base_frequency: float,
    collection_amplitude_scale: float,
    collection_feedback_scale: float,
    termination_reason: str,
    stop_at_excitation_start: bool,
) -> None:
    """Persist raw UART capture arrays together with a compact JSON summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{output_prefix}.npz"
    json_path = output_dir / f"{output_prefix}.json"

    q = np.asarray(q_log, dtype=np.float64).reshape(-1, 7)
    qd = np.asarray(qd_log, dtype=np.float64).reshape(-1, 7)
    tau_measured = np.asarray(tau_measured_log, dtype=np.float64).reshape(-1, 7)
    tau_command = np.asarray(tau_command_log, dtype=np.float64).reshape(-1, 7)
    excitation_valid = np.asarray(excitation_valid_log, dtype=bool).reshape(-1)
    excitation_blocked = np.asarray(excitation_blocked_log, dtype=bool).reshape(-1, 7)
    mos_temp = np.asarray(mos_temp_log, dtype=np.float64).reshape(-1, 7)
    coil_temp = np.asarray(coil_temp_log, dtype=np.float64).reshape(-1, 7)

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
        tau_measured=tau_measured,
        tau_command=tau_command,
        excitation_valid=excitation_valid,
        excitation_blocked=excitation_blocked,
        mos_temperature=mos_temp,
        coil_temperature=coil_temp,
        ee_pos=ee_pos,
        ee_quat=ee_quat,
        uart_cycle_hz=np.asarray(uart_cycle_hz_log, dtype=np.float64),
        uart_latency_ms=np.asarray(uart_latency_ms_log, dtype=np.float64),
        uart_transfer_kbps=np.asarray(uart_transfer_kbps_log, dtype=np.float64),
    )

    summary = {
        "control_mode": control_mode,
        "control_output_mode": control_output_mode,
        "send_enabled": bool(send_enabled),
        "send_bias_compensation_only": bool(send_bias_compensation_only),
        "collection_controller": collection_controller,
        "collection_base_frequency": float(collection_base_frequency),
        "collection_amplitude_scale": float(collection_amplitude_scale),
        "collection_feedback_scale": float(collection_feedback_scale),
        "sample_count": int(len(time_log)),
        "duration_s": float(time_log[-1]) if time_log else 0.0,
        "termination_reason": termination_reason,
        "stop_at_excitation_start": bool(stop_at_excitation_start),
        "excitation_filter_limits_rad": np.column_stack(get_real_excitation_limits()).tolist(),
        "excitation_valid_samples": int(np.count_nonzero(excitation_valid)),
        "excitation_blocked_samples": int(np.count_nonzero(np.any(excitation_blocked, axis=1))),
        "mean_uart_cycle_hz": float(np.mean(uart_cycle_hz_log)) if uart_cycle_hz_log else 0.0,
        "max_uart_cycle_hz": float(np.max(uart_cycle_hz_log)) if uart_cycle_hz_log else 0.0,
        "mean_uart_latency_ms": float(np.mean(uart_latency_ms_log)) if uart_latency_ms_log else 0.0,
        "mean_uart_transfer_kbps": float(np.mean(uart_transfer_kbps_log)) if uart_transfer_kbps_log else 0.0,
    }
    write_json(json_path, summary)

    log_info(f"采集结果已保存: {npz_path}")
    log_info(f"采集摘要已保存: {json_path}")
