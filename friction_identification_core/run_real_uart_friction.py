#!/usr/bin/env python3

from __future__ import annotations

"""真机 UART 入口层，仅保留串口/可视化/保存等外设侧差异。"""

import argparse
from dataclasses import dataclass
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from friction_identification_core.config import DEFAULT_FRICTION_CONFIG
from friction_identification_core.estimator import fit_multijoint_friction
from friction_identification_core.models import JointFrictionParameters
from friction_identification_core.mujoco_driver import MujocoFrictionCollector
from friction_identification_core.real_serial_protocol import (
    RECV_FRAME_SIZE,
    SEND_FRAME_SIZE,
    SerialFrameReader,
    TorqueCommandFramePacker,
)
from friction_identification_core.shared_logic import (
    ReferenceTrajectory,
    build_position_window_mask,
    build_residual_clean_sample_mask,
    find_joint_limit_violation,
    predict_friction_compensation_torque,
    sample_reference_trajectory,
    shape_limit_aware_torque_command,
    shrink_joint_limit_window,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI options for real robot collection or compensation mode."""

    parser = argparse.ArgumentParser(description="Run real UART friction-compensation forwarding for AM-D02.")
    parser.add_argument(
        "--control-mode",
        choices=("collect", "compensate"),
        default="collect",
        help="`collect` sends excitation torques and identifies from real data; `compensate` sends identified friction compensation.",
    )
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Serial port for the lower controller.")
    parser.add_argument("--baudrate", type=int, default=115200, help="UART baudrate.")
    parser.add_argument("--duration", type=float, default=42.0, help="Run duration in seconds, 0 means until Ctrl+C.")
    parser.add_argument(
        "--render",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open the MuJoCo viewer and drive it from the received motor state.",
    )
    parser.add_argument(
        "--spawn-rerun",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Spawn a local Rerun viewer.",
    )
    parser.add_argument(
        "--summary-path",
        default=str(PROJECT_ROOT / "results" / "real_friction_identification_summary.json"),
        help="Summary JSON path used by `compensate` mode and written by `collect` identification output.",
    )
    parser.add_argument(
        "--output-prefix",
        default="real_uart_capture",
        help="Basename for saved capture files under results/.",
    )
    parser.add_argument(
        "--ident-output-prefix",
        default="real_friction_identification",
        help="Basename for identified real-data friction result files under results/.",
    )
    parser.add_argument(
        "--rerun-log-stride",
        type=int,
        default=1,
        help="Log every N completed 7-axis UART cycles to Rerun.",
    )
    parser.add_argument(
        "--uart-text-log-interval",
        type=int,
        default=100,
        help="Emit RX/TX UART text log to Rerun every N cycles.",
    )
    parser.add_argument(
        "--serial-idle-sleep",
        type=float,
        default=0.0005,
        help="Sleep duration when UART is idle, in seconds.",
    )
    parser.add_argument(
        "--viewer-fps",
        type=float,
        default=30.0,
        help="Maximum MuJoCo viewer refresh rate in real mode.",
    )
    parser.add_argument(
        "--collection-base-frequency",
        type=float,
        default=DEFAULT_FRICTION_CONFIG.collection.base_frequency,
        help="Base frequency in Hz for the shared MuJoCo excitation trajectory.",
    )
    parser.add_argument(
        "--collection-amplitude-scale",
        type=float,
        default=DEFAULT_FRICTION_CONFIG.collection.amplitude_scale,
        help="Trajectory amplitude scale shared with the MuJoCo collector.",
    )
    parser.add_argument(
        "--collection-feedback-scale",
        type=float,
        default=DEFAULT_FRICTION_CONFIG.collection.feedback_scale,
        help="PD feedback mix shared with the MuJoCo collector.",
    )
    parser.add_argument(
        "--tx-debug-frames",
        type=int,
        default=3,
        help="Print the first N transmitted UART torque frames for verification.",
    )
    return parser.parse_args()


def log_info(message: str) -> None:
    """Emit a flushed info log line."""

    print(f"[INFO] {message}", flush=True)


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
    )


def sample_collection_reference(
    controller: RealCollectionController,
    elapsed_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按当前实时时间采样共享参考轨迹。"""

    return sample_reference_trajectory(
        controller.reference,
        elapsed_s,
        wrap=controller.wrap_reference,
    )


class SafetyLimitExceededError(RuntimeError):
    """Raised when measured joints exceed the configured safety envelope."""

    pass


SAFETY_LIMIT_MARGIN_RAD = np.deg2rad(5.0)
REAL_EXCITATION_LIMIT_MARGIN_RAD = np.deg2rad(8.0)
EXCITATION_SOFT_MARGIN_RAD = np.deg2rad(18.0)
EXCITATION_RECENTER_TORQUE_RATIO = 0.16


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


def shape_excitation_command_by_real_limits(
    q: np.ndarray,
    torque_command: np.ndarray,
    torque_limits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """复用共享限位整形逻辑，对真机采集力矩做安全处理。"""

    lower, upper = get_real_excitation_limits()
    return shape_limit_aware_torque_command(
        q=q,
        torque_command=torque_command,
        torque_limits=torque_limits,
        lower=lower,
        upper=upper,
        soft_margin=EXCITATION_SOFT_MARGIN_RAD,
        recenter_torque_ratio=EXCITATION_RECENTER_TORQUE_RATIO,
    )


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
        from friction_identification_core.real_pose_estimator import RealPoseEstimator

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

    torque_limits = DEFAULT_FRICTION_CONFIG.model.torque_limits
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

    q_cmd_ref, qd_cmd_ref, qdd_cmd_ref = sample_collection_reference(collection_controller, elapsed_s)
    tau_feedforward, tau_feedback, raw_tau_command = collection_controller.collector.compute_tracking_torque(
        q_cmd=q_cmd_ref,
        qd_cmd=qd_cmd_ref,
        qdd_cmd=qdd_cmd_ref,
        q_curr=q,
        qd_curr=qd,
        feedback_scale=collection_controller.feedback_scale,
    )
    raw_tau_command = np.clip(raw_tau_command, -torque_limits, torque_limits)
    tau_command, blocked_mask, scale_factors = shape_excitation_command_by_real_limits(
        q,
        raw_tau_command,
        torque_limits,
    )
    excitation_valid = bool(build_real_excitation_valid_mask(q.reshape(1, -1))[0])
    return RealControlCommand(
        tau_command=tau_command,
        tau_feedforward=tau_feedforward,
        tau_feedback=tau_feedback,
        raw_tau_command=raw_tau_command,
        blocked_mask=blocked_mask,
        scale_factors=scale_factors,
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

    from friction_identification_core.real_dynamics_estimator import RealDynamicsEstimator

    model_config = DEFAULT_FRICTION_CONFIG.model
    fit_config = DEFAULT_FRICTION_CONFIG.fit
    sample_filter = DEFAULT_FRICTION_CONFIG.sample_filter

    # Reconstruct acceleration numerically, then subtract rigid-body inverse dynamics.
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

    # Only fit on the cleaned subset, but save both raw and filtered arrays.
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
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

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
        # Rendering may be disabled, so keep array shapes stable with neutral pose defaults.
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
        "collection_controller": collection_controller,
        "collection_base_frequency": float(collection_base_frequency),
        "collection_amplitude_scale": float(collection_amplitude_scale),
        "collection_feedback_scale": float(collection_feedback_scale),
        "sample_count": int(len(time_log)),
        "duration_s": float(time_log[-1]) if time_log else 0.0,
        "termination_reason": termination_reason,
        "excitation_filter_limits_rad": np.column_stack(get_real_excitation_limits()).tolist(),
        "excitation_valid_samples": int(np.count_nonzero(excitation_valid)),
        "excitation_blocked_samples": int(np.count_nonzero(np.any(excitation_blocked, axis=1))),
        "mean_uart_cycle_hz": float(np.mean(uart_cycle_hz_log)) if uart_cycle_hz_log else 0.0,
        "max_uart_cycle_hz": float(np.max(uart_cycle_hz_log)) if uart_cycle_hz_log else 0.0,
        "mean_uart_latency_ms": float(np.mean(uart_latency_ms_log)) if uart_latency_ms_log else 0.0,
        "mean_uart_transfer_kbps": float(np.mean(uart_transfer_kbps_log)) if uart_transfer_kbps_log else 0.0,
    }
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    log_info(f"采集结果已保存: {npz_path}")
    log_info(f"采集摘要已保存: {json_path}")


def main() -> None:
    """Run the realtime UART loop, then save capture and optional fit outputs."""

    args = parse_args()
    summary_path = Path(args.summary_path).resolve()
    output_dir = PROJECT_ROOT / "results"
    parameters = None
    collection_controller = None
    if args.control_mode == "compensate":
        # Compensation mode only replays previously identified friction parameters.
        parameters = load_identified_friction_parameters(summary_path)
        log_info(f"已加载辨识参数: {summary_path}")
        for idx, param in enumerate(parameters, start=1):
            log_info(
                f"J{idx}: fc={param.coulomb:.6f}, fv={param.viscous:.6f}, "
                f"offset={param.offset:.6f}, v_scale={param.velocity_scale:.6f}"
            )
    else:
        # Collection mode synthesizes excitation torques online and later fits new parameters.
        log_info("当前运行模式: collect，将下发采集激励力矩并基于真机数据做辨识。")

    try:
        import serial
    except ImportError as exc:
        raise RuntimeError("缺少 pyserial，请先安装 requirements.txt 中的依赖。") from exc

    try:
        ser = serial.Serial(args.port, args.baudrate, timeout=0)
        ser.reset_input_buffer()
    except Exception as exc:
        raise RuntimeError(f"无法打开串口 {args.port}: {exc}") from exc

    log_info(f"串口已连接: {args.port} @ {args.baudrate}")
    log_info(
        f"UART 接收帧长={RECV_FRAME_SIZE} bytes, 力矩发送帧长={SEND_FRAME_SIZE} bytes, "
        "发送协议=dm_motor_uart_rx_frame_t(mode1 torque[7])"
    )
    if args.control_mode == "collect":
        log_info(
            "采集控制配置: 复用 MuJoCo 轨迹与控制律, "
            f"base_frequency={args.collection_base_frequency:.3f}Hz, "
            f"amplitude_scale={args.collection_amplitude_scale:.3f}, "
            f"feedback_scale={args.collection_feedback_scale:.3f}"
        )
        collection_controller = build_real_collection_controller(
            duration_s=args.duration,
            base_frequency=args.collection_base_frequency,
            amplitude_scale=args.collection_amplitude_scale,
            feedback_scale=args.collection_feedback_scale,
        )
        log_info(
            "采集模式已切换为 MuJoCo 参考轨迹跟踪: "
            f"reference_duration={collection_controller.reference_duration:.3f}s, "
            f"reference_sample_rate={collection_controller.reference_sample_rate:.1f}Hz"
        )

    reporter = None
    if args.spawn_rerun:
        from friction_identification_core.real_rerun_reporter import RealTimeRerunReporter

        reporter = RealTimeRerunReporter(
            app_name="AM-D02 Real UART Friction Compensation",
            joint_names=DEFAULT_FRICTION_CONFIG.model.joint_names,
            spawn=True,
        )
        reporter.init()

    pose_estimator = maybe_build_pose_estimator(render=args.render, viewer_fps=args.viewer_fps)
    if args.render:
        if pose_estimator is not None:
            log_info("MuJoCo 仿真窗口已启动，将使用接收的电机状态实时驱动机械臂。")
        else:
            log_info("MuJoCo 仿真窗口未启动，程序继续执行串口采集与力矩下发。")

    frame_reader = SerialFrameReader()
    frame_packer = TorqueCommandFramePacker()
    zero_torque_frame = frame_packer.pack(np.zeros(7, dtype=np.float32))

    q = np.zeros(7, dtype=np.float64)
    qd = np.zeros(7, dtype=np.float64)
    tau_measured = np.zeros(7, dtype=np.float64)
    mos_temp = np.zeros(7, dtype=np.float64)
    coil_temp = np.zeros(7, dtype=np.float64)
    complete_feedback_mask = (1 << 7) - 1
    feedback_mask = 0
    bytes_per_cycle = RECV_FRAME_SIZE * 7 + SEND_FRAME_SIZE

    time_log: list[float] = []
    q_log: list[np.ndarray] = []
    qd_log: list[np.ndarray] = []
    tau_measured_log: list[np.ndarray] = []
    tau_command_log: list[np.ndarray] = []
    excitation_valid_log: list[bool] = []
    excitation_blocked_log: list[np.ndarray] = []
    mos_temp_log: list[np.ndarray] = []
    coil_temp_log: list[np.ndarray] = []
    ee_pos_log: list[np.ndarray] = []
    ee_quat_log: list[np.ndarray] = []
    uart_cycle_hz_log: list[float] = []
    uart_latency_ms_log: list[float] = []
    uart_transfer_kbps_log: list[float] = []

    start_time = time.perf_counter()
    last_cycle_end = None
    step_index = 0
    termination_reason = "completed"
    safety_error_message = None
    tx_debug_remaining = max(int(args.tx_debug_frames), 0)
    excitation_filter_log_remaining = 5
    control_log_remaining = 5

    try:
        while True:
            if args.duration > 0.0 and (time.perf_counter() - start_time) >= args.duration:
                termination_reason = "duration_reached"
                break

            bytes_waiting = frame_reader.read_available(ser)
            emitted_sample = False

            while True:
                frame = frame_reader.pop_frame()
                if frame is None:
                    break

                if not 1 <= frame.motor_id <= 7:
                    continue

                joint_idx = frame.motor_id - 1
                q[joint_idx] = frame.position
                qd[joint_idx] = frame.velocity
                tau_measured[joint_idx] = frame.torque
                mos_temp[joint_idx] = frame.mos_temperature
                coil_temp[joint_idx] = frame.coil_temperature
                feedback_mask |= 1 << joint_idx

                if feedback_mask != complete_feedback_mask:
                    continue

                # Only compute and send a torque command after all 7 motor updates arrive.
                feedback_mask = 0
                emitted_sample = True
                cycle_end = time.perf_counter()
                elapsed_s = cycle_end - start_time
                check_joint_limits_or_raise(q)
                command = compute_realtime_control_command(
                    control_mode=args.control_mode,
                    q=q,
                    qd=qd,
                    elapsed_s=elapsed_s,
                    parameters=parameters,
                    collection_controller=collection_controller,
                )
                tau_command = command.tau_command
                tau_feedforward = command.tau_feedforward
                blocked_mask = command.blocked_mask
                excitation_valid = command.excitation_valid
                q_cmd_ref = command.q_cmd_ref
                if args.control_mode == "collect":
                    assert command.qd_cmd_ref is not None
                    if control_log_remaining > 0:
                        log_info(
                            "MuJoCo 采集控制: "
                            f"q_cmd={[round(float(x), 4) for x in command.q_cmd_ref]}, "
                            f"qd_cmd={[round(float(x), 4) for x in command.qd_cmd_ref]}, "
                            f"tau_ff={[round(float(x), 4) for x in command.tau_feedforward]}, "
                            f"tau_fb={[round(float(x), 4) for x in command.tau_feedback]}, "
                            f"tau_ctrl={[round(float(x), 4) for x in command.raw_tau_command]}"
                        )
                        control_log_remaining -= 1
                    if np.any(command.blocked_mask) and excitation_filter_log_remaining > 0:
                        lower, upper = get_real_excitation_limits()
                        log_info(
                            "力矩整形: 当前回传位置靠近真实限位，已衰减或改写朝外的总下发力矩，避免继续推向限位。"
                            f" q={[round(float(x), 4) for x in q]},"
                            f" q_cmd={[round(float(x), 4) for x in command.q_cmd_ref]},"
                            f" feedforward_tau={[round(float(x), 4) for x in command.tau_feedforward]},"
                            f" raw_total_tau={[round(float(x), 4) for x in command.raw_tau_command]},"
                            f" shaped_tau={[round(float(x), 4) for x in command.tau_command]},"
                            f" scale={[round(float(x), 3) for x in command.scale_factors]},"
                            f" blocked_joints={[idx + 1 for idx, flag in enumerate(command.blocked_mask) if flag]},"
                            f" valid_range_low={[round(float(x), 4) for x in lower]},"
                            f" valid_range_high={[round(float(x), 4) for x in upper]}"
                        )
                        excitation_filter_log_remaining -= 1
                tx_frame = frame_packer.pack(tau_command)
                ser.write(tx_frame)
                if tx_debug_remaining > 0:
                    log_info(
                        "TX frame debug: torque=["
                        + ", ".join(f"{value:.5f}" for value in tau_command)
                        + f"], hex={tx_frame.hex()}"
                    )
                    tx_debug_remaining -= 1

                uart_latency_ms = 0.0
                uart_cycle_hz = 0.0
                uart_transfer_kbps = 0.0
                if last_cycle_end is not None:
                    cycle_dt = cycle_end - last_cycle_end
                    if cycle_dt > 0.0:
                        uart_latency_ms = cycle_dt * 1000.0
                        uart_cycle_hz = 1.0 / cycle_dt
                        uart_transfer_kbps = (bytes_per_cycle * 8.0) / cycle_dt / 1000.0
                last_cycle_end = cycle_end

                ee_pos = None
                ee_quat = None
                if pose_estimator is not None:
                    ee_pos, ee_quat = pose_estimator.update(q)
                    ee_pos_log.append(np.asarray(ee_pos, dtype=np.float64))
                    ee_quat_log.append(np.asarray(ee_quat, dtype=np.float64))

                time_log.append(float(elapsed_s))
                q_log.append(q.copy())
                qd_log.append(qd.copy())
                tau_measured_log.append(tau_measured.copy())
                tau_command_log.append(tau_command.copy())
                excitation_valid_log.append(excitation_valid)
                excitation_blocked_log.append(blocked_mask.copy())
                mos_temp_log.append(mos_temp.copy())
                coil_temp_log.append(coil_temp.copy())
                uart_cycle_hz_log.append(float(uart_cycle_hz))
                uart_latency_ms_log.append(float(uart_latency_ms))
                uart_transfer_kbps_log.append(float(uart_transfer_kbps))

                if reporter is not None and (
                    args.rerun_log_stride <= 1 or step_index % args.rerun_log_stride == 0
                ):
                    rx_text = None
                    tx_text = None
                    if args.uart_text_log_interval <= 1 or step_index % args.uart_text_log_interval == 0:
                        rx_text = "q=[" + ", ".join(f"{value:.4f}" for value in q) + "]"
                        tx_text = "torque=[" + ", ".join(f"{value:.4f}" for value in tau_command) + "]"

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
                        rx_text=rx_text,
                        tx_text=tx_text,
                    )

                if step_index % 100 == 0:
                    step_log = (
                        f"step={step_index}, uart={uart_cycle_hz:.2f} Hz, "
                        f"q1={q[0]:.4f}, qd1={qd[0]:.4f}, tau_cmd1={tau_command[0]:.4f}"
                    )
                    if args.control_mode == "collect" and q_cmd_ref is not None:
                        step_log += f", q_cmd1={q_cmd_ref[0]:.4f}, tau_ff1={tau_feedforward[0]:.4f}"
                    log_info(step_log)
                step_index += 1

            if bytes_waiting == 0 and not emitted_sample and not frame_reader.has_complete_frame():
                # Back off slightly when UART is idle so the loop does not spin at 100% CPU.
                time.sleep(max(args.serial_idle_sleep, 0.0))

    except KeyboardInterrupt:
        termination_reason = "keyboard_interrupt"
        log_info("收到 Ctrl+C，准备保存实时采集结果。")
    except SafetyLimitExceededError as exc:
        termination_reason = f"safety_stop: {exc}"
        log_info(str(exc))
        try:
            ser.write(zero_torque_frame)
        except Exception:
            pass
        safety_error_message = str(exc)
    finally:
        try:
            ser.close()
        except Exception:
            pass
        if pose_estimator is not None:
            pose_estimator.close()
        if collection_controller is not None:
            collection_controller.collector.close()
        save_capture(
            output_dir=output_dir,
            output_prefix=args.output_prefix,
            control_mode=args.control_mode,
            time_log=time_log,
            q_log=q_log,
            qd_log=qd_log,
            tau_measured_log=tau_measured_log,
            tau_command_log=tau_command_log,
            excitation_valid_log=excitation_valid_log,
            excitation_blocked_log=excitation_blocked_log,
            mos_temp_log=mos_temp_log,
            coil_temp_log=coil_temp_log,
            ee_pos_log=ee_pos_log,
            ee_quat_log=ee_quat_log,
            uart_cycle_hz_log=uart_cycle_hz_log,
            uart_latency_ms_log=uart_latency_ms_log,
            uart_transfer_kbps_log=uart_transfer_kbps_log,
            collection_controller="mujoco_reference_tracking" if args.control_mode == "collect" else "compensate_only",
            collection_base_frequency=args.collection_base_frequency if args.control_mode == "collect" else 0.0,
            collection_amplitude_scale=args.collection_amplitude_scale if args.control_mode == "collect" else 0.0,
            collection_feedback_scale=args.collection_feedback_scale if args.control_mode == "collect" else 0.0,
            termination_reason=termination_reason,
        )
        if reporter is not None:
            reporter.close()

    if args.control_mode == "collect" and time_log:
        # Run identification after capture so the saved summary can be reused in compensate mode.
        identify_real_friction_from_capture(
            time_s=np.asarray(time_log, dtype=np.float64),
            q=np.asarray(q_log, dtype=np.float64),
            qd=np.asarray(qd_log, dtype=np.float64),
            tau_measured=np.asarray(tau_measured_log, dtype=np.float64),
            output_dir=output_dir,
            capture_prefix=args.output_prefix,
            output_prefix=args.ident_output_prefix,
            summary_path=summary_path,
        )

    if safety_error_message is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
