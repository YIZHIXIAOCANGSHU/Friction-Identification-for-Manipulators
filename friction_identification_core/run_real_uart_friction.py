#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from friction_identification_core.config import DEFAULT_FRICTION_CONFIG
from friction_identification_core.models import JointFrictionParameters
from friction_identification_core.real_serial_protocol import (
    RECV_FRAME_SIZE,
    SEND_FRAME_SIZE,
    SerialFrameReader,
    TorqueCommandFramePacker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real UART friction-compensation forwarding for AM-D02.")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Serial port for the lower controller.")
    parser.add_argument("--baudrate", type=int, default=115200, help="UART baudrate.")
    parser.add_argument("--duration", type=float, default=0.0, help="Run duration in seconds, 0 means until Ctrl+C.")
    parser.add_argument(
        "--spawn-rerun",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Spawn a local Rerun viewer.",
    )
    parser.add_argument(
        "--summary-path",
        default=str(PROJECT_ROOT / "results" / "friction_identification_summary.json"),
        help="Path to the latest friction identification summary JSON.",
    )
    parser.add_argument(
        "--output-prefix",
        default="real_uart_capture",
        help="Basename for saved capture files under results/.",
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
    return parser.parse_args()


def log_info(message: str) -> None:
    print(f"[INFO] {message}", flush=True)


def load_identified_friction_parameters(summary_path: Path) -> list[JointFrictionParameters]:
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
    velocity = np.asarray(velocity, dtype=np.float64).reshape(-1)
    if velocity.size != len(parameters):
        raise ValueError("velocity size must match the number of friction parameter groups.")

    torque = np.zeros_like(velocity)
    for idx, param in enumerate(parameters):
        scale = max(float(param.velocity_scale), 1e-6)
        torque[idx] = param.coulomb * np.tanh(velocity[idx] / scale) + param.viscous * velocity[idx] + param.offset

    torque_limits = DEFAULT_FRICTION_CONFIG.model.torque_limits
    return np.clip(torque, -torque_limits, torque_limits)


def maybe_build_pose_estimator():
    try:
        from friction_identification_core.real_pose_estimator import RealPoseEstimator

        model = DEFAULT_FRICTION_CONFIG.model
        return RealPoseEstimator(
            model_path=str(model.urdf_path),
            joint_names=list(model.joint_names),
            end_effector_body=model.end_effector_body,
            tcp_offset=model.tcp_offset,
        )
    except Exception as exc:
        log_info(f"MuJoCo 位姿估计不可用，将仅记录关节与力矩数据: {exc}")
        return None


def save_capture(
    *,
    output_dir: Path,
    output_prefix: str,
    summary_path: Path,
    parameters: list[JointFrictionParameters],
    time_log: list[float],
    q_log: list[np.ndarray],
    qd_log: list[np.ndarray],
    tau_measured_log: list[np.ndarray],
    tau_command_log: list[np.ndarray],
    mos_temp_log: list[np.ndarray],
    coil_temp_log: list[np.ndarray],
    ee_pos_log: list[np.ndarray],
    ee_quat_log: list[np.ndarray],
    uart_cycle_hz_log: list[float],
    uart_latency_ms_log: list[float],
    uart_transfer_kbps_log: list[float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{output_prefix}.npz"
    json_path = output_dir / f"{output_prefix}.json"

    q = np.asarray(q_log, dtype=np.float64).reshape(-1, 7)
    qd = np.asarray(qd_log, dtype=np.float64).reshape(-1, 7)
    tau_measured = np.asarray(tau_measured_log, dtype=np.float64).reshape(-1, 7)
    tau_command = np.asarray(tau_command_log, dtype=np.float64).reshape(-1, 7)
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
        mos_temperature=mos_temp,
        coil_temperature=coil_temp,
        ee_pos=ee_pos,
        ee_quat=ee_quat,
        uart_cycle_hz=np.asarray(uart_cycle_hz_log, dtype=np.float64),
        uart_latency_ms=np.asarray(uart_latency_ms_log, dtype=np.float64),
        uart_transfer_kbps=np.asarray(uart_transfer_kbps_log, dtype=np.float64),
    )

    summary = {
        "summary_path": str(summary_path),
        "sample_count": int(len(time_log)),
        "duration_s": float(time_log[-1]) if time_log else 0.0,
        "mean_uart_cycle_hz": float(np.mean(uart_cycle_hz_log)) if uart_cycle_hz_log else 0.0,
        "max_uart_cycle_hz": float(np.max(uart_cycle_hz_log)) if uart_cycle_hz_log else 0.0,
        "mean_uart_latency_ms": float(np.mean(uart_latency_ms_log)) if uart_latency_ms_log else 0.0,
        "mean_uart_transfer_kbps": float(np.mean(uart_transfer_kbps_log)) if uart_transfer_kbps_log else 0.0,
        "friction_parameters": [
            {
                "joint": DEFAULT_FRICTION_CONFIG.model.joint_names[idx],
                "coulomb": param.coulomb,
                "viscous": param.viscous,
                "offset": param.offset,
                "velocity_scale": param.velocity_scale,
            }
            for idx, param in enumerate(parameters)
        ],
    }
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    log_info(f"采集结果已保存: {npz_path}")
    log_info(f"采集摘要已保存: {json_path}")


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_path).resolve()
    output_dir = PROJECT_ROOT / "results"

    parameters = load_identified_friction_parameters(summary_path)
    log_info(f"已加载辨识参数: {summary_path}")
    for idx, param in enumerate(parameters, start=1):
        log_info(
            f"J{idx}: fc={param.coulomb:.6f}, fv={param.viscous:.6f}, "
            f"offset={param.offset:.6f}, v_scale={param.velocity_scale:.6f}"
        )

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

    reporter = None
    if args.spawn_rerun:
        from friction_identification_core.real_rerun_reporter import RealTimeRerunReporter

        reporter = RealTimeRerunReporter(
            app_name="AM-D02 Real UART Friction Compensation",
            joint_names=DEFAULT_FRICTION_CONFIG.model.joint_names,
            spawn=True,
        )
        reporter.init()

    pose_estimator = maybe_build_pose_estimator()

    frame_reader = SerialFrameReader()
    frame_packer = TorqueCommandFramePacker()

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

    try:
        while True:
            if args.duration > 0.0 and (time.perf_counter() - start_time) >= args.duration:
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

                feedback_mask = 0
                emitted_sample = True
                cycle_end = time.perf_counter()
                elapsed_s = cycle_end - start_time
                tau_command = predict_joint_friction_torque(qd, parameters)
                ser.write(frame_packer.pack(tau_command))

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
                    ee_pos, ee_quat = pose_estimator.compute(q)
                    ee_pos_log.append(np.asarray(ee_pos, dtype=np.float64))
                    ee_quat_log.append(np.asarray(ee_quat, dtype=np.float64))

                time_log.append(float(elapsed_s))
                q_log.append(q.copy())
                qd_log.append(qd.copy())
                tau_measured_log.append(tau_measured.copy())
                tau_command_log.append(tau_command.copy())
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
                    log_info(
                        f"step={step_index}, uart={uart_cycle_hz:.2f} Hz, "
                        f"q1={q[0]:.4f}, qd1={qd[0]:.4f}, tau_cmd1={tau_command[0]:.4f}"
                    )
                step_index += 1

            if bytes_waiting == 0 and not emitted_sample and not frame_reader.has_complete_frame():
                time.sleep(max(args.serial_idle_sleep, 0.0))

    except KeyboardInterrupt:
        log_info("收到 Ctrl+C，准备保存实时采集结果。")
    finally:
        try:
            ser.close()
        except Exception:
            pass
        save_capture(
            output_dir=output_dir,
            output_prefix=args.output_prefix,
            summary_path=summary_path,
            parameters=parameters,
            time_log=time_log,
            q_log=q_log,
            qd_log=qd_log,
            tau_measured_log=tau_measured_log,
            tau_command_log=tau_command_log,
            mos_temp_log=mos_temp_log,
            coil_temp_log=coil_temp_log,
            ee_pos_log=ee_pos_log,
            ee_quat_log=ee_quat_log,
            uart_cycle_hz_log=uart_cycle_hz_log,
            uart_latency_ms_log=uart_latency_ms_log,
            uart_transfer_kbps_log=uart_transfer_kbps_log,
        )
        if reporter is not None:
            reporter.close()


if __name__ == "__main__":
    main()
