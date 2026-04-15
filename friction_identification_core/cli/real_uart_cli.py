#!/usr/bin/env python3

from __future__ import annotations

"""Real UART collection/compensation CLI for AM-D02."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from friction_identification_core.config import DEFAULT_FRICTION_CONFIG
from friction_identification_core.real_serial_protocol import (
    RECV_FRAME_SIZE,
    SEND_FRAME_SIZE,
    SerialFrameReader,
    TorqueCommandFramePacker,
)
from friction_identification_core.real_support import (
    SafetyLimitExceededError,
    build_real_collection_controller,
    check_joint_limits_or_raise,
    compute_collect_bias_compensation_torque,
    compute_realtime_control_command,
    get_real_excitation_limits,
    has_completed_startup_transition,
    identify_real_friction_from_capture,
    load_identified_friction_parameters,
    maybe_build_pose_estimator,
    save_capture,
)
from friction_identification_core.runtime import ensure_results_dir, log_info


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
        "--transition-max-ee-speed",
        type=float,
        default=DEFAULT_FRICTION_CONFIG.collection.transition_max_ee_speed,
        help="Maximum end-effector speed in m/s used for the startup point-to-point transition.",
    )
    parser.add_argument(
        "--transition-min-duration",
        type=float,
        default=DEFAULT_FRICTION_CONFIG.collection.transition_min_duration,
        help="Minimum duration in seconds for the startup point-to-point transition.",
    )
    parser.add_argument(
        "--transition-settle-duration",
        type=float,
        default=DEFAULT_FRICTION_CONFIG.collection.transition_settle_duration,
        help="Extra settle time in seconds after reaching the excitation start pose.",
    )
    parser.add_argument(
        "--stop-at-excitation-start",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="In collect mode, stop after reaching the planned excitation start pose instead of entering the excitation stage.",
    )
    parser.add_argument(
        "--tx-debug-frames",
        type=int,
        default=3,
        help="Print the first N transmitted UART torque frames for verification.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the realtime UART loop, then save capture and optional fit outputs."""

    args = parse_args()
    send_enabled = bool(DEFAULT_FRICTION_CONFIG.real_uart.send_enabled)
    send_bias_compensation_only = bool(DEFAULT_FRICTION_CONFIG.real_uart.send_bias_compensation_only)
    bias_compensation_only_active = (
        args.control_mode == "collect" and (not send_enabled) and send_bias_compensation_only
    )
    if args.control_mode == "collect":
        if send_enabled:
            control_output_mode = "full_excitation"
        elif bias_compensation_only_active:
            control_output_mode = "gravity_coriolis_compensation_only"
        else:
            control_output_mode = "disabled_full_excitation_compute_only"
    else:
        control_output_mode = (
            "identified_friction_compensation" if send_enabled else "disabled_friction_compensation_compute_only"
        )
    summary_path = Path(args.summary_path).resolve()
    output_dir = ensure_results_dir()
    parameters = None
    collection_controller = None
    if args.control_mode != "collect" and args.stop_at_excitation_start:
        log_info("stop_at_excitation_start 仅对 collect 模式生效，当前 compensate 模式会忽略该参数。")
    if args.control_mode == "compensate":
        parameters = load_identified_friction_parameters(summary_path)
        log_info(f"已加载辨识参数: {summary_path}")
        if send_bias_compensation_only:
            log_info("send_bias_compensation_only 仅对 collect 模式生效，当前 compensate 模式会忽略该配置。")
        for idx, param in enumerate(parameters, start=1):
            log_info(
                f"J{idx}: fc={param.coulomb:.6f}, fv={param.viscous:.6f}, "
                f"offset={param.offset:.6f}, v_scale={param.velocity_scale:.6f}"
            )
    else:
        if send_enabled:
            log_info("当前运行模式: collect，将下发采集激励力矩并基于真机数据做辨识。")
            if send_bias_compensation_only:
                log_info("检测到 send_enabled=True，send_bias_compensation_only 将被忽略，仍发送完整激励。")
        elif bias_compensation_only_active:
            log_info("当前运行模式: collect，不发送完整激励；仅向真机下发重力+科氏补偿，并记录完整激励计算结果。")
        else:
            log_info("当前运行模式: collect，已关闭真机力矩下发，只接收状态、计算控制量并记录辨识数据。")

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
    log_info(
        "真机力矩下发配置: "
        f"send_enabled={send_enabled}, "
        f"send_bias_compensation_only={send_bias_compensation_only}, "
        f"control_output_mode={control_output_mode}"
    )
    if args.control_mode == "collect":
        log_info(
            "采集控制配置: 复用 MuJoCo 轨迹与控制律, "
            f"base_frequency={args.collection_base_frequency:.3f}Hz, "
            f"amplitude_scale={args.collection_amplitude_scale:.3f}, "
            f"feedback_scale={args.collection_feedback_scale:.3f}, "
            f"transition_max_ee_speed={args.transition_max_ee_speed:.3f}m/s, "
            f"transition_min_duration={args.transition_min_duration:.3f}s, "
            f"transition_settle_duration={args.transition_settle_duration:.3f}s, "
            f"stop_at_excitation_start={args.stop_at_excitation_start}"
        )
        collection_controller = build_real_collection_controller(
            duration_s=args.duration,
            base_frequency=args.collection_base_frequency,
            amplitude_scale=args.collection_amplitude_scale,
            feedback_scale=args.collection_feedback_scale,
            transition_max_ee_speed=args.transition_max_ee_speed,
            transition_min_duration=args.transition_min_duration,
            transition_settle_duration=args.transition_settle_duration,
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
            if send_enabled or bias_compensation_only_active:
                log_info("MuJoCo 仿真窗口未启动，程序继续执行串口采集与力矩下发。")
            else:
                log_info("MuJoCo 仿真窗口未启动，程序继续执行串口采集，但不会向下位机下发力矩。")

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
    stop_requested = False

    try:
        while True:
            if stop_requested:
                break
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

                feedback_mask = 0
                emitted_sample = True
                cycle_end = time.perf_counter()
                elapsed_s = cycle_end - start_time

                check_joint_limits_or_raise(q)
                if (
                    args.control_mode == "collect"
                    and args.stop_at_excitation_start
                    and collection_controller is not None
                    and has_completed_startup_transition(
                        collection_controller,
                        elapsed_s=elapsed_s,
                        q=q,
                    )
                ):
                    termination_reason = "reached_excitation_start"
                    log_info("已到达预期激励起始姿态，按照 stop_at_excitation_start 配置停止 UART 运行。")
                    if send_enabled or bias_compensation_only_active:
                        ser.write(zero_torque_frame)
                    stop_requested = True
                    break
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
                if bias_compensation_only_active:
                    tau_command = compute_collect_bias_compensation_torque(
                        q=q,
                        qd=qd,
                        collection_controller=collection_controller,
                    )
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
                    if bias_compensation_only_active and control_log_remaining > 0:
                        log_info(
                            "仅补偿下发: "
                            f"tau_bias={[round(float(x), 4) for x in tau_command]}"
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
                if send_enabled or bias_compensation_only_active:
                    ser.write(tx_frame)
                if tx_debug_remaining > 0:
                    debug_prefix = (
                        "TX frame debug"
                        if (send_enabled or bias_compensation_only_active)
                        else "TX frame skipped"
                    )
                    log_info(
                        f"{debug_prefix}: torque=["
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
                time.sleep(max(args.serial_idle_sleep, 0.0))

    except KeyboardInterrupt:
        termination_reason = "keyboard_interrupt"
        log_info("收到 Ctrl+C，准备保存实时采集结果。")
    except SafetyLimitExceededError as exc:
        termination_reason = f"safety_stop: {exc}"
        log_info(str(exc))
        if send_enabled or bias_compensation_only_active:
            try:
                ser.write(zero_torque_frame)
            except Exception:
                pass
        else:
            log_info("安全停机时未发送零力矩帧，因为 send_enabled=False。")
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
            control_output_mode=control_output_mode,
            send_enabled=send_enabled,
            send_bias_compensation_only=send_bias_compensation_only,
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
            stop_at_excitation_start=bool(args.stop_at_excitation_start),
        )
        if reporter is not None:
            reporter.close()

    if args.control_mode == "collect" and time_log and not args.stop_at_excitation_start:
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
