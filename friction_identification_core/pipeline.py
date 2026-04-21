from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from friction_identification_core.config import Config
from friction_identification_core.controller import SingleMotorController
from friction_identification_core.identification import identify_motor_friction
from friction_identification_core.models import RoundCapture
from friction_identification_core.results import ResultStore, RoundArtifact, SummaryPaths
from friction_identification_core.runtime import log_info, utc_now_iso8601
from friction_identification_core.serial_protocol import MotorSequenceChecker, SerialFrameParser, SingleMotorCommandAdapter
from friction_identification_core.trajectory import build_reference_trajectory
from friction_identification_core.transport import SerialTransport, open_serial_transport
from friction_identification_core.visualization import RerunRecorder


@dataclass(frozen=True)
class SequentialRunResult:
    artifacts: tuple[RoundArtifact, ...]
    summary_paths: SummaryPaths


def _capture_round(
    *,
    config: Config,
    transport: SerialTransport,
    parser: SerialFrameParser,
    sequence_checker: MotorSequenceChecker,
    command_adapter: SingleMotorCommandAdapter,
    controller: SingleMotorController,
    target_motor_id: int,
    group_index: int,
    round_index: int,
    rerun_recorder: RerunRecorder,
) -> tuple[RoundCapture, bool]:
    motor_name = config.motors.name_for(target_motor_id)
    target_index = config.motor_index(target_motor_id)
    target_max_velocity = float(config.control.max_velocity[target_index])
    target_max_torque = float(config.control.max_torque[target_index])
    reference = build_reference_trajectory(
        config.excitation,
        max_velocity=target_max_velocity,
    )

    if config.serial.flush_input_before_round:
        transport.reset_input_buffer()
        sequence_checker.reset()

    time_log: list[float] = []
    motor_id_log: list[int] = []
    position_log: list[float] = []
    velocity_log: list[float] = []
    torque_log: list[float] = []
    command_raw_log: list[float] = []
    command_log: list[float] = []
    position_cmd_log: list[float] = []
    velocity_cmd_log: list[float] = []
    acceleration_cmd_log: list[float] = []
    phase_log: list[str] = []
    state_log: list[int] = []
    mos_temperature_log: list[float] = []
    id_match_log: list[bool] = []
    sequence_error_count = 0
    observed_frame_count = 0
    target_frame_count = 0
    target_frame_goal = max(int(reference.time.size), 1)
    sync_required_frames = max(len(config.enabled_motor_ids) * int(config.serial.sync_cycles_required), 1)
    synced_before_capture = False
    sync_wait_duration_s = 0.0
    interrupted = False
    capture_started_at: str | None = None
    round_started_at = utc_now_iso8601()
    sync_started_monotonic = time.monotonic()
    capture_started_monotonic: float | None = None
    consecutive_sync_frames = 0

    try:
        while True:
            now = time.monotonic()
            if synced_before_capture and capture_started_monotonic is not None:
                if (now - capture_started_monotonic) >= reference.duration_s:
                    break
            elif (now - sync_started_monotonic) >= float(config.serial.sync_timeout):
                break

            chunk = transport.read(config.serial.read_chunk_size)
            if chunk:
                parser.feed(chunk)

            saw_frame = False
            capture_complete = False
            while True:
                frame = parser.pop_frame()
                if frame is None:
                    break
                saw_frame = True
                observed_frame_count += 1
                sequence_ok = sequence_checker.observe(frame.motor_id)
                if not sequence_ok:
                    sequence_error_count += 1
                    consecutive_sync_frames = 0
                else:
                    consecutive_sync_frames += 1

                if not synced_before_capture:
                    if frame.motor_id == target_motor_id:
                        transport.write(command_adapter.pack(target_motor_id, 0.0))
                    rerun_recorder.log_live_motor_sample(
                        group_index=int(group_index),
                        round_index=int(round_index),
                        active_motor_id=int(target_motor_id),
                        motor_id=int(frame.motor_id),
                        position=float(frame.position),
                        velocity=float(frame.velocity),
                        target_torque=0.0,
                        feedback_torque=float(frame.torque),
                        state=int(frame.state),
                        mos_temperature=float(frame.mos_temperature),
                        phase_name="sync_wait",
                    )
                    if sequence_ok and consecutive_sync_frames >= sync_required_frames:
                        synced_before_capture = True
                        sync_wait_duration_s = time.monotonic() - sync_started_monotonic
                        capture_started_monotonic = time.monotonic()
                        capture_started_at = utc_now_iso8601()
                        consecutive_sync_frames = 0
                        sequence_checker.reset()
                    continue

                assert capture_started_monotonic is not None
                elapsed_s = time.monotonic() - capture_started_monotonic
                if elapsed_s >= reference.duration_s:
                    capture_complete = True
                    break
                reference_sample = reference.sample(elapsed_s)

                command_raw = 0.0
                command = 0.0
                if frame.motor_id == target_motor_id:
                    command_raw, command_limited = controller.update(target_motor_id, reference_sample, frame)
                    command = command_adapter.limit_command(target_motor_id, command_limited)
                    transport.write(command_adapter.pack(target_motor_id, command))
                    target_frame_count += 1

                    time_log.append(float(elapsed_s))
                    motor_id_log.append(int(frame.motor_id))
                    position_log.append(float(frame.position))
                    velocity_log.append(float(frame.velocity))
                    torque_log.append(float(frame.torque))
                    command_raw_log.append(float(command_raw))
                    command_log.append(float(command))
                    position_cmd_log.append(float(reference_sample.position_cmd))
                    velocity_cmd_log.append(float(reference_sample.velocity_cmd))
                    acceleration_cmd_log.append(float(reference_sample.acceleration_cmd))
                    phase_log.append(str(reference_sample.phase_name))
                    state_log.append(int(frame.state))
                    mos_temperature_log.append(float(frame.mos_temperature))
                    id_match_log.append(True)
                    rerun_recorder.log_live_sample(
                        group_index=int(group_index),
                        round_index=int(round_index),
                        motor_id=int(frame.motor_id),
                        motor_name=motor_name,
                        sample_index=len(time_log) - 1,
                        elapsed_s=float(elapsed_s),
                        position=float(frame.position),
                        position_cmd=float(reference_sample.position_cmd),
                        velocity=float(frame.velocity),
                        velocity_cmd=float(reference_sample.velocity_cmd),
                        command_raw=float(command_raw),
                        command=float(command),
                        torque_feedback=float(frame.torque),
                        phase_name=str(reference_sample.phase_name),
                    )

                    rerun_recorder.log_live_motor_sample(
                        group_index=int(group_index),
                        round_index=int(round_index),
                        active_motor_id=int(target_motor_id),
                        motor_id=int(frame.motor_id),
                    position=float(frame.position),
                    velocity=float(frame.velocity),
                    target_torque=float(command),
                    feedback_torque=float(frame.torque),
                    state=int(frame.state),
                    mos_temperature=float(frame.mos_temperature),
                        phase_name=str(reference_sample.phase_name),
                    )

            if capture_complete:
                break

            if not saw_frame and not chunk:
                time.sleep(max(float(config.serial.read_timeout), 1.0e-3))
    except KeyboardInterrupt:
        interrupted = True
    finally:
        transport.write(command_adapter.pack(target_motor_id, 0.0))
        rerun_recorder.log_round_stop(
            group_index=int(group_index),
            round_index=int(round_index),
            motor_id=int(target_motor_id),
            phase_name="interrupted" if interrupted else "round_complete",
        )

    sequence_error_ratio = float(sequence_error_count / observed_frame_count) if observed_frame_count else 0.0
    target_frame_ratio = min(float(target_frame_count / target_frame_goal), 1.0) if target_frame_goal else 0.0
    stop_reason = "interrupted" if interrupted else "completed"
    if not synced_before_capture and not interrupted:
        stop_reason = "sync_timeout"
    metadata = {
        "group_index": int(group_index),
        "round_index": int(round_index),
        "target_motor_id": int(target_motor_id),
        "enabled_motor_ids": list(config.enabled_motor_ids),
        "excitation_config": asdict(config.excitation),
        "start_time": capture_started_at or round_started_at,
        "round_start_time": round_started_at,
        "stop_reason": stop_reason,
        "synced_before_capture": bool(synced_before_capture),
        "sync_wait_duration_s": float(sync_wait_duration_s),
        "sync_timeout": float(config.serial.sync_timeout),
        "sync_required_frames": int(sync_required_frames),
        "observed_frame_count": int(observed_frame_count),
        "sequence_error_count": int(sequence_error_count),
        "sequence_error_ratio": float(sequence_error_ratio),
        "target_frame_goal": int(target_frame_goal),
        "target_frame_count": int(target_frame_count),
        "target_frame_ratio": float(target_frame_ratio),
        "target_max_velocity": target_max_velocity,
        "target_max_torque": target_max_torque,
    }
    capture = RoundCapture(
        group_index=int(group_index),
        round_index=int(round_index),
        target_motor_id=int(target_motor_id),
        motor_name=motor_name,
        time=np.asarray(time_log, dtype=np.float64),
        motor_id=np.asarray(motor_id_log, dtype=np.int64),
        position=np.asarray(position_log, dtype=np.float64),
        velocity=np.asarray(velocity_log, dtype=np.float64),
        torque_feedback=np.asarray(torque_log, dtype=np.float64),
        command_raw=np.asarray(command_raw_log, dtype=np.float64),
        command=np.asarray(command_log, dtype=np.float64),
        position_cmd=np.asarray(position_cmd_log, dtype=np.float64),
        velocity_cmd=np.asarray(velocity_cmd_log, dtype=np.float64),
        acceleration_cmd=np.asarray(acceleration_cmd_log, dtype=np.float64),
        phase_name=np.asarray(phase_log),
        state=np.asarray(state_log, dtype=np.uint8),
        mos_temperature=np.asarray(mos_temperature_log, dtype=np.float64),
        id_match_ok=np.asarray(id_match_log, dtype=bool),
        metadata=metadata,
    )
    return capture, interrupted


def run_sequential_identification(
    config: Config,
    *,
    transport_factory: Callable[[], SerialTransport] | None = None,
) -> SequentialRunResult:
    store = ResultStore(config)
    parser = SerialFrameParser(max_motor_id=max(config.motor_ids))
    sequence_checker = MotorSequenceChecker(config.motor_ids)
    command_adapter = SingleMotorCommandAdapter(
        motor_count=max(config.motor_ids),
        torque_limits=config.control.max_torque,
    )
    controller = SingleMotorController(config)
    rerun_recorder = RerunRecorder(
        store.rerun_recording_path,
        motor_ids=config.motor_ids,
        motor_names={motor_id: config.motors.name_for(motor_id) for motor_id in config.motor_ids},
    )
    artifacts: list[RoundArtifact] = []

    transport = transport_factory() if transport_factory is not None else open_serial_transport(config.serial)
    try:
        total_rounds = int(config.group_count) * len(config.enabled_motor_ids)
        current_round = 0
        stop_requested = False
        for group_index in range(1, int(config.group_count) + 1):
            for target_motor_id in config.enabled_motor_ids:
                current_round += 1
                log_info(
                    "Starting sequential identification round "
                    f"{current_round}/{total_rounds}: "
                    f"group={group_index}, motor_id={target_motor_id}"
                )
                capture, interrupted = _capture_round(
                    config=config,
                    transport=transport,
                    parser=parser,
                    sequence_checker=sequence_checker,
                    command_adapter=command_adapter,
                    controller=controller,
                    target_motor_id=int(target_motor_id),
                    group_index=int(group_index),
                    round_index=int(current_round),
                    rerun_recorder=rerun_recorder,
                )
                identification = identify_motor_friction(
                    config.identification,
                    capture,
                    max_torque=float(config.control.max_torque[config.motor_index(target_motor_id)]),
                    max_velocity=float(config.control.max_velocity[config.motor_index(target_motor_id)]),
                )
                capture_path = store.save_capture(capture)
                identification_path = store.save_identification(capture, identification)
                rerun_recorder.log_identification(capture, identification)
                artifacts.append(
                    RoundArtifact(
                        capture=capture,
                        identification=identification,
                        capture_path=capture_path,
                        identification_path=identification_path,
                    )
                )
                if interrupted:
                    stop_requested = True
                    break
            if stop_requested:
                break
        summary_paths = store.save_summary(artifacts)
        rerun_recorder.log_summary(
            summary_path=summary_paths.run_summary_path,
            report_path=summary_paths.run_summary_report_path,
        )
        return SequentialRunResult(artifacts=tuple(artifacts), summary_paths=summary_paths)
    finally:
        rerun_recorder.close()
        transport.close()
