from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from friction_identification_core.config import Config
from friction_identification_core.controller import SingleMotorController
from friction_identification_core.identification import identify_motor_friction
from friction_identification_core.models import MotorCompensationParameters, RoundCapture
from friction_identification_core.results import ResultStore, RoundArtifact, SummaryPaths
from friction_identification_core.runtime import log_info, utc_now_iso8601
from friction_identification_core.serial_protocol import SerialFrameParser, SingleMotorCommandAdapter
from friction_identification_core.trajectory import build_reference_trajectory
from friction_identification_core.transport import SerialTransport, open_serial_transport
from friction_identification_core.visualization import RerunRecorder


@dataclass(frozen=True)
class SequentialRunResult:
    artifacts: tuple[RoundArtifact, ...]
    summary_paths: SummaryPaths | None
    manifest_path: Path


@dataclass(frozen=True)
class _AbortEvent:
    reason: str
    motor_id: int
    group_index: int
    round_index: int
    phase_name: str
    observed_velocity: float | None = None
    velocity_limit: float | None = None
    feedback_torque: float | None = None
    torque_limit: float | None = None

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "reason": str(self.reason),
            "motor_id": int(self.motor_id),
            "group_index": int(self.group_index),
            "round_index": int(self.round_index),
            "phase_name": str(self.phase_name),
        }
        if self.observed_velocity is not None:
            payload["observed_velocity"] = float(self.observed_velocity)
        if self.velocity_limit is not None:
            payload["velocity_limit"] = float(self.velocity_limit)
        if self.feedback_torque is not None:
            payload["feedback_torque"] = float(self.feedback_torque)
        if self.torque_limit is not None:
            payload["torque_limit"] = float(self.torque_limit)
        return payload

    def error_message(self) -> str:
        parts = [
            f"reason={self.reason}",
            f"motor_id={self.motor_id}",
            f"group_index={self.group_index}",
            f"round_index={self.round_index}",
            f"phase_name={self.phase_name}",
        ]
        if self.observed_velocity is not None:
            parts.append(f"observed_velocity={self.observed_velocity:.6f}")
        if self.velocity_limit is not None:
            parts.append(f"velocity_limit={self.velocity_limit:.6f}")
        if self.feedback_torque is not None:
            parts.append(f"feedback_torque={self.feedback_torque:.6f}")
        if self.torque_limit is not None:
            parts.append(f"torque_limit={self.torque_limit:.6f}")
        return "Runtime safety abort: " + ", ".join(parts)


class _SafetyAbortError(ValueError):
    def __init__(self, event: _AbortEvent) -> None:
        self.event = event
        super().__init__(event.error_message())


def _root_compensation_summary_path(config: Config) -> Path:
    return (config.results_dir / config.output.summary_filename).resolve()


def _latest_identify_summary_path(config: Config) -> Path | None:
    runs_dir = config.results_dir / "runs"
    if not runs_dir.exists():
        return None

    candidates: list[tuple[datetime, str, Path]] = []
    for manifest_path in runs_dir.glob("*_identify/run_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue

        if str(manifest.get("mode", "")) != "identify":
            continue

        end_time_raw = manifest.get("end_time")
        if not end_time_raw:
            continue

        summary_files = manifest.get("summary_files")
        if not isinstance(summary_files, dict):
            continue
        summary_path_raw = summary_files.get("run_summary_path")
        if not summary_path_raw:
            continue

        try:
            end_time = datetime.fromisoformat(str(end_time_raw))
        except ValueError:
            continue

        summary_path = Path(summary_path_raw).resolve()
        if not summary_path.exists():
            continue

        run_label = str(manifest.get("run_label") or manifest_path.parent.name)
        candidates.append((end_time, run_label, summary_path))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def _resolve_compensation_summary_path(
    config: Config,
    *,
    parameters_path: Path | None = None,
) -> tuple[Path, str]:
    if parameters_path is not None:
        return Path(parameters_path).resolve(), "explicit parameters_path"

    latest_summary_path = _latest_identify_summary_path(config)
    if latest_summary_path is not None:
        return latest_summary_path, "latest identify run summary"

    return _root_compensation_summary_path(config), "root snapshot summary"


def _expected_velocity_vector(config: Config, *, target_index: int, target_velocity: float) -> np.ndarray:
    expected = np.zeros(config.motor_count, dtype=np.float64)
    expected[target_index] = float(target_velocity)
    return expected


def _sent_command_vector(config: Config, *, target_index: int, target_command: float) -> np.ndarray:
    sent_commands = np.zeros(config.motor_count, dtype=np.float64)
    sent_commands[target_index] = float(target_command)
    return sent_commands


def _capture_compensation_metrics(capture: RoundCapture) -> dict[str, float]:
    velocity = np.asarray(capture.velocity, dtype=np.float64)
    velocity_cmd = np.asarray(capture.velocity_cmd, dtype=np.float64)
    error = velocity - velocity_cmd
    finite = np.isfinite(error)
    if not np.any(finite):
        return {
            "tracking_velocity_rmse": float("nan"),
            "tracking_velocity_mae": float("nan"),
            "tracking_velocity_max_abs": float("nan"),
        }
    error = error[finite]
    return {
        "tracking_velocity_rmse": float(np.sqrt(np.mean(error**2))),
        "tracking_velocity_mae": float(np.mean(np.abs(error))),
        "tracking_velocity_max_abs": float(np.max(np.abs(error))),
    }


def _phase_velocity_limit(
    config: Config,
    *,
    phase_name: str,
    reference_velocity: float,
    target_max_velocity: float,
) -> float:
    phase_name = str(phase_name)
    if phase_name.startswith("hold_"):
        return 0.0
    if phase_name.startswith("transition_"):
        return abs(float(reference_velocity))
    if phase_name.startswith("settle_") or phase_name.startswith("steady_"):
        try:
            platform_index = int(phase_name.rsplit("_", 1)[-1]) - 1
        except ValueError:
            return abs(float(reference_velocity))
        if 0 <= platform_index < len(config.excitation.platforms):
            return abs(float(config.excitation.platforms[platform_index].resolve_speed(max_velocity=target_max_velocity)))
    return abs(float(reference_velocity))


def _build_abort_event(
    config: Config,
    *,
    frame,
    reference_velocity: float,
    phase_name: str,
    target_motor_id: int,
    group_index: int,
    round_index: int,
    target_max_velocity: float,
    target_max_torque: float,
) -> _AbortEvent | None:
    velocity_limit = _phase_velocity_limit(
        config,
        phase_name=phase_name,
        reference_velocity=reference_velocity,
        target_max_velocity=target_max_velocity,
    )
    velocity_threshold = float(config.identification.zero_velocity_threshold)
    if abs(float(frame.velocity)) > velocity_limit + velocity_threshold:
        return _AbortEvent(
            reason="velocity_limit_exceeded",
            motor_id=int(target_motor_id),
            group_index=int(group_index),
            round_index=int(round_index),
            phase_name=str(phase_name),
            observed_velocity=float(frame.velocity),
            velocity_limit=float(velocity_limit),
        )

    torque_epsilon = float(np.finfo(np.float32).eps * max(abs(float(target_max_torque)), 1.0))
    if abs(float(frame.torque)) > float(target_max_torque) + torque_epsilon:
        return _AbortEvent(
            reason="torque_limit_exceeded",
            motor_id=int(target_motor_id),
            group_index=int(group_index),
            round_index=int(round_index),
            phase_name=str(phase_name),
            feedback_torque=float(frame.torque),
            torque_limit=float(target_max_torque),
        )
    return None


def _load_compensation_parameters(
    config: Config,
    *,
    parameters_path: Path | None = None,
) -> tuple[Path, str, dict[int, MotorCompensationParameters]]:
    resolved_path, source_label = _resolve_compensation_summary_path(
        config,
        parameters_path=parameters_path,
    )
    if not resolved_path.exists():
        raise ValueError(f"Compensation summary file not found: {resolved_path}")

    with np.load(resolved_path, allow_pickle=False) as summary:
        required_fields = ("motor_ids", "recommended_for_runtime", "coulomb", "viscous", "offset", "velocity_scale")
        missing_fields = [field for field in required_fields if field not in summary.files]
        if missing_fields:
            raise ValueError(
                "Compensation summary is missing fields: " + ", ".join(missing_fields)
            )

        motor_ids = np.asarray(summary["motor_ids"], dtype=np.int64)
        motor_names = (
            np.asarray(summary["motor_names"]).astype(str)
            if "motor_names" in summary.files
            else np.asarray([config.motors.name_for(int(motor_id)) for motor_id in motor_ids])
        )
        recommended_for_runtime = np.asarray(summary["recommended_for_runtime"], dtype=bool)
        coulomb = np.asarray(summary["coulomb"], dtype=np.float64)
        viscous = np.asarray(summary["viscous"], dtype=np.float64)
        offset = np.asarray(summary["offset"], dtype=np.float64)
        velocity_scale = np.asarray(summary["velocity_scale"], dtype=np.float64)

    motor_index = {int(motor_id): index for index, motor_id in enumerate(motor_ids.tolist())}
    missing_motor_ids: list[int] = []
    not_recommended_motor_ids: list[int] = []
    invalid_motor_ids: list[int] = []
    parameters_by_motor: dict[int, MotorCompensationParameters] = {}

    for motor_id in config.enabled_motor_ids:
        index = motor_index.get(int(motor_id))
        if index is None:
            missing_motor_ids.append(int(motor_id))
            continue
        values = (
            float(coulomb[index]),
            float(viscous[index]),
            float(offset[index]),
            float(velocity_scale[index]),
        )
        if not bool(recommended_for_runtime[index]):
            not_recommended_motor_ids.append(int(motor_id))
            continue
        if not np.all(np.isfinite(values)):
            invalid_motor_ids.append(int(motor_id))
            continue
        parameters_by_motor[int(motor_id)] = MotorCompensationParameters(
            motor_id=int(motor_id),
            motor_name=str(motor_names[index]),
            coulomb=float(coulomb[index]),
            viscous=float(viscous[index]),
            offset=float(offset[index]),
            velocity_scale=float(velocity_scale[index]),
        )

    problems: list[str] = []
    if missing_motor_ids:
        problems.append("missing: " + ",".join(str(motor_id) for motor_id in missing_motor_ids))
    if not_recommended_motor_ids:
        problems.append("not recommended: " + ",".join(str(motor_id) for motor_id in not_recommended_motor_ids))
    if invalid_motor_ids:
        problems.append("non-finite: " + ",".join(str(motor_id) for motor_id in invalid_motor_ids))
    if problems:
        raise ValueError("Compensation parameters are unavailable for selected motors (" + "; ".join(problems) + ").")

    return resolved_path, source_label, parameters_by_motor


def _capture_round(
    *,
    config: Config,
    transport: SerialTransport,
    parser: SerialFrameParser,
    command_adapter: SingleMotorCommandAdapter,
    controller: SingleMotorController,
    target_motor_id: int,
    group_index: int,
    round_index: int,
    rerun_recorder: RerunRecorder,
    mode: str,
    compensation: MotorCompensationParameters | None = None,
) -> tuple[RoundCapture, bool]:
    motor_name = config.motors.name_for(target_motor_id)
    target_index = config.motor_index(target_motor_id)
    target_max_velocity = float(config.control.max_velocity[target_index])
    target_max_torque = float(config.control.max_torque[target_index])
    reference = build_reference_trajectory(config.excitation, max_velocity=target_max_velocity)
    planned_duration_s = float(reference.duration_s)

    if config.serial.flush_input_before_round:
        transport.reset_input_buffer()
        parser.reset()

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
    observed_frame_count = 0
    target_frame_count = 0
    target_frame_goal = max(int(reference.time.size), 1)
    sync_required_target_frames = max(int(config.serial.sync_cycles_required), 1)
    synced_before_capture = False
    sync_wait_duration_s = 0.0
    interrupted = False
    capture_started_at: str | None = None
    round_started_at = utc_now_iso8601()
    sync_started_monotonic = time.monotonic()
    capture_started_monotonic: float | None = None
    target_sync_frame_count = 0
    current_target_command = 0.0
    abort_event: _AbortEvent | None = None

    rerun_recorder.log_round_timing(
        group_index=int(group_index),
        round_index=int(round_index),
        active_motor_id=int(target_motor_id),
        planned_duration_s=planned_duration_s,
        actual_capture_duration_s=0.0,
        sync_wait_duration_s=0.0,
        round_total_duration_s=0.0,
    )

    def _log_live_command(target_command: float, target_velocity: float, raw_packet: bytes) -> None:
        rerun_recorder.log_live_command_packet(
            group_index=int(group_index),
            round_index=int(round_index),
            active_motor_id=int(target_motor_id),
            sent_commands=_sent_command_vector(config, target_index=target_index, target_command=target_command),
            expected_velocities=_expected_velocity_vector(
                config,
                target_index=target_index,
                target_velocity=target_velocity,
            ),
            raw_packet=raw_packet,
        )

    def _start_capture() -> None:
        nonlocal synced_before_capture, sync_wait_duration_s, capture_started_monotonic, capture_started_at
        synced_before_capture = True
        sync_wait_duration_s = time.monotonic() - sync_started_monotonic
        capture_started_monotonic = time.monotonic()
        capture_started_at = utc_now_iso8601()
        rerun_recorder.log_round_timing(
            group_index=int(group_index),
            round_index=int(round_index),
            active_motor_id=int(target_motor_id),
            planned_duration_s=planned_duration_s,
            actual_capture_duration_s=0.0,
            sync_wait_duration_s=sync_wait_duration_s,
            round_total_duration_s=sync_wait_duration_s,
        )

    try:
        while True:
            now = time.monotonic()
            if synced_before_capture and capture_started_monotonic is not None:
                if (now - capture_started_monotonic) >= planned_duration_s:
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

                if not synced_before_capture:
                    if frame.motor_id == target_motor_id:
                        target_sync_frame_count += 1
                        current_target_command = 0.0
                        command_packet = command_adapter.pack(target_motor_id, 0.0)
                        transport.write(command_packet)
                        _log_live_command(0.0, 0.0, command_packet)
                    rerun_recorder.log_live_motor_sample(
                        group_index=int(group_index),
                        round_index=int(round_index),
                        active_motor_id=int(target_motor_id),
                        motor_id=int(frame.motor_id),
                        position=float(frame.position),
                        velocity=float(frame.velocity),
                        expected_velocity=0.0,
                        target_torque=0.0,
                        feedback_torque=float(frame.torque),
                        state=int(frame.state),
                        mos_temperature=float(frame.mos_temperature),
                        phase_name="sync_wait",
                    )
                    rerun_recorder.log_round_timing(
                        group_index=int(group_index),
                        round_index=int(round_index),
                        active_motor_id=int(target_motor_id),
                        planned_duration_s=planned_duration_s,
                        actual_capture_duration_s=0.0,
                        sync_wait_duration_s=time.monotonic() - sync_started_monotonic,
                        round_total_duration_s=time.monotonic() - sync_started_monotonic,
                    )
                    if target_sync_frame_count >= sync_required_target_frames:
                        _start_capture()
                    continue

                assert capture_started_monotonic is not None
                elapsed_s = time.monotonic() - capture_started_monotonic
                if elapsed_s >= planned_duration_s:
                    capture_complete = True
                    break

                reference_sample = reference.sample(elapsed_s)
                expected_velocity = float(reference_sample.velocity_cmd) if frame.motor_id == target_motor_id else 0.0
                phase_name = str(reference_sample.phase_name) if frame.motor_id == target_motor_id else "idle"

                if frame.motor_id == target_motor_id:
                    abort_event = _build_abort_event(
                        config,
                        frame=frame,
                        reference_velocity=float(reference_sample.velocity_cmd),
                        phase_name=str(reference_sample.phase_name),
                        target_motor_id=int(target_motor_id),
                        group_index=int(group_index),
                        round_index=int(round_index),
                        target_max_velocity=target_max_velocity,
                        target_max_torque=target_max_torque,
                    )
                    if abort_event is not None:
                        raise _SafetyAbortError(abort_event)
                    if reference_sample.phase_name == "hold_start":
                        command_raw = 0.0
                        current_target_command = 0.0
                    else:
                        command_raw, command_limited = controller.update(
                            target_motor_id,
                            reference_sample,
                            frame,
                            compensation=compensation,
                        )
                        current_target_command = command_adapter.limit_command(target_motor_id, command_limited)
                    command_packet = command_adapter.pack(target_motor_id, current_target_command)
                    transport.write(command_packet)
                    _log_live_command(current_target_command, float(reference_sample.velocity_cmd), command_packet)
                    target_frame_count += 1

                    time_log.append(float(elapsed_s))
                    motor_id_log.append(int(frame.motor_id))
                    position_log.append(float(frame.position))
                    velocity_log.append(float(frame.velocity))
                    torque_log.append(float(frame.torque))
                    command_raw_log.append(float(command_raw))
                    command_log.append(float(current_target_command))
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
                        command=float(current_target_command),
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
                    expected_velocity=expected_velocity,
                    target_torque=float(current_target_command if frame.motor_id == target_motor_id else 0.0),
                    feedback_torque=float(frame.torque),
                    state=int(frame.state),
                    mos_temperature=float(frame.mos_temperature),
                    phase_name=phase_name,
                )
                rerun_recorder.log_round_timing(
                    group_index=int(group_index),
                    round_index=int(round_index),
                    active_motor_id=int(target_motor_id),
                    planned_duration_s=planned_duration_s,
                    actual_capture_duration_s=float(elapsed_s),
                    sync_wait_duration_s=sync_wait_duration_s,
                    round_total_duration_s=time.monotonic() - sync_started_monotonic,
                )

            if capture_complete:
                break

            if not saw_frame and not chunk:
                time.sleep(max(float(config.serial.read_timeout), 1.0e-3))
    except KeyboardInterrupt:
        interrupted = True
    finally:
        command_packet = command_adapter.pack(target_motor_id, 0.0)
        transport.write(command_packet)
        _log_live_command(0.0, 0.0, command_packet)
        rerun_recorder.log_round_stop(
            group_index=int(group_index),
            round_index=int(round_index),
            motor_id=int(target_motor_id),
            phase_name=(
                "interrupted"
                if interrupted
                else ("aborted" if abort_event is not None else "round_complete")
            ),
        )

    actual_capture_duration_s = 0.0
    if synced_before_capture and capture_started_monotonic is not None:
        actual_capture_duration_s = min(time.monotonic() - capture_started_monotonic, planned_duration_s)
    round_total_duration_s = time.monotonic() - sync_started_monotonic
    rerun_recorder.log_round_timing(
        group_index=int(group_index),
        round_index=int(round_index),
        active_motor_id=int(target_motor_id),
        planned_duration_s=planned_duration_s,
        actual_capture_duration_s=actual_capture_duration_s,
        sync_wait_duration_s=sync_wait_duration_s,
        round_total_duration_s=round_total_duration_s,
    )

    target_frame_ratio = min(float(target_frame_count / target_frame_goal), 1.0) if target_frame_goal else 0.0
    stop_reason = "interrupted" if interrupted else "completed"
    if not synced_before_capture and not interrupted:
        stop_reason = "sync_timeout"
    metadata = {
        "mode": str(mode),
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
        "sync_required_target_frames": int(sync_required_target_frames),
        "target_sync_frame_count": int(target_sync_frame_count),
        "observed_frame_count": int(observed_frame_count),
        "sequence_error_count": 0,
        "sequence_error_ratio": 0.0,
        "target_frame_goal": int(target_frame_goal),
        "target_frame_count": int(target_frame_count),
        "target_frame_ratio": float(target_frame_ratio),
        "target_max_velocity": target_max_velocity,
        "target_max_torque": target_max_torque,
        "planned_duration_s": float(planned_duration_s),
        "actual_capture_duration_s": float(actual_capture_duration_s),
        "round_total_duration_s": float(round_total_duration_s),
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
    show_rerun_viewer: bool = False,
) -> SequentialRunResult:
    store = ResultStore(config, mode="identify")
    parser = SerialFrameParser(max_motor_id=max(config.motor_ids))
    command_adapter = SingleMotorCommandAdapter(
        motor_count=max(config.motor_ids),
        torque_limits=config.control.max_torque,
    )
    controller = SingleMotorController(config)
    rerun_recorder = RerunRecorder(
        store.rerun_recording_path,
        motor_ids=config.motor_ids,
        motor_names={motor_id: config.motors.name_for(motor_id) for motor_id in config.motor_ids},
        mode="identify",
        show_viewer=show_rerun_viewer,
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
                    "Starting identify round "
                    f"{current_round}/{total_rounds}: "
                    f"group={group_index}, motor_id={target_motor_id}"
                )
                capture, interrupted = _capture_round(
                    config=config,
                    transport=transport,
                    parser=parser,
                    command_adapter=command_adapter,
                    controller=controller,
                    target_motor_id=int(target_motor_id),
                    group_index=int(group_index),
                    round_index=int(current_round),
                    rerun_recorder=rerun_recorder,
                    mode="identify",
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
                log_info(
                    "Finished identify round "
                    f"group={group_index}, motor_id={target_motor_id}, "
                    f"planned={capture.metadata['planned_duration_s']:.3f}s, "
                    f"sync_wait={capture.metadata['sync_wait_duration_s']:.3f}s, "
                    f"capture={capture.metadata['actual_capture_duration_s']:.3f}s, "
                    f"total={capture.metadata['round_total_duration_s']:.3f}s"
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
        return SequentialRunResult(
            artifacts=tuple(artifacts),
            summary_paths=summary_paths,
            manifest_path=store.manifest_path,
        )
    except _SafetyAbortError as exc:
        store.record_abort_event(exc.event.to_payload())
        if artifacts:
            summary_paths = store.save_summary(artifacts)
            rerun_recorder.log_summary(
                summary_path=summary_paths.run_summary_path,
                report_path=summary_paths.run_summary_report_path,
            )
        else:
            store.finalize()
        raise
    finally:
        rerun_recorder.close()
        transport.close()


def run_compensation_validation(
    config: Config,
    *,
    transport_factory: Callable[[], SerialTransport] | None = None,
    show_rerun_viewer: bool = False,
    parameters_path: Path | None = None,
) -> SequentialRunResult:
    resolved_parameters_path, parameters_source, parameters_by_motor = _load_compensation_parameters(
        config,
        parameters_path=parameters_path,
    )
    log_info(
        "Compensation parameters source: "
        f"{parameters_source} ({resolved_parameters_path})"
    )
    store = ResultStore(config, mode="compensate")
    parser = SerialFrameParser(max_motor_id=max(config.motor_ids))
    command_adapter = SingleMotorCommandAdapter(
        motor_count=max(config.motor_ids),
        torque_limits=config.control.max_torque,
    )
    controller = SingleMotorController(config)
    rerun_recorder = RerunRecorder(
        store.rerun_recording_path,
        motor_ids=config.motor_ids,
        motor_names={motor_id: config.motors.name_for(motor_id) for motor_id in config.motor_ids},
        mode="compensate",
        show_viewer=show_rerun_viewer,
    )
    artifacts: list[RoundArtifact] = []

    for motor_id, parameters in parameters_by_motor.items():
        rerun_recorder.log_compensation_reference(
            motor_id=int(motor_id),
            parameters=parameters,
            parameters_path=resolved_parameters_path,
        )

    transport = transport_factory() if transport_factory is not None else open_serial_transport(config.serial)
    try:
        total_rounds = int(config.group_count) * len(config.enabled_motor_ids)
        current_round = 0
        stop_requested = False
        for group_index in range(1, int(config.group_count) + 1):
            for target_motor_id in config.enabled_motor_ids:
                current_round += 1
                log_info(
                    "Starting compensate round "
                    f"{current_round}/{total_rounds}: "
                    f"group={group_index}, motor_id={target_motor_id}"
                )
                capture, interrupted = _capture_round(
                    config=config,
                    transport=transport,
                    parser=parser,
                    command_adapter=command_adapter,
                    controller=controller,
                    target_motor_id=int(target_motor_id),
                    group_index=int(group_index),
                    round_index=int(current_round),
                    rerun_recorder=rerun_recorder,
                    mode="compensate",
                    compensation=parameters_by_motor[int(target_motor_id)],
                )
                capture = replace(
                    capture,
                    metadata={
                        **capture.metadata,
                        **_capture_compensation_metrics(capture),
                        "compensation_parameters_path": str(resolved_parameters_path),
                        "compensation_parameters": asdict(parameters_by_motor[int(target_motor_id)]),
                    },
                )
                capture_path = store.save_capture(capture)
                artifacts.append(
                    RoundArtifact(
                        capture=capture,
                        identification=None,
                        capture_path=capture_path,
                        identification_path=None,
                    )
                )
                log_info(
                    "Finished compensate round "
                    f"group={group_index}, motor_id={target_motor_id}, "
                    f"planned={capture.metadata['planned_duration_s']:.3f}s, "
                    f"sync_wait={capture.metadata['sync_wait_duration_s']:.3f}s, "
                    f"capture={capture.metadata['actual_capture_duration_s']:.3f}s, "
                    f"total={capture.metadata['round_total_duration_s']:.3f}s, "
                    f"velocity_rmse={capture.metadata['tracking_velocity_rmse']:.6f}"
                )
                if interrupted:
                    stop_requested = True
                    break
            if stop_requested:
                break
        store.finalize(compensation_parameters_path=resolved_parameters_path)
        return SequentialRunResult(
            artifacts=tuple(artifacts),
            summary_paths=None,
            manifest_path=store.manifest_path,
        )
    except _SafetyAbortError as exc:
        store.record_abort_event(exc.event.to_payload())
        store.finalize(compensation_parameters_path=resolved_parameters_path)
        raise
    finally:
        rerun_recorder.close()
        transport.close()
