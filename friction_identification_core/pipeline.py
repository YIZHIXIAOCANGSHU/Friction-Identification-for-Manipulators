from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from friction_identification_core.config import Config
from friction_identification_core.controller import SingleMotorController
from friction_identification_core.identification import identify_motor_friction, identify_motor_friction_lugre
from friction_identification_core.models import (
    MotorCompensationParameters,
    ReferenceSample,
    ReferenceTrajectory,
    RoundCapture,
)
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
    stage: str
    motor_id: int
    group_index: int
    round_index: int
    phase_name: str
    observed_velocity: float | None = None
    velocity_limit: float | None = None
    feedback_torque: float | None = None
    torque_limit: float | None = None
    feedback_position: float | None = None
    position_limit: float | None = None
    detail: str | None = None

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "reason": str(self.reason),
            "stage": str(self.stage),
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
        if self.feedback_position is not None:
            payload["feedback_position"] = float(self.feedback_position)
        if self.position_limit is not None:
            payload["position_limit"] = float(self.position_limit)
        if self.detail:
            payload["detail"] = str(self.detail)
        return payload

    def error_message(self) -> str:
        parts = [
            f"reason={self.reason}",
            f"stage={self.stage}",
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
        if self.feedback_position is not None:
            parts.append(f"feedback_position={self.feedback_position:.6f}")
        if self.position_limit is not None:
            parts.append(f"position_limit={self.position_limit:.6f}")
        if self.detail:
            parts.append(f"detail={self.detail}")
        return "Runtime abort: " + ", ".join(parts)


class _RuntimeAbortError(ValueError):
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


def _prebuild_references(config: Config) -> dict[int, ReferenceTrajectory]:
    references: dict[int, ReferenceTrajectory] = {}
    for motor_id in config.enabled_motor_ids:
        motor_index = config.motor_index(motor_id)
        references[int(motor_id)] = build_reference_trajectory(
            config.excitation,
            max_velocity=float(config.control.max_velocity[motor_index]),
        )
    return references


def _expected_velocity_vector(config: Config, *, target_index: int, target_velocity: float) -> np.ndarray:
    expected = np.zeros(config.motor_count, dtype=np.float64)
    expected[target_index] = float(target_velocity)
    return expected


def _expected_position_vector(config: Config, *, target_index: int, target_position: float) -> np.ndarray:
    expected = np.zeros(config.motor_count, dtype=np.float64)
    expected[target_index] = float(target_position)
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


def _send_zero_command(
    *,
    transport: SerialTransport,
    command_adapter: SingleMotorCommandAdapter,
    target_motor_id: int,
    rerun_recorder: RerunRecorder,
    config: Config,
    target_index: int,
) -> None:
    _send_target_command(
        transport=transport,
        command_adapter=command_adapter,
        target_motor_id=int(target_motor_id),
        target_command=0.0,
        rerun_recorder=rerun_recorder,
        config=config,
        target_index=target_index,
    )


def _send_target_command(
    *,
    transport: SerialTransport,
    command_adapter: SingleMotorCommandAdapter,
    target_motor_id: int,
    target_command: float,
    rerun_recorder: RerunRecorder,
    config: Config,
    target_index: int,
) -> None:
    packet = command_adapter.pack(int(target_motor_id), float(target_command))
    transport.write(packet)
    rerun_recorder.log_live_command_packet(
        sent_commands=_sent_command_vector(config, target_index=target_index, target_command=float(target_command)),
        expected_positions=_expected_position_vector(config, target_index=target_index, target_position=0.0),
        expected_velocities=_expected_velocity_vector(config, target_index=target_index, target_velocity=0.0),
        raw_packet=packet,
    )


def _take_control_with_zero_command(
    *,
    config: Config,
    transport: SerialTransport,
    parser: SerialFrameParser,
    command_adapter: SingleMotorCommandAdapter,
    rerun_recorder: RerunRecorder,
) -> None:
    target_motor_id = int(config.enabled_motor_ids[0])
    target_index = config.motor_index(target_motor_id)
    _send_zero_command(
        transport=transport,
        command_adapter=command_adapter,
        target_motor_id=target_motor_id,
        rerun_recorder=rerun_recorder,
        config=config,
        target_index=target_index,
    )
    if config.serial.flush_input_before_round:
        # Drop frames that may still reflect retained commands before this process took over.
        transport.reset_input_buffer()
        parser.reset()


def _phase_theoretical_velocity(
    reference: ReferenceTrajectory,
    *,
    phase_name: str,
    feedback_position: float,
    reference_index: int,
    zero_target_velocity_threshold: float,
) -> tuple[float, int]:
    phase_mask = np.asarray(reference.phase_name).astype(str) == str(phase_name)
    candidate_indices = np.flatnonzero(phase_mask)
    if candidate_indices.size == 0:
        return float(reference.velocity_cmd[reference_index]), int(reference_index)

    reference_velocity = float(reference.velocity_cmd[reference_index])
    if abs(reference_velocity) > float(zero_target_velocity_threshold):
        sign_mask = np.sign(reference.velocity_cmd[candidate_indices]) == np.sign(reference_velocity)
        if np.any(sign_mask):
            candidate_indices = candidate_indices[sign_mask]

    candidate_positions = np.asarray(reference.position_cmd[candidate_indices], dtype=np.float64)
    distances = np.abs(candidate_positions - float(feedback_position))
    min_distance = float(np.min(distances))
    closest_mask = np.isclose(distances, min_distance, rtol=0.0, atol=1.0e-12)
    closest_indices = candidate_indices[closest_mask]
    if closest_indices.size > 1:
        closest_indices = closest_indices[np.argsort(np.abs(closest_indices - int(reference_index)))]
    matched_index = int(closest_indices[0])
    return float(reference.velocity_cmd[matched_index]), matched_index


def _zeroing_theoretical_velocity_from_position(
    *,
    filtered_position: float,
    zeroing_position_gain: float,
    zeroing_velocity_gain: float,
    zeroing_hard_velocity_limit: float,
) -> float:
    if float(zeroing_position_gain) <= 1.0e-9:
        return 0.0
    if float(zeroing_velocity_gain) <= 1.0e-9:
        return float(zeroing_hard_velocity_limit)
    return min(
        float(zeroing_hard_velocity_limit),
        abs(float(zeroing_position_gain) * float(filtered_position) / float(zeroing_velocity_gain)),
    )


def _safety_margin_text(
    *,
    velocity_limit: float,
    observed_velocity: float,
    torque_limit: float,
    feedback_torque: float,
    position_limit: float,
    feedback_position: float,
) -> str:
    return (
        f"velocity_margin={velocity_limit - abs(float(observed_velocity)):+.6f}, "
        f"torque_margin={torque_limit - abs(float(feedback_torque)):+.6f}, "
        f"position_margin={position_limit - abs(float(feedback_position)):+.6f}"
    )


def _runtime_abort_from_frame(
    *,
    frame,
    stage: str,
    target_motor_id: int,
    group_index: int,
    round_index: int,
    phase_name: str,
    velocity_limit: float,
    torque_limit: float,
    position_limit: float,
) -> _AbortEvent | None:
    if abs(float(frame.velocity)) > float(velocity_limit):
        return _AbortEvent(
            reason="velocity_limit_exceeded",
            stage=stage,
            motor_id=int(target_motor_id),
            group_index=int(group_index),
            round_index=int(round_index),
            phase_name=str(phase_name),
            observed_velocity=float(frame.velocity),
            velocity_limit=float(velocity_limit),
        )
    if abs(float(frame.torque)) > float(torque_limit):
        return _AbortEvent(
            reason="torque_limit_exceeded",
            stage=stage,
            motor_id=int(target_motor_id),
            group_index=int(group_index),
            round_index=int(round_index),
            phase_name=str(phase_name),
            feedback_torque=float(frame.torque),
            torque_limit=float(torque_limit),
        )
    effective_position_limit = float(position_limit)
    if np.isfinite(effective_position_limit) and effective_position_limit > 0.0:
        if abs(float(frame.position)) > 1.10 * effective_position_limit:
            return _AbortEvent(
                reason="position_limit_exceeded",
                stage=stage,
                motor_id=int(target_motor_id),
                group_index=int(group_index),
                round_index=int(round_index),
                phase_name=str(phase_name),
                feedback_position=float(frame.position),
                position_limit=float(1.10 * effective_position_limit),
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
            raise ValueError("Compensation summary is missing fields: " + ", ".join(missing_fields))

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


def _perform_zeroing(
    *,
    config: Config,
    transport: SerialTransport,
    parser: SerialFrameParser,
    command_adapter: SingleMotorCommandAdapter,
    controller: SingleMotorController,
    rerun_recorder: RerunRecorder,
) -> None:
    for zeroing_index, target_motor_id in enumerate(config.enabled_motor_ids, start=1):
        if config.serial.flush_input_before_round:
            transport.reset_input_buffer()
            parser.reset()

        motor_index = config.motor_index(target_motor_id)
        max_torque = float(config.control.max_torque[motor_index])
        zeroing_position_gain = float(config.control.zeroing_position_gain[motor_index])
        zeroing_velocity_gain = float(config.control.zeroing_velocity_gain[motor_index])
        zeroing_hard_velocity_limit = float(config.control.zeroing_hard_velocity_limit[motor_index])
        zeroing_velocity_limit = float(config.control.zeroing_velocity_limit[motor_index])
        position_tolerance = float(config.control.zeroing_position_tolerance[motor_index])
        velocity_tolerance = float(config.control.zeroing_velocity_tolerance[motor_index])
        # Zeroing may need to recover from a pose outside the excitation envelope,
        # so do not reuse excitation.position_limit as a zeroing abort guard.
        position_abort_limit = float("nan")
        success_count = 0
        recent_position: deque[float] = deque(maxlen=5)
        recent_velocity: deque[float] = deque(maxlen=5)
        last_raw_position = float("nan")
        last_raw_velocity = float("nan")
        last_filtered_position = float("nan")
        last_filtered_velocity = float("nan")
        last_zeroing_command = 0.0
        zeroing_velocity_violation_streak = 0
        start_monotonic = time.monotonic()
        feedback_request_interval_s = max(float(config.serial.read_timeout), 5.0e-3)
        last_feedback_request_monotonic = 0.0

        def _request_feedback() -> None:
            nonlocal last_feedback_request_monotonic
            _send_target_command(
                transport=transport,
                command_adapter=command_adapter,
                target_motor_id=int(target_motor_id),
                target_command=float(last_zeroing_command),
                rerun_recorder=rerun_recorder,
                config=config,
                target_index=motor_index,
            )
            last_feedback_request_monotonic = time.monotonic()

        rerun_recorder.log_zeroing_event(
            event="zeroing_start",
            motor_id=int(target_motor_id),
            detail=f"index={zeroing_index}",
        )
        _request_feedback()

        while True:
            elapsed = time.monotonic() - start_monotonic
            if elapsed > float(config.control.zeroing_timeout):
                detail_parts = [f"elapsed={elapsed:.3f}"]
                if np.isfinite(last_raw_position):
                    detail_parts.append(f"raw_position={last_raw_position:.6f}")
                if np.isfinite(last_raw_velocity):
                    detail_parts.append(f"raw_velocity={last_raw_velocity:.6f}")
                if np.isfinite(last_filtered_position):
                    detail_parts.append(f"filtered_position={last_filtered_position:.6f}")
                if np.isfinite(last_filtered_velocity):
                    detail_parts.append(f"filtered_velocity={last_filtered_velocity:.6f}")
                detail_parts.append(
                    f"success_count={int(success_count)}/{int(config.control.zeroing_required_frames)}"
                )
                detail = ", ".join(detail_parts)
                _send_zero_command(
                    transport=transport,
                    command_adapter=command_adapter,
                    target_motor_id=int(target_motor_id),
                    rerun_recorder=rerun_recorder,
                    config=config,
                    target_index=motor_index,
                )
                rerun_recorder.log_zeroing_event(
                    event="zeroing_timeout",
                    motor_id=int(target_motor_id),
                    detail=detail,
                )
                raise _RuntimeAbortError(
                    _AbortEvent(
                        reason="zeroing_timeout",
                        stage="zeroing",
                        motor_id=int(target_motor_id),
                        group_index=0,
                        round_index=0,
                        phase_name="zeroing",
                        detail=detail,
                    )
                )

            chunk = transport.read(config.serial.read_chunk_size)
            if chunk:
                parser.feed(chunk)
            saw_frame = False
            saw_target_frame = False
            while True:
                frame = parser.pop_frame()
                if frame is None:
                    break
                saw_frame = True
                if int(frame.motor_id) != int(target_motor_id):
                    continue
                saw_target_frame = True

                last_raw_position = float(frame.position)
                last_raw_velocity = float(frame.velocity)
                recent_position.append(float(frame.position))
                recent_velocity.append(float(frame.velocity))
                filtered_position = float(np.median(np.asarray(recent_position, dtype=np.float64)))
                filtered_velocity = float(np.median(np.asarray(recent_velocity, dtype=np.float64)))
                last_filtered_position = filtered_position
                last_filtered_velocity = filtered_velocity
                zeroing_theoretical_velocity = _zeroing_theoretical_velocity_from_position(
                    filtered_position=filtered_position,
                    zeroing_position_gain=zeroing_position_gain,
                    zeroing_velocity_gain=zeroing_velocity_gain,
                    zeroing_hard_velocity_limit=zeroing_hard_velocity_limit,
                )
                # Keep a dedicated zeroing abort guard during the return motion and
                # only use zeroing_velocity_limit to decide when the final near-zero
                # lock criteria are allowed to engage.
                active_velocity_limit = zeroing_hard_velocity_limit
                inside_zeroing_velocity_window = float(zeroing_theoretical_velocity) <= float(zeroing_velocity_limit)
                if abs(float(frame.velocity)) > float(active_velocity_limit):
                    zeroing_velocity_violation_streak += 1
                else:
                    zeroing_velocity_violation_streak = 0
                abort_event = _runtime_abort_from_frame(
                    frame=frame,
                    stage="zeroing",
                    target_motor_id=int(target_motor_id),
                    group_index=0,
                    round_index=0,
                    phase_name="zeroing",
                    # Zeroing feedback can contain isolated velocity spikes even when
                    # the motor is stationary; require two consecutive violations
                    # before treating it as a true overspeed event.
                    velocity_limit=(
                        float(active_velocity_limit)
                        if zeroing_velocity_violation_streak >= 2
                        else float("inf")
                    ),
                    torque_limit=float(max_torque),
                    position_limit=float(position_abort_limit),
                )
                if abort_event is not None:
                    _send_zero_command(
                        transport=transport,
                        command_adapter=command_adapter,
                        target_motor_id=int(target_motor_id),
                        rerun_recorder=rerun_recorder,
                        config=config,
                        target_index=motor_index,
                    )
                    rerun_recorder.log_zeroing_event(
                        event="zeroing_abort",
                        motor_id=int(target_motor_id),
                        detail=abort_event.reason,
                    )
                    raise _RuntimeAbortError(abort_event)

                zero_reference = ReferenceSample(0.0, 0.0, 0.0, "zeroing")
                command_raw, command = controller.update(
                    int(target_motor_id),
                    zero_reference,
                    frame,
                    position_gain=zeroing_position_gain,
                    velocity_gain=zeroing_velocity_gain,
                )
                last_zeroing_command = float(command)
                _send_target_command(
                    transport=transport,
                    command_adapter=command_adapter,
                    target_motor_id=int(target_motor_id),
                    target_command=float(command),
                    rerun_recorder=rerun_recorder,
                    config=config,
                    target_index=motor_index,
                )

                inside_entry = (
                    inside_zeroing_velocity_window
                    and abs(filtered_position) <= position_tolerance
                    and abs(filtered_velocity) <= velocity_tolerance
                )
                outside_exit = (
                    abs(filtered_position) > 1.25 * position_tolerance
                    or abs(filtered_velocity) > 1.25 * velocity_tolerance
                )
                if inside_entry:
                    success_count += 1
                elif success_count > 0 and outside_exit:
                    success_count = 0

                rerun_recorder.log_zeroing_sample(
                    motor_id=int(target_motor_id),
                    raw_position=float(frame.position),
                    raw_velocity=float(frame.velocity),
                    filtered_position=filtered_position,
                    filtered_velocity=filtered_velocity,
                    position_error=float(-filtered_position),
                    velocity_error=float(-filtered_velocity),
                    success_count=int(success_count),
                    required_frames=int(config.control.zeroing_required_frames),
                    inside_entry_band=bool(inside_entry),
                    inside_exit_band=not bool(outside_exit),
                    command_raw=float(command_raw),
                    command=float(command),
                    feedback_torque=float(frame.torque),
                    torque_limit=float(max_torque),
                    velocity_limit=float(active_velocity_limit),
                    position_limit=float(position_abort_limit),
                )
                if success_count >= int(config.control.zeroing_required_frames):
                    _send_zero_command(
                        transport=transport,
                        command_adapter=command_adapter,
                        target_motor_id=int(target_motor_id),
                        rerun_recorder=rerun_recorder,
                        config=config,
                        target_index=motor_index,
                    )
                    rerun_recorder.log_zeroing_event(
                        event="zeroing_locked",
                        motor_id=int(target_motor_id),
                        detail=f"elapsed={elapsed:.3f}",
                    )
                    break
            if success_count >= int(config.control.zeroing_required_frames):
                break
            if not saw_target_frame and (time.monotonic() - last_feedback_request_monotonic) >= feedback_request_interval_s:
                _request_feedback()
            if not saw_frame and not chunk:
                time.sleep(max(float(config.serial.read_timeout), 1.0e-3))


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
    reference: ReferenceTrajectory,
    compensation: MotorCompensationParameters | None = None,
) -> RoundCapture:
    motor_name = config.motors.name_for(target_motor_id)
    target_index = config.motor_index(target_motor_id)
    target_max_velocity = float(config.control.max_velocity[target_index])
    target_max_torque = float(config.control.max_torque[target_index])
    position_limit = float(config.excitation.position_limit)
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
    sync_wait_duration_s = 0.0
    round_started_at = utc_now_iso8601()
    sync_started_monotonic = time.monotonic()
    capture_started_monotonic: float | None = None
    capture_started_at: str | None = None
    target_sync_frame_count = 0
    feedback_request_interval_s = max(float(config.serial.read_timeout), 5.0e-3)
    last_feedback_request_monotonic = 0.0

    def _request_sync_feedback() -> None:
        nonlocal last_feedback_request_monotonic
        _send_zero_command(
            transport=transport,
            command_adapter=command_adapter,
            target_motor_id=int(target_motor_id),
            rerun_recorder=rerun_recorder,
            config=config,
            target_index=target_index,
        )
        last_feedback_request_monotonic = time.monotonic()

    rerun_recorder.log_round_timing(
        group_index=int(group_index),
        round_index=int(round_index),
        active_motor_id=int(target_motor_id),
        planned_duration_s=planned_duration_s,
        actual_capture_duration_s=0.0,
        sync_wait_duration_s=0.0,
        round_total_duration_s=0.0,
    )
    _request_sync_feedback()

    try:
        while True:
            now = time.monotonic()
            if capture_started_monotonic is not None:
                elapsed_s = now - capture_started_monotonic
                if elapsed_s >= planned_duration_s:
                    break
            elif (now - sync_started_monotonic) >= float(config.serial.sync_timeout):
                raise _RuntimeAbortError(
                    _AbortEvent(
                        reason="sync_timeout",
                        stage=str(mode),
                        motor_id=int(target_motor_id),
                        group_index=int(group_index),
                        round_index=int(round_index),
                        phase_name="sync_wait",
                    )
                )

            chunk = transport.read(config.serial.read_chunk_size)
            if chunk:
                parser.feed(chunk)
            saw_frame = False
            saw_target_frame = False
            while True:
                frame = parser.pop_frame()
                if frame is None:
                    break
                saw_frame = True
                observed_frame_count += 1

                if capture_started_monotonic is None:
                    if int(frame.motor_id) == int(target_motor_id):
                        saw_target_frame = True
                        target_sync_frame_count += 1
                        _send_zero_command(
                            transport=transport,
                            command_adapter=command_adapter,
                            target_motor_id=int(target_motor_id),
                            rerun_recorder=rerun_recorder,
                            config=config,
                            target_index=target_index,
                        )
                    if target_sync_frame_count >= sync_required_target_frames:
                        sync_wait_duration_s = time.monotonic() - sync_started_monotonic
                        capture_started_monotonic = time.monotonic()
                        capture_started_at = utc_now_iso8601()
                    continue

                elapsed_s = time.monotonic() - capture_started_monotonic
                reference_index = reference.index_at(elapsed_s)
                reference_sample = reference.sample(elapsed_s)
                phase_name = str(reference_sample.phase_name) if int(frame.motor_id) == int(target_motor_id) else "idle"
                expected_velocity = float(reference_sample.velocity_cmd) if int(frame.motor_id) == int(target_motor_id) else 0.0
                expected_position = float(reference_sample.position_cmd) if int(frame.motor_id) == int(target_motor_id) else 0.0
                expected_acceleration = float(reference_sample.acceleration_cmd) if int(frame.motor_id) == int(target_motor_id) else 0.0
                command_raw = 0.0
                command = 0.0
                velocity_limit = float(config.control.low_speed_abort_limit[target_index])

                if int(frame.motor_id) == int(target_motor_id):
                    v_theory, _matched_index = _phase_theoretical_velocity(
                        reference,
                        phase_name=str(reference_sample.phase_name),
                        feedback_position=float(frame.position),
                        reference_index=reference_index,
                        zero_target_velocity_threshold=float(config.control.zero_target_velocity_threshold[target_index]),
                    )
                    velocity_limit = max(
                        float(config.control.low_speed_abort_limit[target_index]),
                        float(config.control.speed_abort_ratio[target_index]) * abs(float(v_theory)),
                    )
                    abort_event = _runtime_abort_from_frame(
                        frame=frame,
                        stage=str(mode),
                        target_motor_id=int(target_motor_id),
                        group_index=int(group_index),
                        round_index=int(round_index),
                        phase_name=str(reference_sample.phase_name),
                        velocity_limit=float(velocity_limit),
                        torque_limit=float(target_max_torque),
                        position_limit=float(position_limit),
                    )
                    if abort_event is not None:
                        raise _RuntimeAbortError(abort_event)

                    command_raw, limited_command = controller.update(
                        int(target_motor_id),
                        reference_sample,
                        frame,
                        compensation=compensation,
                    )
                    command = command_adapter.limit_command(int(target_motor_id), limited_command)
                    packet = command_adapter.pack(int(target_motor_id), command)
                    transport.write(packet)
                    rerun_recorder.log_live_command_packet(
                        sent_commands=_sent_command_vector(config, target_index=target_index, target_command=float(command)),
                        expected_positions=_expected_position_vector(
                            config,
                            target_index=target_index,
                            target_position=float(reference_sample.position_cmd),
                        ),
                        expected_velocities=_expected_velocity_vector(config, target_index=target_index, target_velocity=float(reference_sample.velocity_cmd)),
                        raw_packet=packet,
                    )
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

                rerun_recorder.log_live_motor_sample(
                    group_index=int(group_index),
                    round_index=int(round_index),
                    active_motor_id=int(target_motor_id),
                    motor_id=int(frame.motor_id),
                    position=float(frame.position),
                    velocity=float(frame.velocity),
                    feedback_torque=float(frame.torque),
                    command_raw=float(command_raw),
                    command=float(command),
                    reference_position=float(expected_position),
                    reference_velocity=float(expected_velocity),
                    reference_acceleration=float(expected_acceleration),
                    velocity_limit=float(velocity_limit),
                    torque_limit=float(target_max_torque),
                    position_limit=float(position_limit),
                    phase_name=str(phase_name),
                    stage=str(mode),
                    safety_margin_text=_safety_margin_text(
                        velocity_limit=float(velocity_limit),
                        observed_velocity=float(frame.velocity),
                        torque_limit=float(target_max_torque),
                        feedback_torque=float(frame.torque),
                        position_limit=float(position_limit),
                        feedback_position=float(frame.position),
                    ),
                )
                rerun_recorder.log_round_timing(
                    group_index=int(group_index),
                    round_index=int(round_index),
                    active_motor_id=int(target_motor_id),
                    planned_duration_s=planned_duration_s,
                    actual_capture_duration_s=float(elapsed_s),
                    sync_wait_duration_s=float(sync_wait_duration_s),
                    round_total_duration_s=float(time.monotonic() - sync_started_monotonic),
                )
            if capture_started_monotonic is None and not saw_target_frame:
                if (time.monotonic() - last_feedback_request_monotonic) >= feedback_request_interval_s:
                    _request_sync_feedback()
            if not saw_frame and not chunk:
                time.sleep(max(float(config.serial.read_timeout), 1.0e-3))
    except KeyboardInterrupt as exc:
        raise _RuntimeAbortError(
            _AbortEvent(
                reason="interrupted",
                stage=str(mode),
                motor_id=int(target_motor_id),
                group_index=int(group_index),
                round_index=int(round_index),
                phase_name="interrupted",
            )
        ) from exc
    finally:
        _send_zero_command(
            transport=transport,
            command_adapter=command_adapter,
            target_motor_id=int(target_motor_id),
            rerun_recorder=rerun_recorder,
            config=config,
            target_index=target_index,
        )

    actual_capture_duration_s = min(time.monotonic() - (capture_started_monotonic or time.monotonic()), planned_duration_s)
    round_total_duration_s = time.monotonic() - sync_started_monotonic
    rerun_recorder.log_round_stop(
        group_index=int(group_index),
        round_index=int(round_index),
        motor_id=int(target_motor_id),
        phase_name="completed",
        stage=str(mode),
    )

    target_frame_ratio = min(float(target_frame_count / target_frame_goal), 1.0) if target_frame_goal else 0.0
    metadata = {
        "mode": str(mode),
        "group_index": int(group_index),
        "round_index": int(round_index),
        "target_motor_id": int(target_motor_id),
        "enabled_motor_ids": list(config.enabled_motor_ids),
        "excitation_config": asdict(config.excitation),
        "start_time": capture_started_at or round_started_at,
        "round_start_time": round_started_at,
        "stop_reason": "completed",
        "synced_before_capture": True,
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
        "target_max_velocity": float(target_max_velocity),
        "target_max_torque": float(target_max_torque),
        "position_limit": float(position_limit),
        "planned_duration_s": float(planned_duration_s),
        "actual_capture_duration_s": float(actual_capture_duration_s),
        "round_total_duration_s": float(round_total_duration_s),
    }
    return RoundCapture(
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


def run_sequential_identification(
    config: Config,
    *,
    transport_factory: Callable[[], SerialTransport] | None = None,
    show_rerun_viewer: bool = False,
) -> SequentialRunResult:
    references = _prebuild_references(config)
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
        _take_control_with_zero_command(
            config=config,
            transport=transport,
            parser=parser,
            command_adapter=command_adapter,
            rerun_recorder=rerun_recorder,
        )
        _perform_zeroing(
            config=config,
            transport=transport,
            parser=parser,
            command_adapter=command_adapter,
            controller=controller,
            rerun_recorder=rerun_recorder,
        )
        total_rounds = int(config.group_count) * len(config.enabled_motor_ids)
        current_round = 0
        for group_index in range(1, int(config.group_count) + 1):
            for target_motor_id in config.enabled_motor_ids:
                current_round += 1
                log_info(
                    "Starting identify round "
                    f"{current_round}/{total_rounds}: "
                    f"group={group_index}, motor_id={target_motor_id}"
                )
                capture = _capture_round(
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
                    reference=references[int(target_motor_id)],
                )
                identification = identify_motor_friction(
                    config.identification,
                    capture,
                    max_torque=float(config.control.max_torque[config.motor_index(target_motor_id)]),
                    max_velocity=float(config.control.max_velocity[config.motor_index(target_motor_id)]),
                )
                dynamic_identification = identify_motor_friction_lugre(
                    config.identification,
                    capture,
                    identification,
                )
                capture_path = store.save_capture(capture)
                identification_path = store.save_identification(capture, identification)
                dynamic_identification_path = store.save_dynamic_identification(capture, dynamic_identification)
                rerun_recorder.log_identification(capture, identification, dynamic_identification)
                artifacts.append(
                    RoundArtifact(
                        capture=capture,
                        identification=identification,
                        dynamic_identification=dynamic_identification,
                        capture_path=capture_path,
                        identification_path=identification_path,
                        dynamic_identification_path=dynamic_identification_path,
                    )
                )
        summary_paths = store.save_summary(artifacts)
        rerun_recorder.log_summary(
            summary_path=summary_paths.run_summary_path,
            report_path=summary_paths.run_summary_report_path,
            dynamic_summary_path=summary_paths.dynamic_run_summary_path,
            dynamic_report_path=summary_paths.dynamic_run_summary_report_path,
        )
        return SequentialRunResult(
            artifacts=tuple(artifacts),
            summary_paths=summary_paths,
            manifest_path=store.manifest_path,
        )
    except _RuntimeAbortError as exc:
        rerun_recorder.log_abort_event(exc.event.to_payload())
        store.record_abort_event(exc.event.to_payload())
        if artifacts:
            summary_paths = store.save_summary(artifacts)
            rerun_recorder.log_summary(
                summary_path=summary_paths.run_summary_path,
                report_path=summary_paths.run_summary_report_path,
                dynamic_summary_path=summary_paths.dynamic_run_summary_path,
                dynamic_report_path=summary_paths.dynamic_run_summary_report_path,
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
    references = _prebuild_references(config)
    resolved_parameters_path, parameters_source, parameters_by_motor = _load_compensation_parameters(
        config,
        parameters_path=parameters_path,
    )
    log_info("Compensation parameters source: " f"{parameters_source} ({resolved_parameters_path})")
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
        for group_index in range(1, int(config.group_count) + 1):
            for target_motor_id in config.enabled_motor_ids:
                current_round += 1
                log_info(
                    "Starting compensate round "
                    f"{current_round}/{total_rounds}: "
                    f"group={group_index}, motor_id={target_motor_id}"
                )
                capture = _capture_round(
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
                    reference=references[int(target_motor_id)],
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
                        dynamic_identification=None,
                        capture_path=capture_path,
                        identification_path=None,
                        dynamic_identification_path=None,
                    )
                )
        store.finalize(compensation_parameters_path=resolved_parameters_path)
        return SequentialRunResult(
            artifacts=tuple(artifacts),
            summary_paths=None,
            manifest_path=store.manifest_path,
        )
    except _RuntimeAbortError as exc:
        rerun_recorder.log_abort_event(exc.event.to_payload())
        store.record_abort_event(exc.event.to_payload())
        store.finalize(compensation_parameters_path=resolved_parameters_path)
        raise
    finally:
        rerun_recorder.close()
        transport.close()
