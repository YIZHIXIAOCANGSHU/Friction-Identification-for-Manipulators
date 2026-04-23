from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from friction_identification_core.models import (
    MotorCompensationParameters,
    MotorDynamicIdentificationResult,
    MotorIdentificationResult,
    RoundCapture,
)

try:
    import rerun as rr
except ImportError:  # pragma: no cover - optional at import time
    rr = None


class RerunRecorder:
    def __init__(
        self,
        recording_path: Path,
        *,
        motor_ids: tuple[int, ...],
        motor_names: Mapping[int, str],
        mode: str = "identify",
        show_viewer: bool = False,
    ) -> None:
        self.recording_path = Path(recording_path)
        self._mode = str(mode)
        self._motor_ids = tuple(int(motor_id) for motor_id in motor_ids)
        self._motor_names = {int(motor_id): str(motor_names[motor_id]) for motor_id in self._motor_ids}
        self._motor_index = {motor_id: index for index, motor_id in enumerate(self._motor_ids)}
        self._recording = None if rr is None else rr.RecordingStream("friction_identification")
        self._initialized_paths: set[str] = set()
        self._sequence = 0
        motor_count = len(self._motor_ids)
        self._state = {
            "stage": "zeroing" if self._mode == "identify" else self._mode,
            "group_index": 0,
            "round_index": 0,
            "active_motor_id": 0,
            "current_phase": "-",
            "reference_position": 0.0,
            "reference_velocity": 0.0,
            "reference_acceleration": 0.0,
            "velocity_limit": np.nan,
            "torque_limit": np.nan,
            "position_limit": np.nan,
            "last_abort_reason": "-",
            "planned_duration_s": np.nan,
            "actual_capture_duration_s": np.nan,
            "sync_wait_duration_s": np.nan,
            "round_total_duration_s": np.nan,
        }
        self._live_position = np.full(motor_count, np.nan, dtype=np.float64)
        self._live_velocity = np.full(motor_count, np.nan, dtype=np.float64)
        self._live_feedback_torque = np.full(motor_count, np.nan, dtype=np.float64)
        self._live_command_raw = np.zeros(motor_count, dtype=np.float64)
        self._live_command = np.zeros(motor_count, dtype=np.float64)
        self._live_reference_position = np.zeros(motor_count, dtype=np.float64)
        self._live_reference_velocity = np.zeros(motor_count, dtype=np.float64)
        self._live_reference_acceleration = np.zeros(motor_count, dtype=np.float64)
        self._live_velocity_limit = np.full(motor_count, np.nan, dtype=np.float64)
        self._live_torque_limit = np.full(motor_count, np.nan, dtype=np.float64)
        self._live_position_limit = np.full(motor_count, np.nan, dtype=np.float64)
        self._live_phase = ["-"] * motor_count
        self._live_safety_margin = ["-"] * motor_count
        self._zeroing_summary = {
            motor_id: {
                "raw_position": np.nan,
                "raw_velocity": np.nan,
                "filtered_position": np.nan,
                "filtered_velocity": np.nan,
                "position_error": np.nan,
                "velocity_error": np.nan,
                "success_count": 0,
                "required_frames": 0,
                "inside_entry_band": False,
                "inside_exit_band": False,
            }
            for motor_id in self._motor_ids
        }
        self._latest_identification: dict[int, dict[str, object]] = {}
        self._latest_dynamic_identification: dict[int, dict[str, object]] = {}
        self._latest_raw_command_packet = b""
        if self._recording is None:
            return
        self._recording.save(self.recording_path)
        if show_viewer:
            self._spawn_viewer()
        self._send_default_blueprint()
        self._refresh_snapshot()

    def _spawn_viewer(self) -> None:
        if self._recording is None:
            return
        spawn = getattr(self._recording, "spawn", None)
        if callable(spawn):
            try:
                spawn(connect=True, detach_process=True)
            except Exception:
                return

    def _motor_entity_name(self, motor_id: int) -> str:
        return f"motor_{int(motor_id):02d}"

    def _live_motor_root(self, motor_id: int) -> str:
        return f"live/motors/{self._motor_entity_name(motor_id)}"

    def _zeroing_motor_root(self, motor_id: int) -> str:
        return f"live/zeroing/{self._motor_entity_name(motor_id)}"

    def _round_root(self, *, group_index: int, round_index: int, motor_id: int) -> str:
        return f"rounds/group_{int(group_index):02d}/motor_{int(motor_id):02d}/round_{int(round_index):02d}"

    def _send_default_blueprint(self) -> None:
        if self._recording is None:
            return
        rrb = rr.blueprint
        by_motor_tabs = [
            rrb.Vertical(
                rrb.TextDocumentView(origin=f"/{self._live_motor_root(motor_id)}/status", name="Status"),
                rrb.Horizontal(
                    rrb.TimeSeriesView(origin=f"/{self._live_motor_root(motor_id)}/signals/position", name="Position"),
                    rrb.TimeSeriesView(origin=f"/{self._live_motor_root(motor_id)}/signals/velocity", name="Velocity"),
                    rrb.TimeSeriesView(origin=f"/{self._live_motor_root(motor_id)}/signals/torque", name="Torque"),
                ),
                name=f"M{int(motor_id):02d} {self._motor_names[int(motor_id)]}",
            )
            for motor_id in self._motor_ids
        ]
        blueprint = rrb.Blueprint(
            rrb.Tabs(
                rrb.Vertical(
                    rrb.TextDocumentView(origin="/live/overview/current_state", name="Overview"),
                    rrb.TextDocumentView(origin="/live/overview/raw_command_packet", name="Raw Packet"),
                    name="Overview",
                ),
                rrb.Vertical(
                    *[
                        rrb.TextDocumentView(origin=f"/{self._zeroing_motor_root(motor_id)}/status", name=f"M{int(motor_id):02d}")
                        for motor_id in self._motor_ids
                    ],
                    name="Zeroing",
                ),
                rrb.Tabs(*by_motor_tabs, name="Live"),
                rrb.Vertical(
                    rrb.TextDocumentView(origin="/summary/static", name="Static Summary"),
                    rrb.TextDocumentView(origin="/summary/dynamic", name="Dynamic Summary"),
                    name="Summary",
                ),
            ),
            auto_views=False,
        )
        self._recording.send_blueprint(blueprint, make_active=True, make_default=True)

    def _format_float(self, value: float, *, digits: int = 6) -> str:
        value = float(value)
        if not np.isfinite(value):
            return "-"
        return f"{value:.{digits}f}"

    def _ensure_live_series(self) -> None:
        if self._recording is None:
            return
        for motor_id in self._motor_ids:
            root = self._live_motor_root(motor_id)
            series = {
                f"{root}/signals/position": ["feedback", "reference"],
                f"{root}/signals/velocity": ["actual", "theoretical", "velocity_limit"],
                f"{root}/signals/torque": ["raw_command", "sent_command", "feedback", "torque_limit"],
            }
            for path, names in series.items():
                if path in self._initialized_paths:
                    continue
                self._recording.log(path, rr.SeriesLines(names=names), static=True)
                self._initialized_paths.add(path)
        for motor_id in self._motor_ids:
            root = self._zeroing_motor_root(motor_id)
            zeroing_series = {
                f"{root}/signals/raw_position": ["raw_position"],
                f"{root}/signals/raw_velocity": ["raw_velocity"],
                f"{root}/signals/filtered_position": ["filtered_position"],
                f"{root}/signals/filtered_velocity": ["filtered_velocity"],
                f"{root}/signals/success_count": ["success_count", "required_frames"],
            }
            for path, names in zeroing_series.items():
                if path in self._initialized_paths:
                    continue
                self._recording.log(path, rr.SeriesLines(names=names), static=True)
                self._initialized_paths.add(path)

    def _set_time(self) -> None:
        if self._recording is None:
            return
        self._recording.set_time("sample", sequence=int(self._sequence))
        self._sequence += 1

    def _overview_markdown(self) -> str:
        state = self._state
        lines = [
            "# Live Overview",
            "",
            f"- stage: `{state['stage']}`",
            f"- active_motor_id: `{int(state['active_motor_id'])}`",
            f"- active_phase: `{state['current_phase']}`",
            f"- reference_position: `{self._format_float(float(state['reference_position']))}`",
            f"- reference_velocity: `{self._format_float(float(state['reference_velocity']))}`",
            f"- reference_acceleration: `{self._format_float(float(state['reference_acceleration']))}`",
            f"- dynamic_velocity_threshold: `{self._format_float(float(state['velocity_limit']))}`",
            f"- torque_limit: `{self._format_float(float(state['torque_limit']))}`",
            f"- position_limit: `{self._format_float(float(state['position_limit']))}`",
            f"- last_abort_reason: `{state['last_abort_reason']}`",
            f"- planned_duration_s: `{self._format_float(float(state['planned_duration_s']))}`",
            f"- actual_capture_duration_s: `{self._format_float(float(state['actual_capture_duration_s']))}`",
            f"- sync_wait_duration_s: `{self._format_float(float(state['sync_wait_duration_s']))}`",
            f"- round_total_duration_s: `{self._format_float(float(state['round_total_duration_s']))}`",
        ]
        return "\n".join(lines)

    def _raw_packet_markdown(self) -> str:
        packet = bytes(self._latest_raw_command_packet)
        return "\n".join(
            [
                "# Raw Command Packet",
                "",
                f"- packet_size_bytes: `{len(packet)}`",
                "",
                "```text",
                packet.hex(" ").upper() if packet else "-",
                "```",
            ]
        )

    def _motor_status_markdown(self, motor_id: int) -> str:
        index = self._motor_index[int(motor_id)]
        static_result = self._latest_identification.get(int(motor_id), {})
        dynamic_result = self._latest_dynamic_identification.get(int(motor_id), {})
        lines = [
            f"# Motor {int(motor_id):02d} {self._motor_names[int(motor_id)]}",
            "",
            f"- feedback_position: `{self._format_float(self._live_position[index])}`",
            f"- reference_position: `{self._format_float(self._live_reference_position[index])}`",
            f"- actual_velocity: `{self._format_float(self._live_velocity[index])}`",
            f"- theoretical_velocity: `{self._format_float(self._live_reference_velocity[index])}`",
            f"- velocity_limit: `{self._format_float(self._live_velocity_limit[index])}`",
            f"- raw_command: `{self._format_float(self._live_command_raw[index])}`",
            f"- sent_command: `{self._format_float(self._live_command[index])}`",
            f"- feedback_torque: `{self._format_float(self._live_feedback_torque[index])}`",
            f"- torque_limit: `{self._format_float(self._live_torque_limit[index])}`",
            f"- phase: `{self._live_phase[index]}`",
            f"- safety_margin: `{self._live_safety_margin[index]}`",
        ]
        if static_result:
            lines.extend(
                [
                    f"- static_status: `{static_result.get('status', '-')}`",
                    f"- static_valid_rmse: `{self._format_float(float(static_result.get('valid_rmse', np.nan)))}`",
                    f"- static_conclusion: `{static_result.get('conclusion_level', '-')}`",
                ]
            )
        if dynamic_result:
            lines.extend(
                [
                    f"- dynamic_status: `{dynamic_result.get('status', '-')}`",
                    f"- dynamic_valid_rmse: `{self._format_float(float(dynamic_result.get('valid_rmse', np.nan)))}`",
                ]
            )
        return "\n".join(lines)

    def _zeroing_status_markdown(self, motor_id: int) -> str:
        item = self._zeroing_summary[int(motor_id)]
        return "\n".join(
            [
                f"# Zeroing Motor {int(motor_id):02d} {self._motor_names[int(motor_id)]}",
                "",
                f"- raw_position: `{self._format_float(float(item['raw_position']))}`",
                f"- raw_velocity: `{self._format_float(float(item['raw_velocity']))}`",
                f"- filtered_position: `{self._format_float(float(item['filtered_position']))}`",
                f"- filtered_velocity: `{self._format_float(float(item['filtered_velocity']))}`",
                f"- position_error: `{self._format_float(float(item['position_error']))}`",
                f"- velocity_error: `{self._format_float(float(item['velocity_error']))}`",
                f"- success_count: `{int(item['success_count'])}`",
                f"- required_frames: `{int(item['required_frames'])}`",
                f"- inside_entry_band: `{bool(item['inside_entry_band'])}`",
                f"- inside_exit_band: `{bool(item['inside_exit_band'])}`",
            ]
        )

    def _refresh_snapshot(self) -> None:
        if self._recording is None:
            return
        self._recording.log(
            "live/overview/current_state",
            rr.TextDocument(self._overview_markdown(), media_type="text/markdown"),
        )
        self._recording.log(
            "live/overview/raw_command_packet",
            rr.TextDocument(self._raw_packet_markdown(), media_type="text/markdown"),
        )
        for motor_id in self._motor_ids:
            self._recording.log(
                f"{self._live_motor_root(motor_id)}/status",
                rr.TextDocument(self._motor_status_markdown(motor_id), media_type="text/markdown"),
            )
            self._recording.log(
                f"{self._zeroing_motor_root(motor_id)}/status",
                rr.TextDocument(self._zeroing_status_markdown(motor_id), media_type="text/markdown"),
            )

    def log_round_timing(
        self,
        *,
        group_index: int,
        round_index: int,
        active_motor_id: int,
        planned_duration_s: float,
        actual_capture_duration_s: float,
        sync_wait_duration_s: float,
        round_total_duration_s: float,
    ) -> None:
        self._state.update(
            {
                "group_index": int(group_index),
                "round_index": int(round_index),
                "active_motor_id": int(active_motor_id),
                "planned_duration_s": float(planned_duration_s),
                "actual_capture_duration_s": float(actual_capture_duration_s),
                "sync_wait_duration_s": float(sync_wait_duration_s),
                "round_total_duration_s": float(round_total_duration_s),
            }
        )
        self._refresh_snapshot()

    def log_live_command_packet(
        self,
        *,
        sent_commands: np.ndarray,
        expected_positions: np.ndarray,
        expected_velocities: np.ndarray,
        raw_packet: bytes | None = None,
    ) -> None:
        if self._recording is None:
            return
        self._ensure_live_series()
        sent_commands = np.asarray(sent_commands, dtype=np.float64).reshape(-1)
        expected_positions = np.asarray(expected_positions, dtype=np.float64).reshape(-1)
        expected_velocities = np.asarray(expected_velocities, dtype=np.float64).reshape(-1)
        self._live_command[:] = sent_commands
        self._live_reference_position[:] = expected_positions
        self._live_reference_velocity[:] = expected_velocities
        active_motor_id = int(self._state.get("active_motor_id", 0))
        active_index = self._motor_index.get(active_motor_id)
        if active_index is not None:
            self._state["reference_position"] = float(self._live_reference_position[active_index])
            self._state["reference_velocity"] = float(self._live_reference_velocity[active_index])
        if raw_packet is not None:
            self._latest_raw_command_packet = bytes(raw_packet)
        self._set_time()
        for index, motor_id in enumerate(self._motor_ids):
            root = self._live_motor_root(motor_id)
            self._recording.log(
                f"{root}/signals/position",
                rr.Scalars(
                    [
                        float(self._live_position[index]),
                        float(self._live_reference_position[index]),
                    ]
                ),
            )
            self._recording.log(
                f"{root}/signals/torque",
                rr.Scalars(
                    [
                        float(self._live_command_raw[index]),
                        float(self._live_command[index]),
                        float(self._live_feedback_torque[index]),
                        float(self._live_torque_limit[index]),
                    ]
                ),
            )
            self._recording.log(
                f"{root}/signals/velocity",
                rr.Scalars(
                    [
                        float(self._live_velocity[index]),
                        float(self._live_reference_velocity[index]),
                        float(self._live_velocity_limit[index]),
                    ]
                ),
            )
        self._refresh_snapshot()

    def log_live_motor_sample(
        self,
        *,
        group_index: int,
        round_index: int,
        active_motor_id: int,
        motor_id: int,
        position: float,
        velocity: float,
        feedback_torque: float,
        command_raw: float,
        command: float,
        reference_position: float,
        reference_velocity: float,
        reference_acceleration: float,
        velocity_limit: float,
        torque_limit: float,
        position_limit: float,
        phase_name: str,
        stage: str,
        safety_margin_text: str,
    ) -> None:
        if self._recording is None:
            return
        self._ensure_live_series()
        motor_id = int(motor_id)
        active_motor_id = int(active_motor_id)
        index = self._motor_index[motor_id]
        self._state.update(
            {
                "stage": str(stage),
                "group_index": int(group_index),
                "round_index": int(round_index),
                "active_motor_id": active_motor_id,
            }
        )
        if motor_id == active_motor_id:
            self._state.update(
                {
                    "current_phase": str(phase_name),
                    "reference_position": float(reference_position),
                    "reference_velocity": float(reference_velocity),
                    "reference_acceleration": float(reference_acceleration),
                    "velocity_limit": float(velocity_limit),
                    "torque_limit": float(torque_limit),
                    "position_limit": float(position_limit),
                }
            )
        self._live_position[index] = float(position)
        self._live_velocity[index] = float(velocity)
        self._live_feedback_torque[index] = float(feedback_torque)
        self._live_command_raw[index] = float(command_raw)
        self._live_command[index] = float(command)
        self._live_reference_position[index] = float(reference_position)
        self._live_reference_velocity[index] = float(reference_velocity)
        self._live_reference_acceleration[index] = float(reference_acceleration)
        self._live_velocity_limit[index] = float(velocity_limit)
        self._live_torque_limit[index] = float(torque_limit)
        self._live_position_limit[index] = float(position_limit)
        self._live_phase[index] = str(phase_name)
        self._live_safety_margin[index] = str(safety_margin_text)
        self._set_time()
        root = self._live_motor_root(motor_id)
        self._recording.log(
            f"{root}/signals/position",
            rr.Scalars([float(position), float(reference_position)]),
        )
        self._recording.log(
            f"{root}/signals/velocity",
            rr.Scalars([float(velocity), float(reference_velocity), float(velocity_limit)]),
        )
        self._recording.log(
            f"{root}/signals/torque",
            rr.Scalars([float(command_raw), float(command), float(feedback_torque), float(torque_limit)]),
        )
        self._refresh_snapshot()

    def log_zeroing_sample(
        self,
        *,
        motor_id: int,
        raw_position: float,
        raw_velocity: float,
        filtered_position: float,
        filtered_velocity: float,
        position_error: float,
        velocity_error: float,
        success_count: int,
        required_frames: int,
        inside_entry_band: bool,
        inside_exit_band: bool,
        command_raw: float,
        command: float,
        feedback_torque: float,
        torque_limit: float,
        velocity_limit: float,
        position_limit: float,
    ) -> None:
        if self._recording is None:
            return
        self._ensure_live_series()
        motor_id = int(motor_id)
        self._state["stage"] = "zeroing"
        self._state["active_motor_id"] = int(motor_id)
        self._state["current_phase"] = "zeroing"
        self._state["reference_position"] = 0.0
        self._state["reference_velocity"] = 0.0
        self._state["reference_acceleration"] = 0.0
        self._state["velocity_limit"] = float(velocity_limit)
        self._state["torque_limit"] = float(torque_limit)
        self._state["position_limit"] = float(position_limit)
        self._zeroing_summary[motor_id] = {
            "raw_position": float(raw_position),
            "raw_velocity": float(raw_velocity),
            "filtered_position": float(filtered_position),
            "filtered_velocity": float(filtered_velocity),
            "position_error": float(position_error),
            "velocity_error": float(velocity_error),
            "success_count": int(success_count),
            "required_frames": int(required_frames),
            "inside_entry_band": bool(inside_entry_band),
            "inside_exit_band": bool(inside_exit_band),
        }
        index = self._motor_index[motor_id]
        self._live_position[index] = float(raw_position)
        self._live_velocity[index] = float(raw_velocity)
        self._live_feedback_torque[index] = float(feedback_torque)
        self._live_command_raw[index] = float(command_raw)
        self._live_command[index] = float(command)
        self._live_velocity_limit[index] = float(velocity_limit)
        self._live_torque_limit[index] = float(torque_limit)
        self._live_position_limit[index] = float(position_limit)
        self._live_phase[index] = "zeroing"
        self._live_safety_margin[index] = (
            f"zeroing_count={int(success_count)}/{int(required_frames)}, "
            f"entry={bool(inside_entry_band)}, exit={bool(inside_exit_band)}"
        )
        self._set_time()
        root = self._zeroing_motor_root(motor_id)
        self._recording.log(f"{root}/signals/raw_position", rr.Scalars([float(raw_position)]))
        self._recording.log(f"{root}/signals/raw_velocity", rr.Scalars([float(raw_velocity)]))
        self._recording.log(f"{root}/signals/filtered_position", rr.Scalars([float(filtered_position)]))
        self._recording.log(f"{root}/signals/filtered_velocity", rr.Scalars([float(filtered_velocity)]))
        self._recording.log(
            f"{root}/signals/success_count",
            rr.Scalars([float(success_count), float(required_frames)]),
        )
        self._refresh_snapshot()

    def log_zeroing_event(self, *, event: str, motor_id: int, detail: str = "") -> None:
        if self._recording is None:
            return
        self._state["stage"] = "zeroing"
        self._state["active_motor_id"] = int(motor_id)
        self._recording.log(
            f"{self._zeroing_motor_root(int(motor_id))}/events",
            rr.TextLog(f"{event}: {detail}".strip()),
        )
        self._refresh_snapshot()

    def log_abort_event(self, payload: dict[str, object]) -> None:
        self._state["stage"] = "aborted"
        self._state["last_abort_reason"] = str(payload.get("reason", "-"))
        if self._recording is None:
            return
        self._recording.log(
            "live/overview/abort_event",
            rr.TextDocument(str(payload), media_type="text/plain"),
        )
        self._refresh_snapshot()

    def log_round_stop(
        self,
        *,
        group_index: int,
        round_index: int,
        motor_id: int,
        phase_name: str,
        stage: str,
    ) -> None:
        self._state["stage"] = "completed" if str(phase_name) == "completed" else str(stage)
        self._state["group_index"] = int(group_index)
        self._state["round_index"] = int(round_index)
        self._state["active_motor_id"] = int(motor_id)
        self._state["current_phase"] = str(phase_name)
        if self._recording is None:
            return
        self._recording.log(
            f"{self._round_root(group_index=group_index, round_index=round_index, motor_id=motor_id)}/events",
            rr.TextLog(str(phase_name)),
        )
        self._refresh_snapshot()

    def log_identification(
        self,
        capture: RoundCapture,
        result: MotorIdentificationResult,
        dynamic_result: MotorDynamicIdentificationResult,
    ) -> None:
        self._latest_identification[int(capture.target_motor_id)] = {
            "status": str(result.metadata.get("status", "unknown")),
            "valid_rmse": float(result.valid_rmse),
            "conclusion_level": str(result.metadata.get("conclusion_level", "-")),
        }
        self._latest_dynamic_identification[int(capture.target_motor_id)] = {
            "status": str(dynamic_result.metadata.get("status", "unknown")),
            "valid_rmse": float(dynamic_result.valid_rmse),
        }
        if self._recording is None:
            return
        round_root = self._round_root(
            group_index=int(capture.group_index),
            round_index=int(capture.round_index),
            motor_id=int(capture.target_motor_id),
        )
        for path, names in {
            f"{round_root}/masks": ["window", "selected", "train", "valid", "tracking_ok", "saturation_ok"],
            f"{round_root}/static_residual": ["static_residual"],
            f"{round_root}/dynamic_residual": ["dynamic_residual"],
        }.items():
            if path not in self._initialized_paths:
                self._recording.log(path, rr.SeriesLines(names=names), static=True)
                self._initialized_paths.add(path)

        phase_names = np.asarray(capture.phase_name).astype(str)
        cycle_labels = [phase for phase in phase_names if str(phase).startswith("excitation_cycle_")]
        for cycle_label in tuple(dict.fromkeys(cycle_labels)):
            self._recording.log(f"{round_root}/cycle_events", rr.TextLog(str(cycle_label)))

        velocity_band_text = (
            f"train={','.join(result.metadata.get('train_velocity_bands', [])) or '-'}; "
            f"valid={','.join(result.metadata.get('valid_velocity_bands', [])) or '-'}"
        )
        self._recording.log(f"{round_root}/velocity_bands", rr.TextLog(velocity_band_text))
        for sample_index in range(capture.sample_count):
            self._set_time()
            self._recording.log(
                f"{round_root}/masks",
                rr.Scalars(
                    [
                        float(result.identification_window_mask[sample_index]),
                        float(result.sample_mask[sample_index]),
                        float(result.train_mask[sample_index]),
                        float(result.valid_mask[sample_index]),
                        float(result.tracking_ok_mask[sample_index]),
                        float(result.saturation_ok_mask[sample_index]),
                    ]
                ),
            )
            self._recording.log(
                f"{round_root}/static_residual",
                rr.Scalars([float(result.torque_target[sample_index] - result.torque_pred[sample_index])]),
            )
            dynamic_residual = np.nan
            if np.isfinite(dynamic_result.torque_pred[sample_index]):
                dynamic_residual = float(dynamic_result.torque_target[sample_index] - dynamic_result.torque_pred[sample_index])
            self._recording.log(
                f"{round_root}/dynamic_residual",
                rr.Scalars([float(dynamic_residual)]),
            )
        self._recording.log(
            f"{round_root}/quality",
            rr.TextDocument(
                "\n".join(
                    [
                        "# Validation",
                        "",
                        f"- static_valid_rmse: `{self._format_float(float(result.valid_rmse))}`",
                        f"- dynamic_valid_rmse: `{self._format_float(float(dynamic_result.valid_rmse))}`",
                        f"- validation_mode: `{result.metadata.get('validation_mode', '-')}`",
                        f"- dynamic_validation_mode: `{dynamic_result.metadata.get('validation_mode', '-')}`",
                    ]
                ),
                media_type="text/markdown",
            ),
        )
        self._refresh_snapshot()

    def log_compensation_reference(
        self,
        *,
        motor_id: int,
        parameters: MotorCompensationParameters,
        parameters_path: Path,
    ) -> None:
        self._latest_identification[int(motor_id)] = {
            "status": "loaded",
            "valid_rmse": np.nan,
            "conclusion_level": str(parameters_path),
        }
        if self._recording is None:
            return
        self._recording.log(
            f"{self._live_motor_root(int(motor_id))}/compensation",
            rr.TextDocument(
                "\n".join(
                    [
                        f"# Compensation Motor {int(motor_id):02d}",
                        "",
                        f"- parameters_path: `{parameters_path}`",
                        f"- coulomb: `{self._format_float(float(parameters.coulomb))}`",
                        f"- viscous: `{self._format_float(float(parameters.viscous))}`",
                        f"- offset: `{self._format_float(float(parameters.offset))}`",
                        f"- velocity_scale: `{self._format_float(float(parameters.velocity_scale))}`",
                    ]
                ),
                media_type="text/markdown",
            ),
        )
        self._refresh_snapshot()

    def log_summary(
        self,
        *,
        summary_path: Path,
        report_path: Path,
        dynamic_summary_path: Path | None = None,
        dynamic_report_path: Path | None = None,
    ) -> None:
        self._state["stage"] = "completed"
        if self._recording is None:
            return
        static_text = report_path.read_text(encoding="utf-8") if report_path.exists() else str(summary_path)
        dynamic_text = ""
        if dynamic_report_path is not None:
            dynamic_text = dynamic_report_path.read_text(encoding="utf-8") if dynamic_report_path.exists() else str(dynamic_summary_path)
        self._recording.log(
            "summary/static",
            rr.TextDocument(static_text, media_type="text/markdown"),
        )
        self._recording.log(
            "summary/dynamic",
            rr.TextDocument(dynamic_text or "-", media_type="text/markdown"),
        )
        self._refresh_snapshot()

    def close(self) -> None:
        if self._recording is None:
            return
        disconnect = getattr(self._recording, "disconnect", None)
        if callable(disconnect):
            disconnect()
