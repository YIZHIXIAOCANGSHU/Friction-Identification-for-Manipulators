from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from friction_identification_core.models import MotorIdentificationResult, RoundCapture

try:
    import rerun as rr
except ImportError:  # pragma: no cover - optional at import time
    rr = None


class RerunRecorder:
    def __init__(self, recording_path: Path, *, motor_ids: tuple[int, ...], motor_names: Mapping[int, str]) -> None:
        self.recording_path = Path(recording_path)
        self._motor_ids = tuple(int(motor_id) for motor_id in motor_ids)
        self._motor_names = {int(motor_id): str(motor_names[motor_id]) for motor_id in self._motor_ids}
        self._motor_index = {motor_id: index for index, motor_id in enumerate(self._motor_ids)}
        self._recording = None if rr is None else rr.RecordingStream("friction_identification")
        self._initialized_paths: set[str] = set()
        self._last_phase_by_round: dict[str, str] = {}
        self._live_sample_index = 0
        self._identification_batch_by_motor = {motor_id: 0 for motor_id in self._motor_ids}
        motor_count = len(self._motor_ids)
        self._live_position = np.full(motor_count, np.nan, dtype=np.float64)
        self._live_velocity = np.full(motor_count, np.nan, dtype=np.float64)
        self._live_target_torque = np.zeros(motor_count, dtype=np.float64)
        self._live_feedback_torque = np.full(motor_count, np.nan, dtype=np.float64)
        self._live_state = ["-"] * motor_count
        self._live_temperature = np.full(motor_count, np.nan, dtype=np.float64)
        self._live_phase = ["-"] * motor_count
        self._latest_context = {
            "group_index": 0,
            "round_index": 0,
            "active_motor_id": 0,
        }
        self._latest_identification: dict[int, dict[str, object]] = {}
        if self._recording is None:
            return
        self._recording.save(self.recording_path)
        self._send_default_blueprint()
        self._recording.log(
            "run/overview",
            rr.TextDocument(
                "\n".join(
                    [
                        "# Sequential Friction Identification",
                        "",
                        "- `live/overview/current_state`: all motor live snapshot",
                        "- `live/all/signals/*`: all motor position / velocity / target torque / actual torque",
                        "- `live/motors/motor_xx/*`: single motor live curves and latest identification result",
                        "- `rounds/*/signals/*`: target motor round detail, including velocity error and saturation",
                        "- `rounds/*/identification/*`: torque fit, residual and sample masks",
                        "- `rounds/*/quality/summary`: round quality summary",
                        "- `summary/*`: final summary bar charts, conclusions and report",
                    ]
                ),
                media_type="text/markdown",
            ),
        )
        self._recording.log(
            "live/overview/current_state",
            rr.TextDocument(self._build_current_state_markdown(), media_type="text/markdown"),
        )
        for motor_id in self._motor_ids:
            self._recording.log(
                f"{self._live_motor_root(motor_id)}/status",
                rr.TextDocument(self._build_motor_status_markdown(motor_id), media_type="text/markdown"),
            )
            self._recording.log(
                f"{self._live_motor_root(motor_id)}/identification/summary",
                rr.TextDocument(self._build_live_identification_markdown(motor_id), media_type="text/markdown"),
            )

    @property
    def enabled(self) -> bool:
        return self._recording is not None

    def _motor_entity_name(self, motor_id: int) -> str:
        return f"motor_{int(motor_id):02d}"

    def _motor_label(self, motor_id: int) -> str:
        return f"M{int(motor_id):02d} {self._motor_names[int(motor_id)]}"

    def _live_motor_root(self, motor_id: int) -> str:
        return f"live/motors/{self._motor_entity_name(motor_id)}"

    def _round_root(self, *, group_index: int, round_index: int, motor_id: int) -> str:
        return f"rounds/group_{int(group_index):02d}/motor_{int(motor_id):02d}/round_{int(round_index):02d}"

    def _send_default_blueprint(self) -> None:
        if self._recording is None:
            return

        rrb = rr.blueprint
        by_motor_tabs = [
            rrb.Vertical(
                rrb.TextDocumentView(origin=f"/{self._live_motor_root(motor_id)}/status", name="Current State"),
                rrb.Horizontal(
                    rrb.TimeSeriesView(origin=f"/{self._live_motor_root(motor_id)}/signals/position", name="Position"),
                    rrb.TimeSeriesView(origin=f"/{self._live_motor_root(motor_id)}/signals/velocity", name="Velocity"),
                ),
                rrb.Horizontal(
                    rrb.TimeSeriesView(origin=f"/{self._live_motor_root(motor_id)}/signals/torque", name="Target vs Actual Torque"),
                    rrb.TextDocumentView(
                        origin=f"/{self._live_motor_root(motor_id)}/identification/summary",
                        name="Identification",
                    ),
                ),
                name=self._motor_label(motor_id),
            )
            for motor_id in self._motor_ids
        ]

        overview_tab = rrb.Vertical(
            rrb.TextDocumentView(origin="/live/overview/current_state", name="Current State"),
            rrb.Horizontal(
                rrb.TimeSeriesView(origin="/live/all/signals/position", name="All Motor Position"),
                rrb.TimeSeriesView(origin="/live/all/signals/velocity", name="All Motor Velocity"),
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(origin="/live/all/signals/target_torque", name="Target Torque"),
                rrb.TimeSeriesView(origin="/live/all/signals/feedback_torque", name="Actual Torque"),
            ),
            name="Overview",
        )
        identification_tab = rrb.Vertical(
            rrb.Horizontal(
                rrb.BarChartView(origin="/summary/coulomb", name="Coulomb"),
                rrb.BarChartView(origin="/summary/viscous", name="Viscous"),
                rrb.BarChartView(origin="/summary/offset", name="Offset"),
            ),
            rrb.Horizontal(
                rrb.BarChartView(origin="/summary/velocity_scale", name="Velocity Scale"),
                rrb.BarChartView(origin="/summary/validation_rmse", name="Validation RMSE"),
                rrb.BarChartView(origin="/summary/validation_r2", name="Validation R2"),
            ),
            rrb.Horizontal(
                rrb.BarChartView(origin="/summary/high_speed_valid_rmse", name="High-Speed Valid RMSE"),
                rrb.BarChartView(origin="/summary/saturation_ratio", name="Saturation Ratio"),
                rrb.BarChartView(origin="/summary/tracking_error_ratio", name="Tracking Error Ratio"),
            ),
            rrb.Horizontal(
                rrb.BarChartView(origin="/summary/recommended_for_runtime", name="Recommended For Runtime"),
                rrb.TextDocumentView(origin="/summary/conclusions", name="Conclusions"),
            ),
            rrb.TextDocumentView(origin="/summary/report", name="Summary Report"),
            name="Identification",
        )
        blueprint = rrb.Blueprint(
            rrb.Tabs(
                overview_tab,
                rrb.Tabs(*by_motor_tabs, name="By Motor"),
                identification_tab,
            ),
            auto_views=False,
        )
        self._recording.send_blueprint(blueprint, make_active=True, make_default=True)

    def _format_float(self, value: float, *, digits: int = 6) -> str:
        value = float(value)
        if not np.isfinite(value):
            return "-"
        return f"{value:.{digits}f}"

    def _build_current_state_markdown(self) -> str:
        context = self._latest_context
        lines = [
            "# Live Motor State",
            "",
            f"- active_group: `{int(context['group_index'])}`",
            f"- active_round: `{int(context['round_index'])}`",
            f"- active_motor_id: `{int(context['active_motor_id'])}`",
            "",
            "| motor | position | velocity | target_torque | actual_torque | state | temp_c | phase | identified | coulomb | viscous | offset | valid_rmse |",
            "| --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for motor_id in self._motor_ids:
            index = self._motor_index[motor_id]
            identification = self._latest_identification.get(motor_id, {})
            lines.append(
                "| "
                f"{self._motor_label(motor_id)} | "
                f"{self._format_float(self._live_position[index])} | "
                f"{self._format_float(self._live_velocity[index])} | "
                f"{self._format_float(self._live_target_torque[index])} | "
                f"{self._format_float(self._live_feedback_torque[index])} | "
                f"{self._live_state[index]} | "
                f"{self._format_float(self._live_temperature[index], digits=2)} | "
                f"{self._live_phase[index]} | "
                f"{identification.get('identified', '-')} | "
                f"{self._format_float(float(identification.get('coulomb', np.nan)))} | "
                f"{self._format_float(float(identification.get('viscous', np.nan)))} | "
                f"{self._format_float(float(identification.get('offset', np.nan)))} | "
                f"{self._format_float(float(identification.get('valid_rmse', np.nan)))} |"
            )
        return "\n".join(lines)

    def _build_motor_status_markdown(self, motor_id: int) -> str:
        index = self._motor_index[int(motor_id)]
        context = self._latest_context
        identification = self._latest_identification.get(int(motor_id), {})
        lines = [
            f"# {self._motor_label(motor_id)}",
            "",
            f"- current_position: `{self._format_float(self._live_position[index])}`",
            f"- current_velocity: `{self._format_float(self._live_velocity[index])}`",
            f"- target_torque: `{self._format_float(self._live_target_torque[index])}`",
            f"- actual_torque: `{self._format_float(self._live_feedback_torque[index])}`",
            f"- state: `{self._live_state[index]}`",
            f"- mos_temperature_c: `{self._format_float(self._live_temperature[index], digits=2)}`",
            f"- phase: `{self._live_phase[index]}`",
            f"- active_group: `{int(context['group_index'])}`",
            f"- active_round: `{int(context['round_index'])}`",
            f"- active_motor_id: `{int(context['active_motor_id'])}`",
            f"- identified: `{identification.get('identified', 'pending')}`",
        ]
        if identification:
            lines.extend(
                [
                    f"- coulomb: `{self._format_float(float(identification.get('coulomb', np.nan)))}`",
                    f"- viscous: `{self._format_float(float(identification.get('viscous', np.nan)))}`",
                    f"- offset: `{self._format_float(float(identification.get('offset', np.nan)))}`",
                    f"- velocity_scale: `{self._format_float(float(identification.get('velocity_scale', np.nan)))}`",
                    f"- valid_rmse: `{self._format_float(float(identification.get('valid_rmse', np.nan)))}`",
                    f"- status: `{identification.get('status', 'unknown')}`",
                    f"- conclusion_level: `{identification.get('conclusion_level', 'unknown')}`",
                    f"- recommended_for_runtime: `{identification.get('recommended_for_runtime', False)}`",
                ]
            )
        return "\n".join(lines)

    def _build_identification_summary_lines(
        self,
        *,
        motor_id: int,
        motor_name: str,
        group_index: int,
        round_index: int,
        status: str,
        result: MotorIdentificationResult,
    ) -> list[str]:
        return [
            f"# Motor {int(motor_id):02d} Identification",
            "",
            f"- motor_name: `{motor_name}`",
            f"- group_index: `{int(group_index)}`",
            f"- round_index: `{int(round_index)}`",
            f"- status: `{status}`",
            f"- identified: `{bool(result.identified)}`",
            f"- coulomb: `{float(result.coulomb):.6f}`",
            f"- viscous: `{float(result.viscous):.6f}`",
            f"- offset: `{float(result.offset):.6f}`",
            f"- velocity_scale: `{float(result.velocity_scale):.6f}`",
            f"- train_rmse: `{float(result.train_rmse):.6f}`",
            f"- valid_rmse: `{float(result.valid_rmse):.6f}`",
            f"- train_r2: `{float(result.train_r2):.6f}`",
            f"- valid_r2: `{float(result.valid_r2):.6f}`",
            f"- valid_sample_ratio: `{float(result.valid_sample_ratio):.4f}`",
            f"- sample_count: `{int(result.sample_count)}`",
            f"- conclusion_level: `{result.metadata.get('conclusion_level', 'unknown')}`",
            f"- recommended_for_runtime: `{bool(result.metadata.get('recommended_for_runtime', False))}`",
            f"- high_speed_valid_rmse: `{self._format_float(float(result.metadata.get('high_speed_valid_rmse', np.nan)))}`",
        ]

    def _build_quality_summary_lines(self, result: MotorIdentificationResult) -> list[str]:
        return [
            "# Round Quality Summary",
            "",
            f"- status: `{result.metadata.get('status', 'unknown')}`",
            f"- identified: `{bool(result.identified)}`",
            f"- recommended_for_runtime: `{bool(result.metadata.get('recommended_for_runtime', False))}`",
            f"- conclusion_level: `{result.metadata.get('conclusion_level', 'unknown')}`",
            f"- dropped_platforms: `{', '.join(result.metadata.get('dropped_platforms', [])) or '-'}`",
            f"- steady_sample_count: `{int(result.metadata.get('steady_sample_count', 0))}`",
            f"- high_speed_platform_count: `{int(result.metadata.get('high_speed_platform_count', 0))}`",
            f"- validation_mode: `{result.metadata.get('validation_mode', '-')}`",
            f"- valid_rmse: `{self._format_float(float(result.valid_rmse))}`",
            f"- high_speed_valid_rmse: `{self._format_float(float(result.metadata.get('high_speed_valid_rmse', np.nan)))}`",
            f"- saturation_ratio: `{self._format_float(float(result.metadata.get('saturation_ratio', np.nan)), digits=4)}`",
            f"- tracking_error_ratio: `{self._format_float(float(result.metadata.get('tracking_error_ratio', np.nan)), digits=4)}`",
        ]

    def _build_summary_conclusions_markdown(self, summary: Mapping[str, np.ndarray]) -> str:
        lines = [
            "# Runtime Conclusions",
            "",
        ]
        motor_ids = np.asarray(summary["motor_ids"], dtype=np.int64)
        motor_names = np.asarray(summary["motor_names"]).astype(str)
        for index, motor_id in enumerate(motor_ids):
            lines.extend(
                [
                    f"## Motor {int(motor_id):02d} {motor_names[index]}",
                    "",
                    f"- motor_id / motor_name: `{int(motor_id)} / {motor_names[index]}`",
                    f"- recommendation: `{str(summary['conclusion_level'][index])}`",
                    (
                        f"- 核心参数: `coulomb={float(summary['coulomb'][index]):.6f}, "
                        f"viscous={float(summary['viscous'][index]):.6f}, "
                        f"offset={float(summary['offset'][index]):.6f}, "
                        f"velocity_scale={float(summary['velocity_scale'][index]):.6f}`"
                    ),
                    (
                        f"- 是否覆盖高速段: `"
                        f"{'yes' if int(summary['high_speed_platform_count'][index]) >= 2 else 'no'}`"
                    ),
                    f"- 主要原因: `{str(summary['conclusion_text'][index])}`",
                    "",
                ]
            )
        return "\n".join(lines)

    def _build_live_identification_markdown(self, motor_id: int) -> str:
        identification = self._latest_identification.get(int(motor_id))
        if identification is None:
            return "\n".join(
                [
                    f"# Motor {int(motor_id):02d} Identification",
                    "",
                    "- status: `pending`",
                ]
            )
        return "\n".join(identification["summary_lines"])

    def _ensure_live_series(self) -> None:
        if self._recording is None:
            return

        global_series = {
            "live/all/signals/position": [self._motor_label(motor_id) for motor_id in self._motor_ids],
            "live/all/signals/velocity": [self._motor_label(motor_id) for motor_id in self._motor_ids],
            "live/all/signals/target_torque": [self._motor_label(motor_id) for motor_id in self._motor_ids],
            "live/all/signals/feedback_torque": [self._motor_label(motor_id) for motor_id in self._motor_ids],
        }
        for path, names in global_series.items():
            if path in self._initialized_paths:
                continue
            self._recording.log(path, rr.SeriesLines(names=names), static=True)
            self._initialized_paths.add(path)

    def _ensure_live_motor_series(self, motor_id: int) -> None:
        if self._recording is None:
            return

        motor_root = self._live_motor_root(motor_id)
        series = {
            f"{motor_root}/signals/position": ["position"],
            f"{motor_root}/signals/velocity": ["velocity"],
            f"{motor_root}/signals/torque": ["target", "feedback"],
        }
        for path, names in series.items():
            if path in self._initialized_paths:
                continue
            self._recording.log(path, rr.SeriesLines(names=names), static=True)
            self._initialized_paths.add(path)

    def _ensure_identification_history_series(self, motor_id: int) -> None:
        if self._recording is None:
            return

        for metric in ("coulomb", "viscous", "offset", "velocity_scale", "validation_rmse", "validation_r2"):
            path = f"identification/history/{metric}/{self._motor_entity_name(motor_id)}"
            if path in self._initialized_paths:
                continue
            self._recording.log(path, rr.SeriesLines(names=[self._motor_label(motor_id)]), static=True)
            self._initialized_paths.add(path)

    def _log_live_overview_documents(self, motor_id: int) -> None:
        if self._recording is None:
            return

        self._recording.log(
            "live/overview/current_state",
            rr.TextDocument(self._build_current_state_markdown(), media_type="text/markdown"),
        )
        self._recording.log(
            f"{self._live_motor_root(motor_id)}/status",
            rr.TextDocument(self._build_motor_status_markdown(motor_id), media_type="text/markdown"),
        )
        self._recording.log(
            f"{self._live_motor_root(motor_id)}/identification/summary",
            rr.TextDocument(self._build_live_identification_markdown(motor_id), media_type="text/markdown"),
        )

    def _ensure_round_series(self, round_root: str) -> None:
        if self._recording is None:
            return

        position_path = f"{round_root}/signals/position"
        if position_path not in self._initialized_paths:
            self._recording.log(
                position_path,
                rr.SeriesLines(names=["feedback", "reference"]),
                static=True,
            )
            self._initialized_paths.add(position_path)

        velocity_path = f"{round_root}/signals/velocity"
        if velocity_path not in self._initialized_paths:
            self._recording.log(
                velocity_path,
                rr.SeriesLines(names=["feedback", "reference"]),
                static=True,
            )
            self._initialized_paths.add(velocity_path)

        torque_path = f"{round_root}/signals/torque"
        if torque_path not in self._initialized_paths:
            self._recording.log(
                torque_path,
                rr.SeriesLines(names=["raw_command", "sent_command", "feedback"]),
                static=True,
            )
            self._initialized_paths.add(torque_path)

        fit_path = f"{round_root}/identification/torque_fit"
        if fit_path not in self._initialized_paths:
            self._recording.log(
                fit_path,
                rr.SeriesLines(names=["target", "prediction"]),
                static=True,
            )
            self._initialized_paths.add(fit_path)

        velocity_error_path = f"{round_root}/signals/velocity_error"
        if velocity_error_path not in self._initialized_paths:
            self._recording.log(
                velocity_error_path,
                rr.SeriesLines(names=["velocity_error"]),
                static=True,
            )
            self._initialized_paths.add(velocity_error_path)

        saturation_flag_path = f"{round_root}/signals/saturation_flag"
        if saturation_flag_path not in self._initialized_paths:
            self._recording.log(
                saturation_flag_path,
                rr.SeriesLines(names=["saturation_flag"]),
                static=True,
            )
            self._initialized_paths.add(saturation_flag_path)

        residual_path = f"{round_root}/identification/residual"
        if residual_path not in self._initialized_paths:
            self._recording.log(
                residual_path,
                rr.SeriesLines(names=["residual"]),
                static=True,
            )
            self._initialized_paths.add(residual_path)

        sample_masks_path = f"{round_root}/identification/sample_masks"
        if sample_masks_path not in self._initialized_paths:
            self._recording.log(
                sample_masks_path,
                rr.SeriesLines(names=["steady", "selected", "train", "valid", "tracking_ok", "saturation_ok"]),
                static=True,
            )
            self._initialized_paths.add(sample_masks_path)

    def log_live_motor_sample(
        self,
        *,
        group_index: int,
        round_index: int,
        active_motor_id: int,
        motor_id: int,
        position: float,
        velocity: float,
        target_torque: float,
        feedback_torque: float,
        state: int,
        mos_temperature: float,
        phase_name: str,
    ) -> None:
        if self._recording is None:
            return

        motor_id = int(motor_id)
        index = self._motor_index[motor_id]
        self._latest_context = {
            "group_index": int(group_index),
            "round_index": int(round_index),
            "active_motor_id": int(active_motor_id),
        }
        self._live_position[index] = float(position)
        self._live_velocity[index] = float(velocity)
        self._live_target_torque[index] = float(target_torque)
        self._live_feedback_torque[index] = float(feedback_torque)
        self._live_state[index] = str(int(state))
        self._live_temperature[index] = float(mos_temperature)
        self._live_phase[index] = str(phase_name)

        self._ensure_live_series()
        self._ensure_live_motor_series(motor_id)
        self._recording.set_time("live_sample", sequence=int(self._live_sample_index))
        self._live_sample_index += 1
        motor_root = self._live_motor_root(motor_id)
        self._recording.log(f"{motor_root}/signals/position", rr.Scalars([float(position)]))
        self._recording.log(f"{motor_root}/signals/velocity", rr.Scalars([float(velocity)]))
        self._recording.log(f"{motor_root}/signals/torque", rr.Scalars([float(target_torque), float(feedback_torque)]))
        self._recording.log("live/all/signals/position", rr.Scalars(self._live_position.copy()))
        self._recording.log("live/all/signals/velocity", rr.Scalars(self._live_velocity.copy()))
        self._recording.log("live/all/signals/target_torque", rr.Scalars(self._live_target_torque.copy()))
        self._recording.log("live/all/signals/feedback_torque", rr.Scalars(self._live_feedback_torque.copy()))
        self._log_live_overview_documents(motor_id)

    def log_round_stop(
        self,
        *,
        group_index: int,
        round_index: int,
        motor_id: int,
        phase_name: str,
    ) -> None:
        if self._recording is None:
            return

        motor_id = int(motor_id)
        index = self._motor_index[motor_id]
        self._latest_context = {
            "group_index": int(group_index),
            "round_index": int(round_index),
            "active_motor_id": int(motor_id),
        }
        self._live_target_torque[index] = 0.0
        self._live_phase[index] = str(phase_name)
        self._ensure_live_series()
        self._ensure_live_motor_series(motor_id)
        self._recording.set_time("live_sample", sequence=int(self._live_sample_index))
        self._live_sample_index += 1
        self._recording.log(
            f"{self._live_motor_root(motor_id)}/signals/torque",
            rr.Scalars([0.0, float(self._live_feedback_torque[index])]),
        )
        self._recording.log("live/all/signals/target_torque", rr.Scalars(self._live_target_torque.copy()))
        self._recording.log("live/all/signals/feedback_torque", rr.Scalars(self._live_feedback_torque.copy()))
        self._log_live_overview_documents(motor_id)

    def log_live_sample(
        self,
        *,
        group_index: int,
        round_index: int,
        motor_id: int,
        motor_name: str,
        sample_index: int,
        elapsed_s: float,
        position: float,
        position_cmd: float,
        velocity: float,
        velocity_cmd: float,
        command_raw: float,
        command: float,
        torque_feedback: float,
        phase_name: str,
    ) -> None:
        if self._recording is None:
            return

        round_root = self._round_root(group_index=group_index, round_index=round_index, motor_id=motor_id)
        self._ensure_round_series(round_root)
        self._recording.set_time("round_sample", sequence=int(sample_index))
        self._recording.set_time("round_time_s", duration=float(elapsed_s))
        self._recording.log(f"{round_root}/signals/position", rr.Scalars([float(position), float(position_cmd)]))
        self._recording.log(f"{round_root}/signals/velocity", rr.Scalars([float(velocity), float(velocity_cmd)]))
        self._recording.log(
            f"{round_root}/signals/torque",
            rr.Scalars([float(command_raw), float(command), float(torque_feedback)]),
        )

        previous_phase = self._last_phase_by_round.get(round_root)
        if previous_phase != phase_name:
            self._recording.log(
                f"{round_root}/events",
                rr.TextLog(f"{motor_name}: phase -> {phase_name}"),
            )
            self._last_phase_by_round[round_root] = str(phase_name)

    def log_identification(self, capture: RoundCapture, result: MotorIdentificationResult) -> None:
        if self._recording is None:
            return

        round_root = self._round_root(
            group_index=capture.group_index,
            round_index=capture.round_index,
            motor_id=capture.target_motor_id,
        )
        self._ensure_round_series(round_root)
        sample_count = min(capture.sample_count, int(result.torque_target.size), int(result.torque_pred.size))
        for sample_index in range(sample_count):
            self._recording.set_time("round_sample", sequence=int(sample_index))
            self._recording.set_time("round_time_s", duration=float(capture.time[sample_index]))
            self._recording.log(
                f"{round_root}/identification/torque_fit",
                rr.Scalars(
                    [
                        float(result.torque_target[sample_index]),
                        float(result.torque_pred[sample_index]),
                    ]
                ),
            )

        status = str(result.metadata.get("status", "unknown"))
        summary_lines = self._build_identification_summary_lines(
            motor_id=capture.target_motor_id,
            motor_name=capture.motor_name,
            group_index=capture.group_index,
            round_index=capture.round_index,
            status=status,
            result=result,
        )
        self._recording.log(
            f"{round_root}/identification/summary",
            rr.TextDocument("\n".join(summary_lines), media_type="text/markdown"),
        )
        quality_summary_lines = self._build_quality_summary_lines(result)
        self._recording.log(
            f"{round_root}/quality/summary",
            rr.TextDocument("\n".join(quality_summary_lines), media_type="text/markdown"),
        )
        velocity_error = np.asarray(capture.velocity, dtype=np.float64) - np.asarray(capture.velocity_cmd, dtype=np.float64)
        saturation_flag = ~np.asarray(result.saturation_ok_mask, dtype=bool)
        residual = np.asarray(result.torque_target, dtype=np.float64) - np.asarray(result.torque_pred, dtype=np.float64)
        self._latest_identification[int(capture.target_motor_id)] = {
            "identified": bool(result.identified),
            "status": status,
            "coulomb": float(result.coulomb),
            "viscous": float(result.viscous),
            "offset": float(result.offset),
            "velocity_scale": float(result.velocity_scale),
            "valid_rmse": float(result.valid_rmse),
            "valid_r2": float(result.valid_r2),
            "conclusion_level": str(result.metadata.get("conclusion_level", "unknown")),
            "recommended_for_runtime": bool(result.metadata.get("recommended_for_runtime", False)),
            "summary_lines": summary_lines,
        }
        for sample_index in range(sample_count):
            self._recording.set_time("round_sample", sequence=int(sample_index))
            self._recording.set_time("round_time_s", duration=float(capture.time[sample_index]))
            self._recording.log(
                f"{round_root}/signals/velocity_error",
                rr.Scalars([float(velocity_error[sample_index])]),
            )
            self._recording.log(
                f"{round_root}/signals/saturation_flag",
                rr.Scalars([float(saturation_flag[sample_index])]),
            )
            self._recording.log(
                f"{round_root}/identification/residual",
                rr.Scalars([float(residual[sample_index])]),
            )
            self._recording.log(
                f"{round_root}/identification/sample_masks",
                rr.Scalars(
                    [
                        float(result.steady_state_mask[sample_index]),
                        float(result.sample_mask[sample_index]),
                        float(result.train_mask[sample_index]),
                        float(result.valid_mask[sample_index]),
                        float(result.tracking_ok_mask[sample_index]),
                        float(result.saturation_ok_mask[sample_index]),
                    ]
                ),
            )
        self._ensure_identification_history_series(int(capture.target_motor_id))
        batch_index = self._identification_batch_by_motor[int(capture.target_motor_id)] + 1
        self._identification_batch_by_motor[int(capture.target_motor_id)] = batch_index
        self._recording.set_time("identification_batch", sequence=int(batch_index))
        for metric, value in (
            ("coulomb", float(result.coulomb)),
            ("viscous", float(result.viscous)),
            ("offset", float(result.offset)),
            ("velocity_scale", float(result.velocity_scale)),
            ("validation_rmse", float(result.valid_rmse)),
            ("validation_r2", float(result.valid_r2)),
        ):
            self._recording.log(
                f"identification/history/{metric}/{self._motor_entity_name(capture.target_motor_id)}",
                rr.Scalars([value]),
            )
        self._log_live_overview_documents(int(capture.target_motor_id))

    def log_summary(self, *, summary_path: Path, report_path: Path) -> None:
        if self._recording is None:
            return

        with np.load(summary_path, allow_pickle=False) as summary:
            motor_ids = np.asarray(summary["motor_ids"], dtype=np.float64).tolist()
            for metric in ("coulomb", "viscous", "offset", "velocity_scale", "validation_rmse", "validation_r2"):
                self._recording.log(
                    f"summary/{metric}",
                    rr.BarChart(np.asarray(summary[metric], dtype=np.float64), abscissa=motor_ids),
                )
            for metric in ("high_speed_valid_rmse", "saturation_ratio", "tracking_error_ratio"):
                self._recording.log(
                    f"summary/{metric}",
                    rr.BarChart(np.asarray(summary[metric], dtype=np.float64), abscissa=motor_ids),
                )
            self._recording.log(
                "summary/recommended_for_runtime",
                rr.BarChart(
                    np.asarray(summary["recommended_for_runtime"], dtype=np.float64),
                    abscissa=motor_ids,
                ),
            )
            self._recording.log(
                "summary/conclusions",
                rr.TextDocument(self._build_summary_conclusions_markdown(summary), media_type="text/markdown"),
            )

        self._recording.log(
            "summary/report",
            rr.TextDocument(report_path.read_text(encoding="utf-8"), media_type="text/markdown"),
        )

    def close(self) -> None:
        if self._recording is None:
            return
        self._recording.disconnect()
