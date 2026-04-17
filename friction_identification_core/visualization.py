from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from friction_identification_core.config import Config
from friction_identification_core.models import FrictionIdentificationResult
from friction_identification_core.mujoco_support import build_am_d02_model
from friction_identification_core.runtime import log_info
from friction_identification_core.status import format_joint_motion_summary

if TYPE_CHECKING:
    from friction_identification_core.results import IdentificationResults


_PHASE_CODE_MAP = {
    "startup_move": 0,
    "startup_settle": 1,
    "home_to_center": 2,
    "center_hold": 2,
    "full_range_sweep": 3,
    "reversal_dither": 4,
    "speed_sweep": 5,
    "hold": 6,
    "collect": 7,
    "compensate": 8,
}


def _joint_colors() -> list[list[int]]:
    return [
        [230, 50, 50],
        [230, 140, 30],
        [210, 200, 30],
        [50, 200, 50],
        [50, 200, 200],
        [50, 80, 230],
        [150, 50, 230],
    ]


def _phase_code(phase_name: str) -> int:
    return int(_PHASE_CODE_MAP.get(str(phase_name), -1))


_JOINT_SERIES_SPECS = (
    ("joint_state/q", "q", "Position", 2.0),
    ("joint_state/qd", "qd", "Velocity", 2.0),
    ("joint_state/q_cmd", "q cmd", "Cmd Position", 1.5),
    ("joint_state/qd_cmd", "qd cmd", "Cmd Velocity", 1.5),
    ("joint_state/rotation_state", "rotation", "Rotation", 2.0),
    ("joint_state/range_ratio", "range", "Range Ratio", 2.0),
    ("joint_state/phase", "phase", "Phase", 1.5),
    ("torque/measured", "measured", "Measured Torque", 2.0),
    ("torque/command", "command", "Command Torque", 2.0),
    ("torque/friction_comp", "friction comp", "Friction Compensation", 2.0),
    ("torque/residual", "residual", "Residual Torque", 2.0),
    ("torque/track_ff", "track ff", "Tracking Feedforward", 1.5),
    ("torque/track_fb", "track fb", "Tracking Feedback", 1.5),
    ("safety/limit_margin_remaining", "margin", "Limit Margin", 2.0),
    ("health/mos_temperature", "MOS temp", "MOS Temperature", 1.5),
    ("health/coil_temperature", "coil temp", "Coil Temperature", 1.5),
)

_RUNTIME_SERIES_SPECS = (
    ("runtime/uart_cycle_hz", [230, 120, 40], "UART cycle rate"),
    ("runtime/uart_latency_ms", [230, 170, 80], "UART cycle period"),
    ("runtime/uart_transfer_kbps", [100, 180, 230], "UART throughput"),
    ("runtime/valid_sample_ratio", [80, 180, 110], "Valid sample ratio"),
    ("runtime/batch_index", [180, 180, 180], "Batch index"),
)

_IDENTIFICATION_METRIC_SPECS = (
    ("coulomb", "Coulomb"),
    ("viscous", "Viscous"),
    ("offset", "Offset"),
    ("validation_rmse", "Val RMSE"),
    ("validation_r2", "Val R2"),
    ("valid_sample_ratio", "Valid Ratio"),
)


def _joint_color(joint_index: int) -> list[int]:
    palette = _joint_colors()
    return list(palette[joint_index % len(palette)])


def _joint_entity_path(prefix: str, joint_id: int) -> str:
    return f"{prefix}/J{joint_id}"


def _identification_history_path(metric_key: str, joint_id: int) -> str:
    return f"identification/history/{metric_key}/J{joint_id}"


class HardwareRerunReporter:
    """Realtime Rerun dashboard for 7-axis collection and compensation runs."""

    def __init__(self, *, app_name: str, joint_names: list[str], spawn: bool = True) -> None:
        self.app_name = app_name
        self.spawn = spawn
        self.joint_names = list(joint_names)
        self._rr = None

    def init(self) -> None:
        import rerun as rr
        import rerun.blueprint as rrb

        self._rr = rr
        rr.init(self.app_name, spawn=self.spawn)
        self._log_static_series_metadata()
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.Tabs(
                    self._build_overview_tab(rrb),
                    self._build_joint_centric_tab(rrb),
                    self._build_motion_tab(rrb),
                    self._build_torque_tab(rrb),
                    self._build_identification_tab(rrb),
                    self._build_runtime_tab(rrb),
                    active_tab="Overview",
                ),
                collapse_panels=True,
            )
        )

    def _log_static_series_metadata(self) -> None:
        if self._rr is None:
            return
        rr = self._rr

        for joint_idx, _joint_name in enumerate(self.joint_names):
            joint_id = joint_idx + 1
            color = _joint_color(joint_idx)
            for prefix, series_name, _view_label, width in _JOINT_SERIES_SPECS:
                rr.log(
                    _joint_entity_path(prefix, joint_id),
                    rr.SeriesLines(colors=[color], names=[f"J{joint_id} {series_name}"], widths=[width]),
                    static=True,
                )
            for metric_key, metric_label in _IDENTIFICATION_METRIC_SPECS:
                rr.log(
                    _identification_history_path(metric_key, joint_id),
                    rr.SeriesLines(colors=[color], names=[f"J{joint_id} {metric_label}"], widths=[2.0]),
                    static=True,
                )

        for path, color, label in _RUNTIME_SERIES_SPECS:
            rr.log(path, rr.SeriesLines(colors=[color], names=[label], widths=[2.0]), static=True)

    def _joint_metric_grid(self, rrb, *, title: str, prefix: str, view_label: str):
        return rrb.Grid(
            *[
                rrb.TimeSeriesView(
                    name=f"J{joint_idx + 1} {view_label}",
                    origin=f"/{_joint_entity_path(prefix, joint_idx + 1)}",
                )
                for joint_idx in range(len(self.joint_names))
            ],
            grid_columns=min(4, max(len(self.joint_names), 1)),
            name=title,
        )

    def _aggregate_identification_grid(self, rrb):
        return rrb.Grid(
            rrb.BarChartView(name="Coulomb", origin="/identification/coulomb"),
            rrb.BarChartView(name="Viscous", origin="/identification/viscous"),
            rrb.BarChartView(name="Offset", origin="/identification/offset"),
            rrb.BarChartView(name="Validation RMSE", origin="/identification/validation_rmse"),
            rrb.BarChartView(name="Validation R2", origin="/identification/validation_r2"),
            rrb.BarChartView(name="Valid Sample Ratio", origin="/identification/valid_sample_ratio"),
            grid_columns=3,
            name="Aggregate Parameters",
        )

    def _identification_history_grid(self, rrb, *, joint_ids: list[int], name: str, grid_columns: int):
        return rrb.Grid(
            *[
                rrb.TimeSeriesView(
                    name=f"J{joint_id} {metric_label}",
                    origin=f"/{_identification_history_path(metric_key, joint_id)}",
                )
                for joint_id in joint_ids
                for metric_key, metric_label in _IDENTIFICATION_METRIC_SPECS
            ],
            grid_columns=max(int(grid_columns), 1),
            name=name,
        )

    def _joint_tab(self, rrb, *, joint_id: int):
        return rrb.Vertical(
            rrb.Grid(
                rrb.TimeSeriesView(name="Position", origin=f"/{_joint_entity_path('joint_state/q', joint_id)}"),
                rrb.TimeSeriesView(name="Velocity", origin=f"/{_joint_entity_path('joint_state/qd', joint_id)}"),
                rrb.TimeSeriesView(name="Cmd Position", origin=f"/{_joint_entity_path('joint_state/q_cmd', joint_id)}"),
                rrb.TimeSeriesView(name="Cmd Velocity", origin=f"/{_joint_entity_path('joint_state/qd_cmd', joint_id)}"),
                rrb.TimeSeriesView(
                    name="Rotation State",
                    origin=f"/{_joint_entity_path('joint_state/rotation_state', joint_id)}",
                ),
                rrb.TimeSeriesView(name="Range Ratio", origin=f"/{_joint_entity_path('joint_state/range_ratio', joint_id)}"),
                rrb.TimeSeriesView(name="Phase", origin=f"/{_joint_entity_path('joint_state/phase', joint_id)}"),
                rrb.TimeSeriesView(
                    name="Limit Margin",
                    origin=f"/{_joint_entity_path('safety/limit_margin_remaining', joint_id)}",
                ),
                grid_columns=4,
                name="Motion",
            ),
            rrb.Grid(
                rrb.TimeSeriesView(name="Measured Torque", origin=f"/{_joint_entity_path('torque/measured', joint_id)}"),
                rrb.TimeSeriesView(name="Command Torque", origin=f"/{_joint_entity_path('torque/command', joint_id)}"),
                rrb.TimeSeriesView(
                    name="Friction Compensation",
                    origin=f"/{_joint_entity_path('torque/friction_comp', joint_id)}",
                ),
                rrb.TimeSeriesView(name="Residual Torque", origin=f"/{_joint_entity_path('torque/residual', joint_id)}"),
                rrb.TimeSeriesView(
                    name="Tracking Feedforward",
                    origin=f"/{_joint_entity_path('torque/track_ff', joint_id)}",
                ),
                rrb.TimeSeriesView(
                    name="Tracking Feedback",
                    origin=f"/{_joint_entity_path('torque/track_fb', joint_id)}",
                ),
                rrb.TimeSeriesView(
                    name="MOS Temperature",
                    origin=f"/{_joint_entity_path('health/mos_temperature', joint_id)}",
                ),
                rrb.TimeSeriesView(
                    name="Coil Temperature",
                    origin=f"/{_joint_entity_path('health/coil_temperature', joint_id)}",
                ),
                grid_columns=4,
                name="Torque & Health",
            ),
            self._identification_history_grid(
                rrb,
                joint_ids=[joint_id],
                name="Identification",
                grid_columns=3,
            ),
            name=f"J{joint_id}",
        )

    def _build_overview_tab(self, rrb):
        return rrb.Vertical(
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.TextDocumentView(name="Runtime Status", origin="/runtime/status"),
                    rrb.TextDocumentView(name="Joint Summary", origin="/runtime/joint_summary"),
                    row_shares=[0.75, 1.25],
                    name="Live Status",
                ),
                rrb.TextDocumentView(name="Identification Summary", origin="/identification/summary"),
                rrb.Spatial3DView(name="EE Pose", origin="/trajectory_3d"),
                column_shares=[1.0, 1.35, 1.1],
                name="Status Overview",
            ),
            rrb.Tabs(
                rrb.Vertical(
                    self._joint_metric_grid(rrb, title="Joint Position", prefix="joint_state/q", view_label="Position"),
                    self._joint_metric_grid(rrb, title="Joint Velocity", prefix="joint_state/qd", view_label="Velocity"),
                    self._joint_metric_grid(
                        rrb,
                        title="Range Ratio",
                        prefix="joint_state/range_ratio",
                        view_label="Range",
                    ),
                    name="Motion Snapshot",
                ),
                rrb.Vertical(
                    self._joint_metric_grid(
                        rrb,
                        title="Measured Torque",
                        prefix="torque/measured",
                        view_label="Measured",
                    ),
                    self._joint_metric_grid(
                        rrb,
                        title="Residual Torque",
                        prefix="torque/residual",
                        view_label="Residual",
                    ),
                    self._joint_metric_grid(
                        rrb,
                        title="Friction Compensation",
                        prefix="torque/friction_comp",
                        view_label="Friction Comp",
                    ),
                    name="Torque Snapshot",
                ),
                rrb.Vertical(
                    self._aggregate_identification_grid(rrb),
                    self._identification_history_grid(
                        rrb,
                        joint_ids=list(range(1, len(self.joint_names) + 1)),
                        name="Per-Joint Parameter Trends",
                        grid_columns=len(_IDENTIFICATION_METRIC_SPECS),
                    ),
                    name="Identification Snapshot",
                ),
                rrb.Grid(
                    rrb.TimeSeriesView(name="UART Frequency", origin="/runtime/uart_cycle_hz"),
                    rrb.TimeSeriesView(name="UART Period", origin="/runtime/uart_latency_ms"),
                    rrb.TimeSeriesView(name="UART Throughput", origin="/runtime/uart_transfer_kbps"),
                    rrb.TimeSeriesView(name="Valid Sample Ratio", origin="/runtime/valid_sample_ratio"),
                    rrb.TimeSeriesView(name="Batch Index", origin="/runtime/batch_index"),
                    grid_columns=3,
                    name="Runtime Trends",
                ),
                name="Quick Panels",
            ),
            name="Overview",
        )

    def _build_joint_centric_tab(self, rrb):
        return rrb.Tabs(
            *[self._joint_tab(rrb, joint_id=joint_idx + 1) for joint_idx in range(len(self.joint_names))],
            active_tab="J1" if self.joint_names else None,
            name="By Joint",
        )

    def _build_motion_tab(self, rrb):
        return rrb.Vertical(
            self._joint_metric_grid(rrb, title="Position", prefix="joint_state/q", view_label="Position"),
            self._joint_metric_grid(rrb, title="Velocity", prefix="joint_state/qd", view_label="Velocity"),
            self._joint_metric_grid(rrb, title="Cmd Position", prefix="joint_state/q_cmd", view_label="Cmd Position"),
            self._joint_metric_grid(rrb, title="Cmd Velocity", prefix="joint_state/qd_cmd", view_label="Cmd Velocity"),
            self._joint_metric_grid(
                rrb,
                title="Rotation State",
                prefix="joint_state/rotation_state",
                view_label="Rotation",
            ),
            self._joint_metric_grid(rrb, title="Range Ratio", prefix="joint_state/range_ratio", view_label="Range"),
            self._joint_metric_grid(rrb, title="Phase", prefix="joint_state/phase", view_label="Phase"),
            self._joint_metric_grid(
                rrb,
                title="Limit Margin Remaining",
                prefix="safety/limit_margin_remaining",
                view_label="Margin",
            ),
            name="Joint Motion",
        )

    def _build_torque_tab(self, rrb):
        return rrb.Vertical(
            self._joint_metric_grid(rrb, title="Measured Torque", prefix="torque/measured", view_label="Measured"),
            self._joint_metric_grid(rrb, title="Command Torque", prefix="torque/command", view_label="Command"),
            self._joint_metric_grid(
                rrb,
                title="Friction Compensation",
                prefix="torque/friction_comp",
                view_label="Friction Comp",
            ),
            self._joint_metric_grid(rrb, title="Residual Torque", prefix="torque/residual", view_label="Residual"),
            self._joint_metric_grid(
                rrb,
                title="Tracking Feedforward",
                prefix="torque/track_ff",
                view_label="Track FF",
            ),
            self._joint_metric_grid(
                rrb,
                title="Tracking Feedback",
                prefix="torque/track_fb",
                view_label="Track FB",
            ),
            name="Torque",
        )

    def _build_identification_tab(self, rrb):
        return rrb.Vertical(
            rrb.Horizontal(
                rrb.TextDocumentView(name="Identification Summary", origin="/identification/summary"),
                self._aggregate_identification_grid(rrb),
                column_shares=[1.35, 1.0],
                name="Aggregate Overview",
            ),
            self._identification_history_grid(
                rrb,
                joint_ids=list(range(1, len(self.joint_names) + 1)),
                name="Per-Joint Parameter Mini Charts",
                grid_columns=len(_IDENTIFICATION_METRIC_SPECS),
            ),
            name="Identification",
        )

    def _build_runtime_tab(self, rrb):
        return rrb.Vertical(
            rrb.Grid(
                rrb.TimeSeriesView(name="UART Frequency", origin="/runtime/uart_cycle_hz"),
                rrb.TimeSeriesView(name="UART Period", origin="/runtime/uart_latency_ms"),
                rrb.TimeSeriesView(name="UART Throughput", origin="/runtime/uart_transfer_kbps"),
                rrb.TimeSeriesView(name="Valid Sample Ratio", origin="/runtime/valid_sample_ratio"),
                rrb.TimeSeriesView(name="Batch Index", origin="/runtime/batch_index"),
                grid_columns=3,
                name="Runtime Telemetry",
            ),
            self._joint_metric_grid(
                rrb,
                title="MOS Temperature",
                prefix="health/mos_temperature",
                view_label="MOS Temp",
            ),
            self._joint_metric_grid(
                rrb,
                title="Coil Temperature",
                prefix="health/coil_temperature",
                view_label="Coil Temp",
            ),
            rrb.Horizontal(
                rrb.TextDocumentView(name="Runtime Status", origin="/runtime/status"),
                rrb.TextDocumentView(name="Joint Summary", origin="/runtime/joint_summary"),
                rrb.Spatial3DView(name="EE Pose", origin="/trajectory_3d"),
                column_shares=[1.0, 1.2, 1.1],
                name="Runtime Overview",
            ),
            name="Runtime",
        )

    def log_step(
        self,
        *,
        elapsed_s: float,
        step_index: int,
        batch_index: int,
        total_batches: int,
        q: np.ndarray,
        qd: np.ndarray,
        q_cmd: np.ndarray,
        qd_cmd: np.ndarray,
        tau_measured: np.ndarray,
        tau_command: np.ndarray,
        tau_track_ff: np.ndarray,
        tau_track_fb: np.ndarray,
        tau_friction_comp: np.ndarray,
        tau_residual: np.ndarray,
        rotation_state: np.ndarray,
        range_ratio: np.ndarray,
        limit_margin_remaining: np.ndarray,
        mos_temperature: np.ndarray,
        coil_temperature: np.ndarray,
        uart_cycle_hz: float,
        uart_latency_ms: float,
        uart_transfer_kbps: float,
        valid_sample_ratio: float,
        phase_name: str,
        ee_pos: np.ndarray | None,
        ee_quat: np.ndarray | None,
    ) -> None:
        if self._rr is None:
            return
        rr = self._rr
        rr.set_time_seconds("time", float(elapsed_s))
        rr.set_time_sequence("step", int(step_index))

        for idx in range(q.shape[0]):
            joint_id = idx + 1
            rr.log(f"joint_state/q/J{joint_id}", rr.Scalars(float(q[idx])))
            rr.log(f"joint_state/qd/J{joint_id}", rr.Scalars(float(qd[idx])))
            rr.log(f"joint_state/q_cmd/J{joint_id}", rr.Scalars(float(q_cmd[idx])))
            rr.log(f"joint_state/qd_cmd/J{joint_id}", rr.Scalars(float(qd_cmd[idx])))
            rr.log(f"joint_state/rotation_state/J{joint_id}", rr.Scalars(int(rotation_state[idx])))
            rr.log(f"joint_state/range_ratio/J{joint_id}", rr.Scalars(float(range_ratio[idx])))
            rr.log(f"joint_state/phase/J{joint_id}", rr.Scalars(_phase_code(phase_name)))

            rr.log(f"torque/measured/J{joint_id}", rr.Scalars(float(tau_measured[idx])))
            rr.log(f"torque/command/J{joint_id}", rr.Scalars(float(tau_command[idx])))
            rr.log(f"torque/friction_comp/J{joint_id}", rr.Scalars(float(tau_friction_comp[idx])))
            rr.log(f"torque/residual/J{joint_id}", rr.Scalars(float(tau_residual[idx])))
            rr.log(f"torque/track_ff/J{joint_id}", rr.Scalars(float(tau_track_ff[idx])))
            rr.log(f"torque/track_fb/J{joint_id}", rr.Scalars(float(tau_track_fb[idx])))

            rr.log(f"safety/limit_margin_remaining/J{joint_id}", rr.Scalars(float(limit_margin_remaining[idx])))
            rr.log(f"health/mos_temperature/J{joint_id}", rr.Scalars(float(mos_temperature[idx])))
            rr.log(f"health/coil_temperature/J{joint_id}", rr.Scalars(float(coil_temperature[idx])))

        rr.log("runtime/uart_cycle_hz", rr.Scalars(float(uart_cycle_hz)))
        rr.log("runtime/uart_latency_ms", rr.Scalars(float(uart_latency_ms)))
        rr.log("runtime/uart_transfer_kbps", rr.Scalars(float(uart_transfer_kbps)))
        rr.log("runtime/valid_sample_ratio", rr.Scalars(float(valid_sample_ratio)))
        rr.log("runtime/batch_index", rr.Scalars(float(batch_index)))

        rr.log(
            "runtime/status",
            rr.TextDocument(
                "\n".join(
                    [
                        f"batch: {batch_index}/{total_batches}",
                        f"phase: {phase_name or 'unknown'}",
                        f"valid_sample_ratio: {valid_sample_ratio:.3f}",
                        f"uart_cycle_hz: {uart_cycle_hz:.2f}",
                        f"uart_latency_ms: {uart_latency_ms:.2f}",
                        f"uart_transfer_kbps: {uart_transfer_kbps:.2f}",
                    ]
                ),
                media_type="text/plain",
            ),
        )
        rr.log(
            "runtime/joint_summary",
            rr.TextDocument(
                format_joint_motion_summary(self.joint_names, rotation_state, range_ratio, qd),
                media_type="text/plain",
            ),
        )

        if ee_pos is not None:
            rr.log(
                "trajectory_3d/ee",
                rr.Points3D(
                    [np.asarray(ee_pos, dtype=np.float64)],
                    colors=[[230, 100, 50]],
                    radii=[0.012],
                    labels=["ee"],
                ),
            )

    def log_identification_summary(
        self,
        result: FrictionIdentificationResult,
        *,
        active_joint_names: list[str],
        active_joint_indices: list[int] | np.ndarray | None = None,
    ) -> None:
        self._log_identification_metrics(
            title="Identification Summary",
            joint_names=active_joint_names,
            coulomb=np.asarray([param.coulomb for param in result.parameters], dtype=np.float64),
            viscous=np.asarray([param.viscous for param in result.parameters], dtype=np.float64),
            offset=np.asarray([param.offset for param in result.parameters], dtype=np.float64),
            validation_rmse=np.asarray(result.validation_rmse, dtype=np.float64),
            validation_r2=np.asarray(result.validation_r2, dtype=np.float64),
            joint_indices=active_joint_indices,
        )

    def log_loaded_summary(self, summary: "IdentificationResults") -> None:
        joint_names = [joint.joint_name for joint in summary.joint_results]
        self._log_identification_metrics(
            title="Hardware Parallel Identification Summary",
            joint_names=joint_names,
            coulomb=np.asarray([joint.coulomb for joint in summary.joint_results], dtype=np.float64),
            viscous=np.asarray([joint.viscous for joint in summary.joint_results], dtype=np.float64),
            offset=np.asarray([joint.offset for joint in summary.joint_results], dtype=np.float64),
            validation_rmse=np.asarray([joint.validation_rmse for joint in summary.joint_results], dtype=np.float64),
            validation_r2=np.asarray([joint.validation_r2 for joint in summary.joint_results], dtype=np.float64),
            valid_sample_ratio=np.asarray(
                [joint.valid_sample_ratio for joint in summary.joint_results],
                dtype=np.float64,
            ),
            sample_count=np.asarray([joint.sample_count for joint in summary.joint_results], dtype=np.int64),
            joint_indices=np.asarray([joint.joint_index for joint in summary.joint_results], dtype=np.int64),
            batch_coulomb=summary.batch_coulomb,
            batch_viscous=summary.batch_viscous,
            batch_offset=summary.batch_offset,
            batch_validation_rmse=summary.batch_validation_rmse,
            batch_validation_r2=summary.batch_validation_r2,
            batch_valid_sample_ratio=summary.batch_valid_sample_ratio,
        )

    def _log_identification_metrics(
        self,
        *,
        title: str,
        joint_names: list[str],
        coulomb: np.ndarray,
        viscous: np.ndarray,
        offset: np.ndarray,
        validation_rmse: np.ndarray,
        validation_r2: np.ndarray,
        valid_sample_ratio: np.ndarray | None = None,
        sample_count: np.ndarray | None = None,
        joint_indices: list[int] | np.ndarray | None = None,
        batch_coulomb: np.ndarray | None = None,
        batch_viscous: np.ndarray | None = None,
        batch_offset: np.ndarray | None = None,
        batch_validation_rmse: np.ndarray | None = None,
        batch_validation_r2: np.ndarray | None = None,
        batch_valid_sample_ratio: np.ndarray | None = None,
    ) -> None:
        if self._rr is None:
            return
        rr = self._rr
        rr.log("identification/validation_rmse", rr.BarChart(np.asarray(validation_rmse, dtype=np.float64)))
        rr.log("identification/validation_r2", rr.BarChart(np.asarray(validation_r2, dtype=np.float64)))
        rr.log("identification/coulomb", rr.BarChart(np.asarray(coulomb, dtype=np.float64)))
        rr.log("identification/viscous", rr.BarChart(np.asarray(viscous, dtype=np.float64)))
        rr.log("identification/offset", rr.BarChart(np.asarray(offset, dtype=np.float64)))
        if valid_sample_ratio is not None:
            rr.log("identification/valid_sample_ratio", rr.BarChart(np.asarray(valid_sample_ratio, dtype=np.float64)))
        if joint_indices is None:
            joint_indices = np.arange(len(joint_names), dtype=np.int64)
        else:
            joint_indices = np.asarray(joint_indices, dtype=np.int64).reshape(-1)
        self._log_identification_history(
            joint_indices=joint_indices,
            batch_values={
                "coulomb": batch_coulomb,
                "viscous": batch_viscous,
                "offset": batch_offset,
                "validation_rmse": batch_validation_rmse,
                "validation_r2": batch_validation_r2,
                "valid_sample_ratio": batch_valid_sample_ratio,
            },
            latest_values={
                "coulomb": np.asarray(coulomb, dtype=np.float64),
                "viscous": np.asarray(viscous, dtype=np.float64),
                "offset": np.asarray(offset, dtype=np.float64),
                "validation_rmse": np.asarray(validation_rmse, dtype=np.float64),
                "validation_r2": np.asarray(validation_r2, dtype=np.float64),
                "valid_sample_ratio": np.asarray(valid_sample_ratio, dtype=np.float64)
                if valid_sample_ratio is not None
                else None,
            },
        )

        lines = [
            f"# {title}",
            "",
            "| Joint | Coulomb | Viscous | Offset | Val RMSE | Val R2 | Sample Count | Valid Ratio |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        if sample_count is None:
            sample_count = np.full(len(joint_names), -1, dtype=np.int64)
        if valid_sample_ratio is None:
            valid_sample_ratio = np.full(len(joint_names), np.nan, dtype=np.float64)
        for joint_name, coulomb_value, viscous_value, offset_value, rmse, r2, count, ratio in zip(
            joint_names,
            coulomb,
            viscous,
            offset,
            validation_rmse,
            validation_r2,
            sample_count,
            valid_sample_ratio,
        ):
            lines.append(
                f"| {joint_name} | {coulomb_value:.5f} | {viscous_value:.5f} | "
                f"{offset_value:.5f} | {rmse:.6f} | {r2:.4f} | {int(count)} | {ratio:.4f} |"
            )
        rr.log("identification/summary", rr.TextDocument("\n".join(lines), media_type="text/markdown"))

    def _log_identification_history(
        self,
        *,
        joint_indices: np.ndarray,
        batch_values: dict[str, np.ndarray | None],
        latest_values: dict[str, np.ndarray | None],
    ) -> None:
        if self._rr is None or joint_indices.size == 0:
            return
        rr = self._rr

        logged_any_history = False
        for metric_key, values in batch_values.items():
            if values is None:
                continue
            series = np.asarray(values, dtype=np.float64)
            if series.ndim != 2:
                continue
            logged_any_history = True
            for batch_idx in range(series.shape[0]):
                rr.set_time_sequence("identification_batch", int(batch_idx + 1))
                for local_idx, joint_idx in enumerate(joint_indices):
                    if joint_idx < 0 or joint_idx >= len(self.joint_names):
                        continue
                    if joint_idx >= series.shape[1]:
                        continue
                    value = float(series[batch_idx, joint_idx])
                    if not np.isfinite(value):
                        continue
                    rr.log(_identification_history_path(metric_key, int(joint_idx) + 1), rr.Scalars(value))

        if logged_any_history:
            return

        rr.set_time_sequence("identification_batch", 1)
        for metric_key, values in latest_values.items():
            if values is None:
                continue
            series = np.asarray(values, dtype=np.float64).reshape(-1)
            if series.size != joint_indices.size:
                continue
            for local_idx, joint_idx in enumerate(joint_indices):
                if joint_idx < 0 or joint_idx >= len(self.joint_names):
                    continue
                value = float(series[local_idx])
                if not np.isfinite(value):
                    continue
                rr.log(_identification_history_path(metric_key, int(joint_idx) + 1), rr.Scalars(value))

    def close(self) -> None:
        if self._rr is not None:
            self._rr.disconnect()
            self._rr = None


class PoseEstimator:
    """Forward-kinematics helper for visualizing live robot joint states in MuJoCo."""

    def __init__(
        self,
        *,
        model_path: str,
        joint_names: list[str],
        end_effector_body: str,
        tcp_offset: np.ndarray,
        render: bool = True,
        viewer_fps: float = 30.0,
    ) -> None:
        import mujoco
        import mujoco.viewer

        self._mujoco = mujoco
        self._viewer_module = mujoco.viewer
        self.model = build_am_d02_model(model_path, np.asarray(tcp_offset, dtype=np.float64))
        self.data = mujoco.MjData(self.model)
        self.qpos_addrs = []
        self.viewer = None
        self._viewer_period_s = 1.0 / max(float(viewer_fps), 1.0)
        self._last_viewer_sync = 0.0

        for name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"找不到关节: {name}")
            self.qpos_addrs.append(self.model.jnt_qposadr[joint_id])

        self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, end_effector_body)
        if self.ee_body_id < 0:
            raise ValueError(f"找不到末端 body: {end_effector_body}")

        if render:
            self.viewer = self._viewer_module.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
            )
            self.viewer.cam.azimuth = 135
            self.viewer.cam.elevation = -20
            self.viewer.cam.distance = 1.8
            self.viewer.cam.lookat[:] = [0.0, 0.0, 1.1]

    def update(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.size != len(self.qpos_addrs):
            raise ValueError("q size does not match the configured robot joints.")

        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        for addr, value in zip(self.qpos_addrs, q):
            self.data.qpos[addr] = value
        self._mujoco.mj_forward(self.model, self.data)

        if self.viewer is not None:
            if not self.viewer.is_running():
                self.viewer.close()
                self.viewer = None
            else:
                now = time.perf_counter()
                if now - self._last_viewer_sync >= self._viewer_period_s:
                    self.viewer.sync()
                    self._last_viewer_sync = now

        return self.data.xpos[self.ee_body_id].copy(), self.data.xquat[self.ee_body_id].copy()

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


def build_hardware_reporter(config: Config):
    if not config.visualization.spawn_rerun:
        return None

    try:
        reporter = HardwareRerunReporter(
            app_name="AM-D02 Parallel Hardware Friction Identification",
            joint_names=list(config.robot.joint_names),
            spawn=True,
        )
        reporter.init()
        return reporter
    except Exception as exc:
        log_info(f"Rerun 真机可视化初始化失败，将继续运行: {exc}")
        return None


def build_pose_estimator(config: Config):
    if not config.visualization.render:
        return None

    try:
        return PoseEstimator(
            model_path=str(config.robot.urdf_path),
            joint_names=list(config.robot.joint_names),
            end_effector_body=config.robot.end_effector_body,
            tcp_offset=config.robot.tcp_offset,
            render=True,
            viewer_fps=config.visualization.viewer_fps,
        )
    except Exception as exc:
        log_info(f"MuJoCo 真机姿态可视化初始化失败，将继续运行: {exc}")
        return None


__all__ = [
    "HardwareRerunReporter",
    "PoseEstimator",
    "build_hardware_reporter",
    "build_pose_estimator",
]
