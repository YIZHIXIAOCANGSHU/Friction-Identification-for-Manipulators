from __future__ import annotations

"""Realtime Rerun dashboards for live UART collection and compensation runs."""

from typing import Sequence

import numpy as np
import rerun as rr
import rerun.blueprint as rrb


class RealTimeRerunReporter:
    """Publish live joint, torque, temperature, and UART diagnostics to Rerun."""

    def __init__(self, *, app_name: str, joint_names: Sequence[str], spawn: bool = True) -> None:
        self.app_name = app_name
        self.spawn = spawn
        self.joint_names = list(joint_names)

    def init(self) -> None:
        """Initialize Rerun streams and send the dashboard layout."""

        rr.init(self.app_name, spawn=self.spawn)

        # Keep one consistent color per joint across all plots.
        joint_colors = [
            [230, 50, 50],
            [230, 140, 30],
            [210, 200, 30],
            [50, 200, 50],
            [50, 200, 200],
            [50, 80, 230],
            [150, 50, 230],
        ]

        for idx, color in enumerate(joint_colors, start=1):
            rr.log(
                f"joint_state/q/J{idx}",
                rr.SeriesLines(colors=[color], names=[f"J{idx} q"], widths=[2.0]),
                static=True,
            )
            rr.log(
                f"joint_state/qd/J{idx}",
                rr.SeriesLines(colors=[color], names=[f"J{idx} qd"], widths=[2.0]),
                static=True,
            )
            rr.log(
                f"torque/measured/J{idx}",
                rr.SeriesLines(colors=[color], names=[f"J{idx} measured"], widths=[2.0]),
                static=True,
            )
            rr.log(
                f"torque/command/J{idx}",
                rr.SeriesLines(colors=[color], names=[f"J{idx} command"], widths=[2.0]),
                static=True,
            )
            rr.log(
                f"torque/gap/J{idx}",
                rr.SeriesLines(colors=[color], names=[f"J{idx} command - measured"], widths=[2.0]),
                static=True,
            )
            rr.log(
                f"temperature/mos/J{idx}",
                rr.SeriesLines(colors=[color], names=[f"J{idx} mos"], widths=[2.0]),
                static=True,
            )
            rr.log(
                f"temperature/coil/J{idx}",
                rr.SeriesLines(colors=[color], names=[f"J{idx} coil"], widths=[2.0]),
                static=True,
            )

        rr.log(
            "performance/uart_cycle_hz",
            rr.SeriesLines(colors=[[230, 120, 40]], names=["UART cycle rate"], widths=[2.0]),
            static=True,
        )
        rr.log(
            "performance/uart_latency_ms",
            rr.SeriesLines(colors=[[230, 170, 80]], names=["UART cycle period"], widths=[2.0]),
            static=True,
        )
        rr.log(
            "performance/uart_transfer_kbps",
            rr.SeriesLines(colors=[[100, 180, 230]], names=["UART throughput"], widths=[2.0]),
            static=True,
        )

        rr.log(
            "trajectory_3d/world",
            rr.Arrows3D(
                vectors=[[0.12, 0.0, 0.0], [0.0, 0.12, 0.0], [0.0, 0.0, 0.12]],
                colors=[[220, 50, 50], [50, 220, 50], [50, 50, 220]],
            ),
            static=True,
        )

        joint_state_views = [
            rrb.TimeSeriesView(name=f"J{idx + 1} Position", origin=f"/joint_state/q/J{idx + 1}")
            for idx in range(len(self.joint_names))
        ]
        joint_vel_views = [
            rrb.TimeSeriesView(name=f"J{idx + 1} Velocity", origin=f"/joint_state/qd/J{idx + 1}")
            for idx in range(len(self.joint_names))
        ]
        measured_torque_views = [
            rrb.TimeSeriesView(name=f"J{idx + 1} Measured", origin=f"/torque/measured/J{idx + 1}")
            for idx in range(len(self.joint_names))
        ]
        command_torque_views = [
            rrb.TimeSeriesView(name=f"J{idx + 1} Command", origin=f"/torque/command/J{idx + 1}")
            for idx in range(len(self.joint_names))
        ]
        torque_gap_views = [
            rrb.TimeSeriesView(name=f"J{idx + 1} Gap", origin=f"/torque/gap/J{idx + 1}")
            for idx in range(len(self.joint_names))
        ]

        blueprint = rrb.Blueprint(
            rrb.Tabs(
                rrb.Grid(*joint_state_views, name="Joint Position"),
                rrb.Grid(*joint_vel_views, name="Joint Velocity"),
                rrb.Grid(*measured_torque_views, name="Measured Torque"),
                rrb.Grid(*command_torque_views, name="Command Torque"),
                rrb.Grid(*torque_gap_views, name="Torque Gap"),
                rrb.Vertical(
                    rrb.TimeSeriesView(name="UART Frequency", origin="/performance/uart_cycle_hz"),
                    rrb.TimeSeriesView(name="UART Period", origin="/performance/uart_latency_ms"),
                    rrb.TimeSeriesView(name="UART Throughput", origin="/performance/uart_transfer_kbps"),
                    rrb.TextLogView(name="UART Frames", origin="/uart_log"),
                    name="UART",
                ),
                rrb.Vertical(
                    rrb.Spatial3DView(name="EE Pose", origin="/trajectory_3d"),
                    rrb.Grid(
                        *[
                            rrb.TimeSeriesView(name=f"J{idx + 1} MOS Temp", origin=f"/temperature/mos/J{idx + 1}")
                            for idx in range(len(self.joint_names))
                        ],
                        name="MOS Temp",
                    ),
                    rrb.Grid(
                        *[
                            rrb.TimeSeriesView(name=f"J{idx + 1} Coil Temp", origin=f"/temperature/coil/J{idx + 1}")
                            for idx in range(len(self.joint_names))
                        ],
                        name="Coil Temp",
                    ),
                    name="3D And Temp",
                ),
            ),
            collapse_panels=True,
        )
        rr.send_blueprint(blueprint)

    def log_step(
        self,
        *,
        elapsed_s: float,
        step_index: int,
        q: np.ndarray,
        qd: np.ndarray,
        tau_measured: np.ndarray,
        tau_command: np.ndarray,
        mos_temperature: np.ndarray,
        coil_temperature: np.ndarray,
        uart_cycle_hz: float,
        uart_latency_ms: float,
        uart_transfer_kbps: float,
        ee_pos: np.ndarray | None,
        ee_quat: np.ndarray | None,
        rx_text: str | None,
        tx_text: str | None,
    ) -> None:
        """Append one completed 7-axis UART cycle to the live dashboard."""

        rr.set_time_seconds("time", float(elapsed_s))
        rr.set_time_sequence("step", int(step_index))

        for idx in range(q.shape[0]):
            rr.log(f"joint_state/q/J{idx + 1}", rr.Scalars(float(q[idx])))
            rr.log(f"joint_state/qd/J{idx + 1}", rr.Scalars(float(qd[idx])))
            rr.log(f"torque/measured/J{idx + 1}", rr.Scalars(float(tau_measured[idx])))
            rr.log(f"torque/command/J{idx + 1}", rr.Scalars(float(tau_command[idx])))
            rr.log(f"torque/gap/J{idx + 1}", rr.Scalars(float(tau_command[idx] - tau_measured[idx])))
            rr.log(f"temperature/mos/J{idx + 1}", rr.Scalars(float(mos_temperature[idx])))
            rr.log(f"temperature/coil/J{idx + 1}", rr.Scalars(float(coil_temperature[idx])))

        rr.log("performance/uart_cycle_hz", rr.Scalars(float(uart_cycle_hz)))
        rr.log("performance/uart_latency_ms", rr.Scalars(float(uart_latency_ms)))
        rr.log("performance/uart_transfer_kbps", rr.Scalars(float(uart_transfer_kbps)))

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
        if rx_text is not None and tx_text is not None:
            rr.log("uart_log", rr.TextLog(f"[{step_index}] RX {rx_text}\n[{step_index}] TX {tx_text}"))

    def close(self) -> None:
        """Disconnect from the Rerun SDK."""

        rr.disconnect()
